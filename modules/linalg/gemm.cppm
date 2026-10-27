/// @file gemm.cppm
/// @brief Matrix multiplication (GEMM) for float with AVX2 and FMA.
///
/// gemm(), gemm_tn() and gemm_nt() compute C = A * B, C = A^T * B and C = A * B^T. They can
/// gemm_batch() computes several independent GEMMs in one parallel region. It can also add a bias row and apply ReLU when it writes C.
///
/// The GEMM uses a modified method of Goto and van de Geijn (BLIS). It has five loops around a small kernel. Each loop level keeps one part of the data in one cache level:
///
/// @code
/// for jc in steps of NC:                  // B~ panel, KC x NC        (L2/L3)
///     for pc in steps of KC:              // depth blocks
///         pack B[pc.., jc..] into B~
///         for ic in steps of MC:          // block of A, MC x KC      (L1/L2)
///             macro_kernel: for each B~ micro-panel, for each MR-row panel of A:
///                 micro_kernel: C tile (MR x NR) += A panel * B~ micro-panel
/// @endcode
///
/// Only B is packed. The kernel reads A where it is: with MR rows, a few rows of A are in
/// use at a time, and the block of A stays in L1/L2 while the kernel uses it with each
/// micro-panel of B~.
///
/// A and B can be transposed. The code reads them through a strided_view, which has a step
/// for the rows and a step for the columns. A transposed matrix only swaps the two steps, so
/// there is no copy of A^T or B^T.
///
/// Each thread does these loops for its own part of C, with its own packing buffer.
/// So the threads do not wait for each other.
///
/// Source: https://www.cs.utexas.edu/~flame/laff/pfhp/LAff-On-PfHP.pdf
module;

#include <immintrin.h> // __m256, _mm256_*
#include <omp.h>       // omp_get_num_threads(), omp_get_thread_num()

export module rho.linalg.gemm;

import std;
import rho.core.matrix;

using rho::core::matrix_view;

export namespace rho::linalg {

enum class gemm_mode : std::uint8_t {
    accumulate, ///< C = C + A * B.
    overwrite   ///< C = A * B.
};

/// @brief One GEMM for gemm_batch(): C = op(A) * op(B) + bias (+ C), then ReLU if relu is true.
///
/// op(A) is A, or A^T if A_transposed is true; op(B) in the same way. op(A) is m x k, op(B)
/// is k x n and C is m x n. The GEMM reads A^T and B^T in place: there is no copy.
struct gemm_task {
    matrix_view<const float> A;             ///< m x k, or k x m if A_transposed
    matrix_view<const float> B;             ///< k x n, or n x k if B_transposed
    matrix_view<float> C;                   ///< m x n
    bool A_transposed = false;              ///< true: op(A) = A^T
    bool B_transposed = false;              ///< true: op(B) = B^T
    gemm_mode mode = gemm_mode::accumulate; ///< add to C, or write over C
    std::span<const float> bias = {};       ///< empty, or n values: added once to each row of C
    bool relu = false;                      ///< true: C = (C > 0 ? C : 0) at the end
};

} // namespace rho::linalg


using rho::linalg::gemm_mode;
using rho::linalg::gemm_task;

/// @brief op(M) of a row-major matrix M: element (i, j) is at data[i * row_step + j * col_step].
///
/// For M, row_step is the row stride and col_step is 1. For M^T the two steps change places,
/// so the GEMM reads M^T in place.
struct strided_view {
    const float *data;
    std::size_t row_step;
    std::size_t col_step;

    /// @return op(M): M, or M^T if transposed is true.
    [[nodiscard]]
    static constexpr strided_view of(matrix_view<const float> m, bool transposed) noexcept {
        return transposed ? strided_view{m.data(), 1, m.stride()}
                          : strided_view{m.data(), m.stride(), 1};
    }

    /// @return The block that starts at element (i, j).
    [[nodiscard]]
    constexpr strided_view block(std::size_t i, std::size_t j) const noexcept {
        return {data + i * row_step + j * col_step, row_step, col_step};
    }
};

#pragma GCC push_options
#pragma GCC target("avx2,fma")

/// Register block: the micro-kernel keeps an MR x NR tile of C in registers.
namespace kernel {
constexpr std::size_t lanes{8}; // floats in one __m256
constexpr std::size_t MR{6};    // rows of the tile: one broadcast of A for each row
constexpr std::size_t NR{16};   // columns of the tile: two __m256 of B for each row
} // namespace kernel

namespace block {
// A block of A (MC x KC floats = 36 KiB) stays in L1/L2. MC must be a multiple of MR.
constexpr std::size_t MC{24};
// Depth of a block (the last block can be smaller). A B~ micro-panel (KC x NR floats =
// 24 KiB) stays in L1.
constexpr std::size_t KC{384};
// B~ (KC x NC floats = 3 MiB) stays in L2/L3. NC must be a multiple of NR.
constexpr std::size_t NC{2048};
} // namespace block

static_assert(kernel::NR == 2 * kernel::lanes);
static_assert(block::MC % kernel::MR == 0);
static_assert(block::NC % kernel::NR == 0);

/// The bias for "no bias": x + -0.0 == x for every x (also for -0.0), so it changes nothing.
static constexpr std::array<float, kernel::NR> no_bias = [] {
    std::array<float, kernel::NR> v{};
    v.fill(-0.0F);
    return v;
}();

/// @return x, rounded up to a multiple of m.
[[nodiscard]]
static constexpr std::size_t round_up(std::size_t x, std::size_t m) noexcept {
    return (x + m - 1) / m * m;
}

/// @return A view of the rows x cols block that starts at element (i, j) of v.
template <class T>
[[nodiscard]]
static constexpr matrix_view<T> block_view(
    matrix_view<T> v, std::size_t i, std::size_t j, std::size_t rows, std::size_t cols) noexcept {
    return {v.data() + i * v.stride() + j, rows, cols, v.stride()};
}

/// @brief Pack a kb x nb block of op(B) into dst, as micro-panels of NR columns.
///
/// The micro-panel of columns j to j + NR - 1 starts at dst[j * kb]. columns
/// row by row: dst[j * kb + p * NR + jj] = op(B)(p, j + jj).
///
/// padded with zeros.
///
/// @param B The block of op(B).
/// @param kb Rows of the block.
/// @param nb Columns of the block.
/// @param dst Output, kb * round_up(nb, NR) floats.
static void pack_b(strided_view B, std::size_t kb, std::size_t nb, float *dst) noexcept {
    for (std::size_t j = 0; j < nb; j += kernel::NR) {
        const std::size_t jb = std::min(kernel::NR, nb - j);
        const float *column_j = B.data + j * B.col_step; // op(B)(0, j)
        float *panel = dst + j * kb;

        if (jb == kernel::NR && B.col_step == 1) { // full micro-panel of B: rows of NR floats
            for (std::size_t p = 0; p < kb; ++p) {
                for (std::size_t jj = 0; jj < kernel::NR; ++jj) { // GCC makes vector moves
                    panel[p * kernel::NR + jj] = column_j[p * B.row_step + jj];
                }
            }
        } else if (jb == kernel::NR) { // full micro-panel of B^T: a column is a row in memory
            for (std::size_t jj = 0; jj < kernel::NR; ++jj) {
                const float *column = column_j + jj * B.col_step;
                for (std::size_t p = 0; p < kb; ++p) {
                    panel[p * kernel::NR + jj] = column[p * B.row_step];
                }
            }
        } else { // last micro-panel of the panel
            for (std::size_t p = 0; p < kb; ++p) {
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    panel[p * kernel::NR + jj] = column_j[p * B.row_step + jj * B.col_step];
                }
                for (std::size_t jj = jb; jj < kernel::NR; ++jj) {
                    panel[p * kernel::NR + jj] = 0.0F;
                }
            }
        }
    }
}

/// @brief C(0:MR, 0:NR) = A * B~ + bias + (C or 0), then ReLU if asked, for one MR x kb
///        panel of op(A) and one packed micro-panel of B.
///
/// A is not packed like in BLIS; the kernel reads it through its strided_view.
///
/// The tile of C is in 12 named ymm registers: c0_lo to c5_hi (row i, columns 0-7 and
/// 8-15). 
///
/// @tparam edge false: a full MR x NR tile. true: an edge tile.
/// @param kb Shared dimension.
/// @param A The MR x kb panel of op(A).
/// @param B Packed kb x NR micro-panel of B (see pack_b), 32-byte aligned.
/// @param bias Bias of the NR columns of the tile (no_bias: no bias).
/// @param C Element (0, 0) of the tile.
/// @param ldc Row stride of C, in elements.
/// @param read All ones: add to C. All zeros: write over C.
/// @param keep_all All zeros: ReLU. All ones: no ReLU.
/// @param rows Rows of the tile in C (only for an edge tile).
/// @param cols Columns of the tile in C (only for an edge tile).
template <bool edge>
static void micro_kernel(std::size_t kb,
                         strided_view A,
                         const float *B,
                         const float *bias,
                         float *C,
                         std::size_t ldc,
                         __m256i read,
                         __m256 keep_all,
                         std::size_t rows,
                         std::size_t cols) noexcept {

    // compile time check just to make sure the kernel size is correct
	static_assert(kernel::MR == 6 && kernel::NR == 16, "the kernel is written for 6 x 16");

    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i cols_lo = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(cols)), lane);
    const __m256i cols_hi = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(cols) - 8), lane);
    
	const auto mask = [&] [[gnu::always_inline]] (std::size_t i, std::size_t half) {
        return _mm256_and_si256(half == 0 ? cols_lo : cols_hi,
                                _mm256_set1_epi32(i < rows ? -1 : 0));
    };
    
	const auto address = [&] [[gnu::always_inline]] (std::size_t i, std::size_t half) {
        return C + (edge && i >= rows ? 0 : i) * ldc + half;
    };
    
	// half is 0 for columns 0-7, and kernel::lanes for columns 8-15.
    const auto load = [&] [[gnu::always_inline]] (std::size_t i, std::size_t half) {
        if constexpr (edge) {
            return _mm256_maskload_ps(address(i, half), _mm256_and_si256(read, mask(i, half)));
        } else {
            return _mm256_and_ps(_mm256_loadu_ps(address(i, half)), _mm256_castsi256_ps(read));
        }
    };
    
	const auto load_bias = [&] [[gnu::always_inline]] (std::size_t half) {
        if constexpr (edge) {
            return _mm256_maskload_ps(bias + half, half == 0 ? cols_lo : cols_hi);
        } else {
            return _mm256_loadu_ps(bias + half);
        }
    };

    // ReLU as in rho.nn.activation: v > 0 ? v : 0. NaN and -0.0 give +0.0.
    const __m256 zero = _mm256_setzero_ps();
    const auto store = [&] [[gnu::always_inline]] (std::size_t i, std::size_t half, __m256 v) {
        v = _mm256_and_ps(v, _mm256_or_ps(_mm256_cmp_ps(v, zero, _CMP_GT_OQ), keep_all));
        if constexpr (edge) {
            _mm256_maskstore_ps(address(i, half), mask(i, half), v);
        } else {
            _mm256_storeu_ps(address(i, half), v);
        }
    };

    // Offsets of rows 1 to 5 of A from row 0. A row that does not exist reads row 0.
    const auto row_offset = [&] [[gnu::always_inline]] (std::size_t i) {
        return (edge && i >= rows ? 0 : i) * A.row_step;
    };
    const std::size_t row1 = row_offset(1), row2 = row_offset(2), row3 = row_offset(3),
                      row4 = row_offset(4), row5 = row_offset(5);

    constexpr std::size_t hi{kernel::lanes}; // 8 for avx2, 16 for avx512

    const __m256 bias_lo = load_bias(0), bias_hi = load_bias(hi);
    __m256 c0_lo = _mm256_add_ps(load(0, 0), bias_lo)
	__m256 c0_hi = _mm256_add_ps(load(0, hi), bias_hi);

    __m256 c1_lo = _mm256_add_ps(load(1, 0), bias_lo)
	__m256 c1_hi = _mm256_add_ps(load(1, hi), bias_hi);

    __m256 c2_lo = _mm256_add_ps(load(2, 0), bias_lo)
	__m256 c2_hi = _mm256_add_ps(load(2, hi), bias_hi);

    __m256 c3_lo = _mm256_add_ps(load(3, 0), bias_lo)
	__m256 c3_hi = _mm256_add_ps(load(3, hi), bias_hi);

    __m256 c4_lo = _mm256_add_ps(load(4, 0), bias_lo)
	__m256 c4_hi = _mm256_add_ps(load(4, hi), bias_hi);

    __m256 c5_lo = _mm256_add_ps(load(5, 0), bias_lo)
	__m256 c5_hi = _mm256_add_ps(load(5, hi), bias_hi);

    const float *a_p = A.data; // A(0, p): moves one column of A in each step
    // 12 accumulators + 2 B vectors + 1 A broadcast = 15 of the 16 ymm registers
    for (std::size_t p = 0; p < kb; ++p) {
        
		// row p of the B micro-panel: B(p, 0:16)
        const __m256 b_lo = _mm256_load_ps(B);
        const __m256 b_hi = _mm256_load_ps(B + hi);
        __m256 a_ip; // A(i, p) in all 8 lanes. Then C(i, 0:16) += A(i, p) * B(p, 0:16).

        a_ip = _mm256_broadcast_ss(a_p);
        c0_lo = _mm256_fmadd_ps(a_ip, b_lo, c0_lo);
        c0_hi = _mm256_fmadd_ps(a_ip, b_hi, c0_hi);

        a_ip = _mm256_broadcast_ss(a_p + row1);
        c1_lo = _mm256_fmadd_ps(a_ip, b_lo, c1_lo);
        c1_hi = _mm256_fmadd_ps(a_ip, b_hi, c1_hi);

        a_ip = _mm256_broadcast_ss(a_p + row2);
        c2_lo = _mm256_fmadd_ps(a_ip, b_lo, c2_lo);
        c2_hi = _mm256_fmadd_ps(a_ip, b_hi, c2_hi);

        a_ip = _mm256_broadcast_ss(a_p + row3);
        c3_lo = _mm256_fmadd_ps(a_ip, b_lo, c3_lo);
        c3_hi = _mm256_fmadd_ps(a_ip, b_hi, c3_hi);

        a_ip = _mm256_broadcast_ss(a_p + row4);
        c4_lo = _mm256_fmadd_ps(a_ip, b_lo, c4_lo);
        c4_hi = _mm256_fmadd_ps(a_ip, b_hi, c4_hi);

        a_ip = _mm256_broadcast_ss(a_p + row5);
        c5_lo = _mm256_fmadd_ps(a_ip, b_lo, c5_lo);
        c5_hi = _mm256_fmadd_ps(a_ip, b_hi, c5_hi);

        a_p += A.col_step;
        B += kernel::NR;
    }

    store(0, 0, c0_lo);
    store(0, hi, c0_hi);
    store(1, 0, c1_lo);
    store(1, hi, c1_hi);
    store(2, 0, c2_lo);
    store(2, hi, c2_hi);
    store(3, 0, c3_lo);
    store(3, hi, c3_hi);
    store(4, 0, c4_lo);
    store(4, hi, c4_hi);
    store(5, 0, c5_lo);
    store(5, hi, c5_hi);
}

/// @brief C = A * B~ + bias + (C or 0), then ReLU if asked, for one block of op(A) and one
///        packed panel of B.
///
/// MR x NR tiles
///
/// @param kb Shared dimension.
/// @param A The block of op(A), C.rows() x kb (not packed).
/// @param B Packed panel of B, kb x C.cols() (see pack_b).
/// @param bias Bias of column 0 of the block. Column j uses bias[j * bias_step].
/// @param bias_step 1 for a bias row, 0 for no_bias.
/// @param relu true: apply ReLU before the store.
/// @param C The block of C.
/// @param load_c true: add to C. 
///				  false: write over C
[[gnu::noinline]]
static void macro_kernel(std::size_t kb,
                         strided_view A,
                         const float *B,
                         const float *bias,
                         std::size_t bias_step,
                         bool relu,
                         matrix_view<float> C,
                         bool load_c) noexcept {

    const std::size_t m = C.rows();
    const std::size_t n = C.cols();
    const std::size_t m_full = m - m % kernel::MR; // rows in full tiles
    const std::size_t n_full = n - n % kernel::NR; // columns in full tiles

    const __m256i read = _mm256_set1_epi32(load_c ? -1 : 0);
    const __m256 keep_all = _mm256_castsi256_ps(_mm256_set1_epi32(relu ? 0 : -1));

    for (std::size_t j = 0; j < n_full; j += kernel::NR) {
        for (std::size_t i = 0; i < m_full; i += kernel::MR) {
            micro_kernel<false>(kb, A.block(i, 0), B + j * kb, bias + j * bias_step, &C(i, j),
                                C.stride(), read, keep_all, kernel::MR, kernel::NR);
        }

        for (std::size_t i = m_full; i < m; i += kernel::MR) { // the last rows
            micro_kernel<true>(kb, A.block(i, 0), B + j * kb, bias + j * bias_step, &C(i, j),
                               C.stride(), read, keep_all, m - i, kernel::NR);
        }
    }

    for (std::size_t j = n_full; j < n; j += kernel::NR) { // the last columns
        for (std::size_t i = 0; i < m; i += kernel::MR) {
            micro_kernel<true>(kb, A.block(i, 0), B + j * kb, bias + j * bias_step, &C(i, j),
                               C.stride(), read, keep_all, std::min(kernel::MR, m - i), n - j);
        }
    }
}

#pragma GCC pop_options

/// @brief The part of GEMM for one thread
///
/// Each thread computes its own part of C. 
/// rows if C has more rows than columns (dW), else columns (forward pass, dx)
static void gemm_part(const gemm_task &task, std::size_t thread, std::size_t threads) noexcept {
    const matrix_view<float> C = task.C;
    const std::size_t m = C.rows();
    const std::size_t n = C.cols();
    const std::size_t k = task.A_transposed ? task.A.rows() : task.A.cols();
    const bool overwrite = task.mode == gemm_mode::overwrite;

    const bool split_rows = m >= n;
    const std::size_t size = split_rows ? m : n;
    const std::size_t tile = split_rows ? kernel::MR : kernel::NR;
    const std::size_t part = round_up((size + threads - 1) / threads, tile);
    const std::size_t first = std::min(thread * part, size);
    
	const std::size_t count = std::min(part, size - first);
    if (count == 0) { // part is empty if C is small
        return;
    }

    const std::size_t i_begin = split_rows ? first : 0;
    const std::size_t i_end = split_rows ? first + count : m;
    const std::size_t j_begin = split_rows ? 0 : first;
    const std::size_t j_end = split_rows ? n : first + count;

    if (k == 0) { // op(A) * op(B) is the zero matrix: C = (C or 0) + bias, then ReLU
        for (std::size_t i = i_begin; i < i_end; ++i) {
            for (std::size_t j = j_begin; j < j_end; ++j) {
                const float v = (overwrite ? 0.0F : C(i, j))
                                + (task.bias.empty() ? -0.0F : task.bias[j]);
                C(i, j) = task.relu ? (v > 0.0F ? v : 0.0F) : v;
            }
        }
        return;
    }

    // The packing buffer of this thread. The thread keeps it for the next call.
    // It only becomes larger, so most calls do not allocate memory.
    thread_local rho::core::aligned_buffer<float> packed_B;
    const std::size_t packed_B_size =
        block::KC * round_up(std::min(block::NC, j_end - j_begin), kernel::NR);

    if (packed_B.size() < packed_B_size) {
        packed_B = rho::core::aligned_buffer<float>(packed_B_size);
    }

    const strided_view op_A = strided_view::of(task.A, task.A_transposed);
    const strided_view op_B = strided_view::of(task.B, task.B_transposed);
    
	// with "No bias" the kernel reads the same 16 values for each column
    const float *bias = task.bias.empty() ? no_bias.data() : task.bias.data();
    const std::size_t bias_step = task.bias.empty() ? 0 : 1;

    for (std::size_t jc = j_begin; jc < j_end; jc += block::NC) {
        const std::size_t nb = std::min(block::NC, j_end - jc);

        for (std::size_t pc = 0; pc < k; pc += block::KC) {
            const std::size_t kb = std::min(block::KC, k - pc);

            // The first depth block adds the bias, and with overwrite it does not read C.
            // The last depth block applies the ReLU.
            const bool first_block = pc == 0;
            const bool last_block = pc + kb == k;
			
            pack_b(op_B.block(pc, jc), kb, nb, packed_B.data());

            for (std::size_t ic = i_begin; ic < i_end; ic += block::MC) {

                const std::size_t mb = std::min(block::MC, i_end - ic);

                macro_kernel(kb, op_A.block(ic, pc), packed_B.data(),
                             first_block ? bias + jc * bias_step : no_bias.data(),
                             first_block ? bias_step : 0, task.relu && last_block,
                             block_view(C, ic, jc, mb, nb), !overwrite || !first_block);
            }
        }
    }
}

export namespace rho::linalg {

/// @brief Compute independent GEMMs in parallel
/// @param tasks The GEMMs.
void gemm_batch(std::span<const gemm_task> tasks) noexcept {
    for (const gemm_task &task : tasks) {
        const std::size_t m = task.A_transposed ? task.A.cols() : task.A.rows();
        const std::size_t k = task.A_transposed ? task.A.rows() : task.A.cols();
        const std::size_t k_of_B = task.B_transposed ? task.B.cols() : task.B.rows();
        const std::size_t n = task.B_transposed ? task.B.rows() : task.B.cols();
        rho::core::require(k == k_of_B && task.C.rows() == m && task.C.cols() == n
                               && (task.bias.empty() || task.bias.size() == n),
                           "gemm: shapes do not match (op(A): m x k, op(B): k x n, C: m x n, "
                           "bias: n)");
    }

#pragma omp parallel
    {
        const auto threads = static_cast<std::size_t>(omp_get_num_threads());
        const auto thread = static_cast<std::size_t>(omp_get_thread_num());
        for (const gemm_task &task : tasks) {
            gemm_part(task, thread, threads);
        }
    }
}

/// @brief GEMM: C += A * B (default), or C = A * B with gemm_mode::overwrite.
/// @param A Input, m x k.
/// @param B Input, k x n.
/// @param C Output, m x n.
/// @param mode accumulate (default) or overwrite. Overwrite is faster than a zero fill+acc
/// @note If the shapes do not match, the program stops (see require()).
void gemm(matrix_view<const float> A,
          matrix_view<const float> B,
          matrix_view<float> C,
          gemm_mode mode = gemm_mode::accumulate) noexcept {
    const std::array task{gemm_task{.A = A, .B = B, .C = C, .mode = mode}};
    gemm_batch(task);
}

/// @brief GEMM with A transposed: C += A^T * B (default), or C = A^T * B with
///        gemm_mode::overwrite.
///
/// The GEMM reads A^T directly from A, so there is no copy of A^T. Duriung the backward pass
/// this gives dW = x^T * dz directly from x.
///
/// @param A Input, k x m (so A^T is m x k).
/// @param B Input, k x n.
/// @param C Output, m x n.
/// @param mode accumulate (default) or overwrite.
void gemm_tn(matrix_view<const float> A,
             matrix_view<const float> B,
             matrix_view<float> C,
             gemm_mode mode = gemm_mode::accumulate) noexcept {
    const std::array task{gemm_task{.A = A, .B = B, .C = C, .A_transposed = true, .mode = mode}};
    gemm_batch(task);
}

/// @brief GEMM with B transposed: C += A * B^T (default), or C = A * B^T with
///        gemm_mode::overwrite.
///
/// The packing step reads B^T directly from B, so there is no copy of B^T. In the backward
/// pass this gives dx = dz * W^T directly from W.
/// @param A Input, m x k.
/// @param B Input, n x k (so B^T is k x n).
/// @param C Output, m x n.
/// @param mode accumulate (default) or overwrite.
void gemm_nt(matrix_view<const float> A,
             matrix_view<const float> B,
             matrix_view<float> C,
             gemm_mode mode = gemm_mode::accumulate) noexcept {
    const std::array task{gemm_task{.A = A, .B = B, .C = C, .B_transposed = true, .mode = mode}};
    gemm_batch(task);
}

} // namespace rho::linalg
