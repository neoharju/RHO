/// @file gemm.cppm
/// @brief Matrix multiplication (GEMM) and matrix transpose.
export module rho.linalg.gemm;

import std;
import rho.core.matrix;

export namespace rho::linalg {

/// @brief Reference GEMM: C += A * B.
///
/// The function adds to C. It does not set C to zero first. Fill C (for example
/// with the bias) before the call.
///
/// Loop order is i-p-j. For one A(i, p), the inner loop reads row p of B and
/// writes row i of C, both in memory order, so the compiler vectorizes it (FMA).
/// The rows of C are divided between the OpenMP threads. Each thread writes
/// only its own rows, so there is no data race.
///
/// Time: O(m * k * n). Extra memory: none.
///
/// @tparam T Element type.
/// @param A Input, m x k.
/// @param B Input, k x n.
/// @param C Input and output, m x n. On return C = C + A * B.
/// @pre A.cols() == B.rows(), C.rows() == A.rows(), C.cols() == B.cols().
/// @pre C does not overlap A or B.
template <class T>
void gemm(rho::core::matrix_view<const T> A,
          rho::core::matrix_view<const T> B,
          rho::core::matrix_view<T> C) noexcept {

    const auto m = A.rows();
    const auto k = A.cols();
    const auto n = B.cols();

#pragma omp parallel for
    for (std::size_t i = 0; i < m; ++i) {
        const auto ci = C.row(i);

        for (std::size_t p{0}; p < k; ++p) {
            const auto bp = B.row(p);
            // take one A element constant for slightly more cache efficiency
            const T aip = A(i, p);

            for (std::size_t j{0}; j < n; ++j) {
                ci[j] = aip * bp[j] + ci[j]; // fma
            }
        }
    }
}

/// @brief Simple transpose: dst = src^T.
///
/// The reads are in memory order. The writes go down one column of dst, so they
/// are not cache friendly for large matrices. For the sizes in this project
/// (at most 784 x 256) this is fast enough.
///
/// @param src Input, r x c.
/// @param dst Output, c x r.
/// @pre dst.rows() == src.cols() and dst.cols() == src.rows().
template <class T>
void transpose(rho::core::matrix_view<const T> src, rho::core::matrix_view<T> dst) noexcept {
#pragma omp parallel for
    for (std::size_t i = 0; i < src.rows(); ++i) {
        for (std::size_t j = 0; j < src.cols(); ++j) {
            dst(j, i) = src(i, j);
        }
    }
}

} // namespace rho::linalg
