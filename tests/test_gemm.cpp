#include <gtest/gtest.h>
#include <omp.h>

import std;
import rho.core.matrix;
import rho.linalg.gemm;
import rho.core.rng;

using rho::core::aligned_vector;
using rho::core::matrix;

TEST(Gemm, IdentityMatrix) {
    const matrix<float> A{2, 2, aligned_vector<float>{1.0F, 2.0F, 3.0F, 4.0F}};
    const matrix<float> B{2, 2, aligned_vector<float>{1.0F, 0.0F, 0.0F, 1.0F}};
    matrix<float> C{2, 2};
    C.fill(0.0F);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    EXPECT_EQ(C(0, 0), 1.0F);
    EXPECT_EQ(C(0, 1), 2.0F);
    EXPECT_EQ(C(1, 0), 3.0F);
    EXPECT_EQ(C(1, 1), 4.0F);
}

TEST(Gemm, HandComputed_A23_B32_C22) {
    const matrix<float> A{2, 3, aligned_vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
    const matrix<float> B{3, 2, aligned_vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
    matrix<float> C{2, 2};
    C.fill(0.0F);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    EXPECT_EQ(C(0, 0), 22.0F);
    EXPECT_EQ(C(0, 1), 28.0F);
    EXPECT_EQ(C(1, 0), 49.0F);
    EXPECT_EQ(C(1, 1), 64.0F);
}

TEST(Gemm, AccumulatesIntoC) {
    const matrix<float> A{1, 2, aligned_vector<float>{1.0F, 2.0F}};
    const matrix<float> B{2, 1, aligned_vector<float>{3.0F, 4.0F}};
    matrix<float> C{1, 1};
    C.fill(100.0F);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    EXPECT_EQ(C(0, 0), 111.0F); // 100 + 1*3 + 2*4
}

// -----------------------------------------------------------------------------
// gemm() and gemm_tn()
// -----------------------------------------------------------------------------
namespace {

// Fill a matrix with uniform values in [-1, 1).
template <class T>
matrix<T> random_matrix(std::size_t rows, std::size_t cols, rho::core::xoshiro256pp &rng) {
    matrix<T> m{rows, cols};
    for (std::size_t i = 0; i < rows; ++i) {
        for (T &v : m.row(i)) {
            v = static_cast<T>(rng.uniform(-1.0, 1.0));
        }
    }
    return m;
}

template <class T>
matrix<T> copy_of(const matrix<T> &from) {
    matrix<T> to{from.rows(), from.cols()};
    for (std::size_t i = 0; i < from.rows(); ++i) {
        std::ranges::copy(from.row(i), to.row(i).begin());
    }
    return to;
}

// Reference C += A * B in long double, as a check for the accuracy.
template <class T>
void long_double_gemm(const matrix<T> &A, const matrix<T> &B, matrix<T> &C) {
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            long double sum = C(i, j);
            for (std::size_t p = 0; p < A.cols(); ++p) {
                sum += static_cast<long double>(A(i, p)) * B(p, j);
            }
            C(i, j) = static_cast<T>(sum);
        }
    }
}

struct Shape {
    std::size_t m, k, n;
};

constexpr std::array shapes{
    Shape{1, 1, 1},   // smaller than one micro-tile
    Shape{5, 3, 7},   // row edge (5 < 6) and column edge (7 < 16)
    Shape{6, 16, 16}, // exactly one micro-tile
    Shape{7, 17, 17}, // one more than a micro-tile in each direction
    Shape{11, 9, 33},
    Shape{24, 256, 16},
    Shape{50, 600, 40},
    Shape{7, 300, 2049}, // 2 B panels (2048 + 1)
    Shape{64, 784, 256}, // MNIST layer 1: 3 depth blocks (384 + 384 + 16)
    Shape{64, 128, 10},  // MNIST output layer
    Shape{784, 64, 256}, // dW of MNIST layer 1 (x^T * dz)
};

// The transpose of a, for the inputs of gemm_tn() and gemm_nt().
template <class T>
matrix<T> transposed(const matrix<T> &a) {
    matrix<T> t{a.cols(), a.rows()};
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            t(j, i) = a(i, j);
        }
    }
    return t;
}

bool bit_identical(const matrix<float> &x, const matrix<float> &y) {
    for (std::size_t i = 0; i < x.rows(); ++i) {
        if (std::memcmp(x.row(i).data(), y.row(i).data(), x.cols() * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

// The schoolboys i-p-j loop (the first GEMM of RHO): C += A * B. The tests compare gemm()
void simple_gemm(const matrix<float> &A, const matrix<float> &B, matrix<float> &C) {
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t p = 0; p < A.cols(); ++p) {
            const float aip = A(i, p);
            for (std::size_t j = 0; j < B.cols(); ++j) {
                C(i, j) = aip * B(p, j) + C(i, j); // fma
            }
        }
    }
}

// gemm() always uses FMA. simple_gemm() uses FMA if the compiler joins a * b + c
#if defined(__OPTIMIZE__) && defined(__FMA__)
constexpr bool expect_bit_identical = true;
#else
constexpr bool expect_bit_identical = false;
#endif

void expect_same(const matrix<float> &actual, const matrix<float> &expected, std::size_t k) {
    if (expect_bit_identical) {
        EXPECT_TRUE(bit_identical(actual, expected));
        return;
    }
    const float tol = 1e-5F * static_cast<float>(k + 1);
    for (std::size_t i = 0; i < actual.rows(); ++i) {
        for (std::size_t j = 0; j < actual.cols(); ++j) {
            ASSERT_NEAR(actual(i, j), expected(i, j), tol);
        }
    }
}

} // namespace

TEST(Gemm, MatchesLongDouble) {
    rho::core::xoshiro256pp rng{11U};
    for (const auto [m, k, n] : shapes) {
        const auto A = random_matrix<float>(m, k, rng);
        const auto B = random_matrix<float>(k, n, rng);
        auto C = random_matrix<float>(m, n, rng); // accumulate into a C that is not 0
        auto R = copy_of(C);

        rho::linalg::gemm(A.view(), B.view(), C.view());
        long_double_gemm(A, B, R);

        // the float rounding error grows about like k * eps * |values|
        const float tol = 1e-5F * static_cast<float>(k + 1);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                ASSERT_NEAR(C(i, j), R(i, j), tol)
                    << m << "x" << k << "x" << n << " at " << i << "," << j;
            }
        }
    }
}

TEST(GemmTn, MatchesLongDouble) {
    rho::core::xoshiro256pp rng{14U};
    for (const auto [m, k, n] : shapes) {
        const auto At = random_matrix<float>(k, m, rng);
        const auto B = random_matrix<float>(k, n, rng);
        auto C = random_matrix<float>(m, n, rng);
        auto R = copy_of(C);

        rho::linalg::gemm_tn(At.view(), B.view(), C.view());
        long_double_gemm(transposed(At), B, R);

        const float tol = 1e-5F * static_cast<float>(k + 1);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                ASSERT_NEAR(C(i, j), R(i, j), tol) << m << "x" << k << "x" << n;
            }
        }
    }
}

TEST(Gemm, BitIdenticalToSimpleLoop) {
    rho::core::xoshiro256pp rng{31U};
    for (const auto [m, k, n] : shapes) {
        SCOPED_TRACE(std::format("{}x{}x{}", m, k, n));
        const auto A = random_matrix<float>(m, k, rng);
        const auto B = random_matrix<float>(k, n, rng);
        const auto C0 = random_matrix<float>(m, n, rng);

        auto expected = copy_of(C0);
        auto actual = copy_of(C0);
        simple_gemm(A, B, expected);
        rho::linalg::gemm(A.view(), B.view(), actual.view());
        expect_same(actual, expected, k);
    }
}

TEST(GemmTn, BitIdenticalToSimpleLoop) {
    rho::core::xoshiro256pp rng{33U};
    for (const auto [m, k, n] : shapes) {
        SCOPED_TRACE(std::format("{}x{}x{}", m, k, n));
        const auto At = random_matrix<float>(k, m, rng);
        const auto B = random_matrix<float>(k, n, rng);
        const auto C0 = random_matrix<float>(m, n, rng);

        auto expected = copy_of(C0);
        auto actual = copy_of(C0);
        simple_gemm(transposed(At), B, expected);
        rho::linalg::gemm_tn(At.view(), B.view(), actual.view());
        expect_same(actual, expected, k);
    }
}

// the result must not depend on the number of threads.
TEST(Gemm, SameResultForAnyThreadCount) {
    rho::core::xoshiro256pp rng{32U};
    for (const auto [m, k, n] : {Shape{64, 784, 256}, Shape{50, 600, 40}, Shape{784, 64, 256}}) {
        const auto A = random_matrix<float>(m, k, rng);
        const auto At = transposed(A);
        const auto B = random_matrix<float>(k, n, rng);
        const int saved = omp_get_max_threads();

        std::optional<matrix<float>> first;
        for (const int threads : {1, 2, 3, 4}) {
            omp_set_num_threads(threads);
            matrix<float> C{m, n};
            rho::linalg::gemm(A.view(), B.view(), C.view(), rho::linalg::gemm_mode::overwrite);
            matrix<float> C_tn{m, n};
            rho::linalg::gemm_tn(
                At.view(), B.view(), C_tn.view(), rho::linalg::gemm_mode::overwrite);
            EXPECT_TRUE(bit_identical(C_tn, C)) << "gemm_tn, " << threads << " threads";
            if (!first) {
                first.emplace(std::move(C));
            } else {
                EXPECT_TRUE(bit_identical(C, *first))
                    << threads << " threads, " << m << "x" << k << "x" << n;
            }
        }
        omp_set_num_threads(saved);
    }
}

TEST(Gemm, EmptyMatricesDoNothing) {
    const matrix<float> A{3, 0};
    const matrix<float> B{0, 5};
    matrix<float> C{3, 5};
    C.fill(2.0F);
    rho::linalg::gemm(A.view(), B.view(), C.view()); // k = 0: C += 0
    EXPECT_EQ(C(2, 4), 2.0F);

    const matrix<float> A2{0, 4};
    const matrix<float> B2{4, 3};
    matrix<float> C2{0, 3}; // m = 0
    rho::linalg::gemm(A2.view(), B2.view(), C2.view());
    EXPECT_EQ(C2.rows(), 0U);
}

// Overwrite: C is empty (NaN). The result must still be the same as accumulate from C = 0,
TEST(Gemm, OverwriteNeverUsesOldC) {
    rho::core::xoshiro256pp rng{16U};
    for (const auto [m, k, n] : shapes) {
        const auto A = random_matrix<float>(m, k, rng);
        const auto At = transposed(A);
        const auto B = random_matrix<float>(k, n, rng);
        matrix<float> from_zero{m, n};
        from_zero.fill(0.0F);
        rho::linalg::gemm(A.view(), B.view(), from_zero.view());

        matrix<float> C{m, n};
        C.fill(std::numeric_limits<float>::quiet_NaN());
        rho::linalg::gemm(A.view(), B.view(), C.view(), rho::linalg::gemm_mode::overwrite);
        EXPECT_TRUE(bit_identical(C, from_zero)) << "gemm " << m << "x" << k << "x" << n;

        C.fill(std::numeric_limits<float>::quiet_NaN());
        rho::linalg::gemm_tn(At.view(), B.view(), C.view(), rho::linalg::gemm_mode::overwrite);
        EXPECT_TRUE(bit_identical(C, from_zero)) << "gemm_tn " << m << "x" << k << "x" << n;
    }
}

TEST(Gemm, OverwriteWithEmptyInnerSizeGivesZero) {
    const matrix<float> A{3, 0};
    const matrix<float> At{0, 3};
    const matrix<float> B{0, 5};
    matrix<float> C{3, 5};
    C.fill(std::numeric_limits<float>::quiet_NaN());
    rho::linalg::gemm(A.view(), B.view(), C.view(), rho::linalg::gemm_mode::overwrite);
    EXPECT_EQ(C(2, 4), 0.0F);
    C.fill(std::numeric_limits<float>::quiet_NaN());
    rho::linalg::gemm_tn(At.view(), B.view(), C.view(), rho::linalg::gemm_mode::overwrite);
    EXPECT_EQ(C(0, 0), 0.0F);
}

TEST(Gemm, ShapeMismatchStops) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const matrix<float> A{2, 3};
    const matrix<float> B{4, 2}; // must be 3 x 2
    matrix<float> C{2, 2};
    EXPECT_DEATH(rho::linalg::gemm(A.view(), B.view(), C.view()), "gemm: shapes do not match");
}

TEST(GemmTn, ShapeMismatchStops) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const matrix<float> At{3, 2};
    const matrix<float> B{4, 2}; // must have 3 rows
    matrix<float> C{2, 2};
    EXPECT_DEATH(rho::linalg::gemm_tn(At.view(), B.view(), C.view()), "gemm: shapes do not match");
}

TEST(GemmNt, BitIdenticalToSimpleLoop) {
    rho::core::xoshiro256pp rng{41U};
    for (const auto [m, k, n] : shapes) {
        SCOPED_TRACE(std::format("{}x{}x{}", m, k, n));
        const auto A = random_matrix<float>(m, k, rng);
        const auto Bt = random_matrix<float>(n, k, rng); // B^T
        const auto C0 = random_matrix<float>(m, n, rng);

        auto expected = copy_of(C0);
        auto actual = copy_of(C0);
        simple_gemm(A, transposed(Bt), expected);
        rho::linalg::gemm_nt(A.view(), Bt.view(), actual.view());
        expect_same(actual, expected, k);
    }
}

TEST(GemmNt, ShapeMismatchStops) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const matrix<float> A{2, 3};
    const matrix<float> Bt{2, 4}; // must have 3 columns
    matrix<float> C{2, 2};
    EXPECT_DEATH(rho::linalg::gemm_nt(A.view(), Bt.view(), C.view()), "gemm: shapes do not match");
}

// C = ReLU(A * B + bias) so the bias must be added once and the ReLU applied at
// the end only.
TEST(GemmBatch, BiasAndReluMatchSimpleLoop) {
    rho::core::xoshiro256pp rng{42U};
    for (const auto [m, k, n] : shapes) {
        SCOPED_TRACE(std::format("{}x{}x{}", m, k, n));
        const auto A = random_matrix<float>(m, k, rng);
        const auto B = random_matrix<float>(k, n, rng);
        const auto bias = random_matrix<float>(1, n, rng);
        for (const bool relu : {false, true}) {
            matrix<float> expected{m, n};
            for (std::size_t i = 0; i < m; ++i) {
                std::ranges::copy(bias.row(0), expected.row(i).begin());
            }
            simple_gemm(A, B, expected);
            if (relu) {
                for (std::size_t i = 0; i < m; ++i) {
                    for (float &v : expected.row(i)) {
                        v = v > 0.0F ? v : 0.0F;
                    }
                }
            }
            matrix<float> actual{m, n};
            actual.fill(std::numeric_limits<float>::quiet_NaN()); // overwrite: not read
            const std::array task{rho::linalg::gemm_task{.A = A.view(),
                                                         .B = B.view(),
                                                         .C = actual.view(),
                                                         .mode = rho::linalg::gemm_mode::overwrite,
                                                         .bias = bias.row(0),
                                                         .relu = relu}};
            rho::linalg::gemm_batch(task);
            expect_same(actual, expected, k);
        }
    }
}

// ReLU as relu() of rho.nn.activation: v > 0 ? v : 0, so NaN and negative values give 0.
// Also for k = 0, where C = ReLU(bias).
TEST(GemmBatch, ReluRule) {
    const matrix<float> A{1, 1, aligned_vector<float>{1.0F}};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const matrix<float> B{1, 3, aligned_vector<float>{nan, -2.0F, 3.0F}};
    matrix<float> C{1, 3};
    const std::array task{rho::linalg::gemm_task{.A = A.view(),
                                                 .B = B.view(),
                                                 .C = C.view(),
                                                 .mode = rho::linalg::gemm_mode::overwrite,
                                                 .relu = true}};
    rho::linalg::gemm_batch(task);
    EXPECT_EQ(C(0, 0), 0.0F);
    EXPECT_EQ(C(0, 1), 0.0F);
    EXPECT_EQ(C(0, 2), 3.0F);

    const matrix<float> A0{2, 0};
    const matrix<float> B0{0, 3};
    const std::array<float, 3> bias{-1.0F, 0.5F, nan};
    matrix<float> C0{2, 3};
    C0.fill(7.0F);
    const std::array task0{rho::linalg::gemm_task{.A = A0.view(),
                                                  .B = B0.view(),
                                                  .C = C0.view(),
                                                  .mode = rho::linalg::gemm_mode::overwrite,
                                                  .bias = bias,
                                                  .relu = true}};
    rho::linalg::gemm_batch(task0);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(C0(i, 0), 0.0F);
        EXPECT_EQ(C0(i, 1), 0.5F);
        EXPECT_EQ(C0(i, 2), 0.0F);
    }
}

// The backward pass of a dense layer: dW = x^T * dz and dx = dz * W^T in one parallel
// region. The result must be the same as two separate calls, for 1 to 4 threads.
TEST(GemmBatch, TwoTasksInOneRegion) {
    rho::core::xoshiro256pp rng{43U};
    const int saved = omp_get_max_threads();
    for (const auto [batch, n_in, n_out] : {Shape{64, 784, 256}, Shape{37, 20, 11}}) {
        const auto x = random_matrix<float>(batch, n_in, rng);
        const auto dz = random_matrix<float>(batch, n_out, rng);
        const auto W = random_matrix<float>(n_in, n_out, rng);
        matrix<float> dW_ref{n_in, n_out};
        matrix<float> dx_ref{batch, n_in};
        rho::linalg::gemm_tn(x.view(), dz.view(), dW_ref.view(), rho::linalg::gemm_mode::overwrite);
        rho::linalg::gemm_nt(dz.view(), W.view(), dx_ref.view(), rho::linalg::gemm_mode::overwrite);
        for (const int threads : {1, 2, 3, 4}) {
            omp_set_num_threads(threads);
            matrix<float> dW{n_in, n_out};
            matrix<float> dx{batch, n_in};
            const std::array tasks{
                rho::linalg::gemm_task{.A = x.view(),
                                       .B = dz.view(),
                                       .C = dW.view(),
                                       .A_transposed = true,
                                       .mode = rho::linalg::gemm_mode::overwrite},
                rho::linalg::gemm_task{.A = dz.view(),
                                       .B = W.view(),
                                       .C = dx.view(),
                                       .B_transposed = true,
                                       .mode = rho::linalg::gemm_mode::overwrite}};
            rho::linalg::gemm_batch(tasks);
            EXPECT_TRUE(bit_identical(dW, dW_ref)) << threads << " threads";
            EXPECT_TRUE(bit_identical(dx, dx_ref)) << threads << " threads";
        }
    }
    omp_set_num_threads(saved);
}

TEST(GemmBatch, BiasSizeMismatchStops) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const matrix<float> A{2, 3};
    const matrix<float> B{3, 4};
    matrix<float> C{2, 4};
    const std::array<float, 3> bias{}; // must have 4 values
    const std::array task{
        rho::linalg::gemm_task{.A = A.view(), .B = B.view(), .C = C.view(), .bias = bias}};
    EXPECT_DEATH(rho::linalg::gemm_batch(task), "gemm: shapes do not match");
}
