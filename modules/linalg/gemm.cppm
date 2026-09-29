export module rho.linalg.gemm;

import std;
import rho.core.matrix;

export namespace rho::linalg {

// Reference GEMM
// C[m x n] = A[m x k] * B[k x n]
template <class T>
void gemm(rho::core::matrix_view<const T> A,
          rho::core::matrix_view<const T> B,
          rho::core::matrix_view<T> C) noexcept {

    const auto m = A.rows();
    const auto k = A.cols();
    const auto n = B.cols();

    for (std::size_t i{0}; i < m; ++i) {
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

// simple transpose for small matrices
// NB: note cache efficient for larger onces, just like GEMM above
template <class T>
void transpose(rho::core::matrix_view<const T> src, rho::core::matrix_view<T> dst) noexcept {
    for (std::size_t i = 0; i < src.rows(); ++i) {
        for (std::size_t j = 0; j < src.cols(); ++j) {
            dst(j, i) = src(i, j);
        }
    }
}

} // namespace rho::linalg
