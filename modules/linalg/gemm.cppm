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

} // namespace rho::linalg
