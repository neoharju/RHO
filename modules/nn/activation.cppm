/// @file activation.cppm
/// @brief ReLU activation: forward pass and backward pass.
export module rho.nn.activation;

import std;

export import rho.core.matrix;

export namespace rho::nn {

/// @brief ReLU(z) = max(z, 0).
/// @return z if z > 0, else 0. For z = -0.0 the result is +0.0.
template <class T>
[[nodiscard]] inline constexpr T relu(T z) noexcept {
    return z > T{0} ? z : T{0};
}

/// @brief Forward pass: a = ReLU(z), element by element.
/// @param z Input (pre-activation), B x n.
/// @param a Output, B x n. It can be the same memory as z.
/// @pre a has the same shape as z.
template <class T>
inline void relu_forward(rho::core::matrix_view<const T> z, rho::core::matrix_view<T> a) noexcept {
    for (std::size_t i = 0; i < z.rows(); ++i) {
        const auto zi = z.row(i);
        const auto ai = a.row(i);
        for (std::size_t j = 0; j < z.cols(); ++j) {
            ai[j] = relu(zi[j]);
        }
    }
}

/// @brief Backward pass: dz = da * ReLU'(z), element by element.
///
/// ReLU'(z) is 1 for z > 0 and 0 for z < 0. At z = 0 any value in [0, 1] is a
/// valid subgradient. We use 0.
///
/// @param z The input of the forward pass, B x n.
/// @param da Gradient of the loss with respect to the output a, B x n.
/// @param dz Output: gradient of the loss with respect to z, B x n.
/// @pre All three have the same shape.
template <class T>
inline void relu_backward(rho::core::matrix_view<const T> z,
                          rho::core::matrix_view<const T> da,
                          rho::core::matrix_view<T> dz) noexcept {
    for (std::size_t i = 0; i < z.rows(); ++i) {
        const auto zi = z.row(i);
        const auto dai = da.row(i);
        const auto dzi = dz.row(i);
        for (std::size_t j = 0; j < z.cols(); ++j) {
            dzi[j] = zi[j] > T{0} ? dai[j] : T{0};
        }
    }
}

} // namespace rho::nn
