export module rho.nn.activation;

import std;

export import rho.core.matrix;

export namespace rho::nn {

	// ReLU(z):  return 0 if lower than 0 else z
	template<class T>
		[[nodiscard]] inline constexpr T relu(T z) noexcept {
			return z > T{0} ? z : T{0};
		}

	template <class T>
		inline void relu_forward(rho::core::matrix_view<const T> z, rho::core::matrix_view<T> a) noexcept {
			for (std::size_t i = 0; i < z.rows(); ++i) {
				const auto zi = z.row(i);
				const auto ai = a.row(i);
				for(std::size_t j = 0; j < z.cols(); ++j){
					ai[j] = relu(zi[j]);
				}
			}
		}


	template <class T>
		inline void relu_backward(rho::core::matrix_view<const T> z, rho::core::matrix_view<const T> da, rho::core::matrix_view<T> dz) noexcept {
			for (std::size_t i = 0; i < z.rows(); ++i) {
				const auto zi = z.row(i);
				const auto dai = da.row(i);
				const auto dzi = dz.row(i);
				for(std::size_t j = 0; j < z.cols(); ++j){
					dzi[i] = zi[j] > T{0} ? dai[j] : T{0};
				}
			}
		}

} // namespace rho::nn
