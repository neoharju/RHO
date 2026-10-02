export module rho.io.normalization;

import std;

export namespace rho::io {
// We need to standardize (z-score) the image pixel values
// (x - mean / stddev) == x * scale + bias
struct normalization {
	float scale{1.0f};
	float bias{};

	[[nodiscard]] static constexpr normalization from_mean_stddev(double mean, double stddev) noexcept {
		const double inv_stddev = stddev > 0.0 ? 1.0 / stddev : 1.0;
		return {static_cast<float>(inv_stddev),			 // scale
				static_cast<float>(-mean * inv_stddev)}; // bias
	}

	[[nodiscard]] constexpr float apply(std::uint8_t x) const noexcept {
		return static_cast<float>(x) * scale + bias;
	}
};

} // namespace rho::io
