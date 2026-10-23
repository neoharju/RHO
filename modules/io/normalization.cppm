/// @file normalization.cppm
/// @brief Z-score normalization of pixel values.
export module rho.io.normalization;

import std;

export namespace rho::io {

/// @brief Z-score normalization: x_norm = (x - mean) / stddev.
///
/// We store it as one multiply and one add: x_norm = x * scale + bias,
/// with scale = 1 / stddev and bias = -mean / stddev.
/// The compiler can make this one FMA instruction.
struct normalization {
    float scale{1.0f}; ///< 1 / stddev.
    float bias{};      ///< -mean / stddev.

    /// @brief Make a normalization from the mean and the standard deviation.
    /// @param mean Mean of the pixel values.
    /// @param stddev Standard deviation of the pixel values. If it is 0 (all pixels
    ///               are equal), scale is 1 and only the mean is subtracted.
    [[nodiscard]] static constexpr normalization from_mean_stddev(double mean,
                                                                  double stddev) noexcept {
        const double inv_stddev = stddev > 0.0 ? 1.0 / stddev : 1.0;
        return {static_cast<float>(inv_stddev),          // scale
                static_cast<float>(-mean * inv_stddev)}; // bias
    }

    /// @brief Normalize one pixel value.
    /// @param x Raw pixel value, 0 to 255.
    /// @return x * scale + bias.
    [[nodiscard]] constexpr float apply(std::uint8_t x) const noexcept {
        return static_cast<float>(x) * scale + bias;
    }
};

} // namespace rho::io
