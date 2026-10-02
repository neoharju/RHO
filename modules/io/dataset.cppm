export module rho.io.dataset;

import std;

export import rho.io.idx;
export import rho.io.normalization;
export import rho.core.rng;

namespace fs = std::filesystem;

export namespace rho::io {
// MNIST specific
inline constexpr std::size_t image_rows{28};
inline constexpr std::size_t image_cols{28};
inline constexpr std::size_t pixels_per_image {image_rows * image_cols}; // 784
inline constexpr std::uint8_t num_classes{10};

class dataset {
// TODO:
// load data into aligned buffers
// check data
// normalize
// shuffle data
// split to training/test/validation
// batch data for efficiency
// shuffle data before every epoch, except validation
};

} // rho::io
