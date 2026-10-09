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
inline constexpr std::size_t pixels_per_image{image_rows * image_cols}; // 784
inline constexpr std::uint8_t num_classes{10};

class dataset {
  public:
    [[nodiscard]]
    static std::expected<dataset, Error> load(const fs::path &images_file,
                                              const fs::path &labels_file);
    [[nodiscard]]
    inline std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(labels_.size());
    }

    [[nodiscard]]
    inline std::span<const std::uint8_t, pixels_per_image> image(std::uint32_t i) const noexcept {
        return pixels_.span().subspan(std::size_t{i} * pixels_per_image).first<pixels_per_image>();
    }

    [[nodiscard]]
    inline std::uint8_t label(std::uint32_t i) const noexcept {
        return labels_span()[i];
    }

  private:
    // This might surprise the reviewer. The constructor is here because
    // the intended construction path is via dataset::load(images_file, labels_file);
    dataset(rho::core::aligned_buffer<std::uint8_t> pixels,
            rho::core::aligned_buffer<std::uint8_t> labels) noexcept
        : pixels_{std::move(pixels)}, labels_{std::move(labels)} {}

    rho::core::aligned_buffer<std::uint8_t> pixels_; // size() * 28 * 28
    rho::core::aligned_buffer<std::uint8_t> labels_; // size()
};

std::expected<dataset, Error> dataset::load(const fs::path &images_file,
                                            const fs::path &labels_file) {
    auto images = read_idx_u8(images_file);
    if (!images) {
        return std::unexpected(std::move(images).error());
    }

    if (images->rank != 3U || images->dims[1] != image_rows || images->dims[2] != image_cols) {
        return std::unexpected(
            BadShapeError{images_file, images->dims, images->rank, "[N, 28, 28]"});
    }

    auto labels = read_idx_u8(labels_file);
    if (!labels) {
        return std::unexpected(std::move(labels).error());
    }

    if (labels->rank != 1U) {
        return std::unexpected(BadShapeError{labels_file, labels->dims, labels->rank, "[N"});
    }

    if (labels->dims[0] != images->dims[0]) {
        return std::unexpected(
            CountMismatchError{images_file, labels_file, images->dims[0], labels->dims[0]});
    }

    const auto label_bytes = labels->data.span();
    const auto bad =
        std::ranges::find_if(label_bytes, [](std::uint8_t v) { return v >= num_classes; });
    if (bad != label_bytes.end()) {
        return std::unexpected(
            BadLabelError{labels_file, static_cast<std::size_t>(bad - label_bytes.begin()), *bad});
    }

    return dataset{std::move(images->data), std::move(labels->data)};
}

// TODO:
// normalize
// shuffle data
// split to training/test/validation
// batch data for efficiency
// shuffle data before every epoch, except validation

} // namespace rho::io
