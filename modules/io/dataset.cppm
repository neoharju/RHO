/// @file dataset.cppm
/// @brief MNIST dataset in memory, normalization statistics, train/validation split,
///        and the mini-batch maker (batcher).
export module rho.io.dataset;

import std;

export import rho.core.matrix;
export import rho.core.rng;
export import rho.io.idx;
export import rho.io.normalization;

namespace fs = std::filesystem;

export namespace rho::io {

// MNIST specific
inline constexpr std::size_t image_rows{28};                            ///< Image height in pixels.
inline constexpr std::size_t image_cols{28};                            ///< Image width in pixels.
inline constexpr std::size_t pixels_per_image{image_rows * image_cols}; ///< 784 pixels.
inline constexpr std::uint8_t num_classes{10};                          ///< Digits 0 to 9.

/// @brief MNIST images and labels in memory, as raw bytes.
///
/// Image i is 784 contiguous bytes (row-major, 0 = white, 255 = black).
/// The only way to make a dataset is load(). So each dataset object is valid:
/// it has N images of 28 x 28 pixels, N labels, and each label is 0 to 9.
class dataset {
  public:
    /// @brief Load and check an MNIST images file and its labels file.
    /// @param images_file IDX file with shape [N, 28, 28].
    /// @param labels_file IDX file with shape [N] and values 0 to 9.
    /// @return The dataset, or an Error: any error of read_idx_u8(), BadShapeError,
    ///         CountMismatchError or BadLabelError.
    [[nodiscard]]
    static std::expected<dataset, Error> load(const fs::path &images_file,
                                              const fs::path &labels_file);

    /// @return Number of samples N.
    [[nodiscard]]
    inline std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(labels_.size());
    }

    /// @brief The 784 pixels of image i.
    /// @pre i < size(). There is no bounds check.
    [[nodiscard]]
    inline std::span<const std::uint8_t, pixels_per_image> image(std::uint32_t i) const noexcept {
        return pixels_.span().subspan(std::size_t{i} * pixels_per_image).first<pixels_per_image>();
    }

    /// @brief The label (0 to 9) of sample i.
    /// @pre i < size(). There is no bounds check.
    [[nodiscard]]
    inline std::uint8_t label(std::uint32_t i) const noexcept {
        return labels_.span()[i];
    }

  private:
    // The constructor is private: the only way to make a dataset is
    // dataset::load(images_file, labels_file), which checks the data first.
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
        return std::unexpected(BadShapeError{labels_file, labels->dims, labels->rank, "[N]"});
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

/// @brief Compute the z-score normalization from the pixels of some samples.
///
/// Use only the training ids. Then no information from the validation or test
/// data goes into the model.
///
/// The sums are exact integers: one image has at most 784 * 255^2 < 2^32 for the
/// sum of squares, and the totals use 64 bits. The variance is E[x^2] - E[x]^2 in double.
///
/// @param ds The dataset.
/// @param ids Sample ids to use. Each id must be < ds.size().
/// @return The normalization. If ids is empty, the identity normalization (scale 1, bias 0).
/// @pre All ids < ds.size(). If not, the program stops.
[[nodiscard]]
normalization compute_normalization(const dataset &ds, std::span<const std::uint32_t> ids) {
    if (ids.empty()) {
        return {};
    }

    rho::core::require(std::ranges::max(ids) < ds.size(),
                       "compute_normalization: id out of range for this dataset");

    std::uint64_t sum{};
    std::uint64_t sum_sq{};

    for (const auto id : ids) {
        std::uint32_t s{};
        std::uint32_t sq{};

        for (const auto v : ds.image(id)) {
            s += v;
            sq += v * v;
        }

        sum += s;
        sum_sq += sq;
    }

    const double n = static_cast<double>(ids.size()) * static_cast<double>(pixels_per_image);
    const double mean = static_cast<double>(sum) / n;

    double var = static_cast<double>(sum_sq) / n - mean * mean;
    // make sure rounding does not make var go below zero
    var = var > 0.0 ? var : 0.0;

    return normalization::from_mean_stddev(mean, std::sqrt(var));
}
// -----------------------------------------------------------------------------

/// Sample ids of the training part and the validation part. Both are sorted.
struct split_ids {
    std::vector<std::uint32_t> train;
    std::vector<std::uint32_t> validation;
};

/// @return The ids 0, 1, ..., ds.size() - 1.
[[nodiscard]] std::vector<std::uint32_t> all_ids(const dataset &ds) {
    std::vector<std::uint32_t> ids(ds.size());
    std::iota(ids.begin(), ids.end(), 0U);
    return ids;
}

/// @brief Split the ids 0 to n-1 randomly into a training part and a validation part.
///
/// The function shuffles the ids, takes the first validation_count ids for validation,
/// and sorts both parts. Sorted ids give sequential memory access in compute_normalization().
/// The batcher shuffles the training ids again in each epoch.
///
/// @param n Number of samples.
/// @param validation_count Number of samples for validation.
/// @param rng The generator. The same seed gives the same split.
/// @return The two parts. They do not overlap, and together they hold all ids.
/// @throws std::invalid_argument if validation_count > n.
[[nodiscard]] split_ids train_validation_split(std::uint32_t n,
                                               std::uint32_t validation_count,
                                               rho::core::xoshiro256pp &rng) {
    if (validation_count > n) {
        throw std::invalid_argument("train_validation_split: validation_count > n");
    }

    std::vector<std::uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0U);
    rho::core::shuffle(ids, rng);
    // cut the validation part
    const auto cut = ids.begin() + static_cast<std::ptrdiff_t>(validation_count);
    split_ids out{{cut, ids.end()}, {ids.begin(), cut}};
    std::ranges::sort(out.train);
    std::ranges::sort(out.validation);
    return out;
}

/// @brief One mini-batch: normalized pixels and labels.
///
/// The views point into the buffers of the batcher. They are valid only until
/// the batcher makes the next batch. Copy the data if you need it longer.
struct batch {
    rho::core::matrix_view<const float> x; ///< samples x 784, normalized pixels.
    std::span<const std::uint8_t> y;       ///< samples labels, 0 to 9.
};

/// The order of the samples in an epoch.
enum class sample_order : std::uint8_t {
    sequential, ///< The order of the ids (use it for evaluation).
    shuffled    ///< A new random order in each epoch (use it for training).
};

/// @brief Makes mini-batches from a set of sample ids of one dataset.
///
/// For each batch, the batcher copies the uint8 pixels of the samples into one
/// float buffer and normalizes them. The buffer is allocated one time, in the
/// constructor, so an epoch does not allocate memory.
///
/// The last batch of an epoch can have fewer rows than batch_size.
///
/// The batcher keeps a pointer to the dataset. The dataset must live longer
/// than the batcher.
class batcher {
  public:
    /// @brief Make a batcher.
    /// @param ds The dataset. It must live longer than the batcher.
    /// @param ids The sample ids to use. Each id must be < ds.size().
    /// @param norm The normalization to apply to each pixel.
    /// @param batch_size Maximum number of samples in one batch. Must be > 0.
    /// @param order sequential or shuffled.
    /// @param seed Seed for the shuffle (only used with sample_order::shuffled).
    /// @throws std::invalid_argument if batch_size is 0.
    /// @throws std::out_of_range if an id is not in the dataset.
    batcher(const dataset &ds,
            std::vector<std::uint32_t> ids,
            normalization norm,
            std::size_t batch_size,
            sample_order order,
            std::uint64_t seed = 0U)
        : ds_{&ds}, ids_{std::move(ids)}, norm_{norm}, batch_size_{batch_size}, order_{order},
          rng_{seed}, x_{batch_size == 0U ? 1U : batch_size, pixels_per_image}, y_(batch_size) {
        if (batch_size == 0U) {
            throw std::invalid_argument("batcher: batch_size must be larger than 0");
        }
        if (!ids_.empty() && std::ranges::max(ids_) >= ds.size()) {
            throw std::out_of_range("batcher: a sample id is not in the dataset");
        }
    }

    /// A temporary dataset is not allowed, because the batcher would keep a dangling
    /// pointer. For example: `batcher b(load_mnist(), ids, norm, 64, sample_order::shuffled);`
    batcher(const dataset &&,
            std::vector<std::uint32_t>,
            normalization,
            std::size_t,
            sample_order,
            std::uint64_t = 0U) = delete;

    /// @brief All batches of one epoch, as a lazy sequence (C++23 coroutine).
    ///
    /// With sample_order::shuffled, the ids get a new random order first.
    /// Each batch overwrites the previous one. Do not keep a batch after the
    /// next iteration, and do not run two epochs of one batcher at the same time.
    ///
    /// @return A generator. Use it in a range-for loop: `for (const batch &b : data.epoch())`.
    [[nodiscard]] std::generator<const batch &> epoch() {

        if (order_ == sample_order::shuffled) {
            rho::core::shuffle(ids_, rng_);
        }

        for (std::size_t first = 0; first < ids_.size(); first += batch_size_) {
            const std::size_t count = std::min(batch_size_, ids_.size() - first);
            const batch b = gather(std::span{ids_}.subspan(first, count));
            co_yield b;
        }
    }

    /// @return The maximum number of samples in one batch.
    [[nodiscard]] std::size_t batch_size() const noexcept {
        return batch_size_;
    }

  private:
    /// Copy and normalize the samples `ids` into rows 0 to ids.size() - 1 of x_,
    /// and their labels into y_.
    /// @pre ids.size() <= batch_size_.
    [[nodiscard]] batch gather(std::span<const std::uint32_t> ids) noexcept {
        for (std::size_t r = 0; r < ids.size(); ++r) {
            const auto pixels = ds_->image(ids[r]);
            const auto row = x_.row(r);
            for (std::size_t p = 0; p < pixels_per_image; ++p) {
                row[p] = norm_.apply(pixels[p]);
            }
            y_[r] = ds_->label(ids[r]);
        }
        return {x_.view().row_range(0, ids.size()), std::span{y_}.first(ids.size())};
    }

    const dataset *ds_;              // not owned; must outlive the batcher
    std::vector<std::uint32_t> ids_; // samples of this batcher
    normalization norm_;
    std::size_t batch_size_;
    sample_order order_;
    rho::core::xoshiro256pp rng_;    // for the shuffle in epoch()
    rho::core::matrix<float> x_;     // batch_size x 784
    std::vector<std::uint8_t> y_;    // batch_size labels
};

} // namespace rho::io
