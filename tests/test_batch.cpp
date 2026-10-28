#include <gtest/gtest.h>

import std;
import rho.train;

namespace fs = std::filesystem;
using rho::io::batcher;
using rho::io::sample_order;

namespace {

[[nodiscard]] constexpr std::uint8_t label_of(std::uint32_t i) noexcept {
    return static_cast<std::uint8_t>((i / 4U) % 10U);
}

// Writes a tiny MNIST-shaped IDX image and label file pair to a temporary directory.
// Image i has pixels 0..699 = i (mod 256) and label (i / 4) % 10.
// Pixel 700 + label is 255, so a network can learn the label.
class TinyIdx {
  public:
    explicit TinyIdx(std::uint32_t n)
        : dir_{fs::temp_directory_path()
               / std::format("rho_test_{}_{}",
                             ::testing::UnitTest::GetInstance()->current_test_info()->name(),
                             n)} {
        fs::create_directories(dir_);
        write(
            images(),
            {0, 0, 8, 3},
            {n, 28, 28},
            [](std::uint32_t i) {
                std::vector<std::uint8_t> pixels(784, static_cast<std::uint8_t>(i % 256U));
                std::fill(pixels.begin() + 700, pixels.end(), std::uint8_t{0});
                pixels[700U + label_of(i)] = 255U;
                return pixels;
            },
            n);
        write(
            labels(),
            {0, 0, 8, 1},
            {n},
            [](std::uint32_t i) { return std::vector<std::uint8_t>{label_of(i)}; },
            n);
    }
    ~TinyIdx() {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    TinyIdx(const TinyIdx &) = delete;
    TinyIdx &operator=(const TinyIdx &) = delete;

    [[nodiscard]] fs::path images() const {
        return dir_ / "images";
    }
    [[nodiscard]] fs::path labels() const {
        return dir_ / "labels";
    }

    [[nodiscard]] rho::io::dataset load() const {
        auto ds = rho::io::dataset::load(images(), labels());
        if (!ds) {
            throw std::runtime_error("TinyIdx: load failed");
        }
        return std::move(*ds);
    }

  private:
    template <class Sample>
    static void write(const fs::path &path,
                      std::array<std::uint8_t, 4> magic,
                      std::initializer_list<std::uint32_t> dims,
                      Sample sample,
                      std::uint32_t n) {
        std::ofstream out{path, std::ios::binary};
        out.write(reinterpret_cast<const char *>(magic.data()), 4);
        for (const std::uint32_t d : dims) {
            const std::array<std::uint8_t, 4> be{static_cast<std::uint8_t>(d >> 24U),
                                                 static_cast<std::uint8_t>(d >> 16U),
                                                 static_cast<std::uint8_t>(d >> 8U),
                                                 static_cast<std::uint8_t>(d)};
            out.write(reinterpret_cast<const char *>(be.data()), 4);
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto bytes = sample(i);
            out.write(reinterpret_cast<const char *>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        }
    }

    fs::path dir_;
};

// Image i has all pixels = i, so the first pixel tells the sample id (for n <= 256)
[[nodiscard]] std::uint32_t
sample_id(const rho::io::batch &b, std::size_t row, const rho::io::normalization &norm) {
    return static_cast<std::uint32_t>(std::lround((b.x(row, 0) - norm.bias) / norm.scale));
}

} // namespace

TEST(Split, DisjointAndComplete) {
    rho::core::xoshiro256pp rng{7};
    const auto s = rho::io::train_validation_split(100, 30, rng);

    EXPECT_EQ(s.train.size(), 70U);
    EXPECT_EQ(s.validation.size(), 30U);

    std::vector<std::uint32_t> all{s.train};
    all.append_range(s.validation);
    std::ranges::sort(all);
    std::vector<std::uint32_t> expected(100);
    std::iota(expected.begin(), expected.end(), 0U);
    EXPECT_EQ(all, expected); // every id once, so the parts are disjoint
}

TEST(Split, TooLargeValidationThrows) {
    rho::core::xoshiro256pp rng{7};
    EXPECT_THROW(std::ignore = rho::io::train_validation_split(10, 11, rng), std::invalid_argument);
}

TEST(Batcher, SequentialKeepsOrder_LastBatchIsPartial) {
    const TinyIdx files{10};
    const auto ds = files.load();
    const rho::io::normalization norm{}; // identity: pixel value as float

    batcher b{ds, rho::io::all_ids(ds), norm, 4, sample_order::sequential};

    std::vector<std::size_t> rows;
    std::vector<std::uint32_t> seen;
    for (const auto &batch : b.epoch()) {
        rows.push_back(batch.x.rows());
        ASSERT_EQ(batch.x.cols(), 784U);
        ASSERT_EQ(batch.y.size(), batch.x.rows());
        for (std::size_t r = 0; r < batch.x.rows(); ++r) {
            const auto id = sample_id(batch, r, norm);
            seen.push_back(id);
            EXPECT_EQ(batch.y[r], label_of(id)); // label stays with its image
        }
    }
    EXPECT_EQ(rows, (std::vector<std::size_t>{4, 4, 2}));
    EXPECT_EQ(seen, (std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(Batcher, ShuffledEpochsArePermutations) {
    const TinyIdx files{50};
    const auto ds = files.load();
    const rho::io::normalization norm{};

    batcher b{ds, rho::io::all_ids(ds), norm, 8, sample_order::shuffled, 3};

    std::vector<std::vector<std::uint32_t>> epochs;
    for (int e = 0; e < 2; ++e) {
        auto &order = epochs.emplace_back();
        for (const auto &batch : b.epoch()) {
            for (std::size_t r = 0; r < batch.x.rows(); ++r) {
                order.push_back(sample_id(batch, r, norm));
                EXPECT_EQ(batch.y[r], label_of(order.back()));
            }
        }
        auto sorted = order;
        std::ranges::sort(sorted);
        std::vector<std::uint32_t> expected(50);
        std::iota(expected.begin(), expected.end(), 0U);
        EXPECT_EQ(sorted, expected); // each sample exactly once per epoch
    }
    EXPECT_NE(epochs[0], epochs[1]); // a new order each epoch
}

TEST(Batcher, SameSeedSameOrder) {
    const TinyIdx files{40};
    const auto ds = files.load();
    const rho::io::normalization norm{};
    batcher a{ds, rho::io::all_ids(ds), norm, 8, sample_order::shuffled, 11};
    batcher b{ds, rho::io::all_ids(ds), norm, 8, sample_order::shuffled, 11};

    // the sample ids of one epoch, in batch order
    const auto order = [&](batcher &x) {
        std::vector<std::uint32_t> out;
        for (const auto &batch : x.epoch()) {
            for (std::size_t r = 0; r < batch.x.rows(); ++r) {
                out.push_back(sample_id(batch, r, norm));
            }
        }
        return out;
    };
    for (int e = 0; e < 2; ++e) {
        EXPECT_EQ(order(a), order(b));
    }
}

TEST(Batcher, NormalizationIsApplied) {
    const TinyIdx files{3};
    const auto ds = files.load();
    const auto norm = rho::io::normalization::from_mean_stddev(1.0, 2.0);

    batcher b{ds, rho::io::all_ids(ds), norm, 3, sample_order::sequential};
    for (const auto &batch : b.epoch()) {
        for (std::size_t r = 0; r < 3; ++r) {
            for (const float v : batch.x.row(r).first(700)) {
                EXPECT_FLOAT_EQ(v, (static_cast<float>(r) - 1.0f) / 2.0f);
            }
            EXPECT_FLOAT_EQ(batch.x(r, 700 + label_of(static_cast<std::uint32_t>(r))), 127.0f);
        }
    }
}

TEST(Batcher, BadArgumentsThrow) {
    const TinyIdx files{5};
    const auto ds = files.load();
    EXPECT_THROW((batcher{ds, rho::io::all_ids(ds), {}, 0, sample_order::sequential}),
                 std::invalid_argument);
    EXPECT_THROW((batcher{ds, {0, 5}, {}, 2, sample_order::sequential}), std::out_of_range);
}

TEST(Batcher, EmptyIdsGiveNoBatches) {
    const TinyIdx files{5};
    const auto ds = files.load();
    batcher b{ds, {}, {}, 2, sample_order::shuffled};
    for (const auto &_ : b.epoch()) {
        ADD_FAILURE() << "an empty batcher gave a batch";
    }
}

TEST(Batcher, LearnsTinyDataset_ReportsEachEpoch) {
    // 40 samples, 10 classes. The marker pixel makes the labels learnable.
    const TinyIdx files{40};
    const auto ds = files.load();
    const auto norm = rho::io::compute_normalization(ds, rho::io::all_ids(ds));
    batcher train{ds, rho::io::all_ids(ds), norm, 8, sample_order::shuffled, 1};
    batcher validation{ds, rho::io::all_ids(ds), norm, 8, sample_order::sequential};

    const std::array<std::size_t, 3> sizes{784, 64, 10};
    rho::nn::mlp<float> net{sizes, 8, 2};
    rho::optim::sgd<float> opt{0.05f};

    std::vector<rho::train::epoch_report> reports;
    rho::train::fit(net,
                    opt,
                    train,
                    validation,
                    {.epochs = 30, .lr_decay = 0.99f},
                    [&](const rho::train::epoch_report &r) { reports.push_back(r); });

    ASSERT_EQ(reports.size(), 30U);
    EXPECT_FLOAT_EQ(reports[0].learning_rate, 0.05f); // starts with the rate of opt
    EXPECT_LT(reports.back().train_loss, reports.front().train_loss);
    EXPECT_EQ(reports.back().validation.accuracy(), 1.0);
    EXPECT_FLOAT_EQ(reports[1].learning_rate, 0.05f * 0.99f);
}
