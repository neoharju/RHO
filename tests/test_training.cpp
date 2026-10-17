#include <gtest/gtest.h>

import std;
import rho.optim;

// Tests recommended to be done by S. Bjorkqvist
// [0]: https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

// 1. the network overfits a tiny dataset

using rho::core::matrix;

namespace {

struct TestData {
    matrix<float> x;
    std::vector<std::uint8_t> y;

    TestData(std::size_t samples,
             std::size_t dim,
             std::size_t classes,
             double spread,
             std::uint64_t seed)
        : x{samples, dim}, y(samples) {
        rho::core::xoshiro256pp rng{seed};
        matrix<double> centre{classes, dim};

        // choose the centre, and do one row per class with dim numbers per row
        for (std::size_t cl = 0; cl < classes; ++cl) {
            // ofr each of its dim coords
            for (double &v : centre.row(cl)) {
                v = rng.normal(); // random num from 0-10
            }
        }

        // make the samples
        for (std::size_t sample = 0; sample < samples; ++sample) {
            y[sample] = static_cast<std::uint8_t>(sample % classes); // labels
            const auto row = x.row(sample);

            for (std::size_t coord = 0; coord < dim; ++coord) {
                // coordinate j of its class' centre + noise (rng.normal)
                // spread is to scale the centre for classes;
                // Spread=0 is not learnable since everything is mixed together
                // i.e. labels will not have any meaning
                row[coord] = static_cast<float>(spread * centre(y[sample], coord) + rng.normal());
            }
        }
    }
    [[nodiscard]] rho::core::matrix_view<const float> rows(std::size_t first,
                                                           std::size_t count) const {
        return rho::core::matrix_view<const float>{
            x.data() + first * x.stride(), count, x.cols(), x.stride()};
    }

    [[nodiscard]] std::span<const std::uint8_t> labels(std::size_t first, std::size_t count) const {
        return std::span<const std::uint8_t>{y}.subspan(first, count);
    }
};

double accuracy(rho::core::matrix_view<const float> logits, std::span<const std::uint8_t> labels) {
    std::size_t correct = 0;
    for (std::size_t r = 0; r < logits.rows(); ++r) {
        const auto row = logits.row(r);
        correct +=
            static_cast<std::size_t>(std::ranges::max_element(row) - row.begin() == labels[r]);
    }
    return static_cast<double>(correct) / static_cast<double>(logits.rows());
}

} // namespace

// ref [0] from Bjorkqvist:
// test 1:
//   - Data points being shuffled during the forward pass. This may happen e.g. if the batch
//   creation is invalid, or if some data grouping or result gathering is done incorrectly.
//   - Data being mangled during the forward pass. One example is if you use a masking function
//   incorrectly.
//   - The network not being expressive enough to solve the task at hand. This can happen for
//   instance if you forget to add non-linearities and stack just linear layers on top of each
//   other. Then the entire network is just a linear function, and it cannot overfit if the target
//   function or decision boundary is non-linear.
//   - Invalid weight initialization. If you for some reason need to write your own weight
//   initialization you might end up with a network that’s not able to learn at all.
//
TEST(Training, OverfitsTinyDataset) {
    constexpr std::size_t samples{48};

    //(std::size_t samples, std::size_t dim, std::size_t classes, double spread, std::uint64_t seed)
    const TestData data{samples, 8, 4, 0.0, 1};
    const std::array<std::size_t, 3> sizes{8, 64, 4};

    rho::nn::mlp<float> net{sizes, samples, 1};
    rho::optim::sgd<float> opt{0.5f};

    float loss = net.loss(data.rows(0, samples), data.labels(0, samples));
    for (std::size_t step = 0; step < 5000 && loss > 0.01f; ++step) {
        loss = net.forward_backward(data.rows(0, samples), data.labels(0, samples));
        opt.step(net.parameters());
        net.after_update();
    }
    loss = net.loss(data.rows(0, samples), data.labels(0, samples));

    EXPECT_LT(loss, 0.01f);
    EXPECT_EQ(accuracy(net.logits(), data.labels(0, samples)), 1.0);
}
