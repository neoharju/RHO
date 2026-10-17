#include <gtest/gtest.h>

import std;
import rho.optim;

// Tests recommended to be done by S. Bjorkqvist
// [0]: https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

// 1. the network overfits a tiny dataset
// 2. gradients are non-zero and the loss decreases

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
			return x.view().row_range(first, count);
		}

		[[nodiscard]] std::span<const std::uint8_t> labels(std::size_t first, std::size_t count) const {
			return std::span<const std::uint8_t>{y}.subspan(first, count);
		}
	};

} // namespace

// ref [0] from Bjorkqvist:
//
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
	EXPECT_EQ(rho::nn::count_correct<float>(net.logits(), data.labels(0, samples)), samples);
}

// test2:
// - run training for some batches and verify that the gradients are non-zero, and that the loss decreases during training.
TEST(Training, GradientsNonZero_LossDecreases) {
	constexpr std::size_t samples{64};
	const TestData data{samples, 8, 4, 3.0, 2};
	const std::array<std::size_t, 3> sizes{8, 16, 4};

	rho::nn::mlp<float> net{sizes, samples, 2};
	rho::optim::sgd<float> opt{0.1f};

	const auto x = data.rows(0, samples);
	const auto y = data.labels(0, samples);

	const float before_loss = net.forward_backward(x, y);

	// sum of squares: 0 if grads are zero
	for (const auto &p : net.parameters()) {
		float sum_sq{0.0f};

		for (std::size_t r = 0; r < p.grad.rows(); ++r) {
			for (const float g : p.grad.row(r)){
				sum_sq += g * g;
			}
		}
		EXPECT_TRUE(std::isfinite(sum_sq) && sum_sq > 0.0f);
	}


	opt.step(net.parameters());
	net.after_update();

	const float loss_after = net.loss(x, y);

	EXPECT_LT(loss_after, before_loss);
}
