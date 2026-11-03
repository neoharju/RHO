#include <gtest/gtest.h>

import std;
import rho.optim;

// Gradient check: compare the gradients from backpropagation with numeric gradients from
// central differences. For each parameter w of the network:
//
//     dL/dw = (L(w + h) - L(w - h)) / (2 * h) + O(h^2)
//
// Source: Stanford CS231n, https://cs231n.github.io/neural-networks-3/#gradcheck

using rho::core::matrix;

namespace {

	// Input 13, hidden layers 17 and 16, and 10 classes. These sizes are not multiples of the
	// 6 x 16 tile of the GEMM, so that the edge kernels also run.
	constexpr std::array<std::size_t, 4> sizes{13, 17, 16, 10};
	constexpr std::size_t batch = 7; // smaller than max_batch (8)

	matrix<float> random_input(std::size_t rows, std::size_t cols, std::uint64_t seed) {
		matrix<float> x{rows, cols};
		rho::core::xoshiro256pp rng{seed};
		for (std::size_t r = 0; r < rows; ++r) {
			for (float &v : x.row(r)) {
				v = static_cast<float>(rng.normal());
			}
		}
		return x;
	}

	/// @brief Mean cross-entropy loss of the network, in double precision with simple loops.
	///
	/// It does the same steps as mlp::loss(): z = a * W + b in each layer, ReLU between the
	/// layers, and softmax cross-entropy after the last layer.
	/// @param params W and b of each layer, in the order of mlp::parameters().
	/// @param x Input batch, one sample in each row.
	/// @param labels One label for each sample.
	/// @return The mean loss of the batch.
	double reference_loss(const std::vector<matrix<double>> &params,
			const matrix<float> &x,
			std::span<const std::uint8_t> labels) {
		double loss = 0.0;
		for (std::size_t s = 0; s < x.rows(); ++s) {
			std::vector<double> a(x.row(s).begin(), x.row(s).end());
			for (std::size_t k = 0; k < params.size(); k += 2) {
				const matrix<double> &W = params[k];
				std::vector<double> z(params[k + 1].row(0).begin(), params[k + 1].row(0).end());
				for (std::size_t i = 0; i < W.rows(); ++i) {
					for (std::size_t j = 0; j < W.cols(); ++j) {
						z[j] += a[i] * W(i, j);
					}
				}
				if (k + 2 < params.size()) { // not the last layer
					for (double &v : z) {
						v = v > 0.0 ? v : 0.0;
					}
				}
				a = std::move(z);
			}
			// a holds the logits: loss = log(sum_j exp(a_j)) - a_label
			const double max_a = std::ranges::max(a);
			double sum = 0.0;
			for (const double v : a) {
				sum += std::exp(v - max_a);
			}
			loss += max_a + std::log(sum) - a[labels[s]];
		}
		return loss / static_cast<double>(x.rows());
	}

} // namespace

TEST(GradientCheck, BackpropMatchesCentralDifferences) {
	rho::nn::mlp<float> net{sizes, batch + 1U, 1U};
	const auto x = random_input(batch, sizes[0], 101U);
	std::vector<std::uint8_t> y(batch);
	for (std::size_t i = 0; i < batch; ++i) {
		y[i] = static_cast<std::uint8_t>(i % 10U);
	}

	// Do one SGD step first, so that the biases are not 0.
	rho::optim::sgd<float> opt{0.1f};
	std::ignore = net.forward_backward(x.view(), y);
	opt.step(net.parameters());

	// The analytic gradients. The reference loss does not change them.
	const float loss = net.forward_backward(x.view(), y);

	std::vector<matrix<double>> params;
	for (const auto &p : net.parameters()) {
		auto &copy = params.emplace_back(p.value.rows(), p.value.cols());
		for (std::size_t r = 0; r < p.value.rows(); ++r) {
			std::ranges::copy(p.value.row(r), copy.row(r).begin());
		}
	}

	// The float forward pass and the reference must give the same loss.
	ASSERT_NEAR(loss, reference_loss(params, x, y), 1e-5);

	constexpr double h = 1e-6;
	std::size_t checked = 0;
	for (std::size_t k = 0; k < params.size(); ++k) {
		const auto analytic = net.parameters()[k].grad;
		for (std::size_t r = 0; r < analytic.rows(); ++r) {
			for (std::size_t c = 0; c < analytic.cols(); ++c) {
				const double w = params[k](r, c);
				params[k](r, c) = w + h;
				const double loss_up = reference_loss(params, x, y);
				params[k](r, c) = w - h;
				const double loss_down = reference_loss(params, x, y);
				params[k](r, c) = w;

				const double numeric = (loss_up - loss_down) / (2.0 * h);
				// The float error of backpropagation is approximately 1e-7.
				ASSERT_NEAR(analytic(r, c), numeric, 1e-6 + 1e-4 * std::abs(numeric))
					<< (k % 2 == 0 ? "W" : "b") << " of layer " << k / 2 << " at (" << r << ", "
					<< c << ")";
				++checked;
			}
		}
	}
	EXPECT_EQ(checked, 696U); // all W and b: 13*17 + 17 + 17*16 + 16 + 16*10 + 10
}
