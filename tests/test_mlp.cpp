#include <gtest/gtest.h>
#include <omp.h>

import std;
import rho.optim;


using rho::core::matrix;

namespace {

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

	// All parameters of a network, flat, for exact comparison.
	std::vector<float> flat_parameters(rho::nn::mlp<float> &net) {
		std::vector<float> out;
		for (const auto &p : net.parameters()) {
			for (std::size_t r = 0; r < p.value.rows(); ++r) {
				out.insert(out.end(), p.value.row(r).begin(), p.value.row(r).end());
			}
		}
		return out;
	}

} // namespace

TEST(Mlp, BadTopologyThrows) {
	const std::vector<std::size_t> empty;
	const std::vector<std::size_t> one{784};
	const std::vector<std::size_t> zero_width{784, 0, 10};
	const std::vector<std::size_t> ok{4, 3};
	EXPECT_THROW((rho::nn::mlp<float>{empty, 8, 1}), std::invalid_argument);
	EXPECT_THROW((rho::nn::mlp<float>{one, 8, 1}), std::invalid_argument);
	EXPECT_THROW((rho::nn::mlp<float>{zero_width, 8, 1}), std::invalid_argument);
	EXPECT_THROW((rho::nn::mlp<float>{ok, 0, 1}), std::invalid_argument);
	EXPECT_NO_THROW((rho::nn::mlp<float>{ok, 1, 1}));
}

TEST(Mlp, TopologyAndInitialization) {
	const std::array<std::size_t, 4> sizes{6, 5, 4, 3};
	rho::nn::mlp<float> a{sizes, 2, 7};
	rho::nn::mlp<float> b{sizes, 2, 7};
	rho::nn::mlp<float> c{sizes, 2, 8};

	EXPECT_EQ(a.topology(), (std::vector<std::size_t>{6, 5, 4, 3}));
	EXPECT_EQ(a.parameters().size(), 6U);              // W and b for 3 layers
	EXPECT_EQ(flat_parameters(a), flat_parameters(b)); // same seed, same network
	EXPECT_NE(flat_parameters(a), flat_parameters(c));

	// Biases start at 0 but weights should not
	for (std::size_t l = 0; l < 3; ++l) {
		const auto bias = a.parameters()[2 * l + 1].value;
		EXPECT_TRUE(std::ranges::all_of(bias.row(0), [](float v) { return v == 0.0f; }));
	}
}

TEST(Mlp, HeInitializationHasExpectedSpread) {
	// Hidden layer 0 uses He normal: stddev sqrt(2 / n_in) = sqrt(2 / 784) = 0.0505.
	const std::array<std::size_t, 3> sizes{784, 256, 10};
	rho::nn::mlp<float> net{sizes, 1, 3};
	const auto w = net.parameters()[0].value;
	double sum = 0.0;
	double sum_sq = 0.0;
	for (std::size_t r = 0; r < w.rows(); ++r) {
		for (const float v : w.row(r)) {
			sum += v;
			sum_sq += static_cast<double>(v) * v;
		}
	}
	const double n = static_cast<double>(w.rows() * w.cols());
	EXPECT_NEAR(sum / n, 0.0, 1e-3);
	EXPECT_NEAR(std::sqrt(sum_sq / n), std::sqrt(2.0 / 784.0), 1e-3);
}

TEST(Mlp, PreconditionsStopTheProgram) {
	GTEST_FLAG_SET(death_test_style, "threadsafe");
	const std::array<std::size_t, 3> sizes{4, 5, 3};
	rho::nn::mlp<float> net{sizes, 2, 1};

	const auto too_many_rows = random_input(3, 4, 1); // max_batch is 2
	const auto wrong_cols = random_input(2, 5, 1);    // input size is 4
	const auto x = random_input(2, 4, 1);
	const std::array<std::uint8_t, 2> labels{0, 2};
	const std::array<std::uint8_t, 1> too_few_labels{0};
	const std::array<std::uint8_t, 2> label_too_big{0, 3}; // only classes 0, 1, 2

	EXPECT_DEATH(static_cast<void>(net.forward(too_many_rows.view())), "row_range");
	EXPECT_DEATH(static_cast<void>(net.forward(wrong_cols.view())), "dense::forward");
	EXPECT_DEATH(static_cast<void>(net.loss(x.view(), too_few_labels)), "softmax_cross_entropy");
	EXPECT_DEATH(static_cast<void>(net.loss(x.view(), label_too_big)),
			"label >= number of classes");
	EXPECT_NO_FATAL_FAILURE(static_cast<void>(net.loss(x.view(), labels)));
}

TEST(Mlp, LossOfEmptyBatchStops) {
	GTEST_FLAG_SET(death_test_style, "threadsafe");
	const matrix<float> logits{0, 3};
	EXPECT_DEATH(
			static_cast<void>(rho::nn::softmax_cross_entropy<float>(
					logits.view(), std::span<const std::uint8_t>{}, rho::core::matrix_view<float>{})),
			"need N > 0 rows");
}

// training must give bit-identical weights for any number of threads.
TEST(Mlp, SameResultForAnyThreadCount) {
	const std::array<std::size_t, 4> sizes{32, 64, 48, 10};
	const auto x = random_input(64, 32, 5);
	std::vector<std::uint8_t> y(64);
	for (std::size_t i = 0; i < y.size(); ++i) {
		y[i] = static_cast<std::uint8_t>(i % 10U);
	}

	auto train = [&](int threads) {
		const int saved = omp_get_max_threads();
		omp_set_num_threads(threads);
		rho::nn::mlp<float> net{sizes, 64, 9};
		rho::optim::sgd<float> opt{0.05f};
		for (int step = 0; step < 10; ++step) {
			static_cast<void>(net.forward_backward(x.view(), y));
			opt.step(net.parameters());
		}
		omp_set_num_threads(saved);
		return flat_parameters(net);
	};

	const auto one = train(1);
	EXPECT_EQ(one, train(2));
	EXPECT_EQ(one, train(4));
}

TEST(Dense, BackwardMatchesSimpleSumsWhenTheBatchGetsSmaller) {
	constexpr std::size_t n_in = 37; // not a multiple of 6 or 16: edge tiles
	constexpr std::size_t n_out = 20;
	rho::nn::dense<float> layer{n_in, n_out};
	rho::core::xoshiro256pp rng{41U};
	layer.initialize(rho::nn::init_scheme::he_normal, rng);
	const auto w = layer.weights();

	for (const std::size_t batch : {8U, 3U}) {
		const auto x = random_input(batch, n_in, 42U + batch);
		const auto dz = random_input(batch, n_out, 43U + batch);
		matrix<float> dx{batch, n_in};
		layer.backward(x.view(), dz.view(), dx.view());

		const auto dw = layer.grad_weights();
		for (std::size_t i = 0; i < n_in; ++i) {
			for (std::size_t j = 0; j < n_out; ++j) {
				double sum = 0.0;
				for (std::size_t r = 0; r < batch; ++r) {
					sum += static_cast<double>(x(r, i)) * dz(r, j);
				}
				ASSERT_NEAR(dw(i, j), sum, 1e-5) << "dW, batch " << batch;
			}
		}
		for (std::size_t r = 0; r < batch; ++r) {
			for (std::size_t i = 0; i < n_in; ++i) {
				double sum = 0.0;
				for (std::size_t j = 0; j < n_out; ++j) {
					sum += static_cast<double>(dz(r, j)) * w(i, j);
				}
				ASSERT_NEAR(dx(r, i), sum, 1e-5) << "dx, batch " << batch;
			}
		}
	}
}
