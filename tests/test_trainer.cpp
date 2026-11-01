#include <gtest/gtest.h>

import std;
import rho.train;

// End-to-end tests of the training loop (rho.train) on MNIST data
//
#ifdef RHO_DATA_DIR
TEST(Mnist, SmallNetworkLearnsToNinetyPercent) {
	namespace fs = std::filesystem;
	const fs::path dir{RHO_DATA_DIR};
	auto train_set =
		rho::io::dataset::load(dir / "train-images-idx3-ubyte", dir / "train-labels-idx1-ubyte");
	auto test_set =
		rho::io::dataset::load(dir / "t10k-images-idx3-ubyte", dir / "t10k-labels-idx1-ubyte");
	if (!train_set || !test_set) {
		GTEST_SKIP() << "MNIST not found in " << RHO_DATA_DIR;
	}

	std::vector<std::uint32_t> ids(10000);
	std::iota(ids.begin(), ids.end(), 0U);
	const auto norm = rho::io::compute_normalization(*train_set, ids);

	using rho::io::sample_order;
	rho::io::batcher train{*train_set, std::move(ids), norm, 32, sample_order::shuffled, 1};
	rho::io::batcher validation{*train_set, {}, norm, 32, sample_order::sequential};
	rho::io::batcher test{
		*test_set, rho::io::all_ids(*test_set), norm, 32, sample_order::sequential};

	const std::array<std::size_t, 3> sizes{784, 64, 10};
	rho::nn::mlp<float> net{sizes, 32, 2};
	rho::optim::sgd<float> opt{0.1f};

	std::vector<double> losses;
	rho::train::fit(net,
			opt,
			train,
			validation,
			{.epochs = 4, .lr_decay = 0.7f},
			[&](const rho::train::epoch_report &r) { losses.push_back(r.train_loss); });

	ASSERT_EQ(losses.size(), 4U);
	EXPECT_TRUE(
			std::ranges::is_sorted(losses, std::ranges::greater{})); // loss goes down each epoch
	const auto result = rho::train::evaluate(net, test);
	EXPECT_EQ(result.total, 10000U);
	std::println("    test accuracy after 4 epochs: {:.2f} %", 100.0 * result.accuracy());
	EXPECT_GT(result.accuracy(), 0.90);
}
#endif
