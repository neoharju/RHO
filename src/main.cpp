import std;

import rho.core.matrix;
import rho.io.dataset;
import rho.linalg.gemm;

int main() { // int argc, char *argv[]) {
	using rho::core::aligned_vector;
	// test basic gemm.
	aligned_vector<float> pixelsA = {1, 2, 3, 4};
	aligned_vector<float> pixelsB = {1, 0, 0, 1}; // identity matrix

	using rho::core::matrix;
	const matrix<float> A{2, 2, std::move(pixelsA)};
	const matrix<float> B{2, 2, std::move(pixelsB)};
	matrix<float> C{2, 2};
	C.fill(0.0f);

	rho::linalg::gemm(A.view(), B.view(), C.view());

	for (std::size_t i = 0; i < C.rows(); ++i) {
		for (std::size_t j = 0; j < C.cols(); ++j) {
			std::print("{} ", C(i, j));
		}
		std::println();
	}

	std::println("test: Importing MNIST dataset");

	const auto train = rho::io::dataset::load(
			"data/train-images-idx3-ubyte",
			"data/train-labels-idx1-ubyte");

	if (!train) {
		rho::io::display_error(train.error());
		return 1;
	}

	std::println("Loaded {} train images", train->size());
	const auto test = rho::io::dataset::load(
			"data/t10k-images-idx3-ubyte",
			"data/t10k-labels-idx1-ubyte");

	if (!test) {
		rho::io::display_error(test.error());
		return 1;
	}
	std::println("Loaded {} test images", test->size());
	// check train norms
	{
		std::vector<std::uint32_t> train_ids(train->size());
		std::iota(train_ids.begin(), train_ids.end(), 0U);
		const auto norm = rho::io::compute_normalization(*train, train_ids);

		const double mean = -norm.bias / norm.scale;
		const double stddev = 1.0 / norm.scale;

		std::println("Train pixels: mean {:.4f}, stddev {:.4f}, normalized range [{:.4f}, {:.4f}]", mean, stddev, norm.apply(0), norm.apply(255));
	}
	// check test norms
	{	
		std::vector<std::uint32_t> test_ids(test->size());
		std::iota(test_ids.begin(), test_ids.end(), 0U);
		const auto norm = rho::io::compute_normalization(*test, test_ids);

		const double mean = -norm.bias / norm.scale;
		const double stddev = 1.0 / norm.scale;

		std::println("Test pixels: mean {:.4f}, stddev {:.4f}, normalized range [{:.4f}, {:.4f}]", mean, stddev, norm.apply(0), norm.apply(255));
	}

}
