import std;

import rho.core.matrix;
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

	std::println("Hello World from main!");
}
