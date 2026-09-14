#include <gtest/gtest.h>

import std;
import rho.core.matrix;
import rho.linalg.gemm;

TEST(Gemm, IdentityMatrix) {
	using rho::core::matrix;

	std::vector<float> pixelsA = {1.0f, 2.0f, 3.0f, 4.0f};

	std::vector<float> pixelsB = {1.0f, 0.0f, 0.0f, 1.0f};

	std::vector<float> pixelsC(4, 0.0f);

	const matrix<float> A{2, 2, std::move(pixelsA)};
	const matrix<float> B{2, 2, std::move(pixelsB)};
	matrix<float> C{2, 2, std::move(pixelsC)};

	rho::linalg::gemm(A.view(), B.view(), C.view());

	ASSERT_EQ(C.rows(), 2);
	ASSERT_EQ(C.cols(), 2);

	EXPECT_FLOAT_EQ(C(0, 0), 1.0f);
	EXPECT_FLOAT_EQ(C(0, 1), 2.0f);
	EXPECT_FLOAT_EQ(C(1, 0), 3.0f);
	EXPECT_FLOAT_EQ(C(1, 1), 4.0f);
}

TEST(Gemm, Int_A23_B32_C22) {

	using rho::core::matrix;
	const matrix<int> A{2, 3, std::vector<int>{1, 2, 3, 4, 5, 6}};
	const matrix<int> B{3, 2, std::vector<int>{1, 2, 3, 4, 5, 6}};
	matrix<int> C{2, 2, std::vector<int>{0, 0, 0, 0, 0, 0}};

	rho::linalg::gemm(A.view(), B.view(), C.view());

	ASSERT_EQ(C.rows(), 2);
	ASSERT_EQ(C.cols(), 2);

	EXPECT_FLOAT_EQ(C(0, 0), 22);
	EXPECT_FLOAT_EQ(C(0, 1), 28);
	EXPECT_FLOAT_EQ(C(1, 0), 49);
	EXPECT_FLOAT_EQ(C(1, 1), 64);
}

TEST(Gemm, Float_A23_B32_C22) {

	using rho::core::matrix;
	const matrix<float> A{2, 3, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
	const matrix<float> B{3, 2, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
	matrix<float> C{2, 2, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

	rho::linalg::gemm(A.view(), B.view(), C.view());

	ASSERT_EQ(C.rows(), 2);
	ASSERT_EQ(C.cols(), 2);

	EXPECT_FLOAT_EQ(C(0, 0), 22.0f);
	EXPECT_FLOAT_EQ(C(0, 1), 28.0f);
	EXPECT_FLOAT_EQ(C(1, 0), 49.0f);
	EXPECT_FLOAT_EQ(C(1, 1), 64.0f);
}

TEST(Gemm, Double_A23_B32_C22) {

	using rho::core::matrix;
	const matrix<double> A{2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
	const matrix<double> B{3, 2, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
	matrix<double> C{2, 2, std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

	rho::linalg::gemm(A.view(), B.view(), C.view());

	ASSERT_EQ(C.rows(), 2);
	ASSERT_EQ(C.cols(), 2);

	EXPECT_FLOAT_EQ(C(0, 0), 22.0);
	EXPECT_FLOAT_EQ(C(0, 1), 28.0);
	EXPECT_FLOAT_EQ(C(1, 0), 49.0);
	EXPECT_FLOAT_EQ(C(1, 1), 64.0);
}
