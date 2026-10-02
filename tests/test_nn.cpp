#include <gtest/gtest.h>

import std;
// import rho.core.matrix;
import rho.nn.activation;
import rho.nn.loss;

using rho::core::aligned_vector;
using rho::core::matrix;
/* ReLU:
 * Every value below zero should return zero, else value or rounded to zero
 * if very close to zero.
 */
TEST(ReLU, ForwardInt) {
	aligned_vector<int> z_values = {-2, -0, 0, 3, static_cast<int>(1e-30)};
	aligned_vector<int> a_values(5, 0.0);

	const matrix<int> Z{1, 5, std::move(z_values)};
	matrix<int> A{1, 5, std::move(a_values)};

	rho::nn::relu_forward<int>(Z.view(), A.view());
	EXPECT_EQ(A(0, 0), 0);
	EXPECT_EQ(A(0, 1), 0);
	EXPECT_EQ(A(0, 2), 0);
	EXPECT_EQ(A(0, 3), 3);

	EXPECT_EQ(A(0, 4), 0);

	EXPECT_FALSE(std::signbit(A(0, 1)));
}

TEST(ReLU, ForwardFloat) {
	aligned_vector<float> z_values = {-2.0f, -0.0f, 0.0f, 3.5f, 1e-30f};
	aligned_vector<float> a_values(5, 0.0f);

	const matrix<float> Z{1, 5, std::move(z_values)};
	matrix<float> A{1, 5, std::move(a_values)};

	rho::nn::relu_forward<float>(Z.view(), A.view());
	EXPECT_EQ(A(0, 0), 0.0f);
	EXPECT_EQ(A(0, 1), 0.0f);
	EXPECT_EQ(A(0, 2), 0.0f);
	EXPECT_EQ(A(0, 3), 3.5f);

	EXPECT_GT(A(0, 4), 0.0f);

	EXPECT_FALSE(std::signbit(A(0, 1)));
}

TEST(ReLU, ForwardDouble) {
	aligned_vector<double> z_values = {-2.0, -0.0, 0.0, 3.5, 1e-30};
	aligned_vector<double> a_values(5, 0.0);

	const matrix<double> Z{1, 5, std::move(z_values)};
	matrix<double> A{1, 5, std::move(a_values)};

	rho::nn::relu_forward<double>(Z.view(), A.view());
	EXPECT_EQ(A(0, 0), 0.0);
	EXPECT_EQ(A(0, 1), 0.0);
	EXPECT_EQ(A(0, 2), 0.0);
	EXPECT_EQ(A(0, 3), 3.5);

	EXPECT_GT(A(0, 4), 0.0);

	EXPECT_FALSE(std::signbit(A(0, 1)));
}
/*
 * The subgradient at zero is any value in [0,1] but we define it as 0
 * ReLUs derative is 1 for for every z > 0
 */
TEST(ReLU, BackwardInt) {
	// 1-e30 gets rounded to 0
	aligned_vector<int> z_values = {-2, -0, 0, static_cast<int>(3.5), static_cast<int>(1e-30)};
	aligned_vector<int> da_values(5, 1);
	aligned_vector<int> dz_values(5, 0);

	const matrix<int> Z{1, 5, std::move(z_values)};
	const matrix<int> dA{1, 5, std::move(da_values)};
	matrix<int> dZ{1, 5, std::move(dz_values)};

	rho::nn::relu_backward<int>(Z.view(), dA.view(), dZ.view());

	EXPECT_EQ(dZ(0, 0), 0);
	EXPECT_EQ(dZ(0, 1), 0);
	EXPECT_EQ(dZ(0, 2), 0);
	EXPECT_EQ(dZ(0, 3), 1);
	EXPECT_EQ(dZ(0, 4), 0); // 1e-30 gets rounded to 0
}

TEST(ReLU, BackwardFloat) {
	aligned_vector<float> z_values = {-2.0f, -0.0f, 0.0f, 3.5f, 1e-30f};
	aligned_vector<float> da_values(5, 1.0f);
	aligned_vector<float> dz_values(5, 0.0f);

	const matrix<float> Z{1, 5, std::move(z_values)};
	const matrix<float> dA{1, 5, std::move(da_values)};
	matrix<float> dZ{1, 5, std::move(dz_values)};

	rho::nn::relu_backward<float>(Z.view(), dA.view(), dZ.view());

	EXPECT_EQ(dZ(0, 0), 0.0f);
	EXPECT_EQ(dZ(0, 1), 0.0f);
	EXPECT_EQ(dZ(0, 2), 0.0f);
	EXPECT_EQ(dZ(0, 3), 1.0f);
	EXPECT_EQ(dZ(0, 4), 1.0f);
}

TEST(ReLU, BackwardDouble) {
	aligned_vector<double> z_values = {-2.0, -0.0, 0.0, 3.5, 1e-30};
	aligned_vector<double> da_values(5, 1.0);
	aligned_vector<double> dz_values(5, 0.0);

	const matrix<double> Z{1, 5, std::move(z_values)};
	const matrix<double> dA{1, 5, std::move(da_values)};
	matrix<double> dZ{1, 5, std::move(dz_values)};

	rho::nn::relu_backward<double>(Z.view(), dA.view(), dZ.view());

	EXPECT_EQ(dZ(0, 0), 0.0);
	EXPECT_EQ(dZ(0, 1), 0.0);
	EXPECT_EQ(dZ(0, 2), 0.0);
	EXPECT_EQ(dZ(0, 3), 1.0);
	EXPECT_EQ(dZ(0, 4), 1.0);
}

// -----------------------------------------------------------------------------
// --- softmax and cross-entropy
// -----------------------------------------------------------------------------

TEST(SoftmaxCrossEntropy, NumericalExtremes) {
	aligned_vector<double> z_values = {1000.0, 1000.0, 1000.0};
	aligned_vector<double> dz_values(3, 0.0);
	const aligned_vector<std::uint8_t> y{1U};

	const matrix<double> Z{1, 3, std::move(z_values)};
	matrix<double> dZ{1, 3, std::move(dz_values)};

	const double loss = rho::nn::softmax_cross_entropy<double>(Z.view(), std::span{y}, dZ.view());

	ASSERT_TRUE(std::isfinite(loss));
	EXPECT_NEAR(loss, std::log(3.0), 1e-6);

	EXPECT_NEAR(dZ(0, 0), 1.0 / 3.0, 1e-6);
	EXPECT_NEAR(dZ(0, 1), 1.0 / 3.0 - 1.0, 1e-6);
}
