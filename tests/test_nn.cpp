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

TEST(SoftmaxCrossEntropy, GradientRowsSumToZero) {
	const matrix<double> Z{2, 3, aligned_vector<double>{1, 2, 3, -1, 0, 4}};
	matrix<double> dZ{2, 3};
	const std::array<std::uint8_t, 2> y{0U, 2U};

	static_cast<void>(rho::nn::softmax_cross_entropy<double>(
				Z.view(), std::span<const std::uint8_t>{y}, dZ.view()));

	for (std::size_t i = 0; i < 2; ++i) {
		EXPECT_NEAR(dZ(i, 0) + dZ(i, 1) + dZ(i, 2), 0.0, 1e-12);
	}
}

TEST(Softmax, RowsSumToOne) {
	const matrix<double> Z{2, 3, aligned_vector<double>{1, 2, 3, -50, 0, 50}};
	matrix<double> P{2, 3};

	rho::nn::softmax<double>(Z.view(), P.view());

	for (std::size_t i = 0; i < 2; ++i) {
		EXPECT_NEAR(P(i, 0) + P(i, 1) + P(i, 2), 1.0, 1e-12);
	}
}

TEST(CrossEntropy, IsMeanMinusLogOfTrueClass) {
	const matrix<double> P{2, 2, aligned_vector<double>{0.25, 0.75, 0.5, 0.5}};
	const std::array<std::uint8_t, 2> y{1U, 0U};

	const double loss = rho::nn::cross_entropy<double>(P.view(), std::span<const std::uint8_t>{y});

	EXPECT_NEAR(loss, (-std::log(0.75) - std::log(0.5)) / 2.0, 1e-12);
}

TEST(ArgmaxRow, LargestScore) {
	// clang-format off
	const matrix<float> s{2, 3, aligned_vector<float>{0.1F, 0.7F, 0.2F,   // largest at 1
		0.5F, 0.5F, 0.1F}}; // tie: 0 wins
							// clang-format on
	EXPECT_EQ(rho::nn::argmax_row<float>(s.view(), 0), 1U);
	EXPECT_EQ(rho::nn::argmax_row<float>(s.view(), 1), 0U);
}

TEST(CountCorrect, LargestScoreIsTheLabel) {
	// clang-format off
	const matrix<float> s{3, 3, aligned_vector<float>{0.1F, 0.7F, 0.2F,   // predicts 1
		0.9F, 0.0F, 0.1F,   // predicts 0
		0.2F, 0.3F, 0.5F}}; // predicts 2
							// clang-format on
	const std::array<std::uint8_t, 3> y{1U, 0U, 0U}; // 2 of 3 right
	EXPECT_EQ(rho::nn::count_correct<float>(s.view(), std::span<const std::uint8_t>{y}), 2U);
}

