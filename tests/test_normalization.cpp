#include <gtest/gtest.h>

import std;
import rho.io.dataset;

namespace fs = std::filesystem;

TEST(Normalization, MeanToZero_OneStddevToOne) {
	// 128 and 64 are exact in float
	const auto n = rho::io::normalization::from_mean_stddev(128.0, 64.0);

	EXPECT_FLOAT_EQ(n.apply(128), 0.0f);
	EXPECT_FLOAT_EQ(n.apply(192), 1.0f);
	EXPECT_FLOAT_EQ(n.apply(0), -2.0f);
}

TEST(Normalization, ZeroStddev) {
	const auto n = rho::io::normalization::from_mean_stddev(5.0, 0.0);

	EXPECT_FLOAT_EQ(n.scale, 1.0f);
	EXPECT_FLOAT_EQ(n.apply(5), 0.0f);
}
