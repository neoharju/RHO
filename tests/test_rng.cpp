#include <gtest/gtest.h>

import std;
import rho.core.rng;


TEST(xoshiro256pp, SameSeedDeterministic) {
	rho::core::xoshiro256pp a{1234U};
	rho::core::xoshiro256pp b{1234U};
	bool differs = false;

	for (int i = 0; i < 1000; ++i)
		ASSERT_EQ(a.next(), b.next()) << " differed at iteration " << i << '\n';
}

TEST(xoshiro256pp, DifferentSeedsDiverge) {
	rho::core::xoshiro256pp a{12345U};
	rho::core::xoshiro256pp b{12346U};
	bool differs = false;
	for (int i = 0; i < 1000; ++i) {
		differs |= (a.next() != b.next());
	}
	EXPECT_TRUE(differs);
}

TEST(Shuffle, SameSeedDeterministic) {
	std::vector<std::uint32_t> v(10);
	std::iota(v.begin(), v.end(), 0U); // 0U,1U,2U,3U,4U,5U,6U,7U,8U,9U
	auto w = v;
	rho::core::xoshiro256pp a{99U};
	rho::core::xoshiro256pp b{99U};
	rho::core::shuffle(v, a);
	rho::core::shuffle(w, b);
	EXPECT_EQ(v, w);
}

TEST(Shuffle, SameSeedDeterministicInt) {
	std::vector<int> v(10);
	std::iota(v.begin(), v.end(), 0); // 0,1,2,3,4,5,6,7,8,9
	auto w = v;
	rho::core::xoshiro256pp a{99U};
	rho::core::xoshiro256pp b{99U};
	rho::core::shuffle(v, a);
	rho::core::shuffle(w, b);
	EXPECT_EQ(v, w);
}

TEST(Shuffle, SameSeedDeterministicFloat) {
	std::vector<float> v(10);
	std::iota(v.begin(), v.end(), 0.0f);
	auto w = v;
	rho::core::xoshiro256pp a{99U};
	rho::core::xoshiro256pp b{99U};
	rho::core::shuffle(v, a);
	rho::core::shuffle(w, b);
	EXPECT_EQ(v, w);
}

TEST(Shuffle, SameSeedDeterministicDouble) {
	std::vector<double> v(10);
	std::iota(v.begin(), v.end(), 0.0);
	auto w = v;
	rho::core::xoshiro256pp a{99U};
	rho::core::xoshiro256pp b{99U};
	rho::core::shuffle(v, a);
	rho::core::shuffle(w, b);
	EXPECT_EQ(v, w);
}
