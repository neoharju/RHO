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

TEST(Shuffle, IsAPermutation) {
    std::vector<std::uint32_t> v(1000);
    std::iota(v.begin(), v.end(), 0U);
    rho::core::xoshiro256pp rng{7U};

    rho::core::shuffle(v, rng);

    EXPECT_FALSE(std::ranges::is_sorted(v)); // it did shuffle
    std::ranges::sort(v);
    for (std::uint32_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i], i); // nothing lost
    }
}

TEST(xoshiro256pp, BoundedStaysInRange) {
    rho::core::xoshiro256pp rng{1U};
    for (int i = 0; i < 10000; ++i) {
        ASSERT_LT(rng.bounded(10U), 10U);
    }
    EXPECT_EQ(rng.bounded(1U), 0U); // the only possible value
}

TEST(xoshiro256pp, Uniform01) {
    rho::core::xoshiro256pp rng{2U};
    for (int i = 0; i < 10000; ++i) {
        const double u = rng.uniform01();
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
    }
}

TEST(xoshiro256pp, NormalHasMeanZeroVarianceOne) {
    rho::core::xoshiro256pp rng{3U};
    constexpr int n = 100000;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double z = rng.normal();
        sum += z;
        sum_sq += z * z;
    }
    const double mean = sum / n;
    EXPECT_NEAR(mean, 0.0, 0.02);
    EXPECT_NEAR(sum_sq / n - mean * mean, 1.0, 0.02);
}
