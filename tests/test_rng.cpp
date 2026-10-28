#include <gtest/gtest.h>

import std;
import rho.core.rng;

TEST(xoshiro256pp, SameSeedDeterministic) {
    rho::core::xoshiro256pp a{1234U};
    rho::core::xoshiro256pp b{1234U};

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

// Chi-square test with a fixed seed. For k - 1 degrees of freedom the value
// must be below the 99.9 % quantile; a biased generator or a wrong shuffle
// gives a much larger value.
namespace {
double chi_square(std::span<const std::size_t> counts, double expected) {
    double chi2 = 0.0;
    for (const auto c : counts) {
        const double d = static_cast<double>(c) - expected;
        chi2 += d * d / expected;
    }
    return chi2;
}
} // namespace

TEST(xoshiro256pp, BoundedIsUniform) {
    rho::core::xoshiro256pp rng{4U};
    constexpr std::size_t draws = 100000;
    std::array<std::size_t, 10> counts{};
    for (std::size_t i = 0; i < draws; ++i) {
        ++counts[rng.bounded(10U)];
    }
    EXPECT_LT(chi_square(counts, draws / 10.0), 27.88); // 9 dof, p = 0.001
}

TEST(Shuffle, AllPermutationsEquallyLikely) {
    // 3 elements have 6 orders. Fisher-Yates gives each with probability 1/6.
    rho::core::xoshiro256pp rng{6U};
    constexpr std::size_t trials = 60000;
    std::array<std::size_t, 6> counts{};
    for (std::size_t t = 0; t < trials; ++t) {
        std::array<int, 3> v{0, 1, 2};
        rho::core::shuffle(v, rng);
        // index of the permutation: 2 * first + (second > third ? 1 : 0), in 0..5
        const auto index = static_cast<std::size_t>(2 * v[0] + (v[1] > v[2] ? 1 : 0));
        ++counts[index];
    }
    EXPECT_LT(chi_square(counts, trials / 6.0), 20.52); // 5 dof, p = 0.001
}

TEST(xoshiro256pp, JumpedDoesNotChangeOriginal) {
    const rho::core::xoshiro256pp a{8U};
    const auto before = a.state();
    const auto b = a.jumped();
    EXPECT_EQ(a.state(), before);
    EXPECT_NE(b.state(), before);

    rho::core::xoshiro256pp c{8U};
    c.jump();
    EXPECT_EQ(c.state(), b.state()); // jumped() == copy + jump()
}

TEST(xoshiro256pp, ZeroSeedIsValid) {
    rho::core::xoshiro256pp rng{0U};
    const auto s = rng.state();
    EXPECT_FALSE(std::ranges::all_of(s, [](std::uint64_t w) { return w == 0U; }));
    EXPECT_NE(rng.next(), rng.next());
}

TEST(xoshiro256pp, BoundedWithHugeRange) {
    // range = 2^63 + 1: about half of all draws are biased and must be redrawn,
    // so this exercises the rejection loop.
    rho::core::xoshiro256pp rng{10U};
    const std::uint64_t range = (std::uint64_t{1} << 63U) + 1U;
    bool upper_half = false;
    for (int i = 0; i < 1000; ++i) {
        const std::uint64_t v = rng.bounded(range);
        ASSERT_LT(v, range);
        upper_half |= v >= (std::uint64_t{1} << 62U);
    }
    EXPECT_TRUE(upper_half);
}

TEST(xoshiro256pp, CallOperatorAndLongJump) {
    rho::core::xoshiro256pp a{11U};
    rho::core::xoshiro256pp b{11U};
    EXPECT_EQ(a(), b.next()); // operator() is next()

    rho::core::xoshiro256pp j{12U};
    rho::core::xoshiro256pp lj{12U};
    j.jump();
    lj.long_jump();
    EXPECT_NE(j.state(), lj.state()); // 2^128 and 2^192 steps give different states
}

TEST(Shuffle, EmptyAndSingleElementUnchanged) {
    rho::core::xoshiro256pp rng{13U};
    std::vector<int> empty;
    rho::core::shuffle(empty, rng);
    EXPECT_TRUE(empty.empty());

    std::vector<int> one{42};
    rho::core::shuffle(one, rng);
    EXPECT_EQ(one, std::vector<int>{42});
}
