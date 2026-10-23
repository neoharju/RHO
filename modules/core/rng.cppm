/// @file rng.cppm
/// @brief Pseudo-random numbers: xoshiro256++, bounded integers, uniform and normal
///        values, and a Fisher-Yates shuffle.
///
/// The generator is not for cryptography. It is fast and has good statistical
/// quality for simulation and machine learning.
///
/// Sources:
///  - xoshiro256++: D. Blackman and S. Vigna, http://arxiv.org/pdf/1805.01407 and
///    https://prng.di.unimi.it/xoshiro256plusplus.c
///  - splitmix64 (makes the start state from one seed): https://prng.di.unimi.it/splitmix64.c
///  - Bounded integers without modulo bias (Lemire):
///    https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
///  - Fast shuffle (Lemire): https://lemire.me/blog/2016/06/30/fast-random-shuffling
///  - Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
export module rho.core.rng;

import std;

export namespace rho::core {

/// @brief xoshiro256++ pseudo-random number generator (Blackman and Vigna).
///
/// The state is four 64-bit words (256 bits). The name means xor, shift, rotate.
/// "++" is the output scrambler: add two words, rotate, add again.
///
/// One step of the generator:
/// @code
/// result = rotl(s[0] + s[3], R) + s[0];   // all additions wrap modulo 2^64
/// t      = s[1] << A;
/// s[2] ^= s[0];   // s[2] and s[3] do not depend on each other,
/// s[3] ^= s[1];   // so the CPU can do them at the same time (ILP)
/// s[1] ^= s[2];
/// s[0] ^= s[3];
/// s[2] ^= t;
/// s[3]  = rotl(s[3], B);
/// @endcode
///
/// The paper gives R = 23, A = 17, B = 45. A and B together define one linear
/// transformation. Do not change them: other values can give a shorter period
/// or worse statistics.
///
/// The period is 2^256 - 1. The all-zero state is not allowed, because the
/// generator cannot leave it.
class xoshiro256pp {
  public:
    /// @return The largest value that next() can return (2^64 - 1).
    static constexpr std::uint64_t max() noexcept {
        return std::numeric_limits<std::uint64_t>::max();
    }

    /// @brief Make the start state from one 64-bit seed with splitmix64.
    ///
    /// Equal seeds give equal number sequences. The constructor never makes
    /// the all-zero state.
    /// @param seed Any 64-bit value.
    explicit xoshiro256pp(std::uint64_t seed) noexcept {
        std::uint64_t x = seed;
        for (auto &w : s_) {
            w = splitmix64(x);
        }
        // Cant state transition from all zero state, as said in
        // 1805.01407 paper, so exclude it
        if (std::ranges::all_of(s_, [](std::uint64_t w) { return w == 0U; })) {
            s_[0] = 0x9E3779B97F4A7C15ULL;
        }
    }

    /// @brief Advance the state one step.
    /// @return 64 random bits.
    std::uint64_t next() noexcept {
        const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17U;

        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    /// Same as next(). It lets you call the generator like a function: rng().
    std::uint64_t operator()() noexcept {
        return next();
    }

    /// @brief Advance the state by 2^128 steps.
    ///
    /// Use it to make streams that do not overlap, for example one per thread.
    void jump() noexcept {
        apply_jump(jump_poly);
    }

    /// @brief Advance the state by 2^192 steps.
    void long_jump() noexcept {
        apply_jump(long_jump_poly);
    }

    /// @return A copy of this generator, advanced by 2^128 steps. This generator does not change.
    [[nodiscard]] xoshiro256pp jumped() const noexcept {
        xoshiro256pp copy{*this};
        copy.jump();
        return copy;
    }

    /// @return The four state words (for tests and checkpoints).
    [[nodiscard]] std::array<std::uint64_t, 4> state() const noexcept {
        return s_;
    }

    /// @brief Uniform random integer in [0, range), without modulo bias (Lemire).
    ///
    /// How it works:
    ///  1. Multiply 64 random bits by range. The result has 128 bits.
    ///  2. The upper 64 bits are the answer.
    ///  3. The lower 64 bits show if the answer is biased. A biased answer has
    ///     probability less than range / 2^64. Then we draw a new number.
    ///
    /// No division happens in the usual case.
    /// @param range Number of possible values. If range is 0, the function returns 0.
    /// @return A value in [0, range).
    std::uint64_t bounded(std::uint64_t range) noexcept {

        __uint128_t m = static_cast<__uint128_t>(next()) * range;
        std::uint64_t low = static_cast<std::uint64_t>(m);

        // this is unlikely, since its essentially
        // happends with probability range/2^64
        if (low < range) [[unlikely]] {
            // threshold = 2^64 mod range
            const std::uint64_t threshold = (max() - range + 1U) % range;
            while (low < threshold) {
                m = static_cast<__uint128_t>(next()) * range;
                low = static_cast<std::uint64_t>(m);
            }
        }
        // The upper 64 bits of random_number * range
        // gives the unbiased random number in [0, range)
        return static_cast<std::uint64_t>(m >> 64U);
    }

    /// @brief Uniform double in [0, 1).
    ///
    /// A double has 53 bits of precision (52 stored fraction bits + 1 hidden bit).
    /// We keep the top 53 random bits (next() >> 11) and multiply by 2^-53.
    /// The result is one of the values {0, 1/2^53, ..., (2^53 - 1)/2^53}.
    /// So it is uniform on this grid, not on all real numbers in [0, 1).
    /// @return A value in [0, 1). It can be 0.0. It is never 1.0.
    [[nodiscard]] inline double uniform01() noexcept {
        return static_cast<double>(next() >> 11U) * 0x1.0p-53;
    }

    /// @brief Uniform double in [lo, hi).
    /// @param lo Lower limit (included).
    /// @param hi Upper limit (not included). Must be > lo.
    [[nodiscard]] double uniform(double lo, double hi) noexcept {
        return lo + (hi - lo) * uniform01();
    }

    /// @brief Standard normal value N(0, 1) with the Box-Muller transform.
    ///
    /// Box-Muller makes two values. We use only one and discard the other.
    /// This keeps the generator stateless, so the sequence is easy to reproduce.
    /// @return A normally distributed value with mean 0 and variance 1.
    [[nodiscard]] double normal() noexcept {
        constexpr double two_pi{2.0 * std::numbers::pi};
        double u1 = uniform01();
        // Chance of getting a 0 from uniform is 1/2^53 but still possible, so
        // we need a redraw if it happens, since log(0) would be -inf
        while (u1 <= 0.0) [[unlikely]] {
            u1 = uniform01();
        }
        const double u2 = uniform01();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
    }

  private:
    /// Rotate x left by k bits. Only for 0 < k < 64 (k = 0 would shift by 64, which is UB).
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << static_cast<unsigned>(k)) | (x >> static_cast<unsigned>(64 - k));
    }

    /// splitmix64: advance x and return one well-mixed 64-bit value.
    /// https://prng.di.unimi.it/splitmix64.c
    static constexpr std::uint64_t splitmix64(std::uint64_t &x) noexcept {
        std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }

    /// Jump with a jump polynomial (from https://prng.di.unimi.it/xoshiro256plusplus.c).
    void apply_jump(const std::array<std::uint64_t, 4> &poly) noexcept {
        std::array<std::uint64_t, 4> acc{};
        for (const std::uint64_t word : poly) {
            for (unsigned b = 0; b < 64U; ++b) {
                if ((word & (std::uint64_t{1} << b)) != 0U) {
                    for (std::size_t i = 0; i < 4U; ++i) {
                        acc[i] ^= s_[i];
                    }
                }
                static_cast<void>(next());
            }
        }
        s_ = acc;
    }

    // magic jump constants for polynomial taken from here
    // https://prng.di.unimi.it/xoshiro256plusplus.c
    static constexpr std::array<std::uint64_t, 4> jump_poly{
        0x180EC6D33CFD0ABAULL, 0xD5A61266F0C9392CULL, 0xA9582618E03FC9AAULL, 0x39ABDC4529B1661CULL};

    static constexpr std::array<std::uint64_t, 4> long_jump_poly{
        0x76E15D3EFEFDCBBFULL, 0xC5004E441C522FB3ULL, 0x77710069854EE241ULL, 0x39109BB02ACBE635ULL};

    std::array<std::uint64_t, 4> s_{};
};

/// @brief Put the elements of a range in a uniformly random order (Fisher-Yates).
///
/// For i = n-1 down to 1: pick j in [0, i] with bounded(), then swap elements i and j.
/// Each of the n! orders has the same probability. Time is O(n). No memory is allocated.
///
/// @param range A random-access range, for example a std::vector.
/// @param rng The generator. Its state changes.
template <std::ranges::random_access_range R>
    requires std::ranges::sized_range<R> && std::permutable<std::ranges::iterator_t<R>>
void shuffle(R &&range, xoshiro256pp &rng) noexcept {

    const auto first = std::ranges::begin(range);
    const auto n = std::ranges::ssize(range);

    if (n < 2)
        return;

    for (auto i = n - 1; i > 0; --i) {
        // pick a random number from 0 to i and store it in j
        const auto j = static_cast<std::ranges::range_difference_t<R>>(
            rng.bounded(static_cast<std::uint64_t>(i) + 1U));
        // make the swap
        std::ranges::iter_swap(first + i, first + j);
    }
}

} // namespace rho::core
