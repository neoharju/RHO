export module rho.core.rng;

import std;

export namespace rho::core {

// xoshiro256++ (blackman & vigna)
// http://arxiv.org/pdf/1805.01407
// https://prng.di.unimi.it/xoshiro256plusplus.c
/*
 *  NB: Addition wraps modulo 2^64 due to uint
 *  result = Add two state words, rotate the result left (bits that fall from
 *   left side come back to right), add the first selected state word again
 *
 * const uint64_t result_plusplus = rotl(s[0] + s[3], R) + s[0];
 * const uint64_t t = s[1] << A;
 * s[2] ^= s[0]; // note: independent dependency paths for s[2] and s[3] for ILP
 * s[3] ^= s[1]; //
 * s[1] ^= s[2];
 * s[0] ^= s[3];
 * s[2] ^= t;
 * s[3] = rotl(s[3], B);
 *
 * xoshiro: xor - shift - rotate - xo-shi-ro; PseudoRNG
 *
 * 256-bits of internal state // the generator has four 64-bit integers: 4 x 64 = 256
 *							  // s[0], s[1], s[2], s[3] // keep as memory
 * ++ output scrambling; (add, rotate), add
 *
 * xoshiro256++ parameters: R=23, A=17, B=45 taken from paper
 *   - A: shift s[1]				   i.e. s[1] << 17;
 *   - B: rotates s[3]				   i.e. rotl(s[3], 45);
 *     - NB: (A, B) define one 256-bit linear transformation, the generator
 *           would change by changing one parameter, which could result in
 *           shorter periods, worse statistical properties etc:
 *           they are algorithm parameter.
 *   - R: rotate sum(s[0], s[3]) by R, i.e. rotl(s[0] + s[3], R)
 *
 *  ---------------------------------------------------------------
 *  splitmix64 for initial state of generators, as recommended in
 *  https://prng.di.unimi.it/
 *  https://prng.di.unimi.it/splitmix64.c
 */

class xoshiro256pp {
  public:
    static constexpr std::uint64_t max() noexcept {
        return std::numeric_limits<std::uint64_t>::max();
    }

    explicit xoshiro256pp(std::uint64_t seed) noexcept {
        std::uint64_t x = seed;
        for (auto &w : s_) {
            w = splitmix64(x);
        }
        // Cant state transition from all zero state, as said in
        // 1805.01407 paper, so exclute it
        if (std::ranges::all_of(s_, [](std::uint64_t w) { return w == 0U; })) {
            s_[0] = 0x9E3779B97F4A7C15ULL;
        }
    }

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

    std::uint64_t operator()() noexcept {
        return next();
    }

    void jump() noexcept {
        apply_jump(jump_poly);
    }
    void long_jump() noexcept {
        apply_jump(long_jump_poly);
    }

    [[nodiscard]] xoshiro256pp jumped() const noexcept {
        xoshiro256pp copy{*this};
        copy.jump();
        return copy;
    }

    // get state
    [[nodiscard]] std::array<std::uint64_t, 4> state() const noexcept {
        return s_;
    }

    // Lemire's multiply-shift with rejection.
    // 1. Generate a random 64-bit number x
    // 2. Cast it to uint 128-bit and multiply with range
    // 3. the upper 64 bits tells us the random answer
    // 4. the lower 64 bits tells us wheter the result is slightly biased
    // 5. If it is biased, get a new number, else return  upper 64 bits
    // Ref: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    //      https://lemire.me/blog/2016/06/30/fast-random-shuffling/
    std::uint64_t bounded(std::uint64_t range) noexcept {

        __uint128_t m = static_cast<__uint128_t>(next()) * range;
        std::uint64_t low = static_cast<std::uint64_t>(m);

        // this is unlikely, since its essentially
        // happends with probability range/2^64
        if (low < range) [[unlikely]] {
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

  private:
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << static_cast<unsigned>(k)) | (x >> static_cast<unsigned>(64 - k));
    }

    // https://prng.di.unimi.it/splitmix64.c
    static constexpr std::uint64_t splitmix64(std::uint64_t &x) noexcept {
        std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }

    // https://prng.di.unimi.it/xoshiro256plusplus.c
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

// Fisher-Yates shuffle
// Ref: https://lemire.me/blog/2016/06/30/fast-random-shuffling/
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
