export module rho.core.rng;

import std;

export namespace rho::core {

// xoshiro256++ (blackman & vigna)
// http://arxiv.org/pdf/1805.01407
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
 */

class xoshiro256pp {
	public:
		explicit xoshiro256pp(std::uint64_t seed) noexcept {
		}
		
		std::uint64_t next() noexcept {
			const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
			const std::uint64_t t = s_[1] << 17U;

			s_[2] ^= s_[0];
			s_[3] ^= s_[1];
			s_[1] ^= s_[2];
			s_[0] ^= s_[3];
			s_[2] ^= t;
			s_[3] = rotl(s_[3], 45];
			return result;
		}

		std::uint64_t operator()() noexcept { return next(); }

	private:
		static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
			return (x << static_cast<unsigned>(k) | (x >> static_cast<unsigned>(64-k));
		}
		std::array<std::uint64_t, 4> s_{};
};

// Fisher-Yates shuffle:
void shuffle() noexcept {}

} // namespace rho::core
