export module rho.io.idx;

import std;

export namespace rho::io {

namespace detail {
// The data is advertised as big-endian which means we need to convert it to
// little endian
//
// hexdump -C -n 64 data/train-images-idx3-ubyte
// 00000000  00 00 08 03 00 00 ea 60  00 00 00 1c 00 00 00 1c  |.......`........|
//
//  00 00 08 03 = 2051:  idx magic number
//  00 00 EA 60 = 60000: number of images
//  00 00 00 1C = 28:    number of rows
//  00 00 00 1C = 28:    number of columns
//
//  So this information matches the advertised values in the website, that
//  corresponds to the MNIST training image dataset

	[[nodiscard]] inline std::uint32_t bigendian32(const std::byte* p) noexcept {
		// We need to byteswap from big endian to little endian
		//
		// eg. 00 00 08 03 big endian to 
		//     03 08 00 00 little endian
		//
		// std::to_integer<std::uint8_t>(p[i]) converts the byte into an 8 bit integer 
		// Then we cast it into a uint32 and shift its position by << amount and
		// combine all four uint32s
		return (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[0])) << 24U) |
			   (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[1])) << 16U) |
			   (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[2])) << 8U) |
			   (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[3])));
	}
} // namespace detail

}
