export module rho.io.idx;

import std;

export import rho.core.memory;

namespace fs = std::filesystem;

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
[[nodiscard]] constexpr std::uint32_t load_bigendian32(const std::byte *b) noexcept {
    // We need to byteswap from big endian to little endian
    //
    // eg. 00 00 08 03 big endian to
    //     03 08 00 00 little endian
    //
    // std::to_integer<std::uint8_t>(p[i]) converts the byte into an 8 bit integer
    // Then we cast it into a uint32 and shift its position by << amount and
    // combine all four uint32s
    return (std::to_integer<std::uint32_t>(b[0]) << 24U)
           | (std::to_integer<std::uint32_t>(b[1]) << 16U)
           | (std::to_integer<std::uint32_t>(b[2]) << 8U) | std::to_integer<std::uint32_t>(b[3]);
}
// check that it works
static_assert(load_bigendian32(std::array{
                  std::byte{0x00}, 
				  std::byte{0x00}, 
				  std::byte{0xEA}, 
				  std::byte{0x60}
				  }.data()) == 60000U);

// Reads bytes.size() bytes
[[nodiscard]] bool read_exact(std::istream &in, std::span<std::byte> bytes) {
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return in.gcount() == static_cast<std::streamsize>(bytes.size());
}

export namespace rho::io {

// The MNIST files have three dimensions [size, 28, 28]) and
// one labels: [size]. With four dims we could have e.g.
// [size, 1 color channel, 28 height, 28 width].
inline constexpr std::size_t max_rank = 4;

struct idx_file {
    std::array<std::uint32_t, max_rank> dims{};
    std::size_t rank{};
    rho::core::aligned_buffer<std::uint8_t> data;
};

// red X
inline constexpr std::string_view FAILURE = "\x1b[1;31m✘\x1b[0m"; // red

// Error types
struct FsError {
    fs::path file;
    std::string message;
    int code;
};
// cant open for reading
struct OpenError {
    fs::path file;
};
// not a valid IDX file, or reading it failed
struct ReadError {
    fs::path file;
    std::string_view reason;
};

struct EmptyDimensionError {
    fs::path file;
    std::size_t dimension;
};

// mnist checks needed in dataset
// Valid IDX, but not MNIST shape
struct BadShapeError {
    fs::path file;
    std::array<std::uint32_t, max_rank> dims;
    std::size_t rank;
    std::string_view expected; // e.g. "[N, 28, 28]"
};
// images and labels differ
struct CountMismatchError {
    fs::path images_file;
    fs::path labels_file;
    std::uint32_t images;
    std::uint32_t labels;
};
// labels do not belong to the MNIST 0-9 range
struct BadLabelError {
    fs::path file;
    std::size_t index;
    std::uint8_t value;
};

using Error = std::variant<FsError,
                           OpenError,
                           ReadError,
                           EmptyDimensionError,
                           BadShapeError,
                           CountMismatchError,
                           BadLabelError>;

// std::visit helper
template <typename... Fs>
struct match : Fs... {
    using Fs::operator()...;
};

void display_error(const Error &error) {
    constexpr auto visitor = match{
        [](const FsError &e) {
            std::println(std::cerr,
                         "{} Cannot open '{}': {} ({})",
                         FAILURE,
                         e.file.string(),
                         e.message,
                         e.code);
        },
        [](const OpenError &e) {
            std::println(std::cerr,
                         "{} Cannot open '{}' for reading (no permission, or not a regular file).",
                         FAILURE,
                         e.file.string());
        },
        [](const ReadError &e) {
            std::println(std::cerr, "{} Cannot read '{}': {}.", FAILURE, e.file.string(), e.reason);
        },
        [](const EmptyDimensionError &e) {
            std::println(std::cerr,
                         "{} '{}': dimension {} has size 0.",
                         FAILURE,
                         e.file.string(),
                         e.dimension);
        },
        [](const BadShapeError &e) {
            const auto shape = std::span{e.dims}.first(std::min(e.rank, e.dims.size()));
            std::println(std::cerr,
                         "{} '{}' has shape {}, but MNIST needs {}.",
                         FAILURE,
                         e.file.string(),
                         shape,
                         e.expected);
        },
        [](const CountMismatchError &e) {
            std::println(std::cerr,
                         "{} '{}' holds {} images, but '{}' has {} labels",
                         FAILURE,
                         e.images_file.string(),
                         e.images,
                         e.labels_file.string(),
                         e.labels);
        },
        [](const BadLabelError &e) {
            std::println(std::cerr,
                         "{} '{}': label {} is {}, but labels must be 0 to 9",
                         FAILURE,
                         e.file.string(),
                         e.index,
                         e.value);
        },
    };
    std::visit(visitor, error);
}

[[nodiscard]] std::expected<idx_file, Error> read_idx_u8(const fs::path &path) {
    constexpr std::size_t max_bytes = std::size_t{1} << 30U; // 1 GiB

    std::error_code ec;
    const fs::file_status status = fs::status(path, ec);
    if (ec) {
        return std::unexpected(FsError{path, ec.message(), ec.value()});
    }
    if (!fs::is_regular_file(status)) {
        return std::unexpected(OpenError{path});
    }

    std::ifstream in{path, std::ios::binary};
    if (!in) {
        return std::unexpected(OpenError{path});
    }

    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // Magic number: 00 00 08 R. 08 = unsigned bytes, R = num of dimensions.
    std::array<std::byte, 4> magic{};
    if (!read_exact(in, magic)) {
        return std::unexpected(ReadError{path, "does not match an IDX header"});
    }
    const std::size_t rank = std::to_integer<std::size_t>(magic[3]);
    if (magic[0] != std::byte{0x00} || magic[1] != std::byte{0x00} || magic[2] != std::byte{0x08}
        || rank == 0U || rank > max_rank) {
        return std::unexpected(ReadError{path, "it is not an IDX file (still zipped?)"});
    }

    // The size of each dimension, 4 bytes each.
    std::array<std::byte, 4 * max_rank> raw{};
    if (!read_exact(in, {raw.data(), 4U * max_rank})) {
        return std::unexpected(ReadError{path, "does not match an IDX header"});
    }

    idx_file out;
    out.rank = rank;
    std::uint64_t data_size = 1U;
    for (std::size_t i = 0; i < rank; ++i) {
        const std::uint32_t d = load_bigendian32(raw.data() + 4U * i);

        if (d == 0U) {
            return std::unexpected(EmptyDimensionError{path, i});
        }
        if (d > (max_bytes / data_size)) {
            return std::unexpected(ReadError{path, "it claims more than 1 GiB of data"});
        }
        // multiply data_size by dimensions
        data_size *= d;
        out.dims[i] = d;
    }

    const std::uint64_t header_size = 4U + 4U * rank;
    if (file_size < 0 || static_cast<std::uint64_t>(file_size) != header_size + data_size) {
        return std::unexpected(ReadError{path, "its size does not match its header"});
    }

    out.data = rho::core::aligned_buffer<std::uint8_t>{static_cast<std::size_t>(data_size)};
    if (!read_exact(in, std::as_writable_bytes(out.data.span()))) {
        return std::unexpected(ReadError{path, "reading ended early"});
    }
    return out;
}
} // namespace rho::io
