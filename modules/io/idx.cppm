/// @file idx.cppm
/// @brief Safe reader for IDX files (the MNIST file format) and the I/O error types.
///
/// IDX format (all integers are big-endian):
/// @code
/// bytes 0-1  : 00 00
/// byte  2    : element type (08 = unsigned byte; only this type is supported)
/// byte  3    : rank R (number of dimensions)
/// next 4 * R : size of each dimension (uint32, big-endian)
/// rest       : the data, row-major
/// @endcode
///
/// Example (hexdump -C -n 16 data/train-images-idx3-ubyte):
/// @code
/// 00 00 08 03 = magic: unsigned bytes, rank 3
/// 00 00 EA 60 = 60000 images
/// 00 00 00 1C = 28 rows
/// 00 00 00 1C = 28 columns
/// @endcode
///
/// The reader treats the file as untrusted input. It checks the magic number,
/// the rank, each dimension, a 1 GiB size limit, and that the file size is
/// exactly header + data. So a bad header cannot cause an overflow or a huge allocation.
export module rho.io.idx;

import std;

export import rho.core.memory;

namespace fs = std::filesystem;

/// @brief Read a big-endian uint32 from 4 bytes.
///
/// It works on any host byte order, because it builds the value with shifts:
/// b[0] is the most significant byte. Example: 00 00 08 03 -> 0x00000803 = 2051.
/// @param b Pointer to 4 readable bytes.
/// @return The value in host byte order.
[[nodiscard]] constexpr std::uint32_t load_bigendian32(const std::byte *b) noexcept {
    return (std::to_integer<std::uint32_t>(b[0]) << 24U)
           | (std::to_integer<std::uint32_t>(b[1]) << 16U)
           | (std::to_integer<std::uint32_t>(b[2]) << 8U) | std::to_integer<std::uint32_t>(b[3]);
}
// check that it works
static_assert(
    load_bigendian32(
        std::array{std::byte{0x00}, std::byte{0x00}, std::byte{0xEA}, std::byte{0x60}}.data())
    == 60000U);

/// @brief Read exactly bytes.size() bytes from a stream.
/// @return true if all bytes were read. false if the stream ended early or failed.
[[nodiscard]] bool read_exact(std::istream &in, std::span<std::byte> bytes) {
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return in.gcount() == static_cast<std::streamsize>(bytes.size());
}

export namespace rho::io {

/// Largest rank that the reader accepts. MNIST uses rank 3 (images) and rank 1 (labels).
/// Rank 4 allows for example [N, channels, height, width].
inline constexpr std::size_t max_rank = 4;

/// @brief Contents of one IDX file with unsigned byte elements.
struct idx_file {
    std::array<std::uint32_t, max_rank> dims{}; ///< Size of each dimension; only the first `rank` are used.
    std::size_t rank{};                         ///< Number of dimensions, 1 to max_rank.
    rho::core::aligned_buffer<std::uint8_t> data; ///< All elements, row-major.
};

/// Red "X" mark for error messages (ANSI terminal colors).
inline constexpr std::string_view FAILURE = "\x1b[1;31m✘\x1b[0m"; // red

// --- Error types -------------------------------------------------------------
// Each error is a small struct with the data that the message needs.
// Error (below) is a std::variant of all of them.

/// The file system could not get the file status (for example, the file does not exist).
struct FsError {
    fs::path file;
    std::string message; ///< Text from std::error_code::message().
    int code;            ///< Native error code from std::error_code::value().
};

/// The path is not a regular file, or the file cannot be opened for reading.
struct OpenError {
    fs::path file;
};

/// The file is not a valid IDX file, or reading it failed.
struct ReadError {
    fs::path file;
    std::string_view reason; ///< Points to a string literal (static lifetime).
};

/// One dimension in the IDX header has size 0.
struct EmptyDimensionError {
    fs::path file;
    std::size_t dimension; ///< Index of the dimension, from 0.
};

/// The file is a valid IDX file, but it does not have the MNIST shape.
struct BadShapeError {
    fs::path file;
    std::array<std::uint32_t, max_rank> dims;
    std::size_t rank;
    std::string_view expected; ///< Expected shape as text, e.g. "[N, 28, 28]".
};

/// The images file and the labels file have a different number of samples.
struct CountMismatchError {
    fs::path images_file;
    fs::path labels_file;
    std::uint32_t images;
    std::uint32_t labels;
};

/// A label is not in the MNIST range 0 to 9.
struct BadLabelError {
    fs::path file;
    std::size_t index;   ///< Index of the first bad label.
    std::uint8_t value;  ///< Its value.
};

/// All errors that loading the data can return.
using Error = std::variant<FsError,
                           OpenError,
                           ReadError,
                           EmptyDimensionError,
                           BadShapeError,
                           CountMismatchError,
                           BadLabelError>;

/// @brief Helper to make one visitor from many lambdas for std::visit.
///
/// Example: `std::visit(match{[](const A &) {...}, [](const B &) {...}}, v);`
template <typename... Fs>
struct match : Fs... {
    using Fs::operator()...;
};

/// @brief Print a human-readable message for an error to std::cerr.
/// @param error The error to print.
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

/// @brief Read an IDX file with unsigned byte elements (type code 0x08).
///
/// Checks, in this order:
///  1. The path is a regular file and it opens for reading.
///  2. The magic number is 00 00 08 R, with 1 <= R <= max_rank.
///  3. No dimension is 0, and the data size is at most 1 GiB (no overflow is possible).
///  4. The file size is exactly 4 + 4 * R + (product of dimensions).
///
/// @param path Path to the (uncompressed) IDX file.
/// @return The file contents, or an Error (FsError, OpenError, ReadError or
///         EmptyDimensionError) that tells what is wrong.
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
    if (!read_exact(in, std::span{raw}.first(4U * rank))) {
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
        // Check before the multiplication, so data_size * d cannot overflow.
        if (d > (max_bytes / data_size)) {
            return std::unexpected(ReadError{path, "it claims more than 1 GiB of data"});
        }
        // multiply data_size by dimensions
        data_size *= d;
        out.dims[i] = d;
    }

    // A file that is too short or too long is rejected before any allocation.
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
