/// @file memory.cppm
/// @brief 64-byte aligned memory, a move-only buffer and a precondition check.
///
/// Why this module exists:
///  - The data is 64-byte aligned. One cache line and one AVX-512 register
///    are 64 bytes, so aligned rows are faster to load and to vectorize.
///  - The buffer can only move. It cannot copy by accident, because a copy
///    of a large matrix costs much time.
///  - A new buffer does not write zeros. The caller writes the values once.
///  - Memory is freed automatically (RAII), as with std::vector.
///
/// The idea comes from the Boost.Align library:
/// https://github.com/boostorg/align/blob/440281d63d1c0b7c7fde63ded67b4860b57d5756/include/boost/align/aligned_allocator.hpp
export module rho.core.memory;

import std;

/// @brief Allocator that returns memory aligned to 64 bytes (or more if T needs it).
///
/// It also changes value-initialization to default-initialization:
/// std::vector<T, aligned_allocator<T>>(n) does not write zeros for trivial
/// types such as float. The elements have indeterminate values until you write them.
///
/// @tparam T Element type.
template <class T>
struct aligned_allocator {
    using value_type = T; // needed by std::allocator_traits (std::vector etc.)

    /// Alignment in bytes: 64 (one cache line), or alignof(T) if T needs more.
    static constexpr std::align_val_t alignment{std::max<std::size_t>(64, alignof(T))};

    aligned_allocator() = default;

    /// Rebind constructor. std::vector uses it to make an allocator for another type.
    template <class U>
    constexpr aligned_allocator(const aligned_allocator<U> &) {}

    /// @brief Allocate memory for n objects. The objects are not constructed.
    /// @param n Number of objects.
    /// @return Pointer to aligned memory.
    /// @throws std::bad_array_new_length if n * sizeof(T) does not fit in std::size_t.
    /// @throws std::bad_alloc if the system has no free memory.
    [[nodiscard]] T *allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length{};
        }

        return static_cast<T *>(::operator new(n * sizeof(T), alignment));
    }

    /// @brief Default-initialize an object at p (no zeroing for trivial types).
    ///
    /// std::vector calls this function for each element of vector(n).
    /// Calls with arguments, for example vector{1, 2, 3}, use std::construct_at instead.
    /// @param p Pointer to memory from allocate().
    template <class U>
    void construct(U *p) {
        ::new (static_cast<void *>(p)) U;
    }

    /// @brief Free memory from allocate().
    /// @param p Pointer that allocate() returned.
    /// @param n The same n that was given to allocate().
    void deallocate(T *p, std::size_t n) noexcept {
        ::operator delete(p, n * sizeof(T), alignment);
    }

    /// All aligned_allocator objects are equal: memory from one can be freed by another.
    friend bool operator==(const aligned_allocator &, const aligned_allocator &) = default;
};

export namespace rho::core {

/// @brief Stop the program if a precondition is false.
///
/// Use it for programmer errors (wrong shapes, wrong indices).
/// Do not use it for bad user input; return an error value for that.
/// The check stays on in Release builds. It costs one branch.
///
/// @param ok The condition that must be true.
/// @param what A short message. The function prints it to std::cerr.
/// @param where The caller's file and line. Do not give this argument.
constexpr void require(bool ok,
                       std::string_view what,
                       std::source_location where = std::source_location::current()) noexcept {
    if (!ok) [[unlikely]] {
        std::println(
            std::cerr, "{}:{}: precondition failed: {}", where.file_name(), where.line(), what);
        std::abort();
    }
}

/// std::vector with 64-byte aligned storage and no zeroing in vector(n).
template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

/// @brief Move-only, 64-byte aligned array with a fixed size.
///
/// A new buffer has indeterminate values (no zeros). Write each element before you read it,
/// or call fill().
///
/// @tparam T Element type.
template <class T>
class aligned_buffer {
  public:
    /// Make an empty buffer (size 0, no memory).
    aligned_buffer() = default;

    /// @brief Allocate n elements. The elements are not set to zero.
    /// @param n Number of elements.
    explicit aligned_buffer(std::size_t n) : v_(n) {}

    /// @brief Take the memory of an aligned_vector. No data is copied.
    /// @param v The vector to move from.
    aligned_buffer(aligned_vector<T> &&v) noexcept : v_{std::move(v)} {}

    /// Copy from a vector is not allowed. Use std::move to give the vector to the buffer.
    aligned_buffer(const aligned_vector<T> &v) = delete;

    aligned_buffer(aligned_buffer &&) noexcept = default;
    aligned_buffer &operator=(aligned_buffer &&) noexcept = default;

    // We only want to move data, not copy
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer &operator=(const aligned_buffer &) = delete;

    /// @return Number of elements.
    [[nodiscard]]
    std::size_t size() const noexcept {
        return v_.size();
    }

    /// @return Size of the data in bytes.
    [[nodiscard]]
    std::size_t size_bytes() const noexcept {
        return v_.size() * sizeof(T);
    }

    /// @return Pointer to the first element (const if the buffer is const).
    [[nodiscard]]
    auto *data(this auto &self) noexcept {
        return self.v_.data();
    }

    /// @return A std::span over all elements (span<const T> if the buffer is const).
    [[nodiscard]]
    auto span(this auto &self) noexcept {
        return std::span{self.v_.data(), self.v_.size()};
    }

    /// @brief Set all elements to the same value.
    /// @param value The new value of each element.
    void fill(const T &value) {
        std::ranges::fill(v_, value);
    }

  private:
    std::vector<T, aligned_allocator<T>> v_;
};

} // namespace rho::core
