export module rho.core.memory;

import std;

/*
 *  The motivation to create a custom allocator and buffer:
 *   - Get the data 64 byte aligned, so it is faster to process with
 *   vectorization
 *   - Make sure the data is not accidentally copied many times (which reduces
 *   performance)
 *   - Avoid writing zeros unnecessarily when allocating a buffer (array)
 *   - Have the benefits of automatic allocation/deallocation like it is for
 *   std::vector
 *
 *   Idea taken from the high performance C++ Boost library:
 *   -
 * https://github.com/boostorg/align/blob/440281d63d1c0b7c7fde63ded67b4860b57d5756/include/boost/align/aligned_allocator.hpp
 */

template <class T>
struct aligned_allocator {
    // needed for std::vector allocator etc
    using value_type = T;
    static constexpr std::align_val_t alignment{std::max<std::size_t>(64, alignof(T))};

    aligned_allocator() = default;

    template <class U>
    constexpr aligned_allocator(const aligned_allocator<U> &) {}

    [[nodiscard]] T *allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length{};
        }

        return static_cast<T *>(::operator new(n * sizeof(T), alignment));
    }

    template <class U>
    void construct(U *p) {
        ::new (static_cast<void *>(p)) U;
    }

    void deallocate(T *p, std::size_t n) noexcept {
        ::operator delete(p, n * sizeof(T), alignment);
    }

    friend bool operator==(const aligned_allocator &, const aligned_allocator &) = default;
};

export namespace rho::core {

template <class T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

template <class T>
class aligned_buffer {
  public:
    aligned_buffer() = default;
    // elements are not zeroed
    explicit aligned_buffer(std::size_t n) : v_(n) {}

    aligned_buffer(aligned_vector<T> &&v) noexcept : v_{std::move(v)} {}

    aligned_buffer(const aligned_vector<T> &v) = delete;

    aligned_buffer(aligned_buffer &&) noexcept = default;
    aligned_buffer &operator=(aligned_buffer &&) noexcept = default;

    // We only want to move data, not copy
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer &operator=(const aligned_buffer &) = delete;

    [[nodiscard]]
    std::size_t size() const noexcept {
        return v_.size();
    }

    [[nodiscard]]
    std::size_t size_bytes() const noexcept {
        return v_.size() * sizeof(T);
    }

    [[nodiscard]]
    auto *data(this auto &self) noexcept {
        return self.v_.data();
    }

    [[nodiscard]]
    auto span(this auto &self) noexcept {
        return std::span{self.v_.data(), self.v_.size()};
    }

    // fill the elements of the buffer with a specific value
    void fill(const T &value) {
        std::ranges::fill(v_, value);
    }

  private:
    std::vector<T, aligned_allocator<T>> v_;
};

} // namespace rho::core
