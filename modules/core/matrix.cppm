export module rho.core.matrix;

import std;

export namespace rho::core {

template <class T>
class matrix_view {
  public:
    constexpr matrix_view() noexcept = default;

    constexpr matrix_view(T *p, std::size_t rows, std::size_t cols, std::size_t stride) noexcept
        : data_{p}, rows_{rows}, cols_{cols}, stride_{stride} {}

    [[nodiscard]]
    constexpr std::size_t rows() const noexcept {
        return rows_;
    }

    [[nodiscard]]
    constexpr std::size_t cols() const noexcept {
        return cols_;
    }

    [[nodiscard]]
    constexpr std::size_t stride() const noexcept {
        return stride_;
    }

    [[nodiscard]]
    constexpr T *data() const noexcept {
        return data_;
    }

    // lets us write e.g. A(3, 5). Note: const only for object not the data
    // unless matrix_view is matrix_view(const float) A;
    [[nodiscard]]
    constexpr T &operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * stride_ + j];
    }

    // Row return
    [[nodiscard]]
    constexpr std::span<T> row(std::size_t i) const noexcept {
        return {data_ + i * stride_, cols_};
    }

  private:
    T *data_{nullptr};
    std::size_t rows_{};
    std::size_t cols_{};
    std::size_t stride_{};
};

template <class T>
class matrix {
  public:
    matrix() noexcept = default;

    matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        : data_{std::move(data)}, rows_{rows}, cols_{cols}, stride_{cols} {}

    [[nodiscard]]
    std::size_t rows() const noexcept {
        return rows_;
    }

    [[nodiscard]]
    std::size_t cols() const noexcept {
        return cols_;
    }

    [[nodiscard]]
    std::size_t stride() const noexcept {
        return stride_;
    }
    // return data non-const matrices
    [[nodiscard]]
    T *data() noexcept {
        return data_.data();
    }
    // return data for const matrices to stay const
    [[nodiscard]]
    const T *data() const noexcept {
        return data_;
    }

    // lets us write e.g. A(3, 5). Note: const only for object not the data
    // unless matrix_view is matrix_view(const float) A;
    [[nodiscard]]
    T &operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * stride_ + j];
    }

    // again for const matrices to stay const
    [[nodiscard]]
    const T &operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * stride_ + j];
    }

    // Row return
    [[nodiscard]]
    std::span<T> row(std::size_t i) noexcept {
        return {data_ + i * stride_, cols_};
    }

    [[nodiscard]]
    constexpr std::span<T> row(std::size_t i) const noexcept {
        return {data_ + i * stride_, cols_};
    }

    void fill(const T &value) {
        std::ranges::fill(data_, value);
    }

    [[nodiscard]]
    matrix_view<T> view() noexcept {
        return {data_.data(), rows_, cols_, stride_};
    }

    [[nodiscard]]
    matrix_view<const T> view() const noexcept {
        return {data_.data(), rows_, cols_, stride_};
    }

  private:
    std::vector<T> data_; // mm_alloc or aligned buffer later?
    std::size_t rows_{};
    std::size_t cols_{};
    std::size_t stride_{};
};

} // namespace rho::core
