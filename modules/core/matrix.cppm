export module rho.core.matrix;

import std;

export import rho.core.memory;

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

    // lets us write e.g. A(3, 5).
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

    matrix(std::size_t rows, std::size_t cols)
        : data_{check_size(rows, cols)}, rows_{rows}, cols_{cols} {}

    matrix(std::size_t rows, std::size_t cols, aligned_buffer<T> data)
        : data_{std::move(data)}, rows_{rows}, cols_{cols} {
        if (data_.size() != check_size(rows, cols)) {
            throw std::invalid_argument("matrix: values.size() must equal rows * cols");
        }
    }
    // move while setting cols and rows 0
    matrix(matrix &&other) noexcept
        : data_{std::move(other.data_)}, rows_{std::exchange(other.rows_, 0)},
          cols_{std::exchange(other.cols_, 0)} {}

    matrix &operator=(matrix &&other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            rows_ = std::exchange(other.rows_, 0);
            cols_ = std::exchange(other.cols_, 0);
        }
        return *this;
    }
    // prevent matrix from being copyable
    matrix(const matrix &) = delete;
    matrix &operator=(const matrix &) = delete;

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
        return cols_;
    }
    // return data non-const matrices
    [[nodiscard]]
    auto *data(this auto &self) noexcept {
        return self.data_.data();
    }

    [[nodiscard]]
    auto &operator()(this auto &self, std::size_t i, std::size_t j) noexcept {
        return self.data_.span()[i * self.cols_ + j];
    }

    // Row return
    [[nodiscard]]
    auto row(this auto &self, std::size_t i) noexcept {
        return self.data_.span().subspan(i * self.cols_, self.cols_);
    }

    [[nodiscard]]
    auto view(this auto &self) noexcept {
        return matrix_view{self.data_.data(), self.rows_, self.cols_, self.cols_};
    }

    // use aligned_buffer.fill to fill array with values
    void fill(const T &value) {
        data_.fill(value);
    }

  private:
    static std::size_t check_size(std::size_t rows, std::size_t cols) {
        // this will never happen for mnist dataset, just a sanity check
        if (cols > 0 && rows > (std::numeric_limits<std::size_t>::max() / cols)) [[unlikely]] {
            throw std::length_error("matrix: rows * cols overflows");
        }
        return rows * cols;
    }
    aligned_buffer<T> data_;
    std::size_t rows_{};
    std::size_t cols_{};
};

} // namespace rho::core
