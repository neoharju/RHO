/// @file matrix.cppm
/// @brief Row-major matrix (owns memory) and matrix_view (does not own memory).
///
/// Element (i, j) is at index i * stride + j. Each row is contiguous in memory.
/// For a matrix, stride == cols. A view of some rows of a matrix keeps the stride
/// of the matrix.
export module rho.core.matrix;

import std;

export import rho.core.memory;

export namespace rho::core {

/// @brief Non-owning view of a row-major matrix.
///
/// A view is a pointer plus a shape. It is cheap to copy (4 words), so give it by value.
/// The view does not keep the memory alive. The owner (for example a matrix) must
/// live longer than all views of it.
///
/// @tparam T Element type. Use `const T` for a read-only view.
template <class T>
class matrix_view {
  public:
    /// Make an empty view (0 x 0, null data).
    constexpr matrix_view() noexcept = default;

    /// @brief Make a view of existing memory.
    /// @param p Pointer to element (0, 0).
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param stride Distance in elements from the start of one row to the start of the
    ///               next row. Must be >= cols.
    constexpr matrix_view(T *p, std::size_t rows, std::size_t cols, std::size_t stride) noexcept
        : data_{p}, rows_{rows}, cols_{cols}, stride_{stride} {}

    /// @brief View of rows [first, first + count). No data is copied.
    /// @param first Index of the first row.
    /// @param count Number of rows.
    /// @return A view with `count` rows and the same columns and stride.
    /// @pre first + count <= rows(). If not, the program stops (see require()).
    [[nodiscard]]
    constexpr matrix_view row_range(std::size_t first, std::size_t count) const noexcept {
        rho::core::require(first <= rows_ && count <= rows_ - first,
                           "matrix_view::row_range: rows out of range");
        return matrix_view{data_ + first * stride_, count, cols_, stride_};
    }

    /// @brief Convert `matrix_view<U>` to `matrix_view<const U>`.
    ///
    /// Only this direction is allowed: a read-only view cannot become a writable view.
    template <class U>
        requires std::same_as<T, const U>
    constexpr matrix_view(matrix_view<U> other) noexcept
        : matrix_view{other.data(), other.rows(), other.cols(), other.stride()} {}

    /// @return Number of rows.
    [[nodiscard]]
    constexpr std::size_t rows() const noexcept {
        return rows_;
    }

    /// @return Number of columns.
    [[nodiscard]]
    constexpr std::size_t cols() const noexcept {
        return cols_;
    }

    /// @return Distance in elements between the starts of two rows that follow each other.
    [[nodiscard]]
    constexpr std::size_t stride() const noexcept {
        return stride_;
    }

    /// @return Pointer to element (0, 0).
    [[nodiscard]]
    constexpr T *data() const noexcept {
        return data_;
    }

    /// @brief Access element (i, j), for example A(3, 5).
    /// @pre i < rows() and j < cols(). There is no bounds check.
    [[nodiscard]]
    constexpr T &operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * stride_ + j];
    }

    /// @brief Row i as a contiguous span of cols() elements.
    /// @pre i < rows(). There is no bounds check.
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

/// @brief Row-major matrix that owns 64-byte aligned memory.
///
/// The matrix can move but cannot copy. A new matrix(rows, cols) has
/// indeterminate values: call fill() or write all elements before you read them.
///
/// @tparam T Element type.
template <class T>
class matrix {
  public:
    /// Make an empty 0 x 0 matrix.
    matrix() noexcept = default;

    /// @brief Allocate a rows x cols matrix. The values are not set.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @throws std::length_error if rows * cols does not fit in std::size_t.
    matrix(std::size_t rows, std::size_t cols)
        : data_{check_size(rows, cols)}, rows_{rows}, cols_{cols} {}

    /// @brief Make a matrix from existing data in row-major order. No data is copied.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param data Buffer with exactly rows * cols elements.
    /// @throws std::invalid_argument if data.size() != rows * cols.
    /// @throws std::length_error if rows * cols does not fit in std::size_t.
    matrix(std::size_t rows, std::size_t cols, aligned_buffer<T> data)
        : data_{std::move(data)}, rows_{rows}, cols_{cols} {
        if (data_.size() != check_size(rows, cols)) {
            throw std::invalid_argument("matrix: values.size() must equal rows * cols");
        }
    }

    /// Move constructor. The source becomes an empty 0 x 0 matrix.
    matrix(matrix &&other) noexcept
        : data_{std::move(other.data_)}, rows_{std::exchange(other.rows_, 0)},
          cols_{std::exchange(other.cols_, 0)} {}

    /// Move assignment. The source becomes an empty 0 x 0 matrix.
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

    /// @return Number of rows.
    [[nodiscard]]
    std::size_t rows() const noexcept {
        return rows_;
    }

    /// @return Number of columns.
    [[nodiscard]]
    std::size_t cols() const noexcept {
        return cols_;
    }

    /// @return Row stride in elements. It is always equal to cols().
    [[nodiscard]]
    std::size_t stride() const noexcept {
        return cols_;
    }

    /// @return Pointer to element (0, 0) (const if the matrix is const).
    [[nodiscard]]
    auto *data(this auto &self) noexcept {
        return self.data_.data();
    }

    /// @brief Access element (i, j).
    /// @pre i < rows() and j < cols(). There is no bounds check.
    [[nodiscard]]
    auto &operator()(this auto &self, std::size_t i, std::size_t j) noexcept {
        return self.data_.span()[i * self.cols_ + j];
    }

    /// @brief Row i as a span of cols() elements.
    /// @pre i < rows().
    [[nodiscard]]
    auto row(this auto &self, std::size_t i) noexcept {
        return self.data_.span().subspan(i * self.cols_, self.cols_);
    }

    /// @return A view of the full matrix (matrix_view<const T> if the matrix is const).
    [[nodiscard]]
    auto view(this auto &self) noexcept {
        return matrix_view{self.data_.data(), self.rows_, self.cols_, self.cols_};
    }

    /// @brief Set all elements to the same value.
    void fill(const T &value) {
        data_.fill(value);
    }

  private:
    /// @return rows * cols.
    /// @throws std::length_error if the product does not fit in std::size_t.
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
