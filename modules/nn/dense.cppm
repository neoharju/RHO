/// @file dense.cppm
/// @brief Fully connected (dense) layer: z = x * W + b.
export module rho.nn.dense;

import std;

export import rho.core.matrix;
export import rho.core.rng;
export import rho.linalg.gemm;

export namespace rho::nn {

/// How to set the start values of the weights.
enum class init_scheme : std::uint8_t {
    he_normal,     ///< Normal(0, sqrt(2 / n_in)). Use it before a ReLU.
    glorot_uniform ///< Uniform(-a, a), a = sqrt(6 / (n_in + n_out)). Use it for the output layer.
};

/// @brief A fully connected layer: z = x * W + b.
///
/// Shapes (B = number of samples in the batch):
///  - x: B x n_in (input), W: n_in x n_out, b: 1 x n_out, z: B x n_out.
///
/// The layer owns its parameters (W, b) and their gradients (dW, db).
/// It also keeps a copy of W^T, because then dx = dz * W^T is a normal GEMM
/// with rows in memory order. The first layer does not need dx, so it has no W^T.
/// backward() updates W^T from W before it uses it, so W^T can never be out of date
/// after an optimizer step. The copy costs O(n_in * n_out), much less than the GEMMs.
///
/// @tparam T Element type (float or double).
template <class T>
class dense {
  public:
    /// @brief Make a layer. All parameters and gradients are 0. Call initialize() next.
    /// @param n_in Number of inputs.
    /// @param n_out Number of outputs.
    /// @param needs_input_grad If true, backward() can compute dx (the layer keeps W^T).
    ///        Use false for the first layer, because nothing comes before it.
    dense(std::size_t n_in, std::size_t n_out, bool needs_input_grad = true)
        : w_{n_in, n_out},
          wt_{needs_input_grad ? rho::core::matrix<T>{n_out, n_in} : rho::core::matrix<T>{}},
          dw_{n_in, n_out}, b_{1U, n_out}, db_{1U, n_out}, n_in_{n_in}, n_out_{n_out} {

        w_.fill(T{0});
        wt_.fill(T{0});
        dw_.fill(T{0});
        b_.fill(T{0});
        db_.fill(T{0});
    }

    /// @return Number of inputs.
    [[nodiscard]] std::size_t n_in() const noexcept {
        return n_in_;
    }
    /// @return Number of outputs.
    [[nodiscard]] std::size_t n_out() const noexcept {
        return n_out_;
    }

    /// @brief Set random start weights and set the biases to 0.
    ///
    /// He: normal(0, sqrt(2 / n_in)), for hidden layers followed by ReLU.
    /// Glorot: uniform(-a, a), a = sqrt(6 / (n_in + n_out)), for the output layer.
    ///
    /// @param scheme The distribution of the weights.
    /// @param rng The generator. The same seed gives the same weights.
    void initialize(init_scheme scheme, rho::core::xoshiro256pp &rng) noexcept {
        const double he_sd = std::sqrt(2.0 / static_cast<double>(n_in_));
        const double glorot = std::sqrt(6.0 / static_cast<double>(n_in_ + n_out_));
        for (std::size_t i = 0; i < n_in_; ++i) {
            for (T &w : w_.row(i)) {
                w = static_cast<T>(scheme == init_scheme::he_normal ? rng.normal() * he_sd
                                                                    : rng.uniform(-glorot, glorot));
            }
        }
        b_.fill(T{0});
    }

    /// @brief Forward pass: z = x * W + b.
    ///
    /// Each row of z is first set to b. Then gemm adds x * W.
    /// So the bias addition costs no extra pass over z.
    ///
    /// @param x Input, B x n_in.
    /// @param z Output, B x n_out.
    /// @pre x.cols() == n_in(), z.rows() == x.rows(), z.cols() == n_out().
    ///      If not, the program stops.
    void forward(rho::core::matrix_view<const T> x, rho::core::matrix_view<T> z) const noexcept {
        rho::core::require(x.cols() == n_in_ && z.rows() == x.rows() && z.cols() == n_out_,
                           "dense::forward: x must be B x n_in and z must be B x n_out");
        const auto bias = b_.row(0U);
        for (std::size_t r = 0; r < z.rows(); ++r) {
            std::ranges::copy(bias, z.row(r).begin());
        }
        rho::linalg::gemm<T>(x, w_.view(), z);
    }

    /// @brief Backward pass. It computes the gradients from dz = dL/dz.
    ///
    ///  - dW = x^T * dz   (stored in the layer, see grad_weights())
    ///  - db = sum of the rows of dz   (stored in the layer, see grad_bias_matrix())
    ///  - dx = dz * W^T   (the gradient for the layer below; skipped if dx is empty)
    ///
    /// The old values of dW and db are overwritten (not added to).
    ///
    /// @param x The input of the forward pass, B x n_in.
    /// @param dz Gradient of the loss with respect to z, B x n_out.
    /// @param dx Output, B x n_in, or an empty view (0 rows) if not needed.
    /// @pre x is B x n_in, dz is B x n_out, and dx is empty or B x n_in.
    /// @pre The layer was made with needs_input_grad == true if dx is not empty.
    ///      If a precondition is false, the program stops.
    void backward(rho::core::matrix_view<const T> x,
                  rho::core::matrix_view<const T> dz,
                  rho::core::matrix_view<T> dx) noexcept {
        rho::core::require(x.cols() == n_in_ && dz.cols() == n_out_ && x.rows() == dz.rows(),
                           "dense::backward: x must be B x n_in and dz must be B x n_out");
        rho::core::require(dx.rows() == 0U
                               || (wt_.rows() != 0U && dx.rows() == dz.rows() && dx.cols() == n_in_),
                           "dense::backward: dx must be empty or B x n_in (needs_input_grad)");

        // Row i of dW is the sum over the batch of x(r, i) * (row r of dz).
        // Each thread writes its own rows of dW, so there is no data race.
#pragma omp parallel for
        for (std::size_t i = 0; i < n_in_; ++i) {
            const auto dw_i = dw_.row(i);
            std::ranges::fill(dw_i, T{0});
            for (std::size_t r = 0; r < dz.rows(); ++r) {
                const T x_ri = x(r, i);
                const auto dz_r = dz.row(r);
                for (std::size_t j = 0; j < n_out_; ++j) {
                    dw_i[j] += x_ri * dz_r[j];
                }
            }
        }

        const auto db = db_.row(0U);
        std::ranges::fill(db, T{0});
        for (std::size_t r = 0; r < dz.rows(); ++r) {
            const auto dz_r = dz.row(r);
            for (std::size_t j = 0; j < n_out_; ++j) {
                db[j] += dz_r[j];
            }
        }

        if (dx.rows() != 0U) {
            refresh_transpose(); // W may have changed since the last call (optimizer step)
            for (std::size_t r = 0; r < dx.rows(); ++r) {
                std::ranges::fill(dx.row(r), T{0}); // gemm adds into dx
            }
            rho::linalg::gemm<T>(dz, wt_.view(), dx);
        }
    }

  private:
    /// Copy W into W^T. It does nothing for a layer without W^T (needs_input_grad == false).
    void refresh_transpose() noexcept {
        if (wt_.rows() == 0U) { // no W^T (needs input grad == false, e.g. layer 0)
            return;
        }
        rho::linalg::transpose<T>(w_.view(), wt_.view());
    }

  public:
    /// @return Writable view of W (n_in x n_out). The optimizer changes it.
    [[nodiscard]] rho::core::matrix_view<T> weights() noexcept {
        return w_.view();
    }
    /// @return View of dW (n_in x n_out), as computed by the last backward().
    [[nodiscard]] rho::core::matrix_view<T> grad_weights() noexcept {
        return dw_.view();
    }
    /// @return Writable view of b (1 x n_out).
    [[nodiscard]] rho::core::matrix_view<T> bias_matrix() noexcept {
        return b_.view();
    }
    /// @return View of db (1 x n_out), as computed by the last backward().
    [[nodiscard]] rho::core::matrix_view<T> grad_bias_matrix() noexcept {
        return db_.view();
    }

  private:
    rho::core::matrix<T> w_;  // weights, n_in x n_out
    rho::core::matrix<T> wt_; // W^T, n_out x n_in; empty if this layer has no dX
    rho::core::matrix<T> dw_; // weight gradients
    rho::core::matrix<T> b_;  // bias, one row
    rho::core::matrix<T> db_; // bias gradients
    std::size_t n_in_ = 0U;
    std::size_t n_out_ = 0U;
};

} // namespace rho::nn
