export module rho.nn.dense;

import std;

export import rho.core.matrix;
export import rho.core.rng;
export import rho.linalg.gemm;

export namespace rho::nn {

// init starting weights
enum class init_scheme : std::uint8_t {
    he_normal,
    glorot_uniform
};

// A fully connected layer: z = x * W + b.
template <class T>
class dense {
  public:
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

    [[nodiscard]] std::size_t n_in() const noexcept {
        return n_in_;
    }
    [[nodiscard]] std::size_t n_out() const noexcept {
        return n_out_;
    }

    // He: normal(0, sqrt(2 / n_in)), for hidden layers followed by ReLU.
    // Glorot: uniform(-a, a), a = sqrt(6 / (n_in + n_out)), for the output layer.
    // Biases at 0
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
        refresh_transpose();
    }

    // z = x * W + b. z starts as the bias and gemm adds x * W 
    // Shapes: x is B x n_in, z is B x n_out (mlp sizes its buffers this way)
    void forward(rho::core::matrix_view<const T> x, rho::core::matrix_view<T> z) const noexcept {
        const auto bias = b_.row(0U);
        for (std::size_t r = 0; r < z.rows(); ++r) {
            std::ranges::copy(bias, z.row(r).begin());
        }
        rho::linalg::gemm<T>(x, w_.view(), z);
    }

    // From dz = dL/dz (B x n_out):
    //   dW = x^T * dz
    //   db = column sums of dz
    //   dx = dz * W^T, the gradient for the layer below (skipped if dx is empty)
    void backward(rho::core::matrix_view<const T> x,
                  rho::core::matrix_view<const T> dz,
                  rho::core::matrix_view<T> dx) noexcept {

        // Row i of dW is the sum over the batch of x(r, i) * (row r of dz)
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
            for (std::size_t r = 0; r < dx.rows(); ++r) {
                std::ranges::fill(dx.row(r), T{0}); // gemm adds into dx
            }
            rho::linalg::gemm<T>(dz, wt_.view(), dx);
        }
    }

    void refresh_transpose() noexcept {
        if (wt_.rows() == 0U) { // no W^T (needs input grad == false, e.g. layer 0)
            return;
        }
        rho::linalg::transpose<T>(w_.view(), wt_.view());
    }

    [[nodiscard]] rho::core::matrix_view<T> weights() noexcept {
        return w_.view();
    }
    [[nodiscard]] rho::core::matrix_view<T> grad_weights() noexcept {
        return dw_.view();
    }
    [[nodiscard]] rho::core::matrix_view<T> bias_matrix() noexcept {
        return b_.view();
    }
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
