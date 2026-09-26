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

    // z = x * W + b. 
    void forward(){}
    void backward(){}

	void refresh_transpose() noexcept {
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

