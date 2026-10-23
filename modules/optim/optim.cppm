/// @file optim.cppm
/// @brief Optimizers. Now only plain stochastic gradient descent (SGD).
export module rho.optim;

import std;

export import rho.nn.mlp;

export namespace rho::optim {

/// @brief Stochastic gradient descent: w = w - lr * dw for each parameter.
///
/// Time: O(P), with P = number of parameters. Extra memory: none.
template <class T>
class sgd {
  public:
    sgd() = default;

    /// @param lr Learning rate. Must be > 0.
    explicit sgd(T lr) noexcept : lr_{lr} {}

    /// @brief Do one update step: value -= lr * grad, for each parameter.
    ///
    /// The gradients must be batch means (mlp::forward_backward() gives batch means).
    /// @param params Parameters and gradients, from mlp::parameters().
    void step(std::span<const rho::nn::param_ref<T>> params) noexcept {
        for (const auto &p : params) {
            for (std::size_t i = 0; i < p.value.rows(); ++i) {
                const auto w = p.value.row(i);
                const auto g = p.grad.row(i);
                for (std::size_t j = 0; j < w.size(); ++j) {
                    w[j] -= lr_ * g[j];
                }
            }
        }
    }

    /// @return The learning rate.
    [[nodiscard]] T learning_rate() const noexcept {
        return lr_;
    }

    /// @param lr The new learning rate.
    void set_learning_rate(T lr) noexcept {
        lr_ = lr;
    }

  private:
    T lr_{};
};

} // namespace rho::optim
