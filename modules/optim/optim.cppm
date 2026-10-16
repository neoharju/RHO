export module rho.optim;

import std;

export import rho.nn.mlp;

export namespace rho::optim {

// stochastic gradient descent: w -= lr * dw 
template <class T>
class sgd {
  public:
    sgd() = default;
    explicit sgd(T lr) noexcept : lr_{lr} {}

    // the gradients are batch means
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

  private:
    T lr_{};
};

} // namespace rho::optim
