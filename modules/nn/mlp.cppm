
export module rho.nn.mlp;

import std;

export import rho.nn.dense;
export import rho.nn.activation;
export import rho.nn.loss;

export namespace rho::nn {

template <class T>
struct param_ref {
    rho::core::matrix_view<T> value;
    rho::core::matrix_view<T> grad;
};

template <class T>
class mlp {
  public:
    mlp(std::span<const std::size_t> sizes, std::size_t max_batch, std::uint64_t seed)
        : max_batch_{max_batch} {
        const std::size_t n_layers = sizes.size() - 1U;

        layers_.reserve(n_layers);
        for (std::size_t l = 0; l < n_layers; ++l) {
            layers_.emplace_back(sizes[l], sizes[l + 1U], l != 0U);
        }

        rho::core::xoshiro256pp rng{seed};
        for (std::size_t l = 0; l < n_layers; ++l) {
            const bool is_output = (l + 1U == n_layers);
            layers_[l].initialize(is_output ? init_scheme::glorot_uniform : init_scheme::he_normal,
                                  rng);
        }

        for (std::size_t l = 0; l < n_layers; ++l) {
            z_.emplace_back(max_batch, sizes[l + 1U]);
            d_.emplace_back(max_batch, sizes[l + 1U]);
            dx_.emplace_back(l == 0U ? rho::core::matrix<T>{}
                                     : rho::core::matrix<T>{max_batch, sizes[l]});
        }
        for (std::size_t l = 0; l + 1U < n_layers; ++l) {
            a_.emplace_back(max_batch, sizes[l + 1U]);
        }

        params_.reserve(2U * n_layers);
        for (auto &layer : layers_) {
            params_.push_back({layer.weights(), layer.grad_weights()});
            params_.push_back({layer.bias_matrix(), layer.grad_bias_matrix()});
        }
    }

    [[nodiscard]] std::span<param_ref<T>> parameters() noexcept {
        return params_;
    }

    [[nodiscard]] std::vector<std::size_t> topology() const {
        std::vector<std::size_t> out;
        out.push_back(layers_.front().n_in());
        for (const auto &layer : layers_) {
            out.push_back(layer.n_out());
        }
        return out;
    }

    void after_update() noexcept {
        for (auto &layer : layers_) {
            layer.refresh_transpose();
        }
    }

    [[nodiscard]] rho::core::matrix_view<const T>
    forward(rho::core::matrix_view<const T> x) noexcept {
        const std::size_t b = x.rows();
        batch_ = b;

        // Hidden layers: dense, ReLU
        rho::core::matrix_view<const T> input = x;
        for (std::size_t l = 0; l + 1U < layers_.size(); ++l) {
            const auto z = first_rows(z_[l], b);
            const auto a = first_rows(a_[l], b);
            layers_[l].forward(input, z);
            relu_forward<T>(z, a);
            input = a;
        }
        // Output layer: dense
        const auto logits = first_rows(z_.back(), b);
        layers_.back().forward(input, logits);
        return logits;
    }

    [[nodiscard]] rho::core::matrix_view<const T> logits() noexcept {
        return first_rows(z_.back(), batch_);
    }

    // Forward pass and loss
    [[nodiscard]] T loss(rho::core::matrix_view<const T> x,
                         std::span<const std::uint8_t> labels) noexcept {
        return softmax_cross_entropy_loss<T>(forward(x), labels);
    }

    // Forward pass, loss, and backpropagation
    [[nodiscard]] T forward_backward(rho::core::matrix_view<const T> x,
                                     std::span<const std::uint8_t> labels) noexcept {
        const auto logits = forward(x);
        const std::size_t b = batch_;

        const T loss_value = softmax_cross_entropy<T>(logits, labels, first_rows(d_.back(), b));

        for (std::size_t l = layers_.size() - 1U; l > 0U; --l) {
            const auto dx = first_rows(dx_[l], b);
            layers_[l].backward(first_rows(a_[l - 1U], b), first_rows(d_[l], b), dx);
            relu_backward<T>(first_rows(z_[l - 1U], b), dx, first_rows(d_[l - 1U], b));
        }
        // bottom layer: weight gradients
        layers_.front().backward(x, first_rows(d_.front(), b), rho::core::matrix_view<T>{});
        return loss_value;
    }

  private:
    [[nodiscard]] static rho::core::matrix_view<T> first_rows(rho::core::matrix<T> &m,
                                                              std::size_t n) noexcept {
        return rho::core::matrix_view<T>{m.data(), n, m.cols(), m.stride()};
    }

    std::vector<dense<T>> layers_;
    std::vector<rho::core::matrix<T>> z_;  // each layer's output, before ReLU
    std::vector<rho::core::matrix<T>> a_;  // after ReLU, hidden layers only
    std::vector<rho::core::matrix<T>> d_;  // loss gradient at each layers output
    std::vector<rho::core::matrix<T>> dx_; // loss gradient at each layers input
    std::vector<param_ref<T>> params_;
    std::size_t max_batch_ = 0U;
    std::size_t batch_ = 0U;
};

} // namespace rho::nn
