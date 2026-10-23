/// @file mlp.cppm
/// @brief Multilayer perceptron: dense layers with ReLU between them, and a
///        softmax cross-entropy loss at the end.
export module rho.nn.mlp;

import std;

export import rho.nn.dense;
export import rho.nn.activation;
export import rho.nn.loss;

export namespace rho::nn {

/// @brief One trainable parameter and its gradient, as views.
///
/// The optimizer uses it: value -= lr * grad. Both views have the same shape.
template <class T>
struct param_ref {
    rho::core::matrix_view<T> value;
    rho::core::matrix_view<T> grad;
};

/// @brief Multilayer perceptron (MLP) for classification.
///
/// For sizes {784, 256, 128, 10} the network is
/// @code
/// x -> dense(784, 256) -> ReLU -> dense(256, 128) -> ReLU -> dense(128, 10) -> logits
/// @endcode
///
/// All activation and gradient buffers are allocated one time in the constructor,
/// for max_batch rows. forward() and forward_backward() do not allocate memory.
///
/// @tparam T Element type (float or double).
template <class T>
class mlp {
  public:
    /// @brief Make the network and set random start weights.
    ///
    /// Hidden layers use He initialization (for ReLU). The output layer uses Glorot.
    ///
    /// @param sizes Layer widths: {inputs, hidden..., outputs}.
    /// @param max_batch The largest batch that forward() will get.
    /// @param seed Seed for the weights. The same seed gives the same network.
    /// @throws std::invalid_argument if sizes.size() < 2, a size is 0, or max_batch is 0.
    mlp(std::span<const std::size_t> sizes, std::size_t max_batch, std::uint64_t seed)
        : max_batch_{check_topology(sizes, max_batch)} {
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

    /// @return All parameters (W and b of each layer) with their gradients, for the optimizer.
    [[nodiscard]] std::span<param_ref<T>> parameters() noexcept {
        return params_;
    }

    /// @return The layer widths, for example {784, 256, 128, 10}.
    [[nodiscard]] std::vector<std::size_t> topology() const {
        std::vector<std::size_t> out;
        out.push_back(layers_.front().n_in());
        for (const auto &layer : layers_) {
            out.push_back(layer.n_out());
        }
        return out;
    }

    /// @return The largest batch size that the buffers can hold.
    [[nodiscard]] std::size_t max_batch() const noexcept {
        return max_batch_;
    }

    /// @brief Forward pass.
    /// @param x Input batch, B x sizes[0], B <= max_batch().
    /// @return View of the logits (B x outputs). It is valid until the next forward().
    /// @pre x.rows() <= max_batch() and x.cols() == sizes[0]. If not, the program stops.
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

    /// @return The logits of the last forward pass.
    [[nodiscard]] rho::core::matrix_view<const T> logits() noexcept {
        return first_rows(z_.back(), batch_);
    }

    /// @brief Forward pass and mean loss. No gradients.
    /// @param x Input batch, B x sizes[0].
    /// @param labels B labels, each < number of outputs.
    /// @return Mean softmax cross-entropy loss.
    /// @pre B > 0 and labels.size() == B. If not, the program stops.
    [[nodiscard]] T loss(rho::core::matrix_view<const T> x,
                         std::span<const std::uint8_t> labels) noexcept {
        return softmax_cross_entropy<T>(forward(x), labels, rho::core::matrix_view<T>{});
    }

    /// @brief Forward pass, loss, and backpropagation.
    ///
    /// After the call, the gradient views in parameters() hold dL/dW and dL/db
    /// for this batch (batch means). logits() holds the logits of this batch.
    ///
    /// Backpropagation, from the top layer l = L-1 down to 1:
    ///   d_[l]      = dL/dz_l   (for the top layer: from softmax_cross_entropy)
    ///   dx_[l]     = d_[l] * W_l^T            (in dense::backward)
    ///   d_[l - 1]  = dx_[l] * ReLU'(z_{l-1})  (in relu_backward)
    /// The first layer computes only dW and db.
    ///
    /// @param x Input batch, B x sizes[0].
    /// @param labels B labels, each < number of outputs.
    /// @return Mean loss of the batch (before the update).
    /// @pre B > 0 and labels.size() == B. If not, the program stops.
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
    /// Check the constructor arguments. It runs in the member initializer list,
    /// so the constructor body never sees a bad topology.
    /// @return max_batch.
    static std::size_t check_topology(std::span<const std::size_t> sizes, std::size_t max_batch) {
        if (sizes.size() < 2U) {
            throw std::invalid_argument("mlp: sizes needs at least an input and an output size");
        }
        if (std::ranges::contains(sizes, std::size_t{0}) || max_batch == 0U) {
            throw std::invalid_argument("mlp: layer sizes and max_batch must be larger than 0");
        }
        return max_batch;
    }

    /// The first n rows of m. Stops the program if n > m.rows().
    [[nodiscard]] static rho::core::matrix_view<T> first_rows(rho::core::matrix<T> &m,
                                                              std::size_t n) noexcept {
        return m.view().row_range(0, n);
    }

    std::vector<dense<T>> layers_;
    std::vector<rho::core::matrix<T>> z_;  // each layer's output, before ReLU
    std::vector<rho::core::matrix<T>> a_;  // after ReLU, hidden layers only
    std::vector<rho::core::matrix<T>> d_;  // loss gradient at each layers output
    std::vector<rho::core::matrix<T>> dx_; // loss gradient at each layers input
    std::vector<param_ref<T>> params_;     // views into layers_ (heap data, stable on move)
    std::size_t max_batch_ = 0U;
    std::size_t batch_ = 0U;               // rows of the last forward()
};

} // namespace rho::nn
