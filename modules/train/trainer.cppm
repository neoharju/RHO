/// @file trainer.cppm
/// @brief Training loop and evaluation.
export module rho.train;

import std;

export import rho.io.dataset;
export import rho.optim;

export namespace rho::train {

/// Result of one pass over a data set.
struct eval_result {
    double loss{};         ///< Mean cross-entropy over all samples.
    std::size_t correct{}; ///< Samples where argmax(logits) == label.
    std::size_t total{};   ///< Number of samples.

    /// @return correct / total, or 0 if total is 0.
    [[nodiscard]] double accuracy() const noexcept {
        return total == 0U ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
    }
};

/// Settings for fit().
struct train_config {
    std::size_t epochs{10}; ///< Number of passes over the training data.
    float lr_decay{1.0f};   ///< After each epoch: lr = lr * lr_decay. 1 means constant lr.
};

/// What fit() gives to the callback after each epoch.
struct epoch_report {
    std::size_t epoch{};      ///< Epoch number, from 1.
    float learning_rate{};    ///< The learning rate of this epoch.
    double train_loss{};      ///< Mean loss of the training batches, each measured before its update.
    double train_accuracy{};  ///< Accuracy of the training batches, measured the same way.
    eval_result validation{}; ///< Result on the validation data after the epoch.
    std::chrono::duration<double> seconds{}; ///< Wall time of the epoch (training + validation).
};

/// @brief Run the network on all batches of `data`, without training.
/// @param net The network. Its weights do not change.
/// @param data The batches to evaluate (use sample_order::sequential).
/// @return Mean loss and accuracy over all samples. Empty data gives {0, 0, 0}.
/// @pre data.batch_size() <= net.max_batch().
[[nodiscard]] eval_result evaluate(rho::nn::mlp<float> &net, rho::io::batcher &data) {
    eval_result out;
    double loss_sum{0.0};

    for (const rho::io::batch &b : data.epoch()) {
        // weight by the batch size, because the last batch can be smaller
        loss_sum += static_cast<double>(net.loss(b.x, b.y)) * static_cast<double>(b.x.rows());
        out.correct += rho::nn::count_correct<float>(net.logits(), b.y);
        out.total += b.x.rows();
    }
    out.loss = out.total == 0U ? 0.0 : loss_sum / static_cast<double>(out.total);
    return out;
}

/// @brief One pass over the training data: forward, backward and one SGD step per batch.
///
/// The loss and accuracy of each batch are measured before its update, so they
/// cost no extra forward pass.
///
/// @param net The network to train.
/// @param opt The optimizer.
/// @param data The training batches (use sample_order::shuffled).
/// @return Mean loss and accuracy over the training batches.
/// @pre data.batch_size() <= net.max_batch().
[[nodiscard]] eval_result
train_epoch(rho::nn::mlp<float> &net, rho::optim::sgd<float> &opt, rho::io::batcher &data) {
    eval_result out;
    double loss_sum{0.0};

    for (const rho::io::batch &b : data.epoch()) {
        const float batch_loss = net.forward_backward(b.x, b.y);
        // forward_backward keeps the logits; read them before the update
        out.correct += rho::nn::count_correct<float>(net.logits(), b.y);
        out.total += b.x.rows();
        loss_sum += static_cast<double>(batch_loss) * static_cast<double>(b.x.rows());

        opt.step(net.parameters());
    }
    out.loss = out.total == 0U ? 0.0 : loss_sum / static_cast<double>(out.total);
    return out;
}

/// @brief Train for config.epochs epochs.
///
/// The first epoch uses the learning rate of `opt`. After each epoch the function
/// evaluates on `validation`, calls on_epoch(report), and multiplies the learning
/// rate by config.lr_decay. An empty validation batcher gives an empty result.
///
/// @param net The network to train.
/// @param opt The optimizer. Its learning rate changes.
/// @param train Training batches.
/// @param validation Validation batches.
/// @param config Number of epochs and learning-rate decay.
/// @param on_epoch Callback, called one time per epoch with an epoch_report.
/// @pre Both batch sizes <= net.max_batch(). Learning rate and lr_decay > 0.
///      If not, the program stops.
template <std::invocable<const epoch_report &> OnEpoch>
void fit(rho::nn::mlp<float> &net,
         rho::optim::sgd<float> &opt,
         rho::io::batcher &train,
         rho::io::batcher &validation,
         const train_config &config,
         OnEpoch &&on_epoch) {
    rho::core::require(train.batch_size() <= net.max_batch()
                           && validation.batch_size() <= net.max_batch(),
                       "fit: batch_size is larger than the network's max_batch");
    float lr = opt.learning_rate();
    rho::core::require(lr > 0.0f && config.lr_decay > 0.0f,
                       "fit: learning rate and lr_decay must be larger than 0");

    for (std::size_t e = 1; e <= config.epochs; ++e) {
        const auto start = std::chrono::steady_clock::now();
        opt.set_learning_rate(lr);

        const eval_result trained = train_epoch(net, opt, train);
        const epoch_report report{
            .epoch = e,
            .learning_rate = lr,
            .train_loss = trained.loss,
            .train_accuracy = trained.accuracy(),
            .validation = evaluate(net, validation),
            .seconds = std::chrono::steady_clock::now() - start,
        };
        on_epoch(report);
        lr *= config.lr_decay;
    }
}

} // namespace rho::train
