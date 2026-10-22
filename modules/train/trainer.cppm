export module rho.train;

import std;

export import rho.io.dataset;
export import rho.optim;

export namespace rho::train {

struct eval_result {
    double loss{};         // mean cross-entropy
    std::size_t correct{}; // samples where argmax(logits) == label
    std::size_t total{};

    [[nodiscard]] double accuracy() const noexcept {
        return total == 0U ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
    }
};

struct train_config {
    std::size_t epochs{10};
    float lr_decay{1.0f}; // after each epoch: lr = lr * lr_decay
};

struct epoch_report {
    std::size_t epoch{};      // starts at 1
    float learning_rate{};    // the rate used in this epoch
    double train_loss{};      // mean loss of the training batches, before each update
    double train_accuracy{};  // same, as accuracy
    eval_result validation{}; // after the epoch
    std::chrono::duration<double> seconds{};
};

// Run the network on all batches of `data` (no training): mean loss and accuracy.
[[nodiscard]] eval_result evaluate(rho::nn::mlp<float> &net, rho::io::batcher &data) {
    eval_result out;
    double loss_sum{0.0};

    for (const rho::io::batch &b : data.epoch()) {
        loss_sum += static_cast<double>(net.loss(b.x, b.y)) * static_cast<double>(b.x.rows());
        out.correct += rho::nn::count_correct<float>(net.logits(), b.y);
        out.total += b.x.rows();
    }
    out.loss = out.total == 0U ? 0.0 : loss_sum / static_cast<double>(out.total);
    return out;
}

// One pass over the training data: forward, backward and one SGD step for each batch.
// Returns loss and accuracy of the batches. Each batch is measured before its update.
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
        net.after_update();
    }
    out.loss = out.total == 0U ? 0.0 : loss_sum / static_cast<double>(out.total);
    return out;
}

// Train for config.epochs epochs, starting with the learning rate of `opt`.
// After each epoch, evaluate on `validation` and call on_epoch(report).
// An empty validation batcher gives an empty result.
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
