/// @file loss.cppm
/// @brief Softmax, cross-entropy, the fused softmax + cross-entropy with its gradient,
///        and accuracy helpers.
///
/// Math (one sample i, logits z_i, true class y_i, N samples in the batch):
///
///   softmax:        p_{i,k} = exp(z_{i,k}) / sum_j exp(z_{i,j})
///   cross-entropy:  CE_i    = -log(p_{i,y_i})
///   batch loss:     L       = (1/N) * sum_i CE_i
///
/// Put softmax into CE:
///
///   CE_i = -log( exp(z_{i,y_i}) / sum_j exp(z_{i,j}) )
///        = log(sum_j exp(z_{i,j})) - z_{i,y_i}
///        = LSE(z_i) - z_{i,y_i}                      (LSE = log-sum-exp)
///
/// Numerical stability: exp(1000) overflows. Let m = max_j z_{i,j}. Then
///
///   LSE(z_i) = m + log(sum_j exp(z_{i,j} - m))
///
/// All exponents are <= 0, so exp() cannot overflow, and the largest term is 1,
/// so the sum is >= 1 and log() is finite. We compute
///
///   CE_i = (m - z_{i,y_i}) + log(sum_j exp(z_{i,j} - m))
///
/// Gradient. Let S = sum_j exp(z_{i,j}).
///
///   d LSE / d z_{i,k} = exp(z_{i,k}) / S = p_{i,k}     (the derivative of LSE is softmax)
///   d(-z_{i,y_i}) / d z_{i,k} = -[k == y_i]             (minus the one-hot vector)
///
/// So dCE_i / dz_{i,k} = p_{i,k} - [k == y_i], and for the mean loss
///
///   dL / dz_{i,k} = (p_{i,k} - [k == y_i]) / N
///
/// "Predicted minus target, divided by the batch size."
export module rho.nn.loss;

import std;

import rho.core.matrix;

export namespace rho::nn {

/// @brief Softmax of each row, with the max-subtraction trick (no overflow).
/// @param logits Input, N x C.
/// @param probabilities Output, N x C. Each row is > 0 and sums to 1.
/// @pre Same shape. logits.cols() >= 1.
template <class T>
void softmax(rho::core::matrix_view<const T> logits, rho::core::matrix_view<T> probabilities) {
    for (std::size_t i = 0; i < logits.rows(); ++i) {
        const auto z_i = logits.row(i);
        const auto p_i = probabilities.row(i);

        // For numerical stability
        T max_z = z_i[0];
        for (std::size_t j = 1; j < logits.cols(); ++j) {
            max_z = std::max(max_z, z_i[j]);
        }

        T sum{0};
        for (std::size_t j = 0; j < logits.cols(); ++j) {
            p_i[j] = std::exp(z_i[j] - max_z);
            sum += p_i[j];
        }

        const T inv_sum = T{1} / sum;
        for (std::size_t j = 0; j < logits.cols(); ++j) {
            p_i[j] *= inv_sum;
        }
    }
}

/// @brief Mean cross-entropy of probabilities: -(1/N) * sum_i log(p_{i, y_i}).
///
/// For training, use softmax_cross_entropy() instead. It is more stable: here
/// a probability that is 0 after rounding gives +inf.
///
/// @param probabilities N x C, each row sums to 1.
/// @param labels N class indices, each < C.
/// @return The mean loss.
/// @pre labels.size() == probabilities.rows() and N > 0. If not, the program stops.
template <class T, class U>
[[nodiscard]] T cross_entropy(rho::core::matrix_view<const T> probabilities,
                              std::span<const U> labels) {
    rho::core::require(probabilities.rows() > 0U && labels.size() == probabilities.rows(),
                       "cross_entropy: need N > 0 rows and N labels");

    T loss{0};
    // assume the probabilities are normalized
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        auto p_i = probabilities.row(i);
        std::size_t true_y = static_cast<std::size_t>(labels[i]);

        loss -= std::log(p_i[true_y]);
    }

    return loss / static_cast<T>(probabilities.rows());
}

/// @brief Fused softmax + cross-entropy, and optionally its gradient.
///
/// One pass over the logits gives the loss. If dlogits is not empty, the same
/// pass also writes dL/dz = (softmax(z) - onehot(y)) / N. The probabilities are
/// never stored, so this needs less memory traffic than softmax() + cross_entropy().
///
/// @param logits Raw scores, N x C.
/// @param labels N class indices, each < C.
/// @param dlogits Output N x C for the gradient, or an empty view to skip it.
/// @return Mean loss over the N samples.
/// @pre labels.size() == logits.rows(), N > 0, each label < C, and dlogits is empty
///      or N x C. If not, the program stops (a bad label would read out of bounds).
template <class T, class U>
[[nodiscard]] T softmax_cross_entropy(rho::core::matrix_view<const T> logits,
                                      std::span<const U> labels,
                                      rho::core::matrix_view<T> dlogits) noexcept {

    // (samples, classes)
    const std::size_t samples = logits.rows();
    const std::size_t classes = logits.cols();
    rho::core::require(samples > 0U && classes > 0U && labels.size() == samples,
                       "softmax_cross_entropy: need N > 0 rows, C > 0 columns and N labels");
    rho::core::require(dlogits.rows() == 0U
                           || (dlogits.rows() == samples && dlogits.cols() == classes),
                       "softmax_cross_entropy: dlogits must be empty or N x C");
    const T inv_samples = T{1} / static_cast<T>(samples);
    const bool take_grad = dlogits.rows() != 0U;
    T loss_sum{0};

    for (std::size_t i = 0; i < samples; ++i) {
        const auto z_i = logits.row(i);
        const std::size_t true_y = static_cast<std::size_t>(labels[i]);
        rho::core::require(true_y < classes, "softmax_cross_entropy: label >= number of classes");

        T max_z = z_i[0];
        for (std::size_t j = 1; j < classes; ++j) {
            max_z = std::max(max_z, z_i[j]);
        }

        T exp_sum = T{0};
        for (std::size_t j = 0; j < classes; ++j) {
            exp_sum += std::exp(z_i[j] - max_z);
        }

        loss_sum += (max_z - z_i[true_y]) + std::log(exp_sum);

        if (take_grad) {
            const auto dl_i = dlogits.row(i);
            const T inv_exp_sum = T{1} / exp_sum;

            for (std::size_t j = 0; j < classes; ++j) {
                // probability_j = softmax(z_j)
                const T probability_j = std::exp(z_i[j] - max_z) * inv_exp_sum;
                dl_i[j] = (probability_j - (j == true_y ? T{1} : T{0})) * inv_samples;
            }
        }
    }

    return loss_sum * inv_samples;
}

/// @brief Index of the largest score in row i (the predicted class).
///
/// If two scores are equal, the smaller index wins.
/// @pre i < scores.rows() and scores.cols() >= 1.
template <class T>
[[nodiscard]] std::size_t argmax_row(rho::core::matrix_view<const T> scores,
                                     std::size_t i) noexcept {
    const auto row = scores.row(i);
    return static_cast<std::size_t>(std::ranges::max_element(row) - row.begin());
}

/// @brief Number of rows where the predicted class (argmax) is equal to the label.
/// @param scores N x C (logits or probabilities).
/// @param labels N labels.
/// @pre labels.size() >= scores.rows(). If not, the program stops.
template <class T>
[[nodiscard]] std::size_t count_correct(rho::core::matrix_view<const T> scores,
                                        std::span<const std::uint8_t> labels) noexcept {
    rho::core::require(labels.size() >= scores.rows(), "count_correct: fewer labels than rows");
    std::size_t correct{};

    for (std::size_t i = 0; i < scores.rows(); ++i) {
        correct += static_cast<std::size_t>(argmax_row<T>(scores, i) == labels[i]);
    }
    return correct;
}
} // namespace rho::nn
