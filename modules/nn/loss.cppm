export module rho.nn.loss;

import std;

import rho.core.matrix;

export namespace rho::nn {

	// softmax(z_i) = exp(z_i) / Sum (exp(z_j)) from j=1 to n
	//
	// Cross-Entropy (CE) = -Sum (y_i * log(\hat{y}_i)) from i=1 to n
	// where y are the true label vector elements (one-hot encoded), and \hat{y}
	// are the predicted probability vector elements
	//
	// We get loss by computing the sum of cross-entropy of the softmax(logits) and
	// dividing by the sample size, where logits are raw scores from the model, 
	// and probabilities = softmax(logits)
	// 
	// Thus, loss_i = cross-entropy_i(probabilities, true labels)
	//
	// And for a batch of size N we get the mean-reduced loss: 
	//              loss = sum( CE_i() ) / N     from i to N
	//
	//
	// Then for a single sample: CE_i = -log(probability_{i,y_i})
	// Substituting softmax in,
	//              CE_i = -log(exp(z_{i,y_i} / sum(exp(z_{i,j}))))
	//				CE_i = log(sum(exp(z_{i,j}))) - log(exp(z_{i,y_i})))
	//				CE_i = log(sum(exp(z_{i,j}))) - z_{i, y_i}
	//	            CE_i = LSE(Z_i) - z_{i, y_i}
	//
	// We will also do a to trick for numerical stability:
	//	LSE(Z_i) = log(sum(exp(z_{i,j})))
	//	LSE(Z_i) = log(exp(m) * sum(exp(z_{i,j} - m)))
	//	LSE(Z_i) = m + log(sum(exp(z_{i,j} - m)))
	//	
	//	where m = max_i (z_i) = max_z , largest z element in vector Z
	//
	//  And change the order slightly, to make log(sum()) more significant
	//  numerically
	//
	//   CE_i = (max_z - z_i{i, y_i}) + log(sum(exp(z_{i,j} - m)))
	//
	// -------------------------------------------------------------------
	// --- Gradient ------------------------------------------------------
	// -------------------------------------------------------------------
	// Starting from 
	//	
	//	            CE_i = LSE(Z_i) - z_{i, y_i}
	//
	//	we differentiate the two terms separetely.
	//
	//	1. Let S = sum_j (exp(z{i,j}), so LSE = log(S), then
	//
	//			d/dz_{i,k} LSE(Z_i) = (1/S) * dS/dz_{i,k}
	//								= (1/S) * exp(z_{i,k})
	//								= exp(z_{i,k}) / (sum_j(exp(z_{i,j})))
	//								= p_{i,k}
	//
	//	   So, the derivative of log-sum-exp is actually softmax.
	//
	//	2. This is simply the true-class logit:
	//
	//			d/dz{i,k} ( -z_{i,y_i} ) = -[k == y_i]
	//
	//	   i.e. one-hot vector y
	//
	//	Thus we get:
	//
	//		dCE_i / dz_{i,k} = p_{i,k} - [k == y_i]
	//
	//  And since batch loss is the mean, L = (1/N) sum_i ( CE_i )
	//
	//					dL/dz_ij = (p_j − [j == y_i]) / N
	//
	//	"predicted minus target, averaged over the batch"
	//
	//	So we need to find dCE_i/dz{i,k} for one row, then scale by 1/N 
	//

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

	template <class T, class U>
		void cross_entropy(rho::core::matrix_view<const T> probabilities, std::span<const U> labels) {


			T loss{0};
			// assume the probabilities are normalized
			for (std::size_t i = 0; i < probabilities.rows(); ++i) {
				auto p_i = probabilities.row(i);
				std::size_t true_y = static_cast<std::size_t>(labels[i]);

				loss -= std::log(p_i[true_y]);
			}

			return loss / static_cast<T>(probabilities.rows());

		}

	// fused cross_entropy( softmax ) for efficiency
	// Return: mean loss over samples
	template <class T, class U>
		[[nodiscard]] T softmax_cross_entropy(
				rho::core::matrix_view<const T> logits,
				std::span< const U> labels,
				rho::core::matrix_view<T> dlogits) noexcept {

			// (samples, classes)
			const std::size_t samples = logits.rows();
			const std::size_t classes = logits.cols();
			const T inv_samples = T{1} / static_cast<T>(samples);
			const bool take_grad = dlogits.rows() != 0U;
			T loss_sum {0};



			for (std::size_t i = 0; i < samples; ++i) {
				const auto z_i = logits.row(i);
				const std::size_t true_y = static_cast<std::size_t>(labels[i]);

				T max_z = z_i[0];
				for (std::size_t j = 1; j < classes; ++j) {
					max_z = std::max(max_z, z_i[j]);
				}

				T exp_sum = T{0};
				for (std::size_t j = 0; j < classes; ++j) {
					exp_sum += std::exp(z_i[j] - max_z);
				}

				loss_sum += (max_z - z_i[true_y]) + std::log(exp_sum);

				if(take_grad) {
					const auto dl_i = dlogits.row(i);
					const T inv_exp_sum = T{1} / exp_sum;

					for(std::size_t j = 0; j < classes; ++j) {
						// probability_j = softmax(z_j)
						const T probability_j = std::exp(z_i[j] - max_z) * inv_exp_sum;
						dl_i[j] = (probability_j - (j == true_y ? T{1} : T{0})) * inv_samples;
					}
				}

			}

			return loss_sum * inv_samples;
		}
} // namespace rho::nn
