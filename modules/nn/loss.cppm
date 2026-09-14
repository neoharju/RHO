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
	//	            CE_i = LSE(Z) - z_{i, y_i}
	//
	// We will also do a to trick for numerical stability:
	//	LSE(Z) = log(sum(exp(z_{i,j})))
	//	LSE(Z) = log(exp(m) * sum(exp(z_{i,j} - m)))
	//	LSE(Z) = m + log(sum(exp(z_{i,j} - m)))
	//	
	//	where m = max_i (z_i) , largest z element in vector Z
	//

	template <class T>
		static inline void softmax(rho::core::matrix_view<const T> logits, rho::core::matrix_view<T> probabilities) {
			for (std::size_t i = 0; i < logits.rows(); ++i) {
				const auto z_i = logits.row(i);
				const auto p_i = probabilities.row(i);

				// For numerical stability
				T max_z = z_i[0];
				for (std::size_t j = 1; j < logits.cols(); ++j) {
					max_z = std::max(max_z, z_i[j])
				}

				T sum{0};
				for (std::size_t j = 0; j < logits.cols(); ++j) {
					p_i[j] = std::exp(z_i[j] - max_z)
						sum += p_i[j];
				}

				const T inv_sum = T{1} / sum;
				for (std::size_t j = 0; j < logits.cols(); ++j) {
					p_i[j] *= inv_sum;
				}
			}
		}

	template <class T, class U>
		static inline void cross_entropy(rho::core::matrix_view<const T> probabilities, std::span<const U> labels) {
		
			
			T loss{0};
			// assume the probabilities are normalized
			for (std::size_t i = 0; i < probabilities.rows(); ++i) {
				auto p_i = probabilities.row(i);
				std::size_t true_y = static_cast<std::size_t>(labels[i]);

				loss -= std::log(p_i[true_y])
			}

			return loss / static_cast<T>(probabilities.rows());
		
		}
} // namespace rho::nn
