import std;

import rho.train;

namespace app {

	// namespace fs; using fs?
	namespace fs = std::filesystem;

	// Setting
	constexpr std::string_view data_dir{"data"};
	// e.g. sequential Dense Network in pytorch
	constexpr std::array<std::size_t, 4> layer_sizes{
		rho::io::pixels_per_image, 
			256, 
			128, 
			rho::io::num_classes
	};

	constexpr std::size_t epochs{10};
	constexpr std::size_t batch_size{64};
	constexpr float learning_rate{0.1f};
	constexpr float lr_decay{0.95f};               // learning rate factor after each epoch
	constexpr std::uint32_t validation_size{5000}; // training samples kept for validation
	constexpr std::uint64_t seed{42};

	static_assert(epochs > 0U && batch_size > 0U);
	static_assert(learning_rate > 0.0f && lr_decay > 0.0f);

	// Error types
	struct DatasetError {
		rho::io::Error error; // cannot read an MNIST file, or it has bad contents
	};
	struct ValidationSizeError {
		std::uint32_t validation;
		std::uint32_t samples;
	};
	using Error = std::variant<DatasetError, ValidationSizeError>;

	void display_error(const Error &error) {
		constexpr auto visitor = rho::io::match{
			[](const DatasetError &e) { rho::io::display_error(e.error); },
				[](const ValidationSizeError &e) {
					std::println(std::cerr,
							"{} Cannot keep {} samples for validation: the training set has {}.",
							rho::io::FAILURE,
							e.validation,
							e.samples);
				},
		};
		std::visit(visitor, error);
	}

	// Load data from data-folder
	[[nodiscard]] std::expected<rho::io::dataset, Error> load(std::string_view name) {
		const fs::path dir{data_dir};
		return rho::io::dataset::load(dir / std::format("{}-images-idx3-ubyte", name),
				dir / std::format("{}-labels-idx1-ubyte", name))
			.transform_error([](rho::io::Error e) -> Error { return DatasetError{std::move(e)}; });
	}

	[[nodiscard]] std::expected<void, Error> run() {
		const auto train_set = load("train");
		if (!train_set) {
			return std::unexpected(train_set.error());
		}
		const auto test_set = load("t10k");
		if (!test_set) {
			return std::unexpected(test_set.error());
		}
		if (validation_size >= train_set->size()) {
			return std::unexpected(ValidationSizeError{validation_size, train_set->size()});
		}

		rho::core::xoshiro256pp seeds{seed};
		rho::core::xoshiro256pp split_rng{seeds()};
		const std::uint64_t net_seed = seeds();
		const std::uint64_t shuffle_seed = seeds();

		auto ids = rho::io::train_validation_split(train_set->size(), validation_size, split_rng);
		const auto norm = rho::io::compute_normalization(*train_set, ids.train);

		std::println("Network {}: {} train, {} validation, {} test samples",
				layer_sizes,
				ids.train.size(),
				ids.validation.size(),
				test_set->size());

		using rho::io::sample_order;
		rho::io::batcher train{
			*train_set, std::move(ids.train), norm, batch_size, sample_order::shuffled, shuffle_seed};
		rho::io::batcher validation{
			*train_set, std::move(ids.validation), norm, batch_size, sample_order::sequential};
		rho::io::batcher test{
			*test_set, rho::io::all_ids(*test_set), norm, batch_size, sample_order::sequential};

		rho::nn::mlp<float> net{layer_sizes, batch_size, net_seed};
		rho::optim::sgd<float> sgd{learning_rate};

		// train
		rho::train::fit(net,
				sgd,
				train,
				validation,
				{.epochs = epochs, .lr_decay = lr_decay},
				[](const rho::train::epoch_report &r) {
				std::println("epoch {:>2}  lr {:.4f}  train loss {:.4f} acc {:.2f}%  "
						"val loss {:.4f} acc {:.2f}%  {:.1f}s",
						r.epoch,
						r.learning_rate,
						r.train_loss,
						100.0 * r.train_accuracy,
						r.validation.loss,
						100.0 * r.validation.accuracy(),
						r.seconds.count());
				});

		// evaluate
		const auto result = rho::train::evaluate(net, test);
		std::println("Test: loss {:.4f}, accuracy {:.2f}% ({} / {})",
				result.loss,
				100.0 * result.accuracy(),
				result.correct,
				result.total);
		return {};
	}

} // namespace app

int main() {
	if (const auto result = app::run(); !result) {
		app::display_error(result.error());
		return 1;
	}
}
