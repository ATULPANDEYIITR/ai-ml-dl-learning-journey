#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * TRANSFER LEARNING
 *
 * C++17 implementation of a transfer-learning engine.
 *
 * Features:
 * - Pretrained model representation
 * - Feature extraction
 * - Frozen and trainable layers
 * - Selective fine-tuning
 * - Target classification head
 * - Dataset compatibility checks
 * - Learning-rate policies
 * - Binary cross-entropy
 * - Evaluation
 * - Experiment state management
 *
 * Compile:
 * g++ -std=c++17 -O2 transfer_learning.cpp -o transfer_learning
 */

namespace transfer {

enum class LayerType {
    Convolution,
    Dense,
    Normalization,
    Activation
};

enum class TransferStrategy {
    FeatureExtraction,
    FineTuning
};

enum class ExperimentState {
    Created,
    Validated,
    Training,
    Evaluated,
    Approved,
    Rejected,
    Failed
};

std::string toString(ExperimentState state) {
    switch (state) {
        case ExperimentState::Created: return "CREATED";
        case ExperimentState::Validated: return "VALIDATED";
        case ExperimentState::Training: return "TRAINING";
        case ExperimentState::Evaluated: return "EVALUATED";
        case ExperimentState::Approved: return "APPROVED";
        case ExperimentState::Rejected: return "REJECTED";
        case ExperimentState::Failed: return "FAILED";
    }

    throw std::logic_error("Unknown experiment state.");
}

std::string toString(TransferStrategy strategy) {
    return strategy == TransferStrategy::FeatureExtraction
        ? "FEATURE_EXTRACTION"
        : "FINE_TUNING";
}

// -----------------------------------------------------
// Numerical functions
// -----------------------------------------------------

double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }

    const double expValue = std::exp(value);
    return expValue / (1.0 + expValue);
}

double binaryCrossEntropy(double probability, int target) {
    constexpr double epsilon = 1e-12;

    probability = std::clamp(
        probability,
        epsilon,
        1.0 - epsilon
    );

    return -(
        target * std::log(probability) +
        (1 - target) * std::log(1.0 - probability)
    );
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

// -----------------------------------------------------
// Domain structures
// -----------------------------------------------------

struct Dataset {
    std::string name;
    std::string domain;
    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    std::size_t dimension() const {
        return features.empty() ? 0 : features.front().size();
    }

    void validate() const {
        require(!features.empty(), "Dataset cannot be empty.");

        require(
            features.size() == labels.size(),
            "Feature and label counts differ."
        );

        require(dimension() > 0, "Feature dimension cannot be zero.");

        for (const auto& row : features) {
            require(
                row.size() == dimension(),
                "Inconsistent feature dimensions."
            );

            for (double value : row) {
                require(
                    std::isfinite(value),
                    "Non-finite feature value."
                );
            }
        }

        for (int label : labels) {
            require(
                label == 0 || label == 1,
                "Binary classification labels must be 0 or 1."
            );
        }
    }
};

struct Layer {
    std::string name;
    LayerType type;
    std::size_t inputDimension;
    std::size_t outputDimension;
    bool trainable = false;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    void initialize(std::mt19937& generator) {
        std::normal_distribution<double> distribution(0.0, 0.1);

        weights.assign(
            outputDimension,
            std::vector<double>(inputDimension)
        );

        biases.assign(outputDimension, 0.0);

        for (auto& row : weights) {
            for (double& weight : row) {
                weight = distribution(generator);
            }
        }
    }

    void validate() const {
        require(inputDimension > 0, "Invalid layer input dimension.");
        require(outputDimension > 0, "Invalid layer output dimension.");

        require(
            weights.size() == outputDimension,
            "Invalid weight matrix."
        );

        require(
            biases.size() == outputDimension,
            "Invalid bias vector."
        );

        for (const auto& row : weights) {
            require(
                row.size() == inputDimension,
                "Invalid weight matrix dimensions."
            );
        }
    }
};

// -----------------------------------------------------
// Pretrained model
// -----------------------------------------------------

class PretrainedModel {
private:
    std::string name_;
    std::string sourceDomain_;
    std::vector<Layer> layers_;
    std::size_t inputDimension_;

public:
    PretrainedModel(
        std::string name,
        std::string sourceDomain,
        std::size_t inputDimension
    )
        : name_(std::move(name)),
          sourceDomain_(std::move(sourceDomain)),
          inputDimension_(inputDimension) {

        require(!name_.empty(), "Model name is required.");
        require(!sourceDomain_.empty(), "Source domain is required.");
        require(inputDimension_ > 0, "Input dimension must be positive.");

        std::mt19937 generator(42);

        Layer first{
            "feature_layer_1",
            LayerType::Dense,
            inputDimension_,
            8
        };

        first.initialize(generator);
        layers_.push_back(first);

        Layer second{
            "feature_layer_2",
            LayerType::Dense,
            8,
            4
        };

        second.initialize(generator);
        layers_.push_back(second);

        for (auto& layer : layers_) {
            layer.validate();
        }
    }

    const std::string& name() const {
        return name_;
    }

    const std::string& sourceDomain() const {
        return sourceDomain_;
    }

    const std::vector<Layer>& layers() const {
        return layers_;
    }

    std::vector<Layer>& mutableLayers() {
        return layers_;
    }

    std::size_t inputDimension() const {
        return inputDimension_;
    }

    std::size_t parameterCount() const {
        std::size_t count = 0;

        for (const auto& layer : layers_) {
            count += layer.outputDimension * layer.inputDimension;
            count += layer.outputDimension;
        }

        return count;
    }

    std::vector<double> extractFeatures(
        const std::vector<double>& input
    ) const {
        require(
            input.size() == inputDimension_,
            "Input dimension mismatch."
        );

        std::vector<double> activation = input;

        for (const auto& layer : layers_) {
            std::vector<double> next(layer.outputDimension);

            for (std::size_t i = 0; i < layer.outputDimension; ++i) {
                double value = layer.biases[i];

                for (std::size_t j = 0; j < layer.inputDimension; ++j) {
                    value += layer.weights[i][j] * activation[j];
                }

                next[i] = std::tanh(value);
            }

            activation = std::move(next);
        }

        return activation;
    }

    void configureTrainability(
        TransferStrategy strategy,
        std::size_t frozenLayers
    ) {
        require(
            frozenLayers <= layers_.size(),
            "Frozen layer count exceeds backbone depth."
        );

        if (strategy == TransferStrategy::FeatureExtraction) {
            require(
                frozenLayers == layers_.size(),
                "Feature extraction must freeze all backbone layers."
            );
        }

        for (std::size_t i = 0; i < layers_.size(); ++i) {
            layers_[i].trainable =
                strategy == TransferStrategy::FineTuning &&
                i >= frozenLayers;
        }
    }
};

// -----------------------------------------------------
// Target classification head
// -----------------------------------------------------

class ClassificationHead {
private:
    std::vector<double> weights_;
    double bias_ = 0.0;

public:
    explicit ClassificationHead(std::size_t inputDimension)
        : weights_(inputDimension, 0.0) {

        require(
            inputDimension > 0,
            "Classification head requires input features."
        );
    }

    double probability(const std::vector<double>& features) const {
        require(
            features.size() == weights_.size(),
            "Classification head dimension mismatch."
        );

        double value = bias_;

        for (std::size_t i = 0; i < weights_.size(); ++i) {
            value += weights_[i] * features[i];
        }

        return sigmoid(value);
    }

    int predict(const std::vector<double>& features) const {
        return probability(features) >= 0.5 ? 1 : 0;
    }

    void update(
        const std::vector<double>& features,
        int target,
        double learningRate
    ) {
        const double error = probability(features) - target;

        for (std::size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= learningRate * error * features[i];
        }

        bias_ -= learningRate * error;
    }

    const std::vector<double>& weights() const {
        return weights_;
    }
};

// -----------------------------------------------------
// Experiment configuration
// -----------------------------------------------------

struct TransferConfig {
    TransferStrategy strategy;
    std::size_t frozenLayers;
    std::size_t epochs;
    double learningRate;
    double minimumAccuracy;

    void validate(const PretrainedModel& model) const {
        require(epochs > 0, "Epoch count must be positive.");

        require(
            std::isfinite(learningRate) && learningRate > 0.0,
            "Learning rate must be positive and finite."
        );

        require(
            minimumAccuracy >= 0.0 && minimumAccuracy <= 1.0,
            "Invalid minimum accuracy."
        );

        require(
            frozenLayers <= model.layers().size(),
            "Invalid frozen layer count."
        );

        if (strategy == TransferStrategy::FeatureExtraction) {
            require(
                frozenLayers == model.layers().size(),
                "Feature extraction requires a frozen backbone."
            );
        }
    }
};

// -----------------------------------------------------
// Evaluation metrics
// -----------------------------------------------------

struct Evaluation {
    double accuracy;
    double loss;
    std::size_t samples;
};

Evaluation evaluate(
    const PretrainedModel& model,
    const ClassificationHead& head,
    const Dataset& dataset
) {
    dataset.validate();

    std::size_t correct = 0;
    double totalLoss = 0.0;

    for (std::size_t i = 0; i < dataset.features.size(); ++i) {
        const auto representation =
            model.extractFeatures(dataset.features[i]);

        const double probability = head.probability(representation);

        const int prediction = probability >= 0.5 ? 1 : 0;

        correct += prediction == dataset.labels[i];

        totalLoss += binaryCrossEntropy(
            probability,
            dataset.labels[i]
        );
    }

    const double count = static_cast<double>(dataset.labels.size());

    return {
        correct / count,
        totalLoss / count,
        dataset.labels.size()
    };
}

// -----------------------------------------------------
// Transfer-learning experiment
// -----------------------------------------------------

class TransferExperiment {
private:
    PretrainedModel& model_;
    ClassificationHead head_;
    Dataset dataset_;
    TransferConfig config_;
    ExperimentState state_ = ExperimentState::Created;

    void transition(ExperimentState next) {
        const std::unordered_map<
            ExperimentState,
            std::vector<ExperimentState>
        > allowed = {
            {ExperimentState::Created,
                {ExperimentState::Validated, ExperimentState::Failed}},
            {ExperimentState::Validated,
                {ExperimentState::Training, ExperimentState::Failed}},
            {ExperimentState::Training,
                {ExperimentState::Evaluated, ExperimentState::Failed}},
            {ExperimentState::Evaluated,
                {ExperimentState::Approved, ExperimentState::Rejected}},
            {ExperimentState::Approved, {}},
            {ExperimentState::Rejected, {}},
            {ExperimentState::Failed, {}}
        };

        const auto& possible = allowed.at(state_);

        require(
            std::find(possible.begin(), possible.end(), next)
                != possible.end(),
            "Invalid experiment state transition."
        );

        state_ = next;
    }

public:
    TransferExperiment(
        PretrainedModel& model,
        Dataset dataset,
        TransferConfig config
    )
        : model_(model),
          head_(model.layers().back().outputDimension),
          dataset_(std::move(dataset)),
          config_(config) {}

    Evaluation run() {
        try {
            dataset_.validate();
            config_.validate(model_);

            require(
                dataset_.dimension() == model_.inputDimension(),
                "Target dataset input dimension is incompatible."
            );

            transition(ExperimentState::Validated);

            model_.configureTrainability(
                config_.strategy,
                config_.frozenLayers
            );

            transition(ExperimentState::Training);

            std::vector<std::size_t> indices(
                dataset_.labels.size()
            );

            std::iota(indices.begin(), indices.end(), 0);

            std::mt19937 generator(123);

            for (std::size_t epoch = 0; epoch < config_.epochs; ++epoch) {
                std::shuffle(
                    indices.begin(),
                    indices.end(),
                    generator
                );

                for (std::size_t index : indices) {
                    const auto features =
                        model_.extractFeatures(dataset_.features[index]);

                    head_.update(
                        features,
                        dataset_.labels[index],
                        config_.learningRate
                    );
                }
            }

            Evaluation result = evaluate(
                model_,
                head_,
                dataset_
            );

            transition(ExperimentState::Evaluated);

            return result;

        } catch (...) {
            if (
                state_ != ExperimentState::Failed &&
                state_ != ExperimentState::Rejected
            ) {
                state_ = ExperimentState::Failed;
            }

            throw;
        }
    }

    bool approve(const Evaluation& evaluation) {
        require(
            state_ == ExperimentState::Evaluated,
            "Only evaluated experiments can be approved."
        );

        if (evaluation.accuracy < config_.minimumAccuracy) {
            transition(ExperimentState::Rejected);
            return false;
        }

        transition(ExperimentState::Approved);
        return true;
    }

    ExperimentState state() const {
        return state_;
    }

    const ClassificationHead& head() const {
        return head_;
    }
};

// -----------------------------------------------------
// Synthetic target dataset
// -----------------------------------------------------

Dataset createDataset(std::size_t count) {
    Dataset dataset;

    dataset.name = "IndustrialSensorClassification";
    dataset.domain = "Industrial Monitoring";

    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.1);

    for (std::size_t i = 0; i < count; ++i) {
        std::vector<double> row(4);

        for (double& value : row) {
            value = distribution(generator);
        }

        const double score =
            1.2 * row[0] -
            0.9 * row[1] +
            0.7 * row[2] +
            0.4 * row[3] +
            noise(generator);

        dataset.features.push_back(row);
        dataset.labels.push_back(score > 0.0 ? 1 : 0);
    }

    return dataset;
}

// -----------------------------------------------------
// Main
// -----------------------------------------------------

int main() {
    try {
        std::cout << std::fixed << std::setprecision(4);

        PretrainedModel pretrained(
            "IndustrialFeatureBackbone",
            "General Sensor Data",
            4
        );

        Dataset dataset = createDataset(500);
        dataset.validate();

        std::cout << "TRANSFER LEARNING ENGINE\n";
        std::cout << "========================\n";

        std::cout << "Pretrained model: "
                  << pretrained.name() << '\n';

        std::cout << "Source domain: "
                  << pretrained.sourceDomain() << '\n';

        std::cout << "Parameters: "
                  << pretrained.parameterCount() << '\n';

        std::cout << "Target samples: "
                  << dataset.features.size() << "\n\n";

        // Feature extraction experiment.
        TransferConfig extractionConfig{
            TransferStrategy::FeatureExtraction,
            2,
            150,
            0.05,
            0.50
        };

        TransferExperiment extraction(
            pretrained,
            dataset,
            extractionConfig
        );

        const Evaluation extractionResult = extraction.run();

        std::cout << "FEATURE EXTRACTION\n";
        std::cout << "Accuracy: "
                  << extractionResult.accuracy << '\n';

        std::cout << "Loss: "
                  << extractionResult.loss << '\n';

        std::cout << "State: "
                  << toString(extraction.state()) << "\n\n";

        // Fine-tuning experiment.
        TransferConfig fineTuningConfig{
            TransferStrategy::FineTuning,
            1,
            150,
            0.01,
            0.50
        };

        TransferExperiment fineTuning(
            pretrained,
            dataset,
            fineTuningConfig
        );

        const Evaluation fineTuningResult = fineTuning.run();

        std::cout << "FINE-TUNING\n";
        std::cout << "Accuracy: "
                  << fineTuningResult.accuracy << '\n';

        std::cout << "Loss: "
                  << fineTuningResult.loss << '\n';

        const bool approved = fineTuning.approve(
            fineTuningResult
        );

        std::cout << "Approval: "
                  << (approved ? "APPROVED" : "REJECTED")
                  << '\n';

        std::cout << "Final state: "
                  << toString(fineTuning.state()) << '\n';

        // Edge-case validation.
        try {
            Dataset invalid;
            invalid.validate();
        } catch (const std::invalid_argument& error) {
            std::cout << "\nInvalid dataset correctly rejected: "
                      << error.what() << '\n';
        }

        try {
            TransferConfig invalidConfig{
                TransferStrategy::FeatureExtraction,
                1,
                100,
                0.01,
                0.5
            };

            invalidConfig.validate(pretrained);
        } catch (const std::invalid_argument& error) {
            std::cout << "Invalid transfer policy rejected: "
                      << error.what() << '\n';
        }

    } catch (const std::exception& error) {
        std::cerr << "Execution failed: "
                  << error.what() << '\n';

        return 1;
    }

    return 0;
}
