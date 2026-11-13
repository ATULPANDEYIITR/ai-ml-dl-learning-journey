#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Forward Propagation and Neural Network Computation
 *
 * C++17 implementation of:
 * - Dense neural network layers
 * - Matrix-vector multiplication
 * - Weight and bias management
 * - Activation functions
 * - Layer composition
 * - Batch inference
 * - Input validation
 * - Numerical error detection
 * - Model parameter counting
 *
 * Weight orientation:
 * W[output_neuron][input_neuron]
 */

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

enum class Activation {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};

class NetworkError : public std::runtime_error {
public:
    explicit NetworkError(const std::string& message)
        : std::runtime_error(message) {}
};

class DimensionError : public NetworkError {
public:
    explicit DimensionError(const std::string& message)
        : NetworkError(message) {}
};

class NumericalError : public NetworkError {
public:
    explicit NumericalError(const std::string& message)
        : NetworkError(message) {}
};

void validateVector(const Vector& values, const std::string& name) {
    if (values.empty()) {
        throw std::invalid_argument(name + " cannot be empty.");
    }

    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                name + " contains a non-finite value."
            );
        }
    }
}

std::string activationName(Activation activation) {
    switch (activation) {
        case Activation::Linear: return "Linear";
        case Activation::ReLU: return "ReLU";
        case Activation::Sigmoid: return "Sigmoid";
        case Activation::Tanh: return "Tanh";
        case Activation::Softmax: return "Softmax";
    }

    throw std::invalid_argument("Unknown activation.");
}

double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }

    const double exponential = std::exp(value);

    return exponential / (1.0 + exponential);
}

Vector softmax(const Vector& values) {
    const double maximum = *std::max_element(
        values.begin(),
        values.end()
    );

    Vector exponentials;
    exponentials.reserve(values.size());

    for (double value : values) {
        exponentials.push_back(std::exp(value - maximum));
    }

    const double total = std::accumulate(
        exponentials.begin(),
        exponentials.end(),
        0.0
    );

    if (!std::isfinite(total) || total <= 0.0) {
        throw NumericalError("Invalid softmax denominator.");
    }

    for (double& value : exponentials) {
        value /= total;
    }

    return exponentials;
}

Vector applyActivation(
    const Vector& values,
    Activation activation
) {
    Vector result;

    switch (activation) {
        case Activation::Linear:
            result = values;
            break;

        case Activation::ReLU:
            result.reserve(values.size());

            for (double value : values) {
                result.push_back(std::max(0.0, value));
            }
            break;

        case Activation::Sigmoid:
            result.reserve(values.size());

            for (double value : values) {
                result.push_back(sigmoid(value));
            }
            break;

        case Activation::Tanh:
            result.reserve(values.size());

            for (double value : values) {
                result.push_back(std::tanh(value));
            }
            break;

        case Activation::Softmax:
            result = softmax(values);
            break;
    }

    for (double value : result) {
        if (!std::isfinite(value)) {
            throw NumericalError(
                "Activation produced a non-finite value."
            );
        }
    }

    return result;
}

struct LayerResult {
    Vector weightedSums;
    Vector outputs;
};

class DenseLayer {
private:
    Matrix weights_;
    Vector biases_;
    Activation activation_;
    std::string name_;
    std::size_t inputSize_;
    std::size_t outputSize_;

public:
    DenseLayer(
        Matrix weights,
        Vector biases,
        Activation activation,
        std::string name
    )
        : weights_(std::move(weights)),
          biases_(std::move(biases)),
          activation_(activation),
          name_(std::move(name)),
          inputSize_(0),
          outputSize_(weights_.size()) {

        if (weights_.empty()) {
            throw std::invalid_argument(
                "Weight matrix cannot be empty."
            );
        }

        validateVector(biases_, "Biases");

        inputSize_ = weights_[0].size();

        if (inputSize_ == 0) {
            throw DimensionError("Weight rows cannot be empty.");
        }

        if (biases_.size() != outputSize_) {
            throw DimensionError(
                "Bias count must equal output neuron count."
            );
        }

        for (const auto& row : weights_) {
            validateVector(row, "Weight row");

            if (row.size() != inputSize_) {
                throw DimensionError(
                    "Weight matrix must be rectangular."
                );
            }
        }
    }

    LayerResult forward(const Vector& inputs) const {
        validateVector(inputs, "Layer input");

        if (inputs.size() != inputSize_) {
            throw DimensionError(
                name_ + ": incompatible input dimensions."
            );
        }

        Vector weightedSums(outputSize_, 0.0);

        // Compute each output neuron's weighted sum.
        for (std::size_t j = 0; j < outputSize_; ++j) {
            double sum = biases_[j];

            for (std::size_t i = 0; i < inputSize_; ++i) {
                sum += weights_[j][i] * inputs[i];
            }

            if (!std::isfinite(sum)) {
                throw NumericalError(
                    "Invalid weighted sum in " + name_ + "."
                );
            }

            weightedSums[j] = sum;
        }

        Vector outputs = applyActivation(
            weightedSums,
            activation_
        );

        return {weightedSums, outputs};
    }

    std::size_t inputSize() const {
        return inputSize_;
    }

    std::size_t outputSize() const {
        return outputSize_;
    }

    std::size_t parameterCount() const {
        return inputSize_ * outputSize_ + outputSize_;
    }

    const std::string& name() const {
        return name_;
    }

    Activation activation() const {
        return activation_;
    }
};

struct TraceEntry {
    std::string layerName;
    Vector weightedSums;
    Vector outputs;
};

struct Prediction {
    Vector values;
    std::vector<TraceEntry> trace;
};

class NeuralNetwork {
private:
    std::vector<DenseLayer> layers_;

public:
    explicit NeuralNetwork(std::vector<DenseLayer> layers)
        : layers_(std::move(layers)) {

        if (layers_.empty()) {
            throw std::invalid_argument(
                "Network requires at least one layer."
            );
        }

        for (std::size_t i = 1; i < layers_.size(); ++i) {
            if (
                layers_[i - 1].outputSize() !=
                layers_[i].inputSize()
            ) {
                throw DimensionError(
                    "Adjacent layers have incompatible dimensions."
                );
            }
        }
    }

    std::size_t inputSize() const {
        return layers_.front().inputSize();
    }

    std::size_t outputSize() const {
        return layers_.back().outputSize();
    }

    std::size_t parameterCount() const {
        std::size_t count = 0;

        for (const auto& layer : layers_) {
            count += layer.parameterCount();
        }

        return count;
    }

    Prediction forward(const Vector& inputs) const {
        validateVector(inputs, "Network input");

        if (inputs.size() != inputSize()) {
            throw DimensionError(
                "Input feature count does not match network."
            );
        }

        Vector current = inputs;
        std::vector<TraceEntry> trace;

        trace.reserve(layers_.size());

        for (const auto& layer : layers_) {
            LayerResult result = layer.forward(current);

            trace.push_back({
                layer.name(),
                result.weightedSums,
                result.outputs
            });

            current = std::move(result.outputs);
        }

        return {current, trace};
    }

    std::vector<Vector> predictBatch(
        const std::vector<Vector>& samples
    ) const {
        if (samples.empty()) {
            throw std::invalid_argument(
                "Batch cannot be empty."
            );
        }

        std::vector<Vector> results;
        results.reserve(samples.size());

        for (const auto& sample : samples) {
            results.push_back(forward(sample).values);
        }

        return results;
    }

    void printSummary() const {
        std::cout << "\nNETWORK ARCHITECTURE\n";

        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];

            std::cout
                << "Layer " << i + 1 << ": "
                << layer.name()
                << " | " << layer.inputSize()
                << " -> " << layer.outputSize()
                << " | " << activationName(layer.activation())
                << '\n';
        }

        std::cout
            << "Trainable parameters: "
            << parameterCount() << '\n';
    }
};

void printVector(const std::string& label, const Vector& values) {
    std::cout << label << ": [";

    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];

        if (i + 1 < values.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}

NeuralNetwork createDemoNetwork() {
    DenseLayer hidden(
        {
            {0.2, 0.4, -0.3},
            {0.5, -0.1, 0.2},
            {-0.2, 0.3, 0.6},
            {0.1, 0.2, 0.4}
        },
        {0.1, -0.1, 0.05, 0.0},
        Activation::ReLU,
        "Hidden Layer"
    );

    DenseLayer output(
        {
            {0.3, -0.2, 0.4, 0.1},
            {-0.1, 0.5, 0.2, -0.3}
        },
        {0.05, -0.05},
        Activation::Softmax,
        "Output Layer"
    );

    return NeuralNetwork({hidden, output});
}

void runTests(const NeuralNetwork& network) {
    DenseLayer linear(
        {{2.0, 3.0}},
        {1.0},
        Activation::Linear,
        "Linear Test"
    );

    auto linearResult = linear.forward({4.0, 5.0});

    if (std::abs(linearResult.outputs[0] - 24.0) > 1e-12) {
        throw NetworkError("Linear computation test failed.");
    }

    DenseLayer relu(
        {{1.0}},
        {-2.0},
        Activation::ReLU,
        "ReLU Test"
    );

    if (relu.forward({1.0}).outputs[0] != 0.0) {
        throw NetworkError("ReLU test failed.");
    }

    Vector probabilities = softmax({1.0, 2.0, 3.0});

    double total = std::accumulate(
        probabilities.begin(),
        probabilities.end(),
        0.0
    );

    if (std::abs(total - 1.0) > 1e-10) {
        throw NetworkError("Softmax normalization failed.");
    }

    auto batch = network.predictBatch({
        {0.2, 0.5, 0.8},
        {0.7, 0.1, 0.4}
    });

    if (batch.size() != 2) {
        throw NetworkError("Batch inference test failed.");
    }

    bool dimensionErrorCaught = false;

    try {
        network.forward({1.0, 2.0});
    } catch (const DimensionError&) {
        dimensionErrorCaught = true;
    }

    if (!dimensionErrorCaught) {
        throw NetworkError("Dimension validation test failed.");
    }

    bool numericalInputRejected = false;

    try {
        network.forward({
            std::numeric_limits<double>::quiet_NaN(),
            1.0,
            2.0
        });
    } catch (const std::invalid_argument&) {
        numericalInputRejected = true;
    }

    if (!numericalInputRejected) {
        throw NetworkError("Input validation test failed.");
    }

    std::cout << "\nAll C++17 tests passed.\n";
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);

        std::cout
            << "FORWARD PROPAGATION ENGINE\n"
            << "==========================\n";

        NeuralNetwork network = createDemoNetwork();

        network.printSummary();

        Vector inputs = {0.2, 0.5, 0.8};

        Prediction result = network.forward(inputs);

        std::cout << "\nFORWARD PROPAGATION TRACE\n";

        printVector("Input", inputs);

        for (std::size_t i = 0; i < result.trace.size(); ++i) {
            const auto& entry = result.trace[i];

            std::cout << "\n" << entry.layerName << '\n';

            printVector(
                "Weighted sums",
                entry.weightedSums
            );

            printVector(
                "Activated outputs",
                entry.outputs
            );
        }

        std::cout << "\nFINAL PREDICTION\n";

        printVector("Output", result.values);

        std::vector<Vector> samples = {
            {0.2, 0.5, 0.8},
            {0.7, 0.1, 0.4},
            {0.9, 0.3, 0.2}
        };

        std::cout << "\nBATCH INFERENCE\n";

        auto batch = network.predictBatch(samples);

        for (std::size_t i = 0; i < batch.size(); ++i) {
            printVector(
                "Sample " + std::to_string(i + 1),
                batch[i]
            );
        }

        runTests(network);

        std::cout << "\nForward propagation completed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Execution failure: "
            << error.what()
            << '\n';

        return 1;
    }
}
