#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Neural Network Mathematics
 *
 * C++17 implementation of neurons, weighted sums,
 * activation functions, dense layers, forward propagation,
 * gradient-based training, and numerical validation.
 */

namespace neural {

// ============================================================
// 1. Domain Types
// ============================================================

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

enum class Activation {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU
};

struct NeuronResult {
    double weightedSum;
    double output;
};

struct TrainingMetrics {
    int epoch;
    double loss;
};

// ============================================================
// 2. Validation
// ============================================================

void validateNumber(double value, const std::string& name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            name + " must be finite."
        );
    }
}

void validateVector(
    const Vector& values,
    std::size_t expected = 0
) {
    if (values.empty()) {
        throw std::invalid_argument(
            "Vector cannot be empty."
        );
    }

    if (expected != 0 && values.size() != expected) {
        throw std::invalid_argument(
            "Vector dimension mismatch."
        );
    }

    for (double value : values) {
        validateNumber(value, "Vector element");
    }
}

// ============================================================
// 3. Activation Functions
// ============================================================

class ActivationFunctions {
public:
    static double apply(double x, Activation activation) {
        validateNumber(x, "Activation input");

        switch (activation) {
            case Activation::Linear:
                return x;

            case Activation::Sigmoid:
                if (x >= 0.0) {
                    return 1.0 / (
                        1.0 + std::exp(-std::min(x, 709.0))
                    );
                } else {
                    double e = std::exp(std::max(x, -709.0));
                    return e / (1.0 + e);
                }

            case Activation::Tanh:
                return std::tanh(x);

            case Activation::ReLU:
                return std::max(0.0, x);

            case Activation::LeakyReLU:
                return x > 0.0 ? x : 0.01 * x;
        }

        throw std::invalid_argument(
            "Unsupported activation."
        );
    }

    static double derivative(
        double x,
        Activation activation
    ) {
        switch (activation) {
            case Activation::Linear:
                return 1.0;

            case Activation::Sigmoid: {
                double output = apply(x, activation);
                return output * (1.0 - output);
            }

            case Activation::Tanh: {
                double output = std::tanh(x);
                return 1.0 - output * output;
            }

            case Activation::ReLU:
                return x > 0.0 ? 1.0 : 0.0;

            case Activation::LeakyReLU:
                return x > 0.0 ? 1.0 : 0.01;
        }

        throw std::invalid_argument(
            "Unsupported activation."
        );
    }

    static std::string name(Activation activation) {
        switch (activation) {
            case Activation::Linear: return "Linear";
            case Activation::Sigmoid: return "Sigmoid";
            case Activation::Tanh: return "Tanh";
            case Activation::ReLU: return "ReLU";
            case Activation::LeakyReLU: return "LeakyReLU";
        }

        return "Unknown";
    }
};

// ============================================================
// 4. Weighted Sum
// ============================================================

double weightedSum(
    const Vector& inputs,
    const Vector& weights,
    double bias
) {
    validateVector(inputs);
    validateVector(weights, inputs.size());
    validateNumber(bias, "Bias");

    double result = bias;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        result += inputs[i] * weights[i];
    }

    validateNumber(result, "Weighted sum");

    return result;
}

// ============================================================
// 5. Artificial Neuron
// ============================================================

class Neuron {
private:
    Vector weights_;
    double bias_;
    Activation activation_;

public:
    Neuron(
        Vector weights,
        double bias,
        Activation activation
    )
        : weights_(std::move(weights)),
          bias_(bias),
          activation_(activation) {

        validateVector(weights_);
        validateNumber(bias_, "Bias");
    }

    NeuronResult forward(const Vector& inputs) const {
        double z = weightedSum(
            inputs,
            weights_,
            bias_
        );

        return {
            z,
            ActivationFunctions::apply(z, activation_)
        };
    }

    const Vector& weights() const {
        return weights_;
    }

    double bias() const {
        return bias_;
    }

    Activation activation() const {
        return activation_;
    }

    void update(
        const Vector& newWeights,
        double newBias
    ) {
        validateVector(newWeights, weights_.size());
        validateNumber(newBias, "New bias");

        weights_ = newWeights;
        bias_ = newBias;
    }
};

// ============================================================
// 6. Dense Neural Network
// ============================================================

class NeuralNetwork {
private:
    std::vector<Matrix> weights_;
    std::vector<Vector> biases_;

    std::vector<Activation> activations_;

    std::vector<Vector> layerOutputs_;
    std::vector<Vector> layerSums_;

    std::vector<std::size_t> architecture_;

public:
    NeuralNetwork(
        const std::vector<std::size_t>& architecture,
        const std::vector<Activation>& activations,
        unsigned int seed = 42
    ) : architecture_(architecture),
        activations_(activations) {

        if (architecture.size() < 2) {
            throw std::invalid_argument(
                "Network needs at least two layers."
            );
        }

        if (activations.size() != architecture.size() - 1) {
            throw std::invalid_argument(
                "Each dense layer requires an activation."
            );
        }

        for (std::size_t size : architecture) {
            if (size == 0) {
                throw std::invalid_argument(
                    "Layer size cannot be zero."
                );
            }
        }

        std::mt19937 generator(seed);

        for (std::size_t layer = 0;
             layer + 1 < architecture.size();
             ++layer) {

            std::size_t fanIn = architecture[layer];
            std::size_t fanOut = architecture[layer + 1];

            // Xavier initialization.
            double limit = std::sqrt(
                6.0 / static_cast<double>(fanIn + fanOut)
            );

            std::uniform_real_distribution<double> distribution(
                -limit,
                limit
            );

            Matrix layerWeights(
                fanOut,
                Vector(fanIn)
            );

            for (auto& row : layerWeights) {
                for (double& weight : row) {
                    weight = distribution(generator);
                }
            }

            weights_.push_back(std::move(layerWeights));
            biases_.push_back(Vector(fanOut, 0.0));
        }
    }

    // --------------------------------------------------------
    // 7. Forward Propagation
    // --------------------------------------------------------

    Vector forward(const Vector& inputs) {
        validateVector(inputs, architecture_[0]);

        Vector current = inputs;

        layerOutputs_.clear();
        layerSums_.clear();

        layerOutputs_.push_back(current);

        for (std::size_t layer = 0;
             layer < weights_.size();
             ++layer) {

            Vector next;
            Vector sums;

            for (std::size_t neuron = 0;
                 neuron < weights_[layer].size();
                 ++neuron) {

                double z = weightedSum(
                    current,
                    weights_[layer][neuron],
                    biases_[layer][neuron]
                );

                sums.push_back(z);

                next.push_back(
                    ActivationFunctions::apply(
                        z,
                        activations_[layer]
                    )
                );
            }

            layerSums_.push_back(std::move(sums));
            layerOutputs_.push_back(next);
            current = std::move(next);
        }

        return current;
    }

    // --------------------------------------------------------
    // 8. Loss Function
    // --------------------------------------------------------

    static double mse(
        const Vector& predictions,
        const Vector& targets
    ) {
        validateVector(targets, predictions.size());
        validateVector(predictions, targets.size());

        double sum = 0.0;

        for (std::size_t i = 0; i < predictions.size(); ++i) {
            double difference = predictions[i] - targets[i];
            sum += difference * difference;
        }

        return sum / static_cast<double>(targets.size());
    }

    // --------------------------------------------------------
    // 9. Backpropagation
    // --------------------------------------------------------

    double trainSample(
        const Vector& inputs,
        const Vector& targets,
        double learningRate
    ) {
        validateNumber(learningRate, "Learning rate");

        if (learningRate <= 0.0) {
            throw std::invalid_argument(
                "Learning rate must be positive."
            );
        }

        Vector predictions = forward(inputs);

        validateVector(targets, predictions.size());

        std::size_t layerCount = weights_.size();

        std::vector<Vector> deltas(layerCount);

        // Output-layer delta.
        std::size_t last = layerCount - 1;

        for (std::size_t neuron = 0;
             neuron < predictions.size();
             ++neuron) {

            double lossGradient =
                2.0 * (predictions[neuron] - targets[neuron]);

            double activationGradient =
                ActivationFunctions::derivative(
                    layerSums_[last][neuron],
                    activations_[last]
                );

            deltas[last].push_back(
                lossGradient * activationGradient
            );
        }

        // Hidden-layer deltas.
        for (int layer = static_cast<int>(layerCount) - 2;
             layer >= 0;
             --layer) {

            for (std::size_t neuron = 0;
                 neuron < weights_[layer].size();
                 ++neuron) {

                double propagated = 0.0;

                for (std::size_t next = 0;
                     next < deltas[layer + 1].size();
                     ++next) {

                    propagated +=
                        weights_[layer + 1][next][neuron] *
                        deltas[layer + 1][next];
                }

                double derivative =
                    ActivationFunctions::derivative(
                        layerSums_[layer][neuron],
                        activations_[layer]
                    );

                deltas[layer].push_back(
                    propagated * derivative
                );
            }
        }

        // Update all parameters after calculating every delta.
        for (std::size_t layer = 0;
             layer < layerCount;
             ++layer) {

            const Vector& previous = layerOutputs_[layer];

            for (std::size_t neuron = 0;
                 neuron < weights_[layer].size();
                 ++neuron) {

                for (std::size_t weight = 0;
                     weight < previous.size();
                     ++weight) {

                    weights_[layer][neuron][weight] -=
                        learningRate *
                        deltas[layer][neuron] *
                        previous[weight];
                }

                biases_[layer][neuron] -=
                    learningRate * deltas[layer][neuron];
            }
        }

        return mse(predictions, targets);
    }

    // --------------------------------------------------------
    // 10. Dataset Training
    // --------------------------------------------------------

    std::vector<TrainingMetrics> train(
        const std::vector<std::pair<Vector, Vector>>& dataset,
        int epochs,
        double learningRate
    ) {
        if (dataset.empty() || epochs <= 0) {
            throw std::invalid_argument(
                "Dataset and epoch count must be valid."
            );
        }

        for (const auto& sample : dataset) {
            validateVector(sample.first, architecture_[0]);
            validateVector(sample.second, architecture_.back());
        }

        std::vector<TrainingMetrics> history;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;

            for (const auto& sample : dataset) {
                trainSample(
                    sample.first,
                    sample.second,
                    learningRate
                );
            }

            for (const auto& sample : dataset) {
                totalLoss += mse(
                    forward(sample.first),
                    sample.second
                );
            }

            totalLoss /= static_cast<double>(dataset.size());

            history.push_back({
                epoch,
                totalLoss
            });

            if (epoch % 1000 == 0) {
                std::cout
                    << "Epoch: " << epoch
                    << " | Loss: " << std::fixed
                    << std::setprecision(8)
                    << totalLoss << '\n';
            }

            if (totalLoss < 0.001) {
                break;
            }
        }

        return history;
    }
};

// ============================================================
// 11. Numerical Gradient Verification
// ============================================================

double numericalDerivative(
    double (*function)(double),
    double x,
    double epsilon = 1e-6
) {
    if (epsilon <= 0.0) {
        throw std::invalid_argument(
            "Epsilon must be positive."
        );
    }

    return (
        function(x + epsilon) -
        function(x - epsilon)
    ) / (2.0 * epsilon);
}

double sigmoidFunction(double x) {
    return ActivationFunctions::apply(
        x,
        Activation::Sigmoid
    );
}

// ============================================================
// 12. Demonstration
// ============================================================

void demonstrateNeuron() {
    std::cout << "\n--- Single Neuron ---\n";

    Neuron neuron(
        {0.5, -0.3, 0.8},
        0.2,
        Activation::Sigmoid
    );

    NeuronResult result = neuron.forward(
        {1.0, 2.0, 0.5}
    );

    std::cout << "Weighted sum: "
              << result.weightedSum << '\n';

    std::cout << "Activation output: "
              << result.output << '\n';
}

void demonstrateActivations() {
    std::cout << "\n--- Activation Functions ---\n";

    for (Activation activation : {
        Activation::Linear,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::ReLU,
        Activation::LeakyReLU
    }) {
        double x = 1.5;

        std::cout
            << ActivationFunctions::name(activation)
            << " | Output: "
            << ActivationFunctions::apply(x, activation)
            << " | Derivative: "
            << ActivationFunctions::derivative(x, activation)
            << '\n';
    }
}

void demonstrateXOR() {
    std::cout << "\n--- XOR Training ---\n";

    std::vector<std::pair<Vector, Vector>> dataset = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    NeuralNetwork network(
        {2, 4, 1},
        {Activation::Tanh, Activation::Sigmoid},
        42
    );

    auto history = network.train(
        dataset,
        10000,
        0.5
    );

    std::cout << "\nFinal loss: "
              << history.back().loss << '\n';

    for (const auto& sample : dataset) {
        Vector prediction = network.forward(sample.first);

        std::cout
            << sample.first[0] << " XOR "
            << sample.first[1] << " = "
            << prediction[0]
            << " | Class: "
            << (prediction[0] >= 0.5 ? 1 : 0)
            << '\n';
    }
}

void demonstrateValidation() {
    std::cout << "\n--- Validation ---\n";

    try {
        weightedSum(
            {1.0, 2.0},
            {0.5},
            0.1
        );
    } catch (const std::exception& error) {
        std::cout << "Expected error: "
                  << error.what() << '\n';
    }

    double x = 0.7;

    double analytical =
        ActivationFunctions::derivative(
            x,
            Activation::Sigmoid
        );

    double numerical = numericalDerivative(
        sigmoidFunction,
        x
    );

    std::cout << "Analytical derivative: "
              << analytical << '\n';

    std::cout << "Numerical derivative: "
              << numerical << '\n';

    std::cout << "Derivative verification: "
              << (std::abs(analytical - numerical) < 1e-5
                      ? "PASSED"
                      : "FAILED")
              << '\n';
}

} // namespace neural

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);

        neural::demonstrateNeuron();
        neural::demonstrateActivations();
        neural::demonstrateValidation();
        neural::demonstrateXOR();

    } catch (const std::exception& error) {
        std::cerr << "Execution error: "
                  << error.what() << '\n';

        return 1;
    }

    return 0;
}
