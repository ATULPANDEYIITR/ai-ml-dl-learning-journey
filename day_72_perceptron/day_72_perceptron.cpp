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

using namespace std;

/*
 * Perceptron Learning Rule and Linear Separability
 *
 * C++17 implementation of a binary classification engine.
 *
 * Features:
 * - Domain modeling for samples and training configuration.
 * - Online perceptron updates.
 * - Linear decision function.
 * - Configurable convergence limits.
 * - Deterministic sample shuffling.
 * - Protected model state through class interfaces.
 * - Confusion matrix and accuracy evaluation.
 * - Repository-style experiment reporting.
 * - Numerical validation and exception handling.
 */

using Vector = vector<double>;
using Dataset = vector<Vector>;

enum class TrainingStatus {
    Converged,
    MaximumEpochsReached
};

struct Sample {
    Vector features;
    int label;
};

struct TrainingConfig {
    double learningRate = 0.1;
    int maxEpochs = 100;
    bool shuffle = false;
    unsigned int seed = 42;
};

struct EpochReport {
    int epoch;
    int mistakes;
    Vector weights;
    double bias;
};

struct Evaluation {
    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;

    double accuracy() const {
        const int total =
            truePositive + trueNegative + falsePositive + falseNegative;

        return total == 0
            ? 0.0
            : static_cast<double>(truePositive + trueNegative) / total;
    }
};

class ValidationError : public runtime_error {
public:
    explicit ValidationError(const string& message)
        : runtime_error(message) {}
};

class Perceptron {
private:
    TrainingConfig config;
    Vector weights;
    double bias = 0.0;
    int dimensions = 0;
    TrainingStatus status = TrainingStatus::MaximumEpochsReached;
    vector<EpochReport> history;

    static void validateSample(const Vector& sample) {
        if (sample.empty()) {
            throw ValidationError("A sample cannot be empty.");
        }

        for (double value : sample) {
            if (!isfinite(value)) {
                throw ValidationError("Features must be finite.");
            }
        }
    }

    void validateDataset(const vector<Sample>& samples) const {
        if (samples.empty()) {
            throw ValidationError("Training dataset cannot be empty.");
        }

        if (samples.front().features.empty()) {
            throw ValidationError("Samples must contain features.");
        }

        const size_t expected = samples.front().features.size();
        bool hasZero = false;
        bool hasOne = false;

        for (const auto& sample : samples) {
            validateSample(sample.features);

            if (sample.features.size() != expected) {
                throw ValidationError("Inconsistent feature dimensions.");
            }

            if (sample.label != 0 && sample.label != 1) {
                throw ValidationError("Labels must be binary.");
            }

            hasZero |= sample.label == 0;
            hasOne |= sample.label == 1;
        }

        if (!hasZero || !hasOne) {
            throw ValidationError("Both classes must be represented.");
        }
    }

public:
    explicit Perceptron(TrainingConfig configuration = {})
        : config(configuration) {
        if (!isfinite(config.learningRate) || config.learningRate <= 0.0) {
            throw ValidationError("Learning rate must be positive.");
        }

        if (config.maxEpochs < 1) {
            throw ValidationError("Maximum epochs must be positive.");
        }
    }

    double netInput(const Vector& sample) const {
        if (sample.size() != weights.size()) {
            throw ValidationError("Prediction dimension mismatch.");
        }

        double result = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * sample[i];
        }

        return result;
    }

    int predict(const Vector& sample) const {
        return netInput(sample) >= 0.0 ? 1 : 0;
    }

    void train(vector<Sample> samples) {
        validateDataset(samples);

        dimensions = static_cast<int>(samples.front().features.size());
        weights.assign(dimensions, 0.0);
        bias = 0.0;
        history.clear();

        mt19937 generator(config.seed);

        for (int epoch = 1; epoch <= config.maxEpochs; ++epoch) {
            if (config.shuffle) {
                shuffle(samples.begin(), samples.end(), generator);
            }

            int mistakes = 0;

            for (const auto& sample : samples) {
                const int prediction = predict(sample.features);
                const int error = sample.label - prediction;

                if (error != 0) {
                    ++mistakes;

                    // The update uses the current observation and its error.
                    for (int j = 0; j < dimensions; ++j) {
                        weights[j] +=
                            config.learningRate *
                            error *
                            sample.features[j];
                    }

                    bias += config.learningRate * error;

                    for (double weight : weights) {
                        if (!isfinite(weight)) {
                            throw overflow_error("Weight overflow detected.");
                        }
                    }

                    if (!isfinite(bias)) {
                        throw overflow_error("Bias overflow detected.");
                    }
                }
            }

            history.push_back({
                epoch,
                mistakes,
                weights,
                bias
            });

            cout << "Epoch " << setw(3) << epoch
                 << " | Mistakes: " << mistakes
                 << " | Weights: ";

            for (double weight : weights) {
                cout << fixed << setprecision(3) << weight << " ";
            }

            cout << "| Bias: " << bias << '\n';

            if (mistakes == 0) {
                status = TrainingStatus::Converged;
                return;
            }
        }

        status = TrainingStatus::MaximumEpochsReached;
    }

    Evaluation evaluate(const vector<Sample>& samples) const {
        if (weights.empty()) {
            throw logic_error("Model has not been trained.");
        }

        if (samples.empty()) {
            throw ValidationError("Evaluation data cannot be empty.");
        }

        Evaluation result;

        for (const auto& sample : samples) {
            validateSample(sample.features);

            if (sample.features.size() != weights.size()) {
                throw ValidationError("Evaluation dimension mismatch.");
            }

            if (sample.label != 0 && sample.label != 1) {
                throw ValidationError("Invalid evaluation label.");
            }

            const int prediction = predict(sample.features);

            if (prediction == 1 && sample.label == 1) {
                ++result.truePositive;
            } else if (prediction == 0 && sample.label == 0) {
                ++result.trueNegative;
            } else if (prediction == 1) {
                ++result.falsePositive;
            } else {
                ++result.falseNegative;
            }
        }

        return result;
    }

    void printModel() const {
        cout << "\nModel parameters\n";
        cout << "Weights: ";

        for (double weight : weights) {
            cout << weight << " ";
        }

        cout << "\nBias: " << bias << '\n';

        cout << "Status: "
             << (status == TrainingStatus::Converged
                 ? "Converged"
                 : "Maximum epochs reached")
             << '\n';
    }

    bool converged() const {
        return status == TrainingStatus::Converged;
    }

    const vector<EpochReport>& getHistory() const {
        return history;
    }

    int getDimensions() const {
        return dimensions;
    }
};

class DatasetFactory {
public:
    static vector<Sample> andGate() {
        return {
            {{0, 0}, 0},
            {{0, 1}, 0},
            {{1, 0}, 0},
            {{1, 1}, 1}
        };
    }

    static vector<Sample> xorGate() {
        return {
            {{0, 0}, 0},
            {{0, 1}, 1},
            {{1, 0}, 1},
            {{1, 1}, 0}
        };
    }
};

class Experiment {
public:
    static void run(
        const string& name,
        const vector<Sample>& samples
    ) {
        cout << "\n====================================\n";
        cout << name << '\n';
        cout << "====================================\n";

        Perceptron model({
            0.1,
            100,
            false,
            42
        });

        model.train(samples);
        model.printModel();

        const Evaluation result = model.evaluate(samples);

        cout << "\nEvaluation\n";
        cout << "Accuracy: " << fixed << setprecision(2)
             << result.accuracy() * 100.0 << "%\n";

        cout << "TP: " << result.truePositive
             << " TN: " << result.trueNegative
             << " FP: " << result.falsePositive
             << " FN: " << result.falseNegative << '\n';

        cout << "\nPredictions\n";

        for (const auto& sample : samples) {
            cout << "(";

            for (size_t i = 0; i < sample.features.size(); ++i) {
                if (i) cout << ", ";
                cout << sample.features[i];
            }

            cout << ") Expected=" << sample.label
                 << " Predicted=" << model.predict(sample.features)
                 << '\n';
        }

        if (name == "AND gate" && !model.converged()) {
            throw runtime_error("AND gate should converge.");
        }

        if (name == "XOR gate" && model.converged()) {
            throw runtime_error("XOR gate should not converge.");
        }
    }
};

int main() {
    try {
        Experiment::run("AND gate", DatasetFactory::andGate());
        Experiment::run("XOR gate", DatasetFactory::xorGate());

        // A constant feature dimension is permitted; its weight has
        // no useful effect when the feature never changes.
        cout << "\nAll C++17 experiments completed.\n";

    } catch (const exception& error) {
        cerr << "Execution error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
