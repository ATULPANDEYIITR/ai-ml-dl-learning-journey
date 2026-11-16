#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/*
 * Gradient Descent Repository
 *
 * C++17 implementation of three optimization strategies:
 * 1. Batch Gradient Descent
 * 2. Stochastic Gradient Descent
 * 3. Mini-Batch Gradient Descent
 *
 * Engineering focus:
 * - Explicit domain types
 * - Efficient vector-based numerical operations
 * - Deterministic shuffling
 * - Gradient norm clipping
 * - L2 regularization
 * - Training-state management
 * - Merge-free optimization strategy comparison
 * - Evaluation and CSV output
 */

enum class OptimizerType {
    Batch,
    Stochastic,
    MiniBatch
};

struct Sample {
    vector<double> features;
    double target;
};

struct TrainingConfig {
    OptimizerType optimizer;
    double learningRate = 0.01;
    double regularization = 0.001;
    double gradientClip = 10.0;
    double tolerance = 1e-10;
    size_t epochs = 150;
    size_t batchSize = 16;
    size_t patience = 25;
    unsigned int seed = 42;
};

struct Gradients {
    vector<double> weights;
    double bias = 0.0;
};

struct EpochMetrics {
    size_t epoch;
    double loss;
    double gradientNorm;
    size_t batchCount;
};

struct TrainingResult {
    string optimizer;
    vector<double> weights;
    double bias;
    vector<EpochMetrics> history;
    bool converged = false;
};

string optimizerName(OptimizerType type) {
    switch (type) {
        case OptimizerType::Batch:
            return "BATCH";

        case OptimizerType::Stochastic:
            return "STOCHASTIC";

        case OptimizerType::MiniBatch:
            return "MINI_BATCH";
    }

    throw invalid_argument("Unknown optimizer.");
}

void validateConfig(const TrainingConfig& config) {
    if (!isfinite(config.learningRate) ||
        config.learningRate <= 0.0) {
        throw invalid_argument("Invalid learning rate.");
    }

    if (!isfinite(config.regularization) ||
        config.regularization < 0.0) {
        throw invalid_argument("Invalid regularization.");
    }

    if (!isfinite(config.gradientClip) ||
        config.gradientClip <= 0.0) {
        throw invalid_argument("Invalid gradient clipping.");
    }

    if (config.epochs == 0 ||
        config.batchSize == 0 ||
        config.patience == 0) {
        throw invalid_argument("Invalid training dimensions.");
    }

    if (config.tolerance < 0.0 ||
        !isfinite(config.tolerance)) {
        throw invalid_argument("Invalid convergence tolerance.");
    }
}

class Dataset {
public:
    static vector<Sample> generate(
        size_t count,
        double noise,
        unsigned int seed
    ) {
        if (count < 2 || noise < 0.0 || !isfinite(noise)) {
            throw invalid_argument("Invalid dataset parameters.");
        }

        mt19937 generator(seed);
        uniform_real_distribution<double> feature(-5.0, 5.0);
        normal_distribution<double> error(0.0, noise);

        vector<Sample> samples;
        samples.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            vector<double> x = {
                feature(generator),
                feature(generator)
            };

            double y =
                2.5 * x[0] -
                1.7 * x[1] +
                4.0 +
                error(generator);

            samples.push_back({move(x), y});
        }

        return samples;
    }
};

class LinearRegression {
private:
    vector<double> weights;
    double bias;

public:
    explicit LinearRegression(size_t dimensions)
        : weights(dimensions, 0.0), bias(0.0) {
        if (dimensions == 0) {
            throw invalid_argument("Model needs features.");
        }
    }

    double predict(const vector<double>& features) const {
        if (features.size() != weights.size()) {
            throw invalid_argument("Feature dimension mismatch.");
        }

        double result = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            if (!isfinite(features[i])) {
                throw invalid_argument("Non-finite feature.");
            }

            result += weights[i] * features[i];
        }

        return result;
    }

    double loss(
        const vector<Sample>& samples,
        double regularization
    ) const {
        if (samples.empty()) {
            throw invalid_argument("Empty training dataset.");
        }

        double squaredError = 0.0;

        for (const auto& sample : samples) {
            double error = predict(sample.features) - sample.target;
            squaredError += error * error;
        }

        double penalty = inner_product(
            weights.begin(),
            weights.end(),
            weights.begin(),
            0.0
        );

        double result =
            squaredError / (2.0 * samples.size()) +
            regularization * penalty / 2.0;

        if (!isfinite(result)) {
            throw overflow_error("Non-finite loss.");
        }

        return result;
    }

    Gradients computeGradients(
        const vector<Sample>& batch,
        double regularization
    ) const {
        if (batch.empty()) {
            throw invalid_argument("Cannot calculate empty gradients.");
        }

        Gradients gradients;
        gradients.weights.assign(weights.size(), 0.0);

        for (const auto& sample : batch) {
            double error = predict(sample.features) - sample.target;

            for (size_t i = 0; i < weights.size(); ++i) {
                gradients.weights[i] += error * sample.features[i];
            }

            gradients.bias += error;
        }

        double count = static_cast<double>(batch.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            gradients.weights[i] =
                gradients.weights[i] / count +
                regularization * weights[i];
        }

        gradients.bias /= count;

        return gradients;
    }

    double update(
        const Gradients& gradients,
        double learningRate,
        double clipping
    ) {
        double squaredNorm = gradients.bias * gradients.bias;

        for (double value : gradients.weights) {
            squaredNorm += value * value;
        }

        double norm = sqrt(squaredNorm);

        if (!isfinite(norm)) {
            throw overflow_error("Gradient norm overflow.");
        }

        double scale = min(
            1.0,
            clipping / max(norm, 1e-15)
        );

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -=
                learningRate * gradients.weights[i] * scale;
        }

        bias -= learningRate * gradients.bias * scale;

        return norm;
    }

    const vector<double>& getWeights() const {
        return weights;
    }

    double getBias() const {
        return bias;
    }
};

class GradientDescentEngine {
private:
    TrainingConfig config;
    mt19937 generator;

    vector<vector<size_t>> createBatches(size_t count) {
        vector<size_t> indices(count);
        iota(indices.begin(), indices.end(), 0);

        shuffle(indices.begin(), indices.end(), generator);

        size_t batchSize;

        switch (config.optimizer) {
            case OptimizerType::Batch:
                batchSize = count;
                break;

            case OptimizerType::Stochastic:
                batchSize = 1;
                break;

            case OptimizerType::MiniBatch:
                batchSize = config.batchSize;
                break;

            default:
                throw invalid_argument("Unsupported optimizer.");
        }

        vector<vector<size_t>> batches;

        for (size_t start = 0; start < count; start += batchSize) {
            size_t end = min(start + batchSize, count);

            batches.emplace_back(
                indices.begin() + start,
                indices.begin() + end
            );
        }

        return batches;
    }

public:
    explicit GradientDescentEngine(TrainingConfig config)
        : config(move(config)), generator(this->config.seed) {
        validateConfig(this->config);
    }

    TrainingResult train(const vector<Sample>& samples) {
        if (samples.empty()) {
            throw invalid_argument("Training dataset is empty.");
        }

        size_t dimensions = samples.front().features.size();

        if (dimensions == 0) {
            throw invalid_argument("Samples require features.");
        }

        for (const auto& sample : samples) {
            if (sample.features.size() != dimensions ||
                !isfinite(sample.target)) {
                throw invalid_argument("Invalid training sample.");
            }

            for (double feature : sample.features) {
                if (!isfinite(feature)) {
                    throw invalid_argument("Invalid feature.");
                }
            }
        }

        LinearRegression model(dimensions);

        TrainingResult result;
        result.optimizer = optimizerName(config.optimizer);

        double bestLoss = numeric_limits<double>::infinity();
        size_t staleEpochs = 0;

        for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
            auto batches = createBatches(samples.size());

            double totalGradientNorm = 0.0;

            for (const auto& batchIndices : batches) {
                vector<Sample> batch;
                batch.reserve(batchIndices.size());

                for (size_t index : batchIndices) {
                    batch.push_back(samples[index]);
                }

                Gradients gradients = model.computeGradients(
                    batch,
                    config.regularization
                );

                totalGradientNorm += model.update(
                    gradients,
                    config.learningRate,
                    config.gradientClip
                );
            }

            double currentLoss = model.loss(
                samples,
                config.regularization
            );

            EpochMetrics metrics{
                epoch,
                currentLoss,
                totalGradientNorm / batches.size(),
                batches.size()
            };

            result.history.push_back(metrics);

            if (epoch == 1 || epoch % 50 == 0) {
                cout << "[" << result.optimizer << "] "
                     << "Epoch " << epoch
                     << " | Loss: " << fixed
                     << setprecision(6)
                     << currentLoss << '\n';
            }

            if (bestLoss - currentLoss > config.tolerance) {
                bestLoss = currentLoss;
                staleEpochs = 0;
            } else {
                ++staleEpochs;
            }

            if (staleEpochs >= config.patience) {
                result.converged = true;
                break;
            }
        }

        result.weights = model.getWeights();
        result.bias = model.getBias();

        return result;
    }
};

struct Evaluation {
    double mse;
    double rSquared;
};

Evaluation evaluate(
    const TrainingResult& model,
    const vector<Sample>& samples
) {
    if (samples.empty()) {
        throw invalid_argument("Evaluation dataset is empty.");
    }

    double mean = 0.0;

    for (const auto& sample : samples) {
        mean += sample.target;
    }

    mean /= samples.size();

    double residual = 0.0;
    double total = 0.0;

    for (const auto& sample : samples) {
        double prediction = model.bias;

        for (size_t i = 0; i < model.weights.size(); ++i) {
            prediction += model.weights[i] * sample.features[i];
        }

        residual += pow(sample.target - prediction, 2);
        total += pow(sample.target - mean, 2);
    }

    double mse = residual / samples.size();

    double score = total == 0.0
        ? (residual == 0.0 ? 1.0 : 0.0)
        : 1.0 - residual / total;

    return {mse, score};
}

void exportCSV(
    const vector<TrainingResult>& results,
    const string& filename
) {
    ofstream file(filename);

    if (!file) {
        throw runtime_error("Unable to open CSV output.");
    }

    file << "optimizer,epoch,loss,gradient_norm,batch_count\n";
    file << setprecision(12);

    for (const auto& result : results) {
        for (const auto& metrics : result.history) {
            file << result.optimizer << ','
                 << metrics.epoch << ','
                 << metrics.loss << ','
                 << metrics.gradientNorm << ','
                 << metrics.batchCount << '\n';
        }
    }

    if (!file.good()) {
        throw runtime_error("CSV writing failed.");
    }
}

int main() {
    try {
        cout << "GRADIENT DESCENT GOVERNANCE ENGINE\n";
        cout << string(65, '=') << "\n\n";

        vector<Sample> training = Dataset::generate(240, 0.5, 42);
        vector<Sample> testing = Dataset::generate(80, 0.5, 99);

        vector<TrainingConfig> configurations = {
            {
                OptimizerType::Batch,
                0.01, 0.001, 10.0, 1e-10,
                150, 16, 25, 42
            },
            {
                OptimizerType::Stochastic,
                0.003, 0.001, 10.0, 1e-10,
                150, 1, 25, 42
            },
            {
                OptimizerType::MiniBatch,
                0.01, 0.001, 10.0, 1e-10,
                150, 16, 25, 42
            }
        };

        vector<TrainingResult> results;

        for (const auto& config : configurations) {
            GradientDescentEngine engine(config);

            TrainingResult result = engine.train(training);

            Evaluation metrics = evaluate(result, testing);

            cout << "\nOptimizer: " << result.optimizer << '\n';

            cout << "Weights: ";
            for (double weight : result.weights) {
                cout << fixed << setprecision(4)
                     << weight << ' ';
            }

            cout << "\nBias: " << result.bias;
            cout << "\nTest MSE: " << metrics.mse;
            cout << "\nR-squared: " << metrics.rSquared;
            cout << "\nEpochs: " << result.history.size();
            cout << "\nConverged: "
                 << boolalpha << result.converged << "\n";

            results.push_back(move(result));
        }

        exportCSV(results, "gradient_descent_results.csv");

        cout << "\nCSV report exported successfully.\n";

        // Validate rejection of an invalid optimization policy.
        try {
            TrainingConfig invalid = configurations.front();
            invalid.learningRate = -0.5;

            validateConfig(invalid);
        } catch (const invalid_argument& error) {
            cout << "Configuration validation: PASSED ("
                 << error.what() << ")\n";
        }

    } catch (const exception& error) {
        cerr << "Execution failure: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
