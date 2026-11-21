#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

/*
 * Regularization in Deep Learning
 *
 * C++17 implementation of a regularized neural-network training engine.
 *
 * Techniques:
 * - Inverted dropout
 * - L2 weight decay
 * - Early stopping
 * - Gaussian data augmentation
 *
 * Scenario:
 * A binary classifier predicts whether a numerical observation belongs
 * to a particular nonlinear class.
 */

enum class RegularizationMode {
    NONE,
    DROPOUT,
    WEIGHT_DECAY,
    COMBINED
};

struct Sample {
    vector<double> features;
    int label;
};

struct Configuration {
    double learningRate = 0.035;
    double dropoutRate = 0.20;
    double weightDecay = 0.005;
    double augmentationNoise = 0.04;

    int maximumEpochs = 120;
    int patience = 12;
    double minimumImprovement = 1e-5;
    int batchSize = 16;
    unsigned int seed = 42;

    void validate() const {
        if (learningRate <= 0 || !isfinite(learningRate))
            throw invalid_argument("Invalid learning rate.");

        if (dropoutRate < 0 || dropoutRate >= 1)
            throw invalid_argument("Invalid dropout rate.");

        if (weightDecay < 0 || augmentationNoise < 0)
            throw invalid_argument("Negative regularization parameter.");

        if (maximumEpochs <= 0 || patience <= 0 || batchSize <= 0)
            throw invalid_argument("Invalid training dimensions.");

        if (minimumImprovement < 0)
            throw invalid_argument("Invalid minimum improvement.");
    }
};

class Dataset {
private:
    vector<Sample> samples;
    size_t dimensions;

public:
    explicit Dataset(vector<Sample> input)
        : samples(move(input)), dimensions(0) {

        if (samples.empty())
            throw invalid_argument("Dataset cannot be empty.");

        dimensions = samples.front().features.size();

        if (dimensions == 0)
            throw invalid_argument("Feature vectors cannot be empty.");

        for (const auto& sample : samples) {
            if (sample.features.size() != dimensions)
                throw invalid_argument("Inconsistent feature dimensions.");

            if (sample.label != 0 && sample.label != 1)
                throw invalid_argument("Labels must be binary.");

            for (double value : sample.features) {
                if (!isfinite(value))
                    throw invalid_argument("Non-finite feature.");
            }
        }
    }

    const vector<Sample>& getSamples() const {
        return samples;
    }

    size_t size() const {
        return samples.size();
    }

    size_t featureCount() const {
        return dimensions;
    }
};

class RandomGenerator {
private:
    mt19937 engine;

public:
    explicit RandomGenerator(unsigned int seed)
        : engine(seed) {}

    double uniform() {
        return generate_canonical<double, 32>(engine);
    }

    double normal(double mean = 0.0, double standardDeviation = 1.0) {
        normal_distribution<double> distribution(
            mean, standardDeviation
        );

        return distribution(engine);
    }

    void shuffle(vector<size_t>& indices) {
        std::shuffle(indices.begin(), indices.end(), engine);
    }
};

class DataAugmenter {
private:
    double noise;
    RandomGenerator random;

public:
    DataAugmenter(double noiseLevel, unsigned int seed)
        : noise(noiseLevel), random(seed) {

        if (noiseLevel < 0)
            throw invalid_argument("Noise cannot be negative.");
    }

    vector<double> transform(const vector<double>& input) {
        vector<double> output = input;

        for (double& value : output)
            value += random.normal(0.0, noise);

        return output;
    }
};

class NeuralNetwork {
private:
    size_t inputSize;
    size_t hiddenSize;

    vector<vector<double>> w1;
    vector<double> b1;

    vector<double> w2;
    double b2 = 0.0;

    RandomGenerator random;

    static double sigmoid(double x) {
        x = max(-500.0, min(500.0, x));
        return 1.0 / (1.0 + exp(-x));
    }

    static double relu(double x) {
        return max(0.0, x);
    }

public:
    NeuralNetwork(size_t inputs, size_t hidden, unsigned int seed)
        : inputSize(inputs),
          hiddenSize(hidden),
          w1(hidden, vector<double>(inputs)),
          b1(hidden, 0.0),
          w2(hidden, 0.0),
          random(seed) {

        if (inputs == 0 || hidden == 0)
            throw invalid_argument("Invalid network dimensions.");

        for (auto& row : w1) {
            for (double& weight : row)
                weight = (random.uniform() - 0.5) * 0.8;
        }

        for (double& weight : w2)
            weight = (random.uniform() - 0.5) * 0.8;
    }

    struct ForwardCache {
        vector<double> hidden;
        vector<double> dropped;
        vector<double> mask;
        double probability;
    };

    ForwardCache forward(
        const vector<double>& input,
        bool training,
        double dropoutRate,
        RandomGenerator& rng
    ) const {

        if (input.size() != inputSize)
            throw invalid_argument("Incorrect input dimension.");

        vector<double> hidden(hiddenSize);
        vector<double> mask(hiddenSize, 1.0);
        vector<double> dropped(hiddenSize);

        for (size_t j = 0; j < hiddenSize; ++j) {
            double z = b1[j];

            for (size_t k = 0; k < inputSize; ++k)
                z += w1[j][k] * input[k];

            hidden[j] = relu(z);
        }

        // Inverted dropout is active only during training.
        if (training && dropoutRate > 0) {
            double scale = 1.0 / (1.0 - dropoutRate);

            for (size_t j = 0; j < hiddenSize; ++j)
                mask[j] = rng.uniform() >= dropoutRate ? scale : 0.0;
        }

        double logit = b2;

        for (size_t j = 0; j < hiddenSize; ++j) {
            dropped[j] = hidden[j] * mask[j];
            logit += w2[j] * dropped[j];
        }

        return {
            hidden,
            dropped,
            mask,
            sigmoid(logit)
        };
    }

    void trainBatch(
        const vector<Sample>& batch,
        const Configuration& config,
        RandomGenerator& rng
    ) {

        if (batch.empty())
            throw invalid_argument("Empty training batch.");

        vector<vector<double>> dw1(
            hiddenSize, vector<double>(inputSize, 0.0)
        );

        vector<double> db1(hiddenSize, 0.0);
        vector<double> dw2(hiddenSize, 0.0);
        double db2 = 0.0;

        for (const auto& sample : batch) {
            auto cache = forward(
                sample.features,
                true,
                config.dropoutRate,
                rng
            );

            double outputError = cache.probability - sample.label;

            for (size_t j = 0; j < hiddenSize; ++j)
                dw2[j] += outputError * cache.dropped[j];

            db2 += outputError;

            for (size_t j = 0; j < hiddenSize; ++j) {
                if (cache.hidden[j] <= 0)
                    continue;

                double hiddenError =
                    outputError * w2[j] * cache.mask[j];

                for (size_t k = 0; k < inputSize; ++k)
                    dw1[j][k] += hiddenError * sample.features[k];

                db1[j] += hiddenError;
            }
        }

        double count = static_cast<double>(batch.size());

        // Apply averaged gradients and L2 regularization.
        for (size_t j = 0; j < hiddenSize; ++j) {
            for (size_t k = 0; k < inputSize; ++k) {
                double gradient = dw1[j][k] / count
                    + config.weightDecay * w1[j][k];

                w1[j][k] -= config.learningRate * gradient;
            }

            b1[j] -= config.learningRate * db1[j] / count;

            double gradient = dw2[j] / count
                + config.weightDecay * w2[j];

            w2[j] -= config.learningRate * gradient;
        }

        b2 -= config.learningRate * db2 / count;
    }

    double loss(const Dataset& dataset) const {
        constexpr double epsilon = 1e-12;
        double total = 0.0;

        RandomGenerator unused(1);

        for (const auto& sample : dataset.getSamples()) {
            double p = forward(
                sample.features, false, 0.0, unused
            ).probability;

            total -= sample.label * log(max(epsilon, p))
                + (1 - sample.label) * log(max(epsilon, 1.0 - p));
        }

        return total / dataset.size();
    }

    double accuracy(const Dataset& dataset) const {
        size_t correct = 0;
        RandomGenerator unused(1);

        for (const auto& sample : dataset.getSamples()) {
            double probability = forward(
                sample.features, false, 0.0, unused
            ).probability;

            int prediction = probability >= 0.5 ? 1 : 0;

            if (prediction == sample.label)
                ++correct;
        }

        return static_cast<double>(correct) / dataset.size();
    }

    void applyWeightDecay(double factor) {
        for (auto& row : w1)
            for (double& weight : row)
                weight *= factor;

        for (double& weight : w2)
            weight *= factor;
    }
};

struct TrainingResult {
    int bestEpoch = 0;
    int stoppedEpoch = 0;
    double bestLoss = numeric_limits<double>::infinity();
    double accuracy = 0.0;
    double trainingLoss = 0.0;
};

class TrainingEngine {
private:
    Configuration config;

public:
    explicit TrainingEngine(Configuration configuration)
        : config(configuration) {
        config.validate();
    }

    TrainingResult train(
        const Dataset& training,
        const Dataset& validation
    ) const {

        if (training.featureCount() != validation.featureCount())
            throw invalid_argument("Dataset feature dimensions differ.");

        NeuralNetwork model(
            training.featureCount(),
            8,
            config.seed
        );

        DataAugmenter augmenter(
            config.augmentationNoise,
            config.seed + 1
        );

        RandomGenerator rng(config.seed);

        // Snapshot of model state is required for checkpoint restoration.
        // A copy of the complete network provides a consistent checkpoint.
        NeuralNetwork bestModel = model;

        double bestLoss = numeric_limits<double>::infinity();
        int bestEpoch = 0;
        int staleEpochs = 0;
        int stoppedEpoch = 0;

        vector<size_t> indices(training.size());

        iota(indices.begin(), indices.end(), 0);

        for (int epoch = 1; epoch <= config.maximumEpochs; ++epoch) {
            rng.shuffle(indices);

            const auto& samples = training.getSamples();

            for (size_t start = 0; start < indices.size();
                 start += config.batchSize) {

                size_t end = min(
                    indices.size(),
                    start + static_cast<size_t>(config.batchSize)
                );

                vector<Sample> batch;

                for (size_t i = start; i < end; ++i) {
                    Sample sample = samples[indices[i]];

                    if (config.augmentationNoise > 0)
                        sample.features = augmenter.transform(sample.features);

                    batch.push_back(move(sample));
                }

                model.trainBatch(batch, config, rng);
            }

            double validationLoss = model.loss(validation);

            if (!isfinite(validationLoss))
                throw runtime_error("Non-finite validation loss.");

            if (validationLoss < bestLoss - config.minimumImprovement) {
                bestLoss = validationLoss;
                bestModel = model;
                bestEpoch = epoch;
                staleEpochs = 0;
            } else {
                ++staleEpochs;
            }

            stoppedEpoch = epoch;

            cout << "Epoch " << setw(3) << epoch
                 << " | Train loss: " << fixed << setprecision(5)
                 << model.loss(training)
                 << " | Validation loss: " << validationLoss
                 << '\n';

            if (staleEpochs >= config.patience)
                break;
        }

        // Return the best checkpoint, not the final epoch parameters.
        return {
            bestEpoch,
            stoppedEpoch,
            bestLoss,
            bestModel.accuracy(validation),
            bestModel.loss(training)
        };
    }
};

vector<Sample> generateSamples(size_t count, unsigned int seed) {
    RandomGenerator rng(seed);
    vector<Sample> samples;

    for (size_t i = 0; i < count; ++i) {
        double x = rng.uniform() * 5.0 - 2.5;
        double y = rng.uniform() * 5.0 - 2.5;

        int label = x * x + y * y < 2.0 ? 1 : 0;

        if (rng.uniform() < 0.04)
            label = 1 - label;

        samples.push_back({{x, y}, label});
    }

    return samples;
}

pair<Dataset, Dataset> splitDataset(
    const vector<Sample>& input,
    unsigned int seed
) {
    vector<size_t> indices(input.size());
    iota(indices.begin(), indices.end(), 0);

    RandomGenerator rng(seed);
    rng.shuffle(indices);

    size_t split = static_cast<size_t>(input.size() * 0.75);

    if (split == 0 || split >= input.size())
        throw invalid_argument("Invalid dataset split.");

    vector<Sample> training;
    vector<Sample> validation;

    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < split)
            training.push_back(input[indices[i]]);
        else
            validation.push_back(input[indices[i]]);
    }

    return {
        Dataset(move(training)),
        Dataset(move(validation))
    };
}

int main() {
    try {
        auto samples = generateSamples(240, 42);

        auto [training, validation] = splitDataset(samples, 42);

        vector<pair<string, Configuration>> experiments;

        Configuration baseline;
        baseline.dropoutRate = 0;
        baseline.weightDecay = 0;
        baseline.augmentationNoise = 0;

        Configuration dropout = baseline;
        dropout.dropoutRate = 0.25;

        Configuration decay = baseline;
        decay.weightDecay = 0.01;

        Configuration combined;
        combined.dropoutRate = 0.20;
        combined.weightDecay = 0.005;
        combined.augmentationNoise = 0.04;

        experiments.push_back({"Baseline", baseline});
        experiments.push_back({"Dropout", dropout});
        experiments.push_back({"Weight Decay", decay});
        experiments.push_back({"Combined", combined});

        vector<pair<string, TrainingResult>> results;

        for (const auto& [name, config] : experiments) {
            cout << "\n========================================\n";
            cout << "Experiment: " << name << '\n';
            cout << "========================================\n";

            TrainingEngine engine(config);
            TrainingResult result = engine.train(training, validation);

            cout << "Best epoch: " << result.bestEpoch << '\n';
            cout << "Stopped epoch: " << result.stoppedEpoch << '\n';
            cout << "Best validation loss: "
                 << result.bestLoss << '\n';
            cout << "Validation accuracy: "
                 << result.accuracy * 100 << "%\n";

            results.push_back({name, result});
        }

        sort(
            results.begin(),
            results.end(),
            [](const auto& a, const auto& b) {
                return a.second.bestLoss < b.second.bestLoss;
            }
        );

        cout << "\nExperiment ranking\n";
        cout << "----------------------------------------\n";

        for (const auto& [name, result] : results) {
            cout << left << setw(20) << name
                 << " Loss: " << result.bestLoss
                 << " Accuracy: " << result.accuracy * 100 << "%\n";
        }

        if (!results.empty()) {
            cout << "\nSelected configuration: "
                 << results.front().first << '\n';
        }

    } catch (const exception& error) {
        cerr << "Training error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
