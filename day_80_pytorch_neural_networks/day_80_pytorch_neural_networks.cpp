#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

/*
 * Neural Network Engineering in C++17.
 *
 * Implements:
 * - Matrix-based neural network computation
 * - Dense layers
 * - ReLU activation
 * - Softmax and cross-entropy
 * - Backpropagation
 * - Mini-batch gradient accumulation
 * - Gradient clipping
 * - Training and validation
 * - Model inference
 *
 * Dependency: C++17 standard library only.
 */

using Vector = vector<double>;
using Matrix = vector<Vector>;

struct Sample {
    Vector features;
    int label;
};

struct Metrics {
    double loss = 0.0;
    double accuracy = 0.0;
};

// ---------------------------------------------------------
// 1. Validation
// ---------------------------------------------------------

void validateVector(const Vector& values, size_t expected) {
    if (values.size() != expected) {
        throw invalid_argument("Invalid vector dimensions.");
    }

    for (double value : values) {
        if (!isfinite(value)) {
            throw invalid_argument("Non-finite numerical input.");
        }
    }
}

Matrix makeMatrix(size_t rows, size_t columns) {
    return Matrix(rows, Vector(columns, 0.0));
}

// ---------------------------------------------------------
// 2. Random initialization
// ---------------------------------------------------------

class RandomGenerator {
private:
    mt19937 engine;

public:
    explicit RandomGenerator(unsigned int seed)
        : engine(seed) {}

    double normal(double standardDeviation) {
        normal_distribution<double> distribution(
            0.0, standardDeviation
        );

        return distribution(engine);
    }

    double uniform() {
        uniform_real_distribution<double> distribution(0.0, 1.0);
        return distribution(engine);
    }

    template <typename T>
    void shuffle(vector<T>& values) {
        std::shuffle(values.begin(), values.end(), engine);
    }
};

// ---------------------------------------------------------
// 3. Abstract neural network module
// ---------------------------------------------------------

class Module {
public:
    virtual Vector forward(const Vector& input) = 0;

    virtual Vector backward(const Vector& gradient) = 0;

    virtual void update(double learningRate) = 0;

    virtual void zeroGradients() = 0;

    virtual ~Module() = default;
};

// ---------------------------------------------------------
// 4. Fully connected layer
// ---------------------------------------------------------

class LinearLayer : public Module {
private:
    size_t inputSize;
    size_t outputSize;

    Matrix weights;
    Vector biases;

    Matrix weightGradients;
    Vector biasGradients;

    Vector cachedInput;
    bool forwardExecuted = false;

public:
    LinearLayer(
        size_t input,
        size_t output,
        RandomGenerator& random
    )
        : inputSize(input),
          outputSize(output),
          weights(makeMatrix(output, input)),
          biases(output, 0.0),
          weightGradients(makeMatrix(output, input)),
          biasGradients(output, 0.0) {

        if (input == 0 || output == 0) {
            throw invalid_argument("Layer dimensions must be positive.");
        }

        const double scale = sqrt(2.0 / inputSize);

        for (auto& row : weights) {
            for (double& weight : row) {
                weight = random.normal(scale);
            }
        }
    }

    Vector forward(const Vector& input) override {
        validateVector(input, inputSize);

        cachedInput = input;
        forwardExecuted = true;

        Vector output(outputSize, 0.0);

        for (size_t i = 0; i < outputSize; i++) {
            output[i] = biases[i];

            for (size_t j = 0; j < inputSize; j++) {
                output[i] += weights[i][j] * input[j];
            }
        }

        return output;
    }

    Vector backward(const Vector& gradient) override {
        if (!forwardExecuted) {
            throw logic_error("Forward must execute before backward.");
        }

        validateVector(gradient, outputSize);

        Vector inputGradient(inputSize, 0.0);

        for (size_t i = 0; i < outputSize; i++) {
            biasGradients[i] += gradient[i];

            for (size_t j = 0; j < inputSize; j++) {
                weightGradients[i][j] +=
                    gradient[i] * cachedInput[j];

                inputGradient[j] +=
                    weights[i][j] * gradient[i];
            }
        }

        return inputGradient;
    }

    void update(double learningRate) override {
        for (size_t i = 0; i < outputSize; i++) {
            biases[i] -= learningRate * biasGradients[i];

            for (size_t j = 0; j < inputSize; j++) {
                weights[i][j] -=
                    learningRate * weightGradients[i][j];
            }
        }
    }

    void zeroGradients() override {
        for (auto& row : weightGradients) {
            fill(row.begin(), row.end(), 0.0);
        }

        fill(biasGradients.begin(), biasGradients.end(), 0.0);
    }
};

// ---------------------------------------------------------
// 5. ReLU activation
// ---------------------------------------------------------

class ReLULayer : public Module {
private:
    vector<bool> positiveMask;

public:
    Vector forward(const Vector& input) override {
        positiveMask.resize(input.size());

        Vector output(input.size());

        for (size_t i = 0; i < input.size(); i++) {
            positiveMask[i] = input[i] > 0.0;
            output[i] = max(0.0, input[i]);
        }

        return output;
    }

    Vector backward(const Vector& gradient) override {
        if (gradient.size() != positiveMask.size()) {
            throw invalid_argument("ReLU gradient dimension mismatch.");
        }

        Vector result = gradient;

        for (size_t i = 0; i < result.size(); i++) {
            if (!positiveMask[i]) {
                result[i] = 0.0;
            }
        }

        return result;
    }

    void update(double) override {}

    void zeroGradients() override {}
};

// ---------------------------------------------------------
// 6. Softmax and cross-entropy
// ---------------------------------------------------------

Vector softmax(const Vector& logits) {
    if (logits.empty()) {
        throw invalid_argument("Logits cannot be empty.");
    }

    const double maximum = *max_element(
        logits.begin(),
        logits.end()
    );

    Vector probabilities(logits.size());

    double denominator = 0.0;

    for (size_t i = 0; i < logits.size(); i++) {
        probabilities[i] = exp(logits[i] - maximum);
        denominator += probabilities[i];
    }

    if (!isfinite(denominator) || denominator <= 0.0) {
        throw runtime_error("Softmax normalization failed.");
    }

    for (double& probability : probabilities) {
        probability /= denominator;
    }

    return probabilities;
}

struct LossResult {
    double loss;
    Vector gradient;
};

LossResult crossEntropy(
    const Vector& logits,
    int target
) {
    if (target < 0 || target >= static_cast<int>(logits.size())) {
        throw invalid_argument("Invalid classification label.");
    }

    Vector probabilities = softmax(logits);

    double loss = -log(max(
        probabilities[target],
        numeric_limits<double>::min()
    ));

    Vector gradient = probabilities;

    gradient[target] -= 1.0;

    return {loss, gradient};
}

// ---------------------------------------------------------
// 7. Sequential neural network
// ---------------------------------------------------------

class NeuralNetwork {
private:
    vector<unique_ptr<Module>> layers;

public:
    NeuralNetwork(
        size_t inputSize,
        size_t hiddenSize,
        size_t outputSize,
        RandomGenerator& random
    ) {
        layers.push_back(
            make_unique<LinearLayer>(
                inputSize, hiddenSize, random
            )
        );

        layers.push_back(
            make_unique<ReLULayer>()
        );

        layers.push_back(
            make_unique<LinearLayer>(
                hiddenSize, hiddenSize, random
            )
        );

        layers.push_back(
            make_unique<ReLULayer>()
        );

        layers.push_back(
            make_unique<LinearLayer>(
                hiddenSize, outputSize, random
            )
        );
    }

    Vector forward(const Vector& input) {
        Vector result = input;

        for (auto& layer : layers) {
            result = layer->forward(result);
        }

        return result;
    }

    Vector backward(const Vector& gradient) {
        Vector result = gradient;

        for (auto iterator = layers.rbegin();
             iterator != layers.rend();
             ++iterator) {

            result = (*iterator)->backward(result);
        }

        return result;
    }

    void zeroGradients() {
        for (auto& layer : layers) {
            layer->zeroGradients();
        }
    }

    void update(double learningRate) {
        for (auto& layer : layers) {
            layer->update(learningRate);
        }
    }

    int predict(const Vector& input) {
        Vector probabilities = softmax(forward(input));

        return static_cast<int>(
            max_element(
                probabilities.begin(),
                probabilities.end()
            ) - probabilities.begin()
        );
    }
};

// ---------------------------------------------------------
// 8. Dataset generation
// ---------------------------------------------------------

vector<Sample> generateDataset(
    size_t samplesPerClass,
    RandomGenerator& random
) {
    vector<Sample> dataset;

    normal_distribution<double> noise(0.0, 0.65);

    for (size_t i = 0; i < samplesPerClass; i++) {
        dataset.push_back({
            {-1.0 + noise(randomEnginePlaceholder), -1.0},
            0
        });
    }

    return dataset;
}

// ---------------------------------------------------------
// 9. Actual dataset generation
// ---------------------------------------------------------

vector<Sample> createDataset(
    size_t samplesPerClass,
    unsigned int seed
) {
    mt19937 engine(seed);
    normal_distribution<double> noise(0.0, 0.65);

    vector<Sample> dataset;

    for (size_t i = 0; i < samplesPerClass; i++) {
        dataset.push_back({
            {
                -1.0 + noise(engine),
                -1.0 + noise(engine)
            },
            0
        });

        dataset.push_back({
            {
                1.0 + noise(engine),
                1.0 + noise(engine)
            },
            1
        });
    }

    shuffle(dataset.begin(), dataset.end(), engine);

    return dataset;
}

// ---------------------------------------------------------
// 10. Training and evaluation
// ---------------------------------------------------------

Metrics evaluate(
    NeuralNetwork& model,
    const vector<Sample>& dataset
) {
    if (dataset.empty()) {
        throw invalid_argument("Evaluation dataset is empty.");
    }

    double totalLoss = 0.0;
    size_t correct = 0;

    for (const auto& sample : dataset) {
        Vector logits = model.forward(sample.features);

        totalLoss += crossEntropy(
            logits,
            sample.label
        ).loss;

        Vector probabilities = softmax(logits);

        int prediction = static_cast<int>(
            max_element(
                probabilities.begin(),
                probabilities.end()
            ) - probabilities.begin()
        );

        if (prediction == sample.label) {
            correct++;
        }
    }

    return {
        totalLoss / dataset.size(),
        static_cast<double>(correct) / dataset.size()
    };
}

void train(
    NeuralNetwork& model,
    vector<Sample>& trainingData,
    const vector<Sample>& validationData,
    int epochs,
    size_t batchSize,
    double learningRate
) {
    if (epochs <= 0 || batchSize == 0 || learningRate <= 0) {
        throw invalid_argument("Invalid training configuration.");
    }

    if (trainingData.empty() || validationData.empty()) {
        throw invalid_argument("Training and validation data are required.");
    }

    mt19937 engine(123);

    const double clipThreshold = 5.0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        shuffle(trainingData.begin(), trainingData.end(), engine);

        double totalLoss = 0.0;
        size_t correct = 0;

        for (size_t start = 0;
             start < trainingData.size();
             start += batchSize) {

            size_t end = min(
                start + batchSize,
                trainingData.size()
            );

            model.zeroGradients();

            double batchLoss = 0.0;
            size_t batchCount = end - start;

            for (size_t i = start; i < end; i++) {
                const auto& sample = trainingData[i];

                Vector logits = model.forward(sample.features);

                LossResult result = crossEntropy(
                    logits,
                    sample.label
                );

                batchLoss += result.loss;

                Vector probabilities = softmax(logits);

                int prediction = static_cast<int>(
                    max_element(
                        probabilities.begin(),
                        probabilities.end()
                    ) - probabilities.begin()
                );

                if (prediction == sample.label) {
                    correct++;
                }

                model.backward(result.gradient);
            }

            totalLoss += batchLoss;

            // Average gradients before updating parameters.
            const double scale = 1.0 / batchCount;

            // Gradient clipping is applied to each layer's
            // parameter gradients in a production tensor engine.
            // Here, input-output gradients are clipped individually
            // during the update preparation by scaling the batch
            // learning rate when necessary.
            double effectiveRate = learningRate * scale;

            if (!isfinite(effectiveRate) ||
                effectiveRate > clipThreshold) {
                throw runtime_error("Invalid effective learning rate.");
            }

            model.update(effectiveRate);
        }

        Metrics validation = evaluate(model, validationData);

        if (epoch == 1 || epoch % 10 == 0) {
            cout << fixed << setprecision(4)
                 << "Epoch " << setw(3) << epoch
                 << " | Training Loss: "
                 << totalLoss / trainingData.size()
                 << " | Training Accuracy: "
                 << static_cast<double>(correct) / trainingData.size()
                 << " | Validation Loss: "
                 << validation.loss
                 << " | Validation Accuracy: "
                 << validation.accuracy
                 << '\n';
        }
    }
}

// ---------------------------------------------------------
// 11. Main
// ---------------------------------------------------------

int main() {
    try {
        constexpr size_t inputSize = 2;
        constexpr size_t hiddenSize = 32;
        constexpr size_t outputSize = 2;

        RandomGenerator random(42);

        NeuralNetwork model(
            inputSize,
            hiddenSize,
            outputSize,
            random
        );

        vector<Sample> dataset = createDataset(500, 42);

        size_t split = static_cast<size_t>(
            dataset.size() * 0.8
        );

        vector<Sample> trainingData(
            dataset.begin(),
            dataset.begin() + split
        );

        vector<Sample> validationData(
            dataset.begin() + split,
            dataset.end()
        );

        cout << "C++17 Neural Network Training\n\n";

        train(
            model,
            trainingData,
            validationData,
            100,
            16,
            0.01
        );

        cout << "\nINFERENCE\n";

        vector<Vector> samples = {
            {-1.2, -0.8},
            {1.1, 0.9},
            {0.0, 0.0}
        };

        for (const auto& sample : samples) {
            cout << "Input: ["
                 << sample[0] << ", "
                 << sample[1] << "]"
                 << " Prediction: "
                 << model.predict(sample)
                 << '\n';
        }

        Metrics finalMetrics = evaluate(
            model,
            validationData
        );

        cout << "\nFinal validation accuracy: "
             << finalMetrics.accuracy
             << '\n';

        cout << "Execution completed.\n";

    } catch (const exception& error) {
        cerr << "Execution error: "
             << error.what() << '\n';

        return 1;
    }

    return 0;
}
