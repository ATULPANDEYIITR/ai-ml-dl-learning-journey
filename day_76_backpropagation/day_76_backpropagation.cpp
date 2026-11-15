#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

/*
 * Backpropagation and computational graph demonstration.
 *
 * Implements:
 * 1. Scalar automatic differentiation.
 * 2. Topological graph traversal.
 * 3. Reverse-mode differentiation.
 * 4. Gradient accumulation.
 * 5. Numerical gradient verification.
 * 6. A multilayer perceptron.
 * 7. XOR training using batch gradient descent.
 */

class GraphNode {
public:
    double value;
    double gradient = 0.0;
    vector<GraphNode*> parents;
    string operation;

    GraphNode(double value, string operation = "")
        : value(value), operation(std::move(operation)) {
        if (!isfinite(value)) {
            throw invalid_argument("Non-finite graph value.");
        }
    }

    static GraphNode* add(GraphNode* a, GraphNode* b) {
        auto* result = new GraphNode(
            a->value + b->value, "add"
        );

        result->parents = {a, b};
        return result;
    }

    static GraphNode* multiply(GraphNode* a, GraphNode* b) {
        auto* result = new GraphNode(
            a->value * b->value, "multiply"
        );

        result->parents = {a, b};
        return result;
    }

    static GraphNode* tanhNode(GraphNode* input) {
        auto* result = new GraphNode(
            tanh(input->value), "tanh"
        );

        result->parents = {input};
        return result;
    }
};

class ComputationalGraph {
private:
    vector<GraphNode*> nodes;

    void visit(
        GraphNode* node,
        vector<GraphNode*>& topology,
        vector<bool>& visited,
        const unordered_map<GraphNode*, size_t>& indices
    ) {
        size_t index = indices.at(node);

        if (visited[index]) return;

        visited[index] = true;

        for (auto* parent : node->parents) {
            visit(parent, topology, visited, indices);
        }

        topology.push_back(node);
    }

public:
    ~ComputationalGraph() {
        for (auto* node : nodes) {
            delete node;
        }
    }

    GraphNode* create(double value, string operation = "") {
        auto* node = new GraphNode(value, std::move(operation));
        nodes.push_back(node);
        return node;
    }

    GraphNode* add(GraphNode* a, GraphNode* b) {
        auto* result = GraphNode::add(a, b);
        nodes.push_back(result);
        return result;
    }

    GraphNode* multiply(GraphNode* a, GraphNode* b) {
        auto* result = GraphNode::multiply(a, b);
        nodes.push_back(result);
        return result;
    }

    GraphNode* tanhNode(GraphNode* input) {
        auto* result = GraphNode::tanhNode(input);
        nodes.push_back(result);
        return result;
    }

    void backward(GraphNode* output) {
        if (!output) {
            throw invalid_argument("Null output node.");
        }

        unordered_map<GraphNode*, size_t> indices;

        for (size_t i = 0; i < nodes.size(); ++i) {
            indices[nodes[i]] = i;
        }

        vector<GraphNode*> topology;
        vector<bool> visited(nodes.size(), false);

        visit(output, topology, visited, indices);

        for (auto* node : topology) {
            node->gradient = 0.0;
        }

        output->gradient = 1.0;

        for (auto it = topology.rbegin();
             it != topology.rend(); ++it) {

            GraphNode* node = *it;

            if (node->operation == "add") {
                node->parents[0]->gradient += node->gradient;
                node->parents[1]->gradient += node->gradient;
            }

            else if (node->operation == "multiply") {
                auto* a = node->parents[0];
                auto* b = node->parents[1];

                a->gradient += b->value * node->gradient;
                b->gradient += a->value * node->gradient;
            }

            else if (node->operation == "tanh") {
                auto* input = node->parents[0];

                input->gradient +=
                    (1.0 - node->value * node->value)
                    * node->gradient;
            }
        }
    }
};

class Neuron {
private:
    vector<double> weights;
    double bias;

public:
    Neuron(size_t inputs, mt19937& generator) {
        uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (size_t i = 0; i < inputs; ++i) {
            weights.push_back(distribution(generator));
        }

        bias = 0.0;
    }

    double forward(const vector<double>& inputs) const {
        if (inputs.size() != weights.size()) {
            throw invalid_argument("Neuron input mismatch.");
        }

        double sum = bias;

        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * inputs[i];
        }

        return tanh(sum);
    }

    const vector<double>& getWeights() const {
        return weights;
    }

    double getBias() const {
        return bias;
    }

    vector<double>& mutableWeights() {
        return weights;
    }

    double& mutableBias() {
        return bias;
    }
};

struct Sample {
    vector<double> input;
    double target;
};

class NeuralNetwork {
private:
    vector<Neuron> hidden;
    Neuron output;
    double learningRate;

    static double squaredError(double prediction, double target) {
        double difference = prediction - target;
        return difference * difference;
    }

public:
    NeuralNetwork(
        size_t inputSize,
        size_t hiddenSize,
        double learningRate,
        mt19937& generator
    )
        : output(hiddenSize, generator),
          learningRate(learningRate) {

        if (inputSize == 0 || hiddenSize == 0 ||
            learningRate <= 0) {
            throw invalid_argument("Invalid network configuration.");
        }

        for (size_t i = 0; i < hiddenSize; ++i) {
            hidden.emplace_back(inputSize, generator);
        }
    }

    double predict(const vector<double>& inputs) const {
        if (inputs.size() != hidden[0].getWeights().size()) {
            throw invalid_argument("Network input mismatch.");
        }

        vector<double> activations;

        for (const auto& neuron : hidden) {
            activations.push_back(neuron.forward(inputs));
        }

        return output.forward(activations);
    }

    double trainBatch(const vector<Sample>& dataset) {
        if (dataset.empty()) {
            throw invalid_argument("Empty training dataset.");
        }

        vector<vector<double>> hiddenWeightGradients(
            hidden.size(),
            vector<double>(hidden[0].getWeights().size(), 0.0)
        );

        vector<double> hiddenBiasGradients(hidden.size(), 0.0);

        vector<double> outputWeightGradients(hidden.size(), 0.0);
        double outputBiasGradient = 0.0;

        double totalLoss = 0.0;

        for (const auto& sample : dataset) {
            if (sample.input.size() != hidden[0].getWeights().size() ||
                !isfinite(sample.target)) {
                throw invalid_argument("Invalid training sample.");
            }

            vector<double> hiddenValues;
            vector<double> hiddenDerivatives;

            for (const auto& neuron : hidden) {
                double value = neuron.forward(sample.input);

                hiddenValues.push_back(value);
                hiddenDerivatives.push_back(1.0 - value * value);
            }

            double prediction = output.forward(hiddenValues);

            totalLoss += squaredError(prediction, sample.target);

            // Derivative of mean squared error.
            double outputDelta =
                2.0 * (prediction - sample.target) /
                static_cast<double>(dataset.size());

            outputBiasGradient += outputDelta;

            for (size_t j = 0; j < hidden.size(); ++j) {
                outputWeightGradients[j] +=
                    outputDelta * hiddenValues[j];

                double hiddenDelta =
                    outputDelta *
                    output.getWeights()[j] *
                    hiddenDerivatives[j];

                hiddenBiasGradients[j] += hiddenDelta;

                for (size_t k = 0; k < sample.input.size(); ++k) {
                    hiddenWeightGradients[j][k] +=
                        hiddenDelta * sample.input[k];
                }
            }
        }

        // Validate the entire update before modifying model state.
        auto finite = [](double value) {
            return isfinite(value);
        };

        for (const auto& row : hiddenWeightGradients) {
            for (double gradient : row) {
                if (!finite(gradient)) {
                    throw runtime_error("Invalid hidden gradient.");
                }
            }
        }

        for (double gradient : outputWeightGradients) {
            if (!finite(gradient)) {
                throw runtime_error("Invalid output gradient.");
            }
        }

        if (!finite(outputBiasGradient)) {
            throw runtime_error("Invalid bias gradient.");
        }

        // Apply batch updates.
        for (size_t j = 0; j < hidden.size(); ++j) {
            auto& weights = hidden[j].mutableWeights();

            for (size_t k = 0; k < weights.size(); ++k) {
                weights[k] -=
                    learningRate * hiddenWeightGradients[j][k];
            }

            hidden[j].mutableBias() -=
                learningRate * hiddenBiasGradients[j];
        }

        auto& outputWeights = output.mutableWeights();

        for (size_t j = 0; j < outputWeights.size(); ++j) {
            outputWeights[j] -=
                learningRate * outputWeightGradients[j];
        }

        output.mutableBias() -=
            learningRate * outputBiasGradient;

        if (!finite(totalLoss)) {
            throw runtime_error("Non-finite training loss.");
        }

        return totalLoss / dataset.size();
    }
};

void demonstrateComputationalGraph() {
    cout << "\n1. Computational Graph\n";

    ComputationalGraph graph;

    auto* x = graph.create(2.0);
    auto* three = graph.create(3.0);
    auto* two = graph.create(2.0);

    auto* square = graph.multiply(x, x);
    auto* linear = graph.multiply(three, x);
    auto* partial = graph.add(square, linear);
    auto* result = graph.add(partial, two);

    graph.backward(result);

    cout << "f(x) = x^2 + 3x + 2\n";
    cout << "f(2) = " << result->value << '\n';
    cout << "df/dx = " << x->gradient << '\n';

    if (abs(x->gradient - 7.0) > 1e-9) {
        throw runtime_error("Gradient test failed.");
    }
}

void demonstrateNumericalGradient() {
    cout << "\n2. Numerical Gradient Verification\n";

    double x = 1.5;
    double epsilon = 1e-6;

    auto function = [](double value) {
        return exp(value * value);
    };

    double numerical =
        (function(x + epsilon) - function(x - epsilon)) /
        (2.0 * epsilon);

    double analytical = 2.0 * x * exp(x * x);

    cout << "Analytical: " << analytical << '\n';
    cout << "Numerical:  " << numerical << '\n';

    if (abs(analytical - numerical) > 1e-3) {
        throw runtime_error("Numerical verification failed.");
    }
}

void trainXOR() {
    cout << "\n3. Neural Network Training: XOR\n";

    mt19937 generator(42);

    NeuralNetwork network(2, 4, 0.08, generator);

    vector<Sample> dataset = {
        {{0.0, 0.0}, 0.0},
        {{0.0, 1.0}, 1.0},
        {{1.0, 0.0}, 1.0},
        {{1.0, 1.0}, 0.0}
    };

    const int epochs = 3000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = network.trainBatch(dataset);

        if (epoch % 500 == 0 || epoch == epochs - 1) {
            cout << "Epoch " << setw(4) << epoch
                 << " | Loss " << fixed << setprecision(6)
                 << loss << '\n';
        }
    }

    cout << "\nPredictions\n";

    for (const auto& sample : dataset) {
        cout << sample.input[0] << ", "
             << sample.input[1] << " -> "
             << network.predict(sample.input) << '\n';
    }
}

int main() {
    try {
        cout << fixed << setprecision(6);

        demonstrateComputationalGraph();
        demonstrateNumericalGradient();
        trainXOR();

        cout << "\nAll demonstrations completed.\n";
    }
    catch (const exception& error) {
        cerr << "Execution error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
