#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * C++17 Activation Function Engine.
 *
 * Implements:
 * - Sigmoid
 * - Tanh
 * - ReLU
 * - Leaky ReLU
 * - GELU
 * - Softmax
 * - Activation derivatives
 * - Softmax Jacobian-vector product
 * - Classification loss
 * - Activation registry
 * - Neural network inference
 */

using namespace std;

enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    GELU,
    SOFTMAX
};

string activationName(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::RELU: return "ReLU";
        case ActivationType::LEAKY_RELU: return "Leaky ReLU";
        case ActivationType::GELU: return "GELU";
        case ActivationType::SOFTMAX: return "Softmax";
    }

    throw invalid_argument("Unknown activation.");
}

void validateNumber(double x) {
    if (!isfinite(x)) {
        throw invalid_argument("Input must be finite.");
    }
}

void validateVector(const vector<double>& values) {
    if (values.empty()) {
        throw invalid_argument("Vector cannot be empty.");
    }

    for (double value : values) {
        validateNumber(value);
    }
}

class ActivationFunction {
public:
    static double sigmoid(double x) {
        validateNumber(x);

        if (x >= 0) {
            double z = exp(-x);
            return 1.0 / (1.0 + z);
        }

        double z = exp(x);
        return z / (1.0 + z);
    }

    static double sigmoidDerivative(double x) {
        double y = sigmoid(x);
        return y * (1.0 - y);
    }

    static double tanhActivation(double x) {
        validateNumber(x);
        return std::tanh(x);
    }

    static double tanhDerivative(double x) {
        double y = tanhActivation(x);
        return 1.0 - y * y;
    }

    static double relu(double x) {
        validateNumber(x);
        return max(0.0, x);
    }

    static double reluDerivative(double x) {
        validateNumber(x);
        return x > 0.0 ? 1.0 : 0.0;
    }

    static double leakyRelu(double x, double alpha = 0.01) {
        validateNumber(x);
        validateNumber(alpha);

        if (alpha < 0) {
            throw invalid_argument("Alpha cannot be negative.");
        }

        return x > 0 ? x : alpha * x;
    }

    static double leakyReluDerivative(
        double x,
        double alpha = 0.01
    ) {
        validateNumber(x);
        validateNumber(alpha);

        if (alpha < 0) {
            throw invalid_argument("Alpha cannot be negative.");
        }

        return x > 0 ? 1.0 : alpha;
    }

    static double gelu(double x) {
        validateNumber(x);

        const double c = sqrt(2.0 / M_PI);
        const double inner = c * (x + 0.044715 * x * x * x);

        return 0.5 * x * (1.0 + tanh(inner));
    }

    static double geluDerivative(double x) {
        validateNumber(x);

        const double c = sqrt(2.0 / M_PI);
        const double inner = c * (x + 0.044715 * x * x * x);

        const double t = tanh(inner);
        const double innerDerivative =
            c * (1.0 + 3.0 * 0.044715 * x * x);

        return 0.5 * (1.0 + t)
            + 0.5 * x * (1.0 - t * t) * innerDerivative;
    }

    static vector<double> softmax(
        const vector<double>& logits
    ) {
        validateVector(logits);

        const double maximum = *max_element(
            logits.begin(), logits.end()
        );

        vector<double> result;
        result.reserve(logits.size());

        for (double value : logits) {
            result.push_back(exp(value - maximum));
        }

        const double denominator = accumulate(
            result.begin(),
            result.end(),
            0.0
        );

        if (!isfinite(denominator) || denominator <= 0) {
            throw runtime_error("Softmax normalization failed.");
        }

        for (double& value : result) {
            value /= denominator;
        }

        return result;
    }

    static vector<vector<double>> softmaxJacobian(
        const vector<double>& probabilities
    ) {
        validateVector(probabilities);

        double total = accumulate(
            probabilities.begin(),
            probabilities.end(),
            0.0
        );

        for (double p : probabilities) {
            if (p < 0) {
                throw invalid_argument("Negative probability.");
            }
        }

        if (abs(total - 1.0) > 1e-8) {
            throw invalid_argument(
                "Probabilities must sum to one."
            );
        }

        const size_t n = probabilities.size();

        vector<vector<double>> jacobian(
            n,
            vector<double>(n, 0.0)
        );

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                jacobian[i][j] =
                    probabilities[i] *
                    ((i == j ? 1.0 : 0.0) - probabilities[j]);
            }
        }

        return jacobian;
    }

    static vector<double> softmaxJVP(
        const vector<double>& probabilities,
        const vector<double>& upstream
    ) {
        validateVector(probabilities);
        validateVector(upstream);

        if (probabilities.size() != upstream.size()) {
            throw invalid_argument(
                "Gradient dimensions do not match."
            );
        }

        double total = accumulate(
            probabilities.begin(),
            probabilities.end(),
            0.0
        );

        for (double p : probabilities) {
            if (p < 0) {
                throw invalid_argument("Negative probability.");
            }
        }

        if (abs(total - 1.0) > 1e-8) {
            throw invalid_argument(
                "Invalid probability distribution."
            );
        }

        double dot = inner_product(
            probabilities.begin(),
            probabilities.end(),
            upstream.begin(),
            0.0
        );

        vector<double> result(probabilities.size());

        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = probabilities[i] * (upstream[i] - dot);
        }

        return result;
    }
};

class ActivationLayer {
private:
    ActivationType type;
    double alpha;

public:
    ActivationLayer(
        ActivationType activation,
        double negativeSlope = 0.01
    ) : type(activation), alpha(negativeSlope) {

        validateNumber(alpha);

        if (alpha < 0) {
            throw invalid_argument(
                "Negative slope cannot be negative."
            );
        }
    }

    vector<double> forward(
        const vector<double>& inputs
    ) const {
        validateVector(inputs);

        if (type == ActivationType::SOFTMAX) {
            return ActivationFunction::softmax(inputs);
        }

        vector<double> output;
        output.reserve(inputs.size());

        for (double x : inputs) {
            switch (type) {
                case ActivationType::SIGMOID:
                    output.push_back(
                        ActivationFunction::sigmoid(x)
                    );
                    break;

                case ActivationType::TANH:
                    output.push_back(
                        ActivationFunction::tanhActivation(x)
                    );
                    break;

                case ActivationType::RELU:
                    output.push_back(
                        ActivationFunction::relu(x)
                    );
                    break;

                case ActivationType::LEAKY_RELU:
                    output.push_back(
                        ActivationFunction::leakyRelu(x, alpha)
                    );
                    break;

                case ActivationType::GELU:
                    output.push_back(
                        ActivationFunction::gelu(x)
                    );
                    break;

                case ActivationType::SOFTMAX:
                    break;
            }
        }

        return output;
    }

    vector<double> backward(
        const vector<double>& inputs,
        const vector<double>& upstream
    ) const {
        validateVector(inputs);
        validateVector(upstream);

        if (inputs.size() != upstream.size()) {
            throw invalid_argument(
                "Gradient dimensions do not match."
            );
        }

        if (type == ActivationType::SOFTMAX) {
            return ActivationFunction::softmaxJVP(
                ActivationFunction::softmax(inputs),
                upstream
            );
        }

        vector<double> result;
        result.reserve(inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            double derivative = 0.0;

            switch (type) {
                case ActivationType::SIGMOID:
                    derivative =
                        ActivationFunction::sigmoidDerivative(inputs[i]);
                    break;

                case ActivationType::TANH:
                    derivative =
                        ActivationFunction::tanhDerivative(inputs[i]);
                    break;

                case ActivationType::RELU:
                    derivative =
                        ActivationFunction::reluDerivative(inputs[i]);
                    break;

                case ActivationType::LEAKY_RELU:
                    derivative =
                        ActivationFunction::leakyReluDerivative(
                            inputs[i], alpha
                        );
                    break;

                case ActivationType::GELU:
                    derivative =
                        ActivationFunction::geluDerivative(inputs[i]);
                    break;

                case ActivationType::SOFTMAX:
                    break;
            }

            result.push_back(derivative * upstream[i]);
        }

        return result;
    }

    ActivationType getType() const {
        return type;
    }
};

class NeuralNetwork {
private:
    vector<ActivationLayer> layers;

public:
    void addLayer(const ActivationLayer& layer) {
        layers.push_back(layer);
    }

    vector<double> predict(
        vector<double> inputs
    ) const {
        validateVector(inputs);

        for (const auto& layer : layers) {
            inputs = layer.forward(inputs);
        }

        return inputs;
    }

    size_t layerCount() const {
        return layers.size();
    }
};

double crossEntropy(
    const vector<double>& probabilities,
    size_t target
) {
    validateVector(probabilities);

    if (target >= probabilities.size()) {
        throw out_of_range("Target class is invalid.");
    }

    for (double p : probabilities) {
        if (p < 0 || p > 1) {
            throw invalid_argument("Invalid probability.");
        }
    }

    return -log(max(1e-15, probabilities[target]));
}

void printVector(
    const string& label,
    const vector<double>& values
) {
    cout << label << ": ";

    for (double value : values) {
        cout << fixed << setprecision(6) << value << " ";
    }

    cout << '\n';
}

void demonstration() {
    vector<double> inputs = {-3, -1, 0, 1, 3};

    vector<ActivationType> types = {
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::RELU,
        ActivationType::LEAKY_RELU,
        ActivationType::GELU
    };

    cout << "ACTIVATION FUNCTION ENGINE\n";
    cout << string(65, '-') << '\n';

    cout << left << setw(15) << "Input";

    for (auto type : types) {
        cout << setw(15) << activationName(type);
    }

    cout << '\n';

    for (double x : inputs) {
        cout << left << setw(15) << x;

        for (auto type : types) {
            ActivationLayer layer(type);
            cout << setw(15) << layer.forward({x})[0];
        }

        cout << '\n';
    }

    cout << "\nSOFTMAX CLASSIFICATION\n";

    vector<double> logits = {2.0, 1.0, 0.1};

    auto probabilities = ActivationFunction::softmax(logits);

    printVector("Probabilities", probabilities);

    cout << "Cross-entropy loss: "
         << crossEntropy(probabilities, 0) << '\n';

    cout << "\nSOFTMAX JACOBIAN\n";

    auto jacobian =
        ActivationFunction::softmaxJacobian(probabilities);

    for (const auto& row : jacobian) {
        printVector("Row", row);
    }

    cout << "\nBACKPROPAGATION\n";

    ActivationLayer reluLayer(ActivationType::RELU);

    printVector(
        "ReLU backward",
        reluLayer.backward(
            inputs,
            vector<double>(inputs.size(), 1.0)
        )
    );

    printVector(
        "Softmax JVP",
        ActivationFunction::softmaxJVP(
            probabilities,
            {1.0, 0.0, -1.0}
        )
    );

    cout << "\nNEURAL NETWORK PIPELINE\n";

    NeuralNetwork network;

    network.addLayer(
        ActivationLayer(ActivationType::RELU)
    );

    network.addLayer(
        ActivationLayer(ActivationType::SOFTMAX)
    );

    printVector(
        "Network output",
        network.predict({-2, 1, 3})
    );

    cout << "Layer count: " << network.layerCount() << '\n';
}

void runTests() {
    using AF = ActivationFunction;

    if (abs(AF::sigmoid(0) - 0.5) > 1e-10) {
        throw runtime_error("Sigmoid test failed.");
    }

    if (AF::relu(-2) != 0) {
        throw runtime_error("ReLU test failed.");
    }

    if (abs(AF::tanhActivation(0)) > 1e-10) {
        throw runtime_error("Tanh test failed.");
    }

    auto p = AF::softmax({1000, 1001, 1002});

    double sum = accumulate(p.begin(), p.end(), 0.0);

    if (abs(sum - 1.0) > 1e-10) {
        throw runtime_error("Softmax normalization test failed.");
    }

    auto gradient = AF::softmaxJVP(p, {1, 0, -1});

    double gradientSum = accumulate(
        gradient.begin(), gradient.end(), 0.0
    );

    if (abs(gradientSum) > 1e-10) {
        throw runtime_error("Softmax gradient test failed.");
    }

    cout << "\nAll tests passed.\n";
}

int main() {
    try {
        runTests();
        demonstration();
    } catch (const exception& error) {
        cerr << "Execution error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
