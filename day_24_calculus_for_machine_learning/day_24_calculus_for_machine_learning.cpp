/*
 * Calculus for Machine Learning
 * C++17 Technical Case Study
 *
 * Scenario:
 * A small production-style binary classification service is modeled using
 * logistic regression. The implementation develops the mathematical
 * foundations from scalar functions through gradients and Hessians, then
 * uses those ideas to train and validate a classifier.
 *
 * The program demonstrates:
 *   - Functions and function objects
 *   - Numerical limits and derivatives
 *   - Partial derivatives
 *   - Gradients
 *   - Hessians
 *   - Vector and matrix operations
 *   - Logistic regression
 *   - Binary cross-entropy
 *   - Gradient descent
 *   - Validation
 *   - Numerical stability
 *   - Accuracy evaluation
 *   - Complexity considerations
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic calculus_for_ml.cpp -o calculus_for_ml
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

constexpr double EPSILON = 1e-12;


// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

void heading(const std::string& title) {
    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(78, '=') << "\n";
}

void subsection(const std::string& title) {
    std::cout << "\n" << std::string(78, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(78, '-') << "\n";
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Dot product requires equal vector dimensions."
        );
    }

    return std::inner_product(
        a.begin(),
        a.end(),
        b.begin(),
        0.0
    );
}

double norm(const Vector& vector) {
    return std::sqrt(dot(vector, vector));
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector subtraction requires equal dimensions."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

Vector scale(const Vector& vector, double scalar) {
    Vector result(vector.size());

    for (std::size_t i = 0; i < vector.size(); ++i) {
        result[i] = scalar * vector[i];
    }

    return result;
}

void printVector(
    const Vector& vector,
    int precision = 6
) {
    std::cout << "[";

    for (std::size_t i = 0; i < vector.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << std::fixed
            << std::setprecision(precision)
            << vector[i];
    }

    std::cout << "]";
}

void printMatrix(
    const Matrix& matrix,
    int precision = 6
) {
    for (const auto& row : matrix) {
        printVector(row, precision);
        std::cout << "\n";
    }
}


// -----------------------------------------------------------------------------
// 1. Scalar calculus
// -----------------------------------------------------------------------------

double numericalDerivative(
    const std::function<double(double)>& function,
    double x,
    double step = 1e-5
) {
    if (step <= 0.0) {
        throw std::invalid_argument(
            "Derivative step must be positive."
        );
    }

    return (
        function(x + step) -
        function(x - step)
    ) / (2.0 * step);
}

double secondDerivative(
    const std::function<double(double)>& function,
    double x,
    double step = 1e-4
) {
    return (
        function(x + step) -
        2.0 * function(x) +
        function(x - step)
    ) / (step * step);
}

void demonstrateScalarCalculus() {
    heading("1. Scalar Calculus");

    auto quadratic = [](double x) {
        return x * x;
    };

    const double x = 3.0;

    std::cout << "Function: f(x) = x^2\n";
    std::cout << "Exact derivative at x=3: 6\n";
    std::cout
        << "Numerical derivative: "
        << numericalDerivative(quadratic, x)
        << "\n";

    subsection("Second derivative");

    auto fourthPower = [](double value) {
        return value * value * value * value;
    };

    std::cout
        << "Numerical second derivative of x^4 at x=2: "
        << secondDerivative(fourthPower, 2.0)
        << "\n";

    subsection("Finite-difference step-size behavior");

    for (double step : {1e-1, 1e-2, 1e-3, 1e-4}) {
        const double estimate =
            numericalDerivative(
                quadratic,
                x,
                step
            );

        std::cout
            << "h=" << std::scientific << step
            << " estimate=" << std::fixed
            << std::setprecision(10)
            << estimate << "\n";
    }
}


// -----------------------------------------------------------------------------
// 2. Multivariable calculus
// -----------------------------------------------------------------------------

double numericalPartialDerivative(
    const std::function<double(const Vector&)>& function,
    const Vector& point,
    std::size_t variable,
    double step = 1e-5
) {
    if (variable >= point.size()) {
        throw std::out_of_range(
            "Partial derivative variable is outside the point."
        );
    }

    Vector plus = point;
    Vector minus = point;

    plus[variable] += step;
    minus[variable] -= step;

    return (
        function(plus) -
        function(minus)
    ) / (2.0 * step);
}

Vector numericalGradient(
    const std::function<double(const Vector&)>& function,
    const Vector& point,
    double step = 1e-5
) {
    Vector result(point.size());

    for (std::size_t index = 0; index < point.size(); ++index) {
        result[index] =
            numericalPartialDerivative(
                function,
                point,
                index,
                step
            );
    }

    return result;
}

Matrix numericalHessian(
    const std::function<double(const Vector&)>& function,
    const Vector& point,
    double step = 1e-4
) {
    const std::size_t dimension = point.size();

    Matrix result(
        dimension,
        Vector(dimension, 0.0)
    );

    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t j = 0; j < dimension; ++j) {
            Vector pp = point;
            Vector pm = point;
            Vector mp = point;
            Vector mm = point;

            pp[i] += step;
            pp[j] += step;

            pm[i] += step;
            pm[j] -= step;

            mp[i] -= step;
            mp[j] += step;

            mm[i] -= step;
            mm[j] -= step;

            result[i][j] = (
                function(pp) -
                function(pm) -
                function(mp) +
                function(mm)
            ) / (4.0 * step * step);
        }
    }

    return result;
}

void demonstrateMultivariableCalculus() {
    heading("2. Partial Derivatives, Gradients, and Hessians");

    // f(x,y) = x^2 + 3xy + y^2
    // df/dx = 2x + 3y
    // df/dy = 3x + 2y
    auto surface = [](const Vector& point) {
        const double x = point[0];
        const double y = point[1];

        return x * x + 3.0 * x * y + y * y;
    };

    Vector point{2.0, 4.0};

    Vector gradient =
        numericalGradient(surface, point);

    std::cout << "Function: f(x,y) = x^2 + 3xy + y^2\n";
    std::cout << "Point: ";
    printVector(point);
    std::cout << "\nGradient: ";
    printVector(gradient);
    std::cout << "\n";

    auto scalarFunction = [](const Vector& coordinates) {
        const double x = coordinates[0];
        const double y = coordinates[1];

        return x * x + 3.0 * y * y + x * y;
    };

    Matrix hessian =
        numericalHessian(
            scalarFunction,
            point
        );

    std::cout << "Hessian:\n";
    printMatrix(hessian);
}


// -----------------------------------------------------------------------------
// 3. Stable logistic function
// -----------------------------------------------------------------------------

double sigmoid(double value) {
    // A piecewise implementation avoids overflow for large negative inputs.
    if (value >= 0.0) {
        const double exponent = std::exp(-value);
        return 1.0 / (1.0 + exponent);
    }

    const double exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}


// -----------------------------------------------------------------------------
// 4. Dataset representation
// -----------------------------------------------------------------------------

struct Dataset {
    Matrix features;
    std::vector<int> targets;

    void validate() const {
        if (features.empty()) {
            throw std::invalid_argument(
                "Dataset cannot be empty."
            );
        }

        if (features.size() != targets.size()) {
            throw std::invalid_argument(
                "Feature rows and targets must have equal size."
            );
        }

        const std::size_t dimension =
            features.front().size();

        if (dimension == 0) {
            throw std::invalid_argument(
                "Feature vectors cannot be empty."
            );
        }

        for (const auto& row : features) {
            if (row.size() != dimension) {
                throw std::invalid_argument(
                    "All feature rows must have equal dimensions."
                );
            }
        }

        for (int target : targets) {
            if (target != 0 && target != 1) {
                throw std::invalid_argument(
                    "Binary targets must be 0 or 1."
                );
            }
        }
    }
};


// -----------------------------------------------------------------------------
// 5. Logistic regression model
// -----------------------------------------------------------------------------

class LogisticRegression {
private:
    Vector weights_;
    double bias_;

public:
    explicit LogisticRegression(std::size_t featureCount)
        : weights_(featureCount, 0.0),
          bias_(0.0) {
        if (featureCount == 0) {
            throw std::invalid_argument(
                "Feature count must be positive."
            );
        }
    }

    double logit(const Vector& features) const {
        if (features.size() != weights_.size()) {
            throw std::invalid_argument(
                "Feature dimension does not match model."
            );
        }

        return dot(weights_, features) + bias_;
    }

    double probability(const Vector& features) const {
        return sigmoid(logit(features));
    }

    int predict(
        const Vector& features,
        double threshold = 0.5
    ) const {
        if (!(threshold > 0.0 && threshold < 1.0)) {
            throw std::invalid_argument(
                "Classification threshold must be between 0 and 1."
            );
        }

        return probability(features) >= threshold
            ? 1
            : 0;
    }

    const Vector& weights() const {
        return weights_;
    }

    double bias() const {
        return bias_;
    }

    void update(
        const Vector& weightGradient,
        double biasGradient,
        double learningRate
    ) {
        if (weightGradient.size() != weights_.size()) {
            throw std::invalid_argument(
                "Gradient dimension does not match model."
            );
        }

        if (learningRate <= 0.0) {
            throw std::invalid_argument(
                "Learning rate must be positive."
            );
        }

        for (std::size_t i = 0; i < weights_.size(); ++i) {
            weights_[i] -= learningRate * weightGradient[i];
        }

        bias_ -= learningRate * biasGradient;
    }
};


// -----------------------------------------------------------------------------
// 6. Binary cross-entropy and gradient
// -----------------------------------------------------------------------------

struct TrainingResult {
    double loss;
    Vector weightGradient;
    double biasGradient;
};

TrainingResult computeLogisticLossAndGradients(
    const LogisticRegression& model,
    const Dataset& dataset
) {
    dataset.validate();

    const std::size_t dimension =
        dataset.features.front().size();

    Vector gradient(dimension, 0.0);
    double biasGradient = 0.0;
    double totalLoss = 0.0;

    for (std::size_t rowIndex = 0;
         rowIndex < dataset.features.size();
         ++rowIndex) {

        const Vector& row =
            dataset.features[rowIndex];

        const int target =
            dataset.targets[rowIndex];

        const double probability =
            model.probability(row);

        // Clipping prevents log(0).
        const double clipped =
            std::clamp(
                probability,
                EPSILON,
                1.0 - EPSILON
            );

        totalLoss -=
            target * std::log(clipped)
            +
            (1 - target) * std::log(1.0 - clipped);

        // For sigmoid followed by BCE:
        // dL/dz = prediction - target.
        const double error =
            probability - static_cast<double>(target);

        for (std::size_t feature = 0;
             feature < dimension;
             ++feature) {
            gradient[feature] +=
                error * row[feature];
        }

        biasGradient += error;
    }

    const double count =
        static_cast<double>(dataset.features.size());

    for (double& value : gradient) {
        value /= count;
    }

    biasGradient /= count;

    return {
        totalLoss / count,
        gradient,
        biasGradient
    };
}


// -----------------------------------------------------------------------------
// 7. Dataset evaluation
// -----------------------------------------------------------------------------

struct Evaluation {
    std::size_t correct;
    std::size_t total;
    double accuracy;
};

Evaluation evaluate(
    const LogisticRegression& model,
    const Dataset& dataset
) {
    dataset.validate();

    std::size_t correct = 0;

    for (std::size_t i = 0;
         i < dataset.features.size();
         ++i) {

        const int prediction =
            model.predict(dataset.features[i]);

        if (prediction == dataset.targets[i]) {
            ++correct;
        }
    }

    const std::size_t total =
        dataset.features.size();

    return {
        correct,
        total,
        static_cast<double>(correct)
        / static_cast<double>(total)
    };
}


// -----------------------------------------------------------------------------
// 8. Training engine
// -----------------------------------------------------------------------------

struct TrainingConfig {
    double learningRate = 0.8;
    std::size_t epochs = 1000;
    std::size_t reportEvery = 100;
};

void train(
    LogisticRegression& model,
    const Dataset& dataset,
    const TrainingConfig& configuration
) {
    dataset.validate();

    if (configuration.learningRate <= 0.0) {
        throw std::invalid_argument(
            "Learning rate must be positive."
        );
    }

    if (configuration.epochs == 0) {
        throw std::invalid_argument(
            "Training must contain at least one epoch."
        );
    }

    for (std::size_t epoch = 1;
         epoch <= configuration.epochs;
         ++epoch) {

        TrainingResult result =
            computeLogisticLossAndGradients(
                model,
                dataset
            );

        model.update(
            result.weightGradient,
            result.biasGradient,
            configuration.learningRate
        );

        if (
            epoch == 1 ||
            epoch == configuration.epochs ||
            (
                configuration.reportEvery > 0 &&
                epoch % configuration.reportEvery == 0
            )
        ) {
            std::cout
                << "epoch="
                << std::setw(4)
                << epoch
                << " loss="
                << std::fixed
                << std::setprecision(8)
                << result.loss
                << " weights=";

            printVector(model.weights(), 4);

            std::cout
                << " bias="
                << std::setprecision(4)
                << model.bias()
                << "\n";
        }
    }
}


// -----------------------------------------------------------------------------
// 9. Computational graph perspective
// -----------------------------------------------------------------------------

double demonstrateComputationalGraph(
    double x,
    double weight,
    double bias
) {
    /*
     * Forward computation:
     *
     * z = weight*x + bias
     * p = sigmoid(z)
     *
     * The dependency chain is:
     *
     * x -> multiplication -> addition -> sigmoid -> p
     *
     * Differentiation reverses this dependency when computing how a loss
     * depends on the parameters.
     */
    const double z = weight * x + bias;
    const double probability = sigmoid(z);

    return probability;
}

void demonstrateChainRule() {
    heading("3. Computational Graph and Chain Rule");

    const double x = 2.0;
    const double weight = 1.5;
    const double bias = -0.5;

    const double probability =
        demonstrateComputationalGraph(
            x,
            weight,
            bias
        );

    std::cout
        << "Input x=" << x
        << ", weight=" << weight
        << ", bias=" << bias
        << "\n";

    std::cout
        << "Output probability="
        << probability
        << "\n";

    std::cout
        << "For sigmoid(z), sigmoid'(z)=sigmoid(z)(1-sigmoid(z)).\n";

    const double sigmoidDerivative =
        probability * (1.0 - probability);

    std::cout
        << "Local sigmoid derivative="
        << sigmoidDerivative
        << "\n";

    std::cout
        << "The chain rule multiplies local derivatives along the "
        << "dependency path.\n";
}


// -----------------------------------------------------------------------------
// 10. Realistic case-study data
// -----------------------------------------------------------------------------

Dataset createCaseStudyDataset() {
    /*
     * Synthetic application:
     *
     * Two normalized measurements are used to classify observations into
     * two categories. The dataset is intentionally small so that the
     * mathematical behavior remains easy to inspect.
     */
    Dataset dataset{
        {
            {0.0, 0.0},
            {0.2, 0.1},
            {0.1, 0.3},
            {0.3, 0.2},
            {0.2, 0.4},
            {0.4, 0.2},
            {0.8, 1.0},
            {1.0, 1.1},
            {1.2, 0.9},
            {0.9, 1.2},
            {1.1, 1.3},
            {1.3, 1.0}
        },
        {
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1
        }
    };

    dataset.validate();
    return dataset;
}


// -----------------------------------------------------------------------------
// 11. Gradient verification
// -----------------------------------------------------------------------------

void verifyAnalyticalGradient() {
    heading("4. Analytical Gradient Verification");

    Dataset dataset = createCaseStudyDataset();
    LogisticRegression model(2);

    TrainingResult analytical =
        computeLogisticLossAndGradients(
            model,
            dataset
        );

    /*
     * Finite differences independently estimate the derivative:
     *
     * dL/dw_i ≈ [L(w_i+h)-L(w_i-h)]/(2h)
     *
     * Comparing the two is useful when implementing a new optimizer.
     */
    const double step = 1e-5;

    Vector numericalGradient(
        model.weights().size(),
        0.0
    );

    for (std::size_t i = 0;
         i < model.weights().size();
         ++i) {

        LogisticRegression plus(2);
        LogisticRegression minus(2);

        // Reconstruct the same parameter state for this verification.
        Vector plusWeights = model.weights();
        Vector minusWeights = model.weights();

        plusWeights[i] += step;
        minusWeights[i] -= step;

        /*
         * The production model intentionally keeps parameter mutation
         * private. This local verification uses a tiny helper object
         * initialized through controlled updates from zero.
         */
        for (std::size_t j = 0; j < plusWeights.size(); ++j) {
            Vector unitGradient(2, 0.0);
            unitGradient[j] = -plusWeights[j];
            plus.update(
                unitGradient,
                0.0,
                1.0
            );

            unitGradient[j] = -minusWeights[j];
            minus.update(
                unitGradient,
                0.0,
                1.0
            );
        }

        // The construction above is awkward for a private model API.
        // Verify the same derivatives directly using a parameterized loss.
        auto lossForWeights =
            [&](const Vector& weights) {
                double total = 0.0;

                for (std::size_t rowIndex = 0;
                     rowIndex < dataset.features.size();
                     ++rowIndex) {

                    const Vector& row =
                        dataset.features[rowIndex];

                    const int target =
                        dataset.targets[rowIndex];

                    const double logit =
                        dot(weights, row);

                    const double probability =
                        sigmoid(logit);

                    const double clipped =
                        std::clamp(
                            probability,
                            EPSILON,
                            1.0 - EPSILON
                        );

                    total -=
                        target * std::log(clipped)
                        +
                        (1 - target)
                        * std::log(1.0 - clipped);
                }

                return total /
                    static_cast<double>(
                        dataset.features.size()
                    );
            };

        numericalGradient[i] =
            (
                lossForWeights(plusWeights) -
                lossForWeights(minusWeights)
            ) / (2.0 * step);
    }

    std::cout << "Analytical gradient: ";
    printVector(analytical.weightGradient, 10);
    std::cout << "\nNumerical gradient:  ";
    printVector(numericalGradient, 10);
    std::cout << "\n";

    for (std::size_t i = 0;
         i < numericalGradient.size();
         ++i) {
        const double error =
            std::abs(
                analytical.weightGradient[i]
                - numericalGradient[i]
            );

        std::cout
            << "parameter "
            << i
            << " absolute error="
            << std::scientific
            << error
            << "\n";

        assert(error < 1e-5);
    }

    std::cout << "Gradient verification passed.\n";
}


// -----------------------------------------------------------------------------
// 12. Train complete case study
// -----------------------------------------------------------------------------

void runCaseStudy() {
    heading("5. Industry-Style Logistic Regression Case Study");

    Dataset dataset = createCaseStudyDataset();

    std::cout
        << "Samples: "
        << dataset.features.size()
        << "\n";

    std::cout
        << "Features per sample: "
        << dataset.features.front().size()
        << "\n";

    std::cout
        << "Objective: minimize binary cross-entropy using gradient descent.\n";

    LogisticRegression model(
        dataset.features.front().size()
    );

    TrainingConfig configuration;
    configuration.learningRate = 0.8;
    configuration.epochs = 1000;
    configuration.reportEvery = 200;

    train(
        model,
        dataset,
        configuration
    );

    Evaluation evaluation =
        evaluate(
            model,
            dataset
        );

    std::cout
        << "\nTraining accuracy: "
        << std::fixed
        << std::setprecision(2)
        << evaluation.accuracy * 100.0
        << "%\n";

    std::cout
        << "Correct predictions: "
        << evaluation.correct
        << "/"
        << evaluation.total
        << "\n";

    subsection("Prediction inspection");

    for (std::size_t i = 0;
         i < dataset.features.size();
         ++i) {

        const double probability =
            model.probability(
                dataset.features[i]
            );

        const int prediction =
            model.predict(
                dataset.features[i]
            );

        std::cout
            << "sample="
            << std::setw(2)
            << i
            << " features=";

        printVector(dataset.features[i], 2);

        std::cout
            << " target="
            << dataset.targets[i]
            << " probability="
            << std::setprecision(5)
            << probability
            << " prediction="
            << prediction
            << "\n";
    }
}


// -----------------------------------------------------------------------------
// 13. Failure conditions and validation
// -----------------------------------------------------------------------------

void demonstrateFailureConditions() {
    heading("6. Failure Conditions and Validation");

    subsection("Invalid feature dimensions");

    try {
        LogisticRegression model(2);
        model.predict({1.0});
    } catch (const std::exception& error) {
        std::cout
            << "Caught expected error: "
            << error.what()
            << "\n";
    }

    subsection("Invalid threshold");

    try {
        LogisticRegression model(2);
        model.predict({1.0, 2.0}, 1.5);
    } catch (const std::exception& error) {
        std::cout
            << "Caught expected error: "
            << error.what()
            << "\n";
    }

    subsection("Invalid dataset labels");

    try {
        Dataset invalid{
            {{1.0, 2.0}},
            {2}
        };

        invalid.validate();
    } catch (const std::exception& error) {
        std::cout
            << "Caught expected error: "
            << error.what()
            << "\n";
    }

    subsection("Invalid learning rate");

    try {
        LogisticRegression model(2);
        model.update({0.0, 0.0}, 0.0, -0.1);
    } catch (const std::exception& error) {
        std::cout
            << "Caught expected error: "
            << error.what()
            << "\n";
    }
}


// -----------------------------------------------------------------------------
// 14. Complexity analysis
// -----------------------------------------------------------------------------

void demonstrateComplexity() {
    heading("7. Computational Complexity");

    std::cout
        << "For n samples and d features:\n"
        << "  Gradient computation: O(n*d)\n"
        << "  One training epoch:   O(n*d)\n"
        << "  Memory for dataset:   O(n*d)\n"
        << "  Model parameters:     O(d)\n"
        << "\n"
        << "A numerical gradient over d parameters requires roughly d "
        << "additional loss evaluations.\n"
        << "Analytical gradients are therefore substantially more efficient "
        << "for large models.\n";

    std::cout
        << "\nA full Hessian contains d*d entries and can become expensive "
        << "in both computation and memory. This is one reason many large "
        << "machine-learning systems rely primarily on first-order methods.\n";
}


// -----------------------------------------------------------------------------
// 15. Numerical-stability demonstration
// -----------------------------------------------------------------------------

void demonstrateNumericalStability() {
    heading("8. Numerical Stability");

    std::cout
        << "Stable sigmoid values:\n";

    for (double value :
         {-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0}) {

        std::cout
            << "sigmoid("
            << value
            << ") = "
            << sigmoid(value)
            << "\n";
    }

    std::cout
        << "\nClipping probabilities before logarithms prevents "
        << "undefined log(0) operations in cross-entropy calculations.\n";
}


// -----------------------------------------------------------------------------
// 16. Tests
// -----------------------------------------------------------------------------

void runTests() {
    heading("9. Tests");

    auto square = [](double x) {
        return x * x;
    };

    const double derivative =
        numericalDerivative(
            square,
            3.0,
            1e-6
        );

    assert(std::abs(derivative - 6.0) < 1e-5);

    auto function = [](const Vector& point) {
        return point[0] * point[0]
            + 3.0 * point[1] * point[1];
    };

    Vector gradient =
        numericalGradient(
            function,
            {2.0, 4.0}
        );

    assert(std::abs(gradient[0] - 4.0) < 1e-5);
    assert(std::abs(gradient[1] - 24.0) < 1e-5);

    assert(std::abs(sigmoid(0.0) - 0.5) < 1e-12);

    Dataset dataset =
        createCaseStudyDataset();

    LogisticRegression model(2);

    TrainingResult result =
        computeLogisticLossAndGradients(
            model,
            dataset
        );

    assert(std::isfinite(result.loss));

    const Evaluation before =
        evaluate(
            model,
            dataset
        );

    assert(before.total == dataset.features.size());

    std::cout << "All C++ tests passed.\n";
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        demonstrateScalarCalculus();
        demonstrateMultivariableCalculus();
        demonstrateChainRule();
        verifyAnalyticalGradient();
        runCaseStudy();
        demonstrateFailureConditions();
        demonstrateComplexity();
        demonstrateNumericalStability();
        runTests();

        std::cout
            << "\nProgram completed successfully.\n";
    }
    catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << "\n";

        return 1;
    }

    return 0;
}
