#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * C++17 Bias-Variance Tradeoff Evaluation Engine
 *
 * Scenario:
 * A predictive-service team evaluates polynomial models for a nonlinear
 * demand signal. Each candidate is trained on independently generated
 * datasets. Repeated predictions on a fixed evaluation grid estimate
 * squared bias and prediction variance.
 *
 * Build:
 * g++ -std=c++17 -O2 -Wall -Wextra -pedantic bias_variance.cpp -o bias_variance
 *
 * Run:
 * ./bias_variance
 * ./bias_variance 100 50
 */

namespace biasvariance {

constexpr double NOISE_STDDEV = 0.20;
constexpr double RIDGE_LAMBDA = 1e-6;
constexpr double EPSILON = 1e-14;

struct Point {
    double x;
    double y;
};

struct ExperimentConfig {
    std::size_t repetitions = 80;
    std::size_t sampleCount = 36;
    std::size_t gridSize = 81;
    unsigned int seed = 20261103;
};

enum class RiskLevel {
    Underfitting,
    Overfitting,
    GeneralizesWell
};

struct Metrics {
    int degree;
    double trainingMse;
    double biasSquared;
    double predictionVariance;
    double irreducibleNoise;
    double signalPredictionMse;
    double expectedTestMse;
    double generalizationGap;
    RiskLevel risk = RiskLevel::GeneralizesWell;
};

double trueSignal(double x) {
    if (!std::isfinite(x) || x < -1.0 || x > 1.0) {
        throw std::invalid_argument("Signal input must be finite and in [-1, 1].");
    }

    return std::sin(2.0 * std::acos(-1.0) * x) + 0.25 * x;
}

void validateConfiguration(const ExperimentConfig& config) {
    if (config.repetitions < 2) {
        throw std::invalid_argument("At least two repetitions are required.");
    }

    if (config.sampleCount < 10) {
        throw std::invalid_argument("At least ten training samples are required.");
    }

    if (config.gridSize < 3) {
        throw std::invalid_argument("Evaluation grid must contain at least three points.");
    }
}

void validatePoints(const std::vector<Point>& points) {
    if (points.size() < 2) {
        throw std::invalid_argument("Training data must contain at least two points.");
    }

    for (const auto& point : points) {
        if (
            !std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            point.x < -1.0 ||
            point.x > 1.0
        ) {
            throw std::invalid_argument("Invalid training observation.");
        }
    }
}

std::vector<double> polynomialFeatures(double x, int degree) {
    if (degree < 0 || degree > 12) {
        throw std::invalid_argument("Polynomial degree must be between 0 and 12.");
    }

    std::vector<double> features(static_cast<std::size_t>(degree + 1), 1.0);

    for (int index = 1; index <= degree; ++index) {
        features[static_cast<std::size_t>(index)] =
            features[static_cast<std::size_t>(index - 1)] * x;
    }

    return features;
}

/*
 * Gaussian elimination with partial pivoting.
 * A small ridge penalty is added before this solver is called.
 */
std::vector<double> solveLinearSystem(
    std::vector<std::vector<double>> matrix,
    std::vector<double> rhs
) {
    const std::size_t size = rhs.size();

    if (size == 0 || matrix.size() != size) {
        throw std::invalid_argument("Invalid linear system dimensions.");
    }

    for (const auto& row : matrix) {
        if (row.size() != size) {
            throw std::invalid_argument("Coefficient matrix must be square.");
        }
    }

    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;

        for (std::size_t row = column + 1; row < size; ++row) {
            if (
                std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])
            ) {
                pivot = row;
            }
        }

        if (std::abs(matrix[pivot][column]) < EPSILON) {
            throw std::runtime_error("Singular regression system.");
        }

        std::swap(matrix[column], matrix[pivot]);
        std::swap(rhs[column], rhs[pivot]);

        const double divisor = matrix[column][column];

        for (std::size_t row = column + 1; row < size; ++row) {
            const double factor = matrix[row][column] / divisor;

            for (std::size_t index = column; index < size; ++index) {
                matrix[row][index] -= factor * matrix[column][index];
            }

            rhs[row] -= factor * rhs[column];
        }
    }

    std::vector<double> solution(size, 0.0);

    for (std::size_t reverse = size; reverse-- > 0;) {
        double value = rhs[reverse];

        for (std::size_t column = reverse + 1; column < size; ++column) {
            value -= matrix[reverse][column] * solution[column];
        }

        solution[reverse] = value / matrix[reverse][reverse];

        if (!std::isfinite(solution[reverse])) {
            throw std::runtime_error("Regression generated a non-finite coefficient.");
        }
    }

    return solution;
}

class PolynomialRegressor {
public:
    explicit PolynomialRegressor(
        int degree,
        double ridgeLambda = RIDGE_LAMBDA
    ) : degree_(degree), ridgeLambda_(ridgeLambda) {
        if (degree < 0 || degree > 12) {
            throw std::invalid_argument("Degree must be between 0 and 12.");
        }

        if (!std::isfinite(ridgeLambda) || ridgeLambda < 0.0) {
            throw std::invalid_argument("Ridge regularization must be non-negative.");
        }
    }

    void fit(const std::vector<Point>& points) {
        validatePoints(points);

        const std::size_t parameterCount =
            static_cast<std::size_t>(degree_ + 1);

        std::vector<std::vector<double>> matrix(
            parameterCount,
            std::vector<double>(parameterCount, 0.0)
        );
        std::vector<double> rhs(parameterCount, 0.0);

        for (const auto& point : points) {
            const auto features = polynomialFeatures(point.x, degree_);

            for (std::size_t row = 0; row < parameterCount; ++row) {
                rhs[row] += features[row] * point.y;

                for (std::size_t column = 0; column < parameterCount; ++column) {
                    matrix[row][column] += features[row] * features[column];
                }
            }
        }

        // Do not penalize the intercept.
        for (std::size_t index = 1; index < parameterCount; ++index) {
            matrix[index][index] += ridgeLambda_;
        }

        coefficients_ = solveLinearSystem(std::move(matrix), std::move(rhs));
        fitted_ = true;
    }

    double predict(double x) const {
        if (!fitted_) {
            throw std::logic_error("Model must be fitted before prediction.");
        }

        if (!std::isfinite(x) || x < -1.0 || x > 1.0) {
            throw std::invalid_argument("Prediction input must be in [-1, 1].");
        }

        const auto features = polynomialFeatures(x, degree_);
        double prediction = 0.0;

        for (std::size_t index = 0; index < coefficients_.size(); ++index) {
            prediction += coefficients_[index] * features[index];
        }

        if (!std::isfinite(prediction)) {
            throw std::runtime_error("Prediction is not finite.");
        }

        return prediction;
    }

private:
    int degree_;
    double ridgeLambda_;
    bool fitted_ = false;
    std::vector<double> coefficients_;
};

class DatasetGenerator {
public:
    explicit DatasetGenerator(unsigned int seed) : engine_(seed) {}

    std::vector<Point> generate(
        std::size_t sampleCount,
        double noiseStddev
    ) {
        if (sampleCount < 2) {
            throw std::invalid_argument("Sample count must be at least two.");
        }

        if (!std::isfinite(noiseStddev) || noiseStddev < 0.0) {
            throw std::invalid_argument("Invalid noise standard deviation.");
        }

        std::uniform_real_distribution<double> xDistribution(-1.0, 1.0);
        std::normal_distribution<double> noiseDistribution(0.0, noiseStddev);

        std::vector<Point> points;
        points.reserve(sampleCount);

        for (std::size_t index = 0; index < sampleCount; ++index) {
            const double x = xDistribution(engine_);
            const double y = trueSignal(x) + noiseDistribution(engine_);
            points.push_back({x, y});
        }

        return points;
    }

private:
    std::mt19937_64 engine_;
};

double meanSquaredError(
    const std::vector<double>& actual,
    const std::vector<double>& predicted
) {
    if (actual.empty() || actual.size() != predicted.size()) {
        throw std::invalid_argument("MSE inputs must have equal non-zero sizes.");
    }

    double sum = 0.0;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = actual[index] - predicted[index];
        sum += difference * difference;
    }

    return sum / static_cast<double>(actual.size());
}

double average(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot average an empty collection.");
    }

    double sum = 0.0;

    for (double value : values) {
        sum += value;
    }

    return sum / static_cast<double>(values.size());
}

std::string riskName(RiskLevel risk) {
    switch (risk) {
        case RiskLevel::Underfitting:
            return "UNDERFITTING";
        case RiskLevel::Overfitting:
            return "OVERFITTING";
        case RiskLevel::GeneralizesWell:
            return "GENERALIZES";
    }

    throw std::logic_error("Unknown model risk.");
}

class ExperimentRunner {
public:
    explicit ExperimentRunner(ExperimentConfig config)
        : config_(config), generator_(config.seed) {
        validateConfiguration(config_);
    }

    std::vector<Metrics> run(const std::vector<int>& degrees) {
        if (degrees.empty()) {
            throw std::invalid_argument("At least one model degree is required.");
        }

        std::vector<int> sortedDegrees = degrees;
        std::sort(sortedDegrees.begin(), sortedDegrees.end());

        if (
            std::adjacent_find(sortedDegrees.begin(), sortedDegrees.end()) !=
            sortedDegrees.end()
        ) {
            throw std::invalid_argument("Duplicate degrees are not allowed.");
        }

        for (int degree : sortedDegrees) {
            if (degree < 0 || degree > 12) {
                throw std::invalid_argument("Invalid model degree.");
            }
        }

        std::vector<double> grid(config_.gridSize);
        std::vector<double> signalValues(config_.gridSize);

        for (std::size_t index = 0; index < config_.gridSize; ++index) {
            grid[index] =
                -1.0 +
                2.0 * static_cast<double>(index) /
                static_cast<double>(config_.gridSize - 1);

            signalValues[index] = trueSignal(grid[index]);
        }

        // predictions[candidate][trial][grid_position]
        std::vector<std::vector<std::vector<double>>> predictions(
            sortedDegrees.size(),
            std::vector<std::vector<double>>(config_.repetitions)
        );

        std::vector<std::vector<double>> trainingErrors(
            sortedDegrees.size()
        );

        for (std::size_t trial = 0; trial < config_.repetitions; ++trial) {
            const auto trainingData = generator_.generate(
                config_.sampleCount,
                NOISE_STDDEV
            );

            for (
                std::size_t candidate = 0;
                candidate < sortedDegrees.size();
                ++candidate
            ) {
                PolynomialRegressor model(sortedDegrees[candidate]);
                model.fit(trainingData);

                std::vector<double> modelGridPredictions;
                modelGridPredictions.reserve(grid.size());

                for (double x : grid) {
                    modelGridPredictions.push_back(model.predict(x));
                }

                predictions[candidate][trial] =
                    std::move(modelGridPredictions);

                std::vector<double> actualTraining;
                std::vector<double> fittedTraining;
                actualTraining.reserve(trainingData.size());
                fittedTraining.reserve(trainingData.size());

                for (const auto& point : trainingData) {
                    actualTraining.push_back(point.y);
                    fittedTraining.push_back(model.predict(point.x));
                }

                trainingErrors[candidate].push_back(
                    meanSquaredError(actualTraining, fittedTraining)
                );
            }
        }

        std::vector<Metrics> results;
        results.reserve(sortedDegrees.size());

        for (
            std::size_t candidate = 0;
            candidate < sortedDegrees.size();
            ++candidate
        ) {
            std::vector<double> meanPredictions(config_.gridSize, 0.0);

            for (std::size_t position = 0; position < config_.gridSize; ++position) {
                for (std::size_t trial = 0; trial < config_.repetitions; ++trial) {
                    meanPredictions[position] +=
                        predictions[candidate][trial][position];
                }

                meanPredictions[position] /=
                    static_cast<double>(config_.repetitions);
            }

            double biasSquared = 0.0;
            double predictionVariance = 0.0;

            for (std::size_t position = 0; position < config_.gridSize; ++position) {
                const double bias =
                    meanPredictions[position] - signalValues[position];

                biasSquared += bias * bias;

                double varianceAtPosition = 0.0;

                for (std::size_t trial = 0; trial < config_.repetitions; ++trial) {
                    const double difference =
                        predictions[candidate][trial][position] -
                        meanPredictions[position];

                    varianceAtPosition += difference * difference;
                }

                predictionVariance +=
                    varianceAtPosition / static_cast<double>(config_.repetitions);
            }

            biasSquared /= static_cast<double>(config_.gridSize);
            predictionVariance /= static_cast<double>(config_.gridSize);

            const double trainingMse = average(trainingErrors[candidate]);
            const double irreducibleNoise = NOISE_STDDEV * NOISE_STDDEV;
            const double signalPredictionMse =
                biasSquared + predictionVariance;
            const double expectedTestMse =
                signalPredictionMse + irreducibleNoise;

            results.push_back({
                sortedDegrees[candidate],
                trainingMse,
                biasSquared,
                predictionVariance,
                irreducibleNoise,
                signalPredictionMse,
                expectedTestMse,
                expectedTestMse - trainingMse,
                RiskLevel::GeneralizesWell
            });
        }

        classify(results);

        return results;
    }

private:
    static void classify(std::vector<Metrics>& results) {
        const double bestTest = std::min_element(
            results.begin(),
            results.end(),
            [](const Metrics& left, const Metrics& right) {
                return left.expectedTestMse < right.expectedTestMse;
            }
        )->expectedTestMse;

        const double bestTraining = std::min_element(
            results.begin(),
            results.end(),
            [](const Metrics& left, const Metrics& right) {
                return left.trainingMse < right.trainingMse;
            }
        )->trainingMse;

        const double testThreshold = std::max(bestTest, 1e-12) * 1.10;
        const double trainingThreshold =
            std::max(bestTraining, 1e-12) * 1.20;

        for (auto& result : results) {
            if (result.expectedTestMse > testThreshold) {
                result.risk = result.trainingMse > trainingThreshold
                    ? RiskLevel::Underfitting
                    : RiskLevel::Overfitting;
            } else {
                result.risk = RiskLevel::GeneralizesWell;
            }
        }
    }

    ExperimentConfig config_;
    DatasetGenerator generator_;
};

std::size_t parsePositiveInteger(
    const std::string& text,
    const std::string& argumentName
) {
    std::size_t parsed = 0;
    std::size_t consumed = 0;

    try {
        parsed = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            argumentName + " must be a positive integer."
        );
    }

    if (consumed != text.size() || parsed == 0) {
        throw std::invalid_argument(
            argumentName + " must be a positive integer."
        );
    }

    return parsed;
}

void validateFailureConditions() {
    try {
        PolynomialRegressor invalid(-1);
        throw std::logic_error("Invalid degree was accepted.");
    } catch (const std::invalid_argument&) {
        // Expected validation failure.
    }

    try {
        PolynomialRegressor model(2);
        model.predict(0.5);
        throw std::logic_error("Unfitted model accepted prediction.");
    } catch (const std::logic_error& error) {
        if (std::string(error.what()) !=
            "Model must be fitted before prediction.") {
            throw;
        }
    }

    PolynomialRegressor constantModel(0);
    constantModel.fit({{-1.0, 4.0}, {1.0, 4.0}});

    if (std::abs(constantModel.predict(0.0) - 4.0) > 1e-9) {
        throw std::logic_error("Constant-model test failed.");
    }
}

}  // namespace biasvariance

int main(int argc, char* argv[]) {
    using namespace biasvariance;

    try {
        if (argc > 3) {
            throw std::invalid_argument(
                "Usage: bias_variance [repetitions] [sample_count]"
            );
        }

        ExperimentConfig config;

        if (argc >= 2) {
            config.repetitions = parsePositiveInteger(
                argv[1],
                "Repetitions"
            );
        }

        if (argc >= 3) {
            config.sampleCount = parsePositiveInteger(
                argv[2],
                "Sample count"
            );
        }

        if (config.repetitions > 1000 || config.sampleCount > 10000) {
            throw std::invalid_argument(
                "Requested experiment exceeds configured resource limits."
            );
        }

        validateFailureConditions();

        ExperimentRunner runner(config);
        const auto results = runner.run({1, 2, 5, 9});

        std::cout << "BIAS-VARIANCE TRADEOFF EXPERIMENT\n";
        std::cout << "Independent training samples: "
                  << config.repetitions << '\n';
        std::cout << "Training observations per sample: "
                  << config.sampleCount << "\n\n";

        std::cout << std::fixed << std::setprecision(5);
        std::cout
            << std::setw(7) << "Degree"
            << std::setw(13) << "Train MSE"
            << std::setw(13) << "Bias^2"
            << std::setw(13) << "Variance"
            << std::setw(11) << "Noise"
            << std::setw(15) << "Expected Test"
            << std::setw(13) << "Gap"
            << std::setw(18) << "Risk"
            << '\n';

        for (const auto& result : results) {
            std::cout
                << std::setw(7) << result.degree
                << std::setw(13) << result.trainingMse
                << std::setw(13) << result.biasSquared
                << std::setw(13) << result.predictionVariance
                << std::setw(11) << result.irreducibleNoise
                << std::setw(15) << result.expectedTestMse
                << std::setw(13) << result.generalizationGap
                << std::setw(18) << riskName(result.risk)
                << '\n';
        }

        const auto best = std::min_element(
            results.begin(),
            results.end(),
            [](const Metrics& left, const Metrics& right) {
                return left.expectedTestMse < right.expectedTestMse;
            }
        );

        std::cout << "\nSelected polynomial degree: " << best->degree << '\n';
        std::cout << "Expected test MSE: "
                  << best->expectedTestMse << '\n';

        /*
         * Approximate complexity:
         * R = number of repetitions
         * K = number of candidate degrees
         * N = training observations
         * D = maximum degree
         * G = evaluation-grid points
         *
         * Fitting with normal equations and elimination is approximately
         * O(R * K * (N*D^2 + D^3)).
         * Prediction and decomposition add O(R * K * G * D).
         * Stored predictions require O(R * K * G) memory.
         *
         * For higher degrees or poorly scaled features, QR or SVD
         * decomposition is preferable to normal equations.
         */

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Experiment failed: " << error.what() << '\n';
        return 1;
    }
}
