#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct Metrics {
    double mae;
    double rmse;
    double r2;
};

class MatrixOperations {
public:
    static Matrix transpose(const Matrix& matrix) {
        if (matrix.empty()) {
            throw std::invalid_argument("Cannot transpose an empty matrix.");
        }

        const std::size_t columns = matrix.front().size();

        for (const auto& row : matrix) {
            if (row.size() != columns) {
                throw std::invalid_argument("Matrix rows have inconsistent sizes.");
            }
        }

        Matrix result(columns, Vector(matrix.size()));

        for (std::size_t row = 0; row < matrix.size(); ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                result[column][row] = matrix[row][column];
            }
        }

        return result;
    }

    static Matrix multiply(const Matrix& left, const Matrix& right) {
        if (left.empty() || right.empty()) {
            throw std::invalid_argument("Matrices cannot be empty.");
        }

        if (left.front().size() != right.size()) {
            throw std::invalid_argument("Incompatible matrix dimensions.");
        }

        Matrix result(
            left.size(),
            Vector(right.front().size(), 0.0)
        );

        for (std::size_t i = 0; i < left.size(); ++i) {
            for (std::size_t k = 0; k < right.size(); ++k) {
                for (std::size_t j = 0; j < right[k].size(); ++j) {
                    result[i][j] += left[i][k] * right[k][j];
                }
            }
        }

        return result;
    }

    static Vector multiply(const Matrix& matrix, const Vector& vector) {
        if (matrix.empty() || matrix.front().size() != vector.size()) {
            throw std::invalid_argument("Matrix/vector dimensions do not match.");
        }

        Vector result(matrix.size(), 0.0);

        for (std::size_t row = 0; row < matrix.size(); ++row) {
            for (std::size_t column = 0; column < vector.size(); ++column) {
                result[row] += matrix[row][column] * vector[column];
            }
        }

        return result;
    }

    static Matrix identity(std::size_t size) {
        Matrix result(size, Vector(size, 0.0));

        for (std::size_t index = 0; index < size; ++index) {
            result[index][index] = 1.0;
        }

        return result;
    }

    static Matrix inverse(Matrix matrix) {
        const std::size_t n = matrix.size();

        if (n == 0) {
            throw std::invalid_argument("Cannot invert an empty matrix.");
        }

        for (const auto& row : matrix) {
            if (row.size() != n) {
                throw std::invalid_argument("Matrix must be square.");
            }
        }

        Matrix augmented(n, Vector(2 * n, 0.0));

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = 0; column < n; ++column) {
                augmented[row][column] = matrix[row][column];
            }

            for (std::size_t column = 0; column < n; ++column) {
                augmented[row][n + column] =
                    row == column ? 1.0 : 0.0;
            }
        }

        for (std::size_t pivot = 0; pivot < n; ++pivot) {
            std::size_t pivotRow = pivot;

            for (std::size_t row = pivot + 1; row < n; ++row) {
                if (
                    std::abs(augmented[row][pivot]) >
                    std::abs(augmented[pivotRow][pivot])
                ) {
                    pivotRow = row;
                }
            }

            if (std::abs(augmented[pivotRow][pivot]) < 1e-12) {
                throw std::runtime_error(
                    "Matrix is singular or numerically unstable."
                );
            }

            std::swap(augmented[pivot], augmented[pivotRow]);

            const double pivotValue = augmented[pivot][pivot];

            for (double& value : augmented[pivot]) {
                value /= pivotValue;
            }

            for (std::size_t row = 0; row < n; ++row) {
                if (row == pivot) {
                    continue;
                }

                const double factor = augmented[row][pivot];

                for (std::size_t column = 0; column < 2 * n; ++column) {
                    augmented[row][column] -=
                        factor * augmented[pivot][column];
                }
            }
        }

        Matrix result(n, Vector(n));

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = 0; column < n; ++column) {
                result[row][column] =
                    augmented[row][n + column];
            }
        }

        return result;
    }
};

class StandardScaler {
private:
    double mean_ = 0.0;
    double scale_ = 1.0;
    bool fitted_ = false;

public:
    void fit(const Vector& values) {
        if (values.empty()) {
            throw std::invalid_argument("Cannot fit scaler to empty data.");
        }

        mean_ =
            std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size());

        double variance = 0.0;

        for (double value : values) {
            variance += std::pow(value - mean_, 2);
        }

        variance /= static_cast<double>(values.size());
        scale_ = std::sqrt(variance);

        if (scale_ < 1e-12) {
            throw std::invalid_argument(
                "Cannot standardize a constant feature."
            );
        }

        fitted_ = true;
    }

    Vector transform(const Vector& values) const {
        if (!fitted_) {
            throw std::logic_error("Scaler has not been fitted.");
        }

        Vector result;
        result.reserve(values.size());

        for (double value : values) {
            result.push_back((value - mean_) / scale_);
        }

        return result;
    }

    Vector fitTransform(const Vector& values) {
        fit(values);
        return transform(values);
    }
};

class PolynomialFeatureBuilder {
private:
    int degree_;

public:
    explicit PolynomialFeatureBuilder(int degree)
        : degree_(degree) {
        if (degree < 0) {
            throw std::invalid_argument(
                "Polynomial degree cannot be negative."
            );
        }
    }

    Matrix transform(const Vector& values) const {
        if (values.empty()) {
            throw std::invalid_argument(
                "Cannot create features from empty data."
            );
        }

        Matrix features;
        features.reserve(values.size());

        for (double value : values) {
            Vector row;

            for (int power = 0; power <= degree_; ++power) {
                row.push_back(std::pow(value, power));
            }

            features.push_back(std::move(row));
        }

        return features;
    }
};

class PolynomialRegression {
private:
    int degree_;
    double lambda_;
    bool scaling_;
    PolynomialFeatureBuilder featureBuilder_;
    StandardScaler scaler_;
    Vector coefficients_;
    bool fitted_ = false;

public:
    PolynomialRegression(
        int degree,
        double lambda,
        bool scaling
    )
        : degree_(degree),
          lambda_(lambda),
          scaling_(scaling),
          featureBuilder_(degree) {
        if (lambda < 0.0) {
            throw std::invalid_argument(
                "Regularization cannot be negative."
            );
        }
    }

    void fit(const Vector& x, const Vector& y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y sizes differ.");
        }

        if (x.size() < static_cast<std::size_t>(degree_ + 1)) {
            throw std::invalid_argument(
                "Insufficient observations for the selected degree."
            );
        }

        Vector transformedX = x;

        if (scaling_) {
            transformedX = scaler_.fitTransform(transformedX);
        }

        Matrix design = featureBuilder_.transform(transformedX);
        Matrix designT = MatrixOperations::transpose(design);
        Matrix normal =
            MatrixOperations::multiply(designT, design);

        // Ridge regularization is applied to polynomial coefficients but
        // not the intercept. This preserves the model's baseline level.
        for (std::size_t index = 1; index < normal.size(); ++index) {
            normal[index][index] += lambda_;
        }

        Vector rightSide =
            MatrixOperations::multiply(designT, y);

        Matrix inverse =
            MatrixOperations::inverse(normal);

        coefficients_ =
            MatrixOperations::multiply(inverse, rightSide);

        fitted_ = true;
    }

    Vector predict(const Vector& x) const {
        if (!fitted_) {
            throw std::logic_error("Model has not been fitted.");
        }

        Vector transformedX = x;

        if (scaling_) {
            transformedX = scaler_.transform(transformedX);
        }

        Matrix design =
            featureBuilder_.transform(transformedX);

        Vector predictions;
        predictions.reserve(design.size());

        for (const auto& row : design) {
            predictions.push_back(
                std::inner_product(
                    row.begin(),
                    row.end(),
                    coefficients_.begin(),
                    0.0
                )
            );
        }

        return predictions;
    }

    Metrics evaluate(
        const Vector& x,
        const Vector& y
    ) const {
        const Vector predictions = predict(x);

        if (predictions.size() != y.size()) {
            throw std::logic_error("Prediction size mismatch.");
        }

        double absoluteError = 0.0;
        double squaredError = 0.0;

        const double targetMean =
            std::accumulate(y.begin(), y.end(), 0.0) /
            static_cast<double>(y.size());

        double totalVariation = 0.0;

        for (std::size_t index = 0; index < y.size(); ++index) {
            const double residual =
                y[index] - predictions[index];

            absoluteError += std::abs(residual);
            squaredError += residual * residual;

            totalVariation +=
                std::pow(y[index] - targetMean, 2);
        }

        Metrics result;

        result.mae =
            absoluteError / static_cast<double>(y.size());

        result.rmse =
            std::sqrt(squaredError / static_cast<double>(y.size()));

        result.r2 =
            totalVariation == 0.0
                ? 0.0
                : 1.0 - squaredError / totalVariation;

        return result;
    }
};

struct EnergyObservation {
    double temperature;
    double demand;
};

std::vector<EnergyObservation> generateEnergyData(
    std::size_t count
) {
    std::mt19937 generator(2026);
    std::normal_distribution<double> noise(0.0, 18.0);

    std::vector<EnergyObservation> data;
    data.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        const double temperature =
            -5.0 +
            45.0 *
            static_cast<double>(index) /
            static_cast<double>(count - 1);

        // Heating and cooling demand create a nonlinear relationship.
        const double demand =
            420.0
            - 13.0 * temperature
            + 0.72 * std::pow(temperature, 2)
            - 0.012 * std::pow(temperature, 3)
            + noise(generator);

        data.push_back({temperature, demand});
    }

    return data;
}

void splitData(
    const std::vector<EnergyObservation>& data,
    std::vector<EnergyObservation>& training,
    std::vector<EnergyObservation>& testing
) {
    std::vector<std::size_t> indices(data.size());

    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(91);
    std::shuffle(indices.begin(), indices.end(), generator);

    const std::size_t testCount =
        static_cast<std::size_t>(
            std::round(data.size() * 0.2)
        );

    std::vector<bool> testFlag(data.size(), false);

    for (std::size_t index = 0; index < testCount; ++index) {
        testFlag[indices[index]] = true;
    }

    for (std::size_t index = 0; index < data.size(); ++index) {
        if (testFlag[index]) {
            testing.push_back(data[index]);
        } else {
            training.push_back(data[index]);
        }
    }
}

void printMetrics(
    const std::string& label,
    const Metrics& metrics
) {
    std::cout
        << std::left
        << std::setw(22)
        << label
        << "MAE=" << std::setw(10)
        << std::fixed
        << std::setprecision(4)
        << metrics.mae
        << "RMSE=" << std::setw(10)
        << metrics.rmse
        << "R2=" << metrics.r2
        << '\n';
}

int main() {
    try {
        std::cout
            << "============================================================\n"
            << "Polynomial Regression Governance Case Study\n"
            << "============================================================\n\n";

        const auto data = generateEnergyData(140);

        std::vector<EnergyObservation> training;
        std::vector<EnergyObservation> testing;

        splitData(data, training, testing);

        Vector trainX;
        Vector trainY;
        Vector testX;
        Vector testY;

        for (const auto& observation : training) {
            trainX.push_back(observation.temperature);
            trainY.push_back(observation.demand);
        }

        for (const auto& observation : testing) {
            testX.push_back(observation.temperature);
            testY.push_back(observation.demand);
        }

        std::cout << "Training observations: "
                  << trainX.size() << '\n';

        std::cout << "Testing observations: "
                  << testX.size() << "\n\n";

        std::cout
            << "Degree comparison demonstrates the bias-variance trade-off:\n";

        for (int degree : {1, 2, 3, 5, 8}) {
            PolynomialRegression model(
                degree,
                0.1,
                true
            );

            model.fit(trainX, trainY);

            const Metrics trainingMetrics =
                model.evaluate(trainX, trainY);

            const Metrics testingMetrics =
                model.evaluate(testX, testY);

            std::cout
                << "Degree " << degree << '\n';

            printMetrics(
                "  training",
                trainingMetrics
            );

            printMetrics(
                "  testing",
                testingMetrics
            );

            std::cout << '\n';
        }

        std::cout
            << "Regularization comparison for a high-degree model:\n";

        for (double lambda : {0.0, 0.01, 0.1, 1.0, 10.0, 100.0}) {
            PolynomialRegression model(
                8,
                lambda,
                true
            );

            model.fit(trainX, trainY);

            const Metrics testingMetrics =
                model.evaluate(testX, testY);

            std::cout
                << "lambda="
                << std::setw(8)
                << lambda
                << "  test RMSE="
                << std::fixed
                << std::setprecision(4)
                << testingMetrics.rmse
                << "  test R2="
                << testingMetrics.r2
                << '\n';
        }

        std::cout
            << "\nSelected production model: degree 3 with ridge "
               "regularization.\n";

        PolynomialRegression finalModel(
            3,
            0.1,
            true
        );

        finalModel.fit(trainX, trainY);

        printMetrics(
            "final training",
            finalModel.evaluate(trainX, trainY)
        );

        printMetrics(
            "final testing",
            finalModel.evaluate(testX, testY)
        );

        std::cout
            << "\nOperational predictions:\n";

        for (double temperature : {-5.0, 5.0, 15.0, 25.0, 35.0, 40.0}) {
            const double prediction =
                finalModel.predict({temperature}).front();

            std::cout
                << "Temperature "
                << std::setw(6)
                << temperature
                << " C -> predicted demand "
                << std::setw(10)
                << std::fixed
                << std::setprecision(2)
                << prediction
                << '\n';
        }

        std::cout
            << "\nExtrapolation test:\n";

        for (double temperature : {45.0, 55.0, 70.0}) {
            const double prediction =
                finalModel.predict({temperature}).front();

            std::cout
                << temperature
                << " C -> "
                << prediction
                << " demand; outside the observed training range.\n";
        }

        std::cout
            << "\nThe system separates nonlinear feature construction from "
               "coefficient estimation. The transformed model is linear "
               "with respect to its coefficients, while powers of the "
               "original variable allow curved predictions. Scaling, "
               "regularization, held-out evaluation, and extrapolation "
               "checks reduce practical risks associated with flexible "
               "polynomial models.\n";
    }
    catch (const std::exception& error) {
        std::cerr
            << "Execution failed: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}
