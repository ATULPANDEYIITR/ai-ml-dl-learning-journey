/*
 * Regression Mathematics Case Study
 *
 * Scenario:
 * A logistics organization evaluates whether a proposed delivery-time model
 * is safe to deploy. The governance engine estimates coefficients using
 * ordinary least squares, calculates residuals and error metrics, evaluates
 * model complexity with adjusted R², and rejects numerically unsafe models.
 *
 * Build:
 *   g++ -std=c++17 -O2 regression_governance.cpp -o regression_governance
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct RegressionResult {
    Vector coefficients;
    Vector predictions;
    Vector residuals;
    double sse{};
    double mse{};
    double rmse{};
    double rSquared{};
    double adjustedRSquared{};
    std::size_t observations{};
    std::size_t predictors{};
};

struct ModelPolicy {
    double minimumRSquared;
    double minimumAdjustedRSquared;
    double maximumRMSE;
    double maximumAbsoluteResidual;
};

class MatrixOperations {
public:
    static Matrix transpose(const Matrix& matrix) {
        if (matrix.empty()) {
            return {};
        }

        Matrix result(
            matrix.front().size(),
            Vector(matrix.size())
        );

        for (std::size_t i = 0; i < matrix.size(); ++i) {
            for (std::size_t j = 0; j < matrix[i].size(); ++j) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    static Matrix multiply(
        const Matrix& left,
        const Matrix& right
    ) {
        if (left.empty() || right.empty()) {
            throw std::invalid_argument(
                "Matrix multiplication requires non-empty matrices."
            );
        }

        if (left.front().size() != right.size()) {
            throw std::invalid_argument(
                "Matrix dimensions are incompatible."
            );
        }

        Matrix result(
            left.size(),
            Vector(right.front().size(), 0.0)
        );

        for (std::size_t i = 0; i < left.size(); ++i) {
            for (std::size_t k = 0; k < right.size(); ++k) {
                for (std::size_t j = 0; j < right.front().size(); ++j) {
                    result[i][j] += left[i][k] * right[k][j];
                }
            }
        }

        return result;
    }

    static Matrix identity(std::size_t size) {
        Matrix result(size, Vector(size, 0.0));

        for (std::size_t i = 0; i < size; ++i) {
            result[i][i] = 1.0;
        }

        return result;
    }

    static Matrix inverse(Matrix matrix) {
        const std::size_t n = matrix.size();

        if (n == 0) {
            throw std::invalid_argument(
                "Cannot invert an empty matrix."
            );
        }

        for (const auto& row : matrix) {
            if (row.size() != n) {
                throw std::invalid_argument(
                    "Only square matrices can be inverted."
                );
            }
        }

        Matrix augmented(n, Vector(2 * n));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                augmented[i][j] = matrix[i][j];
            }

            for (std::size_t j = 0; j < n; ++j) {
                augmented[i][n + j] =
                    (i == j) ? 1.0 : 0.0;
            }
        }

        // Partial pivoting selects the largest available pivot to reduce
        // numerical amplification during Gauss-Jordan elimination.
        for (std::size_t column = 0; column < n; ++column) {
            std::size_t pivotRow = column;

            for (std::size_t row = column + 1; row < n; ++row) {
                if (
                    std::abs(augmented[row][column]) >
                    std::abs(augmented[pivotRow][column])
                ) {
                    pivotRow = row;
                }
            }

            if (
                std::abs(augmented[pivotRow][column]) <
                1e-12
            ) {
                throw std::runtime_error(
                    "Singular or nearly singular design matrix."
                );
            }

            std::swap(
                augmented[column],
                augmented[pivotRow]
            );

            const double pivot =
                augmented[column][column];

            for (double& value : augmented[column]) {
                value /= pivot;
            }

            for (std::size_t row = 0; row < n; ++row) {
                if (row == column) {
                    continue;
                }

                const double factor =
                    augmented[row][column];

                for (std::size_t j = 0; j < 2 * n; ++j) {
                    augmented[row][j] -=
                        factor * augmented[column][j];
                }
            }
        }

        Matrix result(
            n,
            Vector(n)
        );

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                result[i][j] = augmented[i][n + j];
            }
        }

        return result;
    }
};

class RegressionEngine {
public:
    RegressionResult fit(
        const Matrix& features,
        const Vector& response
    ) const {
        validate(features, response);

        Matrix design = buildDesignMatrix(features);
        Matrix xT = MatrixOperations::transpose(design);
        Matrix xTX = MatrixOperations::multiply(xT, design);
        Matrix inverse = MatrixOperations::inverse(xTX);

        Matrix y(response.size(), Vector(1));

        for (std::size_t i = 0; i < response.size(); ++i) {
            y[i][0] = response[i];
        }

        // Ordinary Least Squares:
        //
        // beta_hat = (X^T X)^(-1) X^T y
        Matrix coefficientsMatrix =
            MatrixOperations::multiply(
                MatrixOperations::multiply(inverse, xT),
                y
            );

        Vector coefficients(coefficientsMatrix.size());

        for (std::size_t i = 0; i < coefficients.size(); ++i) {
            coefficients[i] =
                coefficientsMatrix[i][0];
        }

        Vector predictions =
            predict(coefficients, features);

        Vector residuals(response.size());

        for (std::size_t i = 0; i < response.size(); ++i) {
            residuals[i] =
                response[i] - predictions[i];
        }

        const double sse =
            sumSquared(residuals);

        const double mse =
            sse /
            static_cast<double>(
                response.size()
                - features.front().size()
                - 1
            );

        const double predictiveMSE =
            sse /
            static_cast<double>(response.size());

        const double rmse =
            std::sqrt(predictiveMSE);

        const double sst =
            totalSumOfSquares(response);

        if (std::abs(sst) < 1e-12) {
            throw std::runtime_error(
                "R² is undefined for a constant response."
            );
        }

        const double rSquared =
            1.0 - sse / sst;

        const double n =
            static_cast<double>(response.size());

        const double p =
            static_cast<double>(features.front().size());

        const double adjustedRSquared =
            1.0 -
            (1.0 - rSquared) *
            ((n - 1.0) / (n - p - 1.0));

        return {
            coefficients,
            predictions,
            residuals,
            sse,
            mse,
            rmse,
            rSquared,
            adjustedRSquared,
            response.size(),
            features.front().size()
        };
    }

    static Vector predict(
        const Vector& coefficients,
        const Matrix& features
    ) {
        const std::size_t predictors =
            coefficients.size() - 1;

        Vector predictions;
        predictions.reserve(features.size());

        for (const auto& row : features) {
            if (row.size() != predictors) {
                throw std::invalid_argument(
                    "Feature dimension does not match coefficients."
                );
            }

            double prediction = coefficients[0];

            for (std::size_t j = 0; j < predictors; ++j) {
                prediction +=
                    coefficients[j + 1] * row[j];
            }

            predictions.push_back(prediction);
        }

        return predictions;
    }

private:
    static Matrix buildDesignMatrix(
        const Matrix& features
    ) {
        Matrix design(
            features.size(),
            Vector(features.front().size() + 1)
        );

        for (std::size_t i = 0; i < features.size(); ++i) {
            design[i][0] = 1.0;

            for (std::size_t j = 0; j < features[i].size(); ++j) {
                design[i][j + 1] = features[i][j];
            }
        }

        return design;
    }

    static double sumSquared(
        const Vector& values
    ) {
        return std::accumulate(
            values.begin(),
            values.end(),
            0.0,
            [](double sum, double value) {
                return sum + value * value;
            }
        );
    }

    static double totalSumOfSquares(
        const Vector& response
    ) {
        const double average =
            std::accumulate(
                response.begin(),
                response.end(),
                0.0
            ) / static_cast<double>(response.size());

        double total = 0.0;

        for (double value : response) {
            const double difference =
                value - average;

            total += difference * difference;
        }

        return total;
    }

    static void validate(
        const Matrix& features,
        const Vector& response
    ) {
        if (features.empty() || response.empty()) {
            throw std::invalid_argument(
                "Features and response cannot be empty."
            );
        }

        if (features.size() != response.size()) {
            throw std::invalid_argument(
                "Feature and response row counts differ."
            );
        }

        const std::size_t width =
            features.front().size();

        if (width == 0) {
            throw std::invalid_argument(
                "At least one predictor is required."
            );
        }

        if (
            response.size() <= width + 1
        ) {
            throw std::invalid_argument(
                "Insufficient degrees of freedom."
            );
        }

        for (const auto& row : features) {
            if (row.size() != width) {
                throw std::invalid_argument(
                    "Predictor rows have inconsistent widths."
                );
            }

            for (double value : row) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "All predictor values must be finite."
                    );
                }
            }
        }

        for (double value : response) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "All response values must be finite."
                );
            }
        }
    }
};

class GovernanceEngine {
public:
    static bool isEligible(
        const RegressionResult& result,
        const ModelPolicy& policy
    ) {
        const double largestResidual =
            *std::max_element(
                result.residuals.begin(),
                result.residuals.end(),
                [](double a, double b) {
                    return std::abs(a) < std::abs(b);
                }
            );

        const double absoluteLargestResidual =
            std::abs(largestResidual);

        return
            result.rSquared >= policy.minimumRSquared &&
            result.adjustedRSquared >=
                policy.minimumAdjustedRSquared &&
            result.rmse <= policy.maximumRMSE &&
            absoluteLargestResidual <=
                policy.maximumAbsoluteResidual;
    }
};

static void printResult(
    const std::string& name,
    const RegressionResult& result
) {
    std::cout
        << "\n"
        << std::string(72, '=')
        << "\n"
        << name
        << "\n"
        << std::string(72, '=')
        << "\n";

    std::cout << std::fixed
              << std::setprecision(6);

    for (
        std::size_t i = 0;
        i < result.coefficients.size();
        ++i
    ) {
        std::cout
            << (i == 0 ? "intercept" :
                "beta_" + std::to_string(i))
            << " = "
            << result.coefficients[i]
            << "\n";
    }

    std::cout << "SSE = "
              << result.sse
              << "\n";

    std::cout << "Model-error MSE = "
              << result.mse
              << "\n";

    std::cout << "Predictive RMSE = "
              << result.rmse
              << "\n";

    std::cout << "R² = "
              << result.rSquared
              << "\n";

    std::cout << "Adjusted R² = "
              << result.adjustedRSquared
              << "\n";

    std::cout
        << "\nObservation residuals:\n";

    for (
        std::size_t i = 0;
        i < result.residuals.size();
        ++i
    ) {
        std::cout
            << "row "
            << std::setw(2)
            << i
            << " prediction="
            << std::setw(9)
            << result.predictions[i]
            << " residual="
            << std::setw(9)
            << result.residuals[i]
            << "\n";
    }
}

int main() {
    try {
        /*
         * Enterprise scenario:
         *
         * A distribution operation predicts delivery duration using
         * processing time and shipment size. The governance engine evaluates
         * the fitted model against quantitative deployment rules.
         */
        Matrix features = {
            {2.0, 10.0},
            {3.0, 12.0},
            {4.0, 14.0},
            {5.0, 18.0},
            {6.0, 17.0},
            {7.0, 21.0},
            {8.0, 23.0},
            {9.0, 24.0},
            {10.0, 28.0},
            {11.0, 29.0},
            {12.0, 32.0},
            {13.0, 35.0}
        };

        Vector response = {
            8.4,
            10.2,
            11.7,
            14.1,
            14.8,
            17.0,
            18.9,
            19.7,
            23.0,
            24.2,
            26.0,
            29.1
        };

        RegressionEngine engine;
        RegressionResult result =
            engine.fit(features, response);

        printResult(
            "Delivery-Time Regression Governance Case",
            result
        );

        ModelPolicy policy{
            0.90,
            0.88,
            2.0,
            3.0
        };

        const bool eligible =
            GovernanceEngine::isEligible(
                result,
                policy
            );

        std::cout
            << "\nDeployment decision: "
            << (eligible ? "ELIGIBLE" : "NOT ELIGIBLE")
            << "\n";

        /*
         * A prediction for a future shipment uses the learned coefficient
         * vector without refitting the model.
         */
        Matrix future = {
            {6.5, 20.0},
            {9.5, 25.0},
            {14.0, 38.0}
        };

        Vector futurePredictions =
            RegressionEngine::predict(
                result.coefficients,
                future
            );

        std::cout
            << "\nFuture predictions:\n";

        for (
            std::size_t i = 0;
            i < futurePredictions.size();
            ++i
        ) {
            std::cout
                << "case "
                << i + 1
                << ": "
                << futurePredictions[i]
                << " hours\n";
        }

        /*
         * The following dataset contains perfect linear dependence between
         * predictors. XᵀX becomes singular, so the normal equation cannot
         * uniquely estimate coefficients.
         */
        try {
            Matrix collinearFeatures = {
                {1.0, 2.0},
                {2.0, 4.0},
                {3.0, 6.0},
                {4.0, 8.0},
                {5.0, 10.0}
            };

            Vector collinearResponse = {
                2.0, 4.1, 6.0, 8.2, 10.1
            };

            engine.fit(
                collinearFeatures,
                collinearResponse
            );
        }
        catch (const std::exception& error) {
            std::cout
                << "\nCollinearity protection: "
                << error.what()
                << "\n";
        }

        /*
         * R² can be high even when a model is over-parameterized. Adjusted
         * R² introduces a degrees-of-freedom penalty for extra predictors.
         * A governance system should therefore compare both metrics rather
         * than treating raw R² as a universal model-selection criterion.
         */
        std::cout
            << "\nGovernance interpretation:\n"
            << "R² measures relative explanatory power against the "
               "intercept-only baseline.\n"
            << "Adjusted R² accounts for the number of predictors.\n"
            << "MSE and RMSE retain the original response scale after "
               "taking the square root for RMSE.\n"
            << "Residuals reveal observation-level prediction errors.\n";
    }
    catch (const std::exception& error) {
        std::cerr
            << "Fatal regression error: "
            << error.what()
            << "\n";

        return 1;
    }

    return 0;
}
