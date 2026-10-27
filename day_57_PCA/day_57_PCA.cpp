#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class PCAError : public std::runtime_error {
public:
    explicit PCAError(const std::string& message)
        : std::runtime_error(message) {}
};

struct PCAResult {
    Vector means;
    Matrix covariance;
    Vector eigenvalues;
    Matrix eigenvectors;
    Vector explainedVarianceRatio;
    Vector cumulativeExplainedVariance;
    Matrix transformed;
};

class RepositoryGovernancePCA {
private:
    static constexpr double EPSILON = 1e-12;

    static void validateMatrix(const Matrix& data) {
        if (data.size() < 2) {
            throw PCAError("At least two observations are required.");
        }

        if (data.front().empty()) {
            throw PCAError("At least one feature is required.");
        }

        const std::size_t width = data.front().size();

        for (std::size_t i = 0; i < data.size(); ++i) {
            if (data[i].size() != width) {
                throw PCAError("All observations must have equal dimensions.");
            }

            for (double value : data[i]) {
                if (!std::isfinite(value)) {
                    throw PCAError("All observations must be finite.");
                }
            }
        }
    }

    static double dot(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) {
            throw PCAError("Vector dimension mismatch.");
        }

        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }

    static double norm(const Vector& vector) {
        return std::sqrt(dot(vector, vector));
    }

    static Vector normalize(Vector vector) {
        const double length = norm(vector);

        if (length < EPSILON) {
            throw PCAError("Cannot normalize a zero vector.");
        }

        for (double& value : vector) {
            value /= length;
        }

        return vector;
    }

    static Vector matrixVectorMultiply(
        const Matrix& matrix,
        const Vector& vector
    ) {
        Vector result(matrix.size(), 0.0);

        for (std::size_t i = 0; i < matrix.size(); ++i) {
            result[i] = dot(matrix[i], vector);
        }

        return result;
    }

    static Matrix covariance(const Matrix& centered) {
        const std::size_t n = centered.size();
        const std::size_t dimensions = centered.front().size();

        Matrix result(
            dimensions,
            Vector(dimensions, 0.0)
        );

        for (std::size_t i = 0; i < dimensions; ++i) {
            for (std::size_t j = i; j < dimensions; ++j) {
                double sum = 0.0;

                for (const auto& row : centered) {
                    sum += row[i] * row[j];
                }

                const double value = sum / static_cast<double>(n - 1);
                result[i][j] = value;
                result[j][i] = value;
            }
        }

        return result;
    }

    static std::pair<double, Vector> powerIteration(
        const Matrix& matrix,
        std::size_t maximumIterations = 10000
    ) {
        const std::size_t n = matrix.size();

        Vector vector(n);
        for (std::size_t i = 0; i < n; ++i) {
            vector[i] = 1.0 + static_cast<double>(i) / n;
        }

        vector = normalize(vector);

        double eigenvalue = 0.0;

        for (std::size_t iteration = 0;
             iteration < maximumIterations;
             ++iteration) {

            Vector candidate = normalize(
                matrixVectorMultiply(matrix, vector)
            );

            const double newEigenvalue = dot(
                candidate,
                matrixVectorMultiply(matrix, candidate)
            );

            double difference = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double delta = candidate[i] - vector[i];
                difference += delta * delta;
            }

            difference = std::sqrt(difference);
            vector = candidate;

            if (
                std::abs(newEigenvalue - eigenvalue) < EPSILON &&
                difference < EPSILON
            ) {
                eigenvalue = newEigenvalue;
                break;
            }

            eigenvalue = newEigenvalue;

            if (iteration + 1 == maximumIterations) {
                throw PCAError("Power iteration failed to converge.");
            }
        }

        auto pivot = std::max_element(
            vector.begin(),
            vector.end(),
            [](double a, double b) {
                return std::abs(a) < std::abs(b);
            }
        );

        if (*pivot < 0.0) {
            for (double& value : vector) {
                value = -value;
            }
        }

        return {eigenvalue, vector};
    }

    static Matrix deflate(
        const Matrix& matrix,
        double eigenvalue,
        const Vector& eigenvector
    ) {
        Matrix result = matrix;

        for (std::size_t i = 0; i < matrix.size(); ++i) {
            for (std::size_t j = 0; j < matrix.size(); ++j) {
                result[i][j] -=
                    eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }

        return result;
    }

    static std::pair<Vector, Matrix> eigenDecomposition(
        const Matrix& matrix
    ) {
        Matrix working = matrix;
        Vector eigenvalues;
        Matrix eigenvectors;

        for (std::size_t component = 0;
             component < matrix.size();
             ++component) {

            auto [value, vector] = powerIteration(working);

            if (std::abs(value) < EPSILON) {
                value = 0.0;
            }

            eigenvalues.push_back(value);
            eigenvectors.push_back(vector);

            working = deflate(working, value, vector);
        }

        std::vector<std::size_t> order(eigenvalues.size());
        std::iota(order.begin(), order.end(), 0);

        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t a, std::size_t b) {
                return eigenvalues[a] > eigenvalues[b];
            }
        );

        Vector sortedValues;
        Matrix sortedVectors;

        for (std::size_t index : order) {
            sortedValues.push_back(eigenvalues[index]);
            sortedVectors.push_back(eigenvectors[index]);
        }

        return {sortedValues, sortedVectors};
    }

public:
    static PCAResult fit(
        const Matrix& data,
        std::size_t requestedComponents
    ) {
        validateMatrix(data);

        const std::size_t dimensions = data.front().size();

        if (
            requestedComponents == 0 ||
            requestedComponents > dimensions
        ) {
            throw PCAError("Invalid component count.");
        }

        Vector means(dimensions, 0.0);

        for (const auto& row : data) {
            for (std::size_t j = 0; j < dimensions; ++j) {
                means[j] += row[j];
            }
        }

        for (double& mean : means) {
            mean /= static_cast<double>(data.size());
        }

        Matrix centered = data;

        for (auto& row : centered) {
            for (std::size_t j = 0; j < dimensions; ++j) {
                row[j] -= means[j];
            }
        }

        Matrix covarianceMatrix = covariance(centered);
        auto [eigenvalues, eigenvectors] =
            eigenDecomposition(covarianceMatrix);

        double totalVariance = 0.0;

        for (double value : eigenvalues) {
            totalVariance += std::max(0.0, value);
        }

        if (totalVariance < EPSILON) {
            throw PCAError("Dataset contains no measurable variance.");
        }

        Vector explained;
        Vector cumulative;

        double running = 0.0;

        for (double value : eigenvalues) {
            const double ratio =
                std::max(0.0, value) / totalVariance;

            explained.push_back(ratio);
            running += ratio;
            cumulative.push_back(running);
        }

        Matrix transformed;

        for (const auto& row : centered) {
            Vector projected;

            for (std::size_t component = 0;
                 component < requestedComponents;
                 ++component) {

                projected.push_back(
                    dot(row, eigenvectors[component])
                );
            }

            transformed.push_back(projected);
        }

        return {
            means,
            covarianceMatrix,
            eigenvalues,
            eigenvectors,
            explained,
            cumulative,
            transformed
        };
    }

    static std::size_t componentsForVariance(
        const PCAResult& result,
        double threshold
    ) {
        if (threshold <= 0.0 || threshold > 1.0) {
            throw PCAError("Variance threshold must be in (0, 1].");
        }

        for (std::size_t i = 0;
             i < result.cumulativeExplainedVariance.size();
             ++i) {

            if (result.cumulativeExplainedVariance[i] >= threshold) {
                return i + 1;
            }
        }

        return result.eigenvalues.size();
    }
};

static void printMatrix(
    const std::string& title,
    const Matrix& matrix
) {
    std::cout << "\n" << title << "\n";

    for (const auto& row : matrix) {
        std::cout << "  ";

        for (double value : row) {
            std::cout
                << std::setw(11)
                << std::fixed
                << std::setprecision(5)
                << value;
        }

        std::cout << '\n';
    }
}

static void printReport(const PCAResult& result) {
    printMatrix("Covariance matrix", result.covariance);

    std::cout << "\nEigen-analysis\n";

    for (std::size_t i = 0; i < result.eigenvalues.size(); ++i) {
        std::cout
            << "  PC" << i + 1
            << " eigenvalue=" << std::fixed
            << std::setprecision(6)
            << result.eigenvalues[i]
            << " explained="
            << std::setprecision(2)
            << result.explainedVarianceRatio[i] * 100.0
            << "% cumulative="
            << result.cumulativeExplainedVariance[i] * 100.0
            << "%\n";
    }

    std::cout << "\nEigenvectors\n";

    for (std::size_t i = 0; i < result.eigenvectors.size(); ++i) {
        std::cout << "  PC" << i + 1 << ": ";

        for (double value : result.eigenvectors[i]) {
            std::cout
                << std::fixed
                << std::setprecision(6)
                << value << ' ';
        }

        std::cout << '\n';
    }

    printMatrix("Projected observations", result.transformed);
}

int main() {
    try {
        /*
         * Repository governance telemetry:
         * features represent review latency, changed-file count,
         * status-check duration, and review-comment count.
         *
         * The purpose is not to assign a "quality score" directly.
         * PCA identifies correlated operational dimensions that can
         * support subsequent analysis.
         */
        const Matrix pullRequestTelemetry = {
            {2.0,  8.0,  4.0, 3.0},
            {3.0,  9.0,  5.0, 3.5},
            {1.5,  6.0,  3.0, 2.0},
            {5.0, 15.0, 8.0, 7.0},
            {6.0, 18.0, 9.0, 8.0},
            {7.0, 21.0, 11.0, 9.0},
            {2.5,  7.0,  4.0, 2.5},
            {4.5, 13.0, 7.0, 6.0}
        };

        std::cout
            << "PCA repository telemetry analysis\n"
            << "=================================\n"
            << "Observations: " << pullRequestTelemetry.size() << '\n'
            << "Features: "
            << pullRequestTelemetry.front().size() << '\n';

        PCAResult result = RepositoryGovernancePCA::fit(
            pullRequestTelemetry,
            pullRequestTelemetry.front().size()
        );

        printReport(result);

        const std::size_t required =
            RepositoryGovernancePCA::componentsForVariance(
                result,
                0.95
            );

        std::cout
            << "\nComponents required for 95% variance: "
            << required << '\n';

        /*
         * For a covariance matrix, PCA is scale-sensitive.
         * Review latency measured in minutes and file count measured
         * as integers therefore contribute according to their units.
         * A production pipeline should explicitly decide whether
         * standardization is appropriate before PCA.
         */
        const double ratioSum = std::accumulate(
            result.explainedVarianceRatio.begin(),
            result.explainedVarianceRatio.end(),
            0.0
        );

        if (std::abs(ratioSum - 1.0) > 1e-9) {
            throw PCAError(
                "Explained variance ratios do not sum to one."
            );
        }

        try {
            const Matrix invalid = {
                {1.0, 1.0},
                {1.0, 1.0}
            };

            RepositoryGovernancePCA::fit(invalid, 2);
        } catch (const PCAError& error) {
            std::cout
                << "\nExpected failure: "
                << error.what() << '\n';
        }

        std::cout
            << "\nEngineering considerations\n"
            << "- Covariance uses n-1 for sample covariance.\n"
            << "- Symmetry is preserved during covariance construction.\n"
            << "- Power iteration provides transparent eigen-analysis.\n"
            << "- Deflation extracts subsequent principal directions.\n"
            << "- Eigenvector signs are normalized for deterministic output.\n"
            << "- Optimized eigensolvers are preferable for high-dimensional data.\n"
            << "- Sensitive telemetry should be validated and access-controlled.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "PCA analysis failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
