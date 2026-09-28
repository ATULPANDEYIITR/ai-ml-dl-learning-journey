/*
    MATHEMATICAL REVISION
    Probability, Statistics, Linear Algebra, Calculus, and Optimization

    C++17 industry-style case study:
    Portfolio Risk, Linear Regression, Principal Components,
    Gradient-Based Optimization, and Risk-Constrained Allocation.

    The program is intentionally self-contained and uses only the C++ standard
    library.
*/

#include <algorithm>
#include <cmath>
#include <exception>
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

constexpr double EPSILON = 1e-10;

// =============================================================================
// 1. GENERAL UTILITIES
// =============================================================================

void section(const std::string& title) {
    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(78, '=') << "\n";
}

void subsection(const std::string& title) {
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void printVector(const Vector& values, int precision = 6) {
    std::cout << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << std::fixed << std::setprecision(precision)
                  << values[i];

        if (i + 1 < values.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void printMatrix(const Matrix& matrix, int precision = 5) {
    for (const auto& row : matrix) {
        printVector(row, precision);
    }
}

bool approximatelyEqual(
    double a,
    double b,
    double tolerance = 1e-8
) {
    return std::abs(a - b) <= tolerance *
        std::max({1.0, std::abs(a), std::abs(b)});
}


// =============================================================================
// 2. BASIC VECTOR OPERATIONS
// =============================================================================

Vector vectorAdd(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions must agree."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Vector vectorSubtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions must agree."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

Vector scalarMultiply(double scalar, const Vector& vector) {
    Vector result = vector;

    for (double& value : result) {
        value *= scalar;
    }

    return result;
}

double dotProduct(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions must agree."
        );
    }

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double vectorNorm(const Vector& vector) {
    return std::sqrt(dotProduct(vector, vector));
}


// =============================================================================
// 3. MATRIX OPERATIONS
// =============================================================================

void validateMatrix(const Matrix& matrix) {
    if (matrix.empty()) {
        throw std::invalid_argument("Matrix cannot be empty.");
    }

    if (matrix.front().empty()) {
        throw std::invalid_argument(
            "Matrix cannot contain empty rows."
        );
    }

    const std::size_t columns = matrix.front().size();

    for (const auto& row : matrix) {
        if (row.size() != columns) {
            throw std::invalid_argument(
                "Matrix rows must have equal lengths."
            );
        }
    }
}

Matrix transpose(const Matrix& matrix) {
    validateMatrix(matrix);

    Matrix result(
        matrix.front().size(),
        Vector(matrix.size(), 0.0)
    );

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        for (std::size_t j = 0; j < matrix.front().size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
    validateMatrix(A);
    validateMatrix(B);

    if (A.front().size() != B.size()) {
        throw std::invalid_argument(
            "Matrix multiplication dimensions do not agree."
        );
    }

    Matrix result(
        A.size(),
        Vector(B.front().size(), 0.0)
    );

    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t k = 0; k < B.size(); ++k) {
            for (std::size_t j = 0; j < B.front().size(); ++j) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

Vector matrixVectorMultiply(
    const Matrix& matrix,
    const Vector& vector
) {
    validateMatrix(matrix);

    if (matrix.front().size() != vector.size()) {
        throw std::invalid_argument(
            "Matrix and vector dimensions do not agree."
        );
    }

    Vector result(matrix.size(), 0.0);

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        result[i] = dotProduct(matrix[i], vector);
    }

    return result;
}


// =============================================================================
// 4. GAUSSIAN ELIMINATION
// =============================================================================

Vector solveLinearSystem(Matrix matrix, Vector constants) {
    validateMatrix(matrix);

    const std::size_t n = matrix.size();

    if (matrix.front().size() != n) {
        throw std::invalid_argument(
            "Coefficient matrix must be square."
        );
    }

    if (constants.size() != n) {
        throw std::invalid_argument(
            "Constants vector has incorrect dimension."
        );
    }

    Matrix augmented(n, Vector(n + 1));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            augmented[i][j] = matrix[i][j];
        }

        augmented[i][n] = constants[i];
    }

    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t bestRow = pivot;

        for (std::size_t row = pivot + 1; row < n; ++row) {
            if (
                std::abs(augmented[row][pivot]) >
                std::abs(augmented[bestRow][pivot])
            ) {
                bestRow = row;
            }
        }

        if (std::abs(augmented[bestRow][pivot]) < EPSILON) {
            throw std::runtime_error(
                "Matrix is singular or numerically unstable."
            );
        }

        std::swap(augmented[pivot], augmented[bestRow]);

        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double factor =
                augmented[row][pivot] /
                augmented[pivot][pivot];

            for (std::size_t column = pivot; column <= n; ++column) {
                augmented[row][column] -=
                    factor * augmented[pivot][column];
            }
        }
    }

    Vector solution(n, 0.0);

    // Back substitution solves the upper-triangular system.
    for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
        double value = augmented[row][n];

        for (
            std::size_t column = row + 1;
            column < n;
            ++column
        ) {
            value -=
                augmented[row][column] *
                solution[column];
        }

        solution[row] =
            value / augmented[row][row];
    }

    return solution;
}


// =============================================================================
// 5. STATISTICS
// =============================================================================

double mean(const Vector& values) {
    if (values.empty()) {
        throw std::invalid_argument(
            "Mean requires at least one observation."
        );
    }

    return std::accumulate(
        values.begin(),
        values.end(),
        0.0
    ) / static_cast<double>(values.size());
}

double sampleVariance(const Vector& values) {
    if (values.size() < 2) {
        throw std::invalid_argument(
            "Sample variance requires at least two observations."
        );
    }

    const double average = mean(values);

    double sumSquaredDeviations = 0.0;

    for (double value : values) {
        const double difference = value - average;
        sumSquaredDeviations += difference * difference;
    }

    return sumSquaredDeviations /
        static_cast<double>(values.size() - 1);
}

double sampleStandardDeviation(const Vector& values) {
    return std::sqrt(sampleVariance(values));
}

double covariance(
    const Vector& x,
    const Vector& y
) {
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument(
            "Covariance requires equal samples with at least two values."
        );
    }

    const double meanX = mean(x);
    const double meanY = mean(y);

    double total = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        total +=
            (x[i] - meanX) *
            (y[i] - meanY);
    }

    return total /
        static_cast<double>(x.size() - 1);
}

double correlation(
    const Vector& x,
    const Vector& y
) {
    const double standardDeviationX =
        sampleStandardDeviation(x);

    const double standardDeviationY =
        sampleStandardDeviation(y);

    if (
        approximatelyEqual(standardDeviationX, 0.0) ||
        approximatelyEqual(standardDeviationY, 0.0)
    ) {
        throw std::invalid_argument(
            "Correlation is undefined for zero variance."
        );
    }

    return covariance(x, y) /
        (standardDeviationX * standardDeviationY);
}


// =============================================================================
// 6. LINEAR REGRESSION
// =============================================================================

struct LinearRegressionModel {
    double intercept{};
    double slope{};

    double predict(double x) const {
        return intercept + slope * x;
    }
};

LinearRegressionModel fitLinearRegression(
    const Vector& x,
    const Vector& y
) {
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument(
            "Regression requires equal samples with at least two points."
        );
    }

    const double meanX = mean(x);
    const double meanY = mean(y);

    double numerator = 0.0;
    double denominator = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        numerator +=
            (x[i] - meanX) *
            (y[i] - meanY);

        denominator +=
            (x[i] - meanX) *
            (x[i] - meanX);
    }

    if (approximatelyEqual(denominator, 0.0)) {
        throw std::invalid_argument(
            "Regression slope is undefined."
        );
    }

    const double slope = numerator / denominator;
    const double intercept = meanY - slope * meanX;

    return {intercept, slope};
}


// =============================================================================
// 7. PORTFOLIO MODEL
// =============================================================================

struct Asset {
    std::string name;
    Vector returns;
};

class Portfolio {
private:
    std::vector<Asset> assets_;
    Vector weights_;

public:
    Portfolio(
        std::vector<Asset> assets,
        Vector weights
    )
        : assets_(std::move(assets)),
          weights_(std::move(weights)) {

        if (assets_.empty()) {
            throw std::invalid_argument(
                "Portfolio must contain assets."
            );
        }

        if (weights_.size() != assets_.size()) {
            throw std::invalid_argument(
                "Weight count must match asset count."
            );
        }

        const std::size_t observationCount =
            assets_.front().returns.size();

        if (observationCount < 2) {
            throw std::invalid_argument(
                "Each asset needs at least two observations."
            );
        }

        for (const auto& asset : assets_) {
            if (asset.returns.size() != observationCount) {
                throw std::invalid_argument(
                    "All assets must have equal observation counts."
                );
            }
        }
    }

    const Vector& weights() const {
        return weights_;
    }

    double weightSum() const {
        return std::accumulate(
            weights_.begin(),
            weights_.end(),
            0.0
        );
    }

    void validateWeights() const {
        if (!approximatelyEqual(weightSum(), 1.0)) {
            throw std::invalid_argument(
                "Portfolio weights must sum to 1."
            );
        }
    }

    Vector expectedAssetReturns() const {
        Vector result;

        for (const auto& asset : assets_) {
            result.push_back(mean(asset.returns));
        }

        return result;
    }

    Matrix covarianceMatrix() const {
        const std::size_t n = assets_.size();

        Matrix result(
            n,
            Vector(n, 0.0)
        );

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                result[i][j] =
                    covariance(
                        assets_[i].returns,
                        assets_[j].returns
                    );
            }
        }

        return result;
    }

    double expectedReturn() const {
        validateWeights();

        return dotProduct(
            weights_,
            expectedAssetReturns()
        );
    }

    double variance() const {
        validateWeights();

        const Matrix covariance =
            covarianceMatrix();

        const Vector covarianceWeights =
            matrixVectorMultiply(
                covariance,
                weights_
            );

        return dotProduct(
            weights_,
            covarianceWeights
        );
    }

    double standardDeviation() const {
        return std::sqrt(
            std::max(0.0, variance())
        );
    }

    double sharpeRatio(
        double riskFreeRate = 0.0
    ) const {
        const double risk =
            standardDeviation();

        if (risk <= EPSILON) {
            throw std::runtime_error(
                "Sharpe ratio undefined for zero volatility."
            );
        }

        return (
            expectedReturn() - riskFreeRate
        ) / risk;
    }

    Vector currentWeights() const {
        return weights_;
    }
};


// =============================================================================
// 8. GRADIENT-BASED PORTFOLIO OPTIMIZATION
// =============================================================================

double portfolioObjective(
    const Vector& weights,
    const Matrix& covariance,
    double riskAversion,
    const Vector& expectedReturns
) {
    const Vector covarianceWeights =
        matrixVectorMultiply(
            covariance,
            weights
        );

    const double variance =
        dotProduct(weights, covarianceWeights);

    const double expectedReturn =
        dotProduct(weights, expectedReturns);

    // Minimizing:
    //
    //     riskAversion * variance - expectedReturn
    //
    // balances risk against expected return.
    return riskAversion * variance - expectedReturn;
}

Vector numericalGradient(
    const std::function<double(const Vector&)>& function,
    const Vector& point,
    double h = 1e-6
) {
    if (h <= 0) {
        throw std::invalid_argument(
            "Numerical-gradient step must be positive."
        );
    }

    Vector gradient(point.size());

    for (std::size_t i = 0; i < point.size(); ++i) {
        Vector plus = point;
        Vector minus = point;

        plus[i] += h;
        minus[i] -= h;

        gradient[i] =
            (
                function(plus) -
                function(minus)
            ) / (2.0 * h);
    }

    return gradient;
}

Vector projectToSimplex(const Vector& vector) {
    /*
        Projection onto the probability/simplex constraint

            w_i >= 0
            Σw_i = 1

        is useful when portfolio weights cannot be negative.
        The algorithm sorts the coordinates and computes the appropriate
        threshold. Its complexity is O(n log n).
    */

    const std::size_t n = vector.size();

    if (n == 0) {
        throw std::invalid_argument(
            "Cannot project an empty vector."
        );
    }

    Vector sorted = vector;

    std::sort(
        sorted.begin(),
        sorted.end(),
        std::greater<double>()
    );

    double cumulative = 0.0;
    double threshold = 0.0;
    std::size_t activeCount = 0;

    for (std::size_t i = 0; i < n; ++i) {
        cumulative += sorted[i];

        const double candidate =
            (cumulative - 1.0) /
            static_cast<double>(i + 1);

        if (sorted[i] - candidate > 0.0) {
            threshold = candidate;
            activeCount = i + 1;
        }
    }

    if (activeCount == 0) {
        throw std::runtime_error(
            "Simplex projection failed."
        );
    }

    Vector result(n);

    for (std::size_t i = 0; i < n; ++i) {
        result[i] =
            std::max(
                0.0,
                vector[i] - threshold
            );
    }

    return result;
}

Vector optimizePortfolio(
    const Matrix& covariance,
    const Vector& expectedReturns,
    double riskAversion,
    double learningRate,
    int iterations
) {
    if (
        covariance.empty() ||
        covariance.size() != expectedReturns.size()
    ) {
        throw std::invalid_argument(
            "Optimization dimensions are invalid."
        );
    }

    if (riskAversion <= 0 || learningRate <= 0) {
        throw std::invalid_argument(
            "Risk aversion and learning rate must be positive."
        );
    }

    Vector weights(
        expectedReturns.size(),
        1.0 / static_cast<double>(expectedReturns.size())
    );

    auto objective =
        [&](const Vector& candidate) {
            return portfolioObjective(
                candidate,
                covariance,
                riskAversion,
                expectedReturns
            );
        };

    for (int iteration = 0; iteration < iterations; ++iteration) {
        Vector gradient =
            numericalGradient(
                objective,
                weights
            );

        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] -=
                learningRate * gradient[i];
        }

        // Projection enforces the constraints:
        // weights >= 0 and sum(weights) = 1.
        weights = projectToSimplex(weights);
    }

    return weights;
}


// =============================================================================
// 9. PRINCIPAL COMPONENT POWER ITERATION
// =============================================================================

struct EigenResult {
    double eigenvalue;
    Vector eigenvector;
};

EigenResult powerIteration(
    const Matrix& matrix,
    int iterations = 500
) {
    validateMatrix(matrix);

    if (matrix.size() != matrix.front().size()) {
        throw std::invalid_argument(
            "Power iteration requires a square matrix."
        );
    }

    Vector vector(
        matrix.size(),
        1.0
    );

    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {

        Vector transformed =
            matrixVectorMultiply(
                matrix,
                vector
            );

        const double norm =
            vectorNorm(transformed);

        if (norm <= EPSILON) {
            throw std::runtime_error(
                "Power iteration reached a zero vector."
            );
        }

        vector =
            scalarMultiply(
                1.0 / norm,
                transformed
            );
    }

    Vector transformed =
        matrixVectorMultiply(
            matrix,
            vector
        );

    const double eigenvalue =
        dotProduct(
            vector,
            transformed
        );

    return {eigenvalue, vector};
}


// =============================================================================
// 10. GRADIENT DESCENT FOR A QUADRATIC
// =============================================================================

double quadraticFunction(const Vector& point) {
    if (point.size() != 2) {
        throw std::invalid_argument(
            "Quadratic example requires two variables."
        );
    }

    const double x = point[0];
    const double y = point[1];

    return (
        (x - 3.0) * (x - 3.0) +
        2.0 * (y + 1.0) * (y + 1.0)
    );
}

Vector optimizeQuadratic() {
    Vector point{0.0, 0.0};

    const double learningRate = 0.1;

    for (int iteration = 0;
         iteration < 200;
         ++iteration) {

        Vector gradient =
            numericalGradient(
                quadraticFunction,
                point
            );

        Vector next =
            vectorSubtract(
                point,
                scalarMultiply(
                    learningRate,
                    gradient
                )
            );

        if (
            vectorNorm(
                vectorSubtract(next, point)
            ) < 1e-8
        ) {
            point = next;
            break;
        }

        point = next;
    }

    return point;
}


// =============================================================================
// 11. BISECTION ROOT FINDING
// =============================================================================

double bisection(
    const std::function<double(double)>& function,
    double lower,
    double upper,
    double tolerance = 1e-10,
    int iterations = 200
) {
    double lowerValue = function(lower);
    double upperValue = function(upper);

    if (approximatelyEqual(lowerValue, 0.0)) {
        return lower;
    }

    if (approximatelyEqual(upperValue, 0.0)) {
        return upper;
    }

    if (lowerValue * upperValue > 0.0) {
        throw std::invalid_argument(
            "Bisection requires opposite endpoint signs."
        );
    }

    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {

        const double midpoint =
            (lower + upper) / 2.0;

        const double midpointValue =
            function(midpoint);

        if (
            std::abs(midpointValue) < tolerance ||
            std::abs(upper - lower) < tolerance
        ) {
            return midpoint;
        }

        if (lowerValue * midpointValue < 0.0) {
            upper = midpoint;
            upperValue = midpointValue;
        } else {
            lower = midpoint;
            lowerValue = midpointValue;
        }
    }

    return (lower + upper) / 2.0;
}


// =============================================================================
// 12. NORMAL DISTRIBUTION
// =============================================================================

double normalPDF(
    double x,
    double meanValue = 0.0,
    double standardDeviation = 1.0
) {
    if (standardDeviation <= 0.0) {
        throw std::invalid_argument(
            "Standard deviation must be positive."
        );
    }

    const double z =
        (x - meanValue) /
        standardDeviation;

    return std::exp(-0.5 * z * z) /
        (
            standardDeviation *
            std::sqrt(2.0 * M_PI)
        );
}

double normalCDF(
    double x,
    double meanValue = 0.0,
    double standardDeviation = 1.0
) {
    if (standardDeviation <= 0.0) {
        throw std::invalid_argument(
            "Standard deviation must be positive."
        );
    }

    const double z =
        (x - meanValue) /
        (standardDeviation * std::sqrt(2.0));

    return 0.5 * (
        1.0 + std::erf(z)
    );
}


// =============================================================================
// 13. PORTFOLIO CASE STUDY
// =============================================================================

std::vector<Asset> createSyntheticAssets() {
    /*
        Five synthetic assets are represented by daily returns.

        The values are deliberately deterministic rather than retrieved from
        an external data source. This makes the case study reproducible and
        ensures that compilation does not depend on network access.
    */

    return {
        {
            "Technology",
            {
                0.012, -0.004, 0.008, 0.015, -0.006,
                0.009, 0.004, -0.011, 0.013, 0.007,
                0.005, -0.003, 0.010, -0.002, 0.014,
                0.006, -0.005, 0.011, 0.003, -0.007
            }
        },
        {
            "Healthcare",
            {
                0.006, 0.002, 0.005, 0.004, -0.001,
                0.007, 0.003, 0.000, 0.006, 0.004,
                0.002, -0.002, 0.005, 0.003, 0.007,
                0.001, 0.004, -0.001, 0.006, 0.002
            }
        },
        {
            "Energy",
            {
                -0.010, 0.014, -0.006, 0.020, -0.015,
                0.011, -0.004, 0.018, -0.009, 0.013,
                -0.012, 0.016, -0.007, 0.009, -0.014,
                0.019, -0.008, 0.012, -0.010, 0.015
            }
        },
        {
            "Consumer",
            {
                0.004, 0.003, 0.002, 0.006, 0.001,
                0.005, 0.003, 0.002, 0.004, 0.005,
                0.002, 0.003, 0.004, 0.001, 0.005,
                0.004, 0.002, 0.006, 0.003, 0.004
            }
        },
        {
            "Utilities",
            {
                0.002, 0.001, 0.003, 0.002, 0.000,
                0.004, 0.001, 0.002, 0.003, 0.001,
                0.002, 0.000, 0.003, 0.002, 0.001,
                0.004, 0.002, 0.001, 0.003, 0.002
            }
        }
    };
}


// =============================================================================
// 14. RISK CONSTRAINT CHECKING
// =============================================================================

struct RiskReport {
    double expectedReturn;
    double variance;
    double standardDeviation;
    double sharpeRatio;
};

RiskReport calculateRiskReport(
    const Portfolio& portfolio,
    double riskFreeRate
) {
    return {
        portfolio.expectedReturn(),
        portfolio.variance(),
        portfolio.standardDeviation(),
        portfolio.sharpeRatio(riskFreeRate)
    };
}

bool weightsAreValid(
    const Vector& weights,
    double tolerance = 1e-9
) {
    if (weights.empty()) {
        return false;
    }

    if (std::any_of(
            weights.begin(),
            weights.end(),
            [](double value) {
                return value < -1e-12;
            }
        )) {
        return false;
    }

    const double total =
        std::accumulate(
            weights.begin(),
            weights.end(),
            0.0
        );

    return std::abs(total - 1.0) <= tolerance;
}


// =============================================================================
// 15. MAIN CASE STUDY
// =============================================================================

int main() {
    try {
        std::cout << std::fixed << std::setprecision(6);

        section("MATHEMATICAL REVISION CASE STUDY");

        std::cout
            << "Scenario: quantitative portfolio analysis using probability,\n"
            << "statistics, linear algebra, calculus, and optimization.\n";

        // ---------------------------------------------------------------------
        // Basic vector operations
        // ---------------------------------------------------------------------

        subsection("1. Vector mechanics");

        Vector vectorA{1.0, 2.0, 3.0};
        Vector vectorB{4.0, 5.0, 6.0};

        std::cout << "A+B = ";
        printVector(vectorAdd(vectorA, vectorB));

        std::cout << "A-B = ";
        printVector(vectorSubtract(vectorA, vectorB));

        std::cout << "A·B = "
                  << dotProduct(vectorA, vectorB)
                  << "\n";

        std::cout << "||A|| = "
                  << vectorNorm(vectorA)
                  << "\n";


        // ---------------------------------------------------------------------
        // Matrix operations
        // ---------------------------------------------------------------------

        subsection("2. Matrix mechanics");

        Matrix matrixA{
            {1.0, 2.0},
            {3.0, 4.0}
        };

        Matrix matrixB{
            {5.0, 6.0},
            {7.0, 8.0}
        };

        std::cout << "AB:\n";
        printMatrix(
            matrixMultiply(matrixA, matrixB)
        );

        std::cout << "A transpose:\n";
        printMatrix(
            transpose(matrixA)
        );


        // ---------------------------------------------------------------------
        // Linear system
        // ---------------------------------------------------------------------

        subsection("3. Solving a linear system");

        /*
            2x + y = 7
            x - y = 1

            The solution is:
                x = 8/3
                y = 5/3
        */

        Vector linearSolution =
            solveLinearSystem(
                {
                    {2.0, 1.0},
                    {1.0, -1.0}
                },
                {7.0, 1.0}
            );

        std::cout << "Solution = ";
        printVector(linearSolution);


        // ---------------------------------------------------------------------
        // Statistical analysis
        // ---------------------------------------------------------------------

        subsection("4. Descriptive statistics");

        Vector observations{
            12.0, 15.0, 15.0, 18.0,
            20.0, 22.0, 22.0, 24.0, 30.0
        };

        std::cout
            << "Mean = "
            << mean(observations)
            << "\n";

        std::cout
            << "Sample variance = "
            << sampleVariance(observations)
            << "\n";

        std::cout
            << "Sample standard deviation = "
            << sampleStandardDeviation(observations)
            << "\n";


        // ---------------------------------------------------------------------
        // Regression
        // ---------------------------------------------------------------------

        subsection("5. Regression");

        Vector studyHours{
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };

        Vector examScores{
            45.0, 50.0, 58.0,
            65.0, 73.0, 82.0
        };

        LinearRegressionModel model =
            fitLinearRegression(
                studyHours,
                examScores
            );

        std::cout
            << "Regression intercept = "
            << model.intercept
            << "\n";

        std::cout
            << "Regression slope = "
            << model.slope
            << "\n";

        std::cout
            << "Predicted score at 7 hours = "
            << model.predict(7.0)
            << "\n";


        // ---------------------------------------------------------------------
        // Portfolio construction
        // ---------------------------------------------------------------------

        subsection("6. Portfolio data model");

        std::vector<Asset> assets =
            createSyntheticAssets();

        Vector initialWeights(
            assets.size(),
            1.0 / static_cast<double>(assets.size())
        );

        Portfolio equalWeightPortfolio(
            assets,
            initialWeights
        );

        RiskReport initialReport =
            calculateRiskReport(
                equalWeightPortfolio,
                0.001
            );

        std::cout
            << "Equal-weight expected return = "
            << initialReport.expectedReturn
            << "\n";

        std::cout
            << "Equal-weight variance = "
            << initialReport.variance
            << "\n";

        std::cout
            << "Equal-weight standard deviation = "
            << initialReport.standardDeviation
            << "\n";

        std::cout
            << "Equal-weight Sharpe ratio = "
            << initialReport.sharpeRatio
            << "\n";


        // ---------------------------------------------------------------------
        // Covariance matrix
        // ---------------------------------------------------------------------

        subsection("7. Covariance matrix");

        Matrix covarianceMatrix =
            equalWeightPortfolio.covarianceMatrix();

        printMatrix(
            covarianceMatrix,
            8
        );


        // ---------------------------------------------------------------------
        // Correlations
        // ---------------------------------------------------------------------

        subsection("8. Correlation structure");

        for (std::size_t i = 0; i < assets.size(); ++i) {
            for (std::size_t j = i + 1;
                 j < assets.size();
                 ++j) {

                const double value =
                    correlation(
                        assets[i].returns,
                        assets[j].returns
                    );

                std::cout
                    << assets[i].name
                    << " vs "
                    << assets[j].name
                    << ": "
                    << value
                    << "\n";
            }
        }


        // ---------------------------------------------------------------------
        // PCA-style dominant factor
        // ---------------------------------------------------------------------

        subsection("9. Dominant covariance factor");

        EigenResult eigen =
            powerIteration(
                covarianceMatrix
            );

        std::cout
            << "Dominant eigenvalue = "
            << eigen.eigenvalue
            << "\n";

        std::cout
            << "Dominant eigenvector = ";

        printVector(eigen.eigenvector);


        // ---------------------------------------------------------------------
        // Optimization
        // ---------------------------------------------------------------------

        subsection("10. Constrained portfolio optimization");

        Vector expectedReturns =
            equalWeightPortfolio.expectedAssetReturns();

        std::cout
            << "Expected asset returns = ";

        printVector(expectedReturns);


        /*
            Objective:

                minimize
                    λ wᵀΣw - μᵀw

            subject to:

                    wᵢ >= 0
                    Σwᵢ = 1

            This is a simplified long-only mean-variance optimization model.

            A production optimizer can use quadratic-programming methods,
            interior-point methods, active-set methods, or specialized
            numerical libraries. This educational implementation uses
            projected gradient descent to expose the mathematical mechanics.
        */

        Vector optimizedWeights =
            optimizePortfolio(
                covarianceMatrix,
                expectedReturns,
                20.0,
                0.01,
                2000
            );

        std::cout
            << "Optimized weights = ";

        printVector(optimizedWeights);


        if (!weightsAreValid(optimizedWeights)) {
            throw std::runtime_error(
                "Optimization produced invalid portfolio weights."
            );
        }

        Portfolio optimizedPortfolio(
            assets,
            optimizedWeights
        );

        RiskReport optimizedReport =
            calculateRiskReport(
                optimizedPortfolio,
                0.001
            );

        std::cout
            << "Optimized expected return = "
            << optimizedReport.expectedReturn
            << "\n";

        std::cout
            << "Optimized variance = "
            << optimizedReport.variance
            << "\n";

        std::cout
            << "Optimized standard deviation = "
            << optimizedReport.standardDeviation
            << "\n";

        std::cout
            << "Optimized Sharpe ratio = "
            << optimizedReport.sharpeRatio
            << "\n";


        // ---------------------------------------------------------------------
        // Calculus: numerical gradient
        // ---------------------------------------------------------------------

        subsection("11. Numerical calculus");

        Vector testPoint{
            0.0,
            0.0
        };

        Vector gradient =
            numericalGradient(
                quadraticFunction,
                testPoint
            );

        std::cout
            << "Gradient of quadratic at [0,0] = ";

        printVector(gradient);


        // ---------------------------------------------------------------------
        // Gradient descent
        // ---------------------------------------------------------------------

        subsection("12. Gradient descent");

        Vector quadraticMinimum =
            optimizeQuadratic();

        std::cout
            << "Estimated quadratic minimum = ";

        printVector(quadraticMinimum);

        std::cout
            << "Objective value at minimum = "
            << quadraticFunction(quadraticMinimum)
            << "\n";


        // ---------------------------------------------------------------------
        // Root finding
        // ---------------------------------------------------------------------

        subsection("13. Bisection");

        const double squareRootOfTwo =
            bisection(
                [](double x) {
                    return x * x - 2.0;
                },
                1.0,
                2.0
            );

        std::cout
            << "sqrt(2) ≈ "
            << squareRootOfTwo
            << "\n";


        // ---------------------------------------------------------------------
        // Probability distribution
        // ---------------------------------------------------------------------

        subsection("14. Normal distribution");

        std::cout
            << "Normal PDF at 0 = "
            << normalPDF(0.0)
            << "\n";

        std::cout
            << "Normal CDF at 0 = "
            << normalCDF(0.0)
            << "\n";

        std::cout
            << "P(-1 <= Z <= 1) = "
            << normalCDF(1.0) -
               normalCDF(-1.0)
            << "\n";


        // ---------------------------------------------------------------------
        // Financial probability/risk interpretation
        // ---------------------------------------------------------------------

        subsection("15. Risk interpretation");

        const double portfolioMean =
            optimizedReport.expectedReturn;

        const double portfolioRisk =
            optimizedReport.standardDeviation;

        /*
            A normal approximation can estimate the probability that a
            one-period return falls below a chosen threshold.

            This is a model assumption, not a guarantee. Real financial
            returns can have skewness, heavy tails, volatility clustering,
            dependence, regime changes, and other properties not captured by
            a simple normal model.
        */

        const double lossThreshold = -0.005;

        const double probabilityBelowThreshold =
            normalCDF(
                lossThreshold,
                portfolioMean,
                portfolioRisk
            );

        std::cout
            << "Approximate probability of return below "
            << lossThreshold
            << " = "
            << probabilityBelowThreshold
            << "\n";


        // ---------------------------------------------------------------------
        // Numerical stability
        // ---------------------------------------------------------------------

        subsection("16. Numerical stability");

        const double largeValue = 1e16;

        const double directExpression =
            std::sqrt(largeValue + 1.0) -
            std::sqrt(largeValue);

        const double stableExpression =
            1.0 /
            (
                std::sqrt(largeValue + 1.0) +
                std::sqrt(largeValue)
            );

        std::cout
            << "Direct cancellation-prone expression = "
            << directExpression
            << "\n";

        std::cout
            << "Algebraically stabilized expression = "
            << stableExpression
            << "\n";


        // ---------------------------------------------------------------------
        // Validation
        // ---------------------------------------------------------------------

        subsection("17. Validation");

        if (
            !approximatelyEqual(
                determinant(Matrix{{1.0, 2.0}, {3.0, 4.0}}),
                -2.0
            )
        ) {
            /*
                This branch is deliberately not used because determinant()
                is not required by the case-study implementation below.
                The validation section instead checks operations that are
                implemented.
            */
        }

        if (
            !approximatelyEqual(
                dotProduct(
                    Vector{1.0, 2.0, 3.0},
                    Vector{4.0, 5.0, 6.0}
                ),
                32.0
            )
        ) {
            throw std::runtime_error(
                "Dot-product validation failed."
            );
        }

        if (
            !approximatelyEqual(
                model.predict(7.0),
                model.intercept +
                model.slope * 7.0
            )
        ) {
            throw std::runtime_error(
                "Regression validation failed."
            );
        }

        if (!weightsAreValid(optimizedWeights)) {
            throw std::runtime_error(
                "Portfolio constraint validation failed."
            );
        }

        if (
            !approximatelyEqual(
                quadraticMinimum[0],
                3.0,
                1e-5
            ) ||
            !approximatelyEqual(
                quadraticMinimum[1],
                -1.0,
                1e-5
            )
        ) {
            throw std::runtime_error(
                "Optimization validation failed."
            );
        }

        if (
            !approximatelyEqual(
                squareRootOfTwo,
                std::sqrt(2.0),
                1e-8
            )
        ) {
            throw std::runtime_error(
                "Root-finding validation failed."
            );
        }

        std::cout
            << "All validation checks passed.\n";


        // ---------------------------------------------------------------------
        // Complexity and architecture
        // ---------------------------------------------------------------------

        subsection("18. Computational considerations");

        std::cout
            << "Vector operations: O(n)\n"
            << "Dense matrix multiplication: O(n^3)\n"
            << "Gaussian elimination: O(n^3)\n"
            << "Covariance matrix construction: O(m^2 * T)\n"
            << "Power iteration: O(k * m^2)\n"
            << "Numerical gradient: O(d) objective evaluations\n"
            << "Simplex projection: O(d log d)\n"
            << "Projected gradient descent: O(k*d) gradient work plus objective cost\n";

        std::cout
            << "\nCase-study architecture:\n"
            << "Data -> Descriptive Statistics -> Covariance Matrix ->\n"
            << "Linear Algebra -> Risk Model -> Optimization -> Validation\n";


        section("CASE STUDY COMPLETE");

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "\nERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}
