/*
 * Eigenvalues & Eigenvectors
 * ==========================
 *
 * Industry-style C++17 case study:
 *
 * A small spectral analytics engine for a sensor network.
 *
 * The system:
 *   1. Stores observations from multiple sensors.
 *   2. Computes the covariance matrix.
 *   3. Finds the dominant eigenpair using power iteration.
 *   4. Uses eigenvectors as principal directions.
 *   5. Projects observations onto the dominant component.
 *   6. Computes reconstruction error.
 *   7. Builds a sensor-network graph.
 *   8. Computes its graph Laplacian.
 *   9. Demonstrates a stationary distribution for a transition matrix.
 *
 * The implementation intentionally uses the C++ standard library only.
 *
 * Compile:
 *   g++ -std=c++17 -O2 eigenvalues_eigenvectors.cpp -o spectral_engine
 *
 * Run:
 *   ./spectral_engine
 */

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

constexpr double EPSILON = 1e-10;

// -----------------------------------------------------------------------------
// Basic validation
// -----------------------------------------------------------------------------

void validateRectangular(const Matrix& matrix) {
    if (matrix.empty()) {
        throw std::invalid_argument("Matrix cannot be empty.");
    }

    if (matrix.front().empty()) {
        throw std::invalid_argument("Matrix must contain columns.");
    }

    const std::size_t columns = matrix.front().size();

    for (const auto& row : matrix) {
        if (row.size() != columns) {
            throw std::invalid_argument("Matrix must be rectangular.");
        }
    }
}

std::pair<std::size_t, std::size_t> shape(const Matrix& matrix) {
    validateRectangular(matrix);
    return {matrix.size(), matrix.front().size()};
}

void requireSquare(const Matrix& matrix) {
    const auto [rows, columns] = shape(matrix);

    if (rows != columns) {
        throw std::invalid_argument("Matrix must be square.");
    }
}

// -----------------------------------------------------------------------------
// Printing
// -----------------------------------------------------------------------------

void printVector(
    const Vector& vector,
    const std::string& name
) {
    std::cout << name << " = [";

    for (std::size_t i = 0; i < vector.size(); ++i) {
        std::cout << std::fixed
                  << std::setprecision(6)
                  << vector[i];

        if (i + 1 < vector.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}

void printMatrix(
    const Matrix& matrix,
    const std::string& name
) {
    std::cout << name << " =\n";

    for (const auto& row : matrix) {
        std::cout << "  [";

        for (std::size_t j = 0; j < row.size(); ++j) {
            std::cout << std::fixed
                      << std::setprecision(6)
                      << row[j];

            if (j + 1 < row.size()) {
                std::cout << ", ";
            }
        }

        std::cout << "]\n";
    }
}

// -----------------------------------------------------------------------------
// Vector operations
// -----------------------------------------------------------------------------

double dot(
    const Vector& a,
    const Vector& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have equal dimensions."
        );
    }

    return std::inner_product(
        a.begin(),
        a.end(),
        b.begin(),
        0.0
    );
}

double vectorNorm(const Vector& vector) {
    return std::sqrt(dot(vector, vector));
}

Vector normalize(Vector vector) {
    const double length = vectorNorm(vector);

    if (length < EPSILON) {
        throw std::invalid_argument(
            "Cannot normalize a zero vector."
        );
    }

    for (double& value : vector) {
        value /= length;
    }

    return vector;
}

Vector subtract(
    const Vector& a,
    const Vector& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have equal dimensions."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

Vector add(
    const Vector& a,
    const Vector& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have equal dimensions."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Vector scalarMultiply(
    double scalar,
    const Vector& vector
) {
    Vector result = vector;

    for (double& value : result) {
        value *= scalar;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Matrix operations
// -----------------------------------------------------------------------------

Matrix identityMatrix(std::size_t size) {
    Matrix identity(
        size,
        Vector(size, 0.0)
    );

    for (std::size_t i = 0; i < size; ++i) {
        identity[i][i] = 1.0;
    }

    return identity;
}

Matrix transpose(const Matrix& matrix) {
    const auto [rows, columns] = shape(matrix);

    Matrix result(
        columns,
        Vector(rows, 0.0)
    );

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

Vector matrixVectorMultiply(
    const Matrix& matrix,
    const Vector& vector
) {
    const auto [rows, columns] = shape(matrix);

    if (columns != vector.size()) {
        throw std::invalid_argument(
            "Matrix columns must equal vector dimension."
        );
    }

    Vector result(rows, 0.0);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

Matrix matrixMultiply(
    const Matrix& a,
    const Matrix& b
) {
    const auto [rowsA, columnsA] = shape(a);
    const auto [rowsB, columnsB] = shape(b);

    if (columnsA != rowsB) {
        throw std::invalid_argument(
            "Matrix dimensions cannot be multiplied."
        );
    }

    Matrix result(
        rowsA,
        Vector(columnsB, 0.0)
    );

    for (std::size_t i = 0; i < rowsA; ++i) {
        for (std::size_t k = 0; k < columnsA; ++k) {
            for (std::size_t j = 0; j < columnsB; ++j) {
                result[i][j] +=
                    a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

Matrix subtractMatrices(
    const Matrix& a,
    const Matrix& b
) {
    const auto [rowsA, columnsA] = shape(a);
    const auto [rowsB, columnsB] = shape(b);

    if (rowsA != rowsB || columnsA != columnsB) {
        throw std::invalid_argument(
            "Matrices must have equal dimensions."
        );
    }

    Matrix result = a;

    for (std::size_t i = 0; i < rowsA; ++i) {
        for (std::size_t j = 0; j < columnsA; ++j) {
            result[i][j] -= b[i][j];
        }
    }

    return result;
}

Matrix outerProduct(
    const Vector& a,
    const Vector& b
) {
    Matrix result(
        a.size(),
        Vector(b.size(), 0.0)
    );

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// Determinant and characteristic equation
// -----------------------------------------------------------------------------

double determinant2x2(const Matrix& matrix) {
    const auto [rows, columns] = shape(matrix);

    if (rows != 2 || columns != 2) {
        throw std::invalid_argument(
            "This determinant function expects a 2x2 matrix."
        );
    }

    return
        matrix[0][0] * matrix[1][1]
        - matrix[0][1] * matrix[1][0];
}

double trace(const Matrix& matrix) {
    requireSquare(matrix);

    double result = 0.0;

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        result += matrix[i][i];
    }

    return result;
}

std::pair<double, double> eigenvalues2x2(
    const Matrix& matrix
) {
    /*
     * For a 2x2 matrix:
     *
     *     A = [a b]
     *         [c d]
     *
     * characteristic equation:
     *
     *     lambda^2
     *     - (a+d) lambda
     *     + (ad-bc)
     *     = 0
     *
     * The implementation handles real roots. A production general-purpose
     * eigensolver would need a complex-number path for negative discriminants.
     */
    const double tr = trace(matrix);
    const double det = determinant2x2(matrix);

    const double discriminant =
        tr * tr - 4.0 * det;

    if (discriminant < -EPSILON) {
        throw std::runtime_error(
            "Matrix has complex eigenvalues; "
            "this real-valued case study does not represent them."
        );
    }

    const double adjustedDiscriminant =
        std::max(0.0, discriminant);

    const double root =
        std::sqrt(adjustedDiscriminant);

    return {
        (tr + root) / 2.0,
        (tr - root) / 2.0
    };
}

// -----------------------------------------------------------------------------
// Eigenvectors and eigenpair residuals
// -----------------------------------------------------------------------------

Vector eigenvector2x2(
    const Matrix& matrix,
    double eigenvalue
) {
    /*
     * We solve:
     *
     *     (A - lambda I)v = 0.
     *
     * For a non-zero row [a,b], [-b,a] is perpendicular to that row
     * and therefore lies in its null direction when the determinant
     * is approximately zero.
     */
    Matrix shifted = matrix;

    shifted[0][0] -= eigenvalue;
    shifted[1][1] -= eigenvalue;

    Vector row1 = shifted[0];
    Vector row2 = shifted[1];

    Vector candidate1{
        -row1[1],
        row1[0]
    };

    Vector candidate2{
        -row2[1],
        row2[0]
    };

    Vector candidate =
        vectorNorm(candidate1) >
        vectorNorm(candidate2)
            ? candidate1
            : candidate2;

    if (vectorNorm(candidate) < EPSILON) {
        throw std::runtime_error(
            "Could not construct a non-zero eigenvector."
        );
    }

    return normalize(candidate);
}

double eigenpairResidual(
    const Matrix& matrix,
    double eigenvalue,
    const Vector& eigenvector
) {
    const Vector left =
        matrixVectorMultiply(matrix, eigenvector);

    const Vector right =
        scalarMultiply(eigenvalue, eigenvector);

    return vectorNorm(subtract(left, right));
}

// -----------------------------------------------------------------------------
// Power iteration
// -----------------------------------------------------------------------------

struct PowerIterationResult {
    double eigenvalue{};
    Vector eigenvector;
    std::size_t iterations{};
    bool converged{};
};

double rayleighQuotient(
    const Matrix& matrix,
    const Vector& vector
) {
    const double denominator =
        dot(vector, vector);

    if (std::abs(denominator) < EPSILON) {
        throw std::invalid_argument(
            "Rayleigh quotient requires a non-zero vector."
        );
    }

    const Vector transformed =
        matrixVectorMultiply(matrix, vector);

    return dot(vector, transformed) /
           denominator;
}

PowerIterationResult powerIteration(
    const Matrix& matrix,
    Vector vector,
    std::size_t maxIterations = 5000,
    double tolerance = 1e-10
) {
    requireSquare(matrix);

    if (vector.size() != matrix.size()) {
        throw std::invalid_argument(
            "Initial vector has the wrong dimension."
        );
    }

    vector = normalize(std::move(vector));

    double previousEigenvalue =
        std::numeric_limits<double>::quiet_NaN();

    for (std::size_t iteration = 1;
         iteration <= maxIterations;
         ++iteration) {

        Vector transformed =
            matrixVectorMultiply(matrix, vector);

        if (vectorNorm(transformed) < EPSILON) {
            throw std::runtime_error(
                "Power iteration reached the zero vector."
            );
        }

        vector = normalize(std::move(transformed));

        const double eigenvalue =
            rayleighQuotient(matrix, vector);

        if (
            !std::isnan(previousEigenvalue) &&
            std::abs(
                eigenvalue - previousEigenvalue
            ) < tolerance
        ) {
            return {
                eigenvalue,
                vector,
                iteration,
                true
            };
        }

        previousEigenvalue = eigenvalue;
    }

    return {
        rayleighQuotient(matrix, vector),
        vector,
        maxIterations,
        false
    };
}

// -----------------------------------------------------------------------------
// Covariance matrix and PCA
// -----------------------------------------------------------------------------

Vector columnMeans(const Matrix& data) {
    const auto [rows, columns] = shape(data);

    Vector means(
        columns,
        0.0
    );

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0;
             column < columns;
             ++column) {
            means[column] +=
                data[row][column];
        }
    }

    for (double& mean : means) {
        mean /= static_cast<double>(rows);
    }

    return means;
}

Matrix centerData(
    const Matrix& data,
    const Vector& means
) {
    const auto [rows, columns] = shape(data);

    if (means.size() != columns) {
        throw std::invalid_argument(
            "Mean vector has the wrong dimension."
        );
    }

    Matrix centered = data;

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            centered[i][j] -= means[j];
        }
    }

    return centered;
}

Matrix covarianceMatrix(
    const Matrix& data
) {
    const auto [rows, columns] = shape(data);

    if (rows < 2) {
        throw std::invalid_argument(
            "At least two observations are required."
        );
    }

    const Vector means =
        columnMeans(data);

    const Matrix centered =
        centerData(data, means);

    Matrix covariance(
        columns,
        Vector(columns, 0.0)
    );

    /*
     * Cov(X) = X_centered^T X_centered / (n - 1).
     *
     * The covariance matrix is symmetric and positive semidefinite
     * for real-valued data.
     */
    for (std::size_t i = 0; i < columns; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            for (std::size_t observation = 0;
                 observation < rows;
                 ++observation) {

                covariance[i][j] +=
                    centered[observation][i] *
                    centered[observation][j];
            }

            covariance[i][j] /=
                static_cast<double>(rows - 1);
        }
    }

    return covariance;
}

struct PCAResult {
    Vector means;
    Matrix covariance;
    double dominantEigenvalue{};
    Vector principalComponent;
    Vector scores;
    Matrix reconstruction;
    double explainedVarianceRatio{};
    double meanSquaredError{};
};

PCAResult runPCA(
    const Matrix& data
) {
    const Vector means =
        columnMeans(data);

    const Matrix covariance =
        covarianceMatrix(data);

    /*
     * For a two-feature dataset, the covariance matrix is 2x2.
     * Power iteration estimates its largest eigenvalue and corresponding
     * eigenvector. That eigenvector is the first principal component.
     */
    PowerIterationResult power =
        powerIteration(
            covariance,
            Vector(covariance.size(), 1.0)
        );

    const double totalVariance =
        trace(covariance);

    const double explainedVarianceRatio =
        power.eigenvalue /
        totalVariance;

    const Matrix centered =
        centerData(data, means);

    Vector scores;

    scores.reserve(data.size());

    for (const auto& observation : centered) {
        scores.push_back(
            dot(
                observation,
                power.eigenvector
            )
        );
    }

    /*
     * One-component reconstruction:
     *
     *     x_hat = mean + score * principal_component.
     */
    Matrix reconstruction;

    reconstruction.reserve(data.size());

    for (double score : scores) {
        Vector reconstructed =
            means;

        for (std::size_t j = 0;
             j < reconstructed.size();
             ++j) {

            reconstructed[j] +=
                score *
                power.eigenvector[j];
        }

        reconstruction.push_back(
            std::move(reconstructed)
        );
    }

    double squaredError = 0.0;

    for (std::size_t i = 0;
         i < data.size();
         ++i) {

        for (std::size_t j = 0;
             j < data[i].size();
             ++j) {

            const double difference =
                data[i][j] -
                reconstruction[i][j];

            squaredError +=
                difference * difference;
        }
    }

    const double meanSquaredError =
        squaredError /
        static_cast<double>(
            data.size() *
            data.front().size()
        );

    return {
        means,
        covariance,
        power.eigenvalue,
        power.eigenvector,
        scores,
        reconstruction,
        explainedVarianceRatio,
        meanSquaredError
    };
}

// -----------------------------------------------------------------------------
// Sensor network graph
// -----------------------------------------------------------------------------

Matrix graphLaplacian(
    const Matrix& adjacency
) {
    requireSquare(adjacency);

    const std::size_t n =
        adjacency.size();

    Vector degrees(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        degrees[i] =
            std::accumulate(
                adjacency[i].begin(),
                adjacency[i].end(),
                0.0
            );
    }

    Matrix L(
        n,
        Vector(n, 0.0)
    );

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            L[i][j] =
                (i == j ? degrees[i] : 0.0)
                - adjacency[i][j];
        }
    }

    return L;
}

// -----------------------------------------------------------------------------
// Markov-chain stationary distribution
// -----------------------------------------------------------------------------

Vector stationaryDistribution(
    const Matrix& transition,
    std::size_t maxIterations = 10000,
    double tolerance = 1e-12
) {
    requireSquare(transition);

    const std::size_t n =
        transition.size();

    for (const auto& row : transition) {
        double rowSum = 0.0;

        for (double probability : row) {
            if (probability < -EPSILON) {
                throw std::invalid_argument(
                    "Transition probabilities cannot be negative."
                );
            }

            rowSum += probability;
        }

        if (std::abs(rowSum - 1.0) > EPSILON) {
            throw std::invalid_argument(
                "Each transition row must sum to one."
            );
        }
    }

    Vector distribution(
        n,
        1.0 / static_cast<double>(n)
    );

    for (std::size_t iteration = 0;
         iteration < maxIterations;
         ++iteration) {

        Vector next(n, 0.0);

        /*
         * Row-vector convention:
         *
         *     pi_next = pi P.
         */
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                next[j] +=
                    distribution[i] *
                    transition[i][j];
            }
        }

        double difference = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            difference +=
                std::abs(
                    next[i] -
                    distribution[i]
                );
        }

        distribution =
            std::move(next);

        if (difference < tolerance) {
            return distribution;
        }
    }

    return distribution;
}

// -----------------------------------------------------------------------------
// Sensor-network case study
// -----------------------------------------------------------------------------

class SensorAnalyticsSystem {
private:
    Matrix observations;
    Matrix adjacency;

public:
    SensorAnalyticsSystem(
        Matrix sensorObservations,
        Matrix sensorAdjacency
    )
        : observations(
              std::move(sensorObservations)
          ),
          adjacency(
              std::move(sensorAdjacency)
          ) {

        validateRectangular(observations);
        requireSquare(adjacency);

        if (observations.empty()) {
            throw std::invalid_argument(
                "Sensor observations cannot be empty."
            );
        }
    }

    void runPrincipalComponentAnalysis() const {
        std::cout << "\n";
        std::cout << std::string(78, '=') << "\n";
        std::cout << "SENSOR PCA ANALYSIS\n";
        std::cout << std::string(78, '=') << "\n";

        printMatrix(
            observations,
            "Sensor observations"
        );

        PCAResult result =
            runPCA(observations);

        printVector(
            result.means,
            "Feature means"
        );

        printMatrix(
            result.covariance,
            "Covariance matrix"
        );

        std::cout
            << "Dominant eigenvalue = "
            << std::fixed
            << std::setprecision(8)
            << result.dominantEigenvalue
            << "\n";

        printVector(
            result.principalComponent,
            "First principal component"
        );

        std::cout
            << "Explained variance ratio = "
            << std::fixed
            << std::setprecision(4)
            << result.explainedVarianceRatio * 100.0
            << "%\n";

        printVector(
            result.scores,
            "Compressed one-dimensional scores"
        );

        printMatrix(
            result.reconstruction,
            "Reconstructed observations"
        );

        std::cout
            << "Mean squared reconstruction error = "
            << std::fixed
            << std::setprecision(8)
            << result.meanSquaredError
            << "\n";

        /*
         * The principal component is not merely a numerical artifact.
         * It is a direction in feature space that maximizes projected
         * variance. For correlated sensor measurements, it can represent
         * a dominant shared pattern among the sensors.
         */
        const double residual =
            eigenpairResidual(
                result.covariance,
                result.dominantEigenvalue,
                result.principalComponent
            );

        std::cout
            << "Eigenpair residual = "
            << std::scientific
            << residual
            << "\n";
    }

    void analyzeNetworkStructure() const {
        std::cout << "\n";
        std::cout << std::string(78, '=') << "\n";
        std::cout << "SENSOR NETWORK GRAPH ANALYSIS\n";
        std::cout << std::string(78, '=') << "\n";

        printMatrix(
            adjacency,
            "Sensor adjacency matrix"
        );

        Matrix laplacian =
            graphLaplacian(adjacency);

        printMatrix(
            laplacian,
            "Graph Laplacian"
        );

        Vector constant(
            adjacency.size(),
            1.0
        );

        Vector transformed =
            matrixVectorMultiply(
                laplacian,
                constant
            );

        printVector(
            transformed,
            "L * 1"
        );

        std::cout
            << "The constant vector belongs to the zero-eigenvalue "
            << "direction of the Laplacian.\n";

        /*
         * For an undirected graph, the multiplicity of the zero eigenvalue
         * equals the number of connected components. Therefore spectral
         * graph analysis can expose structural properties of the network.
         */
    }

    void analyzeTransitionBehavior() const {
        std::cout << "\n";
        std::cout << std::string(78, '=') << "\n";
        std::cout << "NETWORK STATE TRANSITION ANALYSIS\n";
        std::cout << std::string(78, '=') << "\n";

        /*
         * A row-stochastic transition matrix models movement among
         * network states. Its stationary distribution corresponds to
         * eigenvalue 1 under the row-vector convention.
         */
        Matrix transition{
            {0.90, 0.10, 0.00},
            {0.20, 0.60, 0.20},
            {0.00, 0.30, 0.70}
        };

        printMatrix(
            transition,
            "Transition matrix"
        );

        Vector stationary =
            stationaryDistribution(
                transition
            );

        printVector(
            stationary,
            "Stationary distribution"
        );

        Vector afterTransition =
            matrixVectorMultiply(
                transpose(transition),
                stationary
            );

        /*
         * The transpose is used here because matrixVectorMultiply assumes
         * column-vector multiplication. The stationary calculation itself
         * used the row-vector equation pi P = pi.
         */
        printVector(
            afterTransition,
            "Equivalent column-vector transition"
        );
    }
};

// -----------------------------------------------------------------------------
// Direct mathematical demonstrations
// -----------------------------------------------------------------------------

void demonstrateCharacteristicEquation() {
    std::cout << "\n";
    std::cout << std::string(78, '=') << "\n";
    std::cout << "CHARACTERISTIC EQUATION\n";
    std::cout << std::string(78, '=') << "\n";

    Matrix A{
        {4.0, 1.0},
        {2.0, 3.0}
    };

    printMatrix(A, "A");

    const double tr = trace(A);
    const double det = determinant2x2(A);

    std::cout
        << "trace(A) = "
        << tr
        << "\n";

    std::cout
        << "det(A) = "
        << det
        << "\n";

    std::cout
        << "Characteristic equation: "
        << "lambda^2 - "
        << tr
        << " lambda + "
        << det
        << " = 0\n";

    const auto [lambda1, lambda2] =
        eigenvalues2x2(A);

    std::cout
        << "lambda_1 = "
        << lambda1
        << "\n";

    std::cout
        << "lambda_2 = "
        << lambda2
        << "\n";

    Vector v1 =
        eigenvector2x2(
            A,
            lambda1
        );

    Vector v2 =
        eigenvector2x2(
            A,
            lambda2
        );

    printVector(v1, "v1");
    printVector(v2, "v2");

    std::cout
        << "Residual for first eigenpair = "
        << std::scientific
        << eigenpairResidual(
               A,
               lambda1,
               v1
           )
        << "\n";

    std::cout
        << "Residual for second eigenpair = "
        << std::scientific
        << eigenpairResidual(
               A,
               lambda2,
               v2
           )
        << "\n";
}

void demonstratePowerIteration() {
    std::cout << "\n";
    std::cout << std::string(78, '=') << "\n";
    std::cout << "POWER ITERATION\n";
    std::cout << std::string(78, '=') << "\n";

    Matrix A{
        {5.0, 1.0},
        {1.0, 3.0}
    };

    PowerIterationResult result =
        powerIteration(
            A,
            {1.0, 1.0}
        );

    printMatrix(A, "A");

    std::cout
        << "Dominant eigenvalue = "
        << std::fixed
        << std::setprecision(10)
        << result.eigenvalue
        << "\n";

    printVector(
        result.eigenvector,
        "Dominant eigenvector"
    );

    std::cout
        << "Iterations = "
        << result.iterations
        << "\n";

    std::cout
        << "Converged = "
        << std::boolalpha
        << result.converged
        << "\n";

    std::cout
        << "Residual = "
        << std::scientific
        << eigenpairResidual(
               A,
               result.eigenvalue,
               result.eigenvector
           )
        << "\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout
            << "EIGENVALUES & EIGENVECTORS\n"
            << "Spectral analytics case study\n";

        demonstrateCharacteristicEquation();
        demonstratePowerIteration();

        /*
         * Observations represent repeated measurements from a hypothetical
         * sensor network. The two features can represent correlated
         * measurements such as temperature and pressure.
         */
        Matrix observations{
            {2.0, 1.1},
            {3.0, 1.9},
            {4.0, 3.0},
            {5.0, 3.9},
            {6.0, 5.1},
            {7.0, 5.8},
            {8.0, 7.2},
            {9.0, 8.0}
        };

        /*
         * Undirected sensor topology:
         *
         * Sensor 0 -- Sensor 1 -- Sensor 2 -- Sensor 3
         */
        Matrix adjacency{
            {0.0, 1.0, 0.0, 0.0},
            {1.0, 0.0, 1.0, 0.0},
            {0.0, 1.0, 0.0, 1.0},
            {0.0, 0.0, 1.0, 0.0}
        };

        SensorAnalyticsSystem system(
            observations,
            adjacency
        );

        system.runPrincipalComponentAnalysis();
        system.analyzeNetworkStructure();
        system.analyzeTransitionBehavior();

        std::cout << "\n";
        std::cout << std::string(78, '=') << "\n";
        std::cout << "CASE STUDY COMPLETE\n";
        std::cout << std::string(78, '=') << "\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Execution error: "
            << error.what()
            << "\n";

        return 1;
    }
}
