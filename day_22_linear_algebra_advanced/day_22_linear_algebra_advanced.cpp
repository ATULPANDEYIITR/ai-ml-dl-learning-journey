/*
 * Advanced Linear Algebra: Technical Case Study
 *
 * Scenario:
 *     A scientific analytics engine receives measurements from multiple
 *     sensors. It must:
 *
 *     1. Represent sensor measurements as vectors.
 *     2. Transform measurements using matrices.
 *     3. Solve calibration equations.
 *     4. Estimate parameters using least squares.
 *     5. Analyze correlations through covariance matrices.
 *     6. Extract dominant directions using eigenvalue iteration.
 *     7. Perform dimensionality reduction using PCA.
 *     8. Represent a sensor network using a graph Laplacian.
 *     9. Solve a symmetric positive-definite system using Conjugate Gradient.
 *
 * Build:
 *     g++ -std=c++17 -O2 advanced_linear_algebra.cpp -o linear_algebra
 *
 * Run:
 *     ./linear_algebra
 *
 * The implementation intentionally uses only the C++17 standard library.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;

constexpr double EPSILON = 1e-10;

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions do not match.");
    }

    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double norm(const Vector& vector) {
    return std::sqrt(dot(vector, vector));
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions do not match.");
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions do not match.");
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

Vector scale(double scalar, const Vector& vector) {
    Vector result = vector;

    for (double& value : result) {
        value *= scalar;
    }

    return result;
}

Vector normalize(const Vector& vector) {
    const double length = norm(vector);

    if (length < EPSILON) {
        throw std::invalid_argument("Cannot normalize zero vector.");
    }

    return scale(1.0 / length, vector);
}

void printVector(const std::string& name, const Vector& vector) {
    std::cout << name << " = [";

    for (std::size_t i = 0; i < vector.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6)
                  << vector[i];

        if (i + 1 < vector.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}

class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> values_;

public:
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
        : rows_(rows),
          cols_(cols),
          values_(rows * cols, value) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }
    }

    Matrix(std::initializer_list<std::initializer_list<double>> values) {
        if (values.size() == 0) {
            throw std::invalid_argument("Matrix cannot be empty.");
        }

        rows_ = values.size();
        cols_ = values.begin()->size();

        if (cols_ == 0) {
            throw std::invalid_argument("Matrix cannot have zero columns.");
        }

        values_.reserve(rows_ * cols_);

        for (const auto& row : values) {
            if (row.size() != cols_) {
                throw std::invalid_argument("Matrix must be rectangular.");
            }

            values_.insert(values_.end(), row.begin(), row.end());
        }
    }

    static Matrix identity(std::size_t n) {
        Matrix result(n, n);

        for (std::size_t i = 0; i < n; ++i) {
            result(i, i) = 1.0;
        }

        return result;
    }

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const {
        return cols_;
    }

    double& operator()(std::size_t row, std::size_t col) {
        return values_[row * cols_ + col];
    }

    double operator()(std::size_t row, std::size_t col) const {
        return values_[row * cols_ + col];
    }

    Matrix transpose() const {
        Matrix result(cols_, rows_);

        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions do not match.");
        }

        Matrix result(rows_, cols_);

        for (std::size_t i = 0; i < values_.size(); ++i) {
            result.values_[i] = values_[i] + other.values_[i];
        }

        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument(
                "Matrix multiplication dimensions are incompatible."
            );
        }

        Matrix result(rows_, other.cols_);

        /*
         * Loop order i-k-j improves locality because the same value
         * A[i][k] is reused across the output row.
         *
         * Complexity:
         *     O(rows * shared_dimension * output_columns)
         */
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t k = 0; k < cols_; ++k) {
                const double aik = (*this)(i, k);

                for (std::size_t j = 0; j < other.cols_; ++j) {
                    result(i, j) += aik * other(k, j);
                }
            }
        }

        return result;
    }

    Vector multiply(const Vector& vector) const {
        if (cols_ != vector.size()) {
            throw std::invalid_argument(
                "Matrix/vector dimensions are incompatible."
            );
        }

        Vector result(rows_, 0.0);

        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result[i] += (*this)(i, j) * vector[j];
            }
        }

        return result;
    }

    double trace() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Trace requires square matrix.");
        }

        double result = 0.0;

        for (std::size_t i = 0; i < rows_; ++i) {
            result += (*this)(i, i);
        }

        return result;
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << "\n";
        }

        for (std::size_t i = 0; i < rows_; ++i) {
            std::cout << "[";

            for (std::size_t j = 0; j < cols_; ++j) {
                std::cout << std::setw(12)
                          << std::fixed
                          << std::setprecision(6)
                          << (*this)(i, j);

                if (j + 1 < cols_) {
                    std::cout << " ";
                }
            }

            std::cout << "]\n";
        }
    }
};

Matrix rref(Matrix matrix, std::vector<std::size_t>& pivots) {
    pivots.clear();

    std::size_t pivotRow = 0;

    for (std::size_t column = 0;
         column < matrix.cols() && pivotRow < matrix.rows();
         ++column) {

        std::size_t bestRow = pivotRow;

        for (std::size_t row = pivotRow + 1;
             row < matrix.rows();
             ++row) {

            if (std::abs(matrix(row, column)) >
                std::abs(matrix(bestRow, column))) {
                bestRow = row;
            }
        }

        if (std::abs(matrix(bestRow, column)) < EPSILON) {
            continue;
        }

        if (bestRow != pivotRow) {
            for (std::size_t j = 0; j < matrix.cols(); ++j) {
                std::swap(
                    matrix(bestRow, j),
                    matrix(pivotRow, j)
                );
            }
        }

        const double pivot = matrix(pivotRow, column);

        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            matrix(pivotRow, j) /= pivot;
        }

        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            if (row == pivotRow) {
                continue;
            }

            const double factor = matrix(row, column);

            if (std::abs(factor) < EPSILON) {
                continue;
            }

            for (std::size_t j = 0; j < matrix.cols(); ++j) {
                matrix(row, j) -= factor * matrix(pivotRow, j);
            }
        }

        pivots.push_back(column);
        ++pivotRow;
    }

    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            if (std::abs(matrix(i, j)) < EPSILON) {
                matrix(i, j) = 0.0;
            }
        }
    }

    return matrix;
}

std::size_t rank(const Matrix& matrix) {
    std::vector<std::size_t> pivots;
    rref(matrix, pivots);
    return pivots.size();
}

double determinant(Matrix matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Determinant requires square matrix.");
    }

    const std::size_t n = matrix.rows();
    double determinantValue = 1.0;
    double sign = 1.0;

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivotRow = column;

        for (std::size_t row = column + 1; row < n; ++row) {
            if (std::abs(matrix(row, column)) >
                std::abs(matrix(pivotRow, column))) {
                pivotRow = row;
            }
        }

        if (std::abs(matrix(pivotRow, column)) < EPSILON) {
            return 0.0;
        }

        if (pivotRow != column) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(
                    matrix(pivotRow, j),
                    matrix(column, j)
                );
            }

            sign *= -1.0;
        }

        const double pivot = matrix(column, column);
        determinantValue *= pivot;

        for (std::size_t row = column + 1; row < n; ++row) {
            const double factor = matrix(row, column) / pivot;

            for (std::size_t j = column; j < n; ++j) {
                matrix(row, j) -= factor * matrix(column, j);
            }
        }
    }

    return sign * determinantValue;
}

Matrix inverse(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Inverse requires square matrix.");
    }

    const std::size_t n = matrix.rows();

    Matrix augmented(n, 2 * n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            augmented(i, j) = matrix(i, j);
        }

        for (std::size_t j = 0; j < n; ++j) {
            augmented(i, n + j) = i == j ? 1.0 : 0.0;
        }
    }

    std::vector<std::size_t> pivots;
    Matrix reduced = rref(augmented, pivots);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double expected = i == j ? 1.0 : 0.0;

            if (std::abs(reduced(i, j) - expected) > 1e-8) {
                throw std::runtime_error("Matrix is singular.");
            }
        }
    }

    Matrix result(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result(i, j) = reduced(i, n + j);
        }
    }

    return result;
}

Vector solve(const Matrix& A, const Vector& b) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("Right-hand side dimension mismatch.");
    }

    Matrix augmented(A.rows(), A.cols() + 1);

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            augmented(i, j) = A(i, j);
        }

        augmented(i, A.cols()) = b[i];
    }

    std::vector<std::size_t> pivots;
    Matrix reduced = rref(augmented, pivots);

    for (std::size_t i = 0; i < reduced.rows(); ++i) {
        bool coefficientsZero = true;

        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (std::abs(reduced(i, j)) > EPSILON) {
                coefficientsZero = false;
                break;
            }
        }

        if (coefficientsZero &&
            std::abs(reduced(i, A.cols())) > EPSILON) {
            throw std::runtime_error("System is inconsistent.");
        }
    }

    if (rank(A) < A.cols()) {
        throw std::runtime_error(
            "System does not have a unique solution."
        );
    }

    Vector result(A.cols());

    for (std::size_t i = 0; i < A.cols(); ++i) {
        result[i] = reduced(i, A.cols());
    }

    return result;
}

std::vector<Vector> gramSchmidt(const std::vector<Vector>& vectors) {
    std::vector<Vector> basis;

    for (const Vector& original : vectors) {
        Vector residual = original;

        for (const Vector& q : basis) {
            residual = subtract(
                residual,
                scale(dot(residual, q), q)
            );
        }

        const double length = norm(residual);

        if (length > EPSILON) {
            basis.push_back(scale(1.0 / length, residual));
        }
    }

    return basis;
}

std::pair<Matrix, Matrix> qrDecomposition(const Matrix& A) {
    if (A.rows() < A.cols()) {
        throw std::invalid_argument(
            "QR implementation expects rows >= columns."
        );
    }

    std::vector<Vector> columns;

    for (std::size_t j = 0; j < A.cols(); ++j) {
        Vector column(A.rows());

        for (std::size_t i = 0; i < A.rows(); ++i) {
            column[i] = A(i, j);
        }

        columns.push_back(column);
    }

    std::vector<Vector> qColumns = gramSchmidt(columns);

    if (qColumns.size() != A.cols()) {
        throw std::runtime_error(
            "QR decomposition encountered dependent columns."
        );
    }

    Matrix Q(A.rows(), A.cols());

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            Q(i, j) = qColumns[j][i];
        }
    }

    Matrix R = Q.transpose() * A;

    return {Q, R};
}

Vector leastSquares(const Matrix& A, const Vector& b) {
    /*
     * Normal equations:
     *
     *     (A^T A)x = A^T b
     *
     * This is mathematically correct but can be less stable than QR/SVD
     * because cond(A^T A) is approximately cond(A)^2.
     */
    Matrix At = A.transpose();
    Matrix normalMatrix = At * A;
    Vector normalVector = At.multiply(b);

    return solve(normalMatrix, normalVector);
}

bool isSymmetric(const Matrix& A, double tolerance = 1e-9) {
    if (A.rows() != A.cols()) {
        return false;
    }

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (std::abs(A(i, j) - A(j, i)) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

double quadraticForm(const Matrix& A, const Vector& x) {
    return dot(x, A.multiply(x));
}

double rayleighQuotient(const Matrix& A, const Vector& x) {
    const double denominator = dot(x, x);

    if (denominator < EPSILON) {
        throw std::invalid_argument(
            "Rayleigh quotient requires non-zero vector."
        );
    }

    return quadraticForm(A, x) / denominator;
}

std::pair<double, Vector> powerIteration(
    const Matrix& A,
    std::size_t maxIterations = 1000,
    double tolerance = 1e-10
) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "Power iteration requires square matrix."
        );
    }

    Vector x(A.rows());

    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i + 1);
    }

    x = normalize(x);

    double eigenvalue = 0.0;

    for (std::size_t iteration = 0;
         iteration < maxIterations;
         ++iteration) {

        Vector next = A.multiply(x);

        if (norm(next) < EPSILON) {
            throw std::runtime_error(
                "Power iteration reached zero vector."
            );
        }

        next = normalize(next);

        const double nextEigenvalue =
            rayleighQuotient(A, next);

        if (norm(subtract(next, x)) < tolerance) {
            return {nextEigenvalue, next};
        }

        x = next;
        eigenvalue = nextEigenvalue;
    }

    return {eigenvalue, x};
}

Matrix covarianceMatrix(
    const std::vector<Vector>& observations
) {
    if (observations.size() < 2) {
        throw std::invalid_argument(
            "At least two observations are required."
        );
    }

    const std::size_t dimensions = observations[0].size();

    if (dimensions == 0) {
        throw std::invalid_argument("Observations cannot be empty.");
    }

    for (const auto& observation : observations) {
        if (observation.size() != dimensions) {
            throw std::invalid_argument(
                "Observations must have equal dimensions."
            );
        }
    }

    Vector means(dimensions, 0.0);

    for (const auto& observation : observations) {
        means = add(means, observation);
    }

    means = scale(
        1.0 / static_cast<double>(observations.size()),
        means
    );

    Matrix covariance(dimensions, dimensions);

    for (const auto& observation : observations) {
        Vector centered = subtract(observation, means);

        for (std::size_t i = 0; i < dimensions; ++i) {
            for (std::size_t j = 0; j < dimensions; ++j) {
                covariance(i, j) +=
                    centered[i] * centered[j];
            }
        }
    }

    const double denominator =
        static_cast<double>(observations.size() - 1);

    for (std::size_t i = 0; i < dimensions; ++i) {
        for (std::size_t j = 0; j < dimensions; ++j) {
            covariance(i, j) /= denominator;
        }
    }

    return covariance;
}

Matrix graphLaplacian(const Matrix& adjacency) {
    if (adjacency.rows() != adjacency.cols()) {
        throw std::invalid_argument(
            "Adjacency matrix must be square."
        );
    }

    const std::size_t n = adjacency.rows();
    Matrix result(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        double degree = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            degree += adjacency(i, j);
        }

        for (std::size_t j = 0; j < n; ++j) {
            result(i, j) =
                (i == j ? degree : 0.0) - adjacency(i, j);
        }
    }

    return result;
}

std::pair<Vector, std::size_t> conjugateGradient(
    const Matrix& A,
    const Vector& b,
    double tolerance = 1e-10
) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "Conjugate Gradient requires square matrix."
        );
    }

    if (!isSymmetric(A)) {
        throw std::invalid_argument(
            "Conjugate Gradient requires a symmetric matrix."
        );
    }

    Vector x(A.rows(), 0.0);
    Vector r = subtract(b, A.multiply(x));
    Vector p = r;

    double rsOld = dot(r, r);

    if (std::sqrt(rsOld) < tolerance) {
        return {x, 0};
    }

    /*
     * For a symmetric positive-definite matrix, CG converges in at most
     * n exact arithmetic iterations. Floating-point arithmetic usually
     * requires tolerance-based termination instead.
     */
    for (std::size_t iteration = 1;
         iteration <= A.rows() * 10;
         ++iteration) {

        Vector Ap = A.multiply(p);

        const double denominator = dot(p, Ap);

        if (std::abs(denominator) < EPSILON) {
            throw std::runtime_error(
                "Degenerate CG search direction."
            );
        }

        const double alpha = rsOld / denominator;

        x = add(x, scale(alpha, p));
        r = subtract(r, scale(alpha, Ap));

        const double rsNew = dot(r, r);

        if (std::sqrt(rsNew) < tolerance) {
            return {x, iteration};
        }

        const double beta = rsNew / rsOld;

        p = add(r, scale(beta, p));
        rsOld = rsNew;
    }

    return {x, A.rows() * 10};
}

void printSection(const std::string& title) {
    std::cout << "\n"
              << std::string(78, '=')
              << "\n"
              << title
              << "\n"
              << std::string(78, '=')
              << "\n";
}

void runCaseStudy() {
    printSection("1. Sensor measurement vectors");

    Vector temperatureAndPressure = {
        25.0,
        101.3
    };

    Vector calibrationOffset = {
        1.5,
        -0.3
    };

    printVector(
        "raw measurement",
        temperatureAndPressure
    );

    printVector(
        "calibrated measurement",
        add(
            temperatureAndPressure,
            calibrationOffset
        )
    );

    std::cout << "measurement magnitude = "
              << norm(temperatureAndPressure)
              << "\n";

    printSection("2. Matrix-based calibration");

    /*
     * The matrix models cross-coupling between two sensor channels.
     */
    Matrix calibration{
        {1.02, 0.01},
        {0.02, 0.98}
    };

    calibration.print("Calibration matrix:");

    Vector corrected =
        calibration.multiply(temperatureAndPressure);

    printVector(
        "corrected sensor vector",
        corrected
    );

    printSection("3. Solving a calibration system");

    Matrix equations{
        {2.0, 1.0},
        {1.0, -1.0}
    };

    Vector measurements{
        5.0,
        1.0
    };

    Vector solution = solve(equations, measurements);

    printVector("calibration parameters", solution);

    printSection("4. Rank and invertibility");

    Matrix dependent{
        {1.0, 2.0, 3.0},
        {2.0, 4.0, 6.0},
        {1.0, 1.0, 1.0}
    };

    dependent.print("Dependent measurement model:");

    std::cout << "rank = "
              << rank(dependent)
              << "\n";

    Matrix invertible{
        {4.0, 7.0},
        {2.0, 6.0}
    };

    std::cout << "determinant = "
              << determinant(invertible)
              << "\n";

    inverse(invertible).print("Inverse:");

    printSection("5. Orthogonal coordinate system");

    std::vector<Vector> sensorDirections{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };

    std::vector<Vector> orthonormal =
        gramSchmidt(sensorDirections);

    for (std::size_t i = 0; i < orthonormal.size(); ++i) {
        printVector(
            "orthonormal direction " +
            std::to_string(i + 1),
            orthonormal[i]
        );
    }

    printSection("6. QR decomposition of sensor design matrix");

    Matrix design{
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    auto [Q, R] = qrDecomposition(design);

    Q.print("Q:");
    R.print("R:");

    Matrix reconstructed = Q * R;
    reconstructed.print("Q * R:");

    printSection("7. Least-squares calibration");

    /*
     * Suppose the true relationship is approximately:
     *
     *     measurement = intercept + gain * input
     *
     * Experimental noise prevents all points from lying on one line.
     */
    std::vector<std::pair<double, double>> measurementsForFit{
        {0.0, 1.1},
        {1.0, 2.9},
        {2.0, 5.2},
        {3.0, 6.8},
        {4.0, 9.1}
    };

    Matrix regression(
        measurementsForFit.size(),
        2
    );

    Vector observed(measurementsForFit.size());

    for (std::size_t i = 0;
         i < measurementsForFit.size();
         ++i) {

        const double x = measurementsForFit[i].first;
        const double y = measurementsForFit[i].second;

        regression(i, 0) = 1.0;
        regression(i, 1) = x;
        observed[i] = y;
    }

    Vector coefficients =
        leastSquares(regression, observed);

    printVector(
        "[intercept, gain]",
        coefficients
    );

    printSection("8. Covariance analysis");

    std::vector<Vector> sensorObservations{
        {2.5, 2.4},
        {0.5, 0.7},
        {2.2, 2.9},
        {1.9, 2.2},
        {3.1, 3.0},
        {2.3, 2.7},
        {2.0, 1.6},
        {1.0, 1.1},
        {1.5, 1.6},
        {1.1, 0.9}
    };

    Matrix covariance =
        covarianceMatrix(sensorObservations);

    covariance.print("Covariance matrix:");

    printSection("9. Dominant eigenvalue");

    Matrix dynamics{
        {4.0, 1.0},
        {2.0, 3.0}
    };

    auto [dominantEigenvalue, dominantEigenvector] =
        powerIteration(dynamics);

    std::cout << "dominant eigenvalue = "
              << dominantEigenvalue
              << "\n";

    printVector(
        "dominant eigenvector",
        dominantEigenvector
    );

    printSection("10. Quadratic energy model");

    Matrix energyMatrix{
        {4.0, 1.0},
        {1.0, 3.0}
    };

    Vector state{
        2.0,
        -1.0
    };

    std::cout << "x^T A x = "
              << quadraticForm(energyMatrix, state)
              << "\n";

    std::cout << "symmetric = "
              << (isSymmetric(energyMatrix) ? "true" : "false")
              << "\n";

    printSection("11. Sensor-network graph Laplacian");

    /*
     * Four sensors form an undirected network.
     *
     * A(i,j) = 1 means sensors i and j communicate directly.
     * D contains node degrees.
     * L = D - A.
     */
    Matrix adjacency{
        {0, 1, 1, 0},
        {1, 0, 1, 0},
        {1, 1, 0, 1},
        {0, 0, 1, 0}
    };

    Matrix laplacian =
        graphLaplacian(adjacency);

    adjacency.print("Adjacency matrix:");
    laplacian.print("Graph Laplacian:");

    Vector constantSignal{
        1.0, 1.0, 1.0, 1.0
    };

    printVector(
        "L * constant signal",
        laplacian.multiply(constantSignal)
    );

    printSection("12. Conjugate Gradient system solver");

    /*
     * A is symmetric positive definite.
     *
     * CG avoids explicitly calculating A^-1. This is important for large
     * systems because solving Ax=b is usually preferable to constructing
     * the inverse matrix.
     */
    Matrix positiveDefinite{
        {4.0, 1.0},
        {1.0, 3.0}
    };

    Vector rightHandSide{
        1.0,
        2.0
    };

    auto [cgSolution, iterations] =
        conjugateGradient(
            positiveDefinite,
            rightHandSide
        );

    printVector(
        "CG solution",
        cgSolution
    );

    std::cout << "CG iterations = "
              << iterations
              << "\n";

    Vector directSolution =
        solve(positiveDefinite, rightHandSide);

    printVector(
        "direct solution",
        directSolution
    );
}

void runTests() {
    printSection("13. Automated validation");

    Matrix A{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(std::abs(determinant(A) + 2.0) < 1e-9);

    Matrix identityApprox =
        A * inverse(A);

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            const double expected =
                i == j ? 1.0 : 0.0;

            assert(
                std::abs(
                    identityApprox(i, j) - expected
                ) < 1e-8
            );
        }
    }

    Vector solution =
        solve(
            Matrix{
                {2.0, 1.0},
                {1.0, -1.0}
            },
            {5.0, 1.0}
        );

    assert(std::abs(solution[0] - 2.0) < 1e-9);
    assert(std::abs(solution[1] - 1.0) < 1e-9);

    std::vector<Vector> vectors{
        {1.0, 1.0},
        {1.0, -1.0}
    };

    auto orthonormal =
        gramSchmidt(vectors);

    assert(std::abs(norm(orthonormal[0]) - 1.0) < 1e-9);
    assert(std::abs(norm(orthonormal[1]) - 1.0) < 1e-9);
    assert(std::abs(dot(
        orthonormal[0],
        orthonormal[1]
    )) < 1e-8);

    std::cout << "All C++ tests passed.\n";
}

void demonstrateFailureConditions() {
    printSection("14. Failure conditions and edge cases");

    try {
        Matrix singular{
            {1.0, 2.0},
            {2.0, 4.0}
        };

        inverse(singular);
    } catch (const std::exception& error) {
        std::cout << "Singular inverse rejected: "
                  << error.what()
                  << "\n";
    }

    try {
        normalize({0.0, 0.0});
    } catch (const std::exception& error) {
        std::cout << "Zero-vector normalization rejected: "
                  << error.what()
                  << "\n";
    }

    try {
        solve(
            Matrix{
                {1.0, 1.0},
                {2.0, 2.0}
            },
            {1.0, 3.0}
        );
    } catch (const std::exception& error) {
        std::cout << "Inconsistent system rejected: "
                  << error.what()
                  << "\n";
    }

    try {
        Matrix invalid{
            {1.0, 2.0},
            {3.0, 4.0}
        };

        Matrix incompatible{
            {1.0, 2.0, 3.0}
        };

        invalid * incompatible;
    } catch (const std::exception& error) {
        std::cout << "Invalid multiplication rejected: "
                  << error.what()
                  << "\n";
    }
}

int main() {
    try {
        std::cout << "ADVANCED LINEAR ALGEBRA\n";
        std::cout
            << "Industry-style sensor analytics case study\n";

        runCaseStudy();
        runTests();
        demonstrateFailureConditions();

        printSection("15. Computational design considerations");

        std::cout << R"(
Dense matrix multiplication:
    O(mnp) for A(m x n) multiplied by B(n x p).

Gaussian elimination:
    O(n^3) for dense n x n systems.

Matrix storage:
    O(n^2) memory for dense n x n matrices.

Least squares:
    Normal equations are simple but can amplify conditioning problems.
    QR and SVD are generally preferable for numerically sensitive problems.

Eigenvalue computation:
    Power iteration is inexpensive but normally targets the dominant
    eigenvalue and depends on spectral separation.

Conjugate Gradient:
    Appropriate for symmetric positive-definite systems.
    Its practical cost depends strongly on matrix sparsity and conditioning.

Production numerical software:
    A production implementation should normally use highly optimized
    and extensively tested numerical kernels. The educational implementations
    here make mathematical mechanisms explicit rather than maximizing speed.

Floating-point arithmetic:
    Exact symbolic identities become approximate numerical identities.
    Tolerance-based comparisons are therefore necessary.

Security and reliability:
    Numerical input should be validated before calculations.
    Extremely large values can overflow.
    Extremely small values can underflow.
    Ill-conditioned matrices can magnify measurement errors.
)";

        std::cout
            << "\nCase study completed successfully.\n";
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
