#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
    Linear Algebra Industry-Style Case Study

    Scenario:
    A small recommendation/risk-scoring engine receives numerical
    observations for several entities. It represents observations as
    vectors, stores transformation and scoring parameters as matrices,
    solves a calibration system, and uses tensor-like storage for
    historical observations.

    Concepts demonstrated:
      - Scalars
      - Vectors
      - Matrices
      - Matrix multiplication
      - Transpose
      - Determinants
      - Gaussian elimination
      - Matrix inverse
      - Linear systems
      - Rank
      - Linear transformations
      - Dot products
      - Tensor-like multidimensional storage
      - Validation
      - Complexity and numerical considerations

    Compile:
      C++17 or later
*/

using Vector = std::vector<double>;
using MatrixData = std::vector<Vector>;

constexpr double EPSILON = 1e-10;

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

bool nearlyEqual(double a, double b, double tolerance = EPSILON) {
    return std::abs(a - b) <= tolerance;
}

void printSection(const std::string& title) {
    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(78, '=') << "\n";
}

// -----------------------------------------------------------------------------
// Vector class
// -----------------------------------------------------------------------------

class VectorMath {
public:
    static void validate(const Vector& vector, const std::string& name) {
        if (vector.empty()) {
            throw std::invalid_argument(name + " cannot be empty.");
        }

        for (double value : vector) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    name + " contains a non-finite number."
                );
            }
        }
    }

    static Vector add(const Vector& a, const Vector& b) {
        validate(a, "a");
        validate(b, "b");

        if (a.size() != b.size()) {
            throw std::invalid_argument(
                "Vector dimensions must match."
            );
        }

        Vector result(a.size());

        for (std::size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }

        return result;
    }

    static Vector subtract(const Vector& a, const Vector& b) {
        validate(a, "a");
        validate(b, "b");

        if (a.size() != b.size()) {
            throw std::invalid_argument(
                "Vector dimensions must match."
            );
        }

        Vector result(a.size());

        for (std::size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] - b[i];
        }

        return result;
    }

    static Vector scale(double scalar, const Vector& vector) {
        validate(vector, "vector");

        Vector result(vector.size());

        for (std::size_t i = 0; i < vector.size(); ++i) {
            result[i] = scalar * vector[i];
        }

        return result;
    }

    static double dot(const Vector& a, const Vector& b) {
        validate(a, "a");
        validate(b, "b");

        if (a.size() != b.size()) {
            throw std::invalid_argument(
                "Dot product requires equal dimensions."
            );
        }

        double result = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }

        return result;
    }

    static double norm(const Vector& vector) {
        validate(vector, "vector");
        return std::sqrt(dot(vector, vector));
    }

    static double distance(const Vector& a, const Vector& b) {
        return norm(subtract(a, b));
    }

    static void print(const Vector& vector) {
        validate(vector, "vector");

        std::cout << "[ ";

        for (double value : vector) {
            std::cout << std::fixed << std::setprecision(4)
                      << value << " ";
        }

        std::cout << "]\n";
    }
};

// -----------------------------------------------------------------------------
// Matrix class
// -----------------------------------------------------------------------------

class Matrix {
private:
    MatrixData data_;

    void validateRectangular() const {
        if (data_.empty()) {
            throw std::invalid_argument(
                "Matrix must contain at least one row."
            );
        }

        if (data_[0].empty()) {
            throw std::invalid_argument(
                "Matrix must contain at least one column."
            );
        }

        const std::size_t columns = data_[0].size();

        for (const auto& row : data_) {
            if (row.size() != columns) {
                throw std::invalid_argument(
                    "Matrix must be rectangular."
                );
            }

            for (double value : row) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "Matrix contains a non-finite value."
                    );
                }
            }
        }
    }

public:
    Matrix() = delete;

    explicit Matrix(MatrixData data)
        : data_(std::move(data)) {
        validateRectangular();
    }

    static Matrix identity(std::size_t size) {
        if (size == 0) {
            throw std::invalid_argument(
                "Identity matrix size must be positive."
            );
        }

        MatrixData data(
            size,
            Vector(size, 0.0)
        );

        for (std::size_t i = 0; i < size; ++i) {
            data[i][i] = 1.0;
        }

        return Matrix(std::move(data));
    }

    std::size_t rows() const {
        return data_.size();
    }

    std::size_t columns() const {
        return data_[0].size();
    }

    bool isSquare() const {
        return rows() == columns();
    }

    double at(std::size_t row, std::size_t column) const {
        if (row >= rows() || column >= columns()) {
            throw std::out_of_range(
                "Matrix index is outside the valid range."
            );
        }

        return data_[row][column];
    }

    double& at(std::size_t row, std::size_t column) {
        if (row >= rows() || column >= columns()) {
            throw std::out_of_range(
                "Matrix index is outside the valid range."
            );
        }

        return data_[row][column];
    }

    Vector multiply(const Vector& vector) const {
        VectorMath::validate(vector, "vector");

        if (columns() != vector.size()) {
            throw std::invalid_argument(
                "Matrix columns must equal vector dimension."
            );
        }

        Vector result(rows(), 0.0);

        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t j = 0; j < columns(); ++j) {
                result[i] += data_[i][j] * vector[j];
            }
        }

        return result;
    }

    Matrix multiply(const Matrix& other) const {
        if (columns() != other.rows()) {
            throw std::invalid_argument(
                "Matrix multiplication dimensions are incompatible."
            );
        }

        MatrixData result(
            rows(),
            Vector(other.columns(), 0.0)
        );

        // Loop ordering i-k-j reduces repeated indexing and is often
        // friendlier to cache access than a naive i-j-k implementation.
        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t k = 0; k < columns(); ++k) {
                const double leftValue = data_[i][k];

                for (std::size_t j = 0; j < other.columns(); ++j) {
                    result[i][j] += leftValue * other.data_[k][j];
                }
            }
        }

        return Matrix(std::move(result));
    }

    Matrix add(const Matrix& other) const {
        if (rows() != other.rows() ||
            columns() != other.columns()) {
            throw std::invalid_argument(
                "Matrix dimensions must match."
            );
        }

        MatrixData result = data_;

        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t j = 0; j < columns(); ++j) {
                result[i][j] += other.data_[i][j];
            }
        }

        return Matrix(std::move(result));
    }

    Matrix subtract(const Matrix& other) const {
        if (rows() != other.rows() ||
            columns() != other.columns()) {
            throw std::invalid_argument(
                "Matrix dimensions must match."
            );
        }

        MatrixData result = data_;

        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t j = 0; j < columns(); ++j) {
                result[i][j] -= other.data_[i][j];
            }
        }

        return Matrix(std::move(result));
    }

    Matrix scale(double scalar) const {
        MatrixData result = data_;

        for (auto& row : result) {
            for (double& value : row) {
                value *= scalar;
            }
        }

        return Matrix(std::move(result));
    }

    Matrix transpose() const {
        MatrixData result(
            columns(),
            Vector(rows(), 0.0)
        );

        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t j = 0; j < columns(); ++j) {
                result[j][i] = data_[i][j];
            }
        }

        return Matrix(std::move(result));
    }

    double determinant() const {
        if (!isSquare()) {
            throw std::invalid_argument(
                "Determinant requires a square matrix."
            );
        }

        Matrix work = *this;
        double result = 1.0;

        for (std::size_t column = 0; column < rows(); ++column) {
            std::size_t pivotRow = column;

            for (std::size_t row = column + 1;
                 row < rows();
                 ++row) {

                if (std::abs(work.at(row, column)) >
                    std::abs(work.at(pivotRow, column))) {
                    pivotRow = row;
                }
            }

            if (nearlyEqual(work.at(pivotRow, column), 0.0)) {
                return 0.0;
            }

            if (pivotRow != column) {
                for (std::size_t j = 0; j < columns(); ++j) {
                    std::swap(
                        work.at(pivotRow, j),
                        work.at(column, j)
                    );
                }

                result *= -1.0;
            }

            const double pivot = work.at(column, column);
            result *= pivot;

            for (std::size_t row = column + 1;
                 row < rows();
                 ++row) {

                const double factor =
                    work.at(row, column) / pivot;

                for (std::size_t j = column + 1;
                     j < columns();
                     ++j) {

                    work.at(row, j) -=
                        factor * work.at(column, j);
                }
            }
        }

        return result;
    }

    Matrix rref() const {
        Matrix work = *this;

        std::size_t pivotRow = 0;

        for (std::size_t column = 0;
             column < columns() && pivotRow < rows();
             ++column) {

            std::size_t bestRow = pivotRow;

            for (std::size_t row = pivotRow + 1;
                 row < rows();
                 ++row) {

                if (std::abs(work.at(row, column)) >
                    std::abs(work.at(bestRow, column))) {
                    bestRow = row;
                }
            }

            if (std::abs(work.at(bestRow, column)) <= EPSILON) {
                continue;
            }

            for (std::size_t j = 0; j < columns(); ++j) {
                std::swap(
                    work.at(pivotRow, j),
                    work.at(bestRow, j)
                );
            }

            const double pivot =
                work.at(pivotRow, column);

            for (std::size_t j = 0; j < columns(); ++j) {
                work.at(pivotRow, j) /= pivot;
            }

            for (std::size_t row = 0; row < rows(); ++row) {
                if (row == pivotRow) {
                    continue;
                }

                const double factor =
                    work.at(row, column);

                if (std::abs(factor) <= EPSILON) {
                    continue;
                }

                for (std::size_t j = 0; j < columns(); ++j) {
                    work.at(row, j) -=
                        factor * work.at(pivotRow, j);
                }
            }

            ++pivotRow;
        }

        for (std::size_t i = 0; i < rows(); ++i) {
            for (std::size_t j = 0; j < columns(); ++j) {
                if (std::abs(work.at(i, j)) <= EPSILON) {
                    work.at(i, j) = 0.0;
                }
            }
        }

        return work;
    }

    std::size_t rank() const {
        Matrix reduced = rref();

        std::size_t rankValue = 0;

        for (std::size_t row = 0; row < reduced.rows(); ++row) {
            bool nonZero = false;

            for (std::size_t column = 0;
                 column < reduced.columns();
                 ++column) {

                if (std::abs(reduced.at(row, column)) > EPSILON) {
                    nonZero = true;
                    break;
                }
            }

            if (nonZero) {
                ++rankValue;
            }
        }

        return rankValue;
    }

    Matrix inverse() const {
        if (!isSquare()) {
            throw std::invalid_argument(
                "Only square matrices can be inverted."
            );
        }

        const std::size_t n = rows();

        MatrixData augmented(
            n,
            Vector(2 * n, 0.0)
        );

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                augmented[i][j] = data_[i][j];
            }

            augmented[i][n + i] = 1.0;
        }

        Matrix reduced(std::move(augmented));
        reduced = reduced.rref();

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double expected =
                    i == j ? 1.0 : 0.0;

                if (!nearlyEqual(
                        reduced.at(i, j),
                        expected)) {

                    throw std::runtime_error(
                        "Matrix is singular and has no inverse."
                    );
                }
            }
        }

        MatrixData inverseData(
            n,
            Vector(n, 0.0)
        );

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                inverseData[i][j] =
                    reduced.at(i, n + j);
            }
        }

        return Matrix(std::move(inverseData));
    }

    void print() const {
        for (const auto& row : data_) {
            std::cout << "[ ";

            for (double value : row) {
                std::cout << std::fixed
                          << std::setprecision(4)
                          << value << " ";
            }

            std::cout << "]\n";
        }
    }
};

// -----------------------------------------------------------------------------
// Tensor-like storage
// -----------------------------------------------------------------------------

class Tensor3D {
private:
    std::size_t depth_;
    std::size_t rows_;
    std::size_t columns_;
    std::vector<double> values_;

    std::size_t index(
        std::size_t depth,
        std::size_t row,
        std::size_t column
    ) const {
        if (depth >= depth_ ||
            row >= rows_ ||
            column >= columns_) {

            throw std::out_of_range(
                "Tensor index is outside the valid range."
            );
        }

        // Flatten a 3D coordinate into one contiguous 1D array.
        return depth * rows_ * columns_
             + row * columns_
             + column;
    }

public:
    Tensor3D(
        std::size_t depth,
        std::size_t rows,
        std::size_t columns
    )
        : depth_(depth),
          rows_(rows),
          columns_(columns),
          values_(depth * rows * columns, 0.0) {

        if (depth == 0 || rows == 0 || columns == 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be positive."
            );
        }
    }

    double& at(
        std::size_t depth,
        std::size_t row,
        std::size_t column
    ) {
        return values_[index(depth, row, column)];
    }

    double at(
        std::size_t depth,
        std::size_t row,
        std::size_t column
    ) const {
        return values_[index(depth, row, column)];
    }

    std::size_t size() const {
        return values_.size();
    }

    void fillSequential() {
        for (std::size_t i = 0; i < values_.size(); ++i) {
            values_[i] = static_cast<double>(i + 1);
        }
    }

    void print() const {
        for (std::size_t depth = 0; depth < depth_; ++depth) {
            std::cout << "Depth " << depth << ":\n";

            for (std::size_t row = 0; row < rows_; ++row) {
                std::cout << "[ ";

                for (std::size_t column = 0;
                     column < columns_;
                     ++column) {

                    std::cout << at(depth, row, column)
                              << " ";
                }

                std::cout << "]\n";
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Linear-system solver
// -----------------------------------------------------------------------------

Vector solveLinearSystem(
    const Matrix& coefficients,
    const Vector& constants
) {
    if (!coefficients.isSquare()) {
        throw std::invalid_argument(
            "This solver requires a square coefficient matrix."
        );
    }

    if (coefficients.rows() != constants.size()) {
        throw std::invalid_argument(
            "Coefficient matrix and constants have incompatible sizes."
        );
    }

    MatrixData augmented(
        coefficients.rows(),
        Vector(coefficients.columns() + 1, 0.0)
    );

    for (std::size_t i = 0; i < coefficients.rows(); ++i) {
        for (std::size_t j = 0;
             j < coefficients.columns();
             ++j) {

            augmented[i][j] =
                coefficients.at(i, j);
        }

        augmented[i][coefficients.columns()] =
            constants[i];
    }

    Matrix reduced(std::move(augmented));
    reduced = reduced.rref();

    for (std::size_t row = 0;
         row < reduced.rows();
         ++row) {

        bool coefficientsAreZero = true;

        for (std::size_t column = 0;
             column < coefficients.columns();
             ++column) {

            if (std::abs(reduced.at(row, column)) > EPSILON) {
                coefficientsAreZero = false;
                break;
            }
        }

        if (coefficientsAreZero &&
            std::abs(
                reduced.at(row, coefficients.columns())
            ) > EPSILON) {

            throw std::runtime_error(
                "The linear system is inconsistent."
            );
        }
    }

    if (coefficients.rank() < coefficients.columns()) {
        throw std::runtime_error(
            "The linear system does not have a unique solution."
        );
    }

    Vector solution(coefficients.columns(), 0.0);

    for (std::size_t row = 0;
         row < reduced.rows();
         ++row) {

        std::size_t pivotColumn =
            coefficients.columns();

        for (std::size_t column = 0;
             column < coefficients.columns();
             ++column) {

            if (std::abs(reduced.at(row, column)) > EPSILON) {
                pivotColumn = column;
                break;
            }
        }

        if (pivotColumn < coefficients.columns()) {
            solution[pivotColumn] =
                reduced.at(row, coefficients.columns());
        }
    }

    return solution;
}

// -----------------------------------------------------------------------------
// Industry-style case study
// -----------------------------------------------------------------------------

class RiskScoringEngine {
private:
    Matrix featureTransformation_;
    Vector bias_;
    Vector calibrationWeights_;

public:
    RiskScoringEngine(
        Matrix featureTransformation,
        Vector bias,
        Vector calibrationWeights
    )
        : featureTransformation_(std::move(featureTransformation)),
          bias_(std::move(bias)),
          calibrationWeights_(std::move(calibrationWeights)) {

        if (featureTransformation_.rows() != bias_.size()) {
            throw std::invalid_argument(
                "Transformation output must match bias size."
            );
        }

        if (featureTransformation_.rows() !=
            calibrationWeights_.size()) {

            throw std::invalid_argument(
                "Calibration weights must match transformed feature size."
            );
        }
    }

    double score(const Vector& rawFeatures) const {
        Vector transformed =
            featureTransformation_.multiply(rawFeatures);

        Vector adjusted =
            VectorMath::add(transformed, bias_);

        return VectorMath::dot(
            adjusted,
            calibrationWeights_
        );
    }
};

void demonstrateCaseStudy() {
    printSection("Industry Case Study: Risk-Scoring Pipeline");

    /*
        Raw features:

          x0 = normalized transaction amount
          x1 = transaction frequency
          x2 = account-age indicator

        A matrix transforms the raw feature vector into an engineered
        feature space. The bias shifts the transformed representation.
        A final vector dot product converts those features into a scalar
        score.

        This pattern appears in many numerical systems:
            transformed = W x
            adjusted    = transformed + b
            score       = wᵀ adjusted
    */

    Matrix transformation({
        {1.2, 0.1, -0.2},
        {0.3, 1.1, 0.4},
        {-0.1, 0.2, 1.4}
    });

    Vector bias = {
        0.05,
        -0.10,
        0.20
    };

    Vector weights = {
        0.50,
        0.30,
        0.20
    };

    RiskScoringEngine engine(
        transformation,
        bias,
        weights
    );

    Vector transactionFeatures = {
        0.8,
        0.6,
        0.9
    };

    const double score =
        engine.score(transactionFeatures);

    std::cout << "Raw feature vector:\n";
    VectorMath::print(transactionFeatures);

    std::cout << "Transformation matrix:\n";
    transformation.print();

    std::cout << "Matrix rank: "
              << transformation.rank()
              << "\n";

    std::cout << "Determinant: "
              << transformation.determinant()
              << "\n";

    std::cout << "Calculated score: "
              << std::fixed
              << std::setprecision(6)
              << score
              << "\n";

    /*
        Design trade-off:
        Keeping the mathematical operations inside reusable classes makes
        validation and behavior explicit. For production numerical systems,
        specialized linear algebra libraries are normally used instead of
        hand-written dense matrix multiplication because optimized libraries
        can exploit SIMD instructions, cache locality, multithreading, and
        specialized hardware.
    */
}

// -----------------------------------------------------------------------------
// Demonstrations
// -----------------------------------------------------------------------------

void demonstrateVectors() {
    printSection("1. Scalars and Vectors");

    const double scalar = 4.0;

    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0, 6.0};

    std::cout << "scalar = " << scalar << "\n";

    std::cout << "a = ";
    VectorMath::print(a);

    std::cout << "b = ";
    VectorMath::print(b);

    std::cout << "a + b = ";
    VectorMath::print(VectorMath::add(a, b));

    std::cout << "a - b = ";
    VectorMath::print(VectorMath::subtract(a, b));

    std::cout << "4a = ";
    VectorMath::print(VectorMath::scale(scalar, a));

    std::cout << "a dot b = "
              << VectorMath::dot(a, b)
              << "\n";

    std::cout << "||a|| = "
              << VectorMath::norm(a)
              << "\n";

    std::cout << "distance(a,b) = "
              << VectorMath::distance(a, b)
              << "\n";
}

void demonstrateMatrices() {
    printSection("2. Matrices and Matrix Operations");

    Matrix A({
        {1, 2, 3},
        {4, 5, 6}
    });

    Matrix B({
        {6, 5},
        {4, 3},
        {2, 1}
    });

    std::cout << "A:\n";
    A.print();

    std::cout << "B:\n";
    B.print();

    std::cout << "A * B:\n";
    A.multiply(B).print();

    std::cout << "A transpose:\n";
    A.transpose().print();

    std::cout << "A scaled by 2:\n";
    A.scale(2).print();

    std::cout << "I3:\n";
    Matrix::identity(3).print();
}

void demonstrateDeterminantAndInverse() {
    printSection("3. Determinants and Inverse");

    Matrix A({
        {4, 7},
        {2, 6}
    });

    std::cout << "A:\n";
    A.print();

    std::cout << "det(A) = "
              << A.determinant()
              << "\n";

    Matrix inverse = A.inverse();

    std::cout << "A^-1:\n";
    inverse.print();

    std::cout << "A * A^-1:\n";
    A.multiply(inverse).print();
}

void demonstrateLinearSystem() {
    printSection("4. Solving a Linear System");

    /*
        2x + y = 7
         x - y = 1

        Matrix representation:
            A x = b
    */

    Matrix A({
        {2, 1},
        {1, -1}
    });

    Vector b = {
        7,
        1
    };

    Vector solution =
        solveLinearSystem(A, b);

    std::cout << "Solution = ";
    VectorMath::print(solution);

    Vector verification =
        A.multiply(solution);

    std::cout << "Verification A*x = ";
    VectorMath::print(verification);
}

void demonstrateTensor() {
    printSection("5. Tensor Storage");

    /*
        A tensor is a generalization of scalar/vector/matrix structure.
        This case uses a 3D tensor:
            depth x rows x columns

        The data is stored in one contiguous vector. This illustrates how
        multidimensional mathematical coordinates can map to linear memory.
    */

    Tensor3D tensor(2, 2, 3);
    tensor.fillSequential();

    std::cout << "3D tensor:\n";
    tensor.print();

    std::cout << "Total elements: "
              << tensor.size()
              << "\n";

    tensor.at(1, 0, 2) = 99.0;

    std::cout << "After modifying [1,0,2]:\n";
    tensor.print();
}

void demonstrateEdgeCases() {
    printSection("6. Failure Conditions and Edge Cases");

    try {
        Matrix singular({
            {1, 2},
            {2, 4}
        });

        singular.inverse();
    } catch (const std::exception& error) {
        std::cout
            << "Singular inverse rejected: "
            << error.what()
            << "\n";
    }

    try {
        Matrix incompatible({
            {1, 2}
        });

        Matrix another({
            {1, 2}
        });

        incompatible.multiply(another);
    } catch (const std::exception& error) {
        std::cout
            << "Invalid multiplication rejected: "
            << error.what()
            << "\n";
    }

    try {
        Vector a = {1, 2};
        Vector b = {1, 2, 3};

        VectorMath::add(a, b);
    } catch (const std::exception& error) {
        std::cout
            << "Invalid vector addition rejected: "
            << error.what()
            << "\n";
    }

    /*
        These checks matter in production because a dimension mismatch is
        usually a structural programming error, while a singular matrix
        can be a legitimate property of the input data.
    */
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout
            << "LINEAR ALGEBRA CASE STUDY\n"
            << "Scalars, vectors, matrices, tensors, and matrix operations\n";

        demonstrateVectors();
        demonstrateMatrices();
        demonstrateDeterminantAndInverse();
        demonstrateLinearSystem();
        demonstrateTensor();
        demonstrateCaseStudy();
        demonstrateEdgeCases();

        printSection("Complexity Notes");

        std::cout << "Vector addition: O(n)\n";
        std::cout << "Dot product: O(n)\n";
        std::cout << "Matrix addition: O(m*n)\n";
        std::cout << "Matrix-vector multiplication: O(m*n)\n";
        std::cout << "Dense n x n matrix multiplication: O(n^3)\n";
        std::cout << "Gaussian elimination: O(n^3)\n";
        std::cout << "Dense matrix inversion by elimination: O(n^3)\n";

        std::cout
            << "\nThe program uses explicit dimension validation and partial "
               "pivoting to make numerical behavior safer. Production "
               "systems should normally use tested numerical libraries for "
               "large or numerically sensitive workloads.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << "\n";

        return 1;
    }
}
