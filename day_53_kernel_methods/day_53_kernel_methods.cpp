#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Kernel Methods: repository-scale similarity computation.
 *
 * The implementation treats a kernel as an object that maps two feature
 * vectors to a scalar similarity. The same abstraction supports linear,
 * polynomial, and RBF kernels without explicitly constructing nonlinear
 * feature vectors.
 *
 * The C++ case study emphasizes:
 * - value-oriented domain types,
 * - explicit ownership,
 * - reusable kernel interfaces,
 * - Gram matrices,
 * - numerical validation,
 * - kernel-based classification,
 * - kernel ridge regression.
 */

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

class KernelError : public std::runtime_error {
public:
    explicit KernelError(const std::string& message)
        : std::runtime_error(message) {}
};

void validateVector(const Vector& x, const std::string& name) {
    if (x.empty()) {
        throw KernelError(name + " must not be empty");
    }

    for (double value : x) {
        if (!std::isfinite(value)) {
            throw KernelError(
                name + " contains a non-finite value"
            );
        }
    }
}

void validateSameDimension(
    const Vector& a,
    const Vector& b
) {
    if (a.size() != b.size()) {
        throw KernelError(
            "dimension mismatch: " +
            std::to_string(a.size()) +
            " != " +
            std::to_string(b.size())
        );
    }
}

double dotProduct(
    const Vector& a,
    const Vector& b
) {
    validateSameDimension(a, b);

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double squaredDistance(
    const Vector& a,
    const Vector& b
) {
    validateSameDimension(a, b);

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        result += difference * difference;
    }

    return result;
}

class Kernel {
public:
    virtual ~Kernel() = default;

    virtual double operator()(
        const Vector& a,
        const Vector& b
    ) const = 0;

    virtual std::string name() const = 0;
};

class LinearKernel final : public Kernel {
public:
    double operator()(
        const Vector& a,
        const Vector& b
    ) const override {
        return dotProduct(a, b);
    }

    std::string name() const override {
        return "linear";
    }
};

class PolynomialKernel final : public Kernel {
private:
    int degree_;
    double gamma_;
    double coefficient_;

public:
    PolynomialKernel(
        int degree,
        double gamma,
        double coefficient
    )
        : degree_(degree),
          gamma_(gamma),
          coefficient_(coefficient) {
        if (degree < 1) {
            throw KernelError(
                "polynomial degree must be positive"
            );
        }

        if (!(gamma_ > 0.0) || !std::isfinite(gamma_)) {
            throw KernelError(
                "polynomial gamma must be positive"
            );
        }

        if (
            coefficient_ < 0.0 ||
            !std::isfinite(coefficient_)
        ) {
            throw KernelError(
                "polynomial coefficient must be non-negative"
            );
        }
    }

    double operator()(
        const Vector& a,
        const Vector& b
    ) const override {
        return std::pow(
            gamma_ * dotProduct(a, b) + coefficient_,
            degree_
        );
    }

    std::string name() const override {
        return "polynomial";
    }
};

class RBFKernel final : public Kernel {
private:
    double gamma_;

public:
    explicit RBFKernel(double gamma)
        : gamma_(gamma) {
        if (!(gamma_ > 0.0) || !std::isfinite(gamma_)) {
            throw KernelError(
                "RBF gamma must be positive"
            );
        }
    }

    double operator()(
        const Vector& a,
        const Vector& b
    ) const override {
        return std::exp(
            -gamma_ * squaredDistance(a, b)
        );
    }

    std::string name() const override {
        return "rbf";
    }
};

Matrix gramMatrix(
    const std::vector<Vector>& samples,
    const Kernel& kernel
) {
    if (samples.empty()) {
        throw KernelError(
            "Gram matrix requires at least one sample"
        );
    }

    const std::size_t dimension = samples.front().size();

    for (std::size_t i = 0; i < samples.size(); ++i) {
        validateVector(
            samples[i],
            "sample[" + std::to_string(i) + "]"
        );

        if (samples[i].size() != dimension) {
            throw KernelError(
                "all samples must have the same dimension"
            );
        }
    }

    Matrix matrix(
        samples.size(),
        std::vector<double>(samples.size(), 0.0)
    );

    /*
     * Only the upper triangle is calculated. Kernel symmetry then supplies
     * the lower triangle. This avoids roughly half of the kernel calls.
     */
    for (std::size_t i = 0; i < samples.size(); ++i) {
        for (std::size_t j = i; j < samples.size(); ++j) {
            const double value =
                kernel(samples[i], samples[j]);

            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }

    return matrix;
}

bool isPositiveSemidefinite(
    const Matrix& matrix,
    double tolerance = 1e-9
) {
    const std::size_t n = matrix.size();

    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw KernelError(
                "PSD test requires a square matrix"
            );
        }
    }

    Matrix lower(
        n,
        std::vector<double>(n, 0.0)
    );

    /*
     * A Cholesky-style factorization provides a practical numerical test.
     * Tiny negative diagonal values are treated as floating-point noise.
     */
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = matrix[i][j];

            for (std::size_t k = 0; k < j; ++k) {
                value -= lower[i][k] * lower[j][k];
            }

            if (i == j) {
                if (value < -tolerance) {
                    return false;
                }

                lower[i][j] =
                    std::sqrt(std::max(0.0, value));
            } else {
                const double pivot = lower[j][j];

                if (std::abs(pivot) <= tolerance) {
                    if (std::abs(value) > tolerance) {
                        return false;
                    }

                    lower[i][j] = 0.0;
                } else {
                    lower[i][j] = value / pivot;
                }
            }
        }
    }

    return true;
}

Vector solveLinearSystem(
    Matrix matrix,
    Vector rhs
) {
    const std::size_t n = matrix.size();

    if (n == 0 || rhs.size() != n) {
        throw KernelError(
            "invalid linear system"
        );
    }

    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw KernelError(
                "linear-system matrix must be square"
            );
        }
    }

    /*
     * Gaussian elimination with partial pivoting is sufficient for this
     * small educational model. Production numerical workloads should use
     * specialized, tested linear-algebra libraries.
     */
    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivotRow = column;

        for (std::size_t row = column + 1; row < n; ++row) {
            if (
                std::abs(matrix[row][column]) >
                std::abs(matrix[pivotRow][column])
            ) {
                pivotRow = row;
            }
        }

        if (
            std::abs(matrix[pivotRow][column]) <
            1e-12
        ) {
            throw KernelError(
                "singular or unstable linear system"
            );
        }

        std::swap(
            matrix[column],
            matrix[pivotRow]
        );

        std::swap(
            rhs[column],
            rhs[pivotRow]
        );

        for (std::size_t row = column + 1; row < n; ++row) {
            const double factor =
                matrix[row][column] /
                matrix[column][column];

            for (
                std::size_t k = column;
                k < n;
                ++k
            ) {
                matrix[row][k] -=
                    factor * matrix[column][k];
            }

            rhs[row] -= factor * rhs[column];
        }
    }

    Vector result(n, 0.0);

    for (
        std::size_t row = n;
        row-- > 0;
    ) {
        double remainder = rhs[row];

        for (
            std::size_t column = row + 1;
            column < n;
            ++column
        ) {
            remainder -=
                matrix[row][column] *
                result[column];
        }

        result[row] =
            remainder / matrix[row][row];
    }

    return result;
}

class SimilarityClassifier {
private:
    std::vector<Vector> samples_;
    std::vector<int> labels_;
    const Kernel& kernel_;

public:
    SimilarityClassifier(
        std::vector<Vector> samples,
        std::vector<int> labels,
        const Kernel& kernel
    )
        : samples_(std::move(samples)),
          labels_(std::move(labels)),
          kernel_(kernel) {
        if (samples_.empty()) {
            throw KernelError(
                "classifier requires training samples"
            );
        }

        if (samples_.size() != labels_.size()) {
            throw KernelError(
                "sample and label counts differ"
            );
        }

        const std::size_t dimension =
            samples_.front().size();

        for (const auto& sample : samples_) {
            validateVector(sample, "training sample");

            if (sample.size() != dimension) {
                throw KernelError(
                    "training dimensions differ"
                );
            }
        }
    }

    int predict(
        const Vector& query
    ) const {
        validateVector(query, "query");

        std::map<int, double> scores;

        for (std::size_t i = 0; i < samples_.size(); ++i) {
            scores[labels_[i]] +=
                kernel_(samples_[i], query);
        }

        return std::max_element(
            scores.begin(),
            scores.end(),
            [](
                const auto& left,
                const auto& right
            ) {
                return left.second < right.second;
            }
        )->first;
    }
};

class KernelRidge {
private:
    const Kernel& kernel_;
    double regularization_;
    std::vector<Vector> samples_;
    Vector coefficients_;

public:
    KernelRidge(
        const Kernel& kernel,
        double regularization
    )
        : kernel_(kernel),
          regularization_(regularization) {
        if (!(regularization_ > 0.0)) {
            throw KernelError(
                "regularization must be positive"
            );
        }
    }

    void fit(
        std::vector<Vector> samples,
        const Vector& targets
    ) {
        if (samples.empty()) {
            throw KernelError(
                "training set is empty"
            );
        }

        if (samples.size() != targets.size()) {
            throw KernelError(
                "sample and target counts differ"
            );
        }

        samples_ = std::move(samples);

        Matrix kernelMatrix =
            gramMatrix(samples_, kernel_);

        const std::size_t n = kernelMatrix.size();

        for (std::size_t i = 0; i < n; ++i) {
            kernelMatrix[i][i] +=
                regularization_;
        }

        coefficients_ =
            solveLinearSystem(
                std::move(kernelMatrix),
                targets
            );
    }

    double predict(
        const Vector& query
    ) const {
        if (coefficients_.empty()) {
            throw KernelError(
                "model has not been fitted"
            );
        }

        validateVector(query, "query");

        double result = 0.0;

        for (std::size_t i = 0; i < samples_.size(); ++i) {
            result +=
                coefficients_[i] *
                kernel_(samples_[i], query);
        }

        return result;
    }
};

void printMatrix(const Matrix& matrix) {
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& row : matrix) {
        for (double value : row) {
            std::cout << std::setw(9)
                      << value
                      << ' ';
        }

        std::cout << '\n';
    }
}

int main() {
    try {
        const std::vector<Vector> samples = {
            {-1.0, -1.0},
            {-1.0,  1.0},
            { 1.0, -1.0},
            { 1.0,  1.0},
            {-0.8, -1.1},
            {-1.1,  0.8},
            { 0.9, -1.2},
            { 1.1,  1.0}
        };

        const std::vector<int> labels = {
            0, 1, 1, 0, 0, 1, 1, 0
        };

        LinearKernel linear;
        PolynomialKernel polynomial(
            3,
            0.5,
            1.0
        );
        RBFKernel rbf(2.0);

        std::cout
            << "RBF Gram matrix\n";

        Matrix rbfMatrix =
            gramMatrix(samples, rbf);

        printMatrix(rbfMatrix);

        std::cout
            << "RBF Gram matrix PSD: "
            << std::boolalpha
            << isPositiveSemidefinite(rbfMatrix)
            << "\n\n";

        Matrix polynomialMatrix =
            gramMatrix(samples, polynomial);

        std::cout
            << "Polynomial Gram matrix PSD: "
            << isPositiveSemidefinite(polynomialMatrix)
            << "\n";

        SimilarityClassifier classifier(
            samples,
            labels,
            rbf
        );

        std::cout
            << "\nKernel similarity classification\n";

        const std::vector<Vector> queries = {
            {-0.9, -0.9},
            {-0.9,  0.9},
            { 0.9, -0.9},
            { 0.9,  0.9}
        };

        for (const auto& query : queries) {
            std::cout
                << '('
                << query[0]
                << ", "
                << query[1]
                << ") -> class "
                << classifier.predict(query)
                << '\n';
        }

        RBFKernel regressionKernel(0.7);

        KernelRidge regression(
            regressionKernel,
            0.05
        );

        regression.fit(
            {
                {-2.0},
                {-1.0},
                { 0.0},
                { 1.0},
                { 2.0}
            },
            {
                4.0,
                1.0,
                0.0,
                1.0,
                4.0
            }
        );

        std::cout
            << "\nKernel ridge predictions\n";

        for (double x : {-1.5, -0.5, 0.5, 1.5}) {
            std::cout
                << "x="
                << x
                << " -> "
                << regression.predict({x})
                << '\n';
        }

        try {
            rbfKernel(0, 0);
        } catch (...) {
            /*
             * No invalid construction is possible for this fixed kernel,
             * so this branch intentionally remains unused.
             */
        }

        std::cout
            << "\nKernel governance engine completed.\n";

    } catch (const std::exception& error) {
        std::cerr
            << "Failure: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}
