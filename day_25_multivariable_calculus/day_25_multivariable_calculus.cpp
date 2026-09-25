/*
 * Multivariable Calculus Case Study
 *
 * Topic:
 *   Gradients, Jacobians, Hessians, and Directional Derivatives
 *
 * Scenario:
 *   A two-dimensional autonomous engineering optimization model estimates
 *   the operating cost of a system as a function of two controllable
 *   parameters:
 *
 *       x = processing intensity
 *       y = cooling intensity
 *
 *   The program demonstrates:
 *     - scalar fields
 *     - gradients
 *     - directional derivatives
 *     - Jacobians
 *     - Hessians
 *     - Taylor approximation
 *     - critical-point classification
 *     - gradient descent
 *     - numerical differentiation
 *     - validation and edge cases
 *
 * Compile:
 *   g++ -std=c++17 -O2 multivariable_calculus.cpp -o multivariable_calculus
 *
 * Run:
 *   ./multivariable_calculus
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Dot product requires equal vector dimensions."
        );
    }

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double norm(const Vector& vector) {
    return std::sqrt(dot(vector, vector));
}

Vector normalize(const Vector& vector) {
    const double length = norm(vector);

    if (length == 0.0) {
        throw std::invalid_argument(
            "Cannot normalize the zero vector."
        );
    }

    Vector result = vector;

    for (double& value : result) {
        value /= length;
    }

    return result;
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions do not match."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions do not match."
        );
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

Vector matrixVectorMultiply(
    const Matrix& matrix,
    const Vector& vector
) {
    Vector result;

    for (const Vector& row : matrix) {
        result.push_back(dot(row, vector));
    }

    return result;
}

double quadraticForm(
    const Vector& vector,
    const Matrix& matrix
) {
    return dot(
        vector,
        matrixVectorMultiply(matrix, vector)
    );
}

// ---------------------------------------------------------------------------
// Domain model
// ---------------------------------------------------------------------------

class ProcessModel {
public:
    /*
     * Cost model:
     *
     *     C(x,y) =
     *       4x²
     *       + 2y²
     *       + xy
     *       - 12x
     *       - 8y
     *       + 30
     *
     * This is a smooth scalar field C: R² -> R.
     *
     * The optimization objective is to find the operating point
     * minimizing C.
     */

    double cost(double x, double y) const {
        return
            4.0 * x * x +
            2.0 * y * y +
            x * y -
            12.0 * x -
            8.0 * y +
            30.0;
    }

    Vector gradient(double x, double y) const {
        /*
         * ∂C/∂x = 8x + y - 12
         * ∂C/∂y = 4y + x - 8
         */
        return {
            8.0 * x + y - 12.0,
            4.0 * y + x - 8.0
        };
    }

    Matrix hessian() const {
        /*
         * H =
         *
         * [8 1]
         * [1 4]
         *
         * The Hessian is constant because C is quadratic.
         */
        return {
            {8.0, 1.0},
            {1.0, 4.0}
        };
    }

    Vector outputState(double x, double y) const {
        /*
         * A vector-valued observable:
         *
         * F1 = x² + y
         * F2 = xy
         * F3 = sin(x) + cos(y)
         *
         * This demonstrates why a Jacobian is needed for a
         * vector-valued function.
         */
        return {
            x * x + y,
            x * y,
            std::sin(x) + std::cos(y)
        };
    }

    Matrix jacobian(double x, double y) const {
        /*
         * J =
         *
         * [2x       1      ]
         * [y        x      ]
         * [cos(x)  -sin(y)]
         */
        return {
            {2.0 * x, 1.0},
            {y, x},
            {std::cos(x), -std::sin(y)}
        };
    }
};

// ---------------------------------------------------------------------------
// Numerical derivatives
// ---------------------------------------------------------------------------

Vector numericalGradient(
    const ProcessModel& model,
    const Vector& point,
    double h = 1e-5
) {
    if (point.size() != 2) {
        throw std::invalid_argument(
            "Gradient requires a two-dimensional point."
        );
    }

    const double x = point[0];
    const double y = point[1];

    const double dx =
        (
            model.cost(x + h, y) -
            model.cost(x - h, y)
        ) / (2.0 * h);

    const double dy =
        (
            model.cost(x, y + h) -
            model.cost(x, y - h)
        ) / (2.0 * h);

    return {dx, dy};
}

Matrix numericalHessian(
    const ProcessModel& model,
    const Vector& point,
    double h = 1e-4
) {
    if (point.size() != 2) {
        throw std::invalid_argument(
            "Hessian requires a two-dimensional point."
        );
    }

    const double x = point[0];
    const double y = point[1];

    const double center = model.cost(x, y);

    const double fxx =
        (
            model.cost(x + h, y) -
            2.0 * center +
            model.cost(x - h, y)
        ) / (h * h);

    const double fyy =
        (
            model.cost(x, y + h) -
            2.0 * center +
            model.cost(x, y - h)
        ) / (h * h);

    const double fxy =
        (
            model.cost(x + h, y + h) -
            model.cost(x + h, y - h) -
            model.cost(x - h, y + h) +
            model.cost(x - h, y - h)
        ) / (4.0 * h * h);

    return {
        {fxx, fxy},
        {fxy, fyy}
    };
}

// ---------------------------------------------------------------------------
// Jacobian numerical validation
// ---------------------------------------------------------------------------

Matrix numericalJacobian(
    const ProcessModel& model,
    const Vector& point,
    double h = 1e-5
) {
    const double x = point[0];
    const double y = point[1];

    Vector plusX = model.outputState(x + h, y);
    Vector minusX = model.outputState(x - h, y);

    Vector plusY = model.outputState(x, y + h);
    Vector minusY = model.outputState(x, y - h);

    Matrix result;

    for (std::size_t i = 0; i < plusX.size(); ++i) {
        result.push_back({
            (plusX[i] - minusX[i]) / (2.0 * h),
            (plusY[i] - minusY[i]) / (2.0 * h)
        });
    }

    return result;
}

// ---------------------------------------------------------------------------
// Directional derivative
// ---------------------------------------------------------------------------

double directionalDerivative(
    const ProcessModel& model,
    const Vector& point,
    const Vector& direction
) {
    const Vector unitDirection = normalize(direction);
    const Vector gradient =
        model.gradient(point[0], point[1]);

    return dot(gradient, unitDirection);
}

double numericalDirectionalDerivative(
    const ProcessModel& model,
    const Vector& point,
    const Vector& direction,
    double h = 1e-5
) {
    const Vector unitDirection = normalize(direction);

    Vector forward = add(
        point,
        scale(h, unitDirection)
    );

    Vector backward = subtract(
        point,
        scale(h, unitDirection)
    );

    return (
        model.cost(forward[0], forward[1]) -
        model.cost(backward[0], backward[1])
    ) / (2.0 * h);
}

// ---------------------------------------------------------------------------
// Taylor approximation
// ---------------------------------------------------------------------------

double firstOrderTaylor(
    double value,
    const Vector& gradient,
    const Vector& displacement
) {
    return value + dot(gradient, displacement);
}

double secondOrderTaylor(
    double value,
    const Vector& gradient,
    const Matrix& hessian,
    const Vector& displacement
) {
    return
        value +
        dot(gradient, displacement) +
        0.5 * quadraticForm(displacement, hessian);
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

double determinant2x2(const Matrix& matrix) {
    if (
        matrix.size() != 2 ||
        matrix[0].size() != 2 ||
        matrix[1].size() != 2
    ) {
        throw std::invalid_argument(
            "Expected a 2x2 matrix."
        );
    }

    return
        matrix[0][0] * matrix[1][1] -
        matrix[0][1] * matrix[1][0];
}

std::string classifyCriticalPoint(const Matrix& hessian) {
    const double determinant =
        determinant2x2(hessian);

    const double fxx = hessian[0][0];

    if (determinant > 0.0 && fxx > 0.0) {
        return "local minimum";
    }

    if (determinant > 0.0 && fxx < 0.0) {
        return "local maximum";
    }

    if (determinant < 0.0) {
        return "saddle point";
    }

    return "inconclusive";
}

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------

struct OptimizationResult {
    Vector point;
    std::vector<double> objectiveHistory;
};

OptimizationResult gradientDescent(
    const ProcessModel& model,
    Vector initialPoint,
    double learningRate,
    int iterations
) {
    Vector point = initialPoint;

    std::vector<double> history;
    history.push_back(
        model.cost(point[0], point[1])
    );

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const Vector gradient =
            model.gradient(point[0], point[1]);

        /*
         * The update is:
         *
         *   x_(k+1) = x_k - α ∇C(x_k)
         *
         * The negative sign moves toward decreasing cost.
         */
        point = subtract(
            point,
            scale(learningRate, gradient)
        );

        history.push_back(
            model.cost(point[0], point[1])
        );
    }

    return {point, history};
}

// ---------------------------------------------------------------------------
// Printing utilities
// ---------------------------------------------------------------------------

void printVector(
    const std::string& name,
    const Vector& vector
) {
    std::cout << name << " = [";

    for (std::size_t i = 0; i < vector.size(); ++i) {
        std::cout << std::fixed << std::setprecision(8)
                  << vector[i];

        if (i + 1 < vector.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}

void printMatrix(
    const std::string& name,
    const Matrix& matrix
) {
    std::cout << name << " =\n";

    for (const Vector& row : matrix) {
        std::cout << "  [";

        for (std::size_t i = 0; i < row.size(); ++i) {
            std::cout << std::fixed
                      << std::setprecision(8)
                      << row[i];

            if (i + 1 < row.size()) {
                std::cout << ", ";
            }
        }

        std::cout << "]\n";
    }
}

bool approximatelyEqual(
    double a,
    double b,
    double tolerance = 1e-6
) {
    return std::abs(a - b) <= tolerance;
}

// ---------------------------------------------------------------------------
// Case-study demonstrations
// ---------------------------------------------------------------------------

void demonstrateGradient(const ProcessModel& model) {
    std::cout << "\n--- 1. Gradient ---\n";

    const Vector point = {2.0, 1.0};

    const Vector analytical =
        model.gradient(point[0], point[1]);

    const Vector numerical =
        numericalGradient(model, point);

    printVector("Point", point);
    printVector("Analytical gradient", analytical);
    printVector("Numerical gradient", numerical);

    std::cout
        << "Interpretation: the gradient gives the direction of "
        << "steepest local increase in cost.\n";
}

void demonstrateDirectionalDerivatives(
    const ProcessModel& model
) {
    std::cout << "\n--- 2. Directional derivatives ---\n";

    const Vector point = {2.0, 1.0};

    const std::vector<Vector> directions = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {-1.0, -1.0}
    };

    for (const Vector& direction : directions) {
        const double analytical =
            directionalDerivative(
                model,
                point,
                direction
            );

        const double numerical =
            numericalDirectionalDerivative(
                model,
                point,
                direction
            );

        printVector("Direction", direction);

        std::cout
            << "  Analytical: "
            << analytical << "\n"
            << "  Numerical:  "
            << numerical << "\n";
    }

    std::cout
        << "A direction is normalized before calculating "
        << "D_u f = grad(f) dot u.\n";
}

void demonstrateJacobian(const ProcessModel& model) {
    std::cout << "\n--- 3. Jacobian ---\n";

    const Vector point = {1.5, 0.75};

    const Matrix analytical =
        model.jacobian(point[0], point[1]);

    const Matrix numerical =
        numericalJacobian(model, point);

    printMatrix("Analytical Jacobian", analytical);
    printMatrix("Numerical Jacobian", numerical);

    std::cout
        << "The Jacobian describes the first-order transformation "
        << "from input changes to output changes.\n";
}

void demonstrateHessian(const ProcessModel& model) {
    std::cout << "\n--- 4. Hessian ---\n";

    const Vector point = {2.0, 1.0};

    const Matrix analytical =
        model.hessian();

    const Matrix numerical =
        numericalHessian(model, point);

    printMatrix("Analytical Hessian", analytical);
    printMatrix("Numerical Hessian", numerical);

    std::cout
        << "Critical-point classification: "
        << classifyCriticalPoint(analytical)
        << "\n";

    const double determinant =
        determinant2x2(analytical);

    std::cout
        << "Hessian determinant: "
        << determinant
        << "\n";
}

void demonstrateTaylor(const ProcessModel& model) {
    std::cout << "\n--- 5. Taylor approximation ---\n";

    const Vector basePoint = {1.0, 1.0};
    const Vector displacement = {0.05, -0.03};

    const Vector target =
        add(basePoint, displacement);

    const double baseValue =
        model.cost(basePoint[0], basePoint[1]);

    const Vector gradient =
        model.gradient(
            basePoint[0],
            basePoint[1]
        );

    const Matrix hessian =
        model.hessian();

    const double exact =
        model.cost(target[0], target[1]);

    const double firstOrder =
        firstOrderTaylor(
            baseValue,
            gradient,
            displacement
        );

    const double secondOrder =
        secondOrderTaylor(
            baseValue,
            gradient,
            hessian,
            displacement
        );

    printVector("Base point", basePoint);
    printVector("Displacement", displacement);
    printVector("Target", target);

    std::cout
        << "Exact value:           "
        << exact << "\n"
        << "First-order estimate:  "
        << firstOrder << "\n"
        << "Second-order estimate: "
        << secondOrder << "\n";
}

void demonstrateOptimization(const ProcessModel& model) {
    std::cout << "\n--- 6. Gradient-descent optimization ---\n";

    const Vector initialPoint = {5.0, -4.0};

    const double learningRate = 0.08;
    const int iterations = 100;

    const OptimizationResult result =
        gradientDescent(
            model,
            initialPoint,
            learningRate,
            iterations
        );

    printVector("Initial point", initialPoint);
    printVector("Final point", result.point);

    std::cout
        << "Initial cost: "
        << result.objectiveHistory.front()
        << "\n"
        << "Final cost:   "
        << result.objectiveHistory.back()
        << "\n";

    std::cout
        << "Gradient descent minimizes the objective by "
        << "repeatedly moving opposite the gradient.\n";
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

void runTests(const ProcessModel& model) {
    std::cout << "\n--- Verification tests ---\n";

    const Vector gradient =
        model.gradient(2.0, 1.0);

    if (!approximatelyEqual(gradient[0], 5.0)) {
        throw std::runtime_error(
            "Gradient x-component test failed."
        );
    }

    if (!approximatelyEqual(gradient[1], -2.0)) {
        throw std::runtime_error(
            "Gradient y-component test failed."
        );
    }

    const Matrix hessian = model.hessian();

    if (!approximatelyEqual(hessian[0][0], 8.0) ||
        !approximatelyEqual(hessian[0][1], 1.0) ||
        !approximatelyEqual(hessian[1][0], 1.0) ||
        !approximatelyEqual(hessian[1][1], 4.0)) {
        throw std::runtime_error(
            "Hessian test failed."
        );
    }

    const double directional =
        directionalDerivative(
            model,
            {2.0, 1.0},
            {1.0, 0.0}
        );

    if (!approximatelyEqual(directional, 5.0)) {
        throw std::runtime_error(
            "Directional derivative test failed."
        );
    }

    if (
        classifyCriticalPoint({
            {2.0, 0.0},
            {0.0, 2.0}
        }) != "local minimum"
    ) {
        throw std::runtime_error(
            "Minimum classification failed."
        );
    }

    if (
        classifyCriticalPoint({
            {-2.0, 0.0},
            {0.0, -2.0}
        }) != "local maximum"
    ) {
        throw std::runtime_error(
            "Maximum classification failed."
        );
    }

    if (
        classifyCriticalPoint({
            {2.0, 0.0},
            {0.0, -2.0}
        }) != "saddle point"
    ) {
        throw std::runtime_error(
            "Saddle classification failed."
        );
    }

    try {
        normalize({0.0, 0.0});
        throw std::runtime_error(
            "Zero-vector normalization should fail."
        );
    } catch (const std::invalid_argument&) {
        // Expected failure.
    }

    std::cout << "All tests passed.\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    try {
        std::cout << std::string(78, '=') << "\n";
        std::cout
            << "MULTIVARIABLE CALCULUS CASE STUDY\n"
            << "Gradients, Jacobians, Hessians, and Directional Derivatives\n";
        std::cout << std::string(78, '=') << "\n";

        std::cout
            << "\nScenario:\n"
            << "A two-variable process model is optimized using "
            << "multivariable calculus.\n"
            << "x represents processing intensity and y represents "
            << "cooling intensity.\n";

        ProcessModel model;

        demonstrateGradient(model);
        demonstrateDirectionalDerivatives(model);
        demonstrateJacobian(model);
        demonstrateHessian(model);
        demonstrateTaylor(model);
        demonstrateOptimization(model);
        runTests(model);

        std::cout
            << "\n--- Engineering interpretation ---\n"
            << "Gradient: sensitivity of scalar cost to each control.\n"
            << "Directional derivative: sensitivity along a chosen "
            << "combined change in controls.\n"
            << "Jacobian: sensitivity of multiple system outputs to "
            << "multiple inputs.\n"
            << "Hessian: local curvature of the scalar cost surface.\n"
            << "Gradient descent: iterative optimization using first-order "
            << "derivative information.\n";

        std::cout
            << "\nProgram completed successfully.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Program error: "
            << error.what()
            << "\n";

        return 1;
    }
}
