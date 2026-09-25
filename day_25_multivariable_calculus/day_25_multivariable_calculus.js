/*
 * Multivariable Calculus
 * Gradients, Jacobians, Hessians, and Directional Derivatives
 *
 * Self-contained JavaScript demonstrations progressing from basic
 * multivariable functions to numerical differentiation, Jacobians,
 * Hessians, Taylor approximation, the chain rule, and optimization.
 *
 * Run with:
 *   node multivariable_calculus.js
 */

"use strict";

// ---------------------------------------------------------------------------
// Vector and matrix utilities
// ---------------------------------------------------------------------------

function dot(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function norm(vector) {
    return Math.sqrt(dot(vector, vector));
}

function normalize(vector) {
    const length = norm(vector);

    if (length === 0) {
        throw new Error("The zero vector cannot be normalized.");
    }

    return vector.map(value => value / length);
}

function addVectors(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.map((value, index) => value + b[index]);
}

function subtractVectors(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.map((value, index) => value - b[index]);
}

function matrixVectorMultiply(matrix, vector) {
    return matrix.map(row => dot(row, vector));
}

function quadraticForm(vector, matrix) {
    return dot(vector, matrixVectorMultiply(matrix, vector));
}

// ---------------------------------------------------------------------------
// Scalar-valued functions
// ---------------------------------------------------------------------------

function surface(x, y) {
    // f(x,y) = x² + 3xy + 2y²
    return x * x + 3 * x * y + 2 * y * y;
}

function nonlinearSurface(x, y) {
    // A nonlinear scalar field for numerical derivative examples.
    return Math.sin(x) * Math.exp(y) + x * x * y;
}

// ---------------------------------------------------------------------------
// Vector-valued function
// ---------------------------------------------------------------------------

function vectorFunction(x, y) {
    /*
     * F: R² -> R³
     *
     * F(x,y) =
     * [
     *   x² + y,
     *   xy,
     *   sin(x) + cos(y)
     * ]
     */
    return [
        x * x + y,
        x * y,
        Math.sin(x) + Math.cos(y)
    ];
}

// ---------------------------------------------------------------------------
// Numerical differentiation
// ---------------------------------------------------------------------------

function partialX(f, x, y, h = 1e-5) {
    return (f(x + h, y) - f(x - h, y)) / (2 * h);
}

function partialY(f, x, y, h = 1e-5) {
    return (f(x, y + h) - f(x, y - h)) / (2 * h);
}

function numericalGradient(f, point, h = 1e-5) {
    if (point.length !== 2) {
        throw new Error("This implementation expects a two-dimensional point.");
    }

    const [x, y] = point;

    return [
        partialX(f, x, y, h),
        partialY(f, x, y, h)
    ];
}

// ---------------------------------------------------------------------------
// Analytical gradient and Hessian
// ---------------------------------------------------------------------------

function analyticalGradient(x, y) {
    /*
     * For f = x² + 3xy + 2y²:
     *
     * ∂f/∂x = 2x + 3y
     * ∂f/∂y = 3x + 4y
     */
    return [
        2 * x + 3 * y,
        3 * x + 4 * y
    ];
}

function analyticalHessian() {
    /*
     * H =
     * [2 3]
     * [3 4]
     */
    return [
        [2, 3],
        [3, 4]
    ];
}

// ---------------------------------------------------------------------------
// Directional derivatives
// ---------------------------------------------------------------------------

function directionalDerivative(gradient, direction) {
    // D_u f = ∇f · u, where u must have unit length.
    const unitDirection = normalize(direction);
    return dot(gradient, unitDirection);
}

function numericalDirectionalDerivative(
    f,
    point,
    direction,
    h = 1e-5
) {
    const unitDirection = normalize(direction);

    const forward = point.map(
        (coordinate, index) => coordinate + h * unitDirection[index]
    );

    const backward = point.map(
        (coordinate, index) => coordinate - h * unitDirection[index]
    );

    return (
        f(forward[0], forward[1]) -
        f(backward[0], backward[1])
    ) / (2 * h);
}

// ---------------------------------------------------------------------------
// Jacobian
// ---------------------------------------------------------------------------

function analyticalJacobian(x, y) {
    /*
     * F1 = x² + y
     * F2 = xy
     * F3 = sin(x) + cos(y)
     *
     * J =
     * [2x       1      ]
     * [y        x      ]
     * [cos(x)  -sin(y)]
     */
    return [
        [2 * x, 1],
        [y, x],
        [Math.cos(x), -Math.sin(y)]
    ];
}

function numericalJacobian(f, point, h = 1e-5) {
    if (point.length !== 2) {
        throw new Error("This implementation expects two input variables.");
    }

    const [x, y] = point;

    const plusX = f(x + h, y);
    const minusX = f(x - h, y);
    const plusY = f(x, y + h);
    const minusY = f(x, y - h);

    return plusX.map((_, outputIndex) => [
        (plusX[outputIndex] - minusX[outputIndex]) / (2 * h),
        (plusY[outputIndex] - minusY[outputIndex]) / (2 * h)
    ]);
}

// ---------------------------------------------------------------------------
// Numerical Hessian
// ---------------------------------------------------------------------------

function numericalHessian(f, point, h = 1e-4) {
    const [x, y] = point;

    const center = f(x, y);

    const fxx = (
        f(x + h, y) -
        2 * center +
        f(x - h, y)
    ) / (h * h);

    const fyy = (
        f(x, y + h) -
        2 * center +
        f(x, y - h)
    ) / (h * h);

    const fxy = (
        f(x + h, y + h) -
        f(x + h, y - h) -
        f(x - h, y + h) +
        f(x - h, y - h)
    ) / (4 * h * h);

    return [
        [fxx, fxy],
        [fxy, fyy]
    ];
}

// ---------------------------------------------------------------------------
// Taylor approximations
// ---------------------------------------------------------------------------

function firstOrderTaylor(fValue, gradient, displacement) {
    return fValue + dot(gradient, displacement);
}

function secondOrderTaylor(
    fValue,
    gradient,
    hessian,
    displacement
) {
    return (
        fValue +
        dot(gradient, displacement) +
        0.5 * quadraticForm(displacement, hessian)
    );
}

// ---------------------------------------------------------------------------
// Critical-point classification
// ---------------------------------------------------------------------------

function determinant2x2(matrix) {
    return (
        matrix[0][0] * matrix[1][1] -
        matrix[0][1] * matrix[1][0]
    );
}

function classifyCriticalPoint(hessian) {
    const determinant = determinant2x2(hessian);
    const fxx = hessian[0][0];

    if (determinant > 0 && fxx > 0) {
        return "local minimum";
    }

    if (determinant > 0 && fxx < 0) {
        return "local maximum";
    }

    if (determinant < 0) {
        return "saddle point";
    }

    return "inconclusive";
}

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------

function gradientDescent(
    objective,
    gradientFunction,
    initialPoint,
    learningRate = 0.05,
    iterations = 100
) {
    let point = [...initialPoint];
    const history = [objective(point[0], point[1])];

    for (let iteration = 0; iteration < iterations; iteration++) {
        const gradient = gradientFunction(point[0], point[1]);

        point = point.map(
            (coordinate, index) =>
                coordinate - learningRate * gradient[index]
        );

        history.push(objective(point[0], point[1]));
    }

    return {
        point,
        history
    };
}

// ---------------------------------------------------------------------------
// Chain rule
// ---------------------------------------------------------------------------

function composedFunction(t) {
    const x = t * t;
    const y = Math.sin(t);

    return surface(x, y);
}

function chainRuleDerivative(t) {
    const x = t * t;
    const y = Math.sin(t);

    const gradient = analyticalGradient(x, y);

    // x'(t) = 2t
    // y'(t) = cos(t)
    const velocity = [
        2 * t,
        Math.cos(t)
    ];

    return dot(gradient, velocity);
}

function numericalDerivative(f, x, h = 1e-5) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function printMatrix(matrix) {
    matrix.forEach(row => {
        console.log(
            "  " +
            row.map(value => Number(value).toFixed(8)).join("  ")
        );
    });
}

// ---------------------------------------------------------------------------
// Demonstrations
// ---------------------------------------------------------------------------

function demonstrateGradient() {
    console.log("\n--- Gradient ---");

    const point = [2, 1];

    const analytical = analyticalGradient(...point);
    const numerical = numericalGradient(surface, point);

    console.log("Point:", point);
    console.log("Analytical gradient:", analytical);
    console.log(
        "Numerical gradient:",
        numerical.map(value => value.toFixed(8))
    );

    console.log(
        "The gradient contains all first-order partial derivatives."
    );
}

function demonstrateDirectionalDerivatives() {
    console.log("\n--- Directional derivatives ---");

    const point = [2, 1];
    const gradient = analyticalGradient(...point);

    const directions = [
        [1, 0],
        [0, 1],
        [1, 1],
        [-1, -1]
    ];

    directions.forEach(direction => {
        const analytical = directionalDerivative(
            gradient,
            direction
        );

        const numerical = numericalDirectionalDerivative(
            surface,
            point,
            direction
        );

        console.log(
            `Direction ${JSON.stringify(direction)}: ` +
            `analytical=${analytical.toFixed(8)}, ` +
            `numerical=${numerical.toFixed(8)}`
        );
    });
}

function demonstrateJacobian() {
    console.log("\n--- Jacobian ---");

    const point = [1.5, 0.75];

    const analytical = analyticalJacobian(...point);
    const numerical = numericalJacobian(vectorFunction, point);

    console.log("Analytical Jacobian:");
    printMatrix(analytical);

    console.log("Numerical Jacobian:");
    printMatrix(numerical);
}

function demonstrateHessian() {
    console.log("\n--- Hessian ---");

    const point = [2, 1];

    const analytical = analyticalHessian();
    const numerical = numericalHessian(surface, point);

    console.log("Analytical Hessian:");
    printMatrix(analytical);

    console.log("Numerical Hessian:");
    printMatrix(numerical);

    console.log(
        "Classification:",
        classifyCriticalPoint(analytical)
    );
}

function demonstrateTaylor() {
    console.log("\n--- Taylor approximation ---");

    const basePoint = [1, 1];
    const displacement = [0.05, -0.03];
    const target = addVectors(basePoint, displacement);

    const baseValue = nonlinearSurface(...basePoint);
    const gradient = numericalGradient(
        nonlinearSurface,
        basePoint
    );
    const hessian = numericalHessian(
        nonlinearSurface,
        basePoint
    );

    const exact = nonlinearSurface(...target);

    const firstOrder = firstOrderTaylor(
        baseValue,
        gradient,
        displacement
    );

    const secondOrder = secondOrderTaylor(
        baseValue,
        gradient,
        hessian,
        displacement
    );

    console.log("Exact value:", exact);
    console.log("First-order approximation:", firstOrder);
    console.log("Second-order approximation:", secondOrder);
}

function demonstrateChainRule() {
    console.log("\n--- Chain rule ---");

    const t = 0.8;

    const analytical = chainRuleDerivative(t);
    const numerical = numericalDerivative(
        composedFunction,
        t
    );

    console.log("t:", t);
    console.log(
        "Chain-rule derivative:",
        analytical.toFixed(10)
    );
    console.log(
        "Numerical derivative:",
        numerical.toFixed(10)
    );
}

function demonstrateOptimization() {
    console.log("\n--- Gradient descent ---");

    function bowl(x, y) {
        return 4 * x * x + 2 * y * y + x * y;
    }

    function bowlGradient(x, y) {
        return [
            8 * x + y,
            4 * y + x
        ];
    }

    const result = gradientDescent(
        bowl,
        bowlGradient,
        [5, -4],
        0.08,
        100
    );

    console.log(
        "Final point:",
        result.point.map(value => value.toFixed(10))
    );

    console.log(
        "Initial objective:",
        result.history[0].toFixed(10)
    );

    console.log(
        "Final objective:",
        result.history.at(-1).toFixed(10)
    );
}

function demonstrateEdgeCases() {
    console.log("\n--- Edge cases ---");

    try {
        normalize([0, 0]);
    } catch (error) {
        console.log("Handled zero direction:", error.message);
    }

    try {
        dot([1, 2], [1]);
    } catch (error) {
        console.log("Handled dimension mismatch:", error.message);
    }

    console.log(
        "A Jacobian describes first-order behavior of a vector-valued map."
    );

    console.log(
        "A Hessian describes second-order behavior of a scalar-valued map."
    );
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

function assertClose(actual, expected, tolerance = 1e-6) {
    if (Math.abs(actual - expected) > tolerance) {
        throw new Error(
            `Expected ${expected}, received ${actual}`
        );
    }
}

function runTests() {
    console.log("\n--- Tests ---");

    const gradient = analyticalGradient(2, 1);

    assertClose(gradient[0], 7);
    assertClose(gradient[1], 10);

    const hessian = analyticalHessian();

    assertClose(hessian[0][0], 2);
    assertClose(hessian[0][1], 3);
    assertClose(hessian[1][0], 3);
    assertClose(hessian[1][1], 4);

    const directional = directionalDerivative(
        [1, 0],
        [1, 0]
    );

    assertClose(directional, 1);

    if (classifyCriticalPoint([
        [2, 0],
        [0, 2]
    ]) !== "local minimum") {
        throw new Error("Minimum classification failed.");
    }

    if (classifyCriticalPoint([
        [-2, 0],
        [0, -2]
    ]) !== "local maximum") {
        throw new Error("Maximum classification failed.");
    }

    if (classifyCriticalPoint([
        [2, 0],
        [0, -2]
    ]) !== "saddle point") {
        throw new Error("Saddle classification failed.");
    }

    console.log("All tests passed.");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
    console.log("=".repeat(78));
    console.log(
        "MULTIVARIABLE CALCULUS: GRADIENTS, JACOBIANS, HESSIANS"
    );
    console.log("=".repeat(78));

    console.log(`
Core relationships:

Scalar field:
  f: R^n -> R

Gradient:
  ∇f = [∂f/∂x1, ..., ∂f/∂xn]

Directional derivative:
  D_u f = ∇f · u, where ||u|| = 1

Vector-valued function:
  F: R^n -> R^m

Jacobian:
  J_ij = ∂F_i / ∂x_j

Hessian:
  H_ij = ∂²f / (∂x_i ∂x_j)

First-order approximation:
  f(x + Δx) ≈ f(x) + ∇f(x)^T Δx

Second-order approximation:
  f(x + Δx) ≈ f(x) +
              ∇f(x)^T Δx +
              1/2 Δx^T H(x) Δx
`);

    demonstrateGradient();
    demonstrateDirectionalDerivatives();
    demonstrateJacobian();
    demonstrateHessian();
    demonstrateTaylor();
    demonstrateChainRule();
    demonstrateOptimization();
    demonstrateEdgeCases();
    runTests();

    console.log("\nProgram completed successfully.");
}

main();
