/*
 * Eigenvalues and Eigenvectors
 * =============================
 *
 * A self-contained JavaScript study implementation covering:
 *
 * - vectors and matrices
 * - linear transformations
 * - determinants and characteristic equations
 * - eigenvalues and eigenvectors of 2x2 matrices
 * - eigenpair verification
 * - diagonalization
 * - matrix powers
 * - symmetric matrices
 * - covariance matrices
 * - PCA
 * - Rayleigh quotient
 * - power iteration
 * - Markov chains
 * - graph Laplacians
 * - numerical precision and edge cases
 *
 * The file uses only standard JavaScript and can run with Node.js.
 */

"use strict";

const EPSILON = 1e-10;

// -----------------------------------------------------------------------------
// Basic vector and matrix utilities
// -----------------------------------------------------------------------------

function assertRectangular(matrix) {
    if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new Error("Matrix must be a non-empty array.");
    }

    const columns = matrix[0].length;

    if (columns === 0) {
        throw new Error("Matrix must contain at least one column.");
    }

    for (const row of matrix) {
        if (!Array.isArray(row) || row.length !== columns) {
            throw new Error("Matrix must be rectangular.");
        }
    }
}

function shape(matrix) {
    assertRectangular(matrix);
    return [matrix.length, matrix[0].length];
}

function identityMatrix(size) {
    return Array.from({ length: size }, (_, row) =>
        Array.from({ length: size }, (_, column) =>
            row === column ? 1 : 0
        )
    );
}

function cloneMatrix(matrix) {
    return matrix.map(row => [...row]);
}

function vectorAdd(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.map((value, index) => value + b[index]);
}

function vectorSubtract(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vectors must have equal dimensions.");
    }

    return a.map((value, index) => value - b[index]);
}

function scalarMultiplyVector(scalar, vector) {
    return vector.map(value => scalar * value);
}

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

    if (length < EPSILON) {
        throw new Error("Cannot normalize a zero vector.");
    }

    return vector.map(value => value / length);
}

function matrixVectorMultiply(matrix, vector) {
    const [rows, columns] = shape(matrix);

    if (columns !== vector.length) {
        throw new Error("Matrix and vector dimensions do not match.");
    }

    return matrix.map(row =>
        row.reduce((sum, value, index) => sum + value * vector[index], 0)
    );
}

function matrixMultiply(a, b) {
    const [rowsA, columnsA] = shape(a);
    const [rowsB, columnsB] = shape(b);

    if (columnsA !== rowsB) {
        throw new Error("Matrix dimensions cannot be multiplied.");
    }

    return Array.from({ length: rowsA }, (_, row) =>
        Array.from({ length: columnsB }, (_, column) =>
            Array.from(
                { length: columnsA },
                (_, k) => a[row][k] * b[k][column]
            ).reduce((sum, value) => sum + value, 0)
        )
    );
}

function transpose(matrix) {
    const [rows, columns] = shape(matrix);

    return Array.from({ length: columns }, (_, column) =>
        Array.from({ length: rows }, (_, row) => matrix[row][column])
    );
}

function scalarMultiplyMatrix(scalar, matrix) {
    return matrix.map(row => row.map(value => scalar * value));
}

function matrixSubtract(a, b) {
    const [rowsA, columnsA] = shape(a);
    const [rowsB, columnsB] = shape(b);

    if (rowsA !== rowsB || columnsA !== columnsB) {
        throw new Error("Matrix dimensions must match.");
    }

    return a.map((row, i) =>
        row.map((value, j) => value - b[i][j])
    );
}

function trace(matrix) {
    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new Error("Trace requires a square matrix.");
    }

    let result = 0;

    for (let i = 0; i < rows; i++) {
        result += matrix[i][i];
    }

    return result;
}

function determinant(matrix) {
    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new Error("Determinant requires a square matrix.");
    }

    if (rows === 1) {
        return matrix[0][0];
    }

    if (rows === 2) {
        return matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0];
    }

    let result = 0;

    for (let column = 0; column < columns; column++) {
        const minor = matrix
            .slice(1)
            .map(row =>
                row.filter((_, index) => index !== column)
            );

        result += (
            (column % 2 === 0 ? 1 : -1)
            * matrix[0][column]
            * determinant(minor)
        );
    }

    return result;
}

function matrixPower(matrix, exponent) {
    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new Error("Matrix must be square.");
    }

    if (!Number.isInteger(exponent) || exponent < 0) {
        throw new Error("Exponent must be a non-negative integer.");
    }

    let result = identityMatrix(rows);
    let base = cloneMatrix(matrix);
    let power = exponent;

    // Exponentiation by squaring reduces the number of matrix
    // multiplications from O(k) to O(log k).
    while (power > 0) {
        if (power % 2 === 1) {
            result = matrixMultiply(result, base);
        }

        base = matrixMultiply(base, base);
        power = Math.floor(power / 2);
    }

    return result;
}

// -----------------------------------------------------------------------------
// Characteristic equation and 2x2 eigenvalues
// -----------------------------------------------------------------------------

function eigenvalues2x2(matrix) {
    const [rows, columns] = shape(matrix);

    if (rows !== 2 || columns !== 2) {
        throw new Error("This eigenvalue implementation expects a 2x2 matrix.");
    }

    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[1][0];
    const d = matrix[1][1];

    // Characteristic polynomial:
    //
    // det(A - lambda I)
    // = lambda^2 - (a + d)lambda + (ad - bc).
    //
    // JavaScript Number is a floating-point type. For matrices requiring
    // exact symbolic arithmetic or complex roots, a numerical library or
    // arbitrary-precision implementation would be appropriate.
    const coefficientB = -(a + d);
    const coefficientC = a * d - b * c;

    const discriminant =
        coefficientB * coefficientB - 4 * coefficientC;

    if (discriminant >= 0) {
        const root = Math.sqrt(discriminant);

        return [
            (-coefficientB + root) / 2,
            (-coefficientB - root) / 2
        ];
    }

    // This implementation reports complex roots as {real, imaginary}
    // objects instead of relying on a non-standard complex-number type.
    const imaginaryRoot = Math.sqrt(-discriminant);

    return [
        {
            real: -coefficientB / 2,
            imaginary: imaginaryRoot / 2
        },
        {
            real: -coefficientB / 2,
            imaginary: -imaginaryRoot / 2
        }
    ];
}

function isComplex(value) {
    return typeof value === "object" &&
        value !== null &&
        "real" in value &&
        "imaginary" in value;
}

function complexMagnitude(value) {
    return isComplex(value)
        ? Math.hypot(value.real, value.imaginary)
        : Math.abs(value);
}

function formatNumber(value, digits = 6) {
    if (isComplex(value)) {
        const sign = value.imaginary >= 0 ? "+" : "-";
        return (
            `${value.real.toFixed(digits)}` +
            `${sign}${Math.abs(value.imaginary).toFixed(digits)}i`
        );
    }

    return Number(value).toFixed(digits);
}

// -----------------------------------------------------------------------------
// Eigenvectors for real 2x2 matrices with real eigenvalues
// -----------------------------------------------------------------------------

function eigenvector2x2(matrix, eigenvalue) {
    if (isComplex(eigenvalue)) {
        throw new Error(
            "This educational eigenvector routine handles real eigenvalues only."
        );
    }

    const shifted = [
        [matrix[0][0] - eigenvalue, matrix[0][1]],
        [matrix[1][0], matrix[1][1] - eigenvalue]
    ];

    const row1 = shifted[0];
    const row2 = shifted[1];

    // A vector perpendicular to [a,b] is [-b,a].
    const candidate1 = [-row1[1], row1[0]];
    const candidate2 = [-row2[1], row2[0]];

    let candidate =
        norm(candidate1) > norm(candidate2)
            ? candidate1
            : candidate2;

    if (norm(candidate) < EPSILON) {
        throw new Error(
            "Could not determine a stable non-zero eigenvector."
        );
    }

    return normalize(candidate);
}

function eigenpairs2x2(matrix) {
    const values = eigenvalues2x2(matrix);

    return values.map(value => ({
        eigenvalue: value,
        eigenvector: isComplex(value)
            ? null
            : eigenvector2x2(matrix, value)
    }));
}

function eigenpairResidual(matrix, eigenvalue, eigenvector) {
    if (isComplex(eigenvalue)) {
        return NaN;
    }

    const left = matrixVectorMultiply(matrix, eigenvector);
    const right = scalarMultiplyVector(eigenvalue, eigenvector);

    return norm(vectorSubtract(left, right));
}

// -----------------------------------------------------------------------------
// Matrix inversion and diagonalization
// -----------------------------------------------------------------------------

function inverse(matrix) {
    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new Error("Only square matrices can be inverted.");
    }

    const identity = identityMatrix(rows);

    const augmented = matrix.map((row, i) => [
        ...row,
        ...identity[i]
    ]);

    for (let column = 0; column < rows; column++) {
        let pivotRow = column;

        for (let row = column + 1; row < rows; row++) {
            if (
                Math.abs(augmented[row][column]) >
                Math.abs(augmented[pivotRow][column])
            ) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][column]) < EPSILON) {
            throw new Error("Matrix is singular.");
        }

        [augmented[column], augmented[pivotRow]] =
            [augmented[pivotRow], augmented[column]];

        const pivot = augmented[column][column];

        for (let j = 0; j < 2 * rows; j++) {
            augmented[column][j] /= pivot;
        }

        for (let row = 0; row < rows; row++) {
            if (row === column) {
                continue;
            }

            const factor = augmented[row][column];

            for (let j = 0; j < 2 * rows; j++) {
                augmented[row][j] -= factor * augmented[column][j];
            }
        }
    }

    return augmented.map(row => row.slice(rows));
}

function diagonalMatrix(values) {
    return values.map((value, i) =>
        values.map((_, j) => i === j ? value : 0)
    );
}

function diagonalize2x2(matrix) {
    const pairs = eigenpairs2x2(matrix);

    if (pairs.some(pair => pair.eigenvector === null)) {
        throw new Error(
            "Complex eigenvectors are outside this real-valued example."
        );
    }

    const P = [
        [pairs[0].eigenvector[0], pairs[1].eigenvector[0]],
        [pairs[0].eigenvector[1], pairs[1].eigenvector[1]]
    ];

    const D = diagonalMatrix(
        pairs.map(pair => pair.eigenvalue)
    );

    const PInverse = inverse(P);

    return {
        P,
        D,
        PInverse
    };
}

// -----------------------------------------------------------------------------
// Rayleigh quotient and power iteration
// -----------------------------------------------------------------------------

function rayleighQuotient(matrix, vector) {
    const denominator = dot(vector, vector);

    if (Math.abs(denominator) < EPSILON) {
        throw new Error("Rayleigh quotient is undefined for zero vector.");
    }

    const transformed = matrixVectorMultiply(matrix, vector);

    return dot(vector, transformed) / denominator;
}

function powerIteration(
    matrix,
    initialVector = null,
    maxIterations = 5000,
    tolerance = 1e-10
) {
    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new Error("Power iteration requires a square matrix.");
    }

    let vector = initialVector
        ? [...initialVector]
        : Array(rows).fill(1);

    vector = normalize(vector);

    let previousEigenvalue = null;

    for (let iteration = 1; iteration <= maxIterations; iteration++) {
        const transformed = matrixVectorMultiply(matrix, vector);

        if (norm(transformed) < EPSILON) {
            throw new Error(
                "Iteration reached zero vector."
            );
        }

        vector = normalize(transformed);

        const eigenvalue = rayleighQuotient(matrix, vector);

        if (
            previousEigenvalue !== null &&
            Math.abs(eigenvalue - previousEigenvalue) < tolerance
        ) {
            return {
                eigenvalue,
                eigenvector: vector,
                iterations: iteration,
                converged: true
            };
        }

        previousEigenvalue = eigenvalue;
    }

    return {
        eigenvalue: rayleighQuotient(matrix, vector),
        eigenvector: vector,
        iterations: maxIterations,
        converged: false
    };
}

// -----------------------------------------------------------------------------
// Covariance and PCA
// -----------------------------------------------------------------------------

function columnMeans(data) {
    const [rows, columns] = shape(data);

    return Array.from({ length: columns }, (_, column) =>
        data.reduce((sum, row) => sum + row[column], 0) / rows
    );
}

function centerData(data) {
    const means = columnMeans(data);

    return {
        means,
        centered: data.map(row =>
            row.map((value, column) => value - means[column])
        )
    };
}

function covarianceMatrix(data) {
    const [rows, columns] = shape(data);

    if (rows < 2) {
        throw new Error("At least two observations are required.");
    }

    const { centered, means } = centerData(data);
    const covariance = Array.from(
        { length: columns },
        () => Array(columns).fill(0)
    );

    for (let i = 0; i < columns; i++) {
        for (let j = 0; j < columns; j++) {
            for (let observation = 0; observation < rows; observation++) {
                covariance[i][j] +=
                    centered[observation][i] *
                    centered[observation][j];
            }

            covariance[i][j] /= rows - 1;
        }
    }

    return {
        covariance,
        means
    };
}

function pca2D(data) {
    const { covariance, means } = covarianceMatrix(data);

    const pairs = eigenpairs2x2(covariance)
        .sort((a, b) =>
            Math.abs(b.eigenvalue) - Math.abs(a.eigenvalue)
        );

    if (pairs.some(pair => pair.eigenvector === null)) {
        throw new Error("PCA covariance matrix should have real eigenvectors.");
    }

    const eigenvalues = pairs.map(pair => pair.eigenvalue);
    const eigenvectors = pairs.map(pair => pair.eigenvector);

    const totalVariance = eigenvalues.reduce(
        (sum, value) => sum + value,
        0
    );

    const explainedVarianceRatio = eigenvalues.map(
        value => value / totalVariance
    );

    return {
        means,
        covariance,
        eigenvalues,
        eigenvectors,
        explainedVarianceRatio
    };
}

function projectOntoComponent(data, means, component) {
    return data.map(row => {
        const centered = row.map(
            (value, index) => value - means[index]
        );

        return dot(centered, component);
    });
}

function reconstructFromComponent(
    scores,
    means,
    component
) {
    return scores.map(score =>
        means.map(
            (mean, index) => mean + score * component[index]
        )
    );
}

function reconstructionError(original, reconstructed) {
    const [rows, columns] = shape(original);

    let total = 0;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
            const difference =
                original[i][j] - reconstructed[i][j];

            total += difference * difference;
        }
    }

    return total / (rows * columns);
}

// -----------------------------------------------------------------------------
// Markov chains
// -----------------------------------------------------------------------------

function stationaryDistribution(
    transitionMatrix,
    maxIterations = 10000,
    tolerance = 1e-12
) {
    const [rows, columns] = shape(transitionMatrix);

    if (rows !== columns) {
        throw new Error("Transition matrix must be square.");
    }

    for (const row of transitionMatrix) {
        if (
            row.some(value => value < -EPSILON) ||
            Math.abs(row.reduce((a, b) => a + b, 0) - 1) > EPSILON
        ) {
            throw new Error(
                "Every transition row must be non-negative and sum to 1."
            );
        }
    }

    let distribution = Array(rows).fill(1 / rows);

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const next = Array(columns).fill(0);

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < columns; j++) {
                next[j] += distribution[i] * transitionMatrix[i][j];
            }
        }

        const difference = next.reduce(
            (sum, value, index) =>
                sum + Math.abs(value - distribution[index]),
            0
        );

        distribution = next;

        if (difference < tolerance) {
            return {
                distribution,
                iterations: iteration + 1,
                converged: true
            };
        }
    }

    return {
        distribution,
        iterations: maxIterations,
        converged: false
    };
}

// -----------------------------------------------------------------------------
// Graph Laplacian
// -----------------------------------------------------------------------------

function graphLaplacian(adjacency) {
    const [rows, columns] = shape(adjacency);

    if (rows !== columns) {
        throw new Error("Adjacency matrix must be square.");
    }

    const degrees = adjacency.map(row =>
        row.reduce((sum, value) => sum + value, 0)
    );

    return adjacency.map((row, i) =>
        row.map((value, j) =>
            (i === j ? degrees[i] : 0) - value
        )
    );
}

// -----------------------------------------------------------------------------
// Output helpers
// -----------------------------------------------------------------------------

function printVector(vector, name) {
    console.log(
        `${name} = [${vector.map(value => formatNumber(value)).join(", ")}]`
    );
}

function printMatrix(matrix, name) {
    console.log(`${name} =`);

    for (const row of matrix) {
        console.log(
            `  [${row.map(value => formatNumber(value)).join(", ")}]`
        );
    }
}

// -----------------------------------------------------------------------------
// Demonstrations
// -----------------------------------------------------------------------------

function demonstrateEigenDefinition() {
    console.log("\n" + "=".repeat(78));
    console.log("1. EIGENVALUE EQUATION");
    console.log("=".repeat(78));

    const A = [
        [4, 1],
        [2, 3]
    ];

    const v = [1, 1];
    const lambda = 5;

    const Av = matrixVectorMultiply(A, v);
    const lambdaV = scalarMultiplyVector(lambda, v);

    printMatrix(A, "A");
    printVector(v, "v");
    printVector(Av, "Av");
    printVector(lambdaV, "lambda*v");

    console.log(
        "Because Av = lambda*v, v is an eigenvector and lambda is its eigenvalue."
    );
}

function demonstrateCharacteristicEquation() {
    console.log("\n" + "=".repeat(78));
    console.log("2. CHARACTERISTIC EQUATION");
    console.log("=".repeat(78));

    const A = [
        [4, 1],
        [2, 3]
    ];

    const eigenvalues = eigenvalues2x2(A);

    printMatrix(A, "A");
    console.log(`trace(A) = ${trace(A)}`);
    console.log(`det(A) = ${determinant(A)}`);

    console.log(
        "Characteristic polynomial: lambda^2 - trace(A)lambda + det(A)"
    );

    for (const value of eigenvalues) {
        console.log(`lambda = ${formatNumber(value)}`);
    }
}

function demonstrateEigenpairs() {
    console.log("\n" + "=".repeat(78));
    console.log("3. EIGENVECTORS AND RESIDUAL VALIDATION");
    console.log("=".repeat(78));

    const A = [
        [4, 1],
        [2, 3]
    ];

    const pairs = eigenpairs2x2(A);

    pairs.forEach((pair, index) => {
        console.log(`Eigenpair ${index + 1}`);
        console.log(`lambda = ${formatNumber(pair.eigenvalue)}`);
        printVector(pair.eigenvector, "v");

        console.log(
            `Residual = ${eigenpairResidual(
                A,
                pair.eigenvalue,
                pair.eigenvector
            ).toExponential(3)}`
        );
    });
}

function demonstrateDiagonalization() {
    console.log("\n" + "=".repeat(78));
    console.log("4. DIAGONALIZATION");
    console.log("=".repeat(78));

    const A = [
        [4, 1],
        [2, 3]
    ];

    const { P, D, PInverse } = diagonalize2x2(A);

    printMatrix(P, "P");
    printMatrix(D, "D");
    printMatrix(PInverse, "P^-1");

    const reconstructed = matrixMultiply(
        matrixMultiply(P, D),
        PInverse
    );

    printMatrix(reconstructed, "P D P^-1");

    console.log(
        "The decomposition A = P D P^-1 represents the same transformation "
        + "in an eigenvector-based coordinate system."
    );
}

function demonstrateMatrixPowers() {
    console.log("\n" + "=".repeat(78));
    console.log("5. MATRIX POWERS");
    console.log("=".repeat(78));

    const A = [
        [2, 1],
        [1, 2]
    ];

    const exponent = 5;

    printMatrix(
        matrixPower(A, exponent),
        `A^${exponent}`
    );

    console.log(
        "Diagonalization can make repeated powers easier because "
        + "D^k simply raises each diagonal eigenvalue to the kth power."
    );
}

function demonstratePCA() {
    console.log("\n" + "=".repeat(78));
    console.log("6. PRINCIPAL COMPONENT ANALYSIS");
    console.log("=".repeat(78));

    const data = [
        [2.0, 1.1],
        [3.0, 1.9],
        [4.0, 3.0],
        [5.0, 3.9],
        [6.0, 5.1],
        [7.0, 5.8],
        [8.0, 7.2],
        [9.0, 8.0]
    ];

    const result = pca2D(data);

    printMatrix(result.covariance, "Covariance matrix");
    printVector(result.means, "Means");

    result.eigenvalues.forEach((value, index) => {
        console.log(
            `Component ${index + 1}: ` +
            `eigenvalue=${value.toFixed(6)}, ` +
            `explained variance=${(
                result.explainedVarianceRatio[index] * 100
            ).toFixed(2)}%`
        );

        printVector(
            result.eigenvectors[index],
            `component ${index + 1}`
        );
    });

    const scores = projectOntoComponent(
        data,
        result.means,
        result.eigenvectors[0]
    );

    const reconstructed = reconstructFromComponent(
        scores,
        result.means,
        result.eigenvectors[0]
    );

    printVector(scores, "One-dimensional scores");

    console.log(
        `Mean squared reconstruction error = ${
            reconstructionError(data, reconstructed).toFixed(6)
        }`
    );

    console.log(
        "PCA chooses covariance eigenvectors as principal directions. "
        + "The largest eigenvalue identifies the direction containing "
        + "the greatest variance."
    );
}

function demonstratePowerIteration() {
    console.log("\n" + "=".repeat(78));
    console.log("7. POWER ITERATION");
    console.log("=".repeat(78));

    const A = [
        [5, 1],
        [1, 3]
    ];

    const result = powerIteration(A);

    printMatrix(A, "A");
    console.log(
        `Estimated dominant eigenvalue = ${result.eigenvalue.toFixed(10)}`
    );
    printVector(result.eigenvector, "Estimated eigenvector");
    console.log(`Iterations = ${result.iterations}`);
    console.log(`Converged = ${result.converged}`);

    console.log(
        `Residual = ${eigenpairResidual(
            A,
            result.eigenvalue,
            result.eigenvector
        ).toExponential(3)}`
    );
}

function demonstrateMarkovChain() {
    console.log("\n" + "=".repeat(78));
    console.log("8. MARKOV CHAIN STATIONARY DISTRIBUTION");
    console.log("=".repeat(78));

    const P = [
        [0.90, 0.10, 0.00],
        [0.20, 0.60, 0.20],
        [0.00, 0.30, 0.70]
    ];

    const result = stationaryDistribution(P);

    printVector(
        result.distribution,
        "stationary distribution"
    );

    console.log(`Iterations = ${result.iterations}`);
    console.log(`Converged = ${result.converged}`);

    console.log(
        "A stationary distribution satisfies pi P = pi, which connects "
        + "Markov-chain steady states with eigenvalue 1."
    );
}

function demonstrateGraphLaplacian() {
    console.log("\n" + "=".repeat(78));
    console.log("9. GRAPH LAPLACIAN");
    console.log("=".repeat(78));

    const adjacency = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ];

    const L = graphLaplacian(adjacency);

    printMatrix(L, "L = D - A");

    const ones = [1, 1, 1, 1];

    printVector(
        matrixVectorMultiply(L, ones),
        "L * 1"
    );

    console.log(
        "The zero result reflects the zero eigenvalue associated "
        + "with the constant vector for this connected undirected graph."
    );
}

function demonstrateEdgeCases() {
    console.log("\n" + "=".repeat(78));
    console.log("10. EDGE CASES");
    console.log("=".repeat(78));

    const repeatedEigenvalueMatrix = [
        [2, 1],
        [0, 2]
    ];

    printMatrix(
        repeatedEigenvalueMatrix,
        "Matrix with repeated eigenvalue"
    );

    const values = eigenvalues2x2(repeatedEigenvalueMatrix);

    values.forEach(value => {
        console.log(`lambda = ${formatNumber(value)}`);
    });

    console.log(
        "Repeated eigenvalues do not automatically provide enough "
        + "independent eigenvectors for diagonalization."
    );

    const rotation = [
        [0, -1],
        [1, 0]
    ];

    printMatrix(rotation, "90-degree rotation");

    const rotationEigenvalues = eigenvalues2x2(rotation);

    rotationEigenvalues.forEach(value => {
        console.log(
            `rotation eigenvalue = ${formatNumber(value)}`
        );
    });

    console.log(
        "The rotation demonstrates that real matrices can have "
        + "complex eigenvalues."
    );
}

function main() {
    console.log("EIGENVALUES & EIGENVECTORS");
    console.log(
        "Eigen decomposition, characteristic equations, and ML applications"
    );

    demonstrateEigenDefinition();
    demonstrateCharacteristicEquation();
    demonstrateEigenpairs();
    demonstrateDiagonalization();
    demonstrateMatrixPowers();
    demonstratePCA();
    demonstratePowerIteration();
    demonstrateMarkovChain();
    demonstrateGraphLaplacian();
    demonstrateEdgeCases();

    console.log("\n" + "=".repeat(78));
    console.log("END OF STUDY PROGRAM");
    console.log("=".repeat(78));
}

main();
