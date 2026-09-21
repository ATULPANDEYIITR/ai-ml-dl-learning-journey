/*
 * Linear Algebra:
 * Scalars, vectors, matrices, tensors, and matrix operations
 *
 * This file demonstrates the topic from basic numerical objects to
 * practical matrix and tensor processing. It is executable in Node.js
 * without external packages.
 */

"use strict";

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

function section(title) {
    console.log("\n" + "=".repeat(78));
    console.log(title);
    console.log("=".repeat(78));
}

function nearlyEqual(a, b, tolerance = 1e-9) {
    return Math.abs(a - b) <= tolerance;
}

function validateVector(vector, name = "vector") {
    if (!Array.isArray(vector) || vector.length === 0) {
        throw new TypeError(`${name} must be a non-empty array.`);
    }

    if (!vector.every(value => typeof value === "number" && Number.isFinite(value))) {
        throw new TypeError(`${name} must contain only finite numbers.`);
    }
}

function validateMatrix(matrix, name = "matrix") {
    if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new TypeError(`${name} must contain at least one row.`);
    }

    if (!matrix.every(Array.isArray)) {
        throw new TypeError(`${name} must contain array rows.`);
    }

    const columns = matrix[0].length;

    if (columns === 0) {
        throw new TypeError(`${name} must contain at least one column.`);
    }

    if (!matrix.every(row => row.length === columns)) {
        throw new TypeError(`${name} must be rectangular.`);
    }

    if (!matrix.every(row =>
        row.every(value => typeof value === "number" && Number.isFinite(value))
    )) {
        throw new TypeError(`${name} must contain only finite numbers.`);
    }
}

function shape(matrix) {
    validateMatrix(matrix);
    return [matrix.length, matrix[0].length];
}

function printMatrix(matrix) {
    validateMatrix(matrix);
    for (const row of matrix) {
        console.log("[ " + row.map(value => Number(value.toFixed(6))).join("  ") + " ]");
    }
}

// -----------------------------------------------------------------------------
// Scalars
// -----------------------------------------------------------------------------

function demonstrateScalars() {
    section("1. Scalars");

    const temperature = 25.5;
    const price = 1499;
    const quantity = 7;

    console.log("temperature =", temperature);
    console.log("price =", price);
    console.log("quantity =", quantity);

    console.log("temperature + 2 =", temperature + 2);
    console.log("price * 1.18 =", price * 1.18);

    // A scalar can scale every component of a vector.
    const vector = [1, 2, 3];
    const scaled = vector.map(value => 4 * value);
    console.log("4 * [1,2,3] =", scaled);
}

// -----------------------------------------------------------------------------
// Vectors
// -----------------------------------------------------------------------------

function addVectors(a, b) {
    validateVector(a, "a");
    validateVector(b, "b");

    if (a.length !== b.length) {
        throw new RangeError("Vector dimensions must match.");
    }

    return a.map((value, index) => value + b[index]);
}

function subtractVectors(a, b) {
    validateVector(a, "a");
    validateVector(b, "b");

    if (a.length !== b.length) {
        throw new RangeError("Vector dimensions must match.");
    }

    return a.map((value, index) => value - b[index]);
}

function scaleVector(scalar, vector) {
    validateVector(vector);
    if (typeof scalar !== "number" || !Number.isFinite(scalar)) {
        throw new TypeError("Scalar must be a finite number.");
    }

    return vector.map(value => scalar * value);
}

function dotProduct(a, b) {
    validateVector(a, "a");
    validateVector(b, "b");

    if (a.length !== b.length) {
        throw new RangeError("Dot product requires equal dimensions.");
    }

    return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function norm(vector, p = 2) {
    validateVector(vector);

    if (!Number.isInteger(p) || p <= 0) {
        throw new RangeError("p must be a positive integer.");
    }

    if (p === 1) {
        return vector.reduce((sum, value) => sum + Math.abs(value), 0);
    }

    return Math.pow(
        vector.reduce((sum, value) => sum + Math.pow(Math.abs(value), p), 0),
        1 / p
    );
}

function distance(a, b) {
    return norm(subtractVectors(a, b));
}

function angleInDegrees(a, b) {
    const denominator = norm(a) * norm(b);

    if (nearlyEqual(denominator, 0)) {
        throw new RangeError("Angle is undefined for a zero vector.");
    }

    let cosine = dotProduct(a, b) / denominator;

    // Clamp because floating-point calculations may produce
    // values slightly outside [-1, 1].
    cosine = Math.max(-1, Math.min(1, cosine));

    return Math.acos(cosine) * 180 / Math.PI;
}

function projectVector(a, b) {
    const denominator = dotProduct(b, b);

    if (nearlyEqual(denominator, 0)) {
        throw new RangeError("Cannot project onto a zero vector.");
    }

    return scaleVector(dotProduct(a, b) / denominator, b);
}

function demonstrateVectors() {
    section("2. Vectors");

    const a = [2, -1, 3];
    const b = [4, 5, 1];

    console.log("a =", a);
    console.log("b =", b);
    console.log("a + b =", addVectors(a, b));
    console.log("a - b =", subtractVectors(a, b));
    console.log("3a =", scaleVector(3, a));
    console.log("a · b =", dotProduct(a, b));
    console.log("||a||₂ =", norm(a));
    console.log("distance(a,b) =", distance(a, b));
    console.log("angle(a,b) =", angleInDegrees(a, b));
    console.log("projection of a onto b =", projectVector(a, b));
}

// -----------------------------------------------------------------------------
// Matrices
// -----------------------------------------------------------------------------

function addMatrices(a, b) {
    validateMatrix(a, "a");
    validateMatrix(b, "b");

    const [rowsA, columnsA] = shape(a);
    const [rowsB, columnsB] = shape(b);

    if (rowsA !== rowsB || columnsA !== columnsB) {
        throw new RangeError("Matrix dimensions must match.");
    }

    return a.map((row, i) =>
        row.map((value, j) => value + b[i][j])
    );
}

function subtractMatrices(a, b) {
    validateMatrix(a, "a");
    validateMatrix(b, "b");

    const [rowsA, columnsA] = shape(a);
    const [rowsB, columnsB] = shape(b);

    if (rowsA !== rowsB || columnsA !== columnsB) {
        throw new RangeError("Matrix dimensions must match.");
    }

    return a.map((row, i) =>
        row.map((value, j) => value - b[i][j])
    );
}

function scaleMatrix(scalar, matrix) {
    validateMatrix(matrix);

    return matrix.map(row =>
        row.map(value => scalar * value)
    );
}

function transpose(matrix) {
    validateMatrix(matrix);

    const [rows, columns] = shape(matrix);

    return Array.from(
        { length: columns },
        (_, column) =>
            Array.from(
                { length: rows },
                (_, row) => matrix[row][column]
            )
    );
}

function multiplyMatrices(a, b) {
    validateMatrix(a, "a");
    validateMatrix(b, "b");

    const [aRows, aColumns] = shape(a);
    const [bRows, bColumns] = shape(b);

    if (aColumns !== bRows) {
        throw new RangeError(
            `Cannot multiply ${aRows}x${aColumns} by ${bRows}x${bColumns}.`
        );
    }

    const result = Array.from(
        { length: aRows },
        () => Array(bColumns).fill(0)
    );

    for (let i = 0; i < aRows; i++) {
        for (let k = 0; k < aColumns; k++) {
            for (let j = 0; j < bColumns; j++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

function identityMatrix(size) {
    if (!Number.isInteger(size) || size <= 0) {
        throw new RangeError("Identity size must be positive.");
    }

    return Array.from(
        { length: size },
        (_, row) =>
            Array.from(
                { length: size },
                (_, column) => row === column ? 1 : 0
            )
    );
}

function matrixVectorMultiply(matrix, vector) {
    validateMatrix(matrix);
    validateVector(vector);

    const [, columns] = shape(matrix);

    if (columns !== vector.length) {
        throw new RangeError(
            "Matrix column count must equal vector dimension."
        );
    }

    return matrix.map(row =>
        row.reduce((sum, value, index) => sum + value * vector[index], 0)
    );
}

function demonstrateMatrices() {
    section("3. Matrices");

    const a = [
        [1, 2, 3],
        [4, 5, 6]
    ];

    const b = [
        [6, 5, 4],
        [3, 2, 1]
    ];

    console.log("A:");
    printMatrix(a);

    console.log("B:");
    printMatrix(b);

    console.log("A + B:");
    printMatrix(addMatrices(a, b));

    console.log("3A:");
    printMatrix(scaleMatrix(3, a));

    console.log("Aᵀ:");
    printMatrix(transpose(a));

    const c = [
        [7, 8],
        [9, 10],
        [11, 12]
    ];

    console.log("A @ C:");
    printMatrix(multiplyMatrices(a, c));

    console.log("I₃:");
    printMatrix(identityMatrix(3));
}

// -----------------------------------------------------------------------------
// Determinant and inverse
// -----------------------------------------------------------------------------

function determinant(matrix) {
    validateMatrix(matrix);

    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new RangeError("Determinant requires a square matrix.");
    }

    const work = matrix.map(row => row.map(Number));
    let result = 1;

    for (let column = 0; column < rows; column++) {
        let pivotRow = column;

        for (let row = column + 1; row < rows; row++) {
            if (Math.abs(work[row][column]) >
                Math.abs(work[pivotRow][column])) {
                pivotRow = row;
            }
        }

        if (nearlyEqual(work[pivotRow][column], 0)) {
            return 0;
        }

        if (pivotRow !== column) {
            [work[pivotRow], work[column]] =
                [work[column], work[pivotRow]];
            result *= -1;
        }

        const pivot = work[column][column];
        result *= pivot;

        for (let row = column + 1; row < rows; row++) {
            const factor = work[row][column] / pivot;

            for (let j = column + 1; j < rows; j++) {
                work[row][j] -= factor * work[column][j];
            }
        }
    }

    return result;
}

function rref(matrix, tolerance = 1e-10) {
    validateMatrix(matrix);

    const work = matrix.map(row => row.map(Number));
    let pivotRow = 0;

    for (let column = 0; column < work[0].length; column++) {
        if (pivotRow >= work.length) {
            break;
        }

        let bestRow = pivotRow;

        for (let row = pivotRow + 1; row < work.length; row++) {
            if (Math.abs(work[row][column]) >
                Math.abs(work[bestRow][column])) {
                bestRow = row;
            }
        }

        if (Math.abs(work[bestRow][column]) <= tolerance) {
            continue;
        }

        [work[pivotRow], work[bestRow]] =
            [work[bestRow], work[pivotRow]];

        const pivot = work[pivotRow][column];

        for (let j = 0; j < work[pivotRow].length; j++) {
            work[pivotRow][j] /= pivot;
        }

        for (let row = 0; row < work.length; row++) {
            if (row === pivotRow) {
                continue;
            }

            const factor = work[row][column];

            if (Math.abs(factor) <= tolerance) {
                continue;
            }

            for (let j = 0; j < work[row].length; j++) {
                work[row][j] -= factor * work[pivotRow][j];
            }
        }

        pivotRow++;
    }

    return work.map(row =>
        row.map(value => Math.abs(value) <= tolerance ? 0 : value)
    );
}

function inverse(matrix) {
    validateMatrix(matrix);

    const [rows, columns] = shape(matrix);

    if (rows !== columns) {
        throw new RangeError("Only square matrices can be inverted.");
    }

    const identity = identityMatrix(rows);
    const augmented = matrix.map((row, i) =>
        [...row, ...identity[i]]
    );

    const reduced = rref(augmented);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < rows; j++) {
            const expected = i === j ? 1 : 0;

            if (!nearlyEqual(reduced[i][j], expected)) {
                throw new RangeError("Matrix is singular.");
            }
        }
    }

    return reduced.map(row => row.slice(rows));
}

function demonstrateInverse() {
    section("4. Determinants and Inverse Matrices");

    const matrix = [
        [4, 7],
        [2, 6]
    ];

    console.log("A:");
    printMatrix(matrix);

    console.log("det(A) =", determinant(matrix));

    console.log("A⁻¹:");
    printMatrix(inverse(matrix));

    console.log("A @ A⁻¹:");
    printMatrix(multiplyMatrices(matrix, inverse(matrix)));
}

// -----------------------------------------------------------------------------
// Asynchronous data-processing example
// -----------------------------------------------------------------------------

function loadFeatureVector() {
    // Promise-based code demonstrates how matrix calculations can be placed
    // inside an application that obtains data asynchronously.
    return new Promise(resolve => {
        setTimeout(() => {
            resolve([0.5, 1.0, -0.5]);
        }, 10);
    });
}

async function runDenseLayer() {
    section("5. Asynchronous Application Example");

    const features = await loadFeatureVector();

    const weights = [
        [1.0, 0.2, -0.4],
        [0.5, -0.3, 0.8]
    ];

    const bias = [0.1, -0.2];

    const weighted = matrixVectorMultiply(weights, features);
    const preActivation = addVectors(weighted, bias);

    // ReLU is a common nonlinear operation after a matrix calculation.
    const activation = preActivation.map(value => Math.max(0, value));

    console.log("features =", features);
    console.log("W shape =", shape(weights));
    console.log("weighted result =", weighted);
    console.log("W x + b =", preActivation);
    console.log("ReLU(Wx+b) =", activation);
}

// -----------------------------------------------------------------------------
// Tensor implementation
// -----------------------------------------------------------------------------

class Tensor {
    constructor(data) {
        this.data = data;
        this._shape = Tensor.inferShape(data);
    }

    static inferShape(value) {
        if (!Array.isArray(value)) {
            if (typeof value !== "number" || !Number.isFinite(value)) {
                throw new TypeError("Tensor leaves must be finite numbers.");
            }

            return [];
        }

        if (value.length === 0) {
            throw new RangeError("Tensor dimensions cannot be empty.");
        }

        const childShapes = value.map(item => Tensor.inferShape(item));

        const firstShape = childShapes[0];

        if (!childShapes.every(
            current => JSON.stringify(current) === JSON.stringify(firstShape)
        )) {
            throw new RangeError("Tensor must be rectangular.");
        }

        return [value.length, ...firstShape];
    }

    get shape() {
        return [...this._shape];
    }

    get rank() {
        return this._shape.length;
    }

    mapValues(operation) {
        const visit = value => {
            if (Array.isArray(value)) {
                return value.map(visit);
            }

            return operation(value);
        };

        return new Tensor(visit(this.data));
    }

    add(other) {
        if (JSON.stringify(this.shape) !== JSON.stringify(other.shape)) {
            throw new RangeError("Tensor shapes must match.");
        }

        const addRecursive = (a, b) => {
            if (Array.isArray(a)) {
                return a.map((value, index) =>
                    addRecursive(value, b[index])
                );
            }

            return a + b;
        };

        return new Tensor(addRecursive(this.data, other.data));
    }

    scale(scalar) {
        return this.mapValues(value => scalar * value);
    }

    toString() {
        return `Tensor(shape=[${this.shape.join(", ")}], data=${JSON.stringify(this.data)})`;
    }
}

function demonstrateTensors() {
    section("6. Tensors");

    const scalar = new Tensor(5);
    const vector = new Tensor([1, 2, 3]);
    const matrix = new Tensor([
        [1, 2],
        [3, 4]
    ]);

    const threeDimensional = new Tensor([
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ]);

    console.log("scalar:", scalar.toString());
    console.log("vector:", vector.toString());
    console.log("matrix:", matrix.toString());
    console.log("3D tensor:", threeDimensional.toString());

    const other = new Tensor([
        [10, 20],
        [30, 40]
    ]);

    console.log("tensor addition:", matrix.add(other).toString());
    console.log("tensor scaling:", matrix.scale(0.5).toString());
}

// -----------------------------------------------------------------------------
// Performance demonstration
// -----------------------------------------------------------------------------

function benchmarkMatrixMultiplication(size = 100) {
    const matrixA = Array.from(
        { length: size },
        (_, i) => Array.from(
            { length: size },
            (_, j) => ((i + j) % 7) / 7
        )
    );

    const matrixB = Array.from(
        { length: size },
        (_, i) => Array.from(
            { length: size },
            (_, j) => ((i * 2 + j) % 11) / 11
        )
    );

    const start = performance.now();
    multiplyMatrices(matrixA, matrixB);
    const elapsed = performance.now() - start;

    console.log(
        `Dense ${size}x${size} multiplication took ${elapsed.toFixed(2)} ms`
    );

    console.log(
        "The naive dense algorithm performs O(n³) scalar multiplications/additions."
    );
}

// -----------------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------------

function demonstrateErrors() {
    section("7. Validation and Edge Cases");

    const tests = [
        [
            "Mismatched vector addition",
            () => addVectors([1, 2], [1, 2, 3])
        ],
        [
            "Invalid matrix multiplication",
            () => multiplyMatrices([[1, 2]], [[1, 2]])
        ],
        [
            "Inverse of singular matrix",
            () => inverse([[1, 2], [2, 4]])
        ],
        [
            "Projection onto zero vector",
            () => projectVector([1, 2], [0, 0])
        ]
    ];

    for (const [description, operation] of tests) {
        try {
            console.log(description, "=>", operation());
        } catch (error) {
            console.log(description, "=> correctly rejected:", error.message);
        }
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

async function main() {
    console.log("LINEAR ALGEBRA STUDY PROGRAM");
    console.log("Scalars, vectors, matrices, tensors, and matrix operations");

    demonstrateScalars();
    demonstrateVectors();
    demonstrateMatrices();
    demonstrateInverse();
    demonstrateTensors();
    demonstrateErrors();

    await runDenseLayer();

    section("8. Performance");
    benchmarkMatrixMultiplication(80);

    section("9. Key Structural Relationship");
    console.log("Scalar: rank 0");
    console.log("Vector: rank 1");
    console.log("Matrix: rank 2");
    console.log("Tensor: rank 3 or higher in the common terminology used here");
    console.log(
        "The central discipline is keeping dimensions, shapes, and operation compatibility explicit."
    );
}

main().catch(error => {
    console.error("Program failed:", error.message);
    process.exitCode = 1;
});
