/*
 * Advanced Linear Algebra
 * ========================
 *
 * A self-contained JavaScript study implementation covering:
 * - vectors and inner products
 * - matrices and matrix multiplication
 * - linear systems
 * - Gaussian elimination
 * - determinants and inverses
 * - rank and null spaces
 * - projections and Gram-Schmidt
 * - QR decomposition
 * - least squares
 * - eigenvalue estimation
 * - symmetric eigenproblems
 * - quadratic forms
 * - graph Laplacians
 * - PCA
 * - iterative methods
 * - numerical validation and edge cases
 *
 * Run with:
 *     node advanced_linear_algebra.js
 *
 * No external npm packages are required.
 */

"use strict";

const EPSILON = 1e-10;

function isClose(a, b, tolerance = EPSILON) {
    return Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b));
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
        throw new Error("Cannot normalize the zero vector.");
    }
    return vector.map(value => value / length);
}

function addVectors(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vector dimensions must match.");
    }
    return a.map((value, index) => value + b[index]);
}

function subtractVectors(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vector dimensions must match.");
    }
    return a.map((value, index) => value - b[index]);
}

function scaleVector(scalar, vector) {
    return vector.map(value => scalar * value);
}

function formatVector(vector) {
    return `[${vector.map(value => value.toFixed(6)).join(", ")}]`;
}

class Matrix {
    constructor(data) {
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error("Matrix cannot be empty.");
        }

        const columns = data[0].length;

        if (columns === 0 || data.some(row => row.length !== columns)) {
            throw new Error("Matrix must be rectangular.");
        }

        this.data = data.map(row => row.map(Number));
        this.rows = data.length;
        this.cols = columns;
    }

    static identity(n) {
        return new Matrix(
            Array.from({ length: n }, (_, i) =>
                Array.from({ length: n }, (_, j) => i === j ? 1 : 0)
            )
        );
    }

    static zeros(rows, cols) {
        return new Matrix(
            Array.from({ length: rows }, () => Array(cols).fill(0))
        );
    }

    clone() {
        return new Matrix(this.data.map(row => [...row]));
    }

    get(i, j) {
        return this.data[i][j];
    }

    set(i, j, value) {
        this.data[i][j] = value;
    }

    transpose() {
        return new Matrix(
            Array.from(
                { length: this.cols },
                (_, j) => Array.from(
                    { length: this.rows },
                    (_, i) => this.data[i][j]
                )
            )
        );
    }

    add(other) {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Matrix dimensions must match.");
        }

        return new Matrix(
            this.data.map((row, i) =>
                row.map((value, j) => value + other.data[i][j])
            )
        );
    }

    multiply(other) {
        if (typeof other === "number") {
            return new Matrix(
                this.data.map(row => row.map(value => value * other))
            );
        }

        if (this.cols !== other.rows) {
            throw new Error("Incompatible matrix dimensions.");
        }

        const result = Matrix.zeros(this.rows, other.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let k = 0; k < this.cols; k++) {
                for (let j = 0; j < other.cols; j++) {
                    result.data[i][j] +=
                        this.data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    multiplyVector(vector) {
        if (this.cols !== vector.length) {
            throw new Error("Matrix/vector dimensions are incompatible.");
        }

        return this.data.map(row =>
            row.reduce((sum, value, j) => sum + value * vector[j], 0)
        );
    }

    trace() {
        if (this.rows !== this.cols) {
            throw new Error("Trace requires a square matrix.");
        }

        return this.data.reduce((sum, row, i) => sum + row[i], 0);
    }

    toString() {
        return this.data
            .map(row =>
                `[${row.map(value => value.toFixed(5).padStart(10)).join(" ")}]`
            )
            .join("\n");
    }
}

function rref(matrix) {
    const a = matrix.clone().data;
    const rows = matrix.rows;
    const cols = matrix.cols;
    const pivots = [];
    let pivotRow = 0;

    for (let col = 0; col < cols && pivotRow < rows; col++) {
        let bestRow = pivotRow;

        for (let row = pivotRow + 1; row < rows; row++) {
            if (Math.abs(a[row][col]) > Math.abs(a[bestRow][col])) {
                bestRow = row;
            }
        }

        if (Math.abs(a[bestRow][col]) < EPSILON) {
            continue;
        }

        [a[pivotRow], a[bestRow]] = [a[bestRow], a[pivotRow]];

        const pivot = a[pivotRow][col];

        for (let j = 0; j < cols; j++) {
            a[pivotRow][j] /= pivot;
        }

        for (let row = 0; row < rows; row++) {
            if (row === pivotRow) {
                continue;
            }

            const factor = a[row][col];

            if (Math.abs(factor) < EPSILON) {
                continue;
            }

            for (let j = 0; j < cols; j++) {
                a[row][j] -= factor * a[pivotRow][j];
            }
        }

        pivots.push(col);
        pivotRow++;
    }

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (Math.abs(a[i][j]) < EPSILON) {
                a[i][j] = 0;
            }
        }
    }

    return {
        matrix: new Matrix(a),
        pivots
    };
}

function rank(matrix) {
    return rref(matrix).pivots.length;
}

function determinant(matrix) {
    if (matrix.rows !== matrix.cols) {
        throw new Error("Determinant requires a square matrix.");
    }

    const a = matrix.clone().data;
    const n = matrix.rows;
    let result = 1;
    let sign = 1;

    for (let col = 0; col < n; col++) {
        let pivotRow = col;

        for (let row = col + 1; row < n; row++) {
            if (Math.abs(a[row][col]) > Math.abs(a[pivotRow][col])) {
                pivotRow = row;
            }
        }

        if (Math.abs(a[pivotRow][col]) < EPSILON) {
            return 0;
        }

        if (pivotRow !== col) {
            [a[pivotRow], a[col]] = [a[col], a[pivotRow]];
            sign *= -1;
        }

        const pivot = a[col][col];
        result *= pivot;

        for (let row = col + 1; row < n; row++) {
            const factor = a[row][col] / pivot;

            for (let j = col; j < n; j++) {
                a[row][j] -= factor * a[col][j];
            }
        }
    }

    return sign * result;
}

function inverse(matrix) {
    if (matrix.rows !== matrix.cols) {
        throw new Error("Inverse requires a square matrix.");
    }

    const n = matrix.rows;

    const augmented = matrix.data.map((row, i) => [
        ...row,
        ...Matrix.identity(n).data[i]
    ]);

    const reduced = rref(new Matrix(augmented)).matrix;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const expected = i === j ? 1 : 0;
            if (!isClose(reduced.data[i][j], expected, 1e-8)) {
                throw new Error("Matrix is singular.");
            }
        }
    }

    return new Matrix(reduced.data.map(row => row.slice(n)));
}

function solveLinearSystem(A, b) {
    if (A.rows !== b.length) {
        throw new Error("Right-hand side dimension mismatch.");
    }

    const augmented = new Matrix(
        A.data.map((row, i) => [...row, b[i]])
    );

    const reduced = rref(augmented).matrix;

    for (const row of reduced.data) {
        const allZero = row
            .slice(0, A.cols)
            .every(value => Math.abs(value) < EPSILON);

        if (allZero && Math.abs(row[A.cols]) > EPSILON) {
            throw new Error("System is inconsistent.");
        }
    }

    if (rank(A) < A.cols) {
        throw new Error("System does not have a unique solution.");
    }

    return reduced.data
        .slice(0, A.cols)
        .map(row => row[A.cols]);
}

function projection(vector, direction) {
    const denominator = dot(direction, direction);

    if (denominator < EPSILON) {
        throw new Error("Cannot project onto the zero vector.");
    }

    return scaleVector(
        dot(vector, direction) / denominator,
        direction
    );
}

function gramSchmidt(vectors) {
    const basis = [];

    for (const original of vectors) {
        let residual = [...original];

        for (const q of basis) {
            residual = subtractVectors(
                residual,
                scaleVector(dot(residual, q), q)
            );
        }

        const length = norm(residual);

        if (length > EPSILON) {
            basis.push(scaleVector(1 / length, residual));
        }
    }

    return basis;
}

function qrDecomposition(A) {
    if (A.rows < A.cols) {
        throw new Error("This implementation expects rows >= columns.");
    }

    const columns = Array.from(
        { length: A.cols },
        (_, j) => A.data.map(row => row[j])
    );

    const qColumns = gramSchmidt(columns);

    if (qColumns.length !== A.cols) {
        throw new Error("Columns are linearly dependent.");
    }

    const Q = new Matrix(
        Array.from(
            { length: A.rows },
            (_, i) => qColumns.map(column => column[i])
        )
    );

    const R = Q.transpose().multiply(A);

    return { Q, R };
}

function leastSquares(A, b) {
    /*
     * Normal equations:
     *
     *     A^T A x = A^T b
     *
     * This is conceptually simple, but QR or SVD is preferred for
     * numerically sensitive problems because A^T A squares the
     * condition number.
     */
    const At = A.transpose();
    const normalMatrix = At.multiply(A);
    const normalVector = At.multiplyVector(b);

    return solveLinearSystem(normalMatrix, normalVector);
}

function rayleighQuotient(A, x) {
    const denominator = dot(x, x);

    if (denominator < EPSILON) {
        throw new Error("Rayleigh quotient requires a non-zero vector.");
    }

    return dot(x, A.multiplyVector(x)) / denominator;
}

function powerIteration(A, iterations = 1000, tolerance = 1e-10) {
    if (A.rows !== A.cols) {
        throw new Error("Power iteration requires a square matrix.");
    }

    let x = normalize(
        Array.from({ length: A.rows }, (_, i) => i + 1)
    );

    let eigenvalue = 0;

    for (let iteration = 0; iteration < iterations; iteration++) {
        let next = A.multiplyVector(x);

        if (norm(next) < EPSILON) {
            throw new Error("Power iteration reached the zero vector.");
        }

        next = normalize(next);

        const nextEigenvalue = rayleighQuotient(A, next);

        if (norm(subtractVectors(next, x)) < tolerance) {
            return {
                eigenvalue: nextEigenvalue,
                eigenvector: next,
                iterations: iteration + 1
            };
        }

        x = next;
        eigenvalue = nextEigenvalue;
    }

    return {
        eigenvalue,
        eigenvector: x,
        iterations
    };
}

function quadraticForm(A, x) {
    return dot(x, A.multiplyVector(x));
}

function isSymmetric(A, tolerance = 1e-9) {
    if (A.rows !== A.cols) {
        return false;
    }

    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j < A.cols; j++) {
            if (Math.abs(A.data[i][j] - A.data[j][i]) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

function graphLaplacian(adjacency) {
    if (adjacency.rows !== adjacency.cols) {
        throw new Error("Adjacency matrix must be square.");
    }

    const degrees = adjacency.data.map(row =>
        row.reduce((sum, value) => sum + value, 0)
    );

    return new Matrix(
        adjacency.data.map((row, i) =>
            row.map((value, j) =>
                (i === j ? degrees[i] : 0) - value
            )
        )
    );
}

function covarianceMatrix(data) {
    if (data.length < 2) {
        throw new Error("At least two observations are required.");
    }

    const dimensions = data[0].length;

    if (data.some(row => row.length !== dimensions)) {
        throw new Error("Observations must have equal dimensions.");
    }

    const means = Array.from(
        { length: dimensions },
        (_, j) =>
            data.reduce((sum, row) => sum + row[j], 0) / data.length
    );

    const centered = data.map(row =>
        row.map((value, j) => value - means[j])
    );

    const covariance = Matrix.zeros(dimensions, dimensions);

    for (let i = 0; i < dimensions; i++) {
        for (let j = 0; j < dimensions; j++) {
            covariance.data[i][j] =
                centered.reduce(
                    (sum, row) => sum + row[i] * row[j],
                    0
                ) / (data.length - 1);
        }
    }

    return {
        means,
        centered,
        covariance
    };
}

function pca(data) {
    /*
     * For a complete production PCA, a robust eigensolver or SVD should be
     * used. This function performs the data-centering and covariance stages,
     * which are the conceptual core of PCA.
     */
    const result = covarianceMatrix(data);

    return result;
}

function conjugateGradient(A, b, tolerance = 1e-10) {
    if (A.rows !== A.cols) {
        throw new Error("CG requires a square matrix.");
    }

    if (!isSymmetric(A)) {
        throw new Error("CG requires a symmetric matrix.");
    }

    const n = A.rows;
    let x = Array(n).fill(0);
    let r = subtractVectors(b, A.multiplyVector(x));
    let p = [...r];

    let rsOld = dot(r, r);

    if (Math.sqrt(rsOld) < tolerance) {
        return { solution: x, iterations: 0 };
    }

    for (let iteration = 1; iteration <= n * 10; iteration++) {
        const Ap = A.multiplyVector(p);
        const denominator = dot(p, Ap);

        if (Math.abs(denominator) < EPSILON) {
            throw new Error("Degenerate CG search direction.");
        }

        const alpha = rsOld / denominator;

        x = addVectors(x, scaleVector(alpha, p));
        r = subtractVectors(r, scaleVector(alpha, Ap));

        const rsNew = dot(r, r);

        if (Math.sqrt(rsNew) < tolerance) {
            return {
                solution: x,
                iterations: iteration
            };
        }

        const beta = rsNew / rsOld;

        p = addVectors(r, scaleVector(beta, p));
        rsOld = rsNew;
    }

    return {
        solution: x,
        iterations: n * 10
    };
}

function section(title) {
    console.log("\n" + "=".repeat(78));
    console.log(title);
    console.log("=".repeat(78));
}

function runDemonstrations() {
    section("1. Vectors and inner products");

    const v = [3, 4];
    const w = [1, 2];

    console.log("v =", formatVector(v));
    console.log("w =", formatVector(w));
    console.log("v + w =", formatVector(addVectors(v, w)));
    console.log("v . w =", dot(v, w));
    console.log("||v|| =", norm(v));
    console.log("projection =", formatVector(projection(v, w)));

    section("2. Matrix arithmetic");

    const A = new Matrix([
        [1, 2],
        [3, 4]
    ]);

    const B = new Matrix([
        [5, 6],
        [7, 8]
    ]);

    console.log("A:");
    console.log(A.toString());

    console.log("\nA + B:");
    console.log(A.add(B).toString());

    console.log("\nA * B:");
    console.log(A.multiply(B).toString());

    console.log("\nA^T:");
    console.log(A.transpose().toString());

    console.log("\ndet(A) =", determinant(A));
    console.log("trace(A) =", A.trace());

    section("3. Linear systems");

    const systemMatrix = new Matrix([
        [2, 1],
        [1, -1]
    ]);

    const rhs = [5, 1];

    console.log(
        "solution =",
        formatVector(solveLinearSystem(systemMatrix, rhs))
    );

    section("4. Rank, determinant, and inverse");

    console.log("rank =", rank(A));
    console.log("A^-1:");
    console.log(inverse(A).toString());

    console.log("\nA * A^-1:");
    console.log(A.multiply(inverse(A)).toString());

    section("5. Orthogonality and QR");

    const basis = gramSchmidt([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]);

    basis.forEach((vector, index) => {
        console.log(`q${index + 1} =`, formatVector(vector));
    });

    const QR = qrDecomposition(new Matrix([
        [1, 1],
        [1, 0],
        [0, 1]
    ]));

    console.log("\nQ:");
    console.log(QR.Q.toString());

    console.log("\nR:");
    console.log(QR.R.toString());

    section("6. Least squares");

    const observations = [
        [0, 1.1],
        [1, 2.9],
        [2, 5.2],
        [3, 6.8],
        [4, 9.1]
    ];

    const regressionA = new Matrix(
        observations.map(([x]) => [1, x])
    );

    const regressionB = observations.map(([, y]) => y);

    const coefficients = leastSquares(regressionA, regressionB);

    console.log(
        "intercept, slope =",
        formatVector(coefficients)
    );

    section("7. Eigenvalue estimation");

    const eigenMatrix = new Matrix([
        [4, 1],
        [2, 3]
    ]);

    const eigenResult = powerIteration(eigenMatrix);

    console.log("dominant eigenvalue =", eigenResult.eigenvalue);
    console.log(
        "dominant eigenvector =",
        formatVector(eigenResult.eigenvector)
    );
    console.log("iterations =", eigenResult.iterations);

    section("8. Quadratic forms");

    const positiveMatrix = new Matrix([
        [4, 1],
        [1, 3]
    ]);

    const point = [2, -1];

    console.log(
        "x^T A x =",
        quadraticForm(positiveMatrix, point)
    );
    console.log(
        "symmetric =",
        isSymmetric(positiveMatrix)
    );

    section("9. PCA preprocessing");

    const dataset = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ];

    const pcaResult = pca(dataset);

    console.log("feature means =", formatVector(pcaResult.means));
    console.log("\nCovariance matrix:");
    console.log(pcaResult.covariance.toString());

    section("10. Graph Laplacian");

    const adjacency = new Matrix([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ]);

    const L = graphLaplacian(adjacency);

    console.log("L = D - A:");
    console.log(L.toString());

    console.log(
        "\nL * 1 =",
        formatVector(L.multiplyVector([1, 1, 1, 1]))
    );

    section("11. Conjugate Gradient");

    const spd = new Matrix([
        [4, 1],
        [1, 3]
    ]);

    const cgResult = conjugateGradient(spd, [1, 2]);

    console.log(
        "solution =",
        formatVector(cgResult.solution)
    );
    console.log("iterations =", cgResult.iterations);

    section("12. JavaScript-specific numerical considerations");

    console.log(`
JavaScript Number uses IEEE-754 double-precision floating-point arithmetic.

Consequences:
- Most ordinary calculations have approximately 15-17 significant decimal
  digits of precision.
- Decimal values such as 0.1 cannot generally be represented exactly.
- Equality checks should therefore use tolerances for numerical algorithms.
- Large matrices are expensive because ordinary JavaScript arrays contain
  object-like elements and do not provide the same memory model as a compact
  native matrix representation.
- Typed arrays such as Float64Array are useful for performance-sensitive
  numerical applications.
- Production scientific workloads commonly use specialized numerical
  libraries rather than hand-written dense matrix algorithms.
`);

    section("13. Edge cases");

    try {
        inverse(new Matrix([
            [1, 2],
            [2, 4]
        ]));
    } catch (error) {
        console.log("Singular inverse rejected:", error.message);
    }

    try {
        normalize([0, 0]);
    } catch (error) {
        console.log("Zero-vector normalization rejected:", error.message);
    }
}

function runTests() {
    section("14. Correctness tests");

    const A = new Matrix([
        [1, 2],
        [3, 4]
    ]);

    if (!isClose(determinant(A), -2)) {
        throw new Error("Determinant test failed.");
    }

    const identity = A.multiply(inverse(A));

    for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
            const expected = i === j ? 1 : 0;

            if (!isClose(identity.data[i][j], expected, 1e-8)) {
                throw new Error("Inverse test failed.");
            }
        }
    }

    const solution = solveLinearSystem(
        new Matrix([
            [2, 1],
            [1, -1]
        ]),
        [5, 1]
    );

    if (!isClose(solution[0], 2) || !isClose(solution[1], 1)) {
        throw new Error("Linear-system test failed.");
    }

    const orthonormal = gramSchmidt([
        [1, 1],
        [1, -1]
    ]);

    if (!isClose(norm(orthonormal[0]), 1)) {
        throw new Error("Normalization test failed.");
    }

    if (!isClose(dot(orthonormal[0], orthonormal[1]), 0, 1e-8)) {
        throw new Error("Orthogonality test failed.");
    }

    console.log("All JavaScript tests passed.");
}

runDemonstrations();
runTests();
