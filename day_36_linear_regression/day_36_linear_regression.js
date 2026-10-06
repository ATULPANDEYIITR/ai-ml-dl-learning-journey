"use strict";

/*
 * Linear Regression in JavaScript
 *
 * This Node.js program models a practical revenue forecasting workflow.
 * It demonstrates:
 * - simple regression with one predictor
 * - multiple regression with several predictors
 * - intercept and coefficient estimation
 * - matrix-based ordinary least squares
 * - prediction and residual analysis
 * - R-squared and adjusted R-squared
 * - gradient descent
 * - event-driven model lifecycle
 * - validation and failure handling
 *
 * Run with:
 *   node linear-regression.js
 *
 * No external npm packages are required.
 */

// ---------------------------------------------------------------------------
// Numerical utilities
// ---------------------------------------------------------------------------

function mean(values) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new Error("Mean requires a non-empty array.");
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function dot(a, b) {
    if (a.length !== b.length) {
        throw new Error("Dot product requires equal-length vectors.");
    }

    return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function transpose(matrix) {
    if (matrix.length === 0) {
        return [];
    }

    const width = matrix[0].length;

    if (matrix.some(row => row.length !== width)) {
        throw new Error("Matrix must be rectangular.");
    }

    return Array.from({ length: width }, (_, column) =>
        matrix.map(row => row[column])
    );
}

function matrixMultiply(a, b) {
    if (a.length === 0 || b.length === 0) {
        throw new Error("Matrices cannot be empty.");
    }

    const aWidth = a[0].length;
    const bHeight = b.length;
    const bWidth = b[0].length;

    if (aWidth !== bHeight) {
        throw new Error("Matrix dimensions are incompatible.");
    }

    if (a.some(row => row.length !== aWidth)) {
        throw new Error("Matrix A must be rectangular.");
    }

    if (b.some(row => row.length !== bWidth)) {
        throw new Error("Matrix B must be rectangular.");
    }

    const result = [];

    for (let row = 0; row < a.length; row++) {
        const outputRow = [];

        for (let column = 0; column < bWidth; column++) {
            let total = 0;

            for (let k = 0; k < aWidth; k++) {
                total += a[row][k] * b[k][column];
            }

            outputRow.push(total);
        }

        result.push(outputRow);
    }

    return result;
}

function identityMatrix(size) {
    return Array.from(
        { length: size },
        (_, row) =>
            Array.from(
                { length: size },
                (_, column) => row === column ? 1 : 0
            )
    );
}

function inverse(matrix) {
    const n = matrix.length;

    if (
        n === 0 ||
        matrix.some(row => row.length !== n)
    ) {
        throw new Error("Only non-empty square matrices can be inverted.");
    }

    const identity = identityMatrix(n);

    const augmented = matrix.map((row, index) => [
        ...row.map(Number),
        ...identity[index]
    ]);

    for (let column = 0; column < n; column++) {
        let pivotRow = column;

        for (let row = column + 1; row < n; row++) {
            if (
                Math.abs(augmented[row][column]) >
                Math.abs(augmented[pivotRow][column])
            ) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][column]) < 1e-12) {
            throw new Error(
                "Design matrix is singular or nearly singular."
            );
        }

        if (pivotRow !== column) {
            [augmented[pivotRow], augmented[column]] =
                [augmented[column], augmented[pivotRow]];
        }

        const pivot = augmented[column][column];

        for (let j = 0; j < 2 * n; j++) {
            augmented[column][j] /= pivot;
        }

        for (let row = 0; row < n; row++) {
            if (row === column) {
                continue;
            }

            const factor = augmented[row][column];

            for (let j = 0; j < 2 * n; j++) {
                augmented[row][j] -= factor * augmented[column][j];
            }
        }
    }

    return augmented.map(row => row.slice(n));
}

// ---------------------------------------------------------------------------
// Regression model
// ---------------------------------------------------------------------------

class LinearRegression {
    constructor(featureNames = []) {
        this.featureNames = [...featureNames];
        this.intercept = null;
        this.coefficients = [];
        this.metrics = null;
        this.residuals = [];
        this.predictions = [];
    }

    validate(X, y) {
        if (!Array.isArray(X) || X.length === 0) {
            throw new Error("X must contain observations.");
        }

        if (!Array.isArray(y) || y.length !== X.length) {
            throw new Error("X and y must contain the same number of rows.");
        }

        const featureCount = X[0].length;

        if (featureCount === 0) {
            throw new Error("At least one feature is required.");
        }

        if (X.some(row => row.length !== featureCount)) {
            throw new Error("Every feature row must have equal length.");
        }

        if (X.length <= featureCount) {
            throw new Error(
                "More observations than parameters are required."
            );
        }

        for (const row of X) {
            for (const value of row) {
                if (!Number.isFinite(value)) {
                    throw new Error("Features must contain finite numbers.");
                }
            }
        }

        for (const value of y) {
            if (!Number.isFinite(value)) {
                throw new Error("Target values must be finite numbers.");
            }
        }
    }

    designMatrix(X) {
        return X.map(row => [1, ...row]);
    }

    fit(X, y) {
        this.validate(X, y);

        if (
            this.featureNames.length > 0 &&
            this.featureNames.length !== X[0].length
        ) {
            throw new Error(
                "featureNames does not match the number of predictors."
            );
        }

        if (this.featureNames.length === 0) {
            this.featureNames = X[0].map(
                (_, index) => `x${index + 1}`
            );
        }

        const design = this.designMatrix(X);
        const designT = transpose(design);

        // OLS:
        // beta = (X'X)^(-1) X'y
        const xtx = matrixMultiply(designT, design);
        const xtxInverse = inverse(xtx);

        const yColumn = y.map(value => [value]);
        const xty = matrixMultiply(designT, yColumn);

        const beta = matrixMultiply(
            xtxInverse,
            xty
        ).map(row => row[0]);

        this.intercept = beta[0];
        this.coefficients = beta.slice(1);

        this.predictions = X.map(row => this.predict(row));

        this.residuals = y.map(
            (actual, index) => actual - this.predictions[index]
        );

        this.metrics = calculateMetrics(
            y,
            this.predictions,
            X[0].length
        );

        return this;
    }

    predict(row) {
        if (this.intercept === null) {
            throw new Error("Model has not been fitted.");
        }

        if (row.length !== this.coefficients.length) {
            throw new Error("Prediction feature count does not match model.");
        }

        return this.intercept + dot(this.coefficients, row);
    }

    equation() {
        if (this.intercept === null) {
            return "Model is not fitted.";
        }

        let expression = `y = ${this.intercept.toFixed(4)}`;

        this.coefficients.forEach((coefficient, index) => {
            const sign = coefficient >= 0 ? "+" : "-";
            expression +=
                ` ${sign} ${Math.abs(coefficient).toFixed(4)}` +
                `*${this.featureNames[index]}`;
        });

        return expression;
    }

    summary() {
        console.log("\nRegression equation:");
        console.log(this.equation());

        console.log("\nCoefficient estimates:");

        console.table(
            this.featureNames.map((name, index) => ({
                feature: name,
                coefficient: Number(
                    this.coefficients[index].toFixed(6)
                )
            }))
        );

        console.log("Intercept:", this.intercept.toFixed(6));
        console.log(
            "R-squared:",
            this.metrics.rSquared.toFixed(6)
        );
        console.log(
            "Adjusted R-squared:",
            this.metrics.adjustedRSquared.toFixed(6)
        );
        console.log(
            "RMSE:",
            this.metrics.rmse.toFixed(6)
        );
        console.log(
            "MAE:",
            this.metrics.mae.toFixed(6)
        );
    }
}

function calculateMetrics(y, predictions, predictorCount) {
    const residuals = y.map(
        (actual, index) => actual - predictions[index]
    );

    const sse = residuals.reduce(
        (sum, error) => sum + error * error,
        0
    );

    const yMean = mean(y);

    const sst = y.reduce(
        (sum, actual) =>
            sum + Math.pow(actual - yMean, 2),
        0
    );

    const rSquared =
        sst < 1e-12
            ? (sse < 1e-12 ? 1 : 0)
            : 1 - sse / sst;

    const n = y.length;
    const p = predictorCount;

    const adjustedRSquared =
        n > p + 1
            ? 1 - ((1 - rSquared) * (n - 1)) / (n - p - 1)
            : NaN;

    const mse = sse / n;
    const rmse = Math.sqrt(mse);

    const mae =
        residuals.reduce(
            (sum, error) => sum + Math.abs(error),
            0
        ) / n;

    return {
        sse,
        mse,
        rmse,
        mae,
        rSquared,
        adjustedRSquared
    };
}

// ---------------------------------------------------------------------------
// Simple regression with a direct formula
// ---------------------------------------------------------------------------

function simpleRegression(x, y) {
    if (x.length !== y.length || x.length < 2) {
        throw new Error(
            "Simple regression requires equal-length arrays with at least two observations."
        );
    }

    const xMean = mean(x);
    const yMean = mean(y);

    const numerator = x.reduce(
        (sum, value, index) =>
            sum + (value - xMean) * (y[index] - yMean),
        0
    );

    const denominator = x.reduce(
        (sum, value) =>
            sum + Math.pow(value - xMean, 2),
        0
    );

    if (denominator < 1e-12) {
        throw new Error(
            "Slope cannot be estimated when x is constant."
        );
    }

    const slope = numerator / denominator;
    const intercept = yMean - slope * xMean;

    const predictions = x.map(
        value => intercept + slope * value
    );

    const metrics = calculateMetrics(
        y,
        predictions,
        1
    );

    return {
        intercept,
        slope,
        predictions,
        metrics
    };
}

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------

function standardize(X) {
    const columns = transpose(X);

    const means = columns.map(column => mean(column));

    const scales = columns.map(column => {
        const variance =
            column.reduce(
                (sum, value) =>
                    sum + Math.pow(value - mean(column), 2),
                0
            ) / column.length;

        const scale = Math.sqrt(variance);

        if (scale < 1e-12) {
            throw new Error("Cannot standardize a constant feature.");
        }

        return scale;
    });

    const transformed = X.map(row =>
        row.map(
            (value, index) =>
                (value - means[index]) / scales[index]
        )
    );

    return {
        transformed,
        means,
        scales
    };
}

function gradientDescent(
    X,
    y,
    learningRate = 0.03,
    epochs = 5000
) {
    const n = X.length;
    const p = X[0].length;

    let intercept = 0;
    let coefficients = new Array(p).fill(0);
    const losses = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
        const predictions = X.map(
            row => intercept + dot(coefficients, row)
        );

        const errors = predictions.map(
            (prediction, index) => prediction - y[index]
        );

        const loss =
            errors.reduce(
                (sum, error) => sum + error * error,
                0
            ) / n;

        losses.push(loss);

        const interceptGradient =
            (2 / n) * errors.reduce(
                (sum, error) => sum + error,
                0
            );

        const gradients = X[0].map((_, featureIndex) =>
            (2 / n) *
            errors.reduce(
                (sum, error, rowIndex) =>
                    sum + error * X[rowIndex][featureIndex],
                0
            )
        );

        intercept -= learningRate * interceptGradient;

        coefficients = coefficients.map(
            (coefficient, index) =>
                coefficient -
                learningRate * gradients[index]
        );
    }

    return {
        intercept,
        coefficients,
        losses
    };
}

// ---------------------------------------------------------------------------
// Event-driven workflow model
// ---------------------------------------------------------------------------

class RegressionWorkflow {
    constructor() {
        this.listeners = new Map();
        this.model = null;
    }

    on(eventName, listener) {
        if (!this.listeners.has(eventName)) {
            this.listeners.set(eventName, []);
        }

        this.listeners.get(eventName).push(listener);
    }

    emit(eventName, payload) {
        const listeners = this.listeners.get(eventName) || [];

        for (const listener of listeners) {
            listener(payload);
        }
    }

    train(X, y, featureNames) {
        this.emit("trainingStarted", {
            observations: X.length,
            features: X[0].length
        });

        try {
            this.model = new LinearRegression(featureNames)
                .fit(X, y);

            this.emit("trainingCompleted", {
                equation: this.model.equation(),
                metrics: this.model.metrics
            });

            return this.model;
        } catch (error) {
            this.emit("trainingFailed", {
                message: error.message
            });

            throw error;
        }
    }
}

// ---------------------------------------------------------------------------
// Demonstrations
// ---------------------------------------------------------------------------

function demonstrateSimpleRegression() {
    console.log("=".repeat(78));
    console.log("SIMPLE LINEAR REGRESSION");
    console.log("=".repeat(78));

    const advertisingSpend = [
        10, 20, 30, 40, 50,
        60, 70, 80
    ];

    const sales = [
        110, 125, 142, 157,
        175, 193, 205, 225
    ];

    const result = simpleRegression(
        advertisingSpend,
        sales
    );

    console.log(
        `Intercept: ${result.intercept.toFixed(4)}`
    );

    console.log(
        `Coefficient: ${result.slope.toFixed(4)}`
    );

    console.log(
        `R²: ${result.metrics.rSquared.toFixed(4)}`
    );

    const newSpend = 65;

    console.log(
        `Predicted sales at advertising spend ${newSpend}: ` +
        `${(result.intercept + result.slope * newSpend).toFixed(2)}`
    );
}

function demonstrateMultipleRegression() {
    console.log("\n" + "=".repeat(78));
    console.log("MULTIPLE LINEAR REGRESSION");
    console.log("=".repeat(78));

    // Revenue is modeled from:
    // marketing spend, sales staff count, and website conversion rate.
    const X = [
        [20, 4, 2.1],
        [25, 5, 2.3],
        [30, 5, 2.7],
        [35, 6, 2.9],
        [40, 7, 3.1],
        [45, 7, 3.3],
        [50, 8, 3.5],
        [55, 9, 3.7],
        [60, 9, 4.0],
        [65, 10, 4.2],
        [70, 10, 4.4],
        [75, 11, 4.6]
    ];

    const revenue = [
        180, 205, 230, 252,
        278, 300, 325, 352,
        380, 410, 438, 465
    ];

    const workflow = new RegressionWorkflow();

    workflow.on("trainingStarted", event => {
        console.log(
            `Training started: ${event.observations} observations, ` +
            `${event.features} predictors`
        );
    });

    workflow.on("trainingCompleted", event => {
        console.log(
            `Training completed: ${event.equation}`
        );
    });

    workflow.on("trainingFailed", event => {
        console.error(
            `Training failed: ${event.message}`
        );
    });

    const model = workflow.train(
        X,
        revenue,
        [
            "marketing_spend",
            "sales_staff",
            "conversion_rate"
        ]
    );

    model.summary();

    const newBusiness = [62, 9, 4.1];

    console.log(
        `\nForecast for ${JSON.stringify(newBusiness)}: ` +
        `${model.predict(newBusiness).toFixed(2)}`
    );

    console.log("\nResidual sample:");

    model.residuals.slice(0, 5).forEach(
        (residual, index) => {
            console.log(
                `Observation ${index + 1}: ${residual.toFixed(4)}`
            );
        }
    );
}

function demonstrateGradientDescent() {
    console.log("\n" + "=".repeat(78));
    console.log("GRADIENT DESCENT");
    console.log("=".repeat(78));

    const X = [
        [10, 2],
        [15, 3],
        [20, 4],
        [25, 5],
        [30, 6],
        [35, 7],
        [40, 8],
        [45, 9]
    ];

    const y = [
        55, 68, 82, 94,
        109, 121, 136, 149
    ];

    const scaled = standardize(X);

    const result = gradientDescent(
        scaled.transformed,
        y,
        0.03,
        6000
    );

    console.log(
        `Initial MSE: ${result.losses[0].toFixed(4)}`
    );

    console.log(
        `Final MSE: ${result.losses.at(-1).toFixed(4)}`
    );

    console.log(
        `Standardized intercept: ${result.intercept.toFixed(4)}`
    );

    console.log(
        "Standardized coefficients:",
        result.coefficients.map(
            value => Number(value.toFixed(4))
        )
    );
}

function demonstrateFailures() {
    console.log("\n" + "=".repeat(78));
    console.log("FAILURE HANDLING");
    console.log("=".repeat(78));

    const failures = [
        () =>
            simpleRegression(
                [1, 1, 1],
                [2, 3, 4]
            ),

        () =>
            new LinearRegression()
                .fit(
                    [
                        [1, 2],
                        [2, 4],
                        [3, 6],
                        [4, 8]
                    ],
                    [4, 8, 12, 16]
                ),

        () =>
            new LinearRegression()
                .fit(
                    [[1], [2], [3]],
                    [2, NaN, 6]
                )
    ];

    failures.forEach((operation, index) => {
        try {
            operation();
        } catch (error) {
            console.log(
                `Failure ${index + 1}: ${error.message}`
            );
        }
    });
}

function main() {
    demonstrateSimpleRegression();
    demonstrateMultipleRegression();
    demonstrateGradientDescent();
    demonstrateFailures();

    console.log("\n" + "=".repeat(78));
    console.log("JAVASCRIPT LINEAR REGRESSION DEMONSTRATION COMPLETE");
    console.log("=".repeat(78));
}

main();
