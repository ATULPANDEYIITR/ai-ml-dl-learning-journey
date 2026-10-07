"use strict";

/*
 * Regression Mathematics in JavaScript
 *
 * This file models an event-driven regression workflow for an operations
 * analytics service. It focuses on:
 *   - least-squares coefficient estimation
 *   - residual generation
 *   - MSE and RMSE
 *   - R² and adjusted R²
 *   - model comparison
 *   - validation and numerical failure handling
 *
 * It is executable with Node.js 18+ and uses no external packages.
 */

// -----------------------------------------------------------------------------
// Matrix utilities
// -----------------------------------------------------------------------------

function transpose(matrix) {
    if (matrix.length === 0) return [];

    return matrix[0].map((_, column) =>
        matrix.map(row => row[column])
    );
}

function multiplyMatrices(a, b) {
    if (a.length === 0 || b.length === 0) {
        throw new Error("Matrices cannot be empty.");
    }

    if (a[0].length !== b.length) {
        throw new Error("Incompatible matrix dimensions.");
    }

    const result = Array.from(
        { length: a.length },
        () => Array(b[0].length).fill(0)
    );

    for (let i = 0; i < a.length; i++) {
        for (let k = 0; k < b.length; k++) {
            for (let j = 0; j < b[0].length; j++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

function identityMatrix(size) {
    return Array.from({ length: size }, (_, i) =>
        Array.from({ length: size }, (_, j) => i === j ? 1 : 0)
    );
}

function inverseMatrix(matrix) {
    const n = matrix.length;

    if (
        n === 0 ||
        matrix.some(row => row.length !== n)
    ) {
        throw new Error("Only non-empty square matrices can be inverted.");
    }

    const augmented = matrix.map((row, i) => [
        ...row.map(Number),
        ...identityMatrix(n)[i]
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
                "Singular design matrix: predictors are perfectly or nearly collinear."
            );
        }

        [augmented[column], augmented[pivotRow]] =
            [augmented[pivotRow], augmented[column]];

        const pivot = augmented[column][column];

        for (let j = 0; j < 2 * n; j++) {
            augmented[column][j] /= pivot;
        }

        for (let row = 0; row < n; row++) {
            if (row === column) continue;

            const factor = augmented[row][column];

            for (let j = 0; j < 2 * n; j++) {
                augmented[row][j] -= factor * augmented[column][j];
            }
        }
    }

    return augmented.map(row => row.slice(n));
}

// -----------------------------------------------------------------------------
// Regression metric functions
// -----------------------------------------------------------------------------

function mean(values) {
    if (values.length === 0) {
        throw new Error("Cannot calculate the mean of an empty array.");
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sse(actual, predicted) {
    if (actual.length !== predicted.length) {
        throw new Error("Actual and predicted arrays must have equal length.");
    }

    return actual.reduce(
        (sum, value, index) =>
            sum + Math.pow(value - predicted[index], 2),
        0
    );
}

function mse(actual, predicted) {
    return sse(actual, predicted) / actual.length;
}

function totalSumOfSquares(actual) {
    const average = mean(actual);

    return actual.reduce(
        (sum, value) => sum + Math.pow(value - average, 2),
        0
    );
}

function rSquared(actual, predicted) {
    const total = totalSumOfSquares(actual);

    if (Math.abs(total) < 1e-12) {
        throw new Error(
            "R² is undefined because the response has zero variance."
        );
    }

    return 1 - sse(actual, predicted) / total;
}

function adjustedRSquared(r2, observations, predictors) {
    if (observations <= predictors + 1) {
        throw new Error(
            "Adjusted R² requires n > p + 1."
        );
    }

    return 1 -
        (1 - r2) *
        ((observations - 1) / (observations - predictors - 1));
}

// -----------------------------------------------------------------------------
// Event-driven regression service
// -----------------------------------------------------------------------------

class RegressionModel {
    constructor(coefficients, predictorNames) {
        this.coefficients = Object.freeze([...coefficients]);
        this.predictorNames = Object.freeze([...predictorNames]);

        if (coefficients.length !== predictorNames.length + 1) {
            throw new Error(
                "Coefficient count must equal predictors plus intercept."
            );
        }
    }

    predict(observation) {
        if (observation.length !== this.predictorNames.length) {
            throw new Error(
                `Expected ${this.predictorNames.length} predictors.`
            );
        }

        return this.coefficients[0] +
            observation.reduce(
                (sum, value, index) =>
                    sum + value * this.coefficients[index + 1],
                0
            );
    }

    predictBatch(observations) {
        return observations.map(row => this.predict(row));
    }
}

class RegressionEventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(eventName, listener) {
        if (!this.listeners.has(eventName)) {
            this.listeners.set(eventName, []);
        }

        this.listeners.get(eventName).push(listener);
    }

    emit(eventName, payload) {
        const listeners = this.listeners.get(eventName) ?? [];

        for (const listener of listeners) {
            listener(payload);
        }
    }
}

class RegressionEngine {
    constructor(eventBus) {
        this.eventBus = eventBus;
    }

    fit(features, response, predictorNames) {
        this.validateDataset(features, response, predictorNames);

        const designMatrix = features.map(row => [1, ...row]);
        const xT = transpose(designMatrix);
        const xTX = multiplyMatrices(xT, designMatrix);
        const inverse = inverseMatrix(xTX);
        const y = response.map(value => [value]);

        // Ordinary least squares:
        // beta = (XᵀX)^(-1)Xᵀy
        const coefficientsMatrix = multiplyMatrices(
            multiplyMatrices(inverse, xT),
            y
        );

        const coefficients = coefficientsMatrix.map(row => row[0]);
        const model = new RegressionModel(coefficients, predictorNames);
        const predictions = model.predictBatch(features);
        const residuals = response.map(
            (actual, index) => actual - predictions[index]
        );

        const sumSquaredError = sse(response, predictions);
        const predictionMSE = mse(response, predictions);
        const r2 = rSquared(response, predictions);
        const adjustedR2 = adjustedRSquared(
            r2,
            response.length,
            predictorNames.length
        );

        const result = Object.freeze({
            model,
            coefficients,
            predictions,
            residuals,
            sse: sumSquaredError,
            mse: predictionMSE,
            rmse: Math.sqrt(predictionMSE),
            r2,
            adjustedR2,
            n: response.length,
            p: predictorNames.length
        });

        this.eventBus.emit("model:fitted", result);

        return result;
    }

    validateDataset(features, response, predictorNames) {
        if (!Array.isArray(features) || features.length === 0) {
            throw new Error("Features must contain observations.");
        }

        if (features.length !== response.length) {
            throw new Error("Features and response lengths differ.");
        }

        if (
            !Array.isArray(predictorNames) ||
            predictorNames.length === 0
        ) {
            throw new Error("At least one predictor is required.");
        }

        if (features.some(row =>
            row.length !== predictorNames.length
        )) {
            throw new Error("Predictor dimensions are inconsistent.");
        }

        if (features.length <= predictorNames.length + 1) {
            throw new Error(
                "The dataset does not have enough degrees of freedom."
            );
        }

        for (const row of features) {
            for (const value of row) {
                if (!Number.isFinite(value)) {
                    throw new Error("Predictors must contain finite numbers.");
                }
            }
        }

        for (const value of response) {
            if (!Number.isFinite(value)) {
                throw new Error("Response values must be finite numbers.");
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Model-comparison and diagnostics
// -----------------------------------------------------------------------------

function describeResiduals(result) {
    const residuals = result.residuals;

    return {
        mean: mean(residuals),
        minimum: Math.min(...residuals),
        maximum: Math.max(...residuals),
        meanAbsoluteError:
            residuals.reduce((sum, value) => sum + Math.abs(value), 0) /
            residuals.length
    };
}

function printRegressionResult(title, result) {
    console.log(`\n${"=".repeat(72)}`);
    console.log(title);
    console.log("=".repeat(72));

    result.coefficients.forEach((coefficient, index) => {
        const name = index === 0
            ? "intercept"
            : result.model.predictorNames[index - 1];

        console.log(
            `${name.padStart(18)} : ${coefficient.toFixed(6)}`
        );
    });

    console.log(`SSE`.padStart(18) + ` : ${result.sse.toFixed(6)}`);
    console.log(`MSE`.padStart(18) + ` : ${result.mse.toFixed(6)}`);
    console.log(`RMSE`.padStart(18) + ` : ${result.rmse.toFixed(6)}`);
    console.log(`R²`.padStart(18) + ` : ${result.r2.toFixed(6)}`);
    console.log(
        `Adjusted R²`.padStart(18) +
        ` : ${result.adjustedR2.toFixed(6)}`
    );

    console.log("\nPrediction residuals:");

    result.predictions.forEach((prediction, index) => {
        console.log(
            `row=${String(index).padStart(2)} ` +
            `predicted=${prediction.toFixed(3).padStart(9)} ` +
            `residual=${result.residuals[index].toFixed(3).padStart(9)}`
        );
    });
}

// -----------------------------------------------------------------------------
// Main operational analytics example
// -----------------------------------------------------------------------------

function main() {
    const eventBus = new RegressionEventBus();

    eventBus.on("model:fitted", result => {
        console.log(
            `\nEvent: regression model fitted with ` +
            `${result.n} observations and ${result.p} predictors.`
        );
    });

    eventBus.on("model:fitted", result => {
        if (result.adjustedR2 < result.r2) {
            console.log(
                "Diagnostic: adjusted R² is below R², reflecting model complexity."
            );
        }
    });

    const engine = new RegressionEngine(eventBus);

    // Predict warehouse dispatch time using:
    // processingHours = operational effort
    // itemCount      = workload size
    const features = [
        [2, 10],
        [3, 12],
        [4, 14],
        [5, 18],
        [6, 17],
        [7, 21],
        [8, 23],
        [9, 24],
        [10, 28],
        [11, 29],
        [12, 32],
        [13, 35]
    ];

    const response = [
        8.4,
        10.2,
        11.7,
        14.1,
        14.8,
        17.0,
        18.9,
        19.7,
        23.0,
        24.2,
        26.0,
        29.1
    ];

    const predictorNames = [
        "processingHours",
        "itemCount"
    ];

    const fullModel = engine.fit(
        features,
        response,
        predictorNames
    );

    printRegressionResult(
        "Multiple Regression: Operational Dispatch Time",
        fullModel
    );

    const residualDescription = describeResiduals(fullModel);

    console.log("\nResidual diagnostics:");
    console.log(residualDescription);

    const futureObservations = [
        [6.5, 20],
        [9.5, 25],
        [14, 38]
    ];

    console.log("\nUnseen-case predictions:");

    for (const observation of futureObservations) {
        console.log({
            observation,
            predictedHours: fullModel.model.predict(observation)
        });
    }

    // Fit a reduced model to demonstrate why adjusted R² can be preferable
    // when comparing models with different predictor counts.
    const reducedFeatures = features.map(row => [row[0]]);

    const reducedModel = engine.fit(
        reducedFeatures,
        response,
        ["processingHours"]
    );

    printRegressionResult(
        "Reduced Regression: Processing Hours Only",
        reducedModel
    );

    console.log("\nModel comparison:");

    for (const [name, result] of [
        ["Reduced", reducedModel],
        ["Full", fullModel]
    ]) {
        console.log(
            `${name.padEnd(10)} ` +
            `R²=${result.r2.toFixed(5)} ` +
            `Adjusted R²=${result.adjustedR2.toFixed(5)} ` +
            `MSE=${result.mse.toFixed(5)}`
        );
    }

    // Validation failures are part of a production regression component.
    console.log("\nFailure handling:");

    try {
        engine.fit(
            [
                [1, 2],
                [2, 4],
                [3, 6],
                [4, 8]
            ],
            [2, 4, 6, 8],
            ["x", "twoX"]
        );
    } catch (error) {
        console.log(`Collinearity failure: ${error.message}`);
    }

    try {
        rSquared(
            [5, 5, 5],
            [5, 5, 5]
        );
    } catch (error) {
        console.log(`Zero-variance failure: ${error.message}`);
    }

    // JavaScript's Number type is IEEE-754 double precision. Regression
    // implementations should therefore monitor very large feature scales,
    // because poorly scaled matrices can lose numerical precision.
    console.log(
        "\nNumerical note: standardizing predictors before matrix operations " +
        "can improve conditioning when feature magnitudes differ substantially."
    );
}

main();
