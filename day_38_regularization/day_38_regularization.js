"use strict";

/*
 * Regularization in JavaScript:
 * Ridge, Lasso, Elastic Net, and the Bias-Variance Trade-off.
 *
 * This Node.js program implements a small regression toolkit without
 * third-party dependencies. It emphasizes JavaScript-specific patterns:
 * classes, immutable-style data handling, event-driven model monitoring,
 * validation, Map-based reporting, and asynchronous cross-validation.
 */

const assert = require("node:assert/strict");

function mean(values) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new Error("mean requires a non-empty array");
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function dot(a, b) {
    if (a.length !== b.length) {
        throw new Error("Vector dimensions do not match");
    }
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function mse(actual, predicted) {
    if (actual.length !== predicted.length || actual.length === 0) {
        throw new Error("MSE requires equal non-empty arrays");
    }

    return mean(
        actual.map((value, index) => {
            const error = value - predicted[index];
            return error * error;
        })
    );
}

function rmse(actual, predicted) {
    return Math.sqrt(mse(actual, predicted));
}

function r2(actual, predicted) {
    const baseline = mean(actual);
    const total = actual.reduce(
        (sum, value) => sum + (value - baseline) ** 2,
        0
    );

    if (total === 0) {
        return actual.every(
            (value, index) => Math.abs(value - predicted[index]) < 1e-12
        )
            ? 1
            : 0;
    }

    const residual = actual.reduce(
        (sum, value, index) => sum + (value - predicted[index]) ** 2,
        0
    );

    return 1 - residual / total;
}

function validateMatrix(X, y) {
    if (!Array.isArray(X) || X.length === 0) {
        throw new Error("X must be a non-empty matrix");
    }

    if (!Array.isArray(y) || X.length !== y.length) {
        throw new Error("X and y must have the same number of rows");
    }

    const width = X[0].length;

    if (width === 0) {
        throw new Error("At least one feature is required");
    }

    for (const row of X) {
        if (row.length !== width) {
            throw new Error("All feature rows must have the same width");
        }

        for (const value of row) {
            if (!Number.isFinite(value)) {
                throw new Error("Features must be finite numbers");
            }
        }
    }

    for (const value of y) {
        if (!Number.isFinite(value)) {
            throw new Error("Targets must be finite numbers");
        }
    }
}

class StandardScaler {
    constructor() {
        this.means = [];
        this.scales = [];
        this.fitted = false;
    }

    fit(X) {
        if (!Array.isArray(X) || X.length === 0) {
            throw new Error("Cannot fit scaler to empty data");
        }

        const width = X[0].length;

        this.means = Array.from({ length: width }, (_, column) =>
            mean(X.map(row => row[column]))
        );

        this.scales = this.means.map((columnMean, column) => {
            const variance = mean(
                X.map(row => (row[column] - columnMean) ** 2)
            );
            const standardDeviation = Math.sqrt(variance);

            // A constant feature contains no information about scale.
            // Keeping scale at one avoids division by zero.
            return standardDeviation > 1e-12 ? standardDeviation : 1;
        });

        this.fitted = true;
        return this;
    }

    transform(X) {
        if (!this.fitted) {
            throw new Error("Scaler has not been fitted");
        }

        return X.map(row =>
            row.map(
                (value, column) =>
                    (value - this.means[column]) / this.scales[column]
            )
        );
    }

    fitTransform(X) {
        return this.fit(X).transform(X);
    }
}

function softThreshold(value, threshold) {
    if (value > threshold) {
        return value - threshold;
    }

    if (value < -threshold) {
        return value + threshold;
    }

    return 0;
}

class LinearRegression {
    constructor({
        learningRate = 0.003,
        maxIterations = 8000,
        tolerance = 1e-8
    } = {}) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.intercept = 0;
        this.coefficients = [];
    }

    fit(X, y) {
        validateMatrix(X, y);

        const n = X.length;
        const p = X[0].length;

        this.coefficients = Array(p).fill(0);
        this.intercept = mean(y);

        let previousLoss = Infinity;

        for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
            const predictions = X.map(row =>
                this.intercept + dot(row, this.coefficients)
            );

            const errors = predictions.map(
                (prediction, index) => prediction - y[index]
            );

            const interceptGradient = 2 * mean(errors);

            const gradients = Array.from({ length: p }, (_, column) =>
                2 *
                mean(
                    errors.map(
                        (error, rowIndex) =>
                            error * X[rowIndex][column]
                    )
                )
            );

            this.intercept -= this.learningRate * interceptGradient;

            this.coefficients = this.coefficients.map(
                (coefficient, column) =>
                    coefficient - this.learningRate * gradients[column]
            );

            const loss = mean(errors.map(error => error ** 2));

            if (Math.abs(previousLoss - loss) < this.tolerance) {
                break;
            }

            previousLoss = loss;
        }

        return this;
    }

    predict(X) {
        return X.map(
            row => this.intercept + dot(row, this.coefficients)
        );
    }
}

class RidgeRegression extends LinearRegression {
    constructor({
        alpha = 1,
        learningRate = 0.003,
        maxIterations = 10000,
        tolerance = 1e-8
    } = {}) {
        super({ learningRate, maxIterations, tolerance });

        if (alpha < 0) {
            throw new Error("alpha must be non-negative");
        }

        this.alpha = alpha;
    }

    fit(X, y) {
        validateMatrix(X, y);

        const p = X[0].length;
        this.coefficients = Array(p).fill(0);
        this.intercept = mean(y);

        let previousLoss = Infinity;

        for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
            const predictions = X.map(row =>
                this.intercept + dot(row, this.coefficients)
            );

            const errors = predictions.map(
                (prediction, index) => prediction - y[index]
            );

            this.intercept -=
                this.learningRate * 2 * mean(errors);

            this.coefficients = this.coefficients.map(
                (coefficient, column) => {
                    const dataGradient =
                        2 *
                        mean(
                            errors.map(
                                (error, rowIndex) =>
                                    error * X[rowIndex][column]
                            )
                        );

                    const penaltyGradient = 2 * this.alpha * coefficient;

                    return (
                        coefficient -
                        this.learningRate *
                            (dataGradient + penaltyGradient)
                    );
                }
            );

            const dataLoss = mean(errors.map(error => error ** 2));
            const penalty =
                this.alpha *
                this.coefficients.reduce(
                    (sum, coefficient) => sum + coefficient ** 2,
                    0
                );

            const loss = dataLoss + penalty;

            if (Math.abs(previousLoss - loss) < this.tolerance) {
                break;
            }

            previousLoss = loss;
        }

        return this;
    }
}

class LassoRegression {
    constructor({
        alpha = 0.1,
        maxIterations = 4000,
        tolerance = 1e-7
    } = {}) {
        if (alpha < 0) {
            throw new Error("alpha must be non-negative");
        }

        this.alpha = alpha;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.intercept = 0;
        this.coefficients = [];
    }

    fit(X, y) {
        validateMatrix(X, y);

        const n = X.length;
        const p = X[0].length;

        this.coefficients = Array(p).fill(0);
        this.intercept = mean(y);

        let residuals = y.map(target => target - this.intercept);

        for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
            const old = [...this.coefficients];

            const residualMean = mean(residuals);
            this.intercept += residualMean;
            residuals = residuals.map(value => value - residualMean);

            for (let column = 0; column < p; column++) {
                const feature = X.map(row => row[column]);

                // Restore the old contribution before optimizing this
                // coordinate. This is the core coordinate-descent update.
                residuals = residuals.map(
                    (residual, rowIndex) =>
                        residual +
                        feature[rowIndex] * this.coefficients[column]
                );

                const rho = feature.reduce(
                    (sum, value, rowIndex) =>
                        sum + value * residuals[rowIndex],
                    0
                );

                const denominator = feature.reduce(
                    (sum, value) => sum + value * value,
                    0
                );

                this.coefficients[column] =
                    denominator === 0
                        ? 0
                        : softThreshold(
                              rho,
                              (this.alpha * n) / 2
                          ) / denominator;

                residuals = residuals.map(
                    (residual, rowIndex) =>
                        residual -
                        feature[rowIndex] * this.coefficients[column]
                );
            }

            const largestChange = Math.max(
                ...this.coefficients.map(
                    (coefficient, index) =>
                        Math.abs(coefficient - old[index])
                )
            );

            if (largestChange < this.tolerance) {
                break;
            }
        }

        return this;
    }

    predict(X) {
        return X.map(
            row => this.intercept + dot(row, this.coefficients)
        );
    }
}

class ElasticNetRegression extends LassoRegression {
    constructor({
        alpha = 0.1,
        l1Ratio = 0.5,
        maxIterations = 5000,
        tolerance = 1e-7
    } = {}) {
        if (alpha < 0) {
            throw new Error("alpha must be non-negative");
        }

        if (l1Ratio < 0 || l1Ratio > 1) {
            throw new Error("l1Ratio must be between zero and one");
        }

        super({
            alpha,
            maxIterations,
            tolerance
        });

        this.l1Ratio = l1Ratio;
    }

    fit(X, y) {
        validateMatrix(X, y);

        const n = X.length;
        const p = X[0].length;

        this.coefficients = Array(p).fill(0);
        this.intercept = mean(y);

        let residuals = y.map(target => target - this.intercept);

        const l1 = this.alpha * this.l1Ratio;
        const l2 = this.alpha * (1 - this.l1Ratio);

        for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
            const old = [...this.coefficients];

            const residualMean = mean(residuals);
            this.intercept += residualMean;
            residuals = residuals.map(value => value - residualMean);

            for (let column = 0; column < p; column++) {
                const feature = X.map(row => row[column]);

                residuals = residuals.map(
                    (residual, rowIndex) =>
                        residual +
                        feature[rowIndex] * this.coefficients[column]
                );

                const rho = feature.reduce(
                    (sum, value, rowIndex) =>
                        sum + value * residuals[rowIndex],
                    0
                );

                const denominator =
                    feature.reduce(
                        (sum, value) => sum + value * value,
                        0
                    ) +
                    n * l2;

                this.coefficients[column] =
                    denominator === 0
                        ? 0
                        : softThreshold(
                              rho,
                              (n * l1) / 2
                          ) / denominator;

                residuals = residuals.map(
                    (residual, rowIndex) =>
                        residual -
                        feature[rowIndex] * this.coefficients[column]
                );
            }

            const largestChange = Math.max(
                ...this.coefficients.map(
                    (coefficient, index) =>
                        Math.abs(coefficient - old[index])
                )
            );

            if (largestChange < this.tolerance) {
                break;
            }
        }

        return this;
    }
}

/*
 * A tiny event-driven layer demonstrates a useful JavaScript-specific
 * perspective: model training can emit progress events while another part
 * of the application observes convergence without owning the algorithm.
 */
class TrainingMonitor extends require("node:events").EventEmitter {
    report(modelName, metrics) {
        this.emit("trainingComplete", {
            modelName,
            ...metrics
        });
    }
}

function generateData(samples = 180, seed = 13) {
    let state = seed >>> 0;

    const random = () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };

    const gaussian = () => {
        let u = 0;
        let v = 0;

        while (u === 0) u = random();
        while (v === 0) v = random();

        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };

    const X = [];
    const y = [];

    for (let i = 0; i < samples; i++) {
        const x1 = 20 + gaussian() * 4;
        const x2 = 55 + gaussian() * 10;
        const x3 = 1000 + x1 * 2 + gaussian() * 3;
        const x4 = random() * 100;
        const x5 = random() * 30;
        const x6 = random() < 0.2 ? 1 : 0;
        const x7 = 50 + gaussian() * 7;
        const x8 = 500 + gaussian() * 100;
        const x9 = gaussian();
        const x10 = gaussian();
        const x11 = gaussian();
        const x12 = gaussian();

        X.push([
            x1, x2, x3, x4, x5, x6,
            x7, x8, x9, x10, x11, x12
        ]);

        y.push(
            80 +
            2.8 * x1 -
            0.9 * x2 +
            0.6 * x4 +
            1.2 * x5 +
            18 * x6 -
            0.5 * x7 +
            0.04 * x8 +
            gaussian() * 8
        );
    }

    return { X, y };
}

function splitData(X, y, testRatio = 0.25) {
    const indices = Array.from(
        { length: X.length },
        (_, index) => index
    );

    // Fisher-Yates shuffle creates a random partition without dependencies.
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const testSize = Math.max(1, Math.round(X.length * testRatio));
    const testIndices = new Set(indices.slice(0, testSize));

    const XTrain = [];
    const yTrain = [];
    const XTest = [];
    const yTest = [];

    for (let i = 0; i < X.length; i++) {
        if (testIndices.has(i)) {
            XTest.push([...X[i]]);
            yTest.push(y[i]);
        } else {
            XTrain.push([...X[i]]);
            yTrain.push(y[i]);
        }
    }

    return { XTrain, yTrain, XTest, yTest };
}

function evaluate(name, model, XTrain, yTrain, XTest, yTest, monitor) {
    model.fit(XTrain, yTrain);

    const trainPrediction = model.predict(XTrain);
    const testPrediction = model.predict(XTest);

    const metrics = {
        trainRMSE: rmse(yTrain, trainPrediction),
        testRMSE: rmse(yTest, testPrediction),
        trainR2: r2(yTrain, trainPrediction),
        testR2: r2(yTest, testPrediction),
        zeroCoefficients: model.coefficients.filter(
            coefficient => Math.abs(coefficient) < 1e-8
        ).length
    };

    monitor.report(name, metrics);

    return model;
}

async function crossValidate(ModelFactory, X, y, alphas, folds = 5) {
    const foldSize = Math.floor(X.length / folds);
    const results = [];

    for (const alpha of alphas) {
        const validationErrors = [];

        for (let fold = 0; fold < folds; fold++) {
            const validationStart = fold * foldSize;
            const validationEnd =
                fold === folds - 1
                    ? X.length
                    : validationStart + foldSize;

            const XTrain = [];
            const yTrain = [];
            const XValid = [];
            const yValid = [];

            for (let index = 0; index < X.length; index++) {
                if (index >= validationStart && index < validationEnd) {
                    XValid.push(X[index]);
                    yValid.push(y[index]);
                } else {
                    XTrain.push(X[index]);
                    yTrain.push(y[index]);
                }
            }

            /*
             * Promise.resolve makes each fold asynchronous from the caller's
             * perspective. It models how a production application can run
             * validation jobs through a task queue without changing the
             * mathematical training algorithm.
             */
            const error = await Promise.resolve().then(() => {
                const model = ModelFactory(alpha);
                model.fit(XTrain, yTrain);
                return mse(yValid, model.predict(XValid));
            });

            validationErrors.push(error);
        }

        results.push({
            alpha,
            validationMSE: mean(validationErrors)
        });
    }

    return results.sort(
        (a, b) => a.validationMSE - b.validationMSE
    );
}

async function main() {
    console.log("=".repeat(72));
    console.log("REGULARIZATION IN JAVASCRIPT");
    console.log("=".repeat(72));

    const { X, y } = generateData();
    const { XTrain, yTrain, XTest, yTest } = splitData(X, y);

    const scaler = new StandardScaler();
    const scaledTrain = scaler.fitTransform(XTrain);
    const scaledTest = scaler.transform(XTest);

    const monitor = new TrainingMonitor();

    monitor.on("trainingComplete", event => {
        console.log(
            `${event.modelName.padEnd(12)} ` +
            `train RMSE=${event.trainRMSE.toFixed(4)} ` +
            `test RMSE=${event.testRMSE.toFixed(4)} ` +
            `test R²=${event.testR2.toFixed(4)} ` +
            `zeros=${event.zeroCoefficients}`
        );
    });

    const models = new Map([
        [
            "OLS",
            () =>
                new LinearRegression({
                    learningRate: 0.003,
                    maxIterations: 10000
                })
        ],
        [
            "Ridge",
            () =>
                new RidgeRegression({
                    alpha: 0.8,
                    learningRate: 0.003,
                    maxIterations: 12000
                })
        ],
        [
            "Lasso",
            () =>
                new LassoRegression({
                    alpha: 0.08,
                    maxIterations: 5000
                })
        ],
        [
            "Elastic Net",
            () =>
                new ElasticNetRegression({
                    alpha: 0.08,
                    l1Ratio: 0.5,
                    maxIterations: 5000
                })
        ]
    ]);

    for (const [name, factory] of models) {
        evaluate(
            name,
            factory(),
            scaledTrain,
            yTrain,
            scaledTest,
            yTest,
            monitor
        );
    }

    console.log("\nCoefficient sparsity");

    for (const [name, factory] of models) {
        const model = factory();
        model.fit(scaledTrain, yTrain);

        const coefficients = model.coefficients.map(
            coefficient => Number(coefficient.toFixed(5))
        );

        console.log(`${name.padEnd(12)} ${coefficients.join(", ")}`);
    }

    console.log("\nRidge cross-validation");

    const ridgeScores = await crossValidate(
        alpha =>
            new RidgeRegression({
                alpha,
                learningRate: 0.003,
                maxIterations: 10000
            }),
        scaledTrain,
        yTrain,
        [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    );

    for (const result of ridgeScores) {
        console.log(
            `alpha=${String(result.alpha).padEnd(6)} ` +
            `validation MSE=${result.validationMSE.toFixed(5)}`
        );
    }

    console.log(
        `Selected Ridge alpha: ${ridgeScores[0].alpha}`
    );

    console.log("\nLasso sparsity path");

    for (const alpha of [0.005, 0.02, 0.05, 0.1, 0.2, 0.5]) {
        const model = new LassoRegression({
            alpha,
            maxIterations: 5000
        });

        model.fit(scaledTrain, yTrain);

        const activeFeatures = model.coefficients
            .map((coefficient, index) => ({
                index,
                coefficient
            }))
            .filter(item => Math.abs(item.coefficient) > 1e-7)
            .map(item => `x${item.index + 1}`);

        console.log(
            `alpha=${alpha.toString().padEnd(6)} ` +
            `active=${activeFeatures.join(", ")}`
        );
    }

    console.log("\nBias-variance simulation");

    const predictionSamples = new Map([
        ["OLS", []],
        ["Ridge", []],
        ["Lasso", []],
        ["Elastic Net", []]
    ]);

    for (let sample = 0; sample < 25; sample++) {
        const indices = Array.from(
            { length: Math.floor(X.length / 2) },
            () => Math.floor(Math.random() * X.length)
        );

        const sampleX = indices.map(index => scaledTrain[index % scaledTrain.length]);
        const sampleY = indices.map(index => yTrain[index % yTrain.length]);
        const observation = scaledTest.slice(0, 1);

        const sampleModels = [
            ["OLS", new LinearRegression()],
            ["Ridge", new RidgeRegression({ alpha: 1 })],
            ["Lasso", new LassoRegression({ alpha: 0.08 })],
            [
                "Elastic Net",
                new ElasticNetRegression({
                    alpha: 0.08,
                    l1Ratio: 0.5
                })
            ]
        ];

        for (const [name, model] of sampleModels) {
            model.fit(sampleX, sampleY);
            predictionSamples.get(name).push(
                model.predict(observation)[0]
            );
        }
    }

    for (const [name, predictions] of predictionSamples) {
        const predictionMean = mean(predictions);
        const predictionVariance = mean(
            predictions.map(
                prediction => (prediction - predictionMean) ** 2
            )
        );

        console.log(
            `${name.padEnd(12)} ` +
            `mean=${predictionMean.toFixed(4)} ` +
            `variance=${predictionVariance.toFixed(4)}`
        );
    }

    console.log("\nValidation behavior");

    assert.throws(
        () => new RidgeRegression({ alpha: -1 }),
        /non-negative/
    );

    assert.throws(
        () =>
            new ElasticNetRegression({
                alpha: 0.1,
                l1Ratio: 1.5
            }),
        /between zero and one/
    );

    console.log("Invalid regularization parameters are rejected.");

    console.log("\nPractical interpretation");
    console.log(
        "Ridge distributes shrinkage across correlated predictors and " +
        "usually retains all features."
    );
    console.log(
        "Lasso uses an L1 penalty that can make coefficients exactly zero, " +
        "creating a sparse model."
    );
    console.log(
        "Elastic Net combines the sparse behavior of L1 with the stability " +
        "of L2 for correlated predictors."
    );
    console.log(
        "Increasing regularization normally raises bias while reducing " +
        "variance, so alpha must be selected using validation data."
    );
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
