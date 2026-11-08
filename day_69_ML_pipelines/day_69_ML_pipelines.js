"use strict";

/*
 * ML Pipelines: Event-driven preprocessing and model workflow.
 *
 * Implements:
 * - Schema validation
 * - Median imputation
 * - Standardization
 * - Categorical encoding
 * - Transformer composition
 * - Logistic regression with gradient descent
 * - Training and validation
 * - Model evaluation
 * - Pipeline state management
 * - Asynchronous artifact persistence
 * - Prediction using the fitted pipeline
 *
 * Runtime: Node.js 18+
 * Dependencies: None
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { EventEmitter } = require("node:events");

const CONFIG = Object.freeze({
    learningRate: 0.08,
    iterations: 2500,
    validationRatio: 0.20,
    seed: 42,
    outputDirectory: "js_ml_artifacts"
});

class PipelineError extends Error {
    constructor(message, stage) {
        super(message);
        this.name = "PipelineError";
        this.stage = stage;
    }
}

const assert = (condition, message, stage = "validation") => {
    if (!condition) {
        throw new PipelineError(message, stage);
    }
};

function randomGenerator(seed) {
    let state = seed >>> 0;

    return () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function sigmoid(value) {
    if (value >= 0) {
        return 1 / (1 + Math.exp(-Math.min(value, 500)));
    }

    const exp = Math.exp(Math.max(value, -500));
    return exp / (1 + exp);
}

function createDataset(size = 500) {
    assert(size >= 100, "Dataset must contain at least 100 rows");

    const random = randomGenerator(CONFIG.seed);
    const regions = ["north", "south", "east", "west"];
    const contracts = ["monthly", "annual", "two_year"];

    const rows = [];

    for (let i = 0; i < size; i++) {
        const age = 18 + Math.floor(random() * 55);
        const spend = 25 + random() * 100;
        const tenure = 1 + Math.floor(random() * 72);

        const contract = contracts[
            Math.floor(random() * contracts.length)
        ];

        const region = regions[
            Math.floor(random() * regions.length)
        ];

        const score =
            -1.0 +
            (spend - 65) * 0.012 -
            tenure * 0.025 +
            (contract === "monthly" ? 1.2 : 0) +
            (contract === "two_year" ? -1.0 : 0) +
            (age < 25 ? 0.4 : 0);

        const probability = sigmoid(score);

        rows.push({
            age: random() < 0.03 ? null : age,
            monthlySpend: random() < 0.03 ? null : spend,
            tenure: tenure,
            contract: random() < 0.03 ? null : contract,
            region,
            target: random() < probability ? 1 : 0
        });
    }

    return rows;
}

class SchemaValidator {
    constructor(numericColumns, categoricalColumns) {
        this.numericColumns = [...numericColumns];
        this.categoricalColumns = [...categoricalColumns];

        this.columns = [
            ...this.numericColumns,
            ...this.categoricalColumns
        ];

        assert(
            new Set(this.columns).size === this.columns.length,
            "Feature names must be unique"
        );
    }

    validate(rows, requireTarget = false) {
        assert(Array.isArray(rows) && rows.length > 0,
            "Input must be a non-empty array");

        for (const [index, row] of rows.entries()) {
            assert(
                row !== null && typeof row === "object",
                `Invalid record at index ${index}`
            );

            for (const column of this.columns) {
                assert(
                    Object.hasOwn(row, column),
                    `Missing feature ${column} at row ${index}`
                );

                const value = row[column];

                if (value !== null && value !== undefined) {
                    if (this.numericColumns.includes(column)) {
                        assert(
                            typeof value === "number" &&
                            Number.isFinite(value),
                            `Invalid numeric feature ${column}`
                        );
                    } else {
                        assert(
                            typeof value === "string" &&
                            value.length > 0,
                            `Invalid categorical feature ${column}`
                        );
                    }
                }
            }

            if (requireTarget) {
                assert(
                    row.target === 0 || row.target === 1,
                    `Invalid binary target at row ${index}`
                );
            }
        }
    }
}

class Transformer {
    constructor(name) {
        this.name = name;
        this.fitted = false;
    }

    fit() {
        throw new Error("fit must be implemented");
    }

    transform() {
        throw new Error("transform must be implemented");
    }

    fitTransform(data) {
        this.fit(data);
        return this.transform(data);
    }
}

class MedianImputer extends Transformer {
    constructor(columns) {
        super("median_imputer");
        this.columns = columns;
        this.medians = new Map();
    }

    fit(rows) {
        for (const column of this.columns) {
            const values = rows
                .map(row => row[column])
                .filter(value =>
                    typeof value === "number" &&
                    Number.isFinite(value)
                )
                .sort((a, b) => a - b);

            assert(
                values.length > 0,
                `Cannot impute entirely missing column ${column}`
            );

            const middle = Math.floor(values.length / 2);

            const median = values.length % 2 === 0
                ? (values[middle - 1] + values[middle]) / 2
                : values[middle];

            this.medians.set(column, median);
        }

        this.fitted = true;
        return this;
    }

    transform(rows) {
        assert(this.fitted, "Imputer is not fitted", this.name);

        return rows.map(row => {
            const output = { ...row };

            for (const column of this.columns) {
                if (
                    output[column] === null ||
                    output[column] === undefined
                ) {
                    output[column] = this.medians.get(column);
                }
            }

            return output;
        });
    }
}

class StandardScaler extends Transformer {
    constructor(columns) {
        super("standard_scaler");
        this.columns = columns;
        this.statistics = new Map();
    }

    fit(rows) {
        for (const column of this.columns) {
            const values = rows.map(row => row[column]);

            assert(
                values.every(Number.isFinite),
                `Non-finite values in ${column}`
            );

            const mean = values.reduce((sum, value) =>
                sum + value, 0) / values.length;

            const variance = values.reduce((sum, value) =>
                sum + (value - mean) ** 2, 0) / values.length;

            this.statistics.set(column, {
                mean,
                deviation: Math.sqrt(variance)
            });
        }

        this.fitted = true;
        return this;
    }

    transform(rows) {
        assert(this.fitted, "Scaler is not fitted", this.name);

        return rows.map(row => {
            const output = { ...row };

            for (const column of this.columns) {
                const { mean, deviation } =
                    this.statistics.get(column);

                output[column] = deviation === 0
                    ? 0
                    : (output[column] - mean) / deviation;
            }

            return output;
        });
    }
}

class OneHotEncoder extends Transformer {
    constructor(columns) {
        super("one_hot_encoder");
        this.columns = columns;
        this.categories = new Map();
    }

    fit(rows) {
        for (const column of this.columns) {
            const categories = [...new Set(
                rows.map(row => row[column])
            )].sort();

            assert(
                categories.length > 0,
                `No categories found for ${column}`
            );

            this.categories.set(column, categories);
        }

        this.fitted = true;
        return this;
    }

    transform(rows) {
        assert(this.fitted, "Encoder is not fitted", this.name);

        return rows.map(row => {
            const output = { ...row };

            for (const column of this.columns) {
                const categories = this.categories.get(column);

                for (const category of categories) {
                    const featureName = `${column}__${category}`;

                    output[featureName] =
                        row[column] === category ? 1 : 0;
                }

                // Unknown categories become an all-zero encoded group.
                delete output[column];
            }

            return output;
        });
    }
}

class Pipeline extends EventEmitter {
    constructor(transformers, estimator) {
        super();

        this.transformers = transformers;
        this.estimator = estimator;
        this.state = "created";
        this.fitted = false;
    }

    transition(nextState) {
        const allowed = {
            created: ["fitting", "failed"],
            fitting: ["fitted", "failed"],
            fitted: ["predicting", "fitting", "failed"],
            predicting: ["fitted", "failed"],
            failed: []
        };

        assert(
            allowed[this.state]?.includes(nextState),
            `Invalid state transition: ${this.state} -> ${nextState}`
        );

        const previous = this.state;
        this.state = nextState;

        this.emit("stateChanged", {
            previous,
            current: nextState
        });
    }

    fit(rows) {
        assert(
            this.state !== "failed",
            "A failed pipeline instance cannot be reused"
        );

        this.transition("fitting");

        try {
            let transformed = rows.map(row => ({ ...row }));

            for (const transformer of this.transformers) {
                transformed = transformer.fitTransform(transformed);

                this.emit("transformerCompleted", {
                    name: transformer.name,
                    rows: transformed.length
                });
            }

            this.estimator.fit(transformed);
            this.fitted = true;
            this.transition("fitted");

            return this;
        } catch (error) {
            this.state = "failed";
            this.emit("pipelineError", error);
            throw error;
        }
    }

    predict(rows) {
        assert(this.fitted, "Pipeline must be fitted before prediction");

        this.transition("predicting");

        try {
            let transformed = rows.map(row => ({ ...row }));

            for (const transformer of this.transformers) {
                transformed = transformer.transform(transformed);
            }

            const predictions = this.estimator.predict(transformed);

            this.transition("fitted");

            return predictions;
        } catch (error) {
            this.state = "failed";
            this.emit("pipelineError", error);
            throw error;
        }
    }

    predictProbability(rows) {
        assert(this.fitted, "Pipeline must be fitted");

        let transformed = rows.map(row => ({ ...row }));

        for (const transformer of this.transformers) {
            transformed = transformer.transform(transformed);
        }

        return this.estimator.predictProbability(transformed);
    }

    serialize() {
        assert(this.fitted, "Cannot serialize an unfitted pipeline");

        return {
            version: 1,
            state: this.state,
            transformers: this.transformers.map(transformer => ({
                name: transformer.name,
                fitted: transformer.fitted,
                columns: transformer.columns,
                medians: transformer.medians
                    ? Object.fromEntries(transformer.medians)
                    : undefined,
                statistics: transformer.statistics
                    ? Object.fromEntries(transformer.statistics)
                    : undefined,
                categories: transformer.categories
                    ? Object.fromEntries(transformer.categories)
                    : undefined
            })),
            estimator: this.estimator.serialize()
        };
    }
}

class LogisticRegressionModel {
    constructor({
        learningRate = CONFIG.learningRate,
        iterations = CONFIG.iterations,
        regularization = 0.01
    } = {}) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.regularization = regularization;
        this.weights = [];
        this.featureNames = [];
        this.fitted = false;
    }

    extractFeatures(rows) {
        return rows.map(row => this.featureNames.map(name => {
            const value = row[name] ?? 0;

            assert(
                Number.isFinite(value),
                `Invalid model input ${name}`
            );

            return value;
        }));
    }

    fit(rows) {
        assert(rows.length > 0, "Training data is empty");

        const names = Object.keys(rows[0])
            .filter(name => name !== "target")
            .sort();

        assert(names.length > 0, "No model features available");

        this.featureNames = names;

        const X = this.extractFeatures(rows);
        const y = rows.map(row => row.target);

        assert(
            new Set(y).size === 2,
            "Training requires both target classes"
        );

        this.weights = new Array(names.length + 1).fill(0);

        // Batch gradient descent with L2 regularization.
        for (let iteration = 0;
             iteration < this.iterations;
             iteration++) {

            const gradients = new Array(this.weights.length).fill(0);

            for (let i = 0; i < X.length; i++) {
                const score = this.weights[0] +
                    X[i].reduce(
                        (sum, value, j) =>
                            sum + value * this.weights[j + 1],
                        0
                    );

                const error = sigmoid(score) - y[i];

                gradients[0] += error;

                for (let j = 0; j < X[i].length; j++) {
                    gradients[j + 1] += error * X[i][j];
                }
            }

            const scale = 1 / X.length;

            this.weights[0] -=
                this.learningRate * gradients[0] * scale;

            for (let j = 1; j < this.weights.length; j++) {
                const regularizationGradient =
                    this.regularization * this.weights[j];

                this.weights[j] -= this.learningRate *
                    (gradients[j] * scale + regularizationGradient);
            }
        }

        this.fitted = true;
        return this;
    }

    predictProbability(rows) {
        assert(this.fitted, "Estimator has not been trained");

        return this.extractFeatures(rows).map(features => {
            const score = this.weights[0] +
                features.reduce(
                    (sum, value, index) =>
                        sum + value * this.weights[index + 1],
                    0
                );

            return sigmoid(score);
        });
    }

    predict(rows) {
        return this.predictProbability(rows).map(
            probability => probability >= 0.5 ? 1 : 0
        );
    }

    serialize() {
        return {
            learningRate: this.learningRate,
            iterations: this.iterations,
            regularization: this.regularization,
            weights: this.weights,
            featureNames: this.featureNames,
            fitted: this.fitted
        };
    }
}

function splitDataset(rows, validationRatio = 0.20) {
    assert(
        validationRatio > 0 && validationRatio < 1,
        "Invalid validation ratio"
    );

    const random = randomGenerator(CONFIG.seed);
    const shuffled = [...rows];

    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    const validationSize = Math.floor(
        shuffled.length * validationRatio
    );

    return {
        training: shuffled.slice(validationSize),
        validation: shuffled.slice(0, validationSize)
    };
}

function calculateMetrics(actual, predicted, probabilities) {
    assert(
        actual.length === predicted.length &&
        actual.length === probabilities.length,
        "Metric input sizes do not match"
    );

    let correct = 0;
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;

    for (let i = 0; i < actual.length; i++) {
        if (actual[i] === predicted[i]) correct++;

        if (actual[i] === 1 && predicted[i] === 1) tp++;
        else if (actual[i] === 0 && predicted[i] === 1) fp++;
        else if (actual[i] === 1 && predicted[i] === 0) fn++;
        else tn++;
    }

    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);

    const f1 = precision + recall === 0
        ? 0
        : 2 * precision * recall / (precision + recall);

    const ranked = actual.map((label, index) => ({
        label,
        probability: probabilities[index]
    })).sort((a, b) => a.probability - b.probability);

    const positives = actual.filter(value => value === 1).length;
    const negatives = actual.length - positives;

    let auc = null;

    if (positives > 0 && negatives > 0) {
        let positiveRankSum = 0;

        for (let i = 0; i < ranked.length; i++) {
            if (ranked[i].label === 1) {
                positiveRankSum += i + 1;
            }
        }

        // For this implementation, tied scores are rare but rank ties
        // should be averaged in a production statistical implementation.
        auc = (
            positiveRankSum -
            positives * (positives + 1) / 2
        ) / (positives * negatives);
    }

    return {
        accuracy: correct / actual.length,
        precision,
        recall,
        f1,
        rocAuc: auc,
        confusionMatrix: [[tn, fp], [fn, tp]]
    };
}

async function persistArtifact(artifact) {
    const directory = path.resolve(CONFIG.outputDirectory);
    const destination = path.join(directory, "pipeline.json");
    const temporary = `${destination}.tmp`;

    await fs.mkdir(directory, { recursive: true });

    try {
        await fs.writeFile(
            temporary,
            JSON.stringify(artifact, null, 2),
            { encoding: "utf8", mode: 0o600 }
        );

        await fs.rename(temporary, destination);

        return destination;
    } catch (error) {
        await fs.rm(temporary, { force: true });
        throw new PipelineError(
            `Artifact persistence failed: ${error.message}`,
            "persistence"
        );
    }
}

async function main() {
    const numericColumns = [
        "age",
        "monthlySpend",
        "tenure"
    ];

    const categoricalColumns = [
        "contract",
        "region"
    ];

    const dataset = createDataset();
    const validator = new SchemaValidator(
        numericColumns,
        categoricalColumns
    );

    validator.validate(dataset, true);

    const { training, validation } = splitDataset(
        dataset,
        CONFIG.validationRatio
    );

    // Fit all transformations using training data only.
    const featureColumns = [
        new MedianImputer(numericColumns),
        new StandardScaler(numericColumns),
        new OneHotEncoder(categoricalColumns)
    ];

    const estimator = new LogisticRegressionModel();

    const pipeline = new Pipeline(
        featureColumns,
        estimator
    );

    pipeline.on("stateChanged", event => {
        console.log(
            `[STATE] ${event.previous} -> ${event.current}`
        );
    });

    pipeline.on("transformerCompleted", event => {
        console.log(
            `[TRANSFORM] ${event.name}: ${event.rows} rows`
        );
    });

    pipeline.on("pipelineError", error => {
        console.error("[PIPELINE ERROR]", error.message);
    });

    console.log("Training ML pipeline...");

    pipeline.fit(training);

    const predictions = pipeline.predict(validation);
    const probabilities = pipeline.predictProbability(validation);

    const actual = validation.map(row => row.target);

    const metrics = calculateMetrics(
        actual,
        predictions,
        probabilities
    );

    console.log("\nEvaluation metrics:");
    console.log(JSON.stringify(metrics, null, 2));

    const artifact = {
        metadata: {
            model: "LogisticRegression",
            pipelineVersion: 1,
            trainingSamples: training.length,
            validationSamples: validation.length,
            createdAt: new Date().toISOString()
        },
        metrics,
        pipeline: pipeline.serialize()
    };

    const artifactPath = await persistArtifact(artifact);

    console.log("\nArtifact persisted:", artifactPath);

    const inferenceRecords = [
        {
            age: 28,
            monthlySpend: 90,
            tenure: 6,
            contract: "monthly",
            region: "north"
        },
        {
            age: 55,
            monthlySpend: 50,
            tenure: 65,
            contract: "two_year",
            region: "south"
        }
    ];

    validator.validate(inferenceRecords);

    const results = inferenceRecords.map((record, index) => ({
        customer: index + 1,
        probability: pipeline.predictProbability([record])[0],
        prediction: pipeline.predict([record])[0]
    }));

    console.log("\nInference:");
    console.log(JSON.stringify(results, null, 2));

    assert(pipeline.state === "fitted", "Unexpected final state");

    console.log("\nAll pipeline operations completed.");
}

main().catch(error => {
    console.error(
        `[FATAL] ${error.name}: ${error.message}`
    );

    if (error instanceof PipelineError) {
        console.error("Failed stage:", error.stage);
    }

    process.exitCode = 1;
});
