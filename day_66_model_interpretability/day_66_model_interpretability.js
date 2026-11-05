"use strict";

/*
 * Model Interpretability
 *
 * Node.js implementation focused on:
 * - Model metadata and feature contracts
 * - Coefficient-based global importance
 * - Repeated permutation experiments
 * - Partial dependence analysis
 * - Event-driven interpretation workflows
 * - Audit events and policy validation
 *
 * Run: node model_interpretability.js
 */

const fs = require("node:fs/promises");
const crypto = require("node:crypto");

class InterpretabilityError extends Error {
    constructor(message, code = "INTERPRETABILITY_ERROR") {
        super(message);
        this.name = "InterpretabilityError";
        this.code = code;
    }
}

const Method = Object.freeze({
    COEFFICIENT: "coefficient",
    PERMUTATION: "permutation",
    PARTIAL_DEPENDENCE: "partial_dependence"
});

const Metric = Object.freeze({
    MSE: "mse",
    MAE: "mae",
    RMSE: "rmse"
});

function validateNumber(value, label) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new InterpretabilityError(
            `${label} must be a finite number.`,
            "INVALID_NUMBER"
        );
    }
}

function validateDataset(X, y, featureCount) {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length === 0) {
        throw new InterpretabilityError("Dataset cannot be empty.");
    }

    if (X.length !== y.length) {
        throw new InterpretabilityError("Feature and target counts differ.");
    }

    for (const row of X) {
        if (!Array.isArray(row) || row.length !== featureCount) {
            throw new InterpretabilityError(
                "Invalid feature dimensions."
            );
        }

        row.forEach((value, index) =>
            validateNumber(value, `Feature ${index}`)
        );
    }

    y.forEach((value, index) =>
        validateNumber(value, `Target ${index}`)
    );
}

function mean(values) {
    if (values.length === 0) {
        throw new InterpretabilityError("Cannot average empty values.");
    }

    return values.reduce((sum, value) => sum + value, 0) /
        values.length;
}

function variance(values) {
    const average = mean(values);

    return mean(values.map(value => (value - average) ** 2));
}

function mse(actual, predicted) {
    if (actual.length !== predicted.length || actual.length === 0) {
        throw new InterpretabilityError("Invalid evaluation arrays.");
    }

    return mean(actual.map(
        (value, index) => (value - predicted[index]) ** 2
    ));
}

function evaluate(actual, predicted) {
    const errors = actual.map(
        (value, index) => value - predicted[index]
    );

    const meanSquaredError = mean(errors.map(error => error ** 2));

    return Object.freeze({
        mse: meanSquaredError,
        mae: mean(errors.map(Math.abs)),
        rmse: Math.sqrt(meanSquaredError),
        samples: errors.length
    });
}

class LinearModel {
    constructor(features, intercept, coefficients, means, scales) {
        if (
            !Array.isArray(features) ||
            features.length === 0 ||
            new Set(features).size !== features.length
        ) {
            throw new InterpretabilityError(
                "Feature names must be unique."
            );
        }

        if (
            coefficients.length !== features.length ||
            means.length !== features.length ||
            scales.length !== features.length
        ) {
            throw new InterpretabilityError(
                "Model dimensions are inconsistent."
            );
        }

        validateNumber(intercept, "Intercept");

        coefficients.forEach((value, i) =>
            validateNumber(value, `Coefficient ${i}`)
        );

        this.features = Object.freeze([...features]);
        this.intercept = intercept;
        this.coefficients = Object.freeze([...coefficients]);
        this.means = Object.freeze([...means]);
        this.scales = Object.freeze([...scales]);

        Object.freeze(this);
    }

    predictOne(row) {
        if (!Array.isArray(row) || row.length !== this.features.length) {
            throw new InterpretabilityError("Invalid prediction input.");
        }

        let prediction = this.intercept;

        for (let i = 0; i < row.length; i++) {
            validateNumber(row[i], this.features[i]);

            prediction += this.coefficients[i] *
                ((row[i] - this.means[i]) / this.scales[i]);
        }

        return prediction;
    }

    predict(X) {
        return X.map(row => this.predictOne(row));
    }
}

class FeatureRegistry {
    #features = new Map();

    register(feature) {
        if (!feature.name || !feature.description) {
            throw new InterpretabilityError(
                "Feature metadata is incomplete."
            );
        }

        if (this.#features.has(feature.name)) {
            throw new InterpretabilityError(
                `Feature already registered: ${feature.name}`
            );
        }

        this.#features.set(
            feature.name,
            Object.freeze({ ...feature })
        );
    }

    get(name) {
        const feature = this.#features.get(name);

        if (!feature) {
            throw new InterpretabilityError(
                `Unknown feature: ${name}`
            );
        }

        return feature;
    }

    list() {
        return [...this.#features.values()];
    }
}

class DeterministicRandom {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        // Mulberry32-style deterministic generator.
        this.state = (this.state + 0x6D2B79F5) >>> 0;

        let t = this.state;

        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    shuffle(values) {
        const result = [...values];

        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }

        return result;
    }
}

class InterpretationPolicy {
    constructor({
        minimumSamples = 20,
        maximumFeatures = 100,
        minimumRepeats = 5,
        maximumRepeats = 100
    } = {}) {
        this.minimumSamples = minimumSamples;
        this.maximumFeatures = maximumFeatures;
        this.minimumRepeats = minimumRepeats;
        this.maximumRepeats = maximumRepeats;
    }

    validate(model, X, y, repeats) {
        validateDataset(X, y, model.features.length);

        if (X.length < this.minimumSamples) {
            throw new InterpretabilityError(
                "Insufficient samples for interpretation."
            );
        }

        if (model.features.length > this.maximumFeatures) {
            throw new InterpretabilityError(
                "Feature count exceeds policy."
            );
        }

        if (
            !Number.isInteger(repeats) ||
            repeats < this.minimumRepeats ||
            repeats > this.maximumRepeats
        ) {
            throw new InterpretabilityError(
                "Permutation repeat count violates policy."
            );
        }
    }
}

class InterpretationService {
    constructor(model, policy = new InterpretationPolicy()) {
        this.model = model;
        this.policy = policy;
    }

    coefficientImportance() {
        const raw = this.model.coefficients.map(Math.abs);
        const total = raw.reduce((a, b) => a + b, 0);

        return this.model.features.map((feature, index) => ({
            feature,
            score: raw[index],
            normalized: total === 0 ? 0 : raw[index] / total,
            method: Method.COEFFICIENT
        })).sort((a, b) => b.score - a.score);
    }

    permutationImportance(X, y, {
        repeats = 10,
        seed = 42
    } = {}) {
        this.policy.validate(this.model, X, y, repeats);

        const baseline = mse(y, this.model.predict(X));
        const random = new DeterministicRandom(seed);
        const results = [];

        for (let featureIndex = 0;
            featureIndex < this.model.features.length;
            featureIndex++) {

            const changes = [];

            for (let repeat = 0; repeat < repeats; repeat++) {
                const values = random.shuffle(
                    X.map(row => row[featureIndex])
                );

                const modified = X.map((row, index) => {
                    const copy = [...row];
                    copy[featureIndex] = values[index];
                    return copy;
                });

                const changedMse = mse(
                    y,
                    this.model.predict(modified)
                );

                changes.push(changedMse - baseline);
            }

            results.push({
                feature: this.model.features[featureIndex],
                score: mean(changes),
                standardDeviation: Math.sqrt(variance(changes)),
                method: Method.PERMUTATION
            });
        }

        const positiveTotal = results.reduce(
            (sum, item) => sum + Math.max(0, item.score),
            0
        );

        for (const item of results) {
            item.normalized = positiveTotal === 0
                ? 0
                : Math.max(0, item.score) / positiveTotal;
        }

        return results.sort((a, b) => b.score - a.score);
    }

    partialDependence(X, featureIndex, gridPoints = 10) {
        validateDataset(
            X,
            X.map(() => 0),
            this.model.features.length
        );

        if (
            !Number.isInteger(featureIndex) ||
            featureIndex < 0 ||
            featureIndex >= this.model.features.length
        ) {
            throw new InterpretabilityError("Invalid feature index.");
        }

        if (!Number.isInteger(gridPoints) || gridPoints < 2) {
            throw new InterpretabilityError(
                "At least two grid points are required."
            );
        }

        const values = X.map(row => row[featureIndex]);
        const min = Math.min(...values);
        const max = Math.max(...values);

        const grid = min === max
            ? [min]
            : Array.from(
                { length: gridPoints },
                (_, index) => min + (max - min) *
                    index / (gridPoints - 1)
            );

        const predictions = grid.map(value => {
            const modified = X.map(row => {
                const copy = [...row];
                copy[featureIndex] = value;
                return copy;
            });

            return mean(this.model.predict(modified));
        });

        return Object.freeze({
            feature: this.model.features[featureIndex],
            grid: Object.freeze(grid),
            predictions: Object.freeze(predictions),
            method: Method.PARTIAL_DEPENDENCE
        });
    }
}

class InterpretationEventBus {
    #listeners = new Map();

    subscribe(event, listener) {
        if (typeof listener !== "function") {
            throw new InterpretabilityError("Listener must be a function.");
        }

        if (!this.#listeners.has(event)) {
            this.#listeners.set(event, new Set());
        }

        this.#listeners.get(event).add(listener);

        return () => this.#listeners.get(event)?.delete(listener);
    }

    async publish(event, payload) {
        const listeners = [...(this.#listeners.get(event) ?? [])];

        const outcomes = await Promise.allSettled(
            listeners.map(listener =>
                Promise.resolve().then(() => listener(payload))
            )
        );

        return outcomes;
    }
}

class InterpretationWorkflow {
    constructor(service) {
        this.service = service;
        this.events = new InterpretationEventBus();
        this.history = [];
    }

    async execute(X, y) {
        const executionId = crypto.randomUUID();
        const startedAt = new Date().toISOString();

        await this.events.publish("interpretation.started", {
            executionId,
            startedAt
        });

        try {
            const result = {
                executionId,
                startedAt,
                evaluation: evaluate(y, this.service.model.predict(X)),
                coefficients: this.service.coefficientImportance(),
                permutation: this.service.permutationImportance(X, y),
                partialDependence: this.service.model.features.map(
                    (_, index) =>
                        this.service.partialDependence(X, index)
                )
            };

            const safeResult = JSON.parse(JSON.stringify(result));

            this.history.push({
                executionId,
                status: "COMPLETED",
                timestamp: new Date().toISOString()
            });

            await this.events.publish("interpretation.completed", {
                executionId
            });

            return safeResult;
        } catch (error) {
            this.history.push({
                executionId,
                status: "FAILED",
                message: error.message,
                timestamp: new Date().toISOString()
            });

            await this.events.publish("interpretation.failed", {
                executionId,
                error: error.message
            });

            throw error;
        }
    }
}

function createDataset(samples = 240, seed = 42) {
    const random = new DeterministicRandom(seed);
    const X = [];
    const y = [];

    for (let i = 0; i < samples; i++) {
        const revenue = 100 + random.next() * 900;
        const debt = 0.05 + random.next() * 0.9;
        const retention = 0.4 + random.next() * 0.59;

        // Deterministic noise avoids external random dependencies.
        const noise = (random.next() - 0.5) * 16;

        X.push([revenue, debt, retention]);

        y.push(
            0.035 * revenue -
            120 * debt +
            180 * retention +
            noise
        );
    }

    return { X, y };
}

function fitDemonstrationModel(X, y) {
    const featureCount = X[0].length;
    const means = [];
    const scales = [];

    for (let j = 0; j < featureCount; j++) {
        const column = X.map(row => row[j]);
        means.push(mean(column));
        scales.push(Math.sqrt(variance(column)) || 1);
    }

    const design = X.map(row => [
        1,
        ...row.map((value, j) =>
            (value - means[j]) / scales[j]
        )
    ]);

    const n = design.length;
    const p = featureCount + 1;

    const augmented = Array.from(
        { length: p },
        (_, i) => [
            ...Array.from(
                { length: p },
                (_, j) => design.reduce(
                    (sum, row) => sum + row[i] * row[j],
                    0
                )
            ),
            design.reduce(
                (sum, row, index) => sum + row[i] * y[index],
                0
            )
        ]
    );

    // Gaussian elimination with partial pivoting.
    for (let col = 0; col < p; col++) {
        let pivot = col;

        for (let row = col + 1; row < p; row++) {
            if (
                Math.abs(augmented[row][col]) >
                Math.abs(augmented[pivot][col])
            ) {
                pivot = row;
            }
        }

        if (Math.abs(augmented[pivot][col]) < 1e-12) {
            throw new InterpretabilityError("Singular regression matrix.");
        }

        [augmented[col], augmented[pivot]] =
            [augmented[pivot], augmented[col]];

        const divisor = augmented[col][col];

        for (let j = col; j <= p; j++) {
            augmented[col][j] /= divisor;
        }

        for (let row = 0; row < p; row++) {
            if (row === col) continue;

            const factor = augmented[row][col];

            for (let j = col; j <= p; j++) {
                augmented[row][j] -= factor * augmented[col][j];
            }
        }
    }

    const solution = augmented.map(row => row[p]);

    return new LinearModel(
        ["revenue", "debt_ratio", "customer_retention"],
        solution[0],
        solution.slice(1),
        means,
        scales
    );
}

async function main() {
    const registry = new FeatureRegistry();

    registry.register({
        name: "revenue",
        description: "Annual business revenue",
        unit: "thousand USD"
    });

    registry.register({
        name: "debt_ratio",
        description: "Debt relative to assets",
        unit: "ratio"
    });

    registry.register({
        name: "customer_retention",
        description: "Customer retention rate",
        unit: "ratio"
    });

    const { X, y } = createDataset();
    const model = fitDemonstrationModel(X, y);

    const service = new InterpretationService(model);
    const workflow = new InterpretationWorkflow(service);

    workflow.events.subscribe(
        "interpretation.completed",
        ({ executionId }) => {
            process.stdout.write(
                `Completed interpretation ${executionId}\n`
            );
        }
    );

    const result = await workflow.execute(X, y);

    console.log("\nMODEL EVALUATION");
    console.table(result.evaluation);

    console.log("\nCOEFFICIENT IMPORTANCE");
    console.table(result.coefficients);

    console.log("\nPERMUTATION IMPORTANCE");
    console.table(result.permutation);

    console.log("\nPARTIAL DEPENDENCE");
    for (const item of result.partialDependence) {
        console.log(`\n${item.feature}`);

        console.table(
            item.grid.map((value, index) => ({
                featureValue: value,
                averagePrediction: item.predictions[index]
            }))
        );
    }

    const output = JSON.stringify(
        {
            metadata: registry.list(),
            result,
            auditHistory: workflow.history
        },
        null,
        2
    );

    await fs.writeFile(
        "javascript_interpretability_report.json",
        output,
        { encoding: "utf8", mode: 0o600 }
    );

    console.log("\nReport saved.");
}

main().catch(error => {
    console.error({
        error: error.name,
        code: error.code ?? "UNEXPECTED_ERROR",
        message: error.message
    });

    process.exitCode = 1;
});
