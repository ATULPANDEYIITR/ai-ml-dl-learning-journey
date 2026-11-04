"use strict";

/*
 * Feature Selection Engine
 *
 * Event-driven feature-selection system for Node.js.
 *
 * Implements:
 * - Filter selection using variance, correlation and mutual information.
 * - Wrapper selection using greedy subset evaluation.
 * - Embedded selection using regularized linear coefficients.
 * - Recursive Feature Elimination.
 * - Asynchronous experiment execution.
 * - Selection policies and audit events.
 *
 * Run: node feature-selection.js
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { EventEmitter } = require("node:events");

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class FeatureSelectionError extends Error {
    constructor(message) {
        super(message);
        this.name = "FeatureSelectionError";
    }
}

const SelectionMethod = Object.freeze({
    FILTER: "FILTER",
    WRAPPER: "WRAPPER",
    EMBEDDED: "EMBEDDED",
    RFE: "RFE"
});

function mean(values) {
    if (!values.length) {
        throw new ValidationError("Cannot calculate an empty mean.");
    }

    return values.reduce((sum, value) => sum + value, 0) /
        values.length;
}

function variance(values) {
    const center = mean(values);

    return values.reduce(
        (sum, value) => sum + (value - center) ** 2,
        0
    ) / values.length;
}

function correlation(x, y) {
    const mx = mean(x);
    const my = mean(y);

    let numerator = 0;
    let sx = 0;
    let sy = 0;

    for (let i = 0; i < x.length; i++) {
        const dx = x[i] - mx;
        const dy = y[i] - my;

        numerator += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }

    const denominator = Math.sqrt(sx * sy);

    return denominator < 1e-12 ? 0 : numerator / denominator;
}

function entropy(counts) {
    const total = counts.reduce((a, b) => a + b, 0);

    if (!total) return 0;

    return counts.reduce((result, count) => {
        if (!count) return result;

        const probability = count / total;
        return result - probability * Math.log2(probability);
    }, 0);
}

function mutualInformation(values, labels, bins = 5) {
    if (bins < 2) {
        throw new ValidationError("At least two bins are required.");
    }

    const sorted = [...values].sort((a, b) => a - b);

    const boundaries = Array.from(
        { length: bins - 1 },
        (_, i) => sorted[
            Math.floor((i + 1) * sorted.length / bins)
        ]
    );

    const joint = new Map();
    const xCounts = new Map();
    const yCounts = new Map();

    for (let i = 0; i < values.length; i++) {
        const category = boundaries.filter(
            boundary => values[i] >= boundary
        ).length;

        const key = `${category}:${labels[i]}`;

        joint.set(key, (joint.get(key) || 0) + 1);
        xCounts.set(category, (xCounts.get(category) || 0) + 1);
        yCounts.set(labels[i], (yCounts.get(labels[i]) || 0) + 1);
    }

    const n = values.length;
    let score = 0;

    for (const [key, count] of joint) {
        const [category, label] = key.split(":").map(Number);

        const pxy = count / n;
        const px = xCounts.get(category) / n;
        const py = yCounts.get(label) / n;

        score += pxy * Math.log2(pxy / (px * py));
    }

    return Math.max(0, score);
}

class Dataset {
    constructor(features, rows, labels) {
        if (!Array.isArray(features) || features.length === 0) {
            throw new ValidationError("Feature names are required.");
        }

        if (!Array.isArray(rows) || rows.length < 3) {
            throw new ValidationError("At least three rows are required.");
        }

        if (rows.length !== labels.length) {
            throw new ValidationError("Rows and labels must match.");
        }

        if (new Set(features).size !== features.length) {
            throw new ValidationError("Feature names must be unique.");
        }

        if (new Set(labels).size < 2 ||
            labels.some(label => label !== 0 && label !== 1)) {
            throw new ValidationError("Labels must contain both binary classes.");
        }

        this.features = [...features];
        this.rows = rows.map(row => {
            if (!Array.isArray(row) || row.length !== features.length) {
                throw new ValidationError("Invalid row dimensions.");
            }

            if (row.some(value =>
                typeof value !== "number" || !Number.isFinite(value)
            )) {
                throw new ValidationError("Feature values must be finite numbers.");
            }

            return [...row];
        });

        this.labels = [...labels];
        Object.freeze(this.features);
    }

    get featureCount() {
        return this.features.length;
    }

    column(index) {
        if (!Number.isInteger(index) ||
            index < 0 || index >= this.featureCount) {
            throw new ValidationError("Invalid feature index.");
        }

        return this.rows.map(row => row[index]);
    }

    subset(indices) {
        if (!indices.length) {
            throw new ValidationError("An empty subset is not permitted.");
        }

        return this.rows.map(
            row => indices.map(index => row[index])
        );
    }
}

class SelectionPolicy {
    constructor({
        maxFeatures = 3,
        varianceThreshold = 1e-8,
        correlationThreshold = 0.95,
        minimumScore = 0
    } = {}) {
        if (!Number.isInteger(maxFeatures) || maxFeatures < 1) {
            throw new ValidationError("maxFeatures must be positive.");
        }

        if (varianceThreshold < 0 ||
            correlationThreshold < 0 ||
            correlationThreshold > 1 ||
            minimumScore < 0) {
            throw new ValidationError("Invalid selection policy.");
        }

        Object.assign(this, {
            maxFeatures,
            varianceThreshold,
            correlationThreshold,
            minimumScore
        });

        Object.freeze(this);
    }
}

class FeatureSelectionEngine extends EventEmitter {
    constructor(dataset, policy = new SelectionPolicy()) {
        super();

        if (!(dataset instanceof Dataset)) {
            throw new ValidationError("A valid Dataset is required.");
        }

        this.dataset = dataset;
        this.policy = policy;
        this.auditLog = [];
        this.running = false;
    }

    emitAudit(event, details) {
        const record = {
            event,
            timestamp: new Date().toISOString(),
            details
        };

        this.auditLog.push(record);
        this.emit("audit", record);
    }

    filter() {
        const candidates = [];

        for (let i = 0; i < this.dataset.featureCount; i++) {
            const values = this.dataset.column(i);
            const featureVariance = variance(values);

            if (featureVariance <= this.policy.varianceThreshold) {
                continue;
            }

            candidates.push({
                index: i,
                name: this.dataset.features[i],
                variance: featureVariance,
                mutualInformation: mutualInformation(
                    values, this.dataset.labels
                )
            });
        }

        candidates.sort(
            (a, b) =>
                b.mutualInformation - a.mutualInformation ||
                a.name.localeCompare(b.name)
        );

        const selected = [];

        for (const candidate of candidates) {
            const redundant = selected.some(index =>
                Math.abs(correlation(
                    this.dataset.column(candidate.index),
                    this.dataset.column(index)
                )) >= this.policy.correlationThreshold
            );

            if (!redundant &&
                candidate.mutualInformation >= this.policy.minimumScore) {
                selected.push(candidate.index);
            }

            if (selected.length >= this.policy.maxFeatures) {
                break;
            }
        }

        if (!selected.length) {
            throw new FeatureSelectionError(
                "No features passed the filter policy."
            );
        }

        const result = {
            method: SelectionMethod.FILTER,
            features: selected.map(i => this.dataset.features[i]),
            ranking: candidates
        };

        this.emitAudit("FILTER_COMPLETED", result);

        return result;
    }

    /*
     * A simple nearest-centroid evaluator provides a distinct
     * model-based wrapper objective.
     */
    evaluate(indices) {
        const rows = this.dataset.subset(indices);
        const labels = this.dataset.labels;

        const means = new Map();

        for (const label of [0, 1]) {
            const group = rows.filter(
                (_, index) => labels[index] === label
            );

            means.set(
                label,
                indices.map((_, column) =>
                    mean(group.map(row => row[column]))
                )
            );
        }

        let correct = 0;

        for (let i = 0; i < rows.length; i++) {
            let distances = [0, 0];

            for (const label of [0, 1]) {
                const center = means.get(label);

                distances[label] = rows[i].reduce(
                    (sum, value, column) =>
                        sum + (value - center[column]) ** 2,
                    0
                );
            }

            const prediction = distances[0] <= distances[1] ? 0 : 1;

            if (prediction === labels[i]) correct++;
        }

        return correct / rows.length;
    }

    wrapper() {
        const selected = [];
        const remaining = new Set(
            Array.from(
                { length: this.dataset.featureCount },
                (_, index) => index
            )
        );

        const history = [];

        while (
            remaining.size &&
            selected.length < this.policy.maxFeatures
        ) {
            let bestIndex = -1;
            let bestScore = -Infinity;

            for (const index of remaining) {
                const score = this.evaluate([...selected, index]);

                if (
                    score > bestScore ||
                    (score === bestScore && index < bestIndex)
                ) {
                    bestIndex = index;
                    bestScore = score;
                }
            }

            if (bestIndex < 0) break;

            selected.push(bestIndex);
            remaining.delete(bestIndex);

            history.push({
                feature: this.dataset.features[bestIndex],
                accuracy: bestScore
            });
        }

        const result = {
            method: SelectionMethod.WRAPPER,
            features: selected.map(i => this.dataset.features[i]),
            history
        };

        this.emitAudit("WRAPPER_COMPLETED", result);

        return result;
    }

    embedded() {
        /*
         * Standardized univariate class-separation coefficients
         * provide an inexpensive embedded-style scoring model.
         * The regularization penalty shrinks weak coefficients.
         */
        const coefficients = [];

        for (let i = 0; i < this.dataset.featureCount; i++) {
            const values = this.dataset.column(i);

            const positive = values.filter(
                (_, index) => this.dataset.labels[index] === 1
            );

            const negative = values.filter(
                (_, index) => this.dataset.labels[index] === 0
            );

            const pooledScale = Math.sqrt(
                (variance(positive) + variance(negative)) / 2
            );

            const difference = pooledScale < 1e-12
                ? 0
                : Math.abs(mean(positive) - mean(negative)) /
                  pooledScale;

            const coefficient = Math.max(0, difference - 0.2);

            coefficients.push({
                index: i,
                name: this.dataset.features[i],
                coefficient
            });
        }

        coefficients.sort(
            (a, b) => b.coefficient - a.coefficient
        );

        const selected = coefficients
            .filter(item => item.coefficient > 0)
            .slice(0, this.policy.maxFeatures);

        if (!selected.length) {
            throw new FeatureSelectionError(
                "No features passed embedded selection."
            );
        }

        const result = {
            method: SelectionMethod.EMBEDDED,
            features: selected.map(item => item.name),
            coefficients
        };

        this.emitAudit("EMBEDDED_COMPLETED", result);

        return result;
    }

    rfe(targetCount = this.policy.maxFeatures) {
        if (!Number.isInteger(targetCount) ||
            targetCount < 1 ||
            targetCount > this.dataset.featureCount) {
            throw new ValidationError("Invalid RFE target count.");
        }

        const active = Array.from(
            { length: this.dataset.featureCount },
            (_, index) => index
        );

        const history = [];

        while (active.length > targetCount) {
            const ranked = active.map(index => {
                const values = this.dataset.column(index);

                const positive = values.filter(
                    (_, i) => this.dataset.labels[i] === 1
                );

                const negative = values.filter(
                    (_, i) => this.dataset.labels[i] === 0
                );

                const scale = Math.sqrt(
                    (variance(positive) + variance(negative)) / 2
                );

                return {
                    index,
                    score: scale < 1e-12
                        ? 0
                        : Math.abs(mean(positive) - mean(negative)) / scale
                };
            });

            ranked.sort(
                (a, b) => a.score - b.score || a.index - b.index
            );

            const removed = ranked[0];

            active.splice(active.indexOf(removed.index), 1);

            history.push({
                feature: this.dataset.features[removed.index],
                score: removed.score,
                remaining: active.length
            });
        }

        const result = {
            method: SelectionMethod.RFE,
            features: active.map(i => this.dataset.features[i]),
            eliminationHistory: history
        };

        this.emitAudit("RFE_COMPLETED", result);

        return result;
    }

    async runExperiment() {
        if (this.running) {
            throw new FeatureSelectionError(
                "An experiment is already running."
            );
        }

        this.running = true;
        this.emitAudit("EXPERIMENT_STARTED", {});

        try {
            // Yield between methods so asynchronous consumers can process events.
            const results = [];

            for (const method of [
                () => this.filter(),
                () => this.wrapper(),
                () => this.embedded(),
                () => this.rfe()
            ]) {
                await new Promise(resolve => setImmediate(resolve));
                results.push(method());
            }

            this.emitAudit("EXPERIMENT_COMPLETED", {
                methods: results.length
            });

            return results;
        } catch (error) {
            this.emitAudit("EXPERIMENT_FAILED", {
                message: error.message
            });

            throw error;
        } finally {
            this.running = false;
        }
    }

    async saveReport(results, directory = "feature_selection_output") {
        await fs.mkdir(directory, { recursive: true });

        const report = {
            createdAt: new Date().toISOString(),
            featureNames: this.dataset.features,
            sampleCount: this.dataset.rows.length,
            results,
            auditLog: this.auditLog
        };

        const destination = path.resolve(
            directory,
            "feature_selection_report.json"
        );

        await fs.writeFile(
            destination,
            JSON.stringify(report, null, 2),
            { encoding: "utf8", flag: "w" }
        );

        return destination;
    }
}

function generateDataset(samples = 180, seed = 42) {
    let state = seed >>> 0;

    function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    }

    const features = [
        "income",
        "age",
        "balance",
        "transactions",
        "income_duplicate",
        "constant"
    ];

    const rows = [];
    const labels = [];

    for (let i = 0; i < samples; i++) {
        const income = 20000 + random() * 130000;
        const age = 20 + random() * 50;
        const balance = income * 0.5 + random() * 10000;
        const transactions = random() * 100;
        const duplicate = income + (random() - 0.5) * 200;

        const signal =
            -2 +
            0.000025 * (income - 85000) +
            0.025 * (age - 45) +
            0.00003 * (balance - 40000);

        const probability = 1 / (1 + Math.exp(-signal));

        rows.push([
            income,
            age,
            balance,
            transactions,
            duplicate,
            10
        ]);

        labels.push(random() < probability ? 1 : 0);
    }

    if (new Set(labels).size < 2) {
        throw new FeatureSelectionError(
            "Synthetic data contains only one class."
        );
    }

    return new Dataset(features, rows, labels);
}

async function main() {
    const dataset = generateDataset();

    const policy = new SelectionPolicy({
        maxFeatures: 3,
        varianceThreshold: 1e-8,
        correlationThreshold: 0.95,
        minimumScore: 0
    });

    const engine = new FeatureSelectionEngine(dataset, policy);

    engine.on("audit", event => {
        console.log(`[${event.event}]`);
    });

    const results = await engine.runExperiment();

    console.log("\nFEATURE SELECTION RESULTS");

    for (const result of results) {
        console.log(
            `${result.method}: ${result.features.join(", ")}`
        );
    }

    const reportPath = await engine.saveReport(results);

    console.log(`\nReport saved: ${reportPath}`);

    // Input validation test.
    try {
        new Dataset(["x"], [[1, 2]], [0]);
        throw new Error("Validation test failed.");
    } catch (error) {
        if (!(error instanceof ValidationError)) {
            throw error;
        }
    }

    console.log("Validation tests passed.");
}

main().catch(error => {
    console.error("Feature selection failed:", error.message);
    process.exitCode = 1;
});
