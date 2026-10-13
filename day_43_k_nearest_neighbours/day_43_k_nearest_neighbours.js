"use strict";

/*
 * knn_workflow.js
 *
 * Event-driven k-NN classification and regression using modern JavaScript.
 * The implementation emphasizes reusable distance functions, neighborhood
 * inspection, streaming training updates, reproducible predictions, and
 * cross-validation without third-party packages.
 *
 * Run with Node.js 18 or later.
 */

const assert = require("node:assert/strict");

function validateVector(vector, name = "vector") {
    if (!Array.isArray(vector) || vector.length === 0) {
        throw new TypeError(`${name} must be a nonempty array.`);
    }

    vector.forEach((value, index) => {
        if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new TypeError(`${name}[${index}] must be a finite number.`);
        }
    });
}

function validateDataset(X, y = undefined) {
    if (!Array.isArray(X) || X.length === 0) {
        throw new TypeError("X must contain at least one sample.");
    }

    const dimensions = X[0].length;
    if (dimensions === 0) {
        throw new TypeError("Samples must contain at least one feature.");
    }

    X.forEach((sample, index) => {
        validateVector(sample, `X[${index}]`);
        if (sample.length !== dimensions) {
            throw new TypeError("All samples must have equal dimensions.");
        }
    });

    if (y !== undefined) {
        if (!Array.isArray(y) || y.length !== X.length) {
            throw new TypeError("X and y must have equal sample counts.");
        }
        if (y.some(target => target === null || target === undefined)) {
            throw new TypeError("Targets must not be null or undefined.");
        }
    }

    return dimensions;
}

function euclideanDistance(a, b) {
    validateVector(a, "a");
    validateVector(b, "b");
    if (a.length !== b.length) {
        throw new RangeError("Distance operands have different dimensions.");
    }
    return Math.hypot(...a.map((value, index) => value - b[index]));
}

function manhattanDistance(a, b) {
    validateVector(a, "a");
    validateVector(b, "b");
    if (a.length !== b.length) {
        throw new RangeError("Distance operands have different dimensions.");
    }
    return a.reduce((sum, value, index) => sum + Math.abs(value - b[index]), 0);
}

function minkowskiDistance(a, b, p = 3) {
    if (!Number.isFinite(p) || p < 1) {
        throw new RangeError("Minkowski p must be finite and at least 1.");
    }
    validateVector(a, "a");
    validateVector(b, "b");
    if (a.length !== b.length) {
        throw new RangeError("Distance operands have different dimensions.");
    }
    return a.reduce(
        (sum, value, index) => sum + Math.abs(value - b[index]) ** p,
        0
    ) ** (1 / p);
}

function cosineDistance(a, b) {
    validateVector(a, "a");
    validateVector(b, "b");
    if (a.length !== b.length) {
        throw new RangeError("Distance operands have different dimensions.");
    }

    const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
    const normA = Math.hypot(...a);
    const normB = Math.hypot(...b);

    if (normA === 0 || normB === 0) {
        throw new RangeError("Cosine distance is undefined for zero vectors.");
    }

    const similarity = Math.max(-1, Math.min(1, dot / (normA * normB)));
    return 1 - similarity;
}

const METRICS = Object.freeze({
    euclidean: euclideanDistance,
    manhattan: manhattanDistance,
    minkowski: minkowskiDistance,
    cosine: cosineDistance
});

class OnlineStandardScaler {
    constructor() {
        this.means = [];
        this.scales = [];
        this.fitted = false;
    }

    fit(X) {
        const dimensions = validateDataset(X);
        this.means = [];
        this.scales = [];

        for (let column = 0; column < dimensions; column++) {
            const values = X.map(row => row[column]);
            const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
            const variance = values.reduce(
                (sum, value) => sum + (value - mean) ** 2,
                0
            ) / values.length;
            const deviation = Math.sqrt(variance);

            this.means.push(mean);
            this.scales.push(deviation > 0 ? deviation : 1);
        }

        this.fitted = true;
        return this;
    }

    transform(X) {
        if (!this.fitted) {
            throw new Error("Fit the scaler before transforming samples.");
        }
        validateDataset(X);

        return X.map((row, rowIndex) => {
            if (row.length !== this.means.length) {
                throw new RangeError(`Feature mismatch in row ${rowIndex}.`);
            }
            return row.map(
                (value, column) => (value - this.means[column]) / this.scales[column]
            );
        });
    }
}

class KNNClassifier {
    constructor({
        k = 3,
        metric = "euclidean",
        weights = "uniform",
        minkowskiP = 3
    } = {}) {
        if (!Number.isInteger(k) || k < 1) {
            throw new RangeError("k must be a positive integer.");
        }
        if (!Object.hasOwn(METRICS, metric)) {
            throw new RangeError(`Unknown distance metric: ${metric}`);
        }
        if (!["uniform", "distance"].includes(weights)) {
            throw new RangeError("weights must be uniform or distance.");
        }
        if (!Number.isFinite(minkowskiP) || minkowskiP < 1) {
            throw new RangeError("minkowskiP must be finite and at least 1.");
        }

        this.k = k;
        this.metric = metric;
        this.weights = weights;
        this.minkowskiP = minkowskiP;
        this.samples = [];
        this.classes = [];
        this.dimensions = 0;
        this.fitted = false;
    }

    fit(X, y) {
        this.dimensions = validateDataset(X, y);
        if (this.k > X.length) {
            throw new RangeError("k cannot exceed the training sample count.");
        }

        this.samples = X.map((features, index) => ({
            features: [...features],
            label: y[index],
            index
        }));

        this.classes = [...new Set(y)].sort((a, b) =>
            String(a).localeCompare(String(b))
        );
        this.fitted = true;
        return this;
    }

    distance(a, b) {
        const fn = METRICS[this.metric];
        const result = this.metric === "minkowski"
            ? fn(a, b, this.minkowskiP)
            : fn(a, b);

        if (!Number.isFinite(result) || result < 0) {
            throw new RangeError("Distance calculation returned an invalid value.");
        }
        return result;
    }

    kneighbors(query, count = this.k) {
        if (!this.fitted) {
            throw new Error("Train the classifier before querying neighbors.");
        }
        validateVector(query, "query");

        if (query.length !== this.dimensions) {
            throw new RangeError("Query dimensions do not match training data.");
        }
        if (!Number.isInteger(count) || count < 1 || count > this.samples.length) {
            throw new RangeError("Invalid requested neighbor count.");
        }

        return this.samples
            .map(sample => ({
                ...sample,
                distance: this.distance(query, sample.features)
            }))
            .sort((a, b) => a.distance - b.distance || a.index - b.index)
            .slice(0, count);
    }

    predictOne(query) {
        const neighbors = this.kneighbors(query);
        const exactMatches = neighbors.filter(neighbor => neighbor.distance === 0);
        const selected = this.weights === "distance" && exactMatches.length
            ? exactMatches
            : neighbors;

        const totals = new Map();
        const distanceTotals = new Map();

        for (const neighbor of selected) {
            // Exact matches receive finite equal weights rather than Infinity.
            const weight = this.weights === "uniform"
                ? 1
                : neighbor.distance === 0
                    ? 1
                    : 1 / neighbor.distance;

            totals.set(
                neighbor.label,
                (totals.get(neighbor.label) || 0) + weight
            );
            distanceTotals.set(
                neighbor.label,
                (distanceTotals.get(neighbor.label) || 0) + neighbor.distance
            );
        }

        return [...this.classes].sort((a, b) => {
            const weightDifference =
                (totals.get(b) || 0) - (totals.get(a) || 0);
            if (weightDifference !== 0) return weightDifference;

            const distanceDifference =
                (distanceTotals.get(a) || 0) - (distanceTotals.get(b) || 0);
            if (distanceDifference !== 0) return distanceDifference;

            return String(a).localeCompare(String(b));
        })[0];
    }

    predict(X) {
        validateDataset(X);
        return X.map(query => this.predictOne(query));
    }

    predictProbabilities(query) {
        const neighbors = this.kneighbors(query);
        const exactMatches = neighbors.filter(neighbor => neighbor.distance === 0);
        const selected = this.weights === "distance" && exactMatches.length
            ? exactMatches
            : neighbors;

        const totals = new Map(this.classes.map(label => [label, 0]));

        for (const neighbor of selected) {
            const weight = this.weights === "uniform"
                ? 1
                : neighbor.distance === 0
                    ? 1
                    : 1 / neighbor.distance;
            totals.set(neighbor.label, totals.get(neighbor.label) + weight);
        }

        const totalWeight = [...totals.values()].reduce((sum, value) => sum + value, 0);
        return Object.fromEntries(
            this.classes.map(label => [label, totals.get(label) / totalWeight])
        );
    }

    // A training event updates the local model only after input validation.
    addTrainingExample(features, label) {
        if (!this.fitted) {
            throw new Error("Fit an initial dataset before adding examples.");
        }
        validateVector(features, "features");
        if (features.length !== this.dimensions) {
            throw new RangeError("New sample has the wrong feature count.");
        }
        if (label === null || label === undefined) {
            throw new TypeError("Label must not be null or undefined.");
        }

        this.samples.push({
            features: [...features],
            label,
            index: this.samples.length
        });
        this.classes = [...new Set(this.samples.map(sample => sample.label))]
            .sort((a, b) => String(a).localeCompare(String(b)));
    }
}

class KNNRegressor {
    constructor({ k = 3, metric = "euclidean", weights = "distance" } = {}) {
        if (!Number.isInteger(k) || k < 1) {
            throw new RangeError("k must be a positive integer.");
        }
        if (!["euclidean", "manhattan", "cosine"].includes(metric)) {
            throw new RangeError("Unsupported regression distance metric.");
        }
        if (!["uniform", "distance"].includes(weights)) {
            throw new RangeError("Unsupported weighting mode.");
        }

        this.k = k;
        this.metric = metric;
        this.weights = weights;
        this.X = [];
        this.y = [];
        this.dimensions = 0;
    }

    fit(X, y) {
        this.dimensions = validateDataset(X, y);
        if (this.k > X.length) {
            throw new RangeError("k exceeds training sample count.");
        }
        if (y.some(value => typeof value !== "number" || !Number.isFinite(value))) {
            throw new TypeError("Regression targets must be finite numbers.");
        }

        this.X = X.map(row => [...row]);
        this.y = [...y];
        return this;
    }

    predictOne(query) {
        validateVector(query, "query");
        if (query.length !== this.dimensions) {
            throw new RangeError("Query dimensions do not match training data.");
        }

        const distanceFn = METRICS[this.metric];
        const neighbors = this.X.map((features, index) => ({
            index,
            distance: distanceFn(query, features),
            target: this.y[index]
        })).sort((a, b) => a.distance - b.distance || a.index - b.index)
            .slice(0, this.k);

        const exact = neighbors.filter(neighbor => neighbor.distance === 0);
        const selected = this.weights === "distance" && exact.length ? exact : neighbors;

        const numerator = selected.reduce((sum, neighbor) => {
            const weight = this.weights === "uniform"
                ? 1
                : neighbor.distance === 0
                    ? 1
                    : 1 / neighbor.distance;
            return sum + weight * neighbor.target;
        }, 0);

        const denominator = selected.reduce((sum, neighbor) => {
            return sum + (
                this.weights === "uniform"
                    ? 1
                    : neighbor.distance === 0
                        ? 1
                        : 1 / neighbor.distance
            );
        }, 0);

        return numerator / denominator;
    }

    predict(X) {
        validateDataset(X);
        return X.map(query => this.predictOne(query));
    }
}

function accuracy(actual, predicted) {
    if (actual.length === 0 || actual.length !== predicted.length) {
        throw new RangeError("Inputs must have equal nonzero lengths.");
    }
    return actual.filter((value, index) => value === predicted[index]).length
        / actual.length;
}

function meanAbsoluteError(actual, predicted) {
    if (actual.length === 0 || actual.length !== predicted.length) {
        throw new RangeError("Inputs must have equal nonzero lengths.");
    }
    return actual.reduce(
        (sum, value, index) => sum + Math.abs(value - predicted[index]),
        0
    ) / actual.length;
}

function confusionMatrix(actual, predicted) {
    if (actual.length === 0 || actual.length !== predicted.length) {
        throw new RangeError("Inputs must have equal nonzero lengths.");
    }

    const labels = [...new Set([...actual, ...predicted])]
        .sort((a, b) => String(a).localeCompare(String(b)));

    const matrix = new Map(
        labels.map(actualLabel => [
            actualLabel,
            new Map(labels.map(predictedLabel => [predictedLabel, 0]))
        ])
    );

    actual.forEach((actualLabel, index) => {
        const row = matrix.get(actualLabel);
        row.set(predicted[index], row.get(predicted[index]) + 1);
    });

    return Object.fromEntries(
        [...matrix.entries()].map(([label, row]) => [
            label,
            Object.fromEntries(row.entries())
        ])
    );
}

function seededShuffle(values, seed = 42) {
    const result = [...values];
    let state = seed >>> 0;

    function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 0x100000000;
    }

    for (let index = result.length - 1; index > 0; index--) {
        const swapIndex = Math.floor(random() * (index + 1));
        [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
    }

    return result;
}

function crossValidate(X, y, candidateKValues, folds = 4) {
    validateDataset(X, y);
    if (!Number.isInteger(folds) || folds < 2 || folds > X.length) {
        throw new RangeError("Invalid number of folds.");
    }

    const shuffled = seededShuffle(
        X.map((_, index) => index),
        91
    );
    const partitions = Array.from({ length: folds }, () => []);

    shuffled.forEach((index, position) => {
        partitions[position % folds].push(index);
    });

    return candidateKValues.map(k => {
        const scores = [];

        for (const validationIndices of partitions) {
            const validationSet = new Set(validationIndices);
            const trainIndices = shuffled.filter(index => !validationSet.has(index));

            if (k > trainIndices.length) {
                throw new RangeError("Candidate k exceeds fold training size.");
            }

            const XTrain = trainIndices.map(index => X[index]);
            const yTrain = trainIndices.map(index => y[index]);
            const XValidation = validationIndices.map(index => X[index]);
            const yValidation = validationIndices.map(index => y[index]);

            // Preprocessing is learned inside each fold, never from validation rows.
            const scaler = new OnlineStandardScaler().fit(XTrain);
            const model = new KNNClassifier({
                k,
                metric: "euclidean",
                weights: "distance"
            }).fit(scaler.transform(XTrain), yTrain);

            scores.push(
                accuracy(yValidation, model.predict(scaler.transform(XValidation)))
            );
        }

        return {
            k,
            meanAccuracy: scores.reduce((sum, value) => sum + value, 0) / scores.length
        };
    }).sort((a, b) => b.meanAccuracy - a.meanAccuracy || a.k - b.k);
}

async function evaluateModelAsync(model, queries) {
    // Promise-based orchestration keeps the interface composable with asynchronous
    // data-loading pipelines; the brute-force neighbor search itself is synchronous.
    await Promise.resolve();
    return queries.map(query => ({
        query,
        prediction: model.predictOne(query),
        probabilities: model.predictProbabilities(query)
    }));
}

async function main() {
    console.log("Distance metric behavior");
    console.log("Euclidean:", euclideanDistance([1, 2], [4, 6]).toFixed(3));
    console.log("Manhattan:", manhattanDistance([1, 2], [4, 6]).toFixed(3));
    console.log("Minkowski p=3:", minkowskiDistance([1, 2], [4, 6], 3).toFixed(3));
    console.log("Cosine:", cosineDistance([1, 0], [0, 1]).toFixed(3));

    const X = [
        [22, 28000], [24, 32000], [27, 35000], [29, 41000],
        [35, 52000], [38, 57000], [42, 63000], [45, 71000],
        [48, 76000], [52, 85000], [55, 91000], [58, 98000]
    ];
    const y = [
        "entry", "entry", "entry", "entry",
        "professional", "professional", "professional", "professional",
        "senior", "senior", "senior", "senior"
    ];

    const scaler = new OnlineStandardScaler().fit(X);
    const normalizedX = scaler.transform(X);

    const classifier = new KNNClassifier({
        k: 3,
        metric: "euclidean",
        weights: "distance"
    }).fit(normalizedX, y);

    const queries = scaler.transform([[36, 54000], [51, 83000]]);
    console.log("\nNeighbor inspection");
    for (const query of queries) {
        console.log({
            neighbors: classifier.kneighbors(query),
            probabilities: classifier.predictProbabilities(query),
            prediction: classifier.predictOne(query)
        });
    }

    const predictions = classifier.predict(normalizedX);
    console.log("\nTraining-set diagnostic accuracy:", accuracy(y, predictions));
    console.log("Confusion matrix:", confusionMatrix(y, predictions));

    console.log("\nCross-validation");
    console.log(crossValidate(X, y, [1, 3, 5], 4));

    const regression = new KNNRegressor({ k: 3, weights: "distance" })
        .fit(
            [[45, 8], [55, 10], [60, 5], [70, 12], [80, 7], [90, 4]],
            [32, 39, 44, 49, 55, 63]
        );

    const actualPrices = [44, 49];
    const predictedPrices = regression.predict([[60, 5], [70, 12]]);
    console.log("\nRegression predictions:", predictedPrices);
    console.log("MAE:", meanAbsoluteError(actualPrices, predictedPrices));

    console.log("\nAsynchronous evaluation");
    console.log(await evaluateModelAsync(classifier, queries));

    // A new labeled observation can update the in-memory reference set.
    classifier.addTrainingExample(scaler.transform([[31, 45000]])[0], "professional");
    console.log("Updated training records:", classifier.samples.length);
}

function runTests() {
    assert.equal(euclideanDistance([0, 0], [3, 4]), 5);
    assert.equal(manhattanDistance([0, 0], [3, 4]), 7);
    assert.equal(minkowskiDistance([0, 0], [3, 4], 1), 7);
    assert.equal(cosineDistance([1, 0], [0, 1]), 1);

    const exactMatchClassifier = new KNNClassifier({
        k: 3,
        weights: "distance"
    }).fit([[0], [0], [10]], ["approved", "rejected", "rejected"]);

    assert.equal(exactMatchClassifier.predictOne([0]), "approved");

    const regressor = new KNNRegressor({ k: 2, weights: "uniform" })
        .fit([[0], [2]], [10, 20]);
    assert.equal(regressor.predictOne([1]), 15);

    assert.throws(
        () => new KNNClassifier({ k: 4 }).fit([[0], [1], [2]], ["a", "b", "c"]),
        RangeError
    );
    assert.throws(() => cosineDistance([0, 0], [1, 1]), RangeError);
    assert.throws(() => euclideanDistance([1], [1, 2]), RangeError);

    console.log("All k-NN validation tests passed.");
}

runTests();
main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
