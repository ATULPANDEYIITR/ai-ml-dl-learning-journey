"use strict";

/*
 * Decision tree optimization in JavaScript.
 *
 * This implementation focuses on a production-oriented model lifecycle:
 * validation, depth-constrained tree induction, asynchronous evaluation,
 * validation-based depth selection, post-pruning, and feature importance.
 *
 * Runtime: Node.js 18 or later.
 */

const { setTimeout: delay } = require("node:timers/promises");

function assertFiniteNumber(value, description) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new TypeError(`${description} must be a finite number.`);
    }
}

function validateSamples(samples) {
    if (!Array.isArray(samples) || samples.length === 0) {
        throw new TypeError("Training samples must be a non-empty array.");
    }

    const width = samples[0].features.length;
    if (!Number.isInteger(width) || width < 1) {
        throw new TypeError("Each sample must have at least one feature.");
    }

    for (const [index, sample] of samples.entries()) {
        if (
            !sample ||
            !Array.isArray(sample.features) ||
            sample.features.length !== width ||
            typeof sample.label !== "string" ||
            sample.label.length === 0
        ) {
            throw new TypeError(`Malformed sample at index ${index}.`);
        }

        sample.features.forEach((value, featureIndex) => {
            assertFiniteNumber(value, `Feature ${featureIndex}`);
        });
    }

    return width;
}

function gini(labels) {
    const counts = new Map();

    for (const label of labels) {
        counts.set(label, (counts.get(label) || 0) + 1);
    }

    const total = labels.length;
    if (total === 0) return 0;

    let squaredProbabilities = 0;
    for (const count of counts.values()) {
        squaredProbabilities += (count / total) ** 2;
    }

    return 1 - squaredProbabilities;
}

function majorityLabel(labels) {
    const counts = new Map();

    for (const label of labels) {
        counts.set(label, (counts.get(label) || 0) + 1);
    }

    return [...counts.entries()]
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
}

function partition(samples, featureIndex, threshold) {
    const left = [];
    const right = [];

    for (const sample of samples) {
        if (sample.features[featureIndex] <= threshold) {
            left.push(sample);
        } else {
            right.push(sample);
        }
    }

    return { left, right };
}

function bestSplit(samples, featureCount, minSamplesLeaf) {
    const labels = samples.map(sample => sample.label);
    const parentImpurity = gini(labels);
    let winner = null;

    for (let featureIndex = 0; featureIndex < featureCount; featureIndex++) {
        const values = [...new Set(
            samples.map(sample => sample.features[featureIndex])
        )].sort((a, b) => a - b);

        for (let i = 0; i < values.length - 1; i++) {
            const lower = values[i];
            const upper = values[i + 1];
            const threshold = lower + (upper - lower) / 2;
            const { left, right } = partition(
                samples,
                featureIndex,
                threshold
            );

            if (
                left.length < minSamplesLeaf ||
                right.length < minSamplesLeaf
            ) {
                continue;
            }

            const weightedImpurity =
                left.length / samples.length * gini(left.map(s => s.label)) +
                right.length / samples.length * gini(right.map(s => s.label));

            const gain = parentImpurity - weightedImpurity;

            if (!winner || gain > winner.gain) {
                winner = { featureIndex, threshold, left, right, gain };
            }
        }
    }

    return winner;
}

class DecisionTree {
    constructor({
        maxDepth = 4,
        minSamplesSplit = 4,
        minSamplesLeaf = 2,
        minGain = 0.001
    } = {}) {
        if (maxDepth !== null && (!Number.isInteger(maxDepth) || maxDepth < 0)) {
            throw new RangeError("maxDepth must be a non-negative integer or null.");
        }
        if (!Number.isInteger(minSamplesSplit) || minSamplesSplit < 2) {
            throw new RangeError("minSamplesSplit must be at least 2.");
        }
        if (!Number.isInteger(minSamplesLeaf) || minSamplesLeaf < 1) {
            throw new RangeError("minSamplesLeaf must be at least 1.");
        }
        assertFiniteNumber(minGain, "minGain");
        if (minGain < 0) throw new RangeError("minGain cannot be negative.");

        this.options = { maxDepth, minSamplesSplit, minSamplesLeaf, minGain };
        this.root = null;
        this.featureCount = 0;
    }

    fit(samples) {
        this.featureCount = validateSamples(samples);
        this.root = this.#grow(samples, 0);
        return this;
    }

    #grow(samples, depth) {
        const labels = samples.map(sample => sample.label);
        const node = {
            prediction: majorityLabel(labels),
            sampleCount: samples.length,
            impurity: gini(labels),
            depth,
            featureIndex: null,
            threshold: null,
            gain: 0,
            left: null,
            right: null
        };

        const { maxDepth, minSamplesSplit, minSamplesLeaf, minGain } =
            this.options;

        if (
            node.impurity === 0 ||
            (maxDepth !== null && depth >= maxDepth) ||
            samples.length < minSamplesSplit
        ) {
            return node;
        }

        const split = bestSplit(samples, this.featureCount, minSamplesLeaf);

        if (!split || split.gain <= 0 || split.gain < minGain) {
            return node;
        }

        Object.assign(node, {
            featureIndex: split.featureIndex,
            threshold: split.threshold,
            gain: split.gain,
            left: this.#grow(split.left, depth + 1),
            right: this.#grow(split.right, depth + 1)
        });

        return node;
    }

    #requireRoot() {
        if (!this.root) {
            throw new Error("The tree has not been fitted.");
        }
        return this.root;
    }

    predictOne(features) {
        if (!Array.isArray(features) || features.length !== this.featureCount) {
            throw new TypeError("Prediction feature count does not match training.");
        }

        features.forEach((value, index) => {
            assertFiniteNumber(value, `Prediction feature ${index}`);
        });

        let node = this.#requireRoot();

        while (node.featureIndex !== null) {
            node = features[node.featureIndex] <= node.threshold
                ? node.left
                : node.right;
        }

        return node.prediction;
    }

    predict(samples) {
        return samples.map(features => this.predictOne(features));
    }

    getStatistics() {
        const root = this.#requireRoot();
        const importanceTotals = Array(this.featureCount).fill(0);
        let nodes = 0;
        let leaves = 0;
        let maxDepth = 0;

        const visit = node => {
            nodes++;
            maxDepth = Math.max(maxDepth, node.depth);

            if (node.featureIndex === null) {
                leaves++;
                return;
            }

            importanceTotals[node.featureIndex] +=
                node.sampleCount * node.gain;

            visit(node.left);
            visit(node.right);
        };

        visit(root);

        const totalImportance = importanceTotals.reduce((a, b) => a + b, 0);
        const featureImportances = importanceTotals.map(value =>
            totalImportance === 0 ? 0 : value / totalImportance
        );

        return { nodes, leaves, depth: maxDepth, featureImportances };
    }

    prune(alpha) {
        assertFiniteNumber(alpha, "alpha");
        if (alpha < 0) throw new RangeError("alpha cannot be negative.");

        const leafCount = node =>
            node.featureIndex === null
                ? 1
                : leafCount(node.left) + leafCount(node.right);

        const subtreeRisk = node =>
            node.featureIndex === null
                ? node.sampleCount * node.impurity
                : subtreeRisk(node.left) + subtreeRisk(node.right);

        const visit = node => {
            if (node.featureIndex === null) return node;

            node.left = visit(node.left);
            node.right = visit(node.right);

            const leaves = leafCount(node);
            if (leaves <= 1) return node;

            const leafRisk = node.sampleCount * node.impurity;
            const effectiveAlpha =
                (leafRisk - subtreeRisk(node)) / (leaves - 1);

            if (effectiveAlpha <= alpha) {
                node.featureIndex = null;
                node.threshold = null;
                node.gain = 0;
                node.left = null;
                node.right = null;
            }

            return node;
        };

        this.root = visit(this.#requireRoot());
        this.#refreshDepths();
        return this;
    }

    #refreshDepths() {
        const visit = (node, depth) => {
            node.depth = depth;
            if (node.featureIndex !== null) {
                visit(node.left, depth + 1);
                visit(node.right, depth + 1);
            }
        };

        visit(this.#requireRoot(), 0);
    }
}

function accuracy(actual, predicted) {
    if (actual.length === 0 || actual.length !== predicted.length) {
        throw new RangeError("Accuracy requires equally sized, non-empty arrays.");
    }

    return actual.reduce(
        (correct, label, index) => correct + Number(label === predicted[index]),
        0
    ) / actual.length;
}

function makeEquipmentData(size = 360, seed = 2718) {
    // A small seeded generator keeps the demonstration repeatable.
    let state = seed >>> 0;
    const random = () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };

    const samples = [];

    for (let i = 0; i < size; i++) {
        const vibration = 0.2 + random() * 8.8;
        const temperature = 30 + random() * 80;
        const serviceHours = random() * 18;
        const priorFaults = Math.floor(random() * 5);

        const score =
            0.9 * vibration +
            0.035 * temperature +
            0.045 * serviceHours +
            0.9 * priorFaults +
            (random() - 0.5) * 2.4;

        samples.push({
            features: [vibration, temperature, serviceHours, priorFaults],
            label: score > 8.8 ? "failure" : "normal"
        });
    }

    return samples;
}

function splitData(samples, validationFraction = 0.25) {
    if (!(validationFraction > 0 && validationFraction < 1)) {
        throw new RangeError("validationFraction must be between zero and one.");
    }

    // Interleave both classes so validation is less sensitive to source order.
    const normal = samples.filter(sample => sample.label === "normal");
    const failures = samples.filter(sample => sample.label === "failure");

    const training = [];
    const validation = [];

    for (const group of [normal, failures]) {
        const boundary = Math.max(
            1,
            Math.min(group.length - 1, Math.floor(group.length * (1 - validationFraction)))
        );

        training.push(...group.slice(0, boundary));
        validation.push(...group.slice(boundary));
    }

    if (!training.length || !validation.length) {
        throw new Error("Insufficient examples to create a train-validation split.");
    }

    return { training, validation };
}

async function evaluateCandidates(training, validation) {
    const actual = validation.map(sample => sample.label);
    const results = [];

    for (const maxDepth of [1, 2, 3, 4, 6, null]) {
        // Yield between candidates so a larger evaluation job can remain
        // responsive to Node.js event-loop work.
        await delay(0);

        const model = new DecisionTree({
            maxDepth,
            minSamplesSplit: 6,
            minSamplesLeaf: 3,
            minGain: 0.001
        }).fit(training);

        const predictions = model.predict(
            validation.map(sample => sample.features)
        );

        const statistics = model.getStatistics();

        results.push({
            maxDepth: maxDepth === null ? "unlimited" : maxDepth,
            trainAccuracy: accuracy(
                training.map(sample => sample.label),
                model.predict(training.map(sample => sample.features))
            ),
            validationAccuracy: accuracy(actual, predictions),
            ...statistics,
            model
        });
    }

    return results;
}

async function main() {
    console.log("Decision Tree Optimization: Equipment Failure Classification");
    console.log("=============================================================");

    const samples = makeEquipmentData();
    const { training, validation } = splitData(samples);

    console.log(
        `Training records: ${training.length}; validation records: ${validation.length}`
    );

    const candidates = await evaluateCandidates(training, validation);

    console.log("\nDepth selection results");
    console.table(candidates.map(({ model, ...result }) => ({
        ...result,
        featureImportances: undefined
    })));

    // Validation performance selects a model; tree size breaks ties in favor
    // of a simpler explanation. The test set should remain untouched until
    // the final model and all optimization settings have been chosen.
    const selected = [...candidates].sort((a, b) =>
        b.validationAccuracy - a.validationAccuracy ||
        a.nodes - b.nodes
    )[0];

    console.log(`\nSelected maximum depth: ${selected.maxDepth}`);
    console.log("Selected tree statistics:", selected.model.getStatistics());

    const postPruned = new DecisionTree({
        maxDepth: null,
        minSamplesSplit: 4,
        minSamplesLeaf: 2,
        minGain: 0
    }).fit(training);

    console.log("\nBefore cost-complexity pruning:");
    console.log(postPruned.getStatistics());

    postPruned.prune(0.012);

    console.log("\nAfter cost-complexity pruning:");
    console.log(postPruned.getStatistics());

    const sampleAsset = [7.4, 96, 11, 3];
    console.log("\nAsset inspection:", {
        vibration: sampleAsset[0],
        temperature: sampleAsset[1],
        serviceHours: sampleAsset[2],
        priorFaults: sampleAsset[3],
        predictedCondition: selected.model.predictOne(sampleAsset)
    });

    try {
        selected.model.predictOne([7.4, Infinity, 11, 3]);
    } catch (error) {
        console.log("\nInvalid-input protection:", error.message);
    }

    console.log(
        "\nFeature importance is an internal measure of impurity reduction. " +
        "Correlated measurements can share or distort importance, and a high " +
        "importance score does not establish a causal relationship."
    );
}

main().catch(error => {
    console.error("Model evaluation failed:", error.message);
    process.exitCode = 1;
});
