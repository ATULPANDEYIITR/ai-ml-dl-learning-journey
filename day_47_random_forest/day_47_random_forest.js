"use strict";

/*
 * Random Forest: bootstrap aggregation and random feature selection.
 *
 * This file uses a JavaScript-specific event-driven training workflow.
 * Run with Node.js 18 or later. No third-party dependencies are required.
 */

const { EventEmitter } = require("node:events");

function createRandom(seed) {
    // A deterministic generator makes experiments repeatable.
    let state = seed >>> 0;
    return function random() {
        state = (Math.imul(1664525, state) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function validateRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        throw new TypeError("Training rows must be a non-empty array.");
    }

    const featureCount = rows[0].features.length;
    if (!Number.isInteger(featureCount) || featureCount < 1) {
        throw new TypeError("Every row needs numerical features.");
    }

    for (const row of rows) {
        if (
            !Array.isArray(row.features) ||
            row.features.length !== featureCount ||
            !row.features.every(Number.isFinite) ||
            typeof row.label !== "string" ||
            row.label.length === 0
        ) {
            throw new TypeError("Invalid training observation.");
        }
    }

    return featureCount;
}

function majorityVote(labels) {
    const counts = new Map();
    for (const label of labels) {
        counts.set(label, (counts.get(label) || 0) + 1);
    }

    const highest = Math.max(...counts.values());
    return [...counts.entries()]
        .filter(([, count]) => count === highest)
        .map(([label]) => label)
        .sort()[0];
}

function gini(labels) {
    if (labels.length === 0) return 0;

    const counts = new Map();
    for (const label of labels) {
        counts.set(label, (counts.get(label) || 0) + 1);
    }

    let impurity = 1;
    for (const count of counts.values()) {
        const probability = count / labels.length;
        impurity -= probability * probability;
    }
    return impurity;
}

function bootstrapRows(rows, random) {
    const selectedIndices = [];
    const outOfBagIndices = new Set();

    for (let i = 0; i < rows.length; i++) {
        selectedIndices.push(Math.floor(random() * rows.length));
    }

    const selectedSet = new Set(selectedIndices);
    for (let i = 0; i < rows.length; i++) {
        if (!selectedSet.has(i)) outOfBagIndices.add(i);
    }

    // Duplicate row references are intentional: bootstrap sampling is
    // sampling with replacement, not a random split without replacement.
    return {
        rows: selectedIndices.map(index => rows[index]),
        outOfBagIndices
    };
}

class ClassificationTree {
    constructor(options = {}) {
        this.maxDepth = options.maxDepth ?? 5;
        this.minLeaf = options.minLeaf ?? 2;
        this.maxFeatures = options.maxFeatures ?? null;
        this.random = createRandom(options.seed ?? 1);
        this.importance = new Map();
        this.root = null;
        this.featureCount = 0;

        if (this.maxDepth < 0 || this.minLeaf < 1) {
            throw new RangeError("Invalid tree depth or leaf size.");
        }
    }

    fit(rows) {
        this.featureCount = validateRows(rows);
        this.importance.clear();
        this.root = this.grow(rows, 0);
        return this;
    }

    grow(rows, depth) {
        const labels = rows.map(row => row.label);
        const counts = new Map();
        for (const label of labels) {
            counts.set(label, (counts.get(label) || 0) + 1);
        }

        const node = {
            prediction: majorityVote(labels),
            feature: null,
            threshold: null,
            left: null,
            right: null
        };

        if (
            depth >= this.maxDepth ||
            rows.length < this.minLeaf * 2 ||
            counts.size === 1
        ) {
            return node;
        }

        const requested = this.maxFeatures ?? this.featureCount;
        if (requested > this.featureCount || requested < 1) {
            throw new RangeError("Invalid random feature count.");
        }

        const candidates = Array.from(
            { length: this.featureCount },
            (_, index) => index
        );

        // Fisher-Yates shuffling selects a random subset without duplicates.
        for (let i = candidates.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
        }

        let best = null;
        const parentImpurity = gini(labels);

        for (const feature of candidates.slice(0, requested)) {
            const values = [...new Set(
                rows.map(row => row.features[feature])
            )].sort((a, b) => a - b);

            for (let i = 0; i < values.length - 1; i++) {
                const threshold = (values[i] + values[i + 1]) / 2;
                const left = [];
                const right = [];

                for (const row of rows) {
                    if (row.features[feature] <= threshold) left.push(row);
                    else right.push(row);
                }

                if (left.length < this.minLeaf || right.length < this.minLeaf) {
                    continue;
                }

                const weighted =
                    left.length / rows.length * gini(left.map(row => row.label)) +
                    right.length / rows.length * gini(right.map(row => row.label));

                const gain = parentImpurity - weighted;
                if (gain > 1e-12 && (!best || gain > best.gain)) {
                    best = { feature, threshold, left, right, gain };
                }
            }
        }

        if (!best) return node;

        node.feature = best.feature;
        node.threshold = best.threshold;
        node.left = this.grow(best.left, depth + 1);
        node.right = this.grow(best.right, depth + 1);

        this.importance.set(
            best.feature,
            (this.importance.get(best.feature) || 0) +
            best.gain * rows.length
        );

        return node;
    }

    predict(features) {
        if (!this.root) throw new Error("Tree has not been trained.");
        if (
            !Array.isArray(features) ||
            features.length !== this.featureCount ||
            !features.every(Number.isFinite)
        ) {
            throw new TypeError("Invalid prediction features.");
        }

        let node = this.root;
        while (node.feature !== null) {
            node = features[node.feature] <= node.threshold
                ? node.left
                : node.right;
        }
        return node.prediction;
    }
}

class RandomForest extends EventEmitter {
    constructor(options = {}) {
        super();
        this.treeCount = options.treeCount ?? 31;
        this.maxDepth = options.maxDepth ?? 5;
        this.maxFeatures = options.maxFeatures ?? 2;
        this.seed = options.seed ?? 42;
        this.trees = [];
        this.oobPredictions = new Map();
        this.importances = new Map();

        if (!Number.isInteger(this.treeCount) || this.treeCount < 1) {
            throw new RangeError("treeCount must be a positive integer.");
        }
    }

    fit(rows) {
        const featureCount = validateRows(rows);
        if (this.maxFeatures > featureCount || this.maxFeatures < 1) {
            throw new RangeError("maxFeatures must fit the feature dimension.");
        }

        this.trees = [];
        this.oobPredictions.clear();
        this.importances.clear();

        const votes = new Map();
        const masterRandom = createRandom(this.seed);

        for (let treeIndex = 0; treeIndex < this.treeCount; treeIndex++) {
            const treeSeed = Math.floor(masterRandom() * 4294967296);
            const sample = bootstrapRows(rows, createRandom(treeSeed));

            const tree = new ClassificationTree({
                maxDepth: this.maxDepth,
                minLeaf: 2,
                maxFeatures: this.maxFeatures,
                seed: treeSeed ^ 0x9e3779b9
            }).fit(sample.rows);

            this.trees.push(tree);

            for (const [feature, value] of tree.importance) {
                this.importances.set(
                    feature,
                    (this.importances.get(feature) || 0) + value
                );
            }

            for (const index of sample.outOfBagIndices) {
                if (!votes.has(index)) votes.set(index, []);
                votes.get(index).push(tree.predict(rows[index].features));
            }

            // Listeners can monitor training without coupling the model to
            // a particular user interface or logging implementation.
            this.emit("treeTrained", {
                index: treeIndex,
                total: this.treeCount,
                oobRows: sample.outOfBagIndices.size
            });
        }

        for (const [index, predictions] of votes) {
            this.oobPredictions.set(index, majorityVote(predictions));
        }

        const totalImportance = [...this.importances.values()]
            .reduce((sum, value) => sum + value, 0);

        if (totalImportance > 0) {
            for (const [feature, value] of this.importances) {
                this.importances.set(feature, value / totalImportance);
            }
        }

        this.emit("trainingComplete", {
            trees: this.trees.length,
            oobRows: this.oobPredictions.size
        });

        return this;
    }

    predict(features) {
        if (this.trees.length === 0) {
            throw new Error("Train the forest before prediction.");
        }
        return majorityVote(this.trees.map(tree => tree.predict(features)));
    }

    oobAccuracy(rows) {
        if (this.oobPredictions.size === 0) return null;

        let correct = 0;
        for (const [index, prediction] of this.oobPredictions) {
            if (prediction === rows[index].label) correct++;
        }
        return correct / this.oobPredictions.size;
    }
}

function createInspectionRows(count = 180, seed = 2026) {
    const random = createRandom(seed);
    const rows = [];

    for (let i = 0; i < count; i++) {
        const vibration = random() * 10;
        const temperature = 35 + random() * 75;
        const operatingHours = random() * 12000;
        const pressureDeviation = random() * 8;

        const risk =
            0.28 * vibration +
            0.045 * Math.max(0, temperature - 70) +
            0.00025 * operatingHours +
            0.65 * pressureDeviation +
            0.12 * vibration * pressureDeviation;

        rows.push({
            features: [
                vibration,
                temperature,
                operatingHours,
                pressureDeviation
            ],
            label: risk > 8 ? "maintenance_required" : "normal"
        });
    }

    return rows;
}

function main() {
    const rows = createInspectionRows();
    const split = Math.floor(rows.length * 0.8);
    const training = rows.slice(0, split);
    const testing = rows.slice(split);

    const forest = new RandomForest({
        treeCount: 41,
        maxDepth: 6,
        maxFeatures: 2,
        seed: 71
    });

    forest.on("treeTrained", event => {
        if ((event.index + 1) % 10 === 0) {
            console.log(`Trained ${event.index + 1}/${event.total} trees`);
        }
    });

    forest.on("trainingComplete", event => {
        console.log(
            `Training complete: ${event.trees} trees, ` +
            `${event.oobRows} observations with OOB predictions`
        );
    });

    forest.fit(training);

    const correct = testing.filter(
        row => forest.predict(row.features) === row.label
    ).length;

    console.log(`Test accuracy: ${(correct / testing.length).toFixed(3)}`);

    const oob = forest.oobAccuracy(training);
    console.log(
        `Out-of-bag accuracy: ${oob === null ? "unavailable" : oob.toFixed(3)}`
    );

    const names = [
        "vibration",
        "temperature",
        "operating hours",
        "pressure deviation"
    ];

    console.log("\nNormalized split importance:");
    for (const [feature, importance] of [...forest.importances.entries()]
        .sort((a, b) => b[1] - a[1])) {
        console.log(`${names[feature]}: ${importance.toFixed(4)}`);
    }

    const asset = [8.1, 97.0, 9900, 6.4];
    console.log("\nAsset inspection:");
    console.log(`Prediction: ${forest.predict(asset)}`);

    // Invalid dimensions are rejected before traversal.
    try {
        forest.predict([8.1, 97.0]);
    } catch (error) {
        console.log(`Validation caught: ${error.message}`);
    }
}

if (require.main === module) {
    main();
}

module.exports = {
    ClassificationTree,
    RandomForest,
    bootstrapRows,
    gini,
    majorityVote
};
