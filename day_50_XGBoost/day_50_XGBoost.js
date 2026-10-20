"use strict";

const fs = require("node:fs");

function sigmoid(x) {
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
}

function validateDataset(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        throw new Error("Dataset must contain at least one row.");
    }

    const width = rows[0].features.length;
    for (const row of rows) {
        if (!Array.isArray(row.features) || row.features.length !== width) {
            throw new Error("All rows must have the same feature count.");
        }
        if (row.label !== 0 && row.label !== 1) {
            throw new Error("Labels must be binary.");
        }
        if (row.features.some(Number.isNaN)) {
            throw new Error("Features cannot contain NaN.");
        }
    }
}

class BoostingTree {
    constructor(feature, threshold, left, right, weight = 0) {
        this.feature = feature;
        this.threshold = threshold;
        this.left = left;
        this.right = right;
        this.weight = weight;
    }

    predict(features) {
        if (this.feature === null) return this.weight;
        return features[this.feature] <= this.threshold
            ? this.left.predict(features)
            : this.right.predict(features);
    }
}

class XGBoostPolicyModel {
    constructor(options = {}) {
        this.maxDepth = options.maxDepth ?? 2;
        this.learningRate = options.learningRate ?? 0.1;
        this.nEstimators = options.nEstimators ?? 50;
        this.lambda = options.lambda ?? 1.0;
        this.alpha = options.alpha ?? 0.0;
        this.gamma = options.gamma ?? 0.0;
        this.minChildWeight = options.minChildWeight ?? 1.0;
        this.subsample = options.subsample ?? 1.0;
        this.baseScore = options.baseScore ?? 0.5;
        this.trees = [];
        this.featureCount = 0;
    }

    leafWeight(gradients, hessians) {
        let g = gradients.reduce((a, b) => a + b, 0);
        const h = hessians.reduce((a, b) => a + b, 0);

        if (h < this.minChildWeight) return 0;

        if (this.alpha > 0) {
            if (g > this.alpha) g -= this.alpha;
            else if (g < -this.alpha) g += this.alpha;
            else g = 0;
        }

        return -g / (h + this.lambda);
    }

    gain(gLeft, hLeft, gRight, hRight, gParent, hParent) {
        if (
            hLeft < this.minChildWeight ||
            hRight < this.minChildWeight
        ) {
            return -Infinity;
        }

        const score = (g, h) => {
            if (this.alpha > 0) {
                if (g > this.alpha) g -= this.alpha;
                else if (g < -this.alpha) g += this.alpha;
                else g = 0;
            }
            return (g * g) / (h + this.lambda);
        };

        return (
            0.5 *
            (score(gLeft, hLeft) +
                score(gRight, hRight) -
                score(gParent, hParent)) -
            this.gamma
        );
    }

    buildTree(rows, gradients, hessians, depth) {
        const featureCount = rows[0].features.length;
        const leaf = new BoostingTree(
            null,
            null,
            null,
            null,
            this.leafWeight(gradients, hessians)
        );

        if (depth >= this.maxDepth || rows.length < 2) return leaf;

        let best = null;
        const parentG = gradients.reduce((a, b) => a + b, 0);
        const parentH = hessians.reduce((a, b) => a + b, 0);

        for (let feature = 0; feature < featureCount; feature++) {
            const values = [...new Set(rows.map(r => r.features[feature]))]
                .sort((a, b) => a - b);

            for (let i = 0; i < values.length - 1; i++) {
                const threshold = (values[i] + values[i + 1]) / 2;
                const left = [];
                const right = [];

                for (let j = 0; j < rows.length; j++) {
                    if (rows[j].features[feature] <= threshold) {
                        left.push(j);
                    } else {
                        right.push(j);
                    }
                }

                if (!left.length || !right.length) continue;

                const gLeft = left.reduce((s, j) => s + gradients[j], 0);
                const hLeft = left.reduce((s, j) => s + hessians[j], 0);
                const gRight = right.reduce((s, j) => s + gradients[j], 0);
                const hRight = right.reduce((s, j) => s + hessians[j], 0);

                const gain = this.gain(
                    gLeft,
                    hLeft,
                    gRight,
                    hRight,
                    parentG,
                    parentH
                );

                if (!best || gain > best.gain) {
                    best = { feature, threshold, left, right, gain };
                }
            }
        }

        if (!best || best.gain <= this.gamma) return leaf;

        const leftRows = best.left.map(i => rows[i]);
        const rightRows = best.right.map(i => rows[i]);
        const leftG = best.left.map(i => gradients[i]);
        const rightG = best.right.map(i => gradients[i]);
        const leftH = best.left.map(i => hessians[i]);
        const rightH = best.right.map(i => hessians[i]);

        return new BoostingTree(
            best.feature,
            best.threshold,
            this.buildTree(leftRows, leftG, leftH, depth + 1),
            this.buildTree(rightRows, rightG, rightH, depth + 1)
        );
    }

    fit(rows) {
        validateDataset(rows);

        this.featureCount = rows[0].features.length;
        const initialMargin = Math.log(
            this.baseScore / (1 - this.baseScore)
        );
        const margins = new Array(rows.length).fill(initialMargin);

        for (let round = 0; round < this.nEstimators; round++) {
            const probabilities = margins.map(sigmoid);
            const gradients = rows.map(
                (row, i) => probabilities[i] - row.label
            );
            const hessians = probabilities.map(
                p => Math.max(p * (1 - p), 1e-6)
            );

            let indexes = rows.map((_, i) => i);
            if (this.subsample < 1) {
                const target = Math.max(
                    1,
                    Math.round(rows.length * this.subsample)
                );
                indexes = indexes
                    .sort(() => Math.random() - 0.5)
                    .slice(0, target);
            }

            const sampledRows = indexes.map(i => rows[i]);
            const sampledG = indexes.map(i => gradients[i]);
            const sampledH = indexes.map(i => hessians[i]);

            const tree = this.buildTree(
                sampledRows,
                sampledG,
                sampledH,
                0
            );

            this.trees.push(tree);

            for (let i = 0; i < rows.length; i++) {
                margins[i] += this.learningRate * tree.predict(rows[i].features);
            }
        }

        return this;
    }

    predictProbability(features) {
        if (features.length !== this.featureCount) {
            throw new Error("Feature count mismatch.");
        }

        let margin = Math.log(
            this.baseScore / (1 - this.baseScore)
        );

        for (const tree of this.trees) {
            margin += this.learningRate * tree.predict(features);
        }

        return sigmoid(margin);
    }

    predict(features, threshold = 0.5) {
        if (!(threshold > 0 && threshold < 1)) {
            throw new Error("Threshold must be between zero and one.");
        }
        return this.predictProbability(features) >= threshold ? 1 : 0;
    }
}

class ModelEventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(event, listener) {
        if (!this.listeners.has(event)) this.listeners.set(event, []);
        this.listeners.get(event).push(listener);
    }

    emit(event, payload) {
        for (const listener of this.listeners.get(event) ?? []) {
            listener(payload);
        }
    }
}

function logLoss(rows, model) {
    const epsilon = 1e-15;
    let total = 0;

    for (const row of rows) {
        const p = Math.min(
            1 - epsilon,
            Math.max(epsilon, model.predictProbability(row.features))
        );
        total -= row.label * Math.log(p) +
            (1 - row.label) * Math.log(1 - p);
    }

    return total / rows.length;
}

const dataset = [
    [1, 0, 0.10, 3, 0], [2, 0, 0.20, 4, 0],
    [3, 0, 0.15, 5, 0], [4, 1, 0.40, 3, 1],
    [5, 1, 0.55, 2, 1], [6, 2, 0.65, 2, 1],
    [7, 2, 0.80, 1, 1], [8, 3, 0.90, 1, 1],
    [2, 0, 0.05, 6, 0], [3, 1, 0.35, 4, 0],
    [5, 0, 0.30, 5, 1], [6, 1, 0.50, 3, 1],
    [9, 3, 0.95, 0, 1], [1, 0, 0.05, 7, 0],
    [4, 1, 0.45, 3, 1], [7, 1, 0.70, 2, 1]
].map(r => ({
    features: r.slice(0, 4).map(Number),
    label: r[4]
}));

const train = dataset.slice(0, 12);
const test = dataset.slice(12);

const events = new ModelEventBus();
events.on("prediction", result => {
    console.log(
        `prediction probability=${result.probability.toFixed(3)} ` +
        `class=${result.prediction}`
    );
});

const model = new XGBoostPolicyModel({
    maxDepth: 2,
    learningRate: 0.12,
    nEstimators: 60,
    lambda: 2,
    alpha: 0.05,
    gamma: 0.02,
    minChildWeight: 1,
    subsample: 0.9
});

model.fit(train);

const results = test.map(row => {
    const probability = model.predictProbability(row.features);
    const prediction = model.predict(row.features);
    events.emit("prediction", { probability, prediction });
    return { actual: row.label, prediction };
});

const accuracy =
    results.filter(r => r.actual === r.prediction).length / results.length;

const metadata = {
    model: "second-order gradient boosting",
    estimators: model.nEstimators,
    maxDepth: model.maxDepth,
    learningRate: model.learningRate,
    regularization: {
        lambda: model.lambda,
        alpha: model.alpha,
        gamma: model.gamma
    }
};

fs.writeFileSync(
    "xgboost-policy-model.json",
    JSON.stringify(metadata, null, 2),
    "utf8"
);

console.log(`test log loss=${logLoss(test, model).toFixed(4)}`);
console.log(`test accuracy=${(accuracy * 100).toFixed(2)}%`);
