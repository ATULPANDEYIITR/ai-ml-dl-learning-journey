"use strict";

/*
 * Decision tree splitting in JavaScript.
 *
 * This implementation focuses on the event-driven side of a model-building
 * workflow: candidate split evaluation, deterministic tree construction,
 * classification, and traceable split decisions.
 *
 * Run with Node.js:
 *   node decision-tree.js
 */

const assert = require("node:assert/strict");

function countClasses(labels) {
  const counts = new Map();
  for (const label of labels) {
    if (typeof label !== "string" || label.length === 0) {
      throw new TypeError("Class labels must be nonempty strings.");
    }
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return counts;
}

function entropy(labels) {
  if (labels.length === 0) return 0;
  const counts = countClasses(labels);
  let result = 0;

  for (const count of counts.values()) {
    const probability = count / labels.length;
    result -= probability * Math.log2(probability);
  }

  return result;
}

function gini(labels) {
  if (labels.length === 0) return 0;
  const counts = countClasses(labels);
  let squaredProbabilities = 0;

  for (const count of counts.values()) {
    const probability = count / labels.length;
    squaredProbabilities += probability * probability;
  }

  return 1 - squaredProbabilities;
}

function weightedImpurity(groups, impurityFunction) {
  const total = groups.reduce((sum, group) => sum + group.length, 0);
  if (total === 0) throw new RangeError("Cannot score an empty partition.");

  return groups.reduce(
    (sum, group) =>
      sum + (group.length / total) * impurityFunction(group),
    0
  );
}

function splitLabels(rows, labels, feature, threshold) {
  const left = [];
  const right = [];

  rows.forEach((row, index) => {
    (row[feature] <= threshold ? left : right).push(labels[index]);
  });

  return { left, right };
}

function scoreThreshold(rows, labels, feature, threshold, criterion) {
  const groups = splitLabels(rows, labels, feature, threshold);
  const impurity = criterion === "entropy" ? entropy : gini;

  if (!groups.left.length || !groups.right.length) {
    return { threshold, gain: -Infinity, ...groups };
  }

  const gain =
    impurity(labels) -
    weightedImpurity([groups.left, groups.right], impurity);

  return { threshold, gain, ...groups };
}

function candidateThresholds(values) {
  const sorted = [...new Set(values)].sort((a, b) => a - b);
  const thresholds = [];

  for (let index = 0; index < sorted.length - 1; index += 1) {
    // The midpoint separates distinct observed values. It avoids testing
    // thresholds that cannot change the partition.
    thresholds.push(sorted[index] + (sorted[index + 1] - sorted[index]) / 2);
  }

  return thresholds;
}

function validateTrainingData(rows, labels, features) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new TypeError("Training rows must be a nonempty array.");
  }
  if (!Array.isArray(labels) || labels.length !== rows.length) {
    throw new RangeError("Each training row must have one label.");
  }
  if (!Array.isArray(features) || features.length === 0) {
    throw new TypeError("At least one feature is required.");
  }
  if (new Set(features).size !== features.length) {
    throw new RangeError("Feature names must be unique.");
  }

  countClasses(labels);

  rows.forEach((row, index) => {
    if (row === null || typeof row !== "object" || Array.isArray(row)) {
      throw new TypeError(`Row ${index} must be an object.`);
    }

    for (const feature of features) {
      const value = row[feature];

      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new TypeError(
          `Feature ${feature} in row ${index} must be a finite number.`
        );
      }
    }
  });
}

class DecisionTree {
  constructor({
    criterion = "gini",
    maxDepth = 3,
    minSamplesLeaf = 2,
    minGain = 1e-12,
  } = {}) {
    if (!["gini", "entropy"].includes(criterion)) {
      throw new RangeError("criterion must be gini or entropy.");
    }
    if (!Number.isInteger(maxDepth) || maxDepth < 0) {
      throw new RangeError("maxDepth must be a nonnegative integer.");
    }
    if (!Number.isInteger(minSamplesLeaf) || minSamplesLeaf < 1) {
      throw new RangeError("minSamplesLeaf must be positive.");
    }

    this.criterion = criterion;
    this.maxDepth = maxDepth;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minGain = minGain;
    this.root = null;
    this.features = [];
    this.trace = [];
  }

  impurity(labels) {
    return this.criterion === "entropy" ? entropy(labels) : gini(labels);
  }

  fit(rows, labels, features) {
    validateTrainingData(rows, labels, features);
    this.features = [...features];
    this.trace = [];

    const build = (indices, depth) => {
      const localRows = indices.map((index) => rows[index]);
      const localLabels = indices.map((index) => labels[index]);
      const counts = countClasses(localLabels);

      // Sorting makes ties reproducible rather than dependent on insertion
      // order or incidental Map iteration behavior.
      const majorityCount = Math.max(...counts.values());
      const prediction = [...counts.entries()]
        .filter(([, count]) => count === majorityCount)
        .map(([label]) => label)
        .sort()[0];

      const node = {
        prediction,
        counts: Object.fromEntries(counts),
        samples: indices.length,
        impurity: this.impurity(localLabels),
        depth,
        feature: null,
        threshold: null,
        gain: 0,
        left: null,
        right: null,
      };

      if (depth >= this.maxDepth || node.impurity === 0) {
        return node;
      }

      let best = null;

      for (const feature of this.features) {
        const thresholds = candidateThresholds(
          localRows.map((row) => row[feature])
        );

        for (const threshold of thresholds) {
          const candidate = scoreThreshold(
            localRows,
            localLabels,
            feature,
            threshold,
            this.criterion
          );

          const leftIndices = [];
          const rightIndices = [];

          localRows.forEach((row, localIndex) => {
            (row[feature] <= threshold ? leftIndices : rightIndices).push(
              indices[localIndex]
            );
          });

          if (
            leftIndices.length < this.minSamplesLeaf ||
            rightIndices.length < this.minSamplesLeaf
          ) {
            continue;
          }

          candidate.feature = feature;
          candidate.leftIndices = leftIndices;
          candidate.rightIndices = rightIndices;

          if (
            best === null ||
            candidate.gain > best.gain + 1e-15
          ) {
            best = candidate;
          }
        }
      }

      if (best === null || best.gain <= this.minGain) {
        return node;
      }

      node.feature = best.feature;
      node.threshold = best.threshold;
      node.gain = best.gain;

      // The trace is an audit trail of decisions made during fitting.
      this.trace.push({
        depth,
        feature: best.feature,
        threshold: best.threshold,
        gain: best.gain,
        samples: indices.length,
      });

      node.left = build(best.leftIndices, depth + 1);
      node.right = build(best.rightIndices, depth + 1);
      return node;
    };

    this.root = build(rows.map((_, index) => index), 0);
    return this;
  }

  predictOne(row) {
    if (this.root === null) {
      throw new Error("Fit the tree before prediction.");
    }

    for (const feature of this.features) {
      if (typeof row[feature] !== "number" || !Number.isFinite(row[feature])) {
        throw new TypeError(`Prediction feature ${feature} must be finite.`);
      }
    }

    let node = this.root;

    while (node.feature !== null) {
      node =
        row[node.feature] <= node.threshold ? node.left : node.right;
    }

    return node.prediction;
  }

  predict(rows) {
    return rows.map((row) => this.predictOne(row));
  }

  toJSON() {
    if (this.root === null) {
      throw new Error("No fitted tree to serialize.");
    }

    // Serialization emits model data only. Never evaluate serialized strings
    // as JavaScript source.
    return JSON.stringify({
      criterion: this.criterion,
      features: this.features,
      root: this.root,
      trace: this.trace,
    }, null, 2);
  }
}

function buildReleaseDataset() {
  const rows = [];
  const labels = [];

  for (const coverage of [48, 61, 69, 76, 83, 91, 97]) {
    for (const failedTests of [0, 1, 3]) {
      for (const securityFindings of [0, 1]) {
        const row = { coverage, failedTests, securityFindings };
        const requiresReview =
          coverage < 75 || failedTests > 0 || securityFindings > 0;

        rows.push(row);
        labels.push(requiresReview ? "manual_review" : "eligible");
      }
    }
  }

  return { rows, labels };
}

function evaluate(actual, predicted) {
  if (actual.length === 0 || actual.length !== predicted.length) {
    throw new RangeError("Evaluation requires matching nonempty arrays.");
  }

  const matrix = new Map();

  actual.forEach((label, index) => {
    if (!matrix.has(label)) matrix.set(label, new Map());
    const row = matrix.get(label);
    const guess = predicted[index];
    row.set(guess, (row.get(guess) ?? 0) + 1);
  });

  const correct = actual.reduce(
    (sum, label, index) => sum + Number(label === predicted[index]),
    0
  );

  return {
    accuracy: correct / actual.length,
    confusionMatrix: Object.fromEntries(
      [...matrix.entries()].map(([label, counts]) => [
        label,
        Object.fromEntries(counts),
      ])
    ),
  };
}

function main() {
  const mixed = ["eligible", "eligible", "eligible", "manual_review"];
  assert.equal(entropy(["eligible", "eligible"]), 0);
  assert.equal(gini(["eligible", "eligible"]), 0);
  assert.ok(entropy(mixed) > 0);
  assert.ok(gini(mixed) > 0);

  const { rows, labels } = buildReleaseDataset();
  const tree = new DecisionTree({
    criterion: "entropy",
    maxDepth: 3,
    minSamplesLeaf: 2,
  }).fit(rows, labels, [
    "coverage",
    "failedTests",
    "securityFindings",
  ]);

  const predictions = tree.predict(rows);
  const report = evaluate(labels, predictions);

  console.log("Root node:", tree.root);
  console.log("Split trace:", tree.trace);
  console.log("Training diagnostic:", report);

  const releaseCandidate = {
    coverage: 89,
    failedTests: 0,
    securityFindings: 0,
  };

  console.log(
    "Release candidate classification:",
    tree.predictOne(releaseCandidate)
  );

  const serialized = tree.toJSON();
  assert.equal(JSON.parse(serialized).criterion, "entropy");
  console.log("Serialized model characters:", serialized.length);

  assert.throws(
    () => tree.predictOne({ coverage: Infinity, failedTests: 0, securityFindings: 0 }),
    /finite/
  );

  console.log("Decision-tree assertions passed.");
}

main();
