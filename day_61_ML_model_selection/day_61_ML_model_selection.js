"use strict";

/*
 * ML Model Selection
 * Cross-validation, K-fold, stratified CV, and repeated CV.
 *
 * Node.js standard library only.
 *
 * This implementation uses an event-driven evaluation pipeline:
 * 1. Generate a binary classification dataset.
 * 2. Separate training and holdout data.
 * 3. Build stratified folds.
 * 4. Evaluate multiple models asynchronously.
 * 5. Repeat validation with different fold assignments.
 * 6. Select a model without touching the holdout set.
 * 7. Evaluate the selected model once on the holdout set.
 */

const { EventEmitter } = require("node:events");

class ValidationError extends Error {}

class Dataset {
  constructor(features, labels) {
    this.features = features;
    this.labels = labels;
    this.validate();
  }

  validate() {
    if (!Array.isArray(this.features) || !Array.isArray(this.labels)) {
      throw new ValidationError("Features and labels must be arrays.");
    }

    if (this.features.length === 0 || this.features.length !== this.labels.length) {
      throw new ValidationError("Features and labels must have equal non-zero lengths.");
    }

    const width = this.features[0].length;
    if (width === 0) {
      throw new ValidationError("Features cannot be empty.");
    }

    for (const row of this.features) {
      if (!Array.isArray(row) || row.length !== width) {
        throw new ValidationError("Feature dimensions are inconsistent.");
      }

      if (row.some((value) => !Number.isFinite(value))) {
        throw new ValidationError("Features must contain finite numbers.");
      }
    }

    for (const label of this.labels) {
      if (label !== 0 && label !== 1) {
        throw new ValidationError("This implementation requires binary labels.");
      }
    }
  }

  subset(indices) {
    return new Dataset(
      indices.map((i) => this.features[i]),
      indices.map((i) => this.labels[i])
    );
  }
}

function seededRandom(seed) {
  let state = seed >>> 0;

  return function random() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function shuffle(array, random) {
  const result = [...array];

  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
}

function accuracy(actual, predicted) {
  return actual.reduce((sum, value, i) => sum + Number(value === predicted[i]), 0) /
    actual.length;
}

function precision(actual, predicted) {
  let tp = 0;
  let fp = 0;

  for (let i = 0; i < actual.length; i++) {
    if (predicted[i] === 1 && actual[i] === 1) tp++;
    if (predicted[i] === 1 && actual[i] === 0) fp++;
  }

  return tp + fp === 0 ? 0 : tp / (tp + fp);
}

function recall(actual, predicted) {
  let tp = 0;
  let fn = 0;

  for (let i = 0; i < actual.length; i++) {
    if (predicted[i] === 1 && actual[i] === 1) tp++;
    if (predicted[i] === 0 && actual[i] === 1) fn++;
  }

  return tp + fn === 0 ? 0 : tp / (tp + fn);
}

function f1(actual, predicted) {
  const p = precision(actual, predicted);
  const r = recall(actual, predicted);
  return p + r === 0 ? 0 : (2 * p * r) / (p + r);
}

function logLoss(actual, probabilities) {
  if (actual.length !== probabilities.length || actual.length === 0) {
    throw new ValidationError("Invalid log-loss inputs.");
  }

  let total = 0;

  for (let i = 0; i < actual.length; i++) {
    const p = Math.min(Math.max(probabilities[i], 1e-15), 1 - 1e-15);
    total -= actual[i] * Math.log(p) + (1 - actual[i]) * Math.log(1 - p);
  }

  return total / actual.length;
}

class LogisticRegression {
  constructor({ learningRate = 0.08, epochs = 500, l2 = 0.02 } = {}) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.l2 = l2;
    this.weights = [];
    this.bias = 0;
  }

  sigmoid(value) {
    const bounded = Math.max(-60, Math.min(60, value));
    return 1 / (1 + Math.exp(-bounded));
  }

  fit(X, y) {
    const width = X[0].length;
    this.weights = Array(width).fill(0);
    this.bias = 0;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const gradient = Array(width).fill(0);
      let biasGradient = 0;

      for (let i = 0; i < X.length; i++) {
        const linear = this.bias +
          X[i].reduce((sum, value, j) => sum + value * this.weights[j], 0);

        const error = this.sigmoid(linear) - y[i];

        for (let j = 0; j < width; j++) {
          gradient[j] += error * X[i][j];
        }

        biasGradient += error;
      }

      for (let j = 0; j < width; j++) {
        gradient[j] = gradient[j] / X.length + this.l2 * this.weights[j];
        this.weights[j] -= this.learningRate * gradient[j];
      }

      this.bias -= this.learningRate * biasGradient / X.length;
    }
  }

  predictProba(X) {
    return X.map((row) => {
      const linear = this.bias +
        row.reduce((sum, value, j) => sum + value * this.weights[j], 0);
      return this.sigmoid(linear);
    });
  }

  predict(X) {
    return this.predictProba(X).map((value) => Number(value >= 0.5));
  }
}

class DecisionTree {
  constructor({ maxDepth = 4, minSamplesSplit = 6 } = {}) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.root = null;
  }

  gini(labels) {
    if (labels.length === 0) return 0;

    const positive = labels.reduce((sum, value) => sum + value, 0) / labels.length;
    return 2 * positive * (1 - positive);
  }

  majority(labels) {
    const positive = labels.reduce((sum, value) => sum + value, 0);
    return Number(positive * 2 >= labels.length);
  }

  bestSplit(X, y, indices) {
    const parent = this.gini(indices.map((i) => y[i]));
    let best = null;

    for (let feature = 0; feature < X[0].length; feature++) {
      const values = [...new Set(indices.map((i) => X[i][feature]))].sort((a, b) => a - b);

      for (let j = 0; j < values.length - 1; j++) {
        const threshold = (values[j] + values[j + 1]) / 2;
        const left = indices.filter((i) => X[i][feature] <= threshold);
        const right = indices.filter((i) => X[i][feature] > threshold);

        if (!left.length || !right.length) continue;

        const weighted =
          (left.length * this.gini(left.map((i) => y[i])) +
            right.length * this.gini(right.map((i) => y[i]))) /
          indices.length;

        const gain = parent - weighted;

        if (best === null || gain > best.gain) {
          best = { feature, threshold, gain };
        }
      }
    }

    return best;
  }

  build(X, y, indices, depth) {
    const labels = indices.map((i) => y[i]);
    const prediction = this.majority(labels);

    if (
      depth >= this.maxDepth ||
      indices.length < this.minSamplesSplit ||
      new Set(labels).size === 1
    ) {
      return { type: "leaf", prediction };
    }

    const split = this.bestSplit(X, y, indices);

    if (!split || split.gain <= 0) {
      return { type: "leaf", prediction };
    }

    const left = indices.filter((i) => X[i][split.feature] <= split.threshold);
    const right = indices.filter((i) => X[i][split.feature] > split.threshold);

    return {
      type: "node",
      feature: split.feature,
      threshold: split.threshold,
      left: this.build(X, y, left, depth + 1),
      right: this.build(X, y, right, depth + 1)
    };
  }

  fit(X, y) {
    this.root = this.build(X, y, [...X.keys()], 0);
  }

  predictOne(row, node) {
    if (node.type === "leaf") return node.prediction;

    return row[node.feature] <= node.threshold
      ? this.predictOne(row, node.left)
      : this.predictOne(row, node.right);
  }

  predict(X) {
    if (!this.root) throw new Error("Tree has not been fitted.");
    return X.map((row) => this.predictOne(row, this.root));
  }

  predictProba(X) {
    return this.predict(X).map((value) => value ? 0.85 : 0.15);
  }
}

function stratifiedKFold(labels, k, seed) {
  if (!Number.isInteger(k) || k < 2 || k > labels.length) {
    throw new ValidationError("Invalid number of folds.");
  }

  const random = seededRandom(seed);
  const classBuckets = new Map();

  labels.forEach((label, index) => {
    if (!classBuckets.has(label)) classBuckets.set(label, []);
    classBuckets.get(label).push(index);
  });

  const folds = Array.from({ length: k }, () => []);

  for (const indices of classBuckets.values()) {
    const shuffled = shuffle(indices, random);

    shuffled.forEach((index, position) => {
      folds[position % k].push(index);
    });
  }

  folds.forEach((fold) => {
    const reordered = shuffle(fold, random);
    fold.splice(0, fold.length, ...reordered);
  });

  return folds.map((validation) => {
    const validationSet = new Set(validation);
    const train = [];

    folds.forEach((fold) => {
      fold.forEach((index) => {
        if (!validationSet.has(index)) train.push(index);
      });
    });

    return { train, validation };
  });
}

function generateDataset(size = 180, seed = 11) {
  const random = seededRandom(seed);

  function normal() {
    const u = Math.max(random(), Number.EPSILON);
    const v = Math.max(random(), Number.EPSILON);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const X = [];
  const y = [];

  for (let i = 0; i < size; i++) {
    const x1 = normal();
    const x2 = normal();
    const x3 = normal();
    const x4 = normal();

    const score =
      1.6 * x1 -
      1.1 * x2 +
      0.7 * x3 +
      0.2 * x4 +
      0.8 * normal();

    X.push([x1, x2, x3, x4]);
    y.push(Number(score > 0.35));
  }

  return new Dataset(X, y);
}

function holdoutSplit(dataset, ratio = 0.2, seed = 42) {
  const random = seededRandom(seed);
  const indices = shuffle([...dataset.labels.keys()], random);
  const testSize = Math.max(1, Math.round(indices.length * ratio));

  return {
    train: dataset.subset(indices.slice(testSize)),
    test: dataset.subset(indices.slice(0, testSize))
  };
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values) {
  if (values.length < 2) return 0;

  const average = mean(values);
  const variance = mean(values.map((value) => (value - average) ** 2));
  return Math.sqrt(variance);
}

async function evaluateModel(modelFactory, dataset, folds) {
  const scores = [];
  const accuracies = [];
  const losses = [];

  for (const fold of folds) {
    const train = dataset.subset(fold.train);
    const validation = dataset.subset(fold.validation);

    const model = modelFactory();
    model.fit(train.features, train.labels);

    const predictions = model.predict(validation.features);
    const probabilities = model.predictProba(validation.features);

    scores.push(f1(validation.labels, predictions));
    accuracies.push(accuracy(validation.labels, predictions));
    losses.push(logLoss(validation.labels, probabilities));

    await new Promise((resolve) => setImmediate(resolve));
  }

  return {
    meanF1: mean(scores),
    stdF1: standardDeviation(scores),
    meanAccuracy: mean(accuracies),
    meanLogLoss: mean(losses),
    scores
  };
}

class ModelSelectionPipeline extends EventEmitter {
  constructor(dataset) {
    super();
    this.dataset = dataset;
    this.models = new Map([
      [
        "logistic-regression",
        () => new LogisticRegression({
          learningRate: 0.08,
          epochs: 600,
          l2: 0.02
        })
      ],
      [
        "decision-tree",
        () => new DecisionTree({
          maxDepth: 4,
          minSamplesSplit: 6
        })
      ]
    ]);
  }

  async run() {
    const split = holdoutSplit(this.dataset, 0.2, 42);

    this.emit("phase", "holdout separated");

    const results = new Map();
    const folds = stratifiedKFold(split.train.labels, 5, 42);

    for (const [name, factory] of this.models) {
      const result = await evaluateModel(factory, split.train, folds);
      results.set(name, result);
      this.emit("model-evaluated", { name, result });
    }

    let selectedName = null;
    let selectedResult = null;

    for (const [name, result] of results) {
      if (
        selectedResult === null ||
        result.meanF1 > selectedResult.meanF1 ||
        (
          result.meanF1 === selectedResult.meanF1 &&
          result.stdF1 < selectedResult.stdF1
        )
      ) {
        selectedName = name;
        selectedResult = result;
      }
    }

    const repeatedScores = [];

    for (let repetition = 0; repetition < 3; repetition++) {
      const repeatedFolds = stratifiedKFold(
        split.train.labels,
        5,
        42 + repetition
      );

      const repeated = await evaluateModel(
        this.models.get(selectedName),
        split.train,
        repeatedFolds
      );

      repeatedScores.push(repeated.meanF1);
    }

    const finalModel = this.models.get(selectedName)();
    finalModel.fit(split.train.features, split.train.labels);

    const predictions = finalModel.predict(split.test.features);
    const probabilities = finalModel.predictProba(split.test.features);

    const holdout = {
      accuracy: accuracy(split.test.labels, predictions),
      precision: precision(split.test.labels, predictions),
      recall: recall(split.test.labels, predictions),
      f1: f1(split.test.labels, predictions),
      logLoss: logLoss(split.test.labels, probabilities)
    };

    this.emit("complete", {
      selectedName,
      selectedResult,
      repeatedScores,
      holdout
    });

    return {
      selectedName,
      selectedResult,
      repeatedScores,
      holdout
    };
  }
}

async function main() {
  const dataset = generateDataset();

  const pipeline = new ModelSelectionPipeline(dataset);

  pipeline.on("phase", (message) => {
    console.log(`[phase] ${message}`);
  });

  pipeline.on("model-evaluated", ({ name, result }) => {
    console.log(`\n${name}`);
    console.log(`  Fold F1: ${result.scores.map((x) => x.toFixed(4)).join(", ")}`);
    console.log(`  Mean F1: ${result.meanF1.toFixed(4)}`);
    console.log(`  F1 std: ${result.stdF1.toFixed(4)}`);
    console.log(`  Accuracy: ${result.meanAccuracy.toFixed(4)}`);
    console.log(`  Log loss: ${result.meanLogLoss.toFixed(4)}`);
  });

  const result = await pipeline.run();

  console.log("\nREPEATED STRATIFIED CV");
  console.log(
    `F1 by repetition: ${result.repeatedScores.map((x) => x.toFixed(4)).join(", ")}`
  );

  console.log("\nFINAL HOLDOUT");
  console.log(`Selected model: ${result.selectedName}`);
  console.log(`Accuracy: ${result.holdout.accuracy.toFixed(4)}`);
  console.log(`Precision: ${result.holdout.precision.toFixed(4)}`);
  console.log(`Recall: ${result.holdout.recall.toFixed(4)}`);
  console.log(`F1: ${result.holdout.f1.toFixed(4)}`);
  console.log(`Log loss: ${result.holdout.logLoss.toFixed(4)}`);
}

main().catch((error) => {
  console.error(`Evaluation failed: ${error.message}`);
  process.exitCode = 1;
});
