"use strict";

/*
 * Gradient Boosting in JavaScript.
 *
 * This implementation focuses on a JavaScript-specific event-driven view of
 * boosting. The model emits progress events while shallow regression trees
 * are fitted sequentially to residuals.
 *
 * No external packages are required.
 */

const EventEmitter = require("node:events");

function mean(values) {
  if (values.length === 0) {
    throw new Error("mean requires at least one value");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function mse(actual, predicted) {
  if (actual.length !== predicted.length || actual.length === 0) {
    throw new Error("MSE requires equally sized non-empty arrays");
  }

  return actual.reduce(
    (sum, value, index) => sum + (value - predicted[index]) ** 2,
    0
  ) / actual.length;
}

function validateDataset(features, targets) {
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error("features must be a non-empty array");
  }

  if (features.length !== targets.length) {
    throw new Error("features and targets must have equal lengths");
  }

  const width = features[0].length;

  if (width === 0) {
    throw new Error("each feature row needs at least one value");
  }

  for (const row of features) {
    if (row.length !== width) {
      throw new Error("feature rows must have equal widths");
    }

    if (row.some((value) => !Number.isFinite(value))) {
      throw new Error("features must contain finite numbers");
    }
  }

  if (targets.some((value) => !Number.isFinite(value))) {
    throw new Error("targets must contain finite numbers");
  }
}

function squaredError(values) {
  if (values.length === 0) {
    return 0;
  }

  const center = mean(values);
  return values.reduce(
    (sum, value) => sum + (value - center) ** 2,
    0
  );
}

function seededRandom(seed) {
  let state = seed >>> 0;

  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function sampleWithoutReplacement(items, count, random) {
  const copy = [...items];

  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }

  return copy.slice(0, count);
}

class TreeNode {
  constructor(prediction, feature = null, threshold = null) {
    this.prediction = prediction;
    this.feature = feature;
    this.threshold = threshold;
    this.left = null;
    this.right = null;
  }

  get isLeaf() {
    return this.feature === null;
  }
}

class RegressionTree {
  constructor({ maxDepth = 2, minSamplesLeaf = 2 } = {}) {
    if (maxDepth < 0) {
      throw new Error("maxDepth cannot be negative");
    }

    if (minSamplesLeaf < 1) {
      throw new Error("minSamplesLeaf must be positive");
    }

    this.maxDepth = maxDepth;
    this.minSamplesLeaf = minSamplesLeaf;
    this.root = null;
    this.featureImportance = [];
  }

  bestSplit(features, residuals, indices, featureIndices) {
    if (indices.length < this.minSamplesLeaf * 2) {
      return null;
    }

    const parentError = squaredError(
      indices.map((index) => residuals[index])
    );

    let best = null;

    for (const feature of featureIndices) {
      const ordered = [...indices].sort(
        (a, b) => features[a][feature] - features[b][feature]
      );

      for (
        let position = this.minSamplesLeaf;
        position <= ordered.length - this.minSamplesLeaf;
        position++
      ) {
        const left = ordered.slice(0, position);
        const right = ordered.slice(position);

        const leftValue = features[left[left.length - 1]][feature];
        const rightValue = features[right[0]][feature];

        if (leftValue === rightValue) {
          continue;
        }

        const threshold = (leftValue + rightValue) / 2;

        const leftError = squaredError(
          left.map((index) => residuals[index])
        );

        const rightError = squaredError(
          right.map((index) => residuals[index])
        );

        const improvement =
          parentError - leftError - rightError;

        if (
          improvement > 1e-12 &&
          (best === null || improvement > best.improvement)
        ) {
          best = { feature, threshold, improvement };
        }
      }
    }

    return best;
  }

  fit(features, residuals) {
    validateDataset(features, residuals);

    this.featureImportance = Array(features[0].length).fill(0);

    const featureIndices = features[0].map((_, index) => index);
    const allIndices = features.map((_, index) => index);

    const build = (indices, depth) => {
      const prediction = mean(
        indices.map((index) => residuals[index])
      );

      if (
        depth >= this.maxDepth ||
        indices.length < this.minSamplesLeaf * 2
      ) {
        return new TreeNode(prediction);
      }

      const split = this.bestSplit(
        features,
        residuals,
        indices,
        featureIndices
      );

      if (split === null) {
        return new TreeNode(prediction);
      }

      const leftIndices = indices.filter(
        (index) => features[index][split.feature] <= split.threshold
      );

      const rightIndices = indices.filter(
        (index) => features[index][split.feature] > split.threshold
      );

      if (
        leftIndices.length < this.minSamplesLeaf ||
        rightIndices.length < this.minSamplesLeaf
      ) {
        return new TreeNode(prediction);
      }

      this.featureImportance[split.feature] += split.improvement;

      const node = new TreeNode(
        prediction,
        split.feature,
        split.threshold
      );

      node.left = build(leftIndices, depth + 1);
      node.right = build(rightIndices, depth + 1);

      return node;
    };

    this.root = build(allIndices, 0);
    return this;
  }

  predictOne(row) {
    if (this.root === null) {
      throw new Error("tree has not been fitted");
    }

    let node = this.root;

    while (!node.isLeaf) {
      node =
        row[node.feature] <= node.threshold
          ? node.left
          : node.right;
    }

    return node.prediction;
  }

  predict(features) {
    return features.map((row) => this.predictOne(row));
  }
}

class GradientBoostingRegressor extends EventEmitter {
  constructor({
    nEstimators = 100,
    learningRate = 0.05,
    maxDepth = 2,
    minSamplesLeaf = 2,
    subsample = 1,
    randomSeed = 42,
  } = {}) {
    super();

    if (nEstimators < 1) {
      throw new Error("nEstimators must be positive");
    }

    if (learningRate <= 0) {
      throw new Error("learningRate must be positive");
    }

    if (subsample <= 0 || subsample > 1) {
      throw new Error("subsample must be in (0, 1]");
    }

    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
    this.minSamplesLeaf = minSamplesLeaf;
    this.subsample = subsample;
    this.random = seededRandom(randomSeed);
    this.initialPrediction = 0;
    this.trees = [];
    this.lossHistory = [];
  }

  fit(features, targets) {
    validateDataset(features, targets);

    this.initialPrediction = mean(targets);

    const prediction = Array(targets.length).fill(
      this.initialPrediction
    );

    this.trees = [];
    this.lossHistory = [];

    for (let iteration = 0; iteration < this.nEstimators; iteration++) {
      /*
       * Gradient boosting with squared loss has a particularly simple
       * negative gradient: target - current prediction.
       */
      const residuals = targets.map(
        (target, index) => target - prediction[index]
      );

      const allIndices = features.map((_, index) => index);
      const sampleCount = Math.max(
        this.minSamplesLeaf * 2,
        Math.floor(features.length * this.subsample)
      );

      const sampledIndices = sampleWithoutReplacement(
        allIndices,
        Math.min(sampleCount, features.length),
        this.random
      );

      const treeFeatures = sampledIndices.map(
        (index) => features[index]
      );

      const treeResiduals = sampledIndices.map(
        (index) => residuals[index]
      );

      const tree = new RegressionTree({
        maxDepth: this.maxDepth,
        minSamplesLeaf: this.minSamplesLeaf,
      });

      tree.fit(treeFeatures, treeResiduals);

      const correction = tree.predict(features);

      for (let index = 0; index < prediction.length; index++) {
        prediction[index] +=
          this.learningRate * correction[index];
      }

      this.trees.push(tree);

      const loss = mse(targets, prediction);
      this.lossHistory.push(loss);

      this.emit("iteration", {
        iteration: iteration + 1,
        loss,
        treeCount: this.trees.length,
      });
    }

    this.emit("complete", {
      treeCount: this.trees.length,
      finalLoss: this.lossHistory.at(-1),
    });

    return this;
  }

  predict(features) {
    if (this.trees.length === 0) {
      throw new Error("model has not been fitted");
    }

    const prediction = Array(features.length).fill(
      this.initialPrediction
    );

    for (const tree of this.trees) {
      const correction = tree.predict(features);

      for (let index = 0; index < prediction.length; index++) {
        prediction[index] +=
          this.learningRate * correction[index];
      }
    }

    return prediction;
  }

  stagedPredict(features) {
    if (this.trees.length === 0) {
      throw new Error("model has not been fitted");
    }

    const prediction = Array(features.length).fill(
      this.initialPrediction
    );

    return this.trees.map((tree) => {
      const correction = tree.predict(features);

      for (let index = 0; index < prediction.length; index++) {
        prediction[index] +=
          this.learningRate * correction[index];
      }

      return [...prediction];
    });
  }
}

function createEnergyDataset() {
  return {
    features: [
      [72, 2, 18, 5],
      [110, 4, 12, 4],
      [150, 5, 8, 3],
      [190, 6, 5, 2],
      [230, 7, 3, 1],
      [260, 8, 2, 1],
      [90, 3, 16, 5],
      [130, 4, 10, 4],
      [170, 5, 7, 3],
      [210, 6, 4, 2],
      [250, 8, 2, 1],
      [280, 9, 1, 1],
      [60, 2, 25, 7],
      [100, 3, 15, 5],
      [145, 4, 9, 4],
      [185, 6, 5, 2],
      [225, 7, 3, 2],
      [275, 9, 1, 1],
    ],
    targets: [
      410,
      650,
      830,
      1030,
      1230,
      1370,
      500,
      720,
      920,
      1100,
      1310,
      1450,
      360,
      570,
      790,
      1010,
      1190,
      1430,
    ],
  };
}

async function run() {
  const { features, targets } = createEnergyDataset();

  const model = new GradientBoostingRegressor({
    nEstimators: 80,
    learningRate: 0.04,
    maxDepth: 2,
    minSamplesLeaf: 2,
    subsample: 0.8,
    randomSeed: 13,
  });

  model.on("iteration", ({ iteration, loss }) => {
    if (iteration === 1 || iteration % 10 === 0) {
      console.log(
        `boosting iteration=${iteration} training_mse=${loss.toFixed(3)}`
      );
    }
  });

  model.on("complete", ({ treeCount, finalLoss }) => {
    console.log(
      `completed trees=${treeCount} final_training_mse=${finalLoss.toFixed(3)}`
    );
  });

  model.fit(features, targets);

  const predictions = model.predict(features);

  console.log("\nEnergy-demand predictions");

  for (let index = 0; index < predictions.length; index++) {
    console.log(
      `actual=${targets[index].toFixed(1)} ` +
      `predicted=${predictions[index].toFixed(1)}`
    );
  }

  const staged = model.stagedPredict(features);

  console.log("\nContribution of the ensemble");

  for (const checkpoint of [1, 5, 20, 40, 80]) {
    const prediction = staged[checkpoint - 1];

    console.log(
      `trees=${checkpoint} ` +
      `mse=${mse(targets, prediction).toFixed(3)}`
    );
  }

  /*
   * The event interface is useful in a web service because training progress
   * can be forwarded to a dashboard, log stream, or monitoring system without
   * coupling the model to a particular UI.
   */
}

run().catch((error) => {
  console.error(`Training failed: ${error.message}`);
  process.exitCode = 1;
});
