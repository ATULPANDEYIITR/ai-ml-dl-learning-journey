"use strict";

/*
 * LightGBM-style histogram boosting plus CatBoost-style ordered
 * categorical statistics.
 *
 * Node.js 18+.
 *
 * The implementation intentionally exposes the algorithms rather than
 * depending on external machine-learning packages.
 */

const DATA = [
  { age: 23, income: 31000, tenure: 4, region: "north", plan: "basic", churn: 1 },
  { age: 25, income: 42000, tenure: 8, region: "south", plan: "basic", churn: 1 },
  { age: 29, income: 58000, tenure: 18, region: "west", plan: "pro", churn: 0 },
  { age: 31, income: 61000, tenure: 24, region: "north", plan: "pro", churn: 0 },
  { age: 35, income: 72000, tenure: 30, region: "east", plan: "pro", churn: 0 },
  { age: 41, income: 83000, tenure: 44, region: "west", plan: "enterprise", churn: 0 },
  { age: 48, income: 91000, tenure: 52, region: "north", plan: "enterprise", churn: 0 },
  { age: 52, income: 39000, tenure: 9, region: "south", plan: "basic", churn: 1 },
  { age: 27, income: 36000, tenure: 6, region: "east", plan: "basic", churn: 1 },
  { age: 33, income: 69000, tenure: 27, region: "south", plan: "pro", churn: 0 },
  { age: 38, income: 51000, tenure: 15, region: "east", plan: "basic", churn: 1 },
  { age: 45, income: 87000, tenure: 47, region: "west", plan: "enterprise", churn: 0 },
  { age: 22, income: 28000, tenure: 3, region: "north", plan: "basic", churn: 1 },
  { age: 57, income: 99000, tenure: 72, region: "north", plan: "enterprise", churn: 0 },
  { age: 30, income: 47000, tenure: 11, region: "west", plan: "pro", churn: 0 },
  { age: 36, income: 45000, tenure: 10, region: "south", plan: "basic", churn: 1 },
  { age: 43, income: 78000, tenure: 36, region: "east", plan: "pro", churn: 0 },
  { age: 50, income: 54000, tenure: 21, region: "south", plan: "pro", churn: 1 },
  { age: 28, income: 62000, tenure: 22, region: "north", plan: "pro", churn: 0 },
  { age: 39, income: 34000, tenure: 7, region: "east", plan: "basic", churn: 1 }
];

function sigmoid(x) {
  const bounded = Math.max(-35, Math.min(35, x));
  return 1 / (1 + Math.exp(-bounded));
}

function logLoss(y, probabilities) {
  const epsilon = 1e-12;
  let loss = 0;

  for (let i = 0; i < y.length; i++) {
    const p = Math.max(epsilon, Math.min(1 - epsilon, probabilities[i]));
    loss -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
  }

  return loss / y.length;
}

function deterministicShuffle(array, seed = 17) {
  const result = [...array];
  let state = seed >>> 0;

  function random() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  }

  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
}

class OrderedTargetEncoder {
  constructor(columns, priorWeight = 5, seed = 17) {
    this.columns = columns;
    this.priorWeight = priorWeight;
    this.seed = seed;
    this.globalMean = 0;
    this.maps = new Map();
  }

  fitTransform(rows, targets) {
    if (rows.length !== targets.length || rows.length === 0) {
      throw new Error("Rows and targets must have equal non-zero length.");
    }

    this.globalMean =
      targets.reduce((sum, value) => sum + value, 0) / targets.length;

    this.maps = new Map(this.columns.map(column => [column, new Map()]));
    const order = deterministicShuffle(
      [...Array(rows.length).keys()],
      this.seed
    );

    const encoded = rows.map(row => [
      row.age,
      row.income,
      row.tenure
    ]);

    for (const rowIndex of order) {
      for (const column of this.columns) {
        const category = rows[rowIndex][column];
        const map = this.maps.get(column);
        const state = map.get(category) || { sum: 0, count: 0 };

        const statistic =
          (state.sum + this.priorWeight * this.globalMean) /
          (state.count + this.priorWeight);

        encoded[rowIndex].push(statistic);
      }

      for (const column of this.columns) {
        const category = rows[rowIndex][column];
        const map = this.maps.get(column);
        const state = map.get(category) || { sum: 0, count: 0 };

        map.set(category, {
          sum: state.sum + targets[rowIndex],
          count: state.count + 1
        });
      }
    }

    return encoded;
  }

  transform(rows) {
    return rows.map(row => {
      const encoded = [row.age, row.income, row.tenure];

      for (const column of this.columns) {
        const category = row[column];
        const map = this.maps.get(column);
        const state = map.get(category) || { sum: 0, count: 0 };

        encoded.push(
          (state.sum + this.priorWeight * this.globalMean) /
          (state.count + this.priorWeight)
        );
      }

      return encoded;
    });
  }
}

class HistogramTree {
  constructor(maxDepth = 3, maxBins = 8, minLeaf = 2) {
    this.maxDepth = maxDepth;
    this.maxBins = maxBins;
    this.minLeaf = minLeaf;
    this.edges = [];
    this.root = null;
  }

  buildEdges(X) {
    const featureCount = X[0].length;
    this.edges = [];

    for (let feature = 0; feature < featureCount; feature++) {
      const values = [...new Set(X.map(row => row[feature]))].sort((a, b) => a - b);

      if (values.length <= this.maxBins) {
        this.edges.push(values);
        continue;
      }

      const edges = [];
      for (let i = 1; i < this.maxBins; i++) {
        const position = Math.min(
          values.length - 1,
          Math.max(0, Math.floor(i * values.length / this.maxBins))
        );
        edges.push(values[position]);
      }

      this.edges.push([...new Set(edges)].sort((a, b) => a - b));
    }
  }

  bin(feature, value) {
    const edges = this.edges[feature];

    for (let i = 0; i < edges.length; i++) {
      if (value <= edges[i]) return i;
    }

    return edges.length;
  }

  gain(gradients, hessians, left, right) {
    if (left.length < this.minLeaf || right.length < this.minLeaf) {
      return -Infinity;
    }

    const score = indices => {
      let g = 0;
      let h = 0;

      for (const index of indices) {
        g += gradients[index];
        h += hessians[index];
      }

      return g * g / (h + 1);
    };

    return 0.5 * (score(left) + score(right));
  }

  fit(X, gradients, hessians) {
    if (!X.length) throw new Error("Cannot fit an empty matrix.");

    this.buildEdges(X);

    const grow = (indices, depth) => {
      let gradientSum = 0;
      let hessianSum = 0;

      for (const index of indices) {
        gradientSum += gradients[index];
        hessianSum += hessians[index];
      }

      const node = {
        value: -gradientSum / (hessianSum + 1),
        feature: null,
        threshold: null,
        left: null,
        right: null
      };

      if (depth >= this.maxDepth || indices.length < this.minLeaf * 2) {
        return node;
      }

      let bestGain = 0;
      let bestFeature = null;
      let bestThreshold = null;
      let bestLeft = null;
      let bestRight = null;

      for (let feature = 0; feature < X[0].length; feature++) {
        const buckets = new Map();

        for (const index of indices) {
          const bucket = this.bin(feature, X[index][feature]);

          if (!buckets.has(bucket)) buckets.set(bucket, []);
          buckets.get(bucket).push(index);
        }

        const orderedBuckets = [...buckets.keys()].sort((a, b) => a - b);
        if (orderedBuckets.length < 2) continue;

        const left = [];

        for (let position = 0; position < orderedBuckets.length - 1; position++) {
          left.push(...buckets.get(orderedBuckets[position]));

          const right = [];

          for (
            let rightPosition = position + 1;
            rightPosition < orderedBuckets.length;
            rightPosition++
          ) {
            right.push(...buckets.get(orderedBuckets[rightPosition]));
          }

          const currentGain = this.gain(
            gradients,
            hessians,
            left,
            right
          );

          if (currentGain > bestGain) {
            bestGain = currentGain;
            bestFeature = feature;
            bestThreshold = this.edges[feature][orderedBuckets[position]];
            bestLeft = [...left];
            bestRight = right;
          }
        }
      }

      if (bestFeature === null) return node;

      node.feature = bestFeature;
      node.threshold = bestThreshold;
      node.left = grow(bestLeft, depth + 1);
      node.right = grow(bestRight, depth + 1);

      return node;
    };

    this.root = grow([...Array(X.length).keys()], 0);
  }

  predict(row) {
    if (!this.root) throw new Error("Tree has not been fitted.");

    let node = this.root;

    while (node.feature !== null) {
      node = row[node.feature] <= node.threshold
        ? node.left
        : node.right;
    }

    return node.value;
  }
}

class HistogramBoostingClassifier {
  constructor({
    estimators = 30,
    learningRate = 0.08,
    maxDepth = 3,
    maxBins = 8
  } = {}) {
    if (estimators <= 0 || learningRate <= 0) {
      throw new Error("estimators and learningRate must be positive.");
    }

    this.estimators = estimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
    this.maxBins = maxBins;
    this.baseScore = 0;
    this.trees = [];
  }

  fit(X, y) {
    if (X.length === 0 || X.length !== y.length) {
      throw new Error("X and y must contain equal non-zero lengths.");
    }

    if (y.some(value => value !== 0 && value !== 1)) {
      throw new Error("Binary targets must be 0 or 1.");
    }

    const rate = Math.min(
      1 - 1e-6,
      Math.max(1e-6, y.reduce((a, b) => a + b, 0) / y.length)
    );

    this.baseScore = Math.log(rate / (1 - rate));
    this.trees = [];

    const raw = Array(X.length).fill(this.baseScore);

    for (let iteration = 0; iteration < this.estimators; iteration++) {
      const gradients = [];
      const hessians = [];

      for (let i = 0; i < y.length; i++) {
        const probability = sigmoid(raw[i]);
        gradients.push(probability - y[i]);
        hessians.push(Math.max(1e-6, probability * (1 - probability)));
      }

      const tree = new HistogramTree(
        this.maxDepth,
        this.maxBins,
        2
      );

      tree.fit(X, gradients, hessians);
      this.trees.push(tree);

      for (let i = 0; i < X.length; i++) {
        raw[i] += this.learningRate * tree.predict(X[i]);
      }
    }
  }

  predictProba(X) {
    return X.map(row => {
      let raw = this.baseScore;

      for (const tree of this.trees) {
        raw += this.learningRate * tree.predict(row);
      }

      return sigmoid(raw);
    });
  }

  predict(X, threshold = 0.5) {
    if (!(threshold > 0 && threshold < 1)) {
      throw new Error("threshold must be between 0 and 1.");
    }

    return this.predictProba(X).map(value => Number(value >= threshold));
  }
}

function main() {
  const split = 15;
  const trainRows = DATA.slice(0, split);
  const testRows = DATA.slice(split);

  const yTrain = trainRows.map(row => row.churn);
  const yTest = testRows.map(row => row.churn);

  const encoder = new OrderedTargetEncoder(["region", "plan"], 5, 23);
  const XTrain = encoder.fitTransform(trainRows, yTrain);
  const XTest = encoder.transform(testRows);

  const model = new HistogramBoostingClassifier({
    estimators: 35,
    learningRate: 0.06,
    maxDepth: 3,
    maxBins: 6
  });

  model.fit(XTrain, yTrain);

  const probabilities = model.predictProba(XTest);
  const predictions = model.predict(XTest);

  console.log("CatBoost-style categorical encoding + histogram boosting");
  console.log("----------------------------------------------------------");
  console.log(`Training rows: ${trainRows.length}`);
  console.log(`Testing rows : ${testRows.length}`);
  console.log(`Feature count: ${XTrain[0].length}`);
  console.log(`Log loss     : ${logLoss(yTest, probabilities).toFixed(4)}`);
  console.log("Predictions  :", predictions);
  console.log(
    "Probabilities:",
    probabilities.map(value => Number(value.toFixed(3)))
  );

  console.log("\nOperational safeguards");
  console.log("----------------------");
  console.log("* Unknown categories fall back to the global target prior.");
  console.log("* Ordered statistics avoid directly using a row's own target.");
  console.log("* Histogram binning reduces repeated continuous split evaluation.");
  console.log("* Deep trees and many boosting rounds can overfit.");
  console.log("* Production systems require validation, early stopping,");
  console.log("  reproducible preprocessing, monitoring, and model versioning.");
}

main();
