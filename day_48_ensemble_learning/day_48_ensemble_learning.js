"use strict";

/*
 * Ensemble Learning in JavaScript.
 *
 * Run with Node.js 18 or later:
 *     node ensemble_learning.js
 *
 * Demonstrates hard voting, probability averaging, bootstrap bagging,
 * AdaBoost, and validation-set blending with independent base models.
 */

const assert = require("node:assert/strict");

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-Math.min(value, 700));
    return 1 / (1 + z);
  }
  const z = Math.exp(Math.max(value, -700));
  return z / (1 + z);
}

function createRandom(seed) {
  // A seeded generator makes experiments reproducible across runs.
  let state = seed >>> 0;
  return function random() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeDataset(seed = 2026, count = 300) {
  const random = createRandom(seed);
  const gaussian = () => {
    const u1 = Math.max(random(), Number.EPSILON);
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  return Array.from({ length: count }, () => {
    const x = random() * 6 - 3;
    const y = random() * 6 - 3;
    const score = x * x + 0.7 * y - 1.5 + gaussian() * 0.9;

    return { features: [x, y], label: Number(score > 0) };
  });
}

function splitDataset(data, fraction = 0.25) {
  if (!(fraction > 0 && fraction < 1)) {
    throw new RangeError("Test fraction must be between zero and one.");
  }

  const shuffled = [...data];
  const random = createRandom(71);

  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const boundary = Math.floor(shuffled.length * (1 - fraction));
  return [shuffled.slice(0, boundary), shuffled.slice(boundary)];
}

function accuracy(actual, predicted) {
  if (actual.length === 0 || actual.length !== predicted.length) {
    throw new RangeError("Labels must be nonempty and have equal lengths.");
  }

  return actual.reduce(
    (sum, label, index) => sum + Number(label === predicted[index]),
    0
  ) / actual.length;
}

function binaryLogLoss(actual, probabilities) {
  if (actual.length === 0 || actual.length !== probabilities.length) {
    throw new RangeError("Labels and probabilities must have equal lengths.");
  }

  return actual.reduce((loss, label, index) => {
    const p = probabilities[index];

    if (![0, 1].includes(label) || !Number.isFinite(p) || p < 0 || p > 1) {
      throw new RangeError("Invalid binary label or probability.");
    }

    const clipped = Math.max(1e-15, Math.min(1 - 1e-15, p));
    return loss - label * Math.log(clipped) -
      (1 - label) * Math.log(1 - clipped);
  }, 0) / actual.length;
}

class LogisticModel {
  constructor({ rate = 0.15, epochs = 400, lambda = 0.02 } = {}) {
    if (rate <= 0 || epochs < 1 || lambda < 0) {
      throw new RangeError("Invalid logistic model configuration.");
    }

    this.rate = rate;
    this.epochs = epochs;
    this.lambda = lambda;
    this.weights = [];
    this.bias = 0;
  }

  fit(rows) {
    if (!rows.length) throw new Error("Training data is empty.");

    const dimensions = rows[0].features.length;
    if (!dimensions || rows.some(row =>
      row.features.length !== dimensions ||
      ![0, 1].includes(row.label) ||
      row.features.some(x => !Number.isFinite(x))
    )) {
      throw new TypeError("Inconsistent features or invalid labels.");
    }

    this.weights = Array(dimensions).fill(0);
    this.bias = 0;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const gradients = Array(dimensions).fill(0);
      let biasGradient = 0;

      for (const row of rows) {
        const score = this.bias + row.features.reduce(
          (sum, x, j) => sum + x * this.weights[j], 0
        );
        const error = sigmoid(score) - row.label;

        row.features.forEach((x, j) => {
          gradients[j] += error * x;
        });
        biasGradient += error;
      }

      this.weights = this.weights.map((weight, j) =>
        weight - this.rate * (
          gradients[j] / rows.length + this.lambda * weight
        )
      );

      this.bias -= this.rate * biasGradient / rows.length;
    }

    return this;
  }

  predictProbability(features) {
    if (features.length !== this.weights.length) {
      throw new RangeError("Unexpected feature dimension.");
    }

    return sigmoid(this.bias + features.reduce(
      (sum, x, j) => sum + x * this.weights[j], 0
    ));
  }

  predict(rows) {
    return rows.map(row => Number(this.predictProbability(row.features) >= 0.5));
  }
}

class Stump {
  fit(rows, sampleWeights = null) {
    if (!rows.length) throw new Error("Cannot fit a stump to empty data.");

    const weights = sampleWeights ?? rows.map(() => 1 / rows.length);

    if (weights.length !== rows.length ||
        weights.some(w => !Number.isFinite(w) || w < 0) ||
        weights.reduce((a, b) => a + b, 0) <= 0) {
      throw new RangeError("Invalid sample weights.");
    }

    const dimensions = rows[0].features.length;
    let best = { error: Infinity };

    for (let feature = 0; feature < dimensions; feature++) {
      const values = [...new Set(rows.map(row => row.features[feature]))]
        .sort((a, b) => a - b);

      const thresholds = values.length === 1
        ? values
        : values.slice(0, -1).map((value, i) =>
          (value + values[i + 1]) / 2
        );

      for (const threshold of thresholds) {
        for (const leftLabel of [0, 1]) {
          const rightLabel = 1 - leftLabel;
          let error = 0;

          rows.forEach((row, i) => {
            const prediction = row.features[feature] <= threshold
              ? leftLabel : rightLabel;

            if (prediction !== row.label) error += weights[i];
          });

          if (error < best.error) {
            best = { error, feature, threshold, leftLabel, rightLabel };
          }
        }
      }
    }

    Object.assign(this, best);
    return this;
  }

  predictOne(features) {
    return features[this.feature] <= this.threshold
      ? this.leftLabel : this.rightLabel;
  }

  predict(rows) {
    return rows.map(row => this.predictOne(row.features));
  }
}

function bootstrapSample(rows, random) {
  if (!rows.length) throw new Error("Cannot sample empty training data.");

  return Array.from({ length: rows.length }, () =>
    rows[Math.floor(random() * rows.length)]
  );
}

class BaggingEnsemble {
  constructor(size = 35, seed = 82) {
    if (!Number.isInteger(size) || size < 1) {
      throw new RangeError("Ensemble size must be a positive integer.");
    }

    this.size = size;
    this.seed = seed;
    this.models = [];
  }

  fit(rows) {
    const random = createRandom(this.seed);
    this.models = Array.from({ length: this.size }, () =>
      new Stump().fit(bootstrapSample(rows, random))
    );
    return this;
  }

  predict(rows) {
    if (!this.models.length) throw new Error("Fit bagging before prediction.");

    return rows.map(row => {
      const votes = this.models.reduce(
        (sum, model) => sum + model.predictOne(row.features), 0
      );

      // A deterministic tie rule avoids unstable behavior.
      return Number(votes > this.models.length / 2);
    });
  }
}

class AdaBoostEnsemble {
  constructor(rounds = 25) {
    if (!Number.isInteger(rounds) || rounds < 1) {
      throw new RangeError("Boosting rounds must be a positive integer.");
    }

    this.rounds = rounds;
    this.models = [];
    this.alphas = [];
  }

  fit(rows) {
    if (!rows.length || rows.some(row => ![0, 1].includes(row.label))) {
      throw new TypeError("AdaBoost requires nonempty binary training data.");
    }

    const n = rows.length;
    let weights = Array(n).fill(1 / n);
    this.models = [];
    this.alphas = [];

    for (let round = 0; round < this.rounds; round++) {
      const model = new Stump().fit(rows, weights);
      const predictions = model.predict(rows);

      const error = predictions.reduce((sum, prediction, i) =>
        sum + (prediction === rows[i].label ? 0 : weights[i]), 0);

      if (error >= 0.5 - 1e-12) break;

      if (error <= 1e-12) {
        this.models.push(model);
        this.alphas.push(10);
        break;
      }

      const alpha = 0.5 * Math.log((1 - error) / error);
      this.models.push(model);
      this.alphas.push(alpha);

      const updated = weights.map((weight, i) => {
        const correct = predictions[i] === rows[i].label;
        return weight * Math.exp(correct ? -alpha : alpha);
      });

      const total = updated.reduce((a, b) => a + b, 0);
      if (!(total > 0) || !Number.isFinite(total)) {
        throw new Error("Boosting weight normalization failed.");
      }

      weights = updated.map(weight => weight / total);
    }

    if (!this.models.length) {
      throw new Error("No weak learner achieved better-than-chance error.");
    }

    return this;
  }

  score(features) {
    return this.models.reduce((sum, model, i) =>
      sum + this.alphas[i] * (model.predictOne(features) ? 1 : -1), 0);
  }

  predict(rows) {
    return rows.map(row => Number(this.score(row.features) >= 0));
  }

  // These transformed margins are scores, not calibrated probabilities.
  predictProbability(features) {
    return sigmoid(2 * this.score(features));
  }
}

function hardVote(predictions, weights = null) {
  if (!predictions.length) throw new Error("Voting requires base predictions.");

  const count = predictions[0].length;
  if (predictions.some(model => model.length !== count ||
      model.some(label => ![0, 1].includes(label)))) {
    throw new RangeError("Voting requires equally sized binary predictions.");
  }

  const modelWeights = weights ?? predictions.map(() => 1);

  if (modelWeights.length !== predictions.length ||
      modelWeights.some(weight => !Number.isFinite(weight) || weight < 0) ||
      modelWeights.reduce((a, b) => a + b, 0) <= 0) {
    throw new RangeError("Voting weights must be nonnegative with positive sum.");
  }

  return Array.from({ length: count }, (_, i) => {
    const positive = predictions.reduce(
      (sum, model, j) => sum + modelWeights[j] * model[i], 0
    );
    const total = modelWeights.reduce((a, b) => a + b, 0);
    return Number(positive > total / 2);
  });
}

function blendProbabilities(validationRows, modelA, modelB) {
  // The blending weight is chosen on validation data, not on the final test set.
  const actual = validationRows.map(row => row.label);
  const probabilitiesA = validationRows.map(row =>
    modelA.predictProbability(row.features)
  );
  const probabilitiesB = validationRows.map(row =>
    modelB.predictProbability(row.features)
  );

  let bestWeight = 0.5;
  let bestLoss = Infinity;

  for (let weightA = 0; weightA <= 1.00001; weightA += 0.05) {
    const probabilities = probabilitiesA.map((a, i) =>
      weightA * a + (1 - weightA) * probabilitiesB[i]
    );

    const loss = binaryLogLoss(actual, probabilities);
    if (loss < bestLoss) {
      bestLoss = loss;
      bestWeight = Math.min(1, weightA);
    }
  }

  return { weightA: bestWeight, weightB: 1 - bestWeight, validationLoss: bestLoss };
}

function run() {
  const dataset = makeDataset();
  const [development, test] = splitDataset(dataset, 0.25);
  const [training, validation] = splitDataset(development, 0.25);

  const logistic = new LogisticModel().fit(training);
  const bagging = new BaggingEnsemble(41).fit(training);
  const boosting = new AdaBoostEnsemble(30).fit(training);
  const stump = new Stump().fit(training);

  const actual = test.map(row => row.label);
  const predictions = [
    logistic.predict(test),
    stump.predict(test),
    bagging.predict(test),
    boosting.predict(test)
  ];

  console.log("Ensemble learning with Node.js");
  console.log("--------------------------------");

  ["Logistic regression", "Decision stump", "Bagging", "AdaBoost"]
    .forEach((name, i) => {
      console.log(`${name.padEnd(23)} accuracy=${accuracy(actual, predictions[i]).toFixed(3)}`);
    });

  console.log(`Hard voting accuracy: ${accuracy(actual, hardVote(predictions)).toFixed(3)}`);

  const logisticProbabilities = test.map(row =>
    logistic.predictProbability(row.features)
  );
  const boostingProbabilities = test.map(row =>
    boosting.predictProbability(row.features)
  );

  const blendedProbabilities = logisticProbabilities.map((p, i) =>
    0.5 * p + 0.5 * boostingProbabilities[i]
  );

  console.log(`Logistic log loss: ${binaryLogLoss(actual, logisticProbabilities).toFixed(4)}`);
  console.log(`Equal blend log loss: ${binaryLogLoss(actual, blendedProbabilities).toFixed(4)}`);

  const validationLogistic = new LogisticModel().fit(training);
  const validationBoosting = new AdaBoostEnsemble(30).fit(training);
  const blend = blendProbabilities(
    validation, validationLogistic, validationBoosting
  );

  const testBlend = logisticProbabilities.map((p, i) =>
    blend.weightA * p + blend.weightB * boostingProbabilities[i]
  );

  console.log(
    `Validation-selected blend: logistic=${blend.weightA.toFixed(2)}, ` +
    `boosting=${blend.weightB.toFixed(2)}`
  );
  console.log(`Test blend log loss: ${binaryLogLoss(actual, testBlend).toFixed(4)}`);

  assert.equal(predictions.length, 4);
  assert.equal(testBlend.length, test.length);
  assert.ok(blend.validationLoss >= 0);
}

run();
