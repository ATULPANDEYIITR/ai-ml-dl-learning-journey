#!/usr/bin/env node
"use strict";

/*
 * Anomaly Detection Policy and Event Engine
 *
 * Implements complementary anomaly-detection workflows using:
 *   1. Isolation Forest
 *   2. Local Outlier Factor
 *   3. One-Class SVM-style RBF similarity
 *
 * JavaScript is used here as an event-driven monitoring service model:
 * telemetry arrives asynchronously, detectors evaluate it, and an event bus
 * emits anomaly decisions for downstream policy handling.
 *
 * No third-party packages are required.
 */

const crypto = require("node:crypto");
const { EventEmitter } = require("node:events");

class ValidationError extends Error {}

function assertFiniteNumber(value, field) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ValidationError(`${field} must be a finite number`);
  }
}

function validateVector(vector, dimensions = null) {
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new ValidationError("Feature vector must be a non-empty array");
  }

  vector.forEach((value, index) => assertFiniteNumber(value, `feature[${index}]`));

  if (dimensions !== null && vector.length !== dimensions) {
    throw new ValidationError("Feature dimension mismatch");
  }
}

function distance(a, b) {
  return Math.sqrt(a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0));
}

class StandardScaler {
  constructor() {
    this.means = [];
    this.stds = [];
  }

  fit(rows) {
    if (!Array.isArray(rows) || rows.length < 2) {
      throw new ValidationError("At least two observations are required");
    }

    const dimensions = rows[0].length;
    rows.forEach(row => validateVector(row, dimensions));

    this.means = Array.from({ length: dimensions }, (_, column) =>
      rows.reduce((sum, row) => sum + row[column], 0) / rows.length
    );

    this.stds = Array.from({ length: dimensions }, (_, column) => {
      const variance =
        rows.reduce(
          (sum, row) => sum + (row[column] - this.means[column]) ** 2,
          0
        ) / rows.length;

      return Math.sqrt(variance) > 1e-12 ? Math.sqrt(variance) : 1;
    });

    return this;
  }

  transform(rows) {
    if (this.means.length === 0) {
      throw new ValidationError("Scaler is not fitted");
    }

    return rows.map(row => {
      validateVector(row, this.means.length);
      return row.map(
        (value, i) => (value - this.means[i]) / this.stds[i]
      );
    });
  }
}

function quantile(values, q) {
  if (values.length === 0 || q < 0 || q > 1) {
    throw new ValidationError("Invalid quantile");
  }

  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);

  if (lower === upper) return sorted[lower];

  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function harmonic(n) {
  if (n <= 1) return 0;
  let result = 0;
  for (let i = 1; i <= n; i++) result += 1 / i;
  return result;
}

function expectedPathLength(n) {
  if (n <= 1) return 0;
  return 2 * harmonic(n - 1) - (2 * (n - 1)) / n;
}

class Random {
  constructor(seed = 42) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (1664525 * this.state + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }

  integer(max) {
    return Math.floor(this.next() * max);
  }

  uniform(min, max) {
    return min + this.next() * (max - min);
  }

  sample(array, size) {
    const copy = [...array];

    for (let i = copy.length - 1; i > 0; i--) {
      const j = this.integer(i + 1);
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }

    return copy.slice(0, size);
  }
}

class IsolationNode {
  constructor(size) {
    this.feature = null;
    this.split = null;
    this.left = null;
    this.right = null;
    this.size = size;
  }
}

class IsolationTree {
  constructor(maxDepth, rng) {
    this.maxDepth = maxDepth;
    this.rng = rng;
    this.root = null;
  }

  fit(rows) {
    this.root = this.build(rows, 0);
  }

  build(rows, depth) {
    const node = new IsolationNode(rows.length);
    const dimensions = rows[0].length;

    if (depth >= this.maxDepth || rows.length <= 1) {
      return node;
    }

    let variable = false;
    for (let column = 0; column < dimensions; column++) {
      const values = rows.map(row => row[column]);
      if (Math.max(...values) - Math.min(...values) > 1e-12) {
        variable = true;
        break;
      }
    }

    if (!variable) return node;

    const feature = this.rng.integer(dimensions);
    const values = rows.map(row => row[feature]);
    const min = Math.min(...values);
    const max = Math.max(...values);

    if (min >= max) return node;

    const split = this.rng.uniform(min, max);
    const left = rows.filter(row => row[feature] < split);
    const right = rows.filter(row => row[feature] >= split);

    if (left.length === 0 || right.length === 0) return node;

    node.feature = feature;
    node.split = split;
    node.left = this.build(left, depth + 1);
    node.right = this.build(right, depth + 1);

    return node;
  }

  pathLength(row) {
    let node = this.root;
    let depth = 0;

    while (
      node &&
      node.feature !== null &&
      node.left &&
      node.right
    ) {
      depth++;
      node = row[node.feature] < node.split ? node.left : node.right;
    }

    return depth + expectedPathLength(node ? node.size : 1);
  }
}

class IsolationForest {
  constructor({ trees = 80, sampleSize = 64, contamination = 0.08, seed = 42 } = {}) {
    if (trees < 1 || sampleSize < 2) {
      throw new ValidationError("Invalid Isolation Forest configuration");
    }

    if (contamination <= 0 || contamination >= 0.5) {
      throw new ValidationError("contamination must be between 0 and 0.5");
    }

    this.trees = trees;
    this.sampleSize = sampleSize;
    this.contamination = contamination;
    this.rng = new Random(seed);
    this.forest = [];
    this.normalizer = 1;
  }

  fit(rows) {
    if (rows.length < 4) {
      throw new ValidationError("Isolation Forest needs at least four rows");
    }

    const sampleSize = Math.min(this.sampleSize, rows.length);
    const maxDepth = Math.ceil(Math.log2(sampleSize));
    this.normalizer = expectedPathLength(sampleSize);
    this.forest = [];

    for (let i = 0; i < this.trees; i++) {
      const tree = new IsolationTree(maxDepth, this.rng);
      tree.fit(this.rng.sample(rows, sampleSize));
      this.forest.push(tree);
    }

    return this;
  }

  score(rows) {
    if (this.forest.length === 0) {
      throw new ValidationError("Forest is not fitted");
    }

    return rows.map(row => {
      const averagePath =
        this.forest.reduce((sum, tree) => sum + tree.pathLength(row), 0) /
        this.forest.length;

      return 2 ** (-averagePath / this.normalizer);
    });
  }

  predict(rows) {
    const scores = this.score(rows);
    const threshold = quantile(scores, 1 - this.contamination);

    return {
      detector: "Isolation Forest",
      threshold,
      scores,
      predictions: scores.map(score => score >= threshold)
    };
  }
}

class LocalOutlierFactor {
  constructor(k = 8) {
    if (k < 2) throw new ValidationError("LOF k must be at least two");
    this.k = k;
    this.training = [];
    this.kDistances = [];
    this.lrd = [];
  }

  neighbors(index) {
    return this.training
      .map((row, i) => ({
        index: i,
        distance: i === index ? Infinity : distance(this.training[index], row)
      }))
      .filter(item => item.index !== index)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, Math.min(this.k, this.training.length - 1));
  }

  fit(rows) {
    this.training = rows.map(row => [...row]);

    const lists = this.training.map((_, i) => this.neighbors(i));

    this.kDistances = lists.map(list =>
      list.length ? list[list.length - 1].distance : 0
    );

    this.lrd = lists.map((list) => {
      const sum = list.reduce(
        (total, item) =>
          total + Math.max(this.kDistances[item.index], item.distance),
        0
      );

      return sum > 1e-12 ? list.length / sum : Infinity;
    });

    return this;
  }

  score(rows) {
    if (this.training.length === 0) {
      throw new ValidationError("LOF is not fitted");
    }

    return rows.map(row => {
      const neighbors = this.training
        .map((train, index) => ({
          index,
          distance: distance(row, train)
        }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, Math.min(this.k, this.training.length));

      const reachability = neighbors.reduce(
        (sum, item) =>
          sum + Math.max(this.kDistances[item.index], item.distance),
        0
      );

      if (reachability <= 1e-12) return 1;

      const queryLrd = neighbors.length / reachability;
      const neighborLrd =
        neighbors.reduce((sum, item) => sum + this.lrd[item.index], 0) /
        neighbors.length;

      return neighborLrd / Math.max(queryLrd, 1e-12);
    });
  }

  predict(rows, threshold = 1.5) {
    const scores = this.score(rows);

    return {
      detector: "Local Outlier Factor",
      threshold,
      scores,
      predictions: scores.map(score => score >= threshold)
    };
  }
}

class OneClassSVM {
  constructor({ gamma = 0.18, nu = 0.08 } = {}) {
    if (gamma <= 0 || nu <= 0 || nu > 0.5) {
      throw new ValidationError("Invalid One-Class SVM configuration");
    }

    this.gamma = gamma;
    this.nu = nu;
    this.training = [];
  }

  kernel(a, b) {
    const squaredDistance = a.reduce(
      (sum, value, i) => sum + (value - b[i]) ** 2,
      0
    );

    return Math.exp(-this.gamma * squaredDistance);
  }

  fit(rows) {
    this.training = rows.map(row => [...row]);
    return this;
  }

  score(rows) {
    if (this.training.length === 0) {
      throw new ValidationError("One-Class SVM is not fitted");
    }

    return rows.map(row => {
      const similarity =
        this.training.reduce(
          (sum, train) => sum + this.kernel(row, train),
          0
        ) / this.training.length;

      return 1 - similarity;
    });
  }

  predict(rows) {
    const scores = this.score(rows);
    const threshold = quantile(scores, 1 - this.nu);

    return {
      detector: "One-Class SVM",
      threshold,
      scores,
      predictions: scores.map(score => score >= threshold)
    };
  }
}

class MonitoringEventBus extends EventEmitter {
  publish(event) {
    this.emit("telemetry", event);
  }
}

class RepositoryTelemetryService {
  constructor(detectors) {
    this.detectors = detectors;
    this.eventBus = new MonitoringEventBus();
    this.events = [];

    this.eventBus.on("telemetry", event => {
      this.events.push(event);
    });
  }

  async ingest(events) {
    for (const event of events) {
      await new Promise(resolve => setImmediate(resolve));

      if (!event.id || !Array.isArray(event.features)) {
        throw new ValidationError("Telemetry requires id and features");
      }

      validateVector(event.features);
      this.eventBus.publish(event);
    }
  }

  evaluate() {
    const rows = this.events.map(event => event.features);

    const results = this.detectors.map(detector => detector.predict(rows));

    return {
      generatedAt: new Date().toISOString(),
      eventCount: this.events.length,
      results
    };
  }
}

function seededTelemetry() {
  const rng = new Random(17);
  const events = [];

  for (let id = 1; id <= 140; id++) {
    events.push({
      id: crypto.createHash("sha256").update(`event-${id}`).digest("hex").slice(0, 12),
      knownAnomaly: false,
      features: [
        Math.max(1, 8 + (rng.next() - 0.5) * 6),
        Math.max(1, 120 + (rng.next() - 0.5) * 90),
        Math.max(0, 55 + (rng.next() - 0.5) * 40),
        Math.max(0.2, 18 + (rng.next() - 0.5) * 14),
        Math.max(0, (rng.next() - 0.5) * 2),
        Math.max(0, 1 + (rng.next() - 0.5) * 1.5)
      ]
    });
  }

  const anomalies = [
    [90, 2600, 1700, 1, 15, 12],
    [2, 5, 1, 250, 0, 0],
    [60, 1900, 80, 2, 16, 10],
    [3, 8, 2, 0.2, 18, 1],
    [80, 3200, 10, 4, 0, 13]
  ];

  anomalies.forEach((features, index) => {
    events.push({
      id: `known-${index + 1}`,
      knownAnomaly: true,
      features
    });
  });

  return events;
}

function printResults(events, evaluation) {
  console.log("ANOMALY MONITORING ENGINE");
  console.log("=".repeat(72));
  console.log(`Telemetry events: ${evaluation.eventCount}`);

  for (const result of evaluation.results) {
    const flagged = result.predictions.filter(Boolean).length;

    console.log(`\n${result.detector}`);
    console.log("-".repeat(72));
    console.log(`Threshold: ${result.threshold.toFixed(4)}`);
    console.log(`Flagged events: ${flagged}`);

    const ranking = result.scores
      .map((score, index) => ({
        score,
        event: events[index]
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 7);

    ranking.forEach(item => {
      console.log(
        `  ${item.event.id.padEnd(14)} ` +
        `score=${item.score.toFixed(4)} ` +
        `${item.event.knownAnomaly ? "KNOWN" : "UNLABELED"}`
      );
    });
  }
}

async function main() {
  const telemetry = seededTelemetry();

  const scaler = new StandardScaler().fit(
    telemetry.map(event => event.features)
  );

  const scaled = scaler.transform(
    telemetry.map(event => event.features)
  );

  const detectors = [
    new IsolationForest({
      trees: 90,
      sampleSize: 64,
      contamination: 0.06,
      seed: 99
    }).fit(scaled),

    new LocalOutlierFactor(9).fit(scaled),

    new OneClassSVM({
      gamma: 0.18,
      nu: 0.08
    }).fit(scaled)
  ];

  const service = new RepositoryTelemetryService(detectors);

  await service.ingest(
    telemetry.map((event, index) => ({
      ...event,
      features: scaled[index]
    }))
  );

  const evaluation = service.evaluate();
  printResults(service.events, evaluation);

  const consensus = evaluation.results[0].predictions.map(
    (_, index) =>
      evaluation.results.filter(result => result.predictions[index]).length >= 2
  );

  console.log("\nTwo-detector consensus");
  console.log("-".repeat(72));
  console.log(
    `Consensus alerts: ${consensus.filter(Boolean).length}`
  );

  console.log("\nOperational controls");
  console.log("-".repeat(72));
  console.log("1. Reject malformed or non-finite telemetry before scoring.");
  console.log("2. Keep feature normalization parameters versioned.");
  console.log("3. Record detector, threshold, timestamp, and feature schema.");
  console.log("4. Require authorization before exposing sensitive telemetry.");
  console.log("5. Treat model disagreement as a review signal.");
  console.log("6. Recalibrate thresholds when workload distributions change.");

  if (evaluation.eventCount !== telemetry.length) {
    throw new Error("Telemetry ingestion count mismatch");
  }
}

main().catch(error => {
  console.error(`Monitoring failure: ${error.message}`);
  process.exitCode = 1;
});
