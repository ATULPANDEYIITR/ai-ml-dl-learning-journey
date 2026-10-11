"use strict";

/*
 * Classification Metrics
 *
 * This implementation models a binary classifier using JavaScript objects,
 * then extends the model to threshold analysis, multiclass confusion
 * matrices, event-driven evaluation, and policy-oriented metric checks.
 *
 * Run with:
 *   node classification_metrics.js
 */

class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValidationError";
  }
}

function validateBinaryLabels(actual, predicted) {
  if (!Array.isArray(actual) || !Array.isArray(predicted)) {
    throw new ValidationError("Labels must be arrays.");
  }

  if (actual.length === 0) {
    throw new ValidationError("At least one observation is required.");
  }

  if (actual.length !== predicted.length) {
    throw new ValidationError("Actual and predicted arrays must have equal lengths.");
  }

  for (const value of [...actual, ...predicted]) {
    if (value !== 0 && value !== 1) {
      throw new ValidationError("Binary labels must be 0 or 1.");
    }
  }
}

function buildBinaryConfusionMatrix(actual, predicted) {
  validateBinaryLabels(actual, predicted);

  const counts = {
    truePositive: 0,
    falsePositive: 0,
    trueNegative: 0,
    falseNegative: 0
  };

  actual.forEach((observed, index) => {
    const estimate = predicted[index];

    if (observed === 1 && estimate === 1) counts.truePositive++;
    else if (observed === 0 && estimate === 1) counts.falsePositive++;
    else if (observed === 0 && estimate === 0) counts.trueNegative++;
    else counts.falseNegative++;
  });

  return counts;
}

function safeDivide(numerator, denominator) {
  return denominator === 0 ? 0 : numerator / denominator;
}

function accuracy(cm) {
  const correct = cm.truePositive + cm.trueNegative;
  const total =
    cm.truePositive +
    cm.falsePositive +
    cm.trueNegative +
    cm.falseNegative;

  return safeDivide(correct, total);
}

function precision(cm) {
  return safeDivide(
    cm.truePositive,
    cm.truePositive + cm.falsePositive
  );
}

function recall(cm) {
  return safeDivide(
    cm.truePositive,
    cm.truePositive + cm.falseNegative
  );
}

function specificity(cm) {
  return safeDivide(
    cm.trueNegative,
    cm.trueNegative + cm.falsePositive
  );
}

function f1(cm) {
  const p = precision(cm);
  const r = recall(cm);
  return safeDivide(2 * p * r, p + r);
}

function metricReport(cm) {
  return {
    accuracy: accuracy(cm),
    precision: precision(cm),
    recall: recall(cm),
    specificity: specificity(cm),
    f1: f1(cm)
  };
}

function printMetricReport(label, report) {
  console.log(`\n${label}`);
  for (const [name, value] of Object.entries(report)) {
    console.log(`${name.padEnd(12)} ${value.toFixed(4)}`);
  }
}

function validateProbabilityScores(scores) {
  if (!Array.isArray(scores) || scores.length === 0) {
    throw new ValidationError("Scores must be a non-empty array.");
  }

  if (scores.some(score => !Number.isFinite(score) || score < 0 || score > 1)) {
    throw new ValidationError("Probability scores must be finite values in [0, 1].");
  }
}

function predictAtThreshold(scores, threshold) {
  validateProbabilityScores(scores);

  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new ValidationError("Threshold must be in [0, 1].");
  }

  return scores.map(score => score >= threshold ? 1 : 0);
}

function evaluateThreshold(actual, scores, threshold) {
  const predicted = predictAtThreshold(scores, threshold);
  const confusion = buildBinaryConfusionMatrix(actual, predicted);

  return {
    threshold,
    confusion,
    metrics: metricReport(confusion)
  };
}

function compareThresholds(actual, scores, thresholds) {
  console.log("\n=== Threshold analysis ===");
  console.log(
    "threshold  accuracy  precision  recall  specificity  f1"
  );

  for (const threshold of thresholds) {
    const result = evaluateThreshold(actual, scores, threshold);
    const m = result.metrics;

    console.log(
      `${threshold.toFixed(2).padStart(9)}  ` +
      `${m.accuracy.toFixed(3).padStart(8)}  ` +
      `${m.precision.toFixed(3).padStart(9)}  ` +
      `${m.recall.toFixed(3).padStart(6)}  ` +
      `${m.specificity.toFixed(3).padStart(11)}  ` +
      `${m.f1.toFixed(3).padStart(5)}`
    );
  }
}

function buildMulticlassConfusionMatrix(actual, predicted, labels) {
  if (!Array.isArray(actual) || !Array.isArray(predicted)) {
    throw new ValidationError("Multiclass labels must be arrays.");
  }

  if (actual.length !== predicted.length || actual.length === 0) {
    throw new ValidationError("Multiclass observations must have equal non-zero lengths.");
  }

  if (new Set(labels).size !== labels.length) {
    throw new ValidationError("Class labels must be unique.");
  }

  const index = new Map(labels.map((label, position) => [label, position]));
  const matrix = labels.map(() => labels.map(() => 0));

  for (let i = 0; i < actual.length; i++) {
    if (!index.has(actual[i]) || !index.has(predicted[i])) {
      throw new ValidationError("Unknown class encountered.");
    }

    matrix[index.get(actual[i])][index.get(predicted[i])]++;
  }

  return matrix;
}

function oneVsRest(matrix, classIndex) {
  const size = matrix.length;

  if (size === 0 || matrix.some(row => row.length !== size)) {
    throw new ValidationError("Confusion matrix must be square.");
  }

  if (classIndex < 0 || classIndex >= size) {
    throw new ValidationError("Invalid class index.");
  }

  let truePositive = matrix[classIndex][classIndex];
  let falsePositive = 0;
  let falseNegative = 0;
  let trueNegative = 0;

  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      if (row === classIndex && column === classIndex) continue;
      if (row === classIndex) falseNegative += matrix[row][column];
      else if (column === classIndex) falsePositive += matrix[row][column];
      else trueNegative += matrix[row][column];
    }
  }

  return {
    truePositive,
    falsePositive,
    trueNegative,
    falseNegative
  };
}

function multiclassMetrics(matrix, labels) {
  const perClass = labels.map((label, index) => {
    const cm = oneVsRest(matrix, index);
    return {
      label,
      precision: precision(cm),
      recall: recall(cm),
      specificity: specificity(cm),
      f1: f1(cm),
      support: matrix[index].reduce((sum, value) => sum + value, 0)
    };
  });

  const total = perClass.reduce((sum, item) => sum + item.support, 0);
  const correct = matrix.reduce((sum, row, index) => sum + row[index], 0);

  const macroF1 =
    perClass.reduce((sum, item) => sum + item.f1, 0) / perClass.length;

  const weightedF1 =
    total === 0
      ? 0
      : perClass.reduce(
          (sum, item) => sum + item.f1 * item.support,
          0
        ) / total;

  return {
    accuracy: safeDivide(correct, total),
    macroF1,
    weightedF1,
    perClass
  };
}

class EvaluationEngine {
  constructor() {
    this.handlers = new Map();
  }

  on(eventName, handler) {
    if (typeof handler !== "function") {
      throw new TypeError("Event handler must be a function.");
    }

    if (!this.handlers.has(eventName)) {
      this.handlers.set(eventName, []);
    }

    this.handlers.get(eventName).push(handler);
  }

  emit(eventName, payload) {
    for (const handler of this.handlers.get(eventName) ?? []) {
      handler(payload);
    }
  }

  evaluate(actual, scores, threshold) {
    const result = evaluateThreshold(actual, scores, threshold);
    this.emit("evaluationComplete", result);
    return result;
  }
}

function demonstrateBinaryWorkflow() {
  const actual = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1];
  const predicted = [1, 0, 0, 0, 1, 1, 1, 0, 0, 1];

  const confusion = buildBinaryConfusionMatrix(actual, predicted);

  console.log("=== Binary confusion matrix ===");
  console.table(confusion);

  printMetricReport(
    "Binary classification metrics",
    metricReport(confusion)
  );
}

function demonstrateThresholdWorkflow() {
  const actual = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1];
  const probabilities = [
    0.91, 0.14, 0.61, 0.48, 0.57,
    0.31, 0.65, 0.79, 0.22, 0.44
  ];

  compareThresholds(actual, probabilities, [0.3, 0.5, 0.7]);
}

function demonstrateMulticlassWorkflow() {
  const labels = ["fraud", "legitimate", "review"];
  const actual = [
    "fraud", "fraud", "legitimate", "legitimate",
    "review", "review", "legitimate", "fraud",
    "review", "legitimate"
  ];
  const predicted = [
    "fraud", "review", "legitimate", "review",
    "review", "legitimate", "legitimate", "fraud",
    "review", "fraud"
  ];

  const matrix = buildMulticlassConfusionMatrix(actual, predicted, labels);

  console.log("\n=== Multiclass confusion matrix ===");
  console.table(
    matrix.map((row, index) => ({
      actual: labels[index],
      ...Object.fromEntries(labels.map((label, i) => [label, row[i]]))
    }))
  );

  const metrics = multiclassMetrics(matrix, labels);

  console.log(`Multiclass accuracy: ${metrics.accuracy.toFixed(4)}`);
  console.log(`Macro F1: ${metrics.macroF1.toFixed(4)}`);
  console.log(`Weighted F1: ${metrics.weightedF1.toFixed(4)}`);

  console.table(metrics.perClass);
}

async function demonstrateAsyncEvaluation() {
  const engine = new EvaluationEngine();

  engine.on("evaluationComplete", result => {
    console.log(
      `\nEvent received: threshold=${result.threshold}, ` +
      `F1=${result.metrics.f1.toFixed(3)}`
    );
  });

  const actual = [1, 0, 1, 0, 1, 0];
  const scores = [0.91, 0.12, 0.63, 0.72, 0.57, 0.18];

  await Promise.resolve();

  engine.evaluate(actual, scores, 0.5);
}

function demonstrateFailureConditions() {
  console.log("\n=== Validation failures ===");

  try {
    buildBinaryConfusionMatrix([1, 0], [1]);
  } catch (error) {
    console.log(`Length validation: ${error.message}`);
  }

  try {
    predictAtThreshold([0.2, 1.4], 0.5);
  } catch (error) {
    console.log(`Score validation: ${error.message}`);
  }

  try {
    predictAtThreshold([0.2, 0.4], 1.5);
  } catch (error) {
    console.log(`Threshold validation: ${error.message}`);
  }
}

async function main() {
  console.log("CLASSIFICATION METRICS LAB");
  console.log("==========================");

  demonstrateBinaryWorkflow();
  demonstrateThresholdWorkflow();
  demonstrateMulticlassWorkflow();
  demonstrateFailureConditions();
  await demonstrateAsyncEvaluation();

  console.log("\nEvaluation completed successfully.");
}

main().catch(error => {
  console.error("Fatal evaluation error:", error);
  process.exitCode = 1;
});
