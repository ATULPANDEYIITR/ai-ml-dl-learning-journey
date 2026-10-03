/**
 * Categorical Data Encoding
 *
 * JavaScript-specific implementation of:
 * - label encoding
 * - one-hot encoding
 * - ordinal encoding
 * - smoothed target encoding
 * - event-driven encoding workflow
 * - asynchronous dataset processing
 * - schema validation
 * - unknown-category handling
 * - leakage-aware training/evaluation separation
 *
 * Run with:
 *   node categorical_encoding.js
 */

"use strict";

const MISSING_TOKEN = "__MISSING__";
const UNKNOWN_TOKEN = "__UNKNOWN__";

function section(title) {
  console.log(`\n${"=".repeat(78)}\n${title}\n${"=".repeat(78)}`);
}

function normalizeCategory(value) {
  if (value === null || value === undefined) {
    return MISSING_TOKEN;
  }

  if (typeof value === "string") {
    const cleaned = value.trim();
    return cleaned === "" ? MISSING_TOKEN : cleaned;
  }

  if (typeof value === "object") {
    throw new TypeError(
      "Categories must be scalar values in this implementation."
    );
  }

  return String(value);
}

function mean(values) {
  if (values.length === 0) {
    throw new Error("Cannot calculate the mean of an empty collection.");
  }

  return values.reduce((sum, value) => sum + Number(value), 0) / values.length;
}

function validateNumericTarget(target) {
  if (!Array.isArray(target) || target.length === 0) {
    throw new TypeError("Target must be a non-empty array.");
  }

  for (const value of target) {
    if (!Number.isFinite(Number(value))) {
      throw new TypeError(`Target value ${value} is not numeric.`);
    }
  }
}

function printRows(headers, rows) {
  const normalizedRows = rows.map((row) => row.map(String));
  const widths = headers.map((header, index) => {
    const values = normalizedRows.map((row) => row[index] ?? "");
    return Math.max(String(header).length, ...values.map((v) => v.length));
  });

  console.log(
    headers.map((header, index) => String(header).padEnd(widths[index])).join(" | ")
  );
  console.log(widths.map((width) => "-".repeat(width)).join("-+-"));

  for (const row of normalizedRows) {
    console.log(
      row.map((value, index) => value.padEnd(widths[index])).join(" | ")
    );
  }
}


class LabelEncoder {
  constructor({ handleUnknown = "error" } = {}) {
    if (!["error", "use_unknown"].includes(handleUnknown)) {
      throw new Error("Invalid unknown-category policy.");
    }

    this.handleUnknown = handleUnknown;
    this.mapping = new Map();
    this.inverseMapping = new Map();
  }

  fit(values) {
    const categories = [...new Set(values.map(normalizeCategory))].sort();

    this.mapping.clear();
    this.inverseMapping.clear();

    categories.forEach((category, index) => {
      this.mapping.set(category, index);
      this.inverseMapping.set(index, category);
    });

    if (this.handleUnknown === "use_unknown") {
      this.mapping.set(UNKNOWN_TOKEN, -1);
      this.inverseMapping.set(-1, UNKNOWN_TOKEN);
    }

    return this;
  }

  transform(values) {
    if (this.mapping.size === 0) {
      throw new Error("LabelEncoder must be fitted first.");
    }

    return values.map((value) => {
      const category = normalizeCategory(value);

      if (this.mapping.has(category)) {
        return this.mapping.get(category);
      }

      if (this.handleUnknown === "use_unknown") {
        return -1;
      }

      throw new Error(`Unknown category: ${category}`);
    });
  }

  inverseTransform(values) {
    return values.map((value) => {
      if (!this.inverseMapping.has(value)) {
        throw new Error(`Unknown encoded value: ${value}`);
      }

      return this.inverseMapping.get(value);
    });
  }
}


class OneHotEncoder {
  constructor({
    handleUnknown = "ignore",
    dropFirst = false,
  } = {}) {
    if (!["ignore", "error"].includes(handleUnknown)) {
      throw new Error("Invalid unknown-category policy.");
    }

    this.handleUnknown = handleUnknown;
    this.dropFirst = dropFirst;
    this.categories = [];
    this.outputCategories = [];
  }

  fit(values) {
    this.categories = [...new Set(values.map(normalizeCategory))].sort();

    this.outputCategories = this.dropFirst
      ? this.categories.slice(1)
      : [...this.categories];

    return this;
  }

  transform(values) {
    if (this.categories.length === 0) {
      throw new Error("OneHotEncoder must be fitted first.");
    }

    const knownCategories = new Set(this.categories);

    return values.map((value) => {
      const category = normalizeCategory(value);

      if (!knownCategories.has(category)) {
        if (this.handleUnknown === "error") {
          throw new Error(`Unknown category: ${category}`);
        }

        return new Array(this.outputCategories.length).fill(0);
      }

      return this.outputCategories.map(
        (outputCategory) => (category === outputCategory ? 1 : 0)
      );
    });
  }

  featureNames(prefix) {
    return this.outputCategories.map(
      (category) => `${prefix}__${category}`
    );
  }
}


class OrdinalEncoder {
  constructor(orderedCategories, { handleUnknown = "error" } = {}) {
    if (!Array.isArray(orderedCategories) || orderedCategories.length === 0) {
      throw new Error("Ordinal encoding requires an explicit category order.");
    }

    const normalized = orderedCategories.map(normalizeCategory);

    if (new Set(normalized).size !== normalized.length) {
      throw new Error("Ordinal categories must be unique.");
    }

    this.categories = normalized;
    this.handleUnknown = handleUnknown;
    this.mapping = new Map(
      normalized.map((category, index) => [category, index])
    );
  }

  transform(values) {
    return values.map((value) => {
      const category = normalizeCategory(value);

      if (this.mapping.has(category)) {
        return this.mapping.get(category);
      }

      if (this.handleUnknown === "use_unknown") {
        return -1;
      }

      throw new Error(
        `Unknown ordinal category '${category}'. ` +
        `Expected: ${this.categories.join(", ")}`
      );
    });
  }
}


class TargetEncoder {
  constructor({ smoothing = 10 } = {}) {
    if (!Number.isFinite(smoothing) || smoothing < 0) {
      throw new Error("Smoothing must be a finite non-negative number.");
    }

    this.smoothing = smoothing;
    this.globalMean = null;
    this.statistics = new Map();
  }

  fit(categories, target) {
    if (categories.length !== target.length) {
      throw new Error("Category and target arrays must have equal length.");
    }

    validateNumericTarget(target);

    if (categories.length === 0) {
      throw new Error("TargetEncoder requires training observations.");
    }

    const normalized = categories.map(normalizeCategory);
    this.globalMean = mean(target.map(Number));
    this.statistics.clear();

    const grouped = new Map();

    normalized.forEach((category, index) => {
      if (!grouped.has(category)) {
        grouped.set(category, []);
      }

      grouped.get(category).push(Number(target[index]));
    });

    for (const [category, values] of grouped.entries()) {
      this.statistics.set(category, {
        count: values.length,
        rawMean: mean(values),
      });
    }

    return this;
  }

  transform(categories) {
    if (this.globalMean === null) {
      throw new Error("TargetEncoder must be fitted before transform().");
    }

    return categories.map((value) => {
      const category = normalizeCategory(value);
      const stats = this.statistics.get(category);

      if (!stats) {
        return this.globalMean;
      }

      const { count, rawMean } = stats;

      return (
        (count * rawMean + this.smoothing * this.globalMean) /
        (count + this.smoothing)
      );
    });
  }

  describe() {
    if (this.globalMean === null) {
      throw new Error("TargetEncoder has not been fitted.");
    }

    return [...this.statistics.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([category, stats]) => {
        const encoded =
          (stats.count * stats.rawMean +
            this.smoothing * this.globalMean) /
          (stats.count + this.smoothing);

        return {
          category,
          count: stats.count,
          rawMean: Number(stats.rawMean.toFixed(4)),
          encoded: Number(encoded.toFixed(4)),
        };
      });
  }
}


class EncodingWorkflow {
  /**
   * Event-driven state management demonstrates a JavaScript-specific concern:
   * preprocessing often sits inside larger application workflows. Events make
   * successful fitting, transformation, and failure observable without
   * coupling the encoder itself to a UI or logging framework.
   */
  constructor() {
    this.listeners = new Map();
  }

  on(eventName, listener) {
    if (!this.listeners.has(eventName)) {
      this.listeners.set(eventName, []);
    }

    this.listeners.get(eventName).push(listener);
  }

  emit(eventName, payload) {
    for (const listener of this.listeners.get(eventName) ?? []) {
      listener(payload);
    }
  }

  async run(featureName, encoder, trainingValues, evaluationValues, target) {
    this.emit("fit:start", { featureName });

    try {
      if (encoder instanceof TargetEncoder) {
        encoder.fit(trainingValues, target);
      } else {
        encoder.fit(trainingValues);
      }

      this.emit("fit:complete", { featureName });

      await Promise.resolve();

      const transformed = encoder.transform(evaluationValues);

      this.emit("transform:complete", {
        featureName,
        rows: transformed.length,
      });

      return transformed;
    } catch (error) {
      this.emit("error", {
        featureName,
        message: error.message,
      });
      throw error;
    }
  }
}


function buildDataset() {
  return {
    train: [
      { region: "North", plan: "Basic", channel: "Organic", segment: "Student", churn: 1 },
      { region: "North", plan: "Standard", channel: "Referral", segment: "Professional", churn: 0 },
      { region: "South", plan: "Premium", channel: "Partner", segment: "Enterprise", churn: 0 },
      { region: "West", plan: "Basic", channel: "Paid Search", segment: "Student", churn: 1 },
      { region: "East", plan: "Standard", channel: "Organic", segment: "Professional", churn: 0 },
      { region: "South", plan: "Basic", channel: "Paid Search", segment: "Student", churn: 1 },
      { region: "West", plan: "Premium", channel: "Referral", segment: "Enterprise", churn: 0 },
      { region: "North", plan: "Basic", channel: "Partner", segment: "Student", churn: 1 },
      { region: "East", plan: "Premium", channel: "Organic", segment: "Enterprise", churn: 0 },
      { region: "South", plan: "Standard", channel: "Referral", segment: "Professional", churn: 0 },
    ],
    evaluation: [
      { region: "West", plan: "Standard", channel: "Organic", segment: "Student", churn: 1 },
      { region: "Central", plan: "Premium", channel: "Partner", segment: "Enterprise", churn: 0 },
      { region: "East", plan: "Basic", channel: "Social", segment: "New Segment", churn: 1 },
    ],
  };
}


function demonstrateLabelEncoding(dataset) {
  section("Label Encoding");

  const encoder = new LabelEncoder({ handleUnknown: "use_unknown" });
  encoder.fit(dataset.train.map((row) => row.region));

  const values = ["North", "South", "Central"];
  const encoded = encoder.transform(values);

  printRows(
    ["Region", "Encoded"],
    values.map((value, index) => [value, encoded[index]])
  );

  console.log(
    "\nThe integer is a category identifier. It does not create a valid " +
    "numeric ordering for nominal regions."
  );
}


function demonstrateOneHotEncoding(dataset) {
  section("One-Hot Encoding");

  const encoder = new OneHotEncoder({
    handleUnknown: "ignore",
  });

  encoder.fit(dataset.train.map((row) => row.region));

  const values = ["North", "South", "Central"];
  const encoded = encoder.transform(values);
  const names = encoder.featureNames("region");

  printRows(
    names,
    encoded
  );

  console.log(
    "\nCentral is unseen during fitting, so its representation is all zeros."
  );
}


function demonstrateOrdinalEncoding(dataset) {
  section("Ordinal Encoding");

  const encoder = new OrdinalEncoder(
    ["Basic", "Standard", "Premium"],
    { handleUnknown: "use_unknown" }
  );

  const values = ["Basic", "Premium", "Standard", "Enterprise"];
  const encoded = encoder.transform(values);

  printRows(
    ["Plan", "Ordinal value"],
    values.map((value, index) => [value, encoded[index]])
  );

  console.log(
    "\nThe numerical order is deliberate because subscription tiers have " +
    "an explicit business hierarchy."
  );
}


function demonstrateTargetEncoding(dataset) {
  section("Target Encoding");

  const encoder = new TargetEncoder({ smoothing: 4 });

  const trainingSegments = dataset.train.map((row) => row.segment);
  const trainingTarget = dataset.train.map((row) => row.churn);

  encoder.fit(trainingSegments, trainingTarget);

  printRows(
    ["Category", "Count", "Raw mean", "Smoothed"],
    encoder.describe().map((row) => [
      row.category,
      row.count,
      row.rawMean,
      row.encoded,
    ])
  );

  const evaluationSegments = dataset.evaluation.map((row) => row.segment);
  const transformed = encoder.transform(evaluationSegments);

  printRows(
    ["Evaluation segment", "Encoded"],
    evaluationSegments.map((value, index) => [
      value,
      transformed[index].toFixed(4),
    ])
  );

  console.log(
    "\nUnseen evaluation categories use the training global mean rather " +
    "than evaluation outcomes."
  );
}


function demonstrateCrossFittedEncoding() {
  section("Cross-Fitted Target Encoding");

  const categories = [
    "Basic", "Basic", "Basic",
    "Premium", "Premium", "Premium",
    "Standard", "Standard", "Standard",
    "Basic", "Premium", "Standard",
  ];

  const target = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0];

  const folds = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
  ];

  const outOfFold = new Array(categories.length);

  for (const validationIndices of folds) {
    const validationSet = new Set(validationIndices);

    const trainingIndices = categories
      .map((_, index) => index)
      .filter((index) => !validationSet.has(index));

    const encoder = new TargetEncoder({ smoothing: 5 });

    encoder.fit(
      trainingIndices.map((index) => categories[index]),
      trainingIndices.map((index) => target[index])
    );

    const transformed = encoder.transform(
      validationIndices.map((index) => categories[index])
    );

    validationIndices.forEach((rowIndex, localIndex) => {
      outOfFold[rowIndex] = transformed[localIndex];
    });
  }

  printRows(
    ["Row", "Category", "Target", "Out-of-fold encoding"],
    categories.map((category, index) => [
      index,
      category,
      target[index],
      outOfFold[index].toFixed(4),
    ])
  );
}


async function demonstrateEventDrivenWorkflow(dataset) {
  section("Event-Driven JavaScript Workflow");

  const workflow = new EncodingWorkflow();

  workflow.on("fit:start", ({ featureName }) => {
    console.log(`Fitting encoder for ${featureName}...`);
  });

  workflow.on("fit:complete", ({ featureName }) => {
    console.log(`Finished fitting ${featureName}.`);
  });

  workflow.on("transform:complete", ({ featureName, rows }) => {
    console.log(`Transformed ${rows} evaluation rows for ${featureName}.`);
  });

  workflow.on("error", ({ featureName, message }) => {
    console.log(`Encoding error in ${featureName}: ${message}`);
  });

  const regionEncoder = new OneHotEncoder({ handleUnknown: "ignore" });

  const result = await workflow.run(
    "region",
    regionEncoder,
    dataset.train.map((row) => row.region),
    dataset.evaluation.map((row) => row.region)
  );

  console.log("Evaluation region matrix:", result);
}


function demonstrateLeakage(dataset) {
  section("Leakage Demonstration");

  const trainSegments = dataset.train.map((row) => row.segment);
  const trainTarget = dataset.train.map((row) => row.churn);

  const evaluationSegments = dataset.evaluation.map((row) => row.segment);
  const evaluationTarget = dataset.evaluation.map((row) => row.churn);

  const safeEncoder = new TargetEncoder({ smoothing: 3 });
  safeEncoder.fit(trainSegments, trainTarget);

  const safe = safeEncoder.transform(evaluationSegments);

  const leakyEncoder = new TargetEncoder({ smoothing: 3 });
  leakyEncoder.fit(
    [...trainSegments, ...evaluationSegments],
    [...trainTarget, ...evaluationTarget]
  );

  const leaky = leakyEncoder.transform(evaluationSegments);

  printRows(
    ["Segment", "Safe", "Leaky"],
    evaluationSegments.map((segment, index) => [
      segment,
      safe[index].toFixed(4),
      leaky[index].toFixed(4),
    ])
  );

  console.log(
    "\nThe leaky version incorporates evaluation outcomes and therefore " +
    "does not represent a legitimate future-prediction pipeline."
  );
}


function demonstrateEdgeCases() {
  section("Validation and Edge Cases");

  const labelEncoder = new LabelEncoder({ handleUnknown: "use_unknown" });
  labelEncoder.fit(["A", null, "B", ""]);

  console.log(
    "Missing values:",
    labelEncoder.transform(["A", null, "", "C"])
  );

  const oneHot = new OneHotEncoder({ handleUnknown: "ignore" });
  oneHot.fit(["A", "B"]);

  console.log("Unknown one-hot category:", oneHot.transform(["C"]));

  try {
    const ordinal = new OrdinalEncoder(
      ["Low", "Medium", "High"],
      { handleUnknown: "error" }
    );
    ordinal.transform(["Extreme"]);
  } catch (error) {
    console.log("Ordinal validation:", error.message);
  }

  try {
    const targetEncoder = new TargetEncoder();
    targetEncoder.fit(["A", "B"], [1]);
  } catch (error) {
    console.log("Target length validation:", error.message);
  }
}


async function main() {
  section("Categorical Data Encoding in JavaScript");

  const dataset = buildDataset();

  demonstrateLabelEncoding(dataset);
  demonstrateOneHotEncoding(dataset);
  demonstrateOrdinalEncoding(dataset);
  demonstrateTargetEncoding(dataset);
  demonstrateCrossFittedEncoding();
  demonstrateLeakage(dataset);
  demonstrateEdgeCases();
  await demonstrateEventDrivenWorkflow(dataset);

  section("Practical JavaScript Design Rules");

  console.log(
    "Keep fitted encoder state separate from raw application data so that " +
    "the same transformation rules can be reused during inference."
  );

  console.log(
    "For browser applications, never send sensitive target-derived statistics " +
    "to an untrusted client merely to perform preprocessing."
  );

  console.log(
    "For asynchronous data pipelines, await preprocessing stages explicitly " +
    "so model inference cannot begin before encoding state is ready."
  );

  console.log(
    "Treat unknown categories as an expected production condition rather " +
    "than an exceptional event when the application receives evolving data."
  );
}


main().catch((error) => {
  console.error("Fatal encoding workflow error:", error);
  process.exitCode = 1;
});
