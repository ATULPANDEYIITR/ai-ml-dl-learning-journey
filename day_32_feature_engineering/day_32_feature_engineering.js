"use strict";

/*
 * Feature Engineering in JavaScript
 *
 * This Node.js program models a time-aware customer feature pipeline.
 * It intentionally complements the Python implementation by emphasizing:
 *
 * - event-driven transaction ingestion
 * - immutable-style feature snapshots
 * - Map-based aggregation
 * - asynchronous feature computation
 * - policy-driven transformations
 * - reviewable feature metadata
 * - temporal leakage prevention
 * - deterministic serialization
 *
 * No npm dependencies are required.
 */

const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");


/* -------------------------------------------------------------------------
 * Domain data
 * ---------------------------------------------------------------------- */

const customers = [
  {
    id: "C001",
    age: 29,
    region: "North",
    channel: "organic",
    annualIncome: 72000,
    creditLimit: 12000,
    signupDate: "2025-01-12",
  },
  {
    id: "C002",
    age: 41,
    region: "South",
    channel: "referral",
    annualIncome: 98000,
    creditLimit: 20000,
    signupDate: "2024-11-04",
  },
  {
    id: "C003",
    age: 35,
    region: "West",
    channel: "paid_search",
    annualIncome: 61000,
    creditLimit: 9000,
    signupDate: "2025-02-20",
  },
  {
    id: "C004",
    age: 52,
    region: "East",
    channel: "partner",
    annualIncome: 145000,
    creditLimit: 30000,
    signupDate: "2024-08-15",
  },
];

const transactions = [
  {
    id: "T001",
    customerId: "C001",
    timestamp: "2025-04-01T09:30:00Z",
    amount: 120.50,
    category: "groceries",
    paymentMethod: "card",
    merchantRisk: 0.10,
  },
  {
    id: "T002",
    customerId: "C001",
    timestamp: "2025-04-02T11:10:00Z",
    amount: 42,
    category: "transport",
    paymentMethod: "wallet",
    merchantRisk: 0.05,
  },
  {
    id: "T003",
    customerId: "C001",
    timestamp: "2025-04-05T19:20:00Z",
    amount: 680,
    category: "electronics",
    paymentMethod: "card",
    merchantRisk: 0.40,
  },
  {
    id: "T004",
    customerId: "C001",
    timestamp: "2025-04-09T13:45:00Z",
    amount: 90,
    category: "dining",
    paymentMethod: "wallet",
    merchantRisk: 0.12,
  },
  {
    id: "T005",
    customerId: "C001",
    timestamp: "2025-04-14T22:15:00Z",
    amount: 1250,
    category: "travel",
    paymentMethod: "card",
    merchantRisk: 0.25,
  },

  {
    id: "T006",
    customerId: "C002",
    timestamp: "2025-04-01T10:20:00Z",
    amount: 250,
    category: "groceries",
    paymentMethod: "card",
    merchantRisk: 0.08,
  },
  {
    id: "T007",
    customerId: "C002",
    timestamp: "2025-04-03T16:20:00Z",
    amount: 1100,
    category: "travel",
    paymentMethod: "card",
    merchantRisk: 0.22,
  },
  {
    id: "T008",
    customerId: "C002",
    timestamp: "2025-04-06T08:30:00Z",
    amount: 340,
    category: "utilities",
    paymentMethod: "bank_transfer",
    merchantRisk: 0.04,
  },
  {
    id: "T009",
    customerId: "C002",
    timestamp: "2025-04-12T20:00:00Z",
    amount: 85,
    category: "dining",
    paymentMethod: "card",
    merchantRisk: 0.10,
  },

  {
    id: "T010",
    customerId: "C003",
    timestamp: "2025-04-02T09:15:00Z",
    amount: 55,
    category: "transport",
    paymentMethod: "wallet",
    merchantRisk: 0.07,
  },
  {
    id: "T011",
    customerId: "C003",
    timestamp: "2025-04-04T12:00:00Z",
    amount: 230,
    category: "groceries",
    paymentMethod: "card",
    merchantRisk: 0.10,
  },
  {
    id: "T012",
    customerId: "C003",
    timestamp: "2025-04-08T18:40:00Z",
    amount: 430,
    category: "electronics",
    paymentMethod: "card",
    merchantRisk: 0.35,
  },
  {
    id: "T013",
    customerId: "C003",
    timestamp: "2025-04-13T21:30:00Z",
    amount: 70,
    category: "dining",
    paymentMethod: "wallet",
    merchantRisk: 0.11,
  },

  {
    id: "T014",
    customerId: "C004",
    timestamp: "2025-04-01T08:00:00Z",
    amount: 450,
    category: "utilities",
    paymentMethod: "bank_transfer",
    merchantRisk: 0.03,
  },
  {
    id: "T015",
    customerId: "C004",
    timestamp: "2025-04-05T15:30:00Z",
    amount: 2100,
    category: "travel",
    paymentMethod: "card",
    merchantRisk: 0.28,
  },
  {
    id: "T016",
    customerId: "C004",
    timestamp: "2025-04-10T17:45:00Z",
    amount: 850,
    category: "electronics",
    paymentMethod: "card",
    merchantRisk: 0.38,
  },
];


/* -------------------------------------------------------------------------
 * Small reusable feature functions
 * ---------------------------------------------------------------------- */

function ageBand(age) {
  if (!Number.isFinite(age) || age <= 0) {
    throw new RangeError("Age must be a positive number.");
  }

  if (age < 25) return "under_25";
  if (age < 35) return "25_34";
  if (age < 45) return "35_44";
  if (age < 55) return "45_54";
  return "55_plus";
}


function daysBetween(startDate, endDate) {
  const start = new Date(startDate);
  const end = new Date(endDate);

  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    throw new TypeError("Both dates must be valid ISO date strings.");
  }

  return Math.floor((end - start) / 86400000);
}


function safeRatio(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) {
    throw new TypeError("Ratio inputs must be finite numbers.");
  }

  return denominator === 0 ? 0 : numerator / denominator;
}


function logTransform(value) {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("Log transformation requires a non-negative value.");
  }

  return Math.log1p(value);
}


/* -------------------------------------------------------------------------
 * Event-driven ingestion
 * ---------------------------------------------------------------------- */

/*
 * A tiny event emitter illustrates a JavaScript-specific design choice:
 * transactions can arrive asynchronously, while feature state is maintained
 * incrementally instead of recomputing the entire history after every event.
 */
class TransactionStream {
  constructor() {
    this.listeners = new Set();
  }

  subscribe(listener) {
    if (typeof listener !== "function") {
      throw new TypeError("Listener must be a function.");
    }

    this.listeners.add(listener);

    return () => {
      this.listeners.delete(listener);
    };
  }

  async publish(transaction) {
    for (const listener of this.listeners) {
      await listener(transaction);
    }
  }
}


/* -------------------------------------------------------------------------
 * Incremental aggregation state
 * ---------------------------------------------------------------------- */

class CustomerAggregator {
  constructor(customerIds) {
    this.state = new Map();

    for (const customerId of customerIds) {
      this.state.set(customerId, {
        transactionCount: 0,
        totalSpend: 0,
        maximumTransaction: 0,
        riskySpend: 0,
        categories: new Map(),
        paymentMethods: new Map(),
      });
    }
  }

  accept(transaction) {
    if (!this.state.has(transaction.customerId)) {
      throw new Error(
        `Transaction references unknown customer ${transaction.customerId}`
      );
    }

    if (!Number.isFinite(transaction.amount) || transaction.amount < 0) {
      throw new RangeError("Transaction amount must be non-negative.");
    }

    const state = this.state.get(transaction.customerId);

    state.transactionCount += 1;
    state.totalSpend += transaction.amount;
    state.maximumTransaction = Math.max(
      state.maximumTransaction,
      transaction.amount
    );

    if (transaction.merchantRisk >= 0.30) {
      state.riskySpend += transaction.amount;
    }

    state.categories.set(
      transaction.category,
      (state.categories.get(transaction.category) || 0) + transaction.amount
    );

    state.paymentMethods.set(
      transaction.paymentMethod,
      (state.paymentMethods.get(transaction.paymentMethod) || 0) + 1
    );
  }

  snapshot(customerId) {
    const state = this.state.get(customerId);

    if (!state) {
      throw new Error(`Unknown customer ${customerId}`);
    }

    const average = safeRatio(
      state.totalSpend,
      state.transactionCount
    );

    const categoryProbabilities = [...state.categories.values()]
      .map((amount) => safeRatio(amount, state.totalSpend))
      .filter((probability) => probability > 0);

    const categoryEntropy = categoryProbabilities.reduce(
      (entropy, probability) => entropy - probability * Math.log2(probability),
      0
    );

    const cardCount = state.paymentMethods.get("card") || 0;

    return {
      transactions30d: state.transactionCount,
      spend30d: state.totalSpend,
      averageTransaction30d: average,
      maximumTransaction30d: state.maximumTransaction,
      riskySpend30d: state.riskySpend,
      uniqueCategories30d: state.categories.size,
      categoryEntropy30d: categoryEntropy,
      cardUsageRatio: safeRatio(cardCount, state.transactionCount),
      largestPurchaseShare: safeRatio(
        state.maximumTransaction,
        state.totalSpend
      ),
    };
  }
}


/* -------------------------------------------------------------------------
 * Time-aware feature construction
 * ---------------------------------------------------------------------- */

function isWithinWindow(timestamp, predictionTime, days) {
  const transactionTime = new Date(timestamp);
  const prediction = new Date(predictionTime);

  if (
    Number.isNaN(transactionTime.getTime()) ||
    Number.isNaN(prediction.getTime())
  ) {
    throw new TypeError("Invalid timestamp in temporal feature calculation.");
  }

  const lowerBound = prediction.getTime() - days * 86400000;

  return (
    transactionTime.getTime() <= prediction.getTime() &&
    transactionTime.getTime() >= lowerBound
  );
}


function createTemporalFeatures(transaction) {
  const timestamp = new Date(transaction.timestamp);

  if (Number.isNaN(timestamp.getTime())) {
    throw new TypeError(`Invalid timestamp for ${transaction.id}`);
  }

  const hour = timestamp.getUTCHours();
  const day = timestamp.getUTCDay();

  return {
    transactionId: transaction.id,
    dayOfWeek: day,
    isWeekend: Number(day === 0 || day === 6),
    hour,
    isBusinessHour: Number(hour >= 9 && hour < 18),
    isNight: Number(hour < 6 || hour >= 22),
    month: timestamp.getUTCMonth() + 1,
  };
}


/* -------------------------------------------------------------------------
 * Policy-driven transformations
 * ---------------------------------------------------------------------- */

const featurePolicies = Object.freeze({
  annualIncome: {
    transform: (value) => logTransform(value),
    outputName: "logIncome",
  },

  creditLimit: {
    transform: (value) => logTransform(value),
    outputName: "logCreditLimit",
  },
});


function applyNumericPolicies(customer) {
  const result = {};

  for (const [source, policy] of Object.entries(featurePolicies)) {
    const value = customer[source];

    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(`${source} must be non-negative.`);
    }

    result[policy.outputName] = policy.transform(value);
  }

  return result;
}


/* -------------------------------------------------------------------------
 * One-hot encoding with explicit unknown handling
 * ---------------------------------------------------------------------- */

function fitVocabulary(rows, field) {
  const vocabulary = new Set();

  for (const row of rows) {
    if (row[field] !== undefined && row[field] !== null) {
      vocabulary.add(String(row[field]));
    }
  }

  if (vocabulary.size === 0) {
    throw new Error(`No values found for ${field}.`);
  }

  return [...vocabulary].sort();
}


function encodeCategory(value, vocabulary, prefix) {
  const observed = String(value);
  const encoded = {};

  for (const category of vocabulary) {
    encoded[`${prefix}__${category}`] =
      Number(observed === category);
  }

  encoded[`${prefix}__unknown`] =
    Number(!vocabulary.includes(observed));

  return encoded;
}


/* -------------------------------------------------------------------------
 * Complete prediction snapshot
 * ---------------------------------------------------------------------- */

function buildPredictionSnapshot(
  customerList,
  transactionList,
  predictionTime
) {
  const customerMap = new Map(
    customerList.map((customer) => [customer.id, customer])
  );

  const aggregate = new CustomerAggregator(customerMap.keys());

  /*
   * Filtering by prediction time before aggregation is a direct defense
   * against temporal target leakage. A feature builder must not use events
   * that were unavailable when the prediction would have been made.
   */
  for (const transaction of transactionList) {
    if (isWithinWindow(transaction.timestamp, predictionTime, 30)) {
      aggregate.accept(transaction);
    }
  }

  return customerList.map((customer) => {
    const aggregation = aggregate.snapshot(customer.id);

    const tenureDays = daysBetween(
      customer.signupDate,
      predictionTime
    );

    if (tenureDays < 0) {
      throw new Error(
        `Customer ${customer.id} was not registered at prediction time.`
      );
    }

    const domainFeatures = {
      spendToCreditLimit: safeRatio(
        aggregation.spend30d,
        customer.creditLimit
      ),

      annualizedSpendToIncome: safeRatio(
        aggregation.spend30d * 12,
        customer.annualIncome
      ),

      merchantRiskIntensity: safeRatio(
        aggregation.riskySpend30d,
        aggregation.spend30d
      ),

      highValuePurchase: Number(
        aggregation.maximumTransaction30d >= 1000
      ),

      activeCustomer: Number(
        aggregation.transactions30d >= 3
      ),
    };

    return {
      customerId: customer.id,
      age: customer.age,
      ageBand: ageBand(customer.age),
      region: customer.region,
      acquisitionChannel: customer.channel,
      annualIncome: customer.annualIncome,
      creditLimit: customer.creditLimit,
      tenureDays,
      tenureYears: tenureDays / 365.25,
      ...applyNumericPolicies(customer),
      ...aggregation,
      ...domainFeatures,
    };
  });
}


/* -------------------------------------------------------------------------
 * Asynchronous pipeline
 * ---------------------------------------------------------------------- */

async function buildStreamingFeatures(customerList, transactionList, predictionTime) {
  const aggregator = new CustomerAggregator(
    customerList.map((customer) => customer.id)
  );

  const stream = new TransactionStream();

  /*
   * The subscriber represents a streaming feature service. The asynchronous
   * interface makes it possible to replace the in-memory listener with a
   * database, message broker, or online feature store integration later.
   */
  stream.subscribe(async (transaction) => {
    if (isWithinWindow(transaction.timestamp, predictionTime, 30)) {
      aggregator.accept(transaction);
    }
  });

  for (const transaction of transactionList) {
    await stream.publish(transaction);
  }

  return customerList.map((customer) => ({
    customerId: customer.id,
    ...aggregator.snapshot(customer.id),
  }));
}


/* -------------------------------------------------------------------------
 * Export
 * ---------------------------------------------------------------------- */

async function writeFeatureArtifacts(features, outputDirectory) {
  await fs.mkdir(outputDirectory, { recursive: true });

  const jsonPath = path.join(
    outputDirectory,
    "customer_features.json"
  );

  await fs.writeFile(
    jsonPath,
    JSON.stringify(features, null, 2),
    "utf8"
  );

  return jsonPath;
}


/* -------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------- */

async function runTests() {
  const predictionTime = "2025-04-15T23:59:00Z";

  const baseline = buildPredictionSnapshot(
    customers,
    transactions,
    predictionTime
  );

  const futureTransaction = {
    id: "FUTURE",
    customerId: "C001",
    timestamp: "2025-04-20T10:00:00Z",
    amount: 999999,
    category: "electronics",
    paymentMethod: "card",
    merchantRisk: 1.0,
  };

  const withFuture = buildPredictionSnapshot(
    customers,
    [...transactions, futureTransaction],
    predictionTime
  );

  assert.equal(
    baseline[0].spend30d,
    withFuture[0].spend30d,
    "Future transactions must not alter historical features."
  );

  assert.equal(
    baseline[0].ageBand,
    "25_34"
  );

  assert.equal(
    baseline[0].highValuePurchase,
    1
  );

  assert.equal(
    baseline[0].activeCustomer,
    1
  );

  const temporal = createTemporalFeatures(transactions[0]);

  assert.equal(
    temporal.isBusinessHour,
    1
  );

  const vocabulary = fitVocabulary(
    customers,
    "region"
  );

  const unknown = encodeCategory(
    "Central",
    vocabulary,
    "region"
  );

  assert.equal(
    unknown.region__unknown,
    1
  );

  const streaming = await buildStreamingFeatures(
    customers,
    transactions,
    predictionTime
  );

  assert.equal(
    streaming[0].spend30d,
    baseline[0].spend30d
  );

  console.log("All JavaScript feature-engineering tests passed.");
}


/* -------------------------------------------------------------------------
 * Demonstration
 * ---------------------------------------------------------------------- */

async function main() {
  const predictionTime = "2025-04-15T23:59:00Z";

  console.log("FEATURE ENGINEERING IN JAVASCRIPT");
  console.log("=".repeat(72));

  const features = buildPredictionSnapshot(
    customers,
    transactions,
    predictionTime
  );

  console.log("\nCustomer features");

  for (const feature of features) {
    console.log({
      customerId: feature.customerId,
      ageBand: feature.ageBand,
      spend30d: Number(feature.spend30d.toFixed(2)),
      spendToCreditLimit:
        Number(feature.spendToCreditLimit.toFixed(4)),
      merchantRiskIntensity:
        Number(feature.merchantRiskIntensity.toFixed(4)),
    });
  }

  console.log("\nCategory encoding");
  const regionVocabulary = fitVocabulary(customers, "region");

  for (const customer of customers.slice(0, 2)) {
    console.log(
      customer.id,
      encodeCategory(
        customer.region,
        regionVocabulary,
        "region"
      )
    );
  }

  console.log("\nTemporal feature example");
  console.log(
    createTemporalFeatures(transactions[0])
  );

  console.log("\nStreaming aggregation");
  const streamingFeatures = await buildStreamingFeatures(
    customers,
    transactions,
    predictionTime
  );

  for (const feature of streamingFeatures) {
    console.log(
      feature.customerId,
      {
        transactions30d: feature.transactions30d,
        spend30d: Number(feature.spend30d.toFixed(2)),
      }
    );
  }

  const outputDirectory = path.join(
    process.cwd(),
    "feature-output"
  );

  const outputPath = await writeFeatureArtifacts(
    features,
    outputDirectory
  );

  console.log(`\nFeature matrix written to ${outputPath}`);

  await runTests();
}


main().catch((error) => {
  console.error("Feature-engineering pipeline failed:", error.message);
  process.exitCode = 1;
});
