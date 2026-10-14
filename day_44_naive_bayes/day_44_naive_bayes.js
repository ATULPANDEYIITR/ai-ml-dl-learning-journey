"use strict";

/*
 * Naive Bayes: event-driven classification and model diagnostics.
 *
 * Run with Node.js 18 or later:
 *     node naive_bayes.js
 *
 * This implementation emphasizes JavaScript-specific event handling,
 * asynchronous message ingestion, immutable model snapshots, and binary
 * versus count-based text features.
 */

const assert = require("node:assert/strict");

function logSumExpNormalize(logScores) {
  const entries = Object.entries(logScores);
  if (entries.length === 0) {
    throw new Error("At least one class score is required.");
  }

  const maximum = Math.max(...entries.map(([, score]) => score));
  if (!Number.isFinite(maximum)) {
    throw new Error("Scores must have a finite maximum.");
  }

  const exponentials = Object.fromEntries(
    entries.map(([label, score]) => [label, Math.exp(score - maximum)])
  );
  const denominator = Object.values(exponentials).reduce(
    (total, value) => total + value,
    0
  );

  return Object.fromEntries(
    Object.entries(exponentials).map(([label, value]) => [
      label,
      value / denominator,
    ])
  );
}

function tokenize(text) {
  return text
    .normalize("NFKC")
    .toLocaleLowerCase("en")
    .match(/[\p{L}\p{N}]+/gu) ?? [];
}

function validateDataset(documents, labels) {
  if (!Array.isArray(documents) || !Array.isArray(labels)) {
    throw new TypeError("Documents and labels must be arrays.");
  }
  if (documents.length === 0 || documents.length !== labels.length) {
    throw new Error("Documents and labels must be non-empty and aligned.");
  }
  if (documents.some((item) => typeof item !== "string")) {
    throw new TypeError("Every document must be a string.");
  }
  if (labels.some((label) => typeof label !== "string" || !label.trim())) {
    throw new TypeError("Every label must be a non-empty string.");
  }
}

class MultinomialTextClassifier {
  constructor(alpha = 1) {
    if (!Number.isFinite(alpha) || alpha <= 0) {
      throw new RangeError("alpha must be finite and positive.");
    }

    this.alpha = alpha;
    this.snapshot = null;
  }

  fit(documents, labels) {
    validateDataset(documents, labels);

    const grouped = new Map();
    const vocabulary = new Set();

    documents.forEach((document, index) => {
      const label = labels[index];
      if (!grouped.has(label)) grouped.set(label, []);
      const tokens = tokenize(document);
      grouped.get(label).push(tokens);
      tokens.forEach((token) => vocabulary.add(token));
    });

    if (vocabulary.size === 0) {
      throw new Error("Training documents contain no tokens.");
    }

    const classes = [...grouped.keys()].sort();
    const logPriors = {};
    const logLikelihoods = {};
    const totalDocuments = documents.length;

    for (const label of classes) {
      const classDocuments = grouped.get(label);
      const counts = new Map();
      let tokenTotal = 0;

      for (const tokens of classDocuments) {
        for (const token of tokens) {
          counts.set(token, (counts.get(token) ?? 0) + 1);
          tokenTotal += 1;
        }
      }

      const denominator = tokenTotal + this.alpha * vocabulary.size;
      logPriors[label] = Math.log(
        classDocuments.length / totalDocuments
      );
      logLikelihoods[label] = {};

      for (const token of vocabulary) {
        const count = counts.get(token) ?? 0;
        logLikelihoods[label][token] = Math.log(
          (count + this.alpha) / denominator
        );
      }
    }

    // Replace the model snapshot only after all calculations succeed.
    // Consumers never observe a partially fitted model.
    this.snapshot = Object.freeze({
      classes: Object.freeze(classes),
      vocabulary: Object.freeze([...vocabulary]),
      logPriors: Object.freeze(logPriors),
      logLikelihoods: Object.freeze(logLikelihoods),
    });

    return this;
  }

  score(document) {
    if (!this.snapshot) {
      throw new Error("The classifier has not been fitted.");
    }

    const tokenCounts = new Map();
    for (const token of tokenize(document)) {
      tokenCounts.set(token, (tokenCounts.get(token) ?? 0) + 1);
    }

    const scores = {};
    for (const label of this.snapshot.classes) {
      let score = this.snapshot.logPriors[label];

      for (const [token, count] of tokenCounts) {
        const likelihood = this.snapshot.logLikelihoods[label][token];
        if (likelihood !== undefined) {
          score += count * likelihood;
        }
      }

      scores[label] = score;
    }

    return scores;
  }

  predictProba(document) {
    return logSumExpNormalize(this.score(document));
  }

  predict(document) {
    const scores = this.score(document);
    return Object.keys(scores).reduce((best, label) =>
      scores[label] > scores[best] ? label : best
    );
  }
}

class BernoulliTextClassifier {
  constructor(alpha = 1) {
    if (!Number.isFinite(alpha) || alpha <= 0) {
      throw new RangeError("alpha must be finite and positive.");
    }
    this.alpha = alpha;
    this.snapshot = null;
  }

  fit(documents, labels) {
    validateDataset(documents, labels);

    const grouped = new Map();
    const vocabulary = new Set();

    documents.forEach((document, index) => {
      const features = new Set(tokenize(document));
      const label = labels[index];

      if (!grouped.has(label)) grouped.set(label, []);
      grouped.get(label).push(features);
      features.forEach((token) => vocabulary.add(token));
    });

    if (vocabulary.size === 0) {
      throw new Error("Training documents contain no tokens.");
    }

    const classes = [...grouped.keys()].sort();
    const logPriors = {};
    const logPresent = {};
    const logAbsent = {};

    for (const label of classes) {
      const examples = grouped.get(label);
      logPriors[label] = Math.log(examples.length / documents.length);
      logPresent[label] = {};
      logAbsent[label] = {};

      for (const token of vocabulary) {
        const count = examples.reduce(
          (total, features) => total + Number(features.has(token)),
          0
        );
        const probability =
          (count + this.alpha) / (examples.length + 2 * this.alpha);

        logPresent[label][token] = Math.log(probability);
        logAbsent[label][token] = Math.log1p(-probability);
      }
    }

    this.snapshot = Object.freeze({
      classes: Object.freeze(classes),
      vocabulary: Object.freeze([...vocabulary]),
      logPriors: Object.freeze(logPriors),
      logPresent: Object.freeze(logPresent),
      logAbsent: Object.freeze(logAbsent),
    });

    return this;
  }

  score(document) {
    if (!this.snapshot) {
      throw new Error("The classifier has not been fitted.");
    }

    const present = new Set(tokenize(document));
    const scores = {};

    for (const label of this.snapshot.classes) {
      let score = this.snapshot.logPriors[label];

      for (const token of this.snapshot.vocabulary) {
        score += present.has(token)
          ? this.snapshot.logPresent[label][token]
          : this.snapshot.logAbsent[label][token];
      }

      scores[label] = score;
    }

    return scores;
  }

  predictProba(document) {
    return logSumExpNormalize(this.score(document));
  }

  predict(document) {
    const scores = this.score(document);
    return Object.keys(scores).reduce((best, label) =>
      scores[label] > scores[best] ? label : best
    );
  }
}

class WorkflowEventBus {
  constructor() {
    this.listeners = new Map();
  }

  on(eventName, listener) {
    if (typeof listener !== "function") {
      throw new TypeError("Listener must be a function.");
    }

    if (!this.listeners.has(eventName)) {
      this.listeners.set(eventName, new Set());
    }

    this.listeners.get(eventName).add(listener);

    // Return an unsubscribe function so long-lived services can clean up.
    return () => this.listeners.get(eventName)?.delete(listener);
  }

  emit(eventName, payload) {
    const listeners = this.listeners.get(eventName);
    if (!listeners) return;

    // Snapshot the set so subscription changes do not alter this dispatch.
    for (const listener of [...listeners]) {
      listener(payload);
    }
  }
}

class SupportRoutingService {
  constructor(classifier, threshold = 0.55) {
    if (!(classifier instanceof MultinomialTextClassifier) &&
        !(classifier instanceof BernoulliTextClassifier)) {
      throw new TypeError("A fitted text classifier is required.");
    }
    if (!(threshold >= 0 && threshold <= 1)) {
      throw new RangeError("threshold must be between zero and one.");
    }

    this.classifier = classifier;
    this.threshold = threshold;
    this.events = new WorkflowEventBus();
  }

  async classifyMessage(message) {
    if (typeof message !== "string" || !message.trim()) {
      throw new TypeError("Message must contain non-whitespace text.");
    }

    // Asynchronous callers can feed messages from queues or HTTP handlers.
    await Promise.resolve();

    const probabilities = this.classifier.predictProba(message);
    const [category, confidence] = Object.entries(probabilities).reduce(
      (best, entry) => entry[1] > best[1] ? entry : best
    );

    const result = Object.freeze({
      category,
      confidence,
      needsHumanReview: confidence < this.threshold,
      probabilities: Object.freeze(probabilities),
    });

    this.events.emit("message:classified", result);
    return result;
  }
}

async function main() {
  console.log("Naive Bayes event-driven support routing\n");

  const documents = [
    "server timeout latency connection failure",
    "database server connection error",
    "network latency server",
    "invoice payment receipt billing",
    "overdue invoice account payment",
    "billing receipt transaction",
    "password reset login account",
    "authentication password security",
    "login access password",
  ];

  const labels = [
    "technical", "technical", "technical",
    "billing", "billing", "billing",
    "account", "account", "account",
  ];

  const multinomial = new MultinomialTextClassifier(1).fit(
    documents,
    labels
  );
  const bernoulli = new BernoulliTextClassifier(1).fit(documents, labels);

  const message = "server latency timeout after deployment";
  console.log("Multinomial prediction:", multinomial.predict(message));
  console.log("Multinomial probabilities:", multinomial.predictProba(message));
  console.log("Bernoulli prediction:", bernoulli.predict(message));
  console.log("Bernoulli probabilities:", bernoulli.predictProba(message));

  const router = new SupportRoutingService(multinomial, 0.55);
  const unsubscribe = router.events.on("message:classified", (result) => {
    console.log(
      `Event received: ${result.category}, ` +
      `confidence=${result.confidence.toFixed(3)}, ` +
      `humanReview=${result.needsHumanReview}`
    );
  });

  const messages = [
    "database connection timeout",
    "invoice payment overdue",
    "unrecognized phrase involving unrelated vocabulary",
  ];

  const results = await Promise.all(
    messages.map((item) => router.classifyMessage(item))
  );

  assert.equal(results.length, messages.length);
  assert.ok(results.every((item) => item.confidence >= 0));
  assert.ok(
    results.every((item) =>
      Math.abs(
        Object.values(item.probabilities).reduce((a, b) => a + b, 0) - 1
      ) < 1e-10
    )
  );

  unsubscribe();

  assert.equal(tokenize("Server, SERVER!").length, 2);
  assert.throws(
    () => new MultinomialTextClassifier(0),
    RangeError
  );
  assert.throws(
    () => new MultinomialTextClassifier().fit([""], ["empty"]),
    /no tokens/
  );

  console.log("\nAll assertions passed.");
}

main().catch((error) => {
  console.error("Execution failed:", error.message);
  process.exitCode = 1;
});
