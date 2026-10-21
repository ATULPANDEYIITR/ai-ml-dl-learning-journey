
# Educational histogram gradient boosting and ordered categorical statistics.

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic probability."""
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Labels and probabilities must have equal nonzero lengths.")
    epsilon = 1e-15
    return -sum(
        y * math.log(max(epsilon, min(1 - epsilon, p)))
        + (1 - y) * math.log(max(epsilon, min(1 - epsilon, 1 - p)))
        for y, p in zip(labels, probabilities)
    ) / len(labels)


@dataclass(frozen=True)
class Customer:
    tenure: float
    monthly_charge: float
    region: str
    plan: str
    churned: int

    def __post_init__(self) -> None:
        if self.tenure < 0 or self.monthly_charge < 0:
            raise ValueError("Tenure and monthly charge must be nonnegative.")
        if self.churned not in (0, 1):
            raise ValueError("The target must be binary.")
        if not self.region or not self.plan:
            raise ValueError("Categorical features cannot be empty.")


class OrderedTargetEncoder:
   Encode categories using only preceding labels in a deterministic order.

    def __init__(self, smoothing: float = 5.0, seed: int = 17) -> None:
        if smoothing <= 0:
            raise ValueError("Smoothing must be positive.")
        self.smoothing = smoothing
        self.seed = seed
        self.global_mean = 0.5
        self.statistics: dict[str, dict[str, tuple[float, int]]] = {}

    def fit_transform(
        self, rows: Sequence[Customer], feature: str
    ) -> list[float]:
        if not rows:
            raise ValueError("Cannot encode an empty dataset.")
        if feature not in {"region", "plan"}:
            raise ValueError("Unsupported categorical feature.")

        self.global_mean = sum(row.churned for row in rows) / len(rows)
        state: dict[str, tuple[float, int]] = {}
        output = [0.0] * len(rows)
        order = list(range(len(rows)))
        random.Random(self.seed).shuffle(order)

        for index in order:
            category = getattr(rows[index], feature)
            target_sum, count = state.get(category, (0.0, 0))
            output[index] = (
                target_sum + self.smoothing * self.global_mean
            ) / (count + self.smoothing)
            state[category] = (target_sum + rows[index].churned, count + 1)

        self.statistics[feature] = state
        return output

    def transform(self, categories: Sequence[str], feature: str) -> list[float]:
        state = self.statistics.get(feature, {})
        result = []
        for category in categories:
            target_sum, count = state.get(category, (0.0, 0))
            result.append(
                (target_sum + self.smoothing * self.global_mean)
                / (count + self.smoothing)
            )
        return result


@dataclass
class TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class HistogramTree:
    """Depth-constrained second-order tree using quantized feature values."""

    def __init__(
        self,
        max_bins: int = 8,
        max_depth: int = 3,
        min_leaf: int = 3,
        l2: float = 1.0,
        min_gain: float = 1e-8,
    ) -> None:
        if max_bins < 2 or max_depth < 0 or min_leaf < 1 or l2 < 0:
            raise ValueError("Invalid tree configuration.")
        self.max_bins = max_bins
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.l2 = l2
        self.min_gain = min_gain
        self.boundaries: list[list[float]] = []
        self.root: TreeNode | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        gradients: Sequence[float],
        hessians: Sequence[float],
    ) -> None:
        if not X or len(X) != len(gradients) or len(X) != len(hessians):
            raise ValueError("Inconsistent or empty training arrays.")
        width = len(X[0])
        if width == 0 or any(len(row) != width for row in X):
            raise ValueError("Feature rows must have consistent nonzero width.")
        if any(not math.isfinite(v) for row in X for v in row):
            raise ValueError("Features must be finite.")
        if any(not math.isfinite(g) for g in gradients):
            raise ValueError("Gradients must be finite.")
        if any(not math.isfinite(h) or h < 0 for h in hessians):
            raise ValueError("Hessians must be finite and nonnegative.")

        self.boundaries = []
        for feature in range(width):
            values = sorted({row[feature] for row in X})
            if len(values) <= self.max_bins:
                edges = values[:-1]
            else:
                edges = sorted({
                    values[min(len(values) - 1, i * len(values) // self.max_bins)]
                    for i in range(1, self.max_bins)
                })
            self.boundaries.append(edges)

        self.root = self._build(
            X, list(range(len(X))), gradients, hessians, 0
        )

    def _leaf_value(self, indices, gradients, hessians) -> float:
        gradient_sum = sum(gradients[i] for i in indices)
        hessian_sum = sum(hessians[i] for i in indices)
        return -gradient_sum / (hessian_sum + self.l2 + 1e-15)

    def _build(self, X, indices, gradients, hessians, depth) -> TreeNode:
        leaf_value = self._leaf_value(indices, gradients, hessians)
        node = TreeNode(leaf_value)
        if depth >= self.max_depth or len(indices) < 2 * self.min_leaf:
            return node

        parent_g = sum(gradients[i] for i in indices)
        parent_h = sum(hessians[i] for i in indices)
        parent_score = parent_g**2 / (parent_h + self.l2 + 1e-15)
        best_gain = self.min_gain
        best_split = None

        for feature, edges in enumerate(self.boundaries):
            for threshold in edges:
                left = [i for i in indices if X[i][feature] <= threshold]
                if len(left) < self.min_leaf:
                    continue
                if len(indices) - len(left) < self.min_leaf:
                    continue

                left_set = set(left)
                right = [i for i in indices if i not in left_set]
                lg = sum(gradients[i] for i in left)
                lh = sum(hessians[i] for i in left)
                rg = parent_g - lg
                rh = parent_h - lh
                gain = (
                    lg**2 / (lh + self.l2 + 1e-15)
                    + rg**2 / (rh + self.l2 + 1e-15)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, left, right)

        if best_split is None:
            return node

        feature, threshold, left, right = best_split
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X, left, gradients, hessians, depth + 1)
        node.right = self._build(X, right, gradients, hessians, depth + 1)
        return node

    def predict_one(self, row: Sequence[float]) -> float:
        if self.root is None:
            raise RuntimeError("The tree has not been fitted.")
        if len(row) != len(self.boundaries):
            raise ValueError("Prediction feature count mismatch.")
        node = self.root
        while not node.is_leaf:
            node = (
                node.left if row[node.feature] <= node.threshold else node.right
            )
            if node is None:
                raise RuntimeError("Invalid tree structure.")
        return node.value


class LightGBMStyleClassifier:
    """Compact histogram-based additive logistic boosting model."""

    def __init__(
        self,
        iterations: int = 12,
        learning_rate: float = 0.1,
        max_bins: int = 8,
        max_depth: int = 3,
        min_leaf: int = 3,
        l2: float = 1.0,
    ) -> None:
        if iterations < 1 or not 0 < learning_rate <= 1:
            raise ValueError("Invalid boosting configuration.")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.tree_config = (max_bins, max_depth, min_leaf, l2)
        self.initial_score = 0.0
        self.trees: list[HistogramTree] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        if not X or len(X) != len(y):
            raise ValueError("Features and labels must have equal nonzero lengths.")
        if any(label not in (0, 1) for label in y):
            raise ValueError("Labels must be binary.")
        if any(not math.isfinite(v) for row in X for v in row):
            raise ValueError("Features must be finite.")

        rate = sum(y) / len(y)
        rate = min(1 - 1e-6, max(1e-6, rate))
        self.initial_score = math.log(rate / (1 - rate))
        scores = [self.initial_score] * len(y)
        self.trees = []

        for _ in range(self.iterations):
            probabilities = [sigmoid(score) for score in scores]
            gradients = [p - target for p, target in zip(probabilities, y)]
            hessians = [max(1e-6, p * (1 - p)) for p in probabilities]
            tree = HistogramTree(*self.tree_config)
            tree.fit(X, gradients, hessians)
            self.trees.append(tree)
            scores = [
                score + self.learning_rate * tree.predict_one(row)
                for score, row in zip(scores, X)
            ]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        if not self.trees:
            raise RuntimeError("The classifier has not been fitted.")
        probabilities = []
        for row in X:
            score = self.initial_score
            for tree in self.trees:
                score += self.learning_rate * tree.predict_one(row)
            probabilities.append(sigmoid(score))
        return probabilities


def make_customers() -> list[Customer]:
    """Build a deterministic example dataset with categorical and numeric signals."""
    regions = ["north", "south", "east", "west"]
    plans = ["basic", "standard", "premium"]
    rows = []
    for i in range(120):
        tenure = float((i * 7) % 61)
        charge = float(25 + (i * 13) % 95)
        region = regions[(i * 3) % len(regions)]
        plan = plans[(i * 5) % len(plans)]
        churn = int(
            (tenure < 12 and charge > 65)
            or (plan == "basic" and charge > 90)
            or (region == "south" and tenure < 6)
        )
        rows.append(Customer(tenure, charge, region, plan, churn))
    return rows


def main() -> None:
    customers = make_customers()
    encoder = OrderedTargetEncoder(smoothing=5.0, seed=17)
    region_stat = encoder.fit_transform(customers, "region")
    plan_stat = encoder.fit_transform(customers, "plan")

    X = [
        [row.tenure, row.monthly_charge, region_stat[i], plan_stat[i]]
        for i, row in enumerate(customers)
    ]
    y = [row.churned for row in customers]

    # Keep a separate holdout partition for evaluation.
    split = int(len(X) * 0.8)
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    model = LightGBMStyleClassifier(
        iterations=15, learning_rate=0.1, max_bins=8, max_depth=3, min_leaf=3
    )
    model.fit(train_X, train_y)
    probabilities = model.predict_proba(test_X)
    accuracy = sum(
        (p >= 0.5) == bool(label)
        for p, label in zip(probabilities, test_y)
    ) / len(test_y)

    print(f"Training observations: {len(train_X)}")
    print(f"Test observations: {len(test_X)}")
    print(f"Test log loss: {log_loss(test_y, probabilities):.4f}")
    print(f"Test accuracy: {accuracy:.3f}")
    for index, probability in enumerate(probabilities[:5]):
        print(
            f"Customer {split + index}: "
            f"actual={test_y[index]}, churn_probability={probability:.3f}"
        )


if __name__ == "__main__":
    main()
```

```javascript
"use strict";

/**
 * Educational histogram boosting and ordered categorical statistics.
 * Run with Node.js 18 or later. No external packages are required.
 */

const sigmoid = (value) => {
  if (value >= 0) {
    const z = Math.exp(-Math.min(value, 700));
    return 1 / (1 + z);
  }
  const z = Math.exp(Math.max(value, -700));
  return z / (1 + z);
};

function logLoss(labels, probabilities) {
  if (!labels.length || labels.length !== probabilities.length) {
    throw new Error("Labels and probabilities must have equal nonzero lengths.");
  }
  const epsilon = 1e-15;
  return -labels.reduce((total, label, i) => {
    const p = Math.max(epsilon, Math.min(1 - epsilon, probabilities[i]));
    return total + label * Math.log(p) + (1 - label) * Math.log(1 - p);
  }, 0) / labels.length;
}

class Customer {
  constructor(tenure, monthlyCharge, region, plan, churned) {
    if (!Number.isFinite(tenure) || tenure < 0 ||
        !Number.isFinite(monthlyCharge) || monthlyCharge < 0) {
      throw new TypeError("Numeric features must be finite and nonnegative.");
    }
    if (!region || !plan || ![0, 1].includes(churned)) {
      throw new TypeError("Invalid category or binary target.");
    }
    Object.assign(this, { tenure, monthlyCharge, region, plan, churned });
    Object.freeze(this);
  }
}

class OrderedTargetEncoder {
  constructor(smoothing = 5, seed = 17) {
    if (!(smoothing > 0)) throw new RangeError("Smoothing must be positive.");
    this.smoothing = smoothing;
    this.seed = seed;
    this.globalMean = 0.5;
    this.statistics = new Map();
  }

  // Mulberry32 supplies repeatable pseudo-random ordering for the demonstration.
  shuffledIndices(length) {
    let state = this.seed >>> 0;
    const random = () => {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
    const indices = Array.from({ length }, (_, i) => i);
    for (let i = length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
  }

  fitTransform(rows, feature) {
    if (!rows.length) throw new Error("Cannot encode an empty dataset.");
    if (!["region", "plan"].includes(feature)) {
      throw new Error("Unsupported categorical feature.");
    }

    this.globalMean = rows.reduce((sum, row) => sum + row.churned, 0) / rows.length;
    const state = new Map();
    const output = new Array(rows.length);
    for (const index of this.shuffledIndices(rows.length)) {
      const row = rows[index];
      const category = row[feature];
      const current = state.get(category) || { sum: 0, count: 0 };
      output[index] = (
        current.sum + this.smoothing * this.globalMean
      ) / (current.count + this.smoothing);

      // The current label is added only after its encoding is computed.
      state.set(category, {
        sum: current.sum + row.churned,
        count: current.count + 1
      });
    }
    this.statistics.set(feature, state);
    return output;
  }

  transform(categories, feature) {
    const state = this.statistics.get(feature) || new Map();
    return categories.map((category) => {
      const current = state.get(category) || { sum: 0, count: 0 };
      return (
        current.sum + this.smoothing * this.globalMean
      ) / (current.count + this.smoothing);
    });
  }
}

class HistogramTree {
  constructor({ maxBins = 8, maxDepth = 3, minLeaf = 3, l2 = 1 } = {}) {
    if (maxBins < 2 || maxDepth < 0 || minLeaf < 1 || l2 < 0) {
      throw new RangeError("Invalid histogram tree configuration.");
    }
    this.maxBins = maxBins;
    this.maxDepth = maxDepth;
    this.minLeaf = minLeaf;
    this.l2 = l2;
    this.boundaries = [];
    this.root = null;
  }

  fit(X, gradients, hessians) {
    if (!X.length || X.length !== gradients.length ||
        X.length !== hessians.length) {
      throw new Error("Training arrays must have equal nonzero lengths.");
    }
    const width = X[0].length;
    if (!width || X.some(row => row.length !== width ||
        row.some(value => !Number.isFinite(value)))) {
      throw new Error("Features must be finite and have consistent dimensions.");
    }
    if (gradients.some(value => !Number.isFinite(value)) ||
        hessians.some(value => !Number.isFinite(value) || value < 0)) {
      throw new Error("Invalid gradient or Hessian.");
    }

    this.boundaries = Array.from({ length: width }, (_, feature) => {
      const values = [...new Set(X.map(row => row[feature]))].sort((a, b) => a - b);
      if (values.length <= this.maxBins) return values.slice(0, -1);
      const edges = new Set();
      for (let i = 1; i < this.maxBins; i++) {
        edges.add(values[Math.min(values.length - 1,
          Math.floor(i * values.length / this.maxBins))]);
      }
      return [...edges].sort((a, b) => a - b);
    });

    this.root = this.build(X, X.map((_, i) => i), gradients, hessians, 0);
  }

  leafValue(indices, gradients, hessians) {
    const g = indices.reduce((sum, i) => sum + gradients[i], 0);
    const h = indices.reduce((sum, i) => sum + hessians[i], 0);
    return -g / (h + this.l2 + 1e-15);
  }

  build(X, indices, gradients, hessians, depth) {
    const node = {
      value: this.leafValue(indices, gradients, hessians),
      feature: -1, threshold: 0, left: null, right: null
    };
    if (depth >= this.maxDepth || indices.length < 2 * this.minLeaf) return node;

    const sum = (items, values) => items.reduce((total, i) => total + values[i], 0);
    const parentG = sum(indices, gradients);
    const parentH = sum(indices, hessians);
    const parentScore = parentG ** 2 / (parentH + this.l2 + 1e-15);
    let bestGain = 1e-8;
    let bestSplit = null;

    for (let feature = 0; feature < this.boundaries.length; feature++) {
      for (const threshold of this.boundaries[feature]) {
        const left = indices.filter(i => X[i][feature] <= threshold);
        const right = indices.filter(i => X[i][feature] > threshold);
        if (left.length < this.minLeaf || right.length < this.minLeaf) continue;

        const lg = sum(left, gradients);
        const lh = sum(left, hessians);
        const rg = parentG - lg;
        const rh = parentH - lh;
        const gain = lg ** 2 / (lh + this.l2 + 1e-15)
          + rg ** 2 / (rh + this.l2 + 1e-15) - parentScore;

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = { feature, threshold, left, right };
        }
      }
    }

    if (!bestSplit) return node;
    node.feature = bestSplit.feature;
    node.threshold = bestSplit.threshold;
    node.left = this.build(X, bestSplit.left, gradients, hessians, depth + 1);
    node.right = this.build(X, bestSplit.right, gradients, hessians, depth + 1);
    return node;
  }

  predictOne(row) {
    if (!this.root) throw new Error("Tree has not been fitted.");
    if (row.length !== this.boundaries.length ||
        row.some(value => !Number.isFinite(value))) {
      throw new Error("Invalid prediction feature vector.");
    }
    let node = this.root;
    while (node.left !== null) {
      node = row[node.feature] <= node.threshold ? node.left : node.right;
    }
    return node.value;
  }
}

class HistogramBoostingClassifier {
  constructor({ iterations = 15, learningRate = 0.1, ...treeConfig } = {}) {
    if (iterations < 1 || !(learningRate > 0 && learningRate <= 1)) {
      throw new RangeError("Invalid boosting configuration.");
    }
    this.iterations = iterations;
    this.learningRate = learningRate;
    this.treeConfig = treeConfig;
    this.initialScore = 0;
    this.trees = [];
  }

  fit(X, y) {
    if (!X.length || X.length !== y.length ||
        y.some(label => ![0, 1].includes(label))) {
      throw new Error("Expected matching nonempty features and binary labels.");
    }
    const rate = Math.max(1e-6, Math.min(1 - 1e-6,
      y.reduce((sum, label) => sum + label, 0) / y.length));
    this.initialScore = Math.log(rate / (1 - rate));
    let scores = X.map(() => this.initialScore);
    this.trees = [];

    for (let iteration = 0; iteration < this.iterations; iteration++) {
      const probabilities = scores.map(sigmoid);
      const gradients = probabilities.map((p, i) => p - y[i]);
      const hessians = probabilities.map(p => Math.max(1e-6, p * (1 - p)));
      const tree = new HistogramTree(this.treeConfig);
      tree.fit(X, gradients, hessians);
      this.trees.push(tree);
      scores = scores.map((score, i) =>
        score + this.learningRate * tree.predictOne(X[i]));
    }
  }

  predictProba(X) {
    if (!this.trees.length) throw new Error("Model has not been fitted.");
    return X.map(row => {
      let score = this.initialScore;
      for (const tree of this.trees) {
        score += this.learningRate * tree.predictOne(row);
      }
      return sigmoid(score);
    });
  }
}

function createCustomers() {
  const regions = ["north", "south", "east", "west"];
  const plans = ["basic", "standard", "premium"];
  return Array.from({ length: 120 }, (_, i) => {
    const tenure = (i * 7) % 61;
    const monthlyCharge = 25 + (i * 13) % 95;
    const region = regions[(i * 3) % regions.length];
    const plan = plans[(i * 5) % plans.length];
    const churned = Number(
      (tenure < 12 && monthlyCharge > 65) ||
      (plan === "basic" && monthlyCharge > 90) ||
      (region === "south" && tenure < 6)
    );
    return new Customer(tenure, monthlyCharge, region, plan, churned);
  });
}

function main() {
  const customers = createCustomers();
  const encoder = new OrderedTargetEncoder(5, 17);
  const regions = encoder.fitTransform(customers, "region");
  const plans = encoder.fitTransform(customers, "plan");

  const X = customers.map((row, i) => [
    row.tenure, row.monthlyCharge, regions[i], plans[i]
  ]);
  const y = customers.map(row => row.churned);
  const split = Math.floor(X.length * 0.8);

  const model = new HistogramBoostingClassifier({
    iterations: 15, learningRate: 0.1, maxBins: 8, maxDepth: 3, minLeaf: 3
  });
  model.fit(X.slice(0, split), y.slice(0, split));

  const actual = y.slice(split);
  const probabilities = model.predictProba(X.slice(split));
  const accuracy = actual.reduce((sum, label, i) =>
    sum + Number((probabilities[i] >= 0.5) === Boolean(label)), 0) / actual.length;

  console.log(`Training observations: ${split}`);
  console.log(`Test observations: ${actual.length}`);
  console.log(`Test log loss: ${logLoss(actual, probabilities).toFixed(4)}`);
  console.log(`Test accuracy: ${accuracy.toFixed(3)}`);
  probabilities.slice(0, 5).forEach((probability, i) => {
    console.log(`Customer ${split + i}: actual=${actual[i]}, churn_probability=${probability.toFixed(3)}`);
  });
}

if (require.main === module) {
  main();
}
```

```cpp
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct Customer {
    double tenure;
    double monthlyCharge;
    string region;
    string plan;
    int churned;

    Customer(double t, double charge, string r, string p, int y)
        : tenure(t), monthlyCharge(charge), region(move(r)),
          plan(move(p)), churned(y) {
        if (!isfinite(tenure) || tenure < 0 ||
            !isfinite(monthlyCharge) || monthlyCharge < 0 ||
            region.empty() || plan.empty() || (churned != 0 && churned != 1)) {
            throw invalid_argument("Invalid customer observation.");
        }
    }
};

double sigmoid(double value) {
    if (value >= 0) {
        return 1.0 / (1.0 + exp(-min(value, 700.0)));
    }
    double z = exp(max(value, -700.0));
    return z / (1.0 + z);
}

double logLoss(const vector<int>& y, const vector<double>& probabilities) {
    if (y.empty() || y.size() != probabilities.size()) {
        throw invalid_argument("Invalid evaluation arrays.");
    }
    constexpr double epsilon = 1e-15;
    double loss = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        double p = clamp(probabilities[i], epsilon, 1.0 - epsilon);
        loss -= y[i] * log(p) + (1 - y[i]) * log(1.0 - p);
    }
    return loss / y.size();
}

class OrderedTargetEncoder {
    double smoothing;
    double globalMean = 0.5;
    map<string, pair<double, int>> statistics;

public:
    explicit OrderedTargetEncoder(double alpha = 5.0) : smoothing(alpha) {
        if (alpha <= 0) throw invalid_argument("Smoothing must be positive.");
    }

    vector<double> fitTransform(const vector<Customer>& rows, bool usePlan,
                                unsigned seed = 17) {
        if (rows.empty()) throw invalid_argument("Empty training dataset.");
        globalMean = accumulate(rows.begin(), rows.end(), 0.0,
            [](double sum, const Customer& row) { return sum + row.churned; })
            / rows.size();

        vector<size_t> order(rows.size());
        iota(order.begin(), order.end(), 0);
        mt19937 generator(seed);
        shuffle(order.begin(), order.end(), generator);

        statistics.clear();
        vector<double> encoded(rows.size());
        for (size_t index : order) {
            const Customer& row = rows[index];
            const string& category = usePlan ? row.plan : row.region;
            auto [targetSum, count] = statistics[category];
            encoded[index] = (targetSum + smoothing * globalMean)
                           / (count + smoothing);

            // Add the current target only after computing its representation.
            statistics[category] = {targetSum + row.churned, count + 1};
        }
        return encoded;
    }

    double transform(const string& category) const {
        auto it = statistics.find(category);
        if (it == statistics.end()) return globalMean;
        const auto& [targetSum, count] = it->second;
        return (targetSum + smoothing * globalMean) / (count + smoothing);
    }
};

struct TreeNode {
    double value = 0;
    int feature = -1;
    double threshold = 0;
    int left = -1;
    int right = -1;

    bool isLeaf() const { return left == -1; }
};

class HistogramTree {
    int maxBins;
    int maxDepth;
    int minLeaf;
    double l2;
    vector<vector<double>> boundaries;
    vector<TreeNode> nodes;

    double leafValue(const vector<int>& indices,
                     const vector<double>& gradients,
                     const vector<double>& hessians) const {
        double g = 0, h = 0;
        for (int i : indices) {
            g += gradients[i];
            h += hessians[i];
        }
        return -g / (h + l2 + 1e-15);
    }

    int build(const vector<vector<double>>& X, const vector<int>& indices,
              const vector<double>& gradients, const vector<double>& hessians,
              int depth) {
        TreeNode node;
        node.value = leafValue(indices, gradients, hessians);
        int nodeIndex = static_cast<int>(nodes.size());
        nodes.push_back(node);

        if (depth >= maxDepth || static_cast<int>(indices.size()) < 2 * minLeaf) {
            return nodeIndex;
        }

        double parentG = 0, parentH = 0;
        for (int i : indices) {
            parentG += gradients[i];
            parentH += hessians[i];
        }
        double parentScore = parentG * parentG / (parentH + l2 + 1e-15);
        double bestGain = 1e-8;
        int bestFeature = -1;
        double bestThreshold = 0;

        for (size_t feature = 0; feature < boundaries.size(); ++feature) {
            for (double threshold : boundaries[feature]) {
                double lg = 0, lh = 0;
                int leftCount = 0;
                for (int i : indices) {
                    if (X[i][feature] <= threshold) {
                        lg += gradients[i];
                        lh += hessians[i];
                        ++leftCount;
                    }
                }
                int rightCount = static_cast<int>(indices.size()) - leftCount;
                if (leftCount < minLeaf || rightCount < minLeaf) continue;

                double rg = parentG - lg, rh = parentH - lh;
                double gain = lg * lg / (lh + l2 + 1e-15)
                            + rg * rg / (rh + l2 + 1e-15) - parentScore;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = static_cast<int>(feature);
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) return nodeIndex;

        vector<int> leftIndices, rightIndices;
        for (int i : indices) {
            (X[i][bestFeature] <= bestThreshold ? leftIndices : rightIndices)
                .push_back(i);
        }

        int leftNode = build(X, leftIndices, gradients, hessians, depth + 1);
        int rightNode = build(X, rightIndices, gradients, hessians, depth + 1);

        // Store child indices after recursion; vector growth may invalidate references.
        nodes[nodeIndex].feature = bestFeature;
        nodes[nodeIndex].threshold = bestThreshold;
        nodes[nodeIndex].left = leftNode;
        nodes[nodeIndex].right = rightNode;
        return nodeIndex;
    }

public:
    HistogramTree(int bins = 8, int depth = 3, int leaf = 3, double regularization = 1)
        : maxBins(bins), maxDepth(depth), minLeaf(leaf), l2(regularization) {
        if (bins < 2 || depth < 0 || leaf < 1 || regularization < 0) {
            throw invalid_argument("Invalid tree configuration.");
        }
    }

    void fit(const vector<vector<double>>& X, const vector<double>& gradients,
             const vector<double>& hessians) {
        if (X.empty() || X.size() != gradients.size() ||
            X.size() != hessians.size() || X.front().empty()) {
            throw invalid_argument("Inconsistent or empty training arrays.");
        }
        size_t width = X.front().size();
        for (const auto& row : X) {
            if (row.size() != width) throw invalid_argument("Feature dimension mismatch.");
            for (double value : row) {
                if (!isfinite(value)) throw invalid_argument("Non-finite feature.");
            }
        }
        for (size_t i = 0; i < X.size(); ++i) {
            if (!isfinite(gradients[i]) || !isfinite(hessians[i]) || hessians[i] < 0) {
                throw invalid_argument("Invalid gradient or Hessian.");
            }
        }

        boundaries.assign(width, {});
        for (size_t feature = 0; feature < width; ++feature) {
            vector<double> values;
            for (const auto& row : X) values.push_back(row[feature]);
            sort(values.begin(), values.end());
            values.erase(unique(values.begin(), values.end()), values.end());

            if (static_cast<int>(values.size()) <= maxBins) {
                boundaries[feature].assign(values.begin(),
                    values.empty() ? values.end() : values.end() - 1);
            } else {
                set<double> edges;
                for (int i = 1; i < maxBins; ++i) {
                    size_t position = min(values.size() - 1,
                        static_cast<size_t>(i) * values.size() / maxBins);
                    edges.insert(values[position]);
                }
                boundaries[feature].assign(edges.begin(), edges.end());
            }
        }

        nodes.clear();
        vector<int> indices(X.size());
        iota(indices.begin(), indices.end(), 0);
        build(X, indices, gradients, hessians, 0);
    }

    double predictOne(const vector<double>& row) const {
        if (nodes.empty()) throw logic_error("Tree has not been fitted.");
        int nodeIndex = 0;
        while (!nodes[nodeIndex].isLeaf()) {
            const TreeNode& node = nodes[nodeIndex];
            if (row.size() != boundaries.size()) {
                throw invalid_argument("Prediction feature count mismatch.");
            }
            nodeIndex = row[node.feature] <= node.threshold ? node.left : node.right;
        }
        return nodes[nodeIndex].value;
    }
};

class BoostingClassifier {
    int iterations;
    double learningRate;
    int bins, depth, minLeaf;
    double l2;
    double initialScore = 0;
    vector<HistogramTree> trees;

public:
    BoostingClassifier(int rounds = 15, double rate = 0.1, int maxBins = 8,
                       int maxDepth = 3, int minimumLeaf = 3, double regularization = 1)
        : iterations(rounds), learningRate(rate), bins(maxBins),
          depth(maxDepth), minLeaf(minimumLeaf), l2(regularization) {
        if (rounds < 1 || rate <= 0 || rate > 1) {
            throw invalid_argument("Invalid boosting configuration.");
        }
    }

    void fit(const vector<vector<double>>& X, const vector<int>& y) {
        if (X.empty() || X.size() != y.size()) {
            throw invalid_argument("Inconsistent training arrays.");
        }
        if (any_of(y.begin(), y.end(), [](int value) { return value != 0 && value != 1; })) {
            throw invalid_argument("Labels must be binary.");
        }

        double rate = accumulate(y.begin(), y.end(), 0.0) / y.size();
        rate = clamp(rate, 1e-6, 1.0 - 1e-6);
        initialScore = log(rate / (1 - rate));
        vector<double> scores(X.size(), initialScore);
        trees.clear();

        for (int iteration = 0; iteration < iterations; ++iteration) {
            vector<double> gradients(X.size()), hessians(X.size());
            for (size_t i = 0; i < X.size(); ++i) {
                double p = sigmoid(scores[i]);
                gradients[i] = p - y[i];
                hessians[i] = max(1e-6, p * (1 - p));
            }
            HistogramTree tree(bins, depth, minLeaf, l2);
            tree.fit(X, gradients, hessians);
            for (size_t i = 0; i < X.size(); ++i) {
                scores[i] += learningRate * tree.predictOne(X[i]);
            }
            trees.push_back(move(tree));
        }
    }

    vector<double> predictProba(const vector<vector<double>>& X) const {
        if (trees.empty()) throw logic_error("Model has not been fitted.");
        vector<double> output;
        for (const auto& row : X) {
            double score = initialScore;
            for (const auto& tree : trees) score += learningRate * tree.predictOne(row);
            output.push_back(sigmoid(score));
        }
        return output;
    }
};

vector<Customer> makeCustomers() {
    const vector<string> regions = {"north", "south", "east", "west"};
    const vector<string> plans = {"basic", "standard", "premium"};
    vector<Customer> rows;
    for (int i = 0; i < 120; ++i) {
        double tenure = (i * 7) % 61;
        double charge = 25 + (i * 13) % 95;
        string region = regions[(i * 3) % regions.size()];
        string plan = plans[(i * 5) % plans.size()];
        int churn = (tenure < 12 && charge > 65)
                 || (plan == "basic" && charge > 90)
                 || (region == "south" && tenure < 6);
        rows.emplace_back(tenure, charge, region, plan, churn);
    }
    return rows;
}

int main() {
    try {
        auto customers = makeCustomers();
        OrderedTargetEncoder regionEncoder, planEncoder;
        auto regions = regionEncoder.fitTransform(customers, false);
        auto plans = planEncoder.fitTransform(customers, true);

        vector<vector<double>> X;
        vector<int> y;
        for (size_t i = 0; i < customers.size(); ++i) {
            X.push_back({customers[i].tenure, customers[i].monthlyCharge,
                         regions[i], plans[i]});
            y.push_back(customers[i].churned);
        }

        size_t split = X.size() * 4 / 5;
        vector<vector<double>> trainX(X.begin(), X.begin() + split);
        vector<vector<double>> testX(X.begin() + split, X.end());
        vector<int> trainY(y.begin(), y.begin() + split);
        vector<int> testY(y.begin() + split, y.end());

        BoostingClassifier model(15, 0.1, 8, 3, 3, 1.0);
        model.fit(trainX, trainY);
        auto probabilities = model.predictProba(testX);

        int correct = 0;
        for (size_t i = 0; i < testY.size(); ++i) {
            correct += ((probabilities[i] >= 0.5) == (testY[i] == 1));
        }

        cout << fixed << setprecision(4);
        cout << "Training observations: " << trainY.size() << '\n';
        cout << "Test observations: " << testY.size() << '\n';
        cout << "Test log loss: " << logLoss(testY, probabilities) << '\n';
        cout << "Test accuracy: " << static_cast<double>(correct) / testY.size() << '\n';
        for (size_t i = 0; i < min<size_t>(5, testY.size()); ++i) {
            cout << "Customer " << split + i << ": actual=" << testY[i]
                 << ", churn_probability=" << probabilities[i] << '\n';
        }
    } catch (const exception& error) {
        cerr << "Training failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
```

```java
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Enterprise-oriented demonstration of histogram gradient boosting and
 * ordered categorical statistics. Requires Java 17 or later.
 */
public class BoostingEnterpriseDemo {
    record Customer(double tenure, double monthlyCharge, String region,
                    String plan, int churned) {
        Customer {
            if (!Double.isFinite(tenure) || tenure < 0
                    || !Double.isFinite(monthlyCharge) || monthlyCharge < 0) {
                throw new IllegalArgumentException("Invalid numeric features.");
            }
            if (region == null || region.isBlank() || plan == null || plan.isBlank()) {
                throw new IllegalArgumentException("Categories cannot be empty.");
            }
            if (churned != 0 && churned != 1) {
                throw new IllegalArgumentException("Target must be binary.");
            }
        }
    }

    interface FeatureEncoder {
        double[] fitTransform(List<Customer> customers, String feature);
        double[] transform(List<String> categories, String feature);
    }

    static double sigmoid(double value) {
        if (value >= 0) {
            return 1.0 / (1.0 + Math.exp(-Math.min(value, 700)));
        }
        double z = Math.exp(Math.max(value, -700));
        return z / (1.0 + z);
    }

    static double logLoss(int[] labels, double[] probabilities) {
        if (labels.length == 0 || labels.length != probabilities.length) {
            throw new IllegalArgumentException("Invalid evaluation arrays.");
        }
        double total = 0;
        for (int i = 0; i < labels.length; i++) {
            double p = Math.max(1e-15, Math.min(1 - 1e-15, probabilities[i]));
            total -= labels[i] * Math.log(p) + (1 - labels[i]) * Math.log(1 - p);
        }
        return total / labels.length;
    }

    static final class OrderedCategoricalEncoder implements FeatureEncoder {
        private final double smoothing;
        private final long seed;
        private double globalMean = 0.5;
        private final Map<String, Map<String, CategoryState>> statistics = new HashMap<>();

        private record CategoryState(double sum, int count) {}

        OrderedCategoricalEncoder(double smoothing, long seed) {
            if (!(smoothing > 0)) {
                throw new IllegalArgumentException("Smoothing must be positive.");
            }
            this.smoothing = smoothing;
            this.seed = seed;
        }

        @Override
        public double[] fitTransform(List<Customer> customers, String feature) {
            if (customers.isEmpty() || !List.of("region", "plan").contains(feature)) {
                throw new IllegalArgumentException("Invalid dataset or categorical feature.");
            }
            globalMean = customers.stream().mapToInt(Customer::churned).average().orElse(0.5);

            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < customers.size(); i++) order.add(i);
            Collections.shuffle(order, new Random(seed));

            Map<String, CategoryState> state = new HashMap<>();
            double[] output = new double[customers.size()];
            for (int index : order) {
                Customer customer = customers.get(index);
                String category = feature.equals("region") ? customer.region() : customer.plan();
                CategoryState current = state.getOrDefault(category, new CategoryState(0, 0));

                output[index] = (current.sum() + smoothing * globalMean)
                        / (current.count() + smoothing);

                // Current label is not available to its own encoded feature.
                state.put(category, new CategoryState(
                        current.sum() + customer.churned(), current.count() + 1));
            }
            statistics.put(feature, Map.copyOf(state));
            return output;
        }

        @Override
        public double[] transform(List<String> categories, String feature) {
            Map<String, CategoryState> state = statistics.getOrDefault(feature, Map.of());
            double[] output = new double[categories.size()];
            for (int i = 0; i < categories.size(); i++) {
                CategoryState current = state.getOrDefault(
                        categories.get(i), new CategoryState(0, 0));
                output[i] = (current.sum() + smoothing * globalMean)
                        / (current.count() + smoothing);
            }
            return output;
        }
    }

    static final class TreeNode {
        double value;
        int feature = -1;
        double threshold;
        TreeNode left;
        TreeNode right;

        boolean isLeaf() {
            return left == null;
        }
    }

    static final class HistogramTree {
        private final int maxBins;
        private final int maxDepth;
        private final int minLeaf;
        private final double l2;
        private double[][] boundaries;
        private TreeNode root;

        HistogramTree(int maxBins, int maxDepth, int minLeaf, double l2) {
            if (maxBins < 2 || maxDepth < 0 || minLeaf < 1 || l2 < 0) {
                throw new IllegalArgumentException("Invalid tree configuration.");
            }
            this.maxBins = maxBins;
            this.maxDepth = maxDepth;
            this.minLeaf = minLeaf;
            this.l2 = l2;
        }

        void fit(double[][] X, double[] gradients, double[] hessians) {
            if (X.length == 0 || X.length != gradients.length
                    || X.length != hessians.length || X[0].length == 0) {
                throw new IllegalArgumentException("Inconsistent training arrays.");
            }
            int width = X[0].length;
            for (double[] row : X) {
                if (row.length != width) throw new IllegalArgumentException("Feature width mismatch.");
                for (double value : row) {
                    if (!Double.isFinite(value)) throw new IllegalArgumentException("Non-finite feature.");
                }
            }
            for (int i = 0; i < X.length; i++) {
                if (!Double.isFinite(gradients[i]) || !Double.isFinite(hessians[i])
                        || hessians[i] < 0) {
                    throw new IllegalArgumentException("Invalid gradient or Hessian.");
                }
            }

            boundaries = new double[width][];
            for (int feature = 0; feature < width; feature++) {
                double[] values = new double[X.length];
                for (int i = 0; i < X.length; i++) values[i] = X[i][feature];
                java.util.Arrays.sort(values);

                List<Double> unique = new ArrayList<>();
                for (double value : values) {
                    if (unique.isEmpty() || value != unique.get(unique.size() - 1)) {
                        unique.add(value);
                    }
                }

                List<Double> edges = new ArrayList<>();
                if (unique.size() <= maxBins) {
                    edges.addAll(unique.subList(0, Math.max(0, unique.size() - 1)));
                } else {
                    for (int bin = 1; bin < maxBins; bin++) {
                        int position = Math.min(unique.size() - 1,
                                bin * unique.size() / maxBins);
                        double edge = unique.get(position);
                        if (!edges.contains(edge)) edges.add(edge);
                    }
                    Collections.sort(edges);
                }
                boundaries[feature] = edges.stream().mapToDouble(Double::doubleValue).toArray();
            }

            int[] indices = new int[X.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            root = build(X, indices, gradients, hessians, 0);
        }

        private double leafValue(int[] indices, double[] gradients, double[] hessians) {
            double g = 0, h = 0;
            for (int i : indices) {
                g += gradients[i];
                h += hessians[i];
            }
            return -g / (h + l2 + 1e-15);
        }

        private TreeNode build(double[][] X, int[] indices, double[] gradients,
                               double[] hessians, int depth) {
            TreeNode node = new TreeNode();
            node.value = leafValue(indices, gradients, hessians);
            if (depth >= maxDepth || indices.length < 2 * minLeaf) return node;

            double parentG = 0, parentH = 0;
            for (int i : indices) {
                parentG += gradients[i];
                parentH += hessians[i];
            }
            double parentScore = parentG * parentG / (parentH + l2 + 1e-15);
            double bestGain = 1e-8;
            int bestFeature = -1;
            double bestThreshold = 0;

            for (int feature = 0; feature < boundaries.length; feature++) {
                for (double threshold : boundaries[feature]) {
                    double lg = 0, lh = 0;
                    int leftCount = 0;
                    for (int i : indices) {
                        if (X[i][feature] <= threshold) {
                            lg += gradients[i];
                            lh += hessians[i];
                            leftCount++;
                        }
                    }
                    int rightCount = indices.length - leftCount;
                    if (leftCount < minLeaf || rightCount < minLeaf) continue;

                    double rg = parentG - lg;
                    double rh = parentH - lh;
                    double gain = lg * lg / (lh + l2 + 1e-15)
                            + rg * rg / (rh + l2 + 1e-15) - parentScore;
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = threshold;
                    }
                }
            }

            if (bestFeature < 0) return node;
            List<Integer> leftList = new ArrayList<>();
            List<Integer> rightList = new ArrayList<>();
            for (int i : indices) {
                (X[i][bestFeature] <= bestThreshold ? leftList : rightList).add(i);
            }
            node.feature = bestFeature;
            node.threshold = bestThreshold;
            node.left = build(X, leftList.stream().mapToInt(Integer::intValue).toArray(),
                    gradients, hessians, depth + 1);
            node.right = build(X, rightList.stream().mapToInt(Integer::intValue).toArray(),
                    gradients, hessians, depth + 1);
            return node;
        }

        double predictOne(double[] row) {
            if (root == null) throw new IllegalStateException("Tree has not been fitted.");
            if (row.length != boundaries.length) {
                throw new IllegalArgumentException("Prediction feature count mismatch.");
            }
            TreeNode node = root;
            while (!node.isLeaf()) {
                node = row[node.feature] <= node.threshold ? node.left : node.right;
            }
            return node.value;
        }
    }

    static final class BoostingModel {
        private final int iterations;
        private final double learningRate;
        private final int bins, depth, minLeaf;
        private final double l2;
        private double initialScore;
        private List<HistogramTree> trees = new ArrayList<>();

        BoostingModel(int iterations, double learningRate, int bins,
                      int depth, int minLeaf, double l2) {
            if (iterations < 1 || !(learningRate > 0 && learningRate <= 1)) {
                throw new IllegalArgumentException("Invalid boosting configuration.");
            }
            this.iterations = iterations;
            this.learningRate = learningRate;
            this.bins = bins;
            this.depth = depth;
            this.minLeaf = minLeaf;
            this.l2 = l2;
        }

        void fit(double[][] X, int[] y) {
            if (X.length == 0 || X.length != y.length) {
                throw new IllegalArgumentException("Invalid training dimensions.");
            }
            for (int label : y) {
                if (label != 0 && label != 1) {
                    throw new IllegalArgumentException("Labels must be binary.");
                }
            }
            double rate = Math.max(1e-6, Math.min(1 - 1e-6,
                    java.util.Arrays.stream(y).average().orElseThrow()));
            initialScore = Math.log(rate / (1 - rate));
            double[] scores = new double[y.length];
            java.util.Arrays.fill(scores, initialScore);
            trees = new ArrayList<>();

            for (int iteration = 0; iteration < iterations; iteration++) {
                double[] gradients = new double[y.length];
                double[] hessians = new double[y.length];
                for (int i = 0; i < y.length; i++) {
                    double p = sigmoid(scores[i]);
                    gradients[i] = p - y[i];
                    hessians[i] = Math.max(1e-6, p * (1 - p));
                }

                HistogramTree tree = new HistogramTree(bins, depth, minLeaf, l2);
                tree.fit(X, gradients, hessians);
                trees.add(tree);
                for (int i = 0; i < y.length; i++) {
                    scores[i] += learningRate * tree.predictOne(X[i]);
                }
            }
        }

        double[] predictProba(double[][] X) {
            if (trees.isEmpty()) throw new IllegalStateException("Model has not been fitted.");
            double[] output = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                double score = initialScore;
                for (HistogramTree tree : trees) {
                    score += learningRate * tree.predictOne(X[i]);
                }
                output[i] = sigmoid(score);
            }
            return output;
        }
    }

    static List<Customer> makeCustomers() {
        String[] regions = {"north", "south", "east", "west"};
        String[] plans = {"basic", "standard", "premium"};
        List<Customer> customers = new ArrayList<>();
        for (int i = 0; i < 120; i++) {
            double tenure = (i * 7) % 61;
            double charge = 25 + (i * 13) % 95;
            String region = regions[(i * 3) % regions.length];
            String plan = plans[(i * 5) % plans.length];
            int churn = (tenure < 12 && charge > 65)
                    || (plan.equals("basic") && charge > 90)
                    || (region.equals("south") && tenure < 6) ? 1 : 0;
            customers.add(new Customer(tenure, charge, region, plan, churn));
        }
        return customers;
    }

    public static void main(String[] args) {
        List<Customer> customers = makeCustomers();
        OrderedCategoricalEncoder regionEncoder = new OrderedCategoricalEncoder(5, 17);
        OrderedCategoricalEncoder planEncoder = new OrderedCategoricalEncoder(5, 23);
        double[] regions = regionEncoder.fitTransform(customers, "region");
        double[] plans = planEncoder.fitTransform(customers, "plan");

        double[][] X = new double[customers.size()][4];
        int[] y = new int[customers.size()];
        for (int i = 0; i < customers.size(); i++) {
            Customer customer = customers.get(i);
            X[i] = new double[] {
                    customer.tenure(), customer.monthlyCharge(), regions[i], plans[i]
            };
            y[i] = customer.churned();
        }

        int split = X.length * 4 / 5;
        double[][] trainX = java.util.Arrays.copyOfRange(X, 0, split);
        double[][] testX = java.util.Arrays.copyOfRange(X, split, X.length);
        int[] trainY = java.util.Arrays.copyOfRange(y, 0, split);
        int[] testY = java.util.Arrays.copyOfRange(y, split, y.length);

        BoostingModel model = new BoostingModel(15, 0.1, 8, 3, 3, 1.0);
        model.fit(trainX, trainY);
        double[] probabilities = model.predictProba(testX);

        int correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if ((probabilities[i] >= 0.5) == (testY[i] == 1)) correct++;
        }

        System.out.printf("Training observations: %d%n", trainY.length);
        System.out.printf("Test observations: %d%n", testY.length);
        System.out.printf("Test log loss: %.4f%n", logLoss(testY, probabilities));
        System.out.printf("Test accuracy: %.3f%n", (double) correct / testY.length);
        for (int i = 0; i < Math.min(5, testY.length); i++) {
            System.out.printf("Customer %d: actual=%d, churn_probability=%.3f%n",
                    split + i, testY[i], probabilities[i]);
        }
    }
}
```

```sql
BEGIN;

DROP VIEW IF EXISTS customer_churn_summary CASCADE;
DROP VIEW IF EXISTS experiment_evaluation_report CASCADE;
DROP TABLE IF EXISTS model_evaluation CASCADE;
DROP TABLE IF EXISTS tree_split CASCADE;
DROP TABLE IF EXISTS boosting_tree CASCADE;
DROP TABLE IF EXISTS categorical_statistic CASCADE;
DROP TABLE IF EXISTS boosting_experiment CASCADE;
DROP TABLE IF EXISTS customer_observation CASCADE;
DROP TABLE IF EXISTS dataset CASCADE;

CREATE TABLE dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    task_type TEXT NOT NULL CHECK (task_type IN ('binary_classification')),
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_observation (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES dataset(dataset_id) ON DELETE CASCADE,
    tenure_months INTEGER NOT NULL CHECK (tenure_months >= 0),
    monthly_charge NUMERIC(10,2) NOT NULL CHECK (monthly_charge >= 0),
    region TEXT NOT NULL CHECK (length(trim(region)) > 0),
    plan TEXT NOT NULL CHECK (length(trim(plan)) > 0),
    churned SMALLINT NOT NULL CHECK (churned IN (0, 1)),
    data_partition TEXT NOT NULL DEFAULT 'train'
        CHECK (data_partition IN ('train', 'validation', 'test')),
    UNIQUE (dataset_id, observation_id)
);

CREATE INDEX customer_observation_dataset_target_idx
    ON customer_observation(dataset_id, churned);

CREATE INDEX customer_observation_region_plan_idx
    ON customer_observation(dataset_id, region, plan);

CREATE INDEX customer_observation_partition_idx
    ON customer_observation(dataset_id, data_partition);

CREATE TABLE boosting_experiment (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES dataset(dataset_id),
    model_family TEXT NOT NULL CHECK (model_family IN ('lightgbm_style', 'catboost_style')),
    learning_rate NUMERIC(8,6) NOT NULL
        CHECK (learning_rate > 0 AND learning_rate <= 1),
    iterations INTEGER NOT NULL CHECK (iterations > 0),
    max_depth INTEGER NOT NULL CHECK (max_depth BETWEEN 0 AND 32),
    max_bins INTEGER NOT NULL CHECK (max_bins BETWEEN 2 AND 65536),
    min_leaf_size INTEGER NOT NULL CHECK (min_leaf_size > 0),
    l2_regularization NUMERIC(12,6) NOT NULL CHECK (l2_regularization >= 0),
    smoothing_alpha NUMERIC(12,6) NOT NULL DEFAULT 5
        CHECK (smoothing_alpha > 0),
    random_seed INTEGER NOT NULL DEFAULT 17,
    experiment_status TEXT NOT NULL DEFAULT 'configured'
        CHECK (experiment_status IN ('configured', 'running', 'completed', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX boosting_experiment_dataset_family_idx
    ON boosting_experiment(dataset_id, model_family, created_at DESC);

CREATE TABLE categorical_statistic (
    statistic_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES boosting_experiment(experiment_id)
        ON DELETE CASCADE,
    feature_name TEXT NOT NULL CHECK (feature_name IN ('region', 'plan')),
    category_value TEXT NOT NULL,
    permutation_position INTEGER NOT NULL CHECK (permutation_position >= 0),
    prior_count INTEGER NOT NULL CHECK (prior_count >= 0),
    prior_target_sum NUMERIC(14,6) NOT NULL
        CHECK (prior_target_sum >= 0 AND prior_target_sum <= prior_count),
    global_target_mean NUMERIC(10,8) NOT NULL
        CHECK (global_target_mean BETWEEN 0 AND 1),
    smoothing_alpha NUMERIC(12,6) NOT NULL CHECK (smoothing_alpha > 0),
    encoded_value NUMERIC(12,8) NOT NULL CHECK (encoded_value BETWEEN 0 AND 1),
    UNIQUE (experiment_id, feature_name, permutation_position)
);

CREATE INDEX categorical_statistic_lookup_idx
    ON categorical_statistic(experiment_id, feature_name, category_value);

CREATE TABLE boosting_tree (
    tree_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES boosting_experiment(experiment_id)
        ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL CHECK (iteration_number > 0),
    tree_depth INTEGER NOT NULL CHECK (tree_depth >= 0),
    leaf_count INTEGER NOT NULL CHECK (leaf_count > 0),
    training_rows INTEGER NOT NULL CHECK (training_rows > 0),
    UNIQUE (experiment_id, iteration_number)
);

CREATE TABLE tree_split (
    split_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tree_id BIGINT NOT NULL REFERENCES boosting_tree(tree_id) ON DELETE CASCADE,
    node_number INTEGER NOT NULL CHECK (node_number >= 0),
    feature_name TEXT NOT NULL CHECK (
        feature_name IN ('tenure_months', 'monthly_charge', 'region_stat', 'plan_stat')
    ),
    threshold_value NUMERIC(16,6) NOT NULL,
    split_gain NUMERIC(16,8) NOT NULL CHECK (split_gain >= 0),
    left_rows INTEGER NOT NULL CHECK (left_rows >= 0),
    right_rows INTEGER NOT NULL CHECK (right_rows >= 0),
    UNIQUE (tree_id, node_number),
    CHECK (left_rows + right_rows > 0)
);

CREATE INDEX tree_split_feature_gain_idx
    ON tree_split(feature_name, split_gain DESC);

CREATE TABLE model_evaluation (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES boosting_experiment(experiment_id)
        ON DELETE CASCADE,
    data_partition TEXT NOT NULL
        CHECK (data_partition IN ('train', 'validation', 'test')),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    log_loss NUMERIC(12,8) NOT NULL CHECK (log_loss >= 0),
    accuracy NUMERIC(8,7) NOT NULL CHECK (accuracy BETWEEN 0 AND 1),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, data_partition)
);

CREATE OR REPLACE FUNCTION calculate_ordered_category_statistics(
    p_experiment_id BIGINT,
    p_feature_name TEXT
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_dataset_id BIGINT;
    v_alpha NUMERIC;
    v_global_mean NUMERIC;
    v_inserted INTEGER;
BEGIN
    IF p_feature_name NOT IN ('region', 'plan') THEN
        RAISE EXCEPTION 'Unsupported categorical feature: %', p_feature_name;
    END IF;

    SELECT e.dataset_id, e.smoothing_alpha
    INTO v_dataset_id, v_alpha
    FROM boosting_experiment e
    WHERE e.experiment_id = p_experiment_id
      AND e.experiment_status IN ('configured', 'running');

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Experiment % does not exist or cannot be updated',
            p_experiment_id;
    END IF;

    SELECT AVG(churned::NUMERIC)
    INTO v_global_mean
    FROM customer_observation
    WHERE dataset_id = v_dataset_id
      AND data_partition = 'train';

    IF v_global_mean IS NULL THEN
        RAISE EXCEPTION 'Training partition is empty for dataset %', v_dataset_id;
    END IF;

    -- ROWS ... 1 PRECEDING excludes the current row's target.
    -- observation_id provides a deterministic demonstration order.
    WITH ordered AS (
        SELECT
            observation_id,
            CASE WHEN p_feature_name = 'region' THEN region ELSE plan END AS category,
            churned,
            COALESCE(
                SUM(churned) OVER (
                    PARTITION BY
                        CASE WHEN p_feature_name = 'region' THEN region ELSE plan END
                    ORDER BY observation_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0
            ) AS prior_sum,
            COUNT(*) OVER (
                PARTITION BY
                    CASE WHEN p_feature_name = 'region' THEN region ELSE plan END
                ORDER BY observation_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS prior_count,
            ROW_NUMBER() OVER (ORDER BY observation_id) - 1 AS permutation_position
        FROM customer_observation
        WHERE dataset_id = v_dataset_id
          AND data_partition = 'train'
    ),
    encoded AS (
        SELECT
            observation_id,
            category,
            permutation_position,
            prior_sum,
            prior_count,
            (
                prior_sum + v_alpha * v_global_mean
            ) / (prior_count + v_alpha) AS encoded_value
        FROM ordered
    )
    INSERT INTO categorical_statistic (
        experiment_id, feature_name, category_value, permutation_position,
        prior_count, prior_target_sum, global_target_mean, smoothing_alpha,
        encoded_value
    )
    SELECT
        p_experiment_id, p_feature_name, category, permutation_position,
        prior_count, prior_sum, v_global_mean, v_alpha, encoded_value
    FROM encoded
    ON CONFLICT (experiment_id, feature_name, permutation_position)
    DO UPDATE SET
        category_value = EXCLUDED.category_value,
        prior_count = EXCLUDED.prior_count,
        prior_target_sum = EXCLUDED.prior_target_sum,
        global_target_mean = EXCLUDED.global_target_mean,
        smoothing_alpha = EXCLUDED.smoothing_alpha,
        encoded_value = EXCLUDED.encoded_value;

    GET DIAGNOSTICS v_inserted = ROW_COUNT;
    RETURN v_inserted;
END;
$$;

CREATE OR REPLACE VIEW customer_churn_summary AS
SELECT
    dataset_id,
    region,
    plan,
    data_partition,
    COUNT(*) AS customers,
    AVG(churned::NUMERIC)::NUMERIC(8,5) AS observed_churn_rate,
    AVG(tenure_months)::NUMERIC(10,2) AS average_tenure_months,
    AVG(monthly_charge)::NUMERIC(10,2) AS average_monthly_charge
FROM customer_observation
GROUP BY dataset_id, region, plan, data_partition;

CREATE OR REPLACE VIEW experiment_evaluation_report AS
SELECT
    e.experiment_id,
    d.dataset_name,
    e.model_family,
    e.learning_rate,
    e.iterations,
    e.max_depth,
    e.max_bins,
    e.experiment_status,
    m.data_partition,
    m.sample_count,
    m.log_loss,
    m.accuracy,
    m.evaluated_at
FROM boosting_experiment e
JOIN dataset d ON d.dataset_id = e.dataset_id
LEFT JOIN model_evaluation m ON m.experiment_id = e.experiment_id;

INSERT INTO dataset (dataset_name, task_type, row_count)
VALUES ('customer_churn_demo', 'binary_classification', 12);

INSERT INTO customer_observation
    (dataset_id, tenure_months, monthly_charge, region, plan, churned, data_partition)
VALUES
    (1, 2, 89.00, 'south', 'basic', 1, 'train'),
    (1, 8, 72.00, 'north', 'standard', 1, 'train'),
    (1, 24, 45.00, 'east', 'premium', 0, 'train'),
    (1, 4, 105.00, 'west', 'basic', 1, 'train'),
    (1, 36, 39.00, 'north', 'premium', 0, 'train'),
    (1, 10, 84.00, 'south', 'standard', 1, 'train'),
    (1, 48, 65.00, 'east', 'standard', 0, 'train'),
    (1, 6, 96.00, 'west', 'basic', 1, 'train'),
    (1, 18, 52.00, 'north', 'premium', 0, 'train'),
    (1, 15, 73.00, 'south', 'standard', 0, 'validation'),
    (1, 30, 58.00, 'east', 'premium', 0, 'test'),
    (1, 3, 110.00, 'west', 'basic', 1, 'test');

INSERT INTO boosting_experiment (
    dataset_id, model_family, learning_rate, iterations, max_depth,
    max_bins, min_leaf_size, l2_regularization, smoothing_alpha
)
VALUES
    (1, 'lightgbm_style', 0.100000, 50, 6, 64, 10, 1.000000, 5.000000),
    (1, 'catboost_style', 0.050000, 80, 6, 64, 10, 3.000000, 5.000000);

INSERT INTO boosting_tree (
    experiment_id, iteration_number, tree_depth, leaf_count, training_rows
)
VALUES
    (1, 1, 2, 4, 9),
    (1, 2, 3, 6, 9),
    (2, 1, 2, 4, 9);

INSERT INTO tree_split (
    tree_id, node_number, feature_name, threshold_value,
    split_gain, left_rows, right_rows
)
VALUES
    (1, 0, 'tenure_months', 12, 2.41000000, 4, 5),
    (1, 1, 'monthly_charge', 75, 1.22000000, 2, 2),
    (2, 0, 'region_stat', 0.500000, 1.83000000, 3, 6),
    (3, 0, 'plan_stat', 0.460000, 1.57000000, 4, 5);

INSERT INTO model_evaluation (
    experiment_id, data_partition, sample_count, log_loss, accuracy
)
VALUES
    (1, 'train', 9, 0.28000000, 0.8888889),
    (1, 'validation', 1, 0.41000000, 1.0000000),
    (1, 'test', 2, 0.32000000, 1.0000000),
    (2, 'train', 9, 0.25000000, 0.8888889),
    (2, 'validation', 1, 0.39000000, 1.0000000),
    (2, 'test', 2, 0.30000000, 1.0000000);

-- Run each encoder in a transaction-compatible workflow.
SELECT calculate_ordered_category_statistics(2, 'region');
SELECT calculate_ordered_category_statistics(2, 'plan');

-- Aggregate observed churn rates by category and partition.
SELECT *
FROM customer_churn_summary
WHERE dataset_id = 1
ORDER BY data_partition, region, plan;

-- Compare evaluation metrics by model family and data partition.
SELECT *
FROM experiment_evaluation_report
ORDER BY model_family, data_partition;

-- Inspect category encodings without including the current observation's target.
SELECT
    feature_name,
    category_value,
    permutation_position,
    prior_count,
    prior_target_sum,
    global_target_mean,
    encoded_value
FROM categorical_statistic
WHERE experiment_id = 2
ORDER BY feature_name, permutation_position;

-- Validate split partition accounting.
SELECT
    tree_id,
    node_number,
    left_rows,
    right_rows,
    left_rows + right_rows AS partitioned_rows,
    split_gain
FROM tree_split
ORDER BY tree_id, node_number;

-- Invalid examples are intentionally not executed because constraints reject them:
-- INSERT INTO customer_observation
-- (dataset_id, tenure_months, monthly_charge, region, plan, churned)
-- VALUES (1, -1, 40, 'north', 'basic', 2);
--
-- INSERT INTO boosting_experiment
-- (dataset_id, model_family, learning_rate, iterations, max_depth,
--  max_bins, min_leaf_size, l2_regularization)
-- VALUES (1, 'lightgbm_style', 1.5, 10, 4, 32, 2, 1);

COMMIT;
```

```markdown
# LightGBM and CatBoost: Histogram Boosting and Categorical Features

## Project scope

This project examines two efficient gradient-boosting design families through a binary customer-churn prediction scenario. The implementations focus on numerical split search, second-order optimization, categorical target statistics, model evaluation, and experiment tracking.

The distinction between the model families is important:

| Mechanism | LightGBM-style approach | CatBoost-style approach |
|---|---|---|
| Numerical split search | Histogram-based candidate thresholds | Numerical quantization is also used in production CatBoost |
| Tree construction | Leaf-wise growth is a characteristic of LightGBM | Production CatBoost uses its own structured boosting and tree-building methods |
| Categorical representation | Depends on the selected preprocessing and categorical handling | Ordered categorical statistics and other categorical transformations |
| Leakage control | Depends on the complete preprocessing workflow | Ordered statistics help prevent a row's target from leaking into its own encoding |
| Efficiency objective | Reduce split-search cost through binning and optimized training | Handle categorical features effectively while maintaining efficient boosting |
| Important risk | Excessive tree complexity and overfitting | Target leakage, high category cardinality, and category drift |

The source files provide compact educational models. They do not reproduce the full production implementations of LightGBM or CatBoost.

## Gradient boosting for binary classification

Gradient boosting builds an additive prediction function:

`F(x) = F₀(x) + ηT₁(x) + ηT₂(x) + ... + ηTₘ(x)`

Here, `F₀(x)` is the initial score, `Tₘ(x)` is a decision tree, `η` is the learning rate, and `m` is the boosting iteration.

For binary classification, the score is converted into a probability through the logistic sigmoid:

`p = 1 / (1 + exp(-F(x)))`

The initial score is the log-odds of the positive-class frequency. Each subsequent tree attempts to improve the current model by fitting the gradients of the logistic objective.

For an observation with target `y` and predicted probability `p`, the first- and second-order terms are:

- Gradient: `g = p - y`
- Hessian: `h = p(1 - p)`

A regularized leaf value is approximated by:

`leaf_value = -Σg / (Σh + λ)`

The regularization term `λ` discourages excessively large leaf updates. The learning rate scales each tree's contribution to the ensemble.

## Histogram boosting

### Why binning is useful

A numerical feature can contain many distinct values. An exact split search may need to examine many thresholds, increasing training cost as dataset size and feature cardinality grow.

Histogram boosting groups numerical values into a bounded number of bins. The learner evaluates candidate splits using aggregated gradient and Hessian information rather than repeatedly considering every distinct raw value.

The conceptual workflow is:

Raw feature values → quantization boundaries → histogram statistics → candidate split evaluation → tree construction.

The implementations create feature-specific candidate boundaries from the training data. The number of bins controls the granularity of the search.

| Configuration choice | Benefit | Trade-off |
|---|---|---|
| Fewer bins | Smaller candidate search and less histogram state | Coarser numerical thresholds |
| More bins | More precise candidate boundaries | Greater computational and memory cost |
| Shallow trees | Lower model complexity and reduced overfitting risk | May miss complex interactions |
| More boosting iterations | More opportunities to correct residual errors | Longer training and greater model capacity |
| Stronger L2 regularization | More conservative leaf updates | May limit useful corrections |

The supplied histogram learners use constrained recursive trees. Their split-search loops prioritize clarity over the performance optimizations found in production libraries.

## LightGBM-style implementation

The Python, JavaScript, C++, and Java implementations construct a compact histogram-based logistic boosting model.

Each boosting iteration follows this sequence:

1. Convert current scores into probabilities.
2. Calculate the gradients and Hessians of the binary logistic objective.
3. Build numerical split candidates for each feature.
4. Evaluate candidate partitions using gradient and Hessian aggregates.
5. Reject partitions that violate the minimum leaf-size constraint.
6. Select the split with the largest positive gain.
7. Calculate regularized leaf values and update the additive scores.

The implementation limits tree depth, minimum leaf size, and histogram resolution. These controls help constrain model complexity and computation.

### Tree growth and LightGBM terminology

Leaf-wise growth is a characteristic of LightGBM. Instead of expanding every node at the same level, a leaf-wise learner can expand the leaf with the strongest available objective improvement.

This can yield effective trees with fewer nodes, but it can also produce highly specialized leaves. Production models therefore require suitable depth or leaf-count limits, minimum observations per leaf, regularization, validation monitoring, and early stopping where appropriate.

The supplied implementations use depth-constrained recursive growth. They are LightGBM-style educational models, not exact implementations of LightGBM's leaf-wise training engine.

## CatBoost-style categorical processing

### Why ordinary integer encoding can be misleading

A categorical variable such as `region` represents labels rather than ordered numerical quantities. Assigning integers to categories can accidentally imply a numerical ordering that has no semantic meaning.

For example, encoding north as `0`, south as `1`, east as `2`, and west as `3` does not establish that west is greater than north.

Target statistics offer an alternative representation. They summarize the relationship between a category and the target, but naïve target encoding introduces leakage when a row's own target contributes to its encoded feature.

### Ordered target statistics

The categorical encoders maintain a deterministic pseudo-random ordering of training observations. For each observation, the category statistic is calculated from preceding observations in that ordering.

The smoothed estimate is:

`encoded_value = (prior_sum + α × global_mean) / (prior_count + α)`

Where:

- `prior_sum` is the sum of preceding target values for the category.
- `prior_count` is the number of preceding observations with that category.
- `global_mean` is the training target rate.
- `α` is the smoothing strength.

The current target is added to the category state only after the current encoding is computed.

This ordering reduces direct self-target leakage. Smoothing also limits extreme estimates for categories with little historical evidence.

For example, if a category has not appeared previously in the permutation, its statistic equals the smoothed global prior rather than an estimate based on a single observation.

### Training and inference are different operations

During ordered training encoding, each observation receives a representation based on preceding labels. At prediction time, labels for new observations are unavailable. The encoder therefore uses the stored training category statistics and global prior.

An unseen category falls back to the global prior. This makes prediction possible for new categories, although a large increase in unseen categories can indicate category drift or an upstream data-quality problem.

The educational encoders demonstrate ordered statistics, not the complete CatBoost training algorithm. Production CatBoost includes additional ordered boosting, quantization, categorical transformations, permutation handling, missing-value processing, and inference optimizations.

## Python implementation

The Python source defines an immutable `Customer` record, an `OrderedTargetEncoder`, a `HistogramTree`, and a `LightGBMStyleClassifier`.

The `Customer` record validates nonnegative tenure and monthly charges, nonempty category labels, and a binary target. The encoder stores category target sums and counts and generates ordered representations using a seeded permutation.

`HistogramTree` builds numerical candidate boundaries, evaluates split gains using gradient and Hessian aggregates, and constructs a depth-constrained tree. The classifier initializes raw scores from the training class frequency, fits successive trees, and accumulates their contributions.

The example creates a deterministic synthetic churn dataset with numeric and categorical features. It reserves the final portion for a holdout evaluation and reports binary log loss, accuracy, and individual predicted probabilities.

The script uses only Python's standard library.

## JavaScript implementation

The JavaScript source uses Node.js classes and `Map` collections to represent customer records and categorical state.

`OrderedTargetEncoder` creates a seeded ordering, maintains category sums and counts, and updates state only after encoding each row. `HistogramTree` discovers candidate boundaries and evaluates candidate partitions. `HistogramBoostingClassifier` manages the additive score updates across boosting iterations.

The implementation validates dimensions, finite feature values, binary targets, tree parameters, and evaluation inputs. Its deterministic synthetic dataset provides a repeatable example without third-party packages.

Run the source with Node.js 18 or later.

## C++ implementation

The C++17 source emphasizes explicit data structures and computational control.

`Customer` validates each observation. `OrderedTargetEncoder` stores category statistics in an ordered map. `HistogramTree` represents nodes using a vector of `TreeNode` structures, where child indices refer to other elements in the same vector.

The array-oriented node representation avoids maintaining a separate heap allocation for every node. Recursive construction stores node indices and writes child references after recursion, so vector reallocation cannot invalidate a node reference.

The tree validates training dimensions and numerical inputs before constructing boundaries. `BoostingClassifier` calculates gradients and Hessians and accumulates each fitted tree's contribution. The application boundary catches exceptions and reports training failures.

Compile with a C++17-compatible compiler. The implementation uses standard-library containers and algorithms.

## Java implementation

The Java 17 source separates domain modeling, feature preparation, tree construction, and ensemble training.

| Component | Responsibility |
|---|---|
| `Customer` record | Immutable observation and input validation |
| `FeatureEncoder` interface | Contract for fitting and applying feature encodings |
| `OrderedCategoricalEncoder` | Ordered category statistics and smoothing |
| `TreeNode` | Split information, leaf value, and child references |
| `HistogramTree` | Histogram boundaries, split evaluation, and tree prediction |
| `BoostingModel` | Initial score, training iterations, and probability prediction |

Records represent observations and category statistics. The encoder uses immutable snapshots of category state after training. Lists and maps manage the training order and categorical statistics, while arrays provide direct access to numerical feature and gradient values.

The model validates the binary target and training dimensions. The example uses separate training and test partitions and reports log loss and accuracy.

Save the source as `BoostingEnterpriseDemo.java`, then compile and run it with Java 17 or later.

## PostgreSQL experiment model

The SQL source represents model development as a relational experiment-tracking workflow.

| Table | Purpose |
|---|---|
| `dataset` | Dataset identity and task type |
| `customer_observation` | Numerical features, categorical features, target, and data partition |
| `boosting_experiment` | Model family and hyperparameters |
| `categorical_statistic` | Ordered categorical statistics |
| `boosting_tree` | Boosting iteration metadata |
| `tree_split` | Split thresholds, feature names, gains, and partition sizes |
| `model_evaluation` | Log loss and accuracy by data partition |

Primary keys identify records, foreign keys maintain relationships, and check constraints reject invalid targets, negative measurements, unsupported model families, and invalid hyperparameters.

Indexes support dataset-target filtering, category-based grouping, partition selection, and experiment reporting. The views aggregate observed churn patterns and combine model configurations with evaluation metrics.

### Ordered statistics in SQL

The `calculate_ordered_category_statistics` function uses PostgreSQL window functions to compute the target sum and count from preceding observations within each category.

The frame `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` excludes the current row. The ordering by observation identifier makes this demonstration deterministic.

The function records the prior count, prior target sum, global target mean, smoothing strength, and encoded value. It rejects unsupported features, missing experiments, and datasets without training observations.

The supplied SQL data includes training, validation, and test partitions, two model-family configurations, tree metadata, split records, and example evaluation results.

The evaluation values are illustrative records for demonstrating the schema and reporting workflow. They are not measured results produced by executing the Python, JavaScript, C++, or Java models.

Run the script in PostgreSQL. Its transaction groups schema creation, sample data, statistic generation, and reporting queries. The initial `DROP ... CASCADE` statements make the demonstration repeatable but also remove existing objects with these names and their dependent objects, so use a dedicated database or schema.

## Evaluation metrics

Binary log loss measures the quality of predicted probabilities:

`loss = -(y × log(p) + (1 - y) × log(1 - p))`

A lower log loss indicates better probabilistic predictions. Predictions close to zero or one are penalized heavily when they are confidently wrong.

Accuracy measures the fraction of classifications that match the target after applying a threshold, which is `0.5` in these examples. Accuracy does not measure probability calibration and can be misleading when the positive class is rare.

Reliable model comparison should use validation data to select hyperparameters and an untouched test set for final assessment. The small synthetic examples illustrate training mechanics, not production performance or generalization guarantees.

## Edge cases and operational considerations

- **Empty data:** Training and encoding reject empty datasets.
- **Dimension mismatch:** Feature, gradient, Hessian, and target lengths must agree.
- **Invalid targets:** Binary classifiers accept only targets in `{0, 1}`.
- **Non-finite values:** Numerical inputs are validated before histogram construction.
- **Rare categories:** Smoothing prevents extreme estimates from small category counts.
- **Unknown categories:** Inference falls back toward the global target prior.
- **Histogram resolution:** Fewer bins reduce candidate thresholds but can lose numerical detail.
- **Tree complexity:** Depth and minimum leaf size constrain recursive growth.
- **Numerical stability:** Sigmoid and log-loss calculations use bounded exponentials or probability clipping.
- **Target leakage:** Ordered encoding excludes the current row's target from its own statistic.
- **Data partitioning:** Validation and test targets must not enter training-time category statistics.
- **Category drift:** Production systems should monitor unseen categories and changes in category frequencies.
- **Schema consistency:** Training and inference must use the same feature definitions, order, and preprocessing configuration.

## Performance considerations

Histogram boosting reduces split-search work by evaluating a limited set of candidate boundaries. A rough conceptual model of the search space depends on the number of features and bins rather than the number of unique raw values.

The educational tree implementations repeatedly scan observations for candidate splits. They are deliberately straightforward, so their runtime should not be used to estimate production LightGBM or CatBoost performance.

Production efficiency also depends on histogram reuse, memory layout, sparse-feature handling, thread parallelism, feature sampling, cache locality, categorical cardinality, tree constraints, and the number of boosting iterations.

The best bin count balances split precision with computation and memory. The appropriate tree complexity and learning rate depend on the data and must be selected using validation measurements.

## Security, reproducibility, and model governance

Training data should be validated against a versioned schema before fitting or inference. Categorical fields need explicit handling for null values, unexpected strings, excessive cardinality, and changes in meaning. Prediction services should reject incompatible feature dimensions rather than silently changing feature order.

A reproducible model record should include:

- Dataset version and partition definitions
- Feature names and transformations
- Categorical encoding and smoothing configuration
- Histogram binning settings
- Learning rate, iteration count, and tree constraints
- Random seeds and relevant software versions
- Validation and test metrics
- Model artifact version and creation time

Target statistics must be derived only from the training partition. Leakage prevention is a data-processing requirement, not a substitute for proper validation design.

## Relationship between the two approaches

Histogram boosting and ordered categorical statistics solve different problems within gradient-boosted tree systems.

Histogram construction reduces the number of numerical split candidates, improving the efficiency of tree training. Ordered categorical statistics use preceding observations to represent categories while reducing direct target leakage.

The mechanisms can coexist in a boosting pipeline, but they should not be confused: numerical quantization controls split-search resolution, while ordered statistics control how categorical target information is constructed. The five implementations expose these mechanisms through separate computational, object-oriented, and relational perspectives.
```
