/*
 * Information Theory in JavaScript
 *
 * Topics:
 * - Probability distributions
 * - Self-information
 * - Shannon entropy
 * - Cross-entropy
 * - KL divergence
 * - Joint and marginal distributions
 * - Conditional entropy
 * - Mutual information
 * - Information gain
 * - Stable softmax and cross-entropy from logits
 * - Classification losses
 * - Empirical distributions
 * - Edge cases and validation
 * - Practical data-analysis patterns
 *
 * The file uses only standard JavaScript and can run with Node.js.
 */

"use strict";

// ============================================================================
// 1. VALIDATION AND NUMERICAL HELPERS
// ============================================================================

const EPSILON = 1e-15;

function assertClose(actual, expected, tolerance = 1e-10) {
    if (Math.abs(actual - expected) > tolerance) {
        throw new Error(`Expected ${expected}, received ${actual}`);
    }
}

function validateDistribution(distribution, tolerance = 1e-9) {
    if (!(distribution instanceof Map) || distribution.size === 0) {
        throw new Error("Distribution must be a non-empty Map.");
    }

    let total = 0;

    for (const [outcome, probability] of distribution) {
        if (!Number.isFinite(probability) || probability < 0) {
            throw new Error(`Invalid probability for ${String(outcome)}.`);
        }

        total += probability;
    }

    if (Math.abs(total - 1) > tolerance) {
        throw new Error(`Probabilities must sum to 1; received ${total}.`);
    }
}

function logBase(value, base) {
    if (value <= 0) {
        throw new Error("Logarithm requires a positive value.");
    }

    return Math.log(value) / Math.log(base);
}

function selfInformation(probability, base = 2) {
    if (!(probability > 0 && probability <= 1)) {
        throw new Error("Probability must be in (0, 1].");
    }

    return -logBase(probability, base);
}

// ============================================================================
// 2. SHANNON ENTROPY
// ============================================================================

function entropy(distribution, base = 2) {
    validateDistribution(distribution);

    let result = 0;

    for (const probability of distribution.values()) {
        if (probability > 0) {
            result -= probability * logBase(probability, base);
        }
    }

    return result;
}

function maximumEntropy(numberOfOutcomes, base = 2) {
    if (!Number.isInteger(numberOfOutcomes) || numberOfOutcomes <= 0) {
        throw new Error("Number of outcomes must be positive.");
    }

    return logBase(numberOfOutcomes, base);
}

// ============================================================================
// 3. CROSS-ENTROPY
// ============================================================================

function crossEntropy(trueDistribution, predictedDistribution, base = 2) {
    validateDistribution(trueDistribution);
    validateDistribution(predictedDistribution);

    let result = 0;

    for (const [outcome, trueProbability] of trueDistribution) {
        if (trueProbability === 0) {
            continue;
        }

        const predictedProbability =
            predictedDistribution.get(outcome) ?? 0;

        if (predictedProbability === 0) {
            return Infinity;
        }

        result -= trueProbability * logBase(predictedProbability, base);
    }

    return result;
}

// ============================================================================
// 4. KL DIVERGENCE
// ============================================================================

function klDivergence(p, q, base = 2) {
    validateDistribution(p);
    validateDistribution(q);

    let result = 0;

    for (const [outcome, pProbability] of p) {
        if (pProbability === 0) {
            continue;
        }

        const qProbability = q.get(outcome) ?? 0;

        if (qProbability === 0) {
            return Infinity;
        }

        result += pProbability *
            logBase(pProbability / qProbability, base);
    }

    return result;
}

// ============================================================================
// 5. JOINT DISTRIBUTION REPRESENTATION
// ============================================================================

function pairKey(x, y) {
    // JSON encoding prevents ambiguity between values such as ["A|B", "C"]
    // and ["A", "B|C"].
    return JSON.stringify([x, y]);
}

function tripleKey(x, y, z) {
    return JSON.stringify([x, y, z]);
}

function makeJoint(entries) {
    /*
     * Entries are objects with:
     * { x: ..., y: ..., probability: ... }
     */
    const joint = new Map();

    for (const entry of entries) {
        joint.set(
            pairKey(entry.x, entry.y),
            {
                x: entry.x,
                y: entry.y,
                probability: entry.probability
            }
        );
    }

    const total = [...joint.values()]
        .reduce((sum, entry) => sum + entry.probability, 0);

    if (Math.abs(total - 1) > 1e-9) {
        throw new Error("Joint probabilities must sum to 1.");
    }

    return joint;
}

function marginalX(joint) {
    const result = new Map();

    for (const entry of joint.values()) {
        result.set(
            entry.x,
            (result.get(entry.x) ?? 0) + entry.probability
        );
    }

    return result;
}

function marginalY(joint) {
    const result = new Map();

    for (const entry of joint.values()) {
        result.set(
            entry.y,
            (result.get(entry.y) ?? 0) + entry.probability
        );
    }

    return result;
}

function conditionalYGivenX(joint, xValue) {
    const px = marginalX(joint);
    const denominator = px.get(xValue) ?? 0;

    if (denominator === 0) {
        throw new Error(`P(X=${xValue}) is zero.`);
    }

    const result = new Map();

    for (const entry of joint.values()) {
        if (entry.x === xValue) {
            result.set(
                entry.y,
                entry.probability / denominator
            );
        }
    }

    return result;
}

// ============================================================================
// 6. JOINT AND CONDITIONAL ENTROPY
// ============================================================================

function jointEntropy(joint) {
    const distribution = new Map();

    for (const entry of joint.values()) {
        distribution.set(
            pairKey(entry.x, entry.y),
            entry.probability
        );
    }

    return entropy(distribution);
}

function conditionalEntropyYGivenX(joint) {
    const px = marginalX(joint);

    let result = 0;

    for (const [xValue, probabilityX] of px) {
        const conditional = conditionalYGivenX(joint, xValue);
        result += probabilityX * entropy(conditional);
    }

    return result;
}

// ============================================================================
// 7. MUTUAL INFORMATION
// ============================================================================

function mutualInformation(joint) {
    const px = marginalX(joint);
    const py = marginalY(joint);

    let result = 0;

    for (const entry of joint.values()) {
        const pxy = entry.probability;

        if (pxy === 0) {
            continue;
        }

        const independentProbability =
            px.get(entry.x) * py.get(entry.y);

        result += pxy * logBase(
            pxy / independentProbability,
            2
        );
    }

    // Tiny negative values can appear because of floating-point arithmetic.
    return Math.max(0, result);
}

// ============================================================================
// 8. STABLE SOFTMAX
// ============================================================================

function stableSoftmax(logits) {
    if (!Array.isArray(logits) || logits.length === 0) {
        throw new Error("Logits must be a non-empty array.");
    }

    const maximum = Math.max(...logits);

    const exponentials = logits.map(
        value => Math.exp(value - maximum)
    );

    const denominator = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    return exponentials.map(value => value / denominator);
}

function crossEntropyFromLogits(trueClassIndex, logits) {
    if (
        !Number.isInteger(trueClassIndex) ||
        trueClassIndex < 0 ||
        trueClassIndex >= logits.length
    ) {
        throw new Error("Invalid true class index.");
    }

    const maximum = Math.max(...logits);

    // log-sum-exp is numerically safer than directly exponentiating
    // potentially huge logits.
    const logSumExp =
        maximum +
        Math.log(
            logits.reduce(
                (sum, value) => sum + Math.exp(value - maximum),
                0
            )
        );

    return logSumExp - logits[trueClassIndex];
}

// ============================================================================
// 9. BINARY CROSS-ENTROPY
// ============================================================================

function binaryCrossEntropy(labels, predictions) {
    if (labels.length !== predictions.length || labels.length === 0) {
        throw new Error("Labels and predictions must have equal non-zero length.");
    }

    let totalLoss = 0;

    for (let index = 0; index < labels.length; index++) {
        const y = labels[index];
        const p = predictions[index];

        if (y !== 0 && y !== 1) {
            throw new Error("Binary labels must be 0 or 1.");
        }

        if (p < 0 || p > 1) {
            throw new Error("Predictions must be in [0, 1].");
        }

        // Clipping prevents log(0), which would otherwise produce Infinity.
        const clipped = Math.min(
            Math.max(p, EPSILON),
            1 - EPSILON
        );

        totalLoss -=
            y * Math.log(clipped) +
            (1 - y) * Math.log(1 - clipped);
    }

    return totalLoss / labels.length;
}

// ============================================================================
// 10. EMPIRICAL DISTRIBUTIONS
// ============================================================================

function empiricalDistribution(values) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new Error("Values must be a non-empty array.");
    }

    const counts = new Map();

    for (const value of values) {
        counts.set(value, (counts.get(value) ?? 0) + 1);
    }

    const distribution = new Map();

    for (const [value, count] of counts) {
        distribution.set(value, count / values.length);
    }

    return distribution;
}

// ============================================================================
// 11. INFORMATION GAIN
// ============================================================================

function informationGain(parentLabels, childGroups) {
    const parentDistribution = empiricalDistribution(parentLabels);
    const parentEntropy = entropy(parentDistribution);

    let weightedChildEntropy = 0;

    for (const child of childGroups) {
        if (child.length === 0) {
            continue;
        }

        const childDistribution = empiricalDistribution(child);
        const weight = child.length / parentLabels.length;

        weightedChildEntropy += weight * entropy(childDistribution);
    }

    return parentEntropy - weightedChildEntropy;
}

// ============================================================================
// 12. SAMPLE-BASED MUTUAL INFORMATION
// ============================================================================

function mutualInformationFromSamples(xValues, yValues) {
    if (
        xValues.length !== yValues.length ||
        xValues.length === 0
    ) {
        throw new Error("Sample arrays must have equal non-zero length.");
    }

    const counts = new Map();

    for (let index = 0; index < xValues.length; index++) {
        const key = pairKey(xValues[index], yValues[index]);

        if (!counts.has(key)) {
            counts.set(key, {
                x: xValues[index],
                y: yValues[index],
                count: 0
            });
        }

        counts.get(key).count++;
    }

    const total = xValues.length;

    return mutualInformation(
        new Map(
            [...counts.entries()].map(([key, entry]) => [
                key,
                {
                    x: entry.x,
                    y: entry.y,
                    probability: entry.count / total
                }
            ])
        )
    );
}

// ============================================================================
// 13. NORMALIZED MUTUAL INFORMATION
// ============================================================================

function normalizedMutualInformation(joint) {
    const hx = entropy(marginalX(joint));
    const hy = entropy(marginalY(joint));

    if (hx === 0 || hy === 0) {
        return 0;
    }

    return mutualInformation(joint) / Math.sqrt(hx * hy);
}

// ============================================================================
// 14. PRACTICAL CLASSIFICATION ANALYZER
// ============================================================================

class ClassificationAnalyzer {
    constructor(labels, probabilities) {
        if (labels.length !== probabilities.length) {
            throw new Error("Labels and predictions must match.");
        }

        this.labels = labels;
        this.probabilities = probabilities;
    }

    categoricalCrossEntropy() {
        if (this.labels.length === 0) {
            throw new Error("Dataset cannot be empty.");
        }

        let total = 0;

        for (let index = 0; index < this.labels.length; index++) {
            const label = this.labels[index];
            const probabilities = this.probabilities[index];

            if (
                !Number.isInteger(label) ||
                label < 0 ||
                label >= probabilities.length
            ) {
                throw new Error("Invalid class label.");
            }

            const probability =
                Math.min(
                    Math.max(probabilities[label], EPSILON),
                    1
                );

            total -= Math.log(probability);
        }

        return total / this.labels.length;
    }

    accuracy() {
        let correct = 0;

        for (let index = 0; index < this.labels.length; index++) {
            const prediction = this.probabilities[index];

            const predictedClass = prediction.reduce(
                (bestIndex, probability, classIndex) =>
                    probability > prediction[bestIndex]
                        ? classIndex
                        : bestIndex,
                0
            );

            if (predictedClass === this.labels[index]) {
                correct++;
            }
        }

        return correct / this.labels.length;
    }
}

// ============================================================================
// 15. DEMONSTRATIONS
// ============================================================================

function runDemonstrations() {
    console.log("=".repeat(78));
    console.log("INFORMATION THEORY IN JAVASCRIPT");
    console.log("=".repeat(78));

    // Self-information.
    console.log("\n1. SELF-INFORMATION");
    for (const probability of [1, 0.5, 0.25, 0.01]) {
        console.log(
            `P=${probability}: ${selfInformation(probability).toFixed(6)} bits`
        );
    }

    // Entropy.
    console.log("\n2. ENTROPY");

    const fairCoin = new Map([
        ["heads", 0.5],
        ["tails", 0.5]
    ]);

    const biasedCoin = new Map([
        ["heads", 0.9],
        ["tails", 0.1]
    ]);

    console.log(
        `Fair coin: ${entropy(fairCoin).toFixed(6)} bits`
    );

    console.log(
        `Biased coin: ${entropy(biasedCoin).toFixed(6)} bits`
    );

    // Cross-entropy and KL.
    console.log("\n3. CROSS-ENTROPY AND KL DIVERGENCE");

    const p = new Map([
        ["A", 0.5],
        ["B", 0.3],
        ["C", 0.2]
    ]);

    const q = new Map([
        ["A", 0.4],
        ["B", 0.4],
        ["C", 0.2]
    ]);

    const hp = entropy(p);
    const hpq = crossEntropy(p, q);
    const kl = klDivergence(p, q);

    console.log(`H(P): ${hp.toFixed(6)}`);
    console.log(`H(P,Q): ${hpq.toFixed(6)}`);
    console.log(`KL(P||Q): ${kl.toFixed(6)}`);
    console.log(`H(P)+KL(P||Q): ${(hp + kl).toFixed(6)}`);

    // Joint distribution.
    console.log("\n4. MUTUAL INFORMATION");

    const correlated = makeJoint([
        { x: "sunny", y: "sunny", probability: 0.5 },
        { x: "rainy", y: "rainy", probability: 0.5 }
    ]);

    const independent = makeJoint([
        { x: "sunny", y: "sunny", probability: 0.25 },
        { x: "sunny", y: "rainy", probability: 0.25 },
        { x: "rainy", y: "sunny", probability: 0.25 },
        { x: "rainy", y: "rainy", probability: 0.25 }
    ]);

    console.log(
        `Correlated I(X;Y): ${mutualInformation(correlated).toFixed(6)} bits`
    );

    console.log(
        `Independent I(X;Y): ${mutualInformation(independent).toFixed(6)} bits`
    );

    console.log(
        `H(Y|X), correlated: ${
            conditionalEntropyYGivenX(correlated).toFixed(6)
        } bits`
    );

    // Stable softmax.
    console.log("\n5. NUMERICAL STABILITY");

    const logits = [1000, 999, 998];
    const probabilities = stableSoftmax(logits);

    console.log("Stable softmax:", probabilities);
    console.log(
        "Cross-entropy from logits:",
        crossEntropyFromLogits(0, logits).toFixed(6)
    );

    // Binary classification.
    console.log("\n6. BINARY CROSS-ENTROPY");

    console.log(
        binaryCrossEntropy(
            [1, 0, 1, 1],
            [0.9, 0.1, 0.8, 0.7]
        ).toFixed(6),
        "nats"
    );

    // Empirical entropy.
    console.log("\n7. EMPIRICAL ENTROPY");

    const observations = [
        "red", "red", "red",
        "blue", "blue",
        "green", "green", "green", "green"
    ];

    const empirical = empiricalDistribution(observations);

    console.log(
        "Estimated distribution:",
        [...empirical.entries()]
    );

    console.log(
        "Estimated entropy:",
        entropy(empirical).toFixed(6),
        "bits"
    );

    // Information gain.
    console.log("\n8. INFORMATION GAIN");

    const parent = [
        "yes", "yes", "yes", "no",
        "no", "no", "yes", "no"
    ];

    const childA = ["yes", "yes", "yes", "yes"];
    const childB = ["no", "no", "no", "no"];

    console.log(
        "Information gain:",
        informationGain(parent, [childA, childB]).toFixed(6),
        "bits"
    );

    // Feature relevance.
    console.log("\n9. MUTUAL INFORMATION FROM SAMPLES");

    const feature = ["A", "A", "B", "B", "A", "B", "A", "B"];
    const target = ["0", "0", "1", "1", "0", "1", "0", "1"];

    console.log(
        "Feature-target MI:",
        mutualInformationFromSamples(feature, target).toFixed(6),
        "bits"
    );

    // Normalized MI.
    console.log("\n10. NORMALIZED MUTUAL INFORMATION");

    console.log(
        normalizedMutualInformation(correlated).toFixed(6)
    );

    // Object-oriented analysis.
    console.log("\n11. CLASSIFICATION ANALYZER");

    const analyzer = new ClassificationAnalyzer(
        [0, 2, 1],
        [
            [0.8, 0.15, 0.05],
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2]
        ]
    );

    console.log(
        "Cross-entropy:",
        analyzer.categoricalCrossEntropy().toFixed(6),
        "nats"
    );

    console.log(
        "Accuracy:",
        analyzer.accuracy().toFixed(6)
    );
}

// ============================================================================
// 16. CORRECTNESS TESTS
// ============================================================================

function runTests() {
    const fairCoin = new Map([
        ["H", 0.5],
        ["T", 0.5]
    ]);

    assertClose(entropy(fairCoin), 1);

    assertClose(
        klDivergence(fairCoin, fairCoin),
        0
    );

    const independent = makeJoint([
        { x: "0", y: "0", probability: 0.25 },
        { x: "0", y: "1", probability: 0.25 },
        { x: "1", y: "0", probability: 0.25 },
        { x: "1", y: "1", probability: 0.25 }
    ]);

    assertClose(
        mutualInformation(independent),
        0
    );

    const perfectlyCorrelated = makeJoint([
        { x: "0", y: "0", probability: 0.5 },
        { x: "1", y: "1", probability: 0.5 }
    ]);

    assertClose(
        mutualInformation(perfectlyCorrelated),
        1
    );

    const p = new Map([
        ["A", 0.5],
        ["B", 0.5]
    ]);

    const q = new Map([
        ["A", 0.25],
        ["B", 0.75]
    ]);

    assertClose(
        crossEntropy(p, q),
        entropy(p) + klDivergence(p, q)
    );

    // Stable softmax must not overflow for large logits.
    const stable = stableSoftmax([1000, 999, 998]);

    assertClose(
        stable.reduce((sum, value) => sum + value, 0),
        1
    );

    console.log("\nAll JavaScript mathematical checks passed.");
}

// ============================================================================
// 17. PROGRAM ENTRY POINT
// ============================================================================

runDemonstrations();
runTests();
