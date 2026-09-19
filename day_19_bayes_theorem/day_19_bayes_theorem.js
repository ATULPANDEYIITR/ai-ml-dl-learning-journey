/*
 * Bayes' Theorem: Practical JavaScript Demonstrations
 *
 * This file complements the Python implementation with JavaScript-specific
 * examples involving objects, classes, higher-order functions, Maps,
 * validation, browser-compatible execution, asynchronous evidence collection,
 * classification, and practical Bayesian workflows.
 *
 * Run with:
 *   node bayes_theorem.js
 *
 * The file also avoids external npm dependencies.
 */


// ---------------------------------------------------------------------------
// 1. VALIDATION AND BASIC PROBABILITY
// ---------------------------------------------------------------------------

function validateProbability(value, name = "probability") {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new RangeError(`${name} must be between 0 and 1.`);
    }
}

function conditionalProbability(jointProbability, conditionProbability) {
    validateProbability(jointProbability, "P(A and B)");
    validateProbability(conditionProbability, "P(B)");

    if (conditionProbability === 0) {
        throw new RangeError("P(B) must be greater than zero.");
    }

    if (jointProbability > conditionProbability) {
        throw new RangeError("P(A and B) cannot exceed P(B).");
    }

    return jointProbability / conditionProbability;
}

function intersectionProbability(probabilityA, probabilityBGivenA) {
    validateProbability(probabilityA, "P(A)");
    validateProbability(probabilityBGivenA, "P(B | A)");

    return probabilityA * probabilityBGivenA;
}

function bayesBinary(prior, likelihood, competingLikelihood) {
    validateProbability(prior, "P(H)");
    validateProbability(likelihood, "P(E | H)");
    validateProbability(competingLikelihood, "P(E | not H)");

    const complementaryPrior = 1 - prior;

    // The denominator is the total probability of observing E.
    const evidence =
        likelihood * prior +
        competingLikelihood * complementaryPrior;

    if (evidence === 0) {
        throw new RangeError(
            "The evidence has zero probability under the supplied model."
        );
    }

    const posterior = likelihood * prior / evidence;

    return {
        prior,
        likelihood,
        competingLikelihood,
        evidence,
        posterior
    };
}


// ---------------------------------------------------------------------------
// 2. BASIC CONDITIONAL PROBABILITY
// ---------------------------------------------------------------------------

function demonstrateConditionalProbability() {
    console.log("\n=== CONDITIONAL PROBABILITY ===");

    const probabilityPremium = 0.40;
    const probabilityServiceGivenPremium = 0.70;

    const jointProbability = intersectionProbability(
        probabilityPremium,
        probabilityServiceGivenPremium
    );

    const recoveredConditional = conditionalProbability(
        jointProbability,
        probabilityPremium
    );

    console.log("P(Premium) =", probabilityPremium);
    console.log(
        "P(Service | Premium) =",
        probabilityServiceGivenPremium
    );
    console.log(
        "P(Premium and Service) =",
        jointProbability
    );
    console.log(
        "Recovered conditional probability =",
        recoveredConditional
    );
}


// ---------------------------------------------------------------------------
// 3. BAYES' THEOREM
// ---------------------------------------------------------------------------

function demonstrateBayesTheorem() {
    console.log("\n=== BAYES' THEOREM ===");

    const result = bayesBinary(
        0.10,  // Prior P(H)
        0.80,  // Likelihood P(E | H)
        0.05   // P(E | not H)
    );

    console.table(result);
    console.log(
        `Posterior percentage: ${(result.posterior * 100).toFixed(2)}%`
    );
}


// ---------------------------------------------------------------------------
// 4. MULTIPLE HYPOTHESES
// ---------------------------------------------------------------------------

function posteriorDistribution(hypotheses, priors, likelihoods) {
    if (
        hypotheses.length !== priors.length ||
        priors.length !== likelihoods.length
    ) {
        throw new RangeError("All arrays must have equal lengths.");
    }

    if (hypotheses.length === 0) {
        throw new RangeError("At least one hypothesis is required.");
    }

    const priorTotal = priors.reduce((sum, value) => sum + value, 0);

    if (Math.abs(priorTotal - 1) > 1e-10) {
        throw new RangeError("Priors must sum to 1.");
    }

    priors.forEach((value) => validateProbability(value, "prior"));
    likelihoods.forEach((value) =>
        validateProbability(value, "likelihood")
    );

    const unnormalized = hypotheses.map((hypothesis, index) => ({
        hypothesis,
        prior: priors[index],
        likelihood: likelihoods[index],
        unnormalizedPosterior: priors[index] * likelihoods[index]
    }));

    const evidence = unnormalized.reduce(
        (sum, item) => sum + item.unnormalizedPosterior,
        0
    );

    if (evidence === 0) {
        throw new RangeError("Evidence probability is zero.");
    }

    return unnormalized.map((item) => ({
        ...item,
        posterior: item.unnormalizedPosterior / evidence
    }));
}

function demonstrateMultipleHypotheses() {
    console.log("\n=== MULTIPLE HYPOTHESES ===");

    const result = posteriorDistribution(
        ["Server A", "Server B", "Server C"],
        [0.50, 0.30, 0.20],
        [0.20, 0.70, 0.40]
    );

    console.table(result);
}


// ---------------------------------------------------------------------------
// 5. FREQUENCY TABLE
// ---------------------------------------------------------------------------

function diagnosticFrequencyTable(
    population,
    prevalence,
    sensitivity,
    falsePositiveRate
) {
    validateProbability(prevalence, "prevalence");
    validateProbability(sensitivity, "sensitivity");
    validateProbability(falsePositiveRate, "false positive rate");

    if (!Number.isInteger(population) || population <= 0) {
        throw new RangeError("Population must be a positive integer.");
    }

    const diseased = population * prevalence;
    const healthy = population - diseased;

    const truePositives = diseased * sensitivity;
    const falseNegatives = diseased - truePositives;
    const falsePositives = healthy * falsePositiveRate;
    const trueNegatives = healthy - falsePositives;

    const positiveTests = truePositives + falsePositives;

    return {
        population,
        diseased,
        healthy,
        truePositives,
        falsePositives,
        falseNegatives,
        trueNegatives,
        probabilityDiseaseGivenPositive:
            positiveTests === 0
                ? 0
                : truePositives / positiveTests
    };
}

function demonstrateDiagnosticTesting() {
    console.log("\n=== DIAGNOSTIC TESTING ===");

    const table = diagnosticFrequencyTable(
        100000,
        0.01,
        0.99,
        0.05
    );

    console.table(table);
}


// ---------------------------------------------------------------------------
// 6. ODDS FORM
// ---------------------------------------------------------------------------

function probabilityToOdds(probability) {
    validateProbability(probability);

    if (probability === 0) {
        return 0;
    }

    if (probability === 1) {
        return Infinity;
    }

    return probability / (1 - probability);
}

function oddsToProbability(odds) {
    if (odds < 0 || Number.isNaN(odds)) {
        throw new RangeError("Odds cannot be negative.");
    }

    if (odds === Infinity) {
        return 1;
    }

    return odds / (1 + odds);
}

function demonstrateOddsForm() {
    console.log("\n=== ODDS FORM ===");

    const prior = 0.10;
    const positiveLikelihood = 0.80;
    const negativeLikelihood = 0.05;

    const likelihoodRatio =
        positiveLikelihood / negativeLikelihood;

    const priorOdds = probabilityToOdds(prior);
    const posteriorOdds = priorOdds * likelihoodRatio;
    const posterior = oddsToProbability(posteriorOdds);

    console.log("Prior odds:", priorOdds);
    console.log("Likelihood ratio:", likelihoodRatio);
    console.log("Posterior odds:", posteriorOdds);
    console.log("Posterior probability:", posterior);
}


// ---------------------------------------------------------------------------
// 7. SEQUENTIAL BAYESIAN UPDATING
// ---------------------------------------------------------------------------

function updateBelief(
    prior,
    positiveEvidenceLikelihood,
    positiveEvidenceLikelihoodUnderNotH,
    observationIsPositive
) {
    const likelihoodH = observationIsPositive
        ? positiveEvidenceLikelihood
        : 1 - positiveEvidenceLikelihood;

    const likelihoodNotH = observationIsPositive
        ? positiveEvidenceLikelihoodUnderNotH
        : 1 - positiveEvidenceLikelihoodUnderNotH;

    return bayesBinary(
        prior,
        likelihoodH,
        likelihoodNotH
    ).posterior;
}

function sequentialBayesianUpdate(
    prior,
    positiveEvidenceLikelihood,
    positiveEvidenceLikelihoodUnderNotH,
    observations
) {
    let posterior = prior;
    const history = [posterior];

    for (const observation of observations) {
        posterior = updateBelief(
            posterior,
            positiveEvidenceLikelihood,
            positiveEvidenceLikelihoodUnderNotH,
            observation
        );

        history.push(posterior);
    }

    return history;
}

function demonstrateSequentialUpdating() {
    console.log("\n=== SEQUENTIAL UPDATING ===");

    const observations = [
        true,
        true,
        false,
        true
    ];

    const history = sequentialBayesianUpdate(
        0.10,
        0.80,
        0.05,
        observations
    );

    history.forEach((posterior, index) => {
        console.log(
            `After ${index} observation(s): ` +
            `${(posterior * 100).toFixed(3)}%`
        );
    });
}


// ---------------------------------------------------------------------------
// 8. NAIVE BAYES CLASSIFIER
// ---------------------------------------------------------------------------

class NaiveBayesClassifier {
    constructor() {
        this.documentCounts = new Map();
        this.wordCounts = new Map();
        this.totalWords = new Map();
        this.vocabulary = new Set();
        this.totalDocuments = 0;
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^\p{L}\p{N}\s]/gu, " ")
            .split(/\s+/)
            .filter(Boolean);
    }

    fit(documents) {
        if (!Array.isArray(documents) || documents.length === 0) {
            throw new RangeError("Training data cannot be empty.");
        }

        for (const document of documents) {
            const { label, text } = document;

            this.totalDocuments += 1;

            this.documentCounts.set(
                label,
                (this.documentCounts.get(label) || 0) + 1
            );

            if (!this.wordCounts.has(label)) {
                this.wordCounts.set(label, new Map());
                this.totalWords.set(label, 0);
            }

            for (const word of this.tokenize(text)) {
                this.vocabulary.add(word);

                const counts = this.wordCounts.get(label);

                counts.set(
                    word,
                    (counts.get(word) || 0) + 1
                );

                this.totalWords.set(
                    label,
                    this.totalWords.get(label) + 1
                );
            }
        }
    }

    logPrior(label) {
        const count = this.documentCounts.get(label);

        if (!count) {
            throw new Error(`Unknown label: ${label}`);
        }

        return Math.log(count / this.totalDocuments);
    }

    logWordProbability(word, label) {
        const counts = this.wordCounts.get(label);

        const wordCount = counts.get(word) || 0;

        // Laplace smoothing prevents log(0) for unseen words.
        const numerator = wordCount + 1;
        const denominator =
            this.totalWords.get(label) +
            this.vocabulary.size;

        return Math.log(numerator / denominator);
    }

    score(text) {
        const words = this.tokenize(text);
        const scores = new Map();

        for (const label of this.documentCounts.keys()) {
            let score = this.logPrior(label);

            for (const word of words) {
                if (this.vocabulary.has(word)) {
                    score += this.logWordProbability(word, label);
                }
            }

            scores.set(label, score);
        }

        return scores;
    }

    predict(text) {
        const scores = this.score(text);

        let bestLabel = null;
        let bestScore = -Infinity;

        for (const [label, score] of scores) {
            if (score > bestScore) {
                bestScore = score;
                bestLabel = label;
            }
        }

        return bestLabel;
    }
}

function demonstrateNaiveBayes() {
    console.log("\n=== NAIVE BAYES CLASSIFICATION ===");

    const trainingData = [
        { label: "spam", text: "free prize win money" },
        { label: "spam", text: "claim free money now" },
        { label: "spam", text: "win free prize" },
        { label: "ham", text: "project meeting tomorrow" },
        { label: "ham", text: "review project report" },
        { label: "ham", text: "meeting schedule report" }
    ];

    const classifier = new NaiveBayesClassifier();
    classifier.fit(trainingData);

    for (const message of [
        "free money prize",
        "project meeting",
        "free project",
        "report meeting"
    ]) {
        console.log(
            `Message: "${message}" -> ${classifier.predict(message)}`
        );

        console.log(
            "Log scores:",
            Object.fromEntries(classifier.score(message))
        );
    }
}


// ---------------------------------------------------------------------------
// 9. ASYNCHRONOUS EVIDENCE COLLECTION
// ---------------------------------------------------------------------------

function delayedEvidence(name, probabilityOfPositive, delay) {
    validateProbability(
        probabilityOfPositive,
        `${name} positive probability`
    );

    return new Promise((resolve) => {
        setTimeout(() => {
            // Deterministic pseudo-observation for reproducible demonstration.
            // In a real application this value would come from an external
            // measurement, database, sensor, API, or user action.
            resolve({
                name,
                positive: probabilityOfPositive >= 0.5
            });
        }, delay);
    });
}

async function collectEvidence() {
    console.log("\n=== ASYNCHRONOUS EVIDENCE ===");

    const evidenceSources = await Promise.all([
        delayedEvidence("Sensor A", 0.90, 20),
        delayedEvidence("Sensor B", 0.80, 10),
        delayedEvidence("Sensor C", 0.20, 15)
    ]);

    console.table(evidenceSources);
}


// ---------------------------------------------------------------------------
// 10. CONTINUOUS LIKELIHOOD
// ---------------------------------------------------------------------------

function normalPdf(x, mean, standardDeviation) {
    if (standardDeviation <= 0) {
        throw new RangeError(
            "Standard deviation must be positive."
        );
    }

    const coefficient =
        1 /
        (standardDeviation * Math.sqrt(2 * Math.PI));

    const exponent =
        -0.5 *
        Math.pow(
            (x - mean) / standardDeviation,
            2
        );

    return coefficient * Math.exp(exponent);
}

function continuousPosterior(
    observation,
    hypotheses
) {
    const weighted = hypotheses.map((hypothesis) => {
        const density = normalPdf(
            observation,
            hypothesis.mean,
            hypothesis.standardDeviation
        );

        return {
            ...hypothesis,
            likelihoodDensity: density,
            unnormalizedPosterior:
                hypothesis.prior * density
        };
    });

    const evidence = weighted.reduce(
        (sum, item) => sum + item.unnormalizedPosterior,
        0
    );

    return weighted.map((item) => ({
        ...item,
        posterior:
            item.unnormalizedPosterior / evidence
    }));
}

function demonstrateContinuousBayes() {
    console.log("\n=== CONTINUOUS BAYESIAN INFERENCE ===");

    const result = continuousPosterior(
        72,
        [
            {
                name: "Machine A",
                prior: 0.60,
                mean: 70,
                standardDeviation: 3
            },
            {
                name: "Machine B",
                prior: 0.40,
                mean: 80,
                standardDeviation: 4
            }
        ]
    );

    console.table(result);
}


// ---------------------------------------------------------------------------
// 11. DECISION THEORY
// ---------------------------------------------------------------------------

function minimumExpectedLoss(posterior, lossMatrix) {
    const actions = Object.keys(lossMatrix);

    let bestAction = null;
    let lowestExpectedLoss = Infinity;

    for (const action of actions) {
        let expectedLoss = 0;

        for (const state of Object.keys(posterior)) {
            expectedLoss +=
                posterior[state] *
                lossMatrix[action][state];
        }

        if (expectedLoss < lowestExpectedLoss) {
            lowestExpectedLoss = expectedLoss;
            bestAction = action;
        }
    }

    return {
        action: bestAction,
        expectedLoss: lowestExpectedLoss
    };
}

function demonstrateDecisionTheory() {
    console.log("\n=== DECISION THEORY ===");

    const posterior = {
        fraud: 0.30,
        legitimate: 0.70
    };

    const lossMatrix = {
        approve: {
            fraud: 100,
            legitimate: 0
        },
        review: {
            fraud: 10,
            legitimate: 2
        },
        decline: {
            fraud: 0,
            legitimate: 20
        }
    };

    console.log(
        minimumExpectedLoss(
            posterior,
            lossMatrix
        )
    );
}


// ---------------------------------------------------------------------------
// 12. EDGE CASES AND DEBUGGING
// ---------------------------------------------------------------------------

function demonstrateEdgeCases() {
    console.log("\n=== EDGE CASES ===");

    console.log(
        "Certain hypothesis:",
        bayesBinary(1, 0.8, 0.1).posterior
    );

    console.log(
        "Impossible hypothesis:",
        bayesBinary(0, 0.8, 0.1).posterior
    );

    try {
        conditionalProbability(0.2, 0);
    } catch (error) {
        console.log(
            "Expected validation error:",
            error.message
        );
    }

    try {
        bayesBinary(0.5, 0, 0);
    } catch (error) {
        console.log(
            "Expected zero-evidence error:",
            error.message
        );
    }

    // Important distinction:
    // P(H | E) and P(E | H) answer different questions.
    console.log("P(H | E) is not generally equal to P(E | H).");
}


// ---------------------------------------------------------------------------
// 13. BRIER SCORE
// ---------------------------------------------------------------------------

function brierScore(probabilities, outcomes) {
    if (probabilities.length !== outcomes.length) {
        throw new RangeError(
            "Probabilities and outcomes must have equal lengths."
        );
    }

    if (probabilities.length === 0) {
        throw new RangeError("Inputs cannot be empty.");
    }

    let total = 0;

    for (let index = 0; index < probabilities.length; index += 1) {
        validateProbability(probabilities[index]);

        if (
            outcomes[index] !== 0 &&
            outcomes[index] !== 1
        ) {
            throw new RangeError(
                "Binary outcomes must be 0 or 1."
            );
        }

        total += Math.pow(
            probabilities[index] - outcomes[index],
            2
        );
    }

    return total / probabilities.length;
}

function demonstrateBrierScore() {
    console.log("\n=== BRIER SCORE ===");

    console.log(
        brierScore(
            [0.9, 0.8, 0.2, 0.1],
            [1, 1, 0, 1]
        )
    );
}


// ---------------------------------------------------------------------------
// 14. MAIN PROGRAM
// ---------------------------------------------------------------------------

async function main() {
    console.log("=".repeat(78));
    console.log("BAYES' THEOREM: JAVASCRIPT STUDY IMPLEMENTATION");
    console.log("=".repeat(78));

    demonstrateConditionalProbability();
    demonstrateBayesTheorem();
    demonstrateMultipleHypotheses();
    demonstrateDiagnosticTesting();
    demonstrateOddsForm();
    demonstrateSequentialUpdating();
    demonstrateNaiveBayes();
    await collectEvidence();
    demonstrateContinuousBayes();
    demonstrateDecisionTheory();
    demonstrateEdgeCases();
    demonstrateBrierScore();

    console.log("\n=== FORMULA REFERENCE ===");
    console.log("P(A | B) = P(A and B) / P(B)");
    console.log("P(A and B) = P(A) P(B | A)");
    console.log("P(H | E) = P(E | H) P(H) / P(E)");
    console.log("Posterior odds = Prior odds × Likelihood ratio");
}

main().catch((error) => {
    console.error("Program failed:", error.message);
    process.exitCode = 1;
});
