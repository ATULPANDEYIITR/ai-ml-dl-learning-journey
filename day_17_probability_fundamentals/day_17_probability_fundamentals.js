/*
 * Probability Fundamentals
 *
 * A self-contained JavaScript study file covering:
 * - probability spaces
 * - outcomes and events
 * - event set operations
 * - probability axioms
 * - conditional probability
 * - Bayes' theorem
 * - law of total probability
 * - independence
 * - pairwise versus mutual independence
 * - discrete random variables
 * - expectation and variance
 * - Monte Carlo simulation
 * - sampling without replacement
 * - reliability modeling
 * - validation and numerical considerations
 *
 * Runtime: modern Node.js or a modern browser console.
 */

"use strict";

// -----------------------------------------------------------------------------
// 1. Utility functions
// -----------------------------------------------------------------------------

function section(title) {
    console.log("\n" + "=".repeat(78));
    console.log(title);
    console.log("=".repeat(78));
}

function assertCondition(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

function approximatelyEqual(a, b, tolerance = 1e-12) {
    return Math.abs(a - b) <= tolerance;
}

// -----------------------------------------------------------------------------
// 2. Finite probability space
// -----------------------------------------------------------------------------

class FiniteProbabilitySpace {
    /*
     * A finite probability space maps each elementary outcome to its
     * probability.
     *
     * JavaScript's Map is useful here because outcomes can be strings,
     * numbers, or object references without requiring object-key coercion.
     */
    constructor(probabilities) {
        if (!(probabilities instanceof Map)) {
            throw new TypeError("probabilities must be a Map");
        }

        if (probabilities.size === 0) {
            throw new Error("A probability space cannot be empty");
        }

        let total = 0;

        for (const [outcome, probability] of probabilities) {
            if (!Number.isFinite(probability) || probability < 0) {
                throw new Error(
                    `Invalid probability for ${String(outcome)}`
                );
            }
            total += probability;
        }

        if (!approximatelyEqual(total, 1, 1e-12)) {
            throw new Error(
                `Probabilities must sum to 1; received ${total}`
            );
        }

        this.probabilities = new Map(probabilities);
    }

    get outcomes() {
        return new Set(this.probabilities.keys());
    }

    validateEvent(event) {
        const result = new Set(event);

        for (const outcome of result) {
            if (!this.probabilities.has(outcome)) {
                throw new Error(
                    `Event contains an outcome outside the sample space: ${String(outcome)}`
                );
            }
        }

        return result;
    }

    probability(event) {
        const validatedEvent = this.validateEvent(event);
        let total = 0;

        for (const outcome of validatedEvent) {
            total += this.probabilities.get(outcome);
        }

        return total;
    }

    complement(event) {
        const validatedEvent = this.validateEvent(event);
        const result = new Set();

        for (const outcome of this.outcomes) {
            if (!validatedEvent.has(outcome)) {
                result.add(outcome);
            }
        }

        return result;
    }

    intersection(eventA, eventB) {
        const a = this.validateEvent(eventA);
        const b = this.validateEvent(eventB);
        const result = new Set();

        for (const outcome of a) {
            if (b.has(outcome)) {
                result.add(outcome);
            }
        }

        return result;
    }

    union(eventA, eventB) {
        const a = this.validateEvent(eventA);
        const b = this.validateEvent(eventB);

        return new Set([...a, ...b]);
    }

    conditionalProbability(event, condition) {
        const a = this.validateEvent(event);
        const b = this.validateEvent(condition);

        const pB = this.probability(b);

        if (approximatelyEqual(pB, 0, 1e-15)) {
            throw new Error(
                "P(A | B) is undefined when P(B) = 0"
            );
        }

        return this.probability(this.intersection(a, b)) / pB;
    }

    areIndependent(eventA, eventB) {
        const a = this.validateEvent(eventA);
        const b = this.validateEvent(eventB);

        const pA = this.probability(a);
        const pB = this.probability(b);
        const pAB = this.probability(this.intersection(a, b));

        return approximatelyEqual(pAB, pA * pB);
    }
}

// -----------------------------------------------------------------------------
// 3. Probability-space fundamentals
// -----------------------------------------------------------------------------

function probabilitySpaceDemo() {
    section("Probability spaces");

    const coin = new FiniteProbabilitySpace(
        new Map([
            ["H", 0.5],
            ["T", 0.5]
        ])
    );

    const heads = new Set(["H"]);

    console.log("Omega:", [...coin.outcomes]);
    console.log("Event A = heads:", [...heads]);
    console.log("P(A):", coin.probability(heads));
    console.log("P(A complement):", coin.probability(coin.complement(heads)));
}

// -----------------------------------------------------------------------------
// 4. Event operations
// -----------------------------------------------------------------------------

function eventOperationsDemo() {
    section("Events and set operations");

    const die = new FiniteProbabilitySpace(
        new Map(
            Array.from({ length: 6 }, (_, index) => [index + 1, 1 / 6])
        )
    );

    const even = new Set([2, 4, 6]);
    const greaterThanThree = new Set([4, 5, 6]);

    const intersection = die.intersection(even, greaterThanThree);
    const union = die.union(even, greaterThanThree);
    const complement = die.complement(even);

    console.log("A:", [...even]);
    console.log("B:", [...greaterThanThree]);
    console.log("A intersection B:", [...intersection]);
    console.log("A union B:", [...union]);
    console.log("A complement:", [...complement]);

    const pA = die.probability(even);
    const pB = die.probability(greaterThanThree);
    const pAB = die.probability(intersection);
    const pUnion = die.probability(union);

    console.log("P(A):", pA);
    console.log("P(B):", pB);
    console.log("P(A intersection B):", pAB);
    console.log("P(A union B):", pUnion);

    // Inclusion-exclusion:
    // P(A union B) = P(A) + P(B) - P(A intersection B)
    assertCondition(
        approximatelyEqual(pUnion, pA + pB - pAB),
        "inclusion-exclusion"
    );
}

// -----------------------------------------------------------------------------
// 5. Counting
// -----------------------------------------------------------------------------

function factorial(n) {
    if (!Number.isInteger(n) || n < 0) {
        throw new RangeError("factorial requires a non-negative integer");
    }

    let result = 1;

    for (let i = 2; i <= n; i++) {
        result *= i;
    }

    return result;
}

function combinations(n, k) {
    if (!Number.isInteger(n) || !Number.isInteger(k)) {
        throw new TypeError("n and k must be integers");
    }

    if (k < 0 || k > n) {
        return 0;
    }

    // Use the smaller side to reduce multiplication and improve numerical
    // behavior for moderate values.
    k = Math.min(k, n - k);

    let result = 1;

    for (let i = 1; i <= k; i++) {
        result = result * (n - k + i) / i;
    }

    return Math.round(result);
}

function countingDemo() {
    section("Counting and equally likely outcomes");

    const allTwoDice = [];

    for (let first = 1; first <= 6; first++) {
        for (let second = 1; second <= 6; second++) {
            allTwoDice.push([first, second]);
        }
    }

    const favorable = allTwoDice.filter(
        ([first, second]) => first + second >= 9
    );

    console.log("Total ordered outcomes:", allTwoDice.length);
    console.log("Favorable outcomes:", favorable.length);
    console.log("P(sum >= 9):", favorable.length / allTwoDice.length);

    const ways = combinations(5, 3);
    const exactThreeHeads = ways / (2 ** 5);

    console.log("C(5, 3):", ways);
    console.log("P(exactly 3 heads in 5 flips):", exactThreeHeads);
}

// -----------------------------------------------------------------------------
// 6. Conditional probability
// -----------------------------------------------------------------------------

function conditionalProbabilityDemo() {
    section("Conditional probability");

    const die = new FiniteProbabilitySpace(
        new Map(
            Array.from({ length: 6 }, (_, index) => [index + 1, 1 / 6])
        )
    );

    const even = new Set([2, 4, 6]);
    const greaterThanThree = new Set([4, 5, 6]);

    const pA = die.probability(even);
    const pB = die.probability(greaterThanThree);
    const pAB = die.probability(
        die.intersection(even, greaterThanThree)
    );
    const pAGivenB = die.conditionalProbability(
        even,
        greaterThanThree
    );

    console.log("P(A):", pA);
    console.log("P(B):", pB);
    console.log("P(A intersection B):", pAB);
    console.log("P(A | B):", pAGivenB);

    // Multiplication rule:
    // P(A intersection B) = P(A | B) P(B)
    assertCondition(
        approximatelyEqual(pAB, pAGivenB * pB),
        "conditional multiplication rule"
    );
}

// -----------------------------------------------------------------------------
// 7. Bayes' theorem
// -----------------------------------------------------------------------------

function bayesDemo() {
    section("Bayes' theorem");

    const prevalence = 0.01;
    const sensitivity = 0.95;
    const falsePositiveRate = 0.10;

    // Law of total probability:
    //
    // P(positive) =
    //     P(positive | disease)P(disease)
    //     + P(positive | no disease)P(no disease)
    const pPositive =
        sensitivity * prevalence +
        falsePositiveRate * (1 - prevalence);

    // Bayes:
    //
    // P(disease | positive) =
    //     P(positive | disease)P(disease)
    //     / P(positive)
    const posterior =
        sensitivity * prevalence / pPositive;

    console.log("P(disease):", prevalence);
    console.log("P(positive):", pPositive);
    console.log("P(disease | positive):", posterior);
}

// -----------------------------------------------------------------------------
// 8. Total probability
// -----------------------------------------------------------------------------

function totalProbabilityDemo() {
    section("Law of total probability");

    const factories = new Map([
        ["A", { probability: 0.50, defectRate: 0.01 }],
        ["B", { probability: 0.30, defectRate: 0.03 }],
        ["C", { probability: 0.20, defectRate: 0.05 }]
    ]);

    let defectProbability = 0;

    for (const factory of factories.values()) {
        defectProbability +=
            factory.probability * factory.defectRate;
    }

    console.log("Overall defect probability:", defectProbability);
}

// -----------------------------------------------------------------------------
// 9. Independence
// -----------------------------------------------------------------------------

function independenceDemo() {
    section("Independence");

    const die = new FiniteProbabilitySpace(
        new Map(
            Array.from({ length: 6 }, (_, index) => [index + 1, 1 / 6])
        )
    );

    const even = new Set([2, 4, 6]);
    const greaterThanThree = new Set([4, 5, 6]);

    const pA = die.probability(even);
    const pB = die.probability(greaterThanThree);
    const pAB = die.probability(
        die.intersection(even, greaterThanThree)
    );

    console.log("P(A):", pA);
    console.log("P(B):", pB);
    console.log("P(A intersection B):", pAB);
    console.log("P(A)P(B):", pA * pB);
    console.log(
        "Independent:",
        die.areIndependent(even, greaterThanThree)
    );

    /*
     * Mutually exclusive events with positive probabilities cannot be
     * independent:
     *
     * If A and B are mutually exclusive:
     *     P(A intersection B) = 0
     *
     * Independence would require:
     *     P(A)P(B) = 0
     *
     * Therefore at least one event must have probability zero.
     */
}

// -----------------------------------------------------------------------------
// 10. Pairwise but not mutual independence
// -----------------------------------------------------------------------------

function pairwiseNotMutualDemo() {
    section("Pairwise versus mutual independence");

    /*
     * Outcomes are two fair binary variables:
     *
     * A: first bit is 1
     * B: second bit is 1
     * C: XOR of the two bits is 1
     *
     * Every pair is independent, but all three together are not mutually
     * independent.
     */

    const outcomes = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ];

    const space = new FiniteProbabilitySpace(
        new Map(outcomes.map(outcome => [JSON.stringify(outcome), 0.25]))
    );

    const encode = outcome => JSON.stringify(outcome);

    const A = new Set(
        outcomes
            .filter(([x]) => x === 1)
            .map(encode)
    );

    const B = new Set(
        outcomes
            .filter(([, y]) => y === 1)
            .map(encode)
    );

    const C = new Set(
        outcomes
            .filter(([x, y]) => (x ^ y) === 1)
            .map(encode)
    );

    console.log("A and B independent:", space.areIndependent(A, B));
    console.log("A and C independent:", space.areIndependent(A, C));
    console.log("B and C independent:", space.areIndependent(B, C));

    const tripleIntersection = new Set(
        [...A].filter(value => B.has(value) && C.has(value))
    );

    const pTriple = space.probability(tripleIntersection);
    const product =
        space.probability(A) *
        space.probability(B) *
        space.probability(C);

    console.log("P(A intersection B intersection C):", pTriple);
    console.log("P(A)P(B)P(C):", product);
}

// -----------------------------------------------------------------------------
// 11. Random variables
// -----------------------------------------------------------------------------

function expectation(distribution, valueFunction = x => x) {
    let result = 0;

    for (const [outcome, probability] of distribution) {
        result += probability * valueFunction(outcome);
    }

    return result;
}

function variance(distribution, valueFunction = x => x) {
    const mean = expectation(distribution, valueFunction);

    let result = 0;

    for (const [outcome, probability] of distribution) {
        const value = valueFunction(outcome);
        result += probability * (value - mean) ** 2;
    }

    return result;
}

function randomVariableDemo() {
    section("Discrete random variables");

    const die = new Map(
        Array.from({ length: 6 }, (_, index) => [index + 1, 1 / 6])
    );

    const mean = expectation(die);
    const varianceValue = variance(die);

    console.log("E[X]:", mean);
    console.log("Var(X):", varianceValue);
    console.log("Std(X):", Math.sqrt(varianceValue));

    const event = new Set([4, 5, 6]);

    // For an indicator random variable I_A:
    //
    // I_A = 1 when A occurs
    // I_A = 0 otherwise
    //
    // E[I_A] = P(A)
    const indicatorExpectation = expectation(
        die,
        outcome => event.has(outcome) ? 1 : 0
    );

    console.log("E[I_A]:", indicatorExpectation);
    console.log("P(A):", 3 / 6);
}

// -----------------------------------------------------------------------------
// 12. Monte Carlo simulation
// -----------------------------------------------------------------------------

function createRandomGenerator(seed) {
    /*
     * A small deterministic pseudo-random generator is used so the example
     * can be reproduced without external packages.
     *
     * This is educational and is not intended for cryptographic security.
     */
    let state = seed >>> 0;

    return function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function estimateProbability(
    trials,
    eventGenerator,
    seed = 42
) {
    if (!Number.isInteger(trials) || trials <= 0) {
        throw new RangeError("trials must be a positive integer");
    }

    const random = createRandomGenerator(seed);
    let successes = 0;

    for (let i = 0; i < trials; i++) {
        if (eventGenerator(random)) {
            successes++;
        }
    }

    return successes / trials;
}

function monteCarloDemo() {
    section("Monte Carlo simulation");

    for (const trials of [100, 1000, 10000, 100000]) {
        const estimate = estimateProbability(
            trials,
            random => random() < 0.5
        );

        console.log(
            `n=${String(trials).padStart(6)} ` +
            `estimate=${estimate.toFixed(5)} ` +
            `error=${Math.abs(estimate - 0.5).toFixed(5)}`
        );
    }

    /*
     * Simulation estimates a probability. It does not replace an exact
     * analytical calculation when an exact calculation is available.
     *
     * Standard Monte Carlo error generally scales approximately as:
     *
     *     O(1 / sqrt(n))
     *
     * Therefore much larger samples are required for progressively smaller
     * random error.
     */
}

// -----------------------------------------------------------------------------
// 13. Sampling without replacement
// -----------------------------------------------------------------------------

function withoutReplacementDemo() {
    section("Sampling without replacement");

    const pFirstHeart = 13 / 52;
    const pSecondHeartGivenFirst = 12 / 51;
    const pTwoHearts =
        pFirstHeart * pSecondHeartGivenFirst;

    console.log("P(first heart):", pFirstHeart);
    console.log(
        "P(second heart | first heart):",
        pSecondHeartGivenFirst
    );
    console.log("P(two consecutive hearts):", pTwoHearts);
}

// -----------------------------------------------------------------------------
// 14. Reliability systems
// -----------------------------------------------------------------------------

class Component {
    constructor(name, reliability) {
        if (!Number.isFinite(reliability) ||
            reliability < 0 ||
            reliability > 1) {
            throw new RangeError(
                "Reliability must be between 0 and 1"
            );
        }

        this.name = name;
        this.reliability = reliability;
    }
}

function seriesReliability(components) {
    /*
     * A series system works only if every component works.
     *
     * Under independence:
     *     R = R1 * R2 * ... * Rn
     */
    return components.reduce(
        (result, component) =>
            result * component.reliability,
        1
    );
}

function parallelReliability(components) {
    /*
     * A parallel system works when at least one component works.
     *
     * Calculate the complement:
     *
     *     R = 1 - P(all fail)
     */
    const probabilityAllFail = components.reduce(
        (result, component) =>
            result * (1 - component.reliability),
        1
    );

    return 1 - probabilityAllFail;
}

function reliabilityDemo() {
    section("Reliability case study");

    const servers = [
        new Component("server-1", 0.98),
        new Component("server-2", 0.97)
    ];

    const database = new Component("database", 0.995);

    const serverSubsystem =
        parallelReliability(servers);

    const totalReliability =
        serverSubsystem * database.reliability;

    console.log(
        "Server subsystem reliability:",
        serverSubsystem
    );

    console.log(
        "Database reliability:",
        database.reliability
    );

    console.log(
        "Total system reliability:",
        totalReliability
    );

    console.log(
        "Failure probability:",
        1 - totalReliability
    );
}

// -----------------------------------------------------------------------------
// 15. Edge cases
// -----------------------------------------------------------------------------

function edgeCasesDemo() {
    section("Edge cases and validation");

    try {
        new FiniteProbabilitySpace(
            new Map([
                ["A", 0.8],
                ["B", 0.5]
            ])
        );
    } catch (error) {
        console.log("Invalid total probability rejected:", error.message);
    }

    try {
        new FiniteProbabilitySpace(
            new Map([
                ["A", -0.1],
                ["B", 1.1]
            ])
        );
    } catch (error) {
        console.log("Negative/out-of-range probability rejected:", error.message);
    }

    const space = new FiniteProbabilitySpace(
        new Map([
            ["A", 1],
            ["B", 0]
        ])
    );

    try {
        space.conditionalProbability(
            new Set(["A"]),
            new Set(["B"])
        );
    } catch (error) {
        console.log("Zero-condition probability rejected:", error.message);
    }

    try {
        space.probability(new Set(["C"]));
    } catch (error) {
        console.log("Unknown outcome rejected:", error.message);
    }
}

// -----------------------------------------------------------------------------
// 16. Law and formula demonstrations
// -----------------------------------------------------------------------------

function formulaChecksDemo() {
    section("Core formula checks");

    const pA = 0.4;
    const pB = 0.3;
    const pAB = 0.1;

    // Inclusion-exclusion.
    const pUnion = pA + pB - pAB;

    // Complement.
    const pNotA = 1 - pA;

    // Fréchet bounds.
    const lowerBound = Math.max(0, pA + pB - 1);
    const upperBound = Math.min(pA, pB);

    console.log("P(A union B):", pUnion);
    console.log("P(not A):", pNotA);
    console.log("P(A intersection B) lower bound:", lowerBound);
    console.log("P(A intersection B) upper bound:", upperBound);

    assertCondition(
        pAB >= lowerBound && pAB <= upperBound,
        "intersection must satisfy probability bounds"
    );
}

// -----------------------------------------------------------------------------
// 17. Self-tests
// -----------------------------------------------------------------------------

function runTests() {
    section("Self-tests");

    const die = new FiniteProbabilitySpace(
        new Map(
            Array.from({ length: 6 }, (_, index) => [index + 1, 1 / 6])
        )
    );

    const all = die.outcomes;
    const empty = new Set();

    assertCondition(
        approximatelyEqual(die.probability(all), 1),
        "P(Omega) = 1"
    );

    assertCondition(
        approximatelyEqual(die.probability(empty), 0),
        "P(empty) = 0"
    );

    const even = new Set([2, 4, 6]);

    assertCondition(
        approximatelyEqual(
            die.probability(die.complement(even)),
            1 - die.probability(even)
        ),
        "complement rule"
    );

    const b = new Set([4, 5, 6]);

    const pAB = die.probability(
        die.intersection(even, b)
    );

    const pAGivenB =
        die.conditionalProbability(even, b);

    assertCondition(
        approximatelyEqual(
            pAB,
            pAGivenB * die.probability(b)
        ),
        "multiplication rule"
    );

    assertCondition(
        !die.areIndependent(even, b),
        "selected events are dependent"
    );

    console.log("All tests passed.");
}

// -----------------------------------------------------------------------------
// 18. Main
// -----------------------------------------------------------------------------

function main() {
    section("Probability fundamentals");

    console.log(`
Probability provides a mathematical framework for uncertainty.

The central objects are:
  - sample space Omega: all possible elementary outcomes
  - event: a subset of Omega
  - probability P(A): a value from 0 to 1
  - conditional probability P(A | B)
  - independence: P(A intersection B) = P(A)P(B)

The examples progress from finite probability spaces to Bayes' theorem,
independence, random variables, simulation, reliability and validation.
`);

    probabilitySpaceDemo();
    eventOperationsDemo();
    countingDemo();
    conditionalProbabilityDemo();
    bayesDemo();
    totalProbabilityDemo();
    independenceDemo();
    pairwiseNotMutualDemo();
    randomVariableDemo();
    monteCarloDemo();
    withoutReplacementDemo();
    reliabilityDemo();
    edgeCasesDemo();
    formulaChecksDemo();
    runTests();

    section("End of probability fundamentals demonstration");
}

main();
