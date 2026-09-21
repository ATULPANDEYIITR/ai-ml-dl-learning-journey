/*
 * STATISTICAL INFERENCE
 * Sampling, confidence intervals, hypothesis testing, and p-values
 *
 * Self-contained JavaScript study file.
 *
 * This implementation emphasizes:
 *   - JavaScript numerical behavior
 *   - reusable statistical functions
 *   - validation
 *   - simulation
 *   - bootstrap inference
 *   - permutation inference
 *   - practical interpretation
 *
 * Compatible with modern Node.js.
 */

"use strict";

// ============================================================================
// 1. BASIC DESCRIPTIVE STATISTICS
// ============================================================================

function assertNonEmpty(values, name = "values") {
    if (!Array.isArray(values) || values.length === 0) {
        throw new Error(`${name} must be a non-empty array.`);
    }

    if (!values.every(Number.isFinite)) {
        throw new Error(`${name} must contain only finite numbers.`);
    }
}

function mean(values) {
    assertNonEmpty(values);
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleVariance(values) {
    assertNonEmpty(values);

    if (values.length < 2) {
        throw new Error("At least two observations are required.");
    }

    const average = mean(values);

    return values.reduce(
        (total, value) => total + (value - average) ** 2,
        0
    ) / (values.length - 1);
}

function sampleStandardDeviation(values) {
    return Math.sqrt(sampleVariance(values));
}

function standardError(values) {
    return sampleStandardDeviation(values) / Math.sqrt(values.length);
}


// ============================================================================
// 2. REPRODUCIBLE RANDOM NUMBER GENERATOR
// ============================================================================

class SeededRandom {
    /*
     * JavaScript's Math.random() does not provide a standard way to set a seed.
     * This small linear congruential generator makes educational simulations
     * reproducible.
     */
    constructor(seed = 123456789) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (
            (1664525 * this.state + 1013904223)
            >>> 0
        );

        return this.state / 4294967296;
    }

    integer(min, max) {
        return Math.floor(
            this.next() * (max - min + 1)
        ) + min;
    }

    normal(meanValue = 0, standardDeviation = 1) {
        /*
         * Box-Muller transformation:
         *
         * Z = sqrt(-2 ln U1) cos(2πU2)
         *
         * converts two independent uniform random variables into
         * approximately standard normal variables.
         */
        let u1 = this.next();
        let u2 = this.next();

        while (u1 === 0) {
            u1 = this.next();
        }

        const z = Math.sqrt(-2 * Math.log(u1))
            * Math.cos(2 * Math.PI * u2);

        return meanValue + standardDeviation * z;
    }

    choice(values) {
        if (values.length === 0) {
            throw new Error("Cannot choose from an empty array.");
        }

        return values[
            Math.floor(this.next() * values.length)
        ];
    }

    shuffle(values) {
        const result = [...values];

        // Fisher-Yates shuffle gives an unbiased permutation when the
        // underlying random generator is uniform.
        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }

        return result;
    }
}


// ============================================================================
// 3. SAMPLING
// ============================================================================

function simpleRandomSample(population, sampleSize, rng) {
    if (!Array.isArray(population)) {
        throw new Error("Population must be an array.");
    }

    if (
        sampleSize <= 0 ||
        sampleSize > population.length
    ) {
        throw new Error("Invalid sample size.");
    }

    return rng.shuffle(population).slice(0, sampleSize);
}

function bootstrapResample(values, rng) {
    return Array.from(
        { length: values.length },
        () => rng.choice(values)
    );
}


// ============================================================================
// 4. NORMAL DISTRIBUTION
// ============================================================================

function normalCDF(x, meanValue = 0, standardDeviation = 1) {
    if (standardDeviation <= 0) {
        throw new Error("Standard deviation must be positive.");
    }

    /*
     * JavaScript does not provide erf() in every target runtime.
     * Abramowitz-Stegun approximation is sufficient for educational
     * statistical calculations.
     */
    function erf(z) {
        const sign = z >= 0 ? 1 : -1;
        z = Math.abs(z);

        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;

        const t = 1 / (1 + p * z);

        const polynomial =
            (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;

        return sign * (1 - polynomial * Math.exp(-z * z));
    }

    return 0.5 * (
        1 + erf(
            (x - meanValue) /
            (standardDeviation * Math.sqrt(2))
        )
    );
}

function normalQuantile(probability) {
    if (!(probability > 0 && probability < 1)) {
        throw new Error("Probability must be between 0 and 1.");
    }

    // Binary search is slower than a closed-form approximation but is
    // transparent and reliable enough for this educational implementation.
    let low = -10;
    let high = 10;

    for (let i = 0; i < 100; i++) {
        const mid = (low + high) / 2;

        if (normalCDF(mid) < probability) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return (low + high) / 2;
}


// ============================================================================
// 5. CONFIDENCE INTERVAL FOR A MEAN
// ============================================================================

function meanConfidenceIntervalKnownSigma(
    values,
    populationStandardDeviation,
    confidenceLevel = 0.95
) {
    assertNonEmpty(values);

    if (populationStandardDeviation <= 0) {
        throw new Error("Population standard deviation must be positive.");
    }

    const alpha = 1 - confidenceLevel;
    const zCritical = normalQuantile(1 - alpha / 2);
    const estimate = mean(values);

    const margin =
        zCritical
        * populationStandardDeviation
        / Math.sqrt(values.length);

    return {
        estimate,
        lower: estimate - margin,
        upper: estimate + margin,
        confidenceLevel,
        method: "z interval with known population standard deviation"
    };
}


// ============================================================================
// 6. APPROXIMATE t DISTRIBUTION
// ============================================================================

function gamma(x) {
    /*
     * Lanczos approximation for the gamma function.
     *
     * Gamma(n) = (n-1)! for positive integers.
     */
    const coefficients = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ];

    if (x < 0.5) {
        return Math.PI / (
            Math.sin(Math.PI * x)
            * gamma(1 - x)
        );
    }

    x -= 1;

    let value = 0.99999999999980993;

    for (let i = 0; i < coefficients.length; i++) {
        value += coefficients[i] / (x + i + 1);
    }

    const t = x + coefficients.length - 0.5;

    return Math.sqrt(2 * Math.PI)
        * t ** (x + 0.5)
        * Math.exp(-t)
        * value;
}

function tPDF(t, degreesOfFreedom) {
    const df = degreesOfFreedom;

    if (df <= 0) {
        throw new Error("Degrees of freedom must be positive.");
    }

    const numerator = gamma((df + 1) / 2);
    const denominator =
        Math.sqrt(df * Math.PI)
        * gamma(df / 2);

    return (
        numerator / denominator
    ) * (
        1 + t * t / df
    ) ** (-(df + 1) / 2);
}

function numericalIntegral(functionToIntegrate, lower, upper, steps = 5000) {
    const width = (upper - lower) / steps;

    let total =
        0.5 * (
            functionToIntegrate(lower)
            + functionToIntegrate(upper)
        );

    for (let i = 1; i < steps; i++) {
        total += functionToIntegrate(
            lower + i * width
        );
    }

    return total * width;
}

function tCDF(t, degreesOfFreedom) {
    if (t === 0) {
        return 0.5;
    }

    if (t > 0) {
        return 0.5 + numericalIntegral(
            value => tPDF(value, degreesOfFreedom),
            0,
            t
        );
    }

    return 1 - tCDF(-t, degreesOfFreedom);
}

function tQuantile(probability, degreesOfFreedom) {
    if (!(probability > 0 && probability < 1)) {
        throw new Error("Probability must be between 0 and 1.");
    }

    let low = -20;
    let high = 20;

    for (let i = 0; i < 80; i++) {
        const mid = (low + high) / 2;

        if (tCDF(mid, degreesOfFreedom) < probability) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return (low + high) / 2;
}

function meanConfidenceIntervalT(values, confidenceLevel = 0.95) {
    assertNonEmpty(values);

    if (values.length < 2) {
        throw new Error("At least two observations are required.");
    }

    const alpha = 1 - confidenceLevel;
    const degreesOfFreedom = values.length - 1;

    const criticalValue = tQuantile(
        1 - alpha / 2,
        degreesOfFreedom
    );

    const estimate = mean(values);

    const margin =
        criticalValue * standardError(values);

    return {
        estimate,
        lower: estimate - margin,
        upper: estimate + margin,
        confidenceLevel,
        degreesOfFreedom,
        method: "Student's t confidence interval"
    };
}


// ============================================================================
// 7. HYPOTHESIS TESTING
// ============================================================================

function pValueFromZ(z, alternative = "two-sided") {
    if (alternative === "two-sided") {
        return 2 * (
            1 - normalCDF(Math.abs(z))
        );
    }

    if (alternative === "greater") {
        return 1 - normalCDF(z);
    }

    if (alternative === "less") {
        return normalCDF(z);
    }

    throw new Error(
        "Alternative must be two-sided, greater, or less."
    );
}

function oneSampleZTest(
    values,
    nullMean,
    populationStandardDeviation,
    alternative = "two-sided",
    alpha = 0.05
) {
    assertNonEmpty(values);

    const z =
        (mean(values) - nullMean)
        / (
            populationStandardDeviation
            / Math.sqrt(values.length)
        );

    const pValue = pValueFromZ(
        z,
        alternative
    );

    return {
        statistic: z,
        pValue,
        alpha,
        rejectNull: pValue < alpha,
        alternative,
        method: "One-sample z test"
    };
}

function pValueFromT(
    tStatistic,
    degreesOfFreedom,
    alternative = "two-sided"
) {
    const cdf = tCDF(
        tStatistic,
        degreesOfFreedom
    );

    if (alternative === "two-sided") {
        return 2 * Math.min(cdf, 1 - cdf);
    }

    if (alternative === "greater") {
        return 1 - cdf;
    }

    if (alternative === "less") {
        return cdf;
    }

    throw new Error(
        "Alternative must be two-sided, greater, or less."
    );
}

function oneSampleTTest(
    values,
    nullMean,
    alternative = "two-sided",
    alpha = 0.05
) {
    assertNonEmpty(values);

    if (values.length < 2) {
        throw new Error("At least two observations are required.");
    }

    const tStatistic =
        (mean(values) - nullMean)
        / standardError(values);

    const degreesOfFreedom = values.length - 1;

    const pValue = pValueFromT(
        tStatistic,
        degreesOfFreedom,
        alternative
    );

    return {
        statistic: tStatistic,
        pValue,
        alpha,
        rejectNull: pValue < alpha,
        alternative,
        degreesOfFreedom,
        method: "One-sample t test"
    };
}


// ============================================================================
// 8. WELCH TWO-SAMPLE TEST
// ============================================================================

function welchTTest(
    groupA,
    groupB,
    alternative = "two-sided",
    alpha = 0.05
) {
    assertNonEmpty(groupA, "groupA");
    assertNonEmpty(groupB, "groupB");

    if (groupA.length < 2 || groupB.length < 2) {
        throw new Error("Both groups require at least two observations.");
    }

    const varianceA = sampleVariance(groupA);
    const varianceB = sampleVariance(groupB);

    const termA = varianceA / groupA.length;
    const termB = varianceB / groupB.length;

    const standardErrorDifference =
        Math.sqrt(termA + termB);

    const tStatistic =
        (mean(groupA) - mean(groupB))
        / standardErrorDifference;

    const numerator =
        (termA + termB) ** 2;

    const denominator =
        termA ** 2 / (groupA.length - 1)
        + termB ** 2 / (groupB.length - 1);

    const degreesOfFreedom =
        numerator / denominator;

    const usableDegreesOfFreedom =
        Math.max(1, Math.round(degreesOfFreedom));

    const pValue = pValueFromT(
        tStatistic,
        usableDegreesOfFreedom,
        alternative
    );

    return {
        statistic: tStatistic,
        pValue,
        alpha,
        rejectNull: pValue < alpha,
        alternative,
        degreesOfFreedom: usableDegreesOfFreedom,
        method: "Welch two-sample t test"
    };
}


// ============================================================================
// 9. EFFECT SIZE
// ============================================================================

function cohensD(groupA, groupB) {
    const n1 = groupA.length;
    const n2 = groupB.length;

    const pooledStandardDeviation = Math.sqrt(
        (
            (n1 - 1) * sampleVariance(groupA)
            + (n2 - 1) * sampleVariance(groupB)
        )
        / (n1 + n2 - 2)
    );

    if (pooledStandardDeviation === 0) {
        return 0;
    }

    return (
        mean(groupA) - mean(groupB)
    ) / pooledStandardDeviation;
}


// ============================================================================
// 10. PROPORTION INFERENCE
// ============================================================================

function proportionConfidenceInterval(
    successes,
    trials,
    confidenceLevel = 0.95
) {
    if (
        !Number.isInteger(successes)
        || !Number.isInteger(trials)
        || trials <= 0
        || successes < 0
        || successes > trials
    ) {
        throw new Error("Invalid binomial counts.");
    }

    const pHat = successes / trials;
    const zCritical = normalQuantile(
        1 - (1 - confidenceLevel) / 2
    );

    const standardErrorProportion =
        Math.sqrt(
            pHat * (1 - pHat) / trials
        );

    const margin =
        zCritical * standardErrorProportion;

    return {
        estimate: pHat,
        lower: Math.max(0, pHat - margin),
        upper: Math.min(1, pHat + margin),
        confidenceLevel,
        method: "Wald proportion confidence interval"
    };
}

function oneProportionZTest(
    successes,
    trials,
    nullProportion,
    alternative = "two-sided",
    alpha = 0.05
) {
    const observedProportion = successes / trials;

    const standardErrorUnderNull =
        Math.sqrt(
            nullProportion
            * (1 - nullProportion)
            / trials
        );

    const z =
        (
            observedProportion
            - nullProportion
        ) / standardErrorUnderNull;

    const pValue = pValueFromZ(
        z,
        alternative
    );

    return {
        statistic: z,
        pValue,
        alpha,
        rejectNull: pValue < alpha,
        alternative,
        method: "One-proportion z test"
    };
}


// ============================================================================
// 11. BOOTSTRAP
// ============================================================================

function percentile(sortedValues, probability) {
    if (
        probability < 0
        || probability > 1
        || sortedValues.length === 0
    ) {
        throw new Error("Invalid percentile request.");
    }

    const position =
        (sortedValues.length - 1) * probability;

    const lower = Math.floor(position);
    const upper = Math.ceil(position);

    if (lower === upper) {
        return sortedValues[lower];
    }

    const weight = position - lower;

    return (
        sortedValues[lower] * (1 - weight)
        + sortedValues[upper] * weight
    );
}

function bootstrapMeanConfidenceInterval(
    values,
    confidenceLevel = 0.95,
    repetitions = 5000,
    seed = 42
) {
    assertNonEmpty(values);

    const rng = new SeededRandom(seed);
    const estimates = [];

    for (let i = 0; i < repetitions; i++) {
        const resample = bootstrapResample(values, rng);
        estimates.push(mean(resample));
    }

    estimates.sort((a, b) => a - b);

    const alpha = 1 - confidenceLevel;

    return {
        estimate: mean(values),
        lower: percentile(estimates, alpha / 2),
        upper: percentile(estimates, 1 - alpha / 2),
        confidenceLevel,
        method: "Percentile bootstrap confidence interval"
    };
}


// ============================================================================
// 12. PERMUTATION TEST
// ============================================================================

function permutationTestDifferenceInMeans(
    groupA,
    groupB,
    repetitions = 5000,
    seed = 42
) {
    assertNonEmpty(groupA, "groupA");
    assertNonEmpty(groupB, "groupB");

    const rng = new SeededRandom(seed);

    const observedDifference =
        mean(groupA) - mean(groupB);

    const pooled = [...groupA, ...groupB];

    let extremeCount = 0;

    for (let i = 0; i < repetitions; i++) {
        const shuffled = rng.shuffle(pooled);

        const permutedA =
            shuffled.slice(0, groupA.length);

        const permutedB =
            shuffled.slice(groupA.length);

        const difference =
            mean(permutedA) - mean(permutedB);

        if (
            Math.abs(difference)
            >= Math.abs(observedDifference)
        ) {
            extremeCount++;
        }
    }

    return (
        (extremeCount + 1)
        / (repetitions + 1)
    );
}


// ============================================================================
// 13. CENTRAL LIMIT THEOREM SIMULATION
// ============================================================================

function centralLimitTheoremSimulation(
    population,
    sampleSize,
    repetitions,
    seed = 42
) {
    const rng = new SeededRandom(seed);
    const sampleMeans = [];

    for (let i = 0; i < repetitions; i++) {
        const sample = Array.from(
            { length: sampleSize },
            () => rng.choice(population)
        );

        sampleMeans.push(mean(sample));
    }

    return {
        populationMean: mean(population),
        sampleMeansMean: mean(sampleMeans),
        sampleMeansSD: sampleStandardDeviation(sampleMeans),
        theoreticalSE:
            sampleStandardDeviation(population)
            / Math.sqrt(sampleSize)
    };
}


// ============================================================================
// 14. POWER SIMULATION
// ============================================================================

function estimatePower({
    trueMean,
    nullMean,
    populationStandardDeviation,
    sampleSize,
    alpha = 0.05,
    repetitions = 3000,
    seed = 42
}) {
    const rng = new SeededRandom(seed);

    const criticalValue =
        normalQuantile(1 - alpha / 2);

    let rejections = 0;

    for (let i = 0; i < repetitions; i++) {
        const sample = Array.from(
            { length: sampleSize },
            () => rng.normal(
                trueMean,
                populationStandardDeviation
            )
        );

        const z =
            (
                mean(sample)
                - nullMean
            )
            / (
                populationStandardDeviation
                / Math.sqrt(sampleSize)
            );

        if (Math.abs(z) > criticalValue) {
            rejections++;
        }
    }

    return rejections / repetitions;
}


// ============================================================================
// 15. MULTIPLE COMPARISONS
// ============================================================================

function bonferroniCorrection(
    pValues,
    familywiseAlpha = 0.05
) {
    const numberOfTests = pValues.length;

    return pValues.map(pValue => {
        const adjustedPValue = Math.min(
            1,
            pValue * numberOfTests
        );

        return {
            originalPValue: pValue,
            adjustedPValue,
            rejectNull:
                adjustedPValue < familywiseAlpha
        };
    });
}


// ============================================================================
// 16. PRACTICAL CASE STUDY
// ============================================================================

function productQualityCaseStudy() {
    /*
     * A production team historically reports an average processing time of
     * 50 minutes. A process improvement is introduced.
     *
     * The sample is treated as a random sample from the production process.
     *
     * The analysis asks:
     *
     * H0: μ = 50
     * HA: μ < 50
     *
     * The direction matters because the business question is whether
     * processing became faster.
     */
    const processingTimes = [
        46, 48, 51, 47, 45,
        49, 52, 50, 44, 47,
        46, 48, 45, 49, 47,
        43, 51, 46, 48, 44,
        45, 47, 46, 49, 45
    ];

    console.log("\n=== Product Quality Case Study ===");

    console.log("Sample size:", processingTimes.length);
    console.log("Sample mean:", mean(processingTimes).toFixed(4));
    console.log(
        "Sample standard deviation:",
        sampleStandardDeviation(processingTimes).toFixed(4)
    );
    console.log(
        "Standard error:",
        standardError(processingTimes).toFixed(4)
    );

    console.log(
        "95% t confidence interval:",
        meanConfidenceIntervalT(processingTimes)
    );

    console.log(
        "One-sample t test:",
        oneSampleTTest(
            processingTimes,
            50,
            "less",
            0.05
        )
    );

    console.log(
        "Bootstrap interval:",
        bootstrapMeanConfidenceInterval(
            processingTimes,
            0.95,
            3000,
            42
        )
    );
}


// ============================================================================
// 17. ERROR HANDLING EXAMPLE
// ============================================================================

function demonstrateValidation() {
    console.log("\n=== Validation ===");

    try {
        mean([]);
    } catch (error) {
        console.log("Empty sample rejected:", error.message);
    }

    try {
        sampleVariance([10]);
    } catch (error) {
        console.log("Single-value variance rejected:", error.message);
    }

    try {
        oneProportionZTest(120, 100, 0.5);
    } catch (error) {
        console.log(
            "Invalid proportion input detected:",
            error.message
        );
    }
}


// ============================================================================
// 18. MAIN
// ============================================================================

function main() {
    console.log("=".repeat(78));
    console.log("STATISTICAL INFERENCE: JAVASCRIPT STUDY");
    console.log("=".repeat(78));

    const sample = [
        18, 21, 19, 23, 20,
        22, 24, 17, 21, 19
    ];

    console.log("\n=== Descriptive Statistics ===");
    console.log("Sample:", sample);
    console.log("Mean:", mean(sample));
    console.log(
        "Sample variance:",
        sampleVariance(sample)
    );
    console.log(
        "Sample SD:",
        sampleStandardDeviation(sample)
    );
    console.log(
        "Standard error:",
        standardError(sample)
    );

    console.log("\n=== Confidence Intervals ===");

    console.log(
        meanConfidenceIntervalKnownSigma(
            sample,
            3
        )
    );

    console.log(
        meanConfidenceIntervalT(sample)
    );

    console.log(
        proportionConfidenceInterval(
            62,
            100
        )
    );

    console.log("\n=== Hypothesis Tests ===");

    console.log(
        oneSampleZTest(
            sample,
            20,
            3
        )
    );

    console.log(
        oneSampleTTest(
            sample,
            20
        )
    );

    const groupA = [
        72, 75, 71, 74,
        78, 76, 73, 77
    ];

    const groupB = [
        68, 70, 69, 71,
        67, 72, 70, 68
    ];

    console.log(
        welchTTest(groupA, groupB)
    );

    console.log(
        "Cohen's d:",
        cohensD(groupA, groupB)
    );

    console.log("\n=== Proportion Test ===");

    console.log(
        oneProportionZTest(
            56,
            100,
            0.50
        )
    );

    console.log("\n=== Bootstrap ===");

    console.log(
        bootstrapMeanConfidenceInterval(
            sample,
            0.95,
            3000,
            42
        )
    );

    console.log("\n=== Permutation Test ===");

    console.log(
        "Permutation p-value:",
        permutationTestDifferenceInMeans(
            groupA,
            groupB,
            3000,
            42
        )
    );

    console.log("\n=== Central Limit Theorem ===");

    const skewedPopulation = [
        ...Array(80).fill(1),
        ...Array(15).fill(10),
        ...Array(5).fill(100)
    ];

    console.log(
        centralLimitTheoremSimulation(
            skewedPopulation,
            30,
            2000,
            42
        )
    );

    console.log("\n=== Statistical Power ===");

    console.log(
        "Estimated power:",
        estimatePower({
            trueMean: 103,
            nullMean: 100,
            populationStandardDeviation: 15,
            sampleSize: 100,
            repetitions: 2000,
            seed: 42
        })
    );

    console.log("\n=== Multiple Testing ===");

    console.table(
        bonferroniCorrection([
            0.001,
            0.012,
            0.021,
            0.04,
            0.20
        ])
    );

    productQualityCaseStudy();
    demonstrateValidation();

    console.log("\nKey interpretation rules:");
    console.log(
        "1. A p-value is not P(H0 is true)."
    );
    console.log(
        "2. Statistical significance does not measure effect size."
    );
    console.log(
        "3. Failure to reject H0 does not prove H0."
    );
    console.log(
        "4. Confidence level describes the long-run behavior of the interval procedure."
    );
    console.log(
        "5. Random sampling supports population generalization; random assignment supports causal inference."
    );

    console.log("\n" + "=".repeat(78));
    console.log("END OF JAVASCRIPT STATISTICAL INFERENCE STUDY");
    console.log("=".repeat(78));
}

main();
