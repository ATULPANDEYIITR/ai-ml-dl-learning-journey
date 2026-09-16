/*
 * Statistics for AI
 * ==================
 * Population, sample, mean, median, mode, variance, and standard deviation.
 *
 * This file uses modern JavaScript and has no external dependencies.
 * Run with:
 *     node statistics_for_ai.js
 *
 * The examples progress from basic descriptive statistics to streaming
 * statistics and an AI-oriented model-monitoring case study.
 */


// ---------------------------------------------------------------------------
// 1. VALIDATION UTILITIES
// ---------------------------------------------------------------------------

function validateNumericData(values, allowEmpty = false) {
    if (!Array.isArray(values)) {
        throw new TypeError("Data must be an array.");
    }

    if (!allowEmpty && values.length === 0) {
        throw new RangeError("Data cannot be empty.");
    }

    for (const value of values) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new TypeError(
                "Every observation must be a finite JavaScript number."
            );
        }
    }
}


// ---------------------------------------------------------------------------
// 2. MEAN
// ---------------------------------------------------------------------------

function mean(values) {
    validateNumericData(values);

    return values.reduce(
        (total, value) => total + value,
        0
    ) / values.length;
}

console.log("Mean:", mean([10, 20, 30, 40, 50]));


// ---------------------------------------------------------------------------
// 3. MEDIAN
// ---------------------------------------------------------------------------

function median(values) {
    validateNumericData(values);

    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 1) {
        return sorted[middle];
    }

    return (sorted[middle - 1] + sorted[middle]) / 2;
}

console.log("Median:", median([5, 1, 3, 2, 4]));
console.log("Even-count median:", median([1, 2, 3, 4]));


// ---------------------------------------------------------------------------
// 4. MODE
// ---------------------------------------------------------------------------
//
// JavaScript Map is useful for frequency counting.
//
// The function returns every mode when multiple values have the same highest
// frequency. If all observations occur once, the dataset has no repeated mode.

function modes(values) {
    validateNumericData(values);

    const frequencies = new Map();

    for (const value of values) {
        frequencies.set(
            value,
            (frequencies.get(value) ?? 0) + 1
        );
    }

    const highestFrequency = Math.max(...frequencies.values());

    if (highestFrequency === 1) {
        return [];
    }

    return [...frequencies.entries()]
        .filter(([, frequency]) => frequency === highestFrequency)
        .map(([value]) => value)
        .sort((a, b) => a - b);
}

console.log("Modes:", modes([1, 2, 2, 3, 3, 4]));


// ---------------------------------------------------------------------------
// 5. POPULATION VARIANCE
// ---------------------------------------------------------------------------

function populationVariance(values) {
    validateNumericData(values);

    const average = mean(values);

    return values.reduce(
        (sum, value) => sum + (value - average) ** 2,
        0
    ) / values.length;
}


// ---------------------------------------------------------------------------
// 6. SAMPLE VARIANCE
// ---------------------------------------------------------------------------
//
// Bessel's correction uses n - 1.
//
// This estimates the population variance from a sample.

function sampleVariance(values) {
    validateNumericData(values);

    if (values.length < 2) {
        throw new RangeError(
            "Sample variance requires at least two observations."
        );
    }

    const average = mean(values);

    return values.reduce(
        (sum, value) => sum + (value - average) ** 2,
        0
    ) / (values.length - 1);
}


// ---------------------------------------------------------------------------
// 7. STANDARD DEVIATION
// ---------------------------------------------------------------------------

function populationStandardDeviation(values) {
    return Math.sqrt(populationVariance(values));
}

function sampleStandardDeviation(values) {
    return Math.sqrt(sampleVariance(values));
}


const examScores = [60, 70, 80, 90, 100];

console.log("\nExam statistics");
console.log("----------------");
console.log("Mean:", mean(examScores));
console.log("Median:", median(examScores));
console.log("Mode:", modes(examScores));
console.log("Population variance:", populationVariance(examScores));
console.log(
    "Population standard deviation:",
    populationStandardDeviation(examScores)
);
console.log("Sample variance:", sampleVariance(examScores));
console.log(
    "Sample standard deviation:",
    sampleStandardDeviation(examScores)
);


// ---------------------------------------------------------------------------
// 8. DESCRIPTIVE STATISTICS OBJECT
// ---------------------------------------------------------------------------

function describe(values) {
    validateNumericData(values);

    const result = {
        count: values.length,
        minimum: Math.min(...values),
        maximum: Math.max(...values),
        range: Math.max(...values) - Math.min(...values),
        mean: mean(values),
        median: median(values),
        modes: modes(values),
        populationVariance: populationVariance(values),
        populationStandardDeviation:
            populationStandardDeviation(values)
    };

    if (values.length >= 2) {
        result.sampleVariance = sampleVariance(values);
        result.sampleStandardDeviation =
            sampleStandardDeviation(values);
    } else {
        result.sampleVariance = null;
        result.sampleStandardDeviation = null;
    }

    return result;
}

console.log("\nDescriptive report");
console.log("------------------");
console.log(describe([4, 5, 5, 7, 8, 10]));


// ---------------------------------------------------------------------------
// 9. OUTLIER SENSITIVITY
// ---------------------------------------------------------------------------

const ordinaryValues = [48, 49, 50, 51, 52];
const valuesWithOutlier = [48, 49, 50, 51, 500];

console.log("\nOutlier sensitivity");
console.log("-------------------");

for (const [label, values] of [
    ["Ordinary", ordinaryValues],
    ["With outlier", valuesWithOutlier]
]) {
    console.log(label);
    console.log("  mean:", mean(values));
    console.log("  median:", median(values));
    console.log(
        "  standard deviation:",
        populationStandardDeviation(values)
    );
}


// ---------------------------------------------------------------------------
// 10. Z-SCORE STANDARDIZATION
// ---------------------------------------------------------------------------
//
// z = (x - mean) / standard deviation
//
// Standardization puts observations onto a common scale. It is frequently
// useful in machine-learning preprocessing, though the appropriate scaler
// depends on the model and data distribution.

function zScores(values) {
    validateNumericData(values);

    const average = mean(values);
    const standardDeviation = populationStandardDeviation(values);

    if (standardDeviation === 0) {
        throw new RangeError(
            "Z-scores are undefined when all values are identical."
        );
    }

    return values.map(
        value => (value - average) / standardDeviation
    );
}

const temperatures = [10, 20, 30, 40, 50];
const standardizedTemperatures = zScores(temperatures);

console.log("\nZ-score standardization");
console.log("-----------------------");
console.log("Original:", temperatures);
console.log("Z-scores:", standardizedTemperatures);


// ---------------------------------------------------------------------------
// 11. JAVASCRIPT-SPECIFIC SORTING PITFALL
// ---------------------------------------------------------------------------
//
// Array.prototype.sort() converts elements to strings unless a comparator is
// supplied. Therefore [2, 10, 3].sort() can produce [10, 2, 3].
//
// Statistical numeric sorting should use (a, b) => a - b.

console.log("\nNumeric sorting");
console.log("----------------");
console.log("[2, 10, 3].sort() ->", [2, 10, 3].sort());
console.log(
    "[2, 10, 3].sort((a, b) => a - b) ->",
    [2, 10, 3].sort((a, b) => a - b)
);


// ---------------------------------------------------------------------------
// 12. ONLINE / STREAMING STATISTICS
// ---------------------------------------------------------------------------
//
// For very large AI data streams, calculating statistics after storing every
// observation may be unnecessary.
//
// Welford's algorithm maintains:
//   count
//   mean
//   M2
//
// It provides stable incremental variance calculations.

class OnlineStatistics {
    constructor() {
        this.count = 0;
        this.meanValue = 0;
        this.m2 = 0;
    }

    update(value) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new TypeError("Value must be a finite number.");
        }

        this.count += 1;

        const delta = value - this.meanValue;
        this.meanValue += delta / this.count;
        const deltaAfterUpdate = value - this.meanValue;

        this.m2 += delta * deltaAfterUpdate;
    }

    get mean() {
        if (this.count === 0) {
            throw new RangeError("No observations available.");
        }

        return this.meanValue;
    }

    get populationVariance() {
        if (this.count === 0) {
            throw new RangeError("No observations available.");
        }

        return this.m2 / this.count;
    }

    get sampleVariance() {
        if (this.count < 2) {
            throw new RangeError(
                "At least two observations are required."
            );
        }

        return this.m2 / (this.count - 1);
    }

    get populationStandardDeviation() {
        return Math.sqrt(this.populationVariance);
    }

    get sampleStandardDeviation() {
        return Math.sqrt(this.sampleVariance);
    }
}

const stream = new OnlineStatistics();

for (const value of [10, 12, 14, 16, 18]) {
    stream.update(value);
}

console.log("\nStreaming statistics");
console.log("--------------------");
console.log("Count:", stream.count);
console.log("Mean:", stream.mean);
console.log("Population variance:", stream.populationVariance);
console.log("Sample variance:", stream.sampleVariance);
console.log(
    "Population standard deviation:",
    stream.populationStandardDeviation
);


// ---------------------------------------------------------------------------
// 13. ASYNCHRONOUS AI-METRIC STREAM
// ---------------------------------------------------------------------------
//
// JavaScript is often used in event-driven systems. This example shows how
// observations could arrive asynchronously from a service, sensor, browser
// event, or model-monitoring pipeline.

async function* simulatedMetricStream(values, delayMilliseconds = 10) {
    for (const value of values) {
        await new Promise(
            resolve => setTimeout(resolve, delayMilliseconds)
        );
        yield value;
    }
}

async function analyzeMetricStream(values) {
    const statistics = new OnlineStatistics();

    for await (const value of simulatedMetricStream(values)) {
        statistics.update(value);
    }

    return {
        count: statistics.count,
        mean: statistics.mean,
        standardDeviation:
            statistics.populationStandardDeviation
    };
}


// ---------------------------------------------------------------------------
// 14. AI MODEL MONITORING CASE STUDY
// ---------------------------------------------------------------------------
//
// Suppose a classification service emits prediction errors over time.
// Mean error indicates average magnitude, while standard deviation indicates
// variability.
//
// A monitoring system can compare these descriptive measurements against
// previously established operational thresholds.

async function runModelMonitoringExample() {
    const productionErrors = [
        0.08, 0.10, 0.09, 0.11, 0.12,
        0.09, 0.10, 0.14, 0.13, 0.08
    ];

    const report = await analyzeMetricStream(productionErrors);

    console.log("\nAI model monitoring");
    console.log("-------------------");
    console.log("Observations:", report.count);
    console.log("Mean error:", report.mean);
    console.log("Error standard deviation:", report.standardDeviation);

    // These thresholds are illustrative operational values, not universal
    // definitions of acceptable model quality.
    const meanErrorThreshold = 0.15;
    const standardDeviationThreshold = 0.05;

    const meanStatus =
        report.mean <= meanErrorThreshold ? "within threshold" : "above threshold";

    const variabilityStatus =
        report.standardDeviation <= standardDeviationThreshold
            ? "within threshold"
            : "above threshold";

    console.log("Mean status:", meanStatus);
    console.log("Variability status:", variabilityStatus);
}

runModelMonitoringExample().catch(error => {
    console.error("Monitoring error:", error.message);
});


// ---------------------------------------------------------------------------
// 15. GROUPED MODEL ERROR ANALYSIS
// ---------------------------------------------------------------------------
//
// Similar average errors can conceal different variability patterns.

const modelErrors = {
    groupA: [0.10, 0.12, 0.09, 0.11, 0.10],
    groupB: [0.10, 0.30, 0.08, 0.25, 0.12]
};

console.log("\nGrouped error analysis");
console.log("----------------------");

for (const [group, errors] of Object.entries(modelErrors)) {
    console.log(group, {
        mean: mean(errors),
        median: median(errors),
        standardDeviation:
            populationStandardDeviation(errors)
    });
}


// ---------------------------------------------------------------------------
// 16. ERROR HANDLING AND EDGE CASES
// ---------------------------------------------------------------------------

function demonstrateErrors() {
    const cases = [
        () => mean([]),
        () => median([]),
        () => populationVariance([]),
        () => sampleVariance([10]),
        () => mean([1, Number.NaN]),
        () => mean([1, Infinity]),
        () => zScores([5, 5, 5])
    ];

    console.log("\nExpected error cases");
    console.log("--------------------");

    for (const operation of cases) {
        try {
            operation();
        } catch (error) {
            console.log(`${error.constructor.name}: ${error.message}`);
        }
    }
}

demonstrateErrors();


// ---------------------------------------------------------------------------
// 17. NUMERICAL PRECISION
// ---------------------------------------------------------------------------
//
// JavaScript uses IEEE 754 double-precision floating-point numbers for its
// ordinary Number type. Many decimal fractions therefore have approximate
// binary representations.

console.log("\nFloating-point precision");
console.log("------------------------");
console.log("0.1 + 0.2 =", 0.1 + 0.2);
console.log(
    "Approximate comparison:",
    Math.abs((0.1 + 0.2) - 0.3) < Number.EPSILON
);


// ---------------------------------------------------------------------------
// 18. BASIC STATISTICAL TESTS
// ---------------------------------------------------------------------------

function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

function approximatelyEqual(a, b, tolerance = 1e-12) {
    return Math.abs(a - b) <= tolerance;
}

function runTests() {
    const values = [1, 2, 3, 4, 5];

    assert(
        mean(values) === 3,
        "Mean should equal 3."
    );

    assert(
        median(values) === 3,
        "Median should equal 3."
    );

    assert(
        modes(values).length === 0,
        "There should be no repeated mode."
    );

    assert(
        populationVariance(values) === 2,
        "Population variance should equal 2."
    );

    assert(
        sampleVariance(values) === 2.5,
        "Sample variance should equal 2.5."
    );

    assert(
        approximatelyEqual(
            populationStandardDeviation(values),
            Math.sqrt(2)
        ),
        "Population standard deviation is incorrect."
    );

    assert(
        median([1, 2, 3, 4]) === 2.5,
        "Even median is incorrect."
    );

    assert(
        JSON.stringify(modes([1, 1, 2, 2, 3])) === "[1,2]",
        "Multiple modes are incorrect."
    );

    const constantData = [7, 7, 7];
    assert(
        populationVariance(constantData) === 0,
        "Constant variance should be zero."
    );

    const online = new OnlineStatistics();
    for (const value of values) {
        online.update(value);
    }

    assert(
        approximatelyEqual(online.mean, 3),
        "Online mean is incorrect."
    );

    assert(
        approximatelyEqual(online.populationVariance, 2),
        "Online population variance is incorrect."
    );

    console.log("\nAll JavaScript tests passed.");
}

runTests();


// ---------------------------------------------------------------------------
// 19. PRACTICAL INTERPRETATION
// ---------------------------------------------------------------------------
//
// Mean:
//   Arithmetic center, sensitive to extreme values.
//
// Median:
//   Middle ordered observation, generally less sensitive to outliers.
//
// Mode:
//   Most frequent observation(s).
//
// Population variance:
//   Average squared deviation when the dataset is the complete population.
//
// Sample variance:
//   n - 1 corrected estimator used when estimating population variance from
//   a sample.
//
// Standard deviation:
//   Square root of variance, expressed in the original unit.
//
// In AI:
//   These measurements support exploratory analysis, preprocessing,
//   monitoring, anomaly investigation, experiment analysis, and model
//   diagnostics.
//
// They do not by themselves describe skewness, tails, multimodality,
// correlations, causality, or the complete probability distribution.
