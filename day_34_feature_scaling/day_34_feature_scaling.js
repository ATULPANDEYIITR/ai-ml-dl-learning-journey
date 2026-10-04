/**
 * Feature Scaling: Standardization, Normalization, Robust Scaling,
 * and Transformations.
 *
 * Self-contained Node.js implementation with no external dependencies.
 *
 * The program demonstrates JavaScript-specific representations of:
 * - Standardization using mean and standard deviation
 * - Min-max normalization
 * - Robust median/IQR scaling
 * - Max-absolute scaling
 * - Logarithmic and signed-log transformations
 * - Quantile/rank transformation
 * - Event-driven preprocessing
 * - Training-only parameter fitting
 * - Data leakage detection
 * - Distance effects
 * - A composable preprocessing pipeline
 * - Asynchronous inference using Promise-based events
 */

"use strict";

// -----------------------------------------------------------------------------
// Validation and numerical helpers
// -----------------------------------------------------------------------------

function assertFiniteNumber(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new TypeError(`${name} must be a finite number.`);
    }
}

function validateMatrix(matrix, name = "matrix") {
    if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new Error(`${name} must contain at least one row.`);
    }

    if (!Array.isArray(matrix[0]) || matrix[0].length === 0) {
        throw new Error(`${name} must contain at least one feature.`);
    }

    const width = matrix[0].length;

    matrix.forEach((row, rowIndex) => {
        if (!Array.isArray(row) || row.length !== width) {
            throw new Error(
                `${name} is not rectangular at row ${rowIndex}.`
            );
        }

        row.forEach((value, columnIndex) => {
            assertFiniteNumber(
                value,
                `${name}[${rowIndex}][${columnIndex}]`
            );
        });
    });
}

function getColumn(matrix, index) {
    return matrix.map(row => row[index]);
}

function mean(values) {
    if (values.length === 0) {
        throw new Error("Cannot calculate the mean of an empty array.");
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function populationStd(values) {
    const average = mean(values);
    const variance = mean(
        values.map(value => (value - average) ** 2)
    );
    return Math.sqrt(variance);
}

function median(values) {
    if (values.length === 0) {
        throw new Error("Cannot calculate the median of an empty array.");
    }

    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[middle - 1] + sorted[middle]) / 2;
    }

    return sorted[middle];
}

function quantile(values, probability) {
    if (values.length === 0) {
        throw new Error("Cannot calculate a quantile of an empty array.");
    }

    if (probability < 0 || probability > 1) {
        throw new RangeError("Quantile probability must be in [0, 1].");
    }

    const sorted = [...values].sort((a, b) => a - b);

    if (sorted.length === 1) {
        return sorted[0];
    }

    const position = (sorted.length - 1) * probability;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);

    if (lower === upper) {
        return sorted[lower];
    }

    const fraction = position - lower;
    return (
        sorted[lower] +
        fraction * (sorted[upper] - sorted[lower])
    );
}

function cloneMatrix(matrix) {
    return matrix.map(row => [...row]);
}

function formatMatrix(matrix, digits = 4) {
    return matrix
        .map(row => row.map(value => value.toFixed(digits)).join("  "))
        .join("\n");
}

function euclideanDistance(a, b) {
    if (a.length !== b.length) {
        throw new Error("Distance vectors must have equal length.");
    }

    return Math.sqrt(
        a.reduce(
            (sum, value, index) => sum + (value - b[index]) ** 2,
            0
        )
    );
}

// -----------------------------------------------------------------------------
// Event-driven workflow model
// -----------------------------------------------------------------------------

class EventBus {
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
        const listeners = this.listeners.get(eventName) || [];

        for (const listener of listeners) {
            listener(payload);
        }
    }
}

// -----------------------------------------------------------------------------
// Standardization
// -----------------------------------------------------------------------------

class StandardScaler {
    constructor() {
        this.means = null;
        this.scales = null;
        this.featureCount = null;
    }

    fit(matrix) {
        validateMatrix(matrix, "training matrix");

        this.featureCount = matrix[0].length;
        this.means = [];
        this.scales = [];

        for (let columnIndex = 0; columnIndex < this.featureCount; columnIndex++) {
            const values = getColumn(matrix, columnIndex);
            const featureMean = mean(values);
            const standardDeviation = populationStd(values);

            this.means.push(featureMean);

            // Constant features cannot be divided by zero. A unit fallback
            // leaves their transformed value at zero.
            this.scales.push(
                standardDeviation === 0 ? 1 : standardDeviation
            );
        }

        return this;
    }

    transform(matrix) {
        this.#requireFitted();
        this.#validateFeatureCount(matrix);

        return matrix.map(row =>
            row.map(
                (value, columnIndex) =>
                    (value - this.means[columnIndex]) /
                    this.scales[columnIndex]
            )
        );
    }

    inverseTransform(matrix) {
        this.#requireFitted();
        this.#validateFeatureCount(matrix);

        return matrix.map(row =>
            row.map(
                (value, columnIndex) =>
                    value * this.scales[columnIndex] +
                    this.means[columnIndex]
            )
        );
    }

    #requireFitted() {
        if (this.means === null) {
            throw new Error("StandardScaler must be fitted first.");
        }
    }

    #validateFeatureCount(matrix) {
        validateMatrix(matrix);

        if (matrix[0].length !== this.featureCount) {
            throw new Error(
                `Expected ${this.featureCount} features, ` +
                `received ${matrix[0].length}.`
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Min-max normalization
// -----------------------------------------------------------------------------

class MinMaxScaler {
    constructor(lower = 0, upper = 1) {
        if (lower >= upper) {
            throw new RangeError("lower must be smaller than upper.");
        }

        this.lower = lower;
        this.upper = upper;
        this.minimums = null;
        this.maximums = null;
        this.featureCount = null;
    }

    fit(matrix) {
        validateMatrix(matrix);

        this.featureCount = matrix[0].length;
        this.minimums = [];
        this.maximums = [];

        for (let index = 0; index < this.featureCount; index++) {
            const values = getColumn(matrix, index);
            this.minimums.push(Math.min(...values));
            this.maximums.push(Math.max(...values));
        }

        return this;
    }

    transform(matrix) {
        this.#requireFitted();
        this.#validateFeatureCount(matrix);

        return matrix.map(row =>
            row.map((value, index) => {
                const minimum = this.minimums[index];
                const maximum = this.maximums[index];

                if (minimum === maximum) {
                    return this.lower;
                }

                const ratio =
                    (value - minimum) / (maximum - minimum);

                return (
                    this.lower +
                    ratio * (this.upper - this.lower)
                );
            })
        );
    }

    inverseTransform(matrix) {
        this.#requireFitted();
        this.#validateFeatureCount(matrix);

        return matrix.map(row =>
            row.map((value, index) => {
                const minimum = this.minimums[index];
                const maximum = this.maximums[index];

                if (minimum === maximum) {
                    return minimum;
                }

                const ratio =
                    (value - this.lower) /
                    (this.upper - this.lower);

                return minimum + ratio * (maximum - minimum);
            })
        );
    }

    #requireFitted() {
        if (this.minimums === null) {
            throw new Error("MinMaxScaler must be fitted first.");
        }
    }

    #validateFeatureCount(matrix) {
        validateMatrix(matrix);

        if (matrix[0].length !== this.featureCount) {
            throw new Error("Feature count does not match fitted scaler.");
        }
    }
}

// -----------------------------------------------------------------------------
// Robust scaling
// -----------------------------------------------------------------------------

class RobustScaler {
    constructor(qLow = 0.25, qHigh = 0.75) {
        if (qLow < 0 || qHigh > 1 || qLow >= qHigh) {
            throw new RangeError(
                "Robust scaling quantiles must satisfy 0 <= qLow < qHigh <= 1."
            );
        }

        this.qLow = qLow;
        this.qHigh = qHigh;
        this.medians = null;
        this.iqrs = null;
        this.featureCount = null;
    }

    fit(matrix) {
        validateMatrix(matrix);

        this.featureCount = matrix[0].length;
        this.medians = [];
        this.iqrs = [];

        for (let index = 0; index < this.featureCount; index++) {
            const values = getColumn(matrix, index);
            const q1 = quantile(values, this.qLow);
            const q3 = quantile(values, this.qHigh);
            const iqr = q3 - q1;

            this.medians.push(median(values));
            this.iqrs.push(iqr === 0 ? 1 : iqr);
        }

        return this;
    }

    transform(matrix) {
        this.#requireFitted();
        this.#validateFeatureCount(matrix);

        return matrix.map(row =>
            row.map(
                (value, index) =>
                    (value - this.medians[index]) /
                    this.iqrs[index]
            )
        );
    }

    inverseTransform(matrix) {
        this.#requireFitted();
        this.#validateFeatureCount(matrix);

        return matrix.map(row =>
            row.map(
                (value, index) =>
                    value * this.iqrs[index] +
                    this.medians[index]
            )
        );
    }

    #requireFitted() {
        if (this.medians === null) {
            throw new Error("RobustScaler must be fitted first.");
        }
    }

    #validateFeatureCount(matrix) {
        validateMatrix(matrix);

        if (matrix[0].length !== this.featureCount) {
            throw new Error("Feature count does not match fitted scaler.");
        }
    }
}

// -----------------------------------------------------------------------------
// Transformations
// -----------------------------------------------------------------------------

function logTransform(values) {
    return values.map(value => {
        if (value <= 0) {
            throw new RangeError(
                "Natural logarithm requires strictly positive values."
            );
        }

        return Math.log(value);
    });
}

function signedLogTransform(values) {
    return values.map(value => {
        if (value === 0) {
            return 0;
        }

        return Math.sign(value) * Math.log1p(Math.abs(value));
    });
}

function sqrtTransform(values) {
    return values.map(value => {
        if (value < 0) {
            throw new RangeError(
                "Square-root transformation requires non-negative values."
            );
        }

        return Math.sqrt(value);
    });
}

function yeoJohnsonStyleTransform(values, lambda) {
    return values.map(x => {
        if (x >= 0) {
            if (Math.abs(lambda) < 1e-12) {
                return Math.log1p(x);
            }

            return ((x + 1) ** lambda - 1) / lambda;
        }

        const exponent = 2 - lambda;

        if (Math.abs(exponent) < 1e-12) {
            return -Math.log1p(-x);
        }

        return -(((-x + 1) ** exponent - 1) / exponent);
    });
}

function rankToUniform(values) {
    const indexed = values.map((value, index) => ({
        value,
        index
    }));

    indexed.sort((a, b) => a.value - b.value);

    const result = Array(values.length).fill(0);
    const denominator = Math.max(values.length - 1, 1);

    let start = 0;

    while (start < indexed.length) {
        let end = start + 1;

        while (
            end < indexed.length &&
            indexed[end].value === indexed[start].value
        ) {
            end++;
        }

        const averageRank = (start + end - 1) / 2;

        for (let position = start; position < end; position++) {
            result[indexed[position].index] =
                averageRank / denominator;
        }

        start = end;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Pipeline
// -----------------------------------------------------------------------------

class FeaturePipeline {
    constructor({ columnTransforms = new Map(), scaler }) {
        this.columnTransforms = columnTransforms;
        this.scaler = scaler;
    }

    fit(matrix) {
        const transformed = this.#applyColumnTransforms(matrix);
        this.scaler.fit(transformed);
        return this;
    }

    transform(matrix) {
        const transformed = this.#applyColumnTransforms(matrix);
        return this.scaler.transform(transformed);
    }

    #applyColumnTransforms(matrix) {
        validateMatrix(matrix);

        const result = cloneMatrix(matrix);

        for (const [columnIndex, transform] of this.columnTransforms) {
            if (
                columnIndex < 0 ||
                columnIndex >= matrix[0].length
            ) {
                throw new RangeError(
                    `Transformation references feature ${columnIndex}.`
                );
            }

            const transformedColumn = transform(
                getColumn(matrix, columnIndex)
            );

            transformedColumn.forEach((value, rowIndex) => {
                result[rowIndex][columnIndex] = value;
            });
        }

        return result;
    }
}

// -----------------------------------------------------------------------------
// Training-only parameter discipline
// -----------------------------------------------------------------------------

class TrainingPreprocessor {
    constructor(scaler) {
        this.scaler = scaler;
        this.fittedOnRows = 0;
    }

    fit(trainingData) {
        this.scaler.fit(trainingData);
        this.fittedOnRows = trainingData.length;
        return this;
    }

    transformProductionData(data) {
        if (this.fittedOnRows === 0) {
            throw new Error(
                "Production transformation attempted before training fit."
            );
        }

        return this.scaler.transform(data);
    }
}

// -----------------------------------------------------------------------------
// Asynchronous preprocessing service
// -----------------------------------------------------------------------------

class PreprocessingService {
    constructor() {
        this.events = new EventBus();
        this.scaler = null;
    }

    async train(trainingData, scaler) {
        this.events.emit("fit:start", {
            rows: trainingData.length
        });

        await Promise.resolve();

        this.scaler = scaler.fit(trainingData);

        this.events.emit("fit:complete", {
            rows: trainingData.length,
            featureCount: trainingData[0].length
        });

        return this;
    }

    async transform(data) {
        if (!this.scaler) {
            throw new Error(
                "Cannot transform data before preprocessing is fitted."
            );
        }

        this.events.emit("transform:start", {
            rows: data.length
        });

        // The Promise boundary represents an asynchronous inference service.
        // The scaling itself is CPU-local, but a production preprocessing
        // service might obtain batches from a queue, API, or stream.
        await Promise.resolve();

        const result = this.scaler.transform(data);

        this.events.emit("transform:complete", {
            rows: result.length
        });

        return result;
    }
}

// -----------------------------------------------------------------------------
// Realistic customer data
// -----------------------------------------------------------------------------

function customerData() {
    return [
        [28000, 8, 1200, 14],
        [35000, 12, 1450, 20],
        [42000, 15, 1700, 28],
        [48000, 20, 2100, 35],
        [55000, 24, 2300, 42],
        [63000, 31, 2600, 51],
        [75000, 38, 3000, 66],
        [95000, 47, 3500, 82],
        [140000, 75, 5100, 120],
        [850000, 110, 8900, 144]
    ];
}

// -----------------------------------------------------------------------------
// Demonstrations
// -----------------------------------------------------------------------------

function demonstrateScalers(data) {
    console.log("\nRAW CUSTOMER FEATURES");
    console.log(formatMatrix(data, 2));

    const scalers = [
        ["Standardization", new StandardScaler()],
        ["Min-max normalization", new MinMaxScaler()],
        ["Robust scaling", new RobustScaler()]
    ];

    for (const [name, scaler] of scalers) {
        const transformed = scaler.fit(data).transform(data);

        console.log(`\n${name}`);
        console.log(formatMatrix(transformed, 3));
    }
}

function demonstrateTransformations() {
    const revenue = [
        100,
        250,
        500,
        1000,
        5000,
        25000,
        250000
    ];

    const signedChanges = [
        -500,
        -100,
        -10,
        0,
        15,
        200,
        2000
    ];

    console.log("\nTRANSFORMATIONS");

    console.log(
        "Log revenue:",
        logTransform(revenue).map(value => value.toFixed(4))
    );

    console.log(
        "Signed-log changes:",
        signedLogTransform(signedChanges).map(
            value => value.toFixed(4)
        )
    );

    console.log(
        "Yeo-Johnson-style mixed feature:",
        yeoJohnsonStyleTransform(
            signedChanges,
            0.5
        ).map(value => value.toFixed(4))
    );

    console.log(
        "Rank-to-uniform revenue:",
        rankToUniform(revenue).map(
            value => value.toFixed(4)
        )
    );

    console.log(
        "Square-root transaction counts:",
        sqrtTransform([0, 1, 4, 9, 25, 100, 400])
            .map(value => value.toFixed(4))
    );
}

function demonstrateDistanceEffect(data) {
    const first = data[0];
    const second = data[1];

    console.log("\nDISTANCE EFFECT");

    console.log(
        "Raw Euclidean distance:",
        euclideanDistance(first, second).toFixed(4)
    );

    for (const [name, scaler] of [
        ["Standardized", new StandardScaler()],
        ["Min-max", new MinMaxScaler()],
        ["Robust", new RobustScaler()]
    ]) {
        const transformed = scaler.fit(data).transform(data);

        console.log(
            `${name} distance:`,
            euclideanDistance(
                transformed[0],
                transformed[1]
            ).toFixed(4)
        );
    }
}

function demonstrateOutliers() {
    const ordinary = [
        [30000],
        [32000],
        [35000],
        [37000],
        [40000],
        [42000]
    ];

    const withOutlier = [
        ...ordinary,
        [2000000]
    ];

    const standardWithout =
        new StandardScaler().fit(ordinary).transform(ordinary);

    const standardWith =
        new StandardScaler().fit(withOutlier).transform(ordinary);

    const robustWithout =
        new RobustScaler().fit(ordinary).transform(ordinary);

    const robustWith =
        new RobustScaler().fit(withOutlier).transform(ordinary);

    console.log("\nOUTLIER EFFECT");

    console.log(
        "Standard first observation without outlier:",
        standardWithout[0][0].toFixed(4)
    );

    console.log(
        "Standard first observation with outlier:",
        standardWith[0][0].toFixed(4)
    );

    console.log(
        "Robust first observation without outlier:",
        robustWithout[0][0].toFixed(4)
    );

    console.log(
        "Robust first observation with outlier:",
        robustWith[0][0].toFixed(4)
    );
}

function demonstratePipeline() {
    const training = [
        [30000, 8, 100],
        [35000, 10, 250],
        [40000, 15, 500],
        [50000, 20, 1000],
        [65000, 25, 2500],
        [80000, 32, 5000]
    ];

    const inference = [
        [55000, 18, 750],
        [100000, 40, 15000]
    ];

    const pipeline = new FeaturePipeline({
        columnTransforms: new Map([
            [
                2,
                values => values.map(value => Math.log1p(value))
            ]
        ]),
        scaler: new StandardScaler()
    });

    pipeline.fit(training);

    console.log("\nPIPELINE: LOG1P THEN STANDARDIZATION");
    console.log(
        formatMatrix(pipeline.transform(inference), 4)
    );
}

function demonstrateLeakage() {
    const training = [
        [20000, 10],
        [30000, 12],
        [40000, 14],
        [50000, 15]
    ];

    const test = [
        [55000, 16],
        [200000, 17]
    ];

    const correctScaler =
        new StandardScaler().fit(training);

    const correct =
        correctScaler.transform(test);

    const incorrectScaler =
        new StandardScaler().fit([...training, ...test]);

    const leaked =
        incorrectScaler.transform(test);

    console.log("\nTRAINING-ONLY FITTING");
    console.log("Correct transformation:");
    console.log(formatMatrix(correct, 4));

    console.log(
        "Incorrect transformation after test-set leakage:"
    );
    console.log(formatMatrix(leaked, 4));
}

function demonstrateInverseTransformation() {
    const data = [
        [10, 100],
        [20, 200],
        [30, 300]
    ];

    const scaler = new StandardScaler().fit(data);
    const scaled = scaler.transform(data);
    const restored = scaler.inverseTransform(scaled);

    console.log("\nINVERSE TRANSFORMATION");
    console.log("Scaled:");
    console.log(formatMatrix(scaled, 4));

    console.log("Restored:");
    console.log(formatMatrix(restored, 4));
}

function demonstrateEventDrivenService() {
    return (async () => {
        const service = new PreprocessingService();

        service.events.on("fit:start", payload => {
            console.log(
                `\nEVENT fit:start rows=${payload.rows}`
            );
        });

        service.events.on("fit:complete", payload => {
            console.log(
                `EVENT fit:complete features=${payload.featureCount}`
            );
        });

        service.events.on("transform:start", payload => {
            console.log(
                `EVENT transform:start rows=${payload.rows}`
            );
        });

        service.events.on("transform:complete", payload => {
            console.log(
                `EVENT transform:complete rows=${payload.rows}`
            );
        });

        const training = [
            [1000, 10],
            [2000, 20],
            [3000, 30],
            [4000, 40]
        ];

        const inference = [
            [2500, 25],
            [5000, 50]
        ];

        await service.train(
            training,
            new StandardScaler()
        );

        const result = await service.transform(inference);

        console.log("Asynchronous inference result:");
        console.log(formatMatrix(result, 4));
    })();
}

function demonstrateValidation() {
    console.log("\nVALIDATION");

    try {
        new StandardScaler().transform([[1, 2]]);
    } catch (error) {
        console.log("Unfitted scaler:", error.message);
    }

    try {
        logTransform([10, 0, 20]);
    } catch (error) {
        console.log("Invalid log input:", error.message);
    }

    try {
        sqrtTransform([4, -1]);
    } catch (error) {
        console.log("Invalid square-root input:", error.message);
    }

    try {
        validateMatrix([
            [1, 2],
            [3]
        ]);
    } catch (error) {
        console.log("Non-rectangular matrix:", error.message);
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

async function main() {
    console.log("=".repeat(78));
    console.log(
        "FEATURE SCALING: STANDARDIZATION, NORMALIZATION, ROBUST SCALING"
    );
    console.log("AND TRANSFORMATIONS");
    console.log("=".repeat(78));

    const data = customerData();

    demonstrateScalers(data);
    demonstrateTransformations();
    demonstrateDistanceEffect(data);
    demonstrateOutliers();
    demonstratePipeline();
    demonstrateLeakage();
    demonstrateInverseTransformation();
    demonstrateValidation();
    await demonstrateEventDrivenService();

    console.log("\nExecution completed successfully.");
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
