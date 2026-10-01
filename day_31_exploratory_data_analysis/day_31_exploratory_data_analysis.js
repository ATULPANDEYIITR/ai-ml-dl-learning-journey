"use strict";

/*
 * Exploratory Data Analysis
 * Univariate | Bivariate | Multivariate
 *
 * This Node.js program models an EDA workflow for a sales dataset.
 * It intentionally uses JavaScript-specific capabilities:
 * - Map-based frequency analysis
 * - Array transformations
 * - event-driven EDA stages
 * - asynchronous report generation
 * - object-oriented dataset profiling
 * - correlation and regression calculations
 * - policy-style validation
 *
 * Runtime: Node.js 18+
 */

const fs = require("node:fs/promises");
const EventEmitter = require("node:events");

// -----------------------------------------------------------------------------
// Dataset generation
// -----------------------------------------------------------------------------

class SeededRandom {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (1664525 * this.state + 1013904223) >>> 0;
        return this.state / 0x100000000;
    }

    normal(mean = 0, standardDeviation = 1) {
        let u1 = 0;
        let u2 = 0;

        while (u1 === 0) {
            u1 = this.next();
        }

        u2 = this.next();

        const magnitude = Math.sqrt(-2 * Math.log(u1));
        const z = magnitude * Math.cos(2 * Math.PI * u2);

        return mean + z * standardDeviation;
    }

    choice(values) {
        return values[Math.floor(this.next() * values.length)];
    }
}

function generateDataset(size = 160, seed = 42) {
    const rng = new SeededRandom(seed);

    const regions = ["North", "South", "East", "West"];
    const channels = ["Online", "Retail", "Partner"];
    const products = ["Alpha", "Beta", "Gamma", "Delta"];

    const basePrices = {
        Alpha: 72,
        Beta: 105,
        Gamma: 145,
        Delta: 185
    };

    const rows = [];

    for (let id = 1; id <= size; id += 1) {
        const region = rng.choice(regions);
        const channel = rng.choice(channels);
        const product = rng.choice(products);

        const marketingSpend = Math.max(
            50,
            rng.normal(950, 300)
        );

        const discountPct = Math.min(
            30,
            Math.max(0, rng.normal(10, 5))
        );

        const channelEffect = {
            Online: 30,
            Retail: 10,
            Partner: -5
        }[channel];

        const expectedUnits =
            20 +
            marketingSpend * 0.018 +
            discountPct * 1.4 +
            channelEffect;

        const unitsSold = Math.max(
            1,
            Math.round(rng.normal(expectedUnits, 8))
        );

        const deliveryDays = Math.max(
            1,
            rng.normal(
                {
                    Online: 3.5,
                    Retail: 2.5,
                    Partner: 4.5
                }[channel],
                1
            )
        );

        const customerRating = Math.min(
            5,
            Math.max(
                1,
                4.9 -
                deliveryDays * 0.22 +
                rng.normal(0, 0.45)
            )
        );

        const effectivePrice =
            basePrices[product] *
            (1 - discountPct / 100);

        const revenue = Math.max(
            0,
            unitsSold * effectivePrice +
            rng.normal(0, basePrices[product] * 2)
        );

        const returnProbability = Math.min(
            0.65,
            0.025 +
            Math.max(0, 3.4 - customerRating) * 0.08 +
            Math.max(0, deliveryDays - 4) * 0.015
        );

        rows.push({
            transactionId: id,
            region,
            channel,
            product,
            marketingSpend: round(marketingSpend),
            discountPct: round(discountPct),
            unitsSold,
            customerRating: round(customerRating),
            deliveryDays: round(deliveryDays),
            revenue: round(revenue),
            returned: rng.next() < returnProbability
        });
    }

    // Missing values make missingness profiling an actual part of EDA.
    rows[17].customerRating = null;
    rows[53].marketingSpend = null;
    rows[111].deliveryDays = null;

    // A high-value transaction creates a meaningful outlier for inspection.
    rows[99].marketingSpend = 4800;
    rows[99].unitsSold = 170;
    rows[99].revenue = 22000;

    return rows;
}

function round(value, digits = 2) {
    const factor = 10 ** digits;
    return Math.round(value * factor) / factor;
}

// -----------------------------------------------------------------------------
// Generic statistical helpers
// -----------------------------------------------------------------------------

function numericValues(rows, column) {
    return rows
        .map(row => row[column])
        .filter(value =>
            typeof value === "number" &&
            Number.isFinite(value)
        );
}

function pairedValues(rows, xColumn, yColumn) {
    return rows
        .filter(row =>
            typeof row[xColumn] === "number" &&
            Number.isFinite(row[xColumn]) &&
            typeof row[yColumn] === "number" &&
            Number.isFinite(row[yColumn])
        )
        .map(row => [row[xColumn], row[yColumn]]);
}

function mean(values) {
    if (values.length === 0) {
        throw new Error("Mean requires at least one observation.");
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
    if (values.length === 0) {
        throw new Error("Median requires at least one observation.");
    }

    const ordered = [...values].sort((a, b) => a - b);
    const middle = Math.floor(ordered.length / 2);

    return ordered.length % 2 === 0
        ? (ordered[middle - 1] + ordered[middle]) / 2
        : ordered[middle];
}

function quantile(values, probability) {
    if (values.length === 0) {
        throw new Error("Quantile requires observations.");
    }

    const ordered = [...values].sort((a, b) => a - b);
    const position = (ordered.length - 1) * probability;
    const lower = Math.floor(position);
    const upper = Math.min(lower + 1, ordered.length - 1);
    const fraction = position - lower;

    return ordered[lower] +
        (ordered[upper] - ordered[lower]) * fraction;
}

function variance(values) {
    if (values.length < 2) {
        throw new Error("Sample variance requires at least two values.");
    }

    const average = mean(values);

    return values.reduce(
        (sum, value) => sum + (value - average) ** 2,
        0
    ) / (values.length - 1);
}

function standardDeviation(values) {
    return Math.sqrt(variance(values));
}

// -----------------------------------------------------------------------------
// Univariate analysis
// -----------------------------------------------------------------------------

function summarizeNumeric(rows, column) {
    const values = numericValues(rows, column);

    const missing = rows.filter(
        row => row[column] === null || row[column] === undefined
    ).length;

    if (values.length === 0) {
        return {
            count: 0,
            missing,
            minimum: null,
            maximum: null,
            mean: null,
            median: null,
            standardDeviation: null,
            q1: null,
            q3: null,
            iqr: null
        };
    }

    const q1 = quantile(values, 0.25);
    const q3 = quantile(values, 0.75);

    return {
        count: values.length,
        missing,
        minimum: Math.min(...values),
        maximum: Math.max(...values),
        mean: mean(values),
        median: median(values),
        standardDeviation:
            values.length >= 2
                ? standardDeviation(values)
                : null,
        q1,
        q3,
        iqr: q3 - q1
    };
}

function frequencyTable(rows, column) {
    const frequencies = new Map();

    for (const row of rows) {
        const value = row[column];

        if (value === null || value === undefined) {
            continue;
        }

        frequencies.set(
            value,
            (frequencies.get(value) ?? 0) + 1
        );
    }

    return [...frequencies.entries()]
        .sort((a, b) =>
            b[1] - a[1] ||
            String(a[0]).localeCompare(String(b[0]))
        );
}

function describeCategorical(rows, column) {
    const table = frequencyTable(rows, column);
    const total = table.reduce(
        (sum, [, count]) => sum + count,
        0
    );

    return table.map(([value, count]) => ({
        value,
        count,
        percentage: total === 0
            ? 0
            : (count / total) * 100
    }));
}

function histogram(values, binCount = 8) {
    if (values.length === 0) {
        return [];
    }

    const minimum = Math.min(...values);
    const maximum = Math.max(...values);

    if (minimum === maximum) {
        return [{
            lower: minimum,
            upper: maximum,
            count: values.length
        }];
    }

    const width = (maximum - minimum) / binCount;
    const bins = Array.from(
        { length: binCount },
        (_, index) => ({
            lower: minimum + index * width,
            upper: minimum + (index + 1) * width,
            count: 0
        })
    );

    for (const value of values) {
        let index = Math.floor(
            (value - minimum) / width
        );

        index = Math.min(
            index,
            bins.length - 1
        );

        bins[index].count += 1;
    }

    return bins;
}

// -----------------------------------------------------------------------------
// Bivariate analysis
// -----------------------------------------------------------------------------

function covariance(xValues, yValues) {
    if (xValues.length !== yValues.length) {
        throw new Error(
            "Covariance requires paired arrays of equal length."
        );
    }

    if (xValues.length < 2) {
        throw new Error(
            "Covariance requires at least two observations."
        );
    }

    const xMean = mean(xValues);
    const yMean = mean(yValues);

    return xValues.reduce(
        (sum, x, index) =>
            sum + (x - xMean) * (yValues[index] - yMean),
        0
    ) / (xValues.length - 1);
}

function pearsonCorrelation(xValues, yValues) {
    if (xValues.length !== yValues.length) {
        throw new Error(
            "Correlation requires paired arrays of equal length."
        );
    }

    const xMean = mean(xValues);
    const yMean = mean(yValues);

    let numerator = 0;
    let xSumSquares = 0;
    let ySumSquares = 0;

    for (let index = 0; index < xValues.length; index += 1) {
        const xDeviation = xValues[index] - xMean;
        const yDeviation = yValues[index] - yMean;

        numerator += xDeviation * yDeviation;
        xSumSquares += xDeviation ** 2;
        ySumSquares += yDeviation ** 2;
    }

    const denominator =
        Math.sqrt(xSumSquares * ySumSquares);

    if (denominator === 0) {
        throw new Error(
            "Pearson correlation is undefined for a constant variable."
        );
    }

    return numerator / denominator;
}

function simpleRegression(xValues, yValues) {
    const xMean = mean(xValues);
    const yMean = mean(yValues);

    const numerator = xValues.reduce(
        (sum, x, index) =>
            sum + (x - xMean) * (yValues[index] - yMean),
        0
    );

    const denominator = xValues.reduce(
        (sum, x) => sum + (x - xMean) ** 2,
        0
    );

    if (denominator === 0) {
        throw new Error(
            "Regression predictor has zero variance."
        );
    }

    const slope = numerator / denominator;
    const intercept = yMean - slope * xMean;

    const predictions = xValues.map(
        x => intercept + slope * x
    );

    const totalSumSquares = yValues.reduce(
        (sum, y) => sum + (y - yMean) ** 2,
        0
    );

    const residualSumSquares = yValues.reduce(
        (sum, y, index) =>
            sum + (y - predictions[index]) ** 2,
        0
    );

    const rSquared =
        totalSumSquares === 0
            ? 0
            : 1 - residualSumSquares / totalSumSquares;

    return {
        slope,
        intercept,
        rSquared,
        correlation: pearsonCorrelation(
            xValues,
            yValues
        )
    };
}

// -----------------------------------------------------------------------------
// Outlier analysis
// -----------------------------------------------------------------------------

function iqrOutliers(values) {
    const q1 = quantile(values, 0.25);
    const q3 = quantile(values, 0.75);
    const iqr = q3 - q1;

    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;

    const outliers = values.filter(
        value =>
            value < lowerFence ||
            value > upperFence
    );

    return {
        q1,
        q3,
        iqr,
        lowerFence,
        upperFence,
        outliers
    };
}

function zScoreOutliers(values, threshold = 3) {
    if (values.length < 2) {
        return [];
    }

    const average = mean(values);
    const sd = standardDeviation(values);

    if (sd === 0) {
        return [];
    }

    return values.filter(
        value =>
            Math.abs((value - average) / sd) > threshold
    );
}

// -----------------------------------------------------------------------------
// Multivariate analysis
// -----------------------------------------------------------------------------

function correlationMatrix(rows, columns) {
    const matrix = {};

    for (const xColumn of columns) {
        matrix[xColumn] = {};

        for (const yColumn of columns) {
            const pairs = pairedValues(
                rows,
                xColumn,
                yColumn
            );

            if (pairs.length < 2) {
                matrix[xColumn][yColumn] = null;
                continue;
            }

            try {
                matrix[xColumn][yColumn] =
                    pearsonCorrelation(
                        pairs.map(pair => pair[0]),
                        pairs.map(pair => pair[1])
                    );
            } catch {
                matrix[xColumn][yColumn] = null;
            }
        }
    }

    return matrix;
}

function groupedStatistics(rows, groupColumn, valueColumn) {
    const groups = new Map();

    for (const row of rows) {
        const group = row[groupColumn];
        const value = row[valueColumn];

        if (
            group === null ||
            group === undefined ||
            typeof value !== "number" ||
            !Number.isFinite(value)
        ) {
            continue;
        }

        if (!groups.has(group)) {
            groups.set(group, []);
        }

        groups.get(group).push(value);
    }

    return [...groups.entries()]
        .map(([group, values]) => ({
            group,
            count: values.length,
            mean: mean(values),
            median: median(values),
            standardDeviation:
                values.length > 1
                    ? standardDeviation(values)
                    : 0
        }))
        .sort((a, b) =>
            String(a.group).localeCompare(
                String(b.group)
            )
        );
}

function crossTabulation(rows, rowColumn, columnColumn) {
    const table = new Map();

    for (const row of rows) {
        const rowValue = row[rowColumn];
        const columnValue = row[columnColumn];

        if (
            rowValue === null ||
            rowValue === undefined ||
            columnValue === null ||
            columnValue === undefined
        ) {
            continue;
        }

        if (!table.has(rowValue)) {
            table.set(rowValue, new Map());
        }

        const rowMap = table.get(rowValue);

        rowMap.set(
            columnValue,
            (rowMap.get(columnValue) ?? 0) + 1
        );
    }

    return table;
}

// -----------------------------------------------------------------------------
// Data-quality validation
// -----------------------------------------------------------------------------

function validateDataset(rows) {
    const errors = [];

    rows.forEach((row, index) => {
        const position = index + 1;

        if (
            row.customerRating !== null &&
            (
                row.customerRating < 1 ||
                row.customerRating > 5
            )
        ) {
            errors.push(
                `row ${position}: customerRating outside [1, 5]`
            );
        }

        if (
            row.discountPct !== null &&
            (
                row.discountPct < 0 ||
                row.discountPct > 100
            )
        ) {
            errors.push(
                `row ${position}: discountPct outside [0, 100]`
            );
        }

        if (
            row.deliveryDays !== null &&
            row.deliveryDays < 0
        ) {
            errors.push(
                `row ${position}: deliveryDays is negative`
            );
        }

        if (
            row.revenue !== null &&
            row.revenue < 0
        ) {
            errors.push(
                `row ${position}: revenue is negative`
            );
        }
    });

    return errors;
}

// -----------------------------------------------------------------------------
// Event-driven EDA workflow
// -----------------------------------------------------------------------------

class EDAEngine extends EventEmitter {
    constructor(rows) {
        super();
        this.rows = rows;
        this.numericColumns = [
            "marketingSpend",
            "discountPct",
            "unitsSold",
            "customerRating",
            "deliveryDays",
            "revenue"
        ];
        this.categoricalColumns = [
            "region",
            "channel",
            "product",
            "returned"
        ];
        this.results = {};
    }

    inspect() {
        const columns = Object.keys(this.rows[0] ?? {});

        const missing = Object.fromEntries(
            columns.map(column => [
                column,
                this.rows.filter(
                    row =>
                        row[column] === null ||
                        row[column] === undefined
                ).length
            ])
        );

        this.results.profile = {
            observations: this.rows.length,
            variables: columns.length,
            columns,
            missing
        };

        this.emit("stageComplete", {
            stage: "dataset inspection",
            result: this.results.profile
        });
    }

    analyzeUnivariate() {
        this.results.univariate = {
            numerical: Object.fromEntries(
                this.numericColumns.map(column => [
                    column,
                    summarizeNumeric(
                        this.rows,
                        column
                    )
                ])
            ),
            categorical: Object.fromEntries(
                this.categoricalColumns.map(column => [
                    column,
                    describeCategorical(
                        this.rows,
                        column
                    )
                ])
            ),
            revenueHistogram: histogram(
                numericValues(
                    this.rows,
                    "revenue"
                ),
                10
            )
        };

        this.emit("stageComplete", {
            stage: "univariate analysis",
            result: this.results.univariate
        });
    }

    analyzeBivariate() {
        const pairs = pairedValues(
            this.rows,
            "marketingSpend",
            "revenue"
        );

        const xValues = pairs.map(pair => pair[0]);
        const yValues = pairs.map(pair => pair[1]);

        this.results.bivariate = {
            marketingToRevenue: {
                covariance: covariance(
                    xValues,
                    yValues
                ),
                correlation: pearsonCorrelation(
                    xValues,
                    yValues
                ),
                regression: simpleRegression(
                    xValues,
                    yValues
                )
            },
            channelRevenue: groupedStatistics(
                this.rows,
                "channel",
                "revenue"
            ),
            regionRating: groupedStatistics(
                this.rows,
                "region",
                "customerRating"
            )
        };

        this.emit("stageComplete", {
            stage: "bivariate analysis",
            result: this.results.bivariate
        });
    }

    analyzeMultivariate() {
        const columns = this.numericColumns;

        this.results.multivariate = {
            correlationMatrix: correlationMatrix(
                this.rows,
                columns
            ),
            regionChannel: crossTabulation(
                this.rows,
                "region",
                "channel"
            ),
            productChannel: crossTabulation(
                this.rows,
                "product",
                "channel"
            )
        };

        this.emit("stageComplete", {
            stage: "multivariate analysis",
            result: this.results.multivariate
        });
    }

    analyzeOutliers() {
        this.results.outliers = {};

        for (const column of this.numericColumns) {
            const values = numericValues(
                this.rows,
                column
            );

            if (values.length < 4) {
                this.results.outliers[column] = null;
                continue;
            }

            this.results.outliers[column] = {
                iqr: iqrOutliers(values),
                zScore: zScoreOutliers(values)
            };
        }

        this.emit("stageComplete", {
            stage: "outlier analysis",
            result: this.results.outliers
        });
    }

    validate() {
        this.results.validation =
            validateDataset(this.rows);

        this.emit("stageComplete", {
            stage: "data validation",
            result: this.results.validation
        });
    }

    async run() {
        this.emit("started");

        this.inspect();
        this.analyzeUnivariate();

        // setImmediate demonstrates that an EDA pipeline can yield control
        // between expensive analytical stages in an event-driven runtime.
        await new Promise(resolve => setImmediate(resolve));

        this.analyzeBivariate();
        this.analyzeMultivariate();
        this.analyzeOutliers();
        this.validate();

        this.emit("completed", this.results);

        return this.results;
    }
}

// -----------------------------------------------------------------------------
// Report formatting
// -----------------------------------------------------------------------------

function printTitle(title) {
    console.log(`\n${"=".repeat(78)}`);
    console.log(title);
    console.log("=".repeat(78));
}

function printNumericResults(results) {
    printTitle("UNIVARIATE NUMERICAL ANALYSIS");

    console.log(
        [
            "Variable".padEnd(22),
            "N".padStart(8),
            "Missing".padStart(10),
            "Mean".padStart(14),
            "Median".padStart(14),
            "Std Dev".padStart(14)
        ].join("")
    );

    for (const [column, summary] of Object.entries(results)) {
        console.log(
            [
                column.padEnd(22),
                String(summary.count).padStart(8),
                String(summary.missing).padStart(10),
                summary.mean === null
                    ? "NA".padStart(14)
                    : summary.mean.toFixed(2).padStart(14),
                summary.median === null
                    ? "NA".padStart(14)
                    : summary.median.toFixed(2).padStart(14),
                summary.standardDeviation === null
                    ? "NA".padStart(14)
                    : summary.standardDeviation
                        .toFixed(2)
                        .padStart(14)
            ].join("")
        );
    }
}

function printCategoricalResults(results) {
    printTitle("UNIVARIATE CATEGORICAL ANALYSIS");

    for (const [column, values] of Object.entries(results)) {
        console.log(`\n${column}`);

        for (const item of values) {
            console.log(
                `  ${String(item.value).padEnd(14)}` +
                `count=${String(item.count).padEnd(5)}` +
                `share=${item.percentage.toFixed(2)}%`
            );
        }
    }
}

function printBivariateResults(result) {
    printTitle("BIVARIATE ANALYSIS");

    const relationship = result.marketingToRevenue;

    console.log(
        `Marketing spend covariance: ` +
        `${relationship.covariance.toFixed(4)}`
    );

    console.log(
        `Marketing spend correlation: ` +
        `${relationship.correlation.toFixed(4)}`
    );

    console.log(
        `Regression slope: ` +
        `${relationship.regression.slope.toFixed(4)}`
    );

    console.log(
        `Regression intercept: ` +
        `${relationship.regression.intercept.toFixed(4)}`
    );

    console.log(
        `R-squared: ` +
        `${relationship.regression.rSquared.toFixed(4)}`
    );

    console.log("\nRevenue by channel:");

    for (const group of result.channelRevenue) {
        console.log(
            `  ${group.group.padEnd(10)}` +
            `n=${String(group.count).padEnd(5)}` +
            `mean=${group.mean.toFixed(2)}`
        );
    }

    console.log("\nCustomer rating by region:");

    for (const group of result.regionRating) {
        console.log(
            `  ${group.group.padEnd(10)}` +
            `n=${String(group.count).padEnd(5)}` +
            `mean=${group.mean.toFixed(2)}`
        );
    }
}

function printCorrelationMatrix(matrix) {
    printTitle("MULTIVARIATE CORRELATION MATRIX");

    const columns = Object.keys(matrix);

    console.log(
        "Variable".padEnd(22) +
        columns
            .map(column => column.slice(0, 9).padStart(11))
            .join("")
    );

    for (const row of columns) {
        let line = row.padEnd(22);

        for (const column of columns) {
            const value = matrix[row][column];

            line += (
                value === null
                    ? "NA".padStart(11)
                    : value.toFixed(2).padStart(11)
            );
        }

        console.log(line);
    }
}

function printOutliers(results) {
    printTitle("OUTLIER ANALYSIS");

    for (const [column, result] of Object.entries(results)) {
        if (!result) {
            continue;
        }

        console.log(
            `${column}: ` +
            `IQR=${result.iqr.outliers.length}, ` +
            `Z-score=${result.zScore.length}`
        );
    }
}

// -----------------------------------------------------------------------------
// JSON export
// -----------------------------------------------------------------------------

async function writeReport(path, report) {
    await fs.writeFile(
        path,
        JSON.stringify(report, null, 2),
        "utf8"
    );
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

async function main() {
    const rows = generateDataset(160, 42);
    const engine = new EDAEngine(rows);

    engine.on("started", () => {
        printTitle("EXPLORATORY DATA ANALYSIS");
        console.log(
            "Univariate | Bivariate | Multivariate"
        );
    });

    engine.on("stageComplete", event => {
        console.log(
            `Completed: ${event.stage}`
        );
    });

    engine.on("completed", () => {
        console.log(
            "\nEDA pipeline completed successfully."
        );
    });

    const results = await engine.run();

    printTitle("DATASET PROFILE");

    console.log(
        `Observations: ${results.profile.observations}`
    );

    console.log(
        `Variables: ${results.profile.variables}`
    );

    console.log("\nMissing values:");

    for (const [column, count] of Object.entries(
        results.profile.missing
    )) {
        console.log(
            `  ${column.padEnd(22)} ${count}`
        );
    }

    printNumericResults(
        results.univariate.numerical
    );

    printCategoricalResults(
        results.univariate.categorical
    );

    printBivariateResults(
        results.bivariate
    );

    printCorrelationMatrix(
        results.multivariate.correlationMatrix
    );

    printOutliers(
        results.outliers
    );

    printTitle("DATA-QUALITY VALIDATION");

    if (results.validation.length === 0) {
        console.log(
            "No business-rule violations were detected."
        );
    } else {
        results.validation.forEach(error => {
            console.log(error);
        });
    }

    const report = {
        generatedAt: new Date().toISOString(),
        methodology: {
            univariate:
                "Distribution and frequency analysis of individual variables",
            bivariate:
                "Paired-variable relationships and grouped comparisons",
            multivariate:
                "Joint relationships among multiple numerical and categorical variables"
        },
        results
    };

    await writeReport(
        "eda-report.json",
        report
    );

    console.log(
        "\nMachine-readable report written to eda-report.json"
    );

    console.log(
        "\nInterpretation rule: correlation measures linear association, " +
        "not causation. Missing observations, outliers, confounding, " +
        "measurement definitions, and sampling design must be considered " +
        "before treating EDA patterns as evidence for a causal explanation."
    );
}

main().catch(error => {
    console.error(
        "EDA execution failed:",
        error.message
    );
    process.exitCode = 1;
});
