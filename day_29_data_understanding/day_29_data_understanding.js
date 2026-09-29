/*
 * Data Understanding: Features, Labels, Observations, Variables,
 * and Target Variables
 *
 * Self-contained JavaScript study program.
 *
 * Runtime:
 *   Node.js 18+ recommended.
 *
 * The examples progress from basic data terminology to:
 * - observations and units of analysis
 * - variables and variable types
 * - features, labels, and targets
 * - classification and regression
 * - validation
 * - missing values and duplicates
 * - categorical encoding
 * - feature engineering
 * - train/test splitting
 * - stratification
 * - leakage
 * - temporal considerations
 * - data contracts
 * - production-oriented data profiling
 */

"use strict";


// -----------------------------------------------------------------------------
// 1. BASIC DATASET
// -----------------------------------------------------------------------------

console.log("=".repeat(80));
console.log("DATA UNDERSTANDING: FOUNDATIONS");
console.log("=".repeat(80));

const customers = [
    {
        customerId: "C001",
        age: 28,
        annualIncome: 52000,
        city: "Lucknow",
        membership: "Basic",
        visitsPerMonth: 3,
        purchased: 0
    },
    {
        customerId: "C002",
        age: 35,
        annualIncome: 76000,
        city: "Delhi",
        membership: "Premium",
        visitsPerMonth: 8,
        purchased: 1
    },
    {
        customerId: "C003",
        age: 42,
        annualIncome: 91000,
        city: "Mumbai",
        membership: "Premium",
        visitsPerMonth: 10,
        purchased: 1
    },
    {
        customerId: "C004",
        age: 23,
        annualIncome: 41000,
        city: "Lucknow",
        membership: "Basic",
        visitsPerMonth: 2,
        purchased: 0
    }
];

console.table(customers);


// -----------------------------------------------------------------------------
// 2. OBSERVATIONS AND VARIABLES
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("OBSERVATIONS AND VARIABLES");
console.log("=".repeat(80));

/*
 * An observation is one recorded instance.
 *
 * In this dataset:
 *   one row = one customer observation
 *
 * A variable is a characteristic recorded for observations.
 *
 * Examples:
 *   age
 *   annualIncome
 *   city
 *   membership
 *   purchased
 *
 * JavaScript objects are convenient for teaching because each property
 * corresponds to a variable and each object corresponds to an observation.
 */

const firstObservation = customers[0];

console.log("First observation:", firstObservation);
console.log("Variable names:", Object.keys(firstObservation));


// -----------------------------------------------------------------------------
// 3. FEATURES AND TARGET
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("FEATURES AND TARGET");
console.log("=".repeat(80));

const featureColumns = [
    "age",
    "annualIncome",
    "city",
    "membership",
    "visitsPerMonth"
];

const targetColumn = "purchased";

console.log("Features:", featureColumns);
console.log("Target:", targetColumn);
console.log("Identifier:", "customerId");

/*
 * A feature is an input used to make a prediction.
 *
 * A target is the outcome the system is trying to predict.
 *
 * customerId identifies a record but does not automatically become a
 * meaningful predictive feature.
 */


// -----------------------------------------------------------------------------
// 4. GENERIC FEATURE-TARGET SPLIT
// -----------------------------------------------------------------------------

function splitFeaturesAndTarget(rows, featureNames, targetName) {
    if (!Array.isArray(rows) || rows.length === 0) {
        throw new Error("Dataset must contain at least one observation.");
    }

    if (featureNames.includes(targetName)) {
        throw new Error("Target cannot also be a feature.");
    }

    const columns = Object.keys(rows[0]);

    for (const featureName of featureNames) {
        if (!columns.includes(featureName)) {
            throw new Error(`Missing feature column: ${featureName}`);
        }
    }

    if (!columns.includes(targetName)) {
        throw new Error(`Missing target column: ${targetName}`);
    }

    const features = rows.map(row => {
        const featureObject = {};

        for (const featureName of featureNames) {
            featureObject[featureName] = row[featureName];
        }

        return featureObject;
    });

    const targets = rows.map(row => row[targetName]);

    return { features, targets };
}

const split = splitFeaturesAndTarget(
    customers,
    featureColumns,
    targetColumn
);

console.log("Feature records:");
console.table(split.features);
console.log("Targets:", split.targets);


// -----------------------------------------------------------------------------
// 5. VARIABLE CLASSIFICATION
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("VARIABLE CLASSIFICATION");
console.log("=".repeat(80));

const variableTypes = {
    customerId: "identifier",
    age: "discrete numerical",
    annualIncome: "continuous numerical",
    city: "nominal categorical",
    membership: "nominal categorical",
    visitsPerMonth: "discrete numerical",
    purchased: "binary target"
};

console.table(variableTypes);

/*
 * Classification of a variable depends on its meaning, not merely the
 * JavaScript type.
 *
 * For example:
 *   1, 2, 3
 *
 * could represent:
 *   a numerical quantity
 *   ordered categories
 *   arbitrary category codes
 *
 * Context determines the correct interpretation.
 */


// -----------------------------------------------------------------------------
// 6. NOMINAL AND ORDINAL CATEGORIES
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("NOMINAL AND ORDINAL VARIABLES");
console.log("=".repeat(80));

const cities = ["Delhi", "Mumbai", "Lucknow"];
const serviceLevels = ["Low", "Medium", "High"];

console.log("Nominal categories:", cities);
console.log("Ordinal categories:", serviceLevels);

/*
 * Nominal:
 *   categories have no intrinsic order.
 *
 * Ordinal:
 *   categories have meaningful order.
 *
 * Low < Medium < High
 *
 * The ordering does not necessarily imply equal distances between levels.
 */


// -----------------------------------------------------------------------------
// 7. TARGET TYPES
// -----------------------------------------------------------------------------

function classifyTarget(values) {
    if (!Array.isArray(values) || values.length === 0) {
        return "empty";
    }

    const uniqueValues = [...new Set(values)];

    if (values.every(value => typeof value === "boolean")) {
        return "binary categorical";
    }

    if (
        values.every(
            value =>
                typeof value === "number" &&
                Number.isFinite(value)
        )
    ) {
        if (uniqueValues.length === 2) {
            return "binary numerical encoding";
        }

        return "numerical target";
    }

    return "categorical target";
}

console.log("Binary:", classifyTarget([0, 1, 0, 1]));
console.log("Regression:", classifyTarget([125.5, 220.2, 310.8]));
console.log(
    "Multiclass:",
    classifyTarget(["low", "medium", "high", "medium"])
);


// -----------------------------------------------------------------------------
// 8. CLASSIFICATION VS REGRESSION
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("CLASSIFICATION VS REGRESSION");
console.log("=".repeat(80));

const classificationProblem = {
    features: {
        age: 35,
        annualIncome: 76000
    },
    target: "purchased",
    targetType: "binary classification"
};

const regressionProblem = {
    features: {
        areaSquareFeet: 1500,
        bedrooms: 3
    },
    target: "housePrice",
    targetType: "regression"
};

console.log("Classification:", classificationProblem);
console.log("Regression:", regressionProblem);


// -----------------------------------------------------------------------------
// 9. MULTICLASS AND MULTI-LABEL
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("MULTICLASS AND MULTI-LABEL");
console.log("=".repeat(80));

const multiclassTargets = [
    "billing",
    "technical",
    "account",
    "technical"
];

const multiLabelTargets = [
    ["finance", "risk"],
    ["technology"],
    ["finance", "technology"]
];

console.log("Multiclass:", multiclassTargets);
console.log("Multi-label:", multiLabelTargets);

/*
 * Multiclass:
 *   one observation has one class from several possible classes.
 *
 * Multi-label:
 *   one observation may have several labels simultaneously.
 */


// -----------------------------------------------------------------------------
// 10. NUMERICAL PROFILING
// -----------------------------------------------------------------------------

function mean(values) {
    if (values.length === 0) {
        throw new Error("Cannot calculate mean of an empty array.");
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function numericalSummary(values) {
    if (values.length === 0) {
        throw new Error("No numerical observations.");
    }

    const average = mean(values);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);

    return {
        count: values.length,
        minimum,
        maximum,
        mean: average,
        range: maximum - minimum
    };
}

const incomes = customers.map(customer => customer.annualIncome);
const ages = customers.map(customer => customer.age);

console.log("Income summary:", numericalSummary(incomes));
console.log("Age summary:", numericalSummary(ages));


// -----------------------------------------------------------------------------
// 11. MISSING VALUE DETECTION
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("MISSING VALUES");
console.log("=".repeat(80));

const incompleteCustomers = [
    {
        age: 28,
        income: 52000,
        city: "Lucknow"
    },
    {
        age: null,
        income: 76000,
        city: "Delhi"
    },
    {
        age: 42,
        income: undefined,
        city: "Mumbai"
    }
];

function findMissingFields(row) {
    return Object.entries(row)
        .filter(([, value]) => value === null || value === undefined)
        .map(([key]) => key);
}

for (const row of incompleteCustomers) {
    console.log(
        row,
        "missing:",
        findMissingFields(row)
    );
}

/*
 * null and undefined are both relevant when inspecting JavaScript data.
 *
 * Missing values should not automatically be converted to zero.
 * Zero is a legitimate value for many variables.
 */


// -----------------------------------------------------------------------------
// 12. DUPLICATE DETECTION
// -----------------------------------------------------------------------------

function findDuplicateRows(rows) {
    const seen = new Set();
    const duplicates = [];

    rows.forEach((row, index) => {
        const signature = JSON.stringify(
            Object.keys(row)
                .sort()
                .reduce((ordered, key) => {
                    ordered[key] = row[key];
                    return ordered;
                }, {})
        );

        if (seen.has(signature)) {
            duplicates.push(index);
        } else {
            seen.add(signature);
        }
    });

    return duplicates;
}

const duplicateRows = [
    { id: 1, age: 25, income: 40000 },
    { id: 2, age: 30, income: 60000 },
    { id: 2, age: 30, income: 60000 }
];

console.log(
    "Duplicate indices:",
    findDuplicateRows(duplicateRows)
);


// -----------------------------------------------------------------------------
// 13. RANGE AND DOMAIN VALIDATION
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("DATA VALIDATION");
console.log("=".repeat(80));

function validateCustomer(row) {
    const errors = [];

    if (
        typeof row.age !== "number" ||
        !Number.isFinite(row.age)
    ) {
        errors.push("age must be a finite number");
    } else if (row.age < 0 || row.age > 120) {
        errors.push("age must be between 0 and 120");
    }

    if (
        typeof row.annualIncome !== "number" ||
        !Number.isFinite(row.annualIncome)
    ) {
        errors.push("annualIncome must be a finite number");
    } else if (row.annualIncome < 0) {
        errors.push("annualIncome cannot be negative");
    }

    if (!["Basic", "Premium"].includes(row.membership)) {
        errors.push("membership must be Basic or Premium");
    }

    if (![0, 1].includes(row.purchased)) {
        errors.push("purchased must be 0 or 1");
    }

    return errors;
}

const validCustomer = customers[0];

const invalidCustomer = {
    age: -5,
    annualIncome: -100,
    membership: "Unknown",
    purchased: 7
};

console.log(
    "Valid customer errors:",
    validateCustomer(validCustomer)
);

console.log(
    "Invalid customer errors:",
    validateCustomer(invalidCustomer)
);


// -----------------------------------------------------------------------------
// 14. VALUE COUNTS
// -----------------------------------------------------------------------------

function valueCounts(rows, column) {
    const counts = new Map();

    for (const row of rows) {
        const value = row[column];
        counts.set(value, (counts.get(value) || 0) + 1);
    }

    return Object.fromEntries(counts);
}

console.log("City counts:", valueCounts(customers, "city"));
console.log(
    "Membership counts:",
    valueCounts(customers, "membership")
);
console.log(
    "Target counts:",
    valueCounts(customers, "purchased")
);


// -----------------------------------------------------------------------------
// 15. CLASS DISTRIBUTION
// -----------------------------------------------------------------------------

function classDistribution(values) {
    if (values.length === 0) {
        return {};
    }

    const counts = new Map();

    for (const value of values) {
        counts.set(value, (counts.get(value) || 0) + 1);
    }

    return Object.fromEntries(
        [...counts.entries()].map(
            ([key, count]) => [key, count / values.length]
        )
    );
}

console.log(
    "Target distribution:",
    classDistribution(customers.map(row => row.purchased))
);


// -----------------------------------------------------------------------------
// 16. ONE-HOT ENCODING
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("ONE-HOT ENCODING");
console.log("=".repeat(80));

function oneHotEncode(rows, column) {
    const categories = [
        ...new Set(rows.map(row => row[column]))
    ].sort();

    return rows.map(row => {
        const transformed = { ...row };

        delete transformed[column];

        for (const category of categories) {
            transformed[`${column}_${category}`] =
                row[column] === category ? 1 : 0;
        }

        return transformed;
    });
}

const encodedCustomers = oneHotEncode(
    customers,
    "membership"
);

console.table(encodedCustomers);

/*
 * One-hot encoding creates separate binary variables.
 *
 * Example:
 *   Basic   -> membership_Basic = 1
 *   Premium -> membership_Premium = 1
 *
 * This avoids falsely implying:
 *   Basic < Premium
 *
 * when membership is merely nominal.
 */


// -----------------------------------------------------------------------------
// 17. ORDINAL ENCODING
// -----------------------------------------------------------------------------

const priorityMapping = {
    Low: 1,
    Medium: 2,
    High: 3
};

const priorities = [
    "Low",
    "High",
    "Medium",
    "Low"
];

const encodedPriorities = priorities.map(
    priority => priorityMapping[priority]
);

console.log("Original priorities:", priorities);
console.log("Encoded priorities:", encodedPriorities);

/*
 * This encoding is meaningful because the categories have an established
 * order.
 */


// -----------------------------------------------------------------------------
// 18. FEATURE ENGINEERING
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("FEATURE ENGINEERING");
console.log("=".repeat(80));

function enrichCustomer(customer) {
    if (customer.visitsPerMonth <= 0) {
        throw new Error(
            "visitsPerMonth must be greater than zero."
        );
    }

    return {
        ...customer,
        annualIncomePerVisit:
            customer.annualIncome / customer.visitsPerMonth,
        isHighIncome:
            customer.annualIncome >= 75000,
        isFrequentVisitor:
            customer.visitsPerMonth >= 6
    };
}

const enriched = customers.map(enrichCustomer);

console.table(enriched);

/*
 * Feature engineering transforms available information into useful
 * representations.
 *
 * A critical rule:
 *   engineered features must use only information that is legitimately
 *   available at prediction time.
 */


// -----------------------------------------------------------------------------
// 19. MIN-MAX SCALING
// -----------------------------------------------------------------------------

function minMaxScale(values) {
    if (values.length === 0) {
        return [];
    }

    const minimum = Math.min(...values);
    const maximum = Math.max(...values);

    if (minimum === maximum) {
        return values.map(() => 0);
    }

    return values.map(
        value => (value - minimum) / (maximum - minimum)
    );
}

console.log(
    "Scaled income:",
    minMaxScale(incomes)
);


// -----------------------------------------------------------------------------
// 20. STANDARDIZATION
// -----------------------------------------------------------------------------

function standardize(values) {
    if (values.length === 0) {
        return [];
    }

    const average = mean(values);

    const variance = mean(
        values.map(value => (value - average) ** 2)
    );

    const standardDeviation = Math.sqrt(variance);

    if (standardDeviation === 0) {
        return values.map(() => 0);
    }

    return values.map(
        value => (value - average) / standardDeviation
    );
}

console.log(
    "Standardized income:",
    standardize(incomes)
);

/*
 * Scaling parameters should normally be learned from training data only.
 *
 * Correct conceptual sequence:
 *
 *   training data
 *       |
 *       +-- learn preprocessing parameters
 *       |
 *       +-- transform training data
 *
 *   validation/test/new data
 *       |
 *       +-- apply the already learned parameters
 *
 * Calculating preprocessing parameters from the full dataset can leak
 * information from evaluation observations.
 */


// -----------------------------------------------------------------------------
// 21. TRAIN/VALIDATION/TEST SPLIT
// -----------------------------------------------------------------------------

function seededRandom(seed) {
    let state = seed >>> 0;

    return function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function shuffle(array, seed = 42) {
    const result = [...array];
    const random = seededRandom(seed);

    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));

        [result[i], result[j]] =
            [result[j], result[i]];
    }

    return result;
}

function splitIndices(
    rowCount,
    trainFraction = 0.7,
    validationFraction = 0.15,
    seed = 42
) {
    if (rowCount < 3) {
        throw new Error(
            "At least three observations are required."
        );
    }

    const indices = shuffle(
        Array.from(
            { length: rowCount },
            (_, index) => index
        ),
        seed
    );

    const trainEnd = Math.floor(
        rowCount * trainFraction
    );

    const validationEnd = Math.floor(
        rowCount *
        (trainFraction + validationFraction)
    );

    return {
        train: indices.slice(0, trainEnd),
        validation: indices.slice(trainEnd, validationEnd),
        test: indices.slice(validationEnd)
    };
}

console.log(
    "Dataset split:",
    splitIndices(20)
);


// -----------------------------------------------------------------------------
// 22. STRATIFIED SPLIT
// -----------------------------------------------------------------------------

function stratifiedSplit(
    labels,
    testFraction = 0.25,
    seed = 42
) {
    if (
        !Array.isArray(labels) ||
        labels.length === 0
    ) {
        throw new Error("Labels must not be empty.");
    }

    const groups = new Map();

    labels.forEach((label, index) => {
        if (!groups.has(label)) {
            groups.set(label, []);
        }

        groups.get(label).push(index);
    });

    const random = seededRandom(seed);

    const train = [];
    const test = [];

    for (const indices of groups.values()) {
        const shuffled = [...indices];

        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));

            [shuffled[i], shuffled[j]] =
                [shuffled[j], shuffled[i]];
        }

        const testCount = Math.max(
            1,
            Math.round(shuffled.length * testFraction)
        );

        test.push(...shuffled.slice(0, testCount));
        train.push(...shuffled.slice(testCount));
    }

    return {
        train: shuffle(train, seed + 1),
        test: shuffle(test, seed + 2)
    };
}

const targetLabels = [
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
];

const stratified = stratifiedSplit(targetLabels);

console.log("Stratified train:", stratified.train);
console.log("Stratified test:", stratified.test);


// -----------------------------------------------------------------------------
// 23. TIME-ORDERED DATA
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("TEMPORAL DATA");
console.log("=".repeat(80));

const timeSeries = [
    {
        timestamp: "2026-01-01T09:00:00Z",
        sales: 10
    },
    {
        timestamp: "2026-01-02T09:00:00Z",
        sales: 13
    },
    {
        timestamp: "2026-01-03T09:00:00Z",
        sales: 15
    }
];

const parsedTimeSeries = timeSeries.map(row => ({
    ...row,
    timestamp: new Date(row.timestamp)
}));

console.table(parsedTimeSeries);

/*
 * In time-dependent prediction, future information must not be used to
 * construct features for past predictions.
 *
 * A random split can be inappropriate when temporal order matters.
 */


// -----------------------------------------------------------------------------
// 24. DATA LEAKAGE EXAMPLE
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("DATA LEAKAGE");
console.log("=".repeat(80));

const loan = {
    income: 80000,
    creditScore: 730,
    employmentYears: 5,
    collectionActionTaken: false,
    defaulted: false
};

console.log(loan);

/*
 * If the objective is to predict default at application time:
 *
 * Valid:
 *   income
 *   credit score
 *   employment history
 *
 * Potentially invalid:
 *   collectionActionTaken
 *   information created after the prediction point
 *
 * Leakage can produce artificially strong validation results.
 */


// -----------------------------------------------------------------------------
// 25. PREDICTION-TIME FEATURE CONTRACT
// -----------------------------------------------------------------------------

class FeatureDefinition {
    constructor(
        name,
        type,
        role,
        required,
        nullable,
        availableAtPrediction,
        description
    ) {
        this.name = name;
        this.type = type;
        this.role = role;
        this.required = required;
        this.nullable = nullable;
        this.availableAtPrediction =
            availableAtPrediction;
        this.description = description;
    }
}

const customerSchema = [
    new FeatureDefinition(
        "customerId",
        "string",
        "identifier",
        true,
        false,
        true,
        "Unique customer identifier"
    ),
    new FeatureDefinition(
        "age",
        "number",
        "feature",
        true,
        false,
        true,
        "Customer age"
    ),
    new FeatureDefinition(
        "annualIncome",
        "number",
        "feature",
        true,
        false,
        true,
        "Annual income"
    ),
    new FeatureDefinition(
        "city",
        "string",
        "feature",
        true,
        false,
        true,
        "Customer city"
    ),
    new FeatureDefinition(
        "membership",
        "string",
        "feature",
        true,
        false,
        true,
        "Membership tier"
    ),
    new FeatureDefinition(
        "visitsPerMonth",
        "number",
        "feature",
        true,
        false,
        true,
        "Average monthly visits"
    ),
    new FeatureDefinition(
        "purchased",
        "number",
        "target",
        true,
        false,
        false,
        "Purchase outcome"
    )
];

console.table(customerSchema);


// -----------------------------------------------------------------------------
// 26. SCHEMA VALIDATION
// -----------------------------------------------------------------------------

function validateSchema(row, schema) {
    const errors = [];

    for (const specification of schema) {
        const exists =
            Object.prototype.hasOwnProperty.call(
                row,
                specification.name
            );

        if (!exists) {
            if (specification.required) {
                errors.push(
                    `Missing required field: ${specification.name}`
                );
            }

            continue;
        }

        const value = row[specification.name];

        if (value === null || value === undefined) {
            if (!specification.nullable) {
                errors.push(
                    `${specification.name} cannot be null`
                );
            }

            continue;
        }

        const actualType = typeof value;

        if (
            specification.type === "number" &&
            (
                actualType !== "number" ||
                !Number.isFinite(value)
            )
        ) {
            errors.push(
                `${specification.name} must be a finite number`
            );
        }

        if (
            specification.type === "string" &&
            actualType !== "string"
        ) {
            errors.push(
                `${specification.name} must be a string`
            );
        }
    }

    return errors;
}

console.log(
    "Schema errors:",
    validateSchema(customers[0], customerSchema)
);


// -----------------------------------------------------------------------------
// 27. SIMPLE PREDICTIVE SCORING
// -----------------------------------------------------------------------------

function purchaseScore(customer) {
    let score = 0;

    if (customer.age >= 30) {
        score += 1;
    }

    if (customer.annualIncome >= 70000) {
        score += 1;
    }

    if (customer.membership === "Premium") {
        score += 1.5;
    }

    if (customer.visitsPerMonth >= 6) {
        score += 1.5;
    }

    return score;
}

for (const customer of customers) {
    const score = purchaseScore(customer);
    const prediction = score >= 2.5 ? 1 : 0;

    console.log({
        customerId: customer.customerId,
        score,
        prediction,
        actualTarget: customer.purchased
    });
}

/*
 * The scoring function intentionally does not receive `purchased`.
 *
 * That illustrates a basic supervised-learning separation:
 *
 *   X -> prediction
 *
 * while:
 *
 *   y -> observed outcome used during training/evaluation
 */


// -----------------------------------------------------------------------------
// 28. FEATURE MATRIX
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("FEATURE MATRIX");
console.log("=".repeat(80));

const numericFeatures = [
    "age",
    "annualIncome",
    "visitsPerMonth"
];

const featureMatrix = customers.map(customer =>
    numericFeatures.map(
        feature => customer[feature]
    )
);

console.log("Columns:", numericFeatures);
console.table(featureMatrix);


// -----------------------------------------------------------------------------
// 29. CORRELATION
// -----------------------------------------------------------------------------

function pearsonCorrelation(x, y) {
    if (x.length !== y.length) {
        throw new Error(
            "Correlation requires equal-length arrays."
        );
    }

    if (x.length < 2) {
        throw new Error(
            "At least two observations are required."
        );
    }

    const xMean = mean(x);
    const yMean = mean(y);

    let numerator = 0;
    let xVariation = 0;
    let yVariation = 0;

    for (let i = 0; i < x.length; i++) {
        const xDifference = x[i] - xMean;
        const yDifference = y[i] - yMean;

        numerator +=
            xDifference * yDifference;

        xVariation +=
            xDifference ** 2;

        yVariation +=
            yDifference ** 2;
    }

    const denominator =
        Math.sqrt(xVariation * yVariation);

    if (denominator === 0) {
        throw new Error(
            "Correlation is undefined for a constant variable."
        );
    }

    return numerator / denominator;
}

const visits = [1, 2, 4, 7, 9];
const purchases = [0, 0, 1, 1, 1];

console.log(
    "Feature-target correlation:",
    pearsonCorrelation(visits, purchases)
);

/*
 * Correlation describes association under a particular statistical measure.
 * It does not prove causation.
 */


// -----------------------------------------------------------------------------
// 30. OUTLIER IDENTIFICATION
// -----------------------------------------------------------------------------

function percentile(values, fraction) {
    const sorted = [...values].sort(
        (a, b) => a - b
    );

    if (
        fraction < 0 ||
        fraction > 1 ||
        sorted.length === 0
    ) {
        throw new Error("Invalid percentile request.");
    }

    const position =
        (sorted.length - 1) * fraction;

    const lower = Math.floor(position);
    const upper = Math.ceil(position);

    if (lower === upper) {
        return sorted[lower];
    }

    const weight = position - lower;

    return (
        sorted[lower] * (1 - weight) +
        sorted[upper] * weight
    );
}

function iqrBounds(values) {
    const q1 = percentile(values, 0.25);
    const q3 = percentile(values, 0.75);
    const iqr = q3 - q1;

    return {
        lower: q1 - 1.5 * iqr,
        upper: q3 + 1.5 * iqr
    };
}

const incomeWithOutlier = [
    40000,
    42000,
    45000,
    47000,
    50000,
    1000000
];

const bounds = iqrBounds(incomeWithOutlier);

console.log("IQR bounds:", bounds);
console.log(
    "Potential outliers:",
    incomeWithOutlier.filter(
        value =>
            value < bounds.lower ||
            value > bounds.upper
    )
);


// -----------------------------------------------------------------------------
// 31. OBSERVATION-LEVEL DATA QUALITY REPORT
// -----------------------------------------------------------------------------

function profileDataset(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        throw new Error("Cannot profile an empty dataset.");
    }

    const columns = Object.keys(rows[0]);

    return columns.map(column => {
        const values = rows.map(row => row[column]);

        const missingCount = values.filter(
            value =>
                value === null ||
                value === undefined
        ).length;

        const nonMissing = values.filter(
            value =>
                value !== null &&
                value !== undefined
        );

        const uniqueCount =
            new Set(nonMissing).size;

        return {
            column,
            rowCount: rows.length,
            missingCount,
            uniqueCount,
            missingRate:
                missingCount / rows.length
        };
    });
}

console.table(profileDataset(customers));


// -----------------------------------------------------------------------------
// 32. UNIT OF ANALYSIS EXAMPLE
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("UNIT OF ANALYSIS");
console.log("=".repeat(80));

const transactions = [
    {
        transactionId: "T001",
        customerId: "C001",
        amount: 1200
    },
    {
        transactionId: "T002",
        customerId: "C001",
        amount: 800
    },
    {
        transactionId: "T003",
        customerId: "C002",
        amount: 2200
    }
];

console.table(transactions);

console.log(
    "Customer C001 appears multiple times because the unit is a transaction."
);


// -----------------------------------------------------------------------------
// 33. TARGET AVAILABILITY CHECK
// -----------------------------------------------------------------------------

function checkPredictionAvailability(schema) {
    return schema
        .filter(
            field =>
                field.role === "feature" &&
                !field.availableAtPrediction
        )
        .map(field => ({
            field: field.name,
            issue: "Potential data leakage"
        }));
}

console.log(
    "Unavailable prediction features:",
    checkPredictionAvailability(customerSchema)
);


// -----------------------------------------------------------------------------
// 34. EDGE CASES
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("EDGE CASES");
console.log("=".repeat(80));

const edgeCases = {
    emptyDataset: [],
    constantFeature: [10, 10, 10, 10],
    oneClassTarget: [1, 1, 1, 1],
    missingTarget: [0, null, 1],
    duplicateIds: ["C1", "C1", "C2"]
};

console.table(edgeCases);

/*
 * Important edge cases:
 *
 * - empty datasets
 * - one observation
 * - constant features
 * - constant target
 * - missing targets
 * - duplicate IDs
 * - unexpected categories
 * - invalid ranges
 * - extreme values
 * - future information
 * - inconsistent units
 */


// -----------------------------------------------------------------------------
// 35. COMMON MISTAKES
// -----------------------------------------------------------------------------

const commonMistakes = [
    "Treating every column as a feature.",
    "Using the target as an input.",
    "Treating an arbitrary ID as a quantitative feature.",
    "Ignoring the unit of analysis.",
    "Assuming every integer is continuous.",
    "Encoding nominal categories as ordered numbers.",
    "Replacing every missing value with zero.",
    "Deleting every outlier without investigation.",
    "Calculating preprocessing parameters using all data.",
    "Allowing future information into features.",
    "Evaluating imbalanced classification with accuracy alone.",
    "Interpreting correlation as proof of causation.",
    "Ignoring production-time availability."
];

console.log("\nCommon mistakes:");

commonMistakes.forEach(
    (mistake, index) =>
        console.log(`${index + 1}. ${mistake}`)
);


// -----------------------------------------------------------------------------
// 36. PRODUCTION DATA CONTRACT
// -----------------------------------------------------------------------------

class DataContract {
    constructor({
        name,
        required,
        type,
        nullable,
        role,
        availableAtPrediction
    }) {
        this.name = name;
        this.required = required;
        this.type = type;
        this.nullable = nullable;
        this.role = role;
        this.availableAtPrediction =
            availableAtPrediction;
    }
}

const productionContract = [
    new DataContract({
        name: "customerId",
        required: true,
        type: "string",
        nullable: false,
        role: "identifier",
        availableAtPrediction: true
    }),
    new DataContract({
        name: "age",
        required: true,
        type: "number",
        nullable: false,
        role: "feature",
        availableAtPrediction: true
    }),
    new DataContract({
        name: "annualIncome",
        required: true,
        type: "number",
        nullable: false,
        role: "feature",
        availableAtPrediction: true
    }),
    new DataContract({
        name: "purchased",
        required: true,
        type: "number",
        nullable: false,
        role: "target",
        availableAtPrediction: false
    })
];

console.table(productionContract);


// -----------------------------------------------------------------------------
// 37. ASYNCHRONOUS DATA INGESTION
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("ASYNCHRONOUS DATA INGESTION");
console.log("=".repeat(80));

/*
 * JavaScript is especially useful for demonstrating application-level
 * asynchronous data processing.
 *
 * A production service might retrieve data from an API before validating it.
 * This self-contained example uses an in-memory Promise rather than an
 * external service.
 */

function fetchTrainingData() {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve(customers);
        }, 10);
    });
}

async function inspectAsyncDataset() {
    const rows = await fetchTrainingData();

    return {
        observations: rows.length,
        variables: Object.keys(rows[0]).length,
        targetDistribution:
            valueCounts(rows, "purchased")
    };
}

inspectAsyncDataset()
    .then(report => {
        console.log("Asynchronous profile:", report);
    })
    .catch(error => {
        console.error("Data inspection failed:", error);
    });


// -----------------------------------------------------------------------------
// 38. FINAL CONCEPT MAP
// -----------------------------------------------------------------------------

console.log("\n" + "=".repeat(80));
console.log("CONCEPT MAP");
console.log("=".repeat(80));

console.log(`
Dataset
|
+-- Observations
|   |
|   +-- Rows representing a defined unit of analysis
|
+-- Variables
    |
    +-- Features
    |   +-- Numerical
    |   +-- Categorical
    |   +-- Temporal
    |   +-- Text
    |   +-- Engineered
    |
    +-- Target / Label
    |   +-- Binary
    |   +-- Multiclass
    |   +-- Multi-label
    |   +-- Regression
    |   +-- Multi-output
    |
    +-- Identifiers
        +-- Usually used for identity, not prediction
`);

console.log("=".repeat(80));
console.log("END OF DATA UNDERSTANDING STUDY PROGRAM");
console.log("=".repeat(80));
