"use strict";

/**
 * Explainable AI: SHAP-style attribution and LIME-style local
 * surrogate analysis for a credit-risk scoring service.
 *
 * Node.js 18+.
 * No external dependencies.
 */

const fs = require("node:fs/promises");

class ValidationError extends Error {}
class ExplanationError extends Error {}

const FEATURES = Object.freeze({
    income: {
        min: 10000,
        max: 200000,
        default: 60000,
        coefficient: 0.000015
    },
    debtRatio: {
        min: 0,
        max: 1,
        default: 0.35,
        coefficient: -2.2
    },
    creditHistory: {
        min: 0,
        max: 30,
        default: 8,
        coefficient: 0.09
    },
    missedPayments: {
        min: 0,
        max: 20,
        default: 1,
        coefficient: -0.45
    },
    employmentYears: {
        min: 0,
        max: 40,
        default: 5,
        coefficient: 0.055
    }
});

class CreditRiskModel {
    constructor() {
        this.intercept = -0.5;
        this.featureNames = Object.keys(FEATURES);
    }

    validate(sample) {
        if (
            sample === null ||
            typeof sample !== "object" ||
            Array.isArray(sample)
        ) {
            throw new ValidationError("Sample must be an object");
        }

        const supplied = Object.keys(sample).sort();
        const expected = [...this.featureNames].sort();

        if (JSON.stringify(supplied) !== JSON.stringify(expected)) {
            throw new ValidationError(
                "Sample must contain exactly the required features"
            );
        }

        for (const name of this.featureNames) {
            const value = sample[name];
            const definition = FEATURES[name];

            if (
                typeof value !== "number" ||
                !Number.isFinite(value) ||
                value < definition.min ||
                value > definition.max
            ) {
                throw new ValidationError(
                    `Invalid value for ${name}`
                );
            }
        }
    }

    probability(sample) {
        this.validate(sample);

        let score = this.intercept;

        for (const name of this.featureNames) {
            score += sample[name] * FEATURES[name].coefficient;
        }

        if (score >= 0) {
            return 1 / (1 + Math.exp(-score));
        }

        const exponential = Math.exp(score);
        return exponential / (1 + exponential);
    }

    classify(sample, threshold = 0.5) {
        if (
            typeof threshold !== "number" ||
            threshold <= 0 ||
            threshold >= 1
        ) {
            throw new ValidationError(
                "Threshold must be between zero and one"
            );
        }

        return this.probability(sample) >= threshold ? 1 : 0;
    }
}

class DeterministicRandom {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (
            1664525 * this.state + 1013904223
        ) >>> 0;

        return this.state / 4294967296;
    }

    normal() {
        const first = Math.max(this.next(), 1e-12);
        const second = this.next();

        return Math.sqrt(-2 * Math.log(first)) *
            Math.cos(2 * Math.PI * second);
    }
}

class ExplanationService {
    constructor(model, background) {
        if (!Array.isArray(background) || background.length === 0) {
            throw new ValidationError(
                "Background observations are required"
            );
        }

        background.forEach(sample => model.validate(sample));

        this.model = model;
        this.background = background.map(
            sample => Object.freeze({ ...sample })
        );
        this.names = model.featureNames;
    }

    mean(values) {
        if (values.length === 0) {
            throw new ExplanationError("Cannot average empty data");
        }

        return values.reduce((sum, value) => sum + value, 0) /
            values.length;
    }

    coalitionPrediction(sample, coalition) {
        const predictions = this.background.map(background => {
            const hybrid = {};

            for (const name of this.names) {
                hybrid[name] = coalition.has(name)
                    ? sample[name]
                    : background[name];
            }

            return this.model.probability(hybrid);
        });

        return this.mean(predictions);
    }

    shap(sample) {
        this.model.validate(sample);

        const n = this.names.length;
        const values = new Map();

        const enumerate = (index, selected) => {
            if (index === n) {
                const key = [...selected].sort().join("|");
                values.set(
                    key,
                    this.coalitionPrediction(
                        sample,
                        new Set(selected)
                    )
                );
                return;
            }

            enumerate(index + 1, selected);

            selected.push(this.names[index]);
            enumerate(index + 1, selected);
            selected.pop();
        };

        enumerate(0, []);

        const factorial = value => {
            let result = 1;

            for (let i = 2; i <= value; i++) {
                result *= i;
            }

            return result;
        };

        const contributions = {};

        for (const feature of this.names) {
            const remaining = this.names.filter(
                name => name !== feature
            );

            let attribution = 0;

            const subsets = (index, selected) => {
                if (index === remaining.length) {
                    const size = selected.length;

                    const baseKey = [...selected].sort().join("|");
                    const withKey = [...selected, feature]
                        .sort()
                        .join("|");

                    const weight =
                        factorial(size) *
                        factorial(n - size - 1) /
                        factorial(n);

                    attribution += weight * (
                        values.get(withKey) - values.get(baseKey)
                    );

                    return;
                }

                subsets(index + 1, selected);

                selected.push(remaining[index]);
                subsets(index + 1, selected);
                selected.pop();
            };

            subsets(0, []);
            contributions[feature] = attribution;
        }

        const baseline = values.get("");
        const prediction = this.model.probability(sample);

        const reconstructed = baseline +
            Object.values(contributions).reduce(
                (sum, value) => sum + value,
                0
            );

        return {
            type: "SHAP",
            baseline,
            prediction,
            contributions,
            reconstructed,
            additivityError: Math.abs(
                prediction - reconstructed
            )
        };
    }

    solveLinearSystem(matrix, vector) {
        const n = vector.length;
        const a = matrix.map(
            (row, i) => [...row, vector[i]]
        );

        for (let col = 0; col < n; col++) {
            let pivot = col;

            for (let row = col + 1; row < n; row++) {
                if (
                    Math.abs(a[row][col]) >
                    Math.abs(a[pivot][col])
                ) {
                    pivot = row;
                }
            }

            if (Math.abs(a[pivot][col]) < 1e-12) {
                throw new ExplanationError(
                    "Weighted regression is singular"
                );
            }

            [a[col], a[pivot]] = [a[pivot], a[col]];

            const divisor = a[col][col];

            for (let j = col; j <= n; j++) {
                a[col][j] /= divisor;
            }

            for (let row = 0; row < n; row++) {
                if (row === col) continue;

                const factor = a[row][col];

                for (let j = col; j <= n; j++) {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }

        return a.map(row => row[n]);
    }

    lime(sample, options = {}) {
        this.model.validate(sample);

        const count = options.samples ?? 1500;
        const kernelWidth = options.kernelWidth ?? 0.75;
        const random = new DeterministicRandom(
            options.seed ?? 42
        );

        if (!Number.isInteger(count) || count < 20 || count > 100000) {
            throw new ValidationError("Invalid perturbation count");
        }

        if (
            !Number.isFinite(kernelWidth) ||
            kernelWidth <= 0
        ) {
            throw new ValidationError("Invalid kernel width");
        }

        const dimension = this.names.length;
        const size = dimension + 1;

        const matrix = Array.from(
            { length: size },
            () => Array(size).fill(0)
        );

        const vector = Array(size).fill(0);

        const ranges = this.names.map(name =>
            FEATURES[name].max - FEATURES[name].min
        );

        let totalWeight = 0;

        for (let iteration = 0; iteration < count; iteration++) {
            const row = [1];
            const perturbed = {};

            for (let i = 0; i < dimension; i++) {
                const name = this.names[i];
                const definition = FEATURES[name];

                const value = Math.max(
                    definition.min,
                    Math.min(
                        definition.max,
                        sample[name] + random.normal() * ranges[i] * 0.15
                    )
                );

                perturbed[name] = value;
                row.push((value - sample[name]) / ranges[i]);
            }

            const distance = Math.sqrt(
                row.slice(1).reduce(
                    (sum, value) => sum + value * value,
                    0
                )
            );

            const weight = Math.exp(
                -(distance * distance) /
                (kernelWidth * kernelWidth)
            );

            const target = this.model.probability(perturbed);
            totalWeight += weight;

            for (let i = 0; i < size; i++) {
                vector[i] += weight * row[i] * target;

                for (let j = 0; j < size; j++) {
                    matrix[i][j] +=
                        weight * row[i] * row[j];
                }
            }
        }

        for (let i = 1; i < size; i++) {
            matrix[i][i] += 1e-6;
        }

        const coefficients = this.solveLinearSystem(
            matrix,
            vector
        );

        const contributions = {};

        for (let i = 0; i < dimension; i++) {
            contributions[this.names[i]] = coefficients[i + 1];
        }

        return {
            type: "LIME",
            localIntercept: coefficients[0],
            localCoefficients: contributions,
            prediction: this.model.probability(sample),
            perturbations: count,
            kernelWidth,
            totalWeight
        };
    }

    globalImportance(dataset) {
        if (!Array.isArray(dataset) || dataset.length === 0) {
            throw new ValidationError(
                "Global analysis requires observations"
            );
        }

        const totals = new Map(
            this.names.map(name => [name, 0])
        );

        for (const sample of dataset) {
            const result = this.shap(sample);

            for (const [name, value] of Object.entries(
                result.contributions
            )) {
                totals.set(
                    name,
                    totals.get(name) + Math.abs(value)
                );
            }
        }

        return [...totals.entries()]
            .map(([feature, value]) => ({
                feature,
                meanAbsoluteSHAP: value / dataset.length
            }))
            .sort(
                (a, b) =>
                    b.meanAbsoluteSHAP - a.meanAbsoluteSHAP
            );
    }
}

class ExplanationAudit {
    constructor() {
        this.events = [];
    }

    record(type, details) {
        this.events.push({
            type,
            timestamp: new Date().toISOString(),
            details
        });
    }

    export() {
        return this.events.map(event => ({ ...event }));
    }
}

async function main() {
    const background = [
        {
            income: 45000,
            debtRatio: 0.55,
            creditHistory: 3,
            missedPayments: 4,
            employmentYears: 2
        },
        {
            income: 70000,
            debtRatio: 0.25,
            creditHistory: 12,
            missedPayments: 0,
            employmentYears: 10
        },
        {
            income: 55000,
            debtRatio: 0.40,
            creditHistory: 7,
            missedPayments: 2,
            employmentYears: 5
        },
        {
            income: 95000,
            debtRatio: 0.20,
            creditHistory: 18,
            missedPayments: 0,
            employmentYears: 15
        }
    ];

    const applicant = {
        income: 85000,
        debtRatio: 0.30,
        creditHistory: 14,
        missedPayments: 1,
        employmentYears: 9
    };

    const model = new CreditRiskModel();
    const service = new ExplanationService(model, background);
    const audit = new ExplanationAudit();

    const prediction = model.probability(applicant);
    const shap = service.shap(applicant);
    const lime = service.lime(applicant);

    const global = service.globalImportance([
        applicant,
        ...background,
        {
            income: 120000,
            debtRatio: 0.15,
            creditHistory: 20,
            missedPayments: 0,
            employmentYears: 18
        }
    ]);

    audit.record("PREDICTION", {
        probability: prediction,
        classification: model.classify(applicant)
    });

    audit.record("SHAP_EXPLANATION", {
        additivityError: shap.additivityError
    });

    audit.record("LIME_EXPLANATION", {
        perturbations: lime.perturbations
    });

    if (shap.additivityError > 1e-9) {
        throw new ExplanationError(
            "SHAP additivity validation failed"
        );
    }

    const report = {
        generatedAt: new Date().toISOString(),
        applicant,
        prediction: {
            probability: prediction,
            classification: model.classify(applicant)
        },
        shap,
        lime,
        globalImportance: global,
        audit: audit.export()
    };

    await fs.writeFile(
        "explainability_report.json",
        JSON.stringify(report, null, 2),
        { encoding: "utf8", mode: 0o600 }
    );

    console.log("EXPLAINABLE AI CREDIT RISK REPORT");
    console.log("================================");

    console.log("Probability:", prediction.toFixed(6));
    console.log("Classification:", model.classify(applicant));

    console.log("\nSHAP VALUES");

    for (const [name, value] of Object.entries(
        shap.contributions
    )) {
        console.log(`${name}: ${value.toFixed(6)}`);
    }

    console.log("\nLIME COEFFICIENTS");

    for (const [name, value] of Object.entries(
        lime.localCoefficients
    )) {
        console.log(`${name}: ${value.toFixed(6)}`);
    }

    console.log("\nGLOBAL IMPORTANCE");

    for (const item of global) {
        console.log(
            `${item.feature}: ${item.meanAbsoluteSHAP.toFixed(6)}`
        );
    }

    console.log("\nReport saved to explainability_report.json");
}

main().catch(error => {
    console.error(`${error.name}: ${error.message}`);
    process.exitCode = 1;
});
