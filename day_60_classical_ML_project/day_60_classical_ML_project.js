"use strict";

/*
 * Model evaluation and deployment-gate service.
 *
 * Consumes metrics.json written by classical_ml_project.py, checks
 * model quality against explicit deployment policy thresholds, and
 * emits events for auditability.
 *
 * Run:
 *   node ml_deployment_gate.js
 *   node ml_deployment_gate.js classical_ml_results/metrics.json
 *   node ml_deployment_gate.js classical_ml_results/metrics.json gate-report.json
 *
 * No npm dependencies are required.
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { EventEmitter } = require("node:events");

const MAX_INPUT_BYTES = 5 * 1024 * 1024;

const DEFAULT_POLICY = Object.freeze({
    name: "Malignant Classification Quality Gate",
    minimumRecall: 0.90,
    minimumPrecision: 0.80,
    minimumRocAuc: 0.95,
    minimumCrossValidationF1: 0.90,
    maximumCrossValidationF1Std: 0.05,
    requiredPositiveClass: "malignant",
});

const DEMONSTRATION_DATA = Object.freeze({
    schema_version: 1,
    dataset: {
        name: "Demonstration binary classification dataset",
        positive_class: "malignant",
    },
    runs: [
        {
            model: "LogisticRegression",
            positive_class: "malignant",
            metrics: {
                accuracy: 0.956,
                precision: 0.951,
                recall: 0.929,
                f1: 0.940,
                roc_auc: 0.987,
            },
            cross_validation: {
                f1_mean: 0.934,
                f1_std: 0.018,
            },
        },
        {
            model: "RandomForest",
            positive_class: "malignant",
            metrics: {
                accuracy: 0.886,
                precision: 0.892,
                recall: 0.786,
                f1: 0.835,
                roc_auc: 0.925,
            },
            cross_validation: {
                f1_mean: 0.860,
                f1_std: 0.040,
            },
        },
    ],
});

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class DeploymentPolicy {
    constructor(configuration) {
        this.name = configuration.name;
        this.minimumRecall = configuration.minimumRecall;
        this.minimumPrecision = configuration.minimumPrecision;
        this.minimumRocAuc = configuration.minimumRocAuc;
        this.minimumCrossValidationF1 =
            configuration.minimumCrossValidationF1;
        this.maximumCrossValidationF1Std =
            configuration.maximumCrossValidationF1Std;
        this.requiredPositiveClass = configuration.requiredPositiveClass;

        this.validate();
        Object.freeze(this);
    }

    validate() {
        const thresholds = [
            this.minimumRecall,
            this.minimumPrecision,
            this.minimumRocAuc,
            this.minimumCrossValidationF1,
            this.maximumCrossValidationF1Std,
        ];

        for (const threshold of thresholds) {
            if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
                throw new ValidationError(
                    "Every policy threshold must be between zero and one."
                );
            }
        }

        if (
            typeof this.requiredPositiveClass !== "string" ||
            this.requiredPositiveClass.trim().length === 0
        ) {
            throw new ValidationError("The positive class must be specified.");
        }
    }

    evaluate(run) {
        const reasons = [];
        const metrics = run.metrics;
        const cv = run.cross_validation;

        if (run.positive_class !== this.requiredPositiveClass) {
            reasons.push(
                `Expected positive class '${this.requiredPositiveClass}', ` +
                `received '${run.positive_class}'.`
            );
        }

        const checks = [
            {
                name: "recall",
                actual: metrics.recall,
                required: this.minimumRecall,
                comparison: (actual, required) => actual >= required,
                failure: `Recall is below ${this.minimumRecall}.`,
            },
            {
                name: "precision",
                actual: metrics.precision,
                required: this.minimumPrecision,
                comparison: (actual, required) => actual >= required,
                failure: `Precision is below ${this.minimumPrecision}.`,
            },
            {
                name: "roc_auc",
                actual: metrics.roc_auc,
                required: this.minimumRocAuc,
                comparison: (actual, required) => actual >= required,
                failure: `ROC AUC is below ${this.minimumRocAuc}.`,
            },
            {
                name: "cross_validation_f1",
                actual: cv.f1_mean,
                required: this.minimumCrossValidationF1,
                comparison: (actual, required) => actual >= required,
                failure:
                    `Mean cross-validation F1 is below ` +
                    `${this.minimumCrossValidationF1}.`,
            },
            {
                name: "cross_validation_f1_std",
                actual: cv.f1_std,
                required: this.maximumCrossValidationF1Std,
                comparison: (actual, required) => actual <= required,
                failure:
                    `Cross-validation F1 standard deviation exceeds ` +
                    `${this.maximumCrossValidationF1Std}.`,
            },
        ];

        const checkResults = checks.map((check) => {
            const passed = check.comparison(
                check.actual,
                check.required
            );

            if (!passed) {
                reasons.push(check.failure);
            }

            return {
                name: check.name,
                actual: check.actual,
                threshold: check.required,
                passed,
            };
        });

        return {
            model: run.model,
            decision: reasons.length === 0 ? "ELIGIBLE" : "BLOCKED",
            checks: checkResults,
            reasons,
        };
    }
}

function requireObject(value, description) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        throw new ValidationError(`${description} must be an object.`);
    }
}

function requireMetric(value, description) {
    if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        value < 0 ||
        value > 1
    ) {
        throw new ValidationError(
            `${description} must be a finite number between zero and one.`
        );
    }
}

function validateRun(run, index) {
    requireObject(run, `Run ${index}`);
    requireObject(run.metrics, `Run ${index} metrics`);
    requireObject(run.cross_validation, `Run ${index} cross-validation`);

    if (typeof run.model !== "string" || run.model.trim() === "") {
        throw new ValidationError(`Run ${index} has no model name.`);
    }

    if (typeof run.positive_class !== "string") {
        throw new ValidationError(`Run ${index} has no positive-class label.`);
    }

    for (const metricName of [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]) {
        requireMetric(
            run.metrics[metricName],
            `Run ${index} metrics.${metricName}`
        );
    }

    requireMetric(
        run.cross_validation.f1_mean,
        `Run ${index} cross_validation.f1_mean`
    );
    requireMetric(
        run.cross_validation.f1_std,
        `Run ${index} cross_validation.f1_std`
    );
}

function validateExperiment(payload) {
    requireObject(payload, "Experiment report");
    requireObject(payload.dataset, "Dataset metadata");

    if (payload.schema_version !== 1) {
        throw new ValidationError("Unsupported experiment report schema.");
    }

    if (!Array.isArray(payload.runs) || payload.runs.length === 0) {
        throw new ValidationError("The experiment report contains no runs.");
    }

    const names = new Set();

    payload.runs.forEach((run, index) => {
        validateRun(run, index);

        if (names.has(run.model)) {
            throw new ValidationError(
                `Duplicate model name found: ${run.model}.`
            );
        }

        names.add(run.model);
    });

    return payload;
}

class ExperimentGate extends EventEmitter {
    constructor(policy) {
        super();
        this.policy = policy;
    }

    async evaluateRun(run) {
        // Yield between runs so event listeners can process updates.
        await new Promise((resolve) => setImmediate(resolve));

        const decision = this.policy.evaluate(run);
        this.emit("model:evaluated", decision);
        return decision;
    }

    async evaluateExperiment(payload) {
        this.emit("experiment:loaded", {
            dataset: payload.dataset.name || "Unnamed dataset",
            runCount: payload.runs.length,
        });

        const decisions = await Promise.all(
            payload.runs.map((run) => this.evaluateRun(run))
        );

        const eligible = decisions.filter(
            (decision) => decision.decision === "ELIGIBLE"
        );

        const report = {
            schema_version: 1,
            policy: {
                name: this.policy.name,
                minimum_recall: this.policy.minimumRecall,
                minimum_precision: this.policy.minimumPrecision,
                minimum_roc_auc: this.policy.minimumRocAuc,
                minimum_cross_validation_f1:
                    this.policy.minimumCrossValidationF1,
                maximum_cross_validation_f1_std:
                    this.policy.maximumCrossValidationF1Std,
            },
            dataset: payload.dataset,
            evaluated_models: decisions.length,
            eligible_models: eligible.length,
            blocked_models: decisions.length - eligible.length,
            decisions,
        };

        this.emit("experiment:completed", {
            eligible: report.eligible_models,
            blocked: report.blocked_models,
        });

        return report;
    }
}

async function loadExperiment(inputPath) {
    if (!inputPath) {
        return DEMONSTRATION_DATA;
    }

    const absolutePath = path.resolve(inputPath);
    const fileStats = await fs.stat(absolutePath);

    if (!fileStats.isFile()) {
        throw new ValidationError("The input path must identify a file.");
    }

    if (fileStats.size > MAX_INPUT_BYTES) {
        throw new ValidationError("The experiment report exceeds the size limit.");
    }

    const contents = await fs.readFile(absolutePath, "utf8");

    try {
        return JSON.parse(contents);
    } catch (error) {
        throw new ValidationError(
            `Invalid JSON in experiment report: ${error.message}`
        );
    }
}

async function main() {
    const inputPath = process.argv[2];
    const outputPath = path.resolve(
        process.argv[3] || "deployment_gate_report.json"
    );

    const payload = validateExperiment(
        await loadExperiment(inputPath)
    );

    const policy = new DeploymentPolicy(DEFAULT_POLICY);
    const gate = new ExperimentGate(policy);

    gate.on("experiment:loaded", (event) => {
        console.log(
            `Loaded ${event.dataset}: ${event.runCount} model runs.`
        );
    });

    gate.on("model:evaluated", (decision) => {
        console.log(
            `${decision.model}: ${decision.decision}`
        );

        for (const reason of decision.reasons) {
            console.log(`  - ${reason}`);
        }
    });

    gate.on("experiment:completed", (event) => {
        console.log(
            `Completed: ${event.eligible} eligible, ` +
            `${event.blocked} blocked.`
        );
    });

    const report = await gate.evaluateExperiment(payload);

    await fs.writeFile(
        outputPath,
        JSON.stringify(
            {
                generated_at: new Date().toISOString(),
                ...report,
            },
            null,
            2
        ),
        { encoding: "utf8", flag: "w" }
    );

    console.log(`Report written to ${outputPath}`);
}

main().catch((error) => {
    console.error(`${error.name}: ${error.message}`);
    process.exitCode = 1;
});
