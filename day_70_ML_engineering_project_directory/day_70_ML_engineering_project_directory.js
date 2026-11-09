"use strict";

/**
 * Production-oriented ML pipeline orchestration and release control.
 *
 * Runtime: Node.js 18+
 *
 * Demonstrates:
 * - Event-driven pipeline orchestration
 * - Dataset contracts
 * - Idempotent execution
 * - Retry handling
 * - Pipeline state transitions
 * - Quality gates
 * - Model registry
 * - Deployment promotion
 * - Drift monitoring
 * - Audit events
 *
 * Run:
 *   node ml_orchestrator.js
 */

const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const path = require("node:path");
const EventEmitter = require("node:events");

// -------------------------------------------------------
// Configuration
// -------------------------------------------------------

const CONFIG = Object.freeze({
    minimumRows: 50,
    minimumRocAuc: 0.75,
    maximumLatencyMs: 250,
    maximumFeatureDrift: 0.20,
    minimumRecall: 0.60,
    maximumRetries: 2,
    artifactDirectory: "ml_js_artifacts"
});

const PipelineState = Object.freeze({
    CREATED: "CREATED",
    VALIDATING: "VALIDATING",
    VALIDATED: "VALIDATED",
    TRAINING: "TRAINING",
    EVALUATING: "EVALUATING",
    APPROVED: "APPROVED",
    DEPLOYING: "DEPLOYING",
    DEPLOYED: "DEPLOYED",
    FAILED: "FAILED",
    REJECTED: "REJECTED"
});

const ALLOWED_TRANSITIONS = Object.freeze({
    CREATED: ["VALIDATING", "FAILED"],
    VALIDATING: ["VALIDATED", "FAILED"],
    VALIDATED: ["TRAINING", "FAILED"],
    TRAINING: ["EVALUATING", "FAILED"],
    EVALUATING: ["APPROVED", "REJECTED", "FAILED"],
    APPROVED: ["DEPLOYING", "FAILED"],
    DEPLOYING: ["DEPLOYED", "FAILED"],
    DEPLOYED: [],
    FAILED: [],
    REJECTED: []
});

// -------------------------------------------------------
// Domain exceptions
// -------------------------------------------------------

class ContractError extends Error {
    constructor(message) {
        super(message);
        this.name = "ContractError";
    }
}

class QualityGateError extends Error {
    constructor(message) {
        super(message);
        this.name = "QualityGateError";
    }
}

class StateTransitionError extends Error {
    constructor(message) {
        super(message);
        this.name = "StateTransitionError";
    }
}

// -------------------------------------------------------
// Utility functions
// -------------------------------------------------------

function hashObject(value) {
    return crypto
        .createHash("sha256")
        .update(JSON.stringify(value))
        .digest("hex");
}

function percentile(values, percentage) {
    if (!values.length) {
        throw new RangeError("Cannot calculate empty percentile.");
    }

    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * percentage;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);

    if (lower === upper) {
        return sorted[lower];
    }

    return (
        sorted[lower] * (upper - position) +
        sorted[upper] * (position - lower)
    );
}

// -------------------------------------------------------
// Dataset contract
// -------------------------------------------------------

class DatasetContract {
    constructor({
        requiredFeatures,
        target,
        minimumRows = CONFIG.minimumRows
    }) {
        this.requiredFeatures = new Set(requiredFeatures);
        this.target = target;
        this.minimumRows = minimumRows;
    }

    validate(records, training = true) {
        if (!Array.isArray(records)) {
            throw new ContractError("Dataset must be an array.");
        }

        if (records.length < this.minimumRows) {
            throw new ContractError("Insufficient dataset rows.");
        }

        const identifiers = new Set();

        for (const [index, record] of records.entries()) {
            if (
                record === null ||
                typeof record !== "object" ||
                Array.isArray(record)
            ) {
                throw new ContractError(
                    `Invalid record at index ${index}.`
                );
            }

            if (
                !Object.hasOwn(record, "customer_id") ||
                record.customer_id === null ||
                record.customer_id === undefined
            ) {
                throw new ContractError(
                    `Missing customer identifier at row ${index}.`
                );
            }

            const identifier = String(record.customer_id);

            if (identifiers.has(identifier)) {
                throw new ContractError(
                    `Duplicate customer identifier: ${identifier}`
                );
            }

            identifiers.add(identifier);

            for (const feature of this.requiredFeatures) {
                if (
                    !Object.hasOwn(record, feature) ||
                    record[feature] === null ||
                    record[feature] === undefined
                ) {
                    throw new ContractError(
                        `Missing ${feature} at row ${index}.`
                    );
                }
            }

            if (training) {
                if (
                    ![0, 1].includes(record[this.target])
                ) {
                    throw new ContractError(
                        `Invalid target at row ${index}.`
                    );
                }
            }
        }

        return {
            rows: records.length,
            features: [...this.requiredFeatures],
            fingerprint: hashObject(records)
        };
    }
}

// -------------------------------------------------------
// Pipeline execution state
// -------------------------------------------------------

class PipelineRun extends EventEmitter {
    constructor(runId) {
        super();

        this.runId = runId;
        this.state = PipelineState.CREATED;
        this.history = [];
        this.createdAt = new Date().toISOString();
        this.error = null;
    }

    transition(nextState, details = {}) {
        const allowed = ALLOWED_TRANSITIONS[this.state];

        if (!allowed || !allowed.includes(nextState)) {
            throw new StateTransitionError(
                `Invalid transition: ${this.state} -> ${nextState}`
            );
        }

        const event = Object.freeze({
            runId: this.runId,
            previousState: this.state,
            nextState,
            timestamp: new Date().toISOString(),
            details
        });

        this.state = nextState;
        this.history.push(event);

        this.emit("stateChanged", event);
    }

    fail(error) {
        if (
            ![
                PipelineState.DEPLOYED,
                PipelineState.FAILED,
                PipelineState.REJECTED
            ].includes(this.state)
        ) {
            this.error = error.message;
            this.transition(PipelineState.FAILED, {
                message: error.message
            });
        }
    }
}

// -------------------------------------------------------
// Retry and timeout handling
// -------------------------------------------------------

async function retry(operation, attempts = CONFIG.maximumRetries) {
    let lastError;

    for (let attempt = 0; attempt <= attempts; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error;

            if (attempt === attempts) {
                break;
            }

            const delay = 100 * (2 ** attempt);

            await new Promise(resolve =>
                setTimeout(resolve, delay)
            );
        }
    }

    throw lastError;
}

async function withTimeout(promise, milliseconds) {
    let timer;

    try {
        return await Promise.race([
            promise,
            new Promise((_, reject) => {
                timer = setTimeout(
                    () => reject(new Error("Operation timed out.")),
                    milliseconds
                );
            })
        ]);
    } finally {
        clearTimeout(timer);
    }
}

// -------------------------------------------------------
// Metrics and model evaluation
// -------------------------------------------------------

function confusionMetrics(actual, predicted) {
    if (actual.length !== predicted.length || !actual.length) {
        throw new RangeError("Invalid evaluation arrays.");
    }

    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    for (let i = 0; i < actual.length; i++) {
        if (actual[i] === 1 && predicted[i] === 1) tp++;
        else if (actual[i] === 0 && predicted[i] === 0) tn++;
        else if (actual[i] === 0 && predicted[i] === 1) fp++;
        else fn++;
    }

    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);

    return {
        tp,
        tn,
        fp,
        fn,
        accuracy: (tp + tn) / actual.length,
        precision,
        recall,
        f1: precision + recall === 0
            ? 0
            : 2 * precision * recall / (precision + recall)
    };
}

function calculateAuc(actual, probabilities) {
    if (
        actual.length !== probabilities.length ||
        actual.length === 0
    ) {
        throw new RangeError("Invalid ROC-AUC inputs.");
    }

    const positives = actual.filter(value => value === 1).length;
    const negatives = actual.length - positives;

    if (!positives || !negatives) {
        throw new RangeError(
            "ROC-AUC requires both target classes."
        );
    }

    const observations = actual.map((label, index) => ({
        label,
        score: probabilities[index]
    }));

    observations.sort((a, b) => a.score - b.score);

    let rankSum = 0;
    let index = 0;

    while (index < observations.length) {
        let end = index + 1;

        while (
            end < observations.length &&
            observations[end].score === observations[index].score
        ) {
            end++;
        }

        const averageRank = ((index + 1) + end) / 2;

        for (let j = index; j < end; j++) {
            if (observations[j].label === 1) {
                rankSum += averageRank;
            }
        }

        index = end;
    }

    return (
        rankSum - positives * (positives + 1) / 2
    ) / (positives * negatives);
}

// -------------------------------------------------------
// Release policy
// -------------------------------------------------------

class ReleasePolicy {
    constructor(configuration = CONFIG) {
        this.configuration = configuration;
    }

    evaluate(metrics) {
        const failures = [];

        if (metrics.rocAuc < this.configuration.minimumRocAuc) {
            failures.push("ROC-AUC below minimum.");
        }

        if (metrics.recall < this.configuration.minimumRecall) {
            failures.push("Recall below minimum.");
        }

        if (
            metrics.latencyMs >
            this.configuration.maximumLatencyMs
        ) {
            failures.push("Prediction latency exceeds limit.");
        }

        if (
            metrics.featureDrift >
            this.configuration.maximumFeatureDrift
        ) {
            failures.push("Feature drift exceeds limit.");
        }

        return Object.freeze({
            eligible: failures.length === 0,
            failures,
            evaluatedAt: new Date().toISOString()
        });
    }
}

// -------------------------------------------------------
// Model registry
// -------------------------------------------------------

class ModelRegistry extends EventEmitter {
    constructor() {
        super();
        this.models = new Map();
        this.productionVersion = null;
    }

    register(model) {
        if (
            !model.version ||
            !model.runId ||
            !model.metrics
        ) {
            throw new TypeError("Invalid model registration.");
        }

        if (this.models.has(model.version)) {
            throw new Error("Model version already exists.");
        }

        const entry = Object.freeze({
            ...model,
            registeredAt: new Date().toISOString()
        });

        this.models.set(model.version, entry);

        this.emit("registered", entry);

        return entry;
    }

    promote(version) {
        const model = this.models.get(version);

        if (!model || !model.releaseEligible) {
            throw new QualityGateError(
                "Model is not eligible for production."
            );
        }

        const previousVersion = this.productionVersion;

        this.productionVersion = version;

        this.emit("promoted", {
            previousVersion,
            newVersion: version,
            timestamp: new Date().toISOString()
        });

        return version;
    }

    getProductionModel() {
        if (!this.productionVersion) {
            throw new Error("No production model is registered.");
        }

        return this.models.get(this.productionVersion);
    }
}

// -------------------------------------------------------
// Drift monitoring
// -------------------------------------------------------

function calculatePopulationStabilityIndex(
    reference,
    current,
    buckets = 10
) {
    if (
        reference.length < buckets ||
        current.length < buckets ||
        reference.some(value => !Number.isFinite(value)) ||
        current.some(value => !Number.isFinite(value))
    ) {
        throw new RangeError("Invalid drift-monitoring samples.");
    }

    const boundaries = Array.from(
        { length: buckets - 1 },
        (_, index) => percentile(reference, (index + 1) / buckets)
    );

    const counts = values => {
        const result = Array(buckets).fill(0);

        for (const value of values) {
            let bin = boundaries.findIndex(
                boundary => value <= boundary
            );

            if (bin === -1) bin = buckets - 1;

            result[bin]++;
        }

        return result;
    };

    const expected = counts(reference);
    const observed = counts(current);

    const epsilon = 0.000001;

    let psi = 0;

    for (let i = 0; i < buckets; i++) {
        const expectedRate =
            Math.max(expected[i] / reference.length, epsilon);

        const observedRate =
            Math.max(observed[i] / current.length, epsilon);

        psi += (
            observedRate - expectedRate
        ) * Math.log(observedRate / expectedRate);
    }

    return psi;
}

// -------------------------------------------------------
// Audit storage
// -------------------------------------------------------

class AuditLog {
    constructor() {
        this.entries = [];
    }

    record(event, details = {}) {
        this.entries.push(Object.freeze({
            event,
            timestamp: new Date().toISOString(),
            details: structuredClone(details)
        }));
    }

    async persist(directory) {
        await fs.mkdir(directory, { recursive: true });

        const destination = path.join(
            directory,
            "audit.json"
        );

        const temporary = destination + ".tmp";

        await fs.writeFile(
            temporary,
            JSON.stringify(this.entries, null, 2),
            { mode: 0o600 }
        );

        await fs.rename(temporary, destination);
    }
}

// -------------------------------------------------------
// Pipeline orchestration
// -------------------------------------------------------

class MLPipeline {
    constructor({
        contract,
        policy,
        registry,
        audit
    }) {
        this.contract = contract;
        this.policy = policy;
        this.registry = registry;
        this.audit = audit;
        this.completedRuns = new Map();
    }

    async execute({
        runId,
        records,
        trainer,
        evaluator
    }) {
        if (this.completedRuns.has(runId)) {
            return this.completedRuns.get(runId);
        }

        const run = new PipelineRun(runId);

        run.on("stateChanged", event => {
            this.audit.record("STATE_CHANGED", event);
            console.log(
                `[${event.runId}] ${event.previousState} -> ${event.nextState}`
            );
        });

        try {
            run.transition(PipelineState.VALIDATING);

            const dataset = this.contract.validate(records);

            run.transition(
                PipelineState.VALIDATED,
                dataset
            );

            run.transition(PipelineState.TRAINING);

            const model = await retry(() =>
                withTimeout(
                    trainer(records),
                    30000
                )
            );

            run.transition(PipelineState.EVALUATING);

            const metrics = await evaluator(model, records);

            const decision = this.policy.evaluate(metrics);

            if (!decision.eligible) {
                run.transition(
                    PipelineState.REJECTED,
                    decision
                );

                const result = {
                    runId,
                    state: run.state,
                    decision,
                    history: run.history
                };

                this.audit.record("MODEL_REJECTED", result);
                this.completedRuns.set(runId, result);

                return result;
            }

            run.transition(
                PipelineState.APPROVED,
                decision
            );

            const registered = this.registry.register({
                version: model.version,
                runId,
                metrics,
                releaseEligible: true,
                datasetFingerprint: dataset.fingerprint
            });

            run.transition(PipelineState.DEPLOYING);

            this.registry.promote(registered.version);

            run.transition(PipelineState.DEPLOYED);

            const result = {
                runId,
                state: run.state,
                model: registered,
                decision,
                history: run.history
            };

            this.audit.record("MODEL_DEPLOYED", {
                runId,
                version: registered.version
            });

            this.completedRuns.set(runId, result);

            return result;

        } catch (error) {
            run.fail(error);

            this.audit.record("PIPELINE_FAILED", {
                runId,
                error: error.message
            });

            throw error;
        }
    }
}

// -------------------------------------------------------
// Demonstration
// -------------------------------------------------------

function createDemonstrationDataset(rows = 200) {
    return Array.from({ length: rows }, (_, index) => ({
        customer_id: index + 1,
        tenure: 1 + (index % 60),
        monthly_charges: 30 + (index % 100),
        support_calls: index % 6,
        churn: index % 5 === 0 ? 1 : 0
    }));
}

async function main() {
    const records = createDemonstrationDataset();

    const contract = new DatasetContract({
        requiredFeatures: [
            "tenure",
            "monthly_charges",
            "support_calls"
        ],
        target: "churn"
    });

    const registry = new ModelRegistry();
    const audit = new AuditLog();

    const policy = new ReleasePolicy({
        ...CONFIG,
        minimumRocAuc: 0.70,
        minimumRecall: 0.50
    });

    const pipeline = new MLPipeline({
        contract,
        policy,
        registry,
        audit
    });

    // The trainer and evaluator are adapters.
    // The actual statistical model can run in a Python service.
    const trainer = async () => ({
        version: "churn-1.0.0",
        algorithm: "logistic-regression",
        trainedAt: new Date().toISOString()
    });

    const evaluator = async () => ({
        rocAuc: 0.86,
        recall: 0.78,
        precision: 0.81,
        f1: 0.795,
        latencyMs: 35,
        featureDrift: 0.08
    });

    const result = await pipeline.execute({
        runId: crypto.randomUUID(),
        records,
        trainer,
        evaluator
    });

    console.log("\nPipeline result:");
    console.log(JSON.stringify(result, null, 2));

    if (result.state === PipelineState.DEPLOYED) {
        console.log(
            "\nProduction model:",
            registry.getProductionModel().version
        );
    }

    const reference = records.map(
        record => record.monthly_charges
    );

    const current = records.map(
        record => record.monthly_charges * 1.05
    );

    console.log(
        "\nPopulation Stability Index:",
        calculatePopulationStabilityIndex(
            reference,
            current
        ).toFixed(4)
    );

    await audit.persist(CONFIG.artifactDirectory);
}

main().catch(error => {
    console.error("Pipeline failure:", error.message);
    process.exitCode = 1;
});
