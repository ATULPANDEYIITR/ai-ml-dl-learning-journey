"use strict";

/**
 * Imbalanced Learning: JavaScript implementation.
 *
 * Focus:
 * - Streaming-style dataset processing
 * - Class distribution monitoring
 * - Random over/undersampling
 * - SMOTE interpolation
 * - Cost-sensitive classification
 * - Threshold-based decision policies
 * - Asynchronous experiment execution
 *
 * Runtime: Node.js 18+
 */

const { randomUUID } = require("node:crypto");

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class Dataset {
    constructor(samples) {
        this.samples = samples.map(sample => ({
            features: [...sample.features],
            label: sample.label
        }));

        this.validate();
    }

    validate() {
        if (this.samples.length === 0) {
            throw new ValidationError("Dataset is empty.");
        }

        const dimensions = this.samples[0].features.length;

        if (dimensions === 0) {
            throw new ValidationError("Features cannot be empty.");
        }

        for (const sample of this.samples) {
            if (
                sample.label !== 0 &&
                sample.label !== 1
            ) {
                throw new ValidationError("Labels must be 0 or 1.");
            }

            if (
                sample.features.length !== dimensions ||
                sample.features.some(
                    value => !Number.isFinite(value)
                )
            ) {
                throw new ValidationError("Invalid feature vector.");
            }
        }

        const distribution = this.distribution();

        if (distribution.size !== 2) {
            throw new ValidationError(
                "Both classes must be represented."
            );
        }
    }

    distribution() {
        const counts = new Map([[0, 0], [1, 0]]);

        for (const sample of this.samples) {
            counts.set(
                sample.label,
                counts.get(sample.label) + 1
            );
        }

        return counts;
    }

    get size() {
        return this.samples.length;
    }

    clone() {
        return new Dataset(this.samples);
    }
}

class RandomGenerator {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (
            Math.imul(this.state, 1664525) + 1013904223
        ) >>> 0;

        return this.state / 4294967296;
    }

    integer(maximum) {
        if (!Number.isInteger(maximum) || maximum <= 0) {
            throw new ValidationError("Invalid random range.");
        }

        return Math.floor(this.next() * maximum);
    }

    shuffle(items) {
        const result = [...items];

        for (let i = result.length - 1; i > 0; i--) {
            const j = this.integer(i + 1);
            [result[i], result[j]] = [result[j], result[i]];
        }

        return result;
    }
}

function createSyntheticDataset(
    majorityCount = 200,
    minorityCount = 25,
    seed = 42
) {
    if (
        majorityCount < 2 ||
        minorityCount < 2 ||
        !Number.isInteger(majorityCount) ||
        !Number.isInteger(minorityCount)
    ) {
        throw new ValidationError("Invalid sample counts.");
    }

    const rng = new RandomGenerator(seed);
    const samples = [];

    // Box-Muller normal random generation.
    function normal(mean, standardDeviation) {
        const u1 = Math.max(rng.next(), Number.EPSILON);
        const u2 = rng.next();

        const z = Math.sqrt(-2 * Math.log(u1))
            * Math.cos(2 * Math.PI * u2);

        return mean + standardDeviation * z;
    }

    for (let i = 0; i < majorityCount; i++) {
        samples.push({
            features: [
                normal(0, 1.2),
                normal(0, 1.2)
            ],
            label: 0
        });
    }

    for (let i = 0; i < minorityCount; i++) {
        samples.push({
            features: [
                normal(2, 1),
                normal(2, 1)
            ],
            label: 1
        });
    }

    return new Dataset(rng.shuffle(samples));
}

function stratifiedSplit(dataset, testFraction = 0.25, seed = 42) {
    if (!(testFraction > 0 && testFraction < 1)) {
        throw new ValidationError("Invalid test fraction.");
    }

    const rng = new RandomGenerator(seed);
    const groups = new Map([[0, []], [1, []]]);

    for (const sample of dataset.samples) {
        groups.get(sample.label).push(sample);
    }

    const training = [];
    const testing = [];

    for (const group of groups.values()) {
        if (group.length < 2) {
            throw new ValidationError(
                "Each class requires at least two records."
            );
        }

        const shuffled = rng.shuffle(group);

        const testSize = Math.max(
            1,
            Math.min(
                group.length - 1,
                Math.floor(group.length * testFraction)
            )
        );

        testing.push(...shuffled.slice(0, testSize));
        training.push(...shuffled.slice(testSize));
    }

    return {
        training: new Dataset(rng.shuffle(training)),
        testing: new Dataset(rng.shuffle(testing))
    };
}

function randomOversample(dataset, seed = 42) {
    const rng = new RandomGenerator(seed);
    const groups = [0, 1].map(label =>
        dataset.samples.filter(sample => sample.label === label)
    );

    const target = Math.max(...groups.map(group => group.length));
    const result = [...dataset.samples];

    for (const group of groups) {
        while (group.length + result.length - dataset.size < target - Math.min(...groups.map(g => g.length))) {
            break;
        }

        const additions = target - group.length;

        for (let i = 0; i < additions; i++) {
            const original = group[rng.integer(group.length)];
            result.push({
                features: [...original.features],
                label: original.label
            });
        }
    }

    return new Dataset(result);
}

function randomUndersample(dataset, seed = 42) {
    const rng = new RandomGenerator(seed);
    const groups = [0, 1].map(label =>
        rng.shuffle(
            dataset.samples.filter(sample => sample.label === label)
        )
    );

    const target = Math.min(...groups.map(group => group.length));

    return new Dataset(
        groups.flatMap(group => group.slice(0, target))
    );
}

function distance(a, b) {
    return Math.sqrt(
        a.reduce((sum, value, index) =>
            sum + (value - b[index]) ** 2, 0
        )
    );
}

function smote(dataset, neighbors = 5, seed = 42) {
    const rng = new RandomGenerator(seed);

    const minority = dataset.samples.filter(s => s.label === 1);
    const majority = dataset.samples.filter(s => s.label === 0);

    if (minority.length < 2) {
        throw new ValidationError(
            "SMOTE requires at least two minority samples."
        );
    }

    const k = Math.min(neighbors, minority.length - 1);
    const result = [...dataset.samples];

    const required = majority.length - minority.length;

    for (let i = 0; i < required; i++) {
        const source = minority[rng.integer(minority.length)];

        const nearest = minority
            .filter(candidate => candidate !== source)
            .map(candidate => ({
                sample: candidate,
                distance: distance(
                    source.features,
                    candidate.features
                )
            }))
            .sort((a, b) => a.distance - b.distance)
            .slice(0, k);

        const neighbor = nearest[
            rng.integer(nearest.length)
        ].sample;

        const factor = rng.next();

        const features = source.features.map(
            (value, index) =>
                value + factor * (
                    neighbor.features[index] - value
                )
        );

        result.push({ features, label: 1 });
    }

    return new Dataset(rng.shuffle(result));
}

function calculateClassWeights(dataset) {
    const counts = dataset.distribution();
    const total = dataset.size;

    return new Map([
        [0, total / (2 * counts.get(0))],
        [1, total / (2 * counts.get(1))]
    ]);
}

class CostSensitiveClassifier {
    constructor(options = {}) {
        this.learningRate = options.learningRate ?? 0.08;
        this.epochs = options.epochs ?? 1000;
        this.regularization = options.regularization ?? 0.001;

        if (
            this.learningRate <= 0 ||
            !Number.isInteger(this.epochs) ||
            this.epochs <= 0
        ) {
            throw new ValidationError("Invalid model configuration.");
        }

        this.weights = [];
        this.bias = 0;
        this.fitted = false;
    }

    static sigmoid(value) {
        const bounded = Math.max(-500, Math.min(500, value));
        return 1 / (1 + Math.exp(-bounded));
    }

    fit(dataset, classWeights = new Map([[0, 1], [1, 1]])) {
        const dimensions = dataset.samples[0].features.length;

        this.weights = Array(dimensions).fill(0);
        this.bias = 0;

        const totalWeight = dataset.samples.reduce(
            (sum, sample) => {
                const weight = classWeights.get(sample.label);

                if (!Number.isFinite(weight) || weight <= 0) {
                    throw new ValidationError("Invalid class weight.");
                }

                return sum + weight;
            },
            0
        );

        for (let epoch = 0; epoch < this.epochs; epoch++) {
            const gradients = Array(dimensions).fill(0);
            let biasGradient = 0;

            for (const sample of dataset.samples) {
                const score = sample.features.reduce(
                    (sum, feature, index) =>
                        sum + feature * this.weights[index],
                    this.bias
                );

                const error = (
                    CostSensitiveClassifier.sigmoid(score)
                    - sample.label
                );

                const sampleWeight = classWeights.get(sample.label);

                sample.features.forEach((feature, index) => {
                    gradients[index] += (
                        error * feature * sampleWeight
                    );
                });

                biasGradient += error * sampleWeight;
            }

            this.weights = this.weights.map(
                (weight, index) => weight
                    - this.learningRate * (
                        gradients[index] / totalWeight
                        + this.regularization * weight
                    )
            );

            this.bias -= (
                this.learningRate * biasGradient / totalWeight
            );
        }

        this.fitted = true;
        return this;
    }

    predictProbability(features) {
        if (!this.fitted) {
            throw new Error("Model has not been trained.");
        }

        if (
            features.length !== this.weights.length ||
            features.some(value => !Number.isFinite(value))
        ) {
            throw new ValidationError("Invalid prediction features.");
        }

        const score = features.reduce(
            (sum, value, index) =>
                sum + value * this.weights[index],
            this.bias
        );

        return CostSensitiveClassifier.sigmoid(score);
    }

    predict(features, threshold = 0.5) {
        if (!(threshold > 0 && threshold < 1)) {
            throw new ValidationError("Invalid classification threshold.");
        }

        return Number(
            this.predictProbability(features) >= threshold
        );
    }
}

function evaluate(dataset, model, threshold = 0.5) {
    const matrix = {
        tp: 0,
        tn: 0,
        fp: 0,
        fn: 0
    };

    for (const sample of dataset.samples) {
        const prediction = model.predict(
            sample.features,
            threshold
        );

        if (sample.label === 1 && prediction === 1) matrix.tp++;
        else if (sample.label === 0 && prediction === 0) matrix.tn++;
        else if (sample.label === 0 && prediction === 1) matrix.fp++;
        else matrix.fn++;
    }

    const precision = matrix.tp + matrix.fp
        ? matrix.tp / (matrix.tp + matrix.fp) : 0;

    const recall = matrix.tp + matrix.fn
        ? matrix.tp / (matrix.tp + matrix.fn) : 0;

    const specificity = matrix.tn + matrix.fp
        ? matrix.tn / (matrix.tn + matrix.fp) : 0;

    const f1 = precision + recall
        ? 2 * precision * recall / (precision + recall) : 0;

    return {
        ...matrix,
        precision,
        recall,
        specificity,
        f1,
        balancedAccuracy: (recall + specificity) / 2
    };
}

class ExperimentRunner {
    constructor(training, testing) {
        this.training = training;
        this.testing = testing;
        this.results = new Map();
        this.events = [];
    }

    record(event, details = {}) {
        this.events.push({
            id: randomUUID(),
            event,
            timestamp: new Date().toISOString(),
            details
        });
    }

    async run(name, dataset, weights = null) {
        this.record("experiment_started", { name });

        try {
            // Yield control to support asynchronous orchestration.
            await Promise.resolve();

            const model = new CostSensitiveClassifier();
            model.fit(
                dataset,
                weights ?? new Map([[0, 1], [1, 1]])
            );

            const metrics = evaluate(this.testing, model);

            this.results.set(name, metrics);

            this.record("experiment_completed", {
                name,
                f1: metrics.f1,
                recall: metrics.recall
            });

            return metrics;
        } catch (error) {
            this.record("experiment_failed", {
                name,
                message: error.message
            });

            throw error;
        }
    }

    auditLog() {
        return [...this.events];
    }
}

async function main() {
    console.log("IMBALANCED LEARNING EXPERIMENT");
    console.log("=".repeat(75));

    const dataset = createSyntheticDataset();
    const { training, testing } = stratifiedSplit(dataset);

    console.log("Original:", Object.fromEntries(dataset.distribution()));
    console.log("Training:", Object.fromEntries(training.distribution()));
    console.log("Testing:", Object.fromEntries(testing.distribution()));

    const runner = new ExperimentRunner(training, testing);

    const strategies = [
        {
            name: "Baseline",
            dataset: training,
            weights: null
        },
        {
            name: "Random Oversampling",
            dataset: randomOversample(training)
        },
        {
            name: "Random Undersampling",
            dataset: randomUndersample(training)
        },
        {
            name: "SMOTE",
            dataset: smote(training)
        },
        {
            name: "Class Weighting",
            dataset: training,
            weights: calculateClassWeights(training)
        }
    ];

    console.log("\nEXPERIMENT RESULTS");
    console.log("-".repeat(75));

    for (const strategy of strategies) {
        const metrics = await runner.run(
            strategy.name,
            strategy.dataset,
            strategy.weights ?? null
        );

        console.log(
            strategy.name.padEnd(24),
            `Precision=${metrics.precision.toFixed(3)}`,
            `Recall=${metrics.recall.toFixed(3)}`,
            `F1=${metrics.f1.toFixed(3)}`,
            `Balanced Accuracy=${metrics.balancedAccuracy.toFixed(3)}`
        );
    }

    console.log("\nEVENT AUDIT");
    for (const event of runner.auditLog()) {
        console.log(
            event.event,
            event.details.name
        );
    }

    console.log("\nVALIDATION");
    if (smote(training).size < training.size) {
        throw new Error("SMOTE validation failed.");
    }

    if (randomUndersample(training).size > training.size) {
        throw new Error("Undersampling validation failed.");
    }

    console.log("All validation checks passed.");

    console.log("\nOPERATIONAL CONTROLS");
    console.log("- Preserve the original test distribution.");
    console.log("- Apply sampling only inside training partitions.");
    console.log("- Prevent duplicate records across data partitions.");
    console.log("- Audit generated synthetic observations.");
    console.log("- Tune classification thresholds independently.");
    console.log("- Monitor minority-class precision and recall.");
}

main().catch(error => {
    console.error("Experiment failed:", error.message);
    process.exitCode = 1;
});
