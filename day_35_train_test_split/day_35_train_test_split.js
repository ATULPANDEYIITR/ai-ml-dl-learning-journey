/**
 * Train-Test Split: a JavaScript implementation of an experiment pipeline.
 *
 * This Node.js program focuses on:
 * - deterministic pseudo-random splitting
 * - stratification
 * - chronological validation
 * - preprocessing fitted only on training data
 * - event-driven experiment lifecycle
 * - validation-based model selection
 * - final test evaluation
 * - leakage and reproducibility checks
 *
 * Run with:
 *   node train-test-split.js
 */

"use strict";

class SeededRandom {
    constructor(seed) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (1664525 * this.state + 1013904223) >>> 0;
        return this.state / 4294967296;
    }

    shuffle(items) {
        const copy = [...items];
        for (let i = copy.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [copy[i], copy[j]] = [copy[j], copy[i]];
        }
        return copy;
    }
}

function makeDataset(seed = 2026, size = 180) {
    const rng = new SeededRandom(seed);
    const authors = Array.from({ length: 12 }, (_, i) => `developer-${String(i).padStart(2, "0")}`);
    const start = new Date("2026-01-01T09:00:00Z");

    return Array.from({ length: size }, (_, id) => {
        const authorId = authors[Math.floor(rng.next() * authors.length)];
        const experience = 1 + Math.floor(rng.next() * 10);
        const filesChanged = 1 + Math.floor(-Math.log(1 - rng.next()) * 3);
        const linesChanged = Math.max(5, Math.floor(-Math.log(1 - rng.next()) * 70));
        const reviewComments = Math.floor(-Math.log(1 - rng.next()) * 2.5);
        const priorFailures = Math.floor(-Math.log(1 - rng.next()) * 1.3);

        const score =
            0.025 * linesChanged +
            0.55 * filesChanged +
            0.75 * reviewComments +
            0.9 * priorFailures -
            0.65 * experience;

        const label = score > 3 ? 1 : 0;

        return Object.freeze({
            id,
            linesChanged,
            filesChanged,
            reviewComments,
            priorFailures,
            authorExperience: experience,
            authorId,
            timestamp: new Date(start.getTime() + id * 8 * 60 * 60 * 1000),
            label
        });
    });
}

function splitByRatios(rows, seed, trainRatio = 0.7, validationRatio = 0.15) {
    if (trainRatio + validationRatio >= 1) {
        throw new RangeError("Train and validation ratios must leave a test partition.");
    }

    const shuffled = new SeededRandom(seed).shuffle(rows);
    const trainEnd = Math.floor(shuffled.length * trainRatio);
    const validationEnd = trainEnd + Math.floor(shuffled.length * validationRatio);

    return {
        train: shuffled.slice(0, trainEnd),
        validation: shuffled.slice(trainEnd, validationEnd),
        test: shuffled.slice(validationEnd)
    };
}

function stratifiedSplit(rows, seed) {
    const groups = new Map();

    for (const row of rows) {
        if (!groups.has(row.label)) groups.set(row.label, []);
        groups.get(row.label).push(row);
    }

    const rng = new SeededRandom(seed);
    const result = { train: [], validation: [], test: [] };

    for (const classRows of groups.values()) {
        const shuffled = rng.shuffle(classRows);
        const trainEnd = Math.floor(shuffled.length * 0.7);
        const validationEnd = trainEnd + Math.floor(shuffled.length * 0.15);

        result.train.push(...shuffled.slice(0, trainEnd));
        result.validation.push(...shuffled.slice(trainEnd, validationEnd));
        result.test.push(...shuffled.slice(validationEnd));
    }

    result.train = rng.shuffle(result.train);
    result.validation = rng.shuffle(result.validation);
    result.test = rng.shuffle(result.test);

    return result;
}

function chronologicalSplit(rows) {
    const ordered = [...rows].sort((a, b) => a.timestamp - b.timestamp);
    const trainEnd = Math.floor(ordered.length * 0.7);
    const validationEnd = trainEnd + Math.floor(ordered.length * 0.15);

    return {
        train: ordered.slice(0, trainEnd),
        validation: ordered.slice(trainEnd, validationEnd),
        test: ordered.slice(validationEnd)
    };
}

function featureVector(row) {
    return [
        row.linesChanged,
        row.filesChanged,
        row.reviewComments,
        row.priorFailures,
        row.authorExperience
    ];
}

class StandardScaler {
    constructor() {
        this.means = null;
        this.stds = null;
    }

    fit(matrix) {
        if (!matrix.length) throw new Error("Cannot fit on empty data.");

        const width = matrix[0].length;
        this.means = Array.from({ length: width }, (_, column) =>
            matrix.reduce((sum, row) => sum + row[column], 0) / matrix.length
        );

        this.stds = Array.from({ length: width }, (_, column) => {
            const variance = matrix.reduce(
                (sum, row) => sum + (row[column] - this.means[column]) ** 2,
                0
            ) / matrix.length;

            const standardDeviation = Math.sqrt(variance);
            return standardDeviation === 0 ? 1 : standardDeviation;
        });

        return this;
    }

    transform(matrix) {
        if (!this.means) throw new Error("Scaler must be fitted first.");

        return matrix.map(row =>
            row.map((value, column) =>
                (value - this.means[column]) / this.stds[column]
            )
        );
    }

    fitTransform(matrix) {
        return this.fit(matrix).transform(matrix);
    }
}

class NearestCentroidClassifier {
    constructor() {
        this.centroids = new Map();
    }

    fit(features, targets) {
        const grouped = new Map();

        targets.forEach((target, index) => {
            if (!grouped.has(target)) grouped.set(target, []);
            grouped.get(target).push(features[index]);
        });

        for (const [label, rows] of grouped.entries()) {
            const centroid = rows[0].map((_, column) =>
                rows.reduce((sum, row) => sum + row[column], 0) / rows.length
            );
            this.centroids.set(label, centroid);
        }

        return this;
    }

    predict(features) {
        if (!this.centroids.size) {
            throw new Error("Classifier has not been trained.");
        }

        return features.map(row => {
            let bestLabel = null;
            let bestDistance = Infinity;

            for (const [label, centroid] of this.centroids.entries()) {
                const distance = row.reduce(
                    (sum, value, i) => sum + (value - centroid[i]) ** 2,
                    0
                );

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }

            return bestLabel;
        });
    }
}

function metrics(actual, predicted) {
    if (actual.length !== predicted.length || actual.length === 0) {
        throw new Error("Metric inputs must have equal non-zero length.");
    }

    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    actual.forEach((value, i) => {
        const prediction = predicted[i];

        if (value === 1 && prediction === 1) tp++;
        else if (value === 0 && prediction === 0) tn++;
        else if (value === 0 && prediction === 1) fp++;
        else if (value === 1 && prediction === 0) fn++;
    });

    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0
        ? 0
        : (2 * precision * recall) / (precision + recall);

    return {
        accuracy: (tp + tn) / actual.length,
        precision,
        recall,
        f1,
        tp,
        tn,
        fp,
        fn
    };
}

function positiveRate(rows) {
    if (!rows.length) return 0;
    return rows.reduce((sum, row) => sum + row.label, 0) / rows.length;
}

class ExperimentRunner {
    constructor(rows) {
        this.rows = rows;
        this.events = new EventTarget();
        this.history = [];
    }

    on(type, callback) {
        this.events.addEventListener(type, callback);
    }

    emit(type, detail) {
        this.events.dispatchEvent(new CustomEvent(type, { detail }));
    }

    run(split) {
        this.emit("split-created", {
            train: split.train.length,
            validation: split.validation.length,
            test: split.test.length
        });

        const scaler = new StandardScaler();

        // Only training data can influence learned preprocessing parameters.
        const trainFeatures = scaler.fitTransform(split.train.map(featureVector));
        const validationFeatures = scaler.transform(split.validation.map(featureVector));

        const classifier = new NearestCentroidClassifier()
            .fit(trainFeatures, split.train.map(row => row.label));

        const validationPredictions = classifier.predict(validationFeatures);
        const validationMetrics = metrics(
            split.validation.map(row => row.label),
            validationPredictions
        );

        this.history.push({
            stage: "validation",
            metrics: validationMetrics
        });

        this.emit("validation-complete", validationMetrics);

        // Test data enters only after the configuration decision is complete.
        const testFeatures = scaler.transform(split.test.map(featureVector));
        const testPredictions = classifier.predict(testFeatures);
        const testMetrics = metrics(
            split.test.map(row => row.label),
            testPredictions
        );

        this.history.push({
            stage: "test",
            metrics: testMetrics
        });

        this.emit("test-complete", testMetrics);

        return { scaler, classifier, validationMetrics, testMetrics };
    }
}

function detectAuthorOverlap(split) {
    const sets = {
        train: new Set(split.train.map(row => row.authorId)),
        validation: new Set(split.validation.map(row => row.authorId)),
        test: new Set(split.test.map(row => row.authorId))
    };

    return {
        trainValidation: [...sets.train].filter(x => sets.validation.has(x)),
        trainTest: [...sets.train].filter(x => sets.test.has(x)),
        validationTest: [...sets.validation].filter(x => sets.test.has(x))
    };
}

function demonstrateLeakage(rows) {
    const split = splitByRatios(rows, 41);

    const proper = new StandardScaler()
        .fit(split.train.map(featureVector));

    // This is intentionally incorrect. It learns distribution parameters from
    // held-out observations before the experiment is evaluated.
    const contaminated = new StandardScaler()
        .fit([
            ...split.train.map(featureVector),
            ...split.validation.map(featureVector),
            ...split.test.map(featureVector)
        ]);

    console.log("\nLeakage demonstration");
    console.log("Training-only means:", proper.means.map(x => x.toFixed(2)));
    console.log("All-data means:", contaminated.means.map(x => x.toFixed(2)));
    console.log(
        "Preprocessing leakage detected:",
        JSON.stringify(proper.means) !== JSON.stringify(contaminated.means)
    );
}

function main() {
    console.log("TRAIN / VALIDATION / TEST EXPERIMENT");
    console.log("====================================");

    const rows = makeDataset();

    const random = splitByRatios(rows, 42);
    const stratified = stratifiedSplit(rows, 42);
    const chronological = chronologicalSplit(rows);

    console.log("\nPartition sizes");
    console.log("Random:", random.train.length, random.validation.length, random.test.length);
    console.log("Stratified:", stratified.train.length, stratified.validation.length, stratified.test.length);
    console.log("Chronological:", chronological.train.length, chronological.validation.length, chronological.test.length);

    console.log("\nClass distribution");
    console.log("Random train positive rate:", positiveRate(random.train).toFixed(3));
    console.log("Random test positive rate:", positiveRate(random.test).toFixed(3));
    console.log("Stratified train positive rate:", positiveRate(stratified.train).toFixed(3));
    console.log("Stratified test positive rate:", positiveRate(stratified.test).toFixed(3));

    const runner = new ExperimentRunner(rows);

    runner.on("split-created", event =>
        console.log("\nExperiment split:", event.detail)
    );

    runner.on("validation-complete", event =>
        console.log("Validation F1:", event.detail.f1.toFixed(4))
    );

    runner.on("test-complete", event =>
        console.log("Final test F1:", event.detail.f1.toFixed(4))
    );

    const result = runner.run(stratified);

    console.log("\nFinal test metrics:", result.testMetrics);

    const overlap = detectAuthorOverlap(stratified);
    console.log("\nAuthor overlap in ordinary stratification:");
    console.log(overlap);

    demonstrateLeakage(rows);

    const reproducibleA = splitByRatios(rows, 777);
    const reproducibleB = splitByRatios(rows, 777);

    console.log(
        "\nReproducible partition:",
        reproducibleA.train.map(x => x.id).join(",") ===
        reproducibleB.train.map(x => x.id).join(",")
    );

    console.log("\nInterpretation");
    console.log("Validation performance supports model decisions.");
    console.log("Test performance estimates generalization after those decisions.");
    console.log("A random split is not automatically valid for temporal or grouped data.");
    console.log("A fixed seed controls stochastic partitioning but does not repair data leakage.");
}

main();
