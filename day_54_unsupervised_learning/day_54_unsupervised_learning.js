"use strict";

const fs = require("node:fs");

class ValidationError extends Error {}

class Observation {
    constructor(id, features) {
        if (!id || typeof id !== "string") {
            throw new ValidationError("Observation id must be a non-empty string.");
        }
        if (!Array.isArray(features) || features.length === 0) {
            throw new ValidationError("Observation requires numeric features.");
        }
        if (!features.every(Number.isFinite)) {
            throw new ValidationError("All features must be finite numbers.");
        }

        this.id = id;
        this.features = Object.freeze([...features]);
        Object.freeze(this);
    }
}

class EventBus {
    constructor() {
        this.handlers = new Map();
    }

    on(eventName, handler) {
        if (!this.handlers.has(eventName)) {
            this.handlers.set(eventName, new Set());
        }
        this.handlers.get(eventName).add(handler);
    }

    emit(eventName, payload) {
        const handlers = this.handlers.get(eventName) ?? new Set();
        for (const handler of handlers) {
            handler(payload);
        }
    }
}

function validateDataset(observations) {
    if (!Array.isArray(observations) || observations.length === 0) {
        throw new ValidationError("Dataset cannot be empty.");
    }

    const dimensions = observations[0].features.length;
    const identifiers = new Set();

    for (const observation of observations) {
        if (identifiers.has(observation.id)) {
            throw new ValidationError(`Duplicate observation: ${observation.id}`);
        }

        if (observation.features.length !== dimensions) {
            throw new ValidationError("All observations must have equal dimensions.");
        }

        identifiers.add(observation.id);
    }
}

function distance(a, b) {
    if (a.length !== b.length) {
        throw new ValidationError("Distance vectors must have equal dimensions.");
    }

    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const delta = a[i] - b[i];
        sum += delta * delta;
    }

    return Math.sqrt(sum);
}

function standardize(observations) {
    validateDataset(observations);

    const dimensions = observations[0].features.length;
    const means = [];
    const deviations = [];

    for (let d = 0; d < dimensions; d++) {
        const values = observations.map(o => o.features[d]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance =
            values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
            values.length;

        means.push(mean);
        deviations.push(Math.sqrt(variance));
    }

    return observations.map(observation => {
        const features = observation.features.map((value, index) => {
            return deviations[index] === 0
                ? 0
                : (value - means[index]) / deviations[index];
        });

        return new Observation(observation.id, features);
    });
}

class KMeansEngine {
    constructor(k, maxIterations = 100, tolerance = 1e-7, events = new EventBus()) {
        if (!Number.isInteger(k) || k < 1) {
            throw new ValidationError("k must be a positive integer.");
        }

        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.events = events;
    }

    fit(observations) {
        validateDataset(observations);

        if (this.k > observations.length) {
            throw new ValidationError("k cannot exceed dataset size.");
        }

        const dimensions = observations[0].features.length;
        let centroids = observations
            .slice(0, this.k)
            .map(o => [...o.features]);

        let assignments = Array.from({ length: this.k }, () => []);

        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            assignments = Array.from({ length: this.k }, () => []);

            for (const observation of observations) {
                const distances = centroids.map(c =>
                    distance(observation.features, c)
                );

                const clusterIndex = distances.indexOf(Math.min(...distances));
                assignments[clusterIndex].push(observation);
            }

            const nextCentroids = [];

            for (let cluster = 0; cluster < this.k; cluster++) {
                const members = assignments[cluster];

                if (members.length === 0) {
                    nextCentroids.push([...centroids[cluster]]);
                    continue;
                }

                const centroid = Array(dimensions).fill(0);

                for (const observation of members) {
                    for (let d = 0; d < dimensions; d++) {
                        centroid[d] += observation.features[d];
                    }
                }

                for (let d = 0; d < dimensions; d++) {
                    centroid[d] /= members.length;
                }

                nextCentroids.push(centroid);
            }

            const movement = Math.max(
                ...centroids.map((oldCentroid, index) =>
                    distance(oldCentroid, nextCentroids[index])
                )
            );

            this.events.emit("iteration", {
                iteration: iteration + 1,
                movement
            });

            centroids = nextCentroids;

            if (movement <= this.tolerance) {
                break;
            }
        }

        const result = centroids.map((centroid, index) => ({
            id: index,
            centroid,
            members: assignments[index]
        }));

        this.events.emit("fitted", result);
        return result;
    }
}

function detectAnomalies(observations, multiplier = 2.5) {
    validateDataset(observations);

    if (!Number.isFinite(multiplier) || multiplier < 0) {
        throw new ValidationError("Anomaly multiplier must be non-negative.");
    }

    const dimensions = observations[0].features.length;
    const centroid = Array(dimensions).fill(0);

    for (const observation of observations) {
        observation.features.forEach((value, index) => {
            centroid[index] += value;
        });
    }

    centroid.forEach((_, index) => {
        centroid[index] /= observations.length;
    });

    const distances = observations.map(observation => ({
        id: observation.id,
        distance: distance(observation.features, centroid)
    }));

    const mean =
        distances.reduce((sum, item) => sum + item.distance, 0) /
        distances.length;

    const std = Math.sqrt(
        distances.reduce(
            (sum, item) => sum + (item.distance - mean) ** 2,
            0
        ) / distances.length
    );

    const threshold = mean + multiplier * std;

    return distances.filter(item => item.distance > threshold).map(item => ({
        ...item,
        threshold
    }));
}

async function runWorkflow() {
    const observations = [
        new Observation("customer-01", [1.0, 1.2, 0.9]),
        new Observation("customer-02", [1.1, 0.8, 1.0]),
        new Observation("customer-03", [0.9, 1.0, 1.2]),
        new Observation("customer-04", [7.8, 8.1, 7.5]),
        new Observation("customer-05", [8.2, 7.9, 8.4]),
        new Observation("customer-06", [7.5, 8.3, 8.0]),
        new Observation("customer-07", [4.2, 4.0, 4.1]),
        new Observation("customer-08", [4.5, 4.3, 4.2]),
        new Observation("customer-09", [4.0, 4.4, 4.5]),
        new Observation("customer-10", [25.0, 1.0, 0.5])
    ];

    const events = new EventBus();

    events.on("iteration", event => {
        console.log(
            `iteration=${event.iteration} centroid_movement=${event.movement.toFixed(6)}`
        );
    });

    events.on("fitted", clusters => {
        console.log(`model produced ${clusters.length} clusters`);
    });

    const normalized = standardize(observations);
    const engine = new KMeansEngine(3, 100, 1e-7, events);
    const clusters = engine.fit(normalized);

    console.log("\nClusters");
    for (const cluster of clusters) {
        const members = cluster.members.map(member => member.id).join(", ");
        console.log(
            `Cluster ${cluster.id}: ${members} | centroid=${cluster.centroid
                .map(value => value.toFixed(3))
                .join(", ")}`
        );
    }

    console.log("\nAnomalies");
    const anomalies = detectAnomalies(normalized, 2.0);
    for (const anomaly of anomalies) {
        console.log(
            `${anomaly.id}: distance=${anomaly.distance.toFixed(3)}, ` +
            `threshold=${anomaly.threshold.toFixed(3)}`
        );
    }

    // Node.js standard library is used for a lightweight audit artifact.
    const audit = {
        generatedAt: new Date().toISOString(),
        observationCount: observations.length,
        clusterCount: clusters.length,
        anomalyCount: anomalies.length
    };

    fs.writeFileSync(
        "unsupervised-learning-audit.json",
        JSON.stringify(audit, null, 2),
        { encoding: "utf8", flag: "w" }
    );
}

runWorkflow().catch(error => {
    console.error(`Workflow failed: ${error.message}`);
    process.exitCode = 1;
});
