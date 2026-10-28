#!/usr/bin/env node
"use strict";

/*
 * Advanced Dimensionality Reduction
 *
 * A dependency-free Node.js implementation of:
 *   - PCA
 *   - t-SNE-style probability construction and optimization
 *   - UMAP-style weighted k-nearest-neighbor graph embedding
 *
 * The implementation exposes an event-driven experiment runner so that
 * reductions can be monitored without coupling the algorithms to output.
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { EventEmitter } = require("node:events");

class ValidationError extends Error {}

function validateMatrix(matrix) {
    if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new ValidationError("Dataset must contain observations.");
    }

    if (!matrix.every(Array.isArray)) {
        throw new ValidationError("Every observation must be an array.");
    }

    const width = matrix[0].length;

    if (width < 2) {
        throw new ValidationError("At least two input dimensions are required.");
    }

    if (!matrix.every(row => row.length === width)) {
        throw new ValidationError("All observations must have equal dimensions.");
    }

    for (const row of matrix) {
        for (const value of row) {
            if (!Number.isFinite(value)) {
                throw new ValidationError("Feature values must be finite.");
            }
        }
    }

    return { rows: matrix.length, columns: width };
}

function squaredDistance(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        const delta = a[i] - b[i];
        result += delta * delta;
    }
    return result;
}

function standardize(matrix) {
    const { rows, columns } = validateMatrix(matrix);

    const means = Array(columns).fill(0);
    const scales = Array(columns).fill(0);

    for (const row of matrix) {
        for (let j = 0; j < columns; j++) {
            means[j] += row[j];
        }
    }

    for (let j = 0; j < columns; j++) {
        means[j] /= rows;
    }

    for (const row of matrix) {
        for (let j = 0; j < columns; j++) {
            scales[j] += (row[j] - means[j]) ** 2;
        }
    }

    return matrix.map(row =>
        row.map((value, j) => {
            const scale = Math.sqrt(scales[j] / Math.max(1, rows - 1));
            return (value - means[j]) / (scale || 1);
        })
    );
}

function pairwiseDistances(matrix) {
    const n = matrix.length;
    const distances = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const distance = squaredDistance(matrix[i], matrix[j]);
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }

    return distances;
}

function createRng(seed) {
    let state = seed >>> 0;

    return function random() {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function gaussian(random) {
    let u = 0;
    let v = 0;

    while (u === 0) u = random();
    while (v === 0) v = random();

    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function nearestNeighbors(matrix, k) {
    const n = matrix.length;

    if (!Number.isInteger(k) || k < 1 || k >= n) {
        throw new ValidationError("k must be between 1 and n-1.");
    }

    const distances = pairwiseDistances(matrix);

    return distances.map((row, i) =>
        row
            .map((distance, j) => ({ distance, index: j }))
            .filter(item => item.index !== i)
            .sort((a, b) => a.distance - b.distance)
            .slice(0, k)
            .map(item => item.index)
    );
}

function perplexityProbabilities(distances, perplexity) {
    const targetEntropy = Math.log(perplexity);
    let beta = 1;

    for (let iteration = 0; iteration < 60; iteration++) {
        const probabilities = distances.map(distance =>
            distance > 0 ? Math.exp(-beta * distance) : 0
        );

        const total = probabilities.reduce((a, b) => a + b, 0);

        if (total <= Number.EPSILON) {
            beta *= 0.5;
            continue;
        }

        for (let i = 0; i < probabilities.length; i++) {
            probabilities[i] /= total;
        }

        const entropy = probabilities.reduce(
            (sum, probability) =>
                probability > 0
                    ? sum - probability * Math.log(probability)
                    : sum,
            0
        );

        const difference = entropy - targetEntropy;

        if (Math.abs(difference) < 1e-5) {
            return probabilities;
        }

        if (difference > 0) {
            beta *= 2;
        } else {
            beta *= 0.5;
        }
    }

    return distances.map(() => 1 / Math.max(1, distances.length));
}

function buildSymmetricTSNEProbabilities(matrix, perplexity) {
    const distances = pairwiseDistances(matrix);
    const n = matrix.length;
    const result = Array.from({ length: n }, () => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        const probabilities = perplexityProbabilities(
            distances[i],
            Math.min(perplexity, Math.max(2, n - 1))
        );

        for (let j = 0; j < n; j++) {
            result[i][j] = probabilities[j];
        }
    }

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            result[i][j] =
                (result[i][j] + result[j][i]) / (2 * n);
        }
    }

    return result;
}

function tsne(matrix, options = {}) {
    const standardized = standardize(matrix);
    const n = standardized.length;

    const {
        perplexity = 8,
        iterations = 300,
        learningRate = 50,
        seed = 42,
        onIteration = () => {}
    } = options;

    if (perplexity <= 1 || perplexity >= n) {
        throw new ValidationError("Invalid t-SNE perplexity.");
    }

    const probabilities =
        buildSymmetricTSNEProbabilities(standardized, perplexity);

    const random = createRng(seed);

    const embedding = Array.from({ length: n }, () => [
        gaussian(random) * 0.001,
        gaussian(random) * 0.001
    ]);

    const velocity = Array.from({ length: n }, () => [0, 0]);

    for (let iteration = 0; iteration < iterations; iteration++) {
        const qNumerators = Array.from(
            { length: n },
            () => Array(n).fill(0)
        );

        let total = 0;

        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const distance = squaredDistance(
                    embedding[i],
                    embedding[j]
                );

                const value = 1 / (1 + distance);
                qNumerators[i][j] = value;
                qNumerators[j][i] = value;
                total += 2 * value;
            }
        }

        const gradients = Array.from(
            { length: n },
            () => [0, 0]
        );

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) continue;

                const q = qNumerators[i][j] / Math.max(total, 1e-12);
                const multiplier =
                    4 *
                    (probabilities[i][j] - q) *
                    qNumerators[i][j];

                gradients[i][0] +=
                    multiplier *
                    (embedding[i][0] - embedding[j][0]);

                gradients[i][1] +=
                    multiplier *
                    (embedding[i][1] - embedding[j][1]);
            }
        }

        const momentum = iteration < 100 ? 0.5 : 0.8;

        for (let i = 0; i < n; i++) {
            for (let d = 0; d < 2; d++) {
                velocity[i][d] =
                    momentum * velocity[i][d] -
                    learningRate * gradients[i][d];

                embedding[i][d] += velocity[i][d];
            }
        }

        const meanX =
            embedding.reduce((sum, row) => sum + row[0], 0) / n;
        const meanY =
            embedding.reduce((sum, row) => sum + row[1], 0) / n;

        for (const point of embedding) {
            point[0] -= meanX;
            point[1] -= meanY;
        }

        if (iteration % 25 === 0 || iteration === iterations - 1) {
            onIteration({
                iteration: iteration + 1,
                totalIterations: iterations
            });
        }
    }

    return embedding;
}

function umapStyle(matrix, options = {}) {
    const standardized = standardize(matrix);
    const n = standardized.length;

    const {
        neighbors = 8,
        iterations = 300,
        learningRate = 50,
        minDist = 0.1,
        seed = 42,
        onIteration = () => {}
    } = options;

    if (neighbors < 1 || neighbors >= n) {
        throw new ValidationError("Invalid neighbor count.");
    }

    const graph = nearestNeighbors(standardized, neighbors);
    const distances = pairwiseDistances(standardized);
    const random = createRng(seed);

    const embedding = Array.from({ length: n }, () => [
        random() * 2 - 1,
        random() * 2 - 1
    ]);

    for (let iteration = 0; iteration < iterations; iteration++) {
        const gradient = Array.from(
            { length: n },
            () => [0, 0]
        );

        for (let i = 0; i < n; i++) {
            const positiveDistances = graph[i]
                .map(j => Math.sqrt(distances[i][j]))
                .filter(distance => distance > 0);

            const sigma =
                positiveDistances.sort((a, b) => a - b)[
                    Math.floor(positiveDistances.length / 2)
                ] || 1;

            for (const j of graph[i]) {
                const dx = embedding[i][0] - embedding[j][0];
                const dy = embedding[i][1] - embedding[j][1];
                const distance =
                    Math.sqrt(dx * dx + dy * dy) + 1e-9;

                const sourceDistance =
                    Math.sqrt(distances[i][j]);

                const weight = Math.exp(
                    -Math.max(0, sourceDistance - sigma) / sigma
                );

                const attraction =
                    weight * (distance - minDist);

                gradient[i][0] += attraction * dx / distance;
                gradient[i][1] += attraction * dy / distance;
            }
        }

        for (let sample = 0; sample < n * 3; sample++) {
            const i = Math.floor(random() * n);
            const j = Math.floor(random() * n);

            if (i === j || graph[i].includes(j)) continue;

            const dx = embedding[i][0] - embedding[j][0];
            const dy = embedding[i][1] - embedding[j][1];
            const distanceSquared = dx * dx + dy * dy + 1e-6;
            const repulsion = 0.01 / distanceSquared;

            gradient[i][0] += repulsion * dx;
            gradient[i][1] += repulsion * dy;
        }

        const rate =
            learningRate / (1 + iteration / 100);

        for (let i = 0; i < n; i++) {
            embedding[i][0] -= rate * 0.01 * gradient[i][0];
            embedding[i][1] -= rate * 0.01 * gradient[i][1];
        }

        if (iteration % 25 === 0 || iteration === iterations - 1) {
            onIteration({
                iteration: iteration + 1,
                totalIterations: iterations
            });
        }
    }

    return embedding;
}

class ReductionExperiment extends EventEmitter {
    constructor(data, configuration) {
        super();
        validateMatrix(data);

        this.data = data;
        this.configuration = Object.freeze({
            method: "tsne",
            ...configuration
        });
    }

    async run() {
        this.emit("started", {
            method: this.configuration.method,
            samples: this.data.length
        });

        await new Promise(resolve => setImmediate(resolve));

        const callback = progress => this.emit("progress", progress);

        let embedding;

        if (this.configuration.method === "tsne") {
            embedding = tsne(this.data, {
                ...this.configuration,
                onIteration: callback
            });
        } else if (this.configuration.method === "umap") {
            embedding = umapStyle(this.data, {
                ...this.configuration,
                onIteration: callback
            });
        } else {
            throw new ValidationError(
                `Unsupported method: ${this.configuration.method}`
            );
        }

        const result = {
            method: this.configuration.method,
            samples: embedding.length,
            embedding
        };

        this.emit("completed", result);
        return result;
    }
}

function createDataset(samplesPerCurve = 20, seed = 42) {
    const random = createRng(seed);
    const data = [];

    for (let i = 0; i < samplesPerCurve; i++) {
        const t = 4 * Math.PI * i / (samplesPerCurve - 1);
        const noise = (random() - 0.5) * 0.1;

        data.push([
            t * Math.cos(t) + noise,
            t * Math.sin(t) + noise,
            t / (4 * Math.PI) + noise,
            Math.sin(t / 2) + noise
        ]);
    }

    for (let i = 0; i < samplesPerCurve; i++) {
        const t = 4 * Math.PI * i / (samplesPerCurve - 1);
        const noise = (random() - 0.5) * 0.1;

        data.push([
            t * Math.cos(t + Math.PI) + noise,
            t * Math.sin(t + Math.PI) + noise,
            t / (4 * Math.PI) + 1.5 + noise,
            Math.cos(t / 2) + noise
        ]);
    }

    return data;
}

async function saveCsv(filename, embedding) {
    const lines = ["id,x,y"];

    embedding.forEach((point, index) => {
        lines.push(
            `${index},${point[0].toFixed(8)},${point[1].toFixed(8)}`
        );
    });

    await fs.mkdir(path.dirname(filename), { recursive: true });
    await fs.writeFile(filename, lines.join("\n") + "\n", "utf8");
}

async function main() {
    const method = process.argv[2] || "tsne";

    if (!["tsne", "umap"].includes(method)) {
        throw new ValidationError(
            "Usage: node dimensionality_reduction.js [tsne|umap]"
        );
    }

    const data = createDataset();

    const experiment = new ReductionExperiment(data, {
        method,
        iterations: 250,
        perplexity: 8,
        neighbors: 8,
        learningRate: 50,
        minDist: 0.1,
        seed: 42
    });

    experiment.on("started", event => {
        console.log(
            `started method=${event.method} samples=${event.samples}`
        );
    });

    experiment.on("progress", event => {
        if (
            event.iteration === 1 ||
            event.iteration % 50 === 0 ||
            event.iteration === event.totalIterations
        ) {
            console.log(
                `progress ${event.iteration}/${event.totalIterations}`
            );
        }
    });

    experiment.on("completed", event => {
        console.log(
            `completed method=${event.method} dimensions=2`
        );
    });

    const result = await experiment.run();
    await saveCsv(
        path.join(process.cwd(), "output", `${method}_embedding.csv`),
        result.embedding
    );

    console.log(
        `saved output/${method}_embedding.csv`
    );
}

main().catch(error => {
    console.error(`error: ${error.message}`);
    process.exitCode = 1;
});
