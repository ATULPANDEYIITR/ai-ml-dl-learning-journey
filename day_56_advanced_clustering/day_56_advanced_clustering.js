"use strict";

const fs = require("node:fs/promises");

class ClusteringError extends Error {}

class Point {
    constructor(id, x, y, label = "") {
        if (!Number.isInteger(id) || id < 0) {
            throw new ClusteringError("Point ID must be a non-negative integer.");
        }
        if (![x, y].every(Number.isFinite)) {
            throw new ClusteringError(`Invalid coordinates for point ${id}.`);
        }

        this.id = id;
        this.x = x;
        this.y = y;
        this.label = label;
    }

    distance(other) {
        return Math.hypot(this.x - other.x, this.y - other.y);
    }
}

class EventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
    }

    emit(event, payload) {
        for (const listener of this.listeners.get(event) ?? []) {
            listener(payload);
        }
    }
}

function validatePoints(points) {
    if (!Array.isArray(points) || points.length === 0) {
        throw new ClusteringError("At least one point is required.");
    }

    const ids = new Set();

    for (const point of points) {
        if (!(point instanceof Point)) {
            throw new ClusteringError("All observations must be Point instances.");
        }
        if (ids.has(point.id)) {
            throw new ClusteringError(`Duplicate point ID: ${point.id}`);
        }
        ids.add(point.id);
    }
}

function linkageDistance(left, right, type) {
    const distances = [];

    for (const a of left) {
        for (const b of right) {
            distances.push(a.distance(b));
        }
    }

    if (type === "single") return Math.min(...distances);
    if (type === "complete") return Math.max(...distances);
    if (type === "average") {
        return distances.reduce((sum, value) => sum + value, 0) / distances.length;
    }

    throw new ClusteringError("Unknown linkage method.");
}

function hierarchicalClustering(points, targetClusters, linkage = "average") {
    validatePoints(points);

    if (!Number.isInteger(targetClusters) ||
        targetClusters < 1 ||
        targetClusters > points.length) {
        throw new ClusteringError("Invalid target cluster count.");
    }

    const clusters = points.map(point => [point]);

    while (clusters.length > targetClusters) {
        let bestI = -1;
        let bestJ = -1;
        let bestDistance = Infinity;

        for (let i = 0; i < clusters.length; i++) {
            for (let j = i + 1; j < clusters.length; j++) {
                const distance = linkageDistance(
                    clusters[i],
                    clusters[j],
                    linkage
                );

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestI < 0) {
            throw new ClusteringError("Unable to find a merge candidate.");
        }

        clusters[bestI].push(...clusters[bestJ]);
        clusters.splice(bestJ, 1);
    }

    return clusters;
}

function dbscan(points, eps, minSamples, eventBus = null) {
    validatePoints(points);

    if (!(eps > 0) || !Number.isFinite(eps)) {
        throw new ClusteringError("eps must be a positive finite number.");
    }
    if (!Number.isInteger(minSamples) || minSamples < 1) {
        throw new ClusteringError("minSamples must be a positive integer.");
    }

    const neighborhoods = new Map();
    const labels = new Map();
    const corePoints = new Set();
    const noisePoints = new Set();

    function getNeighbors(point) {
        const result = [];

        for (const candidate of points) {
            if (point.distance(candidate) <= eps) {
                result.push(candidate.id);
            }
        }

        return result;
    }

    for (const point of points) {
        neighborhoods.set(point.id, getNeighbors(point));
    }

    for (const [id, neighbors] of neighborhoods) {
        if (neighbors.length >= minSamples) {
            corePoints.add(id);
        }
    }

    const visited = new Set();
    let clusterId = 0;

    for (const point of points) {
        if (visited.has(point.id)) continue;

        visited.add(point.id);
        const neighbors = neighborhoods.get(point.id);

        if (neighbors.length < minSamples) {
            labels.set(point.id, -1);
            noisePoints.add(point.id);
            eventBus?.emit("noise-detected", { pointId: point.id });
            continue;
        }

        labels.set(point.id, clusterId);
        eventBus?.emit("cluster-created", {
            clusterId,
            seedPoint: point.id
        });

        const queue = [...neighbors];

        while (queue.length > 0) {
            const candidateId = queue.shift();

            if (!visited.has(candidateId)) {
                visited.add(candidateId);
                const candidateNeighbors = neighborhoods.get(candidateId);

                if (candidateNeighbors.length >= minSamples) {
                    for (const neighborId of candidateNeighbors) {
                        if (!queue.includes(neighborId)) {
                            queue.push(neighborId);
                        }
                    }
                }
            }

            if (!labels.has(candidateId) || labels.get(candidateId) === -1) {
                labels.set(candidateId, clusterId);
                noisePoints.delete(candidateId);
            }
        }

        clusterId++;
    }

    return { labels, corePoints, noisePoints };
}

function silhouetteScore(points, labels) {
    validatePoints(points);

    const groups = new Map();

    for (const point of points) {
        const label = labels.get(point.id);
        if (label === undefined || label === -1) continue;

        if (!groups.has(label)) groups.set(label, []);
        groups.get(label).push(point);
    }

    if (groups.size < 2) return 0;

    const scores = [];

    for (const point of points) {
        const label = labels.get(point.id);
        if (label === undefined || label === -1) continue;

        const own = groups.get(label);

        const ownDistances = own
            .filter(other => other.id !== point.id)
            .map(other => point.distance(other));

        const a = ownDistances.length === 0
            ? 0
            : ownDistances.reduce((sum, d) => sum + d, 0) / ownDistances.length;

        let b = Infinity;

        for (const [otherLabel, cluster] of groups) {
            if (otherLabel === label) continue;

            const mean = cluster
                .map(other => point.distance(other))
                .reduce((sum, d) => sum + d, 0) / cluster.length;

            b = Math.min(b, mean);
        }

        const denominator = Math.max(a, b);
        scores.push(denominator === 0 ? 0 : (b - a) / denominator);
    }

    return scores.length === 0
        ? 0
        : scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

async function saveReport(filename, payload) {
    const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
    await fs.writeFile(
        safeName,
        JSON.stringify(payload, null, 2),
        { encoding: "utf8", mode: 0o600 }
    );
}

function createDataset() {
    return [
        new Point(1, 1.0, 1.0, "A"),
        new Point(2, 1.2, 0.9, "A"),
        new Point(3, 0.8, 1.1, "A"),
        new Point(4, 1.1, 1.3, "A"),
        new Point(5, 10.0, 10.0, "B"),
        new Point(6, 10.2, 10.1, "B"),
        new Point(7, 9.8, 10.3, "B"),
        new Point(8, 10.1, 9.7, "B"),
        new Point(9, 20.0, 2.0, "noise")
    ];
}

function labelsFromClusters(clusters) {
    const labels = new Map();

    clusters.forEach((cluster, clusterId) => {
        for (const point of cluster) {
            labels.set(point.id, clusterId);
        }
    });

    return labels;
}

async function main() {
    const bus = new EventBus();

    bus.on("cluster-created", event => {
        console.log(
            `DBSCAN cluster ${event.clusterId} started at point ${event.seedPoint}`
        );
    });

    bus.on("noise-detected", event => {
        console.log(`Noise candidate detected: point ${event.pointId}`);
    });

    const points = createDataset();

    const hierarchical = hierarchicalClustering(
        points,
        2,
        "average"
    );

    console.log("\nHierarchical clusters:");
    hierarchical.forEach((cluster, index) => {
        console.log(
            `  ${index}:`,
            cluster.map(point => point.id).sort((a, b) => a - b)
        );
    });

    const hierarchicalLabels = labelsFromClusters(hierarchical);

    console.log(
        "Hierarchical silhouette:",
        silhouetteScore(points, hierarchicalLabels).toFixed(4)
    );

    const density = dbscan(points, 1.0, 3, bus);

    console.log("\nDBSCAN clusters:");
    const grouped = new Map();

    for (const [pointId, clusterId] of density.labels) {
        if (!grouped.has(clusterId)) grouped.set(clusterId, []);
        grouped.get(clusterId).push(pointId);
    }

    for (const [clusterId, ids] of grouped) {
        console.log(
            `  ${clusterId === -1 ? "Noise" : `Cluster ${clusterId}`}:`,
            ids.sort((a, b) => a - b)
        );
    }

    console.log(
        "DBSCAN silhouette:",
        silhouetteScore(points, density.labels).toFixed(4)
    );

    await saveReport("clustering-report.json", {
        hierarchical: hierarchical.map(cluster =>
            cluster.map(point => point.id)
        ),
        dbscan: {
            labels: Object.fromEntries(density.labels),
            corePoints: [...density.corePoints].sort((a, b) => a - b),
            noisePoints: [...density.noisePoints].sort((a, b) => a - b)
        }
    });

    console.log("\nReport written to clustering-report.json");
}

main().catch(error => {
    console.error(`Clustering failed: ${error.message}`);
    process.exitCode = 1;
});
