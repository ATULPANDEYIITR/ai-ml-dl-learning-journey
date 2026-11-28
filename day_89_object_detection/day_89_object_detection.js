"use strict";

/*
 * Object Detection Pipeline
 * Bounding boxes, IoU, NMS, Soft-NMS, asynchronous processing,
 * event-driven detection processing and evaluation.
 *
 * Runtime: Node.js 18+
 * Dependencies: None
 */

const { EventEmitter } = require("node:events");
const fs = require("node:fs/promises");
const path = require("node:path");
const crypto = require("node:crypto");

class DetectionError extends Error {
    constructor(message, code = "DETECTION_ERROR") {
        super(message);
        this.name = "DetectionError";
        this.code = code;
    }
}

function finiteNumber(value, field) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new DetectionError(`${field} must be a finite number.`);
    }
    return value;
}

class BoundingBox {
    constructor(x1, y1, x2, y2) {
        this.x1 = finiteNumber(x1, "x1");
        this.y1 = finiteNumber(y1, "y1");
        this.x2 = finiteNumber(x2, "x2");
        this.y2 = finiteNumber(y2, "y2");

        if (x2 <= x1 || y2 <= y1) {
            throw new DetectionError(
                "Bounding box dimensions must be positive."
            );
        }

        Object.freeze(this);
    }

    get width() {
        return this.x2 - this.x1;
    }

    get height() {
        return this.y2 - this.y1;
    }

    get area() {
        return this.width * this.height;
    }

    get center() {
        return [
            (this.x1 + this.x2) / 2,
            (this.y1 + this.y2) / 2
        ];
    }

    clip(width, height) {
        if (
            !Number.isInteger(width) ||
            !Number.isInteger(height) ||
            width <= 0 ||
            height <= 0
        ) {
            throw new DetectionError("Invalid image dimensions.");
        }

        const x1 = Math.max(0, Math.min(width, this.x1));
        const y1 = Math.max(0, Math.min(height, this.y1));
        const x2 = Math.max(0, Math.min(width, this.x2));
        const y2 = Math.max(0, Math.min(height, this.y2));

        if (x2 <= x1 || y2 <= y1) {
            return null;
        }

        return new BoundingBox(x1, y1, x2, y2);
    }

    toJSON() {
        return {
            x1: this.x1,
            y1: this.y1,
            x2: this.x2,
            y2: this.y2
        };
    }

    static fromXYWH(x, y, width, height) {
        return new BoundingBox(x, y, x + width, y + height);
    }
}

class Detection {
    constructor({
        box,
        confidence,
        classId,
        label,
        imageId
    }) {
        if (!(box instanceof BoundingBox)) {
            throw new DetectionError("A valid BoundingBox is required.");
        }

        confidence = finiteNumber(confidence, "confidence");

        if (confidence < 0 || confidence > 1) {
            throw new DetectionError("Confidence must be between 0 and 1.");
        }

        if (!Number.isInteger(classId) || classId < 0) {
            throw new DetectionError("Invalid class ID.");
        }

        if (typeof label !== "string" || !label.trim()) {
            throw new DetectionError("Class label is required.");
        }

        if (typeof imageId !== "string" || !imageId.trim()) {
            throw new DetectionError("Image ID is required.");
        }

        this.box = box;
        this.confidence = confidence;
        this.classId = classId;
        this.label = label;
        this.imageId = imageId;

        Object.freeze(this);
    }

    withConfidence(confidence) {
        return new Detection({
            box: this.box,
            confidence,
            classId: this.classId,
            label: this.label,
            imageId: this.imageId
        });
    }

    withBox(box) {
        return new Detection({
            box,
            confidence: this.confidence,
            classId: this.classId,
            label: this.label,
            imageId: this.imageId
        });
    }

    toJSON() {
        return {
            imageId: this.imageId,
            classId: this.classId,
            label: this.label,
            confidence: this.confidence,
            bbox: this.box.toJSON()
        };
    }
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersection =
        Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    const union = a.area + b.area - intersection;

    return union > 0 ? intersection / union : 0;
}

function generalizedIoU(a, b) {
    const overlap = iou(a, b);

    const x1 = Math.min(a.x1, b.x1);
    const y1 = Math.min(a.y1, b.y1);
    const x2 = Math.max(a.x2, b.x2);
    const y2 = Math.max(a.y2, b.y2);

    const enclosingArea = (x2 - x1) * (y2 - y1);
    const union = a.area + b.area -
        Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) *
        Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));

    return enclosingArea > 0
        ? overlap - (enclosingArea - union) / enclosingArea
        : overlap;
}

function validateDetections(detections) {
    if (!Array.isArray(detections)) {
        throw new DetectionError("Detections must be an array.");
    }

    for (const detection of detections) {
        if (!(detection instanceof Detection)) {
            throw new DetectionError("Invalid detection object.");
        }
    }

    return detections;
}

function confidenceFilter(detections, threshold) {
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
        throw new DetectionError("Invalid confidence threshold.");
    }

    return validateDetections(detections).filter(
        detection => detection.confidence >= threshold
    );
}

/*
 * Sorting is performed once. The algorithm repeatedly retains the
 * highest-confidence box and removes excessive overlaps.
 */
function nms(
    detections,
    {
        iouThreshold = 0.45,
        classAgnostic = false,
        maxDetections = 100
    } = {}
) {
    validateDetections(detections);

    if (
        !Number.isFinite(iouThreshold) ||
        iouThreshold < 0 ||
        iouThreshold > 1
    ) {
        throw new DetectionError("Invalid IoU threshold.");
    }

    if (!Number.isInteger(maxDetections) || maxDetections < 1) {
        throw new DetectionError("Invalid maximum detection count.");
    }

    let candidates = [...detections].sort(
        (a, b) => b.confidence - a.confidence
    );

    const selected = [];

    while (candidates.length && selected.length < maxDetections) {
        const best = candidates.shift();
        selected.push(best);

        candidates = candidates.filter(candidate => {
            if (
                !classAgnostic &&
                candidate.classId !== best.classId
            ) {
                return true;
            }

            return iou(best.box, candidate.box) <= iouThreshold;
        });
    }

    return selected;
}

function softNMS(
    detections,
    {
        iouThreshold = 0.45,
        sigma = 0.5,
        scoreThreshold = 0.001,
        maxDetections = 100
    } = {}
) {
    validateDetections(detections);

    if (
        !Number.isFinite(sigma) ||
        sigma <= 0 ||
        !Number.isFinite(scoreThreshold) ||
        scoreThreshold < 0 ||
        scoreThreshold > 1
    ) {
        throw new DetectionError("Invalid Soft-NMS configuration.");
    }

    const candidates = detections.map(detection => ({
        detection,
        score: detection.confidence
    }));

    const selected = [];

    while (candidates.length && selected.length < maxDetections) {
        candidates.sort((a, b) => b.score - a.score);

        const best = candidates.shift();

        if (best.score < scoreThreshold) break;

        selected.push(best.detection.withConfidence(best.score));

        for (let i = candidates.length - 1; i >= 0; i--) {
            const candidate = candidates[i];

            if (
                candidate.detection.classId !==
                best.detection.classId
            ) {
                continue;
            }

            const overlap = iou(
                best.detection.box,
                candidate.detection.box
            );

            if (overlap > iouThreshold) {
                candidate.score *= Math.exp(
                    -(overlap * overlap) / sigma
                );
            }

            if (candidate.score < scoreThreshold) {
                candidates.splice(i, 1);
            }
        }
    }

    return selected;
}

function averagePrecision(
    predictions,
    groundTruth,
    iouThreshold = 0.5
) {
    validateDetections(predictions);
    validateDetections(groundTruth);

    if (groundTruth.length === 0) return 0;

    const ordered = [...predictions].sort(
        (a, b) => b.confidence - a.confidence
    );

    const matched = new Set();
    const precision = [0];
    const recall = [0];

    let tp = 0;
    let fp = 0;

    for (const prediction of ordered) {
        let bestIndex = -1;
        let bestOverlap = -1;

        groundTruth.forEach((target, index) => {
            if (
                matched.has(index) ||
                target.imageId !== prediction.imageId ||
                target.classId !== prediction.classId
            ) {
                return;
            }

            const overlap = iou(prediction.box, target.box);

            if (overlap >= iouThreshold && overlap > bestOverlap) {
                bestOverlap = overlap;
                bestIndex = index;
            }
        });

        if (bestIndex >= 0) {
            matched.add(bestIndex);
            tp++;
        } else {
            fp++;
        }

        precision.push(tp / (tp + fp));
        recall.push(tp / groundTruth.length);
    }

    precision.push(0);
    recall.push(1);

    for (let i = precision.length - 2; i >= 0; i--) {
        precision[i] = Math.max(precision[i], precision[i + 1]);
    }

    let ap = 0;

    for (let i = 1; i < recall.length; i++) {
        ap += (recall[i] - recall[i - 1]) * precision[i];
    }

    return ap;
}

class DetectionPipeline extends EventEmitter {
    constructor({
        confidenceThreshold = 0.25,
        iouThreshold = 0.45,
        maxDetections = 100,
        classAgnostic = false,
        concurrency = 2
    } = {}) {
        super();

        if (
            !Number.isInteger(concurrency) ||
            concurrency < 1
        ) {
            throw new DetectionError("Invalid concurrency setting.");
        }

        this.config = Object.freeze({
            confidenceThreshold,
            iouThreshold,
            maxDetections,
            classAgnostic
        });

        this.concurrency = concurrency;
        this.active = 0;
        this.queue = [];
        this.closed = false;

        this.metrics = {
            imagesProcessed: 0,
            detectionsReceived: 0,
            detectionsReturned: 0,
            failures: 0
        };
    }

    process(imageId, detections, width, height) {
        if (this.closed) {
            return Promise.reject(
                new DetectionError("Pipeline is closed.")
            );
        }

        return new Promise((resolve, reject) => {
            this.queue.push({
                imageId,
                detections,
                width,
                height,
                resolve,
                reject
            });

            this.drain();
        });
    }

    drain() {
        while (
            !this.closed &&
            this.active < this.concurrency &&
            this.queue.length
        ) {
            const task = this.queue.shift();
            this.active++;

            this.execute(task)
                .then(task.resolve, task.reject)
                .finally(() => {
                    this.active--;
                    this.drain();
                });
        }
    }

    async execute(task) {
        try {
            const {
                imageId,
                detections,
                width,
                height
            } = task;

            validateDetections(detections);

            if (!imageId || !Number.isInteger(width) ||
                !Number.isInteger(height) || width <= 0 ||
                height <= 0) {
                throw new DetectionError("Invalid image metadata.");
            }

            if (detections.some(d => d.imageId !== imageId)) {
                throw new DetectionError("Mixed image identifiers.");
            }

            const start = process.hrtime.bigint();

            const clipped = detections
                .map(detection => {
                    const box = detection.box.clip(width, height);
                    return box ? detection.withBox(box) : null;
                })
                .filter(Boolean);

            const filtered = confidenceFilter(
                clipped,
                this.config.confidenceThreshold
            );

            const output = nms(filtered, {
                iouThreshold: this.config.iouThreshold,
                classAgnostic: this.config.classAgnostic,
                maxDetections: this.config.maxDetections
            });

            const elapsedMs = Number(
                process.hrtime.bigint() - start
            ) / 1e6;

            this.metrics.imagesProcessed++;
            this.metrics.detectionsReceived += detections.length;
            this.metrics.detectionsReturned += output.length;

            const result = Object.freeze({
                imageId,
                detections: output,
                elapsedMs
            });

            this.emit("processed", result);

            return result;
        } catch (error) {
            this.metrics.failures++;
            this.emit("pipelineError", {
                imageId: task.imageId,
                error
            });
            throw error;
        }
    }

    close() {
        this.closed = true;

        const error = new DetectionError("Pipeline closed.");

        for (const task of this.queue.splice(0)) {
            task.reject(error);
        }
    }

    getReport() {
        return {
            ...this.metrics,
            queuedTasks: this.queue.length,
            activeTasks: this.active
        };
    }
}

async function saveResults(detections, filename) {
    validateDetections(detections);

    const resolved = path.resolve(filename);

    if (path.basename(resolved) !== filename) {
        throw new DetectionError(
            "Only a filename in the current directory is allowed."
        );
    }

    const temporary = `${resolved}.${crypto.randomUUID()}.tmp`;

    const content = JSON.stringify(
        detections.map(d => d.toJSON()),
        null,
        2
    );

    try {
        await fs.writeFile(temporary, content, {
            encoding: "utf8",
            flag: "wx",
            mode: 0o600
        });

        await fs.rename(temporary, resolved);
    } catch (error) {
        await fs.rm(temporary, { force: true }).catch(() => {});
        throw error;
    }
}

async function main() {
    const boxA = new BoundingBox(50, 40, 200, 220);
    const boxB = new BoundingBox(55, 45, 198, 218);

    console.assert(iou(boxA, boxA) === 1);
    console.assert(iou(
        new BoundingBox(0, 0, 10, 10),
        new BoundingBox(10, 0, 20, 10)
    ) === 0);

    console.assert(generalizedIoU(boxA, boxA) === 1);

    const detections = [
        new Detection({
            box: boxA,
            confidence: 0.96,
            classId: 0,
            label: "person",
            imageId: "image-001"
        }),
        new Detection({
            box: boxB,
            confidence: 0.89,
            classId: 0,
            label: "person",
            imageId: "image-001"
        }),
        new Detection({
            box: new BoundingBox(300, 100, 430, 260),
            confidence: 0.91,
            classId: 2,
            label: "car",
            imageId: "image-001"
        }),
        new Detection({
            box: new BoundingBox(500, 50, 580, 130),
            confidence: 0.18,
            classId: 0,
            label: "person",
            imageId: "image-001"
        })
    ];

    const pipeline = new DetectionPipeline({
        confidenceThreshold: 0.25,
        iouThreshold: 0.45,
        concurrency: 2
    });

    pipeline.on("processed", result => {
        console.log(
            `Processed ${result.imageId} in ${result.elapsedMs.toFixed(3)} ms`
        );
    });

    pipeline.on("pipelineError", event => {
        console.error(event.imageId, event.error.message);
    });

    const result = await pipeline.process(
        "image-001",
        detections,
        640,
        480
    );

    console.log("\nDETECTIONS");

    for (const detection of result.detections) {
        console.log(detection.toJSON());
    }

    const targets = [
        new Detection({
            box: new BoundingBox(52, 42, 202, 222),
            confidence: 1,
            classId: 0,
            label: "person",
            imageId: "image-001"
        })
    ];

    console.log("\nAP:", averagePrecision(
        result.detections.filter(d => d.classId === 0),
        targets,
        0.5
    ));

    console.log("\nPIPELINE METRICS:", pipeline.getReport());

    await saveResults(result.detections, "javascript-detections.json");

    pipeline.close();
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
