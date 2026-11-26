"use strict";

/**
 * Computer Vision Pipeline
 * Event-driven image preprocessing, augmentation, feature extraction,
 * classification, and batch processing.
 *
 * Runtime: Node.js 18+
 * Dependencies: Node.js standard library only.
 *
 * Supported image format: P3 ASCII PPM.
 */

const fs = require("node:fs/promises");
const path = require("node:path");
const { EventEmitter } = require("node:events");
const crypto = require("node:crypto");

class PipelineError extends Error {
    constructor(message, code = "PIPELINE_ERROR") {
        super(message);
        this.name = "PipelineError";
        this.code = code;
    }
}

class ValidationError extends PipelineError {
    constructor(message) {
        super(message, "VALIDATION_ERROR");
        this.name = "ValidationError";
    }
}

function validateDimension(value, name) {
    if (!Number.isInteger(value) || value <= 0 || value > 4096) {
        throw new ValidationError(`${name} must be between 1 and 4096.`);
    }
}

function validateImage(image) {
    if (!image || typeof image !== "object") {
        throw new ValidationError("Image must be an object.");
    }

    validateDimension(image.width, "Width");
    validateDimension(image.height, "Height");

    if (!Number.isInteger(image.maxValue) ||
        image.maxValue < 1 ||
        image.maxValue > 65535) {
        throw new ValidationError("Invalid image maximum value.");
    }

    if (!Array.isArray(image.pixels) ||
        image.pixels.length !== image.height) {
        throw new ValidationError("Invalid image height.");
    }

    for (const row of image.pixels) {
        if (!Array.isArray(row) || row.length !== image.width) {
            throw new ValidationError("Invalid image width.");
        }

        for (const pixel of row) {
            if (!Array.isArray(pixel) || pixel.length !== 3) {
                throw new ValidationError("Every pixel must contain RGB.");
            }

            if (pixel.some(value =>
                !Number.isInteger(value) ||
                value < 0 ||
                value > image.maxValue
            )) {
                throw new ValidationError("Invalid pixel channel.");
            }
        }
    }
}

function createImage(width, height, pixels, maxValue = 255) {
    const image = { width, height, pixels, maxValue };
    validateImage(image);
    return image;
}

class PPMCodec {
    static async read(filePath) {
        const content = await fs.readFile(filePath, "utf8");

        const tokens = content
            .replace(/#[^\r\n]*/g, "")
            .trim()
            .split(/\s+/);

        if (tokens.length < 4 || tokens[0] !== "P3") {
            throw new PipelineError("Only P3 PPM images are supported.");
        }

        const width = Number(tokens[1]);
        const height = Number(tokens[2]);
        const maxValue = Number(tokens[3]);

        validateDimension(width, "Width");
        validateDimension(height, "Height");

        if (!Number.isInteger(maxValue) ||
            maxValue < 1 ||
            maxValue > 65535) {
            throw new ValidationError("Invalid PPM maximum value.");
        }

        const expected = width * height * 3;
        const values = tokens.slice(4).map(Number);

        if (values.length !== expected) {
            throw new PipelineError(
                `Expected ${expected} channel values; received ${values.length}.`
            );
        }

        if (values.some(value =>
            !Number.isInteger(value) ||
            value < 0 ||
            value > maxValue
        )) {
            throw new ValidationError("PPM channel outside permitted range.");
        }

        const pixels = [];

        for (let y = 0; y < height; y++) {
            const row = [];

            for (let x = 0; x < width; x++) {
                const index = (y * width + x) * 3;
                row.push(values.slice(index, index + 3));
            }

            pixels.push(row);
        }

        return createImage(width, height, pixels, maxValue);
    }

    static async write(image, filePath) {
        validateImage(image);

        const lines = [
            "P3",
            `${image.width} ${image.height}`,
            String(image.maxValue)
        ];

        for (const row of image.pixels) {
            lines.push(row.flat().join(" "));
        }

        await fs.writeFile(filePath, lines.join("\n") + "\n", {
            encoding: "ascii",
            mode: 0o600
        });
    }
}

class ImagePreprocessor {
    static resize(image, targetWidth, targetHeight) {
        validateImage(image);
        validateDimension(targetWidth, "Target width");
        validateDimension(targetHeight, "Target height");

        const pixels = [];

        for (let y = 0; y < targetHeight; y++) {
            const sourceY = Math.min(
                image.height - 1,
                Math.floor(y * image.height / targetHeight)
            );

            const row = [];

            for (let x = 0; x < targetWidth; x++) {
                const sourceX = Math.min(
                    image.width - 1,
                    Math.floor(x * image.width / targetWidth)
                );

                row.push([...image.pixels[sourceY][sourceX]]);
            }

            pixels.push(row);
        }

        return createImage(
            targetWidth,
            targetHeight,
            pixels,
            image.maxValue
        );
    }

    static normalize(image, mode = "zero_one") {
        validateImage(image);

        if (!["zero_one", "minus_one_one"].includes(mode)) {
            throw new ValidationError("Unsupported normalization mode.");
        }

        return image.pixels.map(row =>
            row.map(pixel =>
                pixel.map(channel => {
                    const value = channel / image.maxValue;

                    return mode === "zero_one"
                        ? value
                        : value * 2 - 1;
                })
            )
        );
    }

    static grayscale(image) {
        validateImage(image);

        return image.pixels.map(row =>
            row.map(pixel =>
                (
                    0.2126 * pixel[0] +
                    0.7152 * pixel[1] +
                    0.0722 * pixel[2]
                ) / image.maxValue
            )
        );
    }

    static standardize(values) {
        if (!Array.isArray(values) || values.length === 0) {
            throw new ValidationError("Cannot standardize empty data.");
        }

        if (!values.every(Number.isFinite)) {
            throw new ValidationError("Values must be finite.");
        }

        const mean = values.reduce((a, b) => a + b, 0) / values.length;

        const variance = values.reduce(
            (sum, value) => sum + (value - mean) ** 2,
            0
        ) / values.length;

        const deviation = Math.sqrt(variance);

        return values.map(value =>
            deviation === 0 ? 0 : (value - mean) / deviation
        );
    }
}

class ImageAugmentor {
    constructor(seed = 42) {
        this.seed = seed >>> 0;
        this.state = this.seed;
    }

    random() {
        // Deterministic linear congruential generator.
        this.state = (
            Math.imul(1664525, this.state) + 1013904223
        ) >>> 0;

        return this.state / 4294967296;
    }

    horizontalFlip(image) {
        validateImage(image);

        return createImage(
            image.width,
            image.height,
            image.pixels.map(row =>
                [...row].reverse().map(pixel => [...pixel])
            ),
            image.maxValue
        );
    }

    verticalFlip(image) {
        validateImage(image);

        return createImage(
            image.width,
            image.height,
            [...image.pixels].reverse().map(row =>
                row.map(pixel => [...pixel])
            ),
            image.maxValue
        );
    }

    rotate90(image) {
        validateImage(image);

        const pixels = [];

        for (let x = 0; x < image.width; x++) {
            const row = [];

            for (let y = image.height - 1; y >= 0; y--) {
                row.push([...image.pixels[y][x]]);
            }

            pixels.push(row);
        }

        return createImage(
            image.height,
            image.width,
            pixels,
            image.maxValue
        );
    }

    brightness(image, factor) {
        validateImage(image);

        if (!Number.isFinite(factor) || factor < 0 || factor > 3) {
            throw new ValidationError("Invalid brightness factor.");
        }

        return createImage(
            image.width,
            image.height,
            image.pixels.map(row =>
                row.map(pixel =>
                    pixel.map(channel =>
                        Math.max(
                            0,
                            Math.min(
                                image.maxValue,
                                Math.round(channel * factor)
                            )
                        )
                    )
                )
            ),
            image.maxValue
        );
    }

    randomTransform(image) {
        let result = image;

        if (this.random() > 0.5) {
            result = this.horizontalFlip(result);
        }

        if (this.random() > 0.5) {
            result = this.verticalFlip(result);
        }

        return this.brightness(
            result,
            0.8 + this.random() * 0.4
        );
    }
}

class FeatureExtractor {
    static extract(image) {
        validateImage(image);

        const total = image.width * image.height;
        const means = [0, 0, 0];
        const squares = [0, 0, 0];

        for (const row of image.pixels) {
            for (const pixel of row) {
                for (let channel = 0; channel < 3; channel++) {
                    const value = pixel[channel] / image.maxValue;

                    means[channel] += value;
                    squares[channel] += value * value;
                }
            }
        }

        for (let i = 0; i < 3; i++) {
            means[i] /= total;
            squares[i] = Math.sqrt(
                Math.max(0, squares[i] / total - means[i] ** 2)
            );
        }

        const sum = Math.max(means.reduce((a, b) => a + b, 0), 1e-12);

        return {
            redMean: means[0],
            greenMean: means[1],
            blueMean: means[2],
            redStd: squares[0],
            greenStd: squares[1],
            blueStd: squares[2],
            redRatio: means[0] / sum,
            greenRatio: means[1] / sum,
            blueRatio: means[2] / sum
        };
    }
}

class NearestCentroidClassifier {
    constructor() {
        this.centroids = new Map();
        this.featureNames = null;
    }

    fit(samples) {
        if (!Array.isArray(samples) || samples.length === 0) {
            throw new ValidationError("Training samples cannot be empty.");
        }

        const grouped = new Map();

        for (const sample of samples) {
            if (!sample || typeof sample.label !== "string" ||
                !sample.label.trim()) {
                throw new ValidationError("Invalid training label.");
            }

            const entries = Object.entries(sample.features).sort(
                ([a], [b]) => a.localeCompare(b)
            );

            if (entries.length === 0 ||
                entries.some(([, value]) => !Number.isFinite(value))) {
                throw new ValidationError("Invalid feature vector.");
            }

            const names = entries.map(([name]) => name);

            if (this.featureNames === null) {
                this.featureNames = names;
            } else if (
                JSON.stringify(names) !== JSON.stringify(this.featureNames)
            ) {
                throw new ValidationError("Inconsistent feature dimensions.");
            }

            if (!grouped.has(sample.label)) {
                grouped.set(sample.label, []);
            }

            grouped.get(sample.label).push(
                entries.map(([, value]) => value)
            );
        }

        this.centroids.clear();

        for (const [label, vectors] of grouped) {
            const centroid = this.featureNames.map((_, index) =>
                vectors.reduce(
                    (sum, vector) => sum + vector[index],
                    0
                ) / vectors.length
            );

            this.centroids.set(label, centroid);
        }

        return this;
    }

    predict(features) {
        if (this.centroids.size === 0) {
            throw new PipelineError("Classifier is not trained.");
        }

        const entries = Object.entries(features).sort(
            ([a], [b]) => a.localeCompare(b)
        );

        const names = entries.map(([name]) => name);
        const vector = entries.map(([, value]) => value);

        if (
            JSON.stringify(names) !== JSON.stringify(this.featureNames) ||
            !vector.every(Number.isFinite)
        ) {
            throw new ValidationError("Invalid prediction features.");
        }

        const distances = new Map();

        for (const [label, centroid] of this.centroids) {
            let squaredDistance = 0;

            for (let i = 0; i < vector.length; i++) {
                squaredDistance += (vector[i] - centroid[i]) ** 2;
            }

            distances.set(label, Math.sqrt(squaredDistance));
        }

        const minimum = Math.min(...distances.values());
        const weights = new Map();

        for (const [label, distance] of distances) {
            weights.set(label, Math.exp(-(distance - minimum)));
        }

        const denominator = [...weights.values()].reduce(
            (sum, value) => sum + value,
            0
        );

        const probabilities = Object.fromEntries(
            [...weights.entries()].map(([label, value]) => [
                label,
                value / denominator
            ])
        );

        const predictedClass = Object.entries(probabilities).sort(
            (a, b) => b[1] - a[1]
        )[0][0];

        return Object.freeze({
            predictedClass,
            confidence: probabilities[predictedClass],
            probabilities,
            modelFingerprint: this.fingerprint()
        });
    }

    fingerprint() {
        const data = JSON.stringify({
            names: this.featureNames,
            centroids: [...this.centroids.entries()]
        });

        return crypto.createHash("sha256").update(data).digest("hex");
    }

    async save(filePath) {
        if (this.centroids.size === 0) {
            throw new PipelineError("Cannot save an untrained model.");
        }

        const data = {
            featureNames: this.featureNames,
            centroids: Object.fromEntries(this.centroids)
        };

        await fs.writeFile(
            filePath,
            JSON.stringify(data, null, 2),
            { mode: 0o600 }
        );
    }

    async load(filePath) {
        const data = JSON.parse(
            await fs.readFile(filePath, "utf8")
        );

        if (!Array.isArray(data.featureNames) ||
            data.featureNames.length === 0 ||
            !data.centroids ||
            typeof data.centroids !== "object") {
            throw new ValidationError("Invalid model file.");
        }

        const centroids = new Map();

        for (const [label, vector] of Object.entries(data.centroids)) {
            if (!label ||
                !Array.isArray(vector) ||
                vector.length !== data.featureNames.length ||
                !vector.every(Number.isFinite)) {
                throw new ValidationError("Invalid centroid.");
            }

            centroids.set(label, vector);
        }

        if (centroids.size === 0) {
            throw new ValidationError("Model has no classes.");
        }

        this.featureNames = data.featureNames;
        this.centroids = centroids;
        return this;
    }
}

class ComputerVisionPipeline extends EventEmitter {
    constructor(config = {}) {
        super();

        this.config = Object.freeze({
            width: config.width ?? 32,
            height: config.height ?? 32,
            seed: config.seed ?? 42,
            normalization: config.normalization ?? "zero_one"
        });

        validateDimension(this.config.width, "Width");
        validateDimension(this.config.height, "Height");

        if (!["zero_one", "minus_one_one"].includes(
            this.config.normalization
        )) {
            throw new ValidationError("Invalid normalization configuration.");
        }

        this.augmentor = new ImageAugmentor(this.config.seed);
        this.classifier = new NearestCentroidClassifier();
    }

    prepare(image) {
        const resized = ImagePreprocessor.resize(
            image,
            this.config.width,
            this.config.height
        );

        return resized;
    }

    async train(dataset, options = {}) {
        if (!Array.isArray(dataset) || dataset.length === 0) {
            throw new ValidationError("Training dataset is empty.");
        }

        const samples = [];

        for (let i = 0; i < dataset.length; i++) {
            const item = dataset[i];

            try {
                const processed = this.prepare(item.image);

                samples.push({
                    features: FeatureExtractor.extract(processed),
                    label: item.label
                });

                if (options.augment !== false) {
                    const augmented = this.augmentor.randomTransform(processed);

                    samples.push({
                        features: FeatureExtractor.extract(augmented),
                        label: item.label
                    });
                }

                this.emit("sample", {
                    index: i,
                    total: dataset.length
                });
            } catch (error) {
                this.emit("sampleError", {
                    index: i,
                    message: error.message
                });

                throw error;
            }
        }

        this.classifier.fit(samples);

        const event = {
            sampleCount: samples.length,
            classCount: this.classifier.centroids.size
        };

        this.emit("trained", event);
        return event;
    }

    predict(image) {
        const result = this.classifier.predict(
            FeatureExtractor.extract(this.prepare(image))
        );

        this.emit("prediction", result);
        return result;
    }

    async predictBatch(images, concurrency = 4) {
        if (!Array.isArray(images) ||
            !Number.isInteger(concurrency) ||
            concurrency < 1 ||
            concurrency > 32) {
            throw new ValidationError("Invalid batch configuration.");
        }

        const results = new Array(images.length);
        let nextIndex = 0;

        // Bounded workers prevent uncontrolled resource consumption.
        const workers = Array.from(
            { length: Math.min(concurrency, images.length) },
            async () => {
                while (true) {
                    const index = nextIndex++;

                    if (index >= images.length) {
                        return;
                    }

                    results[index] = this.predict(images[index]);
                }
            }
        );

        await Promise.all(workers);
        return results;
    }
}

function syntheticImage(color, width = 16, height = 16) {
    const pixels = [];

    for (let y = 0; y < height; y++) {
        const row = [];

        for (let x = 0; x < width; x++) {
            const variation = (x + y) % 12;

            row.push(
                color.map(channel => Math.min(255, channel + variation))
            );
        }

        pixels.push(row);
    }

    return createImage(width, height, pixels);
}

async function runDemo() {
    const pipeline = new ComputerVisionPipeline({
        width: 16,
        height: 16,
        seed: 42
    });

    pipeline.on("trained", event => {
        console.log("Training completed:", event);
    });

    pipeline.on("prediction", result => {
        console.log(
            `Classification event: ${result.predictedClass}`
        );
    });

    const dataset = [
        { image: syntheticImage([230, 20, 20]), label: "red" },
        { image: syntheticImage([210, 30, 25]), label: "red" },
        { image: syntheticImage([20, 220, 20]), label: "green" },
        { image: syntheticImage([25, 200, 30]), label: "green" },
        { image: syntheticImage([20, 20, 230]), label: "blue" },
        { image: syntheticImage([25, 30, 210]), label: "blue" }
    ];

    await pipeline.train(dataset);

    const testImages = [
        syntheticImage([225, 25, 25]),
        syntheticImage([25, 215, 25]),
        syntheticImage([25, 25, 220])
    ];

    const results = await pipeline.predictBatch(testImages, 2);

    results.forEach((result, index) => {
        console.log(`Test ${index + 1}:`, result);

        const expected = ["red", "green", "blue"][index];

        if (result.predictedClass !== expected) {
            throw new PipelineError("Classification test failed.");
        }
    });

    await pipeline.classifier.save("vision-model.json");

    const restored = new NearestCentroidClassifier();
    await restored.load("vision-model.json");

    console.log(
        "Model integrity:",
        restored.fingerprint() === pipeline.classifier.fingerprint()
            ? "PASSED"
            : "FAILED"
    );

    const sample = syntheticImage([200, 80, 40]);

    const resized = pipeline.prepare(sample);
    const rotated = pipeline.augmentor.rotate90(resized);
    const flipped = pipeline.augmentor.horizontalFlip(resized);

    if (rotated.width !== resized.height ||
        rotated.height !== resized.width ||
        flipped.width !== resized.width) {
        throw new PipelineError("Transformation test failed.");
    }

    console.log("Preprocessing tests: PASSED");
    console.log("Augmentation tests: PASSED");
}

async function main() {
    const args = process.argv.slice(2);

    try {
        if (args.length > 0 && args[0] === "--predict") {
            if (args.length < 3) {
                throw new ValidationError(
                    "Usage: node vision.js --predict image.ppm model.json"
                );
            }

            const image = await PPMCodec.read(args[1]);
            const classifier = new NearestCentroidClassifier();

            await classifier.load(args[2]);

            const pipeline = new ComputerVisionPipeline();
            pipeline.classifier = classifier;

            console.log(
                JSON.stringify(pipeline.predict(image), null, 2)
            );
        } else {
            await runDemo();
        }
    } catch (error) {
        console.error(
            `[${error.code || "ERROR"}] ${error.message}`
        );

        process.exitCode = 1;
    }
}

if (require.main === module) {
    main();
}

module.exports = {
    PipelineError,
    ValidationError,
    PPMCodec,
    ImagePreprocessor,
    ImageAugmentor,
    FeatureExtractor,
    NearestCentroidClassifier,
    ComputerVisionPipeline
};
