"use strict";

/*
 * CNN Fundamentals: Convolution, Filters, Kernels, Stride and Padding.
 *
 * Node.js implementation using native JavaScript capabilities.
 *
 * Run:
 * node cnn_fundamentals.js
 */

const Padding = Object.freeze({
    VALID: "valid",
    SAME: "same"
});

class CNNError extends Error {}

class ShapeError extends CNNError {}

class ConfigurationError extends CNNError {}


function validateMatrix(matrix) {
    if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new ShapeError("Matrix must contain rows.");
    }

    if (!Array.isArray(matrix[0]) || matrix[0].length === 0) {
        throw new ShapeError("Matrix rows cannot be empty.");
    }

    const width = matrix[0].length;

    for (const row of matrix) {
        if (!Array.isArray(row) || row.length !== width) {
            throw new ShapeError("Matrix must be rectangular.");
        }

        for (const value of row) {
            if (typeof value !== "number" || !Number.isFinite(value)) {
                throw new TypeError("Matrix values must be finite numbers.");
            }
        }
    }

    return {
        height: matrix.length,
        width
    };
}


function validateTensor(tensor) {
    if (!Array.isArray(tensor) || tensor.length === 0) {
        throw new ShapeError("Tensor requires at least one channel.");
    }

    const dimensions = tensor.map(validateMatrix);

    const reference = dimensions[0];

    for (const dimension of dimensions) {
        if (
            dimension.height !== reference.height ||
            dimension.width !== reference.width
        ) {
            throw new ShapeError("Tensor channels must have equal dimensions.");
        }
    }

    return {
        channels: tensor.length,
        height: reference.height,
        width: reference.width
    };
}


function validateStride(stride) {
    if (!Number.isInteger(stride) || stride < 1) {
        throw new ConfigurationError("Stride must be a positive integer.");
    }
}


function calculateOutputDimension(
    inputSize,
    kernelSize,
    stride,
    padding
) {
    if (padding === Padding.SAME) {
        return Math.ceil(inputSize / stride);
    }

    if (padding !== Padding.VALID) {
        throw new ConfigurationError("Unsupported padding mode.");
    }

    if (kernelSize > inputSize) {
        throw new ShapeError("Kernel exceeds input dimensions.");
    }

    return Math.floor((inputSize - kernelSize) / stride) + 1;
}


function calculatePadding(
    height,
    width,
    kernelHeight,
    kernelWidth,
    stride,
    padding
) {
    if (padding === Padding.VALID) {
        return { top: 0, bottom: 0, left: 0, right: 0 };
    }

    const outputHeight = Math.ceil(height / stride);
    const outputWidth = Math.ceil(width / stride);

    const totalHeight = Math.max(
        (outputHeight - 1) * stride + kernelHeight - height,
        0
    );

    const totalWidth = Math.max(
        (outputWidth - 1) * stride + kernelWidth - width,
        0
    );

    const top = Math.floor(totalHeight / 2);
    const left = Math.floor(totalWidth / 2);

    return {
        top,
        bottom: totalHeight - top,
        left,
        right: totalWidth - left
    };
}


function padTensor(tensor, padding) {
    const { top, bottom, left, right } = padding;

    return tensor.map(channel => {
        const originalWidth = channel[0].length;
        const paddedWidth = originalWidth + left + right;

        const result = [];

        for (let i = 0; i < top; i++) {
            result.push(new Array(paddedWidth).fill(0));
        }

        for (const row of channel) {
            result.push([
                ...new Array(left).fill(0),
                ...row,
                ...new Array(right).fill(0)
            ]);
        }

        for (let i = 0; i < bottom; i++) {
            result.push(new Array(paddedWidth).fill(0));
        }

        return result;
    });
}


class Filter {
    constructor(weights, bias = 0) {
        const dimensions = validateTensor(weights);

        if (!Number.isFinite(bias)) {
            throw new ConfigurationError("Filter bias must be finite.");
        }

        this.weights = weights.map(
            channel => channel.map(row => [...row])
        );

        this.bias = bias;

        this.inputChannels = dimensions.channels;
        this.kernelHeight = dimensions.height;
        this.kernelWidth = dimensions.width;
    }
}


class FilterBank {
    constructor(filters) {
        if (!Array.isArray(filters) || filters.length === 0) {
            throw new ConfigurationError("Filter bank cannot be empty.");
        }

        const reference = filters[0];

        if (!(reference instanceof Filter)) {
            throw new TypeError("All entries must be Filter instances.");
        }

        for (const filter of filters) {
            if (!(filter instanceof Filter)) {
                throw new TypeError("Invalid filter object.");
            }

            if (
                filter.inputChannels !== reference.inputChannels ||
                filter.kernelHeight !== reference.kernelHeight ||
                filter.kernelWidth !== reference.kernelWidth
            ) {
                throw new ShapeError("Filter dimensions must match.");
            }
        }

        this.filters = [...filters];

        this.inputChannels = reference.inputChannels;
        this.kernelHeight = reference.kernelHeight;
        this.kernelWidth = reference.kernelWidth;
    }

    get parameterCount() {
        return this.filters.length * (
            this.inputChannels *
            this.kernelHeight *
            this.kernelWidth + 1
        );
    }

    apply(input, options = {}) {
        const {
            stride = 1,
            padding = Padding.VALID
        } = options;

        validateStride(stride);

        const dimensions = validateTensor(input);

        if (dimensions.channels !== this.inputChannels) {
            throw new ShapeError("Input channel count does not match filters.");
        }

        const outputHeight = calculateOutputDimension(
            dimensions.height,
            this.kernelHeight,
            stride,
            padding
        );

        const outputWidth = calculateOutputDimension(
            dimensions.width,
            this.kernelWidth,
            stride,
            padding
        );

        const paddingApplied = calculatePadding(
            dimensions.height,
            dimensions.width,
            this.kernelHeight,
            this.kernelWidth,
            stride,
            padding
        );

        const padded = padTensor(input, paddingApplied);

        // Filters are independent and can be evaluated concurrently.
        const featureMaps = this.filters.map(filter => {
            const output = [];

            for (let row = 0; row < outputHeight; row++) {
                const outputRow = [];

                for (let col = 0; col < outputWidth; col++) {
                    let sum = filter.bias;

                    for (let channel = 0; channel < this.inputChannels; channel++) {
                        const matrix = padded[channel];
                        const kernel = filter.weights[channel];

                        for (let kr = 0; kr < this.kernelHeight; kr++) {
                            for (let kc = 0; kc < this.kernelWidth; kc++) {
                                const inputValue = matrix[
                                    row * stride + kr
                                ][
                                    col * stride + kc
                                ];

                                sum += inputValue * kernel[kr][kc];
                            }
                        }
                    }

                    outputRow.push(sum);
                }

                output.push(outputRow);
            }

            return output;
        });

        return {
            values: featureMaps,
            inputShape: [
                dimensions.channels,
                dimensions.height,
                dimensions.width
            ],
            outputShape: [
                featureMaps.length,
                outputHeight,
                outputWidth
            ],
            paddingApplied,
            stride,
            padding
        };
    }

    async applyAsync(input, options = {}) {
        const tasks = this.filters.map(filter => {
            return Promise.resolve().then(() => {
                return new FilterBank([filter]).apply(input, options);
            });
        });

        const results = await Promise.all(tasks);

        return {
            values: results.map(result => result.values[0]),
            outputShape: [
                this.filters.length,
                ...results[0].outputShape.slice(1)
            ]
        };
    }
}


function correlate2d(matrix, kernel, stride = 1, padding = Padding.VALID) {
    const filter = new Filter([kernel]);

    return new FilterBank([filter]).apply(
        [matrix],
        { stride, padding }
    ).values[0];
}


function mapTensor(tensor, operation) {
    return tensor.map(channel =>
        channel.map(row => row.map(operation))
    );
}


function relu(tensor) {
    return mapTensor(tensor, value => Math.max(0, value));
}


function sigmoid(tensor) {
    return mapTensor(tensor, value => {
        if (value >= 0) {
            return 1 / (1 + Math.exp(-value));
        }

        const exponential = Math.exp(value);
        return exponential / (1 + exponential);
    });
}


function maxPool2D(tensor, poolSize = 2, stride = 2) {
    const dimensions = validateTensor(tensor);

    validateStride(stride);

    if (!Number.isInteger(poolSize) || poolSize < 1) {
        throw new ConfigurationError("Pool size must be positive.");
    }

    if (
        poolSize > dimensions.height ||
        poolSize > dimensions.width
    ) {
        throw new ShapeError("Pooling window exceeds input.");
    }

    const outputHeight = Math.floor(
        (dimensions.height - poolSize) / stride
    ) + 1;

    const outputWidth = Math.floor(
        (dimensions.width - poolSize) / stride
    ) + 1;

    return tensor.map(channel => {
        const output = [];

        for (let row = 0; row < outputHeight; row++) {
            const outputRow = [];

            for (let col = 0; col < outputWidth; col++) {
                let maximum = -Infinity;

                for (let r = 0; r < poolSize; r++) {
                    for (let c = 0; c < poolSize; c++) {
                        maximum = Math.max(
                            maximum,
                            channel[
                                row * stride + r
                            ][
                                col * stride + c
                            ]
                        );
                    }
                }

                outputRow.push(maximum);
            }

            output.push(outputRow);
        }

        return output;
    });
}


class CNNPipeline extends EventTarget {
    constructor(filterBank) {
        super();

        this.filterBank = filterBank;
        this.state = "idle";
        this.executionCount = 0;
    }

    async execute(input, options = {}) {
        if (this.state === "running") {
            throw new CNNError("Pipeline is already running.");
        }

        this.state = "running";

        this.dispatchEvent(new CustomEvent("started", {
            detail: { execution: this.executionCount + 1 }
        }));

        try {
            const convolution = await this.filterBank.applyAsync(
                input,
                options
            );

            const activated = relu(convolution.values);

            const pooled = maxPool2D(activated, 2, 2);

            this.executionCount++;
            this.state = "completed";

            const result = {
                convolution,
                activated,
                pooled,
                executionCount: this.executionCount
            };

            this.dispatchEvent(new CustomEvent("completed", {
                detail: {
                    executionCount: this.executionCount,
                    outputShape: convolution.outputShape
                }
            }));

            return result;

        } catch (error) {
            this.state = "failed";

            this.dispatchEvent(new CustomEvent("failed", {
                detail: { message: error.message }
            }));

            throw error;
        }
    }

    reset() {
        if (this.state === "running") {
            throw new CNNError("Cannot reset a running pipeline.");
        }

        this.state = "idle";
    }
}


function printMatrix(name, matrix) {
    console.log(`\n${name}`);

    for (const row of matrix) {
        console.log(
            row.map(value => value.toFixed(2).padStart(8)).join(" ")
        );
    }
}


function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}


async function main() {
    console.log("CNN FUNDAMENTALS");
    console.log("================");

    const image = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ];

    const edgeKernel = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ];

    printMatrix(
        "VALID convolution",
        correlate2d(image, edgeKernel)
    );

    for (const padding of [Padding.VALID, Padding.SAME]) {
        for (const stride of [1, 2]) {
            const output = correlate2d(
                image,
                edgeKernel,
                stride,
                padding
            );

            console.log(
                `\nPadding=${padding}, stride=${stride}, ` +
                `shape=${output.length}x${output[0].length}`
            );

            printMatrix("Feature map", output);
        }
    }

    const input = [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
        ]
    ];

    const filters = new FilterBank([
        new Filter([
            [[1, 0], [0, -1]],
            [[0, 1], [-1, 0]]
        ], 0.5),

        new Filter([
            [[-1, 0], [0, 1]],
            [[1, 1], [0, 0]]
        ], -1)
    ]);

    const result = filters.apply(input, {
        stride: 1,
        padding: Padding.SAME
    });

    console.log("\nMULTI-FILTER CONVOLUTION");
    console.log("Input shape:", result.inputShape);
    console.log("Output shape:", result.outputShape);
    console.log("Parameters:", filters.parameterCount);

    const pipeline = new CNNPipeline(filters);

    pipeline.addEventListener("started", event => {
        console.log("Pipeline started:", event.detail.execution);
    });

    pipeline.addEventListener("completed", event => {
        console.log("Pipeline completed:", event.detail.outputShape);
    });

    pipeline.addEventListener("failed", event => {
        console.error("Pipeline failed:", event.detail.message);
    });

    const pipelineInput = [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [2, 1, 0, 1],
            [1, 2, 1, 0],
            [0, 1, 2, 1],
            [1, 0, 1, 2]
        ]
    ];

    const pipelineResult = await pipeline.execute(
        pipelineInput,
        { stride: 1, padding: Padding.SAME }
    );

    console.log("Pipeline state:", pipeline.state);

    printMatrix(
        "First activated feature map",
        pipelineResult.activated[0]
    );

    printMatrix(
        "First pooled feature map",
        pipelineResult.pooled[0]
    );

    const expected = correlate2d(
        [[1, 2], [3, 4]],
        [[1, 0], [0, 1]]
    );

    assert(expected[0][0] === 5, "Cross-correlation result");

    const same = correlate2d(
        [[1, 2], [3, 4]],
        [[1]],
        2,
        Padding.SAME
    );

    assert(same.length === 1, "SAME output height");

    try {
        correlate2d([[1, 2], [3]], [[1]]);
        throw new Error("Expected rectangular matrix validation.");
    } catch (error) {
        assert(error instanceof ShapeError, "Ragged matrix rejected");
    }

    try {
        correlate2d([[1]], [[1, 2], [3, 4]]);
        throw new Error("Expected oversized kernel validation.");
    } catch (error) {
        assert(error instanceof ShapeError, "Oversized kernel rejected");
    }

    try {
        correlate2d([[1]], [[1]], 0);
        throw new Error("Expected invalid stride validation.");
    } catch (error) {
        assert(
            error instanceof ConfigurationError,
            "Zero stride rejected"
        );
    }

    console.log("\nAll CNN validation tests passed.");
}

main().catch(error => {
    console.error("Execution error:", error.message);
    process.exitCode = 1;
});
