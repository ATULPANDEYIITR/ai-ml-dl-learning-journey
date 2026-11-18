"use strict";

/*
 * PyTorch Fundamentals: JavaScript Tensor Execution Model
 *
 * A dependency-free CPU tensor engine demonstrating:
 * - Multidimensional tensor storage
 * - Shape and broadcasting validation
 * - Element-wise and matrix operations
 * - Automatic differentiation
 * - Reverse-mode gradient propagation
 * - Device metadata
 * - Event-driven execution
 * - Asynchronous experiment scheduling
 *
 * Runtime: Node.js 18+
 *
 * Run:
 *   node pytorch_fundamentals.js
 *
 * This is an educational CPU implementation, not a replacement
 * for PyTorch or a GPU execution backend.
 */

const { EventEmitter } = require("node:events");
const { performance } = require("node:perf_hooks");
const assert = require("node:assert/strict");

// ------------------------------------------------------------
// Domain errors
// ------------------------------------------------------------

class TensorError extends Error {
    constructor(message) {
        super(message);
        this.name = "TensorError";
    }
}

class ShapeError extends TensorError {
    constructor(message) {
        super(message);
        this.name = "ShapeError";
    }
}

class GradientError extends TensorError {
    constructor(message) {
        super(message);
        this.name = "GradientError";
    }
}

// ------------------------------------------------------------
// Shape utilities
// ------------------------------------------------------------

function validateShape(shape) {
    if (!Array.isArray(shape) || shape.length === 0) {
        throw new ShapeError("Shape must be a nonempty array.");
    }

    if (!shape.every(
        value => Number.isSafeInteger(value) && value > 0
    )) {
        throw new ShapeError(
            "Dimensions must be positive safe integers."
        );
    }

    const size = shape.reduce((a, b) => a * b, 1);

    if (!Number.isSafeInteger(size)) {
        throw new ShapeError("Tensor is too large.");
    }

    return size;
}

function stridesFor(shape) {
    const strides = new Array(shape.length);
    let stride = 1;

    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

function broadcastShape(a, b) {
    const length = Math.max(a.length, b.length);
    const result = [];

    for (let i = 0; i < length; i++) {
        const left = a[a.length - 1 - i] ?? 1;
        const right = b[b.length - 1 - i] ?? 1;

        if (left !== right && left !== 1 && right !== 1) {
            throw new ShapeError(
                `Cannot broadcast ${a} with ${b}.`
            );
        }

        result.unshift(Math.max(left, right));
    }

    return result;
}

function flatIndex(index, shape, strides) {
    if (index.length !== shape.length) {
        throw new ShapeError("Index dimensionality mismatch.");
    }

    return index.reduce((total, coordinate, axis) => {
        if (
            !Number.isInteger(coordinate) ||
            coordinate < 0 ||
            coordinate >= shape[axis]
        ) {
            throw new RangeError("Tensor index out of bounds.");
        }

        return total + coordinate * strides[axis];
    }, 0);
}

// ------------------------------------------------------------
// Tensor class
// ------------------------------------------------------------

class Tensor {
    constructor(data, shape, options = {}) {
        const expected = validateShape(shape);

        if (
            !Array.isArray(data) &&
            !ArrayBuffer.isView(data)
        ) {
            throw new TensorError("Data must be an array.");
        }

        if (data.length !== expected) {
            throw new ShapeError(
                `Expected ${expected} values, got ${data.length}.`
            );
        }

        const values = Array.from(data, Number);

        if (!values.every(Number.isFinite)) {
            throw new TensorError(
                "Tensor values must be finite."
            );
        }

        this.data = Float64Array.from(values);
        this.shape = Object.freeze([...shape]);
        this.strides = Object.freeze(stridesFor(shape));
        this.device = options.device ?? "cpu";
        this.dtype = options.dtype ?? "float64";
        this.requiresGrad = Boolean(options.requiresGrad);
        this.grad = null;

        this.parents = [];
        this.backwardFn = () => {};
        this.operation = "leaf";
    }

    static from(data, shape, options = {}) {
        return new Tensor(data, shape, options);
    }

    static zeros(shape, options = {}) {
        return new Tensor(
            new Float64Array(validateShape(shape)),
            shape,
            options
        );
    }

    static ones(shape, options = {}) {
        return new Tensor(
            new Float64Array(validateShape(shape)).fill(1),
            shape,
            options
        );
    }

    static random(shape, options = {}) {
        const size = validateShape(shape);
        return new Tensor(
            Array.from({ length: size }, () => Math.random()),
            shape,
            options
        );
    }

    clone() {
        return new Tensor(this.data, this.shape, {
            device: this.device,
            dtype: this.dtype,
            requiresGrad: this.requiresGrad
        });
    }

    item() {
        if (this.data.length !== 1) {
            throw new TensorError(
                "item() requires a single-element tensor."
            );
        }

        return this.data[0];
    }

    get(index) {
        return this.data[
            flatIndex(index, this.shape, this.strides)
        ];
    }

    toArray() {
        return Array.from(this.data);
    }

    ensureSameDevice(other) {
        if (this.device !== other.device) {
            throw new TensorError(
                "Operands must reside on the same device."
            );
        }
    }

    elementwise(other, operation, gradientLeft, gradientRight) {
        const right = other instanceof Tensor
            ? other
            : Tensor.from([other], [1], {
                device: this.device
            });

        this.ensureSameDevice(right);

        const outputShape = broadcastShape(
            this.shape,
            right.shape
        );

        const size = validateShape(outputShape);
        const output = new Float64Array(size);

        const outputStrides = stridesFor(outputShape);

        for (let i = 0; i < size; i++) {
            const coordinates = outputShape.map(
                (_, axis) => Math.floor(
                    i / outputStrides[axis]
                ) % outputShape[axis]
            );

            const leftCoordinates = coordinates.slice(
                outputShape.length - this.shape.length
            ).map((value, axis) =>
                this.shape[axis] === 1 ? 0 : value
            );

            const rightCoordinates = coordinates.slice(
                outputShape.length - right.shape.length
            ).map((value, axis) =>
                right.shape[axis] === 1 ? 0 : value
            );

            const leftValue = this.get(leftCoordinates);
            const rightValue = right.get(rightCoordinates);

            output[i] = operation(leftValue, rightValue);
        }

        const result = new Tensor(output, outputShape, {
            device: this.device,
            requiresGrad:
                this.requiresGrad || right.requiresGrad
        });

        result.parents = [this, right];
        result.operation = "elementwise";

        result.backwardFn = () => {
            if (this.requiresGrad) {
                this.accumulateBroadcastGradient(
                    result.grad,
                    outputShape,
                    gradientLeft,
                    right.data,
                    right.shape
                );
            }

            if (right.requiresGrad) {
                right.accumulateBroadcastGradient(
                    result.grad,
                    outputShape,
                    gradientRight,
                    this.data,
                    this.shape
                );
            }
        };

        return result;
    }

    accumulateBroadcastGradient(
        outputGradient,
        outputShape,
        derivative,
        otherData,
        otherShape
    ) {
        const outputStrides = stridesFor(outputShape);
        const inputStrides = this.strides;

        if (this.grad === null) {
            this.grad = new Float64Array(this.data.length);
        }

        for (let i = 0; i < outputGradient.length; i++) {
            const coordinates = outputShape.map(
                (_, axis) =>
                    Math.floor(i / outputStrides[axis]) %
                    outputShape[axis]
            );

            const inputCoordinates = coordinates.slice(
                outputShape.length - this.shape.length
            ).map((value, axis) =>
                this.shape[axis] === 1 ? 0 : value
            );

            const otherCoordinates = coordinates.slice(
                outputShape.length - otherShape.length
            ).map((value, axis) =>
                otherShape[axis] === 1 ? 0 : value
            );

            const target = flatIndex(
                inputCoordinates,
                this.shape,
                inputStrides
            );

            const otherIndex = flatIndex(
                otherCoordinates,
                otherShape,
                stridesFor(otherShape)
            );

            this.grad[target] +=
                outputGradient[i] *
                derivative(
                    this.data[target],
                    otherData[otherIndex]
                );
        }
    }

    add(other) {
        return this.elementwise(
            other,
            (a, b) => a + b,
            () => 1,
            () => 1
        );
    }

    multiply(other) {
        return this.elementwise(
            other,
            (a, b) => a * b,
            (a, b) => b,
            (a, b) => a
        );
    }

    square() {
        return this.multiply(this);
    }

    sum() {
        const value = this.data.reduce(
            (total, current) => total + current,
            0
        );

        const result = new Tensor([value], [1], {
            device: this.device,
            requiresGrad: this.requiresGrad
        });

        result.parents = [this];
        result.operation = "sum";

        result.backwardFn = () => {
            if (this.requiresGrad) {
                if (this.grad === null) {
                    this.grad = new Float64Array(this.data.length);
                }

                for (let i = 0; i < this.grad.length; i++) {
                    this.grad[i] += result.grad[0];
                }
            }
        };

        return result;
    }

    matmul(other) {
        if (!(other instanceof Tensor)) {
            throw new TensorError("matmul requires a Tensor.");
        }

        this.ensureSameDevice(other);

        if (
            this.shape.length !== 2 ||
            other.shape.length !== 2
        ) {
            throw new ShapeError(
                "matmul requires two-dimensional tensors."
            );
        }

        const [m, k] = this.shape;
        const [k2, n] = other.shape;

        if (k !== k2) {
            throw new ShapeError(
                "Matrix multiplication dimensions do not match."
            );
        }

        const output = new Float64Array(m * n);

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                for (let p = 0; p < k; p++) {
                    output[i * n + j] +=
                        this.data[i * k + p] *
                        other.data[p * n + j];
                }
            }
        }

        return new Tensor(output, [m, n], {
            device: this.device,
            requiresGrad:
                this.requiresGrad || other.requiresGrad
        });
    }

    backward(seed = 1) {
        if (this.data.length !== 1) {
            throw new GradientError(
                "backward() requires a scalar output."
            );
        }

        if (!this.requiresGrad) {
            throw new GradientError(
                "Output does not require gradients."
            );
        }

        const topology = [];
        const visited = new Set();

        const visit = node => {
            if (visited.has(node)) return;

            visited.add(node);

            for (const parent of node.parents) {
                visit(parent);
            }

            topology.push(node);
        };

        visit(this);

        for (const node of topology) {
            if (node.requiresGrad) {
                node.grad = new Float64Array(node.data.length);
            }
        }

        this.grad[0] = seed;

        // Reverse topological order ensures each node receives
        // gradients from all downstream operations first.
        for (let i = topology.length - 1; i >= 0; i--) {
            const node = topology[i];

            if (node.requiresGrad) {
                node.backwardFn();
            }
        }
    }

    zeroGrad() {
        this.grad = null;
    }
}

// ------------------------------------------------------------
// Event-driven execution
// ------------------------------------------------------------

class ExperimentRunner extends EventEmitter {
    constructor() {
        super();
        this.results = new Map();
        this.running = false;
    }

    async execute(name, operation) {
        if (this.running) {
            throw new Error("Concurrent execution is disabled.");
        }

        if (this.results.has(name)) {
            throw new Error(`Experiment ${name} already exists.`);
        }

        this.running = true;
        this.emit("started", { name });

        const start = performance.now();

        try {
            await new Promise(resolve => setImmediate(resolve));

            const result = await operation();

            const duration = performance.now() - start;

            const record = Object.freeze({
                name,
                result,
                durationMs: duration,
                status: "completed"
            });

            this.results.set(name, record);
            this.emit("completed", record);

            return record;
        } catch (error) {
            this.emit("failed", {
                name,
                message: error.message
            });

            throw error;
        } finally {
            this.running = false;
        }
    }
}

// ------------------------------------------------------------
// Demonstrations
// ------------------------------------------------------------

function demonstrateTensors() {
    console.log("\n1. TENSOR FUNDAMENTALS");

    const matrix = Tensor.from(
        [1, 2, 3, 4],
        [2, 2]
    );

    const bias = Tensor.from([10, 20], [1, 2]);

    console.log("Matrix:", matrix.toArray());
    console.log("Shape:", matrix.shape);
    console.log("Device:", matrix.device);
    console.log("Broadcast result:",
        matrix.add(bias).toArray()
    );

    console.log("Ones:", Tensor.ones([2, 3]).toArray());
    console.log("Random:", Tensor.random([2, 2]).toArray());

    assert.deepEqual(
        matrix.add(bias).toArray(),
        [11, 22, 13, 24]
    );

    const product = matrix.matmul(
        Tensor.from([5, 6, 7, 8], [2, 2])
    );

    console.log("Matrix product:", product.toArray());

    assert.deepEqual(product.toArray(), [19, 22, 43, 50]);
}

function demonstrateAutograd() {
    console.log("\n2. AUTOMATIC DIFFERENTIATION");

    const x = Tensor.from([3], [1], {
        requiresGrad: true
    });

    const y = x.square().add(x.multiply(2)).add(1);

    console.log("Output:", y.item());

    y.backward();

    console.log("Gradient:", x.grad[0]);

    // dy/dx = 2x + 2 = 8.
    assert.equal(x.grad[0], 8);

    // Gradient accumulation is explicit in this engine.
    const a = Tensor.from([2, 3], [2], {
        requiresGrad: true
    });

    const output = a.square().sum();
    output.backward();

    console.log("Vector gradients:", Array.from(a.grad));

    assert.deepEqual(Array.from(a.grad), [4, 6]);
}

function demonstrateValidation() {
    console.log("\n3. VALIDATION");

    const tests = [
        () => Tensor.from([1, 2], [3]),
        () => Tensor.from([NaN], [1]),
        () => Tensor.from([1], [0]),
        () => Tensor.from([1, 2], [2]).matmul(
            Tensor.from([1, 2], [2])
        )
    ];

    for (const test of tests) {
        try {
            test();
            throw new Error("Expected failure did not occur.");
        } catch (error) {
            if (error.message === "Expected failure did not occur.") {
                throw error;
            }

            console.log(
                `Expected failure: ${error.name}: ${error.message}`
            );
        }
    }
}

async function demonstrateExecution() {
    console.log("\n4. ASYNCHRONOUS EXPERIMENT EXECUTION");

    const runner = new ExperimentRunner();

    runner.on("started", event => {
        console.log("Started:", event.name);
    });

    runner.on("completed", event => {
        console.log(
            `Completed: ${event.name}, ${event.durationMs.toFixed(3)} ms`
        );
    });

    runner.on("failed", event => {
        console.log("Failed:", event.message);
    });

    const experiment = await runner.execute(
        "matrix-multiplication",
        async () => {
            const a = Tensor.random([40, 40]);
            const b = Tensor.random([40, 40]);

            return a.matmul(b).shape;
        }
    );

    console.log("Experiment result:", experiment.result);
    console.log("Stored experiments:", runner.results.size);
}

async function main() {
    console.log("PYTORCH FUNDAMENTALS");
    console.log("Dependency-free JavaScript tensor model");

    demonstrateTensors();
    demonstrateAutograd();
    demonstrateValidation();

    await demonstrateExecution();

    console.log("\nAll JavaScript demonstrations completed.");
}

main().catch(error => {
    console.error("Execution failed:", error);
    process.exitCode = 1;
});
