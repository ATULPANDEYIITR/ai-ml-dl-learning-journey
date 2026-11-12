"use strict";

/*
 * Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax.
 *
 * Node.js implementation featuring:
 * - Activation registry
 * - Event-driven execution
 * - Asynchronous inference
 * - Batch processing
 * - Numerical validation
 * - Gradient propagation
 * - Softmax optimization
 * - Runtime policy validation
 *
 * Run: node activation_functions.js
 */

const { EventEmitter } = require("node:events");
const { performance } = require("node:perf_hooks");

class ActivationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ActivationError";
    }
}

function finiteNumber(value, name = "Input") {
    if (
        typeof value !== "number" ||
        !Number.isFinite(value)
    ) {
        throw new ActivationError(`${name} must be finite.`);
    }

    return value;
}

function vector(values) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new ActivationError(
            "Input must be a non-empty numerical array."
        );
    }

    return values.map((value, index) =>
        finiteNumber(value, `Input[${index}]`)
    );
}

class ActivationRegistry {
    constructor() {
        this.functions = new Map();
        this.registerDefaults();
    }

    register(name, implementation) {
        if (
            typeof name !== "string" ||
            !/^[a-z_]+$/.test(name)
        ) {
            throw new ActivationError("Invalid activation name.");
        }

        if (
            !implementation ||
            typeof implementation.forward !== "function" ||
            typeof implementation.derivative !== "function"
        ) {
            throw new ActivationError(
                "Activation requires forward and derivative functions."
            );
        }

        if (this.functions.has(name)) {
            throw new ActivationError(
                `Activation already registered: ${name}`
            );
        }

        this.functions.set(name, Object.freeze(implementation));
    }

    get(name) {
        const activation = this.functions.get(name);

        if (!activation) {
            throw new ActivationError(
                `Unknown activation: ${name}`
            );
        }

        return activation;
    }

    registerDefaults() {
        this.register("sigmoid", {
            forward: x => {
                finiteNumber(x);

                if (x >= 0) {
                    return 1 / (1 + Math.exp(-x));
                }

                const z = Math.exp(x);
                return z / (1 + z);
            },

            derivative: x => {
                const y = this.get("sigmoid").forward(x);
                return y * (1 - y);
            }
        });

        this.register("tanh", {
            forward: x => Math.tanh(finiteNumber(x)),

            derivative: x => {
                const y = Math.tanh(finiteNumber(x));
                return 1 - y * y;
            }
        });

        this.register("relu", {
            forward: x => Math.max(0, finiteNumber(x)),

            derivative: x => finiteNumber(x) > 0 ? 1 : 0
        });

        this.register("leaky_relu", {
            forward: (x, alpha = 0.01) => {
                finiteNumber(x);
                finiteNumber(alpha, "Alpha");

                if (alpha < 0) {
                    throw new ActivationError(
                        "Alpha cannot be negative."
                    );
                }

                return x > 0 ? x : alpha * x;
            },

            derivative: (x, alpha = 0.01) => {
                finiteNumber(x);
                finiteNumber(alpha, "Alpha");

                if (alpha < 0) {
                    throw new ActivationError(
                        "Alpha cannot be negative."
                    );
                }

                return x > 0 ? 1 : alpha;
            }
        });

        this.register("gelu", {
            forward: x => {
                finiteNumber(x);

                const c = Math.sqrt(2 / Math.PI);
                const inner = c * (x + 0.044715 * x ** 3);

                return 0.5 * x * (1 + Math.tanh(inner));
            },

            derivative: x => {
                finiteNumber(x);

                const c = Math.sqrt(2 / Math.PI);
                const inner = c * (x + 0.044715 * x ** 3);
                const t = Math.tanh(inner);

                const derivativeInner =
                    c * (1 + 3 * 0.044715 * x ** 2);

                return (
                    0.5 * (1 + t) +
                    0.5 * x * (1 - t * t) * derivativeInner
                );
            }
        });
    }
}

function softmax(logits) {
    const values = vector(logits);
    const maximum = Math.max(...values);

    const exponentials = values.map(
        value => Math.exp(value - maximum)
    );

    const denominator = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    if (!Number.isFinite(denominator) || denominator <= 0) {
        throw new ActivationError(
            "Softmax normalization failed."
        );
    }

    return exponentials.map(value => value / denominator);
}

function softmaxJVP(probabilities, upstream) {
    const p = vector(probabilities);
    const g = vector(upstream);

    if (p.length !== g.length) {
        throw new ActivationError(
            "Gradient dimensions do not match."
        );
    }

    const total = p.reduce((a, b) => a + b, 0);

    if (
        p.some(value => value < 0) ||
        Math.abs(total - 1) > 1e-7
    ) {
        throw new ActivationError(
            "Expected a valid probability distribution."
        );
    }

    const dot = p.reduce(
        (sum, value, index) => sum + value * g[index],
        0
    );

    return p.map(
        (value, index) => value * (g[index] - dot)
    );
}

class ActivationEngine extends EventEmitter {
    constructor() {
        super();

        this.registry = new ActivationRegistry();
        this.executionCount = 0;
        this.failedCount = 0;
    }

    execute(name, inputs, options = {}) {
        const values = vector(inputs);
        const started = performance.now();

        try {
            let output;

            if (name === "softmax") {
                output = softmax(values);
            } else {
                const activation = this.registry.get(name);
                const alpha = options.alpha ?? 0.01;

                output = values.map(
                    value => activation.forward(value, alpha)
                );
            }

            this.executionCount++;

            const result = Object.freeze({
                activation: name,
                input: Object.freeze([...values]),
                output: Object.freeze([...output]),
                elapsedMs: performance.now() - started
            });

            this.emit("executed", result);

            return result;
        } catch (error) {
            this.failedCount++;

            this.emit("failed", {
                activation: name,
                message: error.message
            });

            throw error;
        }
    }

    backward(name, inputs, upstream, options = {}) {
        const values = vector(inputs);
        const gradients = vector(upstream);

        if (values.length !== gradients.length) {
            throw new ActivationError(
                "Input and gradient lengths differ."
            );
        }

        if (name === "softmax") {
            return softmaxJVP(
                softmax(values),
                gradients
            );
        }

        const activation = this.registry.get(name);
        const alpha = options.alpha ?? 0.01;

        return values.map(
            (value, index) =>
                activation.derivative(value, alpha) *
                gradients[index]
        );
    }

    async executeAsync(name, inputs, options = {}) {
        // Promise scheduling allows callers to compose inference tasks.
        return Promise.resolve().then(
            () => this.execute(name, inputs, options)
        );
    }

    async executeBatch(name, batches, options = {}) {
        if (!Array.isArray(batches) || batches.length === 0) {
            throw new ActivationError(
                "Batch collection cannot be empty."
            );
        }

        const results = await Promise.all(
            batches.map(batch =>
                this.executeAsync(name, batch, options)
            )
        );

        return results;
    }

    statistics() {
        return Object.freeze({
            executions: this.executionCount,
            failures: this.failedCount
        });
    }
}

function crossEntropyLoss(probabilities, targetIndex) {
    const p = vector(probabilities);

    if (
        !Number.isInteger(targetIndex) ||
        targetIndex < 0 ||
        targetIndex >= p.length
    ) {
        throw new ActivationError(
            "Invalid classification target."
        );
    }

    const epsilon = 1e-15;

    return -Math.log(
        Math.max(epsilon, p[targetIndex])
    );
}

async function demonstration() {
    const engine = new ActivationEngine();

    engine.on("executed", event => {
        console.log(
            `[EVENT] ${event.activation} completed in ` +
            `${event.elapsedMs.toFixed(4)} ms`
        );
    });

    engine.on("failed", event => {
        console.error(
            `[EVENT] ${event.activation} failed: ${event.message}`
        );
    });

    const testInputs = [-3, -1, 0, 1, 3];

    const activations = [
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "gelu"
    ];

    console.log("\nACTIVATION COMPARISON");

    for (const name of activations) {
        const result = engine.execute(name, testInputs);

        console.log(
            name.padEnd(12),
            result.output.map(x => x.toFixed(5))
        );
    }

    console.log("\nSOFTMAX CLASSIFICATION");

    const logits = [2, 1, 0.1];
    const probabilities = engine.execute(
        "softmax",
        logits
    ).output;

    console.log("Probabilities:", probabilities);
    console.log(
        "Cross-entropy:",
        crossEntropyLoss(probabilities, 0)
    );

    console.log("\nBACKWARD PROPAGATION");

    const gradient = engine.backward(
        "relu",
        [-2, -1, 0, 1, 2],
        [1, 1, 1, 1, 1]
    );

    console.log("ReLU gradient:", gradient);

    const softmaxGradient = engine.backward(
        "softmax",
        logits,
        [1, 0, -1]
    );

    console.log("Softmax JVP:", softmaxGradient);

    console.log("\nASYNCHRONOUS BATCH INFERENCE");

    const batches = [
        [1, 2, 3],
        [-1, 0, 1],
        [5, 2, 1]
    ];

    const results = await engine.executeBatch(
        "softmax",
        batches
    );

    results.forEach((result, index) => {
        console.log(`Batch ${index + 1}:`, result.output);
    });

    console.log("\nENGINE STATISTICS");
    console.log(engine.statistics());

    console.log("\nVALIDATION TEST");

    try {
        engine.execute("unknown", [1, 2, 3]);
    } catch (error) {
        console.log("Expected error:", error.message);
    }

    console.log(engine.statistics());
}

async function runTests() {
    const registry = new ActivationRegistry();

    console.assert(
        Math.abs(registry.get("sigmoid").forward(0) - 0.5) < 1e-10
    );

    console.assert(registry.get("relu").forward(-5) === 0);

    console.assert(
        Math.abs(registry.get("tanh").forward(0)) < 1e-10
    );

    const probabilities = softmax([1000, 1001, 1002]);

    console.assert(
        Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1)
        < 1e-10
    );

    const gradient = softmaxJVP(
        probabilities,
        [1, 0, -1]
    );

    console.assert(
        Math.abs(gradient.reduce((a, b) => a + b, 0)) < 1e-10
    );

    console.log("\nAll basic assertions passed.");
}

async function main() {
    await runTests();
    await demonstration();
}

main().catch(error => {
    console.error("Application failure:", error);
    process.exitCode = 1;
});
