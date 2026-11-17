"use strict";

/*
 * Optimizer Event Laboratory
 *
 * Implements Momentum, AdaGrad, RMSProp, Adam, and AdamW.
 * Provides an event-driven training pipeline, parameter validation,
 * gradient clipping, cancellation, and convergence reporting.
 *
 * Runtime: Node.js 18+
 * Execute: node optimizer_lab.js
 */

const { performance } = require("node:perf_hooks");

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

class TrainingError extends Error {
    constructor(message) {
        super(message);
        this.name = "TrainingError";
    }
}

class Vector {
    static validate(values, name = "Vector") {
        if (!Array.isArray(values) || values.length === 0) {
            throw new ValidationError(`${name} must be a non-empty array.`);
        }

        if (!values.every(
            value => typeof value === "number" && Number.isFinite(value)
        )) {
            throw new ValidationError(`${name} contains invalid numbers.`);
        }

        return values;
    }

    static sameDimensions(a, b) {
        if (a.length !== b.length) {
            throw new ValidationError("Vector dimensions do not match.");
        }
    }

    static zeros(length) {
        return Array(length).fill(0);
    }

    static norm(values) {
        return Math.hypot(...values);
    }

    static clip(values, threshold) {
        if (threshold === null || threshold === undefined) {
            return [...values];
        }

        if (!Number.isFinite(threshold) || threshold <= 0) {
            throw new ValidationError("Invalid clipping threshold.");
        }

        const norm = this.norm(values);

        if (norm <= threshold) {
            return [...values];
        }

        const scale = threshold / norm;
        return values.map(value => value * scale);
    }
}

class EventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(event, listener) {
        if (typeof listener !== "function") {
            throw new ValidationError("Listener must be a function.");
        }

        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }

        this.listeners.get(event).add(listener);

        return () => this.listeners.get(event)?.delete(listener);
    }

    emit(event, payload) {
        const listeners = this.listeners.get(event);

        if (!listeners) return;

        for (const listener of listeners) {
            listener(payload);
        }
    }
}

class Optimizer {
    constructor({
        learningRate = 0.01,
        epsilon = 1e-8,
        weightDecay = 0,
        gradientClip = null
    } = {}) {
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new ValidationError("Learning rate must be positive.");
        }

        if (!Number.isFinite(epsilon) || epsilon <= 0) {
            throw new ValidationError("Epsilon must be positive.");
        }

        if (!Number.isFinite(weightDecay) || weightDecay < 0) {
            throw new ValidationError("Invalid weight decay.");
        }

        if (
            gradientClip !== null &&
            (!Number.isFinite(gradientClip) || gradientClip <= 0)
        ) {
            throw new ValidationError("Invalid gradient clipping threshold.");
        }

        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.gradientClip = gradientClip;
        this.iteration = 0;
    }

    prepare(parameters, gradients) {
        Vector.validate(parameters, "Parameters");
        Vector.validate(gradients, "Gradients");
        Vector.sameDimensions(parameters, gradients);

        return Vector.clip(gradients, this.gradientClip);
    }

    validateOutput(parameters) {
        Vector.validate(parameters, "Updated parameters");
        return parameters;
    }

    update() {
        throw new Error("Abstract optimizer method.");
    }
}

class Momentum extends Optimizer {
    constructor(config = {}, beta = 0.9) {
        super(config);

        if (!(beta >= 0 && beta < 1)) {
            throw new ValidationError("Invalid momentum coefficient.");
        }

        this.beta = beta;
        this.velocity = null;
    }

    update(parameters, gradients) {
        gradients = this.prepare(parameters, gradients);

        this.velocity ??= Vector.zeros(parameters.length);
        this.iteration++;

        this.velocity = this.velocity.map(
            (velocity, index) =>
                this.beta * velocity + gradients[index]
        );

        return this.validateOutput(
            parameters.map(
                (parameter, index) =>
                    parameter - this.learningRate * this.velocity[index]
            )
        );
    }
}

class AdaGrad extends Optimizer {
    constructor(config = {}) {
        super(config);
        this.accumulator = null;
    }

    update(parameters, gradients) {
        gradients = this.prepare(parameters, gradients);

        this.accumulator ??= Vector.zeros(parameters.length);
        this.iteration++;

        this.accumulator = this.accumulator.map(
            (value, index) =>
                value + gradients[index] ** 2
        );

        return this.validateOutput(
            parameters.map((parameter, index) => {
                const denominator =
                    Math.sqrt(this.accumulator[index]) + this.epsilon;

                return parameter -
                    this.learningRate * gradients[index] / denominator;
            })
        );
    }
}

class RMSProp extends Optimizer {
    constructor(config = {}, beta = 0.9) {
        super(config);

        if (!(beta >= 0 && beta < 1)) {
            throw new ValidationError("Invalid RMSProp coefficient.");
        }

        this.beta = beta;
        this.squareAverage = null;
    }

    update(parameters, gradients) {
        gradients = this.prepare(parameters, gradients);

        this.squareAverage ??= Vector.zeros(parameters.length);
        this.iteration++;

        this.squareAverage = this.squareAverage.map(
            (average, index) =>
                this.beta * average +
                (1 - this.beta) * gradients[index] ** 2
        );

        return this.validateOutput(
            parameters.map((parameter, index) => {
                const denominator =
                    Math.sqrt(this.squareAverage[index]) + this.epsilon;

                return parameter -
                    this.learningRate * gradients[index] / denominator;
            })
        );
    }
}

class Adam extends Optimizer {
    constructor(config = {}, beta1 = 0.9, beta2 = 0.999) {
        super(config);

        if (
            !(beta1 >= 0 && beta1 < 1) ||
            !(beta2 >= 0 && beta2 < 1)
        ) {
            throw new ValidationError("Invalid Adam coefficients.");
        }

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.firstMoment = null;
        this.secondMoment = null;
    }

    update(parameters, gradients) {
        gradients = this.prepare(parameters, gradients);

        this.firstMoment ??= Vector.zeros(parameters.length);
        this.secondMoment ??= Vector.zeros(parameters.length);

        this.iteration++;

        const t = this.iteration;

        this.firstMoment = this.firstMoment.map(
            (m, index) =>
                this.beta1 * m + (1 - this.beta1) * gradients[index]
        );

        this.secondMoment = this.secondMoment.map(
            (v, index) =>
                this.beta2 * v +
                (1 - this.beta2) * gradients[index] ** 2
        );

        return this.validateOutput(
            parameters.map((parameter, index) => {
                const correctedFirst =
                    this.firstMoment[index] / (1 - this.beta1 ** t);

                const correctedSecond =
                    this.secondMoment[index] / (1 - this.beta2 ** t);

                return parameter -
                    this.learningRate *
                    correctedFirst /
                    (Math.sqrt(correctedSecond) + this.epsilon);
            })
        );
    }
}

class AdamW extends Adam {
    update(parameters, gradients) {
        const adamParameters = super.update(parameters, gradients);

        // Weight decay is independent of the adaptive moment calculation.
        const decay = this.learningRate * this.weightDecay;

        return this.validateOutput(
            parameters.map((parameter, index) =>
                adamParameters[index] - decay * parameter
            )
        );
    }
}

class OptimizerRegistry {
    static factories = new Map([
        ["Momentum", config => new Momentum(config)],
        ["AdaGrad", config => new AdaGrad(config)],
        ["RMSProp", config => new RMSProp(config)],
        ["Adam", config => new Adam(config)],
        ["AdamW", config => new AdamW(config)]
    ]);

    static create(name, config) {
        const factory = this.factories.get(name);

        if (!factory) {
            throw new ValidationError(`Unknown optimizer: ${name}`);
        }

        return factory(config);
    }
}

class Objective {
    constructor(target, curvature) {
        Vector.validate(target, "Target");
        Vector.validate(curvature, "Curvature");
        Vector.sameDimensions(target, curvature);

        if (curvature.some(value => value <= 0)) {
            throw new ValidationError("Curvature must be positive.");
        }

        this.target = [...target];
        this.curvature = [...curvature];
    }

    loss(parameters) {
        Vector.validate(parameters);
        Vector.sameDimensions(parameters, this.target);

        return parameters.reduce(
            (total, parameter, index) =>
                total + 0.5 * this.curvature[index] *
                (parameter - this.target[index]) ** 2,
            0
        );
    }

    gradient(parameters) {
        Vector.validate(parameters);
        Vector.sameDimensions(parameters, this.target);

        return parameters.map(
            (parameter, index) =>
                this.curvature[index] *
                (parameter - this.target[index])
        );
    }
}

class TrainingEngine {
    constructor(objective, optimizer, eventBus = new EventBus()) {
        this.objective = objective;
        this.optimizer = optimizer;
        this.events = eventBus;
        this.cancelled = false;
        this.running = false;
    }

    cancel() {
        this.cancelled = true;
    }

    async train(initialParameters, {
        maxIterations = 1000,
        tolerance = 1e-8,
        yieldEvery = 50
    } = {}) {
        if (this.running) {
            throw new TrainingError("Training is already running.");
        }

        Vector.validate(initialParameters);
        if (!Number.isInteger(maxIterations) || maxIterations <= 0) {
            throw new ValidationError("Invalid iteration count.");
        }

        if (!Number.isFinite(tolerance) || tolerance < 0) {
            throw new ValidationError("Invalid convergence tolerance.");
        }

        if (!Number.isInteger(yieldEvery) || yieldEvery <= 0) {
            throw new ValidationError("Invalid yield interval.");
        }

        this.running = true;
        this.cancelled = false;

        let parameters = [...initialParameters];
        const initialLoss = this.objective.loss(parameters);
        const history = [initialLoss];

        const started = performance.now();

        try {
            for (let step = 1; step <= maxIterations; step++) {
                if (this.cancelled) break;

                const gradients = this.objective.gradient(parameters);

                parameters = this.optimizer.update(parameters, gradients);

                const loss = this.objective.loss(parameters);

                if (!Number.isFinite(loss)) {
                    throw new TrainingError("Non-finite loss detected.");
                }

                history.push(loss);

                this.events.emit("iteration", {
                    step,
                    loss,
                    parameters: [...parameters]
                });

                if (loss <= tolerance) break;

                // Yield to the Node.js event loop for cancellation and monitoring.
                if (step % yieldEvery === 0) {
                    await new Promise(resolve => setImmediate(resolve));
                }
            }

            const result = {
                optimizer: this.optimizer.constructor.name,
                parameters,
                initialLoss,
                finalLoss: history.at(-1),
                iterations: history.length - 1,
                converged: history.at(-1) <= tolerance,
                cancelled: this.cancelled,
                elapsedMilliseconds: performance.now() - started,
                history
            };

            this.events.emit("completed", result);
            return result;
        } catch (error) {
            this.events.emit("error", error);
            throw error;
        } finally {
            this.running = false;
        }
    }
}

async function runExperiment(name, config, objective) {
    const optimizer = OptimizerRegistry.create(name, config);
    const events = new EventBus();

    events.on("completed", result => {
        if (!Number.isFinite(result.finalLoss)) {
            throw new TrainingError("Invalid training result.");
        }
    });

    const engine = new TrainingEngine(objective, optimizer, events);

    return engine.train(
        [10, 8, -5],
        {
            maxIterations: 3000,
            tolerance: 1e-10
        }
    );
}

async function runTests() {
    const config = { learningRate: 0.01 };

    for (const name of OptimizerRegistry.factories.keys()) {
        const optimizer = OptimizerRegistry.create(name, config);

        const updated = optimizer.update(
            [2, -1],
            [0.5, -0.25]
        );

        if (updated.length !== 2) {
            throw new Error(`${name} dimension test failed.`);
        }

        if (!updated.every(Number.isFinite)) {
            throw new Error(`${name} numerical test failed.`);
        }
    }

    try {
        OptimizerRegistry.create("Unknown", config);
        throw new Error("Unknown optimizer was accepted.");
    } catch (error) {
        if (!(error instanceof ValidationError)) throw error;
    }

    try {
        new Adam({ learningRate: -1 });
        throw new Error("Invalid learning rate was accepted.");
    } catch (error) {
        if (!(error instanceof ValidationError)) throw error;
    }

    try {
        new Adam(config).update([1], [1, 2]);
        throw new Error("Dimension mismatch was accepted.");
    } catch (error) {
        if (!(error instanceof ValidationError)) throw error;
    }

    const adamw = new AdamW({
        learningRate: 0.1,
        weightDecay: 0.1
    });

    const decayed = adamw.update([2], [0]);

    if (!(decayed[0] < 2)) {
        throw new Error("AdamW weight decay test failed.");
    }

    console.log("All JavaScript tests passed.");
}

async function main() {
    await runTests();

    const objective = new Objective(
        [2, -3, 1],
        [1, 10, 50]
    );

    const configurations = [
        ["Momentum", { learningRate: 0.005 }],
        ["AdaGrad", { learningRate: 0.8 }],
        ["RMSProp", { learningRate: 0.02 }],
        ["Adam", { learningRate: 0.08 }],
        ["AdamW", {
            learningRate: 0.08,
            weightDecay: 0.001
        }]
    ];

    const results = await Promise.all(
        configurations.map(([name, config]) =>
            runExperiment(name, config, objective)
        )
    );

    console.log("\nOPTIMIZER COMPARISON");
    console.log("=".repeat(90));

    console.log(
        "Optimizer".padEnd(15),
        "Initial Loss".padStart(15),
        "Final Loss".padStart(18),
        "Iterations".padStart(14)
    );

    for (const result of results) {
        console.log(
            result.optimizer.padEnd(15),
            result.initialLoss.toFixed(6).padStart(15),
            result.finalLoss.toFixed(10).padStart(18),
            String(result.iterations).padStart(14)
        );
    }

    console.log("\nFinal parameter states:");

    for (const result of results) {
        console.log(
            `${result.optimizer}:`,
            result.parameters.map(value => value.toFixed(6))
        );
    }
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
