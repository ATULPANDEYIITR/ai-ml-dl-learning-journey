"use strict";

/*
 * Backpropagation using an event-driven computational graph.
 *
 * Features:
 * - Directed acyclic graph construction.
 * - Forward execution.
 * - Reverse-mode gradient propagation.
 * - Topological sorting.
 * - Shared-node gradient accumulation.
 * - Gradient verification.
 * - Asynchronous training.
 * - Training events and validation.
 * - Numerical stability checks.
 */

const { EventEmitter } = require("node:events");
const fs = require("node:fs/promises");

class GraphError extends Error {}

class TensorNode {
    constructor(value, operation = "input", parents = []) {
        if (!Number.isFinite(value)) {
            throw new GraphError("Node value must be finite.");
        }

        this.id = TensorNode.nextId++;
        this.value = value;
        this.grad = 0;
        this.operation = operation;
        this.parents = parents;
        this.backwardRule = () => {};
    }

    static nextId = 1;

    static convert(value) {
        return value instanceof TensorNode
            ? value
            : new TensorNode(Number(value));
    }

    static operation(value, operation, parents, rule) {
        const node = new TensorNode(value, operation, parents);
        node.backwardRule = rule(node);
        return node;
    }

    add(other) {
        other = TensorNode.convert(other);

        return TensorNode.operation(
            this.value + other.value,
            "add",
            [this, other],
            result => () => {
                this.grad += result.grad;
                other.grad += result.grad;
            }
        );
    }

    mul(other) {
        other = TensorNode.convert(other);

        return TensorNode.operation(
            this.value * other.value,
            "multiply",
            [this, other],
            result => () => {
                this.grad += other.value * result.grad;
                other.grad += this.value * result.grad;
            }
        );
    }

    sub(other) {
        return this.add(TensorNode.convert(other).mul(-1));
    }

    div(other) {
        other = TensorNode.convert(other);

        if (other.value === 0) {
            throw new GraphError("Division by zero.");
        }

        return this.mul(other.pow(-1));
    }

    pow(exponent) {
        if (!Number.isFinite(exponent)) {
            throw new GraphError("Invalid exponent.");
        }

        if (
            this.value < 0 &&
            !Number.isInteger(exponent)
        ) {
            throw new GraphError("Invalid fractional power.");
        }

        if (this.value === 0 && exponent < 1) {
            throw new GraphError("Undefined derivative.");
        }

        const output = Math.pow(this.value, exponent);

        return TensorNode.operation(
            output,
            "power",
            [this],
            result => () => {
                this.grad +=
                    exponent *
                    Math.pow(this.value, exponent - 1) *
                    result.grad;
            }
        );
    }

    tanh() {
        const output = Math.tanh(this.value);

        return TensorNode.operation(
            output,
            "tanh",
            [this],
            result => () => {
                this.grad +=
                    (1 - result.value ** 2) * result.grad;
            }
        );
    }

    relu() {
        const output = Math.max(0, this.value);

        return TensorNode.operation(
            output,
            "relu",
            [this],
            result => () => {
                this.grad +=
                    (this.value > 0 ? 1 : 0) * result.grad;
            }
        );
    }

    backward() {
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
            node.grad = 0;
        }

        this.grad = 1;

        for (let i = topology.length - 1; i >= 0; i--) {
            topology[i].backwardRule();
        }

        return topology;
    }
}

class NeuralNetwork extends EventEmitter {
    constructor(options = {}) {
        super();

        const {
            inputSize = 2,
            hiddenSize = 4,
            learningRate = 0.08,
            seed = 42
        } = options;

        if (
            inputSize < 1 ||
            hiddenSize < 1 ||
            learningRate <= 0
        ) {
            throw new GraphError("Invalid network configuration.");
        }

        this.learningRate = learningRate;
        this.random = this.createRandom(seed);

        this.hidden = Array.from(
            { length: hiddenSize },
            () => this.createNeuron(inputSize)
        );

        this.output = this.createNeuron(hiddenSize);
        this.epoch = 0;
    }

    createRandom(seed) {
        let state = seed >>> 0;

        return () => {
            state = (1664525 * state + 1013904223) >>> 0;
            return state / 4294967296;
        };
    }

    createNeuron(inputSize) {
        return {
            weights: Array.from(
                { length: inputSize },
                () => new TensorNode(
                    this.random() * 2 - 1
                )
            ),
            bias: new TensorNode(0)
        };
    }

    activate(neuron, inputs) {
        if (inputs.length !== neuron.weights.length) {
            throw new GraphError("Input dimensions do not match.");
        }

        let result = neuron.bias;

        for (let i = 0; i < inputs.length; i++) {
            result = result.add(
                neuron.weights[i].mul(inputs[i])
            );
        }

        return result.tanh();
    }

    predict(inputs) {
        if (
            !Array.isArray(inputs) ||
            inputs.some(value => !Number.isFinite(value))
        ) {
            throw new GraphError("Invalid input vector.");
        }

        const hidden = this.hidden.map(
            neuron => this.activate(
                neuron,
                inputs.map(value => new TensorNode(value))
            )
        );

        return this.activate(this.output, hidden);
    }

    parameters() {
        return [
            ...this.hidden.flatMap(
                neuron => [...neuron.weights, neuron.bias]
            ),
            ...this.output.weights,
            this.output.bias
        ];
    }

    zeroGrad() {
        for (const parameter of this.parameters()) {
            parameter.grad = 0;
        }
    }

    update() {
        const parameters = this.parameters();

        // Reject non-finite updates before changing any parameter.
        for (const parameter of parameters) {
            const next =
                parameter.value -
                this.learningRate * parameter.grad;

            if (!Number.isFinite(next)) {
                throw new GraphError("Invalid parameter update.");
            }
        }

        for (const parameter of parameters) {
            parameter.value -=
                this.learningRate * parameter.grad;
        }
    }

    async train(dataset, epochs = 3000) {
        if (!Array.isArray(dataset) || dataset.length === 0) {
            throw new GraphError("Training dataset is empty.");
        }

        if (!Number.isInteger(epochs) || epochs < 1) {
            throw new GraphError("Invalid epoch count.");
        }

        for (const sample of dataset) {
            if (
                !Array.isArray(sample.input) ||
                sample.input.length !== this.hidden[0].weights.length ||
                !Number.isFinite(sample.target)
            ) {
                throw new GraphError("Invalid training sample.");
            }
        }

        for (let epoch = 0; epoch < epochs; epoch++) {
            this.zeroGrad();

            let loss = new TensorNode(0);

            for (const sample of dataset) {
                const prediction = this.predict(sample.input);

                const error = prediction.sub(sample.target);

                loss = loss.add(error.mul(error));
            }

            loss = loss.div(dataset.length);

            loss.backward();
            this.update();

            this.epoch++;

            const event = {
                epoch: this.epoch,
                loss: loss.value
            };

            if (!Number.isFinite(loss.value)) {
                throw new GraphError("Non-finite training loss.");
            }

            this.emit("epoch", event);

            if (epoch % 500 === 0) {
                await new Promise(resolve => setImmediate(resolve));
            }
        }

        this.emit("trainingComplete", {
            epochs: this.epoch
        });
    }
}

function verifyGradient() {
    const x = new TensorNode(2);

    const result = x.mul(x).add(x.mul(3)).add(2);

    result.backward();

    const numerical = (
        ((2.000001 ** 2) + 3 * 2.000001 + 2) -
        ((1.999999 ** 2) + 3 * 1.999999 + 2)
    ) / 0.000002;

    if (Math.abs(x.grad - numerical) > 1e-5) {
        throw new GraphError("Gradient verification failed.");
    }

    console.log("Analytical gradient:", x.grad);
    console.log("Numerical gradient:", numerical);
}

async function main() {
    verifyGradient();

    const dataset = [
        { input: [0, 0], target: 0 },
        { input: [0, 1], target: 1 },
        { input: [1, 0], target: 1 },
        { input: [1, 1], target: 0 }
    ];

    const network = new NeuralNetwork();

    network.on("epoch", ({ epoch, loss }) => {
        if (epoch % 500 === 1) {
            console.log(
                `Epoch ${epoch}: loss=${loss.toFixed(6)}`
            );
        }
    });

    network.on("trainingComplete", event => {
        console.log("Training completed:", event);
    });

    await network.train(dataset, 3000);

    console.log("\nPredictions:");

    const results = dataset.map(sample => ({
        input: sample.input,
        target: sample.target,
        prediction: network.predict(sample.input).value
    }));

    for (const result of results) {
        console.log(
            result.input,
            "=>",
            result.prediction.toFixed(4),
            "target:",
            result.target
        );
    }

    // Persist only model parameters, not executable graph functions.
    const checkpoint = {
        format: "scalar-mlp-v1",
        epoch: network.epoch,
        parameters: network.parameters().map(p => p.value)
    };

    const temporaryFile = "backprop-checkpoint.tmp";
    const destination = "backprop-checkpoint.json";

    await fs.writeFile(
        temporaryFile,
        JSON.stringify(checkpoint, null, 2),
        { encoding: "utf8", mode: 0o600 }
    );

    await fs.rename(temporaryFile, destination);

    console.log(`Checkpoint saved: ${destination}`);
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
