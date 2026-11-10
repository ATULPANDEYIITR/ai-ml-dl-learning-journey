"use strict";

/*
 * Neural Network Mathematics
 *
 * Event-driven neuron simulation, weighted aggregation,
 * activation policies, asynchronous batch inference,
 * numerical differentiation, and model validation.
 *
 * Runtime: Node.js 18+
 * Dependencies: Node.js standard library only.
 */

const { EventEmitter } = require("node:events");
const fs = require("node:fs/promises");
const path = require("node:path");

// ============================================================
// 1. Errors and Numeric Validation
// ============================================================

class NeuralNetworkError extends Error {
    constructor(message) {
        super(message);
        this.name = "NeuralNetworkError";
    }
}

function validateNumber(value, name = "Value") {
    if (
        typeof value !== "number" ||
        !Number.isFinite(value)
    ) {
        throw new NeuralNetworkError(
            `${name} must be a finite number.`
        );
    }
}

function validateVector(values, expectedLength = null) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new NeuralNetworkError(
            "Input must be a non-empty array."
        );
    }

    if (
        expectedLength !== null &&
        values.length !== expectedLength
    ) {
        throw new NeuralNetworkError(
            `Expected ${expectedLength} values, received ${values.length}.`
        );
    }

    values.forEach((value, index) => {
        validateNumber(value, `Element ${index}`);
    });
}

// ============================================================
// 2. Activation Registry
// ============================================================

class ActivationRegistry {
    static functions = new Map([
        [
            "linear",
            {
                forward: x => x,
                derivative: () => 1
            }
        ],
        [
            "sigmoid",
            {
                forward: x => {
                    if (x >= 0) {
                        return 1 / (1 + Math.exp(-Math.min(x, 709)));
                    }

                    const exp = Math.exp(Math.max(x, -709));
                    return exp / (1 + exp);
                },
                derivative: x => {
                    const output = ActivationRegistry.get("sigmoid").forward(x);
                    return output * (1 - output);
                }
            }
        ],
        [
            "tanh",
            {
                forward: x => Math.tanh(x),
                derivative: x => 1 - Math.tanh(x) ** 2
            }
        ],
        [
            "relu",
            {
                forward: x => Math.max(0, x),
                derivative: x => x > 0 ? 1 : 0
            }
        ],
        [
            "leaky_relu",
            {
                forward: x => x > 0 ? x : 0.01 * x,
                derivative: x => x > 0 ? 1 : 0.01
            }
        ]
    ]);

    static get(name) {
        const activation = this.functions.get(name);

        if (!activation) {
            throw new NeuralNetworkError(
                `Unknown activation function: ${name}`
            );
        }

        return activation;
    }
}

// ============================================================
// 3. Weighted Sum
// ============================================================

function calculateWeightedSum(inputs, weights, bias) {
    validateVector(inputs);
    validateVector(weights, inputs.length);
    validateNumber(bias, "Bias");

    let result = bias;

    for (let i = 0; i < inputs.length; i++) {
        result += inputs[i] * weights[i];
    }

    validateNumber(result, "Weighted sum");

    return result;
}

// ============================================================
// 4. Event-Driven Neuron
// ============================================================

class Neuron extends EventEmitter {
    #weights;
    #bias;
    #activationName;
    #revision = 0;

    constructor({
        id,
        weights,
        bias = 0,
        activation = "sigmoid"
    }) {
        super();

        if (typeof id !== "string" || !id.trim()) {
            throw new NeuralNetworkError(
                "Neuron identifier is required."
            );
        }

        validateVector(weights);
        validateNumber(bias, "Bias");

        ActivationRegistry.get(activation);

        this.id = id;
        this.#weights = [...weights];
        this.#bias = bias;
        this.#activationName = activation;
    }

    get revision() {
        return this.#revision;
    }

    get configuration() {
        return Object.freeze({
            id: this.id,
            weights: [...this.#weights],
            bias: this.#bias,
            activation: this.#activationName
        });
    }

    calculate(inputs) {
        validateVector(inputs, this.#weights.length);

        const z = calculateWeightedSum(
            inputs,
            this.#weights,
            this.#bias
        );

        const activation = ActivationRegistry.get(
            this.#activationName
        );

        const output = activation.forward(z);

        const result = Object.freeze({
            neuronId: this.id,
            inputs: [...inputs],
            weightedSum: z,
            activation: this.#activationName,
            output,
            revision: this.#revision
        });

        this.emit("calculated", result);

        return result;
    }

    updateParameters({ weights, bias }) {
        validateVector(weights, this.#weights.length);
        validateNumber(bias, "Bias");

        this.#weights = [...weights];
        this.#bias = bias;
        this.#revision++;

        this.emit("parametersUpdated", {
            neuronId: this.id,
            revision: this.#revision
        });
    }
}

// ============================================================
// 5. Dense Layer
// ============================================================

class DenseLayer {
    constructor(neurons) {
        if (
            !Array.isArray(neurons) ||
            neurons.length === 0 ||
            !neurons.every(neuron => neuron instanceof Neuron)
        ) {
            throw new NeuralNetworkError(
                "A dense layer requires at least one valid neuron."
            );
        }

        const inputSizes = new Set(
            neurons.map(neuron => neuron.configuration.weights.length)
        );

        if (inputSizes.size !== 1) {
            throw new NeuralNetworkError(
                "All neurons in a dense layer must accept the same input size."
            );
        }

        this.neurons = [...neurons];
        this.inputSize = [...inputSizes][0];
        this.outputSize = neurons.length;
    }

    forward(inputs) {
        validateVector(inputs, this.inputSize);

        return this.neurons.map(
            neuron => neuron.calculate(inputs).output
        );
    }
}

// ============================================================
// 6. Neural Network
// ============================================================

class NeuralNetwork extends EventEmitter {
    constructor(layers) {
        super();

        if (
            !Array.isArray(layers) ||
            layers.length === 0 ||
            !layers.every(layer => layer instanceof DenseLayer)
        ) {
            throw new NeuralNetworkError(
                "Network requires valid dense layers."
            );
        }

        for (let i = 1; i < layers.length; i++) {
            if (layers[i - 1].outputSize !== layers[i].inputSize) {
                throw new NeuralNetworkError(
                    `Layer ${i} has an incompatible input dimension.`
                );
            }
        }

        this.layers = [...layers];
        this.inferenceCount = 0;
    }

    predict(inputs) {
        let current = [...inputs];

        validateVector(
            current,
            this.layers[0].inputSize
        );

        for (const layer of this.layers) {
            current = layer.forward(current);
        }

        this.inferenceCount++;

        const result = Object.freeze({
            prediction: [...current],
            inferenceNumber: this.inferenceCount
        });

        this.emit("prediction", result);

        return result;
    }

    // Bounded concurrency prevents uncontrolled resource usage.
    async predictBatch(dataset, concurrency = 4) {
        if (!Array.isArray(dataset) || dataset.length === 0) {
            throw new NeuralNetworkError(
                "Batch dataset cannot be empty."
            );
        }

        if (
            !Number.isInteger(concurrency) ||
            concurrency < 1 ||
            concurrency > 64
        ) {
            throw new NeuralNetworkError(
                "Concurrency must be between 1 and 64."
            );
        }

        dataset.forEach(input => {
            validateVector(input, this.layers[0].inputSize);
        });

        const results = new Array(dataset.length);
        let nextIndex = 0;

        const workers = Array.from(
            { length: Math.min(concurrency, dataset.length) },
            async () => {
                while (true) {
                    const index = nextIndex++;

                    if (index >= dataset.length) {
                        break;
                    }

                    // Yield between tasks to support event-loop fairness.
                    await Promise.resolve();

                    results[index] = this.predict(dataset[index]);
                }
            }
        );

        await Promise.all(workers);

        this.emit("batchCompleted", {
            sampleCount: dataset.length
        });

        return results;
    }
}

// ============================================================
// 7. Loss and Numerical Differentiation
// ============================================================

function meanSquaredError(predictions, targets) {
    validateVector(predictions);
    validateVector(targets, predictions.length);

    return predictions.reduce(
        (sum, value, index) =>
            sum + (value - targets[index]) ** 2,
        0
    ) / predictions.length;
}

function numericalDerivative(
    fn,
    x,
    epsilon = 1e-6
) {
    validateNumber(x);
    validateNumber(epsilon);

    if (epsilon <= 0) {
        throw new NeuralNetworkError(
            "Epsilon must be positive."
        );
    }

    return (
        fn(x + epsilon) - fn(x - epsilon)
    ) / (2 * epsilon);
}

function verifySigmoidDerivative() {
    const sigmoid = ActivationRegistry.get("sigmoid");
    const x = 0.7;

    const analytical = sigmoid.derivative(x);

    const numerical = numericalDerivative(
        sigmoid.forward,
        x
    );

    return {
        analytical,
        numerical,
        passed: Math.abs(analytical - numerical) < 1e-5
    };
}

// ============================================================
// 8. Model Serialization
// ============================================================

async function saveModel(network, filename) {
    const model = {
        formatVersion: 1,
        layers: network.layers.map(layer =>
            layer.neurons.map(neuron => neuron.configuration)
        )
    };

    const destination = path.resolve(filename);
    const temporary = `${destination}.tmp`;

    try {
        await fs.writeFile(
            temporary,
            JSON.stringify(model, null, 2),
            { encoding: "utf8", flag: "w" }
        );

        await fs.rename(temporary, destination);
    } catch (error) {
        await fs.rm(temporary, { force: true }).catch(() => {});
        throw new NeuralNetworkError(
            `Model persistence failed: ${error.message}`
        );
    }
}

async function loadModel(filename) {
    const data = JSON.parse(
        await fs.readFile(filename, "utf8")
    );

    if (
        data.formatVersion !== 1 ||
        !Array.isArray(data.layers) ||
        data.layers.length === 0
    ) {
        throw new NeuralNetworkError(
            "Unsupported or invalid model format."
        );
    }

    const layers = data.layers.map(layerData => {
        if (!Array.isArray(layerData)) {
            throw new NeuralNetworkError("Invalid layer data.");
        }

        return new DenseLayer(
            layerData.map(config => new Neuron(config))
        );
    });

    return new NeuralNetwork(layers);
}

// ============================================================
// 9. Demonstration
// ============================================================

async function main() {
    console.log("\nNEURAL NETWORK MATHEMATICS");

    // A neuron computes z = w1*x1 + w2*x2 + b.
    const neuron = new Neuron({
        id: "input-neuron-001",
        weights: [0.5, -0.3],
        bias: 0.2,
        activation: "sigmoid"
    });

    neuron.on("calculated", event => {
        console.log(
            `Neuron ${event.neuronId}:`,
            `z=${event.weightedSum.toFixed(4)}`,
            `output=${event.output.toFixed(4)}`
        );
    });

    const singleResult = neuron.calculate([1.0, 2.0]);

    console.log("Single neuron result:", singleResult);

    // Construct a two-layer XOR network.
    const hidden = new DenseLayer([
        new Neuron({
            id: "hidden-1",
            weights: [0.5, -0.4],
            bias: 0.1,
            activation: "tanh"
        }),
        new Neuron({
            id: "hidden-2",
            weights: [-0.3, 0.7],
            bias: -0.2,
            activation: "tanh"
        })
    ]);

    const output = new DenseLayer([
        new Neuron({
            id: "output-1",
            weights: [0.8, -0.6],
            bias: 0.15,
            activation: "sigmoid"
        })
    ]);

    const network = new NeuralNetwork([
        hidden,
        output
    ]);

    network.on("prediction", event => {
        console.log(
            `Inference ${event.inferenceNumber}:`,
            event.prediction.map(x => x.toFixed(5))
        );
    });

    const inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ];

    const predictions = await network.predictBatch(
        inputs,
        2
    );

    console.log(
        "MSE against demonstration targets:",
        meanSquaredError(
            predictions.map(item => item.prediction[0]),
            [0, 1, 1, 0]
        ).toFixed(6)
    );

    console.log(
        "Sigmoid derivative verification:",
        verifySigmoidDerivative()
    );

    const filename = "neural-network-js-model.json";

    await saveModel(network, filename);

    const restored = await loadModel(filename);

    console.log(
        "Restored model prediction:",
        restored.predict([0, 1]).prediction
    );

    // Invalid dimensions are rejected before computation.
    try {
        neuron.calculate([1, 2, 3]);
    } catch (error) {
        console.log("Expected validation error:", error.message);
    }

    console.log(
        "Total original-network inferences:",
        network.inferenceCount
    );
}

main().catch(error => {
    console.error("Execution failed:", error.message);
    process.exitCode = 1;
});
