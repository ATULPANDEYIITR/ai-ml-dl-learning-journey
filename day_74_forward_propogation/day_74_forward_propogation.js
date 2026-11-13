'use strict';

/*
 * Forward Propagation Engine
 *
 * Demonstrates:
 * - Dense neural network layers
 * - Matrix-vector multiplication
 * - Weights and biases
 * - Activation functions
 * - Event-driven inference
 * - Asynchronous batch prediction
 * - Input validation
 * - Numerical stability
 * - Model configuration validation
 *
 * Runtime: Node.js 18+
 * Dependencies: None
 */

const { EventEmitter } = require('node:events');
const fs = require('node:fs/promises');

class NetworkError extends Error {}
class DimensionError extends NetworkError {}
class NumericalError extends NetworkError {}

const ACTIVATIONS = Object.freeze({
    LINEAR: 'linear',
    RELU: 'relu',
    SIGMOID: 'sigmoid',
    TANH: 'tanh',
    SOFTMAX: 'softmax'
});

function validateVector(values, name) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new TypeError(`${name} must be a non-empty array.`);
    }

    for (const value of values) {
        if (typeof value !== 'number' || !Number.isFinite(value)) {
            throw new TypeError(`${name} contains an invalid number.`);
        }
    }
}

function validateActivation(name) {
    if (!Object.values(ACTIVATIONS).includes(name)) {
        throw new TypeError(`Unsupported activation: ${name}`);
    }
}

function sigmoid(value) {
    if (value >= 0) {
        return 1 / (1 + Math.exp(-value));
    }

    const exponential = Math.exp(value);
    return exponential / (1 + exponential);
}

function softmax(values) {
    const maximum = Math.max(...values);

    const exponentials = values.map(
        value => Math.exp(value - maximum)
    );

    const total = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    if (!Number.isFinite(total) || total === 0) {
        throw new NumericalError('Invalid softmax denominator.');
    }

    return exponentials.map(value => value / total);
}

function applyActivation(values, activation) {
    let result;

    switch (activation) {
        case ACTIVATIONS.LINEAR:
            result = [...values];
            break;

        case ACTIVATIONS.RELU:
            result = values.map(value => Math.max(0, value));
            break;

        case ACTIVATIONS.SIGMOID:
            result = values.map(sigmoid);
            break;

        case ACTIVATIONS.TANH:
            result = values.map(Math.tanh);
            break;

        case ACTIVATIONS.SOFTMAX:
            result = softmax(values);
            break;

        default:
            throw new TypeError('Unknown activation function.');
    }

    if (!result.every(Number.isFinite)) {
        throw new NumericalError('Activation generated invalid output.');
    }

    return result;
}

class DenseLayer {
    #weights;
    #biases;

    constructor({
        name,
        weights,
        biases,
        activation = ACTIVATIONS.RELU
    }) {
        if (!Array.isArray(weights) || weights.length === 0) {
            throw new TypeError('Weights must be a non-empty matrix.');
        }

        validateVector(biases, 'Biases');
        validateActivation(activation);

        const inputSize = weights[0].length;

        if (inputSize === 0) {
            throw new DimensionError('Weight rows cannot be empty.');
        }

        if (weights.length !== biases.length) {
            throw new DimensionError(
                'Bias count must equal the number of output neurons.'
            );
        }

        for (const row of weights) {
            validateVector(row, 'Weight row');

            if (row.length !== inputSize) {
                throw new DimensionError(
                    'Weight matrix must be rectangular.'
                );
            }
        }

        this.name = name;
        this.inputSize = inputSize;
        this.outputSize = weights.length;
        this.activation = activation;

        // Copy parameters to prevent external mutation.
        this.#weights = weights.map(row => Object.freeze([...row]));
        this.#biases = Object.freeze([...biases]);
    }

    forward(input) {
        validateVector(input, 'Layer input');

        if (input.length !== this.inputSize) {
            throw new DimensionError(
                `${this.name} requires ${this.inputSize} inputs.`
            );
        }

        const weightedSums = this.#weights.map((row, index) => {
            let total = this.#biases[index];

            for (let j = 0; j < row.length; j++) {
                total += row[j] * input[j];
            }

            if (!Number.isFinite(total)) {
                throw new NumericalError(
                    `Invalid weighted sum in ${this.name}.`
                );
            }

            return total;
        });

        const output = applyActivation(
            weightedSums,
            this.activation
        );

        return {
            weightedSums,
            output
        };
    }

    getParameterCount() {
        return (
            this.inputSize * this.outputSize +
            this.outputSize
        );
    }

    exportConfiguration() {
        return {
            name: this.name,
            weights: this.#weights.map(row => [...row]),
            biases: [...this.#biases],
            activation: this.activation
        };
    }
}

class NeuralNetwork extends EventEmitter {
    #layers;
    #busy = false;

    constructor(layerConfigurations) {
        super();

        if (
            !Array.isArray(layerConfigurations) ||
            layerConfigurations.length === 0
        ) {
            throw new TypeError('Network requires at least one layer.');
        }

        this.#layers = layerConfigurations.map(
            configuration => new DenseLayer(configuration)
        );

        for (let index = 1; index < this.#layers.length; index++) {
            if (
                this.#layers[index - 1].outputSize !==
                this.#layers[index].inputSize
            ) {
                throw new DimensionError(
                    `Incompatible dimensions at layer ${index + 1}.`
                );
            }
        }

        this.inputSize = this.#layers[0].inputSize;
        this.outputSize = this.#layers.at(-1).outputSize;
    }

    get parameterCount() {
        return this.#layers.reduce(
            (total, layer) => total + layer.getParameterCount(),
            0
        );
    }

    forward(input) {
        if (this.#busy) {
            throw new NetworkError(
                'Concurrent inference is not permitted.'
            );
        }

        this.#busy = true;

        try {
            validateVector(input, 'Network input');

            if (input.length !== this.inputSize) {
                throw new DimensionError(
                    `Expected ${this.inputSize} input features.`
                );
            }

            let current = [...input];
            const trace = [];

            for (const [index, layer] of this.#layers.entries()) {
                const result = layer.forward(current);

                trace.push({
                    layer: index + 1,
                    name: layer.name,
                    weightedSums: result.weightedSums,
                    output: result.output
                });

                current = result.output;
            }

            const response = {
                prediction: current,
                trace
            };

            this.emit('inference', response);

            return response;
        } catch (error) {
            this.emit('inferenceError', error);
            throw error;
        } finally {
            this.#busy = false;
        }
    }

    async predictBatch(samples, concurrency = 4) {
        if (!Array.isArray(samples) || samples.length === 0) {
            throw new TypeError('Samples must be a non-empty array.');
        }

        if (
            !Number.isInteger(concurrency) ||
            concurrency < 1 ||
            concurrency > 32
        ) {
            throw new RangeError('Concurrency must be between 1 and 32.');
        }

        const results = new Array(samples.length);
        let nextIndex = 0;

        // Workers process independent samples without shared mutable
        // network state.
        async function worker() {
            while (true) {
                const index = nextIndex++;

                if (index >= samples.length) {
                    return;
                }

                await Promise.resolve();

                results[index] = this.forward(samples[index]).prediction;
            }
        }

        const workers = Array.from(
            { length: Math.min(concurrency, samples.length) },
            () => worker.call(this)
        );

        await Promise.all(workers);

        return results;
    }

    exportConfiguration() {
        return {
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            layers: this.#layers.map(
                layer => layer.exportConfiguration()
            )
        };
    }
}

async function saveModel(network, filename) {
    const content = JSON.stringify(
        network.exportConfiguration(),
        null,
        2
    );

    await fs.writeFile(filename, content, {
        encoding: 'utf8',
        mode: 0o600
    });
}

async function loadModel(filename) {
    const content = await fs.readFile(filename, 'utf8');
    const configuration = JSON.parse(content);

    return new NeuralNetwork(configuration.layers);
}

function createDemoNetwork() {
    return new NeuralNetwork([
        {
            name: 'Hidden Layer',
            weights: [
                [0.2, 0.4, -0.3],
                [0.5, -0.1, 0.2],
                [-0.2, 0.3, 0.6],
                [0.1, 0.2, 0.4]
            ],
            biases: [0.1, -0.1, 0.05, 0.0],
            activation: ACTIVATIONS.RELU
        },
        {
            name: 'Output Layer',
            weights: [
                [0.3, -0.2, 0.4, 0.1],
                [-0.1, 0.5, 0.2, -0.3]
            ],
            biases: [0.05, -0.05],
            activation: ACTIVATIONS.SOFTMAX
        }
    ]);
}

async function runTests(network) {
    const result = network.forward([0.2, 0.5, 0.8]);

    if (Math.abs(
        result.prediction.reduce((a, b) => a + b, 0) - 1
    ) > 1e-10) {
        throw new Error('Softmax probability test failed.');
    }

    if (result.trace.length !== 2) {
        throw new Error('Layer trace test failed.');
    }

    const batch = await network.predictBatch([
        [0.2, 0.5, 0.8],
        [0.5, 0.2, 0.1]
    ]);

    if (batch.length !== 2) {
        throw new Error('Batch inference test failed.');
    }

    try {
        network.forward([1, 2]);
        throw new Error('Expected dimension error was not raised.');
    } catch (error) {
        if (!(error instanceof DimensionError)) {
            throw error;
        }
    }

    console.log('All JavaScript tests passed.');
}

async function main() {
    const network = createDemoNetwork();

    network.on('inference', event => {
        console.log(
            'Inference event:',
            event.prediction
        );
    });

    network.on('inferenceError', error => {
        console.error('Inference failure:', error.message);
    });

    console.log('FORWARD PROPAGATION ENGINE');
    console.log('Input features:', network.inputSize);
    console.log('Output features:', network.outputSize);
    console.log('Parameters:', network.parameterCount);

    const input = [0.2, 0.5, 0.8];
    const result = network.forward(input);

    console.log('\nLayer computation trace:');

    for (const layer of result.trace) {
        console.log(layer.name);
        console.log('Weighted sums:', layer.weightedSums);
        console.log('Activated output:', layer.output);
    }

    console.log('\nPrediction:', result.prediction);

    const batch = await network.predictBatch([
        [0.2, 0.5, 0.8],
        [0.7, 0.1, 0.4],
        [0.9, 0.3, 0.2]
    ]);

    console.log('\nBatch predictions:', batch);

    await runTests(network);

    const filename = 'forward_network.json';

    await saveModel(network, filename);

    const restored = await loadModel(filename);

    const restoredResult = restored.forward(input).prediction;

    if (
        restoredResult.some(
            (value, index) => value !== result.prediction[index]
        )
    ) {
        throw new Error('Model restoration verification failed.');
    }

    await fs.unlink(filename);

    console.log('Model serialization test passed.');
}

main().catch(error => {
    console.error('Application failure:', error.message);
    process.exitCode = 1;
});
