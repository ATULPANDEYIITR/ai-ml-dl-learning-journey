'use strict';

/*
 * Bias-Variance Tradeoff Experiment
 *
 * Event-driven polynomial-model evaluation using Node.js.
 * Independent datasets are generated for repeated trials.
 *
 * Measurements:
 * - Training MSE
 * - Squared bias
 * - Prediction variance
 * - Irreducible noise
 * - Expected test MSE
 * - Generalization gap
 *
 * Run with: node bias-variance.js
 */

const { EventEmitter } = require('node:events');
const { setImmediate: yieldToEventLoop } = require('node:timers/promises');

const CONFIG = Object.freeze({
    repetitions: 80,
    sampleCount: 36,
    gridSize: 81,
    noiseStdDev: 0.20,
    ridgeLambda: 1e-6,
    degrees: Object.freeze([1, 2, 5, 9]),
    seed: 20261103
});

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ValidationError';
    }
}

function assertFinite(value, name) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new ValidationError(`${name} must be a finite number.`);
    }
}

function validateConfiguration(config) {
    if (!Number.isInteger(config.repetitions) || config.repetitions < 2) {
        throw new ValidationError('At least two repetitions are required.');
    }

    if (!Number.isInteger(config.sampleCount) || config.sampleCount < 10) {
        throw new ValidationError('At least ten training samples are required.');
    }

    if (!Number.isInteger(config.gridSize) || config.gridSize < 3) {
        throw new ValidationError('The evaluation grid needs at least three points.');
    }

    assertFinite(config.noiseStdDev, 'Noise standard deviation');
    assertFinite(config.ridgeLambda, 'Ridge regularization');

    if (config.noiseStdDev < 0 || config.ridgeLambda < 0) {
        throw new ValidationError('Noise and regularization cannot be negative.');
    }

    if (!Array.isArray(config.degrees) || config.degrees.length === 0) {
        throw new ValidationError('At least one model degree is required.');
    }

    const uniqueDegrees = new Set(config.degrees);

    if (uniqueDegrees.size !== config.degrees.length) {
        throw new ValidationError('Duplicate model degrees are not allowed.');
    }

    for (const degree of config.degrees) {
        if (!Number.isInteger(degree) || degree < 0 || degree > 12) {
            throw new ValidationError('Degrees must be integers between 0 and 12.');
        }
    }
}

function trueSignal(x) {
    assertFinite(x, 'Input x');
    return Math.sin(2 * Math.PI * x) + 0.25 * x;
}

/*
 * A reproducible pseudo-random generator avoids dependence on
 * environment-specific randomness during model comparisons.
 */
class RandomSource {
    constructor(seed) {
        if (!Number.isInteger(seed)) {
            throw new ValidationError('The random seed must be an integer.');
        }

        this.state = seed >>> 0;

        if (this.state === 0) {
            this.state = 0x6d2b79f5;
        }

        this.spareNormal = null;
    }

    uniform() {
        this.state ^= this.state << 13;
        this.state ^= this.state >>> 17;
        this.state ^= this.state << 5;
        this.state >>>= 0;

        return this.state / 4294967296;
    }

    normal() {
        if (this.spareNormal !== null) {
            const value = this.spareNormal;
            this.spareNormal = null;
            return value;
        }

        let u1 = this.uniform();

        while (u1 <= Number.EPSILON) {
            u1 = this.uniform();
        }

        const u2 = this.uniform();
        const magnitude = Math.sqrt(-2 * Math.log(u1));
        const angle = 2 * Math.PI * u2;

        this.spareNormal = magnitude * Math.sin(angle);

        return magnitude * Math.cos(angle);
    }
}

function createDataset(sampleCount, noiseStdDev, random) {
    if (!Number.isInteger(sampleCount) || sampleCount < 2) {
        throw new ValidationError('Invalid dataset size.');
    }

    if (noiseStdDev < 0 || !Number.isFinite(noiseStdDev)) {
        throw new ValidationError('Invalid noise level.');
    }

    return Array.from({ length: sampleCount }, () => {
        const x = -1 + 2 * random.uniform();

        return Object.freeze({
            x,
            y: trueSignal(x) + random.normal() * noiseStdDev
        });
    });
}

function polynomialFeatures(x, degree) {
    const result = [1];

    for (let index = 1; index <= degree; index += 1) {
        result.push(result[index - 1] * x);
    }

    return result;
}

function solveSystem(matrix, vector) {
    const size = vector.length;

    if (size === 0 || matrix.length !== size) {
        throw new ValidationError('Invalid linear system dimensions.');
    }

    const augmented = matrix.map((row, index) => {
        if (row.length !== size) {
            throw new ValidationError('The matrix must be square.');
        }

        return [...row, vector[index]];
    });

    for (let column = 0; column < size; column += 1) {
        let pivot = column;

        for (let row = column + 1; row < size; row += 1) {
            if (
                Math.abs(augmented[row][column]) >
                Math.abs(augmented[pivot][column])
            ) {
                pivot = row;
            }
        }

        if (Math.abs(augmented[pivot][column]) < 1e-14) {
            throw new Error('Singular regression system.');
        }

        [augmented[column], augmented[pivot]] =
            [augmented[pivot], augmented[column]];

        const divisor = augmented[column][column];

        for (let row = column + 1; row < size; row += 1) {
            const factor = augmented[row][column] / divisor;

            for (let index = column; index <= size; index += 1) {
                augmented[row][index] -= factor * augmented[column][index];
            }
        }
    }

    const solution = Array(size).fill(0);

    for (let row = size - 1; row >= 0; row -= 1) {
        let value = augmented[row][size];

        for (let column = row + 1; column < size; column += 1) {
            value -= augmented[row][column] * solution[column];
        }

        solution[row] = value / augmented[row][row];
    }

    if (!solution.every(Number.isFinite)) {
        throw new Error('Regression coefficients are not finite.');
    }

    return solution;
}

class PolynomialRegressor {
    constructor(degree, ridgeLambda) {
        if (!Number.isInteger(degree) || degree < 0 || degree > 12) {
            throw new ValidationError('Invalid polynomial degree.');
        }

        if (!Number.isFinite(ridgeLambda) || ridgeLambda < 0) {
            throw new ValidationError('Invalid regularization value.');
        }

        this.degree = degree;
        this.ridgeLambda = ridgeLambda;
        this.coefficients = null;
    }

    fit(points) {
        if (!Array.isArray(points) || points.length < 2) {
            throw new ValidationError('Training data is too small.');
        }

        const parameterCount = this.degree + 1;
        const matrix = Array.from(
            { length: parameterCount },
            () => Array(parameterCount).fill(0)
        );
        const vector = Array(parameterCount).fill(0);

        for (const [index, point] of points.entries()) {
            if (
                !Number.isFinite(point.x) ||
                !Number.isFinite(point.y) ||
                point.x < -1 ||
                point.x > 1
            ) {
                throw new ValidationError(`Invalid training observation ${index}.`);
            }

            const features = polynomialFeatures(point.x, this.degree);

            for (let row = 0; row < parameterCount; row += 1) {
                vector[row] += features[row] * point.y;

                for (let column = 0; column < parameterCount; column += 1) {
                    matrix[row][column] += features[row] * features[column];
                }
            }
        }

        // The intercept is excluded from the L2 penalty.
        for (let index = 1; index < parameterCount; index += 1) {
            matrix[index][index] += this.ridgeLambda;
        }

        this.coefficients = solveSystem(matrix, vector);

        return this;
    }

    predict(x) {
        if (this.coefficients === null) {
            throw new Error('Model has not been fitted.');
        }

        if (!Number.isFinite(x) || x < -1 || x > 1) {
            throw new ValidationError('Prediction input must be in [-1, 1].');
        }

        const features = polynomialFeatures(x, this.degree);

        const prediction = this.coefficients.reduce(
            (sum, coefficient, index) =>
                sum + coefficient * features[index],
            0
        );

        if (!Number.isFinite(prediction)) {
            throw new Error('Prediction is not finite.');
        }

        return prediction;
    }
}

function meanSquaredError(actual, predicted) {
    if (
        !Array.isArray(actual) ||
        actual.length === 0 ||
        actual.length !== predicted.length
    ) {
        throw new ValidationError('MSE arrays must have equal non-zero lengths.');
    }

    return actual.reduce(
        (sum, value, index) => sum + (value - predicted[index]) ** 2,
        0
    ) / actual.length;
}

function average(values) {
    if (values.length === 0) {
        throw new ValidationError('Cannot average an empty collection.');
    }

    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

class ExperimentEvents extends EventEmitter {
    constructor() {
        super();
        this.setMaxListeners(20);
    }
}

class BiasVarianceExperiment {
    constructor(config, events = new ExperimentEvents()) {
        validateConfiguration(config);

        this.config = config;
        this.events = events;
        this.random = new RandomSource(config.seed);
    }

    async evaluateDegree(degree, grid, signalValues) {
        const {
            repetitions,
            sampleCount,
            noiseStdDev,
            ridgeLambda
        } = this.config;

        const predictionRuns = [];
        const trainingErrors = [];

        this.events.emit('candidate:started', {
            degree,
            repetitions
        });

        for (let repetition = 0; repetition < repetitions; repetition += 1) {
            const dataset = createDataset(
                sampleCount,
                noiseStdDev,
                this.random
            );

            const model = new PolynomialRegressor(
                degree,
                ridgeLambda
            ).fit(dataset);

            const predictions = grid.map(x => model.predict(x));
            const trainingPrediction = dataset.map(
                point => model.predict(point.x)
            );

            predictionRuns.push(predictions);
            trainingErrors.push(
                meanSquaredError(
                    dataset.map(point => point.y),
                    trainingPrediction
                )
            );

            // Yield periodically so the event loop can process other tasks.
            if ((repetition + 1) % 10 === 0) {
                await yieldToEventLoop();
            }
        }

        const meanPredictions = grid.map((_, index) =>
            average(
                predictionRuns.map(run => run[index])
            )
        );

        const biasSquared = average(
            meanPredictions.map(
                (prediction, index) =>
                    (prediction - signalValues[index]) ** 2
            )
        );

        let predictionVariance = 0;

        for (let index = 0; index < grid.length; index += 1) {
            const meanPrediction = meanPredictions[index];

            const varianceAtPoint = average(
                predictionRuns.map(
                    run => (run[index] - meanPrediction) ** 2
                )
            );

            predictionVariance += varianceAtPoint;
        }

        predictionVariance /= grid.length;

        const averageTrainingMse = average(trainingErrors);
        const irreducibleNoise = noiseStdDev ** 2;
        const signalPredictionMse = biasSquared + predictionVariance;
        const expectedTestMse = signalPredictionMse + irreducibleNoise;

        return Object.freeze({
            degree,
            averageTrainingMse,
            biasSquared,
            predictionVariance,
            irreducibleNoise,
            signalPredictionMse,
            expectedTestMse,
            generalizationGap: expectedTestMse - averageTrainingMse
        });
    }

    classify(results) {
        const bestTest = Math.min(...results.map(item => item.expectedTestMse));
        const bestTraining = Math.min(
            ...results.map(item => item.averageTrainingMse)
        );

        return results.map(item => {
            let risk = 'GENERALIZES_WELL';

            if (item.expectedTestMse > bestTest * 1.10) {
                risk = item.averageTrainingMse > bestTraining * 1.20
                    ? 'UNDERFITTING'
                    : 'OVERFITTING';
            }

            return Object.freeze({
                ...item,
                risk
            });
        });
    }

    async run() {
        const { gridSize, degrees } = this.config;

        const grid = Array.from(
            { length: gridSize },
            (_, index) => -1 + (2 * index) / (gridSize - 1)
        );

        const signalValues = grid.map(trueSignal);
        const rawResults = [];

        this.events.emit('experiment:started', {
            degrees: [...degrees],
            gridSize
        });

        for (const degree of degrees) {
            const result = await this.evaluateDegree(
                degree,
                grid,
                signalValues
            );

            rawResults.push(result);
            this.events.emit('candidate:completed', result);
        }

        const results = this.classify(rawResults)
            .sort((left, right) => left.degree - right.degree);

        const best = results.reduce(
            (current, candidate) =>
                candidate.expectedTestMse < current.expectedTestMse
                    ? candidate
                    : current
        );

        const report = Object.freeze({
            configuration: {
                repetitions: this.config.repetitions,
                sampleCount: this.config.sampleCount,
                noiseStdDev: this.config.noiseStdDev,
                ridgeLambda: this.config.ridgeLambda
            },
            results,
            selectedDegree: best.degree,
            selectedExpectedTestMse: best.expectedTestMse
        });

        this.events.emit('experiment:completed', report);

        return report;
    }
}

function validateFailureConditions() {
    const untrainedModel = new PolynomialRegressor(2, 1e-6);

    try {
        untrainedModel.predict(0.2);
        throw new Error('Prediction should fail for an untrained model.');
    } catch (error) {
        if (error.message !== 'Model has not been fitted.') {
            throw error;
        }
    }

    try {
        new PolynomialRegressor(13, 1e-6);
        throw new Error('Excessive model complexity was not rejected.');
    } catch (error) {
        if (!(error instanceof ValidationError)) {
            throw error;
        }
    }

    try {
        createDataset(20, -0.1, new RandomSource(1));
        throw new Error('Negative observation noise was not rejected.');
    } catch (error) {
        if (!(error instanceof ValidationError)) {
            throw error;
        }
    }

    const constantModel = new PolynomialRegressor(0, 0);
    constantModel.fit([
        { x: -1, y: 3 },
        { x: 1, y: 3 }
    ]);

    if (Math.abs(constantModel.predict(0) - 3) > 1e-9) {
        throw new Error('Constant-model validation failed.');
    }
}

async function main() {
    validateFailureConditions();

    const events = new ExperimentEvents();

    events.on('experiment:started', event => {
        process.stdout.write(
            `Starting experiment for degrees: ${event.degrees.join(', ')}\n`
        );
    });

    events.on('candidate:started', event => {
        process.stdout.write(
            `Evaluating degree ${event.degree} across ${event.repetitions} trials...\n`
        );
    });

    events.on('candidate:completed', event => {
        process.stdout.write(
            `Degree ${event.degree}: expected test MSE = ` +
            `${event.expectedTestMse.toFixed(5)}\n`
        );
    });

    events.on('experiment:completed', report => {
        process.stdout.write(
            `Selected degree ${report.selectedDegree} ` +
            `with expected test MSE ` +
            `${report.selectedExpectedTestMse.toFixed(5)}\n`
        );
    });

    const experiment = new BiasVarianceExperiment(CONFIG, events);
    const report = await experiment.run();

    console.log('\nFinal evaluation:');
    console.table(
        report.results.map(result => ({
            degree: result.degree,
            trainingMSE: result.averageTrainingMse.toFixed(5),
            biasSquared: result.biasSquared.toFixed(5),
            variance: result.predictionVariance.toFixed(5),
            noise: result.irreducibleNoise.toFixed(5),
            expectedTestMSE: result.expectedTestMse.toFixed(5),
            generalizationGap: result.generalizationGap.toFixed(5),
            risk: result.risk
        }))
    );
}

main().catch(error => {
    console.error(`Experiment failed: ${error.message}`);
    process.exitCode = 1;
});
