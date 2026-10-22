'use strict';

/*
 * Support Vector Machines:
 * an event-driven soft-margin policy evaluator for a linear SVM.
 *
 * The implementation uses the primal objective
 *
 *   1/2 ||w||^2 + C * sum(max(0, 1 - y(w.x + b)))
 *
 * and emits training/evaluation events through Node.js EventEmitter.
 *
 * Run with:
 *   node svm.js
 */

const { EventEmitter } = require('node:events');

class ValidationError extends Error {}

function dot(a, b) {
    if (a.length !== b.length) {
        throw new ValidationError('Vector dimensions must match.');
    }
    return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function vectorNorm(values) {
    return Math.sqrt(dot(values, values));
}

function validateLabel(label) {
    if (label !== -1 && label !== 1) {
        throw new ValidationError('SVM labels must be -1 or +1.');
    }
}

function validateFeatures(features) {
    if (!Array.isArray(features) || features.length === 0) {
        throw new ValidationError('Features must be a non-empty array.');
    }
    if (!features.every(Number.isFinite)) {
        throw new ValidationError('Every feature must be finite.');
    }
}

class TrainingExample {
    constructor(features, label, id) {
        validateFeatures(features);
        validateLabel(label);

        this.id = id;
        this.features = Object.freeze([...features]);
        this.label = label;
        Object.freeze(this);
    }
}

function standardize(dataset) {
    if (dataset.length === 0) {
        throw new ValidationError('Dataset cannot be empty.');
    }

    const dimension = dataset[0].features.length;

    if (dataset.some(item => item.features.length !== dimension)) {
        throw new ValidationError('All observations must have equal dimensions.');
    }

    const means = Array.from({ length: dimension }, (_, j) =>
        dataset.reduce((sum, item) => sum + item.features[j], 0) / dataset.length
    );

    const standardDeviations = Array.from({ length: dimension }, (_, j) => {
        const variance = dataset.reduce((sum, item) => {
            const delta = item.features[j] - means[j];
            return sum + delta * delta;
        }, 0) / dataset.length;

        return Math.sqrt(variance) || 1;
    });

    const transformed = dataset.map(item => new TrainingExample(
        item.features.map(
            (value, j) => (value - means[j]) / standardDeviations[j]
        ),
        item.label,
        item.id
    ));

    return { transformed, means, standardDeviations };
}

class LinearSVM extends EventEmitter {
    constructor({ C = 1, learningRate = 0.002, epochs = 5000 } = {}) {
        super();

        if (!Number.isFinite(C) || C <= 0) {
            throw new ValidationError('C must be positive.');
        }
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new ValidationError('learningRate must be positive.');
        }
        if (!Number.isInteger(epochs) || epochs <= 0) {
            throw new ValidationError('epochs must be a positive integer.');
        }

        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.weights = [];
        this.bias = 0;
    }

    fit(dataset) {
        if (!Array.isArray(dataset) || dataset.length === 0) {
            throw new ValidationError('Training data cannot be empty.');
        }

        const dimension = dataset[0].features.length;

        if (dataset.some(item => item.features.length !== dimension)) {
            throw new ValidationError('Feature dimensions are inconsistent.');
        }

        this.weights = Array(dimension).fill(0);
        this.bias = 0;

        for (let epoch = 1; epoch <= this.epochs; epoch++) {
            for (const example of dataset) {
                const score = this.decisionFunction(example.features);
                const functionalMargin = example.label * score;

                if (functionalMargin >= 1) {
                    // Outside the margin: optimize only L2 regularization.
                    this.weights = this.weights.map(
                        weight => weight - this.learningRate * weight
                    );
                } else {
                    // Inside the margin: hinge loss contributes to the gradient.
                    this.weights = this.weights.map(
                        (weight, j) =>
                            weight - this.learningRate *
                            (weight - this.C * example.label * example.features[j])
                    );

                    this.bias -= this.learningRate *
                        (-this.C * example.label);
                }
            }

            if (epoch % 500 === 0 || epoch === this.epochs) {
                this.emit('epoch', {
                    epoch,
                    objective: this.objective(dataset)
                });
            }
        }

        this.emit('trained', {
            weights: [...this.weights],
            bias: this.bias
        });

        return this;
    }

    decisionFunction(features) {
        validateFeatures(features);

        if (this.weights.length !== features.length) {
            throw new ValidationError('Prediction dimension does not match model.');
        }

        return dot(this.weights, features) + this.bias;
    }

    predict(features) {
        return this.decisionFunction(features) >= 0 ? 1 : -1;
    }

    functionalMargin(example) {
        return example.label * this.decisionFunction(example.features);
    }

    hingeLoss(example) {
        return Math.max(0, 1 - this.functionalMargin(example));
    }

    supportVectors(dataset, tolerance = 0.05) {
        if (tolerance < 0) {
            throw new ValidationError('Tolerance cannot be negative.');
        }

        return dataset.filter(
            example => this.functionalMargin(example) <= 1 + tolerance
        );
    }

    objective(dataset) {
        const regularization =
            0.5 * this.weights.reduce((sum, value) => sum + value * value, 0);

        const hinge =
            dataset.reduce((sum, example) => sum + this.hingeLoss(example), 0);

        return regularization + this.C * hinge;
    }

    accuracy(dataset) {
        const correct = dataset.reduce(
            (sum, example) =>
                sum + (this.predict(example.features) === example.label ? 1 : 0),
            0
        );

        return correct / dataset.length;
    }

    marginWidth() {
        const length = vectorNorm(this.weights);
        return length === 0 ? Infinity : 2 / length;
    }
}

class RepositoryMergeStream extends EventEmitter {
    /*
     * The stream represents candidate changes arriving asynchronously.
     * The SVM is used to classify whether a candidate lies on the
     * acceptable (+1) or remediation-required (-1) side of a learned
     * hyperplane. This is deliberately a classification use case rather
     * than treating SVM output as a merge authorization.
     */
    constructor(model, scaler) {
        super();
        this.model = model;
        this.scaler = scaler;
    }

    evaluate(candidate) {
        const features = candidate.metrics.map(
            (value, index) =>
                (value - this.scaler.means[index]) /
                this.scaler.standardDeviations[index]
        );

        const score = this.model.decisionFunction(features);
        const classification = this.model.predict(features);

        const result = {
            id: candidate.id,
            score,
            classification,
            recommendation:
                classification === 1
                    ? 'acceptable-risk-region'
                    : 'remediation-risk-region'
        };

        this.emit('classified', result);
        return result;
    }
}

function buildDataset() {
    return [
        new TrainingExample([0.4, 0.95], 1, 'PR-101'),
        new TrainingExample([0.7, 0.91], 1, 'PR-102'),
        new TrainingExample([1.0, 0.88], 1, 'PR-103'),
        new TrainingExample([1.2, 0.84], 1, 'PR-104'),
        new TrainingExample([1.5, 0.80], 1, 'PR-105'),
        new TrainingExample([1.8, 0.77], 1, 'PR-106'),
        new TrainingExample([2.0, 0.73], 1, 'PR-107'),
        new TrainingExample([2.4, 0.70], -1, 'PR-108'),
        new TrainingExample([2.8, 0.65], -1, 'PR-109'),
        new TrainingExample([3.0, 0.60], -1, 'PR-110'),
        new TrainingExample([3.4, 0.56], -1, 'PR-111'),
        new TrainingExample([3.8, 0.52], -1, 'PR-112'),
        new TrainingExample([4.1, 0.48], -1, 'PR-113'),
        new TrainingExample([4.5, 0.44], -1, 'PR-114'),
        new TrainingExample([2.1, 0.68], -1, 'PR-115'),
        new TrainingExample([1.7, 0.74], -1, 'PR-116')
    ];
}

function main() {
    const rawDataset = buildDataset();
    const scaler = standardize(rawDataset);
    const dataset = scaler.transformed;

    const model = new LinearSVM({
        C: 1,
        learningRate: 0.002,
        epochs: 5000
    });

    model.on('epoch', event => {
        if (event.epoch === 5000) {
            console.log(`Training objective: ${event.objective.toFixed(4)}`);
        }
    });

    model.on('trained', event => {
        console.log(
            'Model trained:',
            `weights=${event.weights.map(x => x.toFixed(4))}`,
            `bias=${event.bias.toFixed(4)}`
        );
    });

    model.fit(dataset);

    console.log(`Margin width: ${model.marginWidth().toFixed(4)}`);
    console.log(`Training accuracy: ${model.accuracy(dataset).toFixed(3)}`);

    const supportVectors = model.supportVectors(dataset);

    console.log('\nSupport-vector candidates:');
    for (const example of supportVectors) {
        console.log(
            `${example.id}: margin=${model.functionalMargin(example).toFixed(4)}, ` +
            `hinge=${model.hingeLoss(example).toFixed(4)}`
        );
    }

    const stream = new RepositoryMergeStream(model, scaler);

    stream.on('classified', result => {
        console.log(
            `${result.id}: score=${result.score.toFixed(4)} ` +
            `class=${result.classification} ` +
            `decision=${result.recommendation}`
        );
    });

    const incomingCandidates = [
        { id: 'PR-201', metrics: [1.1, 0.86] },
        { id: 'PR-202', metrics: [2.9, 0.62] },
        { id: 'PR-203', metrics: [4.3, 0.50] }
    ];

    console.log('\nAsynchronous candidate evaluation:');

    Promise.all(
        incomingCandidates.map((candidate, index) =>
            new Promise(resolve => {
                setTimeout(() => resolve(stream.evaluate(candidate)), index * 25);
            })
        )
    ).then(results => {
        console.log(`Evaluated ${results.length} candidates.`);
    }).catch(error => {
        console.error(`Evaluation failed: ${error.message}`);
        process.exitCode = 1;
    });
}

main();
