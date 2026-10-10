'use strict';

/*
 * Logistic Regression in JavaScript.
 *
 * This implementation models a loan-risk classification workflow.
 * It emphasizes JavaScript-specific features:
 * - immutable-style data transformation
 * - event-driven training progress
 * - asynchronous prediction service
 * - validation
 * - probability thresholds
 * - logits and sigmoid
 * - model serialization
 * - decision-boundary interpretation
 *
 * Runtime: Node.js 18+
 */

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ValidationError';
    }
}

function sigmoid(logit) {
    // The branch avoids Math.exp() overflow for very negative logits.
    if (logit >= 0) {
        const negativeExponent = Math.exp(-logit);
        return 1 / (1 + negativeExponent);
    }

    const positiveExponent = Math.exp(logit);
    return positiveExponent / (1 + positiveExponent);
}

function dotProduct(left, right) {
    if (left.length !== right.length) {
        throw new ValidationError('Vector dimensions do not match.');
    }

    return left.reduce((sum, value, index) => {
        return sum + value * right[index];
    }, 0);
}

function validateDataset(features, labels) {
    if (!Array.isArray(features) || features.length === 0) {
        throw new ValidationError('Features must be a non-empty array.');
    }

    if (!Array.isArray(labels) || labels.length !== features.length) {
        throw new ValidationError(
            'Labels must contain exactly one value per feature row.'
        );
    }

    const width = features[0].length;

    if (width === 0) {
        throw new ValidationError('Feature rows cannot be empty.');
    }

    for (const row of features) {
        if (!Array.isArray(row) || row.length !== width) {
            throw new ValidationError(
                'All feature rows must have the same dimension.'
            );
        }

        if (row.some(value => !Number.isFinite(value))) {
            throw new ValidationError('Features must contain finite numbers.');
        }
    }

    if (labels.some(label => label !== 0 && label !== 1)) {
        throw new ValidationError('Labels must be binary 0 or 1.');
    }
}

class StandardScaler {
    constructor() {
        this.means = null;
        this.scales = null;
    }

    fit(features) {
        validateDataset(features, features.map(() => 0));

        const width = features[0].length;

        this.means = Array.from({ length: width }, (_, column) => {
            return features.reduce((sum, row) => sum + row[column], 0)
                / features.length;
        });

        this.scales = Array.from({ length: width }, (_, column) => {
            const variance = features.reduce((sum, row) => {
                const difference = row[column] - this.means[column];
                return sum + difference * difference;
            }, 0) / features.length;

            const standardDeviation = Math.sqrt(variance);

            // Constant features receive a unit scale so transformation
            // remains defined without silently dropping the feature.
            return standardDeviation > Number.EPSILON
                ? standardDeviation
                : 1;
        });

        return this;
    }

    transform(features) {
        if (!this.means || !this.scales) {
            throw new Error('StandardScaler must be fitted first.');
        }

        return features.map(row => {
            if (row.length !== this.means.length) {
                throw new ValidationError(
                    'Feature dimension differs from fitted data.'
                );
            }

            return row.map((value, column) => {
                return (value - this.means[column]) / this.scales[column];
            });
        });
    }

    fitTransform(features) {
        return this.fit(features).transform(features);
    }
}

class LogisticRegression {
    constructor({
        learningRate = 0.05,
        epochs = 2000,
        regularization = 0.1,
        threshold = 0.5
    } = {}) {
        if (learningRate <= 0 || !Number.isFinite(learningRate)) {
            throw new ValidationError('learningRate must be positive.');
        }

        if (!Number.isInteger(epochs) || epochs <= 0) {
            throw new ValidationError('epochs must be a positive integer.');
        }

        if (regularization < 0 || !Number.isFinite(regularization)) {
            throw new ValidationError(
                'regularization must be non-negative.'
            );
        }

        if (threshold <= 0 || threshold >= 1) {
            throw new ValidationError(
                'threshold must be strictly between 0 and 1.'
            );
        }

        this.learningRate = learningRate;
        this.epochs = epochs;
        this.regularization = regularization;
        this.threshold = threshold;
        this.weights = [];
        this.bias = 0;
    }

    logit(row) {
        if (this.weights.length === 0) {
            throw new Error('Model has not been trained.');
        }

        return dotProduct(this.weights, row) + this.bias;
    }

    probability(row) {
        return sigmoid(this.logit(row));
    }

    fit(features, labels, onEpoch = null) {
        validateDataset(features, labels);

        const sampleCount = features.length;
        const featureCount = features[0].length;

        this.weights = Array(featureCount).fill(0);
        this.bias = 0;

        for (let epoch = 1; epoch <= this.epochs; epoch += 1) {
            const gradients = Array(featureCount).fill(0);
            let biasGradient = 0;

            features.forEach((row, index) => {
                const probability = sigmoid(
                    dotProduct(this.weights, row) + this.bias
                );

                const error = probability - labels[index];

                row.forEach((value, featureIndex) => {
                    gradients[featureIndex] += error * value;
                });

                biasGradient += error;
            });

            this.weights = this.weights.map((weight, index) => {
                const dataGradient = gradients[index] / sampleCount;
                const penalty = (
                    this.regularization * weight / sampleCount
                );

                return weight - this.learningRate * (dataGradient + penalty);
            });

            this.bias -= (
                this.learningRate * biasGradient / sampleCount
            );

            if (onEpoch && (
                epoch === 1 ||
                epoch % 250 === 0 ||
                epoch === this.epochs
            )) {
                onEpoch({
                    epoch,
                    loss: this.loss(features, labels),
                    accuracy: this.accuracy(features, labels)
                });
            }
        }

        return this;
    }

    loss(features, labels) {
        validateDataset(features, labels);

        const total = features.reduce((sum, row, index) => {
            const probability = Math.min(
                Math.max(this.probability(row), 1e-15),
                1 - 1e-15
            );

            return sum - (
                labels[index] * Math.log(probability) +
                (1 - labels[index]) * Math.log(1 - probability)
            );
        }, 0);

        const penalty = (
            this.regularization *
            this.weights.reduce((sum, weight) => sum + weight * weight, 0)
            / 2
        );

        return (total / features.length) + (
            penalty / features.length
        );
    }

    predictProbabilities(features) {
        return features.map(row => this.probability(row));
    }

    predict(features) {
        return this.predictProbabilities(features).map(probability => {
            return probability >= this.threshold ? 1 : 0;
        });
    }

    accuracy(features, labels) {
        const predictions = this.predict(features);

        return predictions.reduce((correct, prediction, index) => {
            return correct + (prediction === labels[index] ? 1 : 0);
        }, 0) / labels.length;
    }

    decisionBoundary() {
        if (this.weights.length !== 2) {
            throw new Error(
                'Decision-boundary equation requires exactly two features.'
            );
        }

        const [weightX, weightY] = this.weights;

        if (Math.abs(weightY) < 1e-12) {
            return `x = ${(-this.bias / weightX).toFixed(4)}`;
        }

        const slope = -weightX / weightY;
        const intercept = -this.bias / weightY;

        return `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`;
    }

    serialize() {
        return JSON.stringify({
            weights: this.weights,
            bias: this.bias,
            threshold: this.threshold
        });
    }

    static fromSerialized(serialized) {
        const parsed = JSON.parse(serialized);

        if (
            !Array.isArray(parsed.weights) ||
            !Number.isFinite(parsed.bias) ||
            parsed.threshold <= 0 ||
            parsed.threshold >= 1
        ) {
            throw new ValidationError('Invalid serialized model.');
        }

        const model = new LogisticRegression({
            threshold: parsed.threshold
        });

        model.weights = parsed.weights;
        model.bias = parsed.bias;

        return model;
    }
}

function generateLoanDataset(seed = 31, count = 220) {
    /*
     * A deterministic pseudo-random generator keeps this example reproducible.
     * The features represent debt-to-income ratio and annual income in thousands.
     */
    let state = seed;

    function random() {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    }

    const features = [];
    const labels = [];

    for (let i = 0; i < count; i += 1) {
        const debtRatio = 0.05 + random() * 0.75;
        const income = 25 + random() * 175;

        const latentScore =
            5.0 * debtRatio -
            0.018 * income +
            0.2 +
            (random() - 0.5) * 1.2;

        features.push([debtRatio, income]);
        labels.push(latentScore > 0.0 ? 1 : 0);
    }

    return { features, labels };
}

function splitDataset(features, labels, testRatio = 0.25) {
    if (testRatio <= 0 || testRatio >= 1) {
        throw new ValidationError('testRatio must be between 0 and 1.');
    }

    const indices = features.map((_, index) => index);

    // Fisher-Yates produces a real shuffled partition instead of selecting
    // every nth record, which can preserve unwanted ordering patterns.
    for (let i = indices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const testSize = Math.max(1, Math.floor(features.length * testRatio));
    const testIndices = indices.slice(0, testSize);
    const trainIndices = indices.slice(testSize);

    return {
        trainFeatures: trainIndices.map(index => features[index]),
        trainLabels: trainIndices.map(index => labels[index]),
        testFeatures: testIndices.map(index => features[index]),
        testLabels: testIndices.map(index => labels[index])
    };
}

function classificationMetrics(actual, predicted) {
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;

    actual.forEach((label, index) => {
        const prediction = predicted[index];

        if (label === 1 && prediction === 1) tp += 1;
        else if (label === 0 && prediction === 0) tn += 1;
        else if (label === 0 && prediction === 1) fp += 1;
        else if (label === 1 && prediction === 0) fn += 1;
    });

    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0
        ? 0
        : 2 * precision * recall / (precision + recall);

    return {
        truePositive: tp,
        trueNegative: tn,
        falsePositive: fp,
        falseNegative: fn,
        precision,
        recall,
        f1,
        accuracy: (tp + tn) / actual.length
    };
}

class LoanRiskService {
    constructor(model, scaler) {
        this.model = model;
        this.scaler = scaler;
    }

    async evaluate(application) {
        await new Promise(resolve => setImmediate(resolve));

        if (
            !Number.isFinite(application.debtRatio) ||
            !Number.isFinite(application.annualIncomeThousands)
        ) {
            throw new ValidationError(
                'Loan application contains invalid numeric fields.'
            );
        }

        if (
            application.debtRatio < 0 ||
            application.debtRatio > 1 ||
            application.annualIncomeThousands <= 0
        ) {
            throw new ValidationError(
                'Loan application is outside accepted domain limits.'
            );
        }

        const rawFeatures = [[
            application.debtRatio,
            application.annualIncomeThousands
        ]];

        const scaledFeatures = this.scaler.transform(rawFeatures);
        const probability = this.model.predictProbabilities(scaledFeatures)[0];

        return {
            applicantId: application.applicantId,
            defaultProbability: probability,
            decision: probability >= this.model.threshold
                ? 'MANUAL_REVIEW'
                : 'LOW_RISK'
        };
    }
}

async function main() {
    console.log('=== Logistic Regression: Loan Risk Classification ===');

    console.log('\nSigmoid examples:');
    [-8, -2, 0, 2, 8].forEach(value => {
        console.log(
            `logit=${value.toString().padStart(4)} ` +
            `probability=${sigmoid(value).toFixed(6)}`
        );
    });

    const dataset = generateLoanDataset();

    // A deterministic split is not essential to the model itself, but the
    // training dataset is generated deterministically for reproducibility.
    const split = splitDataset(
        dataset.features,
        dataset.labels
    );

    const scaler = new StandardScaler();
    const trainFeatures = scaler.fitTransform(split.trainFeatures);
    const testFeatures = scaler.transform(split.testFeatures);

    const model = new LogisticRegression({
        learningRate: 0.08,
        epochs: 2200,
        regularization: 0.15,
        threshold: 0.5
    });

    model.fit(
        trainFeatures,
        split.trainLabels,
        progress => {
            console.log(
                `epoch=${progress.epoch} ` +
                `loss=${progress.loss.toFixed(5)} ` +
                `accuracy=${progress.accuracy.toFixed(4)}`
            );
        }
    );

    const predictions = model.predict(testFeatures);
    const metrics = classificationMetrics(
        split.testLabels,
        predictions
    );

    console.log('\n=== Evaluation ===');
    console.log(`Accuracy : ${metrics.accuracy.toFixed(4)}`);
    console.log(`Precision: ${metrics.precision.toFixed(4)}`);
    console.log(`Recall   : ${metrics.recall.toFixed(4)}`);
    console.log(`F1       : ${metrics.f1.toFixed(4)}`);
    console.log(`Boundary : ${model.decisionBoundary()}`);

    console.log('\n=== Async Risk Evaluation ===');

    const service = new LoanRiskService(model, scaler);

    const applications = [
        {
            applicantId: 'APP-1042',
            debtRatio: 0.72,
            annualIncomeThousands: 42
        },
        {
            applicantId: 'APP-2091',
            debtRatio: 0.18,
            annualIncomeThousands: 135
        },
        {
            applicantId: 'APP-7714',
            debtRatio: 0.51,
            annualIncomeThousands: 78
        }
    ];

    for (const application of applications) {
        const result = await service.evaluate(application);

        console.log(
            `${result.applicantId}: ` +
            `P(default)=${result.defaultProbability.toFixed(4)} ` +
            `decision=${result.decision}`
        );
    }

    console.log('\n=== Serialized Model ===');
    const serialized = model.serialize();
    console.log(serialized);

    const restored = LogisticRegression.fromSerialized(serialized);
    const sample = scaler.transform([[0.30, 110]]);
    console.log(
        `Restored model probability: ` +
        `${restored.predictProbabilities(sample)[0].toFixed(4)}`
    );

    console.log('\n=== Threshold Sensitivity ===');

    const probabilities = model.predictProbabilities(testFeatures);

    [0.30, 0.50, 0.70, 0.90].forEach(threshold => {
        const thresholdPredictions = probabilities.map(probability => {
            return probability >= threshold ? 1 : 0;
        });

        const result = classificationMetrics(
            split.testLabels,
            thresholdPredictions
        );

        console.log(
            `threshold=${threshold.toFixed(2)} ` +
            `precision=${result.precision.toFixed(3)} ` +
            `recall=${result.recall.toFixed(3)} ` +
            `f1=${result.f1.toFixed(3)}`
        );
    });

    try {
        await service.evaluate({
            applicantId: 'INVALID',
            debtRatio: 1.4,
            annualIncomeThousands: 50
        });
    } catch (error) {
        console.log(
            `\nValidation failure handled safely: ${error.message}`
        );
    }
}

main().catch(error => {
    console.error(`Application failure: ${error.message}`);
    process.exitCode = 1;
});
