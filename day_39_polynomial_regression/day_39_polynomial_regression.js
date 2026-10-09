"use strict";

/*
 * Polynomial Regression: JavaScript Workflow Model
 *
 * This implementation focuses on JavaScript-specific behavior:
 * - objects and classes for model state
 * - event-driven workflow simulation
 * - polynomial feature construction
 * - matrix operations
 * - validation and custom errors
 * - asynchronous model-selection events
 * - cross-validation
 * - regularization
 * - merge-style event processing for a realistic analytics pipeline
 *
 * Run with:
 *   node polynomial-regression.js
 */

class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

function assertFiniteNumber(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ValidationError(`${name} must be a finite number.`);
    }
}

function mean(values) {
    if (!values.length) {
        throw new ValidationError("Cannot calculate a mean of an empty array.");
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function rmse(actual, predicted) {
    if (actual.length !== predicted.length || !actual.length) {
        throw new ValidationError("RMSE requires equally sized non-empty arrays.");
    }

    const squaredError = actual.reduce(
        (sum, value, index) => sum + (value - predicted[index]) ** 2,
        0
    );

    return Math.sqrt(squaredError / actual.length);
}

function mae(actual, predicted) {
    if (actual.length !== predicted.length || !actual.length) {
        throw new ValidationError("MAE requires equally sized non-empty arrays.");
    }

    return actual.reduce(
        (sum, value, index) => sum + Math.abs(value - predicted[index]),
        0
    ) / actual.length;
}

function rSquared(actual, predicted) {
    const actualMean = mean(actual);

    const total = actual.reduce(
        (sum, value) => sum + (value - actualMean) ** 2,
        0
    );

    const residual = actual.reduce(
        (sum, value, index) => sum + (value - predicted[index]) ** 2,
        0
    );

    return total === 0 ? (residual === 0 ? 1 : 0) : 1 - residual / total;
}

function transpose(matrix) {
    if (!matrix.length) {
        throw new ValidationError("Cannot transpose an empty matrix.");
    }

    const width = matrix[0].length;

    if (matrix.some(row => row.length !== width)) {
        throw new ValidationError("Matrix rows must have equal lengths.");
    }

    return Array.from(
        { length: width },
        (_, column) => matrix.map(row => row[column])
    );
}

function matrixMultiply(a, b) {
    if (!a.length || !b.length) {
        throw new ValidationError("Matrices cannot be empty.");
    }

    if (a[0].length !== b.length) {
        throw new ValidationError("Matrix dimensions do not match.");
    }

    const bTransposed = transpose(b);

    return a.map(row =>
        bTransposed.map(column =>
            row.reduce((sum, value, index) => sum + value * column[index], 0)
        )
    );
}

function identity(size) {
    return Array.from(
        { length: size },
        (_, row) =>
            Array.from(
                { length: size },
                (_, column) => row === column ? 1 : 0
            )
    );
}

function inverse(matrix) {
    const n = matrix.length;

    if (!n || matrix.some(row => row.length !== n)) {
        throw new ValidationError("Only non-empty square matrices can be inverted.");
    }

    const augmented = matrix.map((row, index) => [
        ...row.map(Number),
        ...identity(n)[index]
    ]);

    for (let column = 0; column < n; column += 1) {
        let pivotRow = column;

        for (let row = column + 1; row < n; row += 1) {
            if (
                Math.abs(augmented[row][column]) >
                Math.abs(augmented[pivotRow][column])
            ) {
                pivotRow = row;
            }
        }

        if (Math.abs(augmented[pivotRow][column]) < 1e-12) {
            throw new ValidationError(
                "Matrix is singular or numerically unstable."
            );
        }

        [augmented[column], augmented[pivotRow]] =
            [augmented[pivotRow], augmented[column]];

        const pivot = augmented[column][column];

        for (let index = 0; index < augmented[column].length; index += 1) {
            augmented[column][index] /= pivot;
        }

        for (let row = 0; row < n; row += 1) {
            if (row === column) {
                continue;
            }

            const factor = augmented[row][column];

            for (let index = 0; index < augmented[row].length; index += 1) {
                augmented[row][index] -= factor * augmented[column][index];
            }
        }
    }

    return augmented.map(row => row.slice(n));
}

function multiplyMatrixVector(matrix, vector) {
    return matrix.map(row =>
        row.reduce((sum, value, index) => sum + value * vector[index], 0)
    );
}

class PolynomialFeatures {
    constructor(degree) {
        if (!Number.isInteger(degree) || degree < 0) {
            throw new ValidationError("Polynomial degree must be a non-negative integer.");
        }

        this.degree = degree;
    }

    transform(values) {
        if (!values.length) {
            throw new ValidationError("Cannot transform an empty dataset.");
        }

        return values.map(value => {
            assertFiniteNumber(value, "Feature value");

            return Array.from(
                { length: this.degree + 1 },
                (_, power) => value ** power
            );
        });
    }
}

class StandardScaler {
    constructor() {
        this.mean = null;
        this.scale = null;
    }

    fit(values) {
        if (!values.length) {
            throw new ValidationError("Scaler requires data.");
        }

        this.mean = mean(values);

        const variance = mean(
            values.map(value => (value - this.mean) ** 2)
        );

        this.scale = Math.sqrt(variance);

        if (this.scale < 1e-12) {
            throw new ValidationError("A constant feature cannot be standardized.");
        }

        return this;
    }

    transform(values) {
        if (this.mean === null || this.scale === null) {
            throw new ValidationError("Scaler has not been fitted.");
        }

        return values.map(value => (value - this.mean) / this.scale);
    }

    fitTransform(values) {
        return this.fit(values).transform(values);
    }
}

class PolynomialRegression {
    constructor({
        degree,
        regularization = 0,
        scale = true
    }) {
        if (regularization < 0) {
            throw new ValidationError("Regularization cannot be negative.");
        }

        this.degree = degree;
        this.regularization = regularization;
        this.scale = scale;
        this.features = new PolynomialFeatures(degree);
        this.scaler = scale ? new StandardScaler() : null;
        this.coefficients = null;
    }

    fit(x, y) {
        if (x.length !== y.length || x.length < this.degree + 1) {
            throw new ValidationError(
                "Training data is too small or x/y lengths differ."
            );
        }

        let transformedX = [...x];

        if (this.scaler) {
            transformedX = this.scaler.fitTransform(transformedX);
        }

        const design = this.features.transform(transformedX);
        const xT = transpose(design);
        const normalMatrix = matrixMultiply(xT, design);

        for (let index = 1; index < normalMatrix.length; index += 1) {
            normalMatrix[index][index] += this.regularization;
        }

        const target = y.map(value => [value]);
        const rightSide = matrixMultiply(xT, target);
        const inverseMatrix = inverse(normalMatrix);

        this.coefficients = multiplyMatrixVector(
            inverseMatrix,
            rightSide.map(row => row[0])
        );

        return this;
    }

    predict(x) {
        if (!this.coefficients) {
            throw new ValidationError("The model must be fitted before prediction.");
        }

        let values = [...x];

        if (this.scaler) {
            values = this.scaler.transform(values);
        }

        return this.features
            .transform(values)
            .map(row =>
                row.reduce(
                    (sum, value, index) =>
                        sum + value * this.coefficients[index],
                    0
                )
            );
    }

    evaluate(x, y) {
        const predictions = this.predict(x);

        return {
            mae: mae(y, predictions),
            rmse: rmse(y, predictions),
            r2: rSquared(y, predictions)
        };
    }
}

function generateEnergyDemand(count = 120, seed = 17) {
    let state = seed >>> 0;

    const random = () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };

    const gaussianNoise = () => {
        const u1 = Math.max(random(), Number.MIN_VALUE);
        const u2 = random();

        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    const temperature = [];
    const demand = [];

    for (let index = 0; index < count; index += 1) {
        const value = -5 + (45 * index) / (count - 1);

        const target =
            420
            - 13 * value
            + 0.72 * value ** 2
            - 0.012 * value ** 3
            + gaussianNoise() * 18;

        temperature.push(value);
        demand.push(target);
    }

    return { temperature, demand };
}

function shuffledIndices(size, seed = 41) {
    const indices = Array.from({ length: size }, (_, index) => index);
    let state = seed >>> 0;

    const random = () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 4294967296;
    };

    for (let index = indices.length - 1; index > 0; index -= 1) {
        const swapIndex = Math.floor(random() * (index + 1));
        [indices[index], indices[swapIndex]] =
            [indices[swapIndex], indices[index]];
    }

    return indices;
}

function splitData(x, y, testRatio = 0.2) {
    const indices = shuffledIndices(x.length);

    const testSize = Math.max(1, Math.round(x.length * testRatio));
    const testSet = new Set(indices.slice(0, testSize));

    const trainX = [];
    const trainY = [];
    const testX = [];
    const testY = [];

    for (let index = 0; index < x.length; index += 1) {
        if (testSet.has(index)) {
            testX.push(x[index]);
            testY.push(y[index]);
        } else {
            trainX.push(x[index]);
            trainY.push(y[index]);
        }
    }

    return { trainX, trainY, testX, testY };
}

function createFolds(size, foldCount = 5) {
    const indices = shuffledIndices(size, 99);
    const folds = Array.from({ length: foldCount }, () => []);

    indices.forEach((index, position) => {
        folds[position % foldCount].push(index);
    });

    return folds;
}

function crossValidate(x, y, degree, regularization = 0.1, folds = 5) {
    const partitions = createFolds(x.length, folds);
    const scores = [];

    for (const validationIndices of partitions) {
        const validationSet = new Set(validationIndices);

        const trainIndices = Array.from(
            { length: x.length },
            (_, index) => index
        ).filter(index => !validationSet.has(index));

        const trainX = trainIndices.map(index => x[index]);
        const trainY = trainIndices.map(index => y[index]);
        const validationX = validationIndices.map(index => x[index]);
        const validationY = validationIndices.map(index => y[index]);

        const model = new PolynomialRegression({
            degree,
            regularization,
            scale: true
        });

        model.fit(trainX, trainY);
        scores.push(model.evaluate(validationX, validationY).rmse);
    }

    return mean(scores);
}

class ModelSelectionPipeline {
    constructor() {
        this.listeners = new Map();
    }

    on(eventName, listener) {
        if (!this.listeners.has(eventName)) {
            this.listeners.set(eventName, []);
        }

        this.listeners.get(eventName).push(listener);
    }

    emit(eventName, payload) {
        const listeners = this.listeners.get(eventName) || [];

        for (const listener of listeners) {
            listener(payload);
        }
    }

    async selectBestDegree(x, y) {
        this.emit("selectionStarted", {
            candidateDegrees: [1, 2, 3, 4, 5, 6]
        });

        const results = [];

        for (const degree of [1, 2, 3, 4, 5, 6]) {
            await new Promise(resolve => setImmediate(resolve));

            const score = crossValidate(x, y, degree, 0.1, 5);

            const result = { degree, cvRmse: score };
            results.push(result);

            this.emit("candidateEvaluated", result);
        }

        const winner = results.reduce(
            (best, current) =>
                current.cvRmse < best.cvRmse ? current : best
        );

        this.emit("selectionCompleted", winner);

        return winner;
    }
}

async function runPipeline() {
    console.log("=".repeat(78));
    console.log("Polynomial Regression: JavaScript Workflow");
    console.log("=".repeat(78));

    const { temperature, demand } = generateEnergyDemand();

    const {
        trainX,
        trainY,
        testX,
        testY
    } = splitData(temperature, demand);

    const pipeline = new ModelSelectionPipeline();

    pipeline.on("selectionStarted", event => {
        console.log(
            `\nModel selection started for degrees: ${event.candidateDegrees.join(", ")}`
        );
    });

    pipeline.on("candidateEvaluated", event => {
        console.log(
            `Degree ${event.degree}: CV RMSE=${event.cvRmse.toFixed(4)}`
        );
    });

    pipeline.on("selectionCompleted", event => {
        console.log(
            `Selected degree ${event.degree} with CV RMSE=${event.cvRmse.toFixed(4)}`
        );
    });

    const selected = await pipeline.selectBestDegree(trainX, trainY);

    const finalModel = new PolynomialRegression({
        degree: selected.degree,
        regularization: 0.1,
        scale: true
    });

    finalModel.fit(trainX, trainY);

    const trainMetrics = finalModel.evaluate(trainX, trainY);
    const testMetrics = finalModel.evaluate(testX, testY);

    console.log("\nFinal model:");
    console.log(`Degree: ${selected.degree}`);
    console.log(`Training MAE: ${trainMetrics.mae.toFixed(4)}`);
    console.log(`Training RMSE: ${trainMetrics.rmse.toFixed(4)}`);
    console.log(`Training R²: ${trainMetrics.r2.toFixed(4)}`);
    console.log(`Testing MAE: ${testMetrics.mae.toFixed(4)}`);
    console.log(`Testing RMSE: ${testMetrics.rmse.toFixed(4)}`);
    console.log(`Testing R²: ${testMetrics.r2.toFixed(4)}`);

    console.log("\nOperational predictions:");

    for (const value of [-5, 5, 15, 25, 35, 40]) {
        const prediction = finalModel.predict([value])[0];

        console.log(
            `${value.toString().padStart(5)} °C -> ` +
            `${prediction.toFixed(2).padStart(10)} demand`
        );
    }

    console.log("\nOverfitting comparison:");

    for (const degree of [1, 2, 5, 10]) {
        const model = new PolynomialRegression({
            degree,
            regularization: 0.001,
            scale: true
        });

        model.fit(trainX, trainY);

        const trainScore = model.evaluate(trainX, trainY);
        const testScore = model.evaluate(testX, testY);

        console.log(
            `Degree ${degree.toString().padStart(2)} | ` +
            `train RMSE=${trainScore.rmse.toFixed(3)} | ` +
            `test RMSE=${testScore.rmse.toFixed(3)}`
        );
    }

    console.log("\nExtrapolation warning:");

    for (const value of [40, 50, 70]) {
        const prediction = finalModel.predict([value])[0];

        console.log(
            `${value.toString().padStart(5)} °C -> ` +
            `${prediction.toFixed(2).padStart(10)} demand`
        );
    }
}

runPipeline().catch(error => {
    console.error(`Pipeline failed: ${error.name}: ${error.message}`);
    process.exitCode = 1;
});
