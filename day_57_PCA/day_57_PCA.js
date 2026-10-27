'use strict';

/*
 * Principal Component Analysis using covariance, power iteration,
 * eigenvector deflation, and explained-variance analysis.
 *
 * This implementation intentionally uses Node.js standard-library features
 * only. It is suitable for small and medium educational datasets where
 * transparent numerical operations are more important than optimized
 * linear-algebra kernels.
 */

class PCAError extends Error {
    constructor(message) {
        super(message);
        this.name = 'PCAError';
    }
}

function validateMatrix(data) {
    if (!Array.isArray(data) || data.length < 2) {
        throw new PCAError('At least two observations are required.');
    }

    if (!Array.isArray(data[0]) || data[0].length === 0) {
        throw new PCAError('The dataset must contain features.');
    }

    const width = data[0].length;

    data.forEach((row, rowIndex) => {
        if (!Array.isArray(row) || row.length !== width) {
            throw new PCAError(`Invalid feature count at row ${rowIndex + 1}.`);
        }

        row.forEach((value) => {
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                throw new PCAError('Every feature value must be finite.');
            }
        });
    });
}

function meanCenter(data) {
    validateMatrix(data);

    const dimensions = data[0].length;
    const means = Array(dimensions).fill(0);

    for (const row of data) {
        for (let j = 0; j < dimensions; j++) {
            means[j] += row[j];
        }
    }

    for (let j = 0; j < dimensions; j++) {
        means[j] /= data.length;
    }

    const centered = data.map((row) =>
        row.map((value, j) => value - means[j])
    );

    return { means, centered };
}

function dot(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

function norm(vector) {
    return Math.sqrt(dot(vector, vector));
}

function normalize(vector) {
    const length = norm(vector);
    if (length < 1e-14) {
        throw new PCAError('Cannot normalize a zero vector.');
    }
    return vector.map((value) => value / length);
}

function matrixVectorMultiply(matrix, vector) {
    return matrix.map((row) => dot(row, vector));
}

function covarianceMatrix(centered) {
    const n = centered.length;
    const dimensions = centered[0].length;
    const covariance = Array.from(
        { length: dimensions },
        () => Array(dimensions).fill(0)
    );

    for (let i = 0; i < dimensions; i++) {
        for (let j = i; j < dimensions; j++) {
            let sum = 0;
            for (const row of centered) {
                sum += row[i] * row[j];
            }

            const value = sum / (n - 1);
            covariance[i][j] = value;
            covariance[j][i] = value;
        }
    }

    return covariance;
}

function powerIteration(matrix, maxIterations = 10000, tolerance = 1e-12) {
    const n = matrix.length;
    let vector = normalize(
        Array.from({ length: n }, (_, i) => 1 + i / n)
    );

    let eigenvalue = 0;

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const candidate = normalize(matrixVectorMultiply(matrix, vector));
        const newEigenvalue = dot(
            candidate,
            matrixVectorMultiply(matrix, candidate)
        );

        const difference = norm(
            candidate.map((value, i) => value - vector[i])
        );

        vector = candidate;

        if (
            Math.abs(newEigenvalue - eigenvalue) < tolerance &&
            difference < tolerance
        ) {
            eigenvalue = newEigenvalue;
            break;
        }

        eigenvalue = newEigenvalue;

        if (iteration === maxIterations - 1) {
            throw new PCAError('Power iteration did not converge.');
        }
    }

    const pivot = vector.reduce(
        (best, value, index) =>
            Math.abs(value) > Math.abs(vector[best]) ? index : best,
        0
    );

    if (vector[pivot] < 0) {
        vector = vector.map((value) => -value);
    }

    return { eigenvalue, eigenvector: vector };
}

function deflate(matrix, eigenvalue, eigenvector) {
    return matrix.map((row, i) =>
        row.map(
            (value, j) =>
                value - eigenvalue * eigenvector[i] * eigenvector[j]
        )
    );
}

function eigenDecomposition(matrix) {
    let working = matrix.map((row) => row.slice());
    const eigenvalues = [];
    const eigenvectors = [];

    for (let i = 0; i < matrix.length; i++) {
        const result = powerIteration(working);
        let value = result.eigenvalue;

        if (Math.abs(value) < 1e-12) {
            value = 0;
        }

        eigenvalues.push(value);
        eigenvectors.push(result.eigenvector);

        working = deflate(working, value, result.eigenvector);
    }

    const order = eigenvalues
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value);

    return {
        eigenvalues: order.map((entry) => entry.value),
        eigenvectors: order.map((entry) => eigenvectors[entry.index])
    };
}

function project(centered, eigenvectors, components) {
    return centered.map((row) =>
        eigenvectors
            .slice(0, components)
            .map((vector) => dot(row, vector))
    );
}

function fitPCA(data, components = null) {
    const { means, centered } = meanCenter(data);
    const covariance = covarianceMatrix(centered);
    const decomposition = eigenDecomposition(covariance);

    const dimensions = covariance.length;
    const count = components ?? dimensions;

    if (!Number.isInteger(count) || count < 1 || count > dimensions) {
        throw new PCAError('Invalid component count.');
    }

    const positiveEigenvalues = decomposition.eigenvalues.map(
        (value) => Math.max(value, 0)
    );

    const totalVariance = positiveEigenvalues.reduce(
        (sum, value) => sum + value,
        0
    );

    if (totalVariance <= 1e-14) {
        throw new PCAError('The dataset has no measurable variance.');
    }

    const explainedVarianceRatio = positiveEigenvalues.map(
        (value) => value / totalVariance
    );

    const cumulativeExplainedVariance = [];
    let cumulative = 0;

    for (const ratio of explainedVarianceRatio) {
        cumulative += ratio;
        cumulativeExplainedVariance.push(cumulative);
    }

    return {
        means,
        centered,
        covariance,
        eigenvalues: decomposition.eigenvalues,
        eigenvectors: decomposition.eigenvectors,
        explainedVarianceRatio,
        cumulativeExplainedVariance,
        transformed: project(
            centered,
            decomposition.eigenvectors,
            count
        ),
        components: count
    };
}

function chooseComponents(result, threshold = 0.95) {
    if (!(threshold > 0 && threshold <= 1)) {
        throw new PCAError('Threshold must be in (0, 1].');
    }

    for (let i = 0; i < result.cumulativeExplainedVariance.length; i++) {
        if (result.cumulativeExplainedVariance[i] >= threshold) {
            return i + 1;
        }
    }

    return result.eigenvalues.length;
}

function reconstruct(result) {
    return result.transformed.map((scores) => {
        const original = result.means.slice();

        for (let component = 0; component < result.components; component++) {
            for (let feature = 0; feature < result.means.length; feature++) {
                original[feature] +=
                    scores[component] *
                    result.eigenvectors[component][feature];
            }
        }

        return original;
    });
}

function rmse(original, reconstructed) {
    let sum = 0;
    let count = 0;

    for (let i = 0; i < original.length; i++) {
        for (let j = 0; j < original[i].length; j++) {
            const difference = original[i][j] - reconstructed[i][j];
            sum += difference * difference;
            count++;
        }
    }

    return Math.sqrt(sum / count);
}

function printMatrix(title, matrix) {
    console.log(`\n${title}`);
    for (const row of matrix) {
        console.log(
            '  ' +
            row.map((value) => value.toFixed(5).padStart(10)).join(' ')
        );
    }
}

function emitPCAReport(result) {
    printMatrix('Covariance matrix', result.covariance);

    console.log('\nPrincipal components');
    result.eigenvalues.forEach((value, index) => {
        console.log(
            `  PC${index + 1}: eigenvalue=${value.toFixed(6)}, ` +
            `explained=${(result.explainedVarianceRatio[index] * 100).toFixed(2)}%, ` +
            `cumulative=${(result.cumulativeExplainedVariance[index] * 100).toFixed(2)}%`
        );
    });

    console.log('\nEigenvectors');
    result.eigenvectors.forEach((vector, index) => {
        console.log(
            `  PC${index + 1}: ` +
            vector.map((value) => value.toFixed(6)).join(' ')
        );
    });

    printMatrix('Projected observations', result.transformed);
}

async function processDataset(name, data) {
    // Promise-based orchestration makes the workflow easy to connect to
    // asynchronous data ingestion without introducing external packages.
    return Promise.resolve().then(() => {
        const result = fitPCA(data);
        return { name, result };
    });
}

async function main() {
    const measurements = [
        [170, 65, 5.2, 12],
        [172, 67, 5.4, 12.5],
        [168, 61, 5.0, 11],
        [175, 72, 5.8, 14],
        [178, 76, 6.1, 15],
        [180, 79, 6.3, 16],
        [169, 63, 5.1, 11.5],
        [174, 70, 5.7, 13.5]
    ];

    console.log('PCA covariance and eigen-analysis');
    console.log('=================================');

    const { result } = await processDataset(
        'physical-measurements',
        measurements
    );

    emitPCAReport(result);

    const required = chooseComponents(result, 0.95);
    const reduced = fitPCA(measurements, required);
    const reconstructed = reconstruct(reduced);

    console.log(`\nComponents for 95% variance: ${required}`);
    console.log(`Reconstruction RMSE: ${rmse(measurements, reconstructed).toFixed(6)}`);

    const sum = result.explainedVarianceRatio.reduce(
        (accumulator, value) => accumulator + value,
        0
    );

    if (Math.abs(sum - 1) > 1e-9) {
        throw new PCAError('Explained variance ratios do not sum to one.');
    }

    try {
        fitPCA([[1, 1], [1, 1]]);
    } catch (error) {
        console.log(`\nExpected failure: ${error.message}`);
    }

    console.log('\nEngineering considerations');
    console.log('- Covariance PCA is sensitive to feature scale.');
    console.log('- Centering changes the coordinate origin but not pairwise differences.');
    console.log('- Eigenvector signs are arbitrary, so this implementation normalizes their sign.');
    console.log('- Power iteration is transparent but slower than optimized eigensolvers for large matrices.');
    console.log('- Input validation prevents NaN and Infinity from contaminating numerical state.');
}

main().catch((error) => {
    console.error(`PCA processing failed: ${error.message}`);
    process.exitCode = 1;
});
