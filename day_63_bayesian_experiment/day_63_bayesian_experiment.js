'use strict';

/*
 * Bayesian Optimization with:
 * - Gaussian Process surrogate
 * - Expected Improvement acquisition
 * - asynchronous objective evaluation
 * - event-driven optimization progress
 * - validation and bounded candidate search
 *
 * Node.js 18+ is sufficient; no npm packages are required.
 */

const fs = require('node:fs/promises');
const EventEmitter = require('node:events');

class Bounds {
    constructor(low, high) {
        if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
            throw new RangeError('Bounds must be finite and low < high.');
        }
        this.low = low;
        this.high = high;
    }

    sample(rng) {
        return this.low + (this.high - this.low) * rng();
    }
}

class SeededRandom {
    constructor(seed = 42) {
        this.state = seed >>> 0;
    }

    next() {
        this.state = (1664525 * this.state + 1013904223) >>> 0;
        return this.state / 4294967296;
    }
}

class GaussianProcess {
    constructor(lengthScale = 0.8, signalVariance = 1.0, noiseVariance = 1e-5) {
        if (lengthScale <= 0 || signalVariance <= 0 || noiseVariance < 0) {
            throw new RangeError('Invalid Gaussian Process parameters.');
        }

        this.lengthScale = lengthScale;
        this.signalVariance = signalVariance;
        this.noiseVariance = noiseVariance;
        this.x = [];
        this.alpha = [];
        this.L = [];
    }

    kernel(a, b) {
        let distance = 0;
        for (let i = 0; i < a.length; i++) {
            const delta = a[i] - b[i];
            distance += delta * delta;
        }
        return this.signalVariance *
            Math.exp(-0.5 * distance / (this.lengthScale ** 2));
    }

    static cholesky(matrix) {
        const n = matrix.length;
        const L = Array.from({ length: n }, () => Array(n).fill(0));

        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let value = matrix[i][j];

                for (let k = 0; k < j; k++) {
                    value -= L[i][k] * L[j][k];
                }

                if (i === j) {
                    if (value <= 1e-12) {
                        throw new Error('Kernel matrix is not positive definite.');
                    }
                    L[i][j] = Math.sqrt(value);
                } else {
                    L[i][j] = value / L[j][j];
                }
            }
        }

        return L;
    }

    static forward(L, b) {
        const result = [];

        for (let i = 0; i < b.length; i++) {
            let value = b[i];

            for (let j = 0; j < i; j++) {
                value -= L[i][j] * result[j];
            }

            result.push(value / L[i][i]);
        }

        return result;
    }

    static backward(L, b) {
        const result = Array(b.length).fill(0);

        for (let i = b.length - 1; i >= 0; i--) {
            let value = b[i];

            for (let j = i + 1; j < b.length; j++) {
                value -= L[j][i] * result[j];
            }

            result[i] = value / L[i][i];
        }

        return result;
    }

    fit(observations) {
        if (observations.length === 0) {
            throw new Error('At least one observation is required.');
        }

        this.x = observations.map(o => o.x);
        const y = observations.map(o => o.y);
        const n = y.length;

        const matrix = Array.from({ length: n }, (_, i) =>
            Array.from({ length: n }, (_, j) =>
                this.kernel(this.x[i], this.x[j]) +
                (i === j ? this.noiseVariance + 1e-9 : 0)
            )
        );

        this.L = GaussianProcess.cholesky(matrix);
        const forward = GaussianProcess.forward(this.L, y);
        this.alpha = GaussianProcess.backward(this.L, forward);
    }

    predict(point) {
        const k = this.x.map(trainPoint => this.kernel(point, trainPoint));
        const mean = k.reduce(
            (sum, value, index) => sum + value * this.alpha[index],
            0
        );

        const solved = GaussianProcess.forward(this.L, k);
        let variance = this.kernel(point, point);

        for (const value of solved) {
            variance -= value * value;
        }

        return {
            mean,
            standardDeviation: Math.sqrt(Math.max(0, variance))
        };
    }
}

function normalPdf(z) {
    return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

function normalCdf(z) {
    return 0.5 * (1 + erf(z / Math.sqrt(2)));
}

function erf(x) {
    const sign = x < 0 ? -1 : 1;
    const value = Math.abs(x);

    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const t = 1 / (1 + p * value);
    const polynomial =
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;

    return sign * (1 - polynomial * Math.exp(-value * value));
}

function expectedImprovement(mean, standardDeviation, best, exploration = 0.01) {
    if (standardDeviation <= 1e-12) {
        return 0;
    }

    const improvement = mean - best - exploration;
    const z = improvement / standardDeviation;

    return improvement * normalCdf(z) +
        standardDeviation * normalPdf(z);
}

class RepositoryLikeExperiment extends EventEmitter {
    constructor(bounds, objective, seed = 42) {
        super();

        this.bounds = bounds;
        this.objective = objective;
        this.random = new SeededRandom(seed);
        this.gp = new GaussianProcess();
        this.observations = [];
    }

    async evaluate(point) {
        // A Promise models a remote or expensive experiment.
        return new Promise(resolve => {
            setImmediate(() => {
                const value = this.objective(point);

                if (!Number.isFinite(value)) {
                    throw new Error('Objective returned a non-finite value.');
                }

                resolve(value);
            });
        });
    }

    async observe(point) {
        const value = await this.evaluate(point);
        const observation = {
            x: [...point],
            y: value
        };

        this.observations.push(observation);
        this.emit('observation', observation);

        return observation;
    }

    randomPoint() {
        return this.bounds.map(bound => bound.sample(() => this.random.next()));
    }

    candidates(pointsPerDimension) {
        if (pointsPerDimension < 2) {
            throw new RangeError('At least two grid points are required.');
        }

        const axes = this.bounds.map(bound => {
            const values = [];
            for (let i = 0; i < pointsPerDimension; i++) {
                values.push(
                    bound.low +
                    (bound.high - bound.low) *
                    i / (pointsPerDimension - 1)
                );
            }
            return values;
        });

        const result = [];

        const build = (prefix, dimension) => {
            if (dimension === axes.length) {
                result.push(prefix);
                return;
            }

            for (const value of axes[dimension]) {
                build([...prefix, value], dimension + 1);
            }
        };

        build([], 0);
        return result;
    }

    nextPoint(pointsPerDimension = 15) {
        this.gp.fit(this.observations);

        const best = Math.max(...this.observations.map(o => o.y));
        const observed = new Set(
            this.observations.map(o => o.x.map(v => v.toFixed(10)).join(','))
        );

        let bestPoint = null;
        let bestScore = -Infinity;

        for (const candidate of this.candidates(pointsPerDimension)) {
            const key = candidate.map(v => v.toFixed(10)).join(',');

            if (observed.has(key)) {
                continue;
            }

            const prediction = this.gp.predict(candidate);
            const score = expectedImprovement(
                prediction.mean,
                prediction.standardDeviation,
                best
            );

            if (score > bestScore) {
                bestScore = score;
                bestPoint = candidate;
            }
        }

        if (!bestPoint) {
            throw new Error('No unexplored candidate remains.');
        }

        this.emit('candidateSelected', {
            point: bestPoint,
            acquisition: bestScore
        });

        return bestPoint;
    }

    async optimize(initialPoints = 5, iterations = 15) {
        if (initialPoints < 1 || iterations < 1) {
            throw new RangeError('Optimization counts must be positive.');
        }

        for (let i = 0; i < initialPoints; i++) {
            await this.observe(this.randomPoint());
        }

        for (let i = 0; i < iterations; i++) {
            await this.observe(this.nextPoint(17));
        }

        return this.observations.reduce(
            (best, current) => current.y > best.y ? current : best
        );
    }

    async writeHistory(path) {
        const header = 'iteration,x1,x2,objective\n';
        const rows = this.observations.map((observation, index) =>
            `${index + 1},${observation.x[0]},${observation.x[1]},${observation.y}\n`
        );

        await fs.writeFile(path, header + rows.join(''), 'utf8');
    }
}

function objective(point) {
    if (point.length !== 2) {
        throw new RangeError('Objective expects two dimensions.');
    }

    const [x1, x2] = point;

    return (
        1.8 * Math.exp(-((x1 - 1.5) ** 2 + (x2 - 2.0) ** 2) / 1.4) +
        1.25 * Math.exp(-((x1 + 2.0) ** 2 + (x2 + 1.0) ** 2) / 0.8) -
        0.08 * (x1 ** 2 + x2 ** 2) +
        0.015 * Math.sin(17 * x1 + 11 * x2)
    );
}

async function main() {
    const bounds = [
        new Bounds(-4, 4),
        new Bounds(-4, 4)
    ];

    const optimizer = new RepositoryLikeExperiment(bounds, objective, 42);

    optimizer.on('observation', observation => {
        console.log(
            `evaluation x=(${observation.x.map(v => v.toFixed(3)).join(', ')}) ` +
            `y=${observation.y.toFixed(6)}`
        );
    });

    optimizer.on('candidateSelected', event => {
        console.log(
            `candidate acquisition=${event.acquisition.toFixed(6)}`
        );
    });

    const best = await optimizer.optimize(6, 18);

    if (!bounds.every((bound, i) =>
        best.x[i] >= bound.low && best.x[i] <= bound.high)) {
        throw new Error('Best point violates the search domain.');
    }

    console.log('\nBayesian Optimization');
    console.log(`Observations: ${optimizer.observations.length}`);
    console.log(`Best point: (${best.x.join(', ')})`);
    console.log(`Best objective: ${best.y.toFixed(6)}`);

    await optimizer.writeHistory('bayesian_optimization_history.csv');
    console.log('History: bayesian_optimization_history.csv');
}

main().catch(error => {
    console.error(`Optimization failed: ${error.message}`);
    process.exitCode = 1;
});
