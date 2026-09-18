"use strict";

/*
 * Probability Distributions
 * ==========================
 *
 * Practical JavaScript companion implementation for:
 *   Bernoulli
 *   Binomial
 *   Poisson
 *   Gaussian / Normal
 *   Uniform
 *   Exponential
 *
 * This file is executable with a modern Node.js runtime and uses only
 * built-in JavaScript functionality.
 *
 * The implementation emphasizes simulation, application-level modeling,
 * validation, numerical behavior, and JavaScript-specific techniques.
 */

// ============================================================================
// 1. GENERAL UTILITIES
// ============================================================================

function assertProbability(p, name = "probability") {
    if (!Number.isFinite(p) || p < 0 || p > 1) {
        throw new RangeError(`${name} must be finite and between 0 and 1.`);
    }
}

function assertPositive(value, name) {
    if (!Number.isFinite(value) || value <= 0) {
        throw new RangeError(`${name} must be finite and greater than zero.`);
    }
}

function factorial(n) {
    if (!Number.isInteger(n) || n < 0) {
        throw new RangeError("factorial requires a non-negative integer.");
    }

    let result = 1n;
    for (let i = 2n; i <= BigInt(n); i++) {
        result *= i;
    }
    return result;
}

function combination(n, k) {
    if (!Number.isInteger(n) || !Number.isInteger(k)) {
        throw new TypeError("n and k must be integers.");
    }

    if (n < 0 || k < 0 || k > n) {
        return 0n;
    }

    // Compute multiplicatively rather than constructing n! directly.
    const reducedK = Math.min(k, n - k);
    let result = 1n;

    for (let i = 1; i <= reducedK; i++) {
        result = (result * BigInt(n - reducedK + i)) / BigInt(i);
    }

    return result;
}

function logCombination(n, k) {
    if (k < 0 || k > n || !Number.isInteger(k) || !Number.isInteger(n)) {
        return -Infinity;
    }

    return (
        logGamma(n + 1) -
        logGamma(k + 1) -
        logGamma(n - k + 1)
    );
}

// Lanczos approximation for log(Gamma(x)).
function logGamma(x) {
    const coefficients = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7
    ];

    if (x < 0.5) {
        return (
            Math.log(Math.PI) -
            Math.log(Math.sin(Math.PI * x)) -
            logGamma(1 - x)
        );
    }

    let value = 0.9999999999998099;
    const shifted = x - 1;

    for (let i = 0; i < coefficients.length; i++) {
        value += coefficients[i] / (shifted + i + 1);
    }

    const t = shifted + coefficients.length - 0.5;

    return (
        0.5 * Math.log(2 * Math.PI) +
        (shifted + 0.5) * Math.log(t) -
        t +
        Math.log(value)
    );
}

// ============================================================================
// 2. REPRODUCIBLE PSEUDO-RANDOM NUMBER GENERATOR
// ============================================================================

class SeededRandom {
    /*
     * A small deterministic generator is useful for demonstrations because
     * Math.random() cannot be seeded by standard JavaScript APIs.
     *
     * This is suitable for simulation demonstrations, not cryptographic
     * security.
     */
    constructor(seed = 123456789) {
        this.state = seed >>> 0;
    }

    next() {
        let x = this.state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this.state = x >>> 0;
        return this.state / 4294967296;
    }

    uniform(lower = 0, upper = 1) {
        return lower + (upper - lower) * this.next();
    }
}

function randomNormal(rng, mean = 0, standardDeviation = 1) {
    // Box-Muller transformation converts uniform random values into
    // approximately standard-normal values.
    let u1 = rng.next();
    let u2 = rng.next();

    // Avoid log(0).
    u1 = Math.max(Number.MIN_VALUE, u1);

    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + standardDeviation * z;
}

// ============================================================================
// 3. BERNOULLI DISTRIBUTION
// ============================================================================

class BernoulliDistribution {
    constructor(p) {
        assertProbability(p, "p");
        this.p = p;
    }

    pmf(x) {
        if (x === 0) return 1 - this.p;
        if (x === 1) return this.p;
        return 0;
    }

    cdf(x) {
        if (x < 0) return 0;
        if (x < 1) return 1 - this.p;
        return 1;
    }

    mean() {
        return this.p;
    }

    variance() {
        return this.p * (1 - this.p);
    }

    sample(rng = new SeededRandom()) {
        return rng.next() < this.p ? 1 : 0;
    }
}

// ============================================================================
// 4. BINOMIAL DISTRIBUTION
// ============================================================================

class BinomialDistribution {
    constructor(n, p) {
        if (!Number.isInteger(n) || n < 0) {
            throw new RangeError("n must be a non-negative integer.");
        }

        assertProbability(p, "p");

        this.n = n;
        this.p = p;
    }

    pmf(k) {
        if (!Number.isInteger(k) || k < 0 || k > this.n) {
            return 0;
        }

        if (this.p === 0) {
            return k === 0 ? 1 : 0;
        }

        if (this.p === 1) {
            return k === this.n ? 1 : 0;
        }

        const logProbability =
            logCombination(this.n, k) +
            k * Math.log(this.p) +
            (this.n - k) * Math.log1p(-this.p);

        return Math.exp(logProbability);
    }

    cdf(k) {
        if (k < 0) return 0;
        if (k >= this.n) return 1;

        let total = 0;

        for (let i = 0; i <= k; i++) {
            total += this.pmf(i);
        }

        return Math.min(1, total);
    }

    mean() {
        return this.n * this.p;
    }

    variance() {
        return this.n * this.p * (1 - this.p);
    }

    sample(rng = new SeededRandom()) {
        let successes = 0;

        for (let i = 0; i < this.n; i++) {
            if (rng.next() < this.p) {
                successes++;
            }
        }

        return successes;
    }
}

// ============================================================================
// 5. POISSON DISTRIBUTION
// ============================================================================

class PoissonDistribution {
    constructor(rate) {
        if (!Number.isFinite(rate) || rate < 0) {
            throw new RangeError("Poisson rate must be non-negative.");
        }

        this.rate = rate;
    }

    pmf(k) {
        if (!Number.isInteger(k) || k < 0) {
            return 0;
        }

        if (this.rate === 0) {
            return k === 0 ? 1 : 0;
        }

        const logProbability =
            -this.rate +
            k * Math.log(this.rate) -
            logGamma(k + 1);

        return Math.exp(logProbability);
    }

    cdf(k) {
        if (k < 0) return 0;

        let total = 0;

        for (let i = 0; i <= k; i++) {
            total += this.pmf(i);
        }

        return Math.min(1, total);
    }

    mean() {
        return this.rate;
    }

    variance() {
        return this.rate;
    }

    sample(rng = new SeededRandom()) {
        if (this.rate === 0) {
            return 0;
        }

        /*
         * Knuth's algorithm is simple and exact, but its running time grows
         * with lambda. This implementation switches to a Normal approximation
         * for large rates to keep the example computationally practical.
         */
        if (this.rate > 30) {
            return Math.max(
                0,
                Math.round(randomNormal(rng, this.rate, Math.sqrt(this.rate)))
            );
        }

        const threshold = Math.exp(-this.rate);
        let product = 1;
        let count = 0;

        do {
            count++;
            product *= rng.next();
        } while (product > threshold);

        return count - 1;
    }
}

// ============================================================================
// 6. EXPONENTIAL DISTRIBUTION
// ============================================================================

class ExponentialDistribution {
    constructor(rate) {
        assertPositive(rate, "rate");
        this.rate = rate;
    }

    pdf(x) {
        if (x < 0) return 0;
        return this.rate * Math.exp(-this.rate * x);
    }

    cdf(x) {
        if (x < 0) return 0;

        // expm1 improves accuracy for small values of rate*x.
        return -Math.expm1(-this.rate * x);
    }

    survival(x) {
        if (x < 0) return 1;
        return Math.exp(-this.rate * x);
    }

    mean() {
        return 1 / this.rate;
    }

    variance() {
        return 1 / (this.rate ** 2);
    }

    quantile(p) {
        assertProbability(p, "p");

        if (p === 0) return 0;
        if (p === 1) return Infinity;

        return -Math.log1p(-p) / this.rate;
    }

    sample(rng = new SeededRandom()) {
        const u = rng.next();
        return -Math.log1p(-u) / this.rate;
    }
}

// ============================================================================
// 7. UNIFORM DISTRIBUTION
// ============================================================================

class UniformDistribution {
    constructor(lower, upper) {
        if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
            throw new RangeError("Bounds must be finite.");
        }

        if (upper <= lower) {
            throw new RangeError("upper must be greater than lower.");
        }

        this.lower = lower;
        this.upper = upper;
    }

    pdf(x) {
        return x >= this.lower && x <= this.upper
            ? 1 / (this.upper - this.lower)
            : 0;
    }

    cdf(x) {
        if (x < this.lower) return 0;
        if (x >= this.upper) return 1;

        return (x - this.lower) / (this.upper - this.lower);
    }

    mean() {
        return (this.lower + this.upper) / 2;
    }

    variance() {
        return (this.upper - this.lower) ** 2 / 12;
    }

    sample(rng = new SeededRandom()) {
        return rng.uniform(this.lower, this.upper);
    }
}

// ============================================================================
// 8. GAUSSIAN / NORMAL DISTRIBUTION
// ============================================================================

class GaussianDistribution {
    constructor(mean, standardDeviation) {
        if (!Number.isFinite(mean)) {
            throw new RangeError("mean must be finite.");
        }

        assertPositive(standardDeviation, "standardDeviation");

        this.meanValue = mean;
        this.standardDeviation = standardDeviation;
    }

    variance() {
        return this.standardDeviation ** 2;
    }

    zScore(x) {
        return (x - this.meanValue) / this.standardDeviation;
    }

    pdf(x) {
        const z = this.zScore(x);

        return (
            Math.exp(-0.5 * z * z) /
            (this.standardDeviation * Math.sqrt(2 * Math.PI))
        );
    }

    cdf(x) {
        /*
         * JavaScript does not provide erf() as a standard Math method.
         * This approximation is sufficient for demonstration purposes.
         */
        const z = this.zScore(x) / Math.sqrt(2);
        return 0.5 * (1 + erfApproximation(z));
    }

    sample(rng = new SeededRandom()) {
        return randomNormal(
            rng,
            this.meanValue,
            this.standardDeviation
        );
    }
}

function erfApproximation(x) {
    // Abramowitz-Stegun style approximation.
    const sign = x < 0 ? -1 : 1;
    const absoluteX = Math.abs(x);

    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const t = 1 / (1 + p * absoluteX);

    const polynomial =
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;

    const result =
        1 - polynomial * Math.exp(-absoluteX * absoluteX);

    return sign * result;
}

// ============================================================================
// 9. SAMPLE STATISTICS
// ============================================================================

function summarizeSample(values) {
    if (values.length === 0) {
        throw new RangeError("Cannot summarize an empty sample.");
    }

    const mean =
        values.reduce((sum, value) => sum + value, 0) / values.length;

    const variance =
        values.reduce(
            (sum, value) => sum + (value - mean) ** 2,
            0
        ) / values.length;

    return {
        count: values.length,
        mean,
        variance,
        standardDeviation: Math.sqrt(variance),
        minimum: Math.min(...values),
        maximum: Math.max(...values)
    };
}

// ============================================================================
// 10. MONTE CARLO PROBABILITY ESTIMATION
// ============================================================================

function estimateProbability(sampler, condition, trials, rng) {
    if (!Number.isInteger(trials) || trials <= 0) {
        throw new RangeError("trials must be a positive integer.");
    }

    let successes = 0;

    for (let i = 0; i < trials; i++) {
        if (condition(sampler(rng))) {
            successes++;
        }
    }

    return successes / trials;
}

// ============================================================================
// 11. DISPLAY HELPERS
// ============================================================================

function printSection(title) {
    console.log(`\n${"=".repeat(78)}\n${title}\n${"=".repeat(78)}`);
}

function printDistributionTable(rows) {
    console.table(rows);
}

// ============================================================================
// 12. BEGINNER EXAMPLES
// ============================================================================

printSection("1. Bernoulli: one binary experiment");

const coin = new BernoulliDistribution(0.5);

console.log("P(success) =", coin.pmf(1));
console.log("P(failure) =", coin.pmf(0));
console.log("Mean =", coin.mean());
console.log("Variance =", coin.variance());

const rng = new SeededRandom(42);

console.log(
    "Ten simulated coin outcomes:",
    Array.from({ length: 10 }, () => coin.sample(rng))
);

// ============================================================================

printSection("2. Binomial: count successes in fixed trials");

const campaign = new BinomialDistribution(10, 0.3);

console.log("P(X=3) =", campaign.pmf(3));
console.log("P(X<=3) =", campaign.cdf(3));
console.log("Mean =", campaign.mean());
console.log("Variance =", campaign.variance());
console.log("One simulated result =", campaign.sample(rng));

// ============================================================================

printSection("3. Poisson: count events in an interval");

const arrivals = new PoissonDistribution(4);

console.log("P(X=2) =", arrivals.pmf(2));
console.log("P(X<=2) =", arrivals.cdf(2));
console.log("Mean =", arrivals.mean());
console.log("Variance =", arrivals.variance());
console.log("One simulated count =", arrivals.sample(rng));

// ============================================================================

printSection("4. Exponential: waiting time");

const waitingTime = new ExponentialDistribution(2);

console.log("PDF at 0.5 =", waitingTime.pdf(0.5));
console.log("P(T<=0.5) =", waitingTime.cdf(0.5));
console.log("P(T>0.5) =", waitingTime.survival(0.5));
console.log("Mean =", waitingTime.mean());
console.log("Median =", waitingTime.quantile(0.5));
console.log("One waiting-time sample =", waitingTime.sample(rng));

// ============================================================================

printSection("5. Uniform: bounded continuous uncertainty");

const temperature = new UniformDistribution(10, 20);

console.log("PDF at 15 =", temperature.pdf(15));
console.log("P(X<=15) =", temperature.cdf(15));
console.log("Mean =", temperature.mean());
console.log("Variance =", temperature.variance());

// ============================================================================

printSection("6. Gaussian: continuous bell-shaped model");

const scores = new GaussianDistribution(100, 15);

for (const value of [70, 85, 100, 115, 130]) {
    console.log({
        value,
        zScore: scores.zScore(value),
        cdf: scores.cdf(value)
    });
}

// ============================================================================
// 13. DISTRIBUTION COMPARISON
// ============================================================================

printSection("7. Comparing the six distributions");

printDistributionTable([
    {
        distribution: "Bernoulli",
        variable: "Discrete",
        parameter: "p",
        mean: "p",
        variance: "p(1-p)",
        interpretation: "one binary trial"
    },
    {
        distribution: "Binomial",
        variable: "Discrete",
        parameter: "n,p",
        mean: "np",
        variance: "np(1-p)",
        interpretation: "success count"
    },
    {
        distribution: "Poisson",
        variable: "Discrete",
        parameter: "lambda",
        mean: "lambda",
        variance: "lambda",
        interpretation: "event count"
    },
    {
        distribution: "Exponential",
        variable: "Continuous",
        parameter: "rate",
        mean: "1/rate",
        variance: "1/rate²",
        interpretation: "waiting time"
    },
    {
        distribution: "Uniform",
        variable: "Continuous",
        parameter: "lower,upper",
        mean: "(a+b)/2",
        variance: "(b-a)²/12",
        interpretation: "bounded constant density"
    },
    {
        distribution: "Gaussian",
        variable: "Continuous",
        parameter: "mean,SD",
        mean: "mean",
        variance: "SD²",
        interpretation: "symmetric bell curve"
    }
]);

// ============================================================================
// 14. POISSON PROCESS CONNECTION
// ============================================================================

printSection("8. Poisson process and exponential interarrival times");

const requestsPerMinute = 50;
const requestCount = new PoissonDistribution(requestsPerMinute);
const interarrival = new ExponentialDistribution(requestsPerMinute);

console.log(
    "Expected requests in 2 minutes:",
    requestCount.mean() * 2
);

console.log(
    "Expected interarrival time in seconds:",
    interarrival.mean() * 60
);

console.log(
    "P(no request for 0.1 minutes):",
    interarrival.survival(0.1)
);

// ============================================================================
// 15. BINOMIAL TO POISSON APPROXIMATION
// ============================================================================

printSection("9. Binomial-to-Poisson approximation");

const n = 1000;
const p = 0.004;

const binomialModel = new BinomialDistribution(n, p);
const poissonApproximation = new PoissonDistribution(n * p);

const comparison = [];

for (let k = 0; k <= 10; k++) {
    comparison.push({
        k,
        binomial: binomialModel.pmf(k),
        poisson: poissonApproximation.pmf(k),
        absoluteDifference:
            Math.abs(
                binomialModel.pmf(k) -
                poissonApproximation.pmf(k)
            )
    });
}

console.table(comparison);

// ============================================================================
// 16. MONTE CARLO
// ============================================================================

printSection("10. Monte Carlo estimation");

const uniformRng = new SeededRandom(100);

const estimatedProbability = estimateProbability(
    (generator) => temperature.sample(generator),
    (value) => value <= 15,
    100000,
    uniformRng
);

console.log(
    "Estimated P(Uniform(10,20)<=15):",
    estimatedProbability
);

console.log(
    "Exact probability:",
    temperature.cdf(15)
);

// ============================================================================
// 17. LARGE SAMPLE SIMULATION
// ============================================================================

printSection("11. Gaussian sample statistics");

const gaussianRng = new SeededRandom(2026);
const observations = [];

for (let i = 0; i < 10000; i++) {
    observations.push(scores.sample(gaussianRng));
}

console.log(summarizeSample(observations));

// ============================================================================
// 18. CENTRAL LIMIT THEOREM
// ============================================================================

printSection("12. Central Limit Theorem demonstration");

const cltRng = new SeededRandom(2025);
const sampleMeans = [];

for (let repetition = 0; repetition < 5000; repetition++) {
    let total = 0;

    for (let observation = 0; observation < 30; observation++) {
        total += cltRng.next();
    }

    sampleMeans.push(total / 30);
}

console.log("Uniform population mean =", 0.5);
console.log(
    "Observed mean of sample means =",
    summarizeSample(sampleMeans).mean
);
console.log(
    "Observed SD of sample means =",
    summarizeSample(sampleMeans).standardDeviation
);
console.log(
    "Theoretical approximate SD =",
    Math.sqrt((1 / 12) / 30)
);

// ============================================================================
// 19. EXPONENTIAL MEMORYLESS PROPERTY
// ============================================================================

printSection("13. Exponential memorylessness");

const rate = 0.25;
const s = 3;
const t = 2;

const conditionalSurvival =
    Math.exp(-rate * (s + t)) /
    Math.exp(-rate * s);

const directSurvival =
    Math.exp(-rate * t);

console.log("Conditional survival =", conditionalSurvival);
console.log("Direct survival =", directSurvival);
console.log(
    "Difference =",
    Math.abs(conditionalSurvival - directSurvival)
);

// ============================================================================
// 20. PRACTICAL SERVICE CASE STUDY
// ============================================================================

printSection("14. Service monitoring case study");

const averageRequestsPerMinute = 120;
const failureProbability = 0.01;

const oneMinuteRequests =
    new PoissonDistribution(averageRequestsPerMinute);

const fiveHundredRequests =
    new BinomialDistribution(500, failureProbability);

const requestWaitingTime =
    new ExponentialDistribution(averageRequestsPerMinute);

const responseLatency =
    new GaussianDistribution(200, 40);

console.log(
    "P(exactly 120 requests in one minute) =",
    oneMinuteRequests.pmf(120)
);

console.log(
    "P(at least 130 requests in one minute) =",
    1 - oneMinuteRequests.cdf(129)
);

console.log(
    "Expected failures in 500 requests =",
    fiveHundredRequests.mean()
);

console.log(
    "P(no more than 3 failures in 500) =",
    fiveHundredRequests.cdf(3)
);

console.log(
    "Expected interarrival time in seconds =",
    requestWaitingTime.mean() * 60
);

console.log(
    "P(response latency <= 250 ms) =",
    responseLatency.cdf(250)
);

// ============================================================================
// 21. HISTOGRAM
// ============================================================================

printSection("15. Text histogram of Gaussian observations");

function textHistogram(values, bins = 12, width = 45) {
    if (values.length === 0) {
        throw new RangeError("values cannot be empty.");
    }

    const minimum = Math.min(...values);
    const maximum = Math.max(...values);

    if (minimum === maximum) {
        console.log(`${minimum.toFixed(3)} | ${"#".repeat(width)}`);
        return;
    }

    const binWidth = (maximum - minimum) / bins;
    const counts = new Array(bins).fill(0);

    for (const value of values) {
        let index = Math.floor((value - minimum) / binWidth);
        index = Math.min(index, bins - 1);
        counts[index]++;
    }

    const maximumCount = Math.max(...counts);

    counts.forEach((count, index) => {
        const start = minimum + index * binWidth;
        const end = start + binWidth;
        const barLength =
            Math.round(width * count / maximumCount);

        console.log(
            `${start.toFixed(1).padStart(7)} - ` +
            `${end.toFixed(1).padStart(7)} | ` +
            `${"#".repeat(barLength)} ${count}`
        );
    });
}

textHistogram(observations.slice(0, 2000));

// ============================================================================
// 22. EDGE CASES
// ============================================================================

printSection("16. Edge cases");

console.log(
    "Bernoulli p=0, P(X=1):",
    new BernoulliDistribution(0).pmf(1)
);

console.log(
    "Bernoulli p=1, P(X=1):",
    new BernoulliDistribution(1).pmf(1)
);

console.log(
    "Binomial p=0, P(X=0):",
    new BinomialDistribution(20, 0).pmf(0)
);

console.log(
    "Binomial p=1, P(X=20):",
    new BinomialDistribution(20, 1).pmf(20)
);

console.log(
    "Poisson lambda=0, P(X=0):",
    new PoissonDistribution(0).pmf(0)
);

console.log(
    "Exponential CDF at -1:",
    waitingTime.cdf(-1)
);

console.log(
    "Uniform CDF below lower bound:",
    temperature.cdf(0)
);

console.log(
    "Uniform CDF above upper bound:",
    temperature.cdf(100)
);

// ============================================================================
// 23. VALIDATION
// ============================================================================

printSection("17. Validation and error handling");

const invalidCases = [
    () => new BernoulliDistribution(1.5),
    () => new BinomialDistribution(-1, 0.5),
    () => new PoissonDistribution(-3),
    () => new ExponentialDistribution(0),
    () => new UniformDistribution(5, 5),
    () => new GaussianDistribution(0, -1)
];

for (const invalidCase of invalidCases) {
    try {
        invalidCase();
        console.log("Unexpectedly accepted invalid input.");
    } catch (error) {
        console.log("Rejected invalid input:", error.message);
    }
}

// ============================================================================
// 24. PERFORMANCE CONSIDERATIONS
// ============================================================================

printSection("18. Performance notes");

console.log(`
JavaScript numerical modeling considerations:

1. Number uses IEEE-754 double precision for ordinary numeric calculations.
2. Very large integer combinatorial values can exceed safe integer precision.
3. BigInt is used above for exact factorial and combination demonstrations.
4. Probability calculations should normally remain in floating-point form.
5. Logarithmic probability calculations reduce overflow and underflow risks.
6. Repeated JavaScript loops can become expensive for very large simulations.
7. Typed arrays can be useful for large numerical datasets.
8. Browser applications should avoid blocking the main thread with huge
   simulations; Web Workers can move heavy computation away from the UI.
9. Math.random() is not a cryptographic random-number generator.
10. Statistical simulation and security-sensitive randomness are different
    requirements.
`);

// ============================================================================
// 25. DISTRIBUTION SELECTION
// ============================================================================

printSection("19. Distribution selection guide");

const guide = [
    ["Bernoulli", "one binary outcome"],
    ["Binomial", "number of successes in n binary trials"],
    ["Poisson", "number of events in a fixed interval"],
    ["Exponential", "waiting time between Poisson events"],
    ["Uniform", "continuous bounded variable with constant density"],
    ["Gaussian", "continuous symmetric bell-shaped variable"]
];

for (const [name, use] of guide) {
    console.log(`${name.padEnd(12)} -> ${use}`);
}

// ============================================================================
// 26. SIMPLE SELF-TESTS
// ============================================================================

printSection("20. Self-tests");

function approximatelyEqual(a, b, tolerance = 1e-10) {
    return Math.abs(a - b) <= tolerance;
}

if (!approximatelyEqual(coin.pmf(0) + coin.pmf(1), 1)) {
    throw new Error("Bernoulli PMF test failed.");
}

let binomialTotal = 0;

for (let k = 0; k <= campaign.n; k++) {
    binomialTotal += campaign.pmf(k);
}

if (!approximatelyEqual(binomialTotal, 1, 1e-9)) {
    throw new Error("Binomial PMF normalization test failed.");
}

if (!approximatelyEqual(arrivals.mean(), arrivals.variance())) {
    throw new Error("Poisson mean/variance relationship failed.");
}

if (!approximatelyEqual(
    temperature.cdf(15) - temperature.cdf(12),
    0.3
)) {
    throw new Error("Uniform interval probability test failed.");
}

if (!approximatelyEqual(
    waitingTime.quantile(0.5),
    Math.log(2) / waitingTime.rate
)) {
    throw new Error("Exponential median test failed.");
}

console.log("All self-tests passed.");

// ============================================================================
// 27. FINAL REFERENCE
// ============================================================================

printSection("21. Compact mathematical reference");

console.log(`
Bernoulli:
    P(X=x) = p^x (1-p)^(1-x)
    E[X] = p
    Var(X) = p(1-p)

Binomial:
    P(X=k) = C(n,k)p^k(1-p)^(n-k)
    E[X] = np
    Var(X) = np(1-p)

Poisson:
    P(X=k) = exp(-lambda)lambda^k/k!
    E[X] = lambda
    Var(X) = lambda

Exponential:
    f(x) = lambda exp(-lambda x), x >= 0
    E[X] = 1/lambda
    Var(X) = 1/lambda²

Uniform:
    f(x) = 1/(b-a), a <= x <= b
    E[X] = (a+b)/2
    Var(X) = (b-a)²/12

Gaussian:
    f(x) = exp(-(x-mu)²/(2sigma²)) /
           (sigma sqrt(2pi))
    E[X] = mu
    Var(X) = sigma²

The mathematical model must match the data-generating assumptions.
`);
