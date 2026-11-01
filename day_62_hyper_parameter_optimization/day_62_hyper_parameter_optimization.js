"use strict";

/*
 * Hyperparameter Optimization:
 * Grid search, random search, and explicit search-space management.
 *
 * This implementation uses an event-driven trial runner. A configuration
 * is evaluated asynchronously so the orchestration model resembles an
 * actual training service where trials may complete independently.
 */

const SearchStrategy = Object.freeze({
    GRID: "grid",
    RANDOM: "random"
});

class SearchSpace {
    constructor({ learningRates, maxDepths, estimators }) {
        this.learningRates = [...learningRates];
        this.maxDepths = [...maxDepths];
        this.estimators = [...estimators];
        this.validate();
    }

    validate() {
        if (
            this.learningRates.length === 0 ||
            this.maxDepths.length === 0 ||
            this.estimators.length === 0
        ) {
            throw new Error("Every search-space dimension must contain values.");
        }

        if (this.learningRates.some(v => !Number.isFinite(v) || v <= 0)) {
            throw new Error("Learning rates must be positive finite numbers.");
        }

        if (this.maxDepths.some(v => !Number.isInteger(v) || v < 1)) {
            throw new Error("Max depths must be positive integers.");
        }

        if (this.estimators.some(v => !Number.isInteger(v) || v < 1)) {
            throw new Error("Estimator counts must be positive integers.");
        }

        const hasDuplicates = array =>
            new Set(array).size !== array.length;

        if (
            hasDuplicates(this.learningRates) ||
            hasDuplicates(this.maxDepths) ||
            hasDuplicates(this.estimators)
        ) {
            throw new Error("Search-space dimensions must not contain duplicates.");
        }
    }

    get size() {
        return (
            this.learningRates.length *
            this.maxDepths.length *
            this.estimators.length
        );
    }
}

class Configuration {
    constructor(learningRate, maxDepth, estimators) {
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.estimators = estimators;
    }

    key() {
        return `${this.learningRate}|${this.maxDepth}|${this.estimators}`;
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

    choice(values) {
        return values[Math.floor(this.next() * values.length)];
    }

    between(min, max) {
        return min + (max - min) * this.next();
    }
}

class TrialEventBus {
    constructor() {
        this.listeners = new Map();
    }

    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
    }

    emit(event, payload) {
        const listeners = this.listeners.get(event) ?? new Set();

        for (const listener of listeners) {
            listener(payload);
        }
    }
}

class AsyncEvaluator {
    constructor(random) {
        this.random = random;
    }

    evaluate(configuration) {
        return new Promise((resolve, reject) => {
            const delay = 5 + Math.floor(this.random.next() * 15);

            setTimeout(() => {
                if (
                    !Number.isFinite(configuration.learningRate) ||
                    configuration.learningRate <= 0
                ) {
                    reject(new Error("Invalid learning rate."));
                    return;
                }

                const lrPenalty = Math.abs(
                    Math.log10(configuration.learningRate) - Math.log10(0.05)
                );

                const depthPenalty =
                    Math.abs(configuration.maxDepth - 6) * 0.018;

                const estimatorPenalty =
                    Math.abs(configuration.estimators - 200) / 10000;

                const noise = this.random.between(-0.003, 0.003);

                const score = Math.max(
                    0,
                    Math.min(
                        1,
                        0.96 -
                            lrPenalty * 0.09 -
                            depthPenalty -
                            estimatorPenalty +
                            noise
                    )
                );

                const duration =
                    0.5 +
                    configuration.maxDepth * 0.08 +
                    configuration.estimators * 0.004;

                resolve({
                    configuration,
                    score,
                    duration,
                    completedAt: new Date().toISOString()
                });
            }, delay);
        });
    }
}

class HyperparameterOptimizer {
    constructor(searchSpace, evaluator, eventBus, random) {
        this.space = searchSpace;
        this.evaluator = evaluator;
        this.events = eventBus;
        this.random = random;
    }

    gridConfigurations() {
        const configurations = [];

        for (const learningRate of this.space.learningRates) {
            for (const maxDepth of this.space.maxDepths) {
                for (const estimators of this.space.estimators) {
                    configurations.push(
                        new Configuration(
                            learningRate,
                            maxDepth,
                            estimators
                        )
                    );
                }
            }
        }

        return configurations;
    }

    randomConfigurations(count) {
        if (!Number.isInteger(count) || count < 1) {
            throw new Error("Random trial count must be a positive integer.");
        }

        const target = Math.min(count, this.space.size);
        const selected = new Map();

        while (selected.size < target) {
            const configuration = new Configuration(
                this.random.choice(this.space.learningRates),
                this.random.choice(this.space.maxDepths),
                this.random.choice(this.space.estimators)
            );

            selected.set(configuration.key(), configuration);
        }

        return [...selected.values()];
    }

    async run(strategy, options = {}) {
        let configurations;

        if (strategy === SearchStrategy.GRID) {
            configurations = this.gridConfigurations();
        } else if (strategy === SearchStrategy.RANDOM) {
            configurations = this.randomConfigurations(options.trials ?? 10);
        } else {
            throw new Error(`Unknown search strategy: ${strategy}`);
        }

        const results = [];

        // Promise.all models independent trial execution. The evaluator itself
        // remains deterministic enough for reproducible demonstrations.
        const executions = configurations.map(async configuration => {
            this.events.emit("trialStarted", { strategy, configuration });

            try {
                const result = await this.evaluator.evaluate(configuration);
                this.events.emit("trialCompleted", result);
                return result;
            } catch (error) {
                this.events.emit("trialFailed", {
                    configuration,
                    error: error.message
                });
                return null;
            }
        });

        const completed = await Promise.all(executions);

        for (const result of completed) {
            if (result !== null) {
                results.push(result);
            }
        }

        return {
            strategy,
            trials: results,
            best: results.reduce(
                (best, current) =>
                    !best || current.score > best.score ? current : best,
                null
            )
        };
    }
}

function average(results) {
    if (results.length === 0) {
        throw new Error("Cannot calculate average for empty results.");
    }

    return (
        results.reduce((sum, result) => sum + result.score, 0) /
        results.length
    );
}

function printResult(result) {
    console.log(`\n${result.strategy.toUpperCase()} SEARCH`);
    console.log("-".repeat(55));
    console.log(`Trials:        ${result.trials.length}`);
    console.log(`Average score: ${average(result.trials).toFixed(4)}`);

    const best = result.best;

    console.log(`Best score:    ${best.score.toFixed(4)}`);
    console.log(`Learning rate: ${best.configuration.learningRate}`);
    console.log(`Max depth:     ${best.configuration.maxDepth}`);
    console.log(`Estimators:    ${best.configuration.estimators}`);
}

async function main() {
    const space = new SearchSpace({
        learningRates: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        maxDepths: [2, 4, 6, 8, 10],
        estimators: [50, 100, 200, 400]
    });

    const random = new SeededRandom(42);
    const evaluator = new AsyncEvaluator(random);
    const events = new TrialEventBus();

    events.on("trialStarted", ({ strategy, configuration }) => {
        if (strategy === SearchStrategy.RANDOM) {
            console.log(
                `Random trial: lr=${configuration.learningRate}, ` +
                `depth=${configuration.maxDepth}, ` +
                `estimators=${configuration.estimators}`
            );
        }
    });

    events.on("trialFailed", ({ configuration, error }) => {
        console.error(
            `Rejected ${configuration.key()}: ${error}`
        );
    });

    const optimizer = new HyperparameterOptimizer(
        space,
        evaluator,
        events,
        random
    );

    const gridResult = await optimizer.run(SearchStrategy.GRID);

    const randomResult = await optimizer.run(
        SearchStrategy.RANDOM,
        { trials: 20 }
    );

    printResult(gridResult);
    printResult(randomResult);

    console.log("\nSEARCH SPACE");
    console.log("-".repeat(55));
    console.log(`Total possible configurations: ${space.size}`);
    console.log(`Random configurations evaluated: ${randomResult.trials.length}`);

    console.log("\nTop three random trials:");
    [...randomResult.trials]
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .forEach((trial, index) => {
            console.log(
                `${index + 1}. ` +
                `lr=${trial.configuration.learningRate}, ` +
                `depth=${trial.configuration.maxDepth}, ` +
                `estimators=${trial.configuration.estimators}, ` +
                `score=${trial.score.toFixed(4)}`
            );
        });
}

main().catch(error => {
    console.error(`Optimization failed: ${error.message}`);
    process.exitCode = 1;
});
