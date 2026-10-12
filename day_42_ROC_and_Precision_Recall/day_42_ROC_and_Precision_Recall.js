"use strict";

/*
 * ROC & Precision-Recall analysis in Node.js.
 *
 * This implementation focuses on an event-driven threshold workflow:
 * score observations are evaluated at multiple operating points, policy
 * events are emitted when thresholds are selected, and asynchronous
 * evaluation demonstrates how model-scoring infrastructure can feed a
 * metric service.
 *
 * Run with:
 *   node roc_pr_analysis.js
 */

const EventEmitter = require("node:events");

class BinaryDataset {
    constructor(labels, scores) {
        if (!Array.isArray(labels) || !Array.isArray(scores)) {
            throw new TypeError("Labels and scores must be arrays.");
        }
        if (labels.length !== scores.length || labels.length === 0) {
            throw new Error("Labels and scores must have equal non-zero length.");
        }

        for (const label of labels) {
            if (label !== 0 && label !== 1) {
                throw new Error("Labels must contain only 0 or 1.");
            }
        }

        for (const score of scores) {
            if (!Number.isFinite(score) || score < 0 || score > 1) {
                throw new Error("Scores must be finite numbers in [0, 1].");
            }
        }

        if (!labels.includes(0) || !labels.includes(1)) {
            throw new Error("ROC and PR analysis require both classes.");
        }

        this.rows = labels.map((actual, index) => ({
            actual,
            score: scores[index]
        }));
    }

    get prevalence() {
        return this.rows.filter(row => row.actual === 1).length / this.rows.length;
    }

    thresholds() {
        const values = new Set(this.rows.map(row => row.score));
        values.add(0);
        values.add(1);
        return [...values].sort((a, b) => b - a);
    }
}

function confusionMatrix(dataset, threshold) {
    if (!(dataset instanceof BinaryDataset)) {
        throw new TypeError("dataset must be a BinaryDataset.");
    }
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
        throw new RangeError("Threshold must be in [0, 1].");
    }

    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;

    for (const row of dataset.rows) {
        const predictedPositive = row.score >= threshold;

        if (row.actual === 1 && predictedPositive) tp++;
        else if (row.actual === 0 && predictedPositive) fp++;
        else if (row.actual === 0 && !predictedPositive) tn++;
        else fn++;
    }

    return { threshold, tp, fp, tn, fn };
}

function metrics(cm) {
    const precision = cm.tp + cm.fp === 0
        ? 0
        : cm.tp / (cm.tp + cm.fp);

    const recall = cm.tp + cm.fn === 0
        ? 0
        : cm.tp / (cm.tp + cm.fn);

    const specificity = cm.tn + cm.fp === 0
        ? 0
        : cm.tn / (cm.tn + cm.fp);

    const f1 = precision + recall === 0
        ? 0
        : (2 * precision * recall) / (precision + recall);

    return {
        ...cm,
        precision,
        recall,
        specificity,
        fpr: 1 - specificity,
        f1
    };
}

function rocCurve(dataset) {
    return dataset.thresholds()
        .map(threshold => {
            const result = metrics(confusionMatrix(dataset, threshold));
            return {
                threshold,
                fpr: result.fpr,
                tpr: result.recall
            };
        })
        .sort((a, b) => a.fpr - b.fpr || a.tpr - b.tpr);
}

function integrate(points, xKey, yKey) {
    const ordered = [...points].sort((a, b) => a[xKey] - b[xKey]);
    let area = 0;

    for (let index = 1; index < ordered.length; index++) {
        const previous = ordered[index - 1];
        const current = ordered[index];
        area +=
            (current[xKey] - previous[xKey]) *
            (current[yKey] + previous[yKey]) / 2;
    }

    return area;
}

function rocAuc(dataset) {
    return integrate(rocCurve(dataset), "fpr", "tpr");
}

function precisionRecallCurve(dataset) {
    return dataset.thresholds()
        .map(threshold => {
            const result = metrics(confusionMatrix(dataset, threshold));
            return {
                threshold,
                recall: result.recall,
                precision: result.precision
            };
        })
        .sort((a, b) => a.recall - b.recall || a.threshold - b.threshold);
}

function averagePrecision(dataset) {
    const ranked = [...dataset.rows].sort((a, b) => b.score - a.score);
    const totalPositives = ranked.filter(row => row.actual === 1).length;

    if (totalPositives === 0) return 0;

    let tp = 0;
    let fp = 0;
    let previousRecall = 0;
    let result = 0;

    for (const row of ranked) {
        if (row.actual === 1) tp++;
        else fp++;

        const precision = tp / (tp + fp);
        const recall = tp / totalPositives;

        if (row.actual === 1) {
            result += precision * (recall - previousRecall);
            previousRecall = recall;
        }
    }

    return result;
}

function fBeta(result, beta) {
    if (!Number.isFinite(beta) || beta <= 0) {
        throw new RangeError("beta must be positive.");
    }

    const betaSquared = beta * beta;
    const denominator = betaSquared * result.precision + result.recall;

    return denominator === 0
        ? 0
        : ((1 + betaSquared) * result.precision * result.recall) / denominator;
}

function selectThreshold(dataset, objective) {
    let best = null;

    for (const threshold of dataset.thresholds()) {
        const result = metrics(confusionMatrix(dataset, threshold));
        const value = objective(result);

        if (best === null || value > best.value) {
            best = { threshold, value, result };
        }
    }

    return best;
}

function evaluatePolicy(dataset, policy) {
    return selectThreshold(dataset, result => policy(result));
}

class ThresholdMonitor extends EventEmitter {
    constructor(dataset) {
        super();
        this.dataset = dataset;
    }

    evaluate(policyName, objective) {
        const selection = selectThreshold(this.dataset, objective);

        this.emit("threshold-selected", {
            policyName,
            ...selection
        });

        return selection;
    }
}

async function scoreModelAsynchronously(dataset) {
    /*
     * Promise-based scheduling models an application where predictions arrive
     * from an asynchronous inference service. The metric calculations remain
     * deterministic once the scores are available.
     */
    await new Promise(resolve => setTimeout(resolve, 5));
    return dataset;
}

function printRocCurve(dataset) {
    console.log("\nROC operating points");
    console.log("threshold | FPR   | TPR");

    for (const point of rocCurve(dataset)) {
        console.log(
            `${point.threshold.toFixed(2).padStart(9)} | ` +
            `${point.fpr.toFixed(3)} | ${point.tpr.toFixed(3)}`
        );
    }
}

function printPrCurve(dataset) {
    console.log("\nPrecision-Recall operating points");
    console.log("threshold | precision | recall");

    for (const point of precisionRecallCurve(dataset)) {
        console.log(
            `${point.threshold.toFixed(2).padStart(9)} | ` +
            `${point.precision.toFixed(3).padStart(9)} | ` +
            `${point.recall.toFixed(3).padStart(6)}`
        );
    }
}

async function main() {
    const labels = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ];

    const scores = [
        0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.14, 0.16, 0.18, 0.21,
        0.24, 0.27, 0.29, 0.32, 0.34, 0.37, 0.41, 0.45, 0.49, 0.55,
        0.33, 0.48, 0.52, 0.57, 0.61, 0.66, 0.71, 0.77, 0.83, 0.91
    ];

    const dataset = new BinaryDataset(labels, scores);
    const monitor = new ThresholdMonitor(dataset);

    monitor.on("threshold-selected", event => {
        console.log(
            `Policy "${event.policyName}" selected threshold ` +
            `${event.threshold.toFixed(2)} with objective=${event.value.toFixed(4)}`
        );
    });

    console.log("ROC & Precision-Recall analysis");
    console.log(`Samples: ${dataset.rows.length}`);
    console.log(`Positive prevalence: ${dataset.prevalence.toFixed(3)}`);

    printRocCurve(dataset);
    printPrCurve(dataset);

    console.log(`\nROC AUC: ${rocAuc(dataset).toFixed(4)}`);
    console.log(`Average Precision: ${averagePrecision(dataset).toFixed(4)}`);

    const maxF1 = monitor.evaluate(
        "maximum F1",
        result => result.f1
    );

    console.log(
        `F1 operating point: precision=${maxF1.result.precision.toFixed(3)}, ` +
        `recall=${maxF1.result.recall.toFixed(3)}, ` +
        `F1=${maxF1.result.f1.toFixed(3)}`
    );

    const highRecallPolicy = monitor.evaluate(
        "recall subject to FPR <= 0.10",
        result => result.fpr <= 0.10 ? result.recall : -Infinity
    );

    console.log(
        `Constrained operating point: ` +
        `threshold=${highRecallPolicy.threshold.toFixed(2)}, ` +
        `FPR=${highRecallPolicy.result.fpr.toFixed(3)}, ` +
        `recall=${highRecallPolicy.result.recall.toFixed(3)}`
    );

    const costSensitive = monitor.evaluate(
        "false-negative cost = 5",
        result =>
            -(5 * result.fn + 1 * result.fp)
    );

    console.log(
        `Cost-sensitive point: threshold=${costSensitive.threshold.toFixed(2)}, ` +
        `FP=${costSensitive.result.fp}, FN=${costSensitive.result.fn}`
    );

    await scoreModelAsynchronously(dataset);

    console.log("\nInterpretation");
    console.log(
        "ROC AUC evaluates discrimination across thresholds, while the PR " +
        "view exposes the precision-recall trade-off and is strongly affected " +
        "by positive-class prevalence."
    );
    console.log(
        "Threshold optimization converts a ranking model into an operational " +
        "decision rule. The selected threshold must be validated independently " +
        "from the data used to tune it."
    );
}

main().catch(error => {
    console.error(`Analysis failed: ${error.message}`);
    process.exitCode = 1;
});
