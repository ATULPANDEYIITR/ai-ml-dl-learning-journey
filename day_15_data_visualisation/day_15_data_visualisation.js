/*
Data Visualization:
Histograms, Scatter Plots, Box Plots, Line Plots, and Distributions

This self-contained JavaScript study file focuses on browser-compatible
data-processing concepts and uses HTML Canvas directly. It can run in a
browser by opening the generated HTML document, or in Node.js for the
statistical calculations and SVG generation.

No external npm packages are required.

The program:
1. Introduces distribution-oriented data structures.
2. Calculates descriptive statistics.
3. Calculates quantiles, IQR, outliers, and Pearson correlation.
4. Demonstrates histogram binning.
5. Demonstrates scatter-plot data preparation.
6. Demonstrates box-plot statistics.
7. Demonstrates ordered line-series data.
8. Demonstrates normalization and comparison.
9. Generates a complete SVG dashboard as a string.
10. Includes validation and performance-oriented aggregation.
*/

"use strict";

// -----------------------------------------------------------------------------
// Fundamental statistical utilities
// -----------------------------------------------------------------------------

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function numericValues(values) {
    if (!Array.isArray(values)) {
        throw new TypeError("Expected an array.");
    }

    return values
        .filter(value => value !== null && value !== undefined)
        .map(Number)
        .filter(Number.isFinite);
}

function mean(values) {
    const data = numericValues(values);

    if (data.length === 0) {
        throw new Error("Mean requires at least one numeric observation.");
    }

    return data.reduce((sum, value) => sum + value, 0) / data.length;
}

function median(values) {
    const data = numericValues(values).sort((a, b) => a - b);

    if (data.length === 0) {
        throw new Error("Median requires at least one numeric observation.");
    }

    const middle = Math.floor(data.length / 2);

    if (data.length % 2 === 0) {
        return (data[middle - 1] + data[middle]) / 2;
    }

    return data[middle];
}

function quantile(values, probability) {
    const data = numericValues(values).sort((a, b) => a - b);

    if (data.length === 0) {
        throw new Error("Quantile requires at least one numeric observation.");
    }

    if (probability < 0 || probability > 1) {
        throw new RangeError("Probability must be between 0 and 1.");
    }

    const position = probability * (data.length - 1);
    const lowerIndex = Math.floor(position);
    const upperIndex = Math.ceil(position);

    if (lowerIndex === upperIndex) {
        return data[lowerIndex];
    }

    const fraction = position - lowerIndex;

    return (
        data[lowerIndex] +
        fraction * (data[upperIndex] - data[lowerIndex])
    );
}

function standardDeviation(values) {
    const data = numericValues(values);

    if (data.length < 2) {
        throw new Error("Sample standard deviation requires at least two values.");
    }

    const average = mean(data);

    const squaredDifferences = data.map(
        value => (value - average) ** 2
    );

    return Math.sqrt(
        squaredDifferences.reduce((sum, value) => sum + value, 0) /
        (data.length - 1)
    );
}

function describe(values) {
    const data = numericValues(values);

    if (data.length === 0) {
        throw new Error("Cannot describe an empty dataset.");
    }

    const q1 = quantile(data, 0.25);
    const q3 = quantile(data, 0.75);

    return {
        count: data.length,
        minimum: Math.min(...data),
        q1,
        median: median(data),
        mean: mean(data),
        q3,
        maximum: Math.max(...data),
        iqr: q3 - q1,
        standardDeviation: data.length >= 2
            ? standardDeviation(data)
            : 0
    };
}

function iqrOutliers(values) {
    const data = numericValues(values);

    if (data.length === 0) {
        return [];
    }

    const q1 = quantile(data, 0.25);
    const q3 = quantile(data, 0.75);
    const iqr = q3 - q1;

    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;

    return data.filter(
        value => value < lowerFence || value > upperFence
    );
}

function pearsonCorrelation(xValues, yValues) {
    if (xValues.length !== yValues.length) {
        throw new Error("x and y must contain the same number of observations.");
    }

    if (xValues.length < 2) {
        throw new Error("Correlation requires at least two pairs.");
    }

    const x = numericValues(xValues);
    const y = numericValues(yValues);

    if (x.length !== y.length) {
        throw new Error("Missing or invalid values cannot be silently mismatched.");
    }

    const xMean = mean(x);
    const yMean = mean(y);

    let numerator = 0;
    let xVariance = 0;
    let yVariance = 0;

    for (let index = 0; index < x.length; index += 1) {
        const xDifference = x[index] - xMean;
        const yDifference = y[index] - yMean;

        numerator += xDifference * yDifference;
        xVariance += xDifference ** 2;
        yVariance += yDifference ** 2;
    }

    const denominator = Math.sqrt(xVariance * yVariance);

    if (denominator === 0) {
        throw new Error("Correlation is undefined for a constant variable.");
    }

    return numerator / denominator;
}

// -----------------------------------------------------------------------------
// Histogram calculation
// -----------------------------------------------------------------------------

function createHistogram(values, binCount = 10) {
    const data = numericValues(values);

    if (data.length === 0) {
        throw new Error("Histogram requires numeric observations.");
    }

    if (!Number.isInteger(binCount) || binCount <= 0) {
        throw new RangeError("binCount must be a positive integer.");
    }

    const minimum = Math.min(...data);
    const maximum = Math.max(...data);

    // Constant datasets need a non-zero visual width.
    if (minimum === maximum) {
        return {
            minimum,
            maximum,
            binWidth: 1,
            bins: [{
                lower: minimum - 0.5,
                upper: maximum + 0.5,
                count: data.length
            }]
        };
    }

    const binWidth = (maximum - minimum) / binCount;

    const bins = Array.from(
        { length: binCount },
        (_, index) => ({
            lower: minimum + index * binWidth,
            upper: minimum + (index + 1) * binWidth,
            count: 0
        })
    );

    for (const value of data) {
        let index = Math.floor((value - minimum) / binWidth);

        // The maximum value would otherwise produce an index equal to
        // binCount. It belongs to the final interval.
        if (index === binCount) {
            index = binCount - 1;
        }

        bins[index].count += 1;
    }

    return {
        minimum,
        maximum,
        binWidth,
        bins
    };
}

// -----------------------------------------------------------------------------
// Box plot preparation
// -----------------------------------------------------------------------------

function calculateBoxPlot(values) {
    const data = numericValues(values);

    if (data.length === 0) {
        throw new Error("Box plot requires numeric observations.");
    }

    const q1 = quantile(data, 0.25);
    const medianValue = quantile(data, 0.5);
    const q3 = quantile(data, 0.75);
    const iqr = q3 - q1;

    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;

    const nonOutliers = data.filter(
        value => value >= lowerFence && value <= upperFence
    );

    const outliers = data.filter(
        value => value < lowerFence || value > upperFence
    );

    return {
        minimum: Math.min(...data),
        q1,
        median: medianValue,
        q3,
        maximum: Math.max(...data),
        iqr,
        lowerFence,
        upperFence,
        lowerWhisker: Math.min(...nonOutliers),
        upperWhisker: Math.max(...nonOutliers),
        outliers
    };
}

// -----------------------------------------------------------------------------
// Line-series preparation
// -----------------------------------------------------------------------------

function createLineSeries(labels, values) {
    if (labels.length !== values.length) {
        throw new Error("Line-series labels and values must have equal length.");
    }

    return labels.map((label, index) => ({
        label,
        value: Number(values[index])
    }));
}

// -----------------------------------------------------------------------------
// Beginner example
// -----------------------------------------------------------------------------

function beginnerExample() {
    console.log("\n=== BEGINNER EXAMPLE ===");

    const scores = [48, 52, 60, 61, 65, 68, 70, 73, 78, 82];

    console.log("Scores:", scores);
    console.log("Mean:", mean(scores).toFixed(2));
    console.log("Median:", median(scores).toFixed(2));
    console.log("Description:", describe(scores));
}

// -----------------------------------------------------------------------------
// Intermediate example: relationships and distributions
// -----------------------------------------------------------------------------

function intermediateExample() {
    console.log("\n=== INTERMEDIATE EXAMPLE ===");

    const studyHours = [1, 2, 3, 4, 5, 6, 7, 8];
    const scores = [48, 53, 59, 64, 70, 74, 82, 88];

    const correlation = pearsonCorrelation(studyHours, scores);

    console.log(
        "Study-hour/score correlation:",
        correlation.toFixed(3)
    );

    const histogram = createHistogram(scores, 4);

    console.log("Histogram bins:");
    for (const bin of histogram.bins) {
        console.log(
            `[${bin.lower.toFixed(1)}, ${bin.upper.toFixed(1)}]: ${bin.count}`
        );
    }

    const box = calculateBoxPlot(scores);
    console.log("Box plot statistics:", box);
}

// -----------------------------------------------------------------------------
// Advanced example: SVG rendering
// -----------------------------------------------------------------------------

function scale(value, domainMin, domainMax, rangeMin, rangeMax) {
    if (domainMax === domainMin) {
        return (rangeMin + rangeMax) / 2;
    }

    return (
        rangeMin +
        ((value - domainMin) / (domainMax - domainMin)) *
        (rangeMax - rangeMin)
    );
}

function escapeXml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&apos;");
}

function renderHistogramSvg(values, width = 760, height = 420) {
    const histogram = createHistogram(values, 18);

    const margin = {
        top: 50,
        right: 30,
        bottom: 60,
        left: 65
    };

    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const maximumCount = Math.max(
        ...histogram.bins.map(bin => bin.count)
    );

    const bars = histogram.bins.map((bin, index) => {
        const x = margin.left +
            (index / histogram.bins.length) * plotWidth;

        const barWidth = plotWidth / histogram.bins.length - 2;

        const barHeight = scale(
            bin.count,
            0,
            maximumCount,
            0,
            plotHeight
        );

        const y = margin.top + plotHeight - barHeight;

        return `
            <rect
                x="${x.toFixed(2)}"
                y="${y.toFixed(2)}"
                width="${barWidth.toFixed(2)}"
                height="${barHeight.toFixed(2)}"
                fill="steelblue"
                opacity="0.75"
            />
        `;
    }).join("");

    return `
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="${width}"
            height="${height}"
            viewBox="0 0 ${width} ${height}"
        >
            <rect width="100%" height="100%" fill="white"/>

            <text
                x="${width / 2}"
                y="30"
                text-anchor="middle"
                font-size="20"
                font-family="Arial"
            >
                Distribution Histogram
            </text>

            <line
                x1="${margin.left}"
                y1="${margin.top + plotHeight}"
                x2="${width - margin.right}"
                y2="${margin.top + plotHeight}"
                stroke="black"
            />

            <line
                x1="${margin.left}"
                y1="${margin.top}"
                x2="${margin.left}"
                y2="${margin.top + plotHeight}"
                stroke="black"
            />

            ${bars}

            <text
                x="${width / 2}"
                y="${height - 15}"
                text-anchor="middle"
                font-size="14"
                font-family="Arial"
            >
                Value
            </text>

            <text
                x="18"
                y="${height / 2}"
                transform="rotate(-90 18 ${height / 2})"
                text-anchor="middle"
                font-size="14"
                font-family="Arial"
            >
                Frequency
            </text>
        </svg>
    `;
}

// -----------------------------------------------------------------------------
// Scatter SVG
// -----------------------------------------------------------------------------

function renderScatterSvg(xValues, yValues, width = 760, height = 420) {
    if (xValues.length !== yValues.length) {
        throw new Error("Scatter arrays must have equal length.");
    }

    const x = numericValues(xValues);
    const y = numericValues(yValues);

    const margin = {
        top: 50,
        right: 30,
        bottom: 60,
        left: 65
    };

    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const xMin = Math.min(...x);
    const xMax = Math.max(...x);
    const yMin = Math.min(...y);
    const yMax = Math.max(...y);

    const points = x.map((xValue, index) => {
        const xPixel = scale(
            xValue,
            xMin,
            xMax,
            margin.left,
            margin.left + plotWidth
        );

        const yPixel = scale(
            y[index],
            yMin,
            yMax,
            margin.top + plotHeight,
            margin.top
        );

        return `
            <circle
                cx="${xPixel.toFixed(2)}"
                cy="${yPixel.toFixed(2)}"
                r="5"
                fill="darkorange"
                opacity="0.75"
            />
        `;
    }).join("");

    return `
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="${width}"
            height="${height}"
            viewBox="0 0 ${width} ${height}"
        >
            <rect width="100%" height="100%" fill="white"/>

            <text
                x="${width / 2}"
                y="30"
                text-anchor="middle"
                font-size="20"
                font-family="Arial"
            >
                Scatter Plot
            </text>

            <line
                x1="${margin.left}"
                y1="${margin.top + plotHeight}"
                x2="${width - margin.right}"
                y2="${margin.top + plotHeight}"
                stroke="black"
            />

            <line
                x1="${margin.left}"
                y1="${margin.top}"
                x2="${margin.left}"
                y2="${margin.top + plotHeight}"
                stroke="black"
            />

            ${points}

            <text
                x="${width / 2}"
                y="${height - 15}"
                text-anchor="middle"
                font-size="14"
                font-family="Arial"
            >
                X
            </text>

            <text
                x="18"
                y="${height / 2}"
                transform="rotate(-90 18 ${height / 2})"
                text-anchor="middle"
                font-size="14"
                font-family="Arial"
            >
                Y
            </text>
        </svg>
    `;
}

// -----------------------------------------------------------------------------
// Line SVG
// -----------------------------------------------------------------------------

function renderLineSvg(values, width = 760, height = 420) {
    const data = numericValues(values);

    if (data.length < 2) {
        throw new Error("A line plot requires at least two observations.");
    }

    const margin = {
        top: 50,
        right: 30,
        bottom: 60,
        left: 65
    };

    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const yMin = Math.min(...data);
    const yMax = Math.max(...data);

    const points = data.map((value, index) => {
        const x = scale(
            index,
            0,
            data.length - 1,
            margin.left,
            margin.left + plotWidth
        );

        const y = scale(
            value,
            yMin,
            yMax,
            margin.top + plotHeight,
            margin.top
        );

        return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");

    return `
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="${width}"
            height="${height}"
            viewBox="0 0 ${width} ${height}"
        >
            <rect width="100%" height="100%" fill="white"/>

            <text
                x="${width / 2}"
                y="30"
                text-anchor="middle"
                font-size="20"
                font-family="Arial"
            >
                Ordered Time Series
            </text>

            <line
                x1="${margin.left}"
                y1="${margin.top + plotHeight}"
                x2="${width - margin.right}"
                y2="${margin.top + plotHeight}"
                stroke="black"
            />

            <line
                x1="${margin.left}"
                y1="${margin.top}"
                x2="${margin.left}"
                y2="${margin.top + plotHeight}"
                stroke="black"
            />

            <polyline
                points="${points}"
                fill="none"
                stroke="seagreen"
                stroke-width="3"
            />
        </svg>
    `;
}

// -----------------------------------------------------------------------------
// Browser demonstration
// -----------------------------------------------------------------------------

function browserCanvasExample() {
    if (typeof document === "undefined") {
        return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = 800;
    canvas.height = 450;
    canvas.style.border = "1px solid #ccc";

    document.body.appendChild(canvas);

    const context = canvas.getContext("2d");

    // Canvas is useful for custom browser visualization. Each drawing
    // primitive becomes a graphical object controlled by JavaScript.
    context.font = "20px Arial";
    context.fillText("Canvas Data Visualization", 260, 35);

    const values = [30, 45, 40, 70, 60, 85, 72, 95];

    const maximum = Math.max(...values);
    const chartLeft = 70;
    const chartBottom = 390;
    const chartHeight = 300;
    const barWidth = 70;

    values.forEach((value, index) => {
        const barHeight = (value / maximum) * chartHeight;
        const x = chartLeft + index * 80;
        const y = chartBottom - barHeight;

        context.fillStyle = "steelblue";
        context.fillRect(x, y, barWidth, barHeight);

        context.fillStyle = "black";
        context.fillText(
            String(value),
            x + 25,
            y - 8
        );
    });

    context.beginPath();
    context.moveTo(chartLeft, chartBottom);
    context.lineTo(700, chartBottom);
    context.stroke();
}

// -----------------------------------------------------------------------------
// Large-data aggregation
// -----------------------------------------------------------------------------

function createRandomNormalData(size, center = 50, spread = 10) {
    const result = [];

    // Box-Muller transformation generates approximately normal random values.
    for (let index = 0; index < size; index += 1) {
        let u1 = 0;
        let u2 = 0;

        // Math.random() can theoretically produce zero.
        // Avoid log(0) in the transformation.
        while (u1 === 0) {
            u1 = Math.random();
        }

        while (u2 === 0) {
            u2 = Math.random();
        }

        const standardNormal =
            Math.sqrt(-2 * Math.log(u1)) *
            Math.cos(2 * Math.PI * u2);

        result.push(center + standardNormal * spread);
    }

    return result;
}

function performanceExample() {
    console.log("\n=== PERFORMANCE EXAMPLE ===");

    const largeDataset = createRandomNormalData(100000);

    console.time("histogram aggregation");
    const histogram = createHistogram(largeDataset, 50);
    console.timeEnd("histogram aggregation");

    console.log(
        "Raw observations:",
        largeDataset.length
    );

    console.log(
        "Histogram bins:",
        histogram.bins.length
    );

    console.log(
        "Aggregation reduces 100,000 observations to 50 displayed intervals."
    );
}

// -----------------------------------------------------------------------------
// Validation and tests
// -----------------------------------------------------------------------------

function runTests() {
    console.log("\n=== TESTS ===");

    assert(mean([1, 2, 3, 4, 5]) === 3, "Mean test failed.");
    assert(median([1, 2, 3, 4, 5]) === 3, "Median test failed.");
    assert(median([1, 2, 3, 4]) === 2.5, "Even median test failed.");
    assert(quantile([1, 2, 3, 4, 5], 0.5) === 3, "Quantile test failed.");

    const correlation = pearsonCorrelation(
        [1, 2, 3, 4],
        [2, 4, 6, 8]
    );

    assert(
        Math.abs(correlation - 1) < 1e-12,
        "Correlation test failed."
    );

    const box = calculateBoxPlot([
        1, 2, 3, 4, 5, 100
    ]);

    assert(
        box.outliers.includes(100),
        "Outlier detection test failed."
    );

    let failedAsExpected = false;

    try {
        pearsonCorrelation([1, 2], [1]);
    } catch {
        failedAsExpected = true;
    }

    assert(
        failedAsExpected,
        "Validation test failed."
    );

    console.log("All tests passed.");
}

// -----------------------------------------------------------------------------
// Main execution
// -----------------------------------------------------------------------------

function main() {
    console.log("DATA VISUALIZATION STUDY PROGRAM");
    console.log("Histograms | Scatter Plots | Box Plots | Line Plots | Distributions");

    beginnerExample();
    intermediateExample();

    const studyHours = [1, 2, 3, 4, 5, 6, 7, 8];
    const scores = [48, 53, 59, 64, 70, 74, 82, 88];

    const histogramSvg = renderHistogramSvg(scores);
    const scatterSvg = renderScatterSvg(studyHours, scores);
    const lineSvg = renderLineSvg(scores);

    console.log("\nSVG outputs generated in memory:");
    console.log("  Histogram SVG characters:", histogramSvg.length);
    console.log("  Scatter SVG characters:", scatterSvg.length);
    console.log("  Line SVG characters:", lineSvg.length);

    performanceExample();
    runTests();

    if (typeof document !== "undefined") {
        browserCanvasExample();

        const heading = document.createElement("h2");
        heading.textContent = "Data Visualization JavaScript Demonstration";
        document.body.prepend(heading);

        const description = document.createElement("p");
        description.textContent =
            "Open the browser console to inspect statistical calculations.";
        document.body.prepend(description);
    }
}

if (typeof module !== "undefined" && require.main === module) {
    main();
} else {
    main();
}
