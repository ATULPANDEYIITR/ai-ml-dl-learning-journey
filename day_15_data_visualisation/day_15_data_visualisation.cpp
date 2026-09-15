/*
Data Visualization Case Study
=============================

Scenario:
An operations analytics team monitors a manufacturing facility. The team
receives measurements for production time, machine temperature, defect rate,
and daily output.

The program demonstrates how the underlying statistical structures used by
histograms, scatter plots, box plots, line plots, and distributions can be
implemented without an external visualization library.

The program does not attempt to draw a graphical window. Instead, it performs
the analytical preparation and exports a self-contained SVG dashboard.

This design separates:
    Data model
        ->
    Statistical analysis
        ->
    Visualization-ready structures
        ->
    SVG rendering
        ->
    Output file

Compile:
    g++ -std=c++17 -O2 -Wall -Wextra -pedantic visualization_case_study.cpp -o visualization_case_study

Run:
    ./visualization_case_study

Windows:
    visualization_case_study.exe
*/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using std::size_t;

// -----------------------------------------------------------------------------
// Data model
// -----------------------------------------------------------------------------

struct ProductionRecord {
    int day;
    double productionHours;
    double temperature;
    double defectRate;
    int unitsProduced;
};

// -----------------------------------------------------------------------------
// Statistical result structures
// -----------------------------------------------------------------------------

struct DescriptiveStatistics {
    size_t count{};
    double minimum{};
    double q1{};
    double median{};
    double mean{};
    double q3{};
    double maximum{};
    double iqr{};
    double standardDeviation{};
};

struct BoxPlotStatistics {
    double minimum{};
    double q1{};
    double median{};
    double q3{};
    double maximum{};
    double iqr{};
    double lowerFence{};
    double upperFence{};
    double lowerWhisker{};
    double upperWhisker{};
    std::vector<double> outliers;
};

struct HistogramBin {
    double lower{};
    double upper{};
    size_t count{};
};

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

void requireNonEmpty(const std::vector<double>& values,
                     const std::string& operation) {
    if (values.empty()) {
        throw std::invalid_argument(
            operation + " requires at least one observation."
        );
    }
}

void requireEqualLength(const std::vector<double>& first,
                        const std::vector<double>& second,
                        const std::string& operation) {
    if (first.size() != second.size()) {
        throw std::invalid_argument(
            operation + " requires equally sized vectors."
        );
    }
}

void validateProductionRecord(const ProductionRecord& record) {
    if (record.day < 1) {
        throw std::invalid_argument("Day must be positive.");
    }

    if (!std::isfinite(record.productionHours) ||
        record.productionHours < 0 ||
        record.productionHours > 24) {
        throw std::invalid_argument(
            "Production hours must be between 0 and 24."
        );
    }

    if (!std::isfinite(record.temperature)) {
        throw std::invalid_argument("Temperature must be finite.");
    }

    if (!std::isfinite(record.defectRate) ||
        record.defectRate < 0 ||
        record.defectRate > 100) {
        throw std::invalid_argument(
            "Defect rate must be between 0 and 100 percent."
        );
    }

    if (record.unitsProduced < 0) {
        throw std::invalid_argument(
            "Units produced cannot be negative."
        );
    }
}

// -----------------------------------------------------------------------------
// Quantiles and descriptive statistics
// -----------------------------------------------------------------------------

double quantile(std::vector<double> values, double probability) {
    requireNonEmpty(values, "Quantile");

    if (probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument(
            "Quantile probability must be between 0 and 1."
        );
    }

    std::sort(values.begin(), values.end());

    const double position =
        probability * static_cast<double>(values.size() - 1);

    const size_t lower =
        static_cast<size_t>(std::floor(position));

    const size_t upper =
        static_cast<size_t>(std::ceil(position));

    if (lower == upper) {
        return values[lower];
    }

    const double fraction =
        position - static_cast<double>(lower);

    return values[lower] +
           fraction * (values[upper] - values[lower]);
}

double arithmeticMean(const std::vector<double>& values) {
    requireNonEmpty(values, "Mean");

    const double total =
        std::accumulate(values.begin(), values.end(), 0.0);

    return total / static_cast<double>(values.size());
}

double sampleStandardDeviation(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }

    const double average = arithmeticMean(values);

    double squaredDifferenceSum = 0.0;

    for (double value : values) {
        const double difference = value - average;
        squaredDifferenceSum += difference * difference;
    }

    return std::sqrt(
        squaredDifferenceSum /
        static_cast<double>(values.size() - 1)
    );
}

DescriptiveStatistics describe(
    const std::vector<double>& values
) {
    requireNonEmpty(values, "Descriptive statistics");

    const auto [minimumIterator, maximumIterator] =
        std::minmax_element(values.begin(), values.end());

    const double q1 = quantile(values, 0.25);
    const double median = quantile(values, 0.50);
    const double q3 = quantile(values, 0.75);

    DescriptiveStatistics result;

    result.count = values.size();
    result.minimum = *minimumIterator;
    result.q1 = q1;
    result.median = median;
    result.mean = arithmeticMean(values);
    result.q3 = q3;
    result.maximum = *maximumIterator;
    result.iqr = q3 - q1;
    result.standardDeviation = sampleStandardDeviation(values);

    return result;
}

// -----------------------------------------------------------------------------
// Box plot analysis
// -----------------------------------------------------------------------------

BoxPlotStatistics calculateBoxPlot(
    const std::vector<double>& values
) {
    requireNonEmpty(values, "Box plot");

    const DescriptiveStatistics statistics =
        describe(values);

    BoxPlotStatistics result;

    result.minimum = statistics.minimum;
    result.q1 = statistics.q1;
    result.median = statistics.median;
    result.q3 = statistics.q3;
    result.maximum = statistics.maximum;
    result.iqr = statistics.iqr;

    result.lowerFence =
        result.q1 - 1.5 * result.iqr;

    result.upperFence =
        result.q3 + 1.5 * result.iqr;

    bool foundNonOutlier = false;

    double lowerWhisker =
        std::numeric_limits<double>::infinity();

    double upperWhisker =
        -std::numeric_limits<double>::infinity();

    for (double value : values) {
        if (value < result.lowerFence ||
            value > result.upperFence) {
            result.outliers.push_back(value);
            continue;
        }

        foundNonOutlier = true;
        lowerWhisker = std::min(lowerWhisker, value);
        upperWhisker = std::max(upperWhisker, value);
    }

    if (!foundNonOutlier) {
        // This is unusual but possible with pathological input.
        result.lowerWhisker = result.minimum;
        result.upperWhisker = result.maximum;
    } else {
        result.lowerWhisker = lowerWhisker;
        result.upperWhisker = upperWhisker;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Histogram
// -----------------------------------------------------------------------------

std::vector<HistogramBin> createHistogram(
    const std::vector<double>& values,
    size_t binCount
) {
    requireNonEmpty(values, "Histogram");

    if (binCount == 0) {
        throw std::invalid_argument(
            "Histogram must contain at least one bin."
        );
    }

    const auto [minimumIterator, maximumIterator] =
        std::minmax_element(values.begin(), values.end());

    const double minimum = *minimumIterator;
    const double maximum = *maximumIterator;

    if (minimum == maximum) {
        return {
            HistogramBin{
                minimum - 0.5,
                maximum + 0.5,
                values.size()
            }
        };
    }

    const double width =
        (maximum - minimum) /
        static_cast<double>(binCount);

    std::vector<HistogramBin> bins;

    bins.reserve(binCount);

    for (size_t index = 0; index < binCount; ++index) {
        bins.push_back(
            HistogramBin{
                minimum + static_cast<double>(index) * width,
                minimum + static_cast<double>(index + 1) * width,
                0
            }
        );
    }

    for (double value : values) {
        size_t index = static_cast<size_t>(
            std::floor((value - minimum) / width)
        );

        // Maximum can mathematically produce binCount.
        if (index >= binCount) {
            index = binCount - 1;
        }

        ++bins[index].count;
    }

    return bins;
}

// -----------------------------------------------------------------------------
// Correlation
// -----------------------------------------------------------------------------

double pearsonCorrelation(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    requireEqualLength(x, y, "Pearson correlation");

    if (x.size() < 2) {
        throw std::invalid_argument(
            "Pearson correlation requires at least two observations."
        );
    }

    const double xMean = arithmeticMean(x);
    const double yMean = arithmeticMean(y);

    double numerator = 0.0;
    double xSquared = 0.0;
    double ySquared = 0.0;

    for (size_t index = 0; index < x.size(); ++index) {
        const double xDifference = x[index] - xMean;
        const double yDifference = y[index] - yMean;

        numerator += xDifference * yDifference;
        xSquared += xDifference * xDifference;
        ySquared += yDifference * yDifference;
    }

    const double denominator =
        std::sqrt(xSquared * ySquared);

    if (denominator == 0.0) {
        throw std::invalid_argument(
            "Correlation is undefined for a constant variable."
        );
    }

    return numerator / denominator;
}

// -----------------------------------------------------------------------------
// Synthetic production dataset
// -----------------------------------------------------------------------------

std::vector<ProductionRecord> createProductionDataset(
    int numberOfDays
) {
    if (numberOfDays <= 0) {
        throw std::invalid_argument(
            "The dataset must contain at least one day."
        );
    }

    std::mt19937 generator(42);

    std::normal_distribution<double> temperatureNoise(0.0, 2.0);
    std::normal_distribution<double> defectNoise(0.0, 0.25);
    std::normal_distribution<double> productionNoise(0.0, 0.5);

    std::vector<ProductionRecord> records;
    records.reserve(static_cast<size_t>(numberOfDays));

    for (int day = 1; day <= numberOfDays; ++day) {
        const double temperature =
            22.0 +
            0.04 * static_cast<double>(day) +
            temperatureNoise(generator);

        const double productionHours =
            std::clamp(
                8.0 + productionNoise(generator),
                0.0,
                24.0
            );

        const double thermalStress =
            std::abs(temperature - 22.0);

        const double defectRate =
            std::clamp(
                1.0 +
                0.25 * thermalStress +
                defectNoise(generator),
                0.0,
                100.0
            );

        const int unitsProduced =
            std::max(
                0,
                static_cast<int>(
                    std::round(
                        productionHours * 100.0 -
                        defectRate * 8.0
                    )
                )
            );

        ProductionRecord record{
            day,
            productionHours,
            temperature,
            defectRate,
            unitsProduced
        };

        validateProductionRecord(record);
        records.push_back(record);
    }

    // Deliberate anomaly for demonstrating outlier detection.
    if (records.size() >= 50) {
        records[49].defectRate = 12.0;
    }

    return records;
}

// -----------------------------------------------------------------------------
// Data extraction
// -----------------------------------------------------------------------------

std::vector<double> extractTemperatures(
    const std::vector<ProductionRecord>& records
) {
    std::vector<double> values;
    values.reserve(records.size());

    for (const auto& record : records) {
        values.push_back(record.temperature);
    }

    return values;
}

std::vector<double> extractDefectRates(
    const std::vector<ProductionRecord>& records
) {
    std::vector<double> values;
    values.reserve(records.size());

    for (const auto& record : records) {
        values.push_back(record.defectRate);
    }

    return values;
}

std::vector<double> extractProductionHours(
    const std::vector<ProductionRecord>& records
) {
    std::vector<double> values;
    values.reserve(records.size());

    for (const auto& record : records) {
        values.push_back(record.productionHours);
    }

    return values;
}

std::vector<double> extractUnitsProduced(
    const std::vector<ProductionRecord>& records
) {
    std::vector<double> values;
    values.reserve(records.size());

    for (const auto& record : records) {
        values.push_back(
            static_cast<double>(record.unitsProduced)
        );
    }

    return values;
}

// -----------------------------------------------------------------------------
// SVG helpers
// -----------------------------------------------------------------------------

std::string escapeXml(const std::string& text) {
    std::string result;

    for (char character : text) {
        switch (character) {
            case '&':
                result += "&amp;";
                break;
            case '<':
                result += "&lt;";
                break;
            case '>':
                result += "&gt;";
                break;
            case '"':
                result += "&quot;";
                break;
            case '\'':
                result += "&apos;";
                break;
            default:
                result += character;
        }
    }

    return result;
}

double scaleValue(
    double value,
    double domainMinimum,
    double domainMaximum,
    double rangeMinimum,
    double rangeMaximum
) {
    if (domainMaximum == domainMinimum) {
        return (rangeMinimum + rangeMaximum) / 2.0;
    }

    return rangeMinimum +
           ((value - domainMinimum) /
            (domainMaximum - domainMinimum)) *
           (rangeMaximum - rangeMinimum);
}

std::string number(double value, int precision = 2) {
    std::ostringstream stream;
    stream << std::fixed
           << std::setprecision(precision)
           << value;

    return stream.str();
}

// -----------------------------------------------------------------------------
// SVG histogram
// -----------------------------------------------------------------------------

std::string histogramSvg(
    const std::vector<double>& values,
    int width,
    int height,
    const std::string& title
) {
    const auto bins = createHistogram(values, 18);

    const double left = 65.0;
    const double right = 25.0;
    const double top = 55.0;
    const double bottom = 55.0;

    const double plotWidth =
        static_cast<double>(width) - left - right;

    const double plotHeight =
        static_cast<double>(height) - top - bottom;

    size_t maximumCount = 0;

    for (const auto& bin : bins) {
        maximumCount =
            std::max(maximumCount, bin.count);
    }

    std::ostringstream svg;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << width << "\" "
        << "height=\"" << height << "\">";

    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>";

    svg << "<text x=\"" << width / 2
        << "\" y=\"30\" text-anchor=\"middle\" "
        << "font-family=\"Arial\" font-size=\"20\">"
        << escapeXml(title)
        << "</text>";

    svg << "<line x1=\"" << left
        << "\" y1=\"" << top + plotHeight
        << "\" x2=\"" << left + plotWidth
        << "\" y2=\"" << top + plotHeight
        << "\" stroke=\"black\"/>";

    svg << "<line x1=\"" << left
        << "\" y1=\"" << top
        << "\" x2=\"" << left
        << "\" y2=\"" << top + plotHeight
        << "\" stroke=\"black\"/>";

    const double barWidth =
        plotWidth / static_cast<double>(bins.size());

    for (size_t index = 0; index < bins.size(); ++index) {
        const auto& bin = bins[index];

        const double barHeight =
            maximumCount == 0
                ? 0.0
                : (static_cast<double>(bin.count) /
                   static_cast<double>(maximumCount)) *
                  plotHeight;

        const double x =
            left + static_cast<double>(index) * barWidth;

        const double y =
            top + plotHeight - barHeight;

        svg << "<rect x=\"" << number(x)
            << "\" y=\"" << number(y)
            << "\" width=\"" << number(barWidth - 1.0)
            << "\" height=\"" << number(barHeight)
            << "\" fill=\"steelblue\" opacity=\"0.75\"/>";
    }

    svg << "</svg>";

    return svg.str();
}

// -----------------------------------------------------------------------------
// SVG scatter plot
// -----------------------------------------------------------------------------

std::string scatterSvg(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int width,
    int height,
    const std::string& title
) {
    requireEqualLength(x, y, "Scatter plot");

    if (x.empty()) {
        throw std::invalid_argument(
            "Scatter plot cannot be empty."
        );
    }

    const auto [xMinimumIterator, xMaximumIterator] =
        std::minmax_element(x.begin(), x.end());

    const auto [yMinimumIterator, yMaximumIterator] =
        std::minmax_element(y.begin(), y.end());

    const double xMinimum = *xMinimumIterator;
    const double xMaximum = *xMaximumIterator;
    const double yMinimum = *yMinimumIterator;
    const double yMaximum = *yMaximumIterator;

    const double left = 65.0;
    const double right = 25.0;
    const double top = 55.0;
    const double bottom = 55.0;

    const double plotWidth =
        static_cast<double>(width) - left - right;

    const double plotHeight =
        static_cast<double>(height) - top - bottom;

    std::ostringstream svg;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << width << "\" "
        << "height=\"" << height << "\">";

    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>";

    svg << "<text x=\"" << width / 2
        << "\" y=\"30\" text-anchor=\"middle\" "
        << "font-family=\"Arial\" font-size=\"20\">"
        << escapeXml(title)
        << "</text>";

    svg << "<line x1=\"" << left
        << "\" y1=\"" << top + plotHeight
        << "\" x2=\"" << left + plotWidth
        << "\" y2=\"" << top + plotHeight
        << "\" stroke=\"black\"/>";

    svg << "<line x1=\"" << left
        << "\" y1=\"" << top
        << "\" x2=\"" << left
        << "\" y2=\"" << top + plotHeight
        << "\" stroke=\"black\"/>";

    for (size_t index = 0; index < x.size(); ++index) {
        const double xPixel =
            scaleValue(
                x[index],
                xMinimum,
                xMaximum,
                left,
                left + plotWidth
            );

        const double yPixel =
            scaleValue(
                y[index],
                yMinimum,
                yMaximum,
                top + plotHeight,
                top
            );

        svg << "<circle cx=\"" << number(xPixel)
            << "\" cy=\"" << number(yPixel)
            << "\" r=\"4\" fill=\"darkorange\" opacity=\"0.75\"/>";
    }

    svg << "</svg>";

    return svg.str();
}

// -----------------------------------------------------------------------------
// SVG line plot
// -----------------------------------------------------------------------------

std::string lineSvg(
    const std::vector<double>& values,
    int width,
    int height,
    const std::string& title
) {
    requireNonEmpty(values, "Line plot");

    if (values.size() < 2) {
        throw std::invalid_argument(
            "Line plot requires at least two observations."
        );
    }

    const auto [minimumIterator, maximumIterator] =
        std::minmax_element(values.begin(), values.end());

    const double minimum = *minimumIterator;
    const double maximum = *maximumIterator;

    const double left = 65.0;
    const double right = 25.0;
    const double top = 55.0;
    const double bottom = 55.0;

    const double plotWidth =
        static_cast<double>(width) - left - right;

    const double plotHeight =
        static_cast<double>(height) - top - bottom;

    std::ostringstream svg;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << width << "\" "
        << "height=\"" << height << "\">";

    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>";

    svg << "<text x=\"" << width / 2
        << "\" y=\"30\" text-anchor=\"middle\" "
        << "font-family=\"Arial\" font-size=\"20\">"
        << escapeXml(title)
        << "</text>";

    svg << "<line x1=\"" << left
        << "\" y1=\"" << top + plotHeight
        << "\" x2=\"" << left + plotWidth
        << "\" y2=\"" << top + plotHeight
        << "\" stroke=\"black\"/>";

    svg << "<line x1=\"" << left
        << "\" y1=\"" << top
        << "\" x2=\"" << left
        << "\" y2=\"" << top + plotHeight
        << "\" stroke=\"black\"/>";

    svg << "<polyline points=\"";

    for (size_t index = 0; index < values.size(); ++index) {
        const double xPixel =
            scaleValue(
                static_cast<double>(index),
                0.0,
                static_cast<double>(values.size() - 1),
                left,
                left + plotWidth
            );

        const double yPixel =
            scaleValue(
                values[index],
                minimum,
                maximum,
                top + plotHeight,
                top
            );

        svg << number(xPixel)
            << ","
            << number(yPixel)
            << " ";
    }

    svg << "\" fill=\"none\" stroke=\"seagreen\" "
        << "stroke-width=\"3\"/>";

    svg << "</svg>";

    return svg.str();
}

// -----------------------------------------------------------------------------
// Dashboard generation
// -----------------------------------------------------------------------------

void writeDashboard(
    const std::vector<ProductionRecord>& records,
    const std::string& filename
) {
    const auto temperatures =
        extractTemperatures(records);

    const auto defectRates =
        extractDefectRates(records);

    const auto productionHours =
        extractProductionHours(records);

    const auto unitsProduced =
        extractUnitsProduced(records);

    const double temperatureDefectCorrelation =
        pearsonCorrelation(
            temperatures,
            defectRates
        );

    const std::string histogram =
        histogramSvg(
            defectRates,
            760,
            420,
            "Defect Rate Distribution"
        );

    const std::string scatter =
        scatterSvg(
            temperatures,
            defectRates,
            760,
            420,
            "Temperature vs Defect Rate"
        );

    const std::string line =
        lineSvg(
            unitsProduced,
            760,
            420,
            "Daily Production Output"
        );

    const auto statistics =
        describe(defectRates);

    const auto box =
        calculateBoxPlot(defectRates);

    std::ofstream output(filename);

    if (!output) {
        throw std::runtime_error(
            "Unable to open dashboard output file."
        );
    }

    output << "<!DOCTYPE html>";
    output << "<html><head>";
    output << "<meta charset=\"UTF-8\">";
    output << "<title>Production Analytics Dashboard</title>";
    output << "<style>";
    output << "body{font-family:Arial,sans-serif;margin:30px;}";
    output << ".chart{margin-bottom:30px;border:1px solid #ddd;padding:10px;}";
    output << "table{border-collapse:collapse;}";
    output << "td,th{border:1px solid #ccc;padding:8px;}";
    output << "</style>";
    output << "</head><body>";

    output << "<h1>Production Analytics Dashboard</h1>";

    output << "<p>Observations: "
           << records.size()
           << "</p>";

    output << "<p>Temperature/defect correlation: "
           << number(temperatureDefectCorrelation, 3)
           << "</p>";

    output << "<div class=\"chart\">"
           << histogram
           << "</div>";

    output << "<div class=\"chart\">"
           << scatter
           << "</div>";

    output << "<div class=\"chart\">"
           << line
           << "</div>";

    output << "<h2>Defect Rate Statistics</h2>";

    output << "<table>";
    output << "<tr><th>Statistic</th><th>Value</th></tr>";

    output << "<tr><td>Mean</td><td>"
           << number(statistics.mean)
           << "</td></tr>";

    output << "<tr><td>Median</td><td>"
           << number(statistics.median)
           << "</td></tr>";

    output << "<tr><td>Q1</td><td>"
           << number(statistics.q1)
           << "</td></tr>";

    output << "<tr><td>Q3</td><td>"
           << number(statistics.q3)
           << "</td></tr>";

    output << "<tr><td>IQR</td><td>"
           << number(statistics.iqr)
           << "</td></tr>";

    output << "<tr><td>Lower fence</td><td>"
           << number(box.lowerFence)
           << "</td></tr>";

    output << "<tr><td>Upper fence</td><td>"
           << number(box.upperFence)
           << "</td></tr>";

    output << "<tr><td>Potential outliers</td><td>"
           << box.outliers.size()
           << "</td></tr>";

    output << "</table>";

    output << "</body></html>";
}

// -----------------------------------------------------------------------------
// Case-study analysis
// -----------------------------------------------------------------------------

void printCaseStudyAnalysis(
    const std::vector<ProductionRecord>& records
) {
    const auto temperatures =
        extractTemperatures(records);

    const auto defectRates =
        extractDefectRates(records);

    const auto productionHours =
        extractProductionHours(records);

    const auto unitsProduced =
        extractUnitsProduced(records);

    const auto defectStatistics =
        describe(defectRates);

    const auto defectBox =
        calculateBoxPlot(defectRates);

    const double temperatureCorrelation =
        pearsonCorrelation(
            temperatures,
            defectRates
        );

    const double hoursCorrelation =
        pearsonCorrelation(
            productionHours,
            unitsProduced
        );

    std::cout << "\n========================================\n";
    std::cout << "PRODUCTION CASE STUDY\n";
    std::cout << "========================================\n";

    std::cout << "Records: "
              << records.size()
              << "\n";

    std::cout << std::fixed
              << std::setprecision(3);

    std::cout << "Temperature / defect correlation: "
              << temperatureCorrelation
              << "\n";

    std::cout << "Production hours / output correlation: "
              << hoursCorrelation
              << "\n";

    std::cout << "\nDefect rate statistics\n";
    std::cout << "Mean: "
              << defectStatistics.mean
              << "%\n";

    std::cout << "Median: "
              << defectStatistics.median
              << "%\n";

    std::cout << "Q1: "
              << defectStatistics.q1
              << "%\n";

    std::cout << "Q3: "
              << defectStatistics.q3
              << "%\n";

    std::cout << "IQR: "
              << defectStatistics.iqr
              << "\n";

    std::cout << "Potential outliers: "
              << defectBox.outliers.size()
              << "\n";

    for (double outlier : defectBox.outliers) {
        std::cout << "  Outlier candidate: "
                  << outlier
                  << "%\n";
    }

    std::cout << "\nInterpretation rules:\n";
    std::cout << "  Histogram -> distribution shape and concentration.\n";
    std::cout << "  Scatter   -> relationship between paired variables.\n";
    std::cout << "  Box plot  -> quartiles, spread, and potential outliers.\n";
    std::cout << "  Line plot -> ordered behavior over observations.\n";
    std::cout << "  Correlation does not establish causation.\n";
}

// -----------------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------------

void demonstrateEdgeCases() {
    std::cout << "\n========================================\n";
    std::cout << "EDGE CASES\n";
    std::cout << "========================================\n";

    const std::vector<double> constantValues{
        5.0, 5.0, 5.0, 5.0
    };

    const auto constantHistogram =
        createHistogram(constantValues, 10);

    std::cout << "Constant-data histogram bins: "
              << constantHistogram.size()
              << "\n";

    const auto constantBox =
        calculateBoxPlot(constantValues);

    std::cout << "Constant-data IQR: "
              << constantBox.iqr
              << "\n";

    try {
        pearsonCorrelation(
            {1.0, 1.0, 1.0},
            {2.0, 3.0, 4.0}
        );
    } catch (const std::exception& exception) {
        std::cout << "Expected constant-variable error: "
                  << exception.what()
                  << "\n";
    }

    try {
        createHistogram({}, 10);
    } catch (const std::exception& exception) {
        std::cout << "Expected empty-data error: "
                  << exception.what()
                  << "\n";
    }

    try {
        pearsonCorrelation(
            {1.0, 2.0},
            {1.0}
        );
    } catch (const std::exception& exception) {
        std::cout << "Expected length-mismatch error: "
                  << exception.what()
                  << "\n";
    }
}

// -----------------------------------------------------------------------------
// Complexity discussion
// -----------------------------------------------------------------------------

void printComplexityNotes() {
    std::cout << "\n========================================\n";
    std::cout << "COMPLEXITY NOTES\n";
    std::cout << "========================================\n";

    std::cout << "Mean: O(n)\n";
    std::cout << "Standard deviation: O(n)\n";
    std::cout << "Pearson correlation: O(n)\n";
    std::cout << "Histogram construction: O(n + b)\n";
    std::cout << "Quantile in this implementation: O(n log n)\n";
    std::cout << "Box plot calculation: O(n log n)\n";
    std::cout << "SVG generation: proportional to rendered objects.\n";

    std::cout << "\nMemory considerations:\n";
    std::cout << "  Quantile sorting copies the input vector.\n";
    std::cout << "  Histogram bins use O(b) additional storage.\n";
    std::cout << "  Rendering every raw point can become expensive for very large datasets.\n";
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

void runTests() {
    std::cout << "\n========================================\n";
    std::cout << "TESTS\n";
    std::cout << "========================================\n";

    const std::vector<double> values{
        1, 2, 3, 4, 5
    };

    const auto statistics =
        describe(values);

    if (statistics.mean != 3.0) {
        throw std::runtime_error(
            "Mean test failed."
        );
    }

    if (statistics.median != 3.0) {
        throw std::runtime_error(
            "Median test failed."
        );
    }

    if (statistics.iqr != 2.0) {
        throw std::runtime_error(
            "IQR test failed."
        );
    }

    const double correlation =
        pearsonCorrelation(
            {1, 2, 3, 4},
            {2, 4, 6, 8}
        );

    if (std::abs(correlation - 1.0) > 1e-12) {
        throw std::runtime_error(
            "Correlation test failed."
        );
    }

    const auto outlierBox =
        calculateBoxPlot(
            {1, 2, 3, 4, 5, 100}
        );

    if (outlierBox.outliers.empty()) {
        throw std::runtime_error(
            "Outlier test failed."
        );
    }

    std::cout << "All tests passed.\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout << "DATA VISUALIZATION CASE STUDY\n";
        std::cout << "Histograms | Scatter Plots | Box Plots | "
                     "Line Plots | Distributions\n";

        const auto records =
            createProductionDataset(120);

        printCaseStudyAnalysis(records);

        writeDashboard(
            records,
            "production_visualization_dashboard.html"
        );

        demonstrateEdgeCases();
        printComplexityNotes();
        runTests();

        std::cout << "\nDashboard written to:\n";
        std::cout << "  production_visualization_dashboard.html\n";

        std::cout << "\nThe dashboard contains:\n";
        std::cout << "  - Defect-rate histogram\n";
        std::cout << "  - Temperature/defect scatter plot\n";
        std::cout << "  - Daily production line plot\n";
        std::cout << "  - Descriptive statistics\n";
        std::cout << "  - IQR-based outlier information\n";

        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "Fatal error: "
                  << exception.what()
                  << "\n";

        return 1;
    }
}
