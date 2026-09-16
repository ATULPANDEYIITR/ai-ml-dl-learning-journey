/*
 * Statistics for AI: C++ Case Study
 * ==================================
 *
 * Industry-style scenario:
 * ------------------------
 * A machine-learning inference service processes prediction requests.
 * The service records prediction latency in milliseconds for operational
 * monitoring.
 *
 * The system must:
 *   - accept and validate latency observations,
 *   - calculate population statistics for an observed monitoring window,
 *   - calculate sample statistics when treating observations as a sample,
 *   - calculate mean, median, mode, variance, and standard deviation,
 *   - process observations incrementally,
 *   - identify unusually high latency using z-scores,
 *   - report grouped statistics,
 *   - handle invalid input,
 *   - demonstrate algorithmic complexity.
 *
 * Standard: C++17
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Number = double;


// ---------------------------------------------------------------------------
// Data validation
// ---------------------------------------------------------------------------

void validateData(const std::vector<Number>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    for (Number value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Every observation must be finite."
            );
        }
    }
}


// ---------------------------------------------------------------------------
// Mean
// ---------------------------------------------------------------------------

Number mean(const std::vector<Number>& values) {
    validateData(values);

    Number total = std::accumulate(
        values.begin(),
        values.end(),
        0.0
    );

    return total / static_cast<Number>(values.size());
}


// ---------------------------------------------------------------------------
// Median
// ---------------------------------------------------------------------------

Number median(std::vector<Number> values) {
    validateData(values);

    // Passing by value protects the caller's original ordering.
    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    return (values[middle - 1] + values[middle]) / 2.0;
}


// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------
//
// map is chosen for deterministic ordered output.
//
// Time complexity: O(n log n) in the worst case because each insertion into
// std::map is logarithmic.
//
// An unordered_map could provide expected O(n) counting, but output ordering
// would require additional processing.

std::vector<Number> modes(const std::vector<Number>& values) {
    validateData(values);

    std::map<Number, std::size_t> frequency;

    for (Number value : values) {
        ++frequency[value];
    }

    std::size_t highestFrequency = 0;

    for (const auto& [value, count] : frequency) {
        highestFrequency = std::max(highestFrequency, count);
    }

    if (highestFrequency == 1) {
        return {};
    }

    std::vector<Number> result;

    for (const auto& [value, count] : frequency) {
        if (count == highestFrequency) {
            result.push_back(value);
        }
    }

    return result;
}


// ---------------------------------------------------------------------------
// Population variance
// ---------------------------------------------------------------------------

Number populationVariance(const std::vector<Number>& values) {
    validateData(values);

    const Number average = mean(values);

    Number squaredDeviationTotal = 0.0;

    for (Number value : values) {
        const Number deviation = value - average;
        squaredDeviationTotal += deviation * deviation;
    }

    return squaredDeviationTotal /
           static_cast<Number>(values.size());
}


// ---------------------------------------------------------------------------
// Sample variance
// ---------------------------------------------------------------------------
//
// Bessel's correction:
//
//     sample variance = sum((x - x_bar)^2) / (n - 1)
//
// n - 1 is used when estimating population variance from a sample.

Number sampleVariance(const std::vector<Number>& values) {
    validateData(values);

    if (values.size() < 2) {
        throw std::invalid_argument(
            "Sample variance requires at least two observations."
        );
    }

    const Number average = mean(values);

    Number squaredDeviationTotal = 0.0;

    for (Number value : values) {
        const Number deviation = value - average;
        squaredDeviationTotal += deviation * deviation;
    }

    return squaredDeviationTotal /
           static_cast<Number>(values.size() - 1);
}


Number populationStandardDeviation(
    const std::vector<Number>& values
) {
    return std::sqrt(populationVariance(values));
}


Number sampleStandardDeviation(
    const std::vector<Number>& values
) {
    return std::sqrt(sampleVariance(values));
}


// ---------------------------------------------------------------------------
// Statistics report
// ---------------------------------------------------------------------------

struct StatisticsReport {
    std::size_t count;
    Number minimum;
    Number maximum;
    Number meanValue;
    Number medianValue;
    Number range;
    std::vector<Number> modeValues;
    Number populationVarianceValue;
    Number populationStandardDeviationValue;
    std::optional<Number> sampleVarianceValue;
    std::optional<Number> sampleStandardDeviationValue;
};


StatisticsReport describe(const std::vector<Number>& values) {
    validateData(values);

    const auto [minimumIterator, maximumIterator] =
        std::minmax_element(values.begin(), values.end());

    StatisticsReport report{
        values.size(),
        *minimumIterator,
        *maximumIterator,
        mean(values),
        median(values),
        *maximumIterator - *minimumIterator,
        modes(values),
        populationVariance(values),
        populationStandardDeviation(values),
        std::nullopt,
        std::nullopt
    };

    if (values.size() >= 2) {
        report.sampleVarianceValue = sampleVariance(values);
        report.sampleStandardDeviationValue =
            sampleStandardDeviation(values);
    }

    return report;
}


void printModes(const std::vector<Number>& modeValues) {
    if (modeValues.empty()) {
        std::cout << "None";
        return;
    }

    for (std::size_t i = 0; i < modeValues.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        std::cout << modeValues[i];
    }
}


void printReport(
    const std::string& title,
    const StatisticsReport& report
) {
    std::cout << "\n" << title << "\n";
    std::cout << std::string(title.size(), '-') << "\n";

    std::cout << "Count: "
              << report.count << "\n";

    std::cout << "Minimum: "
              << report.minimum << "\n";

    std::cout << "Maximum: "
              << report.maximum << "\n";

    std::cout << "Range: "
              << report.range << "\n";

    std::cout << "Mean: "
              << report.meanValue << "\n";

    std::cout << "Median: "
              << report.medianValue << "\n";

    std::cout << "Mode(s): ";
    printModes(report.modeValues);
    std::cout << "\n";

    std::cout << "Population variance: "
              << report.populationVarianceValue << "\n";

    std::cout << "Population standard deviation: "
              << report.populationStandardDeviationValue
              << "\n";

    if (report.sampleVarianceValue.has_value()) {
        std::cout << "Sample variance: "
                  << *report.sampleVarianceValue << "\n";

        std::cout << "Sample standard deviation: "
                  << *report.sampleStandardDeviationValue
                  << "\n";
    } else {
        std::cout << "Sample variance: undefined\n";
        std::cout << "Sample standard deviation: undefined\n";
    }
}


// ---------------------------------------------------------------------------
// Welford online statistics
// ---------------------------------------------------------------------------
//
// This is appropriate for streaming data because the entire dataset does not
// need to remain in memory.
//
// Memory complexity: O(1).
// Update complexity: O(1) per observation.
//
// The M2 variable stores the accumulated squared deviation information.

class OnlineStatistics {
private:
    std::size_t count_ = 0;
    Number mean_ = 0.0;
    Number m2_ = 0.0;

public:
    void update(Number value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Streaming observation must be finite."
            );
        }

        ++count_;

        const Number delta = value - mean_;

        mean_ += delta /
                 static_cast<Number>(count_);

        const Number deltaAfterUpdate =
            value - mean_;

        m2_ += delta * deltaAfterUpdate;
    }

    std::size_t count() const {
        return count_;
    }

    Number meanValue() const {
        if (count_ == 0) {
            throw std::logic_error(
                "No observations have been processed."
            );
        }

        return mean_;
    }

    Number populationVarianceValue() const {
        if (count_ == 0) {
            throw std::logic_error(
                "No observations have been processed."
            );
        }

        return m2_ /
               static_cast<Number>(count_);
    }

    Number sampleVarianceValue() const {
        if (count_ < 2) {
            throw std::logic_error(
                "At least two observations are required."
            );
        }

        return m2_ /
               static_cast<Number>(count_ - 1);
    }

    Number populationStandardDeviation() const {
        return std::sqrt(populationVarianceValue());
    }

    Number sampleStandardDeviation() const {
        return std::sqrt(sampleVarianceValue());
    }
};


// ---------------------------------------------------------------------------
// Z-score analysis
// ---------------------------------------------------------------------------
//
// z = (x - mean) / standard deviation
//
// A large positive z-score indicates an observation considerably above the
// average. A large negative value indicates an observation considerably below
// the average.
//
// The threshold used here is an operational demonstration, not a universal
// anomaly definition.

struct Anomaly {
    Number observation;
    Number zScore;
};


std::vector<Anomaly> findHighLatencyAnomalies(
    const std::vector<Number>& latencies,
    Number threshold
) {
    validateData(latencies);

    if (threshold <= 0.0) {
        throw std::invalid_argument(
            "Z-score threshold must be positive."
        );
    }

    const Number average = mean(latencies);
    const Number standardDeviation =
        populationStandardDeviation(latencies);

    if (standardDeviation == 0.0) {
        return {};
    }

    std::vector<Anomaly> anomalies;

    for (Number latency : latencies) {
        const Number z =
            (latency - average) / standardDeviation;

        if (z >= threshold) {
            anomalies.push_back({latency, z});
        }
    }

    return anomalies;
}


// ---------------------------------------------------------------------------
// Model monitoring service
// ---------------------------------------------------------------------------

class ModelMonitoringService {
private:
    std::map<std::string, OnlineStatistics> groupStatistics_;

public:
    void record(
        const std::string& modelVersion,
        Number latencyMilliseconds
    ) {
        if (modelVersion.empty()) {
            throw std::invalid_argument(
                "Model version cannot be empty."
            );
        }

        if (latencyMilliseconds < 0.0) {
            throw std::invalid_argument(
                "Latency cannot be negative."
            );
        }

        groupStatistics_[modelVersion].update(
            latencyMilliseconds
        );
    }

    void printReport() const {
        std::cout << "\nModel monitoring by version\n";
        std::cout << "---------------------------\n";

        for (const auto& [modelVersion, statistics]
             : groupStatistics_) {

            std::cout << "\nModel: "
                      << modelVersion << "\n";

            std::cout << "Requests: "
                      << statistics.count() << "\n";

            std::cout << "Mean latency: "
                      << statistics.meanValue()
                      << " ms\n";

            std::cout << "Population variance: "
                      << statistics.populationVarianceValue()
                      << "\n";

            std::cout << "Population standard deviation: "
                      << statistics.populationStandardDeviation()
                      << " ms\n";
        }
    }
};


// ---------------------------------------------------------------------------
// Input processing
// ---------------------------------------------------------------------------
//
// The parser accepts whitespace-separated numeric observations.
//
// Example:
//     100 102 98 101 99
//
// Invalid input is rejected instead of silently converting malformed values.

std::vector<Number> parseInput(const std::string& input) {
    std::istringstream stream(input);
    std::vector<Number> values;

    Number value;

    while (stream >> value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Input contains a non-finite value."
            );
        }

        values.push_back(value);
    }

    if (!stream.eof()) {
        throw std::invalid_argument(
            "Input contains a non-numeric token."
        );
    }

    return values;
}


// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(
            "Test failed: " + message
        );
    }
}


bool approximatelyEqual(
    Number left,
    Number right,
    Number tolerance = 1e-10
) {
    return std::fabs(left - right) <= tolerance;
}


void runTests() {
    const std::vector<Number> values{
        1, 2, 3, 4, 5
    };

    require(
        approximatelyEqual(mean(values), 3.0),
        "Mean should equal 3."
    );

    require(
        approximatelyEqual(median(values), 3.0),
        "Median should equal 3."
    );

    require(
        approximatelyEqual(
            populationVariance(values),
            2.0
        ),
        "Population variance should equal 2."
    );

    require(
        approximatelyEqual(
            sampleVariance(values),
            2.5
        ),
        "Sample variance should equal 2.5."
    );

    const std::vector<Number> repeated{
        1, 1, 2, 2, 3
    };

    const auto repeatedModes = modes(repeated);

    require(
        repeatedModes.size() == 2,
        "There should be two modes."
    );

    require(
        repeatedModes[0] == 1 &&
        repeatedModes[1] == 2,
        "Modes should be 1 and 2."
    );

    const std::vector<Number> constant{
        7, 7, 7, 7
    };

    require(
        populationVariance(constant) == 0.0,
        "Constant data should have zero variance."
    );

    OnlineStatistics online;

    for (Number value : values) {
        online.update(value);
    }

    require(
        approximatelyEqual(online.meanValue(), 3.0),
        "Online mean should equal 3."
    );

    require(
        approximatelyEqual(
            online.populationVarianceValue(),
            2.0
        ),
        "Online population variance should equal 2."
    );

    std::cout << "All C++ tests passed.\n";
}


// ---------------------------------------------------------------------------
// Main case study
// ---------------------------------------------------------------------------

int main() {
    std::cout << std::fixed << std::setprecision(4);

    try {
        runTests();

        // A monitoring window containing request latency measurements.
        //
        // In a real service, these values could arrive from application logs,
        // telemetry, a metrics broker, or an observability platform.
        const std::vector<Number> latencyWindow{
            105, 110, 98, 102, 108,
            115, 101, 99, 104, 109,
            103, 107, 111, 100, 106,
            250
        };

        const StatisticsReport report =
            describe(latencyWindow);

        printReport(
            "Inference latency statistics",
            report
        );

        // Detect observations substantially above the average.
        const Number zThreshold = 2.0;

        const auto anomalies =
            findHighLatencyAnomalies(
                latencyWindow,
                zThreshold
            );

        std::cout << "\nHigh-latency observations\n";
        std::cout << "-------------------------\n";

        if (anomalies.empty()) {
            std::cout << "No observations exceeded the threshold.\n";
        } else {
            for (const auto& anomaly : anomalies) {
                std::cout
                    << "Latency: "
                    << anomaly.observation
                    << " ms, z-score: "
                    << anomaly.zScore
                    << "\n";
            }
        }

        // Streaming analysis.
        OnlineStatistics streaming;

        for (Number latency : latencyWindow) {
            streaming.update(latency);
        }

        std::cout << "\nStreaming calculation\n";
        std::cout << "---------------------\n";
        std::cout << "Count: "
                  << streaming.count()
                  << "\n";
        std::cout << "Mean: "
                  << streaming.meanValue()
                  << " ms\n";
        std::cout << "Population variance: "
                  << streaming.populationVarianceValue()
                  << "\n";
        std::cout << "Population standard deviation: "
                  << streaming.populationStandardDeviation()
                  << " ms\n";

        // Grouped model monitoring.
        ModelMonitoringService monitoring;

        const std::vector<Number> versionA{
            100, 102, 98, 105, 101
        };

        const std::vector<Number> versionB{
            110, 125, 108, 145, 112
        };

        for (Number latency : versionA) {
            monitoring.record("model-v1", latency);
        }

        for (Number latency : versionB) {
            monitoring.record("model-v2", latency);
        }

        monitoring.printReport();

        // Demonstrate parsing external-style input.
        const std::string userInput =
            "10 12 14 16 18";

        const auto parsedValues =
            parseInput(userInput);

        std::cout << "\nParsed input mean: "
                  << mean(parsedValues)
                  << "\n";

        // Demonstrate a controlled failure.
        try {
            parseInput("10 20 invalid 30");
        } catch (const std::exception& error) {
            std::cout << "\nInput validation failure: "
                      << error.what()
                      << "\n";
        }

        // Demonstrate the single-observation sample-variance edge case.
        try {
            sampleVariance({42});
        } catch (const std::exception& error) {
            std::cout
                << "Sample variance edge case: "
                << error.what()
                << "\n";
        }

    } catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << "\n";

        return 1;
    }

    return 0;
}
