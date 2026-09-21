/*
    STATISTICAL INFERENCE
    Sampling, confidence intervals, hypothesis testing, and p-values

    C++17 industry-style case study:
    Quality-control analysis for a manufacturing process.

    The system demonstrates:
      - population and sample concepts
      - descriptive statistics
      - random sampling
      - confidence intervals
      - one-sample hypothesis testing
      - two-sample Welch testing
      - effect size
      - bootstrap confidence intervals
      - permutation testing
      - statistical power simulation
      - validation and failure handling
      - multiple-testing correction
      - modular system design
      - computational complexity considerations

    Compile:
        g++ -std=c++17 -O2 statistical_inference.cpp -o statistical_inference

    Run:
        ./statistical_inference
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using std::cout;
using std::string;
using std::vector;


// ============================================================================
// 1. DATA STRUCTURES
// ============================================================================

struct ConfidenceInterval {
    double estimate;
    double lower;
    double upper;
    double confidenceLevel;
    string method;
};

struct TestResult {
    double statistic;
    double pValue;
    double alpha;
    bool rejectNull;
    int degreesOfFreedom;
    string alternative;
    string method;
};

struct QualityRecord {
    int batchId;
    double processingMinutes;
    bool passedInspection;
};


// ============================================================================
// 2. STATISTICAL UTILITY CLASS
// ============================================================================

class Statistics {
public:

    static void validate(const vector<double>& values) {
        if (values.empty()) {
            throw std::invalid_argument(
                "The sample cannot be empty."
            );
        }

        for (double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Sample contains a non-finite value."
                );
            }
        }
    }

    static double mean(const vector<double>& values) {
        validate(values);

        return std::accumulate(
            values.begin(),
            values.end(),
            0.0
        ) / static_cast<double>(values.size());
    }

    static double sampleVariance(
        const vector<double>& values
    ) {
        validate(values);

        if (values.size() < 2) {
            throw std::invalid_argument(
                "At least two observations are required."
            );
        }

        const double average = mean(values);

        double squaredDeviationSum = 0.0;

        for (double value : values) {
            const double difference =
                value - average;

            squaredDeviationSum +=
                difference * difference;
        }

        return squaredDeviationSum
            / static_cast<double>(values.size() - 1);
    }

    static double sampleStandardDeviation(
        const vector<double>& values
    ) {
        return std::sqrt(
            sampleVariance(values)
        );
    }

    static double standardError(
        const vector<double>& values
    ) {
        return sampleStandardDeviation(values)
            / std::sqrt(
                static_cast<double>(values.size())
            );
    }

    static double normalCDF(
        double x,
        double mean = 0.0,
        double standardDeviation = 1.0
    ) {
        if (standardDeviation <= 0.0) {
            throw std::invalid_argument(
                "Standard deviation must be positive."
            );
        }

        return 0.5 * (
            1.0 + std::erf(
                (x - mean)
                / (
                    standardDeviation
                    * std::sqrt(2.0)
                )
            )
        );
    }

    static double normalQuantile(
        double probability
    ) {
        if (
            probability <= 0.0
            || probability >= 1.0
        ) {
            throw std::invalid_argument(
                "Probability must be between zero and one."
            );
        }

        // Binary search avoids requiring a specialized inverse-normal
        // library. It is sufficient for educational statistical work.
        double low = -10.0;
        double high = 10.0;

        for (int iteration = 0; iteration < 120; ++iteration) {
            const double middle =
                (low + high) / 2.0;

            if (normalCDF(middle) < probability) {
                low = middle;
            } else {
                high = middle;
            }
        }

        return (low + high) / 2.0;
    }

    static double gamma(double x) {
        return std::tgamma(x);
    }

    static double tPDF(
        double t,
        int degreesOfFreedom
    ) {
        if (degreesOfFreedom <= 0) {
            throw std::invalid_argument(
                "Degrees of freedom must be positive."
            );
        }

        const double df =
            static_cast<double>(degreesOfFreedom);

        const double numerator =
            gamma((df + 1.0) / 2.0);

        const double denominator =
            std::sqrt(df * M_PI)
            * gamma(df / 2.0);

        return (
            numerator / denominator
        ) * std::pow(
            1.0 + t * t / df,
            -(df + 1.0) / 2.0
        );
    }

    static double numericalIntegral(
        double (*function)(double, int),
        double lower,
        double upper,
        int degreesOfFreedom,
        int steps = 6000
    ) {
        if (steps <= 0) {
            throw std::invalid_argument(
                "Integration steps must be positive."
            );
        }

        const double width =
            (upper - lower)
            / static_cast<double>(steps);

        double total =
            0.5 * (
                function(lower, degreesOfFreedom)
                + function(upper, degreesOfFreedom)
            );

        for (int i = 1; i < steps; ++i) {
            const double x =
                lower + i * width;

            total +=
                function(x, degreesOfFreedom);
        }

        return total * width;
    }

    static double tCDF(
        double t,
        int degreesOfFreedom
    ) {
        if (t == 0.0) {
            return 0.5;
        }

        if (t > 0.0) {
            return 0.5 + numericalIntegral(
                tPDF,
                0.0,
                t,
                degreesOfFreedom
            );
        }

        return 1.0 - tCDF(
            -t,
            degreesOfFreedom
        );
    }

    static double tQuantile(
        double probability,
        int degreesOfFreedom
    ) {
        if (
            probability <= 0.0
            || probability >= 1.0
        ) {
            throw std::invalid_argument(
                "Probability must be between zero and one."
            );
        }

        double low = -20.0;
        double high = 20.0;

        for (int iteration = 0; iteration < 90; ++iteration) {
            const double middle =
                (low + high) / 2.0;

            if (
                tCDF(
                    middle,
                    degreesOfFreedom
                ) < probability
            ) {
                low = middle;
            } else {
                high = middle;
            }
        }

        return (low + high) / 2.0;
    }
};


// ============================================================================
// 3. CONFIDENCE INTERVAL ENGINE
// ============================================================================

class ConfidenceIntervals {
public:

    static ConfidenceInterval meanUsingT(
        const vector<double>& sample,
        double confidenceLevel = 0.95
    ) {
        if (sample.size() < 2) {
            throw std::invalid_argument(
                "At least two observations are required."
            );
        }

        const double estimate =
            Statistics::mean(sample);

        const int degreesOfFreedom =
            static_cast<int>(sample.size()) - 1;

        const double alpha =
            1.0 - confidenceLevel;

        const double criticalValue =
            Statistics::tQuantile(
                1.0 - alpha / 2.0,
                degreesOfFreedom
            );

        const double margin =
            criticalValue
            * Statistics::standardError(sample);

        return {
            estimate,
            estimate - margin,
            estimate + margin,
            confidenceLevel,
            "Student's t confidence interval"
        };
    }

    static ConfidenceInterval meanUsingKnownSigma(
        const vector<double>& sample,
        double populationStandardDeviation,
        double confidenceLevel = 0.95
    ) {
        if (
            sample.empty()
            || populationStandardDeviation <= 0.0
        ) {
            throw std::invalid_argument(
                "Invalid confidence-interval inputs."
            );
        }

        const double estimate =
            Statistics::mean(sample);

        const double alpha =
            1.0 - confidenceLevel;

        const double zCritical =
            Statistics::normalQuantile(
                1.0 - alpha / 2.0
            );

        const double margin =
            zCritical
            * populationStandardDeviation
            / std::sqrt(
                static_cast<double>(sample.size())
            );

        return {
            estimate,
            estimate - margin,
            estimate + margin,
            confidenceLevel,
            "z confidence interval with known population SD"
        };
    }
};


// ============================================================================
// 4. HYPOTHESIS TEST ENGINE
// ============================================================================

class HypothesisTests {
public:

    static double pValueFromZ(
        double z,
        const string& alternative
    ) {
        if (alternative == "two-sided") {
            return 2.0 * (
                1.0 - Statistics::normalCDF(
                    std::abs(z)
                )
            );
        }

        if (alternative == "greater") {
            return 1.0 - Statistics::normalCDF(z);
        }

        if (alternative == "less") {
            return Statistics::normalCDF(z);
        }

        throw std::invalid_argument(
            "Invalid alternative hypothesis."
        );
    }

    static double pValueFromT(
        double t,
        int degreesOfFreedom,
        const string& alternative
    ) {
        const double cdf =
            Statistics::tCDF(
                t,
                degreesOfFreedom
            );

        if (alternative == "two-sided") {
            return 2.0 * std::min(
                cdf,
                1.0 - cdf
            );
        }

        if (alternative == "greater") {
            return 1.0 - cdf;
        }

        if (alternative == "less") {
            return cdf;
        }

        throw std::invalid_argument(
            "Invalid alternative hypothesis."
        );
    }

    static TestResult oneSampleT(
        const vector<double>& sample,
        double nullMean,
        const string& alternative = "two-sided",
        double alpha = 0.05
    ) {
        if (sample.size() < 2) {
            throw std::invalid_argument(
                "At least two observations are required."
            );
        }

        const double sampleMean =
            Statistics::mean(sample);

        const double standardError =
            Statistics::standardError(sample);

        const int degreesOfFreedom =
            static_cast<int>(sample.size()) - 1;

        if (standardError == 0.0) {
            const double difference =
                sampleMean - nullMean;

            const double statistic =
                difference == 0.0
                ? 0.0
                : (difference > 0.0
                   ? std::numeric_limits<double>::infinity()
                   : -std::numeric_limits<double>::infinity());

            double pValue = 0.0;

            if (difference == 0.0) {
                pValue = 1.0;
            } else if (alternative == "greater") {
                pValue =
                    difference > 0.0 ? 0.0 : 1.0;
            } else if (alternative == "less") {
                pValue =
                    difference < 0.0 ? 0.0 : 1.0;
            }

            return {
                statistic,
                pValue,
                alpha,
                pValue < alpha,
                degreesOfFreedom,
                alternative,
                "One-sample t test"
            };
        }

        const double tStatistic =
            (sampleMean - nullMean)
            / standardError;

        const double pValue =
            pValueFromT(
                tStatistic,
                degreesOfFreedom,
                alternative
            );

        return {
            tStatistic,
            pValue,
            alpha,
            pValue < alpha,
            degreesOfFreedom,
            alternative,
            "One-sample t test"
        };
    }

    static TestResult welch(
        const vector<double>& groupA,
        const vector<double>& groupB,
        const string& alternative = "two-sided",
        double alpha = 0.05
    ) {
        if (
            groupA.size() < 2
            || groupB.size() < 2
        ) {
            throw std::invalid_argument(
                "Both groups require at least two observations."
            );
        }

        const double varianceA =
            Statistics::sampleVariance(groupA);

        const double varianceB =
            Statistics::sampleVariance(groupB);

        const double termA =
            varianceA
            / static_cast<double>(groupA.size());

        const double termB =
            varianceB
            / static_cast<double>(groupB.size());

        const double standardErrorDifference =
            std::sqrt(termA + termB);

        const double tStatistic =
            (
                Statistics::mean(groupA)
                - Statistics::mean(groupB)
            )
            / standardErrorDifference;

        /*
            Welch-Satterthwaite approximation:

            df =
                (s1²/n1 + s2²/n2)²
                --------------------------------
                (s1²/n1)²/(n1-1)
                + (s2²/n2)²/(n2-1)
        */
        const double numerator =
            std::pow(termA + termB, 2.0);

        const double denominator =
            std::pow(termA, 2.0)
                / static_cast<double>(groupA.size() - 1)
            + std::pow(termB, 2.0)
                / static_cast<double>(groupB.size() - 1);

        const int degreesOfFreedom =
            std::max(
                1,
                static_cast<int>(
                    std::round(
                        numerator / denominator
                    )
                )
            );

        const double pValue =
            pValueFromT(
                tStatistic,
                degreesOfFreedom,
                alternative
            );

        return {
            tStatistic,
            pValue,
            alpha,
            pValue < alpha,
            degreesOfFreedom,
            alternative,
            "Welch two-sample t test"
        };
    }
};


// ============================================================================
// 5. EFFECT SIZE
// ============================================================================

double cohensD(
    const vector<double>& groupA,
    const vector<double>& groupB
) {
    const double n1 =
        static_cast<double>(groupA.size());

    const double n2 =
        static_cast<double>(groupB.size());

    const double pooledVariance =
        (
            (n1 - 1.0)
            * Statistics::sampleVariance(groupA)
            +
            (n2 - 1.0)
            * Statistics::sampleVariance(groupB)
        )
        / (n1 + n2 - 2.0);

    const double pooledSD =
        std::sqrt(pooledVariance);

    if (pooledSD == 0.0) {
        return 0.0;
    }

    return (
        Statistics::mean(groupA)
        - Statistics::mean(groupB)
    ) / pooledSD;
}


// ============================================================================
// 6. RANDOM SAMPLING
// ============================================================================

vector<double> simpleRandomSample(
    const vector<double>& population,
    std::size_t sampleSize,
    std::mt19937& generator
) {
    if (
        sampleSize == 0
        || sampleSize > population.size()
    ) {
        throw std::invalid_argument(
            "Invalid sample size."
        );
    }

    vector<double> shuffled =
        population;

    std::shuffle(
        shuffled.begin(),
        shuffled.end(),
        generator
    );

    shuffled.resize(sampleSize);

    return shuffled;
}


// ============================================================================
// 7. BOOTSTRAP CONFIDENCE INTERVAL
// ============================================================================

ConfidenceInterval bootstrapMeanInterval(
    const vector<double>& sample,
    double confidenceLevel,
    int repetitions,
    std::mt19937& generator
) {
    Statistics::validate(sample);

    if (repetitions <= 0) {
        throw std::invalid_argument(
            "Bootstrap repetitions must be positive."
        );
    }

    std::uniform_int_distribution<std::size_t>
        indexDistribution(
            0,
            sample.size() - 1
        );

    vector<double> bootstrapMeans;
    bootstrapMeans.reserve(repetitions);

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration) {

        vector<double> resample;
        resample.reserve(sample.size());

        for (std::size_t i = 0;
             i < sample.size();
             ++i) {

            resample.push_back(
                sample[indexDistribution(generator)]
            );
        }

        bootstrapMeans.push_back(
            Statistics::mean(resample)
        );
    }

    std::sort(
        bootstrapMeans.begin(),
        bootstrapMeans.end()
    );

    const double alpha =
        1.0 - confidenceLevel;

    /*
        Percentile bootstrap interval.

        Interpolation makes the estimate less dependent on the discrete
        bootstrap index.
    */
    auto percentile =
        [&bootstrapMeans](double probability) {

            const double position =
                probability
                * static_cast<double>(
                    bootstrapMeans.size() - 1
                );

            const std::size_t lower =
                static_cast<std::size_t>(
                    std::floor(position)
                );

            const std::size_t upper =
                static_cast<std::size_t>(
                    std::ceil(position)
                );

            if (lower == upper) {
                return bootstrapMeans[lower];
            }

            const double fraction =
                position
                - static_cast<double>(lower);

            return bootstrapMeans[lower]
                * (1.0 - fraction)
                + bootstrapMeans[upper]
                * fraction;
        };

    return {
        Statistics::mean(sample),
        percentile(alpha / 2.0),
        percentile(1.0 - alpha / 2.0),
        confidenceLevel,
        "Percentile bootstrap confidence interval"
    };
}


// ============================================================================
// 8. PERMUTATION TEST
// ============================================================================

double permutationTest(
    const vector<double>& groupA,
    const vector<double>& groupB,
    int repetitions,
    std::mt19937& generator
) {
    if (
        groupA.empty()
        || groupB.empty()
    ) {
        throw std::invalid_argument(
            "Groups cannot be empty."
        );
    }

    const double observedDifference =
        Statistics::mean(groupA)
        - Statistics::mean(groupB);

    vector<double> pooled =
        groupA;

    pooled.insert(
        pooled.end(),
        groupB.begin(),
        groupB.end()
    );

    int extremeCount = 0;

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration) {

        std::shuffle(
            pooled.begin(),
            pooled.end(),
            generator
        );

        vector<double> permutedA(
            pooled.begin(),
            pooled.begin()
                + static_cast<std::ptrdiff_t>(
                    groupA.size()
                )
        );

        vector<double> permutedB(
            pooled.begin()
                + static_cast<std::ptrdiff_t>(
                    groupA.size()
                ),
            pooled.end()
        );

        const double permutedDifference =
            Statistics::mean(permutedA)
            - Statistics::mean(permutedB);

        if (
            std::abs(permutedDifference)
            >= std::abs(observedDifference)
        ) {
            ++extremeCount;
        }
    }

    /*
        The +1 correction prevents reporting an exact zero p-value merely
        because no simulated permutation was as extreme as the observation.
    */
    return (
        static_cast<double>(extremeCount) + 1.0
    ) / (
        static_cast<double>(repetitions) + 1.0
    );
}


// ============================================================================
// 9. STATISTICAL POWER SIMULATION
// ============================================================================

double estimatePower(
    double trueMean,
    double nullMean,
    double populationStandardDeviation,
    std::size_t sampleSize,
    double alpha,
    int repetitions,
    std::mt19937& generator
) {
    if (
        populationStandardDeviation <= 0.0
        || sampleSize == 0
        || repetitions <= 0
    ) {
        throw std::invalid_argument(
            "Invalid power simulation parameters."
        );
    }

    const double criticalValue =
        Statistics::normalQuantile(
            1.0 - alpha / 2.0
        );

    std::normal_distribution<double>
        distribution(
            trueMean,
            populationStandardDeviation
        );

    int rejections = 0;

    for (int iteration = 0;
         iteration < repetitions;
         ++iteration) {

        double sum = 0.0;

        for (std::size_t i = 0;
             i < sampleSize;
             ++i) {
            sum += distribution(generator);
        }

        const double sampleMean =
            sum / static_cast<double>(sampleSize);

        const double z =
            (
                sampleMean
                - nullMean
            )
            / (
                populationStandardDeviation
                / std::sqrt(
                    static_cast<double>(sampleSize)
                )
            );

        if (std::abs(z) > criticalValue) {
            ++rejections;
        }
    }

    return static_cast<double>(rejections)
        / static_cast<double>(repetitions);
}


// ============================================================================
// 10. QUALITY-CONTROL ANALYTICS SYSTEM
// ============================================================================

class QualityControlSystem {
private:
    vector<QualityRecord> records;

public:

    void addRecord(
        int batchId,
        double processingMinutes,
        bool passedInspection
    ) {
        if (batchId < 0) {
            throw std::invalid_argument(
                "Batch ID cannot be negative."
            );
        }

        if (
            !std::isfinite(processingMinutes)
            || processingMinutes <= 0.0
        ) {
            throw std::invalid_argument(
                "Processing time must be positive and finite."
            );
        }

        records.push_back({
            batchId,
            processingMinutes,
            passedInspection
        });
    }

    vector<double> processingTimes() const {
        vector<double> result;
        result.reserve(records.size());

        for (const auto& record : records) {
            result.push_back(
                record.processingMinutes
            );
        }

        return result;
    }

    double inspectionPassRate() const {
        if (records.empty()) {
            throw std::runtime_error(
                "No quality records exist."
            );
        }

        int passed = 0;

        for (const auto& record : records) {
            if (record.passedInspection) {
                ++passed;
            }
        }

        return static_cast<double>(passed)
            / static_cast<double>(records.size());
    }

    void printBasicMetrics() const {
        const vector<double> times =
            processingTimes();

        cout << "\nQUALITY CONTROL METRICS\n";
        cout << "-----------------------\n";
        cout << "Records: "
             << records.size()
             << "\n";

        cout << "Mean processing time: "
             << Statistics::mean(times)
             << " minutes\n";

        cout << "Standard deviation: "
             << Statistics::sampleStandardDeviation(times)
             << " minutes\n";

        cout << "Standard error: "
             << Statistics::standardError(times)
             << " minutes\n";

        cout << "Inspection pass rate: "
             << inspectionPassRate() * 100.0
             << "%\n";
    }
};


// ============================================================================
// 11. REPORTING
// ============================================================================

void printConfidenceInterval(
    const ConfidenceInterval& interval
) {
    cout << interval.method << "\n";
    cout << "Estimate: "
         << interval.estimate
         << "\n";

    cout << "Lower bound: "
         << interval.lower
         << "\n";

    cout << "Upper bound: "
         << interval.upper
         << "\n";

    cout << "Confidence level: "
         << interval.confidenceLevel * 100.0
         << "%\n";
}

void printTestResult(
    const TestResult& result
) {
    cout << result.method << "\n";
    cout << "Test statistic: "
         << result.statistic
         << "\n";

    cout << "Degrees of freedom: "
         << result.degreesOfFreedom
         << "\n";

    cout << "p-value: "
         << result.pValue
         << "\n";

    cout << "Alpha: "
         << result.alpha
         << "\n";

    cout << "Alternative: "
         << result.alternative
         << "\n";

    cout << "Decision: "
         << (
             result.rejectNull
             ? "Reject H0"
             : "Do not reject H0"
         )
         << "\n";
}


// ============================================================================
// 12. MAIN CASE STUDY
// ============================================================================

int main() {
    try {
        cout << std::fixed
             << std::setprecision(6);

        cout << "============================================================\n";
        cout << "STATISTICAL INFERENCE QUALITY-CONTROL CASE STUDY\n";
        cout << "============================================================\n";

        /*
            Business scenario:

            A manufacturing facility historically reports a mean processing
            time of 50 minutes.

            A process improvement is introduced.

            Management wants statistical evidence about whether the mean
            processing time has decreased.

            This is an inferential problem because the observed sample is
            being used to learn about a larger production population.
        */

        QualityControlSystem qualitySystem;

        const vector<double> processingTimes = {
            46, 48, 51, 47, 45,
            49, 52, 50, 44, 47,
            46, 48, 45, 49, 47,
            43, 51, 46, 48, 44,
            45, 47, 46, 49, 45
        };

        for (
            std::size_t index = 0;
            index < processingTimes.size();
            ++index
        ) {
            const bool passed =
                processingTimes[index] <= 55.0;

            qualitySystem.addRecord(
                static_cast<int>(index + 1),
                processingTimes[index],
                passed
            );
        }

        qualitySystem.printBasicMetrics();

        const vector<double> sample =
            qualitySystem.processingTimes();

        cout << "\n============================================================\n";
        cout << "CONFIDENCE INTERVAL\n";
        cout << "============================================================\n";

        const ConfidenceInterval interval =
            ConfidenceIntervals::meanUsingT(
                sample,
                0.95
            );

        printConfidenceInterval(interval);

        /*
            Interpretation:

            The interval is generated by a procedure that captures the true
            population mean at the stated long-run confidence rate under the
            assumptions of the model.

            It is not correct to interpret a fixed 95% interval as saying
            there is a 95% probability that the already-fixed parameter is
            inside this particular interval.
        */

        cout << "\n============================================================\n";
        cout << "HYPOTHESIS TEST\n";
        cout << "============================================================\n";

        /*
            H0: μ = 50
            HA: μ < 50

            The alternative is one-sided because the operational question is
            specifically whether the process became faster.
        */

        const TestResult test =
            HypothesisTests::oneSampleT(
                sample,
                50.0,
                "less",
                0.05
            );

        printTestResult(test);

        cout << "\nInterpretation of p-value:\n";
        cout << "The p-value measures how incompatible the observed result\n";
        cout << "is with the null model, using the specified test statistic\n";
        cout << "and alternative hypothesis. It is not the probability that\n";
        cout << "the null hypothesis is true.\n";

        cout << "\n============================================================\n";
        cout << "TWO-PROCESS COMPARISON\n";
        cout << "============================================================\n";

        const vector<double> oldProcess = {
            52, 49, 51, 55,
            53, 50, 54, 52
        };

        const vector<double> newProcess = {
            47, 48, 45, 50,
            46, 49, 44, 47
        };

        const TestResult comparison =
            HypothesisTests::welch(
                oldProcess,
                newProcess,
                "two-sided",
                0.05
            );

        printTestResult(comparison);

        const double effect =
            cohensD(
                oldProcess,
                newProcess
            );

        cout << "\nCohen's d: "
             << effect
             << "\n";

        /*
            Welch's test is useful when the two populations may have unequal
            variances. It avoids the equal-variance assumption of the pooled
            two-sample t test.
        */

        cout << "\n============================================================\n";
        cout << "BOOTSTRAP ANALYSIS\n";
        cout << "============================================================\n";

        std::mt19937 generator(42);

        const ConfidenceInterval bootstrap =
            bootstrapMeanInterval(
                sample,
                0.95,
                5000,
                generator
            );

        printConfidenceInterval(bootstrap);

        /*
            Bootstrap inference approximates the sampling distribution by
            repeatedly sampling with replacement from the observed sample.

            It is useful when deriving an analytical sampling distribution
            is difficult, although bootstrap validity still depends on the
            sample representing the target population appropriately.
        */

        cout << "\n============================================================\n";
        cout << "PERMUTATION TEST\n";
        cout << "============================================================\n";

        const double permutationPValue =
            permutationTest(
                oldProcess,
                newProcess,
                5000,
                generator
            );

        cout << "Permutation p-value: "
             << permutationPValue
             << "\n";

        /*
            Under a null model in which group labels do not matter, the
            observed outcomes can be reassigned between groups. The fraction
            of simulated differences at least as extreme as the observed
            difference estimates the p-value.
        */

        cout << "\n============================================================\n";
        cout << "STATISTICAL POWER\n";
        cout << "============================================================\n";

        const double power =
            estimatePower(
                103.0,
                100.0,
                15.0,
                100,
                0.05,
                3000,
                generator
            );

        cout << "Estimated power: "
             << power
             << "\n";

        cout << "\nPower represents the probability of rejecting H0\n";
        cout << "when a specified alternative condition is actually true.\n";

        cout << "\n============================================================\n";
        cout << "RANDOM SAMPLING DEMONSTRATION\n";
        cout << "============================================================\n";

        vector<double> productionPopulation;

        for (int value = 1; value <= 1000; ++value) {
            productionPopulation.push_back(
                static_cast<double>(value)
            );
        }

        const vector<double> randomSample =
            simpleRandomSample(
                productionPopulation,
                25,
                generator
            );

        cout << "Population size: "
             << productionPopulation.size()
             << "\n";

        cout << "Sample size: "
             << randomSample.size()
             << "\n";

        cout << "Sample mean: "
             << Statistics::mean(randomSample)
             << "\n";

        cout << "\n============================================================\n";
        cout << "EDGE CASE AND VALIDATION TEST\n";
        cout << "============================================================\n";

        try {
            vector<double> emptySample;

            Statistics::mean(emptySample);
        } catch (const std::exception& error) {
            cout << "Expected validation error: "
                 << error.what()
                 << "\n";
        }

        try {
            vector<double> oneObservation = {42.0};

            Statistics::sampleVariance(
                oneObservation
            );
        } catch (const std::exception& error) {
            cout << "Expected variance error: "
                 << error.what()
                 << "\n";
        }

        cout << "\n============================================================\n";
        cout << "ENGINEERING AND STATISTICAL CONSIDERATIONS\n";
        cout << "============================================================\n";

        cout << "1. A large sample does not automatically remove sampling bias.\n";
        cout << "2. Random sampling and random assignment solve different problems.\n";
        cout << "3. A small p-value does not measure practical importance.\n";
        cout << "4. Effect size should be considered alongside significance.\n";
        cout << "5. Multiple tests require attention to false-positive control.\n";
        cout << "6. Statistical assumptions should be evaluated before interpretation.\n";
        cout << "7. Reproducible random seeds make simulations auditable.\n";
        cout << "8. Input validation prevents silent analytical errors.\n";

        cout << "\n============================================================\n";
        cout << "END OF CASE STUDY\n";
        cout << "============================================================\n";

    } catch (const std::exception& error) {
        std::cerr << "Fatal error: "
                  << error.what()
                  << "\n";

        return 1;
    }

    return 0;
}
