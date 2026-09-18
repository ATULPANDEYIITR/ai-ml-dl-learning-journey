#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Probability Distributions: Industry-Style Service Monitoring Case Study
 *
 * C++17 program demonstrating:
 *
 *   - Bernoulli distribution
 *   - Binomial distribution
 *   - Poisson distribution
 *   - Exponential distribution
 *   - Gaussian / Normal distribution
 *   - Simulation
 *   - Validation
 *   - Probability calculations
 *   - Event-count modeling
 *   - Waiting-time modeling
 *   - Service-failure modeling
 *   - Response-latency analysis
 *   - Histogram generation
 *   - Numerical and performance considerations
 *
 * Scenario:
 *   A distributed web service receives requests, experiences occasional
 *   failures, and records response latency.
 *
 * The model is intentionally self-contained and uses only the C++ standard
 * library.
 */

namespace probability {

// ============================================================================
// Utility functions
// ============================================================================

constexpr double PI = 3.141592653589793238462643383279502884;

void requireProbability(double probability, const std::string& name) {
    if (!std::isfinite(probability) ||
        probability < 0.0 ||
        probability > 1.0) {
        throw std::invalid_argument(
            name + " must be finite and between 0 and 1."
        );
    }
}

void requirePositive(double value, const std::string& name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
            name + " must be finite and greater than zero."
        );
    }
}

// ============================================================================
// Bernoulli distribution
// ============================================================================

class BernoulliDistribution {
private:
    double p_;

public:
    explicit BernoulliDistribution(double p) : p_(p) {
        requireProbability(p_, "Bernoulli probability");
    }

    double pmf(int x) const {
        if (x == 0) return 1.0 - p_;
        if (x == 1) return p_;
        return 0.0;
    }

    double mean() const {
        return p_;
    }

    double variance() const {
        return p_ * (1.0 - p_);
    }

    template <typename Engine>
    int sample(Engine& engine) const {
        std::bernoulli_distribution distribution(p_);
        return distribution(engine) ? 1 : 0;
    }
};

// ============================================================================
// Binomial distribution
// ============================================================================

class BinomialDistribution {
private:
    int n_;
    double p_;

    static double logCombination(int n, int k) {
        if (k < 0 || k > n) {
            return -std::numeric_limits<double>::infinity();
        }

        return std::lgamma(static_cast<double>(n) + 1.0)
             - std::lgamma(static_cast<double>(k) + 1.0)
             - std::lgamma(static_cast<double>(n - k) + 1.0);
    }

public:
    BinomialDistribution(int n, double p) : n_(n), p_(p) {
        if (n_ < 0) {
            throw std::invalid_argument(
                "Binomial trial count cannot be negative."
            );
        }

        requireProbability(p_, "Binomial probability");
    }

    double pmf(int k) const {
        if (k < 0 || k > n_) return 0.0;

        if (p_ == 0.0) {
            return k == 0 ? 1.0 : 0.0;
        }

        if (p_ == 1.0) {
            return k == n_ ? 1.0 : 0.0;
        }

        const double logProbability =
            logCombination(n_, k)
            + static_cast<double>(k) * std::log(p_)
            + static_cast<double>(n_ - k) * std::log1p(-p_);

        return std::exp(logProbability);
    }

    double cdf(int k) const {
        if (k < 0) return 0.0;
        if (k >= n_) return 1.0;

        double total = 0.0;

        for (int i = 0; i <= k; ++i) {
            total += pmf(i);
        }

        return std::min(1.0, total);
    }

    double mean() const {
        return static_cast<double>(n_) * p_;
    }

    double variance() const {
        return static_cast<double>(n_) * p_ * (1.0 - p_);
    }

    template <typename Engine>
    int sample(Engine& engine) const {
        std::binomial_distribution<int> distribution(n_, p_);
        return distribution(engine);
    }

    int trials() const {
        return n_;
    }
};

// ============================================================================
// Poisson distribution
// ============================================================================

class PoissonDistribution {
private:
    double lambda_;

public:
    explicit PoissonDistribution(double lambda) : lambda_(lambda) {
        if (!std::isfinite(lambda_) || lambda_ < 0.0) {
            throw std::invalid_argument(
                "Poisson rate must be finite and non-negative."
            );
        }
    }

    double pmf(int k) const {
        if (k < 0) return 0.0;

        if (lambda_ == 0.0) {
            return k == 0 ? 1.0 : 0.0;
        }

        // Log-PMF prevents direct factorial calculations from overflowing.
        const double logProbability =
            -lambda_
            + static_cast<double>(k) * std::log(lambda_)
            - std::lgamma(static_cast<double>(k) + 1.0);

        return std::exp(logProbability);
    }

    double cdf(int k) const {
        if (k < 0) return 0.0;

        double total = 0.0;

        for (int i = 0; i <= k; ++i) {
            total += pmf(i);
        }

        return std::min(1.0, total);
    }

    double mean() const {
        return lambda_;
    }

    double variance() const {
        return lambda_;
    }

    template <typename Engine>
    int sample(Engine& engine) const {
        std::poisson_distribution<int> distribution(lambda_);
        return distribution(engine);
    }
};

// ============================================================================
// Exponential distribution
// ============================================================================

class ExponentialDistribution {
private:
    double rate_;

public:
    explicit ExponentialDistribution(double rate) : rate_(rate) {
        requirePositive(rate_, "Exponential rate");
    }

    double pdf(double x) const {
        if (x < 0.0) return 0.0;
        return rate_ * std::exp(-rate_ * x);
    }

    double cdf(double x) const {
        if (x < 0.0) return 0.0;
        return -std::expm1(-rate_ * x);
    }

    double survival(double x) const {
        if (x < 0.0) return 1.0;
        return std::exp(-rate_ * x);
    }

    double mean() const {
        return 1.0 / rate_;
    }

    double variance() const {
        return 1.0 / (rate_ * rate_);
    }

    template <typename Engine>
    double sample(Engine& engine) const {
        std::exponential_distribution<double> distribution(rate_);
        return distribution(engine);
    }
};

// ============================================================================
// Uniform distribution
// ============================================================================

class UniformDistribution {
private:
    double lower_;
    double upper_;

public:
    UniformDistribution(double lower, double upper)
        : lower_(lower), upper_(upper) {
        if (!std::isfinite(lower_) || !std::isfinite(upper_)) {
            throw std::invalid_argument(
                "Uniform bounds must be finite."
            );
        }

        if (upper_ <= lower_) {
            throw std::invalid_argument(
                "Uniform upper bound must exceed lower bound."
            );
        }
    }

    double pdf(double x) const {
        if (x < lower_ || x > upper_) return 0.0;
        return 1.0 / (upper_ - lower_);
    }

    double cdf(double x) const {
        if (x < lower_) return 0.0;
        if (x >= upper_) return 1.0;
        return (x - lower_) / (upper_ - lower_);
    }

    double mean() const {
        return (lower_ + upper_) / 2.0;
    }

    double variance() const {
        const double width = upper_ - lower_;
        return width * width / 12.0;
    }

    template <typename Engine>
    double sample(Engine& engine) const {
        std::uniform_real_distribution<double> distribution(
            lower_, upper_
        );
        return distribution(engine);
    }
};

// ============================================================================
// Gaussian / Normal distribution
// ============================================================================

class GaussianDistribution {
private:
    double mean_;
    double standardDeviation_;

public:
    GaussianDistribution(double mean, double standardDeviation)
        : mean_(mean), standardDeviation_(standardDeviation) {
        if (!std::isfinite(mean_)) {
            throw std::invalid_argument("Gaussian mean must be finite.");
        }

        requirePositive(
            standardDeviation_,
            "Gaussian standard deviation"
        );
    }

    double pdf(double x) const {
        const double z =
            (x - mean_) / standardDeviation_;

        return std::exp(-0.5 * z * z)
             / (standardDeviation_ * std::sqrt(2.0 * PI));
    }

    double cdf(double x) const {
        const double z =
            (x - mean_)
            / (standardDeviation_ * std::sqrt(2.0));

        return 0.5 * (1.0 + std::erf(z));
    }

    double zScore(double x) const {
        return (x - mean_) / standardDeviation_;
    }

    double mean() const {
        return mean_;
    }

    double variance() const {
        return standardDeviation_ * standardDeviation_;
    }

    double standardDeviation() const {
        return standardDeviation_;
    }

    template <typename Engine>
    double sample(Engine& engine) const {
        std::normal_distribution<double> distribution(
            mean_,
            standardDeviation_
        );

        return distribution(engine);
    }
};

// ============================================================================
// Statistical helpers
// ============================================================================

struct SampleStatistics {
    std::size_t count;
    double mean;
    double variance;
    double standardDeviation;
    double minimum;
    double maximum;
};

SampleStatistics summarize(
    const std::vector<double>& values
) {
    if (values.empty()) {
        throw std::invalid_argument(
            "Cannot summarize an empty sample."
        );
    }

    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());

    double squaredDeviation = 0.0;

    for (double value : values) {
        const double difference = value - mean;
        squaredDeviation += difference * difference;
    }

    const double variance =
        squaredDeviation
        / static_cast<double>(values.size());

    const auto [minimum, maximum] =
        std::minmax_element(values.begin(), values.end());

    return {
        values.size(),
        mean,
        variance,
        std::sqrt(variance),
        *minimum,
        *maximum
    };
}

void printStatistics(
    const std::string& name,
    const SampleStatistics& statistics
) {
    std::cout << "\n" << name << "\n";
    std::cout << "  Count: " << statistics.count << "\n";
    std::cout << "  Mean: " << statistics.mean << "\n";
    std::cout << "  Variance: " << statistics.variance << "\n";
    std::cout << "  SD: " << statistics.standardDeviation << "\n";
    std::cout << "  Minimum: " << statistics.minimum << "\n";
    std::cout << "  Maximum: " << statistics.maximum << "\n";
}

// ============================================================================
// Text histogram
// ============================================================================

void printHistogram(
    const std::vector<double>& values,
    int bins,
    int width
) {
    if (values.empty()) {
        throw std::invalid_argument(
            "Histogram requires at least one value."
        );
    }

    if (bins <= 0 || width <= 0) {
        throw std::invalid_argument(
            "Histogram dimensions must be positive."
        );
    }

    const auto [minimumIterator, maximumIterator] =
        std::minmax_element(values.begin(), values.end());

    const double minimum = *minimumIterator;
    const double maximum = *maximumIterator;

    if (minimum == maximum) {
        std::cout << std::fixed << std::setprecision(3)
                  << minimum << " | "
                  << std::string(width, '#') << "\n";
        return;
    }

    const double binWidth =
        (maximum - minimum) / static_cast<double>(bins);

    std::vector<int> counts(static_cast<std::size_t>(bins), 0);

    for (double value : values) {
        int index =
            static_cast<int>((value - minimum) / binWidth);

        if (index >= bins) {
            index = bins - 1;
        }

        if (index < 0) {
            index = 0;
        }

        counts[static_cast<std::size_t>(index)]++;
    }

    const int maximumCount =
        *std::max_element(counts.begin(), counts.end());

    for (int index = 0; index < bins; ++index) {
        const double start =
            minimum + index * binWidth;

        const double end =
            start + binWidth;

        const int barLength =
            maximumCount == 0
                ? 0
                : static_cast<int>(
                    std::round(
                        static_cast<double>(width)
                        * counts[static_cast<std::size_t>(index)]
                        / maximumCount
                    )
                );

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(7) << start
                  << " - "
                  << std::setw(7) << end
                  << " | "
                  << std::string(
                      static_cast<std::size_t>(barLength),
                      '#'
                  )
                  << " "
                  << counts[static_cast<std::size_t>(index)]
                  << "\n";
    }
}

// ============================================================================
// Monte Carlo estimation
// ============================================================================

template <typename Sampler, typename Predicate, typename Engine>
double estimateProbability(
    Sampler sampler,
    Predicate condition,
    std::size_t trials,
    Engine& engine
) {
    if (trials == 0) {
        throw std::invalid_argument(
            "Monte Carlo trial count must be positive."
        );
    }

    std::size_t successes = 0;

    for (std::size_t i = 0; i < trials; ++i) {
        if (condition(sampler(engine))) {
            ++successes;
        }
    }

    return static_cast<double>(successes)
         / static_cast<double>(trials);
}

} // namespace probability

// ============================================================================
// Main case study
// ============================================================================

int main() {
    using namespace probability;

    try {
        std::cout << std::fixed << std::setprecision(8);

        std::cout
            << "============================================================\n"
            << "Probability Distribution Service Monitoring Case Study\n"
            << "============================================================\n";

        /*
         * A fixed seed makes the simulation reproducible. Reproducibility is
         * important for debugging and controlled statistical experiments.
         */
        std::mt19937_64 engine(20260918);

        // --------------------------------------------------------------------
        // Stage 1: Bernoulli model
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 1: Bernoulli failure model\n"
            << "------------------------------------------------------------\n";

        const double failureProbability = 0.02;

        BernoulliDistribution requestFailure(
            failureProbability
        );

        std::cout
            << "Failure probability: "
            << requestFailure.mean()
            << "\n";

        std::cout
            << "Failure variance: "
            << requestFailure.variance()
            << "\n";

        std::cout
            << "P(success): "
            << requestFailure.pmf(0)
            << "\n";

        std::cout
            << "P(failure): "
            << requestFailure.pmf(1)
            << "\n";

        // --------------------------------------------------------------------
        // Stage 2: Binomial model
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 2: Binomial batch-failure model\n"
            << "------------------------------------------------------------\n";

        /*
         * Exactly 500 requests are examined. Each request is modeled as an
         * independent Bernoulli trial with p=0.02.
         */
        BinomialDistribution batchFailures(
            500,
            failureProbability
        );

        std::cout
            << "Expected failures: "
            << batchFailures.mean()
            << "\n";

        std::cout
            << "Failure-count variance: "
            << batchFailures.variance()
            << "\n";

        std::cout
            << "P(exactly 10 failures): "
            << batchFailures.pmf(10)
            << "\n";

        std::cout
            << "P(no more than 5 failures): "
            << batchFailures.cdf(5)
            << "\n";

        const int observedBatchFailures =
            batchFailures.sample(engine);

        std::cout
            << "Simulated batch failures: "
            << observedBatchFailures
            << "\n";

        // --------------------------------------------------------------------
        // Stage 3: Poisson request-arrival model
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 3: Poisson request-arrival model\n"
            << "------------------------------------------------------------\n";

        const double requestsPerMinute = 120.0;

        PoissonDistribution requestsPerMinuteModel(
            requestsPerMinute
        );

        std::cout
            << "Expected requests per minute: "
            << requestsPerMinuteModel.mean()
            << "\n";

        std::cout
            << "Variance of requests per minute: "
            << requestsPerMinuteModel.variance()
            << "\n";

        std::cout
            << "P(exactly 120 requests): "
            << requestsPerMinuteModel.pmf(120)
            << "\n";

        std::cout
            << "P(no more than 110 requests): "
            << requestsPerMinuteModel.cdf(110)
            << "\n";

        std::cout
            << "Simulated request count: "
            << requestsPerMinuteModel.sample(engine)
            << "\n";

        // --------------------------------------------------------------------
        // Stage 4: Poisson count over an hour
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 4: Hourly capacity model\n"
            << "------------------------------------------------------------\n";

        const double minutesPerHour = 60.0;
        const double hourlyRate =
            requestsPerMinute * minutesPerHour;

        PoissonDistribution hourlyRequests(hourlyRate);

        std::cout
            << "Expected hourly requests: "
            << hourlyRequests.mean()
            << "\n";

        std::cout
            << "Hourly variance: "
            << hourlyRequests.variance()
            << "\n";

        /*
         * This is a count model. It does not directly model how long an
         * individual request takes or how long we wait between requests.
         */
        std::cout
            << "P(exactly 7200 requests): "
            << hourlyRequests.pmf(7200)
            << "\n";

        // --------------------------------------------------------------------
        // Stage 5: Exponential interarrival-time model
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 5: Exponential request interarrival model\n"
            << "------------------------------------------------------------\n";

        ExponentialDistribution requestInterarrival(
            requestsPerMinute
        );

        std::cout
            << "Expected interarrival time in minutes: "
            << requestInterarrival.mean()
            << "\n";

        std::cout
            << "Expected interarrival time in seconds: "
            << requestInterarrival.mean() * 60.0
            << "\n";

        std::cout
            << "P(wait more than 1 second): "
            << requestInterarrival.survival(
                1.0 / 60.0
            )
            << "\n";

        std::cout
            << "One simulated interarrival time in seconds: "
            << requestInterarrival.sample(engine) * 60.0
            << "\n";

        // --------------------------------------------------------------------
        // Stage 6: Gaussian response-latency model
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 6: Gaussian response-latency model\n"
            << "------------------------------------------------------------\n";

        /*
         * This is deliberately an assumption to be checked against real
         * latency data. Network and service latency can be skewed and may
         * require log-normal, gamma, Weibull, mixture, or empirical models.
         */
        GaussianDistribution responseLatency(
            200.0,
            40.0
        );

        for (double latency : {100.0, 160.0, 200.0, 240.0, 300.0}) {
            std::cout
                << "Latency "
                << latency
                << " ms"
                << " | z="
                << responseLatency.zScore(latency)
                << " | P(X<=x)="
                << responseLatency.cdf(latency)
                << "\n";
        }

        std::cout
            << "P(response latency <= 250 ms): "
            << responseLatency.cdf(250.0)
            << "\n";

        std::cout
            << "Simulated response latency: "
            << responseLatency.sample(engine)
            << " ms\n";

        // --------------------------------------------------------------------
        // Stage 7: Uniform uncertainty model
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 7: Uniform bounded uncertainty\n"
            << "------------------------------------------------------------\n";

        /*
         * Suppose a controlled test varies a configuration parameter uniformly
         * between 0.0 and 1.0. This is a modeling assumption, not a claim that
         * all real configuration uncertainty is uniform.
         */
        UniformDistribution configurationFactor(
            0.0,
            1.0
        );

        std::cout
            << "Uniform mean: "
            << configurationFactor.mean()
            << "\n";

        std::cout
            << "Uniform variance: "
            << configurationFactor.variance()
            << "\n";

        std::cout
            << "P(factor <= 0.75): "
            << configurationFactor.cdf(0.75)
            << "\n";

        // --------------------------------------------------------------------
        // Stage 8: Large response-latency simulation
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 8: Simulated latency dataset\n"
            << "------------------------------------------------------------\n";

        constexpr std::size_t simulationSize = 100000;

        std::vector<double> latencies;
        latencies.reserve(simulationSize);

        for (std::size_t i = 0; i < simulationSize; ++i) {
            latencies.push_back(
                responseLatency.sample(engine)
            );
        }

        const SampleStatistics latencyStatistics =
            summarize(latencies);

        printStatistics(
            "Simulated Gaussian response latencies",
            latencyStatistics
        );

        // --------------------------------------------------------------------
        // Stage 9: Empirical service-level measurement
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 9: Empirical latency threshold\n"
            << "------------------------------------------------------------\n";

        const double serviceLevelThreshold = 250.0;

        const std::size_t withinThreshold =
            static_cast<std::size_t>(
                std::count_if(
                    latencies.begin(),
                    latencies.end(),
                    [serviceLevelThreshold](double value) {
                        return value <= serviceLevelThreshold;
                    }
                )
            );

        const double empiricalServiceLevel =
            static_cast<double>(withinThreshold)
            / static_cast<double>(latencies.size());

        const double theoreticalServiceLevel =
            responseLatency.cdf(
                serviceLevelThreshold
            );

        std::cout
            << "Empirical P(X<=250 ms): "
            << empiricalServiceLevel
            << "\n";

        std::cout
            << "Theoretical P(X<=250 ms): "
            << theoreticalServiceLevel
            << "\n";

        // --------------------------------------------------------------------
        // Stage 10: Monte Carlo rare-event style estimate
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 10: Monte Carlo estimation\n"
            << "------------------------------------------------------------\n";

        const std::size_t monteCarloTrials = 200000;

        std::size_t highLatencyCount = 0;

        for (std::size_t i = 0; i < monteCarloTrials; ++i) {
            const double simulatedLatency =
                responseLatency.sample(engine);

            if (simulatedLatency > 300.0) {
                ++highLatencyCount;
            }
        }

        const double estimatedHighLatencyProbability =
            static_cast<double>(highLatencyCount)
            / static_cast<double>(monteCarloTrials);

        const double exactHighLatencyProbability =
            1.0 - responseLatency.cdf(300.0);

        std::cout
            << "Estimated P(X>300 ms): "
            << estimatedHighLatencyProbability
            << "\n";

        std::cout
            << "Theoretical P(X>300 ms): "
            << exactHighLatencyProbability
            << "\n";

        // --------------------------------------------------------------------
        // Stage 11: Text histogram
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 11: Latency histogram\n"
            << "------------------------------------------------------------\n";

        std::vector<double> histogramData(
            latencies.begin(),
            latencies.begin() + 5000
        );

        printHistogram(
            histogramData,
            15,
            50
        );

        // --------------------------------------------------------------------
        // Stage 12: Binomial and Poisson relationship
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 12: Binomial-to-Poisson approximation\n"
            << "------------------------------------------------------------\n";

        /*
         * When n is large and p is small, Binomial(n,p) can be approximated
         * by Poisson(lambda=np).
         */
        const int approximationN = 1000;
        const double approximationP = 0.004;
        const double approximationLambda =
            approximationN * approximationP;

        BinomialDistribution rareFailures(
            approximationN,
            approximationP
        );

        PoissonDistribution poissonApproximation(
            approximationLambda
        );

        std::cout
            << std::setw(5) << "k"
            << std::setw(18) << "Binomial"
            << std::setw(18) << "Poisson"
            << std::setw(18) << "Difference"
            << "\n";

        for (int k = 0; k <= 12; ++k) {
            const double binomialProbability =
                rareFailures.pmf(k);

            const double poissonProbability =
                poissonApproximation.pmf(k);

            std::cout
                << std::setw(5) << k
                << std::setw(18) << binomialProbability
                << std::setw(18) << poissonProbability
                << std::setw(18)
                << std::abs(
                    binomialProbability -
                    poissonProbability
                )
                << "\n";
        }

        // --------------------------------------------------------------------
        // Stage 13: Exponential memoryless property
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 13: Exponential memorylessness\n"
            << "------------------------------------------------------------\n";

        const double memorylessRate = 0.25;
        const double elapsed = 3.0;
        const double additionalWait = 2.0;

        ExponentialDistribution memorylessModel(
            memorylessRate
        );

        /*
         * P(T>s+t | T>s) = P(T>t)
         */
        const double conditionalSurvival =
            memorylessModel.survival(
                elapsed + additionalWait
            )
            / memorylessModel.survival(elapsed);

        const double directSurvival =
            memorylessModel.survival(additionalWait);

        std::cout
            << "Conditional survival: "
            << conditionalSurvival
            << "\n";

        std::cout
            << "Direct survival: "
            << directSurvival
            << "\n";

        // --------------------------------------------------------------------
        // Stage 14: Operational capacity calculation
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 14: Capacity planning calculation\n"
            << "------------------------------------------------------------\n";

        /*
         * If a service receives 120 requests/minute, the expected count in
         * 10 minutes is 1200. A Poisson model can estimate the probability
         * that demand exceeds a capacity threshold.
         */
        const double tenMinuteRate =
            requestsPerMinute * 10.0;

        PoissonDistribution tenMinuteRequests(
            tenMinuteRate
        );

        const int capacity =
            1300;

        const double probabilityExceedsCapacity =
            1.0 - tenMinuteRequests.cdf(
                capacity - 1
            );

        std::cout
            << "Expected requests in 10 minutes: "
            << tenMinuteRequests.mean()
            << "\n";

        std::cout
            << "Capacity: "
            << capacity
            << "\n";

        std::cout
            << "P(demand >= capacity): "
            << probabilityExceedsCapacity
            << "\n";

        // --------------------------------------------------------------------
        // Stage 15: Error conditions
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 15: Validation tests\n"
            << "------------------------------------------------------------\n";

        try {
            BernoulliDistribution invalidBernoulli(1.2);
        }
        catch (const std::exception& error) {
            std::cout
                << "Rejected invalid Bernoulli model: "
                << error.what()
                << "\n";
        }

        try {
            BinomialDistribution invalidBinomial(-5, 0.5);
        }
        catch (const std::exception& error) {
            std::cout
                << "Rejected invalid Binomial model: "
                << error.what()
                << "\n";
        }

        try {
            PoissonDistribution invalidPoisson(-1.0);
        }
        catch (const std::exception& error) {
            std::cout
                << "Rejected invalid Poisson model: "
                << error.what()
                << "\n";
        }

        try {
            ExponentialDistribution invalidExponential(0.0);
        }
        catch (const std::exception& error) {
            std::cout
                << "Rejected invalid Exponential model: "
                << error.what()
                << "\n";
        }

        try {
            UniformDistribution invalidUniform(5.0, 5.0);
        }
        catch (const std::exception& error) {
            std::cout
                << "Rejected invalid Uniform model: "
                << error.what()
                << "\n";
        }

        try {
            GaussianDistribution invalidGaussian(0.0, -2.0);
        }
        catch (const std::exception& error) {
            std::cout
                << "Rejected invalid Gaussian model: "
                << error.what()
                << "\n";
        }

        // --------------------------------------------------------------------
        // Stage 16: Mathematical consistency checks
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 16: Consistency checks\n"
            << "------------------------------------------------------------\n";

        const BernoulliDistribution fairCoin(0.5);

        const double bernoulliTotal =
            fairCoin.pmf(0)
            + fairCoin.pmf(1);

        std::cout
            << "Bernoulli PMF total: "
            << bernoulliTotal
            << "\n";

        double binomialTotal = 0.0;

        for (int k = 0; k <= batchFailures.trials(); ++k) {
            binomialTotal += batchFailures.pmf(k);
        }

        std::cout
            << "Binomial PMF total: "
            << binomialTotal
            << "\n";

        double poissonPartialTotal = 0.0;

        /*
         * The Poisson distribution has infinite support. A finite partial sum
         * approaches one as the upper limit becomes sufficiently large.
         */
        for (int k = 0; k <= 100; ++k) {
            poissonPartialTotal +=
                requestsPerMinuteModel.pmf(k);
        }

        std::cout
            << "Partial Poisson PMF total for k=0..100: "
            << poissonPartialTotal
            << "\n";

        // --------------------------------------------------------------------
        // Stage 17: Architecture and trade-offs
        // --------------------------------------------------------------------

        std::cout
            << "\n------------------------------------------------------------\n"
            << "Stage 17: Modeling architecture\n"
            << "------------------------------------------------------------\n";

        std::cout
            << R"(
Model map:

  Request outcome
      |
      +--> Bernoulli
      |       one request succeeds or fails
      |
      +--> Binomial
      |       failures among a fixed batch
      |
      +--> Poisson
      |       requests arriving during an interval
      |
      +--> Exponential
      |       time between arrivals
      |
      +--> Gaussian
              response-latency approximation

Important production decisions:

  * Validate independence before applying Bernoulli/Binomial assumptions.
  * Validate a stable event rate before applying a basic Poisson process.
  * Do not infer an Exponential waiting-time model merely because events
    are counts.
  * Validate Gaussian assumptions against observed latency data.
  * Use empirical or alternative distributions when real data contradicts
    the selected model.
  * Use log-probabilities for extreme numerical regimes.
  * Seed simulations when reproducibility is required.
  * Monitor distribution parameters because production systems drift.
)";

        // --------------------------------------------------------------------
        // Final output
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "Case study completed successfully.\n"
            << "============================================================\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << "\n";

        return 1;
    }
}
