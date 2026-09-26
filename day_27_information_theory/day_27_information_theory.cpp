/*
 * Information Theory Case Study in C++17
 *
 * Scenario:
 * A production analytics service receives categorical observations and
 * probability forecasts from several models. The system must:
 *
 * 1. Validate probability distributions.
 * 2. Measure uncertainty using Shannon entropy.
 * 3. Measure prediction cost using cross-entropy.
 * 4. Quantify model mismatch using KL divergence.
 * 5. Measure dependence between categorical variables using mutual information.
 * 6. Evaluate classification predictions.
 * 7. Perform feature-selection analysis using information gain.
 * 8. Handle numerical edge cases safely.
 *
 * The implementation uses only the C++ standard library.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Distribution = std::map<std::string, double>;
using JointKey = std::pair<std::string, std::string>;
using JointDistribution = std::map<JointKey, double>;

constexpr double EPSILON = 1e-15;
constexpr double TOLERANCE = 1e-9;

// ============================================================================
// 1. VALIDATION
// ============================================================================

void validateDistribution(const Distribution& distribution) {
    if (distribution.empty()) {
        throw std::invalid_argument(
            "Probability distribution cannot be empty."
        );
    }

    double total = 0.0;

    for (const auto& [outcome, probability] : distribution) {
        if (!std::isfinite(probability) || probability < 0.0) {
            throw std::invalid_argument(
                "Distribution contains an invalid probability for: " +
                outcome
            );
        }

        total += probability;
    }

    if (std::abs(total - 1.0) > TOLERANCE) {
        throw std::invalid_argument(
            "Distribution probabilities must sum to 1."
        );
    }
}

void validateJointDistribution(const JointDistribution& joint) {
    if (joint.empty()) {
        throw std::invalid_argument(
            "Joint distribution cannot be empty."
        );
    }

    double total = 0.0;

    for (const auto& [key, probability] : joint) {
        if (!std::isfinite(probability) || probability < 0.0) {
            throw std::invalid_argument(
                "Joint distribution contains an invalid probability."
            );
        }

        total += probability;
    }

    if (std::abs(total - 1.0) > TOLERANCE) {
        throw std::invalid_argument(
            "Joint probabilities must sum to 1."
        );
    }
}

// ============================================================================
// 2. BASIC INFORMATION MEASURE
// ============================================================================

double selfInformation(double probability, double base = 2.0) {
    if (!(probability > 0.0 && probability <= 1.0)) {
        throw std::invalid_argument(
            "Probability must be in the interval (0, 1]."
        );
    }

    return -std::log(probability) / std::log(base);
}

// ============================================================================
// 3. SHANNON ENTROPY
// ============================================================================

double entropy(
    const Distribution& distribution,
    double base = 2.0
) {
    validateDistribution(distribution);

    double result = 0.0;

    for (const auto& [outcome, probability] : distribution) {
        if (probability > 0.0) {
            result -= probability *
                      std::log(probability) /
                      std::log(base);
        }
    }

    return result;
}

// ============================================================================
// 4. CROSS-ENTROPY
// ============================================================================

double crossEntropy(
    const Distribution& trueDistribution,
    const Distribution& predictedDistribution,
    double base = 2.0
) {
    validateDistribution(trueDistribution);
    validateDistribution(predictedDistribution);

    double result = 0.0;

    for (const auto& [outcome, trueProbability] : trueDistribution) {
        if (trueProbability == 0.0) {
            continue;
        }

        auto iterator = predictedDistribution.find(outcome);

        if (
            iterator == predictedDistribution.end() ||
            iterator->second == 0.0
        ) {
            return std::numeric_limits<double>::infinity();
        }

        result -= trueProbability *
                  std::log(iterator->second) /
                  std::log(base);
    }

    return result;
}

// ============================================================================
// 5. KL DIVERGENCE
// ============================================================================

double klDivergence(
    const Distribution& p,
    const Distribution& q,
    double base = 2.0
) {
    validateDistribution(p);
    validateDistribution(q);

    double result = 0.0;

    for (const auto& [outcome, pProbability] : p) {
        if (pProbability == 0.0) {
            continue;
        }

        auto iterator = q.find(outcome);

        if (
            iterator == q.end() ||
            iterator->second == 0.0
        ) {
            return std::numeric_limits<double>::infinity();
        }

        result += pProbability *
                  std::log(pProbability / iterator->second) /
                  std::log(base);
    }

    return result;
}

// ============================================================================
// 6. MARGINAL DISTRIBUTIONS
// ============================================================================

Distribution marginalX(const JointDistribution& joint) {
    validateJointDistribution(joint);

    Distribution result;

    for (const auto& [key, probability] : joint) {
        result[key.first] += probability;
    }

    return result;
}

Distribution marginalY(const JointDistribution& joint) {
    validateJointDistribution(joint);

    Distribution result;

    for (const auto& [key, probability] : joint) {
        result[key.second] += probability;
    }

    return result;
}

// ============================================================================
// 7. CONDITIONAL DISTRIBUTION
// ============================================================================

Distribution conditionalYGivenX(
    const JointDistribution& joint,
    const std::string& xValue
) {
    Distribution px = marginalX(joint);

    auto xIterator = px.find(xValue);

    if (xIterator == px.end() || xIterator->second == 0.0) {
        throw std::invalid_argument(
            "Requested X value has zero probability."
        );
    }

    Distribution result;

    for (const auto& [key, probability] : joint) {
        if (key.first == xValue) {
            result[key.second] =
                probability / xIterator->second;
        }
    }

    return result;
}

// ============================================================================
// 8. CONDITIONAL ENTROPY
// ============================================================================

double conditionalEntropyYGivenX(
    const JointDistribution& joint
) {
    Distribution px = marginalX(joint);

    double result = 0.0;

    for (const auto& [xValue, probabilityX] : px) {
        Distribution conditional =
            conditionalYGivenX(joint, xValue);

        result += probabilityX * entropy(conditional);
    }

    return result;
}

// ============================================================================
// 9. MUTUAL INFORMATION
// ============================================================================

double mutualInformation(
    const JointDistribution& joint
) {
    validateJointDistribution(joint);

    Distribution px = marginalX(joint);
    Distribution py = marginalY(joint);

    double result = 0.0;

    for (const auto& [key, probabilityXY] : joint) {
        if (probabilityXY == 0.0) {
            continue;
        }

        const double independentProbability =
            px.at(key.first) * py.at(key.second);

        result += probabilityXY *
                  std::log(
                      probabilityXY /
                      independentProbability
                  ) /
                  std::log(2.0);
    }

    // Floating-point arithmetic can produce tiny negative values.
    return std::max(0.0, result);
}

// ============================================================================
// 10. EMPIRICAL DISTRIBUTION
// ============================================================================

Distribution empiricalDistribution(
    const std::vector<std::string>& observations
) {
    if (observations.empty()) {
        throw std::invalid_argument(
            "Cannot estimate a distribution from empty data."
        );
    }

    std::map<std::string, std::size_t> counts;

    for (const auto& value : observations) {
        ++counts[value];
    }

    Distribution result;

    for (const auto& [value, count] : counts) {
        result[value] =
            static_cast<double>(count) /
            static_cast<double>(observations.size());
    }

    return result;
}

// ============================================================================
// 11. INFORMATION GAIN
// ============================================================================

double informationGain(
    const std::vector<std::string>& parentLabels,
    const std::vector<std::vector<std::string>>& childGroups
) {
    if (parentLabels.empty()) {
        throw std::invalid_argument(
            "Parent dataset cannot be empty."
        );
    }

    if (childGroups.empty()) {
        throw std::invalid_argument(
            "At least one child group is required."
        );
    }

    const Distribution parent =
        empiricalDistribution(parentLabels);

    const double parentEntropy = entropy(parent);

    double weightedChildEntropy = 0.0;

    for (const auto& child : childGroups) {
        if (child.empty()) {
            continue;
        }

        const Distribution childDistribution =
            empiricalDistribution(child);

        const double weight =
            static_cast<double>(child.size()) /
            static_cast<double>(parentLabels.size());

        weightedChildEntropy +=
            weight * entropy(childDistribution);
    }

    return parentEntropy - weightedChildEntropy;
}

// ============================================================================
// 12. CLASSIFICATION RECORD
// ============================================================================

struct ClassificationRecord {
    int actualClass;
    std::vector<double> probabilities;
};

// ============================================================================
// 13. CLASSIFICATION ANALYZER
// ============================================================================

class ClassificationAnalyzer {
private:
    std::vector<ClassificationRecord> records;

    static void validateProbabilities(
        const std::vector<double>& probabilities
    ) {
        if (probabilities.empty()) {
            throw std::invalid_argument(
                "Probability vector cannot be empty."
            );
        }

        double total = 0.0;

        for (double probability : probabilities) {
            if (
                !std::isfinite(probability) ||
                probability < 0.0 ||
                probability > 1.0
            ) {
                throw std::invalid_argument(
                    "Invalid class probability."
                );
            }

            total += probability;
        }

        if (std::abs(total - 1.0) > TOLERANCE) {
            throw std::invalid_argument(
                "Class probabilities must sum to 1."
            );
        }
    }

public:
    explicit ClassificationAnalyzer(
        std::vector<ClassificationRecord> input
    )
        : records(std::move(input)) {
        if (records.empty()) {
            throw std::invalid_argument(
                "Classification dataset cannot be empty."
            );
        }

        for (const auto& record : records) {
            validateProbabilities(record.probabilities);

            if (
                record.actualClass < 0 ||
                record.actualClass >=
                    static_cast<int>(record.probabilities.size())
            ) {
                throw std::invalid_argument(
                    "Actual class is outside the probability vector."
                );
            }
        }
    }

    double crossEntropyLoss() const {
        double totalLoss = 0.0;

        for (const auto& record : records) {
            const double probability =
                std::max(
                    record.probabilities[
                        record.actualClass
                    ],
                    EPSILON
                );

            totalLoss -= std::log(probability);
        }

        return totalLoss /
               static_cast<double>(records.size());
    }

    double accuracy() const {
        std::size_t correct = 0;

        for (const auto& record : records) {
            const auto maximumIterator =
                std::max_element(
                    record.probabilities.begin(),
                    record.probabilities.end()
                );

            const int predictedClass =
                static_cast<int>(
                    std::distance(
                        record.probabilities.begin(),
                        maximumIterator
                    )
                );

            if (predictedClass == record.actualClass) {
                ++correct;
            }
        }

        return static_cast<double>(correct) /
               static_cast<double>(records.size());
    }
};

// ============================================================================
// 14. STABLE SOFTMAX
// ============================================================================

std::vector<double> stableSoftmax(
    const std::vector<double>& logits
) {
    if (logits.empty()) {
        throw std::invalid_argument(
            "Logits cannot be empty."
        );
    }

    const double maximum =
        *std::max_element(logits.begin(), logits.end());

    std::vector<double> exponentials;
    exponentials.reserve(logits.size());

    for (double logit : logits) {
        exponentials.push_back(
            std::exp(logit - maximum)
        );
    }

    const double denominator =
        std::accumulate(
            exponentials.begin(),
            exponentials.end(),
            0.0
        );

    std::vector<double> probabilities;

    for (double value : exponentials) {
        probabilities.push_back(value / denominator);
    }

    return probabilities;
}

// ============================================================================
// 15. STABLE CROSS-ENTROPY FROM LOGITS
// ============================================================================

double crossEntropyFromLogits(
    int trueClass,
    const std::vector<double>& logits
) {
    if (
        trueClass < 0 ||
        trueClass >= static_cast<int>(logits.size())
    ) {
        throw std::invalid_argument(
            "True class index is invalid."
        );
    }

    const double maximum =
        *std::max_element(logits.begin(), logits.end());

    double sumExp = 0.0;

    for (double logit : logits) {
        sumExp += std::exp(logit - maximum);
    }

    const double logSumExp =
        maximum + std::log(sumExp);

    return logSumExp - logits[trueClass];
}

// ============================================================================
// 16. REPORTING
// ============================================================================

void printDistribution(
    const Distribution& distribution
) {
    for (const auto& [outcome, probability] : distribution) {
        std::cout
            << "  "
            << outcome
            << " = "
            << std::fixed
            << std::setprecision(6)
            << probability
            << '\n';
    }
}

void printSection(const std::string& title) {
    std::cout << "\n"
              << std::string(78, '=')
              << "\n"
              << title
              << "\n"
              << std::string(78, '=')
              << "\n";
}

// ============================================================================
// 17. MAIN CASE STUDY
// ============================================================================

int main() {
    try {
        // --------------------------------------------------------------------
        // Stage 1: Model uncertainty in incoming categorical data.
        // --------------------------------------------------------------------

        printSection("1. DATA UNCERTAINTY");

        Distribution trafficDistribution{
            {"low", 0.50},
            {"medium", 0.30},
            {"high", 0.20}
        };

        validateDistribution(trafficDistribution);

        std::cout
            << "Traffic distribution:\n";

        printDistribution(trafficDistribution);

        std::cout
            << "Entropy = "
            << entropy(trafficDistribution)
            << " bits\n";

        // --------------------------------------------------------------------
        // Stage 2: Compare a forecast against observed probabilities.
        // --------------------------------------------------------------------

        printSection("2. MODEL EVALUATION");

        Distribution modelForecast{
            {"low", 0.45},
            {"medium", 0.40},
            {"high", 0.15}
        };

        const double trueEntropy =
            entropy(trafficDistribution);

        const double forecastCrossEntropy =
            crossEntropy(
                trafficDistribution,
                modelForecast
            );

        const double modelKL =
            klDivergence(
                trafficDistribution,
                modelForecast
            );

        std::cout
            << "H(P) = "
            << trueEntropy
            << " bits\n";

        std::cout
            << "H(P,Q) = "
            << forecastCrossEntropy
            << " bits\n";

        std::cout
            << "D_KL(P||Q) = "
            << modelKL
            << " bits\n";

        std::cout
            << "H(P) + D_KL(P||Q) = "
            << trueEntropy + modelKL
            << " bits\n";

        // --------------------------------------------------------------------
        // Stage 3: Detect dependence between two operational variables.
        //
        // X = weather condition
        // Y = traffic condition
        // --------------------------------------------------------------------

        printSection("3. DEPENDENCE ANALYSIS");

        JointDistribution weatherTraffic{
            {{"clear", "low"}, 0.35},
            {{"clear", "high"}, 0.15},
            {{"rain", "low"}, 0.10},
            {{"rain", "high"}, 0.40}
        };

        Distribution weather =
            marginalX(weatherTraffic);

        Distribution traffic =
            marginalY(weatherTraffic);

        std::cout
            << "Weather marginal:\n";

        printDistribution(weather);

        std::cout
            << "\nTraffic marginal:\n";

        printDistribution(traffic);

        const double mi =
            mutualInformation(weatherTraffic);

        std::cout
            << "\nMutual information = "
            << mi
            << " bits\n";

        std::cout
            << "H(Traffic|Weather) = "
            << conditionalEntropyYGivenX(weatherTraffic)
            << " bits\n";

        // --------------------------------------------------------------------
        // Stage 4: Feature-selection case.
        //
        // Suppose a decision tree is deciding whether a transaction is
        // fraudulent. A candidate feature divides transactions into groups.
        // --------------------------------------------------------------------

        printSection("4. FEATURE SELECTION");

        std::vector<std::string> fraudLabels{
            "fraud", "fraud", "legitimate", "legitimate",
            "fraud", "legitimate", "legitimate", "fraud"
        };

        std::vector<std::string> suspiciousDevice{
            "fraud", "fraud", "legitimate", "legitimate"
        };

        std::vector<std::string> normalDevice{
            "fraud", "legitimate", "legitimate", "fraud"
        };

        const double gain =
            informationGain(
                fraudLabels,
                {
                    suspiciousDevice,
                    normalDevice
                }
            );

        std::cout
            << "Information gain = "
            << gain
            << " bits\n";

        // --------------------------------------------------------------------
        // Stage 5: Classification model evaluation.
        // --------------------------------------------------------------------

        printSection("5. CLASSIFICATION MODEL");

        std::vector<ClassificationRecord> predictions{
            {0, {0.80, 0.15, 0.05}},
            {2, {0.10, 0.20, 0.70}},
            {1, {0.20, 0.60, 0.20}},
            {0, {0.65, 0.25, 0.10}}
        };

        ClassificationAnalyzer analyzer(
            predictions
        );

        std::cout
            << "Mean cross-entropy = "
            << analyzer.crossEntropyLoss()
            << " nats\n";

        std::cout
            << "Accuracy = "
            << analyzer.accuracy()
            << "\n";

        // --------------------------------------------------------------------
        // Stage 6: Numerical stability.
        // --------------------------------------------------------------------

        printSection("6. NUMERICAL STABILITY");

        std::vector<double> largeLogits{
            1000.0,
            999.0,
            998.0
        };

        const std::vector<double> probabilities =
            stableSoftmax(largeLogits);

        std::cout
            << "Stable softmax probabilities:\n";

        for (double probability : probabilities) {
            std::cout
                << "  "
                << probability
                << '\n';
        }

        std::cout
            << "Stable cross-entropy from logits = "
            << crossEntropyFromLogits(
                0,
                largeLogits
            )
            << " nats\n";

        // --------------------------------------------------------------------
        // Stage 7: Edge condition.
        // If a model assigns zero probability to an event that actually
        // occurs, the cross-entropy and KL divergence become infinite.
        // --------------------------------------------------------------------

        printSection("7. SUPPORT MISMATCH EDGE CASE");

        Distribution actual{
            {"normal", 0.90},
            {"attack", 0.10}
        };

        Distribution unsafeForecast{
            {"normal", 1.00},
            {"attack", 0.00}
        };

        const double edgeCrossEntropy =
            crossEntropy(
                actual,
                unsafeForecast
            );

        const double edgeKL =
            klDivergence(
                actual,
                unsafeForecast
            );

        std::cout
            << "Cross-entropy with impossible true event: "
            << edgeCrossEntropy
            << '\n';

        std::cout
            << "KL divergence with impossible true event: "
            << edgeKL
            << '\n';

        // --------------------------------------------------------------------
        // Stage 8: Mathematical correctness checks.
        // --------------------------------------------------------------------

        printSection("8. CORRECTNESS CHECKS");

        Distribution fairCoin{
            {"heads", 0.50},
            {"tails", 0.50}
        };

        assert(std::abs(
            entropy(fairCoin) - 1.0
        ) < 1e-12);

        assert(std::abs(
            klDivergence(fairCoin, fairCoin)
        ) < 1e-12);

        JointDistribution independent{
            {{"0", "0"}, 0.25},
            {{"0", "1"}, 0.25},
            {{"1", "0"}, 0.25},
            {{"1", "1"}, 0.25}
        };

        assert(std::abs(
            mutualInformation(independent)
        ) < 1e-12);

        JointDistribution perfectlyCorrelated{
            {{"0", "0"}, 0.50},
            {{"1", "1"}, 0.50}
        };

        assert(std::abs(
            mutualInformation(perfectlyCorrelated) - 1.0
        ) < 1e-12);

        Distribution p{
            {"A", 0.50},
            {"B", 0.50}
        };

        Distribution q{
            {"A", 0.25},
            {"B", 0.75}
        };

        assert(std::abs(
            crossEntropy(p, q) -
            (entropy(p) + klDivergence(p, q))
        ) < 1e-12);

        std::cout
            << "All mathematical checks passed.\n";

        // --------------------------------------------------------------------
        // Production design observations.
        // --------------------------------------------------------------------

        printSection("9. ENGINEERING OBSERVATIONS");

        std::cout
            << "1. Validate distributions before calculating information measures.\n"
            << "2. Zero probabilities require explicit handling.\n"
            << "3. KL divergence is directional and is not a metric.\n"
            << "4. Mutual information is symmetric but requires a joint distribution.\n"
            << "5. Stable log-sum-exp calculations should be used with logits.\n"
            << "6. Cross-entropy is especially useful for probabilistic classifiers.\n"
            << "7. Floating-point tolerance is required for probability-sum checks.\n"
            << "8. Data-driven entropy and mutual information depend on sample quality.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Application error: "
            << error.what()
            << '\n';

        return 1;
    }
}
