#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct ConfusionMatrix {
    long long truePositive = 0;
    long long falsePositive = 0;
    long long trueNegative = 0;
    long long falseNegative = 0;

    long long total() const {
        return truePositive + falsePositive + trueNegative + falseNegative;
    }
};

double safeDivide(double numerator, double denominator) {
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

double accuracy(const ConfusionMatrix& cm) {
    return safeDivide(
        static_cast<double>(cm.truePositive + cm.trueNegative),
        static_cast<double>(cm.total())
    );
}

double precision(const ConfusionMatrix& cm) {
    return safeDivide(
        static_cast<double>(cm.truePositive),
        static_cast<double>(cm.truePositive + cm.falsePositive)
    );
}

double recall(const ConfusionMatrix& cm) {
    return safeDivide(
        static_cast<double>(cm.truePositive),
        static_cast<double>(cm.truePositive + cm.falseNegative)
    );
}

double specificity(const ConfusionMatrix& cm) {
    return safeDivide(
        static_cast<double>(cm.trueNegative),
        static_cast<double>(cm.trueNegative + cm.falsePositive)
    );
}

double f1Score(const ConfusionMatrix& cm) {
    const double p = precision(cm);
    const double r = recall(cm);
    return safeDivide(2.0 * p * r, p + r);
}

void validateBinaryLabels(
    const std::vector<int>& actual,
    const std::vector<int>& predicted
) {
    if (actual.empty()) {
        throw std::invalid_argument("At least one observation is required.");
    }

    if (actual.size() != predicted.size()) {
        throw std::invalid_argument("Actual and predicted sizes differ.");
    }

    auto valid = [](int value) {
        return value == 0 || value == 1;
    };

    if (!std::all_of(actual.begin(), actual.end(), valid) ||
        !std::all_of(predicted.begin(), predicted.end(), valid)) {
        throw std::invalid_argument("Binary labels must be 0 or 1.");
    }
}

ConfusionMatrix buildBinaryConfusionMatrix(
    const std::vector<int>& actual,
    const std::vector<int>& predicted
) {
    validateBinaryLabels(actual, predicted);

    ConfusionMatrix cm;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == 1 && predicted[i] == 1) {
            ++cm.truePositive;
        } else if (actual[i] == 0 && predicted[i] == 1) {
            ++cm.falsePositive;
        } else if (actual[i] == 0 && predicted[i] == 0) {
            ++cm.trueNegative;
        } else {
            ++cm.falseNegative;
        }
    }

    return cm;
}

void printMetrics(const ConfusionMatrix& cm) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy:    " << accuracy(cm) << '\n';
    std::cout << "Precision:   " << precision(cm) << '\n';
    std::cout << "Recall:      " << recall(cm) << '\n';
    std::cout << "Specificity: " << specificity(cm) << '\n';
    std::cout << "F1:          " << f1Score(cm) << '\n';
}

std::vector<int> thresholdPredictions(
    const std::vector<double>& probabilities,
    double threshold
) {
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("Threshold must be between 0 and 1.");
    }

    for (double probability : probabilities) {
        if (probability < 0.0 || probability > 1.0) {
            throw std::invalid_argument(
                "Classification probabilities must be in [0, 1]."
            );
        }
    }

    std::vector<int> predictions;
    predictions.reserve(probabilities.size());

    for (double probability : probabilities) {
        predictions.push_back(probability >= threshold ? 1 : 0);
    }

    return predictions;
}

struct RepositoryChange {
    std::string pullRequest;
    std::string author;
    bool securitySensitive;
    bool documentationOnly;
};

class RepositoryGovernanceEngine {
public:
    explicit RepositoryGovernanceEngine(double securityThreshold)
        : securityThreshold_(securityThreshold) {
        if (securityThreshold < 0.0 || securityThreshold > 1.0) {
            throw std::invalid_argument("Invalid security threshold.");
        }
    }

    std::vector<int> classifySecurityReview(
        const std::vector<double>& riskScores
    ) const {
        return thresholdPredictions(riskScores, securityThreshold_);
    }

    bool mergeEligible(
        bool requiredChecksPassed,
        bool requiredApprovalPresent,
        bool unresolvedConversation,
        bool branchIsProtected,
        bool directPushAttempt
    ) const {
        /*
         * The metric engine is intentionally separate from merge governance.
         * Classification metrics evaluate predictions; policy evaluation
         * determines whether a repository state satisfies merge conditions.
         */
        if (directPushAttempt && branchIsProtected) {
            return false;
        }

        return requiredChecksPassed &&
               requiredApprovalPresent &&
               !unresolvedConversation;
    }

private:
    double securityThreshold_;
};

struct MulticlassMatrix {
    std::vector<std::string> labels;
    std::vector<std::vector<int>> values;

    explicit MulticlassMatrix(const std::vector<std::string>& classLabels)
        : labels(classLabels),
          values(
              classLabels.size(),
              std::vector<int>(classLabels.size(), 0)
          ) {}

    std::size_t indexOf(const std::string& label) const {
        auto iterator = std::find(labels.begin(), labels.end(), label);

        if (iterator == labels.end()) {
            throw std::invalid_argument("Unknown class label: " + label);
        }

        return static_cast<std::size_t>(
            std::distance(labels.begin(), iterator)
        );
    }

    void add(const std::string& actual, const std::string& predicted) {
        values[indexOf(actual)][indexOf(predicted)]++;
    }
};

ConfusionMatrix oneVsRest(
    const MulticlassMatrix& matrix,
    std::size_t classIndex
) {
    if (classIndex >= matrix.labels.size()) {
        throw std::out_of_range("Invalid class index.");
    }

    ConfusionMatrix cm;

    for (std::size_t row = 0; row < matrix.labels.size(); ++row) {
        for (std::size_t column = 0; column < matrix.labels.size(); ++column) {
            const int value = matrix.values[row][column];

            if (row == classIndex && column == classIndex) {
                cm.truePositive += value;
            } else if (row == classIndex) {
                cm.falseNegative += value;
            } else if (column == classIndex) {
                cm.falsePositive += value;
            } else {
                cm.trueNegative += value;
            }
        }
    }

    return cm;
}

void printMulticlassMatrix(const MulticlassMatrix& matrix) {
    std::cout << "\nMulticlass confusion matrix\n";
    std::cout << std::setw(15) << "Actual";

    for (const auto& label : matrix.labels) {
        std::cout << std::setw(15) << label;
    }

    std::cout << '\n';

    for (std::size_t row = 0; row < matrix.labels.size(); ++row) {
        std::cout << std::setw(15) << matrix.labels[row];

        for (std::size_t column = 0; column < matrix.labels.size(); ++column) {
            std::cout << std::setw(15) << matrix.values[row][column];
        }

        std::cout << '\n';
    }
}

void runBinaryCaseStudy() {
    /*
     * The scenario represents an automated security-review classifier.
     * Positive means that a Pull Request should receive specialist security
     * review. False negatives are important because they represent risky
     * changes incorrectly treated as routine changes.
     */
    const std::vector<int> actual{
        1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0
    };

    const std::vector<double> riskScores{
        0.92, 0.12, 0.66, 0.31, 0.27, 0.58,
        0.44, 0.81, 0.73, 0.18, 0.39, 0.08
    };

    std::cout << "=== Security-review classification ===\n";

    for (double threshold : {0.30, 0.50, 0.70}) {
        const auto predicted = thresholdPredictions(riskScores, threshold);
        const auto cm = buildBinaryConfusionMatrix(actual, predicted);

        std::cout << "\nThreshold: " << threshold << '\n';
        std::cout << "TP=" << cm.truePositive
                  << " FP=" << cm.falsePositive
                  << " TN=" << cm.trueNegative
                  << " FN=" << cm.falseNegative << '\n';

        printMetrics(cm);
    }
}

void runMulticlassCaseStudy() {
    const std::vector<std::string> labels{
        "security", "feature", "documentation"
    };

    const std::vector<std::string> actual{
        "security", "feature", "feature", "documentation",
        "security", "documentation", "feature", "security",
        "documentation", "feature"
    };

    const std::vector<std::string> predicted{
        "security", "feature", "security", "documentation",
        "feature", "documentation", "feature", "security",
        "documentation", "security"
    };

    MulticlassMatrix matrix(labels);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        matrix.add(actual[i], predicted[i]);
    }

    printMulticlassMatrix(matrix);

    double macroF1 = 0.0;
    int totalSupport = 0;
    double weightedF1 = 0.0;
    int correct = 0;

    for (std::size_t i = 0; i < labels.size(); ++i) {
        const auto cm = oneVsRest(matrix, i);
        const int support = cm.truePositive + cm.falseNegative;

        macroF1 += f1Score(cm);
        weightedF1 += f1Score(cm) * support;
        totalSupport += support;
        correct += matrix.values[i][i];

        std::cout << "\nClass: " << labels[i] << '\n';
        printMetrics(cm);
    }

    macroF1 /= static_cast<double>(labels.size());
    weightedF1 = safeDivide(weightedF1, totalSupport);

    std::cout << "\nMulticlass accuracy: "
              << safeDivide(correct, totalSupport) << '\n';
    std::cout << "Macro F1: " << macroF1 << '\n';
    std::cout << "Weighted F1: " << weightedF1 << '\n';
}

void runGovernancePolicyCaseStudy() {
    RepositoryGovernanceEngine engine(0.50);

    const bool checksPassed = true;
    const bool approvalPresent = true;
    const bool unresolvedConversation = false;
    const bool protectedBranch = true;
    const bool directPushAttempt = false;

    const bool eligible = engine.mergeEligible(
        checksPassed,
        approvalPresent,
        unresolvedConversation,
        protectedBranch,
        directPushAttempt
    );

    std::cout << "\n=== Repository policy evaluation ===\n";
    std::cout << "Merge eligible: " << std::boolalpha << eligible << '\n';

    const bool blockedDirectPush = engine.mergeEligible(
        true,
        true,
        false,
        true,
        true
    );

    std::cout << "Direct push to protected branch accepted: "
              << blockedDirectPush << '\n';
}

int main() {
    try {
        runBinaryCaseStudy();
        runMulticlassCaseStudy();
        runGovernancePolicyCaseStudy();

        std::cout << "\nCase study completed successfully.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Evaluation failed: " << error.what() << '\n';
        return 1;
    }
}
