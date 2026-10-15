#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Repository release-risk classification case study.
 *
 * A deployment governance team wants to understand how a decision tree
 * partitions historical release observations into low-risk and high-risk
 * groups. The program builds a binary classification tree using either
 * entropy reduction or Gini reduction.
 *
 * Compile:
 *   g++ -std=c++17 -O2 decision_tree.cpp -o decision_tree
 *
 * The sample is illustrative, not a substitute for an enforced release policy.
 */

using Label = std::string;

struct Observation {
    double coverage;
    int failedTests;
    int securityFindings;
    Label decision;
};

struct Counts {
    int eligible = 0;
    int manualReview = 0;

    int total() const {
        return eligible + manualReview;
    }

    void add(const Label& label) {
        if (label == "eligible") {
            ++eligible;
        } else if (label == "manual_review") {
            ++manualReview;
        } else {
            throw std::invalid_argument("Unknown classification label.");
        }
    }

    Label majority() const {
        // Ties are conservatively resolved toward manual review.
        return eligible > manualReview ? "eligible" : "manual_review";
    }
};

enum class Criterion {
    Entropy,
    Gini
};

enum class Feature {
    Coverage,
    FailedTests,
    SecurityFindings
};

std::string featureName(Feature feature) {
    switch (feature) {
        case Feature::Coverage: return "coverage";
        case Feature::FailedTests: return "failed_tests";
        case Feature::SecurityFindings: return "security_findings";
    }
    throw std::logic_error("Unknown feature.");
}

double entropy(const Counts& counts) {
    const double total = static_cast<double>(counts.total());
    if (total == 0.0) return 0.0;

    double result = 0.0;
    for (int count : {counts.eligible, counts.manualReview}) {
        if (count == 0) continue;
        const double probability = count / total;
        result -= probability * std::log2(probability);
    }
    return result;
}

double gini(const Counts& counts) {
    const double total = static_cast<double>(counts.total());
    if (total == 0.0) return 0.0;

    const double eligible = counts.eligible / total;
    const double review = counts.manualReview / total;
    return 1.0 - eligible * eligible - review * review;
}

double impurity(const Counts& counts, Criterion criterion) {
    return criterion == Criterion::Entropy ? entropy(counts) : gini(counts);
}

double valueOf(const Observation& observation, Feature feature) {
    switch (feature) {
        case Feature::Coverage: return observation.coverage;
        case Feature::FailedTests:
            return static_cast<double>(observation.failedTests);
        case Feature::SecurityFindings:
            return static_cast<double>(observation.securityFindings);
    }
    throw std::logic_error("Unknown feature.");
}

struct SplitCandidate {
    Feature feature;
    double threshold;
    double gain;
    std::vector<Observation> left;
    std::vector<Observation> right;
};

struct Node {
    Counts counts;
    Label prediction;
    double nodeImpurity = 0.0;
    double gain = 0.0;
    int depth = 0;
    std::optional<Feature> feature;
    double threshold = 0.0;
    std::optional<std::size_t> left;
    std::optional<std::size_t> right;

    bool isLeaf() const {
        return !feature.has_value();
    }
};

class DecisionTree {
public:
    DecisionTree(
        Criterion criterion,
        int maxDepth,
        std::size_t minSamplesLeaf,
        double minGain
    )
        : criterion_(criterion),
          maxDepth_(maxDepth),
          minSamplesLeaf_(minSamplesLeaf),
          minGain_(minGain) {
        if (maxDepth < 0) {
            throw std::invalid_argument("Maximum depth cannot be negative.");
        }
        if (minSamplesLeaf == 0) {
            throw std::invalid_argument("Minimum leaf size must be positive.");
        }
        if (minGain < 0.0) {
            throw std::invalid_argument("Minimum gain cannot be negative.");
        }
    }

    void fit(const std::vector<Observation>& observations) {
        validate(observations);
        nodes_.clear();
        featureGain_.clear();
        totalTrainingSamples_ = observations.size();
        build(observations, 0);
    }

    Label predict(const Observation& observation) const {
        if (nodes_.empty()) {
            throw std::logic_error("Fit the tree before prediction.");
        }
        validateObservation(observation);

        std::size_t index = 0;
        while (!nodes_.at(index).isLeaf()) {
            const Node& node = nodes_.at(index);
            const double value = valueOf(observation, *node.feature);

            index = value <= node.threshold
                ? *node.left
                : *node.right;
        }
        return nodes_.at(index).prediction;
    }

    void print() const {
        if (nodes_.empty()) {
            throw std::logic_error("There is no fitted tree.");
        }
        printNode(0, "");
    }

    void printFeatureImportance() const {
        double total = 0.0;
        for (const auto& entry : featureGain_) total += entry.second;

        std::cout << "\nWeighted feature importance\n";
        for (Feature feature : {
                 Feature::Coverage,
                 Feature::FailedTests,
                 Feature::SecurityFindings}) {
            const double gain = featureGain_.count(feature)
                ? featureGain_.at(feature)
                : 0.0;
            const double normalized = total > 0.0 ? gain / total : 0.0;

            std::cout << std::setw(22) << std::left << featureName(feature)
                      << std::fixed << std::setprecision(3)
                      << normalized << '\n';
        }
    }

private:
    Criterion criterion_;
    int maxDepth_;
    std::size_t minSamplesLeaf_;
    double minGain_;
    std::size_t totalTrainingSamples_ = 0;
    std::vector<Node> nodes_;
    std::map<Feature, double> featureGain_;

    static void validateObservation(const Observation& observation) {
        if (!std::isfinite(observation.coverage) ||
            observation.coverage < 0.0 ||
            observation.coverage > 100.0) {
            throw std::invalid_argument(
                "Coverage must be finite and between 0 and 100."
            );
        }
        if (observation.failedTests < 0 ||
            observation.securityFindings < 0) {
            throw std::invalid_argument(
                "Failure and security counts cannot be negative."
            );
        }
        if (observation.decision != "eligible" &&
            observation.decision != "manual_review") {
            throw std::invalid_argument("Unsupported decision label.");
        }
    }

    static void validate(
        const std::vector<Observation>& observations
    ) {
        if (observations.empty()) {
            throw std::invalid_argument("Training data cannot be empty.");
        }
        for (const auto& observation : observations) {
            validateObservation(observation);
        }
    }

    static Counts count(const std::vector<Observation>& observations) {
        Counts counts;
        for (const auto& observation : observations) {
            counts.add(observation.decision);
        }
        return counts;
    }

    std::optional<SplitCandidate> bestSplit(
        const std::vector<Observation>& observations
    ) const {
        const Counts parentCounts = count(observations);
        const double parentImpurity = impurity(parentCounts, criterion_);
        std::optional<SplitCandidate> best;

        const std::vector<Feature> features = {
            Feature::Coverage,
            Feature::FailedTests,
            Feature::SecurityFindings
        };

        for (Feature feature : features) {
            std::vector<double> values;
            for (const auto& observation : observations) {
                values.push_back(valueOf(observation, feature));
            }

            std::sort(values.begin(), values.end());
            values.erase(
                std::unique(values.begin(), values.end()),
                values.end()
            );

            for (std::size_t i = 0; i + 1 < values.size(); ++i) {
                const double threshold =
                    values[i] + (values[i + 1] - values[i]) / 2.0;

                std::vector<Observation> left;
                std::vector<Observation> right;

                for (const auto& observation : observations) {
                    (valueOf(observation, feature) <= threshold
                        ? left
                        : right).push_back(observation);
                }

                if (left.size() < minSamplesLeaf_ ||
                    right.size() < minSamplesLeaf_) {
                    continue;
                }

                const double childImpurity =
                    (static_cast<double>(left.size()) / observations.size()) *
                        impurity(count(left), criterion_) +
                    (static_cast<double>(right.size()) / observations.size()) *
                        impurity(count(right), criterion_);

                const double gain = parentImpurity - childImpurity;

                if (!best || gain > best->gain + 1e-15) {
                    best = SplitCandidate{
                        feature,
                        threshold,
                        gain,
                        std::move(left),
                        std::move(right)
                    };
                }
            }
        }

        return best;
    }

    std::size_t build(
        const std::vector<Observation>& observations,
        int depth
    ) {
        const Counts counts = count(observations);

        Node node;
        node.counts = counts;
        node.prediction = counts.majority();
        node.nodeImpurity = impurity(counts, criterion_);
        node.depth = depth;

        // Push the parent before recursion so child indices remain stable
        // even when the vector reallocates during subsequent insertions.
        const std::size_t nodeIndex = nodes_.size();
        nodes_.push_back(node);

        if (depth >= maxDepth_ ||
            node.nodeImpurity == 0.0 ||
            observations.size() < 2 * minSamplesLeaf_) {
            return nodeIndex;
        }

        const auto candidate = bestSplit(observations);
        if (!candidate || candidate->gain <= minGain_) {
            return nodeIndex;
        }

        nodes_[nodeIndex].feature = candidate->feature;
        nodes_[nodeIndex].threshold = candidate->threshold;
        nodes_[nodeIndex].gain = candidate->gain;

        const double sampleWeight =
            static_cast<double>(observations.size()) /
            static_cast<double>(totalTrainingSamples_);

        featureGain_[candidate->feature] +=
            sampleWeight * candidate->gain;

        const std::size_t leftIndex = build(candidate->left, depth + 1);
        const std::size_t rightIndex = build(candidate->right, depth + 1);

        nodes_[nodeIndex].left = leftIndex;
        nodes_[nodeIndex].right = rightIndex;
        return nodeIndex;
    }

    void printNode(std::size_t index, const std::string& prefix) const {
        const Node& node = nodes_.at(index);

        if (node.isLeaf()) {
            std::cout << prefix << "predict=" << node.prediction
                      << ", samples=" << node.counts.total()
                      << ", entropy=" << std::fixed << std::setprecision(3)
                      << node.nodeImpurity << '\n';
            return;
        }

        std::cout << prefix << "if "
                  << featureName(*node.feature) << " <= "
                  << std::fixed << std::setprecision(3) << node.threshold
                  << ", gain=" << node.gain
                  << ", samples=" << node.counts.total() << '\n';

        printNode(*node.left, prefix + "  then ");
        printNode(*node.right, prefix + "  else ");
    }
};

std::vector<Observation> makeReleaseHistory() {
    std::vector<Observation> observations;

    for (double coverage : {45.0, 58.0, 68.0, 76.0, 84.0, 92.0, 98.0}) {
        for (int failedTests : {0, 1, 3}) {
            for (int securityFindings : {0, 1}) {
                const bool risk =
                    coverage < 75.0 ||
                    failedTests > 0 ||
                    securityFindings > 0;

                observations.push_back({
                    coverage,
                    failedTests,
                    securityFindings,
                    risk ? "manual_review" : "eligible"
                });
            }
        }
    }

    return observations;
}

int main() {
    try {
        const Counts pure{8, 0};
        const Counts balanced{4, 4};

        std::cout << "Entropy of a pure node: " << entropy(pure) << '\n';
        std::cout << "Gini of a balanced node: " << gini(balanced) << '\n';

        auto history = makeReleaseHistory();

        DecisionTree tree(Criterion::Entropy, 4, 2, 1e-12);
        tree.fit(history);

        std::cout << "\nLearned release-risk tree\n";
        tree.print();
        tree.printFeatureImportance();

        Observation candidate{
            89.0,
            0,
            0,
            "eligible"
        };

        std::cout << "\nCandidate classification: "
                  << tree.predict(candidate) << '\n';

        candidate.coverage = 120.0;
        try {
            tree.predict(candidate);
        } catch (const std::invalid_argument& error) {
            std::cout << "Invalid prediction rejected: "
                      << error.what() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Decision-tree failure: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
