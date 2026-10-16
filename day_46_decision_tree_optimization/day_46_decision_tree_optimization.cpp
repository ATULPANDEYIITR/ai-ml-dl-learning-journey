#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/*
 * Repository-independent technical case study:
 * optimize a decision tree used to prioritize industrial equipment inspections.
 *
 * The implementation uses C++17, value-owned nodes, explicit training
 * constraints, cost-complexity pruning, and validation-based model selection.
 * Numeric features are split with binary thresholds.
 */

struct Observation {
    vector<double> features;
    string label;
};

struct Split {
    size_t featureIndex;
    double threshold;
    double gain;
    vector<Observation> left;
    vector<Observation> right;
};

struct TreeNode {
    string prediction;
    map<string, size_t> classCounts;
    size_t sampleCount;
    size_t depth;
    double impurity;
    optional<size_t> featureIndex;
    optional<double> threshold;
    double gain = 0.0;
    unique_ptr<TreeNode> left;
    unique_ptr<TreeNode> right;

    bool isLeaf() const {
        return !featureIndex.has_value();
    }
};

class DecisionTree {
public:
    DecisionTree(
        optional<size_t> maxDepth,
        size_t minSamplesSplit,
        size_t minSamplesLeaf,
        double minGain
    )
        : maxDepth_(maxDepth),
          minSamplesSplit_(minSamplesSplit),
          minSamplesLeaf_(minSamplesLeaf),
          minGain_(minGain) {
        if (minSamplesSplit_ < 2) {
            throw invalid_argument("minSamplesSplit must be at least two.");
        }
        if (minSamplesLeaf_ < 1) {
            throw invalid_argument("minSamplesLeaf must be positive.");
        }
        if (!isfinite(minGain_) || minGain_ < 0) {
            throw invalid_argument("minGain must be finite and non-negative.");
        }
    }

    void fit(const vector<Observation>& observations) {
        featureCount_ = validate(observations);
        root_ = build(observations, 0);
        calculateImportances();
    }

    string predict(const vector<double>& features) const {
        if (!root_) {
            throw logic_error("The tree must be fitted before prediction.");
        }
        if (features.size() != featureCount_) {
            throw invalid_argument("Unexpected prediction feature count.");
        }
        for (double value : features) {
            if (!isfinite(value)) {
                throw invalid_argument("Prediction values must be finite.");
            }
        }

        const TreeNode* node = root_.get();
        while (!node->isLeaf()) {
            const size_t feature = *node->featureIndex;
            node = features[feature] <= *node->threshold
                ? node->left.get()
                : node->right.get();
        }
        return node->prediction;
    }

    void prune(double alpha) {
        if (!root_) {
            throw logic_error("Cannot prune an unfitted tree.");
        }
        if (!isfinite(alpha) || alpha < 0) {
            throw invalid_argument("alpha must be finite and non-negative.");
        }

        pruneNode(*root_, alpha);
        refreshDepths(*root_, 0);
        calculateImportances();
    }

    size_t nodeCount() const {
        return countNodes(root_.get());
    }

    size_t leafCount() const {
        return countLeaves(root_.get());
    }

    size_t depth() const {
        return maxDepth(root_.get());
    }

    const vector<double>& importances() const {
        return importances_;
    }

    void printRules(const vector<string>& featureNames) const {
        if (!root_) {
            throw logic_error("No fitted tree is available.");
        }
        printNode(*root_, featureNames, "");
    }

private:
    optional<size_t> maxDepth_;
    size_t minSamplesSplit_;
    size_t minSamplesLeaf_;
    double minGain_;
    size_t featureCount_ = 0;
    unique_ptr<TreeNode> root_;
    vector<double> importances_;

    static size_t validate(const vector<Observation>& data) {
        if (data.empty()) {
            throw invalid_argument("Training data cannot be empty.");
        }

        const size_t width = data.front().features.size();
        if (width == 0) {
            throw invalid_argument("At least one feature is required.");
        }

        for (const auto& observation : data) {
            if (observation.features.size() != width ||
                observation.label.empty()) {
                throw invalid_argument("Malformed training observation.");
            }
            for (double value : observation.features) {
                if (!isfinite(value)) {
                    throw invalid_argument("Training values must be finite.");
                }
            }
        }

        return width;
    }

    static map<string, size_t> countLabels(
        const vector<Observation>& data
    ) {
        map<string, size_t> counts;
        for (const auto& observation : data) {
            ++counts[observation.label];
        }
        return counts;
    }

    static string majority(const map<string, size_t>& counts) {
        string winner;
        size_t largest = 0;

        // Ordered-map iteration provides deterministic alphabetical tie-breaking.
        for (const auto& [label, count] : counts) {
            if (count > largest) {
                winner = label;
                largest = count;
            }
        }

        if (winner.empty()) {
            throw logic_error("Cannot determine a label for an empty node.");
        }
        return winner;
    }

    static double gini(const vector<Observation>& data) {
        if (data.empty()) return 0.0;

        const auto counts = countLabels(data);
        double sumSquares = 0.0;

        for (const auto& [label, count] : counts) {
            const double probability =
                static_cast<double>(count) / data.size();
            sumSquares += probability * probability;
        }

        return 1.0 - sumSquares;
    }

    static pair<vector<Observation>, vector<Observation>> partition(
        const vector<Observation>& data,
        size_t featureIndex,
        double threshold
    ) {
        vector<Observation> left;
        vector<Observation> right;

        for (const auto& observation : data) {
            if (observation.features[featureIndex] <= threshold) {
                left.push_back(observation);
            } else {
                right.push_back(observation);
            }
        }

        return {move(left), move(right)};
    }

    optional<Split> findBestSplit(
        const vector<Observation>& data
    ) const {
        const double parentImpurity = gini(data);
        optional<Split> best;

        for (size_t feature = 0; feature < featureCount_; ++feature) {
            vector<double> values;
            values.reserve(data.size());

            for (const auto& observation : data) {
                values.push_back(observation.features[feature]);
            }

            sort(values.begin(), values.end());
            values.erase(unique(values.begin(), values.end()), values.end());

            for (size_t i = 0; i + 1 < values.size(); ++i) {
                const double threshold =
                    values[i] + (values[i + 1] - values[i]) / 2.0;

                auto [left, right] = partition(data, feature, threshold);

                if (left.size() < minSamplesLeaf_ ||
                    right.size() < minSamplesLeaf_) {
                    continue;
                }

                const double weightedImpurity =
                    static_cast<double>(left.size()) / data.size() * gini(left) +
                    static_cast<double>(right.size()) / data.size() * gini(right);

                const double gain = parentImpurity - weightedImpurity;

                if (!best || gain > best->gain) {
                    best = Split{
                        feature,
                        threshold,
                        gain,
                        move(left),
                        move(right)
                    };
                }
            }
        }

        return best;
    }

    unique_ptr<TreeNode> build(
        const vector<Observation>& data,
        size_t currentDepth
    ) {
        const auto counts = countLabels(data);

        auto node = make_unique<TreeNode>();
        node->prediction = majority(counts);
        node->classCounts = counts;
        node->sampleCount = data.size();
        node->depth = currentDepth;
        node->impurity = gini(data);

        if (node->impurity == 0.0 ||
            (maxDepth_ && currentDepth >= *maxDepth_) ||
            data.size() < minSamplesSplit_) {
            return node;
        }

        auto split = findBestSplit(data);
        if (!split || split->gain <= 0.0 || split->gain < minGain_) {
            return node;
        }

        node->featureIndex = split->featureIndex;
        node->threshold = split->threshold;
        node->gain = split->gain;

        node->left = build(split->left, currentDepth + 1);
        node->right = build(split->right, currentDepth + 1);

        return node;
    }

    static size_t countNodes(const TreeNode* node) {
        if (!node) return 0;
        return 1 + countNodes(node->left.get()) + countNodes(node->right.get());
    }

    static size_t countLeaves(const TreeNode* node) {
        if (!node) return 0;
        if (node->isLeaf()) return 1;
        return countLeaves(node->left.get()) + countLeaves(node->right.get());
    }

    static size_t maxDepth(const TreeNode* node) {
        if (!node) return 0;
        if (node->isLeaf()) return node->depth;
        return max(maxDepth(node->left.get()), maxDepth(node->right.get()));
    }

    static double leafRisk(const TreeNode& node) {
        return static_cast<double>(node.sampleCount) * node.impurity;
    }

    static double subtreeRisk(const TreeNode& node) {
        if (node.isLeaf()) return leafRisk(node);
        return subtreeRisk(*node.left) + subtreeRisk(*node.right);
    }

    static void pruneNode(TreeNode& node, double alpha) {
        if (node.isLeaf()) return;

        pruneNode(*node.left, alpha);
        pruneNode(*node.right, alpha);

        const size_t leaves = countLeaves(&node);
        if (leaves <= 1) return;

        const double effectiveAlpha =
            (leafRisk(node) - subtreeRisk(node)) /
            static_cast<double>(leaves - 1);

        if (effectiveAlpha <= alpha) {
            // The node's majority class becomes the prediction of the new leaf.
            node.featureIndex.reset();
            node.threshold.reset();
            node.left.reset();
            node.right.reset();
            node.gain = 0.0;
        }
    }

    static void refreshDepths(TreeNode& node, size_t currentDepth) {
        node.depth = currentDepth;
        if (!node.isLeaf()) {
            refreshDepths(*node.left, currentDepth + 1);
            refreshDepths(*node.right, currentDepth + 1);
        }
    }

    void calculateImportances() {
        importances_.assign(featureCount_, 0.0);
        if (!root_) return;

        vector<const TreeNode*> pending{root_.get()};

        while (!pending.empty()) {
            const TreeNode* node = pending.back();
            pending.pop_back();

            if (node->isLeaf()) continue;

            importances_[*node->featureIndex] +=
                node->sampleCount * node->gain;

            pending.push_back(node->left.get());
            pending.push_back(node->right.get());
        }

        const double total = accumulate(
            importances_.begin(), importances_.end(), 0.0
        );

        if (total > 0.0) {
            for (double& value : importances_) {
                value /= total;
            }
        }
    }

    static void printNode(
        const TreeNode& node,
        const vector<string>& featureNames,
        const string& indentation
    ) {
        if (node.isLeaf()) {
            cout << indentation << "Predict " << node.prediction
                 << " from " << node.sampleCount << " records\n";
            return;
        }

        const size_t feature = *node.featureIndex;
        const string name = feature < featureNames.size()
            ? featureNames[feature]
            : "feature_" + to_string(feature);

        cout << indentation << "if " << name << " <= "
             << fixed << setprecision(3) << *node.threshold
             << " (gain=" << node.gain << ")\n";

        printNode(*node.left, featureNames, indentation + "  ");
        cout << indentation << "else\n";
        printNode(*node.right, featureNames, indentation + "  ");
    }
};

static vector<Observation> makeMaintenanceData(
    size_t size,
    unsigned int seed
) {
    mt19937 generator(seed);
    uniform_real_distribution<double> vibration(0.2, 9.0);
    uniform_real_distribution<double> temperature(30.0, 115.0);
    uniform_real_distribution<double> hours(0.0, 20.0);
    uniform_int_distribution<int> faults(0, 5);
    normal_distribution<double> noise(0.0, 1.5);

    vector<Observation> data;
    data.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        const double v = vibration(generator);
        const double t = temperature(generator);
        const double h = hours(generator);
        const double f = static_cast<double>(faults(generator));

        const double risk =
            0.95 * v + 0.025 * t + 0.035 * h + 0.8 * f + noise(generator);

        data.push_back({
            {v, t, h, f},
            risk > 10.0 ? "inspection_required" : "normal"
        });
    }

    return data;
}

static double evaluateAccuracy(
    const DecisionTree& tree,
    const vector<Observation>& data
) {
    if (data.empty()) {
        throw invalid_argument("Evaluation data cannot be empty.");
    }

    size_t correct = 0;
    for (const auto& observation : data) {
        correct += tree.predict(observation.features) == observation.label;
    }

    return static_cast<double>(correct) / data.size();
}

int main() {
    try {
        cout << "Equipment Inspection Decision Tree\n";
        cout << "==================================\n";

        auto data = makeMaintenanceData(320, 2026);

        // The held-out partition evaluates generalization rather than training fit.
        const size_t splitIndex = data.size() * 4 / 5;
        vector<Observation> training(data.begin(), data.begin() + splitIndex);
        vector<Observation> validation(data.begin() + splitIndex, data.end());

        const vector<string> featureNames = {
            "vibration_mm_s",
            "temperature_c",
            "operating_hours",
            "previous_fault_count"
        };

        DecisionTree unrestricted(nullopt, 2, 1, 0.0);
        unrestricted.fit(training);

        DecisionTree controlled(4, 8, 4, 0.002);
        controlled.fit(training);

        DecisionTree postPruned(nullopt, 4, 2, 0.0);
        postPruned.fit(training);

        cout << fixed << setprecision(3);
        cout << "\nTraining records: " << training.size()
             << ", validation records: " << validation.size() << "\n";

        auto report = [&](const string& name, const DecisionTree& tree) {
            cout << "\n" << name << "\n";
            cout << "Validation accuracy: "
                 << evaluateAccuracy(tree, validation) << "\n";
            cout << "Nodes: " << tree.nodeCount()
                 << ", leaves: " << tree.leafCount()
                 << ", depth: " << tree.depth() << "\n";
        };

        report("Unrestricted tree", unrestricted);
        report("Pre-pruned tree", controlled);

        postPruned.prune(0.012);
        report("Cost-complexity post-pruned tree", postPruned);

        cout << "\nFeature importance for the controlled tree\n";
        const auto& importance = controlled.importances();

        for (size_t i = 0; i < featureNames.size(); ++i) {
            cout << left << setw(28) << featureNames[i]
                 << right << importance[i] << "\n";
        }

        cout << "\nRules used for an inspection decision\n";
        controlled.printRules(featureNames);

        const vector<double> incomingAsset{7.6, 101.0, 14.5, 3.0};
        cout << "\nPredicted asset state: "
             << controlled.predict(incomingAsset) << "\n";

        // Fail closed on malformed measurements instead of silently routing
        // NaN values down one side of a threshold comparison.
        try {
            controlled.predict({7.6, numeric_limits<double>::quiet_NaN(), 14.5, 3.0});
        } catch (const invalid_argument& error) {
            cout << "Rejected malformed telemetry: " << error.what() << "\n";
        }

        cout << "\nEngineering note: validation accuracy must be interpreted "
             << "alongside missed failures, inspection cost, and tree complexity. "
             << "Impurity importance is predictive, not causal.\n";

        return 0;
    } catch (const exception& error) {
        cerr << "Execution failed: " << error.what() << "\n";
        return 1;
    }
}
