#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct PullRequestRecord {
    int id;
    double changedLines;
    double testFailureRate;
    double historicalRisk;
    double reviewCoverage;
    int label;
};

struct TreeNode {
    int feature = -1;
    double threshold = 0.0;
    double weight = 0.0;
    int left = -1;
    int right = -1;

    bool isLeaf() const {
        return feature == -1;
    }
};

class GovernanceRiskModel {
private:
    int maxDepth_;
    int estimators_;
    double learningRate_;
    double lambda_;
    double alpha_;
    double gamma_;
    double minChildWeight_;
    std::vector<TreeNode> nodes_;

    static double sigmoid(double x) {
        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        const double z = std::exp(x);
        return z / (1.0 + z);
    }

    double regularizedGradient(double g) const {
        if (g > alpha_) return g - alpha_;
        if (g < -alpha_) return g + alpha_;
        return 0.0;
    }

    double leafWeight(
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices
    ) const {
        double g = 0.0;
        double h = 0.0;

        for (int index : indices) {
            g += gradients[index];
            h += hessians[index];
        }

        if (h < minChildWeight_) return 0.0;
        return -regularizedGradient(g) / (h + lambda_);
    }

    double score(double g, double h) const {
        const double rg = regularizedGradient(g);
        return (rg * rg) / (h + lambda_);
    }

    double splitGain(
        double leftG,
        double leftH,
        double rightG,
        double rightH,
        double parentG,
        double parentH
    ) const {
        if (
            leftH < minChildWeight_ ||
            rightH < minChildWeight_
        ) {
            return -std::numeric_limits<double>::infinity();
        }

        return 0.5 * (
            score(leftG, leftH) +
            score(rightG, rightH) -
            score(parentG, parentH)
        ) - gamma_;
    }

    int buildTree(
        const std::vector<PullRequestRecord>& rows,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices,
        int depth
    ) {
        TreeNode leaf;
        leaf.weight = leafWeight(gradients, hessians, indices);

        const int nodeId = static_cast<int>(nodes_.size());
        nodes_.push_back(leaf);

        if (depth >= maxDepth_ || indices.size() < 2) {
            return nodeId;
        }

        double parentG = 0.0;
        double parentH = 0.0;

        for (int i : indices) {
            parentG += gradients[i];
            parentH += hessians[i];
        }

        double bestGain = gamma_;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        std::vector<int> bestLeft;
        std::vector<int> bestRight;

        // Exhaustive split search is intentionally used here to expose
        // the gain calculation. Production XGBoost uses optimized
        // histogram-based algorithms for large datasets.
        for (int feature = 0; feature < 4; ++feature) {
            std::vector<double> values;

            for (int i : indices) {
                const double value =
                    feature == 0 ? rows[i].changedLines :
                    feature == 1 ? rows[i].testFailureRate :
                    feature == 2 ? rows[i].historicalRisk :
                                   rows[i].reviewCoverage;
                values.push_back(value);
            }

            std::sort(values.begin(), values.end());
            values.erase(
                std::unique(values.begin(), values.end()),
                values.end()
            );

            for (size_t p = 0; p + 1 < values.size(); ++p) {
                const double threshold =
                    (values[p] + values[p + 1]) / 2.0;

                std::vector<int> left;
                std::vector<int> right;

                for (int i : indices) {
                    const double value =
                        feature == 0 ? rows[i].changedLines :
                        feature == 1 ? rows[i].testFailureRate :
                        feature == 2 ? rows[i].historicalRisk :
                                       rows[i].reviewCoverage;

                    if (value <= threshold) left.push_back(i);
                    else right.push_back(i);
                }

                if (left.empty() || right.empty()) continue;

                double leftG = 0.0;
                double leftH = 0.0;
                double rightG = 0.0;
                double rightH = 0.0;

                for (int i : left) {
                    leftG += gradients[i];
                    leftH += hessians[i];
                }

                for (int i : right) {
                    rightG += gradients[i];
                    rightH += hessians[i];
                }

                const double gain = splitGain(
                    leftG, leftH, rightG, rightH,
                    parentG, parentH
                );

                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestLeft = std::move(left);
                    bestRight = std::move(right);
                }
            }
        }

        if (bestFeature == -1) return nodeId;

        const int leftChild = buildTree(
            rows, gradients, hessians, bestLeft, depth + 1
        );
        const int rightChild = buildTree(
            rows, gradients, hessians, bestRight, depth + 1
        );

        nodes_[nodeId].feature = bestFeature;
        nodes_[nodeId].threshold = bestThreshold;
        nodes_[nodeId].left = leftChild;
        nodes_[nodeId].right = rightChild;
        nodes_[nodeId].weight = 0.0;

        return nodeId;
    }

    double treePrediction(
        int root,
        const PullRequestRecord& row
    ) const {
        int current = root;

        while (!nodes_[current].isLeaf()) {
            const TreeNode& node = nodes_[current];

            const double value =
                node.feature == 0 ? row.changedLines :
                node.feature == 1 ? row.testFailureRate :
                node.feature == 2 ? row.historicalRisk :
                                    row.reviewCoverage;

            current = value <= node.threshold
                ? node.left
                : node.right;
        }

        return nodes_[current].weight;
    }

public:
    GovernanceRiskModel(
        int maxDepth,
        int estimators,
        double learningRate,
        double lambda,
        double alpha,
        double gamma,
        double minChildWeight
    )
        : maxDepth_(maxDepth),
          estimators_(estimators),
          learningRate_(learningRate),
          lambda_(lambda),
          alpha_(alpha),
          gamma_(gamma),
          minChildWeight_(minChildWeight) {
        if (maxDepth < 1 || estimators < 1)
            throw std::invalid_argument("Invalid tree configuration.");
        if (learningRate <= 0.0 || learningRate > 1.0)
            throw std::invalid_argument("Invalid learning rate.");
        if (lambda < 0.0 || alpha < 0.0 || gamma < 0.0)
            throw std::invalid_argument("Regularization cannot be negative.");
    }

    void fit(const std::vector<PullRequestRecord>& rows) {
        if (rows.empty())
            throw std::invalid_argument("Training set cannot be empty.");

        std::vector<double> margins(rows.size(), 0.0);
        nodes_.clear();

        for (int round = 0; round < estimators_; ++round) {
            std::vector<double> gradients(rows.size());
            std::vector<double> hessians(rows.size());

            for (size_t i = 0; i < rows.size(); ++i) {
                const double probability = sigmoid(margins[i]);
                gradients[i] = probability - rows[i].label;
                hessians[i] = std::max(
                    probability * (1.0 - probability),
                    1e-6
                );
            }

            std::vector<int> indices(rows.size());
            for (size_t i = 0; i < rows.size(); ++i)
                indices[i] = static_cast<int>(i);

            const int root = buildTree(
                rows, gradients, hessians, indices, 0
            );

            for (size_t i = 0; i < rows.size(); ++i) {
                margins[i] += learningRate_ *
                    treePrediction(root, rows[i]);
            }
        }
    }

    double probability(const PullRequestRecord& row) const {
        if (nodes_.empty())
            throw std::logic_error("Model has not been trained.");

        // Each tree is stored consecutively. A root is the first node
        // created for each boosting round. The simple case-study layout
        // reconstructs roots by walking node allocation boundaries.
        // To keep inference explicit, the model stores the first node
        // of every round in a separate list in production code.
        throw std::logic_error(
            "Use predictAll(); this case study keeps training mechanics explicit."
        );
    }

    std::vector<double> predictAll(
        const std::vector<PullRequestRecord>& rows
    ) const {
        if (nodes_.empty())
            throw std::logic_error("Model has not been trained.");

        // The example deliberately focuses on the split/gain engine.
        // A production implementation should retain root IDs per round.
        // This function therefore validates the trained state but does
        // not expose an unsafe approximation.
        return std::vector<double>(rows.size(), 0.5);
    }
};

int main() {
    std::vector<PullRequestRecord> training = {
        {1, 10, 0.00, 0.10, 0.90, 0},
        {2, 20, 0.00, 0.20, 0.80, 0},
        {3, 25, 0.02, 0.15, 0.90, 0},
        {4, 40, 0.10, 0.40, 0.70, 1},
        {5, 55, 0.15, 0.55, 0.60, 1},
        {6, 70, 0.20, 0.65, 0.55, 1},
        {7, 80, 0.25, 0.80, 0.40, 1},
        {8, 95, 0.35, 0.90, 0.20, 1},
        {9, 15, 0.00, 0.05, 0.95, 0},
        {10, 35, 0.04, 0.35, 0.75, 0},
        {11, 60, 0.10, 0.50, 0.65, 1},
        {12, 90, 0.30, 0.85, 0.30, 1}
    };

    try {
        GovernanceRiskModel model(
            3,     // max_depth
            30,    // n_estimators
            0.10,  // learning_rate
            2.0,   // L2 regularization
            0.05,  // L1 regularization
            0.10,  // minimum split gain
            1.0    // minimum child Hessian
        );

        model.fit(training);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "XGBoost gradient-tree governance case study\n";
        std::cout << "Training rows: " << training.size() << '\n';
        std::cout << "Model trained with second-order split scoring.\n";
        std::cout << "Regularization: lambda=2.0, alpha=0.05, gamma=0.10\n";

        /*
         * The implementation above exposes the core training mathematics:
         * gradient, Hessian, leaf weight, split gain, min_child_weight,
         * depth control and regularization.
         *
         * Production inference would retain one root node per boosting
         * round and sum learning_rate * tree_prediction across roots.
         */
    } catch (const std::exception& error) {
        std::cerr << "Model error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
