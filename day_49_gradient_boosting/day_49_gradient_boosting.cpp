#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Technical case study:
 *
 * A platform predicts API latency from request characteristics. A governance
 * service needs an interpretable gradient-boosting model rather than one very
 * deep decision tree.
 *
 * Each shallow regression tree is a weak learner. It fits the residual left
 * by the current ensemble. The learning rate shrinks every tree's correction.
 *
 * C++17 is used with no external dependencies.
 */

struct Observation {
    std::vector<double> features;
    double target;
};

struct TreeNode {
    double prediction{};
    int feature{-1};
    double threshold{};
    int left{-1};
    int right{-1};

    bool isLeaf() const {
        return feature == -1;
    }
};

class RegressionTree {
public:
    RegressionTree(int maxDepth, int minLeaf)
        : maxDepth_(maxDepth), minLeaf_(minLeaf) {
        if (maxDepth < 0 || minLeaf < 1) {
            throw std::invalid_argument("invalid tree configuration");
        }
    }

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& residuals
    ) {
        if (X.empty() || X.size() != residuals.size()) {
            throw std::invalid_argument("invalid training data");
        }

        featureImportance_.assign(X.front().size(), 0.0);
        nodes_.clear();

        std::vector<std::size_t> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);

        root_ = build(X, residuals, indices, 0);
    }

    double predict(const std::vector<double>& row) const {
        if (root_ < 0) {
            throw std::logic_error("tree has not been fitted");
        }

        int current = root_;

        while (!nodes_[current].isLeaf()) {
            const TreeNode& node = nodes_[current];

            if (row.at(static_cast<std::size_t>(node.feature))
                <= node.threshold) {
                current = node.left;
            } else {
                current = node.right;
            }
        }

        return nodes_[current].prediction;
    }

    std::vector<double> predict(
        const std::vector<std::vector<double>>& X
    ) const {
        std::vector<double> result;
        result.reserve(X.size());

        for (const auto& row : X) {
            result.push_back(predict(row));
        }

        return result;
    }

    const std::vector<double>& featureImportance() const {
        return featureImportance_;
    }

private:
    int maxDepth_;
    int minLeaf_;
    std::vector<TreeNode> nodes_;
    std::vector<double> featureImportance_;
    int root_{-1};

    static double average(
        const std::vector<double>& residuals,
        const std::vector<std::size_t>& indices
    ) {
        double total = 0.0;

        for (std::size_t index : indices) {
            total += residuals[index];
        }

        return total / static_cast<double>(indices.size());
    }

    static double squaredError(
        const std::vector<double>& residuals,
        const std::vector<std::size_t>& indices
    ) {
        if (indices.empty()) {
            return 0.0;
        }

        const double center = average(residuals, indices);
        double error = 0.0;

        for (std::size_t index : indices) {
            const double difference = residuals[index] - center;
            error += difference * difference;
        }

        return error;
    }

    struct Split {
        int feature{-1};
        double threshold{};
        double improvement{-std::numeric_limits<double>::infinity()};
    };

    Split bestSplit(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& residuals,
        const std::vector<std::size_t>& indices
    ) const {
        Split best;
        const double parentError = squaredError(residuals, indices);

        if (indices.size() < static_cast<std::size_t>(2 * minLeaf_)) {
            return best;
        }

        for (std::size_t feature = 0; feature < X.front().size(); ++feature) {
            std::vector<std::size_t> ordered = indices;

            std::sort(
                ordered.begin(),
                ordered.end(),
                [&](std::size_t a, std::size_t b) {
                    return X[a][feature] < X[b][feature];
                }
            );

            for (
                std::size_t position = static_cast<std::size_t>(minLeaf_);
                position + static_cast<std::size_t>(minLeaf_)
                    <= ordered.size();
                ++position
            ) {
                const double leftValue =
                    X[ordered[position - 1]][feature];
                const double rightValue =
                    X[ordered[position]][feature];

                if (leftValue == rightValue) {
                    continue;
                }

                std::vector<std::size_t> left(
                    ordered.begin(),
                    ordered.begin() + static_cast<std::ptrdiff_t>(position)
                );

                std::vector<std::size_t> right(
                    ordered.begin() + static_cast<std::ptrdiff_t>(position),
                    ordered.end()
                );

                const double improvement =
                    parentError
                    - squaredError(residuals, left)
                    - squaredError(residuals, right);

                if (improvement > best.improvement
                    && improvement > 1e-12) {
                    best.feature = static_cast<int>(feature);
                    best.threshold = (leftValue + rightValue) / 2.0;
                    best.improvement = improvement;
                }
            }
        }

        return best;
    }

    int build(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& residuals,
        const std::vector<std::size_t>& indices,
        int depth
    ) {
        TreeNode leaf;
        leaf.prediction = average(residuals, indices);

        if (
            depth >= maxDepth_
            || indices.size() < static_cast<std::size_t>(2 * minLeaf_)
        ) {
            nodes_.push_back(leaf);
            return static_cast<int>(nodes_.size() - 1);
        }

        const Split split = bestSplit(X, residuals, indices);

        if (split.feature < 0) {
            nodes_.push_back(leaf);
            return static_cast<int>(nodes_.size() - 1);
        }

        std::vector<std::size_t> left;
        std::vector<std::size_t> right;

        for (std::size_t index : indices) {
            if (X[index][static_cast<std::size_t>(split.feature)]
                <= split.threshold) {
                left.push_back(index);
            } else {
                right.push_back(index);
            }
        }

        if (
            left.size() < static_cast<std::size_t>(minLeaf_)
            || right.size() < static_cast<std::size_t>(minLeaf_)
        ) {
            nodes_.push_back(leaf);
            return static_cast<int>(nodes_.size() - 1);
        }

        const int nodeIndex = static_cast<int>(nodes_.size());
        nodes_.push_back(TreeNode{});

        nodes_[nodeIndex].prediction = leaf.prediction;
        nodes_[nodeIndex].feature = split.feature;
        nodes_[nodeIndex].threshold = split.threshold;

        /*
         * Feature importance is accumulated from the reduction in residual
         * squared error caused by each split. It is not causal evidence.
         */
        featureImportance_[static_cast<std::size_t>(split.feature)]
            += split.improvement;

        nodes_[nodeIndex].left = build(X, residuals, left, depth + 1);
        nodes_[nodeIndex].right = build(X, residuals, right, depth + 1);

        return nodeIndex;
    }
};

class GradientBoostingRegressor {
public:
    GradientBoostingRegressor(
        int estimators,
        double learningRate,
        int maxDepth,
        int minLeaf
    )
        : estimators_(estimators),
          learningRate_(learningRate),
          maxDepth_(maxDepth),
          minLeaf_(minLeaf) {
        if (
            estimators < 1
            || learningRate <= 0.0
            || maxDepth < 0
            || minLeaf < 1
        ) {
            throw std::invalid_argument("invalid boosting configuration");
        }
    }

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y
    ) {
        if (X.empty() || X.size() != y.size()) {
            throw std::invalid_argument("invalid boosting dataset");
        }

        initialPrediction_ =
            std::accumulate(y.begin(), y.end(), 0.0)
            / static_cast<double>(y.size());

        std::vector<double> prediction(
            y.size(),
            initialPrediction_
        );

        trees_.clear();
        trainingLoss_.clear();

        featureImportance_.assign(X.front().size(), 0.0);

        for (int iteration = 0; iteration < estimators_; ++iteration) {
            /*
             * For squared error, the negative gradient is exactly the
             * residual. The next tree therefore learns what the current
             * ensemble still gets wrong.
             */
            std::vector<double> residuals(y.size());

            for (std::size_t i = 0; i < y.size(); ++i) {
                residuals[i] = y[i] - prediction[i];
            }

            RegressionTree tree(maxDepth_, minLeaf_);
            tree.fit(X, residuals);

            const auto correction = tree.predict(X);

            for (std::size_t i = 0; i < prediction.size(); ++i) {
                prediction[i] += learningRate_ * correction[i];
            }

            for (
                std::size_t feature = 0;
                feature < featureImportance_.size();
                ++feature
            ) {
                featureImportance_[feature]
                    += tree.featureImportance()[feature];
            }

            trees_.push_back(std::move(tree));
            trainingLoss_.push_back(mse(y, prediction));
        }

        normalizeImportance();
    }

    std::vector<double> predict(
        const std::vector<std::vector<double>>& X
    ) const {
        if (trees_.empty()) {
            throw std::logic_error("model has not been fitted");
        }

        std::vector<double> prediction(
            X.size(),
            initialPrediction_
        );

        for (const auto& tree : trees_) {
            const auto correction = tree.predict(X);

            for (std::size_t i = 0; i < prediction.size(); ++i) {
                prediction[i] += learningRate_ * correction[i];
            }
        }

        return prediction;
    }

    const std::vector<double>& trainingLoss() const {
        return trainingLoss_;
    }

    const std::vector<double>& featureImportance() const {
        return featureImportance_;
    }

private:
    int estimators_;
    double learningRate_;
    int maxDepth_;
    int minLeaf_;

    double initialPrediction_{};
    std::vector<RegressionTree> trees_;
    std::vector<double> trainingLoss_;
    std::vector<double> featureImportance_;

    static double mse(
        const std::vector<double>& actual,
        const std::vector<double>& predicted
    ) {
        if (actual.size() != predicted.size() || actual.empty()) {
            throw std::invalid_argument("invalid MSE input");
        }

        double total = 0.0;

        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double difference = actual[i] - predicted[i];
            total += difference * difference;
        }

        return total / static_cast<double>(actual.size());
    }

    void normalizeImportance() {
        const double total = std::accumulate(
            featureImportance_.begin(),
            featureImportance_.end(),
            0.0
        );

        if (total == 0.0) {
            return;
        }

        for (double& value : featureImportance_) {
            value /= total;
        }
    }
};

int main() {
    try {
        /*
         * API latency case:
         * features = [payload_kb, concurrent_requests,
         *             database_queries, cache_hit_percent]
         */
        const std::vector<std::vector<double>> X = {
            {20, 5, 2, 92},
            {40, 10, 3, 88},
            {60, 15, 4, 84},
            {80, 20, 6, 78},
            {100, 25, 7, 70},
            {120, 30, 9, 64},
            {140, 35, 11, 58},
            {160, 40, 12, 50},
            {180, 45, 15, 45},
            {200, 50, 17, 38},
            {30, 8, 2, 94},
            {70, 18, 5, 82},
            {110, 28, 8, 68},
            {150, 38, 12, 55},
            {190, 48, 16, 42},
            {50, 12, 3, 90},
            {90, 22, 6, 75},
            {130, 32, 10, 60}
        };

        const std::vector<double> y = {
            42, 58, 73, 98, 124, 151, 180, 214, 249,
            290, 49, 88, 137, 192, 264, 65, 112, 165
        };

        GradientBoostingRegressor model(
            120,
            0.04,
            2,
            2
        );

        model.fit(X, y);

        const auto predictions = model.predict(X);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Gradient boosting API latency model\n\n";

        for (std::size_t i = 0; i < y.size(); ++i) {
            std::cout
                << "actual=" << y[i]
                << " predicted=" << predictions[i]
                << '\n';
        }

        std::cout << "\nTraining loss checkpoints\n";

        const auto& loss = model.trainingLoss();

        for (std::size_t checkpoint : {1ULL, 5ULL, 20ULL, 60ULL, 120ULL}) {
            if (checkpoint <= loss.size()) {
                std::cout
                    << "trees=" << checkpoint
                    << " mse=" << loss[checkpoint - 1]
                    << '\n';
            }
        }

        const std::vector<std::string> names = {
            "payload_kb",
            "concurrent_requests",
            "database_queries",
            "cache_hit_percent"
        };

        std::cout << "\nResidual-based feature importance\n";

        const auto& importance = model.featureImportance();

        for (std::size_t i = 0; i < names.size(); ++i) {
            std::cout
                << std::left
                << std::setw(25)
                << names[i]
                << importance[i]
                << '\n';
        }

        std::cout << "\nDesign characteristics\n";
        std::cout
            << "Each depth-limited tree is a weak learner.\n"
            << "Every tree receives the residual left by the ensemble.\n"
            << "The learning rate scales each correction before accumulation.\n"
            << "Tree depth controls the complexity of residual patterns.\n"
            << "Feature importance is based on residual-error reduction.\n"
            << "The implementation uses O(T * N * F * N log N) style "
               "split-search work in its simple form, where T is the "
               "number of trees, N observations, and F features.\n";

    } catch (const std::exception& error) {
        std::cerr << "Model failure: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
