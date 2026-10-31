#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Features = std::vector<std::vector<double>>;
using Labels = std::vector<int>;

struct Dataset {
    Features X;
    Labels y;

    void validate() const {
        if (X.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid dataset dimensions.");
        }

        const std::size_t width = X.front().size();
        if (width == 0) {
            throw std::invalid_argument("Dataset has no features.");
        }

        for (const auto& row : X) {
            if (row.size() != width) {
                throw std::invalid_argument("Inconsistent feature dimensions.");
            }

            for (double value : row) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Non-finite feature value.");
                }
            }
        }

        for (int label : y) {
            if (label != 0 && label != 1) {
                throw std::invalid_argument("Binary labels must be 0 or 1.");
            }
        }
    }
};

struct Fold {
    std::vector<std::size_t> train;
    std::vector<std::size_t> validation;
};

struct Metrics {
    double accuracy{};
    double precision{};
    double recall{};
    double f1{};
    double log_loss{};
};

struct CVResult {
    std::vector<double> f1_scores;
    double mean_f1{};
    double std_f1{};
    double mean_accuracy{};
    double mean_log_loss{};
};

class LogisticRegression {
public:
    LogisticRegression(double learning_rate, int epochs, double l2)
        : learning_rate_(learning_rate), epochs_(epochs), l2_(l2) {
        if (learning_rate <= 0.0 || epochs <= 0 || l2 < 0.0) {
            throw std::invalid_argument("Invalid logistic-regression parameters.");
        }
    }

    void fit(const Features& X, const Labels& y) {
        if (X.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data.");
        }

        weights_.assign(X.front().size(), 0.0);
        bias_ = 0.0;

        for (int epoch = 0; epoch < epochs_; ++epoch) {
            std::vector<double> gradient(weights_.size(), 0.0);
            double bias_gradient = 0.0;

            for (std::size_t i = 0; i < X.size(); ++i) {
                double linear = bias_;

                for (std::size_t j = 0; j < weights_.size(); ++j) {
                    linear += weights_[j] * X[i][j];
                }

                double probability = sigmoid(linear);
                double error = probability - y[i];

                for (std::size_t j = 0; j < weights_.size(); ++j) {
                    gradient[j] += error * X[i][j];
                }

                bias_gradient += error;
            }

            for (std::size_t j = 0; j < weights_.size(); ++j) {
                gradient[j] =
                    gradient[j] / static_cast<double>(X.size())
                    + l2_ * weights_[j];

                weights_[j] -= learning_rate_ * gradient[j];
            }

            bias_ -= learning_rate_
                * bias_gradient / static_cast<double>(X.size());
        }
    }

    std::vector<double> predict_probability(const Features& X) const {
        if (weights_.empty()) {
            throw std::logic_error("Model has not been fitted.");
        }

        std::vector<double> probabilities;
        probabilities.reserve(X.size());

        for (const auto& row : X) {
            double linear = bias_;

            for (std::size_t j = 0; j < weights_.size(); ++j) {
                linear += weights_[j] * row[j];
            }

            probabilities.push_back(sigmoid(linear));
        }

        return probabilities;
    }

    std::vector<int> predict(const Features& X) const {
        const auto probabilities = predict_probability(X);
        std::vector<int> predictions;

        for (double probability : probabilities) {
            predictions.push_back(probability >= 0.5 ? 1 : 0);
        }

        return predictions;
    }

private:
    static double sigmoid(double value) {
        value = std::max(-60.0, std::min(60.0, value));
        return 1.0 / (1.0 + std::exp(-value));
    }

    double learning_rate_;
    int epochs_;
    double l2_;
    std::vector<double> weights_;
    double bias_{};
};

class DecisionTree {
public:
    DecisionTree(int max_depth, int min_samples_split)
        : max_depth_(max_depth), min_samples_split_(min_samples_split) {
        if (max_depth < 1 || min_samples_split < 2) {
            throw std::invalid_argument("Invalid tree parameters.");
        }
    }

    void fit(const Features& X, const Labels& y) {
        X_ = &X;
        y_ = &y;

        std::vector<std::size_t> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);

        root_ = build(indices, 0);
    }

    std::vector<int> predict(const Features& X) const {
        if (!root_) {
            throw std::logic_error("Tree has not been fitted.");
        }

        std::vector<int> predictions;
        predictions.reserve(X.size());

        for (const auto& row : X) {
            predictions.push_back(predict_node(row, root_.get()));
        }

        return predictions;
    }

    std::vector<double> predict_probability(const Features& X) const {
        const auto predictions = predict(X);
        std::vector<double> probabilities;

        for (int prediction : predictions) {
            probabilities.push_back(prediction ? 0.85 : 0.15);
        }

        return probabilities;
    }

private:
    struct Node {
        bool leaf{};
        int prediction{};
        std::size_t feature{};
        double threshold{};
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    struct Split {
        std::size_t feature{};
        double threshold{};
        double gain{};
        bool valid{};
    };

    double gini(const std::vector<std::size_t>& indices) const {
        if (indices.empty()) return 0.0;

        double positive = 0.0;

        for (std::size_t index : indices) {
            positive += (*y_)[index];
        }

        positive /= static_cast<double>(indices.size());
        return 2.0 * positive * (1.0 - positive);
    }

    int majority(const std::vector<std::size_t>& indices) const {
        std::size_t positive = 0;

        for (std::size_t index : indices) {
            positive += (*y_)[index];
        }

        return positive * 2 >= indices.size() ? 1 : 0;
    }

    Split best_split(const std::vector<std::size_t>& indices) const {
        const double parent = gini(indices);
        Split best{};

        for (std::size_t feature = 0; feature < (*X_)[0].size(); ++feature) {
            std::vector<double> values;

            for (std::size_t index : indices) {
                values.push_back((*X_)[index][feature]);
            }

            std::sort(values.begin(), values.end());
            values.erase(
                std::unique(values.begin(), values.end()),
                values.end()
            );

            for (std::size_t position = 0; position + 1 < values.size(); ++position) {
                const double threshold =
                    (values[position] + values[position + 1]) / 2.0;

                std::vector<std::size_t> left;
                std::vector<std::size_t> right;

                for (std::size_t index : indices) {
                    if ((*X_)[index][feature] <= threshold) {
                        left.push_back(index);
                    } else {
                        right.push_back(index);
                    }
                }

                if (left.empty() || right.empty()) continue;

                const double weighted =
                    (left.size() * gini(left)
                    + right.size() * gini(right))
                    / static_cast<double>(indices.size());

                const double gain = parent - weighted;

                if (!best.valid || gain > best.gain) {
                    best = {feature, threshold, gain, true};
                }
            }
        }

        return best;
    }

    std::unique_ptr<Node> build(
        const std::vector<std::size_t>& indices,
        int depth
    ) {
        auto node = std::make_unique<Node>();
        node->prediction = majority(indices);

        std::set<int> classes;
        for (std::size_t index : indices) {
            classes.insert((*y_)[index]);
        }

        if (
            depth >= max_depth_
            || indices.size() < static_cast<std::size_t>(min_samples_split_)
            || classes.size() == 1
        ) {
            node->leaf = true;
            return node;
        }

        const Split split = best_split(indices);

        if (!split.valid || split.gain <= 0.0) {
            node->leaf = true;
            return node;
        }

        std::vector<std::size_t> left;
        std::vector<std::size_t> right;

        for (std::size_t index : indices) {
            if ((*X_)[index][split.feature] <= split.threshold) {
                left.push_back(index);
            } else {
                right.push_back(index);
            }
        }

        node->feature = split.feature;
        node->threshold = split.threshold;

        node->left = build(left, depth + 1);
        node->right = build(right, depth + 1);

        return node;
    }

    int predict_node(
        const std::vector<double>& row,
        const Node* node
    ) const {
        if (node->leaf) {
            return node->prediction;
        }

        if (row[node->feature] <= node->threshold) {
            return predict_node(row, node->left.get());
        }

        return predict_node(row, node->right.get());
    }

    const Features* X_{};
    const Labels* y_{};
    int max_depth_;
    int min_samples_split_;
    std::unique_ptr<Node> root_;
};

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;

    return std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
}

double standard_deviation(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;

    const double average = mean(values);
    double sum = 0.0;

    for (double value : values) {
        sum += std::pow(value - average, 2);
    }

    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

Metrics calculate_metrics(
    const Labels& actual,
    const std::vector<int>& predicted,
    const std::vector<double>& probabilities
) {
    if (
        actual.size() != predicted.size()
        || actual.size() != probabilities.size()
        || actual.empty()
    ) {
        throw std::invalid_argument("Metric input sizes do not match.");
    }

    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    double loss = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == 1 && predicted[i] == 1) ++tp;
        if (actual[i] == 0 && predicted[i] == 0) ++tn;
        if (actual[i] == 0 && predicted[i] == 1) ++fp;
        if (actual[i] == 1 && predicted[i] == 0) ++fn;

        const double p = std::max(
            1e-15,
            std::min(1.0 - 1e-15, probabilities[i])
        );

        loss -= actual[i] * std::log(p)
            + (1 - actual[i]) * std::log(1.0 - p);
    }

    const double accuracy =
        static_cast<double>(tp + tn) / actual.size();

    const double precision =
        tp + fp == 0 ? 0.0 :
        static_cast<double>(tp) / (tp + fp);

    const double recall =
        tp + fn == 0 ? 0.0 :
        static_cast<double>(tp) / (tp + fn);

    const double f1 =
        precision + recall == 0.0
        ? 0.0
        : 2.0 * precision * recall / (precision + recall);

    return {
        accuracy,
        precision,
        recall,
        f1,
        loss / actual.size()
    };
}

std::vector<Fold> stratified_kfold(
    const Labels& labels,
    int k,
    unsigned seed
) {
    if (k < 2 || k > static_cast<int>(labels.size())) {
        throw std::invalid_argument("Invalid k.");
    }

    std::mt19937 generator(seed);
    std::map<int, std::vector<std::size_t>> by_class;

    for (std::size_t i = 0; i < labels.size(); ++i) {
        by_class[labels[i]].push_back(i);
    }

    std::vector<std::vector<std::size_t>> validation_folds(k);

    for (auto& [label, indices] : by_class) {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (std::size_t i = 0; i < indices.size(); ++i) {
            validation_folds[i % k].push_back(indices[i]);
        }
    }

    std::vector<Fold> folds;

    for (auto& validation : validation_folds) {
        std::shuffle(validation.begin(), validation.end(), generator);

        std::vector<bool> is_validation(labels.size(), false);

        for (std::size_t index : validation) {
            is_validation[index] = true;
        }

        Fold fold;
        fold.validation = validation;

        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (!is_validation[i]) {
                fold.train.push_back(i);
            }
        }

        folds.push_back(std::move(fold));
    }

    return folds;
}

Dataset subset(
    const Dataset& dataset,
    const std::vector<std::size_t>& indices
) {
    Dataset result;

    for (std::size_t index : indices) {
        result.X.push_back(dataset.X[index]);
        result.y.push_back(dataset.y[index]);
    }

    result.validate();
    return result;
}

Dataset generate_dataset(std::size_t size, unsigned seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.8);

    Dataset dataset;

    for (std::size_t i = 0; i < size; ++i) {
        const double x1 = normal(generator);
        const double x2 = normal(generator);
        const double x3 = normal(generator);
        const double x4 = normal(generator);

        const double score =
            1.6 * x1
            - 1.2 * x2
            + 0.7 * x3
            + 0.25 * x4
            + noise(generator);

        dataset.X.push_back({x1, x2, x3, x4});
        dataset.y.push_back(score > 0.35 ? 1 : 0);
    }

    dataset.validate();
    return dataset;
}

using ModelFactory = std::function<std::unique_ptr<LogisticRegression>()>;

CVResult evaluate_logistic(
    const Dataset& dataset,
    const std::vector<Fold>& folds
) {
    CVResult result;

    for (const Fold& fold : folds) {
        Dataset train = subset(dataset, fold.train);
        Dataset validation = subset(dataset, fold.validation);

        LogisticRegression model(0.08, 600, 0.02);
        model.fit(train.X, train.y);

        const auto predictions = model.predict(validation.X);
        const auto probabilities = model.predict_probability(validation.X);

        const Metrics metrics =
            calculate_metrics(validation.y, predictions, probabilities);

        result.f1_scores.push_back(metrics.f1);
    }

    result.mean_f1 = mean(result.f1_scores);
    result.std_f1 = standard_deviation(result.f1_scores);

    return result;
}

CVResult evaluate_tree(
    const Dataset& dataset,
    const std::vector<Fold>& folds
) {
    CVResult result;
    std::vector<double> accuracies;
    std::vector<double> losses;

    for (const Fold& fold : folds) {
        Dataset train = subset(dataset, fold.train);
        Dataset validation = subset(dataset, fold.validation);

        DecisionTree model(4, 6);
        model.fit(train.X, train.y);

        const auto predictions = model.predict(validation.X);
        const auto probabilities = model.predict_probability(validation.X);

        const Metrics metrics =
            calculate_metrics(validation.y, predictions, probabilities);

        result.f1_scores.push_back(metrics.f1);
        accuracies.push_back(metrics.accuracy);
        losses.push_back(metrics.log_loss);
    }

    result.mean_f1 = mean(result.f1_scores);
    result.std_f1 = standard_deviation(result.f1_scores);
    result.mean_accuracy = mean(accuracies);
    result.mean_log_loss = mean(losses);

    return result;
}

int main() {
    try {
        constexpr unsigned seed = 42;

        const Dataset complete = generate_dataset(180, 7);

        std::vector<std::size_t> indices(complete.y.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 split_generator(seed);
        std::shuffle(indices.begin(), indices.end(), split_generator);

        const std::size_t test_size =
            std::max<std::size_t>(1, complete.y.size() / 5);

        const std::vector<std::size_t> test_indices(
            indices.begin(),
            indices.begin() + static_cast<std::ptrdiff_t>(test_size)
        );

        const std::vector<std::size_t> train_indices(
            indices.begin() + static_cast<std::ptrdiff_t>(test_size),
            indices.end()
        );

        const Dataset train = subset(complete, train_indices);
        const Dataset test = subset(complete, test_indices);

        const auto folds = stratified_kfold(train.y, 5, seed);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "ML MODEL SELECTION\n";
        std::cout << "==================\n";
        std::cout << "Samples: " << complete.y.size() << '\n';
        std::cout << "Training: " << train.y.size() << '\n';
        std::cout << "Holdout: " << test.y.size() << "\n\n";

        const CVResult logistic_result =
            evaluate_logistic(train, folds);

        const CVResult tree_result =
            evaluate_tree(train, folds);

        std::cout << "STRATIFIED K-FOLD\n";
        std::cout << "-----------------\n";

        std::cout << "Logistic Regression\n";
        std::cout << "  F1 by fold: ";

        for (double score : logistic_result.f1_scores) {
            std::cout << score << ' ';
        }

        std::cout << "\n  Mean F1: " << logistic_result.mean_f1;
        std::cout << "\n  F1 std: " << logistic_result.std_f1 << "\n\n";

        std::cout << "Decision Tree\n";
        std::cout << "  F1 by fold: ";

        for (double score : tree_result.f1_scores) {
            std::cout << score << ' ';
        }

        std::cout << "\n  Mean F1: " << tree_result.mean_f1;
        std::cout << "\n  F1 std: " << tree_result.std_f1 << "\n\n";

        const bool use_logistic =
            logistic_result.mean_f1 > tree_result.mean_f1
            || (
                logistic_result.mean_f1 == tree_result.mean_f1
                && logistic_result.std_f1 <= tree_result.std_f1
            );

        const std::string selected =
            use_logistic ? "Logistic Regression" : "Decision Tree";

        std::cout << "Selected model: " << selected << "\n\n";

        std::vector<double> repeated_scores;

        for (unsigned repetition = 0; repetition < 3; ++repetition) {
            const auto repeated_folds =
                stratified_kfold(train.y, 5, seed + repetition);

            const CVResult result = use_logistic
                ? evaluate_logistic(train, repeated_folds)
                : evaluate_tree(train, repeated_folds);

            repeated_scores.push_back(result.mean_f1);
        }

        std::cout << "REPEATED STRATIFIED CV\n";
        std::cout << "----------------------\n";
        std::cout << "F1 by repetition: ";

        for (double score : repeated_scores) {
            std::cout << score << ' ';
        }

        std::cout << "\nRepeated mean F1: "
                  << mean(repeated_scores) << "\n";
        std::cout << "Repeated std F1: "
                  << standard_deviation(repeated_scores) << "\n\n";

        Metrics holdout_metrics;

        if (use_logistic) {
            LogisticRegression model(0.08, 600, 0.02);
            model.fit(train.X, train.y);

            const auto predictions = model.predict(test.X);
            const auto probabilities = model.predict_probability(test.X);

            holdout_metrics =
                calculate_metrics(test.y, predictions, probabilities);
        } else {
            DecisionTree model(4, 6);
            model.fit(train.X, train.y);

            const auto predictions = model.predict(test.X);
            const auto probabilities = model.predict_probability(test.X);

            holdout_metrics =
                calculate_metrics(test.y, predictions, probabilities);
        }

        std::cout << "FINAL HOLDOUT\n";
        std::cout << "-------------\n";
        std::cout << "Model: " << selected << '\n';
        std::cout << "Accuracy: " << holdout_metrics.accuracy << '\n';
        std::cout << "Precision: " << holdout_metrics.precision << '\n';
        std::cout << "Recall: " << holdout_metrics.recall << '\n';
        std::cout << "F1: " << holdout_metrics.f1 << '\n';
        std::cout << "Log loss: " << holdout_metrics.log_loss << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Evaluation failed: "
                  << error.what() << '\n';
        return 1;
    }
}
