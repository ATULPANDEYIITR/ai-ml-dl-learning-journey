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
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * Repository-scale customer churn example implemented as a compact
 * LightGBM/CatBoost-inspired boosting engine.
 *
 * The design emphasizes:
 *   - fixed histogram bins
 *   - second-order logistic gradients
 *   - leaf-value optimization
 *   - ordered categorical target statistics
 *   - explicit model-state validation
 *
 * C++17, standard library only.
 */

struct Customer {
    double age;
    double income;
    double tenure;
    std::string region;
    std::string plan;
    int churn;
};

struct EncodedCustomer {
    std::vector<double> features;
    int target;
};

static double sigmoid(double value) {
    value = std::max(-35.0, std::min(35.0, value));
    return 1.0 / (1.0 + std::exp(-value));
}

class OrderedTargetEncoder {
private:
    double priorWeight_;
    double globalMean_{0.0};

    struct Statistics {
        double sum{0.0};
        int count{0};
    };

    std::unordered_map<std::string, Statistics> regionStats_;
    std::unordered_map<std::string, Statistics> planStats_;

    double encode(
        const std::unordered_map<std::string, Statistics>& statistics,
        const std::string& category
    ) const {
        auto it = statistics.find(category);

        if (it == statistics.end()) {
            return globalMean_;
        }

        const Statistics& state = it->second;

        return (state.sum + priorWeight_ * globalMean_) /
               (state.count + priorWeight_);
    }

public:
    explicit OrderedTargetEncoder(double priorWeight = 5.0)
        : priorWeight_(priorWeight) {
        if (priorWeight <= 0.0) {
            throw std::invalid_argument("priorWeight must be positive.");
        }
    }

    std::vector<std::vector<double>> fitTransform(
        const std::vector<Customer>& customers
    ) {
        if (customers.empty()) {
            throw std::invalid_argument("Training data cannot be empty.");
        }

        globalMean_ = std::accumulate(
            customers.begin(),
            customers.end(),
            0.0,
            [](double sum, const Customer& customer) {
                return sum + customer.churn;
            }
        ) / customers.size();

        std::vector<int> order(customers.size());
        std::iota(order.begin(), order.end(), 0);

        std::mt19937 generator(42);
        std::shuffle(order.begin(), order.end(), generator);

        std::vector<std::vector<double>> encoded(customers.size());

        /*
         * Each row receives statistics calculated from previous rows in the
         * permutation. The current target is added only after encoding.
         * This is the key leakage-control idea behind ordered statistics.
         */
        for (int index : order) {
            const Customer& customer = customers[index];

            encoded[index] = {
                customer.age,
                customer.income,
                customer.tenure,
                encode(regionStats_, customer.region),
                encode(planStats_, customer.plan)
            };

            regionStats_[customer.region].sum += customer.churn;
            regionStats_[customer.region].count++;

            planStats_[customer.plan].sum += customer.churn;
            planStats_[customer.plan].count++;
        }

        return encoded;
    }

    std::vector<std::vector<double>> transform(
        const std::vector<Customer>& customers
    ) const {
        std::vector<std::vector<double>> encoded;

        for (const Customer& customer : customers) {
            encoded.push_back({
                customer.age,
                customer.income,
                customer.tenure,
                encode(regionStats_, customer.region),
                encode(planStats_, customer.plan)
            });
        }

        return encoded;
    }
};

struct TreeNode {
    double value{0.0};
    int feature{-1};
    double threshold{0.0};
    int left{-1};
    int right{-1};
};

class HistogramTree {
private:
    int maxDepth_;
    int maxBins_;
    int minLeaf_;

    std::vector<std::vector<double>> edges_;
    std::vector<TreeNode> nodes_;

    int bucket(int feature, double value) const {
        const auto& edge = edges_.at(feature);

        return static_cast<int>(
            std::upper_bound(edge.begin(), edge.end(), value) - edge.begin()
        );
    }

    double gain(
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& left,
        const std::vector<int>& right
    ) const {
        if (
            static_cast<int>(left.size()) < minLeaf_ ||
            static_cast<int>(right.size()) < minLeaf_
        ) {
            return -std::numeric_limits<double>::infinity();
        }

        auto score = [&](const std::vector<int>& indices) {
            double gradientSum = 0.0;
            double hessianSum = 0.0;

            for (int index : indices) {
                gradientSum += gradients[index];
                hessianSum += hessians[index];
            }

            return gradientSum * gradientSum / (hessianSum + 1.0);
        };

        return 0.5 * (score(left) + score(right));
    }

    int grow(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<int>& indices,
        int depth
    ) {
        double gradientSum = 0.0;
        double hessianSum = 0.0;

        for (int index : indices) {
            gradientSum += gradients[index];
            hessianSum += hessians[index];
        }

        TreeNode node;
        node.value = -gradientSum / (hessianSum + 1.0);

        const int nodeId = static_cast<int>(nodes_.size());
        nodes_.push_back(node);

        if (
            depth >= maxDepth_ ||
            static_cast<int>(indices.size()) < minLeaf_ * 2
        ) {
            return nodeId;
        }

        double bestGain = 0.0;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        std::vector<int> bestLeft;
        std::vector<int> bestRight;

        for (int feature = 0; feature < static_cast<int>(X[0].size()); ++feature) {
            std::map<int, std::vector<int>> bins;

            for (int index : indices) {
                bins[bucket(feature, X[index][feature])].push_back(index);
            }

            if (bins.size() < 2) {
                continue;
            }

            std::vector<int> orderedBins;
            for (const auto& entry : bins) {
                orderedBins.push_back(entry.first);
            }

            std::vector<int> left;

            for (size_t position = 0; position + 1 < orderedBins.size(); ++position) {
                const int currentBin = orderedBins[position];

                left.insert(
                    left.end(),
                    bins[currentBin].begin(),
                    bins[currentBin].end()
                );

                std::vector<int> right;

                for (
                    size_t next = position + 1;
                    next < orderedBins.size();
                    ++next
                ) {
                    const auto& source = bins[orderedBins[next]];
                    right.insert(right.end(), source.begin(), source.end());
                }

                double currentGain =
                    gain(gradients, hessians, left, right);

                if (currentGain > bestGain) {
                    bestGain = currentGain;
                    bestFeature = feature;

                    const auto& featureEdges = edges_[feature];

                    if (currentBin < static_cast<int>(featureEdges.size())) {
                        bestThreshold = featureEdges[currentBin];
                    } else {
                        bestThreshold =
                            *std::max_element(
                                indices.begin(),
                                indices.end(),
                                [&](int a, int b) {
                                    return X[a][feature] < X[b][feature];
                                }
                            );
                        bestThreshold = X[bestThreshold][feature];
                    }

                    bestLeft = left;
                    bestRight = right;
                }
            }
        }

        if (bestFeature == -1) {
            return nodeId;
        }

        nodes_[nodeId].feature = bestFeature;
        nodes_[nodeId].threshold = bestThreshold;

        nodes_[nodeId].left = grow(
            X,
            gradients,
            hessians,
            bestLeft,
            depth + 1
        );

        nodes_[nodeId].right = grow(
            X,
            gradients,
            hessians,
            bestRight,
            depth + 1
        );

        return nodeId;
    }

public:
    HistogramTree(int maxDepth, int maxBins, int minLeaf)
        : maxDepth_(maxDepth),
          maxBins_(maxBins),
          minLeaf_(minLeaf) {}

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians
    ) {
        if (X.empty()) {
            throw std::invalid_argument("Cannot train an empty tree.");
        }

        if (
            X.size() != gradients.size() ||
            X.size() != hessians.size()
        ) {
            throw std::invalid_argument(
                "X, gradients and hessians must have equal sizes."
            );
        }

        edges_.clear();
        nodes_.clear();

        for (size_t feature = 0; feature < X[0].size(); ++feature) {
            std::vector<double> values;

            for (const auto& row : X) {
                values.push_back(row[feature]);
            }

            std::sort(values.begin(), values.end());
            values.erase(
                std::unique(values.begin(), values.end()),
                values.end()
            );

            std::vector<double> featureEdges;

            if (values.size() <= static_cast<size_t>(maxBins_)) {
                featureEdges = values;
            } else {
                for (int i = 1; i < maxBins_; ++i) {
                    size_t position =
                        std::min(
                            values.size() - 1,
                            static_cast<size_t>(
                                i * values.size() / maxBins_
                            )
                        );

                    featureEdges.push_back(values[position]);
                }

                std::sort(featureEdges.begin(), featureEdges.end());
                featureEdges.erase(
                    std::unique(featureEdges.begin(), featureEdges.end()),
                    featureEdges.end()
                );
            }

            edges_.push_back(featureEdges);
        }

        std::vector<int> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);

        grow(X, gradients, hessians, indices, 0);
    }

    double predict(const std::vector<double>& row) const {
        if (nodes_.empty()) {
            throw std::logic_error("Tree has not been fitted.");
        }

        int nodeId = 0;

        while (nodes_[nodeId].feature != -1) {
            const TreeNode& node = nodes_[nodeId];

            nodeId =
                row[node.feature] <= node.threshold
                    ? node.left
                    : node.right;
        }

        return nodes_[nodeId].value;
    }
};

class BoostingClassifier {
private:
    int rounds_;
    double learningRate_;
    int maxDepth_;
    int maxBins_;
    double baseScore_{0.0};
    std::vector<HistogramTree> trees_;

public:
    BoostingClassifier(
        int rounds,
        double learningRate,
        int maxDepth,
        int maxBins
    )
        : rounds_(rounds),
          learningRate_(learningRate),
          maxDepth_(maxDepth),
          maxBins_(maxBins) {
        if (rounds <= 0 || learningRate <= 0.0) {
            throw std::invalid_argument(
                "Boosting rounds and learning rate must be positive."
            );
        }
    }

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y
    ) {
        if (X.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training matrix.");
        }

        double positiveRate =
            std::accumulate(y.begin(), y.end(), 0.0) / y.size();

        positiveRate = std::clamp(positiveRate, 1e-6, 1.0 - 1e-6);
        baseScore_ = std::log(positiveRate / (1.0 - positiveRate));

        std::vector<double> raw(X.size(), baseScore_);
        trees_.clear();

        for (int iteration = 0; iteration < rounds_; ++iteration) {
            std::vector<double> gradients(X.size());
            std::vector<double> hessians(X.size());

            for (size_t i = 0; i < X.size(); ++i) {
                double probability = sigmoid(raw[i]);

                gradients[i] = probability - y[i];
                hessians[i] =
                    std::max(1e-6, probability * (1.0 - probability));
            }

            HistogramTree tree(maxDepth_, maxBins_, 2);
            tree.fit(X, gradients, hessians);
            trees_.push_back(std::move(tree));

            for (size_t i = 0; i < X.size(); ++i) {
                raw[i] += learningRate_ * trees_.back().predict(X[i]);
            }
        }
    }

    std::vector<double> predictProbability(
        const std::vector<std::vector<double>>& X
    ) const {
        std::vector<double> probabilities;

        for (const auto& row : X) {
            double raw = baseScore_;

            for (const auto& tree : trees_) {
                raw += learningRate_ * tree.predict(row);
            }

            probabilities.push_back(sigmoid(raw));
        }

        return probabilities;
    }
};

int main() {
    try {
        std::vector<Customer> customers = {
            {23, 31000, 4, "north", "basic", 1},
            {25, 42000, 8, "south", "basic", 1},
            {29, 58000, 18, "west", "pro", 0},
            {31, 61000, 24, "north", "pro", 0},
            {35, 72000, 30, "east", "pro", 0},
            {41, 83000, 44, "west", "enterprise", 0},
            {48, 91000, 52, "north", "enterprise", 0},
            {52, 39000, 9, "south", "basic", 1},
            {27, 36000, 6, "east", "basic", 1},
            {33, 69000, 27, "south", "pro", 0},
            {38, 51000, 15, "east", "basic", 1},
            {45, 87000, 47, "west", "enterprise", 0},
            {22, 28000, 3, "north", "basic", 1},
            {57, 99000, 72, "north", "enterprise", 0},
            {30, 47000, 11, "west", "pro", 0},
            {36, 45000, 10, "south", "basic", 1},
            {43, 78000, 36, "east", "pro", 0},
            {50, 54000, 21, "south", "pro", 1},
            {28, 62000, 22, "north", "pro", 0},
            {39, 34000, 7, "east", "basic", 1}
        };

        const size_t split = 15;

        std::vector<Customer> train(
            customers.begin(),
            customers.begin() + static_cast<long>(split)
        );

        std::vector<Customer> test(
            customers.begin() + static_cast<long>(split),
            customers.end()
        );

        OrderedTargetEncoder encoder(5.0);

        auto trainX = encoder.fitTransform(train);
        auto testX = encoder.transform(test);

        std::vector<int> trainY;
        std::vector<int> testY;

        for (const auto& customer : train) {
            trainY.push_back(customer.churn);
        }

        for (const auto& customer : test) {
            testY.push_back(customer.churn);
        }

        BoostingClassifier model(35, 0.06, 3, 6);
        model.fit(trainX, trainY);

        const auto probabilities =
            model.predictProbability(testX);

        double loss = 0.0;

        for (size_t i = 0; i < testY.size(); ++i) {
            const double p =
                std::clamp(probabilities[i], 1e-12, 1.0 - 1e-12);

            loss -=
                testY[i] * std::log(p) +
                (1 - testY[i]) * std::log(1.0 - p);
        }

        loss /= testY.size();

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Histogram boosting with ordered categorical statistics\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << "Training rows: " << train.size() << '\n';
        std::cout << "Testing rows : " << test.size() << '\n';
        std::cout << "Test log loss: " << loss << '\n';

        std::cout << "Predictions  : ";

        for (double probability : probabilities) {
            std::cout << (probability >= 0.5 ? 1 : 0) << ' ';
        }

        std::cout << "\nProbabilities: ";

        for (double probability : probabilities) {
            std::cout << probability << ' ';
        }

        std::cout << "\n\nDesign constraints\n";
        std::cout << "------------------\n";
        std::cout << "Histogram compression reduces split-search cost.\n";
        std::cout << "Second-order statistics determine leaf values.\n";
        std::cout << "Ordered encoding avoids using each row's own target.\n";
        std::cout << "Tree depth and leaf size constrain overfitting.\n";
        std::cout << "Production implementations require validation, early stopping,\n";
        std::cout << "model serialization, feature monitoring, and drift detection.\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Training failure: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
