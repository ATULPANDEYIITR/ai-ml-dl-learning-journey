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

/*
 * C++17 case study: ensemble-based fraud alert classification.
 *
 * A payment service combines independently trained decision stumps using
 * bagging and AdaBoost. A separate policy layer combines model decisions
 * with configurable voting weights.
 *
 * Compile:
 *   g++ -std=c++17 -O2 ensemble_learning.cpp -o ensemble_learning
 */

struct Transaction {
    double amount;
    double accountAgeDays;
    double transactionsLastHour;
    int fraud;
};

using Dataset = std::vector<Transaction>;

struct Features {
    std::vector<double> values;
    int label;
};

std::vector<Features> encode(const Dataset& data) {
    std::vector<Features> result;
    result.reserve(data.size());

    for (const auto& transaction : data) {
        if (transaction.amount < 0 || transaction.accountAgeDays < 0 ||
            transaction.transactionsLastHour < 0 ||
            (transaction.fraud != 0 && transaction.fraud != 1)) {
            throw std::invalid_argument("Invalid transaction record.");
        }

        // Scaling prevents raw currency amounts from dominating the learner.
        result.push_back({
            {
                std::log1p(transaction.amount) / 10.0,
                std::log1p(transaction.accountAgeDays) / 10.0,
                std::log1p(transaction.transactionsLastHour) / 3.0
            },
            transaction.fraud
        });
    }

    return result;
}

Dataset generateTransactions(std::size_t count, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> amountDistribution(1.0, 3000.0);
    std::uniform_real_distribution<double> ageDistribution(0.0, 2000.0);
    std::uniform_int_distribution<int> velocityDistribution(0, 15);
    std::normal_distribution<double> noise(0.0, 0.9);

    Dataset data;
    data.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const double amount = amountDistribution(generator);
        const double age = ageDistribution(generator);
        const int velocity = velocityDistribution(generator);

        const double score =
            0.0012 * amount -
            0.0010 * age +
            0.32 * velocity -
            1.2 + noise(generator);

        data.push_back({amount, age, static_cast<double>(velocity),
                        score > 1.0 ? 1 : 0});
    }

    return data;
}

struct Split {
    std::vector<Features> training;
    std::vector<Features> testing;
};

Split splitData(std::vector<Features> data, double testFraction,
                unsigned seed) {
    if (!(testFraction > 0.0 && testFraction < 1.0)) {
        throw std::invalid_argument("Invalid test fraction.");
    }

    std::mt19937 generator(seed);
    std::shuffle(data.begin(), data.end(), generator);

    const auto boundary = static_cast<std::size_t>(
        data.size() * (1.0 - testFraction));

    if (boundary == 0 || boundary == data.size()) {
        throw std::invalid_argument("Both partitions must be nonempty.");
    }

    Split split;
    split.training.assign(data.begin(), data.begin() + boundary);
    split.testing.assign(data.begin() + boundary, data.end());
    return split;
}

struct Stump {
    std::size_t feature = 0;
    double threshold = 0.0;
    int leftLabel = 0;
    int rightLabel = 1;

    int predict(const std::vector<double>& features) const {
        if (feature >= features.size()) {
            throw std::invalid_argument("Feature dimension mismatch.");
        }

        return features[feature] <= threshold ? leftLabel : rightLabel;
    }

    void fit(const std::vector<Features>& rows,
             const std::vector<double>& weights) {
        if (rows.empty() || rows.size() != weights.size()) {
            throw std::invalid_argument("Invalid stump training data.");
        }

        const std::size_t dimensions = rows.front().values.size();
        if (dimensions == 0) {
            throw std::invalid_argument("Feature vectors cannot be empty.");
        }

        const double totalWeight =
            std::accumulate(weights.begin(), weights.end(), 0.0);

        if (!(totalWeight > 0.0)) {
            throw std::invalid_argument("Sample weights must sum to a positive value.");
        }

        for (double weight : weights) {
            if (weight < 0 || !std::isfinite(weight)) {
                throw std::invalid_argument("Invalid sample weight.");
            }
        }

        double bestError = std::numeric_limits<double>::infinity();

        for (std::size_t featureIndex = 0;
             featureIndex < dimensions; ++featureIndex) {
            std::vector<double> values;
            values.reserve(rows.size());

            for (const auto& row : rows) {
                if (row.values.size() != dimensions ||
                    (row.label != 0 && row.label != 1)) {
                    throw std::invalid_argument("Inconsistent training rows.");
                }
                values.push_back(row.values[featureIndex]);
            }

            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());

            std::vector<double> thresholds;
            if (values.size() == 1) {
                thresholds.push_back(values.front());
            } else {
                for (std::size_t i = 0; i + 1 < values.size(); ++i) {
                    thresholds.push_back((values[i] + values[i + 1]) / 2.0);
                }
            }

            for (double candidate : thresholds) {
                for (int left : {0, 1}) {
                    const int right = 1 - left;
                    double error = 0.0;

                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        const int predicted =
                            rows[i].values[featureIndex] <= candidate
                            ? left : right;

                        if (predicted != rows[i].label) {
                            error += weights[i];
                        }
                    }

                    if (error < bestError) {
                        bestError = error;
                        feature = featureIndex;
                        threshold = candidate;
                        leftLabel = left;
                        rightLabel = right;
                    }
                }
            }
        }
    }
};

class BaggingModel {
private:
    std::vector<Stump> estimators;

public:
    void fit(const std::vector<Features>& rows,
             std::size_t estimatorCount,
             unsigned seed) {
        if (rows.empty() || estimatorCount == 0) {
            throw std::invalid_argument("Invalid bagging configuration.");
        }

        std::mt19937 generator(seed);
        std::uniform_int_distribution<std::size_t> select(0, rows.size() - 1);

        estimators.clear();

        for (std::size_t modelIndex = 0;
             modelIndex < estimatorCount; ++modelIndex) {
            std::vector<Features> sample;
            sample.reserve(rows.size());

            for (std::size_t i = 0; i < rows.size(); ++i) {
                sample.push_back(rows[select(generator)]);
            }

            std::vector<double> weights(sample.size(), 1.0 / sample.size());
            Stump stump;
            stump.fit(sample, weights);
            estimators.push_back(stump);
        }
    }

    int predict(const std::vector<double>& features) const {
        if (estimators.empty()) {
            throw std::logic_error("Bagging model has not been trained.");
        }

        int positiveVotes = 0;
        for (const auto& model : estimators) {
            positiveVotes += model.predict(features);
        }

        // Strict majority makes ties resolve to the negative class.
        return positiveVotes > static_cast<int>(estimators.size() / 2);
    }
};

class AdaBoostModel {
private:
    std::vector<Stump> estimators;
    std::vector<double> alphas;

public:
    void fit(const std::vector<Features>& rows, std::size_t rounds) {
        if (rows.empty() || rounds == 0) {
            throw std::invalid_argument("Invalid boosting configuration.");
        }

        const std::size_t n = rows.size();
        std::vector<double> weights(n, 1.0 / n);
        estimators.clear();
        alphas.clear();

        for (std::size_t round = 0; round < rounds; ++round) {
            Stump stump;
            stump.fit(rows, weights);

            double error = 0.0;
            std::vector<int> predictions;
            predictions.reserve(n);

            for (const auto& row : rows) {
                const int predicted = stump.predict(row.values);
                predictions.push_back(predicted);
            }

            for (std::size_t i = 0; i < n; ++i) {
                if (predictions[i] != rows[i].label) {
                    error += weights[i];
                }
            }

            // A weak learner with error >= 0.5 cannot improve binary AdaBoost.
            if (error >= 0.5 - 1e-12) {
                break;
            }

            if (error <= 1e-12) {
                estimators.push_back(stump);
                alphas.push_back(10.0);
                break;
            }

            const double alpha = 0.5 * std::log((1.0 - error) / error);
            estimators.push_back(stump);
            alphas.push_back(alpha);

            double normalizer = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const bool correct = predictions[i] == rows[i].label;
                weights[i] *= std::exp(correct ? -alpha : alpha);
                normalizer += weights[i];
            }

            if (!(normalizer > 0.0) || !std::isfinite(normalizer)) {
                throw std::runtime_error("Invalid boosting normalization.");
            }

            for (double& weight : weights) {
                weight /= normalizer;
            }
        }

        if (estimators.empty()) {
            throw std::runtime_error("No useful weak learner was found.");
        }
    }

    double score(const std::vector<double>& features) const {
        double result = 0.0;

        for (std::size_t i = 0; i < estimators.size(); ++i) {
            result += alphas[i] *
                (estimators[i].predict(features) == 1 ? 1.0 : -1.0);
        }

        return result;
    }

    int predict(const std::vector<double>& features) const {
        return score(features) >= 0.0 ? 1 : 0;
    }
};

class WeightedVotingPolicy {
private:
    double baggingWeight;
    double boostingWeight;

public:
    WeightedVotingPolicy(double bagging, double boosting)
        : baggingWeight(bagging), boostingWeight(boosting) {
        if (!std::isfinite(bagging) || !std::isfinite(boosting) ||
            bagging < 0.0 || boosting < 0.0 ||
            bagging + boosting <= 0.0) {
            throw std::invalid_argument("Invalid ensemble voting weights.");
        }
    }

    int decide(int baggingPrediction, int boostingPrediction) const {
        if ((baggingPrediction != 0 && baggingPrediction != 1) ||
            (boostingPrediction != 0 && boostingPrediction != 1)) {
            throw std::invalid_argument("Model predictions must be binary.");
        }

        const double positive =
            baggingWeight * baggingPrediction +
            boostingWeight * boostingPrediction;

        const double total = baggingWeight + boostingWeight;
        return positive > total / 2.0 ? 1 : 0;
    }
};

double accuracy(const std::vector<Features>& rows,
                const std::vector<int>& predictions) {
    if (rows.empty() || rows.size() != predictions.size()) {
        throw std::invalid_argument("Invalid evaluation arrays.");
    }

    std::size_t correct = 0;

    for (std::size_t i = 0; i < rows.size(); ++i) {
        correct += rows[i].label == predictions[i];
    }

    return static_cast<double>(correct) / rows.size();
}

int main() {
    try {
        const auto transactions = generateTransactions(1200, 42);
        const auto encoded = encode(transactions);
        const auto split = splitData(encoded, 0.25, 73);

        Stump baseline;
        baseline.fit(
            split.training,
            std::vector<double>(
                split.training.size(), 1.0 / split.training.size()
            )
        );

        BaggingModel bagging;
        bagging.fit(split.training, 51, 103);

        AdaBoostModel boosting;
        boosting.fit(split.training, 35);

        WeightedVotingPolicy policy(1.0, 1.4);

        std::vector<int> baselinePredictions;
        std::vector<int> baggingPredictions;
        std::vector<int> boostingPredictions;
        std::vector<int> ensemblePredictions;

        for (const auto& row : split.testing) {
            const int baselinePrediction = baseline.predict(row.values);
            const int baggingPrediction = bagging.predict(row.values);
            const int boostingPrediction = boosting.predict(row.values);

            baselinePredictions.push_back(baselinePrediction);
            baggingPredictions.push_back(baggingPrediction);
            boostingPredictions.push_back(boostingPrediction);
            ensemblePredictions.push_back(
                policy.decide(baggingPrediction, boostingPrediction)
            );
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Fraud alert ensemble case study\n";
        std::cout << "Training transactions: " << split.training.size() << '\n';
        std::cout << "Testing transactions:  " << split.testing.size() << "\n\n";

        std::cout << "Decision stump accuracy: "
                  << accuracy(split.testing, baselinePredictions) << '\n';
        std::cout << "Bagging accuracy:        "
                  << accuracy(split.testing, baggingPredictions) << '\n';
        std::cout << "AdaBoost accuracy:       "
                  << accuracy(split.testing, boostingPredictions) << '\n';
        std::cout << "Weighted ensemble:       "
                  << accuracy(split.testing, ensemblePredictions) << '\n';

        std::cout << "\nA positive prediction raises a fraud alert.\n";
        std::cout << "Production systems must also evaluate precision, recall, "
                     "calibration, latency, and false-positive costs.\n";

    } catch (const std::exception& error) {
        std::cerr << "Ensemble execution failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
