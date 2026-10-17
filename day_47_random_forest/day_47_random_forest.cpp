#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct EquipmentRecord {
    vector<double> features;
    string outcome;
};

struct TreeNode {
    int feature = -1;
    double threshold = 0.0;
    string prediction;
    unique_ptr<TreeNode> left;
    unique_ptr<TreeNode> right;

    bool isLeaf() const {
        return feature == -1;
    }
};

// A repository of equipment inspections is modeled as a numerical
// classification problem. The forest estimates whether maintenance is needed.
class InspectionTree {
private:
    mt19937 generator;
    int maxDepth;
    int minLeaf;
    int maxFeatures;
    int featureCount = 0;
    unique_ptr<TreeNode> root;
    vector<double> importance;

    static string majority(const vector<EquipmentRecord>& rows) {
        map<string, int> counts;
        for (const auto& row : rows) {
            ++counts[row.outcome];
        }

        string result;
        int best = -1;
        for (const auto& [label, count] : counts) {
            if (count > best) {
                result = label;
                best = count;
            }
        }
        return result;
    }

    static double gini(const vector<EquipmentRecord>& rows) {
        if (rows.empty()) return 0.0;

        map<string, int> counts;
        for (const auto& row : rows) ++counts[row.outcome];

        double impurity = 1.0;
        for (const auto& [label, count] : counts) {
            (void)label;
            double probability = static_cast<double>(count) / rows.size();
            impurity -= probability * probability;
        }
        return impurity;
    }

    unique_ptr<TreeNode> grow(
        const vector<EquipmentRecord>& rows,
        int depth
    ) {
        auto node = make_unique<TreeNode>();
        node->prediction = majority(rows);

        set<string> classes;
        for (const auto& row : rows) classes.insert(row.outcome);

        if (depth >= maxDepth || rows.size() < static_cast<size_t>(2 * minLeaf)
            || classes.size() == 1) {
            return node;
        }

        vector<int> candidates(featureCount);
        iota(candidates.begin(), candidates.end(), 0);
        shuffle(candidates.begin(), candidates.end(), generator);
        candidates.resize(maxFeatures);

        const double parentImpurity = gini(rows);
        double bestGain = 0.0;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        vector<EquipmentRecord> bestLeft, bestRight;

        for (int feature : candidates) {
            vector<double> values;
            values.reserve(rows.size());

            for (const auto& row : rows) {
                values.push_back(row.features[feature]);
            }

            sort(values.begin(), values.end());
            values.erase(unique(values.begin(), values.end()), values.end());

            for (size_t i = 0; i + 1 < values.size(); ++i) {
                const double threshold = values[i] +
                    (values[i + 1] - values[i]) / 2.0;

                vector<EquipmentRecord> left, right;
                for (const auto& row : rows) {
                    if (row.features[feature] <= threshold) {
                        left.push_back(row);
                    } else {
                        right.push_back(row);
                    }
                }

                if (left.size() < static_cast<size_t>(minLeaf) ||
                    right.size() < static_cast<size_t>(minLeaf)) {
                    continue;
                }

                double weighted =
                    static_cast<double>(left.size()) / rows.size() * gini(left)
                    + static_cast<double>(right.size()) / rows.size() * gini(right);
                double gain = parentImpurity - weighted;

                if (gain > bestGain + 1e-12) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestLeft = move(left);
                    bestRight = move(right);
                }
            }
        }

        if (bestFeature == -1) return node;

        // Importance is accumulated from impurity reduction weighted by
        // the number of observations reaching the split.
        importance[bestFeature] += bestGain * rows.size();

        node->feature = bestFeature;
        node->threshold = bestThreshold;
        node->left = grow(bestLeft, depth + 1);
        node->right = grow(bestRight, depth + 1);
        return node;
    }

public:
    InspectionTree(
        unsigned int seed,
        int depth,
        int leafSize,
        int featureSubset
    ) : generator(seed),
        maxDepth(depth),
        minLeaf(leafSize),
        maxFeatures(featureSubset) {
        if (depth < 0 || leafSize < 1 || featureSubset < 1) {
            throw invalid_argument("Invalid tree configuration.");
        }
    }

    void fit(const vector<EquipmentRecord>& rows) {
        if (rows.empty()) throw invalid_argument("Training set is empty.");

        featureCount = static_cast<int>(rows.front().features.size());
        if (featureCount == 0 || maxFeatures > featureCount) {
            throw invalid_argument("Invalid feature dimension or feature subset.");
        }

        for (const auto& row : rows) {
            if (row.features.size() != static_cast<size_t>(featureCount)
                || row.outcome.empty()) {
                throw invalid_argument("Inconsistent training record.");
            }

            for (double value : row.features) {
                if (!isfinite(value)) {
                    throw invalid_argument("Non-finite feature detected.");
                }
            }
        }

        importance.assign(featureCount, 0.0);
        root = grow(rows, 0);
    }

    string predict(const vector<double>& features) const {
        if (!root) throw logic_error("Tree has not been trained.");
        if (features.size() != static_cast<size_t>(featureCount)) {
            throw invalid_argument("Incorrect prediction feature count.");
        }

        for (double value : features) {
            if (!isfinite(value)) {
                throw invalid_argument("Non-finite prediction feature.");
            }
        }

        const TreeNode* node = root.get();
        while (!node->isLeaf()) {
            node = features[node->feature] <= node->threshold
                ? node->left.get()
                : node->right.get();
        }
        return node->prediction;
    }

    const vector<double>& getImportance() const {
        return importance;
    }
};

class InspectionForest {
private:
    int treeCount;
    int maxDepth;
    int maxFeatures;
    unsigned int seed;
    vector<InspectionTree> trees;
    vector<double> importance;
    vector<set<size_t>> outOfBagIndices;
    double oobAccuracy = -1.0;

public:
    InspectionForest(
        int count,
        int depth,
        int featureSubset,
        unsigned int randomSeed
    ) : treeCount(count),
        maxDepth(depth),
        maxFeatures(featureSubset),
        seed(randomSeed) {
        if (count < 1 || depth < 0 || featureSubset < 1) {
            throw invalid_argument("Invalid forest configuration.");
        }
    }

    void fit(const vector<EquipmentRecord>& dataset) {
        if (dataset.empty()) throw invalid_argument("Dataset is empty.");

        const size_t featureCount = dataset.front().features.size();
        if (maxFeatures > static_cast<int>(featureCount)) {
            throw invalid_argument("Feature subset exceeds feature count.");
        }

        trees.clear();
        outOfBagIndices.clear();
        importance.assign(featureCount, 0.0);

        mt19937 master(seed);
        vector<vector<string>> oobVotes(dataset.size());

        for (int index = 0; index < treeCount; ++index) {
            // Sampling exactly n times with replacement creates a bootstrap
            // dataset of size n. Duplicated rows retain their multiplicity.
            mt19937 sampler(master());
            uniform_int_distribution<size_t> choose(0, dataset.size() - 1);
            vector<EquipmentRecord> bootstrap;
            set<size_t> selected;

            for (size_t i = 0; i < dataset.size(); ++i) {
                size_t selectedIndex = choose(sampler);
                bootstrap.push_back(dataset[selectedIndex]);
                selected.insert(selectedIndex);
            }

            set<size_t> oob;
            for (size_t i = 0; i < dataset.size(); ++i) {
                if (!selected.count(i)) oob.insert(i);
            }

            InspectionTree tree(master(), maxDepth, 2, maxFeatures);
            tree.fit(bootstrap);

            for (size_t i = 0; i < featureCount; ++i) {
                importance[i] += tree.getImportance()[i];
            }

            for (size_t i : oob) {
                oobVotes[i].push_back(tree.predict(dataset[i].features));
            }

            trees.push_back(move(tree));
            outOfBagIndices.push_back(move(oob));
        }

        double total = accumulate(importance.begin(), importance.end(), 0.0);
        if (total > 0.0) {
            for (double& value : importance) value /= total;
        }

        size_t evaluated = 0;
        size_t correct = 0;

        for (size_t i = 0; i < dataset.size(); ++i) {
            if (oobVotes[i].empty()) continue;

            map<string, int> counts;
            for (const string& label : oobVotes[i]) ++counts[label];

            string prediction;
            int highest = -1;
            for (const auto& [label, count] : counts) {
                if (count > highest) {
                    prediction = label;
                    highest = count;
                }
            }

            ++evaluated;
            if (prediction == dataset[i].outcome) ++correct;
        }

        oobAccuracy = evaluated == 0
            ? -1.0
            : static_cast<double>(correct) / evaluated;
    }

    string predict(const vector<double>& features) const {
        if (trees.empty()) throw logic_error("Forest has not been trained.");

        map<string, int> votes;
        for (const auto& tree : trees) ++votes[tree.predict(features)];

        string prediction;
        int highest = -1;
        for (const auto& [label, count] : votes) {
            if (count > highest) {
                prediction = label;
                highest = count;
            }
        }
        return prediction;
    }

    const vector<double>& getImportance() const {
        return importance;
    }

    double getOobAccuracy() const {
        return oobAccuracy;
    }
};

vector<EquipmentRecord> makeInspectionData(
    size_t count,
    unsigned int seed
) {
    mt19937 generator(seed);
    uniform_real_distribution<double> vibrationDistribution(0.0, 10.0);
    uniform_real_distribution<double> temperatureDistribution(35.0, 110.0);
    uniform_real_distribution<double> hoursDistribution(0.0, 12000.0);
    uniform_real_distribution<double> pressureDistribution(0.0, 8.0);
    normal_distribution<double> noise(0.0, 0.8);

    vector<EquipmentRecord> rows;
    rows.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        double vibration = vibrationDistribution(generator);
        double temperature = temperatureDistribution(generator);
        double hours = hoursDistribution(generator);
        double pressure = pressureDistribution(generator);

        double risk =
            0.28 * vibration
            + 0.045 * max(0.0, temperature - 70.0)
            + 0.00025 * hours
            + 0.65 * pressure
            + 0.12 * vibration * pressure
            + noise(generator);

        rows.push_back({
            {vibration, temperature, hours, pressure},
            risk > 8.0 ? "maintenance_required" : "normal"
        });
    }

    shuffle(rows.begin(), rows.end(), generator);
    return rows;
}

int main() {
    try {
        const auto dataset = makeInspectionData(240, 2026);
        const size_t split = static_cast<size_t>(dataset.size() * 0.8);

        vector<EquipmentRecord> training(
            dataset.begin(), dataset.begin() + split
        );
        vector<EquipmentRecord> testing(
            dataset.begin() + split, dataset.end()
        );

        InspectionForest forest(51, 6, 2, 71);
        forest.fit(training);

        size_t correct = 0;
        for (const auto& record : testing) {
            if (forest.predict(record.features) == record.outcome) ++correct;
        }

        cout << fixed << setprecision(3);
        cout << "Equipment inspection Random Forest\n";
        cout << "Trees: 51\n";
        cout << "Training rows: " << training.size() << '\n';
        cout << "Test rows: " << testing.size() << '\n';
        cout << "Test accuracy: "
             << static_cast<double>(correct) / testing.size() << '\n';

        if (forest.getOobAccuracy() >= 0.0) {
            cout << "Out-of-bag accuracy: " << forest.getOobAccuracy() << '\n';
        } else {
            cout << "Out-of-bag accuracy unavailable\n";
        }

        const vector<string> featureNames = {
            "Vibration", "Temperature", "Operating hours", "Pressure deviation"
        };

        vector<pair<double, string>> ranking;
        for (size_t i = 0; i < featureNames.size(); ++i) {
            ranking.emplace_back(
                forest.getImportance()[i],
                featureNames[i]
            );
        }

        sort(ranking.rbegin(), ranking.rend());
        cout << "\nImpurity-based feature importance:\n";
        for (const auto& [importance, name] : ranking) {
            cout << setw(22) << left << name << importance << '\n';
        }

        const vector<double> highRiskAsset = {8.2, 97.0, 9800.0, 6.5};
        cout << "\nHigh-risk asset prediction: "
             << forest.predict(highRiskAsset) << '\n';

        // A wrong dimensionality must fail rather than produce a misleading
        // maintenance decision.
        try {
            forest.predict({8.2, 97.0});
        } catch (const invalid_argument& error) {
            cout << "Validation caught: " << error.what() << '\n';
        }
    } catch (const exception& error) {
        cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
