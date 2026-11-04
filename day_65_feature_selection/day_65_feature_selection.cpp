#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

enum class SelectionMethod {
    FILTER,
    WRAPPER,
    EMBEDDED,
    RFE
};

struct FeatureScore {
    string name;
    double variance;
    double relevance;
    double redundancy;
};

struct SelectionResult {
    string method;
    vector<string> selected;
    map<string, double> scores;
};

class Dataset {
private:
    vector<string> names;
    vector<vector<double>> rows;
    vector<int> labels;

public:
    Dataset(
        vector<string> featureNames,
        vector<vector<double>> observations,
        vector<int> targets
    ) :
        names(move(featureNames)),
        rows(move(observations)),
        labels(move(targets))
    {
        if (names.empty() || rows.size() < 3) {
            throw invalid_argument("Insufficient dataset dimensions.");
        }

        if (rows.size() != labels.size()) {
            throw invalid_argument("Observation and label counts differ.");
        }

        set<string> uniqueNames(names.begin(), names.end());

        if (uniqueNames.size() != names.size()) {
            throw invalid_argument("Feature names must be unique.");
        }

        set<int> classes(labels.begin(), labels.end());

        if (classes != set<int>{0, 1}) {
            throw invalid_argument("Both binary classes 0 and 1 are required.");
        }

        for (const auto& row : rows) {
            if (row.size() != names.size()) {
                throw invalid_argument("Inconsistent row dimensions.");
            }

            for (double value : row) {
                if (!isfinite(value)) {
                    throw invalid_argument("Non-finite feature value.");
                }
            }
        }
    }

    size_t sampleCount() const {
        return rows.size();
    }

    size_t featureCount() const {
        return names.size();
    }

    const vector<string>& featureNames() const {
        return names;
    }

    const vector<int>& targets() const {
        return labels;
    }

    vector<double> column(size_t index) const {
        if (index >= names.size()) {
            throw out_of_range("Invalid feature index.");
        }

        vector<double> result;
        result.reserve(rows.size());

        for (const auto& row : rows) {
            result.push_back(row[index]);
        }

        return result;
    }

    vector<vector<double>> subset(
        const vector<size_t>& indices
    ) const {
        if (indices.empty()) {
            throw invalid_argument("Feature subset cannot be empty.");
        }

        vector<vector<double>> result;

        for (const auto& row : rows) {
            vector<double> selectedRow;

            for (size_t index : indices) {
                if (index >= names.size()) {
                    throw out_of_range("Invalid subset index.");
                }

                selectedRow.push_back(row[index]);
            }

            result.push_back(move(selectedRow));
        }

        return result;
    }
};

double average(const vector<double>& values) {
    if (values.empty()) {
        throw invalid_argument("Empty statistical input.");
    }

    return accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double calculateVariance(const vector<double>& values) {
    double center = average(values);

    double sum = 0.0;

    for (double value : values) {
        sum += pow(value - center, 2);
    }

    return sum / values.size();
}

double pearson(
    const vector<double>& x,
    const vector<double>& y
) {
    if (x.size() != y.size() || x.empty()) {
        throw invalid_argument("Invalid correlation inputs.");
    }

    double mx = average(x);
    double my = average(y);

    double numerator = 0.0;
    double sx = 0.0;
    double sy = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;

        numerator += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }

    double denominator = sqrt(sx * sy);

    return denominator < 1e-12
        ? 0.0
        : numerator / denominator;
}

double mutualInformation(
    const vector<double>& values,
    const vector<int>& labels,
    int bins = 5
) {
    if (values.size() != labels.size() || bins < 2) {
        throw invalid_argument("Invalid mutual information input.");
    }

    vector<double> sorted = values;
    sort(sorted.begin(), sorted.end());

    vector<double> edges;

    for (int i = 1; i < bins; ++i) {
        size_t position = min(
            sorted.size() - 1,
            static_cast<size_t>(i * sorted.size() / bins)
        );

        edges.push_back(sorted[position]);
    }

    map<pair<int, int>, int> joint;
    map<int, int> xCounts;
    map<int, int> yCounts;

    for (size_t i = 0; i < values.size(); ++i) {
        int category = static_cast<int>(
            lower_bound(edges.begin(), edges.end(), values[i])
            - edges.begin()
        );

        joint[{category, labels[i]}]++;
        xCounts[category]++;
        yCounts[labels[i]]++;
    }

    double information = 0.0;
    double n = static_cast<double>(values.size());

    for (const auto& [key, count] : joint) {
        double pxy = count / n;
        double px = xCounts[key.first] / n;
        double py = yCounts[key.second] / n;

        information += pxy * log2(pxy / (px * py));
    }

    return max(0.0, information);
}

class GovernanceEngine {
private:
    const Dataset& dataset;
    double varianceThreshold;
    double correlationThreshold;
    size_t maximumFeatures;

    double relevance(size_t index) const {
        return mutualInformation(
            dataset.column(index),
            dataset.targets()
        );
    }

    double evaluate(const vector<size_t>& indices) const {
        /*
         * Wrapper objective: nearest-centroid classification.
         * This implementation evaluates the complete dataset.
         * Production evaluation must use a separate validation
         * set or nested cross-validation.
         */
        auto rows = dataset.subset(indices);
        const auto& labels = dataset.targets();

        vector<double> center0(indices.size(), 0.0);
        vector<double> center1(indices.size(), 0.0);

        int count0 = 0;
        int count1 = 0;

        for (size_t i = 0; i < rows.size(); ++i) {
            if (labels[i] == 0) {
                count0++;
            } else {
                count1++;
            }

            auto& center = labels[i] == 0
                ? center0
                : center1;

            for (size_t j = 0; j < indices.size(); ++j) {
                center[j] += rows[i][j];
            }
        }

        for (size_t j = 0; j < indices.size(); ++j) {
            center0[j] /= count0;
            center1[j] /= count1;
        }

        int correct = 0;

        for (size_t i = 0; i < rows.size(); ++i) {
            double distance0 = 0.0;
            double distance1 = 0.0;

            for (size_t j = 0; j < indices.size(); ++j) {
                double d0 = rows[i][j] - center0[j];
                double d1 = rows[i][j] - center1[j];

                distance0 += d0 * d0;
                distance1 += d1 * d1;
            }

            int prediction = distance0 <= distance1 ? 0 : 1;

            if (prediction == labels[i]) {
                correct++;
            }
        }

        return static_cast<double>(correct) / rows.size();
    }

public:
    GovernanceEngine(
        const Dataset& data,
        double varianceLimit = 1e-8,
        double correlationLimit = 0.95,
        size_t maximum = 3
    ) :
        dataset(data),
        varianceThreshold(varianceLimit),
        correlationThreshold(correlationLimit),
        maximumFeatures(maximum)
    {
        if (varianceLimit < 0 ||
            correlationLimit < 0 ||
            correlationLimit > 1 ||
            maximum == 0 ||
            maximum > dataset.featureCount()) {
            throw invalid_argument("Invalid selection configuration.");
        }
    }

    SelectionResult filterSelection() const {
        vector<FeatureScore> ranking;

        for (size_t i = 0; i < dataset.featureCount(); ++i) {
            auto values = dataset.column(i);
            double var = calculateVariance(values);

            if (var <= varianceThreshold) {
                continue;
            }

            ranking.push_back({
                dataset.featureNames()[i],
                var,
                relevance(i),
                0.0
            });
        }

        sort(
            ranking.begin(),
            ranking.end(),
            [](const FeatureScore& a, const FeatureScore& b) {
                if (a.relevance != b.relevance) {
                    return a.relevance > b.relevance;
                }

                return a.name < b.name;
            }
        );

        vector<size_t> selected;
        map<string, double> scores;

        for (const auto& candidate : ranking) {
            size_t index = find(
                dataset.featureNames().begin(),
                dataset.featureNames().end(),
                candidate.name
            ) - dataset.featureNames().begin();

            bool redundant = false;

            for (size_t existing : selected) {
                if (abs(pearson(
                    dataset.column(index),
                    dataset.column(existing)
                )) >= correlationThreshold) {
                    redundant = true;
                    break;
                }
            }

            scores[candidate.name] = candidate.relevance;

            if (!redundant && selected.size() < maximumFeatures) {
                selected.push_back(index);
            }
        }

        if (selected.empty()) {
            throw runtime_error("No features passed filter selection.");
        }

        vector<string> names;

        for (size_t index : selected) {
            names.push_back(dataset.featureNames()[index]);
        }

        return {"FILTER", names, scores};
    }

    SelectionResult wrapperSelection() const {
        vector<size_t> selected;
        set<size_t> remaining;

        for (size_t i = 0; i < dataset.featureCount(); ++i) {
            remaining.insert(i);
        }

        map<string, double> scores;

        while (!remaining.empty() &&
               selected.size() < maximumFeatures) {
            size_t best = dataset.featureCount();
            double bestScore = -1.0;

            for (size_t candidate : remaining) {
                auto trial = selected;
                trial.push_back(candidate);

                double score = evaluate(trial);

                if (score > bestScore ||
                    (score == bestScore && candidate < best)) {
                    bestScore = score;
                    best = candidate;
                }
            }

            if (best >= dataset.featureCount()) {
                break;
            }

            selected.push_back(best);
            remaining.erase(best);

            scores[dataset.featureNames()[best]] = bestScore;
        }

        vector<string> names;

        for (size_t index : selected) {
            names.push_back(dataset.featureNames()[index]);
        }

        return {"WRAPPER", names, scores};
    }

    SelectionResult embeddedSelection() const {
        /*
         * Embedded-style scoring:
         * standardized class separation with coefficient shrinkage.
         */
        map<string, double> scores;

        for (size_t i = 0; i < dataset.featureCount(); ++i) {
            auto values = dataset.column(i);

            vector<double> positive;
            vector<double> negative;

            for (size_t j = 0; j < values.size(); ++j) {
                if (dataset.targets()[j] == 1) {
                    positive.push_back(values[j]);
                } else {
                    negative.push_back(values[j]);
                }
            }

            double scale = sqrt(
                (calculateVariance(positive) +
                 calculateVariance(negative)) / 2.0
            );

            double coefficient = scale < 1e-12
                ? 0.0
                : abs(average(positive) - average(negative)) / scale;

            // L1-like shrinkage.
            scores[dataset.featureNames()[i]] =
                max(0.0, coefficient - 0.2);
        }

        vector<pair<string, double>> ranking(
            scores.begin(), scores.end()
        );

        sort(
            ranking.begin(),
            ranking.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) {
                    return a.second > b.second;
                }

                return a.first < b.first;
            }
        );

        vector<string> selected;

        for (const auto& item : ranking) {
            if (item.second > 0 &&
                selected.size() < maximumFeatures) {
                selected.push_back(item.first);
            }
        }

        if (selected.empty()) {
            throw runtime_error("No features passed embedded selection.");
        }

        return {"EMBEDDED", selected, scores};
    }

    SelectionResult recursiveFeatureElimination() const {
        vector<size_t> active;

        for (size_t i = 0; i < dataset.featureCount(); ++i) {
            active.push_back(i);
        }

        while (active.size() > maximumFeatures) {
            vector<pair<size_t, double>> ranking;

            for (size_t index : active) {
                auto values = dataset.column(index);

                vector<double> positive;
                vector<double> negative;

                for (size_t j = 0; j < values.size(); ++j) {
                    if (dataset.targets()[j] == 1) {
                        positive.push_back(values[j]);
                    } else {
                        negative.push_back(values[j]);
                    }
                }

                double scale = sqrt(
                    (calculateVariance(positive) +
                     calculateVariance(negative)) / 2.0
                );

                double score = scale < 1e-12
                    ? 0.0
                    : abs(average(positive) - average(negative)) / scale;

                ranking.push_back({index, score});
            }

            sort(
                ranking.begin(),
                ranking.end(),
                [](const auto& a, const auto& b) {
                    if (a.second != b.second) {
                        return a.second < b.second;
                    }

                    return a.first < b.first;
                }
            );

            size_t weakest = ranking.front().first;

            active.erase(
                remove(active.begin(), active.end(), weakest),
                active.end()
            );
        }

        vector<string> names;
        map<string, double> scores;

        for (size_t index : active) {
            names.push_back(dataset.featureNames()[index]);
            scores[dataset.featureNames()[index]] = relevance(index);
        }

        return {"RFE", names, scores};
    }
};

void printResult(const SelectionResult& result) {
    cout << "\nMethod: " << result.method << "\n";
    cout << "Selected features:\n";

    for (const auto& feature : result.selected) {
        cout << "  " << feature << "\n";
    }

    cout << "Scores:\n";

    for (const auto& [name, score] : result.scores) {
        cout << "  " << setw(24) << left << name
             << fixed << setprecision(5) << score << "\n";
    }
}

Dataset createDataset(size_t samples = 200) {
    mt19937 generator(42);

    uniform_real_distribution<double> incomeDistribution(20000, 150000);
    uniform_real_distribution<double> ageDistribution(20, 70);
    uniform_real_distribution<double> noiseDistribution(-1, 1);
    uniform_real_distribution<double> probabilityDistribution(0, 1);

    vector<string> names = {
        "income",
        "age",
        "balance",
        "transactions",
        "income_duplicate",
        "constant_feature"
    };

    vector<vector<double>> rows;
    vector<int> labels;

    for (size_t i = 0; i < samples; ++i) {
        double income = incomeDistribution(generator);
        double age = ageDistribution(generator);
        double balance = income * 0.45 +
                         noiseDistribution(generator) * 10000;
        double transactions = probabilityDistribution(generator) * 100;
        double duplicate = income +
                           noiseDistribution(generator) * 100;

        double signal =
            -2.0 +
            0.000025 * (income - 85000) +
            0.025 * (age - 45) +
            0.000035 * (balance - 40000);

        double probability = 1.0 / (1.0 + exp(-signal));

        int label = probabilityDistribution(generator) < probability
            ? 1
            : 0;

        rows.push_back({
            income,
            age,
            balance,
            transactions,
            duplicate,
            10.0
        });

        labels.push_back(label);
    }

    return Dataset(names, rows, labels);
}

void runTests(const Dataset& dataset) {
    if (dataset.featureCount() != 6 ||
        dataset.sampleCount() != 200) {
        throw runtime_error("Dataset validation failed.");
    }

    if (calculateVariance(dataset.column(5)) != 0.0) {
        throw runtime_error("Constant feature validation failed.");
    }

    double score = mutualInformation(
        dataset.column(0),
        dataset.targets()
    );

    if (score < 0.0 || !isfinite(score)) {
        throw runtime_error("Mutual information validation failed.");
    }

    if (abs(pearson(
        dataset.column(0),
        dataset.column(4)
    )) < 0.95) {
        throw runtime_error("Redundancy validation failed.");
    }
}

int main() {
    try {
        Dataset dataset = createDataset();

        runTests(dataset);

        GovernanceEngine engine(
            dataset,
            1e-8,
            0.95,
            3
        );

        cout << "FEATURE SELECTION ENGINE\n";
        cout << "Observations: " << dataset.sampleCount() << "\n";
        cout << "Original features: " << dataset.featureCount() << "\n";

        printResult(engine.filterSelection());
        printResult(engine.wrapperSelection());
        printResult(engine.embeddedSelection());
        printResult(engine.recursiveFeatureElimination());

        cout << "\nAll validation tests passed.\n";

    } catch (const exception& error) {
        cerr << "Execution failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
