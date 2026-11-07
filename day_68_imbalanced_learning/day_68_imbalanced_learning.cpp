#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

/*
 * Imbalanced Learning: C++17
 *
 * Domain: Fraud detection in financial transactions.
 *
 * Implements:
 * - Class distribution analysis
 * - Stratified splitting
 * - Random oversampling
 * - Random undersampling
 * - SMOTE
 * - Class-weighted centroid classification
 * - Confusion matrix and evaluation metrics
 */

enum class SamplingMethod {
    NONE,
    OVERSAMPLE,
    UNDERSAMPLE,
    SMOTE
};

struct Record {
    vector<double> features;
    int label;

    Record(vector<double> values, int target)
        : features(move(values)), label(target) {
        if (features.empty()) {
            throw invalid_argument("Empty feature vector.");
        }

        if (label != 0 && label != 1) {
            throw invalid_argument("Invalid binary label.");
        }

        for (double value : features) {
            if (!isfinite(value)) {
                throw invalid_argument("Non-finite feature.");
            }
        }
    }
};

using Dataset = vector<Record>;

struct Distribution {
    size_t majority;
    size_t minority;
    double imbalanceRatio;
    size_t negatives;
    size_t positives;
};

struct Metrics {
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;

    double precision() const {
        int denominator = tp + fp;
        return denominator ? double(tp) / denominator : 0.0;
    }

    double recall() const {
        int denominator = tp + fn;
        return denominator ? double(tp) / denominator : 0.0;
    }

    double specificity() const {
        int denominator = tn + fp;
        return denominator ? double(tn) / denominator : 0.0;
    }

    double f1() const {
        double p = precision();
        double r = recall();

        return p + r ? 2.0 * p * r / (p + r) : 0.0;
    }

    double balancedAccuracy() const {
        return (recall() + specificity()) / 2.0;
    }
};

void validateDataset(const Dataset& data) {
    if (data.empty()) {
        throw invalid_argument("Dataset is empty.");
    }

    size_t dimensions = data.front().features.size();
    int counts[2] = {0, 0};

    for (const auto& record : data) {
        if (record.features.size() != dimensions) {
            throw invalid_argument("Feature dimension mismatch.");
        }

        if (record.label != 0 && record.label != 1) {
            throw invalid_argument("Invalid class label.");
        }

        counts[record.label]++;
    }

    if (counts[0] == 0 || counts[1] == 0) {
        throw invalid_argument("Both classes must exist.");
    }
}

Distribution analyzeDistribution(const Dataset& data) {
    validateDataset(data);

    size_t negatives = 0;
    size_t positives = 0;

    for (const auto& record : data) {
        if (record.label == 0) negatives++;
        else positives++;
    }

    return {
        max(negatives, positives),
        min(negatives, positives),
        double(max(negatives, positives))
            / min(negatives, positives),
        negatives,
        positives
    };
}

double distance(const vector<double>& a,
                const vector<double>& b) {
    if (a.size() != b.size()) {
        throw invalid_argument("Distance dimension mismatch.");
    }

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }

    return sqrt(sum);
}

Dataset generateTransactions(
    size_t majorityCount = 240,
    size_t minorityCount = 30
) {
    mt19937 generator(42);

    normal_distribution<double> normalMajority(0.0, 1.2);
    normal_distribution<double> normalMinority(2.0, 1.0);

    Dataset data;
    data.reserve(majorityCount + minorityCount);

    for (size_t i = 0; i < majorityCount; i++) {
        data.emplace_back(
            vector<double>{
                normalMajority(generator),
                normalMajority(generator)
            },
            0
        );
    }

    for (size_t i = 0; i < minorityCount; i++) {
        data.emplace_back(
            vector<double>{
                normalMinority(generator),
                normalMinority(generator)
            },
            1
        );
    }

    shuffle(data.begin(), data.end(), generator);
    return data;
}

pair<Dataset, Dataset> stratifiedSplit(
    const Dataset& data,
    double testFraction = 0.25
) {
    if (!(testFraction > 0.0 && testFraction < 1.0)) {
        throw invalid_argument("Invalid test fraction.");
    }

    vector<Record> groups[2];

    for (const auto& record : data) {
        groups[record.label].push_back(record);
    }

    mt19937 generator(42);
    Dataset training, testing;

    for (auto& group : groups) {
        if (group.size() < 2) {
            throw invalid_argument(
                "Each class needs at least two records."
            );
        }

        shuffle(group.begin(), group.end(), generator);

        size_t testSize = max(
            size_t(1),
            min(
                group.size() - 1,
                size_t(group.size() * testFraction)
            )
        );

        for (size_t i = 0; i < group.size(); i++) {
            if (i < testSize) testing.push_back(group[i]);
            else training.push_back(group[i]);
        }
    }

    shuffle(training.begin(), training.end(), generator);
    shuffle(testing.begin(), testing.end(), generator);

    return {training, testing};
}

Dataset randomOversample(const Dataset& data) {
    mt19937 generator(42);

    vector<Record> groups[2];

    for (const auto& record : data) {
        groups[record.label].push_back(record);
    }

    size_t target = max(groups[0].size(), groups[1].size());

    Dataset result = data;

    for (const auto& group : groups) {
        while (
            count_if(
                result.begin(), result.end(),
                [&](const Record& r) {
                    return r.label == (&group == &groups[0] ? 0 : 1);
                }
            ) < static_cast<long>(target)
        ) {
            uniform_int_distribution<size_t> distribution(
                0, group.size() - 1
            );

            result.push_back(group[distribution(generator)]);
        }
    }

    shuffle(result.begin(), result.end(), generator);
    return result;
}

Dataset randomUndersample(const Dataset& data) {
    mt19937 generator(42);
    vector<Record> groups[2];

    for (const auto& record : data) {
        groups[record.label].push_back(record);
    }

    size_t target = min(groups[0].size(), groups[1].size());

    Dataset result;

    for (auto& group : groups) {
        shuffle(group.begin(), group.end(), generator);

        result.insert(
            result.end(),
            group.begin(),
            group.begin() + target
        );
    }

    shuffle(result.begin(), result.end(), generator);
    return result;
}

Dataset applySMOTE(
    const Dataset& data,
    size_t neighbors = 5
) {
    mt19937 generator(42);

    vector<Record> minority;
    size_t majorityCount = 0;

    for (const auto& record : data) {
        if (record.label == 1) minority.push_back(record);
        else majorityCount++;
    }

    if (minority.size() < 2) {
        throw invalid_argument(
            "SMOTE requires at least two minority records."
        );
    }

    if (neighbors == 0) {
        throw invalid_argument("Neighbor count must be positive.");
    }

    size_t k = min(neighbors, minority.size() - 1);

    Dataset result = data;

    uniform_int_distribution<size_t> sourceDistribution(
        0, minority.size() - 1
    );

    uniform_real_distribution<double> interpolation(0.0, 1.0);

    size_t required = majorityCount - minority.size();

    for (size_t iteration = 0; iteration < required; iteration++) {
        size_t sourceIndex = sourceDistribution(generator);

        const Record& source = minority[sourceIndex];

        vector<pair<double, size_t>> distances;

        for (size_t j = 0; j < minority.size(); j++) {
            if (j == sourceIndex) continue;

            distances.push_back({
                distance(source.features, minority[j].features),
                j
            });
        }

        sort(distances.begin(), distances.end());

        uniform_int_distribution<size_t> neighborDistribution(
            0, k - 1
        );

        size_t neighborIndex =
            distances[neighborDistribution(generator)].second;

        const Record& neighbor = minority[neighborIndex];

        double factor = interpolation(generator);

        vector<double> synthetic(source.features.size());

        for (size_t j = 0; j < synthetic.size(); j++) {
            synthetic[j] = source.features[j]
                + factor * (
                    neighbor.features[j] - source.features[j]
                );
        }

        result.emplace_back(move(synthetic), 1);
    }

    shuffle(result.begin(), result.end(), generator);
    return result;
}

class WeightedCentroidClassifier {
private:
    vector<double> centroids[2];
    double classWeights[2] = {1.0, 1.0};
    bool trained = false;

public:
    void fit(const Dataset& data, bool weighted = false) {
        validateDataset(data);

        size_t dimensions = data.front().features.size();

        centroids[0].assign(dimensions, 0.0);
        centroids[1].assign(dimensions, 0.0);

        size_t counts[2] = {0, 0};

        for (const auto& record : data) {
            counts[record.label]++;

            for (size_t j = 0; j < dimensions; j++) {
                centroids[record.label][j] +=
                    record.features[j];
            }
        }

        for (int label = 0; label < 2; label++) {
            if (counts[label] == 0) {
                throw runtime_error("Missing training class.");
            }

            for (double& value : centroids[label]) {
                value /= counts[label];
            }
        }

        if (weighted) {
            size_t total = counts[0] + counts[1];

            classWeights[0] =
                double(total) / (2.0 * counts[0]);

            classWeights[1] =
                double(total) / (2.0 * counts[1]);
        } else {
            classWeights[0] = classWeights[1] = 1.0;
        }

        trained = true;
    }

    int predict(const vector<double>& features) const {
        if (!trained) {
            throw runtime_error("Classifier is not trained.");
        }

        if (features.size() != centroids[0].size()) {
            throw invalid_argument("Prediction dimension mismatch.");
        }

        double scores[2];

        for (int label = 0; label < 2; label++) {
            scores[label] =
                distance(features, centroids[label])
                / classWeights[label];
        }

        return scores[1] < scores[0] ? 1 : 0;
    }
};

Metrics evaluate(
    const Dataset& testing,
    const WeightedCentroidClassifier& classifier
) {
    Metrics metrics;

    for (const auto& record : testing) {
        int prediction = classifier.predict(record.features);

        if (record.label == 1 && prediction == 1) metrics.tp++;
        else if (record.label == 0 && prediction == 0) metrics.tn++;
        else if (record.label == 0 && prediction == 1) metrics.fp++;
        else metrics.fn++;
    }

    return metrics;
}

void printDistribution(const string& name, const Dataset& data) {
    auto distribution = analyzeDistribution(data);

    cout << name
         << " | Negative: " << distribution.negatives
         << " | Positive: " << distribution.positives
         << " | Imbalance ratio: "
         << fixed << setprecision(2)
         << distribution.imbalanceRatio << ":1\n";
}

void runExperiment(
    const string& name,
    const Dataset& training,
    const Dataset& testing,
    bool weighted = false
) {
    WeightedCentroidClassifier classifier;
    classifier.fit(training, weighted);

    Metrics metrics = evaluate(testing, classifier);

    cout << left << setw(24) << name
         << " Precision=" << fixed << setprecision(3)
         << metrics.precision()
         << " Recall=" << metrics.recall()
         << " F1=" << metrics.f1()
         << " BalancedAccuracy="
         << metrics.balancedAccuracy()
         << "\n";

    cout << "  TP=" << metrics.tp
         << " TN=" << metrics.tn
         << " FP=" << metrics.fp
         << " FN=" << metrics.fn << "\n";
}

int main() {
    try {
        cout << "FRAUD DETECTION: IMBALANCED LEARNING\n";
        cout << string(75, '=') << "\n";

        Dataset original = generateTransactions();

        auto [training, testing] = stratifiedSplit(original);

        printDistribution("Original", original);
        printDistribution("Training", training);
        printDistribution("Testing", testing);

        cout << "\nMODEL COMPARISON\n";
        cout << string(75, '-') << "\n";

        runExperiment(
            "Baseline",
            training,
            testing
        );

        Dataset oversampled = randomOversample(training);

        runExperiment(
            "Random Oversampling",
            oversampled,
            testing
        );

        Dataset undersampled = randomUndersample(training);

        runExperiment(
            "Random Undersampling",
            undersampled,
            testing
        );

        Dataset synthetic = applySMOTE(training);

        runExperiment(
            "SMOTE",
            synthetic,
            testing
        );

        runExperiment(
            "Class Weighting",
            training,
            testing,
            true
        );

        cout << "\nVALIDATION\n";

        if (oversampled.size() < training.size()) {
            throw runtime_error("Oversampling validation failed.");
        }

        if (undersampled.size() > training.size()) {
            throw runtime_error("Undersampling validation failed.");
        }

        if (synthetic.size() < training.size()) {
            throw runtime_error("SMOTE validation failed.");
        }

        cout << "All validation checks passed.\n";

        cout << "\nENGINEERING CONSIDERATIONS\n";
        cout << "- Normalize features before distance calculations.\n";
        cout << "- Apply sampling only to training partitions.\n";
        cout << "- Preserve original test class proportions.\n";
        cout << "- Check synthetic records against domain constraints.\n";
        cout << "- Use group-aware splits for repeated customers.\n";
        cout << "- Monitor false negatives and false positives.\n";
        cout << "- Keep random seeds and experiment metadata.\n";
        cout << "- Avoid unnecessary copies in large-scale pipelines.\n";

    } catch (const exception& error) {
        cerr << "Execution failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
