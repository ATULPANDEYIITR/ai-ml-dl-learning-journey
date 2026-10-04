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
#include <tuple>
#include <vector>

using namespace std;

struct ChangeRecord {
    int id;
    double linesChanged;
    double filesChanged;
    double reviewComments;
    double previousFailures;
    double developerExperience;
    int target;
    string developer;
    int day;
};

using Features = vector<double>;

class RepositoryRiskDataset {
public:
    static vector<ChangeRecord> generate(size_t count, unsigned seed) {
        mt19937 rng(seed);
        uniform_int_distribution<int> developerDist(0, 9);
        uniform_int_distribution<int> experienceDist(1, 10);
        poisson_distribution<int> filesDist(3);
        poisson_distribution<int> commentsDist(2);
        poisson_distribution<int> failuresDist(1);
        exponential_distribution<double> linesDist(1.0 / 70.0);
        normal_distribution<double> noise(0.0, 2.0);

        vector<ChangeRecord> data;
        data.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            int experience = experienceDist(rng);
            int files = max(1, filesDist(rng));
            int comments = commentsDist(rng);
            int failures = failuresDist(rng);
            int lines = max(5, static_cast<int>(linesDist(rng)));

            double risk =
                0.025 * lines +
                0.55 * files +
                0.75 * comments +
                0.9 * failures -
                0.65 * experience +
                noise(rng);

            data.push_back({
                static_cast<int>(i),
                static_cast<double>(lines),
                static_cast<double>(files),
                static_cast<double>(comments),
                static_cast<double>(failures),
                static_cast<double>(experience),
                risk > 3.0 ? 1 : 0,
                "developer-" + to_string(developerDist(rng)),
                static_cast<int>(i)
            });
        }

        return data;
    }
};

class DatasetSplitter {
public:
    struct Split {
        vector<ChangeRecord> train;
        vector<ChangeRecord> validation;
        vector<ChangeRecord> test;
    };

    static Split randomSplit(const vector<ChangeRecord>& source, unsigned seed) {
        vector<ChangeRecord> shuffled = source;
        mt19937 rng(seed);
        shuffle(shuffled.begin(), shuffled.end(), rng);

        size_t trainEnd = shuffled.size() * 70 / 100;
        size_t validationEnd = shuffled.size() * 85 / 100;

        return {
            vector<ChangeRecord>(shuffled.begin(), shuffled.begin() + trainEnd),
            vector<ChangeRecord>(shuffled.begin() + trainEnd, shuffled.begin() + validationEnd),
            vector<ChangeRecord>(shuffled.begin() + validationEnd, shuffled.end())
        };
    }

    static Split chronologicalSplit(vector<ChangeRecord> source) {
        sort(source.begin(), source.end(),
             [](const auto& a, const auto& b) { return a.day < b.day; });

        size_t trainEnd = source.size() * 70 / 100;
        size_t validationEnd = source.size() * 85 / 100;

        return {
            vector<ChangeRecord>(source.begin(), source.begin() + trainEnd),
            vector<ChangeRecord>(source.begin() + trainEnd, source.begin() + validationEnd),
            vector<ChangeRecord>(source.begin() + validationEnd, source.end())
        };
    }

    static Split groupSplit(
        const vector<ChangeRecord>& source,
        unsigned seed
    ) {
        map<string, vector<ChangeRecord>> groups;

        for (const auto& record : source) {
            groups[record.developer].push_back(record);
        }

        vector<string> developers;
        for (const auto& [developer, _] : groups) {
            developers.push_back(developer);
        }

        mt19937 rng(seed);
        shuffle(developers.begin(), developers.end(), rng);

        Split result;
        const size_t trainTarget = source.size() * 70 / 100;
        const size_t validationTarget = source.size() * 15 / 100;

        for (const auto& developer : developers) {
            const auto& group = groups.at(developer);

            if (result.train.size() + group.size() <= trainTarget) {
                result.train.insert(result.train.end(), group.begin(), group.end());
            } else if (result.validation.size() + group.size() <= validationTarget) {
                result.validation.insert(
                    result.validation.end(), group.begin(), group.end()
                );
            } else {
                result.test.insert(result.test.end(), group.begin(), group.end());
            }
        }

        return result;
    }
};

Features featureVector(const ChangeRecord& record) {
    return {
        record.linesChanged,
        record.filesChanged,
        record.reviewComments,
        record.previousFailures,
        record.developerExperience
    };
}

class StandardScaler {
private:
    Features means;
    Features standardDeviations;
    bool fitted = false;

public:
    void fit(const vector<Features>& data) {
        if (data.empty()) {
            throw invalid_argument("Scaler cannot fit an empty dataset.");
        }

        const size_t width = data.front().size();
        means.assign(width, 0.0);
        standardDeviations.assign(width, 0.0);

        for (const auto& row : data) {
            if (row.size() != width) {
                throw invalid_argument("Feature rows have inconsistent widths.");
            }

            for (size_t i = 0; i < width; ++i) {
                means[i] += row[i];
            }
        }

        for (double& value : means) {
            value /= static_cast<double>(data.size());
        }

        for (const auto& row : data) {
            for (size_t i = 0; i < width; ++i) {
                const double difference = row[i] - means[i];
                standardDeviations[i] += difference * difference;
            }
        }

        for (double& value : standardDeviations) {
            value = sqrt(value / static_cast<double>(data.size()));
            if (value == 0.0) value = 1.0;
        }

        fitted = true;
    }

    vector<Features> transform(const vector<Features>& data) const {
        if (!fitted) {
            throw logic_error("Scaler must be fitted before transformation.");
        }

        vector<Features> transformed;
        transformed.reserve(data.size());

        for (const auto& row : data) {
            if (row.size() != means.size()) {
                throw invalid_argument("Feature width does not match scaler.");
            }

            Features normalized;
            normalized.reserve(row.size());

            for (size_t i = 0; i < row.size(); ++i) {
                normalized.push_back(
                    (row[i] - means[i]) / standardDeviations[i]
                );
            }

            transformed.push_back(move(normalized));
        }

        return transformed;
    }

    const Features& getMeans() const {
        return means;
    }
};

class NearestCentroidModel {
private:
    map<int, Features> centroids;

    static double squaredDistance(
        const Features& left,
        const Features& right
    ) {
        double result = 0.0;

        for (size_t i = 0; i < left.size(); ++i) {
            const double difference = left[i] - right[i];
            result += difference * difference;
        }

        return result;
    }

public:
    void fit(
        const vector<Features>& features,
        const vector<int>& targets
    ) {
        if (features.empty() || features.size() != targets.size()) {
            throw invalid_argument("Training data is empty or misaligned.");
        }

        map<int, vector<Features>> byClass;

        for (size_t i = 0; i < features.size(); ++i) {
            byClass[targets[i]].push_back(features[i]);
        }

        centroids.clear();

        for (const auto& [label, rows] : byClass) {
            Features centroid(rows.front().size(), 0.0);

            for (const auto& row : rows) {
                for (size_t i = 0; i < row.size(); ++i) {
                    centroid[i] += row[i];
                }
            }

            for (double& value : centroid) {
                value /= static_cast<double>(rows.size());
            }

            centroids[label] = centroid;
        }
    }

    vector<int> predict(const vector<Features>& features) const {
        if (centroids.empty()) {
            throw logic_error("Model has not been trained.");
        }

        vector<int> predictions;

        for (const auto& row : features) {
            int bestLabel = -1;
            double bestDistance = numeric_limits<double>::infinity();

            for (const auto& [label, centroid] : centroids) {
                const double distance = squaredDistance(row, centroid);

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }

            predictions.push_back(bestLabel);
        }

        return predictions;
    }
};

struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1;
};

Metrics evaluate(
    const vector<int>& actual,
    const vector<int>& predicted
) {
    if (actual.empty() || actual.size() != predicted.size()) {
        throw invalid_argument("Evaluation vectors are invalid.");
    }

    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == 1 && predicted[i] == 1) ++tp;
        else if (actual[i] == 0 && predicted[i] == 0) ++tn;
        else if (actual[i] == 0 && predicted[i] == 1) ++fp;
        else if (actual[i] == 1 && predicted[i] == 0) ++fn;
    }

    double precision = tp + fp == 0
        ? 0.0
        : static_cast<double>(tp) / (tp + fp);

    double recall = tp + fn == 0
        ? 0.0
        : static_cast<double>(tp) / (tp + fn);

    double f1 = precision + recall == 0
        ? 0.0
        : 2.0 * precision * recall / (precision + recall);

    return {
        static_cast<double>(tp + tn) / actual.size(),
        precision,
        recall,
        f1
    };
}

vector<Features> matrixFor(const vector<ChangeRecord>& rows) {
    vector<Features> matrix;
    matrix.reserve(rows.size());

    for (const auto& row : rows) {
        matrix.push_back(featureVector(row));
    }

    return matrix;
}

vector<int> targetsFor(const vector<ChangeRecord>& rows) {
    vector<int> targets;
    targets.reserve(rows.size());

    for (const auto& row : rows) {
        targets.push_back(row.target);
    }

    return targets;
}

double positiveRate(const vector<ChangeRecord>& rows) {
    if (rows.empty()) return 0.0;

    const int positives = accumulate(
        rows.begin(),
        rows.end(),
        0,
        [](int total, const ChangeRecord& row) {
            return total + row.target;
        }
    );

    return static_cast<double>(positives) / rows.size();
}

set<string> developersOf(const vector<ChangeRecord>& rows) {
    set<string> result;

    for (const auto& row : rows) {
        result.insert(row.developer);
    }

    return result;
}

void demonstrateLeakage(const DatasetSplitter::Split& split) {
    StandardScaler correctScaler;
    correctScaler.fit(matrixFor(split.train));

    StandardScaler contaminatedScaler;

    vector<Features> allData = matrixFor(split.train);
    const auto validation = matrixFor(split.validation);
    const auto test = matrixFor(split.test);

    allData.insert(allData.end(), validation.begin(), validation.end());
    allData.insert(allData.end(), test.begin(), test.end());

    // This intentionally incorrect scaler learns statistics from held-out
    // observations. The resulting evaluation is no longer isolated.
    contaminatedScaler.fit(allData);

    cout << "\nLeakage experiment\n";
    cout << "Training-only first mean: "
         << correctScaler.getMeans().front() << '\n';
    cout << "All-data first mean: "
         << contaminatedScaler.getMeans().front() << '\n';
    cout << "Different means demonstrate that held-out distributions changed "
            "the preprocessing state.\n";
}

void runExperiment(const DatasetSplitter::Split& split) {
    StandardScaler scaler;

    // Only train data is allowed to determine normalization parameters.
    const auto trainFeatures = scaler.transform(
        [&]() {
            scaler.fit(matrixFor(split.train));
            return matrixFor(split.train);
        }()
    );

    const auto validationFeatures = scaler.transform(matrixFor(split.validation));

    NearestCentroidModel model;
    model.fit(trainFeatures, targetsFor(split.train));

    const auto validationPredictions = model.predict(validationFeatures);
    const auto validationMetrics = evaluate(
        targetsFor(split.validation),
        validationPredictions
    );

    cout << "\nValidation F1: "
         << fixed << setprecision(4)
         << validationMetrics.f1 << '\n';

    // The test set is intentionally absent from model selection. It enters
    // only after validation has been used to assess the current configuration.
    const auto testFeatures = scaler.transform(matrixFor(split.test));
    const auto testPredictions = model.predict(testFeatures);

    const auto testMetrics = evaluate(
        targetsFor(split.test),
        testPredictions
    );

    cout << "Final test accuracy: " << testMetrics.accuracy << '\n';
    cout << "Final test precision: " << testMetrics.precision << '\n';
    cout << "Final test recall: " << testMetrics.recall << '\n';
    cout << "Final test F1: " << testMetrics.f1 << '\n';
}

int main() {
    try {
        cout << "REPOSITORY CHANGE RISK: TRAIN / VALIDATION / TEST CASE STUDY\n";
        cout << "=============================================================\n";

        const auto dataset =
            RepositoryRiskDataset::generate(240, 2026);

        cout << "Records: " << dataset.size() << '\n';
        cout << "Positive rate: "
             << fixed << setprecision(3)
             << positiveRate(dataset) << '\n';

        const auto randomSplit =
            DatasetSplitter::randomSplit(dataset, 42);

        cout << "\nRandom split sizes: "
             << randomSplit.train.size() << ", "
             << randomSplit.validation.size() << ", "
             << randomSplit.test.size() << '\n';

        cout << "Random train positive rate: "
             << positiveRate(randomSplit.train) << '\n';
        cout << "Random test positive rate: "
             << positiveRate(randomSplit.test) << '\n';

        const auto timeSplit =
            DatasetSplitter::chronologicalSplit(dataset);

        cout << "\nChronological split\n";
        cout << "Training latest day: "
             << timeSplit.train.back().day << '\n';
        cout << "Validation first day: "
             << timeSplit.validation.front().day << '\n';
        cout << "Test first day: "
             << timeSplit.test.front().day << '\n';

        const auto grouped =
            DatasetSplitter::groupSplit(dataset, 42);

        const auto trainDevelopers = developersOf(grouped.train);
        const auto validationDevelopers = developersOf(grouped.validation);
        const auto testDevelopers = developersOf(grouped.test);

        set<string> trainValidationOverlap;
        set_intersection(
            trainDevelopers.begin(), trainDevelopers.end(),
            validationDevelopers.begin(), validationDevelopers.end(),
            inserter(trainValidationOverlap, trainValidationOverlap.begin())
        );

        set<string> trainTestOverlap;
        set_intersection(
            trainDevelopers.begin(), trainDevelopers.end(),
            testDevelopers.begin(), testDevelopers.end(),
            inserter(trainTestOverlap, trainTestOverlap.begin())
        );

        cout << "\nGroup-aware split\n";
        cout << "Train developers: " << trainDevelopers.size() << '\n';
        cout << "Validation developers: " << validationDevelopers.size() << '\n';
        cout << "Test developers: " << testDevelopers.size() << '\n';
        cout << "Train/validation overlap: "
             << trainValidationOverlap.size() << '\n';
        cout << "Train/test overlap: "
             << trainTestOverlap.size() << '\n';

        cout << "\nModel experiment using stratified-style random partition\n";
        runExperiment(randomSplit);

        demonstrateLeakage(randomSplit);

        const auto repeatA =
            DatasetSplitter::randomSplit(dataset, 777);
        const auto repeatB =
            DatasetSplitter::randomSplit(dataset, 777);

        bool identical = true;

        for (size_t i = 0; i < repeatA.train.size(); ++i) {
            if (repeatA.train[i].id != repeatB.train[i].id) {
                identical = false;
                break;
            }
        }

        cout << "\nReproducibility check\n";
        cout << "Same seed gives identical training order: "
             << boolalpha << identical << '\n';

        cout << "\nDesign decisions\n";
        cout << "Random splitting is suitable only when observations are "
                "approximately independent and identically distributed.\n";
        cout << "Chronological splitting is safer when deployment predicts "
                "future observations.\n";
        cout << "Group-aware splitting prevents repeated entities from "
                "appearing across evaluation boundaries.\n";
        cout << "Preprocessing must be fitted on training data only.\n";
        cout << "The final test set should remain outside model-selection decisions.\n";

    } catch (const exception& error) {
        cerr << "Experiment failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
