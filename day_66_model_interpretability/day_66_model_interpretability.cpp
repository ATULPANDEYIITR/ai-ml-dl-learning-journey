#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

/*
 * Model Interpretability Engine
 *
 * Domain: Business credit-risk prediction.
 *
 * Implements:
 * 1. Standardized coefficient importance.
 * 2. Repeated permutation importance.
 * 3. Partial dependence.
 * 4. Model evaluation.
 * 5. Input validation and interpretation policies.
 *
 * Compile:
 * g++ -std=c++17 -O2 -Wall -Wextra -pedantic model_interpretability.cpp -o interpretability
 */

using namespace std;

enum class ImportanceMethod {
    Coefficient,
    Permutation
};

struct Feature {
    string name;
    string description;
    string unit;
};

struct Evaluation {
    double mse = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    size_t sampleCount = 0;
};

struct Importance {
    string feature;
    double score = 0.0;
    double normalized = 0.0;
    double standardDeviation = 0.0;
};

struct DependencePoint {
    double featureValue;
    double averagePrediction;
};

class ValidationError : public runtime_error {
public:
    explicit ValidationError(const string& message)
        : runtime_error(message) {}
};

class InterpretationPolicy {
public:
    size_t minimumSamples = 20;
    size_t maximumFeatures = 100;
    size_t minimumRepeats = 5;
    size_t maximumRepeats = 100;

    void validate(
        size_t samples,
        size_t features,
        size_t repeats
    ) const {
        if (samples < minimumSamples) {
            throw ValidationError("Insufficient dataset size.");
        }

        if (features == 0 || features > maximumFeatures) {
            throw ValidationError("Feature count violates policy.");
        }

        if (repeats < minimumRepeats ||
            repeats > maximumRepeats) {
            throw ValidationError("Repeat count violates policy.");
        }
    }
};

class Dataset {
private:
    vector<vector<double>> X;
    vector<double> y;

public:
    Dataset(
        vector<vector<double>> features,
        vector<double> targets
    ) : X(move(features)), y(move(targets)) {
        validate();
    }

    void validate() const {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw ValidationError("Invalid dataset dimensions.");
        }

        const size_t featureCount = X.front().size();

        if (featureCount == 0) {
            throw ValidationError("Dataset has no features.");
        }

        for (const auto& row : X) {
            if (row.size() != featureCount) {
                throw ValidationError("Inconsistent feature dimensions.");
            }

            for (double value : row) {
                if (!isfinite(value)) {
                    throw ValidationError("Non-finite feature value.");
                }
            }
        }

        for (double value : y) {
            if (!isfinite(value)) {
                throw ValidationError("Non-finite target value.");
            }
        }
    }

    const vector<vector<double>>& features() const {
        return X;
    }

    const vector<double>& targets() const {
        return y;
    }

    size_t size() const {
        return X.size();
    }

    size_t featureCount() const {
        return X.front().size();
    }
};

class LinearModel {
private:
    vector<string> featureNames;
    vector<double> means;
    vector<double> scales;
    vector<double> coefficients;
    double intercept = 0.0;
    bool fitted = false;

    static vector<double> solve(
        vector<vector<double>> matrix,
        vector<double> rhs
    ) {
        const size_t n = rhs.size();

        for (size_t col = 0; col < n; ++col) {
            size_t pivot = col;

            for (size_t row = col + 1; row < n; ++row) {
                if (abs(matrix[row][col]) >
                    abs(matrix[pivot][col])) {
                    pivot = row;
                }
            }

            if (abs(matrix[pivot][col]) < 1e-12) {
                throw ValidationError(
                    "Singular regression matrix. "
                    "Check feature collinearity."
                );
            }

            swap(matrix[col], matrix[pivot]);
            swap(rhs[col], rhs[pivot]);

            double divisor = matrix[col][col];

            for (size_t j = col; j < n; ++j) {
                matrix[col][j] /= divisor;
            }

            rhs[col] /= divisor;

            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;

                double factor = matrix[row][col];

                for (size_t j = col; j < n; ++j) {
                    matrix[row][j] -= factor * matrix[col][j];
                }

                rhs[row] -= factor * rhs[col];
            }
        }

        return rhs;
    }

public:
    explicit LinearModel(vector<string> names)
        : featureNames(move(names)) {
        if (featureNames.empty()) {
            throw ValidationError("Feature names cannot be empty.");
        }

        unordered_set<string> uniqueNames;

        for (const auto& name : featureNames) {
            if (name.empty() || !uniqueNames.insert(name).second) {
                throw ValidationError("Invalid or duplicate feature name.");
            }
        }
    }

    void fit(const Dataset& dataset) {
        if (dataset.featureCount() != featureNames.size()) {
            throw ValidationError("Model and dataset dimensions differ.");
        }

        const auto& X = dataset.features();
        const auto& y = dataset.targets();

        const size_t n = dataset.size();
        const size_t p = featureNames.size();

        means.assign(p, 0.0);
        scales.assign(p, 1.0);

        // Standardize features to make coefficient magnitudes comparable.
        for (size_t j = 0; j < p; ++j) {
            for (const auto& row : X) {
                means[j] += row[j];
            }

            means[j] /= n;

            double variance = 0.0;

            for (const auto& row : X) {
                double difference = row[j] - means[j];
                variance += difference * difference;
            }

            scales[j] = sqrt(variance / n);

            if (scales[j] < 1e-12) {
                scales[j] = 1.0;
            }
        }

        const size_t dimensions = p + 1;

        vector<vector<double>> normal(
            dimensions,
            vector<double>(dimensions, 0.0)
        );

        vector<double> rhs(dimensions, 0.0);

        // Construct normal equations without storing a second design matrix.
        for (size_t row = 0; row < n; ++row) {
            vector<double> design(dimensions, 1.0);

            for (size_t j = 0; j < p; ++j) {
                design[j + 1] =
                    (X[row][j] - means[j]) / scales[j];
            }

            for (size_t i = 0; i < dimensions; ++i) {
                rhs[i] += design[i] * y[row];

                for (size_t j = 0; j < dimensions; ++j) {
                    normal[i][j] += design[i] * design[j];
                }
            }
        }

        vector<double> solution = solve(normal, rhs);

        intercept = solution[0];
        coefficients.assign(
            solution.begin() + 1,
            solution.end()
        );

        fitted = true;
    }

    double predictOne(const vector<double>& row) const {
        if (!fitted) {
            throw ValidationError("Model is not fitted.");
        }

        if (row.size() != featureNames.size()) {
            throw ValidationError("Invalid prediction dimensions.");
        }

        double result = intercept;

        for (size_t j = 0; j < row.size(); ++j) {
            if (!isfinite(row[j])) {
                throw ValidationError("Invalid prediction feature.");
            }

            result += coefficients[j] *
                ((row[j] - means[j]) / scales[j]);
        }

        return result;
    }

    vector<double> predict(
        const vector<vector<double>>& X
    ) const {
        if (X.empty()) {
            throw ValidationError("Prediction dataset is empty.");
        }

        vector<double> result;
        result.reserve(X.size());

        for (const auto& row : X) {
            result.push_back(predictOne(row));
        }

        return result;
    }

    const vector<string>& names() const {
        return featureNames;
    }

    const vector<double>& weights() const {
        return coefficients;
    }
};

Evaluation evaluate(
    const vector<double>& actual,
    const vector<double>& predicted
) {
    if (actual.empty() || actual.size() != predicted.size()) {
        throw ValidationError("Invalid evaluation arrays.");
    }

    Evaluation result;
    result.sampleCount = actual.size();

    for (size_t i = 0; i < actual.size(); ++i) {
        double error = actual[i] - predicted[i];

        result.mse += error * error;
        result.mae += abs(error);
    }

    result.mse /= actual.size();
    result.mae /= actual.size();
    result.rmse = sqrt(result.mse);

    return result;
}

double average(const vector<double>& values) {
    if (values.empty()) {
        throw ValidationError("Cannot average empty values.");
    }

    return accumulate(values.begin(), values.end(), 0.0) /
        values.size();
}

vector<Importance> coefficientImportance(
    const LinearModel& model
) {
    const auto& weights = model.weights();
    const auto& names = model.names();

    vector<Importance> result;
    double total = 0.0;

    for (double weight : weights) {
        total += abs(weight);
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        result.push_back({
            names[i],
            abs(weights[i]),
            total == 0.0 ? 0.0 : abs(weights[i]) / total,
            0.0
        });
    }

    sort(
        result.begin(),
        result.end(),
        [](const Importance& a, const Importance& b) {
            return a.score > b.score;
        }
    );

    return result;
}

vector<Importance> permutationImportance(
    const LinearModel& model,
    const Dataset& dataset,
    size_t repeats,
    uint32_t seed
) {
    InterpretationPolicy policy;

    policy.validate(
        dataset.size(),
        dataset.featureCount(),
        repeats
    );

    const auto& X = dataset.features();
    const auto& y = dataset.targets();

    const double baseline = evaluate(
        y,
        model.predict(X)
    ).mse;

    mt19937 generator(seed);
    vector<Importance> results;

    for (size_t feature = 0;
         feature < dataset.featureCount();
         ++feature) {

        vector<double> changes;

        for (size_t repeat = 0; repeat < repeats; ++repeat) {
            vector<vector<double>> modified = X;
            vector<double> values;

            for (const auto& row : X) {
                values.push_back(row[feature]);
            }

            shuffle(
                values.begin(),
                values.end(),
                generator
            );

            for (size_t i = 0; i < modified.size(); ++i) {
                modified[i][feature] = values[i];
            }

            double changedMse = evaluate(
                y,
                model.predict(modified)
            ).mse;

            changes.push_back(changedMse - baseline);
        }

        double meanChange = average(changes);
        double variance = 0.0;

        for (double value : changes) {
            variance += (value - meanChange) *
                (value - meanChange);
        }

        variance /= changes.size();

        results.push_back({
            model.names()[feature],
            meanChange,
            0.0,
            sqrt(variance)
        });
    }

    double positiveTotal = 0.0;

    for (const auto& item : results) {
        positiveTotal += max(0.0, item.score);
    }

    for (auto& item : results) {
        item.normalized = positiveTotal == 0.0
            ? 0.0
            : max(0.0, item.score) / positiveTotal;
    }

    sort(
        results.begin(),
        results.end(),
        [](const Importance& a, const Importance& b) {
            return a.score > b.score;
        }
    );

    return results;
}

vector<DependencePoint> partialDependence(
    const LinearModel& model,
    const Dataset& dataset,
    size_t featureIndex,
    size_t gridPoints
) {
    if (featureIndex >= dataset.featureCount()) {
        throw ValidationError("Feature index out of range.");
    }

    if (gridPoints < 2) {
        throw ValidationError("Grid requires at least two points.");
    }

    const auto& X = dataset.features();

    double minimum = X.front()[featureIndex];
    double maximum = minimum;

    for (const auto& row : X) {
        minimum = min(minimum, row[featureIndex]);
        maximum = max(maximum, row[featureIndex]);
    }

    vector<double> grid;

    if (minimum == maximum) {
        grid.push_back(minimum);
    } else {
        for (size_t i = 0; i < gridPoints; ++i) {
            double value = minimum +
                (maximum - minimum) *
                static_cast<double>(i) /
                static_cast<double>(gridPoints - 1);

            grid.push_back(value);
        }
    }

    vector<DependencePoint> result;

    for (double value : grid) {
        vector<vector<double>> modified = X;

        for (auto& row : modified) {
            row[featureIndex] = value;
        }

        double prediction = average(model.predict(modified));

        result.push_back({
            value,
            prediction
        });
    }

    return result;
}

Dataset createBusinessDataset(size_t samples, uint32_t seed) {
    if (samples < 20) {
        throw ValidationError("At least 20 samples are required.");
    }

    mt19937 generator(seed);

    uniform_real_distribution<double> revenue(100.0, 1000.0);
    uniform_real_distribution<double> debt(0.05, 0.95);
    uniform_real_distribution<double> retention(0.40, 0.99);
    normal_distribution<double> noise(0.0, 8.0);

    vector<vector<double>> X;
    vector<double> y;

    for (size_t i = 0; i < samples; ++i) {
        double r = revenue(generator);
        double d = debt(generator);
        double c = retention(generator);

        X.push_back({r, d, c});

        y.push_back(
            0.035 * r -
            120.0 * d +
            180.0 * c +
            noise(generator)
        );
    }

    return Dataset(move(X), move(y));
}

void printImportance(
    const string& heading,
    const vector<Importance>& results
) {
    cout << "\n" << heading << "\n";

    cout << left
         << setw(25) << "Feature"
         << right
         << setw(15) << "Importance"
         << setw(15) << "Normalized"
         << setw(15) << "Std. Dev."
         << "\n";

    for (const auto& item : results) {
        cout << left
             << setw(25) << item.feature
             << right
             << setw(15) << fixed << setprecision(5) << item.score
             << setw(14) << setprecision(2)
             << item.normalized * 100 << "%"
             << setw(15) << setprecision(5)
             << item.standardDeviation
             << "\n";
    }
}

int main() {
    try {
        vector<Feature> features = {
            {
                "revenue",
                "Annual business revenue",
                "thousand USD"
            },
            {
                "debt_ratio",
                "Debt relative to assets",
                "ratio"
            },
            {
                "customer_retention",
                "Customer retention rate",
                "ratio"
            }
        };

        vector<string> names;

        for (const auto& feature : features) {
            names.push_back(feature.name);
        }

        Dataset dataset = createBusinessDataset(240, 42);

        // Deterministic holdout split.
        vector<size_t> indices(dataset.size());

        iota(indices.begin(), indices.end(), 0);

        mt19937 splitGenerator(42);

        shuffle(
            indices.begin(),
            indices.end(),
            splitGenerator
        );

        size_t trainCount = dataset.size() * 75 / 100;

        vector<vector<double>> trainX, testX;
        vector<double> trainY, testY;

        for (size_t i = 0; i < indices.size(); ++i) {
            size_t index = indices[i];

            if (i < trainCount) {
                trainX.push_back(dataset.features()[index]);
                trainY.push_back(dataset.targets()[index]);
            } else {
                testX.push_back(dataset.features()[index]);
                testY.push_back(dataset.targets()[index]);
            }
        }

        Dataset training(move(trainX), move(trainY));
        Dataset testing(move(testX), move(testY));

        LinearModel model(names);
        model.fit(training);

        Evaluation metrics = evaluate(
            testing.targets(),
            model.predict(testing.features())
        );

        cout << "BUSINESS MODEL INTERPRETABILITY\n";
        cout << string(55, '=') << "\n";

        cout << fixed << setprecision(5);

        cout << "\nModel Evaluation\n";
        cout << "MSE:  " << metrics.mse << "\n";
        cout << "MAE:  " << metrics.mae << "\n";
        cout << "RMSE: " << metrics.rmse << "\n";

        printImportance(
            "Standardized Coefficient Importance",
            coefficientImportance(model)
        );

        printImportance(
            "Permutation Importance",
            permutationImportance(
                model,
                testing,
                15,
                42
            )
        );

        cout << "\nPartial Dependence\n";

        for (size_t i = 0; i < features.size(); ++i) {
            cout << "\nFeature: " << features[i].name << "\n";

            auto points = partialDependence(
                model,
                testing,
                i,
                8
            );

            for (const auto& point : points) {
                cout << "Value: "
                     << setw(10) << point.featureValue
                     << " | Average prediction: "
                     << point.averagePrediction
                     << "\n";
            }
        }

        // Behavioral checks.
        if (metrics.sampleCount != testing.size()) {
            throw runtime_error("Evaluation count mismatch.");
        }

        if (!isfinite(metrics.rmse)) {
            throw runtime_error("Invalid evaluation metric.");
        }

        cout << "\nAll interpretation operations completed.\n";
    }
    catch (const exception& error) {
        cerr << "Execution failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
