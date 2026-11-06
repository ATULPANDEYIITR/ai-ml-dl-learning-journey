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
#include <vector>

using namespace std;

enum class ExplanationMethod {
    SHAP,
    LIME,
    GLOBAL
};

struct Feature {
    string name;
    double minimum;
    double maximum;
    double baseline;
    double coefficient;
};

struct Observation {
    vector<double> values;
};

struct Explanation {
    ExplanationMethod method;
    double prediction;
    double baseline;
    map<string, double> contributions;
    double reconstructed;
};

class CreditRiskModel {
private:
    vector<Feature> features = {
        {"income", 10000, 200000, 60000, 0.000015},
        {"debt_ratio", 0, 1, 0.35, -2.2},
        {"credit_history", 0, 30, 8, 0.09},
        {"missed_payments", 0, 20, 1, -0.45},
        {"employment_years", 0, 40, 5, 0.055}
    };

    double intercept = -0.5;

public:
    const vector<Feature>& getFeatures() const {
        return features;
    }

    size_t dimensions() const {
        return features.size();
    }

    void validate(const Observation& observation) const {
        if (observation.values.size() != features.size()) {
            throw invalid_argument(
                "Observation has incorrect feature count"
            );
        }

        for (size_t i = 0; i < features.size(); ++i) {
            double value = observation.values[i];

            if (!isfinite(value)) {
                throw invalid_argument(
                    "Feature value must be finite"
                );
            }

            if (
                value < features[i].minimum ||
                value > features[i].maximum
            ) {
                throw invalid_argument(
                    "Feature outside permitted range: " +
                    features[i].name
                );
            }
        }
    }

    double probability(const Observation& observation) const {
        validate(observation);

        double score = intercept;

        for (size_t i = 0; i < features.size(); ++i) {
            score +=
                observation.values[i] *
                features[i].coefficient;
        }

        if (score >= 0) {
            return 1.0 / (1.0 + exp(-score));
        }

        double value = exp(score);
        return value / (1.0 + value);
    }

    int classify(
        const Observation& observation,
        double threshold = 0.5
    ) const {
        if (
            !isfinite(threshold) ||
            threshold <= 0 ||
            threshold >= 1
        ) {
            throw invalid_argument("Invalid threshold");
        }

        return probability(observation) >= threshold ? 1 : 0;
    }
};

class ExplanationEngine {
private:
    const CreditRiskModel& model;
    vector<Observation> background;

    double mean(const vector<double>& values) const {
        if (values.empty()) {
            throw runtime_error("Cannot calculate empty mean");
        }

        return accumulate(
            values.begin(),
            values.end(),
            0.0
        ) / values.size();
    }

    double coalitionValue(
        const Observation& observation,
        unsigned int mask
    ) const {
        vector<double> predictions;

        for (const auto& reference : background) {
            Observation hybrid = reference;

            for (size_t i = 0; i < model.dimensions(); ++i) {
                if (mask & (1U << i)) {
                    hybrid.values[i] = observation.values[i];
                }
            }

            predictions.push_back(model.probability(hybrid));
        }

        return mean(predictions);
    }

    static double factorial(int value) {
        double result = 1.0;

        for (int i = 2; i <= value; ++i) {
            result *= i;
        }

        return result;
    }

    vector<double> solve(
        vector<vector<double>> matrix,
        vector<double> vectorValues
    ) const {
        const size_t n = vectorValues.size();

        for (size_t col = 0; col < n; ++col) {
            size_t pivot = col;

            for (size_t row = col + 1; row < n; ++row) {
                if (
                    abs(matrix[row][col]) >
                    abs(matrix[pivot][col])
                ) {
                    pivot = row;
                }
            }

            if (abs(matrix[pivot][col]) < 1e-12) {
                throw runtime_error(
                    "Singular weighted regression matrix"
                );
            }

            swap(matrix[col], matrix[pivot]);
            swap(vectorValues[col], vectorValues[pivot]);

            double divisor = matrix[col][col];

            for (size_t j = col; j < n; ++j) {
                matrix[col][j] /= divisor;
            }

            vectorValues[col] /= divisor;

            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;

                double factor = matrix[row][col];

                for (size_t j = col; j < n; ++j) {
                    matrix[row][j] -=
                        factor * matrix[col][j];
                }

                vectorValues[row] -=
                    factor * vectorValues[col];
            }
        }

        return vectorValues;
    }

public:
    ExplanationEngine(
        const CreditRiskModel& modelReference,
        vector<Observation> referenceData
    )
        : model(modelReference),
          background(move(referenceData)) {

        if (background.empty()) {
            throw invalid_argument(
                "Background dataset cannot be empty"
            );
        }

        for (const auto& item : background) {
            model.validate(item);
        }
    }

    Explanation shap(const Observation& observation) const {
        model.validate(observation);

        const size_t n = model.dimensions();

        if (n > 20) {
            throw runtime_error(
                "Exact SHAP limited to 20 features"
            );
        }

        const unsigned int coalitionCount = 1U << n;

        vector<double> values(coalitionCount);

        for (
            unsigned int mask = 0;
            mask < coalitionCount;
            ++mask
        ) {
            values[mask] = coalitionValue(observation, mask);
        }

        map<string, double> contributions;

        for (size_t feature = 0; feature < n; ++feature) {
            double attribution = 0;

            for (
                unsigned int mask = 0;
                mask < coalitionCount;
                ++mask
            ) {
                if (mask & (1U << feature)) continue;

                int size = __builtin_popcount(mask);

                double weight =
                    factorial(size) *
                    factorial(static_cast<int>(n) - size - 1) /
                    factorial(static_cast<int>(n));

                unsigned int withFeature =
                    mask | (1U << feature);

                attribution += weight * (
                    values[withFeature] - values[mask]
                );
            }

            contributions[
                model.getFeatures()[feature].name
            ] = attribution;
        }

        double prediction = model.probability(observation);
        double baseline = values[0];

        double reconstructed = baseline;

        for (const auto& entry : contributions) {
            reconstructed += entry.second;
        }

        return {
            ExplanationMethod::SHAP,
            prediction,
            baseline,
            contributions,
            reconstructed
        };
    }

    Explanation lime(
        const Observation& observation,
        int samples = 1500,
        double kernelWidth = 0.75,
        unsigned int seed = 42
    ) const {
        model.validate(observation);

        if (samples < 20 || samples > 100000) {
            throw invalid_argument("Invalid sample count");
        }

        if (
            !isfinite(kernelWidth) ||
            kernelWidth <= 0
        ) {
            throw invalid_argument("Invalid kernel width");
        }

        const size_t n = model.dimensions();
        const size_t size = n + 1;

        vector<vector<double>> matrix(
            size,
            vector<double>(size, 0)
        );

        vector<double> target(size, 0);

        mt19937 generator(seed);
        normal_distribution<double> normal(0, 1);

        const auto& features = model.getFeatures();

        for (int iteration = 0; iteration < samples; ++iteration) {
            Observation perturbed;
            vector<double> row(size, 1.0);

            double distanceSquared = 0;

            for (size_t i = 0; i < n; ++i) {
                double range =
                    features[i].maximum - features[i].minimum;

                double value = observation.values[i] +
                    normal(generator) * range * 0.15;

                value = max(
                    features[i].minimum,
                    min(features[i].maximum, value)
                );

                perturbed.values.push_back(value);

                row[i + 1] =
                    (value - observation.values[i]) / range;

                distanceSquared +=
                    row[i + 1] * row[i + 1];
            }

            double weight = exp(
                -distanceSquared /
                (kernelWidth * kernelWidth)
            );

            double prediction = model.probability(perturbed);

            for (size_t i = 0; i < size; ++i) {
                target[i] += weight * row[i] * prediction;

                for (size_t j = 0; j < size; ++j) {
                    matrix[i][j] +=
                        weight * row[i] * row[j];
                }
            }
        }

        for (size_t i = 1; i < size; ++i) {
            matrix[i][i] += 1e-6;
        }

        vector<double> coefficients = solve(matrix, target);

        map<string, double> contributions;

        for (size_t i = 0; i < n; ++i) {
            contributions[features[i].name] =
                coefficients[i + 1];
        }

        return {
            ExplanationMethod::LIME,
            model.probability(observation),
            coefficients[0],
            contributions,
            coefficients[0]
        };
    }

    map<string, double> globalImportance(
        const vector<Observation>& dataset
    ) const {
        if (dataset.empty()) {
            throw invalid_argument("Dataset cannot be empty");
        }

        map<string, double> totals;

        for (const auto& feature : model.getFeatures()) {
            totals[feature.name] = 0;
        }

        for (const auto& observation : dataset) {
            Explanation explanation = shap(observation);

            for (const auto& item : explanation.contributions) {
                totals[item.first] += abs(item.second);
            }
        }

        for (auto& item : totals) {
            item.second /= dataset.size();
        }

        return totals;
    }
};

void printContributions(
    const string& title,
    const map<string, double>& contributions
) {
    cout << "\n" << title << "\n";

    vector<pair<string, double>> sorted(
        contributions.begin(),
        contributions.end()
    );

    sort(
        sorted.begin(),
        sorted.end(),
        [](const auto& a, const auto& b) {
            return abs(a.second) > abs(b.second);
        }
    );

    for (const auto& item : sorted) {
        cout << left << setw(22) << item.first
             << showpos << fixed << setprecision(6)
             << item.second << noshowpos << "\n";
    }
}

int main() {
    try {
        CreditRiskModel model;

        vector<Observation> background = {
            {{45000, 0.55, 3, 4, 2}},
            {{70000, 0.25, 12, 0, 10}},
            {{55000, 0.40, 7, 2, 5}},
            {{95000, 0.20, 18, 0, 15}}
        };

        Observation applicant = {
            {85000, 0.30, 14, 1, 9}
        };

        ExplanationEngine engine(model, background);

        double probability = model.probability(applicant);

        cout << "EXPLAINABLE AI CREDIT RISK ENGINE\n";
        cout << "=================================\n";

        cout << fixed << setprecision(6);
        cout << "Risk probability: " << probability << "\n";
        cout << "Classification: "
             << model.classify(applicant) << "\n";

        Explanation shap = engine.shap(applicant);
        Explanation lime = engine.lime(applicant);

        cout << "\nSHAP BASELINE: " << shap.baseline << "\n";

        printContributions(
            "LOCAL SHAP ATTRIBUTIONS",
            shap.contributions
        );

        cout << "Reconstructed prediction: "
             << shap.reconstructed << "\n";

        cout << "Additivity error: "
             << abs(shap.prediction - shap.reconstructed)
             << "\n";

        printContributions(
            "LOCAL LIME COEFFICIENTS",
            lime.contributions
        );

        vector<Observation> evaluation = background;

        evaluation.push_back(applicant);

        evaluation.push_back({
            {120000, 0.15, 20, 0, 18}
        });

        auto global = engine.globalImportance(evaluation);

        printContributions(
            "GLOBAL MEAN ABSOLUTE SHAP IMPORTANCE",
            global
        );

        if (
            abs(shap.prediction - shap.reconstructed) > 1e-9
        ) {
            throw runtime_error("SHAP additivity check failed");
        }

        cout << "\nValidation passed.\n";
        cout << "Exact SHAP enumeration completed.\n";
        cout << "LIME surrogate completed.\n";
        cout << "Global importance completed.\n";

    } catch (const exception& error) {
        cerr << "Execution error: "
             << error.what() << "\n";

        return 1;
    }

    return 0;
}
