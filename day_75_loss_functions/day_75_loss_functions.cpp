#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

/*
 * C++17 Loss Evaluation and Repository-Independent
 * Model Training Governance Engine.
 *
 * Implements:
 * - MSE and MAE
 * - Binary cross-entropy
 * - Categorical cross-entropy
 * - Stable logit calculations
 * - Analytical gradients
 * - Training evaluation
 * - Model acceptance policies
 */

enum class LossType {
    MSE,
    MAE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
};

enum class Reduction {
    NONE,
    SUM,
    MEAN
};

struct LossResult {
    string name;
    vector<double> individual;
    double value;
};

class LossEngine {
private:
    static constexpr double EPSILON = 1e-12;

    static void validateFinite(const vector<double>& values) {
        if (values.empty()) {
            throw invalid_argument("Input cannot be empty.");
        }

        for (double value : values) {
            if (!isfinite(value)) {
                throw invalid_argument("Non-finite input detected.");
            }
        }
    }

    static void validatePair(
        const vector<double>& actual,
        const vector<double>& predicted
    ) {
        validateFinite(actual);
        validateFinite(predicted);

        if (actual.size() != predicted.size()) {
            throw invalid_argument("Input sizes do not match.");
        }
    }

    static double reduce(
        const vector<double>& values,
        Reduction reduction
    ) {
        if (values.empty()) {
            throw invalid_argument("Cannot reduce empty values.");
        }

        double total = accumulate(
            values.begin(),
            values.end(),
            0.0
        );

        if (!isfinite(total)) {
            throw overflow_error("Loss accumulation overflow.");
        }

        switch (reduction) {
            case Reduction::NONE:
                throw invalid_argument(
                    "NONE reduction requires individual values."
                );

            case Reduction::SUM:
                return total;

            case Reduction::MEAN:
                return total / values.size();
        }

        throw invalid_argument("Unknown reduction.");
    }

    static double softplus(double x) {
        return max(x, 0.0) + log1p(exp(-abs(x)));
    }

public:
    static LossResult mse(
        const vector<double>& actual,
        const vector<double>& predicted,
        Reduction reduction = Reduction::MEAN
    ) {
        validatePair(actual, predicted);

        vector<double> values;

        for (size_t i = 0; i < actual.size(); ++i) {
            double error = actual[i] - predicted[i];
            values.push_back(error * error);
        }

        return {
            "MSE",
            values,
            reduction == Reduction::NONE
                ? 0.0
                : reduce(values, reduction)
        };
    }

    static LossResult mae(
        const vector<double>& actual,
        const vector<double>& predicted,
        Reduction reduction = Reduction::MEAN
    ) {
        validatePair(actual, predicted);

        vector<double> values;

        for (size_t i = 0; i < actual.size(); ++i) {
            values.push_back(abs(actual[i] - predicted[i]));
        }

        return {
            "MAE",
            values,
            reduction == Reduction::NONE
                ? 0.0
                : reduce(values, reduction)
        };
    }

    static LossResult binaryCrossEntropy(
        const vector<int>& actual,
        const vector<double>& probabilities,
        Reduction reduction = Reduction::MEAN
    ) {
        if (actual.empty() || actual.size() != probabilities.size()) {
            throw invalid_argument("Invalid binary batch.");
        }

        validateFinite(probabilities);

        vector<double> values;

        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != 0 && actual[i] != 1) {
                throw invalid_argument("Binary label must be 0 or 1.");
            }

            double p = probabilities[i];

            if (p < 0.0 || p > 1.0) {
                throw invalid_argument("Invalid probability.");
            }

            p = clamp(p, EPSILON, 1.0 - EPSILON);

            double loss = -(
                actual[i] * log(p) +
                (1 - actual[i]) * log1p(-p)
            );

            values.push_back(loss);
        }

        return {
            "Binary Cross-Entropy",
            values,
            reduction == Reduction::NONE
                ? 0.0
                : reduce(values, reduction)
        };
    }

    static LossResult binaryCrossEntropyWithLogits(
        const vector<int>& actual,
        const vector<double>& logits,
        Reduction reduction = Reduction::MEAN
    ) {
        if (actual.empty() || actual.size() != logits.size()) {
            throw invalid_argument("Invalid binary batch.");
        }

        validateFinite(logits);

        vector<double> values;

        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != 0 && actual[i] != 1) {
                throw invalid_argument("Invalid binary label.");
            }

            double z = logits[i];

            values.push_back(
                max(z, 0.0) -
                z * actual[i] +
                softplus(-abs(z))
            );
        }

        return {
            "Binary Cross-Entropy with Logits",
            values,
            reduction == Reduction::NONE
                ? 0.0
                : reduce(values, reduction)
        };
    }

    static vector<double> softmax(
        const vector<double>& logits
    ) {
        if (logits.size() < 2) {
            throw invalid_argument("At least two classes required.");
        }

        validateFinite(logits);

        double maximum = *max_element(
            logits.begin(),
            logits.end()
        );

        vector<double> probabilities;
        double denominator = 0.0;

        for (double z : logits) {
            double value = exp(z - maximum);
            probabilities.push_back(value);
            denominator += value;
        }

        for (double& p : probabilities) {
            p /= denominator;
        }

        return probabilities;
    }

    static LossResult categoricalCrossEntropy(
        const vector<int>& actual,
        const vector<vector<double>>& probabilities,
        Reduction reduction = Reduction::MEAN
    ) {
        if (actual.empty() || actual.size() != probabilities.size()) {
            throw invalid_argument("Invalid multiclass batch.");
        }

        size_t classes = probabilities[0].size();

        if (classes < 2) {
            throw invalid_argument("At least two classes required.");
        }

        vector<double> values;

        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] < 0 ||
                static_cast<size_t>(actual[i]) >= classes) {
                throw invalid_argument("Invalid class label.");
            }

            const auto& row = probabilities[i];

            if (row.size() != classes) {
                throw invalid_argument("Inconsistent class dimensions.");
            }

            validateFinite(row);

            double sum = accumulate(row.begin(), row.end(), 0.0);

            if (abs(sum - 1.0) > 1e-8) {
                throw invalid_argument(
                    "Probabilities must sum to one."
                );
            }

            for (double p : row) {
                if (p < 0 || p > 1) {
                    throw invalid_argument("Invalid probability.");
                }
            }

            double correct = max(
                row[actual[i]],
                EPSILON
            );

            values.push_back(-log(correct));
        }

        return {
            "Categorical Cross-Entropy",
            values,
            reduction == Reduction::NONE
                ? 0.0
                : reduce(values, reduction)
        };
    }

    static LossResult categoricalCrossEntropyWithLogits(
        const vector<int>& actual,
        const vector<vector<double>>& logits,
        Reduction reduction = Reduction::MEAN
    ) {
        if (actual.empty() || actual.size() != logits.size()) {
            throw invalid_argument("Invalid multiclass batch.");
        }

        size_t classes = logits[0].size();

        if (classes < 2) {
            throw invalid_argument("At least two classes required.");
        }

        vector<double> values;

        for (size_t i = 0; i < actual.size(); ++i) {
            const auto& row = logits[i];

            if (row.size() != classes ||
                actual[i] < 0 ||
                static_cast<size_t>(actual[i]) >= classes) {
                throw invalid_argument("Invalid multiclass dimensions.");
            }

            validateFinite(row);

            double maximum = *max_element(row.begin(), row.end());

            double sum = 0.0;

            for (double z : row) {
                sum += exp(z - maximum);
            }

            double loss = maximum + log(sum) - row[actual[i]];
            values.push_back(loss);
        }

        return {
            "Categorical Cross-Entropy with Logits",
            values,
            reduction == Reduction::NONE
                ? 0.0
                : reduce(values, reduction)
        };
    }

    static vector<double> mseGradient(
        const vector<double>& actual,
        const vector<double>& predicted
    ) {
        validatePair(actual, predicted);

        vector<double> gradient;

        for (size_t i = 0; i < actual.size(); ++i) {
            gradient.push_back(
                2.0 * (predicted[i] - actual[i]) / actual.size()
            );
        }

        return gradient;
    }

    static vector<double> binaryLogitGradient(
        const vector<int>& actual,
        const vector<double>& logits
    ) {
        if (actual.empty() || actual.size() != logits.size()) {
            throw invalid_argument("Invalid gradient batch.");
        }

        validateFinite(logits);

        vector<double> gradient;

        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != 0 && actual[i] != 1) {
                throw invalid_argument("Invalid binary target.");
            }

            double probability;

            if (logits[i] >= 0) {
                probability = 1.0 / (1.0 + exp(-logits[i]));
            } else {
                double e = exp(logits[i]);
                probability = e / (1.0 + e);
            }

            gradient.push_back(
                (probability - actual[i]) / actual.size()
            );
        }

        return gradient;
    }
};

struct EvaluationPolicy {
    double maximumRegressionLoss;
    double maximumClassificationLoss;

    void validate() const {
        if (!isfinite(maximumRegressionLoss) ||
            !isfinite(maximumClassificationLoss) ||
            maximumRegressionLoss < 0 ||
            maximumClassificationLoss < 0) {
            throw invalid_argument("Invalid evaluation policy.");
        }
    }
};

struct EvaluationRecord {
    string model;
    string metric;
    double loss;
    double threshold;
    bool accepted;
};

class GovernanceEngine {
private:
    EvaluationPolicy policy;
    vector<EvaluationRecord> history;

public:
    explicit GovernanceEngine(EvaluationPolicy policy)
        : policy(policy) {
        this->policy.validate();
    }

    EvaluationRecord evaluate(
        const string& model,
        const LossResult& result,
        bool classification
    ) {
        if (model.empty() || !isfinite(result.value) || result.value < 0) {
            throw invalid_argument("Invalid evaluation record.");
        }

        double threshold = classification
            ? policy.maximumClassificationLoss
            : policy.maximumRegressionLoss;

        EvaluationRecord record {
            model,
            result.name,
            result.value,
            threshold,
            result.value <= threshold
        };

        history.push_back(record);

        return record;
    }

    void report() const {
        cout << "\nMODEL GOVERNANCE REPORT\n";

        for (const auto& record : history) {
            cout << record.model
                 << " | " << record.metric
                 << " | Loss: " << record.loss
                 << " | Threshold: " << record.threshold
                 << " | " << (record.accepted ? "PASS" : "FAIL")
                 << '\n';
        }
    }
};

void printResult(const LossResult& result) {
    cout << result.name
         << " = " << fixed << setprecision(8)
         << result.value << '\n';
}

int main() {
    try {
        cout << "LOSS FUNCTION ENGINE\n";
        cout << "====================\n";

        vector<double> actual = {3.0, -0.5, 2.0, 7.0};
        vector<double> predicted = {2.5, 0.0, 2.0, 8.0};

        LossResult mse = LossEngine::mse(actual, predicted);
        LossResult mae = LossEngine::mae(actual, predicted);

        printResult(mse);
        printResult(mae);

        vector<int> binaryActual = {1, 0, 1, 0};
        vector<double> binaryProbabilities = {
            0.95, 0.10, 0.80, 0.20
        };

        LossResult binary = LossEngine::binaryCrossEntropy(
            binaryActual,
            binaryProbabilities
        );

        printResult(binary);

        vector<double> binaryLogits = {
            3.0, -2.2, 1.4, -1.4
        };

        LossResult binaryStable =
            LossEngine::binaryCrossEntropyWithLogits(
                binaryActual,
                binaryLogits
            );

        printResult(binaryStable);

        vector<int> multiclassActual = {0, 2, 1};

        vector<vector<double>> multiclassProbabilities = {
            {0.70, 0.20, 0.10},
            {0.10, 0.20, 0.70},
            {0.15, 0.75, 0.10}
        };

        LossResult categorical =
            LossEngine::categoricalCrossEntropy(
                multiclassActual,
                multiclassProbabilities
            );

        printResult(categorical);

        vector<vector<double>> multiclassLogits = {
            {3.0, 1.0, 0.2},
            {0.1, 0.3, 2.5},
            {0.2, 2.8, 0.1}
        };

        LossResult stableCategorical =
            LossEngine::categoricalCrossEntropyWithLogits(
                multiclassActual,
                multiclassLogits
            );

        printResult(stableCategorical);

        cout << "\nMSE GRADIENT\n";

        for (double value : LossEngine::mseGradient(actual, predicted)) {
            cout << value << " ";
        }

        cout << "\n\nBINARY LOGIT GRADIENT\n";

        for (double value :
             LossEngine::binaryLogitGradient(
                 binaryActual,
                 binaryLogits
             )) {
            cout << value << " ";
        }

        cout << '\n';

        GovernanceEngine governance({
            1.0,
            0.5
        });

        governance.evaluate("Regression-v2", mse, false);
        governance.evaluate("Classification-v3", binary, true);
        governance.evaluate("Multiclass-v1", categorical, true);

        governance.report();

        // Extreme logits test: stable formulation avoids exp overflow.
        LossResult extreme =
            LossEngine::binaryCrossEntropyWithLogits(
                {1, 0},
                {1000.0, -1000.0}
            );

        cout << "\nExtreme-logit loss: " << extreme.value << '\n';

        // Invalid labels must be rejected.
        try {
            LossEngine::binaryCrossEntropy({2}, {0.5});
        } catch (const invalid_argument& error) {
            cout << "Expected validation failure: "
                 << error.what() << '\n';
        }

        return 0;

    } catch (const exception& error) {
        cerr << "Execution error: " << error.what() << '\n';
        return 1;
    }
}
