#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

/*
 * Classical ML model governance and release engine.
 *
 * Scenario:
 * A customer churn prediction model is evaluated before production release.
 *
 * Responsibilities:
 * - Dataset contract validation
 * - Model evaluation
 * - Classification metrics
 * - Drift assessment
 * - Latency assessment
 * - Fairness assessment
 * - Branch-like model version management
 * - Release policy enforcement
 * - Rollback
 *
 * Compile:
 * g++ -std=c++17 -O2 -Wall -Wextra ml_governance.cpp -o ml_governance
 */

// -------------------------------------------------------
// Enumerations
// -------------------------------------------------------

enum class ModelState {
    REGISTERED,
    VALIDATING,
    EVALUATING,
    APPROVED,
    REJECTED,
    DEPLOYED,
    RETIRED
};

enum class CheckStatus {
    PASSED,
    FAILED,
    WARNING
};

string stateName(ModelState state) {
    switch (state) {
        case ModelState::REGISTERED: return "REGISTERED";
        case ModelState::VALIDATING: return "VALIDATING";
        case ModelState::EVALUATING: return "EVALUATING";
        case ModelState::APPROVED: return "APPROVED";
        case ModelState::REJECTED: return "REJECTED";
        case ModelState::DEPLOYED: return "DEPLOYED";
        case ModelState::RETIRED: return "RETIRED";
    }

    throw logic_error("Unknown model state.");
}

// -------------------------------------------------------
// Domain structures
// -------------------------------------------------------

struct Observation {
    int customerId;
    double tenure;
    double monthlyCharges;
    double supportCalls;
    int actual;
    double probability;
};

struct ModelMetrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double rocAuc = 0.0;
    double drift = 0.0;
    double latencyMs = 0.0;
    double fairnessGap = 0.0;
};

struct QualityCheck {
    string name;
    CheckStatus status;
    string explanation;
};

struct ReleaseDecision {
    bool eligible = false;
    vector<QualityCheck> checks;
};

// -------------------------------------------------------
// Dataset contract
// -------------------------------------------------------

class DatasetValidator {
public:
    static void validate(
        const vector<Observation>& observations
    ) {
        if (observations.size() < 50) {
            throw invalid_argument(
                "Dataset requires at least 50 observations."
            );
        }

        set<int> identifiers;

        int positives = 0;
        int negatives = 0;

        for (const auto& row : observations) {
            if (row.customerId <= 0) {
                throw invalid_argument(
                    "Customer ID must be positive."
                );
            }

            if (!identifiers.insert(row.customerId).second) {
                throw invalid_argument(
                    "Duplicate customer identifier."
                );
            }

            if (
                !isfinite(row.tenure) ||
                !isfinite(row.monthlyCharges) ||
                !isfinite(row.supportCalls) ||
                !isfinite(row.probability)
            ) {
                throw invalid_argument(
                    "Non-finite feature or prediction."
                );
            }

            if (row.tenure < 0 || row.monthlyCharges < 0 ||
                row.supportCalls < 0) {
                throw invalid_argument(
                    "Negative feature value."
                );
            }

            if (row.actual != 0 && row.actual != 1) {
                throw invalid_argument(
                    "Target must be binary."
                );
            }

            if (
                row.probability < 0.0 ||
                row.probability > 1.0
            ) {
                throw invalid_argument(
                    "Prediction probability outside [0,1]."
                );
            }

            row.actual == 1 ? positives++ : negatives++;
        }

        if (positives == 0 || negatives == 0) {
            throw invalid_argument(
                "Both target classes are required."
            );
        }
    }
};

// -------------------------------------------------------
// Classification evaluation
// -------------------------------------------------------

class ModelEvaluator {
public:
    static ModelMetrics evaluate(
        const vector<Observation>& observations,
        double threshold
    ) {
        if (
            threshold < 0.0 ||
            threshold > 1.0
        ) {
            throw invalid_argument("Invalid threshold.");
        }

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        for (const auto& row : observations) {
            int predicted =
                row.probability >= threshold ? 1 : 0;

            if (row.actual == 1 && predicted == 1) tp++;
            else if (row.actual == 0 && predicted == 0) tn++;
            else if (row.actual == 0 && predicted == 1) fp++;
            else fn++;
        }

        ModelMetrics metrics;

        const double total = observations.size();

        metrics.accuracy = (tp + tn) / total;

        metrics.precision =
            tp + fp == 0 ? 0.0 :
            static_cast<double>(tp) / (tp + fp);

        metrics.recall =
            tp + fn == 0 ? 0.0 :
            static_cast<double>(tp) / (tp + fn);

        metrics.f1 =
            metrics.precision + metrics.recall == 0.0
            ? 0.0
            : 2.0 * metrics.precision * metrics.recall /
              (metrics.precision + metrics.recall);

        metrics.rocAuc = calculateAuc(observations);

        return metrics;
    }

private:
    static double calculateAuc(
        const vector<Observation>& observations
    ) {
        vector<Observation> sorted = observations;

        sort(
            sorted.begin(),
            sorted.end(),
            [](const Observation& a, const Observation& b) {
                return a.probability < b.probability;
            }
        );

        long long positives = 0;
        long long negatives = 0;

        for (const auto& row : sorted) {
            row.actual == 1 ? positives++ : negatives++;
        }

        if (positives == 0 || negatives == 0) {
            throw invalid_argument(
                "ROC-AUC requires two classes."
            );
        }

        double rankSum = 0.0;

        size_t i = 0;

        while (i < sorted.size()) {
            size_t j = i + 1;

            while (
                j < sorted.size() &&
                sorted[j].probability ==
                sorted[i].probability
            ) {
                j++;
            }

            // Average ranks handle tied prediction scores.
            double averageRank =
                (static_cast<double>(i + 1) +
                 static_cast<double>(j)) / 2.0;

            for (size_t k = i; k < j; k++) {
                if (sorted[k].actual == 1) {
                    rankSum += averageRank;
                }
            }

            i = j;
        }

        return (
            rankSum -
            positives * (positives + 1) / 2.0
        ) / (positives * negatives);
    }
};

// -------------------------------------------------------
// Fairness and operational assessment
// -------------------------------------------------------

class OperationalEvaluator {
public:
    static double calculateGroupRecall(
        const vector<Observation>& observations,
        int begin,
        int end,
        double threshold
    ) {
        int truePositive = 0;
        int actualPositive = 0;

        for (const auto& row : observations) {
            if (
                row.customerId < begin ||
                row.customerId >= end
            ) {
                continue;
            }

            if (row.actual == 1) {
                actualPositive++;

                if (row.probability >= threshold) {
                    truePositive++;
                }
            }
        }

        if (actualPositive == 0) {
            throw invalid_argument(
                "Fairness group has no positive observations."
            );
        }

        return static_cast<double>(truePositive) /
               actualPositive;
    }

    static double calculateFairnessGap(
        const vector<Observation>& observations,
        double threshold
    ) {
        double first = calculateGroupRecall(
            observations, 1, 51, threshold
        );

        double second = calculateGroupRecall(
            observations, 51, 101, threshold
        );

        return abs(first - second);
    }
};

// -------------------------------------------------------
// Release policy
// -------------------------------------------------------

class ReleasePolicy {
private:
    double minimumAuc;
    double minimumRecall;
    double maximumDrift;
    double maximumLatency;
    double maximumFairnessGap;

public:
    ReleasePolicy(
        double auc,
        double recall,
        double drift,
        double latency,
        double fairness
    ) :
        minimumAuc(auc),
        minimumRecall(recall),
        maximumDrift(drift),
        maximumLatency(latency),
        maximumFairnessGap(fairness) {

        if (
            minimumAuc < 0 || minimumAuc > 1 ||
            minimumRecall < 0 || minimumRecall > 1 ||
            maximumDrift < 0 ||
            maximumLatency <= 0 ||
            maximumFairnessGap < 0
        ) {
            throw invalid_argument(
                "Invalid release policy."
            );
        }
    }

    ReleaseDecision evaluate(
        const ModelMetrics& metrics
    ) const {
        ReleaseDecision decision;

        auto addCheck = [&decision](
            const string& name,
            bool passed,
            const string& explanation
        ) {
            decision.checks.push_back({
                name,
                passed ? CheckStatus::PASSED :
                         CheckStatus::FAILED,
                explanation
            });
        };

        addCheck(
            "ROC-AUC",
            metrics.rocAuc >= minimumAuc,
            "Minimum ROC-AUC requirement."
        );

        addCheck(
            "Recall",
            metrics.recall >= minimumRecall,
            "Minimum positive-class recall."
        );

        addCheck(
            "Feature drift",
            metrics.drift <= maximumDrift,
            "Maximum permitted drift."
        );

        addCheck(
            "Inference latency",
            metrics.latencyMs <= maximumLatency,
            "Maximum prediction latency."
        );

        addCheck(
            "Fairness gap",
            metrics.fairnessGap <= maximumFairnessGap,
            "Maximum group recall difference."
        );

        decision.eligible = all_of(
            decision.checks.begin(),
            decision.checks.end(),
            [](const QualityCheck& check) {
                return check.status == CheckStatus::PASSED;
            }
        );

        return decision;
    }
};

// -------------------------------------------------------
// Model registry
// -------------------------------------------------------

struct Model {
    string version;
    ModelState state;
    ModelMetrics metrics;
};

class ModelRegistry {
private:
    map<string, Model> models;
    string productionVersion;

public:
    void registerModel(Model model) {
        if (model.version.empty()) {
            throw invalid_argument(
                "Model version cannot be empty."
            );
        }

        if (models.count(model.version)) {
            throw invalid_argument(
                "Duplicate model version."
            );
        }

        model.state = ModelState::REGISTERED;
        models.emplace(model.version, model);
    }

    void setState(
        const string& version,
        ModelState next
    ) {
        auto it = models.find(version);

        if (it == models.end()) {
            throw out_of_range("Unknown model version.");
        }

        ModelState current = it->second.state;

        const map<ModelState, set<ModelState>> transitions = {
            {
                ModelState::REGISTERED,
                {ModelState::VALIDATING, ModelState::REJECTED}
            },
            {
                ModelState::VALIDATING,
                {ModelState::EVALUATING, ModelState::REJECTED}
            },
            {
                ModelState::EVALUATING,
                {ModelState::APPROVED, ModelState::REJECTED}
            },
            {
                ModelState::APPROVED,
                {ModelState::DEPLOYED}
            },
            {
                ModelState::DEPLOYED,
                {ModelState::RETIRED}
            },
            {ModelState::REJECTED, {}},
            {ModelState::RETIRED, {}}
        };

        auto rule = transitions.find(current);

        if (
            rule == transitions.end() ||
            !rule->second.count(next)
        ) {
            throw logic_error(
                "Invalid model state transition."
            );
        }

        it->second.state = next;
    }

    void promote(const string& version) {
        auto it = models.find(version);

        if (it == models.end()) {
            throw out_of_range("Unknown model version.");
        }

        if (it->second.state != ModelState::APPROVED) {
            throw logic_error(
                "Only approved models can be deployed."
            );
        }

        if (!productionVersion.empty()) {
            setState(
                productionVersion,
                ModelState::RETIRED
            );
        }

        setState(version, ModelState::DEPLOYED);
        productionVersion = version;
    }

    void rollback(const string& version) {
        promote(version);
    }

    void print() const {
        cout << "\nMODEL REGISTRY\n";

        for (const auto& entry : models) {
            cout << entry.first
                 << " | "
                 << stateName(entry.second.state)
                 << '\n';
        }

        cout << "Production version: "
             << (
                 productionVersion.empty()
                 ? "NONE"
                 : productionVersion
             )
             << '\n';
    }

    const Model& get(const string& version) const {
        auto it = models.find(version);

        if (it == models.end()) {
            throw out_of_range("Model not found.");
        }

        return it->second;
    }
};

// -------------------------------------------------------
// Demonstration dataset
// -------------------------------------------------------

vector<Observation> generateDataset() {
    vector<Observation> observations;

    for (int i = 1; i <= 100; i++) {
        int actual = (i % 4 == 0) ? 1 : 0;

        double probability;

        if (actual == 1) {
            probability = 0.62 + (i % 30) / 100.0;
        } else {
            probability = 0.05 + (i % 30) / 100.0;
        }

        observations.push_back({
            i,
            12.0 + i % 60,
            30.0 + i % 100,
            static_cast<double>(i % 5),
            actual,
            probability
        });
    }

    return observations;
}

// -------------------------------------------------------
// Application entry point
// -------------------------------------------------------

int main() {
    try {
        cout << fixed << setprecision(4);

        auto dataset = generateDataset();

        DatasetValidator::validate(dataset);

        cout << "Dataset validation: PASSED\n";
        cout << "Observation count: "
             << dataset.size() << '\n';

        ModelMetrics metrics =
            ModelEvaluator::evaluate(dataset, 0.50);

        metrics.drift = 0.08;
        metrics.latencyMs = 45.0;

        metrics.fairnessGap =
            OperationalEvaluator::calculateFairnessGap(
                dataset, 0.50
            );

        cout << "\nMODEL EVALUATION\n";

        cout << "Accuracy: "
             << metrics.accuracy << '\n';

        cout << "Precision: "
             << metrics.precision << '\n';

        cout << "Recall: "
             << metrics.recall << '\n';

        cout << "F1: "
             << metrics.f1 << '\n';

        cout << "ROC-AUC: "
             << metrics.rocAuc << '\n';

        cout << "Drift: "
             << metrics.drift << '\n';

        cout << "Latency: "
             << metrics.latencyMs << " ms\n";

        cout << "Fairness gap: "
             << metrics.fairnessGap << '\n';

        ReleasePolicy policy(
            0.70,
            0.50,
            0.20,
            250.0,
            0.25
        );

        ReleaseDecision decision =
            policy.evaluate(metrics);

        cout << "\nRELEASE GATE\n";

        for (const auto& check : decision.checks) {
            cout << check.name
                 << " : "
                 << (
                     check.status == CheckStatus::PASSED
                     ? "PASSED"
                     : "FAILED"
                 )
                 << " | "
                 << check.explanation
                 << '\n';
        }

        ModelRegistry registry;

        registry.registerModel({
            "churn-1.0.0",
            ModelState::REGISTERED,
            metrics
        });

        if (decision.eligible) {
            registry.setState(
                "churn-1.0.0",
                ModelState::VALIDATING
            );

            registry.setState(
                "churn-1.0.0",
                ModelState::EVALUATING
            );

            registry.setState(
                "churn-1.0.0",
                ModelState::APPROVED
            );

            registry.promote("churn-1.0.0");

            cout << "\nMODEL DEPLOYMENT: APPROVED\n";
        } else {
            registry.setState(
                "churn-1.0.0",
                ModelState::REJECTED
            );

            cout << "\nMODEL DEPLOYMENT: REJECTED\n";
        }

        registry.print();

        // Demonstrate a release failure without affecting
        // the production registry.
        ModelMetrics failingMetrics = metrics;
        failingMetrics.rocAuc = 0.40;

        ReleaseDecision failure =
            policy.evaluate(failingMetrics);

        cout << "\nFAILURE SCENARIO\n";
        cout << "Release eligible: "
             << boolalpha
             << failure.eligible
             << '\n';

        // Edge-case validation.
        try {
            DatasetValidator::validate({});
        } catch (const invalid_argument& error) {
            cout << "Empty dataset rejected: "
                 << error.what() << '\n';
        }

        return 0;

    } catch (const exception& error) {
        cerr << "Governance engine failure: "
             << error.what() << '\n';

        return 1;
    }
}
