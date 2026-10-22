#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * Linear soft-margin SVM used as a repository-governance risk classifier.
 *
 * This program intentionally models merge eligibility as a deterministic
 * governance engine rather than pretending that an SVM prediction itself
 * replaces repository policy.
 *
 * Feature vector:
 *   0: normalized security-defect count
 *   1: normalized test-coverage ratio
 *
 * Label:
 *   +1 = acceptable-risk region
 *   -1 = remediation-risk region
 *
 * SVM decision:
 *
 *   f(x) = w.x + b
 *
 * Functional margin:
 *
 *   y f(x)
 *
 * Soft-margin objective:
 *
 *   1/2 ||w||^2 + C * sum(max(0, 1 - y f(x)))
 */

struct PullRequest {
    std::string id;
    std::string sourceBranch;
    std::string targetBranch;
    std::vector<double> features;
    int label;
};

struct Review {
    std::string reviewer;
    bool approved;
    bool stale;
};

struct StatusCheck {
    std::string name;
    bool passed;
};

struct BranchProtection {
    std::string branch;
    int requiredApprovals;
    std::vector<std::string> requiredChecks;
    bool prohibitDirectPush;
    bool prohibitForcePush;
    bool requireResolvedConversations;
};

struct GovernanceResult {
    bool mergeEligible;
    std::vector<std::string> reasons;
};

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions must match.");
    }

    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double norm(const std::vector<double>& values) {
    return std::sqrt(dot(values, values));
}

class StandardScaler {
private:
    std::vector<double> means_;
    std::vector<double> standardDeviations_;

public:
    void fit(const std::vector<std::vector<double>>& matrix) {
        if (matrix.empty()) {
            throw std::invalid_argument("Cannot fit scaler to empty data.");
        }

        const std::size_t dimension = matrix.front().size();

        if (dimension == 0) {
            throw std::invalid_argument("Feature dimension cannot be zero.");
        }

        if (std::any_of(matrix.begin(), matrix.end(),
                        [dimension](const auto& row) {
                            return row.size() != dimension;
                        })) {
            throw std::invalid_argument("Inconsistent feature dimensions.");
        }

        means_.assign(dimension, 0.0);
        standardDeviations_.assign(dimension, 1.0);

        for (const auto& row : matrix) {
            for (std::size_t j = 0; j < dimension; ++j) {
                means_[j] += row[j];
            }
        }

        for (double& mean : means_) {
            mean /= static_cast<double>(matrix.size());
        }

        for (std::size_t j = 0; j < dimension; ++j) {
            double variance = 0.0;

            for (const auto& row : matrix) {
                const double delta = row[j] - means_[j];
                variance += delta * delta;
            }

            variance /= static_cast<double>(matrix.size());
            const double standardDeviation = std::sqrt(variance);

            // A constant feature contributes no directional information.
            standardDeviations_[j] =
                standardDeviation > 1e-12 ? standardDeviation : 1.0;
        }
    }

    std::vector<double> transform(
        const std::vector<double>& row) const {

        if (row.size() != means_.size()) {
            throw std::invalid_argument("Scaler dimension mismatch.");
        }

        std::vector<double> result(row.size());

        for (std::size_t j = 0; j < row.size(); ++j) {
            result[j] =
                (row[j] - means_[j]) / standardDeviations_[j];
        }

        return result;
    }
};

class LinearSoftMarginSVM {
private:
    double C_;
    double learningRate_;
    int epochs_;
    std::vector<double> weights_;
    double bias_ = 0.0;

public:
    LinearSoftMarginSVM(double C, double learningRate, int epochs)
        : C_(C), learningRate_(learningRate), epochs_(epochs) {

        if (C <= 0.0 || learningRate <= 0.0 || epochs <= 0) {
            throw std::invalid_argument("Invalid SVM hyperparameters.");
        }
    }

    void fit(const std::vector<std::pair<std::vector<double>, int>>& data) {
        if (data.empty()) {
            throw std::invalid_argument("Training data cannot be empty.");
        }

        const std::size_t dimension = data.front().first.size();
        weights_.assign(dimension, 0.0);
        bias_ = 0.0;

        for (const auto& [features, label] : data) {
            if (features.size() != dimension) {
                throw std::invalid_argument("Inconsistent feature dimensions.");
            }

            if (label != -1 && label != 1) {
                throw std::invalid_argument("Labels must be -1 or +1.");
            }
        }

        /*
         * Sequential subgradient descent is deliberately used here because
         * the hinge-loss boundary is explicit. Production systems can use
         * optimized batch/coordinate solvers when datasets are large.
         */
        for (int epoch = 0; epoch < epochs_; ++epoch) {
            for (const auto& [features, label] : data) {
                const double score = decision(features);
                const double functionalMargin = label * score;

                if (functionalMargin >= 1.0) {
                    for (std::size_t j = 0; j < weights_.size(); ++j) {
                        weights_[j] -= learningRate_ * weights_[j];
                    }
                } else {
                    for (std::size_t j = 0; j < weights_.size(); ++j) {
                        const double gradient =
                            weights_[j] - C_ * label * features[j];

                        weights_[j] -= learningRate_ * gradient;
                    }

                    bias_ -= learningRate_ * (-C_ * label);
                }
            }
        }
    }

    double decision(const std::vector<double>& features) const {
        if (features.size() != weights_.size()) {
            throw std::invalid_argument("Prediction dimension mismatch.");
        }

        return dot(weights_, features) + bias_;
    }

    int predict(const std::vector<double>& features) const {
        return decision(features) >= 0.0 ? 1 : -1;
    }

    double functionalMargin(
        const std::vector<double>& features,
        int label) const {

        return label * decision(features);
    }

    double hingeLoss(
        const std::vector<double>& features,
        int label) const {

        return std::max(0.0, 1.0 - functionalMargin(features, label));
    }

    double objective(
        const std::vector<std::pair<std::vector<double>, int>>& data) const {

        const double regularization =
            0.5 * dot(weights_, weights_);

        double hinge = 0.0;

        for (const auto& [features, label] : data) {
            hinge += hingeLoss(features, label);
        }

        return regularization + C_ * hinge;
    }

    double marginWidth() const {
        const double weightNorm = norm(weights_);
        if (weightNorm == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return 2.0 / weightNorm;
    }

    std::vector<std::size_t> supportVectorIndexes(
        const std::vector<std::pair<std::vector<double>, int>>& data,
        double tolerance = 0.05) const {

        std::vector<std::size_t> indexes;

        for (std::size_t i = 0; i < data.size(); ++i) {
            const auto& [features, label] = data[i];

            if (functionalMargin(features, label) <= 1.0 + tolerance) {
                indexes.push_back(i);
            }
        }

        return indexes;
    }
};

class MergeEligibilityEngine {
private:
    BranchProtection protection_;

public:
    explicit MergeEligibilityEngine(BranchProtection protection)
        : protection_(std::move(protection)) {}

    GovernanceResult evaluate(
        const PullRequest& pullRequest,
        const std::vector<Review>& reviews,
        const std::vector<StatusCheck>& checks,
        bool conversationsResolved,
        bool sourceSynchronized,
        bool mergeConflict,
        int svmClassification) const {

        GovernanceResult result{true, {}};

        if (pullRequest.targetBranch != protection_.branch) {
            result.mergeEligible = false;
            result.reasons.push_back("Target branch is not the protected branch.");
        }

        if (!sourceSynchronized) {
            result.mergeEligible = false;
            result.reasons.push_back("Source branch is behind the target branch.");
        }

        if (mergeConflict) {
            result.mergeEligible = false;
            result.reasons.push_back("Pull Request has unresolved merge conflicts.");
        }

        if (protection_.requireResolvedConversations &&
            !conversationsResolved) {
            result.mergeEligible = false;
            result.reasons.push_back("Required review conversations are unresolved.");
        }

        int validApprovals = 0;

        for (const auto& review : reviews) {
            if (review.approved && !review.stale) {
                ++validApprovals;
            }
        }

        if (validApprovals < protection_.requiredApprovals) {
            result.mergeEligible = false;
            result.reasons.push_back(
                "Required number of current approvals has not been reached.");
        }

        for (const auto& required : protection_.requiredChecks) {
            auto it = std::find_if(
                checks.begin(),
                checks.end(),
                [&required](const StatusCheck& check) {
                    return check.name == required;
                });

            if (it == checks.end() || !it->passed) {
                result.mergeEligible = false;
                result.reasons.push_back(
                    "Required status check is missing or failing: " + required);
            }
        }

        /*
         * The classifier is a risk signal, not a replacement for repository
         * governance. The protected-branch rules remain authoritative.
         */
        if (svmClassification < 0) {
            result.mergeEligible = false;
            result.reasons.push_back(
                "SVM classified the change in the remediation-risk region.");
        }

        return result;
    }
};

int main() {
    try {
        const std::vector<PullRequest> pullRequests = {
            {"PR-301", "feature/payment-validation", "main", {0.4, 0.95}, 1},
            {"PR-302", "feature/session-hardening", "main", {1.0, 0.88}, 1},
            {"PR-303", "feature/legacy-cleanup", "main", {3.4, 0.56}, -1},
            {"PR-304", "feature/auth-refactor", "main", {2.1, 0.68}, -1},
            {"PR-305", "feature/query-optimization", "main", {1.7, 0.74}, -1}
        };

        std::vector<std::vector<double>> rawFeatures;

        for (const auto& pr : pullRequests) {
            rawFeatures.push_back(pr.features);
        }

        StandardScaler scaler;
        scaler.fit(rawFeatures);

        std::vector<std::pair<std::vector<double>, int>> trainingData;

        for (const auto& pr : pullRequests) {
            trainingData.emplace_back(
                scaler.transform(pr.features),
                pr.label
            );
        }

        LinearSoftMarginSVM svm(1.0, 0.002, 5000);
        svm.fit(trainingData);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Linear soft-margin SVM\n";
        std::cout << "=====================\n";
        std::cout << "Margin width: "
                  << svm.marginWidth() << "\n";
        std::cout << "Objective: "
                  << svm.objective(trainingData) << "\n";

        const auto supportVectors =
            svm.supportVectorIndexes(trainingData);

        std::cout << "\nSupport-vector candidates:\n";

        for (const std::size_t index : supportVectors) {
            const auto& [features, label] = trainingData[index];

            std::cout
                << "  " << pullRequests[index].id
                << " functional_margin="
                << svm.functionalMargin(features, label)
                << " hinge_loss="
                << svm.hingeLoss(features, label)
                << "\n";
        }

        BranchProtection protection{
            "main",
            2,
            {"unit-tests", "security-scan"},
            true,
            true,
            true
        };

        MergeEligibilityEngine governance(protection);

        const std::vector<Review> reviews = {
            {"alice", true, false},
            {"bob", true, false},
            {"carol", true, true}
        };

        const std::vector<StatusCheck> checks = {
            {"unit-tests", true},
            {"security-scan", true},
            {"integration-tests", true}
        };

        std::cout << "\nMerge-eligibility decisions:\n";

        for (const auto& pr : pullRequests) {
            const auto normalized = scaler.transform(pr.features);
            const int classification = svm.predict(normalized);

            const GovernanceResult result = governance.evaluate(
                pr,
                reviews,
                checks,
                true,
                true,
                false,
                classification
            );

            std::cout
                << "\n" << pr.id
                << " SVM=" << classification
                << " merge="
                << (result.mergeEligible ? "eligible" : "blocked")
                << "\n";

            for (const auto& reason : result.reasons) {
                std::cout << "  - " << reason << "\n";
            }
        }

        /*
         * A second governance scenario demonstrates that a passing classifier
         * does not override branch-protection requirements.
         */
        std::cout << "\nPolicy failure scenario:\n";

        const PullRequest candidate = pullRequests[0];

        const GovernanceResult blocked =
            governance.evaluate(
                candidate,
                {{"alice", true, false}},
                checks,
                false,
                true,
                false,
                1
            );

        std::cout
            << candidate.id
            << " merge="
            << (blocked.mergeEligible ? "eligible" : "blocked")
            << "\n";

        for (const auto& reason : blocked.reasons) {
            std::cout << "  - " << reason << "\n";
        }

        std::cout << "\nDesign constraints:\n";
        std::cout << "  * SVM classification is a risk signal.\n";
        std::cout << "  * Protected-branch rules remain authoritative.\n";
        std::cout << "  * Stale approvals do not satisfy the approval threshold.\n";
        std::cout << "  * Required status checks are evaluated independently.\n";
        std::cout << "  * Feature scaling prevents measurement units from dominating\n";
        std::cout << "    the learned hyperplane.\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Execution error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
