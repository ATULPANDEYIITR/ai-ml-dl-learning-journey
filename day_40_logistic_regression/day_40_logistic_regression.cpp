#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

/*
 * Repository Merge Eligibility Engine
 *
 * Technical case study:
 * A repository platform evaluates whether a Pull Request can merge into a
 * protected production branch. Logistic regression is used to estimate the
 * probability that a Pull Request will introduce a post-merge defect.
 *
 * The classification model uses:
 *   z = w.x + b
 *   p = sigmoid(z)
 *
 * The model itself is not the governance policy. The governance engine combines
 * the model's probability with explicit repository rules such as approvals,
 * status checks, branch protection, and review requirements.
 *
 * C++17 standard library only.
 */

enum class PullRequestState {
    Draft,
    Open,
    Closed,
    Merged
};

enum class ReviewState {
    Commented,
    ChangesRequested,
    Approved,
    Dismissed
};

struct Review {
    string reviewer;
    ReviewState state;
    bool eligible;
    string comment;
};

struct StatusCheck {
    string name;
    bool required;
    bool passed;
};

struct BranchProtection {
    string branchName;
    int requiredApprovals;
    bool requireStatusChecks;
    bool requireConversationResolution;
    bool restrictForcePush;
    bool restrictDeletion;
    bool requireLinearHistory;
};

struct PullRequest {
    int id;
    string sourceBranch;
    string targetBranch;
    PullRequestState state;
    int changedFiles;
    int additions;
    int deletions;
    int commits;
    bool conversationsResolved;
    bool branchSynchronized;
    bool mergeConflict;
    vector<Review> reviews;
    vector<StatusCheck> checks;
};

struct MergeDecision {
    bool eligible;
    vector<string> blockingReasons;
    double defectProbability;
};

class LogisticRiskModel {
private:
    vector<double> weights;
    double bias;

    static double sigmoid(double value) {
        if (value >= 0.0) {
            double negative = exp(-value);
            return 1.0 / (1.0 + negative);
        }

        double positive = exp(value);
        return positive / (1.0 + positive);
    }

public:
    LogisticRiskModel(vector<double> learnedWeights, double learnedBias)
        : weights(move(learnedWeights)), bias(learnedBias) {}

    double probability(const vector<double>& features) const {
        if (features.size() != weights.size()) {
            throw invalid_argument(
                "Risk-model feature dimension does not match trained model."
            );
        }

        double logit = bias;

        for (size_t i = 0; i < features.size(); ++i) {
            logit += weights[i] * features[i];
        }

        return sigmoid(logit);
    }

    double logit(const vector<double>& features) const {
        if (features.size() != weights.size()) {
            throw invalid_argument("Feature dimension mismatch.");
        }

        return inner_product(
            features.begin(),
            features.end(),
            weights.begin(),
            bias
        );
    }
};

class MergeEligibilityEngine {
private:
    BranchProtection protection;
    LogisticRiskModel riskModel;
    double riskThreshold;

    static int countValidApprovals(const PullRequest& pullRequest) {
        unordered_map<string, bool> approvedReviewers;

        for (const Review& review : pullRequest.reviews) {
            if (
                review.state == ReviewState::Approved &&
                review.eligible
            ) {
                approvedReviewers[review.reviewer] = true;
            }
        }

        return static_cast<int>(approvedReviewers.size());
    }

    static bool allRequiredChecksPassed(
        const PullRequest& pullRequest
    ) {
        for (const StatusCheck& check : pullRequest.checks) {
            if (check.required && !check.passed) {
                return false;
            }
        }

        return true;
    }

public:
    MergeEligibilityEngine(
        BranchProtection branchProtection,
        LogisticRiskModel model,
        double threshold
    )
        : protection(move(branchProtection)),
          riskModel(move(model)),
          riskThreshold(threshold) {
        if (threshold <= 0.0 || threshold >= 1.0) {
            throw invalid_argument(
                "Risk threshold must be between zero and one."
            );
        }
    }

    MergeDecision evaluate(const PullRequest& pullRequest) const {
        MergeDecision decision{
            true,
            {},
            0.0
        };

        /*
         * The risk model receives normalized operational indicators.
         * Larger change volume and more commits generally increase risk,
         * while synchronized branches and resolved conversations reduce it.
         */
        const double files = min(
            pullRequest.changedFiles / 100.0,
            3.0
        );

        const double additions = min(
            pullRequest.additions / 1000.0,
            3.0
        );

        const double deletions = min(
            pullRequest.deletions / 1000.0,
            3.0
        );

        const double commits = min(
            pullRequest.commits / 20.0,
            3.0
        );

        const double unsynchronized =
            pullRequest.branchSynchronized ? 0.0 : 1.0;

        const double unresolved =
            pullRequest.conversationsResolved ? 0.0 : 1.0;

        const vector<double> features{
            files,
            additions,
            deletions,
            commits,
            unsynchronized,
            unresolved
        };

        decision.defectProbability = riskModel.probability(features);

        if (pullRequest.state != PullRequestState::Open) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Pull Request is not open."
            );
        }

        if (pullRequest.targetBranch != protection.branchName) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Pull Request target does not match the protected branch."
            );
        }

        if (pullRequest.sourceBranch == pullRequest.targetBranch) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Source and target branches must be different."
            );
        }

        if (pullRequest.mergeConflict) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Merge conflicts must be resolved before merging."
            );
        }

        if (!pullRequest.branchSynchronized) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Source branch is behind the protected base branch."
            );
        }

        if (countValidApprovals(pullRequest) <
            protection.requiredApprovals) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Required eligible approvals are missing."
            );
        }

        if (
            protection.requireStatusChecks &&
            !allRequiredChecksPassed(pullRequest)
        ) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "At least one required status check has failed."
            );
        }

        if (
            protection.requireConversationResolution &&
            !pullRequest.conversationsResolved
        ) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Required review conversations are unresolved."
            );
        }

        if (decision.defectProbability >= riskThreshold) {
            decision.eligible = false;
            decision.blockingReasons.push_back(
                "Predicted defect risk exceeds the repository policy threshold."
            );
        }

        return decision;
    }
};

string reviewStateName(ReviewState state) {
    switch (state) {
        case ReviewState::Commented:
            return "commented";
        case ReviewState::ChangesRequested:
            return "changes requested";
        case ReviewState::Approved:
            return "approved";
        case ReviewState::Dismissed:
            return "dismissed";
    }

    return "unknown";
}

void printDecision(
    const PullRequest& pullRequest,
    const MergeDecision& decision
) {
    cout << "\nPull Request #" << pullRequest.id << '\n';
    cout << "Source: " << pullRequest.sourceBranch << '\n';
    cout << "Target: " << pullRequest.targetBranch << '\n';
    cout << "Predicted defect probability: "
         << fixed << setprecision(3)
         << decision.defectProbability << '\n';

    cout << "Merge eligibility: "
         << (decision.eligible ? "ELIGIBLE" : "BLOCKED")
         << '\n';

    for (const string& reason : decision.blockingReasons) {
        cout << "  Blocker: " << reason << '\n';
    }
}

PullRequest makeHealthyPullRequest() {
    PullRequest pullRequest{
        1042,
        "feature/payment-timeout",
        "main",
        PullRequestState::Open,
        14,
        260,
        80,
        5,
        true,
        true,
        false,
        {},
        {}
    };

    pullRequest.reviews = {
        {"reviewer-a", ReviewState::Approved, true,
         "Concurrency handling and rollback path reviewed."},
        {"reviewer-b", ReviewState::Approved, true,
         "Input validation and failure handling reviewed."},
        {"reviewer-c", ReviewState::Commented, true,
         "Minor naming observation."}
    };

    pullRequest.checks = {
        {"unit-tests", true, true},
        {"integration-tests", true, true},
        {"security-scan", true, true},
        {"documentation-check", false, false}
    };

    return pullRequest;
}

PullRequest makeBlockedPullRequest() {
    PullRequest pullRequest{
        1098,
        "feature/large-refactor",
        "main",
        PullRequestState::Open,
        180,
        3400,
        1900,
        34,
        false,
        false,
        true,
        {},
        {}
    };

    pullRequest.reviews = {
        {"reviewer-a", ReviewState::ChangesRequested, true,
         "The transaction boundary is unsafe."},
        {"external-user", ReviewState::Approved, false,
         "Reviewer is not eligible for protected branch approval."}
    };

    pullRequest.checks = {
        {"unit-tests", true, true},
        {"integration-tests", true, false},
        {"security-scan", true, true}
    };

    return pullRequest;
}

int main() {
    try {
        BranchProtection productionPolicy{
            "main",
            2,
            true,
            true,
            true,
            true,
            true
        };

        /*
         * Feature order:
         * change-file volume,
         * additions,
         * deletions,
         * commit volume,
         * unsynchronized state,
         * unresolved review conversations.
         *
         * The negative weights for clean workflow signals make the model's
         * probability interpretable: operationally risky Pull Requests should
         * move toward a larger positive logit.
         */
        LogisticRiskModel riskModel(
            {
                2.4,
                1.5,
                1.0,
                1.2,
                2.0,
                1.8
            },
            -3.0
        );

        MergeEligibilityEngine engine(
            productionPolicy,
            riskModel,
            0.70
        );

        PullRequest healthy = makeHealthyPullRequest();
        PullRequest blocked = makeBlockedPullRequest();

        printDecision(
            healthy,
            engine.evaluate(healthy)
        );

        printDecision(
            blocked,
            engine.evaluate(blocked)
        );

        cout << "\nReview state details:\n";

        for (const Review& review : blocked.reviews) {
            cout << review.reviewer
                 << " -> "
                 << reviewStateName(review.state)
                 << ", eligible="
                 << (review.eligible ? "yes" : "no")
                 << '\n';
        }

        cout << "\nDecision-boundary interpretation:\n";
        cout << "The logistic model separates lower-risk and higher-risk "
             << "Pull Requests at logit zero when the policy threshold is "
             << "0.5. This governance engine intentionally uses a higher "
             << "risk threshold of 0.70 because production merge decisions "
             << "are more conservative than ordinary binary classification.\n";

        return 0;
    }
    catch (const exception& error) {
        cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}
