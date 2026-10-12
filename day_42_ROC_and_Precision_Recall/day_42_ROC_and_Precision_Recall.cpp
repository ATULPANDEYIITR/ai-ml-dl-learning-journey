#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Prediction {
    int actual;
    double score;
};

struct Metrics {
    double threshold{};
    int tp{};
    int fp{};
    int tn{};
    int fn{};

    double precision() const {
        return tp + fp == 0 ? 0.0
                            : static_cast<double>(tp) / (tp + fp);
    }

    double recall() const {
        return tp + fn == 0 ? 0.0
                            : static_cast<double>(tp) / (tp + fn);
    }

    double specificity() const {
        return tn + fp == 0 ? 0.0
                            : static_cast<double>(tn) / (tn + fp);
    }

    double fpr() const {
        return 1.0 - specificity();
    }

    double f1() const {
        const double p = precision();
        const double r = recall();
        return p + r == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
    }
};

class MergeGovernanceModel {
public:
    explicit MergeGovernanceModel(
        std::string repository,
        std::string targetBranch,
        int requiredApprovals,
        double requiredRecall,
        double maximumFpr
    )
        : repository_(std::move(repository)),
          targetBranch_(std::move(targetBranch)),
          requiredApprovals_(requiredApprovals),
          requiredRecall_(requiredRecall),
          maximumFpr_(maximumFpr) {
        if (requiredApprovals_ < 0) {
            throw std::invalid_argument("Required approvals cannot be negative.");
        }
        if (requiredRecall_ < 0.0 || requiredRecall_ > 1.0 ||
            maximumFpr_ < 0.0 || maximumFpr_ > 1.0) {
            throw std::invalid_argument("Policy metrics must be in [0, 1].");
        }
    }

    void addPullRequest(int id, const std::string& branch, bool draft) {
        pullRequests_[id] = {
            id,
            branch,
            targetBranch_,
            draft,
            0,
            false,
            false,
            false
        };
    }

    void addReview(int pullRequestId, bool approval, bool changesRequested) {
        auto it = pullRequests_.find(pullRequestId);
        if (it == pullRequests_.end()) {
            throw std::out_of_range("Pull Request does not exist.");
        }

        auto& pr = it->second;

        if (changesRequested) {
            pr.changesRequested = true;
        }

        if (approval) {
            pr.approvals++;
        }
    }

    void updateStatusChecks(int pullRequestId, bool allPassing) {
        auto it = pullRequests_.find(pullRequestId);
        if (it == pullRequests_.end()) {
            throw std::out_of_range("Pull Request does not exist.");
        }
        it->second.statusChecksPassing = allPassing;
    }

    void synchronizeBranch(int pullRequestId) {
        auto it = pullRequests_.find(pullRequestId);
        if (it == pullRequests_.end()) {
            throw std::out_of_range("Pull Request does not exist.");
        }

        /*
         * Synchronization represents the governance event where the source
         * branch incorporates newer target-branch changes. A real platform
         * would also recalculate merge conflicts and potentially invalidate
         * stale approvals according to repository policy.
         */
        it->second.synchronized = true;
        it->second.approvals = 0;
        it->second.changesRequested = false;
    }

    bool mergeEligible(int pullRequestId) const {
        const auto it = pullRequests_.find(pullRequestId);
        if (it == pullRequests_.end()) {
            throw std::out_of_range("Pull Request does not exist.");
        }

        const auto& pr = it->second;

        return !pr.draft &&
               pr.approvals >= requiredApprovals_ &&
               !pr.changesRequested &&
               pr.statusChecksPassing &&
               pr.synchronized;
    }

    void printPullRequest(int pullRequestId) const {
        const auto it = pullRequests_.find(pullRequestId);
        if (it == pullRequests_.end()) {
            throw std::out_of_range("Pull Request does not exist.");
        }

        const auto& pr = it->second;

        std::cout << "PR #" << pr.id
                  << " " << pr.sourceBranch << " -> " << pr.targetBranch
                  << "\n  draft=" << std::boolalpha << pr.draft
                  << " approvals=" << pr.approvals
                  << " status_checks=" << pr.statusChecksPassing
                  << " synchronized=" << pr.synchronized
                  << " changes_requested=" << pr.changesRequested
                  << "\n  merge_eligible=" << mergeEligible(pr.id)
                  << "\n";
    }

    static double trapezoidalAuc(
        std::vector<std::pair<double, double>> points
    ) {
        if (points.size() < 2) {
            return 0.0;
        }

        std::sort(points.begin(), points.end());

        double area = 0.0;
        for (std::size_t i = 1; i < points.size(); ++i) {
            const auto [x1, y1] = points[i - 1];
            const auto [x2, y2] = points[i];

            area += (x2 - x1) * (y1 + y2) / 2.0;
        }

        return area;
    }

    /*
     * The same governance system contains a review-risk classifier. A high
     * score means "likely to require additional review". ROC/PR analysis is
     * used to choose a threshold before automating a review-routing action.
     */
    static std::vector<double> thresholds(
        const std::vector<Prediction>& predictions
    ) {
        std::vector<double> result{0.0, 1.0};

        for (const auto& prediction : predictions) {
            result.push_back(prediction.score);
        }

        std::sort(result.begin(), result.end());
        result.erase(
            std::unique(result.begin(), result.end()),
            result.end()
        );

        return result;
    }

    static Metrics evaluate(
        const std::vector<Prediction>& predictions,
        double threshold
    ) {
        Metrics result;
        result.threshold = threshold;

        for (const auto& prediction : predictions) {
            const bool predictedPositive = prediction.score >= threshold;

            if (prediction.actual == 1 && predictedPositive) {
                ++result.tp;
            } else if (prediction.actual == 0 && predictedPositive) {
                ++result.fp;
            } else if (prediction.actual == 0 && !predictedPositive) {
                ++result.tn;
            } else {
                ++result.fn;
            }
        }

        return result;
    }

    static double rocAuc(
        const std::vector<Prediction>& predictions
    ) {
        std::vector<std::pair<double, double>> points;

        for (double threshold : thresholds(predictions)) {
            const Metrics m = evaluate(predictions, threshold);
            points.emplace_back(m.fpr(), m.recall());
        }

        return trapezoidalAuc(std::move(points));
    }

    static double averagePrecision(
        const std::vector<Prediction>& predictions
    ) {
        std::vector<Prediction> ranked = predictions;

        std::sort(
            ranked.begin(),
            ranked.end(),
            [](const Prediction& a, const Prediction& b) {
                return a.score > b.score;
            }
        );

        int totalPositive = 0;
        for (const auto& prediction : ranked) {
            totalPositive += prediction.actual;
        }

        if (totalPositive == 0) {
            return 0.0;
        }

        int tp = 0;
        int fp = 0;
        double previousRecall = 0.0;
        double result = 0.0;

        for (const auto& prediction : ranked) {
            if (prediction.actual == 1) {
                ++tp;
            } else {
                ++fp;
            }

            const double precision =
                static_cast<double>(tp) / (tp + fp);
            const double recall =
                static_cast<double>(tp) / totalPositive;

            if (prediction.actual == 1) {
                result += precision * (recall - previousRecall);
                previousRecall = recall;
            }
        }

        return result;
    }

    static Metrics optimizeForCost(
        const std::vector<Prediction>& predictions,
        double falsePositiveCost,
        double falseNegativeCost
    ) {
        if (falsePositiveCost < 0 || falseNegativeCost < 0) {
            throw std::invalid_argument("Error costs cannot be negative.");
        }

        double bestCost = std::numeric_limits<double>::infinity();
        Metrics best{};

        for (double threshold : thresholds(predictions)) {
            const Metrics m = evaluate(predictions, threshold);
            const double cost =
                m.fp * falsePositiveCost +
                m.fn * falseNegativeCost;

            if (cost < bestCost) {
                bestCost = cost;
                best = m;
            }
        }

        return best;
    }

private:
    struct PullRequest {
        int id;
        std::string sourceBranch;
        std::string targetBranch;
        bool draft;
        int approvals;
        bool changesRequested;
        bool statusChecksPassing;
        bool synchronized;
    };

    std::string repository_;
    std::string targetBranch_;
    int requiredApprovals_;
    double requiredRecall_;
    double maximumFpr_;
    std::map<int, PullRequest> pullRequests_;
};

int main() {
    try {
        MergeGovernanceModel governance(
            "payments-api",
            "main",
            2,
            0.90,
            0.10
        );

        governance.addPullRequest(417, "feature/risk-scoring", false);
        governance.updateStatusChecks(417, true);

        governance.addReview(417, true, false);
        governance.printPullRequest(417);

        std::cout << "\nSynchronizing source branch...\n";
        governance.synchronizeBranch(417);
        governance.updateStatusChecks(417, true);
        governance.addReview(417, true, false);
        governance.printPullRequest(417);

        /*
         * A second review-routing model is intentionally separate from merge
         * authorization. Its purpose is to predict whether a change should be
         * routed for additional human review. This distinction prevents an
         * ML threshold from silently becoming a repository governance rule.
         */
        const std::vector<Prediction> reviewRisk = {
            {0, 0.04}, {0, 0.08}, {0, 0.12}, {0, 0.18},
            {0, 0.24}, {0, 0.31}, {0, 0.39}, {0, 0.43},
            {0, 0.51}, {0, 0.57}, {1, 0.36}, {1, 0.55},
            {1, 0.62}, {1, 0.69}, {1, 0.74}, {1, 0.88}
        };

        std::cout << "\nReview-risk classifier\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "ROC AUC: "
                  << MergeGovernanceModel::rocAuc(reviewRisk)
                  << "\n";
        std::cout << "Average Precision: "
                  << MergeGovernanceModel::averagePrecision(reviewRisk)
                  << "\n";

        const Metrics costOptimal =
            MergeGovernanceModel::optimizeForCost(
                reviewRisk,
                1.0,
                4.0
            );

        std::cout << "Cost-sensitive threshold: "
                  << costOptimal.threshold
                  << " precision=" << costOptimal.precision()
                  << " recall=" << costOptimal.recall()
                  << " FPR=" << costOptimal.fpr()
                  << "\n";

        /*
         * This is a policy gate rather than a statistical claim. The chosen
         * threshold must be evaluated against a separate validation/test
         * population before deployment. ROC AUC alone cannot establish that
         * a threshold is safe for repository governance.
         */
        const bool satisfiesRoutingPolicy =
            costOptimal.recall() >= 0.90 &&
            costOptimal.fpr() <= 0.10;

        std::cout << "Routing policy satisfied: "
                  << std::boolalpha
                  << satisfiesRoutingPolicy
                  << "\n";

        std::cout << "\nTechnical interpretation\n";
        std::cout
            << "ROC AUC evaluates ranking discrimination across thresholds. "
            << "Average Precision focuses on the quality of positive retrieval "
            << "and is more sensitive to class imbalance. Threshold selection "
            << "must be separated from AUC measurement because AUC contains "
            << "no single operating threshold.\n";

        std::cout
            << "The governance model keeps merge authorization deterministic: "
            << "reviews, approvals, synchronization, and status checks are "
            << "repository policy conditions rather than predictions.\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
