#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Observation {
    std::string id;
    std::vector<double> features;
};

struct ReviewableCluster {
    int id;
    std::vector<double> centroid;
    std::vector<std::string> members;
};

struct StatusCheck {
    std::string name;
    bool passed;
};

struct MergePolicy {
    int requiredApprovals = 2;
    bool requireAllChecks = true;
    bool requireConversationResolution = true;
    bool allowDirectPush = false;
    bool allowForcePush = false;
    bool requireLinearHistory = true;
};

struct PullRequest {
    int number;
    std::string sourceBranch;
    std::string targetBranch;
    std::vector<std::string> changedFiles;
    int approvals = 0;
    bool conversationsResolved = false;
    bool hasConflicts = false;
    bool draft = false;
    std::vector<StatusCheck> checks;
};

class GovernanceEngine {
public:
    explicit GovernanceEngine(MergePolicy policy)
        : policy_(std::move(policy)) {}

    bool mergeEligible(const PullRequest& pr) const {
        if (pr.draft) {
            return false;
        }

        if (pr.sourceBranch.empty() || pr.targetBranch.empty()) {
            return false;
        }

        if (pr.sourceBranch == pr.targetBranch) {
            return false;
        }

        if (pr.changedFiles.empty()) {
            return false;
        }

        if (pr.hasConflicts) {
            return false;
        }

        if (pr.approvals < policy_.requiredApprovals) {
            return false;
        }

        if (policy_.requireConversationResolution &&
            !pr.conversationsResolved) {
            return false;
        }

        if (policy_.requireAllChecks) {
            for (const auto& check : pr.checks) {
                if (!check.passed) {
                    return false;
                }
            }
        }

        return true;
    }

    std::vector<std::string> blockingReasons(const PullRequest& pr) const {
        std::vector<std::string> reasons;

        if (pr.draft) reasons.push_back("Pull Request is still a draft.");
        if (pr.sourceBranch == pr.targetBranch)
            reasons.push_back("Source and target branches are identical.");
        if (pr.changedFiles.empty())
            reasons.push_back("Pull Request contains no changes.");
        if (pr.hasConflicts)
            reasons.push_back("Branch contains merge conflicts.");
        if (pr.approvals < policy_.requiredApprovals)
            reasons.push_back("Required approvals have not been reached.");
        if (policy_.requireConversationResolution &&
            !pr.conversationsResolved)
            reasons.push_back("Review conversations remain unresolved.");

        if (policy_.requireAllChecks) {
            for (const auto& check : pr.checks) {
                if (!check.passed) {
                    reasons.push_back("Status check failed: " + check.name);
                }
            }
        }

        return reasons;
    }

private:
    MergePolicy policy_;
};

double euclidean(const std::vector<double>& a,
                 const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors have different dimensions.");
    }

    double squared = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double delta = a[i] - b[i];
        squared += delta * delta;
    }

    return std::sqrt(squared);
}

void validateDataset(const std::vector<Observation>& observations) {
    if (observations.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    const std::size_t dimensions = observations.front().features.size();
    if (dimensions == 0) {
        throw std::invalid_argument("Observations need at least one feature.");
    }

    std::set<std::string> ids;

    for (const auto& observation : observations) {
        if (observation.id.empty()) {
            throw std::invalid_argument("Observation identifiers cannot be empty.");
        }

        if (!ids.insert(observation.id).second) {
            throw std::invalid_argument("Duplicate observation identifier.");
        }

        if (observation.features.size() != dimensions) {
            throw std::invalid_argument("Inconsistent feature dimensions.");
        }

        for (double value : observation.features) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Features must be finite.");
            }
        }
    }
}

std::vector<ReviewableCluster> kMeans(
    const std::vector<Observation>& observations,
    int k,
    int maxIterations = 100,
    double tolerance = 1e-7) {

    validateDataset(observations);

    if (k <= 0 || k > static_cast<int>(observations.size())) {
        throw std::invalid_argument("Invalid cluster count.");
    }

    const std::size_t dimensions = observations.front().features.size();

    std::vector<std::vector<double>> centroids;
    for (int i = 0; i < k; ++i) {
        centroids.push_back(observations[i].features);
    }

    std::vector<std::vector<std::string>> assignments(k);

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        assignments.assign(k, {});

        for (const auto& observation : observations) {
            double bestDistance = std::numeric_limits<double>::infinity();
            int bestCluster = 0;

            for (int cluster = 0; cluster < k; ++cluster) {
                const double currentDistance =
                    euclidean(observation.features, centroids[cluster]);

                if (currentDistance < bestDistance) {
                    bestDistance = currentDistance;
                    bestCluster = cluster;
                }
            }

            assignments[bestCluster].push_back(observation.id);
        }

        std::vector<std::vector<double>> nextCentroids(k, std::vector<double>(dimensions));

        for (int cluster = 0; cluster < k; ++cluster) {
            if (assignments[cluster].empty()) {
                nextCentroids[cluster] = centroids[cluster];
                continue;
            }

            for (const auto& id : assignments[cluster]) {
                auto it = std::find_if(
                    observations.begin(),
                    observations.end(),
                    [&](const Observation& observation) {
                        return observation.id == id;
                    });

                for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                    nextCentroids[cluster][dimension] +=
                        it->features[dimension];
                }
            }

            for (double& value : nextCentroids[cluster]) {
                value /= assignments[cluster].size();
            }
        }

        double movement = 0.0;
        for (int cluster = 0; cluster < k; ++cluster) {
            movement = std::max(
                movement,
                euclidean(centroids[cluster], nextCentroids[cluster])
            );
        }

        centroids = std::move(nextCentroids);

        if (movement <= tolerance) {
            break;
        }
    }

    std::vector<ReviewableCluster> result;
    for (int cluster = 0; cluster < k; ++cluster) {
        result.push_back({
            cluster,
            centroids[cluster],
            assignments[cluster]
        });
    }

    return result;
}

int main() {
    try {
        const std::vector<Observation> observations = {
            {"customer-01", {1.0, 1.2, 0.9}},
            {"customer-02", {1.1, 0.8, 1.0}},
            {"customer-03", {0.9, 1.0, 1.2}},
            {"customer-04", {7.8, 8.1, 7.5}},
            {"customer-05", {8.2, 7.9, 8.4}},
            {"customer-06", {7.5, 8.3, 8.0}},
            {"customer-07", {4.2, 4.0, 4.1}},
            {"customer-08", {4.5, 4.3, 4.2}},
            {"customer-09", {4.0, 4.4, 4.5}},
            {"customer-10", {25.0, 1.0, 0.5}}
        };

        const auto clusters = kMeans(observations, 3);

        std::cout << "UNSUPERVISED LEARNING CLUSTERS\n";
        std::cout << "=============================\n";

        for (const auto& cluster : clusters) {
            std::cout << "Cluster " << cluster.id << ": ";

            for (std::size_t i = 0; i < cluster.members.size(); ++i) {
                if (i != 0) std::cout << ", ";
                std::cout << cluster.members[i];
            }

            std::cout << "\nCentroid: ";
            for (double value : cluster.centroid) {
                std::cout << std::fixed << std::setprecision(3)
                          << value << " ";
            }
            std::cout << "\n\n";
        }

        MergePolicy policy;
        policy.requiredApprovals = 2;
        policy.requireAllChecks = true;
        policy.requireConversationResolution = true;
        policy.allowDirectPush = false;
        policy.allowForcePush = false;
        policy.requireLinearHistory = true;

        PullRequest pullRequest{
            1042,
            "feature/customer-segmentation",
            "main",
            {"clustering.cpp", "segmentation.sql"},
            2,
            true,
            false,
            false,
            {
                {"unit-tests", true},
                {"security-scan", true},
                {"data-quality", false}
            }
        };

        GovernanceEngine engine(policy);

        std::cout << "MERGE ELIGIBILITY\n";
        std::cout << "=================\n";

        if (engine.mergeEligible(pullRequest)) {
            std::cout << "Pull Request #" << pullRequest.number
                      << " is eligible for merge.\n";
        } else {
            std::cout << "Pull Request #" << pullRequest.number
                      << " is blocked:\n";

            for (const auto& reason : engine.blockingReasons(pullRequest)) {
                std::cout << " - " << reason << "\n";
            }
        }

        // Complexity of one k-means iteration is approximately O(n * k * d).
        // The policy engine itself is O(r + c), where r is review/check count
        // and c is the number of independent merge conditions.

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }
}
