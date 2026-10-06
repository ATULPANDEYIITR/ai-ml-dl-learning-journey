#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class LinearRegressionEngine {
public:
    struct Model {
        double intercept{};
        Vector coefficients;
        Vector predictions;
        Vector residuals;
        double rSquared{};
        double adjustedRSquared{};
        double rmse{};
        double mae{};
    };

private:
    static constexpr double EPSILON = 1e-12;

    static void validateMatrix(const Matrix& matrix) {
        if (matrix.empty()) {
            throw std::invalid_argument("Design matrix cannot be empty.");
        }

        const std::size_t width = matrix.front().size();

        if (width == 0) {
            throw std::invalid_argument(
                "Design matrix must contain at least one column."
            );
        }

        for (const auto& row : matrix) {
            if (row.size() != width) {
                throw std::invalid_argument(
                    "Matrix rows must have equal length."
                );
            }

            for (double value : row) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "Matrix contains a non-finite value."
                    );
                }
            }
        }
    }

    static Matrix transpose(const Matrix& matrix) {
        validateMatrix(matrix);

        Matrix result(
            matrix.front().size(),
            Vector(matrix.size())
        );

        for (std::size_t row = 0; row < matrix.size(); ++row) {
            for (std::size_t column = 0;
                 column < matrix.front().size();
                 ++column) {
                result[column][row] = matrix[row][column];
            }
        }

        return result;
    }

    static Matrix multiply(
        const Matrix& a,
        const Matrix& b
    ) {
        validateMatrix(a);
        validateMatrix(b);

        if (a.front().size() != b.size()) {
            throw std::invalid_argument(
                "Matrix dimensions are incompatible."
            );
        }

        Matrix result(
            a.size(),
            Vector(b.front().size(), 0.0)
        );

        for (std::size_t i = 0; i < a.size(); ++i) {
            for (std::size_t k = 0; k < b.size(); ++k) {
                for (std::size_t j = 0;
                     j < b.front().size();
                     ++j) {
                    result[i][j] +=
                        a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

    static Matrix identity(std::size_t size) {
        Matrix result(
            size,
            Vector(size, 0.0)
        );

        for (std::size_t i = 0; i < size; ++i) {
            result[i][i] = 1.0;
        }

        return result;
    }

    static Matrix inverse(Matrix matrix) {
        validateMatrix(matrix);

        const std::size_t n = matrix.size();

        if (matrix.front().size() != n) {
            throw std::invalid_argument(
                "Only square matrices can be inverted."
            );
        }

        Matrix augmented(
            n,
            Vector(2 * n, 0.0)
        );

        Matrix identityMatrix = identity(n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] =
                    identityMatrix[i][j];
            }
        }

        // Partial pivoting reduces numerical error when the natural
        // diagonal pivot is small relative to another candidate.
        for (std::size_t column = 0;
             column < n;
             ++column) {

            std::size_t pivotRow = column;

            for (std::size_t row = column + 1;
                 row < n;
                 ++row) {
                if (
                    std::abs(augmented[row][column]) >
                    std::abs(augmented[pivotRow][column])
                ) {
                    pivotRow = row;
                }
            }

            if (
                std::abs(augmented[pivotRow][column]) <
                EPSILON
            ) {
                throw std::runtime_error(
                    "Matrix is singular or nearly singular."
                );
            }

            if (pivotRow != column) {
                std::swap(
                    augmented[pivotRow],
                    augmented[column]
                );
            }

            const double pivot =
                augmented[column][column];

            for (double& value : augmented[column]) {
                value /= pivot;
            }

            for (std::size_t row = 0;
                 row < n;
                 ++row) {
                if (row == column) {
                    continue;
                }

                const double factor =
                    augmented[row][column];

                for (std::size_t j = 0;
                     j < 2 * n;
                     ++j) {
                    augmented[row][j] -=
                        factor * augmented[column][j];
                }
            }
        }

        Matrix result(
            n,
            Vector(n)
        );

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                result[i][j] =
                    augmented[i][j + n];
            }
        }

        return result;
    }

    static Vector columnToVector(
        const Matrix& matrix
    ) {
        Vector result(matrix.size());

        for (std::size_t i = 0;
             i < matrix.size();
             ++i) {
            if (matrix[i].size() != 1) {
                throw std::invalid_argument(
                    "Expected a single-column matrix."
                );
            }

            result[i] = matrix[i][0];
        }

        return result;
    }

    static double dot(
        const Vector& a,
        const Vector& b
    ) {
        if (a.size() != b.size()) {
            throw std::invalid_argument(
                "Vector dimensions do not match."
            );
        }

        return std::inner_product(
            a.begin(),
            a.end(),
            b.begin(),
            0.0
        );
    }

public:
    /*
     * Repository governance case study:
     *
     * Each pull request proposes a changeset against a protected branch.
     * The governance engine evaluates:
     *   - whether the base branch is protected
     *   - required status checks
     *   - required reviewer count
     *   - approval decisions
     *   - unresolved review discussions
     *   - direct-push restrictions
     *
     * The regression component is used by the same analytical system to
     * estimate expected review latency from historical repository activity.
     */
    static Model fit(
        const Matrix& features,
        const Vector& target
    ) {
        if (features.empty()) {
            throw std::invalid_argument(
                "Regression data cannot be empty."
            );
        }

        if (features.size() != target.size()) {
            throw std::invalid_argument(
                "Feature rows and target observations must match."
            );
        }

        const std::size_t predictorCount =
            features.front().size();

        if (predictorCount == 0) {
            throw std::invalid_argument(
                "At least one predictor is required."
            );
        }

        if (features.size() <= predictorCount) {
            throw std::invalid_argument(
                "Insufficient observations for the requested model."
            );
        }

        for (const auto& row : features) {
            if (row.size() != predictorCount) {
                throw std::invalid_argument(
                    "Feature rows must have equal lengths."
                );
            }

            for (double value : row) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "Feature data must be finite."
                    );
                }
            }
        }

        for (double value : target) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Target data must be finite."
                );
            }
        }

        Matrix design(
            features.size(),
            Vector(predictorCount + 1)
        );

        // A leading column of ones gives the model an intercept.
        for (std::size_t i = 0;
             i < features.size();
             ++i) {
            design[i][0] = 1.0;

            for (std::size_t j = 0;
                 j < predictorCount;
                 ++j) {
                design[i][j + 1] =
                    features[i][j];
            }
        }

        Matrix xt = transpose(design);
        Matrix xtx = multiply(xt, design);
        Matrix xtxInverse = inverse(xtx);

        Matrix targetColumn(
            target.size(),
            Vector(1)
        );

        for (std::size_t i = 0;
             i < target.size();
             ++i) {
            targetColumn[i][0] = target[i];
        }

        Matrix xty = multiply(
            xt,
            targetColumn
        );

        Matrix betaMatrix = multiply(
            xtxInverse,
            xty
        );

        Vector beta = columnToVector(betaMatrix);

        Model model;
        model.intercept = beta[0];
        model.coefficients.assign(
            beta.begin() + 1,
            beta.end()
        );

        model.predictions.reserve(features.size());
        model.residuals.reserve(features.size());

        for (std::size_t i = 0;
             i < features.size();
             ++i) {
            double prediction =
                model.intercept +
                dot(model.coefficients, features[i]);

            model.predictions.push_back(prediction);
            model.residuals.push_back(
                target[i] - prediction
            );
        }

        const double targetMean =
            std::accumulate(
                target.begin(),
                target.end(),
                0.0
            ) / target.size();

        double sse = 0.0;
        double sst = 0.0;
        double absoluteError = 0.0;

        for (std::size_t i = 0;
             i < target.size();
             ++i) {
            const double residual =
                model.residuals[i];

            sse += residual * residual;
            absoluteError += std::abs(residual);

            const double centered =
                target[i] - targetMean;

            sst += centered * centered;
        }

        model.rSquared =
            sst < EPSILON
                ? (sse < EPSILON ? 1.0 : 0.0)
                : 1.0 - sse / sst;

        const double n =
            static_cast<double>(target.size());

        const double p =
            static_cast<double>(predictorCount);

        if (n > p + 1.0) {
            model.adjustedRSquared =
                1.0 -
                (1.0 - model.rSquared) *
                ((n - 1.0) / (n - p - 1.0));
        } else {
            model.adjustedRSquared =
                std::numeric_limits<double>::quiet_NaN();
        }

        model.rmse =
            std::sqrt(sse / n);

        model.mae =
            absoluteError / n;

        return model;
    }
};

enum class PullRequestState {
    Draft,
    Open,
    Approved,
    ChangesRequested,
    Merged,
    Closed
};

enum class ReviewDecision {
    Commented,
    Approved,
    ChangesRequested
};

struct StatusCheck {
    std::string name;
    bool required;
    bool passed;
};

struct Review {
    std::string reviewer;
    ReviewDecision decision;
    bool dismissed;
    int unresolvedComments;
};

struct PullRequest {
    int id;
    std::string sourceBranch;
    std::string targetBranch;
    PullRequestState state;
    std::vector<std::string> commits;
    std::vector<Review> reviews;
    std::vector<StatusCheck> checks;
    bool hasMergeConflict;
    bool branchUpToDate;
};

struct BranchProtection {
    bool protectedBranch;
    bool restrictDirectPush;
    bool requirePullRequest;
    bool requireUpToDateBranch;
    bool requireLinearHistory;
    bool allowForcePush;
    bool allowDeletion;
    int requiredApprovals;
    std::vector<std::string> requiredChecks;
};

class MergeEligibilityEngine {
public:
    static bool isEligible(
        const PullRequest& pullRequest,
        const BranchProtection& protection,
        std::string& reason
    ) {
        if (!protection.protectedBranch) {
            reason =
                "Target branch is not protected.";
            return true;
        }

        if (
            pullRequest.state == PullRequestState::Draft
        ) {
            reason =
                "Draft pull requests are not merge-ready.";
            return false;
        }

        if (
            pullRequest.state == PullRequestState::Closed ||
            pullRequest.state == PullRequestState::Merged
        ) {
            reason =
                "Pull request is no longer open for merging.";
            return false;
        }

        if (
            pullRequest.sourceBranch ==
            pullRequest.targetBranch
        ) {
            reason =
                "Source and target branches must differ.";
            return false;
        }

        if (
            protection.requirePullRequest &&
            pullRequest.id <= 0
        ) {
            reason =
                "A valid pull request is required.";
            return false;
        }

        if (
            protection.requireUpToDateBranch &&
            !pullRequest.branchUpToDate
        ) {
            reason =
                "Source branch must be synchronized with the target branch.";
            return false;
        }

        if (pullRequest.hasMergeConflict) {
            reason =
                "Merge conflicts must be resolved.";
            return false;
        }

        for (
            const std::string& required :
            protection.requiredChecks
        ) {
            auto iterator =
                std::find_if(
                    pullRequest.checks.begin(),
                    pullRequest.checks.end(),
                    [&](const StatusCheck& check) {
                        return check.name == required;
                    }
                );

            if (
                iterator == pullRequest.checks.end() ||
                !iterator->passed
            ) {
                reason =
                    "Required status check failed or is missing: " +
                    required;
                return false;
            }
        }

        int validApprovals = 0;

        for (const Review& review :
             pullRequest.reviews) {

            if (
                review.decision ==
                    ReviewDecision::Approved &&
                !review.dismissed
            ) {
                ++validApprovals;
            }

            if (
                review.decision ==
                    ReviewDecision::ChangesRequested &&
                !review.dismissed
            ) {
                reason =
                    "An active reviewer has requested changes.";
                return false;
            }

            if (review.unresolvedComments > 0) {
                reason =
                    "Unresolved review discussions remain.";
                return false;
            }
        }

        if (
            validApprovals <
            protection.requiredApprovals
        ) {
            reason =
                "The required number of active approvals has not been reached.";
            return false;
        }

        reason =
            "All configured merge requirements are satisfied.";
        return true;
    }
};

void printRegressionModel(
    const LinearRegressionEngine::Model& model,
    const std::vector<std::string>& names
) {
    std::cout << std::fixed
              << std::setprecision(4);

    std::cout << "\nRegression equation:\n";
    std::cout << "review_hours = "
              << model.intercept;

    for (std::size_t i = 0;
         i < model.coefficients.size();
         ++i) {
        std::cout
            << (model.coefficients[i] >= 0
                    ? " + "
                    : " - ")
            << std::abs(model.coefficients[i])
            << "*" << names[i];
    }

    std::cout << "\n\nCoefficients:\n";
    for (std::size_t i = 0;
         i < names.size();
         ++i) {
        std::cout
            << "  "
            << names[i]
            << ": "
            << model.coefficients[i]
            << '\n';
    }

    std::cout
        << "  intercept: "
        << model.intercept
        << '\n';

    std::cout
        << "\nR-squared: "
        << model.rSquared
        << "\nAdjusted R-squared: "
        << model.adjustedRSquared
        << "\nRMSE: "
        << model.rmse
        << "\nMAE: "
        << model.mae
        << '\n';
}

void runRepositoryGovernanceCaseStudy() {
    /*
     * Historical observations:
     *
     * features:
     *   changed_files
     *   reviewer_count
     *   unresolved_discussions_at_start
     *
     * target:
     *   review completion time in hours
     *
     * The predictors are intentionally repository-review specific.
     */
    Matrix features = {
        {4, 1, 0},
        {7, 2, 1},
        {10, 2, 1},
        {14, 3, 2},
        {18, 3, 2},
        {22, 4, 3},
        {27, 4, 4},
        {31, 5, 4},
        {36, 5, 5},
        {42, 6, 6},
        {47, 6, 7},
        {52, 7, 8},
        {58, 7, 8},
        {64, 8, 9}
    };

    Vector reviewHours = {
        2.5, 4.2, 5.1, 6.8,
        8.0, 10.1, 12.2, 13.8,
        16.0, 18.9, 20.7, 23.0,
        25.4, 27.1
    };

    auto model =
        LinearRegressionEngine::fit(
            features,
            reviewHours
        );

    printRegressionModel(
        model,
        {
            "changed_files",
            "reviewer_count",
            "initial_unresolved_discussions"
        }
    );

    PullRequest pullRequest{
        1042,
        "feature/payment-refactor",
        "main",
        PullRequestState::Open,
        {"a81c2d", "b71e33", "c92f44"},
        {
            {
                "reviewer-alice",
                ReviewDecision::Approved,
                false,
                0
            },
            {
                "reviewer-bob",
                ReviewDecision::Approved,
                false,
                0
            }
        },
        {
            {"unit-tests", true, true},
            {"security-scan", true, true},
            {"build", true, true}
        },
        false,
        true
    };

    BranchProtection protection{
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        2,
        {"unit-tests", "security-scan", "build"}
    };

    std::string reason;

    const bool eligible =
        MergeEligibilityEngine::isEligible(
            pullRequest,
            protection,
            reason
        );

    std::cout
        << "\nRepository governance decision:\n"
        << "  Pull Request #"
        << pullRequest.id
        << '\n'
        << "  Source: "
        << pullRequest.sourceBranch
        << '\n'
        << "  Target: "
        << pullRequest.targetBranch
        << '\n'
        << "  Merge eligible: "
        << (eligible ? "YES" : "NO")
        << '\n'
        << "  Reason: "
        << reason
        << '\n';

    std::cout
        << "\nPredicted review duration for the current "
        << "repository change profile: ";

    Vector currentFeatures = {
        42,
        4,
        3
    };

    double predicted =
        model.intercept;

    for (std::size_t i = 0;
         i < model.coefficients.size();
         ++i) {
        predicted +=
            model.coefficients[i] *
            currentFeatures[i];
    }

    std::cout
        << predicted
        << " hours\n";
}

int main() {
    try {
        std::cout
            << "LINEAR REGRESSION: REPOSITORY GOVERNANCE CASE STUDY\n"
            << "====================================================\n";

        runRepositoryGovernanceCaseStudy();

        std::cout
            << "\nThe model estimates review latency; it does not "
            << "replace repository governance rules. "
            << "Merge eligibility remains a deterministic policy decision.\n";
    }
    catch (const std::exception& error) {
        std::cerr
            << "Execution failed: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}
