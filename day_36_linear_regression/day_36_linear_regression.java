import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Enterprise-oriented linear regression model.
 *
 * The domain scenario estimates Pull Request review duration from
 * repository-development characteristics. The regression engine and the
 * governance model are deliberately separate: regression estimates a
 * continuous quantity, while governance rules determine whether a
 * Pull Request is eligible for merge.
 *
 * Java 17+.
 */
public class LinearRegressionEnterprise {

    private static final double EPSILON = 1e-12;

    enum PullRequestState {
        DRAFT,
        OPEN,
        CHANGES_REQUESTED,
        APPROVED,
        MERGED,
        CLOSED
    }

    enum ReviewState {
        COMMENTED,
        APPROVED,
        CHANGES_REQUESTED,
        DISMISSED
    }

    record Reviewer(String username, boolean eligible) {
        Reviewer {
            if (username == null || username.isBlank()) {
                throw new IllegalArgumentException(
                    "Reviewer username is required."
                );
            }
        }
    }

    record Review(
        Reviewer reviewer,
        ReviewState state,
        int unresolvedComments
    ) {
        Review {
            Objects.requireNonNull(reviewer);

            if (unresolvedComments < 0) {
                throw new IllegalArgumentException(
                    "Unresolved comments cannot be negative."
                );
            }
        }
    }

    record StatusCheck(
        String name,
        boolean required,
        boolean passed
    ) {
        StatusCheck {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                    "Status check name is required."
                );
            }
        }
    }

    static final class PullRequest {
        private final int number;
        private final String sourceBranch;
        private final String targetBranch;
        private PullRequestState state;
        private final List<String> commits;
        private final List<Review> reviews;
        private final List<StatusCheck> statusChecks;
        private boolean mergeConflict;
        private boolean synchronizedWithBase;

        PullRequest(
            int number,
            String sourceBranch,
            String targetBranch,
            List<String> commits
        ) {
            if (number <= 0) {
                throw new IllegalArgumentException(
                    "Pull Request number must be positive."
                );
            }

            if (
                sourceBranch == null ||
                targetBranch == null ||
                sourceBranch.isBlank() ||
                targetBranch.isBlank()
            ) {
                throw new IllegalArgumentException(
                    "Both source and target branches are required."
                );
            }

            if (sourceBranch.equals(targetBranch)) {
                throw new IllegalArgumentException(
                    "Source and target branches must differ."
                );
            }

            if (commits == null || commits.isEmpty()) {
                throw new IllegalArgumentException(
                    "A Pull Request must contain at least one commit."
                );
            }

            this.number = number;
            this.sourceBranch = sourceBranch;
            this.targetBranch = targetBranch;
            this.commits = List.copyOf(commits);
            this.reviews = new ArrayList<>();
            this.statusChecks = new ArrayList<>();
            this.state = PullRequestState.OPEN;
        }

        int number() {
            return number;
        }

        String sourceBranch() {
            return sourceBranch;
        }

        String targetBranch() {
            return targetBranch;
        }

        PullRequestState state() {
            return state;
        }

        List<Review> reviews() {
            return Collections.unmodifiableList(reviews);
        }

        List<StatusCheck> statusChecks() {
            return Collections.unmodifiableList(statusChecks);
        }

        boolean mergeConflict() {
            return mergeConflict;
        }

        boolean synchronizedWithBase() {
            return synchronizedWithBase;
        }

        void addReview(Review review) {
            Objects.requireNonNull(review);
            reviews.add(review);

            if (review.state() == ReviewState.CHANGES_REQUESTED) {
                state = PullRequestState.CHANGES_REQUESTED;
            }
        }

        void addStatusCheck(StatusCheck check) {
            statusChecks.add(Objects.requireNonNull(check));
        }

        void setMergeConflict(boolean mergeConflict) {
            this.mergeConflict = mergeConflict;
        }

        void synchronizeWithBase() {
            this.synchronizedWithBase = true;
        }

        void markDraft() {
            if (state != PullRequestState.OPEN) {
                throw new IllegalStateException(
                    "Only an open Pull Request can become a draft."
                );
            }

            state = PullRequestState.DRAFT;
        }

        void reopen() {
            if (
                state != PullRequestState.CLOSED &&
                state != PullRequestState.CHANGES_REQUESTED
            ) {
                throw new IllegalStateException(
                    "The current state does not require reopening."
                );
            }

            state = PullRequestState.OPEN;
        }

        void merge() {
            if (
                state == PullRequestState.MERGED ||
                state == PullRequestState.CLOSED
            ) {
                throw new IllegalStateException(
                    "Pull Request cannot be merged from its current state."
                );
            }

            state = PullRequestState.MERGED;
        }
    }

    record BranchProtectionPolicy(
        String branchName,
        boolean requirePullRequest,
        boolean requireUpToDateBranch,
        boolean requireConversationResolution,
        boolean restrictDirectPushes,
        boolean preventForcePush,
        boolean preventDeletion,
        boolean requireLinearHistory,
        int requiredApprovals,
        List<String> requiredChecks
    ) {
        BranchProtectionPolicy {
            if (branchName == null || branchName.isBlank()) {
                throw new IllegalArgumentException(
                    "Protected branch name is required."
                );
            }

            if (requiredApprovals < 0) {
                throw new IllegalArgumentException(
                    "Required approvals cannot be negative."
                );
            }

            requiredChecks = List.copyOf(requiredChecks);
        }
    }

    record MergeDecision(
        boolean eligible,
        List<String> blockers
    ) {}

    static final class GovernanceService {

        MergeDecision evaluate(
            PullRequest pullRequest,
            BranchProtectionPolicy policy
        ) {
            List<String> blockers = new ArrayList<>();

            if (policy.requirePullRequest()) {
                if (pullRequest.number() <= 0) {
                    blockers.add(
                        "A valid Pull Request is required."
                    );
                }
            }

            if (pullRequest.state() == PullRequestState.DRAFT) {
                blockers.add(
                    "Draft Pull Requests are not merge-ready."
                );
            }

            if (
                pullRequest.state() == PullRequestState.CLOSED ||
                pullRequest.state() == PullRequestState.MERGED
            ) {
                blockers.add(
                    "Pull Request is already closed or merged."
                );
            }

            if (pullRequest.mergeConflict()) {
                blockers.add(
                    "Merge conflicts must be resolved."
                );
            }

            if (
                policy.requireUpToDateBranch() &&
                !pullRequest.synchronizedWithBase()
            ) {
                blockers.add(
                    "Source branch is not synchronized with the protected branch."
                );
            }

            int activeApprovals = 0;

            for (Review review : pullRequest.reviews()) {
                if (
                    review.state() == ReviewState.APPROVED &&
                    review.reviewer().eligible()
                ) {
                    activeApprovals++;
                }

                if (
                    review.state() ==
                    ReviewState.CHANGES_REQUESTED
                ) {
                    blockers.add(
                        "An eligible reviewer has requested changes."
                    );
                }

                if (
                    policy.requireConversationResolution() &&
                    review.unresolvedComments() > 0
                ) {
                    blockers.add(
                        "Unresolved review discussions remain."
                    );
                }
            }

            if (
                activeApprovals <
                policy.requiredApprovals()
            ) {
                blockers.add(
                    "Required approval count has not been reached."
                );
            }

            Map<String, StatusCheck> checks =
                new HashMap<>();

            for (StatusCheck check :
                 pullRequest.statusChecks()) {
                checks.put(check.name(), check);
            }

            for (String required :
                 policy.requiredChecks()) {
                StatusCheck check =
                    checks.get(required);

                if (
                    check == null ||
                    !check.passed()
                ) {
                    blockers.add(
                        "Required status check failed or is missing: " +
                        required
                    );
                }
            }

            return new MergeDecision(
                blockers.isEmpty(),
                List.copyOf(blockers)
            );
        }
    }

    static final class Matrix {

        private Matrix() {}

        static double[][] transpose(double[][] matrix) {
            validateRectangular(matrix);

            int rows = matrix.length;
            int columns = matrix[0].length;

            double[][] result =
                new double[columns][rows];

            for (int row = 0; row < rows; row++) {
                for (int column = 0;
                     column < columns;
                     column++) {
                    result[column][row] =
                        matrix[row][column];
                }
            }

            return result;
        }

        static double[][] multiply(
            double[][] a,
            double[][] b
        ) {
            validateRectangular(a);
            validateRectangular(b);

            if (a[0].length != b.length) {
                throw new IllegalArgumentException(
                    "Matrix dimensions are incompatible."
                );
            }

            double[][] result =
                new double[a.length][b[0].length];

            for (int row = 0;
                 row < a.length;
                 row++) {
                for (int k = 0;
                     k < b.length;
                     k++) {
                    for (int column = 0;
                         column < b[0].length;
                         column++) {
                        result[row][column] +=
                            a[row][k] *
                            b[k][column];
                    }
                }
            }

            return result;
        }

        static double[][] inverse(double[][] matrix) {
            validateRectangular(matrix);

            int n = matrix.length;

            if (matrix[0].length != n) {
                throw new IllegalArgumentException(
                    "Only square matrices can be inverted."
                );
            }

            double[][] augmented =
                new double[n][2 * n];

            for (int row = 0; row < n; row++) {
                for (int column = 0;
                     column < n;
                     column++) {
                    augmented[row][column] =
                        matrix[row][column];

                    augmented[row][column + n] =
                        row == column ? 1.0 : 0.0;
                }
            }

            for (int column = 0;
                 column < n;
                 column++) {

                int pivotRow = column;

                for (int row = column + 1;
                     row < n;
                     row++) {
                    if (
                        Math.abs(
                            augmented[row][column]
                        ) >
                        Math.abs(
                            augmented[pivotRow][column]
                        )
                    ) {
                        pivotRow = row;
                    }
                }

                if (
                    Math.abs(
                        augmented[pivotRow][column]
                    ) < EPSILON
                ) {
                    throw new IllegalArgumentException(
                        "Matrix is singular or nearly singular."
                    );
                }

                double[] temporary =
                    augmented[column];

                augmented[column] =
                    augmented[pivotRow];

                augmented[pivotRow] =
                    temporary;

                double pivot =
                    augmented[column][column];

                for (int j = 0;
                     j < 2 * n;
                     j++) {
                    augmented[column][j] /= pivot;
                }

                for (int row = 0;
                     row < n;
                     row++) {
                    if (row == column) {
                        continue;
                    }

                    double factor =
                        augmented[row][column];

                    for (int j = 0;
                         j < 2 * n;
                         j++) {
                        augmented[row][j] -=
                            factor *
                            augmented[column][j];
                    }
                }
            }

            double[][] result =
                new double[n][n];

            for (int row = 0;
                 row < n;
                 row++) {
                System.arraycopy(
                    augmented[row],
                    n,
                    result[row],
                    0,
                    n
                );
            }

            return result;
        }

        private static void validateRectangular(
            double[][] matrix
        ) {
            if (
                matrix == null ||
                matrix.length == 0 ||
                matrix[0].length == 0
            ) {
                throw new IllegalArgumentException(
                    "Matrix cannot be empty."
                );
            }

            int width = matrix[0].length;

            for (double[] row : matrix) {
                if (row.length != width) {
                    throw new IllegalArgumentException(
                        "Matrix must be rectangular."
                    );
                }

                for (double value : row) {
                    if (!Double.isFinite(value)) {
                        throw new IllegalArgumentException(
                            "Matrix contains a non-finite value."
                        );
                    }
                }
            }
        }
    }

    static final class RegressionModel {

        private final double intercept;
        private final List<Double> coefficients;
        private final List<String> featureNames;
        private final double rSquared;
        private final double adjustedRSquared;
        private final double rmse;
        private final double mae;

        RegressionModel(
            double intercept,
            List<Double> coefficients,
            List<String> featureNames,
            double rSquared,
            double adjustedRSquared,
            double rmse,
            double mae
        ) {
            this.intercept = intercept;
            this.coefficients =
                List.copyOf(coefficients);
            this.featureNames =
                List.copyOf(featureNames);
            this.rSquared = rSquared;
            this.adjustedRSquared =
                adjustedRSquared;
            this.rmse = rmse;
            this.mae = mae;
        }

        double predict(List<Double> features) {
            if (
                features.size() !=
                coefficients.size()
            ) {
                throw new IllegalArgumentException(
                    "Feature count does not match model."
                );
            }

            double result = intercept;

            for (int i = 0;
                 i < features.size();
                 i++) {
                result +=
                    coefficients.get(i) *
                    features.get(i);
            }

            return result;
        }

        void printSummary() {
            System.out.println("\nRegression equation:");

            StringBuilder equation =
                new StringBuilder(
                    "review_hours = "
                );

            equation.append(
                String.format(
                    "%.4f",
                    intercept
                )
            );

            for (int i = 0;
                 i < coefficients.size();
                 i++) {

                double coefficient =
                    coefficients.get(i);

                equation.append(
                    coefficient >= 0
                        ? " + "
                        : " - "
                );

                equation.append(
                    String.format(
                        "%.4f*%s",
                        Math.abs(coefficient),
                        featureNames.get(i)
                    )
                );
            }

            System.out.println(equation);

            System.out.println("\nCoefficients:");

            for (int i = 0;
                 i < coefficients.size();
                 i++) {
                System.out.printf(
                    "  %-36s %10.4f%n",
                    featureNames.get(i),
                    coefficients.get(i)
                );
            }

            System.out.printf(
                "  %-36s %10.4f%n",
                "intercept",
                intercept
            );

            System.out.printf(
                "%nR-squared: %.4f%n",
                rSquared
            );

            System.out.printf(
                "Adjusted R-squared: %.4f%n",
                adjustedRSquared
            );

            System.out.printf(
                "RMSE: %.4f%n",
                rmse
            );

            System.out.printf(
                "MAE: %.4f%n",
                mae
            );
        }
    }

    static RegressionModel fitRegression(
        double[][] features,
        double[] target,
        List<String> names
    ) {
        if (
            features == null ||
            features.length == 0
        ) {
            throw new IllegalArgumentException(
                "Features cannot be empty."
            );
        }

        if (features.length != target.length) {
            throw new IllegalArgumentException(
                "Features and target lengths differ."
            );
        }

        int observations = features.length;
        int predictors = features[0].length;

        if (observations <= predictors) {
            throw new IllegalArgumentException(
                "Insufficient observations."
            );
        }

        if (names.size() != predictors) {
            throw new IllegalArgumentException(
                "Feature names do not match predictors."
            );
        }

        double[][] design =
            new double[observations][predictors + 1];

        for (int row = 0;
             row < observations;
             row++) {
            design[row][0] = 1.0;

            for (int column = 0;
                 column < predictors;
                 column++) {
                design[row][column + 1] =
                    features[row][column];
            }
        }

        double[][] xt =
            Matrix.transpose(design);

        double[][] xtx =
            Matrix.multiply(xt, design);

        double[][] inverse =
            Matrix.inverse(xtx);

        double[][] targetColumn =
            new double[observations][1];

        for (int i = 0;
             i < observations;
             i++) {
            targetColumn[i][0] =
                target[i];
        }

        double[][] xty =
            Matrix.multiply(
                xt,
                targetColumn
            );

        double[][] beta =
            Matrix.multiply(
                inverse,
                xty
            );

        double intercept =
            beta[0][0];

        List<Double> coefficients =
            new ArrayList<>();

        for (int i = 1;
             i < beta.length;
             i++) {
            coefficients.add(
                beta[i][0]
            );
        }

        double[] predictions =
            new double[observations];

        double[] residuals =
            new double[observations];

        double targetMean =
            Arrays.stream(target)
                .average()
                .orElseThrow();

        double sse = 0.0;
        double sst = 0.0;
        double absoluteError = 0.0;

        for (int row = 0;
             row < observations;
             row++) {

            double prediction =
                intercept;

            for (int column = 0;
                 column < predictors;
                 column++) {
                prediction +=
                    coefficients.get(column) *
                    features[row][column];
            }

            predictions[row] =
                prediction;

            residuals[row] =
                target[row] - prediction;

            sse +=
                residuals[row] *
                residuals[row];

            absoluteError +=
                Math.abs(residuals[row]);

            sst +=
                Math.pow(
                    target[row] - targetMean,
                    2
                );
        }

        double rSquared =
            sst < EPSILON
                ? (sse < EPSILON ? 1.0 : 0.0)
                : 1.0 - sse / sst;

        double adjusted =
            1.0 -
            (1.0 - rSquared) *
            ((observations - 1.0) /
             (observations - predictors - 1.0));

        double rmse =
            Math.sqrt(
                sse / observations
            );

        double mae =
            absoluteError / observations;

        return new RegressionModel(
            intercept,
            coefficients,
            names,
            rSquared,
            adjusted,
            rmse,
            mae
        );
    }

    public static void main(String[] args) {
        try {
            System.out.println(
                "LINEAR REGRESSION: ENTERPRISE REPOSITORY ANALYTICS"
            );
            System.out.println(
                "=================================================="
            );

            double[][] historicalFeatures = {
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

            double[] reviewHours = {
                2.5, 4.2, 5.1, 6.8,
                8.0, 10.1, 12.2, 13.8,
                16.0, 18.9, 20.7, 23.0,
                25.4, 27.1
            };

            List<String> featureNames =
                List.of(
                    "changed_files",
                    "reviewer_count",
                    "initial_unresolved_discussions"
                );

            RegressionModel model =
                fitRegression(
                    historicalFeatures,
                    reviewHours,
                    featureNames
                );

            model.printSummary();

            PullRequest pullRequest =
                new PullRequest(
                    1042,
                    "feature/payment-refactor",
                    "main",
                    List.of(
                        "a81c2d",
                        "b71e33",
                        "c92f44"
                    )
                );

            pullRequest.synchronizeWithBase();

            Reviewer alice =
                new Reviewer(
                    "alice",
                    true
                );

            Reviewer bob =
                new Reviewer(
                    "bob",
                    true
                );

            pullRequest.addReview(
                new Review(
                    alice,
                    ReviewState.APPROVED,
                    0
                )
            );

            pullRequest.addReview(
                new Review(
                    bob,
                    ReviewState.APPROVED,
                    0
                )
            );

            pullRequest.addStatusCheck(
                new StatusCheck(
                    "unit-tests",
                    true,
                    true
                )
            );

            pullRequest.addStatusCheck(
                new StatusCheck(
                    "security-scan",
                    true,
                    true
                )
            );

            pullRequest.addStatusCheck(
                new StatusCheck(
                    "build",
                    true,
                    true
                )
            );

            BranchProtectionPolicy policy =
                new BranchProtectionPolicy(
                    "main",
                    true,
                    true,
                    true,
                    true,
                    true,
                    true,
                    false,
                    2,
                    List.of(
                        "unit-tests",
                        "security-scan",
                        "build"
                    )
                );

            GovernanceService governance =
                new GovernanceService();

            MergeDecision decision =
                governance.evaluate(
                    pullRequest,
                    policy
                );

            System.out.println(
                "\nGovernance decision for Pull Request #" +
                pullRequest.number() +
                ": " +
                (
                    decision.eligible()
                        ? "MERGE ELIGIBLE"
                        : "BLOCKED"
                )
            );

            for (String blocker :
                 decision.blockers()) {
                System.out.println(
                    "  Blocker: " + blocker
                );
            }

            List<Double> currentProfile =
                List.of(
                    42.0,
                    4.0,
                    3.0
                );

            System.out.printf(
                "%nPredicted review duration: %.2f hours%n",
                model.predict(currentProfile)
            );

            System.out.println(
                "\nThe regression model estimates review duration "
                + "from historical observations. The governance service "
                + "does not use that estimate to approve a merge. "
                + "Merge eligibility remains governed by explicit policy."
            );

        } catch (RuntimeException error) {
            System.err.println(
                "Execution failed: " +
                error.getMessage()
            );

            System.exit(1);
        }
    }
}
