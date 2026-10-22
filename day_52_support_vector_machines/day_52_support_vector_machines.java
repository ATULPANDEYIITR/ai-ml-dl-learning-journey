import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/*
 * Enterprise-oriented linear SVM domain model.
 *
 * The SVM classifies a repository change into an acceptable-risk or
 * remediation-risk region. The governance service separately evaluates
 * repository policy, approval requirements, status checks, and merge state.
 *
 * This separation is important: statistical classification does not itself
 * authorize a protected-branch merge.
 */
public class SvmRepositoryGovernance {

    enum Label {
        ACCEPTABLE(1),
        REMEDIATION(-1);

        private final int value;

        Label(int value) {
            this.value = value;
        }

        int value() {
            return value;
        }
    }

    enum PullRequestState {
        DRAFT,
        OPEN,
        CLOSED,
        MERGED
    }

    enum ReviewState {
        COMMENTED,
        CHANGES_REQUESTED,
        APPROVED,
        DISMISSED
    }

    enum CheckState {
        PENDING,
        PASSED,
        FAILED
    }

    record PullRequest(
            String id,
            String sourceBranch,
            String targetBranch,
            double[] features,
            Label expectedLabel,
            PullRequestState state,
            boolean sourceSynchronized,
            boolean hasConflicts,
            boolean conversationsResolved
    ) {
        PullRequest {
            Objects.requireNonNull(id);
            Objects.requireNonNull(sourceBranch);
            Objects.requireNonNull(targetBranch);
            Objects.requireNonNull(features);
            Objects.requireNonNull(expectedLabel);
            Objects.requireNonNull(state);

            if (id.isBlank() || sourceBranch.isBlank() || targetBranch.isBlank()) {
                throw new IllegalArgumentException("Pull Request fields cannot be blank.");
            }

            if (features.length == 0) {
                throw new IllegalArgumentException("Feature vector cannot be empty.");
            }

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Features must be finite.");
                }
            }
        }
    }

    record Review(
            String reviewer,
            ReviewState state,
            boolean stale
    ) {
        Review {
            Objects.requireNonNull(reviewer);
            Objects.requireNonNull(state);

            if (reviewer.isBlank()) {
                throw new IllegalArgumentException("Reviewer cannot be blank.");
            }
        }

        boolean isCurrentApproval() {
            return state == ReviewState.APPROVED && !stale;
        }
    }

    record StatusCheck(
            String name,
            CheckState state
    ) {
        StatusCheck {
            Objects.requireNonNull(name);
            Objects.requireNonNull(state);

            if (name.isBlank()) {
                throw new IllegalArgumentException("Check name cannot be blank.");
            }
        }

        boolean passed() {
            return state == CheckState.PASSED;
        }
    }

    record BranchProtectionPolicy(
            String branch,
            int requiredApprovals,
            Set<String> requiredChecks,
            boolean requireResolvedConversations,
            boolean prohibitDirectPush,
            boolean prohibitForcePush,
            boolean prohibitDeletion
    ) {
        BranchProtectionPolicy {
            Objects.requireNonNull(branch);
            Objects.requireNonNull(requiredChecks);

            if (branch.isBlank()) {
                throw new IllegalArgumentException("Protected branch cannot be blank.");
            }

            if (requiredApprovals < 0) {
                throw new IllegalArgumentException(
                        "Required approvals cannot be negative.");
            }

            requiredChecks = Set.copyOf(requiredChecks);
        }
    }

    record MergeDecision(
            boolean eligible,
            List<String> blockingReasons
    ) {
        MergeDecision {
            blockingReasons = List.copyOf(blockingReasons);
        }
    }

    static final class StandardScaler {
        private double[] means;
        private double[] standardDeviations;

        void fit(List<PullRequest> pullRequests) {
            if (pullRequests.isEmpty()) {
                throw new IllegalArgumentException("Training data is empty.");
            }

            int dimension = pullRequests.getFirst().features().length;

            means = new double[dimension];
            standardDeviations = new double[dimension];

            for (PullRequest request : pullRequests) {
                if (request.features().length != dimension) {
                    throw new IllegalArgumentException(
                            "All observations must have the same dimension.");
                }

                for (int j = 0; j < dimension; j++) {
                    means[j] += request.features()[j];
                }
            }

            for (int j = 0; j < dimension; j++) {
                means[j] /= pullRequests.size();
            }

            for (int j = 0; j < dimension; j++) {
                double variance = 0.0;

                for (PullRequest request : pullRequests) {
                    double delta = request.features()[j] - means[j];
                    variance += delta * delta;
                }

                variance /= pullRequests.size();

                double standardDeviation = Math.sqrt(variance);
                standardDeviations[j] =
                        standardDeviation > 1e-12
                                ? standardDeviation
                                : 1.0;
            }
        }

        double[] transform(double[] features) {
            if (means == null) {
                throw new IllegalStateException("Scaler has not been fitted.");
            }

            if (features.length != means.length) {
                throw new IllegalArgumentException("Feature dimension mismatch.");
            }

            double[] transformed = new double[features.length];

            for (int j = 0; j < features.length; j++) {
                transformed[j] =
                        (features[j] - means[j]) / standardDeviations[j];
            }

            return transformed;
        }
    }

    static final class LinearSoftMarginSvm {
        private final double C;
        private final double learningRate;
        private final int epochs;

        private double[] weights;
        private double bias;

        LinearSoftMarginSvm(double C, double learningRate, int epochs) {
            if (C <= 0 || learningRate <= 0 || epochs <= 0) {
                throw new IllegalArgumentException(
                        "SVM hyperparameters must be positive.");
            }

            this.C = C;
            this.learningRate = learningRate;
            this.epochs = epochs;
        }

        void fit(List<PullRequest> trainingData, StandardScaler scaler) {
            if (trainingData.isEmpty()) {
                throw new IllegalArgumentException("Training data is empty.");
            }

            int dimension = trainingData.getFirst().features().length;
            weights = new double[dimension];
            bias = 0.0;

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (PullRequest request : trainingData) {
                    double[] x = scaler.transform(request.features());
                    int y = request.expectedLabel().value();

                    double score = decision(x);
                    double margin = y * score;

                    if (margin >= 1.0) {
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] -= learningRate * weights[j];
                        }
                    } else {
                        for (int j = 0; j < weights.length; j++) {
                            double gradient =
                                    weights[j] - C * y * x[j];

                            weights[j] -= learningRate * gradient;
                        }

                        bias -= learningRate * (-C * y);
                    }
                }
            }
        }

        double decision(double[] features) {
            if (weights == null) {
                throw new IllegalStateException("SVM has not been trained.");
            }

            if (features.length != weights.length) {
                throw new IllegalArgumentException("Feature dimension mismatch.");
            }

            double score = bias;

            for (int j = 0; j < features.length; j++) {
                score += weights[j] * features[j];
            }

            return score;
        }

        Label classify(double[] features) {
            return decision(features) >= 0
                    ? Label.ACCEPTABLE
                    : Label.REMEDIATION;
        }

        double functionalMargin(double[] features, Label label) {
            return label.value() * decision(features);
        }

        double marginWidth() {
            double squaredNorm = 0.0;

            for (double weight : weights) {
                squaredNorm += weight * weight;
            }

            double weightNorm = Math.sqrt(squaredNorm);

            return weightNorm == 0.0
                    ? Double.POSITIVE_INFINITY
                    : 2.0 / weightNorm;
        }
    }

    static final class MergeEligibilityService {
        private final BranchProtectionPolicy policy;

        MergeEligibilityService(BranchProtectionPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        MergeDecision evaluate(
                PullRequest pullRequest,
                List<Review> reviews,
                List<StatusCheck> checks,
                Label svmClassification) {

            List<String> reasons = new ArrayList<>();

            if (pullRequest.state() != PullRequestState.OPEN) {
                reasons.add("Pull Request is not open.");
            }

            if (!pullRequest.targetBranch().equals(policy.branch())) {
                reasons.add("Target branch is not the protected branch.");
            }

            if (!pullRequest.sourceSynchronized()) {
                reasons.add("Source branch is behind the target branch.");
            }

            if (pullRequest.hasConflicts()) {
                reasons.add("Pull Request has unresolved merge conflicts.");
            }

            if (policy.requireResolvedConversations()
                    && !pullRequest.conversationsResolved()) {
                reasons.add("Required review conversations are unresolved.");
            }

            long approvals = reviews.stream()
                    .filter(Review::isCurrentApproval)
                    .count();

            if (approvals < policy.requiredApprovals()) {
                reasons.add(
                        "Current approval count is below the required threshold.");
            }

            Set<String> passedChecks = new HashSet<>();

            for (StatusCheck check : checks) {
                if (check.passed()) {
                    passedChecks.add(check.name());
                }
            }

            for (String required : policy.requiredChecks()) {
                if (!passedChecks.contains(required)) {
                    reasons.add("Required status check is not passing: " + required);
                }
            }

            /*
             * The SVM classification is a separate risk gate in this example.
             * It does not erase or override branch-protection requirements.
             */
            if (svmClassification == Label.REMEDIATION) {
                reasons.add(
                        "SVM classified the change in the remediation-risk region.");
            }

            return new MergeDecision(reasons.isEmpty(), reasons);
        }
    }

    private static List<PullRequest> trainingData() {
        return List.of(
                new PullRequest(
                        "PR-401", "feature/payment-validation", "main",
                        new double[]{0.4, 0.95}, Label.ACCEPTABLE,
                        PullRequestState.OPEN, true, false, true),

                new PullRequest(
                        "PR-402", "feature/session-hardening", "main",
                        new double[]{0.7, 0.91}, Label.ACCEPTABLE,
                        PullRequestState.OPEN, true, false, true),

                new PullRequest(
                        "PR-403", "feature/auth-refactor", "main",
                        new double[]{1.2, 0.84}, Label.ACCEPTABLE,
                        PullRequestState.OPEN, true, false, true),

                new PullRequest(
                        "PR-404", "feature/legacy-cleanup", "main",
                        new double[]{3.4, 0.56}, Label.REMEDIATION,
                        PullRequestState.OPEN, true, false, true),

                new PullRequest(
                        "PR-405", "feature/query-risk", "main",
                        new double[]{3.8, 0.52}, Label.REMEDIATION,
                        PullRequestState.OPEN, true, false, true),

                new PullRequest(
                        "PR-406", "feature/boundary-change", "main",
                        new double[]{2.1, 0.68}, Label.REMEDIATION,
                        PullRequestState.OPEN, true, false, true)
        );
    }

    public static void main(String[] args) {
        List<PullRequest> data = trainingData();

        StandardScaler scaler = new StandardScaler();
        scaler.fit(data);

        LinearSoftMarginSvm svm =
                new LinearSoftMarginSvm(1.0, 0.002, 5000);

        svm.fit(data, scaler);

        BranchProtectionPolicy policy =
                new BranchProtectionPolicy(
                        "main",
                        2,
                        Set.of("unit-tests", "security-scan"),
                        true,
                        true,
                        true,
                        true
                );

        MergeEligibilityService service =
                new MergeEligibilityService(policy);

        List<Review> healthyReviews = List.of(
                new Review("alice", ReviewState.APPROVED, false),
                new Review("bob", ReviewState.APPROVED, false),
                new Review("carol", ReviewState.COMMENTED, false)
        );

        List<StatusCheck> passingChecks = List.of(
                new StatusCheck("unit-tests", CheckState.PASSED),
                new StatusCheck("security-scan", CheckState.PASSED),
                new StatusCheck("integration-tests", CheckState.PASSED)
        );

        System.out.printf(
                "SVM margin width: %.4f%n",
                svm.marginWidth()
        );

        System.out.println("\nRisk classifications:");

        for (PullRequest request : data) {
            double[] normalized = scaler.transform(request.features());
            Label classification = svm.classify(normalized);

            System.out.printf(
                    "%s -> %s, score=%.4f, functionalMargin=%.4f%n",
                    request.id(),
                    classification,
                    svm.decision(normalized),
                    svm.functionalMargin(
                            normalized,
                            request.expectedLabel())
            );
        }

        PullRequest mergeCandidate = data.getFirst();

        Label candidateClassification =
                svm.classify(scaler.transform(mergeCandidate.features()));

        MergeDecision healthyDecision =
                service.evaluate(
                        mergeCandidate,
                        healthyReviews,
                        passingChecks,
                        candidateClassification
                );

        System.out.println("\nHealthy governance state:");
        printDecision(healthyDecision);

        /*
         * Stale approval demonstrates that approval count is policy-aware.
         * An approval can exist as a historical review decision while not
         * satisfying the current required-review threshold.
         */
        List<Review> staleReviews = List.of(
                new Review("alice", ReviewState.APPROVED, true),
                new Review("bob", ReviewState.APPROVED, false)
        );

        MergeDecision staleDecision =
                service.evaluate(
                        mergeCandidate,
                        staleReviews,
                        passingChecks,
                        candidateClassification
                );

        System.out.println("\nStale-approval scenario:");
        printDecision(staleDecision);

        /*
         * Failed status checks demonstrate that statistical classification
         * cannot bypass a deterministic repository protection policy.
         */
        List<StatusCheck> failedChecks = List.of(
                new StatusCheck("unit-tests", CheckState.PASSED),
                new StatusCheck("security-scan", CheckState.FAILED)
        );

        MergeDecision failedCheckDecision =
                service.evaluate(
                        mergeCandidate,
                        healthyReviews,
                        failedChecks,
                        candidateClassification
                );

        System.out.println("\nFailed-check scenario:");
        printDecision(failedCheckDecision);
    }

    private static void printDecision(MergeDecision decision) {
        System.out.println(
                "merge=" + (decision.eligible() ? "eligible" : "blocked")
        );

        for (String reason : decision.blockingReasons()) {
            System.out.println("  - " + reason);
        }
    }
}
