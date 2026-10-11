import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class ClassificationMetrics {

    enum ReviewClass {
        SECURITY,
        FEATURE,
        DOCUMENTATION
    }

    enum PullRequestState {
        DRAFT,
        OPEN,
        CHANGES_REQUESTED,
        APPROVED,
        MERGED,
        CLOSED
    }

    record ConfusionMatrix(
            int truePositive,
            int falsePositive,
            int trueNegative,
            int falseNegative) {

        ConfusionMatrix {
            if (truePositive < 0 || falsePositive < 0 ||
                trueNegative < 0 || falseNegative < 0) {
                throw new IllegalArgumentException(
                        "Confusion-matrix counts cannot be negative.");
            }
        }

        int total() {
            return truePositive + falsePositive
                    + trueNegative + falseNegative;
        }
    }

    record Metrics(
            double accuracy,
            double precision,
            double recall,
            double specificity,
            double f1) {
    }

    record PullRequest(
            String id,
            String author,
            ReviewClass predictedRiskClass,
            PullRequestState state,
            boolean draft) {

        PullRequest {
            Objects.requireNonNull(id);
            Objects.requireNonNull(author);
            Objects.requireNonNull(predictedRiskClass);
            Objects.requireNonNull(state);

            if (id.isBlank() || author.isBlank()) {
                throw new IllegalArgumentException(
                        "Pull Request identity fields cannot be blank.");
            }
        }
    }

    interface ReviewPolicy {
        boolean requiresSecurityReview(PullRequest pullRequest);

        int requiredApprovals(PullRequest pullRequest);
    }

    static final class ProtectedBranchPolicy implements ReviewPolicy {
        private final int requiredApprovals;
        private final boolean securityReviewForSecurityChanges;

        ProtectedBranchPolicy(
                int requiredApprovals,
                boolean securityReviewForSecurityChanges) {

            if (requiredApprovals < 0) {
                throw new IllegalArgumentException(
                        "Required approvals cannot be negative.");
            }

            this.requiredApprovals = requiredApprovals;
            this.securityReviewForSecurityChanges =
                    securityReviewForSecurityChanges;
        }

        @Override
        public boolean requiresSecurityReview(PullRequest pullRequest) {
            return securityReviewForSecurityChanges
                    && pullRequest.predictedRiskClass()
                    == ReviewClass.SECURITY;
        }

        @Override
        public int requiredApprovals(PullRequest pullRequest) {
            return requiredApprovals;
        }
    }

    record ReviewDecision(
            String reviewer,
            boolean approved,
            boolean dismissed,
            boolean unresolvedComment) {

        ReviewDecision {
            if (reviewer == null || reviewer.isBlank()) {
                throw new IllegalArgumentException(
                        "Reviewer identity is required.");
            }

            if (dismissed && approved) {
                throw new IllegalArgumentException(
                        "A dismissed review cannot remain an active approval.");
            }
        }
    }

    static final class MetricsEngine {

        private static double divide(double numerator, double denominator) {
            return denominator == 0.0 ? 0.0 : numerator / denominator;
        }

        static ConfusionMatrix buildBinaryMatrix(
                List<Integer> actual,
                List<Integer> predicted) {

            if (actual.isEmpty()) {
                throw new IllegalArgumentException(
                        "At least one classification is required.");
            }

            if (actual.size() != predicted.size()) {
                throw new IllegalArgumentException(
                        "Actual and predicted collections must have equal size.");
            }

            int tp = 0;
            int fp = 0;
            int tn = 0;
            int fn = 0;

            for (int i = 0; i < actual.size(); i++) {
                int a = actual.get(i);
                int p = predicted.get(i);

                if ((a != 0 && a != 1) || (p != 0 && p != 1)) {
                    throw new IllegalArgumentException(
                            "Binary labels must be zero or one.");
                }

                if (a == 1 && p == 1) tp++;
                else if (a == 0 && p == 1) fp++;
                else if (a == 0 && p == 0) tn++;
                else fn++;
            }

            return new ConfusionMatrix(tp, fp, tn, fn);
        }

        static Metrics calculate(ConfusionMatrix cm) {
            double accuracy = divide(
                    cm.truePositive() + cm.trueNegative(),
                    cm.total());

            double precision = divide(
                    cm.truePositive(),
                    cm.truePositive() + cm.falsePositive());

            double recall = divide(
                    cm.truePositive(),
                    cm.truePositive() + cm.falseNegative());

            double specificity = divide(
                    cm.trueNegative(),
                    cm.trueNegative() + cm.falsePositive());

            double f1 = divide(
                    2.0 * precision * recall,
                    precision + recall);

            return new Metrics(
                    accuracy,
                    precision,
                    recall,
                    specificity,
                    f1);
        }

        static List<Integer> threshold(
                List<Double> scores,
                double threshold) {

            if (threshold < 0.0 || threshold > 1.0) {
                throw new IllegalArgumentException(
                        "Threshold must be in [0, 1].");
            }

            List<Integer> predictions = new ArrayList<>();

            for (double score : scores) {
                if (score < 0.0 || score > 1.0) {
                    throw new IllegalArgumentException(
                            "Scores must be probabilities in [0, 1].");
                }

                predictions.add(score >= threshold ? 1 : 0);
            }

            return predictions;
        }
    }

    static final class MergeEligibilityService {

        private final ReviewPolicy policy;

        MergeEligibilityService(ReviewPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        boolean isEligible(
                PullRequest pullRequest,
                List<ReviewDecision> reviews,
                boolean requiredChecksPassed,
                boolean unresolvedConversations,
                boolean branchIsProtected,
                boolean directPushAttempt) {

            if (pullRequest.draft()) {
                return false;
            }

            if (pullRequest.state() == PullRequestState.CLOSED ||
                pullRequest.state() == PullRequestState.MERGED) {
                return false;
            }

            if (branchIsProtected && directPushAttempt) {
                return false;
            }

            if (!requiredChecksPassed || unresolvedConversations) {
                return false;
            }

            long activeApprovals = reviews.stream()
                    .filter(review -> review.approved())
                    .filter(review -> !review.dismissed())
                    .count();

            if (activeApprovals < policy.requiredApprovals(pullRequest)) {
                return false;
            }

            if (policy.requiresSecurityReview(pullRequest)) {
                boolean securityReviewerPresent = reviews.stream()
                        .anyMatch(review ->
                                review.approved() && !review.dismissed());

                if (!securityReviewerPresent) {
                    return false;
                }
            }

            return reviews.stream()
                    .noneMatch(ReviewDecision::unresolvedComment);
        }
    }

    private static void printMetrics(String label, Metrics metrics) {
        System.out.printf(
                "%n%s%nAccuracy=%.4f Precision=%.4f Recall=%.4f "
                        + "Specificity=%.4f F1=%.4f%n",
                label,
                metrics.accuracy(),
                metrics.precision(),
                metrics.recall(),
                metrics.specificity(),
                metrics.f1());
    }

    private static void demonstrateBinaryClassification() {
        List<Integer> actual = List.of(
                1, 1, 1, 0, 0, 0, 1, 0, 1, 0);

        List<Integer> predicted = List.of(
                1, 0, 1, 0, 0, 1, 1, 0, 1, 0);

        ConfusionMatrix matrix =
                MetricsEngine.buildBinaryMatrix(actual, predicted);

        System.out.println("=== Binary security-review classifier ===");
        System.out.printf(
                "TP=%d FP=%d TN=%d FN=%d%n",
                matrix.truePositive(),
                matrix.falsePositive(),
                matrix.trueNegative(),
                matrix.falseNegative());

        printMetrics("Classification metrics", MetricsEngine.calculate(matrix));
    }

    private static void demonstrateThresholdSelection() {
        List<Integer> actual = List.of(
                1, 0, 1, 0, 1, 0, 0, 1, 0, 1);

        List<Double> probabilities = List.of(
                0.91, 0.14, 0.61, 0.48, 0.57,
                0.31, 0.65, 0.79, 0.22, 0.44);

        System.out.println("\n=== Threshold effects ===");

        for (double threshold : List.of(0.30, 0.50, 0.70)) {
            List<Integer> predictions =
                    MetricsEngine.threshold(probabilities, threshold);

            ConfusionMatrix matrix =
                    MetricsEngine.buildBinaryMatrix(actual, predictions);

            Metrics metrics = MetricsEngine.calculate(matrix);

            System.out.printf(
                    "threshold=%.2f precision=%.3f recall=%.3f "
                            + "specificity=%.3f F1=%.3f%n",
                    threshold,
                    metrics.precision(),
                    metrics.recall(),
                    metrics.specificity(),
                    metrics.f1());
        }
    }

    private static void demonstrateGovernanceDomain() {
        PullRequest pullRequest = new PullRequest(
                "PR-184",
                "developer-17",
                ReviewClass.SECURITY,
                PullRequestState.OPEN,
                false);

        ReviewPolicy policy =
                new ProtectedBranchPolicy(2, true);

        List<ReviewDecision> reviews = List.of(
                new ReviewDecision("security-reviewer", true, false, false),
                new ReviewDecision("maintainer", true, false, false));

        MergeEligibilityService service =
                new MergeEligibilityService(policy);

        boolean eligible = service.isEligible(
                pullRequest,
                reviews,
                true,
                false,
                true,
                false);

        System.out.println("\n=== Protected-branch governance ===");
        System.out.println("Pull Request: " + pullRequest.id());
        System.out.println("Security review required: "
                + policy.requiresSecurityReview(pullRequest));
        System.out.println("Required approvals: "
                + policy.requiredApprovals(pullRequest));
        System.out.println("Merge eligible: " + eligible);

        boolean directPushAllowed = service.isEligible(
                pullRequest,
                reviews,
                true,
                false,
                true,
                true);

        System.out.println(
                "Merge workflow accepted after direct-push attempt: "
                        + directPushAllowed);
    }

    private static void demonstrateInvalidState() {
        try {
            new ReviewDecision(
                    "reviewer",
                    true,
                    true,
                    false);
        } catch (IllegalArgumentException error) {
            System.out.println(
                    "\nRejected invalid review state: "
                            + error.getMessage());
        }
    }

    public static void main(String[] args) {
        demonstrateBinaryClassification();
        demonstrateThresholdSelection();
        demonstrateGovernanceDomain();
        demonstrateInvalidState();

        System.out.println("\nEnterprise classification evaluation completed.");
    }
}
