import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Enterprise-oriented ROC/PR threshold service.
 *
 * The domain models a code-review routing system. The statistical model
 * predicts whether a Pull Request should receive additional review attention.
 * Repository merge authorization remains a separate policy domain.
 *
 * Java 17+.
 */
public class RocPrecisionRecallEnterprise {

    enum PullRequestState {
        DRAFT,
        OPEN,
        CLOSED,
        MERGED
    }

    enum ReviewDecision {
        APPROVED,
        CHANGES_REQUESTED,
        COMMENTED
    }

    record Prediction(int actual, double score) {
        Prediction {
            if (actual != 0 && actual != 1) {
                throw new IllegalArgumentException("actual must be 0 or 1");
            }
            if (!Double.isFinite(score) || score < 0 || score > 1) {
                throw new IllegalArgumentException("score must be in [0, 1]");
            }
        }
    }

    record ConfusionMatrix(
        double threshold,
        int truePositive,
        int falsePositive,
        int trueNegative,
        int falseNegative
    ) {
        double precision() {
            return truePositive + falsePositive == 0
                ? 0.0
                : (double) truePositive / (truePositive + falsePositive);
        }

        double recall() {
            return truePositive + falseNegative == 0
                ? 0.0
                : (double) truePositive / (truePositive + falseNegative);
        }

        double specificity() {
            return trueNegative + falsePositive == 0
                ? 0.0
                : (double) trueNegative / (trueNegative + falsePositive);
        }

        double falsePositiveRate() {
            return 1.0 - specificity();
        }

        double f1() {
            double p = precision();
            double r = recall();
            return p + r == 0 ? 0.0 : 2.0 * p * r / (p + r);
        }
    }

    static final class ReviewPolicy {
        private final int requiredApprovals;
        private final boolean requirePassingChecks;
        private final boolean requireResolvedConversations;

        ReviewPolicy(
            int requiredApprovals,
            boolean requirePassingChecks,
            boolean requireResolvedConversations
        ) {
            if (requiredApprovals < 0) {
                throw new IllegalArgumentException(
                    "requiredApprovals cannot be negative"
                );
            }

            this.requiredApprovals = requiredApprovals;
            this.requirePassingChecks = requirePassingChecks;
            this.requireResolvedConversations = requireResolvedConversations;
        }

        boolean canMerge(PullRequest pullRequest) {
            if (pullRequest.state != PullRequestState.OPEN) {
                return false;
            }

            if (pullRequest.approvals < requiredApprovals) {
                return false;
            }

            if (pullRequest.changesRequested) {
                return false;
            }

            if (requirePassingChecks && !pullRequest.statusChecksPassing) {
                return false;
            }

            if (requireResolvedConversations &&
                pullRequest.unresolvedConversations > 0) {
                return false;
            }

            return true;
        }
    }

    static final class PullRequest {
        private final int id;
        private final String sourceBranch;
        private final String targetBranch;

        private PullRequestState state;
        private int approvals;
        private boolean changesRequested;
        private boolean statusChecksPassing;
        private int unresolvedConversations;

        PullRequest(int id, String sourceBranch, String targetBranch) {
            if (id <= 0) {
                throw new IllegalArgumentException("PR id must be positive");
            }

            this.id = id;
            this.sourceBranch = Objects.requireNonNull(sourceBranch);
            this.targetBranch = Objects.requireNonNull(targetBranch);
            this.state = PullRequestState.DRAFT;
        }

        void markReadyForReview() {
            if (state != PullRequestState.DRAFT) {
                throw new IllegalStateException(
                    "Only a draft PR can be marked ready"
                );
            }
            state = PullRequestState.OPEN;
        }

        void recordReview(ReviewDecision decision) {
            if (state != PullRequestState.OPEN) {
                throw new IllegalStateException(
                    "Reviews can only be recorded on an open PR"
                );
            }

            switch (decision) {
                case APPROVED -> approvals++;
                case CHANGES_REQUESTED -> changesRequested = true;
                case COMMENTED -> {
                    // A comment alone is not an approval and does not request changes.
                }
            }
        }

        void resolveChangesRequested() {
            changesRequested = false;
        }

        void setStatusChecks(boolean passing) {
            statusChecksPassing = passing;
        }

        void addConversation() {
            unresolvedConversations++;
        }

        void resolveConversation() {
            if (unresolvedConversations == 0) {
                throw new IllegalStateException(
                    "No unresolved conversation exists"
                );
            }
            unresolvedConversations--;
        }

        /*
         * A synchronization event represents the source branch incorporating
         * new target-branch changes. This implementation invalidates prior
         * approvals, modeling a strict stale-approval policy.
         */
        void synchronizeWithTarget() {
            if (state != PullRequestState.OPEN) {
                throw new IllegalStateException(
                    "Only an open PR can be synchronized"
                );
            }
            approvals = 0;
        }

        void merge(ReviewPolicy policy) {
            if (!policy.canMerge(this)) {
                throw new IllegalStateException(
                    "Pull Request does not satisfy merge policy"
                );
            }
            state = PullRequestState.MERGED;
        }

        @Override
        public String toString() {
            return "PR #" + id +
                " " + sourceBranch + " -> " + targetBranch +
                ", state=" + state +
                ", approvals=" + approvals +
                ", changesRequested=" + changesRequested +
                ", checksPassing=" + statusChecksPassing +
                ", unresolvedConversations=" + unresolvedConversations;
        }
    }

    interface ThresholdObjective {
        double score(ConfusionMatrix matrix);
    }

    static final class MetricService {

        static void validate(List<Prediction> data) {
            if (data.isEmpty()) {
                throw new IllegalArgumentException("Dataset cannot be empty");
            }

            Set<Integer> labels = new HashSet<>();
            data.forEach(row -> labels.add(row.actual()));

            if (!labels.contains(0) || !labels.contains(1)) {
                throw new IllegalArgumentException(
                    "Both classes are required for ROC/PR analysis"
                );
            }
        }

        static List<Double> thresholds(List<Prediction> data) {
            Set<Double> thresholds = new HashSet<>();
            thresholds.add(0.0);
            thresholds.add(1.0);

            for (Prediction prediction : data) {
                thresholds.add(prediction.score());
            }

            return thresholds.stream()
                .sorted(Comparator.reverseOrder())
                .toList();
        }

        static ConfusionMatrix evaluate(
            List<Prediction> data,
            double threshold
        ) {
            int tp = 0;
            int fp = 0;
            int tn = 0;
            int fn = 0;

            for (Prediction prediction : data) {
                boolean positive = prediction.score() >= threshold;

                if (prediction.actual() == 1 && positive) {
                    tp++;
                } else if (prediction.actual() == 0 && positive) {
                    fp++;
                } else if (prediction.actual() == 0) {
                    tn++;
                } else {
                    fn++;
                }
            }

            return new ConfusionMatrix(threshold, tp, fp, tn, fn);
        }

        static double auc(List<double[]> points) {
            points.sort(Comparator.comparingDouble(point -> point[0]));

            double area = 0.0;

            for (int i = 1; i < points.size(); i++) {
                double x1 = points.get(i - 1)[0];
                double y1 = points.get(i - 1)[1];
                double x2 = points.get(i)[0];
                double y2 = points.get(i)[1];

                area += (x2 - x1) * (y1 + y2) / 2.0;
            }

            return area;
        }

        static double rocAuc(List<Prediction> data) {
            validate(data);

            List<double[]> points = new ArrayList<>();

            for (double threshold : thresholds(data)) {
                ConfusionMatrix matrix = evaluate(data, threshold);
                points.add(new double[] {
                    matrix.falsePositiveRate(),
                    matrix.recall()
                });
            }

            return auc(points);
        }

        static double averagePrecision(List<Prediction> data) {
            validate(data);

            List<Prediction> ranked = new ArrayList<>(data);
            ranked.sort(
                Comparator.comparingDouble(Prediction::score).reversed()
            );

            long totalPositive = data.stream()
                .filter(row -> row.actual() == 1)
                .count();

            int tp = 0;
            int fp = 0;
            double previousRecall = 0.0;
            double result = 0.0;

            for (Prediction prediction : ranked) {
                if (prediction.actual() == 1) {
                    tp++;
                } else {
                    fp++;
                }

                double precision = (double) tp / (tp + fp);
                double recall = (double) tp / totalPositive;

                if (prediction.actual() == 1) {
                    result += precision * (recall - previousRecall);
                    previousRecall = recall;
                }
            }

            return result;
        }

        static ConfusionMatrix optimize(
            List<Prediction> data,
            ThresholdObjective objective
        ) {
            validate(data);

            ConfusionMatrix best = null;
            double bestScore = Double.NEGATIVE_INFINITY;

            for (double threshold : thresholds(data)) {
                ConfusionMatrix matrix = evaluate(data, threshold);
                double score = objective.score(matrix);

                if (score > bestScore) {
                    bestScore = score;
                    best = matrix;
                }
            }

            return best;
        }
    }

    public static void main(String[] args) {
        System.out.println("ROC & Precision-Recall Enterprise Model");
        System.out.println("----------------------------------------");

        PullRequest pullRequest = new PullRequest(
            902,
            "feature/payment-risk",
            "main"
        );

        ReviewPolicy branchPolicy = new ReviewPolicy(
            2,
            true,
            true
        );

        pullRequest.markReadyForReview();
        pullRequest.setStatusChecks(true);
        pullRequest.addConversation();

        pullRequest.recordReview(ReviewDecision.APPROVED);
        pullRequest.recordReview(ReviewDecision.COMMENTED);

        System.out.println(pullRequest);
        System.out.println(
            "Merge eligible before second approval: " +
            branchPolicy.canMerge(pullRequest)
        );

        pullRequest.resolveConversation();
        pullRequest.recordReview(ReviewDecision.APPROVED);

        System.out.println(
            "Merge eligible after policy conditions: " +
            branchPolicy.canMerge(pullRequest)
        );

        List<Prediction> validationSet = List.of(
            new Prediction(0, 0.03),
            new Prediction(0, 0.07),
            new Prediction(0, 0.11),
            new Prediction(0, 0.19),
            new Prediction(0, 0.23),
            new Prediction(0, 0.29),
            new Prediction(0, 0.34),
            new Prediction(0, 0.42),
            new Prediction(0, 0.48),
            new Prediction(0, 0.57),
            new Prediction(1, 0.31),
            new Prediction(1, 0.51),
            new Prediction(1, 0.63),
            new Prediction(1, 0.68),
            new Prediction(1, 0.79),
            new Prediction(1, 0.93)
        );

        double rocAuc = MetricService.rocAuc(validationSet);
        double averagePrecision =
            MetricService.averagePrecision(validationSet);

        System.out.printf(
            "ROC AUC: %.4f%n",
            rocAuc
        );

        System.out.printf(
            "Average Precision: %.4f%n",
            averagePrecision
        );

        ConfusionMatrix maximumF1 = MetricService.optimize(
            validationSet,
            ConfusionMatrix::f1
        );

        System.out.printf(
            "Maximum-F1 threshold: %.2f, precision=%.3f, recall=%.3f, F1=%.3f%n",
            maximumF1.threshold(),
            maximumF1.precision(),
            maximumF1.recall(),
            maximumF1.f1()
        );

        /*
         * A constrained objective is preferable when false positives have an
         * operational limit. The classifier is optimized for recall only among
         * thresholds satisfying the specified false-positive constraint.
         */
        ConfusionMatrix constrained = MetricService.optimize(
            validationSet,
            matrix ->
                matrix.falsePositiveRate() <= 0.10
                    ? matrix.recall()
                    : Double.NEGATIVE_INFINITY
        );

        System.out.printf(
            "Recall subject to FPR <= 0.10: threshold=%.2f, recall=%.3f, FPR=%.3f%n",
            constrained.threshold(),
            constrained.recall(),
            constrained.falsePositiveRate()
        );

        System.out.println(
            "The statistical threshold is kept separate from merge policy. " +
            "A strong ROC/PR metric does not itself authorize a merge."
        );
    }
}
