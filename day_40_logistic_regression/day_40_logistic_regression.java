import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/*
 * Enterprise Repository Governance Model
 *
 * Java 17+
 *
 * The model separates:
 * Pull Request state,
 * Code Review state,
 * Approval eligibility,
 * Branch Protection policy,
 * status checks,
 * and merge eligibility.
 *
 * Logistic regression supplies a risk probability. It does not replace
 * repository governance rules. A Pull Request must satisfy both the explicit
 * protection policy and the configured risk threshold.
 */

public class RepositoryGovernanceDemo {

    enum PullRequestStatus {
        DRAFT,
        OPEN,
        CLOSED,
        MERGED
    }

    enum ReviewStatus {
        COMMENTED,
        CHANGES_REQUESTED,
        APPROVED,
        DISMISSED
    }

    enum CheckStatus {
        PASSED,
        FAILED,
        PENDING
    }

    record Review(
        String reviewer,
        ReviewStatus status,
        boolean eligibleForApproval,
        String comment
    ) {
        public Review {
            if (reviewer == null || reviewer.isBlank()) {
                throw new IllegalArgumentException(
                    "Reviewer identity is required."
                );
            }

            if (comment == null) {
                comment = "";
            }
        }
    }

    record StatusCheck(
        String name,
        CheckStatus status,
        boolean required
    ) {
        public StatusCheck {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                    "Status check name is required."
                );
            }
        }
    }

    record BranchProtectionPolicy(
        String protectedBranch,
        int requiredApprovals,
        boolean requireStatusChecks,
        boolean requireConversationResolution,
        boolean restrictForcePush,
        boolean restrictDeletion,
        boolean requireLinearHistory
    ) {
        public BranchProtectionPolicy {
            if (protectedBranch == null || protectedBranch.isBlank()) {
                throw new IllegalArgumentException(
                    "Protected branch is required."
                );
            }

            if (requiredApprovals < 0) {
                throw new IllegalArgumentException(
                    "Required approvals cannot be negative."
                );
            }
        }
    }

    static final class PullRequest {
        private final int number;
        private final String sourceBranch;
        private final String targetBranch;
        private PullRequestStatus status;
        private boolean conversationsResolved;
        private boolean branchSynchronized;
        private boolean mergeConflict;
        private final List<Review> reviews = new ArrayList<>();
        private final List<StatusCheck> checks = new ArrayList<>();

        PullRequest(
            int number,
            String sourceBranch,
            String targetBranch
        ) {
            if (number <= 0) {
                throw new IllegalArgumentException(
                    "Pull Request number must be positive."
                );
            }

            if (sourceBranch == null || targetBranch == null) {
                throw new IllegalArgumentException(
                    "Source and target branches are required."
                );
            }

            if (sourceBranch.equals(targetBranch)) {
                throw new IllegalArgumentException(
                    "Source and target branches must differ."
                );
            }

            this.number = number;
            this.sourceBranch = sourceBranch;
            this.targetBranch = targetBranch;
            this.status = PullRequestStatus.DRAFT;
        }

        void open() {
            if (status != PullRequestStatus.DRAFT) {
                throw new IllegalStateException(
                    "Only a draft Pull Request can be opened."
                );
            }

            status = PullRequestStatus.OPEN;
        }

        void synchronize(boolean synchronizedWithBase) {
            if (status != PullRequestStatus.OPEN) {
                throw new IllegalStateException(
                    "Only an open Pull Request can be synchronized."
                );
            }

            branchSynchronized = synchronizedWithBase;
        }

        void setMergeConflict(boolean conflict) {
            if (status != PullRequestStatus.OPEN) {
                throw new IllegalStateException(
                    "Conflict state can only be changed for an open Pull Request."
                );
            }

            mergeConflict = conflict;
        }

        void resolveConversations() {
            conversationsResolved = true;
        }

        void addReview(Review review) {
            if (status != PullRequestStatus.OPEN) {
                throw new IllegalStateException(
                    "Reviews belong to an open Pull Request."
                );
            }

            reviews.add(review);
        }

        void addStatusCheck(StatusCheck check) {
            checks.add(check);
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

        PullRequestStatus status() {
            return status;
        }

        boolean conversationsResolved() {
            return conversationsResolved;
        }

        boolean branchSynchronized() {
            return branchSynchronized;
        }

        boolean mergeConflict() {
            return mergeConflict;
        }

        List<Review> reviews() {
            return List.copyOf(reviews);
        }

        List<StatusCheck> checks() {
            return List.copyOf(checks);
        }
    }

    record MergeEvaluation(
        boolean eligible,
        double riskProbability,
        List<String> blockers
    ) {}

    static final class LogisticRiskService {
        private final double[] weights;
        private final double bias;

        LogisticRiskService(double[] weights, double bias) {
            this.weights = weights.clone();
            this.bias = bias;
        }

        double probability(double... features) {
            if (features.length != weights.length) {
                throw new IllegalArgumentException(
                    "Risk feature count does not match model."
                );
            }

            double logit = bias;

            for (int index = 0; index < features.length; index++) {
                logit += weights[index] * features[index];
            }

            return sigmoid(logit);
        }

        private static double sigmoid(double value) {
            if (value >= 0) {
                double exp = Math.exp(-value);
                return 1.0 / (1.0 + exp);
            }

            double exp = Math.exp(value);
            return exp / (1.0 + exp);
        }
    }

    static final class MergeEligibilityService {
        private final BranchProtectionPolicy policy;
        private final LogisticRiskService riskService;
        private final double riskThreshold;

        MergeEligibilityService(
            BranchProtectionPolicy policy,
            LogisticRiskService riskService,
            double riskThreshold
        ) {
            if (riskThreshold <= 0 || riskThreshold >= 1) {
                throw new IllegalArgumentException(
                    "Risk threshold must be between zero and one."
                );
            }

            this.policy = policy;
            this.riskService = riskService;
            this.riskThreshold = riskThreshold;
        }

        MergeEvaluation evaluate(
            PullRequest pullRequest,
            int changedFiles,
            int additions,
            int deletions,
            int commits
        ) {
            List<String> blockers = new ArrayList<>();

            if (pullRequest.status() != PullRequestStatus.OPEN) {
                blockers.add("Pull Request is not open.");
            }

            if (!pullRequest.targetBranch()
                .equals(policy.protectedBranch())) {
                blockers.add(
                    "Pull Request does not target the protected branch."
                );
            }

            if (!pullRequest.branchSynchronized()) {
                blockers.add(
                    "Pull Request branch is not synchronized with the base branch."
                );
            }

            if (pullRequest.mergeConflict()) {
                blockers.add(
                    "Merge conflict must be resolved."
                );
            }

            long approvals = pullRequest.reviews()
                .stream()
                .filter(review ->
                    review.status() == ReviewStatus.APPROVED
                    && review.eligibleForApproval()
                )
                .map(Review::reviewer)
                .distinct()
                .count();

            if (approvals < policy.requiredApprovals()) {
                blockers.add(
                    "Required eligible approvals are missing."
                );
            }

            boolean requiredChecksPass = pullRequest.checks()
                .stream()
                .filter(StatusCheck::required)
                .allMatch(check -> check.status() == CheckStatus.PASSED);

            if (policy.requireStatusChecks() && !requiredChecksPass) {
                blockers.add(
                    "One or more required status checks are not passing."
                );
            }

            if (
                policy.requireConversationResolution()
                && !pullRequest.conversationsResolved()
            ) {
                blockers.add(
                    "Required review conversations remain unresolved."
                );
            }

            /*
             * These features represent engineering-change characteristics.
             * They are deliberately kept separate from explicit governance
             * rules so a risk score cannot silently replace required policy.
             */
            double normalizedFiles = Math.min(changedFiles / 100.0, 3.0);
            double normalizedAdditions = Math.min(additions / 1000.0, 3.0);
            double normalizedDeletions = Math.min(deletions / 1000.0, 3.0);
            double normalizedCommits = Math.min(commits / 20.0, 3.0);
            double unsynchronized =
                pullRequest.branchSynchronized() ? 0.0 : 1.0;
            double unresolved =
                pullRequest.conversationsResolved() ? 0.0 : 1.0;

            double probability = riskService.probability(
                normalizedFiles,
                normalizedAdditions,
                normalizedDeletions,
                normalizedCommits,
                unsynchronized,
                unresolved
            );

            if (probability >= riskThreshold) {
                blockers.add(
                    "Predicted defect risk exceeds the configured merge threshold."
                );
            }

            return new MergeEvaluation(
                blockers.isEmpty(),
                probability,
                List.copyOf(blockers)
            );
        }
    }

    private static PullRequest createEnterprisePullRequest() {
        PullRequest pullRequest = new PullRequest(
            1427,
            "feature/payment-retry",
            "main"
        );

        pullRequest.open();
        pullRequest.synchronize(true);

        pullRequest.addReview(
            new Review(
                "platform-reviewer",
                ReviewStatus.APPROVED,
                true,
                "Retry policy and idempotency reviewed."
            )
        );

        pullRequest.addReview(
            new Review(
                "security-reviewer",
                ReviewStatus.APPROVED,
                true,
                "Input validation and credential handling reviewed."
            )
        );

        pullRequest.addReview(
            new Review(
                "observer",
                ReviewStatus.COMMENTED,
                false,
                "Informational observation."
            )
        );

        pullRequest.addStatusCheck(
            new StatusCheck(
                "unit-tests",
                CheckStatus.PASSED,
                true
            )
        );

        pullRequest.addStatusCheck(
            new StatusCheck(
                "integration-tests",
                CheckStatus.PASSED,
                true
            )
        );

        pullRequest.addStatusCheck(
            new StatusCheck(
                "security-scan",
                CheckStatus.PASSED,
                true
            )
        );

        pullRequest.resolveConversations();

        return pullRequest;
    }

    public static void main(String[] args) {
        BranchProtectionPolicy policy =
            new BranchProtectionPolicy(
                "main",
                2,
                true,
                true,
                true,
                true,
                true
            );

        LogisticRiskService riskService =
            new LogisticRiskService(
                new double[] {
                    2.4,
                    1.5,
                    1.0,
                    1.2,
                    2.0,
                    1.8
                },
                -3.0
            );

        MergeEligibilityService eligibilityService =
            new MergeEligibilityService(
                policy,
                riskService,
                0.70
            );

        PullRequest pullRequest = createEnterprisePullRequest();

        MergeEvaluation evaluation =
            eligibilityService.evaluate(
                pullRequest,
                17,
                290,
                95,
                6
            );

        System.out.println("=== Enterprise Repository Governance ===");
        System.out.println("Pull Request: #" + pullRequest.number());
        System.out.println(
            "Source branch: " + pullRequest.sourceBranch()
        );
        System.out.println(
            "Target branch: " + pullRequest.targetBranch()
        );
        System.out.printf(
            "Predicted defect probability: %.4f%n",
            evaluation.riskProbability()
        );
        System.out.println(
            "Merge eligibility: "
            + (evaluation.eligible() ? "ELIGIBLE" : "BLOCKED")
        );

        evaluation.blockers().forEach(
            blocker -> System.out.println("Blocker: " + blocker)
        );

        System.out.println("\n=== Review Decisions ===");

        Map<ReviewStatus, Long> reviewCounts =
            pullRequest.reviews()
                .stream()
                .collect(
                    java.util.stream.Collectors.groupingBy(
                        Review::status,
                        java.util.stream.Collectors.counting()
                    )
                );

        for (ReviewStatus status : EnumSet.allOf(ReviewStatus.class)) {
            System.out.println(
                status + ": " + reviewCounts.getOrDefault(status, 0L)
            );
        }

        System.out.println("\n=== Required Checks ===");

        pullRequest.checks()
            .stream()
            .filter(StatusCheck::required)
            .forEach(check ->
                System.out.println(
                    check.name() + " -> " + check.status()
                )
            );
    }
}
