import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public class UnsupervisedLearningGovernance {

    enum ReviewState {
        PENDING,
        APPROVED,
        CHANGES_REQUESTED,
        COMMENTED,
        DISMISSED
    }

    enum MergeState {
        OPEN,
        BLOCKED,
        MERGEABLE,
        MERGED,
        CLOSED
    }

    enum CheckState {
        PASS,
        FAIL,
        PENDING
    }

    record Observation(String id, List<Double> features) {
        Observation {
            if (id == null || id.isBlank()) {
                throw new ValidationException("Observation id is required.");
            }
            if (features == null || features.isEmpty()) {
                throw new ValidationException("Features are required.");
            }
            if (features.stream().anyMatch(value -> value == null || !Double.isFinite(value))) {
                throw new ValidationException("Features must be finite.");
            }
            features = List.copyOf(features);
        }
    }

    record StatusCheck(String name, CheckState state) {
        StatusCheck {
            if (name == null || name.isBlank()) {
                throw new ValidationException("Status check name is required.");
            }
            Objects.requireNonNull(state, "Status check state");
        }
    }

    record Review(String reviewer, ReviewState state, boolean eligible) {
        Review {
            if (reviewer == null || reviewer.isBlank()) {
                throw new ValidationException("Reviewer identity is required.");
            }
        }
    }

    record BranchPolicy(
            int requiredApprovals,
            boolean requireAllChecks,
            boolean requireResolvedConversations,
            boolean allowDirectPush,
            boolean allowForcePush,
            boolean requireLinearHistory) {

        BranchPolicy {
            if (requiredApprovals < 0) {
                throw new ValidationException("Required approvals cannot be negative.");
            }
        }
    }

    static final class PullRequest {
        private final int number;
        private final String sourceBranch;
        private final String targetBranch;
        private final List<Observation> observations;
        private final List<Review> reviews;
        private final List<StatusCheck> checks;

        private boolean draft;
        private boolean conflicts;
        private boolean conversationsResolved;
        private MergeState state = MergeState.OPEN;

        PullRequest(
                int number,
                String sourceBranch,
                String targetBranch,
                List<Observation> observations,
                List<Review> reviews,
                List<StatusCheck> checks) {

            if (number <= 0) {
                throw new ValidationException("Pull Request number must be positive.");
            }
            if (sourceBranch == null || targetBranch == null ||
                    sourceBranch.isBlank() || targetBranch.isBlank()) {
                throw new ValidationException("Source and target branches are required.");
            }

            this.number = number;
            this.sourceBranch = sourceBranch;
            this.targetBranch = targetBranch;
            this.observations = List.copyOf(observations);
            this.reviews = new ArrayList<>(reviews);
            this.checks = List.copyOf(checks);
        }

        void setDraft(boolean draft) {
            ensureOpen();
            this.draft = draft;
        }

        void setConflicts(boolean conflicts) {
            ensureOpen();
            this.conflicts = conflicts;
        }

        void setConversationsResolved(boolean resolved) {
            ensureOpen();
            this.conversationsResolved = resolved;
        }

        List<Review> reviews() {
            return Collections.unmodifiableList(reviews);
        }

        List<StatusCheck> checks() {
            return checks;
        }

        int number() {
            return number;
        }

        boolean isDraft() {
            return draft;
        }

        boolean hasConflicts() {
            return conflicts;
        }

        boolean conversationsResolved() {
            return conversationsResolved;
        }

        String sourceBranch() {
            return sourceBranch;
        }

        String targetBranch() {
            return targetBranch;
        }

        MergeState state() {
            return state;
        }

        void transitionTo(MergeState newState) {
            if (state == MergeState.MERGED || state == MergeState.CLOSED) {
                throw new ValidationException("Terminal Pull Request cannot change state.");
            }
            state = Objects.requireNonNull(newState);
        }

        private void ensureOpen() {
            if (state != MergeState.OPEN && state != MergeState.BLOCKED) {
                throw new ValidationException("Pull Request is not editable.");
            }
        }
    }

    static final class MergeEligibilityService {
        private final BranchPolicy policy;

        MergeEligibilityService(BranchPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        List<String> blockers(PullRequest pullRequest) {
            List<String> blockers = new ArrayList<>();

            if (pullRequest.isDraft()) {
                blockers.add("Pull Request remains a draft.");
            }

            if (pullRequest.sourceBranch().equals(pullRequest.targetBranch())) {
                blockers.add("Source and target branches are identical.");
            }

            if (pullRequest.hasConflicts()) {
                blockers.add("Merge conflicts must be resolved.");
            }

            long eligibleApprovals = pullRequest.reviews()
                    .stream()
                    .filter(Review::eligible)
                    .filter(review -> review.state() == ReviewState.APPROVED)
                    .count();

            if (eligibleApprovals < policy.requiredApprovals()) {
                blockers.add("Required eligible approvals are missing.");
            }

            boolean changesRequested = pullRequest.reviews()
                    .stream()
                    .anyMatch(review -> review.eligible() &&
                            review.state() == ReviewState.CHANGES_REQUESTED);

            if (changesRequested) {
                blockers.add("An eligible reviewer has requested changes.");
            }

            if (policy.requireResolvedConversations() &&
                    !pullRequest.conversationsResolved()) {
                blockers.add("Review conversations remain unresolved.");
            }

            if (policy.requireAllChecks()) {
                pullRequest.checks().stream()
                        .filter(check -> check.state() != CheckState.PASS)
                        .forEach(check ->
                                blockers.add("Status check is not passing: " + check.name()));
            }

            return List.copyOf(blockers);
        }

        boolean isEligible(PullRequest pullRequest) {
            List<String> blockers = blockers(pullRequest);

            if (blockers.isEmpty()) {
                pullRequest.transitionTo(MergeState.MERGEABLE);
                return true;
            }

            pullRequest.transitionTo(MergeState.BLOCKED);
            return false;
        }
    }

    static final class KMeansService {
        private final int k;
        private final int maxIterations;
        private final double tolerance;

        KMeansService(int k, int maxIterations, double tolerance) {
            if (k <= 0 || maxIterations <= 0 || tolerance < 0) {
                throw new ValidationException("Invalid k-means configuration.");
            }
            this.k = k;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
        }

        Map<Integer, List<String>> cluster(List<Observation> observations) {
            validateObservations(observations);

            if (k > observations.size()) {
                throw new ValidationException("k cannot exceed observation count.");
            }

            int dimensions = observations.get(0).features().size();
            List<List<Double>> centroids = new ArrayList<>();

            for (int i = 0; i < k; i++) {
                centroids.add(new ArrayList<>(observations.get(i).features()));
            }

            Map<Integer, List<String>> assignments = new HashMap<>();

            for (int iteration = 0; iteration < maxIterations; iteration++) {
                assignments.clear();

                for (int cluster = 0; cluster < k; cluster++) {
                    assignments.put(cluster, new ArrayList<>());
                }

                for (Observation observation : observations) {
                    int closest = 0;
                    double best = Double.POSITIVE_INFINITY;

                    for (int cluster = 0; cluster < k; cluster++) {
                        double current =
                                distance(observation.features(), centroids.get(cluster));

                        if (current < best) {
                            best = current;
                            closest = cluster;
                        }
                    }

                    assignments.get(closest).add(observation.id());
                }

                List<List<Double>> next = new ArrayList<>();

                for (int cluster = 0; cluster < k; cluster++) {
                    List<String> members = assignments.get(cluster);

                    if (members.isEmpty()) {
                        next.add(new ArrayList<>(centroids.get(cluster)));
                        continue;
                    }

                    List<Double> centroid = new ArrayList<>(
                            Collections.nCopies(dimensions, 0.0));

                    for (String id : members) {
                        Observation observation = observations.stream()
                                .filter(item -> item.id().equals(id))
                                .findFirst()
                                .orElseThrow();

                        for (int dimension = 0; dimension < dimensions; dimension++) {
                            centroid.set(
                                    dimension,
                                    centroid.get(dimension) +
                                            observation.features().get(dimension));
                        }
                    }

                    for (int dimension = 0; dimension < dimensions; dimension++) {
                        centroid.set(
                                dimension,
                                centroid.get(dimension) / members.size());
                    }

                    next.add(centroid);
                }

                double movement = 0.0;
                for (int cluster = 0; cluster < k; cluster++) {
                    movement = Math.max(
                            movement,
                            distance(centroids.get(cluster), next.get(cluster)));
                }

                centroids = next;

                if (movement <= tolerance) {
                    break;
                }
            }

            return Map.copyOf(assignments);
        }

        private static double distance(List<Double> first, List<Double> second) {
            double sum = 0.0;

            for (int index = 0; index < first.size(); index++) {
                double delta = first.get(index) - second.get(index);
                sum += delta * delta;
            }

            return Math.sqrt(sum);
        }

        private static void validateObservations(List<Observation> observations) {
            if (observations == null || observations.isEmpty()) {
                throw new ValidationException("Observation collection cannot be empty.");
            }

            int dimensions = observations.get(0).features().size();
            Set<String> identifiers = new HashSet<>();

            for (Observation observation : observations) {
                if (observation.features().size() != dimensions) {
                    throw new ValidationException("Feature dimensions must match.");
                }

                if (!identifiers.add(observation.id())) {
                    throw new ValidationException("Duplicate observation: " + observation.id());
                }
            }
        }
    }

    static final class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    public static void main(String[] args) {
        List<Observation> observations = List.of(
                new Observation("customer-01", List.of(1.0, 1.2, 0.9)),
                new Observation("customer-02", List.of(1.1, 0.8, 1.0)),
                new Observation("customer-03", List.of(0.9, 1.0, 1.2)),
                new Observation("customer-04", List.of(7.8, 8.1, 7.5)),
                new Observation("customer-05", List.of(8.2, 7.9, 8.4)),
                new Observation("customer-06", List.of(7.5, 8.3, 8.0)),
                new Observation("customer-07", List.of(4.2, 4.0, 4.1)),
                new Observation("customer-08", List.of(4.5, 4.3, 4.2)),
                new Observation("customer-09", List.of(4.0, 4.4, 4.5)),
                new Observation("customer-10", List.of(25.0, 1.0, 0.5))
        );

        PullRequest pullRequest = new PullRequest(
                1042,
                "feature/customer-segmentation",
                "main",
                observations,
                List.of(
                        new Review("analyst-a", ReviewState.APPROVED, true),
                        new Review("analyst-b", ReviewState.APPROVED, true),
                        new Review("analyst-c", ReviewState.COMMENTED, true)
                ),
                List.of(
                        new StatusCheck("unit-tests", CheckState.PASS),
                        new StatusCheck("security-scan", CheckState.PASS),
                        new StatusCheck("data-quality", CheckState.PASS)
                )
        );

        pullRequest.setDraft(false);
        pullRequest.setConflicts(false);
        pullRequest.setConversationsResolved(true);

        BranchPolicy policy = new BranchPolicy(
                2,
                true,
                true,
                false,
                false,
                true
        );

        MergeEligibilityService eligibilityService =
                new MergeEligibilityService(policy);

        boolean eligible = eligibilityService.isEligible(pullRequest);

        System.out.println("UNSUPERVISED LEARNING + GOVERNANCE MODEL");
        System.out.println("========================================");
        System.out.println("Clusters:");

        Map<Integer, List<String>> clusters =
                new KMeansService(3, 100, 1e-7).cluster(observations);

        clusters.forEach((cluster, members) ->
                System.out.println("Cluster " + cluster + ": " + members));

        System.out.println("\nPull Request #" + pullRequest.number());
        System.out.println("Merge state: " + pullRequest.state());

        if (!eligible) {
            System.out.println("Blocking conditions:");
            eligibilityService.blockers(pullRequest)
                    .forEach(reason -> System.out.println(" - " + reason));
        } else {
            System.out.println("All branch-policy conditions are satisfied.");
        }

        // Streams make policy predicates explicit. Records provide immutable
        // domain values while mutable PullRequest state is isolated to one class.
    }
}
