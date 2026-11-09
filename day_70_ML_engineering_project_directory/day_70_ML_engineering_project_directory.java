import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Enterprise classical machine learning governance system.
 *
 * Scenario:
 * Controlled development, evaluation, approval, registration,
 * deployment, and rollback of customer churn models.
 *
 * Java version: 17
 *
 * Compile:
 * javac MLEnterprisePipeline.java
 *
 * Run:
 * java MLEnterprisePipeline
 */
public class MLEnterprisePipeline {

    // ---------------------------------------------------
    // Enumerations
    // ---------------------------------------------------

    enum RunState {
        CREATED,
        DATA_VALIDATED,
        TRAINING,
        TRAINED,
        EVALUATED,
        APPROVED,
        REJECTED,
        DEPLOYING,
        DEPLOYED,
        FAILED
    }

    enum ReviewDecision {
        APPROVE,
        REQUEST_CHANGES,
        REJECT
    }

    enum Environment {
        DEVELOPMENT,
        STAGING,
        PRODUCTION
    }

    enum CheckResult {
        PASS,
        FAIL,
        WARNING
    }

    // ---------------------------------------------------
    // Exceptions
    // ---------------------------------------------------

    static class PipelineException extends RuntimeException {
        PipelineException(String message) {
            super(message);
        }
    }

    static class ContractException extends PipelineException {
        ContractException(String message) {
            super(message);
        }
    }

    static class GovernanceException extends PipelineException {
        GovernanceException(String message) {
            super(message);
        }
    }

    // ---------------------------------------------------
    // Domain records
    // ---------------------------------------------------

    record Dataset(
        String datasetId,
        List<Observation> observations,
        Instant createdAt
    ) {
        Dataset {
            Objects.requireNonNull(datasetId);
            observations = List.copyOf(observations);
            Objects.requireNonNull(createdAt);
        }
    }

    record Observation(
        int customerId,
        double tenure,
        double monthlyCharges,
        double supportCalls,
        int actual,
        double probability,
        String segment
    ) {
        Observation {
            if (customerId <= 0) {
                throw new ContractException(
                    "Invalid customer identifier."
                );
            }

            if (
                !Double.isFinite(tenure) ||
                !Double.isFinite(monthlyCharges) ||
                !Double.isFinite(supportCalls) ||
                !Double.isFinite(probability)
            ) {
                throw new ContractException(
                    "Non-finite observation values."
                );
            }

            if (
                tenure < 0 ||
                monthlyCharges < 0 ||
                supportCalls < 0
            ) {
                throw new ContractException(
                    "Negative feature values."
                );
            }

            if (actual != 0 && actual != 1) {
                throw new ContractException(
                    "Target must be binary."
                );
            }

            if (probability < 0 || probability > 1) {
                throw new ContractException(
                    "Probability must be between 0 and 1."
                );
            }

            Objects.requireNonNull(segment);
        }
    }

    record ModelMetrics(
        double accuracy,
        double precision,
        double recall,
        double f1,
        double rocAuc,
        double drift,
        double latencyMs,
        double fairnessGap
    ) {
        ModelMetrics {
            double[] values = {
                accuracy,
                precision,
                recall,
                f1,
                rocAuc
            };

            for (double value : values) {
                if (!Double.isFinite(value) ||
                    value < 0 || value > 1) {
                    throw new ContractException(
                        "Invalid classification metric."
                    );
                }
            }

            if (
                !Double.isFinite(drift) ||
                drift < 0 ||
                !Double.isFinite(latencyMs) ||
                latencyMs < 0 ||
                !Double.isFinite(fairnessGap) ||
                fairnessGap < 0 ||
                fairnessGap > 1
            ) {
                throw new ContractException(
                    "Invalid operational metric."
                );
            }
        }
    }

    record QualityResult(
        String policy,
        CheckResult result,
        String details
    ) {}

    record Review(
        String reviewerId,
        ReviewDecision decision,
        String comments,
        Instant reviewedAt
    ) {
        Review {
            Objects.requireNonNull(reviewerId);
            Objects.requireNonNull(decision);
            Objects.requireNonNull(comments);
            Objects.requireNonNull(reviewedAt);
        }
    }

    record ModelVersion(
        String version,
        String runId,
        String datasetId,
        ModelMetrics metrics,
        Instant registeredAt,
        Environment environment
    ) {}

    record AuditEvent(
        String eventId,
        String actor,
        String action,
        String details,
        Instant timestamp
    ) {}

    // ---------------------------------------------------
    // Dataset validation
    // ---------------------------------------------------

    static class DatasetValidator {

        private static final int MINIMUM_ROWS = 50;

        public void validate(Dataset dataset) {
            if (dataset.observations().size() < MINIMUM_ROWS) {
                throw new ContractException(
                    "Insufficient dataset size."
                );
            }

            Set<Integer> identifiers = new HashSet<>();

            long positive = dataset.observations()
                .stream()
                .filter(row -> row.actual() == 1)
                .count();

            long negative = dataset.observations()
                .stream()
                .filter(row -> row.actual() == 0)
                .count();

            for (Observation row : dataset.observations()) {
                if (!identifiers.add(row.customerId())) {
                    throw new ContractException(
                        "Duplicate customer identifier."
                    );
                }
            }

            if (positive == 0 || negative == 0) {
                throw new ContractException(
                    "Both target classes are required."
                );
            }
        }
    }

    // ---------------------------------------------------
    // Model evaluation service
    // ---------------------------------------------------

    static class EvaluationService {

        public ModelMetrics evaluate(
            List<Observation> observations,
            double threshold,
            double drift,
            double latency,
            double fairnessGap
        ) {
            if (
                observations.isEmpty() ||
                threshold < 0 ||
                threshold > 1
            ) {
                throw new ContractException(
                    "Invalid evaluation input."
                );
            }

            int tp = 0;
            int tn = 0;
            int fp = 0;
            int fn = 0;

            for (Observation row : observations) {
                int predicted =
                    row.probability() >= threshold ? 1 : 0;

                if (row.actual() == 1 && predicted == 1) tp++;
                else if (row.actual() == 0 && predicted == 0) tn++;
                else if (row.actual() == 0 && predicted == 1) fp++;
                else fn++;
            }

            double precision =
                tp + fp == 0 ? 0 :
                (double) tp / (tp + fp);

            double recall =
                tp + fn == 0 ? 0 :
                (double) tp / (tp + fn);

            double f1 =
                precision + recall == 0 ? 0 :
                2 * precision * recall /
                (precision + recall);

            double accuracy =
                (double) (tp + tn) / observations.size();

            double auc = calculateAuc(observations);

            return new ModelMetrics(
                accuracy,
                precision,
                recall,
                f1,
                auc,
                drift,
                latency,
                fairnessGap
            );
        }

        private double calculateAuc(
            List<Observation> observations
        ) {
            List<Observation> sorted =
                observations.stream()
                    .sorted((a, b) ->
                        Double.compare(
                            a.probability(),
                            b.probability()
                        )
                    )
                    .toList();

            long positives = sorted.stream()
                .filter(row -> row.actual() == 1)
                .count();

            long negatives = sorted.size() - positives;

            if (positives == 0 || negatives == 0) {
                throw new ContractException(
                    "ROC-AUC requires both classes."
                );
            }

            double rankSum = 0;
            int index = 0;

            while (index < sorted.size()) {
                int end = index + 1;

                while (
                    end < sorted.size() &&
                    Double.compare(
                        sorted.get(end).probability(),
                        sorted.get(index).probability()
                    ) == 0
                ) {
                    end++;
                }

                double averageRank =
                    ((index + 1) + end) / 2.0;

                for (int j = index; j < end; j++) {
                    if (sorted.get(j).actual() == 1) {
                        rankSum += averageRank;
                    }
                }

                index = end;
            }

            return (
                rankSum -
                positives * (positives + 1) / 2.0
            ) / (positives * negatives);
        }
    }

    // ---------------------------------------------------
    // Approval policy
    // ---------------------------------------------------

    static class ApprovalPolicy {

        private final int requiredApprovals;
        private final Set<String> eligibleReviewers;
        private final Set<String> excludedAuthors;

        ApprovalPolicy(
            int requiredApprovals,
            Set<String> eligibleReviewers,
            Set<String> excludedAuthors
        ) {
            if (requiredApprovals < 1) {
                throw new GovernanceException(
                    "At least one approval is required."
                );
            }

            this.requiredApprovals = requiredApprovals;

            this.eligibleReviewers =
                Set.copyOf(eligibleReviewers);

            this.excludedAuthors =
                Set.copyOf(excludedAuthors);
        }

        public boolean isApproved(List<Review> reviews) {
            Map<String, ReviewDecision> latest = new HashMap<>();

            reviews.stream()
                .sorted((a, b) ->
                    a.reviewedAt().compareTo(b.reviewedAt())
                )
                .forEach(review ->
                    latest.put(
                        review.reviewerId(),
                        review.decision()
                    )
                );

            long approvals = latest.entrySet()
                .stream()
                .filter(entry ->
                    eligibleReviewers.contains(entry.getKey())
                )
                .filter(entry ->
                    !excludedAuthors.contains(entry.getKey())
                )
                .filter(entry ->
                    entry.getValue() == ReviewDecision.APPROVE
                )
                .count();

            return approvals >= requiredApprovals;
        }

        public void validateReviews(List<Review> reviews) {
            Set<String> reviewers = new HashSet<>();

            for (Review review : reviews) {
                if (!eligibleReviewers.contains(review.reviewerId())) {
                    throw new GovernanceException(
                        "Ineligible reviewer: " +
                        review.reviewerId()
                    );
                }

                if (excludedAuthors.contains(review.reviewerId())) {
                    throw new GovernanceException(
                        "Authors cannot approve their own model."
                    );
                }

                if (!reviewers.add(review.reviewerId())) {
                    throw new GovernanceException(
                        "Duplicate reviewer entry."
                    );
                }
            }
        }
    }

    // ---------------------------------------------------
    // Release policy
    // ---------------------------------------------------

    static class ReleasePolicy {

        private final double minimumAuc;
        private final double minimumRecall;
        private final double maximumDrift;
        private final double maximumLatency;
        private final double maximumFairnessGap;

        ReleasePolicy(
            double minimumAuc,
            double minimumRecall,
            double maximumDrift,
            double maximumLatency,
            double maximumFairnessGap
        ) {
            this.minimumAuc = minimumAuc;
            this.minimumRecall = minimumRecall;
            this.maximumDrift = maximumDrift;
            this.maximumLatency = maximumLatency;
            this.maximumFairnessGap = maximumFairnessGap;
        }

        public List<QualityResult> evaluate(
            ModelMetrics metrics
        ) {
            List<QualityResult> results = new ArrayList<>();

            add(
                results,
                "ROC-AUC",
                metrics.rocAuc() >= minimumAuc
            );

            add(
                results,
                "Recall",
                metrics.recall() >= minimumRecall
            );

            add(
                results,
                "Feature drift",
                metrics.drift() <= maximumDrift
            );

            add(
                results,
                "Inference latency",
                metrics.latencyMs() <= maximumLatency
            );

            add(
                results,
                "Fairness gap",
                metrics.fairnessGap() <= maximumFairnessGap
            );

            return List.copyOf(results);
        }

        private void add(
            List<QualityResult> results,
            String policy,
            boolean passed
        ) {
            results.add(new QualityResult(
                policy,
                passed ? CheckResult.PASS : CheckResult.FAIL,
                passed ? "Requirement satisfied." :
                         "Release requirement not satisfied."
            ));
        }

        public boolean isEligible(ModelMetrics metrics) {
            return evaluate(metrics)
                .stream()
                .allMatch(result ->
                    result.result() == CheckResult.PASS
                );
        }
    }

    // ---------------------------------------------------
    // State machine
    // ---------------------------------------------------

    static class RunStateMachine {

        private static final Map<RunState, Set<RunState>>
            TRANSITIONS = Map.of(
                RunState.CREATED,
                Set.of(RunState.DATA_VALIDATED, RunState.FAILED),

                RunState.DATA_VALIDATED,
                Set.of(RunState.TRAINING, RunState.FAILED),

                RunState.TRAINING,
                Set.of(RunState.TRAINED, RunState.FAILED),

                RunState.TRAINED,
                Set.of(RunState.EVALUATED, RunState.FAILED),

                RunState.EVALUATED,
                Set.of(RunState.APPROVED, RunState.REJECTED),

                RunState.APPROVED,
                Set.of(RunState.DEPLOYING),

                RunState.DEPLOYING,
                Set.of(RunState.DEPLOYED, RunState.FAILED),

                RunState.DEPLOYED,
                Set.of(),

                RunState.REJECTED,
                Set.of(),

                RunState.FAILED,
                Set.of()
            );

        private RunState state = RunState.CREATED;

        public void transition(RunState next) {
            if (
                !TRANSITIONS.getOrDefault(
                    state,
                    Set.of()
                ).contains(next)
            ) {
                throw new GovernanceException(
                    "Invalid transition: " +
                    state + " -> " + next
                );
            }

            System.out.println(
                "RUN STATE: " + state + " -> " + next
            );

            state = next;
        }

        public RunState getState() {
            return state;
        }
    }

    // ---------------------------------------------------
    // Model registry and deployment
    // ---------------------------------------------------

    static class ModelRegistry {

        private final Map<String, ModelVersion> versions =
            new HashMap<>();

        private String productionVersion;

        public void register(ModelVersion model) {
            if (versions.containsKey(model.version())) {
                throw new GovernanceException(
                    "Model version already registered."
                );
            }

            versions.put(model.version(), model);
        }

        public void deploy(
            String version,
            ApprovalPolicy approvalPolicy,
            List<Review> reviews,
            ReleasePolicy releasePolicy
        ) {
            ModelVersion model = versions.get(version);

            if (model == null) {
                throw new GovernanceException(
                    "Unknown model version."
                );
            }

            approvalPolicy.validateReviews(reviews);

            if (!approvalPolicy.isApproved(reviews)) {
                throw new GovernanceException(
                    "Approval requirements not satisfied."
                );
            }

            if (!releasePolicy.isEligible(model.metrics())) {
                throw new GovernanceException(
                    "Model failed release policy."
                );
            }

            productionVersion = version;

            System.out.println(
                "Production model promoted: " + version
            );
        }

        public String getProductionVersion() {
            return productionVersion;
        }

        public ModelVersion get(String version) {
            ModelVersion model = versions.get(version);

            if (model == null) {
                throw new GovernanceException(
                    "Model not found."
                );
            }

            return model;
        }

        public void rollback(String version) {
            ModelVersion model = versions.get(version);

            if (model == null) {
                throw new GovernanceException(
                    "Rollback target does not exist."
                );
            }

            productionVersion = version;

            System.out.println(
                "Rollback completed: " + version
            );
        }
    }

    // ---------------------------------------------------
    // Audit service
    // ---------------------------------------------------

    static class AuditService {

        private final List<AuditEvent> events =
            new ArrayList<>();

        public void record(
            String actor,
            String action,
            String details
        ) {
            events.add(new AuditEvent(
                UUID.randomUUID().toString(),
                actor,
                action,
                details,
                Instant.now()
            ));
        }

        public List<AuditEvent> getEvents() {
            return Collections.unmodifiableList(events);
        }

        public void print() {
            for (AuditEvent event : events) {
                System.out.println(
                    event.timestamp() +
                    " | " +
                    event.actor() +
                    " | " +
                    event.action() +
                    " | " +
                    event.details()
                );
            }
        }
    }

    // ---------------------------------------------------
    // Enterprise workflow service
    // ---------------------------------------------------

    static class MLWorkflowService {

        private final DatasetValidator validator =
            new DatasetValidator();

        private final EvaluationService evaluator =
            new EvaluationService();

        private final ModelRegistry registry =
            new ModelRegistry();

        private final AuditService audit =
            new AuditService();

        private final ReleasePolicy releasePolicy =
            new ReleasePolicy(
                0.70,
                0.50,
                0.20,
                250,
                0.25
            );

        private final ApprovalPolicy approvalPolicy =
            new ApprovalPolicy(
                2,
                Set.of("reviewer-01", "reviewer-02", "reviewer-03"),
                Set.of("author-01")
            );

        public void execute(Dataset dataset) {
            RunStateMachine machine = new RunStateMachine();

            try {
                validator.validate(dataset);

                machine.transition(RunState.DATA_VALIDATED);

                machine.transition(RunState.TRAINING);

                // In production, this stage calls the model-training
                // service and stores its immutable artifact.

                machine.transition(RunState.TRAINED);

                List<Observation> testObservations =
                    dataset.observations();

                ModelMetrics metrics = evaluator.evaluate(
                    testObservations,
                    0.50,
                    0.08,
                    40.0,
                    0.10
                );

                machine.transition(RunState.EVALUATED);

                List<QualityResult> checks =
                    releasePolicy.evaluate(metrics);

                for (QualityResult check : checks) {
                    System.out.println(
                        check.policy() +
                        ": " +
                        check.result()
                    );
                }

                if (!releasePolicy.isEligible(metrics)) {
                    machine.transition(RunState.REJECTED);

                    audit.record(
                        "system",
                        "MODEL_REJECTED",
                        "Release quality gate failed."
                    );

                    return;
                }

                List<Review> reviews = List.of(
                    new Review(
                        "reviewer-01",
                        ReviewDecision.APPROVE,
                        "Validation evidence reviewed.",
                        Instant.now()
                    ),
                    new Review(
                        "reviewer-02",
                        ReviewDecision.APPROVE,
                        "Operational controls reviewed.",
                        Instant.now()
                    )
                );

                approvalPolicy.validateReviews(reviews);

                if (!approvalPolicy.isApproved(reviews)) {
                    throw new GovernanceException(
                        "Insufficient independent approvals."
                    );
                }

                machine.transition(RunState.APPROVED);

                ModelVersion model = new ModelVersion(
                    "churn-1.0.0",
                    UUID.randomUUID().toString(),
                    dataset.datasetId(),
                    metrics,
                    Instant.now(),
                    Environment.STAGING
                );

                registry.register(model);

                audit.record(
                    "system",
                    "MODEL_REGISTERED",
                    model.version()
                );

                machine.transition(RunState.DEPLOYING);

                registry.deploy(
                    model.version(),
                    approvalPolicy,
                    reviews,
                    releasePolicy
                );

                machine.transition(RunState.DEPLOYED);

                audit.record(
                    "system",
                    "MODEL_DEPLOYED",
                    model.version()
                );

                System.out.println(
                    "\nFinal run state: " +
                    machine.getState()
                );

                System.out.println(
                    "Production version: " +
                    registry.getProductionVersion()
                );

            } catch (RuntimeException error) {
                audit.record(
                    "system",
                    "PIPELINE_ERROR",
                    error.getMessage()
                );

                System.err.println(
                    "Pipeline failure: " + error.getMessage()
                );

                throw error;
            }
        }

        public void printAudit() {
            audit.print();
        }

        public ModelRegistry getRegistry() {
            return registry;
        }
    }

    // ---------------------------------------------------
    // Sample data
    // ---------------------------------------------------

    static Dataset createDataset() {
        List<Observation> observations = new ArrayList<>();

        for (int i = 1; i <= 100; i++) {
            int actual = i % 4 == 0 ? 1 : 0;

            double probability = actual == 1
                ? 0.65 + (i % 20) / 100.0
                : 0.05 + (i % 20) / 100.0;

            observations.add(new Observation(
                i,
                10 + i % 60,
                30 + i % 100,
                i % 5,
                actual,
                probability,
                i % 2 == 0 ? "segment-A" : "segment-B"
            ));
        }

        return new Dataset(
            "customer-churn-dataset-v1",
            observations,
            Instant.now()
        );
    }

    // ---------------------------------------------------
    // Main
    // ---------------------------------------------------

    public static void main(String[] args) {
        try {
            MLWorkflowService workflow =
                new MLWorkflowService();

            Dataset dataset = createDataset();

            workflow.execute(dataset);

            System.out.println("\nAUDIT HISTORY");
            workflow.printAudit();

            // Verify invalid transition handling.
            try {
                RunStateMachine stateMachine =
                    new RunStateMachine();

                stateMachine.transition(RunState.DEPLOYED);

            } catch (GovernanceException error) {
                System.out.println(
                    "\nInvalid transition rejected: " +
                    error.getMessage()
                );
            }

        } catch (RuntimeException error) {
            System.err.println(
                "Enterprise workflow terminated: " +
                error.getMessage()
            );

            System.exit(1);
        }
    }
}
