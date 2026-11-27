import java.util.*;
import java.util.stream.Collectors;

/*
 * TRANSFER LEARNING
 *
 * Enterprise transfer-learning management system.
 *
 * Java 17 implementation featuring:
 * - Pretrained model registry
 * - Immutable model metadata
 * - Dataset validation
 * - Feature extraction policies
 * - Fine-tuning policies
 * - Layer freezing
 * - Learning-rate management
 * - Experiment lifecycle
 * - Evaluation and approval
 * - Model deployment governance
 *
 * Compile:
 * javac TransferLearningEnterprise.java
 *
 * Execute:
 * java TransferLearningEnterprise
 */

public class TransferLearningEnterprise {

    // -------------------------------------------------
    // Enumerations
    // -------------------------------------------------

    enum TransferStrategy {
        FEATURE_EXTRACTION,
        FINE_TUNING
    }

    enum ExperimentState {
        CREATED,
        VALIDATED,
        TRAINING,
        EVALUATED,
        APPROVED,
        REJECTED,
        DEPLOYED,
        FAILED
    }

    enum LayerType {
        CONVOLUTION,
        DENSE,
        NORMALIZATION,
        ACTIVATION,
        POOLING
    }

    // -------------------------------------------------
    // Custom exceptions
    // -------------------------------------------------

    static class TransferException extends RuntimeException {
        public TransferException(String message) {
            super(message);
        }
    }

    static void require(boolean condition, String message) {
        if (!condition) {
            throw new TransferException(message);
        }
    }

    static void requirePositive(double value, String name) {
        require(
            Double.isFinite(value) && value > 0,
            name + " must be positive and finite."
        );
    }

    // -------------------------------------------------
    // Immutable layer metadata
    // -------------------------------------------------

    record LayerDefinition(
        String name,
        LayerType type,
        int inputDimension,
        int outputDimension,
        long parameterCount
    ) {
        LayerDefinition {
            require(name != null && !name.isBlank(),
                "Layer name is required.");

            Objects.requireNonNull(type);

            require(inputDimension > 0,
                "Layer input dimension must be positive.");

            require(outputDimension > 0,
                "Layer output dimension must be positive.");

            require(parameterCount > 0,
                "Layer parameter count must be positive.");
        }
    }

    // -------------------------------------------------
    // Pretrained model metadata
    // -------------------------------------------------

    record PretrainedModel(
        String id,
        String name,
        String architecture,
        String sourceDomain,
        List<Integer> inputShape,
        List<LayerDefinition> layers,
        String weightVersion,
        boolean verified
    ) {
        PretrainedModel {
            require(id != null && !id.isBlank(),
                "Model identifier is required.");

            require(name != null && !name.isBlank(),
                "Model name is required.");

            require(sourceDomain != null && !sourceDomain.isBlank(),
                "Source domain is required.");

            inputShape = List.copyOf(inputShape);
            layers = List.copyOf(layers);

            require(!inputShape.isEmpty(),
                "Model input shape cannot be empty.");

            require(
                inputShape.stream().allMatch(value -> value > 0),
                "Invalid input shape."
            );

            require(!layers.isEmpty(),
                "Model must contain layers.");

            int expectedInput = 1;

            for (LayerDefinition layer : layers) {
                require(
                    layer.inputDimension() == expectedInput,
                    "Layer dimensions are not sequentially compatible."
                );

                expectedInput = layer.outputDimension();
            }
        }

        long totalParameters() {
            return layers.stream()
                .mapToLong(LayerDefinition::parameterCount)
                .sum();
        }
    }

    // -------------------------------------------------
    // Dataset metadata
    // -------------------------------------------------

    record TargetDataset(
        String id,
        String name,
        String domain,
        List<Integer> inputShape,
        int samples,
        int classes,
        String version
    ) {
        TargetDataset {
            require(id != null && !id.isBlank(),
                "Dataset identifier is required.");

            require(name != null && !name.isBlank(),
                "Dataset name is required.");

            require(samples > 0,
                "Dataset sample count must be positive.");

            require(classes >= 2,
                "Classification requires at least two classes.");

            inputShape = List.copyOf(inputShape);

            require(!inputShape.isEmpty(),
                "Dataset input shape is required.");

            require(
                inputShape.stream().allMatch(value -> value > 0),
                "Invalid dataset input dimensions."
            );
        }
    }

    // -------------------------------------------------
    // Transfer policy
    // -------------------------------------------------

    record TransferPolicy(
        TransferStrategy strategy,
        int frozenLayers,
        int epochs,
        double backboneLearningRate,
        double headLearningRate,
        int minimumSamples,
        double minimumAccuracy,
        double maximumLoss
    ) {
        TransferPolicy {
            Objects.requireNonNull(strategy);

            require(frozenLayers >= 0,
                "Frozen layer count cannot be negative.");

            require(epochs > 0,
                "Epoch count must be positive.");

            requirePositive(backboneLearningRate,
                "Backbone learning rate");

            requirePositive(headLearningRate,
                "Classification-head learning rate");

            require(minimumSamples > 0,
                "Minimum samples must be positive.");

            require(
                minimumAccuracy >= 0 && minimumAccuracy <= 1,
                "Invalid accuracy threshold."
            );

            require(maximumLoss >= 0 &&
                    Double.isFinite(maximumLoss),
                "Invalid maximum loss.");

            if (strategy == TransferStrategy.FEATURE_EXTRACTION) {
                require(
                    backboneLearningRate <= 1e-12,
                    "Feature extraction must not update backbone weights."
                );
            }
        }
    }

    // -------------------------------------------------
    // Layer training configuration
    // -------------------------------------------------

    record LayerTrainingPlan(
        String layerName,
        boolean trainable,
        double learningRate
    ) {}

    static class TrainingPlanService {

        List<LayerTrainingPlan> build(
            PretrainedModel model,
            TransferPolicy policy
        ) {
            require(
                policy.frozenLayers() <= model.layers().size(),
                "Frozen layers exceed backbone depth."
            );

            return java.util.stream.IntStream
                .range(0, model.layers().size())
                .mapToObj(index -> {
                    LayerDefinition layer = model.layers().get(index);

                    boolean trainable =
                        policy.strategy() == TransferStrategy.FINE_TUNING
                        && index >= policy.frozenLayers();

                    double learningRate = trainable
                        ? policy.backboneLearningRate()
                        : 0.0;

                    return new LayerTrainingPlan(
                        layer.name(),
                        trainable,
                        learningRate
                    );
                })
                .collect(Collectors.toUnmodifiableList());
        }
    }

    // -------------------------------------------------
    // Evaluation results
    // -------------------------------------------------

    record EvaluationMetrics(
        double accuracy,
        double loss,
        int evaluatedSamples,
        double precision,
        double recall
    ) {
        EvaluationMetrics {
            require(
                accuracy >= 0 && accuracy <= 1,
                "Invalid accuracy."
            );

            require(
                loss >= 0 && Double.isFinite(loss),
                "Invalid loss."
            );

            require(evaluatedSamples > 0,
                "Evaluation requires samples.");

            require(
                precision >= 0 && precision <= 1,
                "Invalid precision."
            );

            require(
                recall >= 0 && recall <= 1,
                "Invalid recall."
            );
        }
    }

    // -------------------------------------------------
    // Experiment aggregate
    // -------------------------------------------------

    static class TransferExperiment {

        private final String id = UUID.randomUUID().toString();

        private final PretrainedModel model;
        private final TargetDataset dataset;
        private final TransferPolicy policy;

        private final List<ExperimentState> history =
            new ArrayList<>();

        private ExperimentState state =
            ExperimentState.CREATED;

        private EvaluationMetrics metrics;

        TransferExperiment(
            PretrainedModel model,
            TargetDataset dataset,
            TransferPolicy policy
        ) {
            this.model = Objects.requireNonNull(model);
            this.dataset = Objects.requireNonNull(dataset);
            this.policy = Objects.requireNonNull(policy);

            history.add(state);
        }

        void transition(ExperimentState next) {
            Map<ExperimentState, Set<ExperimentState>> allowed =
                Map.of(
                    ExperimentState.CREATED,
                    Set.of(
                        ExperimentState.VALIDATED,
                        ExperimentState.FAILED
                    ),
                    ExperimentState.VALIDATED,
                    Set.of(
                        ExperimentState.TRAINING,
                        ExperimentState.FAILED
                    ),
                    ExperimentState.TRAINING,
                    Set.of(
                        ExperimentState.EVALUATED,
                        ExperimentState.FAILED
                    ),
                    ExperimentState.EVALUATED,
                    Set.of(
                        ExperimentState.APPROVED,
                        ExperimentState.REJECTED,
                        ExperimentState.FAILED
                    ),
                    ExperimentState.APPROVED,
                    Set.of(
                        ExperimentState.DEPLOYED,
                        ExperimentState.FAILED
                    ),
                    ExperimentState.REJECTED,
                    Set.of(),
                    ExperimentState.DEPLOYED,
                    Set.of(),
                    ExperimentState.FAILED,
                    Set.of()
                );

            require(
                allowed.get(state).contains(next),
                "Invalid transition: " + state + " -> " + next
            );

            state = next;
            history.add(next);
        }

        void setMetrics(EvaluationMetrics metrics) {
            require(
                state == ExperimentState.TRAINING,
                "Metrics can only be recorded during training."
            );

            this.metrics = Objects.requireNonNull(metrics);
        }

        String id() {
            return id;
        }

        PretrainedModel model() {
            return model;
        }

        TargetDataset dataset() {
            return dataset;
        }

        TransferPolicy policy() {
            return policy;
        }

        ExperimentState state() {
            return state;
        }

        EvaluationMetrics metrics() {
            return metrics;
        }

        List<ExperimentState> history() {
            return List.copyOf(history);
        }
    }

    // -------------------------------------------------
    // Compatibility validation
    // -------------------------------------------------

    static class CompatibilityService {

        void validate(
            PretrainedModel model,
            TargetDataset dataset,
            TransferPolicy policy
        ) {
            require(model.verified(),
                "Unverified pretrained weights are prohibited.");

            require(
                model.inputShape().equals(dataset.inputShape()),
                "Target dataset input shape differs from pretrained model."
            );

            require(
                dataset.samples() >= policy.minimumSamples(),
                "Insufficient target training samples."
            );

            require(
                policy.frozenLayers() <= model.layers().size(),
                "Invalid frozen layer configuration."
            );

            require(
                dataset.classes() >= 2,
                "Invalid target classification classes."
            );
        }
    }

    // -------------------------------------------------
    // Enterprise model registry
    // -------------------------------------------------

    static class ModelRegistry {

        private final Map<String, PretrainedModel> models =
            new HashMap<>();

        void register(PretrainedModel model) {
            require(model.verified(),
                "Model verification is required.");

            require(
                !models.containsKey(model.id()),
                "Duplicate model identifier."
            );

            models.put(model.id(), model);
        }

        PretrainedModel find(String id) {
            PretrainedModel model = models.get(id);

            require(model != null, "Pretrained model not found.");

            return model;
        }

        List<PretrainedModel> list() {
            return List.copyOf(models.values());
        }
    }

    // -------------------------------------------------
    // Experiment service
    // -------------------------------------------------

    static class ExperimentService {

        private final CompatibilityService compatibility =
            new CompatibilityService();

        private final TrainingPlanService trainingPlans =
            new TrainingPlanService();

        private final Map<String, TransferExperiment> experiments =
            new HashMap<>();

        TransferExperiment create(
            PretrainedModel model,
            TargetDataset dataset,
            TransferPolicy policy
        ) {
            TransferExperiment experiment =
                new TransferExperiment(model, dataset, policy);

            require(
                !experiments.containsKey(experiment.id()),
                "Duplicate experiment."
            );

            experiments.put(experiment.id(), experiment);

            return experiment;
        }

        List<LayerTrainingPlan> validateAndPrepare(
            TransferExperiment experiment
        ) {
            compatibility.validate(
                experiment.model(),
                experiment.dataset(),
                experiment.policy()
            );

            List<LayerTrainingPlan> plan =
                trainingPlans.build(
                    experiment.model(),
                    experiment.policy()
                );

            experiment.transition(ExperimentState.VALIDATED);

            return plan;
        }

        void execute(
            TransferExperiment experiment,
            EvaluationMetrics metrics
        ) {
            require(
                experiments.get(experiment.id()) == experiment,
                "Experiment is not registered."
            );

            try {
                if (experiment.state() == ExperimentState.CREATED) {
                    validateAndPrepare(experiment);
                }

                experiment.transition(ExperimentState.TRAINING);

                require(
                    metrics.evaluatedSamples() <=
                        experiment.dataset().samples(),
                    "Evaluation sample count exceeds dataset size."
                );

                experiment.setMetrics(metrics);

                experiment.transition(ExperimentState.EVALUATED);

            } catch (RuntimeException exception) {
                if (
                    experiment.state() != ExperimentState.FAILED &&
                    experiment.state() != ExperimentState.REJECTED
                ) {
                    experiment.transition(ExperimentState.FAILED);
                }

                throw exception;
            }
        }

        boolean approve(TransferExperiment experiment) {
            require(
                experiment.state() == ExperimentState.EVALUATED,
                "Experiment must be evaluated before approval."
            );

            EvaluationMetrics metrics = experiment.metrics();
            TransferPolicy policy = experiment.policy();

            boolean eligible =
                metrics.accuracy() >= policy.minimumAccuracy() &&
                metrics.loss() <= policy.maximumLoss() &&
                metrics.evaluatedSamples() >=
                    experiment.dataset().samples() * 0.95;

            experiment.transition(
                eligible
                    ? ExperimentState.APPROVED
                    : ExperimentState.REJECTED
            );

            return eligible;
        }

        void deploy(TransferExperiment experiment) {
            require(
                experiment.state() == ExperimentState.APPROVED,
                "Only approved experiments can be deployed."
            );

            experiment.transition(ExperimentState.DEPLOYED);
        }

        Collection<TransferExperiment> allExperiments() {
            return List.copyOf(experiments.values());
        }
    }

    // -------------------------------------------------
    // Demonstration
    // -------------------------------------------------

    public static void main(String[] args) {

        System.out.println("TRANSFER LEARNING ENTERPRISE SYSTEM");
        System.out.println("===================================");

        ModelRegistry registry = new ModelRegistry();

        List<LayerDefinition> layers = List.of(
            new LayerDefinition(
                "conv_1",
                LayerType.CONVOLUTION,
                3,
                64,
                12000
            ),
            new LayerDefinition(
                "conv_2",
                LayerType.CONVOLUTION,
                64,
                128,
                48000
            ),
            new LayerDefinition(
                "feature_projection",
                LayerType.DENSE,
                128,
                32,
                4096
            )
        );

        PretrainedModel pretrained = new PretrainedModel(
            "MODEL-001",
            "IndustrialVisionBackbone",
            "Convolutional Neural Network",
            "General Image Recognition",
            List.of(224, 224, 3),
            layers,
            "2.0.0",
            true
        );

        registry.register(pretrained);

        System.out.println("\nRegistered model: " + pretrained.name());

        System.out.println(
            "Parameters: " + pretrained.totalParameters()
        );

        TargetDataset dataset = new TargetDataset(
            "DATASET-001",
            "ManufacturingDefectImages",
            "Industrial Quality Control",
            List.of(224, 224, 3),
            5000,
            4,
            "2026.01"
        );

        TransferPolicy featureExtractionPolicy = new TransferPolicy(
            TransferStrategy.FEATURE_EXTRACTION,
            3,
            20,
            0.0,
            0.001,
            500,
            0.80,
            0.50
        );

        TransferPolicy fineTuningPolicy = new TransferPolicy(
            TransferStrategy.FINE_TUNING,
            2,
            30,
            0.0001,
            0.001,
            500,
            0.85,
            0.30
        );

        ExperimentService service = new ExperimentService();

        // Feature extraction experiment.
        TransferExperiment extraction =
            service.create(
                registry.find("MODEL-001"),
                dataset,
                featureExtractionPolicy
            );

        System.out.println("\nFEATURE EXTRACTION PLAN");

        List<LayerTrainingPlan> extractionPlan =
            service.validateAndPrepare(extraction);

        extractionPlan.forEach(plan ->
            System.out.println(
                plan.layerName() +
                " | Trainable: " + plan.trainable()
            )
        );

        service.execute(
            extraction,
            new EvaluationMetrics(
                0.86,
                0.28,
                5000,
                0.85,
                0.84
            )
        );

        System.out.println(
            "Feature extraction state: " + extraction.state()
        );

        // Fine-tuning experiment.
        TransferExperiment fineTuning =
            service.create(
                registry.find("MODEL-001"),
                dataset,
                fineTuningPolicy
            );

        System.out.println("\nFINE-TUNING PLAN");

        List<LayerTrainingPlan> fineTuningPlan =
            service.validateAndPrepare(fineTuning);

        fineTuningPlan.forEach(plan ->
            System.out.println(
                plan.layerName() +
                " | Trainable: " + plan.trainable() +
                " | Learning rate: " + plan.learningRate()
            )
        );

        service.execute(
            fineTuning,
            new EvaluationMetrics(
                0.92,
                0.18,
                5000,
                0.91,
                0.90
            )
        );

        boolean approved = service.approve(fineTuning);

        System.out.println("\nFine-tuning approved: " + approved);

        if (approved) {
            service.deploy(fineTuning);
        }

        System.out.println(
            "Final experiment state: " + fineTuning.state()
        );

        System.out.println("\nEXPERIMENT HISTORY");

        fineTuning.history().forEach(System.out::println);

        System.out.println("\nREGISTERED MODELS");

        registry.list().forEach(model ->
            System.out.println(
                model.name() + " | " +
                model.architecture() + " | " +
                model.totalParameters() + " parameters"
            )
        );
    }
}
