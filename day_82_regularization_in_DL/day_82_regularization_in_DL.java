import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/*
 * Regularization in Deep Learning
 *
 * Enterprise-oriented regularization experiment management.
 *
 * Demonstrates:
 * - Dropout policy configuration.
 * - L2 and decoupled weight decay distinctions.
 * - Data augmentation policies.
 * - Validation-based early stopping.
 * - Immutable experiment configurations.
 * - Explicit model lifecycle states.
 * - Checkpoint restoration.
 * - Training governance and experiment reporting.
 *
 * Java version: 17
 */

public class RegularizationEnterprise {

    enum RegularizationMethod {
        DROPOUT,
        L2,
        WEIGHT_DECAY,
        DATA_AUGMENTATION,
        EARLY_STOPPING
    }

    enum ExperimentState {
        CREATED,
        VALIDATING,
        TRAINING,
        EARLY_STOPPED,
        COMPLETED,
        FAILED
    }

    enum OptimizerType {
        SGD,
        ADAM,
        ADAMW
    }

    static class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    record RegularizationPolicy(
            double dropoutRate,
            double weightDecay,
            double augmentationNoise,
            int patience,
            double minimumImprovement,
            int maximumEpochs,
            OptimizerType optimizer
    ) {
        RegularizationPolicy {
            Objects.requireNonNull(optimizer);

            if (dropoutRate < 0 || dropoutRate >= 1) {
                throw new ValidationException("Invalid dropout rate.");
            }

            if (weightDecay < 0 || augmentationNoise < 0) {
                throw new ValidationException("Negative regularization value.");
            }

            if (patience <= 0 || maximumEpochs <= 0) {
                throw new ValidationException("Invalid epoch configuration.");
            }

            if (minimumImprovement < 0 ||
                    !Double.isFinite(minimumImprovement)) {
                throw new ValidationException("Invalid improvement threshold.");
            }

            if (!Double.isFinite(weightDecay) ||
                    !Double.isFinite(augmentationNoise)) {
                throw new ValidationException("Non-finite policy value.");
            }

            if (optimizer == OptimizerType.ADAMW &&
                    weightDecay > 0) {
                // AdamW applies decay separately from adaptive gradients.
            }
        }

        boolean usesDecoupledWeightDecay() {
            return optimizer == OptimizerType.ADAMW;
        }
    }

    record EpochMetrics(
            int epoch,
            double trainingLoss,
            double validationLoss,
            double validationAccuracy
    ) {
        EpochMetrics {
            if (epoch <= 0 ||
                    !Double.isFinite(trainingLoss) ||
                    !Double.isFinite(validationLoss) ||
                    !Double.isFinite(validationAccuracy)) {
                throw new ValidationException("Invalid epoch metrics.");
            }

            if (trainingLoss < 0 || validationLoss < 0 ||
                    validationAccuracy < 0 || validationAccuracy > 1) {
                throw new ValidationException("Metric outside valid range.");
            }
        }
    }

    record ModelCheckpoint(
            int epoch,
            List<Double> parameters,
            double validationLoss
    ) {
        ModelCheckpoint {
            if (epoch <= 0 || validationLoss < 0) {
                throw new ValidationException("Invalid checkpoint.");
            }

            parameters = List.copyOf(parameters);
        }
    }

    record ExperimentReport(
            String experimentName,
            ExperimentState finalState,
            int bestEpoch,
            int stoppedEpoch,
            double bestValidationLoss,
            double validationAccuracy,
            List<EpochMetrics> history
    ) {
        ExperimentReport {
            history = List.copyOf(history);
        }
    }

    interface AugmentationStrategy {
        double[] transform(double[] features, Random random);
    }

    static class GaussianAugmentation implements AugmentationStrategy {
        private final double standardDeviation;

        GaussianAugmentation(double standardDeviation) {
            if (standardDeviation < 0 ||
                    !Double.isFinite(standardDeviation)) {
                throw new ValidationException("Invalid noise level.");
            }

            this.standardDeviation = standardDeviation;
        }

        @Override
        public double[] transform(double[] features, Random random) {
            return Arrays.stream(features)
                    .map(value -> value +
                            random.nextGaussian() * standardDeviation)
                    .toArray();
        }
    }

    static class DropoutService {

        double[] apply(
                double[] activations,
                double dropoutRate,
                boolean training,
                Random random
        ) {
            if (dropoutRate < 0 || dropoutRate >= 1) {
                throw new ValidationException("Invalid dropout probability.");
            }

            double[] output = new double[activations.length];

            if (!training || dropoutRate == 0) {
                return activations.clone();
            }

            double scale = 1.0 / (1.0 - dropoutRate);

            for (int i = 0; i < activations.length; i++) {
                output[i] = random.nextDouble() >= dropoutRate
                        ? activations[i] * scale
                        : 0.0;
            }

            return output;
        }
    }

    static class ParameterRegularizer {

        /*
         * L2 contributes lambda * weight to the gradient.
         * Biases are excluded from the penalty.
         */
        void addL2Gradient(
                double[] weights,
                double[] gradients,
                double lambda
        ) {
            if (weights.length != gradients.length || lambda < 0) {
                throw new ValidationException("Invalid L2 parameters.");
            }

            for (int i = 0; i < weights.length; i++) {
                gradients[i] += lambda * weights[i];
            }
        }

        /*
         * Decoupled weight decay modifies parameters independently
         * of the adaptive gradient update.
         */
        void applyDecoupledWeightDecay(
                double[] weights,
                double learningRate,
                double decay
        ) {
            if (learningRate <= 0 || decay < 0) {
                throw new ValidationException("Invalid decay configuration.");
            }

            double factor = 1.0 - learningRate * decay;

            if (factor < 0) {
                throw new ValidationException("Negative decay factor.");
            }

            for (int i = 0; i < weights.length; i++) {
                weights[i] *= factor;
            }
        }
    }

    static class EarlyStoppingService {

        private final int patience;
        private final double minDelta;

        private double bestLoss = Double.POSITIVE_INFINITY;
        private int bestEpoch = 0;
        private int staleEpochs = 0;
        private ModelCheckpoint checkpoint;

        EarlyStoppingService(int patience, double minDelta) {
            if (patience <= 0 || minDelta < 0) {
                throw new ValidationException("Invalid early stopping policy.");
            }

            this.patience = patience;
            this.minDelta = minDelta;
        }

        boolean observe(
                int epoch,
                double validationLoss,
                double[] parameters
        ) {
            if (!Double.isFinite(validationLoss) || validationLoss < 0) {
                throw new ValidationException("Invalid validation loss.");
            }

            if (validationLoss < bestLoss - minDelta) {
                bestLoss = validationLoss;
                bestEpoch = epoch;

                checkpoint = new ModelCheckpoint(
                        epoch,
                        Arrays.stream(parameters).boxed().toList(),
                        validationLoss
                );

                staleEpochs = 0;
            } else {
                staleEpochs++;
            }

            return staleEpochs >= patience;
        }

        ModelCheckpoint restore() {
            if (checkpoint == null) {
                throw new IllegalStateException("No checkpoint available.");
            }

            return checkpoint;
        }

        int getBestEpoch() {
            return bestEpoch;
        }

        double getBestLoss() {
            return bestLoss;
        }
    }

    static class Experiment {

        private final String name;
        private final RegularizationPolicy policy;
        private final Random random;

        private final DropoutService dropoutService =
                new DropoutService();

        private final ParameterRegularizer regularizer =
                new ParameterRegularizer();

        private final AugmentationStrategy augmenter;

        private final EarlyStoppingService earlyStopping;

        private final List<EpochMetrics> history =
                new ArrayList<>();

        private ExperimentState state = ExperimentState.CREATED;

        Experiment(
                String name,
                RegularizationPolicy policy,
                long seed
        ) {
            if (name == null || name.isBlank()) {
                throw new ValidationException("Experiment name is required.");
            }

            this.name = name;
            this.policy = Objects.requireNonNull(policy);
            this.random = new Random(seed);

            this.augmenter = new GaussianAugmentation(
                    policy.augmentationNoise()
            );

            this.earlyStopping = new EarlyStoppingService(
                    policy.patience(),
                    policy.minimumImprovement()
            );
        }

        private void transition(ExperimentState next) {
            Map<ExperimentState, Set<ExperimentState>> allowed =
                    Map.of(
                            ExperimentState.CREATED,
                            Set.of(ExperimentState.VALIDATING,
                                    ExperimentState.FAILED),

                            ExperimentState.VALIDATING,
                            Set.of(ExperimentState.TRAINING,
                                    ExperimentState.FAILED),

                            ExperimentState.TRAINING,
                            Set.of(ExperimentState.EARLY_STOPPED,
                                    ExperimentState.COMPLETED,
                                    ExperimentState.FAILED),

                            ExperimentState.EARLY_STOPPED,
                            Set.of(),

                            ExperimentState.COMPLETED,
                            Set.of(),

                            ExperimentState.FAILED,
                            Set.of()
                    );

            if (!allowed.get(state).contains(next)) {
                throw new IllegalStateException(
                        "Invalid state transition: " + state + " -> " + next
                );
            }

            state = next;
        }

        ExperimentReport execute() {
            try {
                transition(ExperimentState.VALIDATING);

                if (policy.optimizer() == OptimizerType.ADAM &&
                        policy.weightDecay() > 0) {
                    System.out.println(
                            "Weight decay uses gradient-based L2 semantics."
                    );
                }

                transition(ExperimentState.TRAINING);

                double[] parameters = {0.8, -0.5, 0.4, 0.9};

                boolean stopped = false;

                for (int epoch = 1;
                     epoch <= policy.maximumEpochs();
                     epoch++) {

                    double[] input = augmenter.transform(
                            new double[]{0.4, 0.7, -0.2},
                            random
                    );

                    double[] hidden = dropoutService.apply(
                            input,
                            policy.dropoutRate(),
                            true,
                            random
                    );

                    // Illustrative optimization update.
                    for (int i = 0; i < parameters.length; i++) {
                        double gradient =
                                parameters[i] * 0.02 +
                                        hidden[i % hidden.length] * 0.001;

                        parameters[i] -= 0.03 * gradient;
                    }

                    if (policy.usesDecoupledWeightDecay()) {
                        regularizer.applyDecoupledWeightDecay(
                                parameters,
                                0.03,
                                policy.weightDecay()
                        );
                    } else if (policy.weightDecay() > 0) {
                        double[] gradients = new double[parameters.length];

                        regularizer.addL2Gradient(
                                parameters,
                                gradients,
                                policy.weightDecay()
                        );

                        for (int i = 0; i < parameters.length; i++) {
                            parameters[i] -= 0.03 * gradients[i];
                        }
                    }

                    // Deterministic validation curve for policy evaluation.
                    double validationLoss =
                            0.12 +
                            0.45 * Math.exp(-epoch / 5.0) +
                            Math.max(0, epoch - 15) * 0.004;

                    double trainingLoss =
                            validationLoss * 0.82;

                    double accuracy = Math.max(
                            0.0,
                            Math.min(1.0, 0.65 + epoch * 0.012)
                    );

                    EpochMetrics metrics = new EpochMetrics(
                            epoch,
                            trainingLoss,
                            validationLoss,
                            accuracy
                    );

                    history.add(metrics);

                    stopped = earlyStopping.observe(
                            epoch,
                            validationLoss,
                            parameters
                    );

                    if (stopped) {
                        transition(ExperimentState.EARLY_STOPPED);
                        break;
                    }
                }

                if (!stopped) {
                    transition(ExperimentState.COMPLETED);
                }

                // Restore the best validation checkpoint.
                ModelCheckpoint checkpoint = earlyStopping.restore();

                return new ExperimentReport(
                        name,
                        state,
                        checkpoint.epoch(),
                        history.size(),
                        checkpoint.validationLoss(),
                        history.get(checkpoint.epoch() - 1).validationAccuracy(),
                        history
                );

            } catch (RuntimeException exception) {
                if (state != ExperimentState.FAILED &&
                        state != ExperimentState.COMPLETED &&
                        state != ExperimentState.EARLY_STOPPED) {
                    state = ExperimentState.FAILED;
                }

                throw exception;
            }
        }
    }

    static class ExperimentService {

        private final Map<String, RegularizationPolicy> policies =
                new LinkedHashMap<>();

        private final AtomicLong experimentCounter =
                new AtomicLong(0);

        void register(String name, RegularizationPolicy policy) {
            if (name == null || name.isBlank()) {
                throw new ValidationException("Policy name is required.");
            }

            if (policies.putIfAbsent(name, policy) != null) {
                throw new ValidationException(
                        "Duplicate policy name: " + name
                );
            }
        }

        List<ExperimentReport> executeAll() {
            List<ExperimentReport> reports = new ArrayList<>();

            for (var entry : policies.entrySet()) {
                long id = experimentCounter.incrementAndGet();

                System.out.println(
                        "\nExperiment " + id + ": " + entry.getKey()
                );

                Experiment experiment = new Experiment(
                        entry.getKey(),
                        entry.getValue(),
                        42 + id
                );

                reports.add(experiment.execute());
            }

            return reports;
        }
    }

    public static void main(String[] args) {

        ExperimentService service = new ExperimentService();

        service.register(
                "Baseline",
                new RegularizationPolicy(
                        0, 0, 0, 10, 0.00001, 80,
                        OptimizerType.SGD
                )
        );

        service.register(
                "Dropout",
                new RegularizationPolicy(
                        0.30, 0, 0, 10, 0.00001, 80,
                        OptimizerType.SGD
                )
        );

        service.register(
                "L2 Regularization",
                new RegularizationPolicy(
                        0, 0.01, 0, 10, 0.00001, 80,
                        OptimizerType.ADAM
                )
        );

        service.register(
                "AdamW Combined",
                new RegularizationPolicy(
                        0.20, 0.005, 0.04, 10, 0.00001, 80,
                        OptimizerType.ADAMW
                )
        );

        List<ExperimentReport> reports = service.executeAll();

        reports.sort(
                Comparator.comparingDouble(
                        ExperimentReport::bestValidationLoss
                )
        );

        System.out.println("\nENTERPRISE EXPERIMENT REPORT");
        System.out.println("-----------------------------------------------");

        for (ExperimentReport report : reports) {
            System.out.printf(
                    Locale.ROOT,
                    "%-22s Best Epoch: %3d | Loss: %.5f | Accuracy: %.2f%% | State: %s%n",
                    report.experimentName(),
                    report.bestEpoch(),
                    report.bestValidationLoss(),
                    report.validationAccuracy() * 100,
                    report.finalState()
            );
        }

        System.out.println("\nSelected configuration: " +
                reports.get(0).experimentName());
    }
}
