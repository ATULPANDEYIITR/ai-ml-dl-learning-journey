import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/*
 * Enterprise Batch Normalization Service.
 *
 * Domain responsibilities:
 * - Training and inference execution.
 * - Versioned normalization configuration.
 * - Running-statistic management.
 * - Immutable execution results.
 * - Backpropagation.
 * - Parameter updates.
 * - Model lifecycle validation.
 */

public class BatchNormalizationEnterprise {

    enum ExecutionMode {
        TRAINING,
        INFERENCE
    }

    enum ModelStatus {
        UNINITIALIZED,
        TRAINING_READY,
        INFERENCE_READY
    }

    static class NormalizationException extends RuntimeException {
        NormalizationException(String message) {
            super(message);
        }
    }

    static class ConfigurationException extends NormalizationException {
        ConfigurationException(String message) {
            super(message);
        }
    }

    static class BatchException extends NormalizationException {
        BatchException(String message) {
            super(message);
        }
    }

    record Configuration(
        int features,
        double epsilon,
        double momentum
    ) {
        Configuration {
            if (features <= 0) {
                throw new ConfigurationException(
                    "Feature count must be positive"
                );
            }

            if (!Double.isFinite(epsilon) || epsilon <= 0) {
                throw new ConfigurationException(
                    "Epsilon must be finite and positive"
                );
            }

            if (
                !Double.isFinite(momentum) ||
                momentum < 0 ||
                momentum >= 1
            ) {
                throw new ConfigurationException(
                    "Momentum must be in [0, 1)"
                );
            }
        }
    }

    record Statistics(
        List<Double> mean,
        List<Double> variance,
        int sampleCount
    ) {
        Statistics {
            mean = List.copyOf(mean);
            variance = List.copyOf(variance);

            if (sampleCount <= 0) {
                throw new BatchException("Invalid sample count");
            }
        }
    }

    record ForwardResult(
        List<List<Double>> output,
        List<List<Double>> normalized,
        List<Double> inverseStd,
        ExecutionMode mode
    ) {
        ForwardResult {
            output = immutableMatrix(output);
            normalized = immutableMatrix(normalized);
            inverseStd = List.copyOf(inverseStd);
        }
    }

    record GradientResult(
        List<List<Double>> inputGradient,
        List<Double> gammaGradient,
        List<Double> betaGradient
    ) {
        GradientResult {
            inputGradient = immutableMatrix(inputGradient);
            gammaGradient = List.copyOf(gammaGradient);
            betaGradient = List.copyOf(betaGradient);
        }
    }

    private static List<List<Double>> immutableMatrix(
        List<List<Double>> matrix
    ) {
        List<List<Double>> result = new ArrayList<>();

        for (List<Double> row : matrix) {
            result.add(List.copyOf(row));
        }

        return Collections.unmodifiableList(result);
    }

    static final class ModelState {
        private final Configuration configuration;

        private List<Double> gamma;
        private List<Double> beta;

        private List<Double> runningMean;
        private List<Double> runningVariance;

        private int batchesSeen;
        private ModelStatus status;

        ModelState(Configuration configuration) {
            this.configuration = Objects.requireNonNull(configuration);

            int features = configuration.features();

            gamma = new ArrayList<>(
                Collections.nCopies(features, 1.0)
            );

            beta = new ArrayList<>(
                Collections.nCopies(features, 0.0)
            );

            runningMean = new ArrayList<>(
                Collections.nCopies(features, 0.0)
            );

            runningVariance = new ArrayList<>(
                Collections.nCopies(features, 1.0)
            );

            batchesSeen = 0;
            status = ModelStatus.UNINITIALIZED;
        }

        Configuration configuration() {
            return configuration;
        }

        ModelStatus status() {
            return status;
        }

        int batchesSeen() {
            return batchesSeen;
        }

        List<Double> gamma() {
            return List.copyOf(gamma);
        }

        List<Double> beta() {
            return List.copyOf(beta);
        }

        List<Double> runningMean() {
            return List.copyOf(runningMean);
        }

        List<Double> runningVariance() {
            return List.copyOf(runningVariance);
        }
    }

    static final class StatisticsService {

        Statistics calculate(
            List<List<Double>> batch,
            int features
        ) {
            int count = batch.size();

            List<Double> mean = new ArrayList<>(
                Collections.nCopies(features, 0.0)
            );

            List<Double> m2 = new ArrayList<>(
                Collections.nCopies(features, 0.0)
            );

            // Online mean and variance computation.
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < features; j++) {
                    double delta =
                        batch.get(i).get(j) - mean.get(j);

                    mean.set(
                        j,
                        mean.get(j) + delta / (i + 1)
                    );

                    double deltaAfter =
                        batch.get(i).get(j) - mean.get(j);

                    m2.set(
                        j,
                        m2.get(j) + delta * deltaAfter
                    );
                }
            }

            List<Double> variance = new ArrayList<>();

            for (int j = 0; j < features; j++) {
                variance.add(
                    Math.max(0.0, m2.get(j) / count)
                );
            }

            return new Statistics(mean, variance, count);
        }
    }

    static final class NormalizationPolicy {

        void validateBatch(
            List<List<Double>> batch,
            int features
        ) {
            if (batch == null || batch.isEmpty()) {
                throw new BatchException("Empty batch");
            }

            for (int i = 0; i < batch.size(); i++) {
                List<Double> row = batch.get(i);

                if (row == null || row.size() != features) {
                    throw new BatchException(
                        "Invalid feature count at row " + i
                    );
                }

                for (Double value : row) {
                    if (
                        value == null ||
                        !Double.isFinite(value)
                    ) {
                        throw new BatchException(
                            "Invalid numerical input"
                        );
                    }
                }
            }
        }

        void validateGradients(
            List<List<Double>> gradients,
            int features
        ) {
            validateBatch(gradients, features);
        }

        void validateLearningRate(double learningRate) {
            if (
                !Double.isFinite(learningRate) ||
                learningRate <= 0
            ) {
                throw new ConfigurationException(
                    "Learning rate must be positive"
                );
            }
        }
    }

    static final class NormalizationService {

        private final ModelState state;
        private final StatisticsService statisticsService;
        private final NormalizationPolicy policy;

        NormalizationService(
            ModelState state,
            StatisticsService statisticsService,
            NormalizationPolicy policy
        ) {
            this.state = Objects.requireNonNull(state);

            this.statisticsService =
                Objects.requireNonNull(statisticsService);

            this.policy = Objects.requireNonNull(policy);
        }

        private void updateRunningStatistics(
            Statistics statistics
        ) {
            double momentum = state.configuration.momentum();

            List<Double> nextMean = new ArrayList<>();
            List<Double> nextVariance = new ArrayList<>();

            for (int j = 0; j < state.configuration.features(); j++) {
                nextMean.add(
                    momentum * state.runningMean.get(j)
                    + (1 - momentum) * statistics.mean().get(j)
                );

                nextVariance.add(
                    momentum * state.runningVariance.get(j)
                    + (1 - momentum) * statistics.variance().get(j)
                );
            }

            // Atomic logical transition after successful calculations.
            state.runningMean = nextMean;
            state.runningVariance = nextVariance;
            state.batchesSeen++;

            state.status = ModelStatus.TRAINING_READY;
        }

        ForwardResult forward(
            List<List<Double>> batch,
            ExecutionMode mode
        ) {
            policy.validateBatch(
                batch,
                state.configuration.features()
            );

            Objects.requireNonNull(mode);

            List<Double> mean;
            List<Double> variance;

            if (mode == ExecutionMode.TRAINING) {
                Statistics statistics =
                    statisticsService.calculate(
                        batch,
                        state.configuration.features()
                    );

                mean = statistics.mean();
                variance = statistics.variance();

                updateRunningStatistics(statistics);

            } else {
                if (state.batchesSeen == 0) {
                    throw new NormalizationException(
                        "Inference is unavailable before training"
                    );
                }

                mean = state.runningMean;
                variance = state.runningVariance;
            }

            int features = state.configuration.features();

            List<Double> inverseStd = new ArrayList<>();

            for (int j = 0; j < features; j++) {
                inverseStd.add(
                    1.0 / Math.sqrt(
                        variance.get(j) + state.configuration.epsilon()
                    )
                );
            }

            List<List<Double>> normalized = new ArrayList<>();
            List<List<Double>> output = new ArrayList<>();

            for (List<Double> row : batch) {
                List<Double> normalizedRow = new ArrayList<>();
                List<Double> outputRow = new ArrayList<>();

                for (int j = 0; j < features; j++) {
                    double normalizedValue =
                        (row.get(j) - mean.get(j)) *
                        inverseStd.get(j);

                    double result =
                        state.gamma.get(j) * normalizedValue
                        + state.beta.get(j);

                    if (!Double.isFinite(result)) {
                        throw new NormalizationException(
                            "Non-finite output detected"
                        );
                    }

                    normalizedRow.add(normalizedValue);
                    outputRow.add(result);
                }

                normalized.add(normalizedRow);
                output.add(outputRow);
            }

            if (mode == ExecutionMode.INFERENCE) {
                state.status = ModelStatus.INFERENCE_READY;
            }

            return new ForwardResult(
                output,
                normalized,
                inverseStd,
                mode
            );
        }

        GradientResult backward(
            List<List<Double>> upstream,
            ForwardResult cache
        ) {
            if (cache.mode() != ExecutionMode.TRAINING) {
                throw new NormalizationException(
                    "Backward pass requires training mode"
                );
            }

            policy.validateGradients(
                upstream,
                state.configuration.features()
            );

            int count = upstream.size();
            int features = state.configuration.features();

            if (cache.normalized().size() != count) {
                throw new BatchException(
                    "Forward and backward batch sizes differ"
                );
            }

            List<Double> dGamma = new ArrayList<>();
            List<Double> dBeta = new ArrayList<>();

            for (int j = 0; j < features; j++) {
                double gammaGradient = 0;
                double betaGradient = 0;

                for (int i = 0; i < count; i++) {
                    double gradient = upstream.get(i).get(j);

                    gammaGradient +=
                        gradient * cache.normalized().get(i).get(j);

                    betaGradient += gradient;
                }

                dGamma.add(gammaGradient);
                dBeta.add(betaGradient);
            }

            List<List<Double>> dInput = new ArrayList<>();

            for (int i = 0; i < count; i++) {
                List<Double> row = new ArrayList<>();

                for (int j = 0; j < features; j++) {
                    double gradient =
                        state.gamma.get(j)
                        * cache.inverseStd().get(j)
                        / count
                        * (
                            count * upstream.get(i).get(j)
                            - dBeta.get(j)
                            - cache.normalized().get(i).get(j)
                            * dGamma.get(j)
                        );

                    row.add(gradient);
                }

                dInput.add(row);
            }

            return new GradientResult(
                dInput,
                dGamma,
                dBeta
            );
        }

        void updateParameters(
            GradientResult gradients,
            double learningRate
        ) {
            policy.validateLearningRate(learningRate);

            int features = state.configuration.features();

            if (
                gradients.gammaGradient().size() != features ||
                gradients.betaGradient().size() != features
            ) {
                throw new BatchException(
                    "Parameter gradient dimensions mismatch"
                );
            }

            List<Double> nextGamma = new ArrayList<>();
            List<Double> nextBeta = new ArrayList<>();

            for (int j = 0; j < features; j++) {
                double gamma = state.gamma.get(j)
                    - learningRate * gradients.gammaGradient().get(j);

                double beta = state.beta.get(j)
                    - learningRate * gradients.betaGradient().get(j);

                if (!Double.isFinite(gamma) || !Double.isFinite(beta)) {
                    throw new NormalizationException(
                        "Invalid parameter update"
                    );
                }

                nextGamma.add(gamma);
                nextBeta.add(beta);
            }

            state.gamma = nextGamma;
            state.beta = nextBeta;
        }
    }

    public static void main(String[] args) {
        Configuration configuration = new Configuration(
            3,
            1e-5,
            0.9
        );

        ModelState model = new ModelState(configuration);

        NormalizationService service = new NormalizationService(
            model,
            new StatisticsService(),
            new NormalizationPolicy()
        );

        List<List<Double>> trainingBatch = List.of(
            List.of(2.0, 4.0, 6.0),
            List.of(4.0, 8.0, 10.0),
            List.of(6.0, 12.0, 14.0),
            List.of(8.0, 16.0, 18.0)
        );

        System.out.println("BATCH NORMALIZATION ENTERPRISE");

        ForwardResult training = service.forward(
            trainingBatch,
            ExecutionMode.TRAINING
        );

        System.out.println("\nTraining output:");

        training.output().forEach(System.out::println);

        GradientResult gradients = service.backward(
            List.of(
                List.of(1.0, 1.0, 1.0),
                List.of(1.0, 1.0, 1.0),
                List.of(1.0, 1.0, 1.0),
                List.of(1.0, 1.0, 1.0)
            ),
            training
        );

        service.updateParameters(gradients, 0.01);

        System.out.println("\nUpdated gamma: " + model.gamma());
        System.out.println("Updated beta: " + model.beta());

        System.out.println(
            "Running mean: " + model.runningMean()
        );

        System.out.println(
            "Running variance: " + model.runningVariance()
        );

        List<Double> previousMean = model.runningMean();
        List<Double> previousVariance = model.runningVariance();

        ForwardResult inference = service.forward(
            List.of(List.of(5.0, 10.0, 12.0)),
            ExecutionMode.INFERENCE
        );

        System.out.println("\nInference output:");
        inference.output().forEach(System.out::println);

        if (
            !previousMean.equals(model.runningMean()) ||
            !previousVariance.equals(model.runningVariance())
        ) {
            throw new IllegalStateException(
                "Inference changed training statistics"
            );
        }

        System.out.println(
            "Inference state isolation verified."
        );

        try {
            service.forward(
                List.of(List.of(1.0, 2.0)),
                ExecutionMode.TRAINING
            );
        } catch (BatchException exception) {
            System.out.println(
                "Invalid batch rejected: " + exception.getMessage()
            );
        }

        try {
            service.forward(
                List.of(
                    Arrays.asList(1.0, Double.NaN, 3.0)
                ),
                ExecutionMode.TRAINING
            );
        } catch (BatchException exception) {
            System.out.println(
                "Invalid numerical input rejected: "
                + exception.getMessage()
            );
        }

        System.out.println(
            "\nModel lifecycle: " + model.status()
        );

        System.out.println(
            "Processed batches: " + model.batchesSeen()
        );
    }
}
