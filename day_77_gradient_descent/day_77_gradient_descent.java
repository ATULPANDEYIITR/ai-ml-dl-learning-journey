import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Predicate;

/*
 * Enterprise Gradient Descent Training System
 *
 * Java 17 implementation featuring:
 * - Batch, stochastic and mini-batch gradient descent
 * - Immutable domain records
 * - Strategy-based optimizer selection
 * - Training policy validation
 * - Enterprise training-state transitions
 * - L2 regularization
 * - Gradient clipping
 * - Evaluation metrics
 * - CSV audit reporting
 * - Custom exceptions
 * - Policy-based convergence monitoring
 */

public class GradientDescentEnterprise {

    enum OptimizerType {
        BATCH,
        STOCHASTIC,
        MINI_BATCH
    }

    enum TrainingStatus {
        CREATED,
        VALIDATED,
        RUNNING,
        COMPLETED,
        EARLY_STOPPED,
        FAILED
    }

    enum ModelStatus {
        INITIALIZED,
        TRAINED,
        EVALUATED
    }

    record Sample(List<Double> features, double target) {

        Sample {
            if (features == null || features.isEmpty()) {
                throw new ValidationException("Features are required.");
            }

            features = List.copyOf(features);

            features.forEach(value -> requireFinite(value, "Feature"));
            requireFinite(target, "Target");
        }
    }

    record TrainingPolicy(
        OptimizerType optimizer,
        double learningRate,
        double regularization,
        double gradientClip,
        double tolerance,
        int epochs,
        int batchSize,
        int patience,
        long seed
    ) {
        TrainingPolicy {
            if (optimizer == null) {
                throw new ValidationException("Optimizer is required.");
            }

            requireFinite(learningRate, "Learning rate");
            requireFinite(regularization, "Regularization");
            requireFinite(gradientClip, "Gradient clipping");
            requireFinite(tolerance, "Tolerance");

            if (learningRate <= 0 ||
                regularization < 0 ||
                gradientClip <= 0 ||
                tolerance < 0) {
                throw new ValidationException("Invalid numeric policy.");
            }

            if (epochs <= 0 || batchSize <= 0 || patience <= 0) {
                throw new ValidationException("Invalid training limits.");
            }
        }
    }

    record Gradients(double[] weights, double bias) {}

    record EpochMetrics(
        int epoch,
        double loss,
        double gradientNorm,
        int batchCount
    ) {}

    record EvaluationMetrics(
        double meanSquaredError,
        double rSquared
    ) {}

    record TrainingReport(
        OptimizerType optimizer,
        List<Double> weights,
        double bias,
        List<EpochMetrics> history,
        TrainingStatus status,
        ModelStatus modelStatus
    ) {
        TrainingReport {
            weights = List.copyOf(weights);
            history = List.copyOf(history);
        }
    }

    static class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    static class TrainingException extends RuntimeException {
        TrainingException(String message) {
            super(message);
        }
    }

    static void requireFinite(double value, String field) {
        if (!Double.isFinite(value)) {
            throw new ValidationException(field + " must be finite.");
        }
    }

    static class SyntheticDataset {

        static List<Sample> generate(
            int count,
            double noise,
            long seed
        ) {
            if (count < 2 || noise < 0 || !Double.isFinite(noise)) {
                throw new ValidationException("Invalid dataset parameters.");
            }

            Random random = new Random(seed);
            List<Sample> samples = new ArrayList<>();

            for (int i = 0; i < count; i++) {
                double x1 = random.nextDouble() * 10 - 5;
                double x2 = random.nextDouble() * 10 - 5;

                double target =
                    2.5 * x1 -
                    1.7 * x2 +
                    4.0 +
                    random.nextGaussian() * noise;

                samples.add(new Sample(
                    List.of(x1, x2),
                    target
                ));
            }

            return List.copyOf(samples);
        }
    }

    static class LinearRegression {

        private final double[] weights;
        private double bias;

        private ModelStatus status = ModelStatus.INITIALIZED;

        LinearRegression(int dimensions) {
            if (dimensions <= 0) {
                throw new ValidationException("Invalid dimensions.");
            }

            this.weights = new double[dimensions];
            this.bias = 0.0;
        }

        double predict(List<Double> features) {
            if (features.size() != weights.length) {
                throw new ValidationException("Feature mismatch.");
            }

            double result = bias;

            for (int i = 0; i < weights.length; i++) {
                result += weights[i] * features.get(i);
            }

            return result;
        }

        double loss(
            List<Sample> samples,
            double regularization
        ) {
            if (samples.isEmpty()) {
                throw new ValidationException("Empty dataset.");
            }

            double squaredError = 0;

            for (Sample sample : samples) {
                double error = predict(sample.features()) - sample.target();
                squaredError += error * error;
            }

            double penalty = 0;

            for (double weight : weights) {
                penalty += weight * weight;
            }

            double result =
                squaredError / (2.0 * samples.size()) +
                regularization * penalty / 2.0;

            if (!Double.isFinite(result)) {
                throw new TrainingException("Non-finite training loss.");
            }

            return result;
        }

        Gradients gradients(
            List<Sample> samples,
            double regularization
        ) {
            if (samples.isEmpty()) {
                throw new ValidationException("Empty gradient batch.");
            }

            double[] gradients = new double[weights.length];
            double biasGradient = 0;

            for (Sample sample : samples) {
                double error = predict(sample.features()) - sample.target();

                for (int i = 0; i < weights.length; i++) {
                    gradients[i] += error * sample.features().get(i);
                }

                biasGradient += error;
            }

            for (int i = 0; i < weights.length; i++) {
                gradients[i] =
                    gradients[i] / samples.size() +
                    regularization * weights[i];
            }

            biasGradient /= samples.size();

            return new Gradients(gradients, biasGradient);
        }

        double update(
            Gradients gradients,
            TrainingPolicy policy
        ) {
            double normSquared = gradients.bias() * gradients.bias();

            for (double gradient : gradients.weights()) {
                normSquared += gradient * gradient;
            }

            double norm = Math.sqrt(normSquared);

            if (!Double.isFinite(norm)) {
                throw new TrainingException("Gradient norm overflow.");
            }

            double scale = Math.min(
                1.0,
                policy.gradientClip() / Math.max(norm, 1e-15)
            );

            for (int i = 0; i < weights.length; i++) {
                weights[i] -=
                    policy.learningRate() *
                    gradients.weights()[i] * scale;
            }

            bias -= policy.learningRate() *
                gradients.bias() * scale;

            return norm;
        }

        void markTrained() {
            status = ModelStatus.TRAINED;
        }

        void markEvaluated() {
            status = ModelStatus.EVALUATED;
        }

        ModelStatus getStatus() {
            return status;
        }

        List<Double> getWeights() {
            List<Double> result = new ArrayList<>();

            for (double weight : weights) {
                result.add(weight);
            }

            return List.copyOf(result);
        }

        double getBias() {
            return bias;
        }
    }

    interface BatchStrategy {
        List<List<Sample>> create(
            List<Sample> samples,
            int batchSize,
            Random random
        );
    }

    static class BatchOptimizer implements BatchStrategy {

        @Override
        public List<List<Sample>> create(
            List<Sample> samples,
            int batchSize,
            Random random
        ) {
            return List.of(List.copyOf(samples));
        }
    }

    static class StochasticOptimizer implements BatchStrategy {

        @Override
        public List<List<Sample>> create(
            List<Sample> samples,
            int batchSize,
            Random random
        ) {
            List<List<Sample>> result = new ArrayList<>();

            for (Sample sample : samples) {
                result.add(List.of(sample));
            }

            return result;
        }
    }

    static class MiniBatchOptimizer implements BatchStrategy {

        @Override
        public List<List<Sample>> create(
            List<Sample> samples,
            int batchSize,
            Random random
        ) {
            List<Sample> shuffled = new ArrayList<>(samples);
            Collections.shuffle(shuffled, random);

            List<List<Sample>> result = new ArrayList<>();

            for (int start = 0; start < shuffled.size(); start += batchSize) {
                int end = Math.min(start + batchSize, shuffled.size());

                result.add(List.copyOf(
                    shuffled.subList(start, end)
                ));
            }

            return result;
        }
    }

    static class StrategyFactory {

        private static final Map<OptimizerType, BatchStrategy> STRATEGIES =
            new EnumMap<>(OptimizerType.class);

        static {
            STRATEGIES.put(
                OptimizerType.BATCH,
                new BatchOptimizer()
            );

            STRATEGIES.put(
                OptimizerType.STOCHASTIC,
                new StochasticOptimizer()
            );

            STRATEGIES.put(
                OptimizerType.MINI_BATCH,
                new MiniBatchOptimizer()
            );
        }

        static BatchStrategy get(OptimizerType optimizer) {
            BatchStrategy strategy = STRATEGIES.get(optimizer);

            if (strategy == null) {
                throw new ValidationException("Unsupported optimizer.");
            }

            return strategy;
        }
    }

    static class ConvergencePolicy {

        private final Predicate<Double> finiteLoss =
            Double::isFinite;

        boolean isValid(double loss) {
            return finiteLoss.test(loss);
        }

        boolean shouldStop(
            int staleEpochs,
            int patience
        ) {
            return staleEpochs >= patience;
        }
    }

    static class TrainingService {

        private final TrainingPolicy policy;
        private TrainingStatus status = TrainingStatus.CREATED;

        TrainingService(TrainingPolicy policy) {
            this.policy = policy;
        }

        TrainingReport train(List<Sample> samples) {
            try {
                status = TrainingStatus.VALIDATED;

                if (samples == null || samples.isEmpty()) {
                    throw new ValidationException("Samples are required.");
                }

                int dimensions = samples.get(0).features().size();

                if (samples.stream().anyMatch(
                    sample -> sample.features().size() != dimensions
                )) {
                    throw new ValidationException(
                        "Inconsistent feature dimensions."
                    );
                }

                LinearRegression model =
                    new LinearRegression(dimensions);

                BatchStrategy strategy =
                    StrategyFactory.get(policy.optimizer());

                Random random = new Random(policy.seed());
                ConvergencePolicy convergence = new ConvergencePolicy();

                List<EpochMetrics> history = new ArrayList<>();

                double bestLoss = Double.POSITIVE_INFINITY;
                int staleEpochs = 0;

                status = TrainingStatus.RUNNING;

                for (int epoch = 1; epoch <= policy.epochs(); epoch++) {
                    List<List<Sample>> batches = strategy.create(
                        samples,
                        policy.batchSize(),
                        random
                    );

                    double totalGradientNorm = 0;

                    for (List<Sample> batch : batches) {
                        Gradients gradients = model.gradients(
                            batch,
                            policy.regularization()
                        );

                        totalGradientNorm += model.update(
                            gradients,
                            policy
                        );
                    }

                    double loss = model.loss(
                        samples,
                        policy.regularization()
                    );

                    if (!convergence.isValid(loss)) {
                        throw new TrainingException(
                            "Non-finite loss encountered."
                        );
                    }

                    EpochMetrics metrics = new EpochMetrics(
                        epoch,
                        loss,
                        totalGradientNorm / batches.size(),
                        batches.size()
                    );

                    history.add(metrics);

                    if (epoch == 1 || epoch % 50 == 0) {
                        System.out.printf(
                            "[%s] Epoch %d | Loss %.6f%n",
                            policy.optimizer(),
                            epoch,
                            loss
                        );
                    }

                    if (bestLoss - loss > policy.tolerance()) {
                        bestLoss = loss;
                        staleEpochs = 0;
                    } else {
                        staleEpochs++;
                    }

                    if (convergence.shouldStop(
                        staleEpochs,
                        policy.patience()
                    )) {
                        status = TrainingStatus.EARLY_STOPPED;
                        break;
                    }
                }

                model.markTrained();

                if (status != TrainingStatus.EARLY_STOPPED) {
                    status = TrainingStatus.COMPLETED;
                }

                return new TrainingReport(
                    policy.optimizer(),
                    model.getWeights(),
                    model.getBias(),
                    history,
                    status,
                    model.getStatus()
                );

            } catch (RuntimeException exception) {
                status = TrainingStatus.FAILED;
                throw exception;
            }
        }

        TrainingStatus getStatus() {
            return status;
        }
    }

    static class EvaluationService {

        static EvaluationMetrics evaluate(
            TrainingReport model,
            List<Sample> samples
        ) {
            if (samples.isEmpty()) {
                throw new ValidationException("Evaluation data required.");
            }

            double mean = samples.stream()
                .mapToDouble(Sample::target)
                .average()
                .orElseThrow();

            double residual = 0;
            double total = 0;

            for (Sample sample : samples) {
                double prediction = model.bias();

                for (int i = 0; i < model.weights().size(); i++) {
                    prediction +=
                        model.weights().get(i) *
                        sample.features().get(i);
                }

                residual += Math.pow(
                    sample.target() - prediction,
                    2
                );

                total += Math.pow(sample.target() - mean, 2);
            }

            double mse = residual / samples.size();

            double rSquared = total == 0
                ? (residual == 0 ? 1.0 : 0.0)
                : 1.0 - residual / total;

            return new EvaluationMetrics(mse, rSquared);
        }
    }

    static class AuditReportService {

        static void export(
            List<TrainingReport> reports,
            Path destination
        ) throws IOException {

            try (BufferedWriter writer = Files.newBufferedWriter(
                destination,
                StandardCharsets.UTF_8
            )) {
                writer.write(
                    "optimizer,epoch,loss,gradient_norm,batch_count"
                );
                writer.newLine();

                for (TrainingReport report : reports) {
                    for (EpochMetrics metric : report.history()) {
                        writer.write(String.format(
                            java.util.Locale.ROOT,
                            "%s,%d,%.12f,%.12f,%d",
                            report.optimizer(),
                            metric.epoch(),
                            metric.loss(),
                            metric.gradientNorm(),
                            metric.batchCount()
                        ));

                        writer.newLine();
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        try {
            System.out.println(
                "ENTERPRISE GRADIENT DESCENT TRAINING SYSTEM"
            );

            List<Sample> training =
                SyntheticDataset.generate(240, 0.5, 42);

            List<Sample> testing =
                SyntheticDataset.generate(80, 0.5, 99);

            List<TrainingPolicy> policies = List.of(
                new TrainingPolicy(
                    OptimizerType.BATCH,
                    0.01, 0.001, 10.0, 1e-10,
                    150, 16, 25, 42
                ),
                new TrainingPolicy(
                    OptimizerType.STOCHASTIC,
                    0.003, 0.001, 10.0, 1e-10,
                    150, 1, 25, 42
                ),
                new TrainingPolicy(
                    OptimizerType.MINI_BATCH,
                    0.01, 0.001, 10.0, 1e-10,
                    150, 16, 25, 42
                )
            );

            List<TrainingReport> reports = new ArrayList<>();

            for (TrainingPolicy policy : policies) {
                TrainingService service =
                    new TrainingService(policy);

                TrainingReport report = service.train(training);
                EvaluationMetrics metrics =
                    EvaluationService.evaluate(report, testing);

                reports.add(report);

                System.out.println("\nOptimizer: " + report.optimizer());
                System.out.println("Weights: " + report.weights());

                System.out.printf(
                    "Bias: %.4f%n",
                    report.bias()
                );

                System.out.printf(
                    "Test MSE: %.6f%n",
                    metrics.meanSquaredError()
                );

                System.out.printf(
                    "R-squared: %.6f%n",
                    metrics.rSquared()
                );

                System.out.println("Status: " + report.status());
                System.out.println("Model: " + report.modelStatus());
            }

            AuditReportService.export(
                reports,
                Path.of("gradient_descent_results.csv")
            );

            System.out.println("\nTraining audit exported.");

            // Demonstrate enterprise policy validation.
            try {
                new TrainingPolicy(
                    OptimizerType.BATCH,
                    -0.01, 0.0, 10.0, 1e-10,
                    100, 16, 10, 42
                );
            } catch (ValidationException exception) {
                System.out.println(
                    "Invalid policy rejected: " + exception.getMessage()
                );
            }

        } catch (IOException | RuntimeException exception) {
            System.err.println(
                "Training system failure: " + exception.getMessage()
            );

            System.exit(1);
        }
    }
}
