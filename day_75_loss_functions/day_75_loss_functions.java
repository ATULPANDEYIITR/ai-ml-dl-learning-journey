import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.stream.Collectors;

/*
 * Enterprise Loss Evaluation and Model Governance.
 *
 * Java 17 implementation featuring:
 * - Immutable domain records
 * - Regression and classification loss functions
 * - Stable probability calculations
 * - Analytical gradients
 * - Model evaluation services
 * - Configurable acceptance policies
 * - Evaluation state transitions
 * - Audit records
 * - Exception handling
 */

public class LossGovernance {

    private static final double EPSILON = 1e-12;

    enum LossType {
        MSE,
        MAE,
        BINARY_CROSS_ENTROPY,
        CATEGORICAL_CROSS_ENTROPY
    }

    enum EvaluationStatus {
        PENDING,
        ACCEPTED,
        REJECTED
    }

    enum Reduction {
        NONE,
        SUM,
        MEAN
    }

    static class DomainException extends RuntimeException {
        DomainException(String message) {
            super(message);
        }
    }

    static class ValidationException extends DomainException {
        ValidationException(String message) {
            super(message);
        }
    }

    record LossResult(
        LossType type,
        List<Double> individualLosses,
        double value,
        Reduction reduction
    ) {
        LossResult {
            Objects.requireNonNull(type);
            Objects.requireNonNull(reduction);
            individualLosses = List.copyOf(individualLosses);

            if (individualLosses.isEmpty()) {
                throw new ValidationException("Empty loss result.");
            }

            if (!Double.isFinite(value) || value < 0) {
                throw new ValidationException("Invalid loss value.");
            }
        }
    }

    record Model(
        UUID id,
        String name,
        String version,
        String owner
    ) {
        Model {
            Objects.requireNonNull(id);
            name = requireText(name, "Model name");
            version = requireText(version, "Model version");
            owner = requireText(owner, "Model owner");
        }
    }

    record EvaluationPolicy(
        double maximumMse,
        double maximumMae,
        double maximumBinaryCrossEntropy,
        double maximumCategoricalCrossEntropy,
        boolean requireFiniteMetrics
    ) {
        EvaluationPolicy {
            double[] thresholds = {
                maximumMse,
                maximumMae,
                maximumBinaryCrossEntropy,
                maximumCategoricalCrossEntropy
            };

            for (double threshold : thresholds) {
                if (!Double.isFinite(threshold) || threshold < 0) {
                    throw new ValidationException(
                        "Policy thresholds must be finite and nonnegative."
                    );
                }
            }
        }

        double thresholdFor(LossType type) {
            return switch (type) {
                case MSE -> maximumMse;
                case MAE -> maximumMae;
                case BINARY_CROSS_ENTROPY ->
                    maximumBinaryCrossEntropy;
                case CATEGORICAL_CROSS_ENTROPY ->
                    maximumCategoricalCrossEntropy;
            };
        }
    }

    record EvaluationRecord(
        UUID id,
        Model model,
        LossResult loss,
        EvaluationStatus status,
        double threshold,
        Instant evaluatedAt,
        String evaluator
    ) {
        EvaluationRecord {
            Objects.requireNonNull(id);
            Objects.requireNonNull(model);
            Objects.requireNonNull(loss);
            Objects.requireNonNull(status);
            Objects.requireNonNull(evaluatedAt);
            evaluator = requireText(evaluator, "Evaluator");

            if (!Double.isFinite(threshold) || threshold < 0) {
                throw new ValidationException("Invalid threshold.");
            }
        }
    }

    static String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new ValidationException(field + " cannot be empty.");
        }

        return value.trim();
    }

    static void validatePair(double[] actual, double[] predicted) {
        if (actual == null || predicted == null ||
            actual.length == 0 ||
            actual.length != predicted.length) {
            throw new ValidationException("Invalid input dimensions.");
        }

        for (int i = 0; i < actual.length; i++) {
            if (!Double.isFinite(actual[i]) ||
                !Double.isFinite(predicted[i])) {
                throw new ValidationException(
                    "Inputs must contain finite values."
                );
            }
        }
    }

    static LossResult reduce(
        LossType type,
        List<Double> values,
        Reduction reduction
    ) {
        if (values.isEmpty()) {
            throw new ValidationException("Empty loss batch.");
        }

        double sum = values.stream()
            .mapToDouble(Double::doubleValue)
            .sum();

        if (!Double.isFinite(sum)) {
            throw new DomainException("Loss accumulation overflow.");
        }

        double result = switch (reduction) {
            case NONE -> 0.0;
            case SUM -> sum;
            case MEAN -> sum / values.size();
        };

        return new LossResult(type, values, result, reduction);
    }

    static LossResult mse(
        double[] actual,
        double[] predicted,
        Reduction reduction
    ) {
        validatePair(actual, predicted);

        List<Double> values = new ArrayList<>();

        for (int i = 0; i < actual.length; i++) {
            double error = actual[i] - predicted[i];
            values.add(error * error);
        }

        return reduce(LossType.MSE, values, reduction);
    }

    static LossResult mae(
        double[] actual,
        double[] predicted,
        Reduction reduction
    ) {
        validatePair(actual, predicted);

        List<Double> values = new ArrayList<>();

        for (int i = 0; i < actual.length; i++) {
            values.add(Math.abs(actual[i] - predicted[i]));
        }

        return reduce(LossType.MAE, values, reduction);
    }

    static void validateBinary(int[] actual, double[] probabilities) {
        if (actual == null || probabilities == null ||
            actual.length == 0 ||
            actual.length != probabilities.length) {
            throw new ValidationException("Invalid binary dimensions.");
        }

        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != 0 && actual[i] != 1) {
                throw new ValidationException("Invalid binary label.");
            }

            double p = probabilities[i];

            if (!Double.isFinite(p) || p < 0 || p > 1) {
                throw new ValidationException("Invalid probability.");
            }
        }
    }

    static LossResult binaryCrossEntropy(
        int[] actual,
        double[] probabilities,
        Reduction reduction
    ) {
        validateBinary(actual, probabilities);

        List<Double> values = new ArrayList<>();

        for (int i = 0; i < actual.length; i++) {
            double p = Math.max(
                EPSILON,
                Math.min(1.0 - EPSILON, probabilities[i])
            );

            double loss = -(
                actual[i] * Math.log(p) +
                (1 - actual[i]) * Math.log1p(-p)
            );

            values.add(loss);
        }

        return reduce(
            LossType.BINARY_CROSS_ENTROPY,
            values,
            reduction
        );
    }

    static double softplus(double x) {
        return Math.max(x, 0.0) +
            Math.log1p(Math.exp(-Math.abs(x)));
    }

    static LossResult binaryCrossEntropyWithLogits(
        int[] actual,
        double[] logits,
        Reduction reduction
    ) {
        if (actual == null || logits == null ||
            actual.length == 0 || actual.length != logits.length) {
            throw new ValidationException("Invalid binary dimensions.");
        }

        List<Double> values = new ArrayList<>();

        for (int i = 0; i < actual.length; i++) {
            if (actual[i] != 0 && actual[i] != 1) {
                throw new ValidationException("Invalid binary target.");
            }

            double z = logits[i];

            if (!Double.isFinite(z)) {
                throw new ValidationException("Invalid logit.");
            }

            values.add(
                Math.max(z, 0.0) -
                z * actual[i] +
                softplus(-Math.abs(z))
            );
        }

        return reduce(
            LossType.BINARY_CROSS_ENTROPY,
            values,
            reduction
        );
    }

    static void validateMulticlass(
        int[] actual,
        double[][] probabilities
    ) {
        if (actual == null || probabilities == null ||
            actual.length == 0 ||
            actual.length != probabilities.length) {
            throw new ValidationException("Invalid multiclass batch.");
        }

        int classes = probabilities[0].length;

        if (classes < 2) {
            throw new ValidationException("At least two classes required.");
        }

        for (int i = 0; i < actual.length; i++) {
            if (actual[i] < 0 || actual[i] >= classes) {
                throw new ValidationException("Invalid class label.");
            }

            if (probabilities[i].length != classes) {
                throw new ValidationException("Inconsistent dimensions.");
            }

            double sum = 0;

            for (double p : probabilities[i]) {
                if (!Double.isFinite(p) || p < 0 || p > 1) {
                    throw new ValidationException("Invalid probability.");
                }

                sum += p;
            }

            if (Math.abs(sum - 1.0) > 1e-8) {
                throw new ValidationException(
                    "Probabilities must sum to one."
                );
            }
        }
    }

    static LossResult categoricalCrossEntropy(
        int[] actual,
        double[][] probabilities,
        Reduction reduction
    ) {
        validateMulticlass(actual, probabilities);

        List<Double> values = new ArrayList<>();

        for (int i = 0; i < actual.length; i++) {
            double correctProbability = Math.max(
                probabilities[i][actual[i]],
                EPSILON
            );

            values.add(-Math.log(correctProbability));
        }

        return reduce(
            LossType.CATEGORICAL_CROSS_ENTROPY,
            values,
            reduction
        );
    }

    static double[] softmax(double[] logits) {
        if (logits == null || logits.length < 2) {
            throw new ValidationException("At least two logits required.");
        }

        double maximum = Double.NEGATIVE_INFINITY;

        for (double value : logits) {
            if (!Double.isFinite(value)) {
                throw new ValidationException("Invalid logit.");
            }

            maximum = Math.max(maximum, value);
        }

        double[] result = new double[logits.length];
        double sum = 0;

        for (int i = 0; i < logits.length; i++) {
            result[i] = Math.exp(logits[i] - maximum);
            sum += result[i];
        }

        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    static LossResult categoricalCrossEntropyWithLogits(
        int[] actual,
        double[][] logits,
        Reduction reduction
    ) {
        if (actual == null || logits == null ||
            actual.length == 0 || actual.length != logits.length) {
            throw new ValidationException("Invalid multiclass batch.");
        }

        int classes = logits[0].length;

        if (classes < 2) {
            throw new ValidationException("At least two classes required.");
        }

        List<Double> values = new ArrayList<>();

        for (int i = 0; i < actual.length; i++) {
            if (actual[i] < 0 || actual[i] >= classes ||
                logits[i].length != classes) {
                throw new ValidationException("Invalid target dimensions.");
            }

            double maximum = Double.NEGATIVE_INFINITY;

            for (double z : logits[i]) {
                if (!Double.isFinite(z)) {
                    throw new ValidationException("Invalid logit.");
                }

                maximum = Math.max(maximum, z);
            }

            double sum = 0;

            for (double z : logits[i]) {
                sum += Math.exp(z - maximum);
            }

            double logSumExp = maximum + Math.log(sum);

            values.add(logSumExp - logits[i][actual[i]]);
        }

        return reduce(
            LossType.CATEGORICAL_CROSS_ENTROPY,
            values,
            reduction
        );
    }

    static double[] mseGradient(double[] actual, double[] predicted) {
        validatePair(actual, predicted);

        double[] gradient = new double[actual.length];

        for (int i = 0; i < actual.length; i++) {
            gradient[i] =
                2.0 * (predicted[i] - actual[i]) / actual.length;
        }

        return gradient;
    }

    static double[] binaryLogitGradient(
        int[] actual,
        double[] logits
    ) {
        if (actual == null || logits == null ||
            actual.length == 0 || actual.length != logits.length) {
            throw new ValidationException("Invalid gradient dimensions.");
        }

        double[] gradient = new double[actual.length];

        for (int i = 0; i < actual.length; i++) {
            if ((actual[i] != 0 && actual[i] != 1) ||
                !Double.isFinite(logits[i])) {
                throw new ValidationException("Invalid gradient input.");
            }

            double z = logits[i];

            double probability;

            if (z >= 0) {
                probability = 1.0 / (1.0 + Math.exp(-z));
            } else {
                double e = Math.exp(z);
                probability = e / (1.0 + e);
            }

            gradient[i] =
                (probability - actual[i]) / actual.length;
        }

        return gradient;
    }

    static class EvaluationService {

        private final EvaluationPolicy policy;
        private final List<EvaluationRecord> auditLog = new ArrayList<>();

        EvaluationService(EvaluationPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        EvaluationRecord evaluate(
            Model model,
            LossResult result,
            String evaluator
        ) {
            Objects.requireNonNull(model);
            Objects.requireNonNull(result);

            if (policy.requireFiniteMetrics() &&
                !Double.isFinite(result.value())) {
                throw new DomainException("Non-finite evaluation metric.");
            }

            double threshold = policy.thresholdFor(result.type());

            EvaluationStatus status = result.value() <= threshold
                ? EvaluationStatus.ACCEPTED
                : EvaluationStatus.REJECTED;

            EvaluationRecord record = new EvaluationRecord(
                UUID.randomUUID(),
                model,
                result,
                status,
                threshold,
                Instant.now(),
                evaluator
            );

            auditLog.add(record);

            return record;
        }

        List<EvaluationRecord> history() {
            return List.copyOf(auditLog);
        }

        Map<EvaluationStatus, Long> statusReport() {
            return auditLog.stream().collect(
                Collectors.groupingBy(
                    EvaluationRecord::status,
                    Collectors.counting()
                )
            );
        }
    }

    static void print(EvaluationRecord record) {
        System.out.printf(
            "%-24s | %-30s | Loss: %.8f | Threshold: %.4f | %s%n",
            record.model().name(),
            record.loss().type(),
            record.loss().value(),
            record.threshold(),
            record.status()
        );
    }

    public static void main(String[] args) {

        try {
            EvaluationPolicy policy = new EvaluationPolicy(
                1.0,
                1.0,
                0.5,
                0.5,
                true
            );

            EvaluationService service = new EvaluationService(policy);

            Model regressionModel = new Model(
                UUID.randomUUID(),
                "DemandForecast",
                "2.1.0",
                "ForecastingTeam"
            );

            Model binaryModel = new Model(
                UUID.randomUUID(),
                "FraudDetection",
                "3.2.0",
                "RiskAnalytics"
            );

            Model multiclassModel = new Model(
                UUID.randomUUID(),
                "DocumentClassifier",
                "1.4.0",
                "DocumentIntelligence"
            );

            double[] actual = {3.0, -0.5, 2.0, 7.0};
            double[] predicted = {2.5, 0.0, 2.0, 8.0};

            LossResult mse = mse(
                actual,
                predicted,
                Reduction.MEAN
            );

            LossResult mae = mae(
                actual,
                predicted,
                Reduction.MEAN
            );

            int[] binaryLabels = {1, 0, 1, 0};
            double[] binaryProbabilities = {
                0.95, 0.10, 0.80, 0.20
            };

            LossResult binary = binaryCrossEntropy(
                binaryLabels,
                binaryProbabilities,
                Reduction.MEAN
            );

            int[] multiclassLabels = {0, 2};

            double[][] multiclassProbabilities = {
                {0.7, 0.2, 0.1},
                {0.1, 0.2, 0.7}
            };

            LossResult categorical = categoricalCrossEntropy(
                multiclassLabels,
                multiclassProbabilities,
                Reduction.MEAN
            );

            print(service.evaluate(
                regressionModel,
                mse,
                "ValidationPipeline"
            ));

            print(service.evaluate(
                regressionModel,
                mae,
                "ValidationPipeline"
            ));

            print(service.evaluate(
                binaryModel,
                binary,
                "RiskEvaluationService"
            ));

            print(service.evaluate(
                multiclassModel,
                categorical,
                "ClassificationService"
            ));

            LossResult extreme = binaryCrossEntropyWithLogits(
                new int[]{1, 0},
                new double[]{1000, -1000},
                Reduction.MEAN
            );

            System.out.println(
                "\nExtreme-logit loss: " + extreme.value()
            );

            System.out.println(
                "MSE gradient: " +
                java.util.Arrays.toString(
                    mseGradient(actual, predicted)
                )
            );

            System.out.println(
                "Binary gradient: " +
                java.util.Arrays.toString(
                    binaryLogitGradient(
                        binaryLabels,
                        new double[]{3, -2, 1.4, -1.4}
                    )
                )
            );

            System.out.println("\nAUDIT REPORT");

            service.history().forEach(System.out::println);

            System.out.println("\nSTATUS COUNTS");
            System.out.println(service.statusReport());

            // Invalid class labels must not enter the audit history.
            try {
                categoricalCrossEntropy(
                    new int[]{5},
                    new double[][]{{0.7, 0.3}},
                    Reduction.MEAN
                );
            } catch (ValidationException exception) {
                System.out.println(
                    "Rejected invalid input: " + exception.getMessage()
                );
            }

        } catch (DomainException exception) {
            System.err.println("Domain failure: " + exception.getMessage());
        }
    }
}
