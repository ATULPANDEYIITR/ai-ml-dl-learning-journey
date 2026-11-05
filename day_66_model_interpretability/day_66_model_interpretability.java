import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/*
 * Model Interpretability
 *
 * Enterprise-oriented implementation covering:
 * - Standardized feature importance
 * - Permutation importance
 * - Partial dependence
 * - Interpretation governance
 * - Model evaluation
 * - Audit events and workflow states
 *
 * Java 17
 *
 * Run:
 * javac ModelInterpretability.java
 * java ModelInterpretability
 */

public class ModelInterpretability {

    enum AnalysisMethod {
        COEFFICIENT,
        PERMUTATION,
        PARTIAL_DEPENDENCE
    }

    enum ExecutionStatus {
        CREATED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    enum RiskLevel {
        LOW,
        MEDIUM,
        HIGH
    }

    static class InterpretationException extends RuntimeException {
        InterpretationException(String message) {
            super(message);
        }
    }

    record FeatureDefinition(
        String name,
        String description,
        String unit,
        boolean sensitive
    ) {
        FeatureDefinition {
            if (name == null || name.isBlank()) {
                throw new InterpretationException(
                    "Feature name is required."
                );
            }

            if (description == null || description.isBlank()) {
                throw new InterpretationException(
                    "Feature description is required."
                );
            }

            if (unit == null || unit.isBlank()) {
                throw new InterpretationException(
                    "Feature unit is required."
                );
            }
        }
    }

    record EvaluationResult(
        double mse,
        double mae,
        double rmse,
        int sampleCount
    ) {}

    record ImportanceResult(
        String feature,
        AnalysisMethod method,
        double score,
        double normalizedScore,
        double standardDeviation
    ) {}

    record DependencePoint(
        double featureValue,
        double averagePrediction
    ) {}

    record DependenceResult(
        String feature,
        List<DependencePoint> points
    ) {
        DependenceResult {
            points = List.copyOf(points);
        }
    }

    record InterpretationPolicy(
        int minimumSamples,
        int maximumFeatures,
        int minimumRepeats,
        int maximumRepeats,
        boolean permitSensitiveFeatures
    ) {
        InterpretationPolicy {
            if (minimumSamples < 2 ||
                maximumFeatures < 1 ||
                minimumRepeats < 1 ||
                maximumRepeats < minimumRepeats) {
                throw new InterpretationException(
                    "Invalid interpretation policy."
                );
            }
        }

        void validate(
            List<FeatureDefinition> features,
            double[][] X,
            double[] y,
            int repeats
        ) {
            if (features.size() > maximumFeatures) {
                throw new InterpretationException(
                    "Feature limit exceeded."
                );
            }

            if (X.length < minimumSamples) {
                throw new InterpretationException(
                    "Insufficient observations."
                );
            }

            if (repeats < minimumRepeats ||
                repeats > maximumRepeats) {
                throw new InterpretationException(
                    "Repeat count violates policy."
                );
            }

            if (!permitSensitiveFeatures &&
                features.stream().anyMatch(FeatureDefinition::sensitive)) {
                throw new InterpretationException(
                    "Sensitive feature analysis is prohibited."
                );
            }

            DatasetValidator.validate(X, y, features.size());
        }
    }

    static class DatasetValidator {

        static void validate(
            double[][] X,
            double[] y,
            int expectedFeatures
        ) {
            Objects.requireNonNull(X, "Feature matrix is required.");
            Objects.requireNonNull(y, "Target array is required.");

            if (X.length == 0 || X.length != y.length) {
                throw new InterpretationException(
                    "Invalid dataset dimensions."
                );
            }

            for (int i = 0; i < X.length; i++) {
                if (X[i] == null ||
                    X[i].length != expectedFeatures) {
                    throw new InterpretationException(
                        "Inconsistent feature dimensions."
                    );
                }

                for (double value : X[i]) {
                    if (!Double.isFinite(value)) {
                        throw new InterpretationException(
                            "Non-finite feature value."
                        );
                    }
                }

                if (!Double.isFinite(y[i])) {
                    throw new InterpretationException(
                        "Non-finite target value."
                    );
                }
            }
        }

        static double[][] copy(double[][] source) {
            double[][] result = new double[source.length][];

            for (int i = 0; i < source.length; i++) {
                result[i] = source[i].clone();
            }

            return result;
        }
    }

    interface PredictiveModel {
        double predict(double[] row);

        List<String> featureNames();
    }

    static final class LinearRegressionModel implements PredictiveModel {

        private final List<String> featureNames;
        private final double[] means;
        private final double[] scales;
        private final double[] coefficients;
        private final double intercept;

        LinearRegressionModel(
            List<String> featureNames,
            double[] means,
            double[] scales,
            double[] coefficients,
            double intercept
        ) {
            this.featureNames = List.copyOf(featureNames);
            this.means = means.clone();
            this.scales = scales.clone();
            this.coefficients = coefficients.clone();
            this.intercept = intercept;

            if (featureNames.isEmpty() ||
                featureNames.size() != means.length ||
                means.length != scales.length ||
                scales.length != coefficients.length) {
                throw new InterpretationException(
                    "Inconsistent model dimensions."
                );
            }

            if (featureNames.stream().distinct().count()
                != featureNames.size()) {
                throw new InterpretationException(
                    "Duplicate model feature names."
                );
            }

            if (!Double.isFinite(intercept)) {
                throw new InterpretationException("Invalid intercept.");
            }

            for (int i = 0; i < scales.length; i++) {
                if (!Double.isFinite(means[i]) ||
                    !Double.isFinite(scales[i]) ||
                    scales[i] <= 0 ||
                    !Double.isFinite(coefficients[i])) {
                    throw new InterpretationException(
                        "Invalid model parameters."
                    );
                }
            }
        }

        @Override
        public double predict(double[] row) {
            if (row == null || row.length != featureNames.size()) {
                throw new InterpretationException(
                    "Invalid prediction input."
                );
            }

            double result = intercept;

            for (int i = 0; i < row.length; i++) {
                if (!Double.isFinite(row[i])) {
                    throw new InterpretationException(
                        "Invalid feature value."
                    );
                }

                result += coefficients[i] *
                    ((row[i] - means[i]) / scales[i]);
            }

            return result;
        }

        double[] predictAll(double[][] X) {
            return java.util.Arrays.stream(X)
                .mapToDouble(this::predict)
                .toArray();
        }

        @Override
        public List<String> featureNames() {
            return featureNames;
        }

        double[] coefficients() {
            return coefficients.clone();
        }
    }

    static final class ModelTrainer {

        private static double[] solve(
            double[][] matrix,
            double[] rhs
        ) {
            int n = rhs.length;
            double[][] a = DatasetValidator.copy(matrix);
            double[] b = rhs.clone();

            for (int col = 0; col < n; col++) {
                int pivot = col;

                for (int row = col + 1; row < n; row++) {
                    if (Math.abs(a[row][col]) >
                        Math.abs(a[pivot][col])) {
                        pivot = row;
                    }
                }

                if (Math.abs(a[pivot][col]) < 1e-12) {
                    throw new InterpretationException(
                        "Singular regression matrix."
                    );
                }

                double[] temp = a[col];
                a[col] = a[pivot];
                a[pivot] = temp;

                double value = b[col];
                b[col] = b[pivot];
                b[pivot] = value;

                double divisor = a[col][col];

                for (int j = col; j < n; j++) {
                    a[col][j] /= divisor;
                }

                b[col] /= divisor;

                for (int row = 0; row < n; row++) {
                    if (row == col) continue;

                    double factor = a[row][col];

                    for (int j = col; j < n; j++) {
                        a[row][j] -= factor * a[col][j];
                    }

                    b[row] -= factor * b[col];
                }
            }

            return b;
        }

        static LinearRegressionModel train(
            List<FeatureDefinition> features,
            double[][] X,
            double[] y
        ) {
            DatasetValidator.validate(X, y, features.size());

            int n = X.length;
            int p = features.size();

            double[] means = new double[p];
            double[] scales = new double[p];

            for (int j = 0; j < p; j++) {
                final int index = j;

                means[j] = java.util.Arrays.stream(X)
                    .mapToDouble(row -> row[index])
                    .average()
                    .orElseThrow();

                double variance = java.util.Arrays.stream(X)
                    .mapToDouble(row ->
                        Math.pow(row[index] - means[index], 2)
                    )
                    .average()
                    .orElse(0.0);

                scales[j] = Math.sqrt(variance);

                if (scales[j] < 1e-12) {
                    scales[j] = 1.0;
                }
            }

            int dimensions = p + 1;

            double[][] normal = new double[dimensions][dimensions];
            double[] rhs = new double[dimensions];

            for (int i = 0; i < n; i++) {
                double[] row = new double[dimensions];
                row[0] = 1.0;

                for (int j = 0; j < p; j++) {
                    row[j + 1] =
                        (X[i][j] - means[j]) / scales[j];
                }

                for (int j = 0; j < dimensions; j++) {
                    rhs[j] += row[j] * y[i];

                    for (int k = 0; k < dimensions; k++) {
                        normal[j][k] += row[j] * row[k];
                    }
                }
            }

            double[] solution = solve(normal, rhs);

            return new LinearRegressionModel(
                features.stream()
                    .map(FeatureDefinition::name)
                    .toList(),
                means,
                scales,
                java.util.Arrays.copyOfRange(
                    solution,
                    1,
                    solution.length
                ),
                solution[0]
            );
        }
    }

    static final class EvaluationService {

        static EvaluationResult evaluate(
            double[] actual,
            double[] predicted
        ) {
            if (actual.length == 0 ||
                actual.length != predicted.length) {
                throw new InterpretationException(
                    "Invalid evaluation arrays."
                );
            }

            double mse = 0;
            double mae = 0;

            for (int i = 0; i < actual.length; i++) {
                double error = actual[i] - predicted[i];

                mse += error * error;
                mae += Math.abs(error);
            }

            mse /= actual.length;
            mae /= actual.length;

            return new EvaluationResult(
                mse,
                mae,
                Math.sqrt(mse),
                actual.length
            );
        }

        static double mean(double[] values) {
            return java.util.Arrays.stream(values)
                .average()
                .orElseThrow();
        }
    }

    static final class InterpretationService {

        private final PredictiveModel model;
        private final List<FeatureDefinition> features;
        private final InterpretationPolicy policy;

        InterpretationService(
            PredictiveModel model,
            List<FeatureDefinition> features,
            InterpretationPolicy policy
        ) {
            this.model = Objects.requireNonNull(model);
            this.features = List.copyOf(features);
            this.policy = Objects.requireNonNull(policy);

            if (!model.featureNames().equals(
                features.stream()
                    .map(FeatureDefinition::name)
                    .toList()
            )) {
                throw new InterpretationException(
                    "Feature metadata does not match model."
                );
            }
        }

        List<ImportanceResult> coefficientImportance() {
            if (!(model instanceof LinearRegressionModel linear)) {
                throw new InterpretationException(
                    "Coefficient interpretation requires linear regression."
                );
            }

            double[] coefficients = linear.coefficients();

            double total = java.util.Arrays.stream(coefficients)
                .map(Math::abs)
                .sum();

            List<ImportanceResult> results = new ArrayList<>();

            for (int i = 0; i < coefficients.length; i++) {
                double score = Math.abs(coefficients[i]);

                results.add(new ImportanceResult(
                    features.get(i).name(),
                    AnalysisMethod.COEFFICIENT,
                    score,
                    total == 0 ? 0 : score / total,
                    0
                ));
            }

            results.sort(
                Comparator.comparingDouble(
                    ImportanceResult::score
                ).reversed()
            );

            return List.copyOf(results);
        }

        List<ImportanceResult> permutationImportance(
            double[][] X,
            double[] y,
            int repeats,
            long seed
        ) {
            policy.validate(features, X, y, repeats);

            double baseline = EvaluationService.evaluate(
                y,
                predictAll(X)
            ).mse();

            Random random = new Random(seed);
            List<ImportanceResult> results = new ArrayList<>();

            for (int featureIndex = 0;
                 featureIndex < features.size();
                 featureIndex++) {

                List<Double> changes = new ArrayList<>();

                for (int repeat = 0; repeat < repeats; repeat++) {
                    double[][] modified =
                        DatasetValidator.copy(X);

                    List<Double> values = new ArrayList<>();

                    for (double[] row : X) {
                        values.add(row[featureIndex]);
                    }

                    Collections.shuffle(values, random);

                    for (int i = 0; i < modified.length; i++) {
                        modified[i][featureIndex] = values.get(i);
                    }

                    double changed = EvaluationService.evaluate(
                        y,
                        predictAll(modified)
                    ).mse();

                    changes.add(changed - baseline);
                }

                double score = changes.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);

                double variance = changes.stream()
                    .mapToDouble(value ->
                        Math.pow(value - score, 2)
                    )
                    .average()
                    .orElse(0.0);

                results.add(new ImportanceResult(
                    features.get(featureIndex).name(),
                    AnalysisMethod.PERMUTATION,
                    score,
                    0.0,
                    Math.sqrt(variance)
                ));
            }

            double total = results.stream()
                .mapToDouble(item -> Math.max(0, item.score()))
                .sum();

            List<ImportanceResult> normalized = results.stream()
                .map(item -> new ImportanceResult(
                    item.feature(),
                    item.method(),
                    item.score(),
                    total == 0
                        ? 0
                        : Math.max(0, item.score()) / total,
                    item.standardDeviation()
                ))
                .sorted(
                    Comparator.comparingDouble(
                        ImportanceResult::score
                    ).reversed()
                )
                .toList();

            return normalized;
        }

        DependenceResult partialDependence(
            double[][] X,
            int featureIndex,
            int gridPoints
        ) {
            DatasetValidator.validate(
                X,
                new double[X.length],
                features.size()
            );

            if (featureIndex < 0 ||
                featureIndex >= features.size()) {
                throw new InterpretationException(
                    "Feature index out of range."
                );
            }

            if (gridPoints < 2) {
                throw new InterpretationException(
                    "At least two grid points are required."
                );
            }

            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;

            for (double[] row : X) {
                min = Math.min(min, row[featureIndex]);
                max = Math.max(max, row[featureIndex]);
            }

            List<DependencePoint> points = new ArrayList<>();

            int actualPoints = min == max ? 1 : gridPoints;

            for (int i = 0; i < actualPoints; i++) {
                double value = min == max
                    ? min
                    : min + (max - min) * i / (gridPoints - 1);

                double[][] modified =
                    DatasetValidator.copy(X);

                for (double[] row : modified) {
                    row[featureIndex] = value;
                }

                double prediction = EvaluationService.mean(
                    predictAll(modified)
                );

                points.add(new DependencePoint(value, prediction));
            }

            return new DependenceResult(
                features.get(featureIndex).name(),
                points
            );
        }

        private double[] predictAll(double[][] X) {
            return java.util.Arrays.stream(X)
                .mapToDouble(model::predict)
                .toArray();
        }
    }

    record AuditEvent(
        UUID eventId,
        UUID executionId,
        ExecutionStatus status,
        Instant timestamp,
        String detail
    ) {}

    static final class AuditService {

        private final List<AuditEvent> events =
            new CopyOnWriteArrayList<>();

        void record(
            UUID executionId,
            ExecutionStatus status,
            String detail
        ) {
            events.add(new AuditEvent(
                UUID.randomUUID(),
                executionId,
                status,
                Instant.now(),
                detail
            ));
        }

        List<AuditEvent> events() {
            return List.copyOf(events);
        }
    }

    static final class InterpretationWorkflow {

        private final InterpretationService service;
        private final AuditService audit;
        private final List<Consumer<AuditEvent>> listeners =
            new CopyOnWriteArrayList<>();

        InterpretationWorkflow(
            InterpretationService service,
            AuditService audit
        ) {
            this.service = service;
            this.audit = audit;
        }

        void subscribe(Consumer<AuditEvent> listener) {
            listeners.add(Objects.requireNonNull(listener));
        }

        private void transition(
            UUID executionId,
            ExecutionStatus status,
            String detail
        ) {
            audit.record(executionId, status, detail);

            AuditEvent event = audit.events()
                .get(audit.events().size() - 1);

            for (Consumer<AuditEvent> listener : listeners) {
                try {
                    listener.accept(event);
                } catch (RuntimeException exception) {
                    System.err.println(
                        "Audit listener failed: " + exception.getMessage()
                    );
                }
            }
        }

        Map<String, Object> execute(
            double[][] X,
            double[] y
        ) {
            UUID executionId = UUID.randomUUID();

            transition(
                executionId,
                ExecutionStatus.CREATED,
                "Interpretation request accepted."
            );

            transition(
                executionId,
                ExecutionStatus.RUNNING,
                "Analysis started."
            );

            try {
                EvaluationResult evaluation =
                    EvaluationService.evaluate(
                        y,
                        java.util.Arrays.stream(X)
                            .mapToDouble(service.model::predict)
                            .toArray()
                    );

                List<ImportanceResult> coefficients =
                    service.coefficientImportance();

                List<ImportanceResult> permutations =
                    service.permutationImportance(
                        X,
                        y,
                        15,
                        42
                    );

                List<DependenceResult> dependence =
                    java.util.stream.IntStream
                        .range(0, service.features.size())
                        .mapToObj(index ->
                            service.partialDependence(X, index, 8)
                        )
                        .toList();

                transition(
                    executionId,
                    ExecutionStatus.COMPLETED,
                    "All interpretation analyses completed."
                );

                return Map.of(
                    "executionId", executionId.toString(),
                    "evaluation", evaluation,
                    "coefficients", coefficients,
                    "permutation", permutations,
                    "partialDependence", dependence
                );
            } catch (RuntimeException exception) {
                transition(
                    executionId,
                    ExecutionStatus.FAILED,
                    exception.getMessage()
                );

                throw exception;
            }
        }
    }

    static double[][] generateFeatures(int count, long seed) {
        Random random = new Random(seed);
        double[][] X = new double[count][3];

        for (int i = 0; i < count; i++) {
            X[i][0] = 100 + random.nextDouble() * 900;
            X[i][1] = 0.05 + random.nextDouble() * 0.90;
            X[i][2] = 0.40 + random.nextDouble() * 0.59;
        }

        return X;
    }

    static double[] generateTargets(
        double[][] X,
        long seed
    ) {
        Random random = new Random(seed);
        double[] y = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            y[i] =
                0.035 * X[i][0] -
                120.0 * X[i][1] +
                180.0 * X[i][2] +
                random.nextGaussian() * 8.0;
        }

        return y;
    }

    public static void main(String[] args) {

        try {
            List<FeatureDefinition> features = List.of(
                new FeatureDefinition(
                    "revenue",
                    "Annual business revenue",
                    "thousand USD",
                    false
                ),
                new FeatureDefinition(
                    "debt_ratio",
                    "Debt relative to assets",
                    "ratio",
                    false
                ),
                new FeatureDefinition(
                    "customer_retention",
                    "Customer retention rate",
                    "ratio",
                    false
                )
            );

            double[][] X = generateFeatures(240, 42);
            double[] y = generateTargets(X, 84);

            // Training and testing indices are separated deterministically.
            List<Integer> indices = new ArrayList<>();

            for (int i = 0; i < X.length; i++) {
                indices.add(i);
            }

            Collections.shuffle(indices, new Random(42));

            int trainCount = X.length * 75 / 100;

            double[][] trainX = new double[trainCount][];
            double[] trainY = new double[trainCount];

            double[][] testX = new double[X.length - trainCount][];
            double[] testY = new double[X.length - trainCount];

            for (int i = 0; i < indices.size(); i++) {
                int original = indices.get(i);

                if (i < trainCount) {
                    trainX[i] = X[original].clone();
                    trainY[i] = y[original];
                } else {
                    int targetIndex = i - trainCount;

                    testX[targetIndex] = X[original].clone();
                    testY[targetIndex] = y[original];
                }
            }

            LinearRegressionModel model =
                ModelTrainer.train(features, trainX, trainY);

            InterpretationPolicy policy =
                new InterpretationPolicy(
                    20,
                    100,
                    5,
                    100,
                    false
                );

            InterpretationService service =
                new InterpretationService(
                    model,
                    features,
                    policy
                );

            AuditService audit = new AuditService();

            InterpretationWorkflow workflow =
                new InterpretationWorkflow(service, audit);

            workflow.subscribe(event ->
                System.out.printf(
                    "[AUDIT] %s | %s | %s%n",
                    event.timestamp(),
                    event.status(),
                    event.detail()
                )
            );

            Map<String, Object> result =
                workflow.execute(testX, testY);

            System.out.println("\nMODEL EVALUATION");
            System.out.println(result.get("evaluation"));

            System.out.println("\nCOEFFICIENT IMPORTANCE");

            for (ImportanceResult item :
                (List<ImportanceResult>) result.get("coefficients")) {
                System.out.println(item);
            }

            System.out.println("\nPERMUTATION IMPORTANCE");

            for (ImportanceResult item :
                (List<ImportanceResult>) result.get("permutation")) {
                System.out.println(item);
            }

            System.out.println("\nPARTIAL DEPENDENCE");

            for (DependenceResult item :
                (List<DependenceResult>) result.get("partialDependence")) {
                System.out.println("\nFeature: " + item.feature());

                for (DependencePoint point : item.points()) {
                    System.out.printf(
                        "Value: %.4f | Average prediction: %.4f%n",
                        point.featureValue(),
                        point.averagePrediction()
                    );
                }
            }

            System.out.println(
                "\nAudit events: " + audit.events().size()
            );

        } catch (RuntimeException exception) {
            System.err.println(
                "Interpretability workflow failed: " +
                exception.getMessage()
            );

            System.exit(1);
        }
    }
}
