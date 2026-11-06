import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;

/**
 * Enterprise Explainable AI demonstration.
 *
 * Covers:
 * - Model-independent explanation interfaces.
 * - Exact interventional SHAP attribution.
 * - LIME-style local surrogate regression.
 * - Global feature importance.
 * - Explanation policies and audit events.
 * - Immutable domain objects and validation.
 *
 * Java 17, no external dependencies.
 */
public class ExplainableAIEnterprise {

    enum Method {
        SHAP,
        LIME,
        GLOBAL
    }

    enum Decision {
        APPROVED,
        REVIEW_REQUIRED
    }

    record FeatureDefinition(
        String name,
        double minimum,
        double maximum,
        double defaultValue,
        double coefficient
    ) {
        FeatureDefinition {
            Objects.requireNonNull(name);

            if (
                !Double.isFinite(minimum) ||
                !Double.isFinite(maximum) ||
                minimum >= maximum
            ) {
                throw new IllegalArgumentException(
                    "Invalid feature bounds"
                );
            }

            if (
                defaultValue < minimum ||
                defaultValue > maximum
            ) {
                throw new IllegalArgumentException(
                    "Default outside feature range"
                );
            }
        }

        void validate(double value) {
            if (
                !Double.isFinite(value) ||
                value < minimum ||
                value > maximum
            ) {
                throw new ValidationException(
                    "Invalid value for " + name
                );
            }
        }
    }

    static class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    static class ExplanationException extends RuntimeException {
        ExplanationException(String message) {
            super(message);
        }
    }

    record Observation(Map<String, Double> values) {

        Observation {
            Objects.requireNonNull(values);

            values = Collections.unmodifiableMap(
                new LinkedHashMap<>(values)
            );
        }

        double get(String name) {
            Double value = values.get(name);

            if (value == null) {
                throw new ValidationException(
                    "Missing feature: " + name
                );
            }

            return value;
        }
    }

    record ModelPrediction(
        double probability,
        Decision decision,
        Instant timestamp
    ) {}

    record Explanation(
        Method method,
        double prediction,
        double baseline,
        Map<String, Double> contributions,
        double reconstructed,
        Map<String, String> metadata
    ) {
        Explanation {
            contributions = Collections.unmodifiableMap(
                new LinkedHashMap<>(contributions)
            );

            metadata = Collections.unmodifiableMap(
                new LinkedHashMap<>(metadata)
            );
        }
    }

    record ExplanationPolicy(
        int requiredBackgroundSamples,
        int maximumExactFeatures,
        double additivityTolerance,
        int minimumPerturbations
    ) {
        ExplanationPolicy {
            if (
                requiredBackgroundSamples < 1 ||
                maximumExactFeatures < 1 ||
                maximumExactFeatures > 20 ||
                additivityTolerance <= 0 ||
                minimumPerturbations < 20
            ) {
                throw new IllegalArgumentException(
                    "Invalid explanation policy"
                );
            }
        }
    }

    record AuditEvent(
        String eventType,
        Instant timestamp,
        String details
    ) {}

    interface PredictiveModel {
        double probability(Observation observation);

        void validate(Observation observation);
    }

    static final class CreditRiskModel implements PredictiveModel {

        private static final List<FeatureDefinition> DEFINITIONS =
            List.of(
                new FeatureDefinition(
                    "income", 10000, 200000, 60000, 0.000015
                ),
                new FeatureDefinition(
                    "debtRatio", 0, 1, 0.35, -2.2
                ),
                new FeatureDefinition(
                    "creditHistory", 0, 30, 8, 0.09
                ),
                new FeatureDefinition(
                    "missedPayments", 0, 20, 1, -0.45
                ),
                new FeatureDefinition(
                    "employmentYears", 0, 40, 5, 0.055
                )
            );

        private static final Map<String, FeatureDefinition> FEATURES;

        static {
            Map<String, FeatureDefinition> definitions =
                new LinkedHashMap<>();

            for (FeatureDefinition feature : DEFINITIONS) {
                definitions.put(feature.name(), feature);
            }

            FEATURES = Collections.unmodifiableMap(definitions);
        }

        private final double intercept = -0.5;

        List<FeatureDefinition> definitions() {
            return DEFINITIONS;
        }

        @Override
        public void validate(Observation observation) {
            Objects.requireNonNull(observation);

            if (!observation.values().keySet().equals(
                FEATURES.keySet()
            )) {
                throw new ValidationException(
                    "Observation has missing or unexpected features"
                );
            }

            FEATURES.forEach((name, definition) ->
                definition.validate(observation.get(name))
            );
        }

        @Override
        public double probability(Observation observation) {
            validate(observation);

            double score = intercept;

            for (FeatureDefinition feature : DEFINITIONS) {
                score += observation.get(feature.name()) *
                    feature.coefficient();
            }

            if (score >= 0) {
                return 1.0 / (1.0 + Math.exp(-score));
            }

            double exponential = Math.exp(score);
            return exponential / (1.0 + exponential);
        }

        ModelPrediction predict(
            Observation observation,
            double threshold
        ) {
            if (
                !Double.isFinite(threshold) ||
                threshold <= 0 ||
                threshold >= 1
            ) {
                throw new ValidationException(
                    "Invalid classification threshold"
                );
            }

            double probability = probability(observation);

            return new ModelPrediction(
                probability,
                probability >= threshold
                    ? Decision.APPROVED
                    : Decision.REVIEW_REQUIRED,
                Instant.now()
            );
        }
    }

    static final class LinearAlgebra {

        private LinearAlgebra() {}

        static double[] solve(
            double[][] matrix,
            double[] vector
        ) {
            int n = vector.length;
            double[][] a = new double[n][n + 1];

            for (int i = 0; i < n; i++) {
                System.arraycopy(matrix[i], 0, a[i], 0, n);
                a[i][n] = vector[i];
            }

            for (int col = 0; col < n; col++) {
                int pivot = col;

                for (int row = col + 1; row < n; row++) {
                    if (
                        Math.abs(a[row][col]) >
                        Math.abs(a[pivot][col])
                    ) {
                        pivot = row;
                    }
                }

                if (Math.abs(a[pivot][col]) < 1e-12) {
                    throw new ExplanationException(
                        "Singular regression matrix"
                    );
                }

                double[] temporary = a[col];
                a[col] = a[pivot];
                a[pivot] = temporary;

                double divisor = a[col][col];

                for (int j = col; j <= n; j++) {
                    a[col][j] /= divisor;
                }

                for (int row = 0; row < n; row++) {
                    if (row == col) continue;

                    double factor = a[row][col];

                    for (int j = col; j <= n; j++) {
                        a[row][j] -= factor * a[col][j];
                    }
                }
            }

            double[] result = new double[n];

            for (int i = 0; i < n; i++) {
                result[i] = a[i][n];
            }

            return result;
        }
    }

    static final class ExplanationService {

        private final PredictiveModel model;
        private final CreditRiskModel creditModel;
        private final List<Observation> background;
        private final ExplanationPolicy policy;

        ExplanationService(
            CreditRiskModel model,
            List<Observation> background,
            ExplanationPolicy policy
        ) {
            this.model = Objects.requireNonNull(model);
            this.creditModel = model;
            this.policy = Objects.requireNonNull(policy);

            if (
                background == null ||
                background.size() <
                policy.requiredBackgroundSamples()
            ) {
                throw new ValidationException(
                    "Insufficient background observations"
                );
            }

            background.forEach(model::validate);

            this.background = List.copyOf(background);
        }

        private double mean(List<Double> values) {
            return values.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElseThrow(() ->
                    new ExplanationException("Empty mean")
                );
        }

        private double coalitionValue(
            Observation observation,
            int mask
        ) {
            List<Double> predictions = new ArrayList<>();

            List<FeatureDefinition> definitions =
                creditModel.definitions();

            for (Observation reference : background) {
                Map<String, Double> hybrid = new LinkedHashMap<>();

                for (int i = 0; i < definitions.size(); i++) {
                    String name = definitions.get(i).name();

                    hybrid.put(
                        name,
                        (mask & (1 << i)) != 0
                            ? observation.get(name)
                            : reference.get(name)
                    );
                }

                predictions.add(
                    model.probability(new Observation(hybrid))
                );
            }

            return mean(predictions);
        }

        private static double factorial(int n) {
            double value = 1;

            for (int i = 2; i <= n; i++) {
                value *= i;
            }

            return value;
        }

        Explanation shap(Observation observation) {
            model.validate(observation);

            List<FeatureDefinition> definitions =
                creditModel.definitions();

            int n = definitions.size();

            if (n > policy.maximumExactFeatures()) {
                throw new ExplanationException(
                    "Feature count exceeds exact SHAP policy"
                );
            }

            int coalitionCount = 1 << n;
            double[] values = new double[coalitionCount];

            for (int mask = 0; mask < coalitionCount; mask++) {
                values[mask] = coalitionValue(observation, mask);
            }

            Map<String, Double> contributions =
                new LinkedHashMap<>();

            for (int feature = 0; feature < n; feature++) {
                double attribution = 0;

                for (
                    int mask = 0;
                    mask < coalitionCount;
                    mask++
                ) {
                    if ((mask & (1 << feature)) != 0) {
                        continue;
                    }

                    int subsetSize = Integer.bitCount(mask);

                    double weight =
                        factorial(subsetSize) *
                        factorial(n - subsetSize - 1) /
                        factorial(n);

                    attribution += weight * (
                        values[mask | (1 << feature)] -
                        values[mask]
                    );
                }

                contributions.put(
                    definitions.get(feature).name(),
                    attribution
                );
            }

            double prediction = model.probability(observation);
            double baseline = values[0];

            double reconstructed = baseline +
                contributions.values().stream()
                    .mapToDouble(Double::doubleValue)
                    .sum();

            if (
                Math.abs(prediction - reconstructed) >
                policy.additivityTolerance()
            ) {
                throw new ExplanationException(
                    "SHAP additivity policy violated"
                );
            }

            return new Explanation(
                Method.SHAP,
                prediction,
                baseline,
                contributions,
                reconstructed,
                Map.of(
                    "algorithm", "Exact interventional SHAP",
                    "coalitions", String.valueOf(coalitionCount),
                    "backgroundSamples",
                    String.valueOf(background.size())
                )
            );
        }

        Explanation lime(
            Observation observation,
            int perturbations,
            double kernelWidth,
            long seed
        ) {
            model.validate(observation);

            if (
                perturbations < policy.minimumPerturbations() ||
                perturbations > 100000
            ) {
                throw new ValidationException(
                    "Perturbation count violates policy"
                );
            }

            if (
                !Double.isFinite(kernelWidth) ||
                kernelWidth <= 0
            ) {
                throw new ValidationException(
                    "Kernel width must be positive"
                );
            }

            List<FeatureDefinition> definitions =
                creditModel.definitions();

            int n = definitions.size();
            int size = n + 1;

            double[][] matrix = new double[size][size];
            double[] vector = new double[size];

            Random random = new Random(seed);

            for (int iteration = 0; iteration < perturbations; iteration++) {
                Map<String, Double> perturbed =
                    new LinkedHashMap<>();

                double[] row = new double[size];
                row[0] = 1.0;

                double distanceSquared = 0;

                for (int i = 0; i < n; i++) {
                    FeatureDefinition feature = definitions.get(i);

                    double range =
                        feature.maximum() - feature.minimum();

                    double value = observation.get(feature.name()) +
                        random.nextGaussian() * range * 0.15;

                    value = Math.max(
                        feature.minimum(),
                        Math.min(feature.maximum(), value)
                    );

                    perturbed.put(feature.name(), value);

                    row[i + 1] =
                        (value - observation.get(feature.name())) /
                        range;

                    distanceSquared +=
                        row[i + 1] * row[i + 1];
                }

                double weight = Math.exp(
                    -distanceSquared /
                    (kernelWidth * kernelWidth)
                );

                double target = model.probability(
                    new Observation(perturbed)
                );

                for (int i = 0; i < size; i++) {
                    vector[i] += weight * row[i] * target;

                    for (int j = 0; j < size; j++) {
                        matrix[i][j] +=
                            weight * row[i] * row[j];
                    }
                }
            }

            for (int i = 1; i < size; i++) {
                matrix[i][i] += 1e-6;
            }

            double[] coefficients =
                LinearAlgebra.solve(matrix, vector);

            Map<String, Double> contributions =
                new LinkedHashMap<>();

            for (int i = 0; i < n; i++) {
                contributions.put(
                    definitions.get(i).name(),
                    coefficients[i + 1]
                );
            }

            return new Explanation(
                Method.LIME,
                model.probability(observation),
                coefficients[0],
                contributions,
                coefficients[0],
                Map.of(
                    "algorithm", "Weighted local linear regression",
                    "perturbations", String.valueOf(perturbations),
                    "kernelWidth", String.valueOf(kernelWidth),
                    "seed", String.valueOf(seed)
                )
            );
        }

        Map<String, Double> globalImportance(
            List<Observation> dataset
        ) {
            if (dataset == null || dataset.isEmpty()) {
                throw new ValidationException(
                    "Global evaluation dataset cannot be empty"
                );
            }

            Map<String, Double> totals = new LinkedHashMap<>();

            for (FeatureDefinition feature :
                creditModel.definitions()) {
                totals.put(feature.name(), 0.0);
            }

            for (Observation observation : dataset) {
                Explanation explanation = shap(observation);

                explanation.contributions().forEach(
                    (name, value) ->
                        totals.merge(name, Math.abs(value), Double::sum)
                );
            }

            Map<String, Double> result = new LinkedHashMap<>();

            totals.entrySet().stream()
                .sorted(
                    Map.Entry.<String, Double>comparingByValue()
                        .reversed()
                )
                .forEach(entry ->
                    result.put(
                        entry.getKey(),
                        entry.getValue() / dataset.size()
                    )
                );

            return Collections.unmodifiableMap(result);
        }
    }

    static final class AuditService {

        private final List<AuditEvent> events =
            new ArrayList<>();

        void record(String type, String details) {
            events.add(
                new AuditEvent(type, Instant.now(), details)
            );
        }

        List<AuditEvent> events() {
            return List.copyOf(events);
        }
    }

    static Observation observation(
        double income,
        double debtRatio,
        double creditHistory,
        double missedPayments,
        double employmentYears
    ) {
        Map<String, Double> values = new LinkedHashMap<>();

        values.put("income", income);
        values.put("debtRatio", debtRatio);
        values.put("creditHistory", creditHistory);
        values.put("missedPayments", missedPayments);
        values.put("employmentYears", employmentYears);

        return new Observation(values);
    }

    static String toJson(
        Observation applicant,
        ModelPrediction prediction,
        Explanation shap,
        Explanation lime,
        Map<String, Double> global
    ) {
        StringBuilder json = new StringBuilder();

        json.append("{\n");
        json.append("  \"generatedAt\": \"")
            .append(Instant.now()).append("\",\n");

        json.append("  \"prediction\": {\n");
        json.append("    \"probability\": ")
            .append(prediction.probability()).append(",\n");
        json.append("    \"decision\": \"")
            .append(prediction.decision()).append("\"\n");
        json.append("  },\n");

        json.append("  \"applicant\": {");

        int index = 0;

        for (var entry : applicant.values().entrySet()) {
            if (index++ > 0) json.append(",");

            json.append("\n    \"")
                .append(entry.getKey())
                .append("\": ")
                .append(entry.getValue());
        }

        json.append("\n  },\n");

        appendExplanation(json, "shap", shap);
        json.append(",\n");
        appendExplanation(json, "lime", lime);
        json.append(",\n");

        json.append("  \"globalImportance\": {");

        index = 0;

        for (var entry : global.entrySet()) {
            if (index++ > 0) json.append(",");

            json.append("\n    \"")
                .append(entry.getKey())
                .append("\": ")
                .append(entry.getValue());
        }

        json.append("\n  }\n}");
        return json.toString();
    }

    private static void appendExplanation(
        StringBuilder json,
        String name,
        Explanation explanation
    ) {
        json.append("  \"").append(name).append("\": {\n");
        json.append("    \"prediction\": ")
            .append(explanation.prediction()).append(",\n");
        json.append("    \"baseline\": ")
            .append(explanation.baseline()).append(",\n");
        json.append("    \"contributions\": {");

        int index = 0;

        for (var entry : explanation.contributions().entrySet()) {
            if (index++ > 0) json.append(",");

            json.append("\n      \"")
                .append(entry.getKey())
                .append("\": ")
                .append(entry.getValue());
        }

        json.append("\n    }\n  }");
    }

    public static void main(String[] args) {
        try {
            CreditRiskModel model = new CreditRiskModel();

            List<Observation> background = List.of(
                observation(45000, 0.55, 3, 4, 2),
                observation(70000, 0.25, 12, 0, 10),
                observation(55000, 0.40, 7, 2, 5),
                observation(95000, 0.20, 18, 0, 15)
            );

            Observation applicant =
                observation(85000, 0.30, 14, 1, 9);

            ExplanationPolicy policy = new ExplanationPolicy(
                3,
                20,
                1e-9,
                20
            );

            ExplanationService service =
                new ExplanationService(model, background, policy);

            AuditService audit = new AuditService();

            ModelPrediction prediction =
                model.predict(applicant, 0.5);

            Explanation shap = service.shap(applicant);

            Explanation lime = service.lime(
                applicant,
                1500,
                0.75,
                42
            );

            List<Observation> evaluation = new ArrayList<>(
                background
            );

            evaluation.add(applicant);

            evaluation.add(
                observation(120000, 0.15, 20, 0, 18)
            );

            Map<String, Double> global =
                service.globalImportance(evaluation);

            audit.record(
                "PREDICTION",
                prediction.decision().name()
            );

            audit.record(
                "SHAP",
                "Additivity error: " +
                    Math.abs(
                        shap.prediction() -
                        shap.reconstructed()
                    )
            );

            audit.record(
                "LIME",
                "Perturbation-based surrogate completed"
            );

            System.out.println(
                "ENTERPRISE EXPLAINABLE AI REPORT"
            );

            System.out.println(
                "================================"
            );

            System.out.printf(
                "Probability: %.6f%n",
                prediction.probability()
            );

            System.out.println(
                "Decision: " + prediction.decision()
            );

            System.out.printf(
                "%nSHAP baseline: %.6f%n",
                shap.baseline()
            );

            shap.contributions().forEach(
                (name, value) ->
                    System.out.printf(
                        "%-22s %+ .6f%n",
                        name,
                        value
                    )
            );

            System.out.println("\nLIME COEFFICIENTS");

            lime.contributions().forEach(
                (name, value) ->
                    System.out.printf(
                        "%-22s %+ .6f%n",
                        name,
                        value
                    )
            );

            System.out.println("\nGLOBAL IMPORTANCE");

            global.forEach(
                (name, value) ->
                    System.out.printf(
                        "%-22s %.6f%n",
                        name,
                        value
                    )
            );

            Path report = Path.of(
                "explainability_enterprise_report.json"
            );

            Files.writeString(
                report,
                toJson(applicant, prediction, shap, lime, global),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
            );

            System.out.println(
                "\nReport written to " + report.toAbsolutePath()
            );

            System.out.println(
                "Audit events: " + audit.events().size()
            );

        } catch (ValidationException |
                 ExplanationException |
                 IOException error) {

            System.err.println(
                "Enterprise explanation failed: " +
                error.getMessage()
            );

            System.exit(1);
        }
    }
}
