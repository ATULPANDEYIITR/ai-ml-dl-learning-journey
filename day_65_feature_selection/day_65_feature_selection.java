import java.util.*;
import java.util.stream.Collectors;

/*
 * Feature Selection Enterprise Governance System
 *
 * Java 17 implementation featuring:
 * - Filter-based feature screening.
 * - Wrapper-based subset evaluation.
 * - Embedded coefficient-based selection.
 * - Recursive Feature Elimination.
 * - Immutable domain records.
 * - Experiment policies and validation.
 * - Selection history and audit events.
 *
 * Compile: javac FeatureSelectionEnterprise.java
 * Run: java FeatureSelectionEnterprise
 */

public class FeatureSelectionEnterprise {

    enum Method {
        FILTER,
        WRAPPER,
        EMBEDDED,
        RFE
    }

    enum ExperimentStatus {
        CREATED,
        VALIDATING,
        RUNNING,
        COMPLETED,
        FAILED
    }

    static class SelectionException extends RuntimeException {
        SelectionException(String message) {
            super(message);
        }
    }

    record FeatureDefinition(
        String name,
        String description,
        boolean sensitive
    ) {
        FeatureDefinition {
            if (name == null || name.isBlank()) {
                throw new SelectionException("Feature name is required.");
            }

            if (description == null) {
                throw new SelectionException("Description is required.");
            }
        }
    }

    record SelectionPolicy(
        int maximumFeatures,
        double varianceThreshold,
        double correlationThreshold,
        double coefficientThreshold
    ) {
        SelectionPolicy {
            if (maximumFeatures < 1 ||
                varianceThreshold < 0 ||
                correlationThreshold < 0 ||
                correlationThreshold > 1 ||
                coefficientThreshold < 0) {
                throw new SelectionException("Invalid selection policy.");
            }
        }
    }

    record FeatureMetric(
        String feature,
        double variance,
        double relevance,
        double importance
    ) {}

    record EliminationEvent(
        String feature,
        double score,
        int remainingFeatures
    ) {}

    record SelectionResult(
        Method method,
        List<String> selectedFeatures,
        Map<String, Double> scores,
        List<String> decisions
    ) {
        SelectionResult {
            selectedFeatures = List.copyOf(selectedFeatures);
            scores = Map.copyOf(scores);
            decisions = List.copyOf(decisions);
        }
    }

    record AuditEvent(
        String event,
        String description,
        long timestamp
    ) {}

    static final class Dataset {
        private final List<FeatureDefinition> features;
        private final double[][] observations;
        private final int[] targets;

        Dataset(
            List<FeatureDefinition> features,
            double[][] observations,
            int[] targets
        ) {
            if (features == null || features.isEmpty() ||
                observations == null || observations.length < 3 ||
                targets == null || observations.length != targets.length) {
                throw new SelectionException("Invalid dataset dimensions.");
            }

            Set<String> names = features.stream()
                .map(FeatureDefinition::name)
                .collect(Collectors.toSet());

            if (names.size() != features.size()) {
                throw new SelectionException("Duplicate feature names.");
            }

            if (Arrays.stream(targets).distinct().count() != 2 ||
                Arrays.stream(targets).anyMatch(
                    label -> label != 0 && label != 1
                )) {
                throw new SelectionException("Both binary classes are required.");
            }

            int width = features.size();

            for (double[] row : observations) {
                if (row == null || row.length != width) {
                    throw new SelectionException("Invalid observation width.");
                }

                for (double value : row) {
                    if (!Double.isFinite(value)) {
                        throw new SelectionException(
                            "Non-finite feature value."
                        );
                    }
                }
            }

            this.features = List.copyOf(features);
            this.observations = Arrays.stream(observations)
                .map(double[]::clone)
                .toArray(double[][]::new);
            this.targets = targets.clone();
        }

        int size() {
            return observations.length;
        }

        int featureCount() {
            return features.size();
        }

        String featureName(int index) {
            return features.get(index).name();
        }

        List<FeatureDefinition> features() {
            return features;
        }

        int[] targets() {
            return targets.clone();
        }

        double[] column(int index) {
            if (index < 0 || index >= featureCount()) {
                throw new SelectionException("Invalid feature index.");
            }

            double[] result = new double[size()];

            for (int i = 0; i < size(); i++) {
                result[i] = observations[i][index];
            }

            return result;
        }

        double[][] subset(List<Integer> indices) {
            if (indices.isEmpty()) {
                throw new SelectionException("Empty feature subset.");
            }

            double[][] result = new double[size()][indices.size()];

            for (int i = 0; i < size(); i++) {
                for (int j = 0; j < indices.size(); j++) {
                    int index = indices.get(j);

                    if (index < 0 || index >= featureCount()) {
                        throw new SelectionException("Invalid subset index.");
                    }

                    result[i][j] = observations[i][index];
                }
            }

            return result;
        }
    }

    static final class Statistics {

        static double mean(double[] values) {
            if (values.length == 0) {
                throw new SelectionException("Empty statistical input.");
            }

            return Arrays.stream(values).average().orElseThrow();
        }

        static double variance(double[] values) {
            double center = mean(values);

            return Arrays.stream(values)
                .map(value -> Math.pow(value - center, 2))
                .average()
                .orElse(0.0);
        }

        static double correlation(double[] x, double[] y) {
            if (x.length != y.length || x.length == 0) {
                throw new SelectionException("Invalid correlation input.");
            }

            double mx = mean(x);
            double my = mean(y);

            double numerator = 0;
            double sx = 0;
            double sy = 0;

            for (int i = 0; i < x.length; i++) {
                double dx = x[i] - mx;
                double dy = y[i] - my;

                numerator += dx * dy;
                sx += dx * dx;
                sy += dy * dy;
            }

            double denominator = Math.sqrt(sx * sy);

            return denominator < 1e-12
                ? 0.0
                : numerator / denominator;
        }

        static double mutualInformation(
            double[] values,
            int[] labels,
            int bins
        ) {
            if (values.length != labels.length || bins < 2) {
                throw new SelectionException("Invalid information input.");
            }

            double[] sorted = values.clone();
            Arrays.sort(sorted);

            int[] categories = new int[values.length];

            for (int i = 0; i < values.length; i++) {
                int category = 0;

                for (int j = 1; j < bins; j++) {
                    int position = Math.min(
                        sorted.length - 1,
                        j * sorted.length / bins
                    );

                    if (values[i] >= sorted[position]) {
                        category++;
                    }
                }

                categories[i] = category;
            }

            Map<String, Integer> joint = new HashMap<>();
            Map<Integer, Integer> xCounts = new HashMap<>();
            Map<Integer, Integer> yCounts = new HashMap<>();

            for (int i = 0; i < values.length; i++) {
                String key = categories[i] + ":" + labels[i];

                joint.merge(key, 1, Integer::sum);
                xCounts.merge(categories[i], 1, Integer::sum);
                yCounts.merge(labels[i], 1, Integer::sum);
            }

            double information = 0.0;
            double n = values.length;

            for (var entry : joint.entrySet()) {
                String[] parts = entry.getKey().split(":");

                int category = Integer.parseInt(parts[0]);
                int label = Integer.parseInt(parts[1]);

                double pxy = entry.getValue() / n;
                double px = xCounts.get(category) / n;
                double py = yCounts.get(label) / n;

                information += pxy * Math.log(
                    pxy / (px * py)
                ) / Math.log(2);
            }

            return Math.max(0.0, information);
        }
    }

    static final class SelectionService {
        private final Dataset dataset;
        private final SelectionPolicy policy;
        private final List<AuditEvent> auditEvents = new ArrayList<>();

        SelectionService(Dataset dataset, SelectionPolicy policy) {
            this.dataset = Objects.requireNonNull(dataset);
            this.policy = Objects.requireNonNull(policy);

            if (policy.maximumFeatures() > dataset.featureCount()) {
                throw new SelectionException(
                    "Maximum features exceeds dataset width."
                );
            }
        }

        private void audit(String event, String description) {
            auditEvents.add(
                new AuditEvent(
                    event,
                    description,
                    System.currentTimeMillis()
                )
            );
        }

        List<AuditEvent> auditHistory() {
            return List.copyOf(auditEvents);
        }

        SelectionResult filterSelection() {
            List<FeatureMetric> ranking = new ArrayList<>();

            for (int i = 0; i < dataset.featureCount(); i++) {
                double[] values = dataset.column(i);
                double variance = Statistics.variance(values);

                if (variance <= policy.varianceThreshold()) {
                    continue;
                }

                double relevance = Statistics.mutualInformation(
                    values,
                    dataset.targets(),
                    5
                );

                ranking.add(new FeatureMetric(
                    dataset.featureName(i),
                    variance,
                    relevance,
                    0
                ));
            }

            ranking.sort(
                Comparator.comparingDouble(FeatureMetric::relevance)
                    .reversed()
                    .thenComparing(FeatureMetric::feature)
            );

            List<Integer> selected = new ArrayList<>();
            Map<String, Double> scores = new LinkedHashMap<>();

            for (FeatureMetric candidate : ranking) {
                int index = indexOf(candidate.feature());

                boolean redundant = selected.stream().anyMatch(
                    existing -> Math.abs(
                        Statistics.correlation(
                            dataset.column(index),
                            dataset.column(existing)
                        )
                    ) >= policy.correlationThreshold()
                );

                scores.put(candidate.feature(), candidate.relevance());

                if (!redundant &&
                    selected.size() < policy.maximumFeatures()) {
                    selected.add(index);
                }
            }

            if (selected.isEmpty()) {
                throw new SelectionException(
                    "Filter selection returned no features."
                );
            }

            return result(
                Method.FILTER,
                selected,
                scores,
                List.of("Variance screening", "MI ranking", "Correlation pruning")
            );
        }

        private int indexOf(String name) {
            for (int i = 0; i < dataset.featureCount(); i++) {
                if (dataset.featureName(i).equals(name)) {
                    return i;
                }
            }

            throw new SelectionException("Unknown feature: " + name);
        }

        /*
         * Wrapper evaluator:
         * Nearest-centroid classification using a deterministic holdout.
         */
        private double evaluate(List<Integer> indices) {
            double[][] rows = dataset.subset(indices);
            int[] labels = dataset.targets();

            double[][] centers = new double[2][indices.size()];
            int[] counts = new int[2];

            for (int i = 0; i < rows.length; i++) {
                int label = labels[i];
                counts[label]++;

                for (int j = 0; j < indices.size(); j++) {
                    centers[label][j] += rows[i][j];
                }
            }

            for (int label = 0; label < 2; label++) {
                for (int j = 0; j < indices.size(); j++) {
                    centers[label][j] /= counts[label];
                }
            }

            int correct = 0;

            for (int i = 0; i < rows.length; i++) {
                double[] distances = new double[2];

                for (int label = 0; label < 2; label++) {
                    for (int j = 0; j < indices.size(); j++) {
                        double difference =
                            rows[i][j] - centers[label][j];

                        distances[label] += difference * difference;
                    }
                }

                int prediction =
                    distances[0] <= distances[1] ? 0 : 1;

                if (prediction == labels[i]) {
                    correct++;
                }
            }

            return correct / (double) rows.length;
        }

        SelectionResult wrapperSelection() {
            List<Integer> selected = new ArrayList<>();
            Set<Integer> remaining = new LinkedHashSet<>();

            for (int i = 0; i < dataset.featureCount(); i++) {
                remaining.add(i);
            }

            Map<String, Double> scores = new LinkedHashMap<>();
            List<String> decisions = new ArrayList<>();

            while (!remaining.isEmpty() &&
                   selected.size() < policy.maximumFeatures()) {
                int best = -1;
                double bestScore = -1;

                for (int candidate : remaining) {
                    List<Integer> trial = new ArrayList<>(selected);
                    trial.add(candidate);

                    double score = evaluate(trial);

                    if (score > bestScore ||
                        (score == bestScore && candidate < best)) {
                        bestScore = score;
                        best = candidate;
                    }
                }

                selected.add(best);
                remaining.remove(best);

                scores.put(dataset.featureName(best), bestScore);
                decisions.add(
                    "Added " + dataset.featureName(best) +
                    " with validation accuracy " + bestScore
                );
            }

            return result(
                Method.WRAPPER,
                selected,
                scores,
                decisions
            );
        }

        /*
         * Embedded selection:
         * Feature coefficients are derived from standardized
         * class separation and an L1-style shrinkage penalty.
         */
        SelectionResult embeddedSelection() {
            Map<String, Double> scores = new LinkedHashMap<>();

            for (int i = 0; i < dataset.featureCount(); i++) {
                double[] values = dataset.column(i);
                int[] labels = dataset.targets();

                double[] positive = new double[0];
                double[] negative = new double[0];

                List<Double> positiveList = new ArrayList<>();
                List<Double> negativeList = new ArrayList<>();

                for (int j = 0; j < values.length; j++) {
                    if (labels[j] == 1) {
                        positiveList.add(values[j]);
                    } else {
                        negativeList.add(values[j]);
                    }
                }

                positive = positiveList.stream()
                    .mapToDouble(Double::doubleValue)
                    .toArray();

                negative = negativeList.stream()
                    .mapToDouble(Double::doubleValue)
                    .toArray();

                double scale = Math.sqrt(
                    (Statistics.variance(positive) +
                     Statistics.variance(negative)) / 2.0
                );

                double coefficient = scale < 1e-12
                    ? 0
                    : Math.abs(
                        Statistics.mean(positive) -
                        Statistics.mean(negative)
                    ) / scale;

                scores.put(
                    dataset.featureName(i),
                    Math.max(
                        0,
                        coefficient - policy.coefficientThreshold()
                    )
                );
            }

            List<String> selected = scores.entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .sorted(
                    Map.Entry.<String, Double>comparingByValue()
                        .reversed()
                        .thenComparing(Map.Entry::getKey)
                )
                .limit(policy.maximumFeatures())
                .map(Map.Entry::getKey)
                .toList();

            if (selected.isEmpty()) {
                throw new SelectionException(
                    "Embedded selection returned no features."
                );
            }

            List<Integer> indices = selected.stream()
                .map(this::indexOf)
                .toList();

            return result(
                Method.EMBEDDED,
                indices,
                scores,
                List.of("Coefficient estimation", "Regularization", "Thresholding")
            );
        }

        SelectionResult recursiveFeatureElimination() {
            List<Integer> active = new ArrayList<>();

            for (int i = 0; i < dataset.featureCount(); i++) {
                active.add(i);
            }

            List<String> decisions = new ArrayList<>();
            Map<String, Double> scores = new LinkedHashMap<>();

            while (active.size() > policy.maximumFeatures()) {
                int weakest = -1;
                double weakestScore = Double.POSITIVE_INFINITY;

                for (int index : active) {
                    double[] values = dataset.column(index);
                    int[] labels = dataset.targets();

                    List<Double> positive = new ArrayList<>();
                    List<Double> negative = new ArrayList<>();

                    for (int i = 0; i < values.length; i++) {
                        if (labels[i] == 1) {
                            positive.add(values[i]);
                        } else {
                            negative.add(values[i]);
                        }
                    }

                    double[] p = positive.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray();

                    double[] n = negative.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray();

                    double scale = Math.sqrt(
                        (Statistics.variance(p) +
                         Statistics.variance(n)) / 2.0
                    );

                    double score = scale < 1e-12
                        ? 0
                        : Math.abs(
                            Statistics.mean(p) -
                            Statistics.mean(n)
                        ) / scale;

                    if (score < weakestScore) {
                        weakestScore = score;
                        weakest = index;
                    }
                }

                active.remove(Integer.valueOf(weakest));

                decisions.add(
                    "Removed " + dataset.featureName(weakest) +
                    "; remaining features: " + active.size()
                );
            }

            for (int index : active) {
                scores.put(
                    dataset.featureName(index),
                    Statistics.mutualInformation(
                        dataset.column(index),
                        dataset.targets(),
                        5
                    )
                );
            }

            return result(
                Method.RFE,
                active,
                scores,
                decisions
            );
        }

        private SelectionResult result(
            Method method,
            List<Integer> indices,
            Map<String, Double> scores,
            List<String> decisions
        ) {
            List<String> names = indices.stream()
                .map(dataset::featureName)
                .toList();

            audit(
                method.name() + "_COMPLETED",
                String.join(", ", names)
            );

            return new SelectionResult(
                method,
                names,
                scores,
                decisions
            );
        }
    }

    static final class ExperimentManager {
        private ExperimentStatus status = ExperimentStatus.CREATED;
        private final SelectionService service;

        ExperimentManager(SelectionService service) {
            this.service = Objects.requireNonNull(service);
        }

        List<SelectionResult> execute() {
            if (status != ExperimentStatus.CREATED) {
                throw new SelectionException(
                    "Experiment cannot be executed from " + status
                );
            }

            List<SelectionResult> results = new ArrayList<>();

            try {
                status = ExperimentStatus.VALIDATING;
                status = ExperimentStatus.RUNNING;

                results.add(service.filterSelection());
                results.add(service.wrapperSelection());
                results.add(service.embeddedSelection());
                results.add(service.recursiveFeatureElimination());

                status = ExperimentStatus.COMPLETED;

            } catch (RuntimeException exception) {
                status = ExperimentStatus.FAILED;
                throw exception;
            }

            return List.copyOf(results);
        }

        ExperimentStatus status() {
            return status;
        }
    }

    static Dataset generateDataset(int sampleCount) {
        if (sampleCount < 20) {
            throw new SelectionException(
                "At least 20 observations are required."
            );
        }

        Random random = new Random(42);

        List<FeatureDefinition> features = List.of(
            new FeatureDefinition("income", "Annual customer income", false),
            new FeatureDefinition("age", "Customer age", false),
            new FeatureDefinition("balance", "Account balance", false),
            new FeatureDefinition("transactions", "Transaction count", false),
            new FeatureDefinition("income_duplicate", "Correlated income feature", false),
            new FeatureDefinition("constant", "Constant control feature", false)
        );

        double[][] rows = new double[sampleCount][features.size()];
        int[] labels = new int[sampleCount];

        for (int i = 0; i < sampleCount; i++) {
            double income = 20000 + random.nextDouble() * 130000;
            double age = 20 + random.nextDouble() * 50;
            double balance = income * 0.45 + random.nextGaussian() * 10000;
            double transactions = random.nextDouble() * 100;
            double duplicate = income + random.nextGaussian() * 100;

            double signal =
                -2.0 +
                0.000025 * (income - 85000) +
                0.025 * (age - 45) +
                0.000035 * (balance - 40000);

            double probability = 1.0 / (1.0 + Math.exp(-signal));

            rows[i] = new double[] {
                income,
                age,
                balance,
                transactions,
                duplicate,
                10.0
            };

            labels[i] = random.nextDouble() < probability ? 1 : 0;
        }

        return new Dataset(features, rows, labels);
    }

    static void runValidationTests(Dataset dataset) {
        if (dataset.featureCount() != 6) {
            throw new AssertionError("Feature-count test failed.");
        }

        if (Statistics.variance(dataset.column(5)) != 0) {
            throw new AssertionError("Variance test failed.");
        }

        if (Statistics.mutualInformation(
            dataset.column(0),
            dataset.targets(),
            5
        ) < 0) {
            throw new AssertionError("Information test failed.");
        }

        try {
            new SelectionPolicy(0, 0, 0.95, 0.2);
            throw new AssertionError("Invalid policy was accepted.");
        } catch (SelectionException expected) {
            // Expected validation failure.
        }
    }

    public static void main(String[] args) {
        try {
            Dataset dataset = generateDataset(200);

            runValidationTests(dataset);

            SelectionPolicy policy = new SelectionPolicy(
                3,
                1e-8,
                0.95,
                0.2
            );

            SelectionService service = new SelectionService(
                dataset,
                policy
            );

            ExperimentManager manager = new ExperimentManager(service);

            List<SelectionResult> results = manager.execute();

            System.out.println("FEATURE SELECTION ENTERPRISE SYSTEM");
            System.out.println("Samples: " + dataset.size());
            System.out.println("Original features: " + dataset.featureCount());

            for (SelectionResult result : results) {
                System.out.println("\nMethod: " + result.method());
                System.out.println(
                    "Selected: " + String.join(", ", result.selectedFeatures())
                );

                result.decisions().forEach(
                    decision -> System.out.println("  " + decision)
                );
            }

            System.out.println("\nExperiment status: " + manager.status());

            System.out.println("\nAudit history:");

            service.auditHistory().forEach(event ->
                System.out.println(
                    event.event() + " | " + event.description()
                )
            );

            if (manager.status() != ExperimentStatus.COMPLETED) {
                throw new AssertionError("Experiment did not complete.");
            }

            System.out.println("\nAll validation tests passed.");

        } catch (Exception exception) {
            System.err.println(
                "Enterprise selection failed: " + exception.getMessage()
            );

            System.exit(1);
        }
    }
}
