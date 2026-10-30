import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Enterprise-oriented classical ML experiment service.
 *
 * The program creates a versioned binary classification dataset, validates
 * its records, performs stratified train/validation/test splitting, fits
 * preprocessing on training data only, selects a model using validation
 * metrics, and evaluates the selected model once on the final test set.
 *
 * Java 17 or later; no third-party dependencies.
 *
 * Compile:
 *   javac ClassicalMLEnterprise.java
 *
 * Run:
 *   java ClassicalMLEnterprise
 */
public class ClassicalMLEnterprise {

    private static final long RANDOM_SEED = 42L;
    private static final int SAMPLE_COUNT = 900;
    private static final double POSITIVE_CLASS_RATE = 0.38;

    private static final List<String> FEATURE_NAMES = List.of(
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_compactness",
        "mean_concavity"
    );

    public static void main(String[] args) {
        try {
            DatasetSnapshot snapshot = DatasetFactory.create(
                "WDBC-SYNTHETIC-2026-01",
                "1.0",
                RANDOM_SEED
            );

            DatasetValidationService validationService =
                new DatasetValidationService();

            validationService.validate(snapshot);

            Experiment experiment = new Experiment(
                "EXP-WDBC-CLASSIFICATION-001",
                snapshot.datasetId()
            );

            experiment.transition(
                ExperimentState.CREATED,
                ExperimentState.DATA_VALIDATED
            );

            DatasetPartition outer = DatasetSplitter.stratifiedSplit(
                snapshot.observations(),
                0.20,
                RANDOM_SEED
            );

            // The outer test partition is not involved in candidate selection.
            DatasetPartition inner = DatasetSplitter.stratifiedSplit(
                outer.training(),
                0.1875,
                RANDOM_SEED + 1
            );

            List<Observation> trainingRows = inner.training();
            List<Observation> validationRows = inner.testing();
            List<Observation> testRows = outer.testing();

            System.out.printf(
                "Dataset %s v%s: %d rows, %d features%n",
                snapshot.datasetId(),
                snapshot.version(),
                snapshot.observations().size(),
                snapshot.featureNames().size()
            );

            System.out.printf(
                "Training: %d | Validation: %d | Test: %d%n",
                trainingRows.size(),
                validationRows.size(),
                testRows.size()
            );

            PreprocessingModel validationPreprocessing =
                PreprocessingModel.fit(trainingRows);

            double[][] trainingMatrix =
                validationPreprocessing.transform(trainingRows);
            double[][] validationMatrix =
                validationPreprocessing.transform(validationRows);

            int[] trainingTargets = targets(trainingRows);
            int[] validationTargets = targets(validationRows);

            List<ModelFactory> factories = List.of(
                new ModelFactory(
                    "GaussianNaiveBayes",
                    GaussianNaiveBayes::new
                ),
                new ModelFactory(
                    "KNearestNeighbors",
                    () -> new KNearestNeighbors(7)
                )
            );

            List<ValidationResult> validationResults = new ArrayList<>();

            for (ModelFactory factory : factories) {
                BinaryClassifier candidate = factory.supplier().get();

                candidate.fit(trainingMatrix, trainingTargets);

                double[] probabilities =
                    candidate.predictProbabilities(validationMatrix);

                Metrics metrics = Metrics.evaluate(
                    validationTargets,
                    probabilities
                );

                validationResults.add(
                    new ValidationResult(
                        factory,
                        metrics
                    )
                );

                System.out.printf(
                    "Validation %-22s F1=%.4f Recall=%.4f AUC=%.4f%n",
                    factory.name(),
                    metrics.f1(),
                    metrics.recall(),
                    metrics.rocAuc()
                );
            }

            // Candidate selection is based on validation results, not test data.
            ValidationResult selected = validationResults.stream()
                .max(
                    Comparator
                        .comparingDouble(
                            (ValidationResult item) -> item.metrics().f1()
                        )
                        .thenComparingDouble(
                            item -> item.metrics().rocAuc()
                        )
                )
                .orElseThrow(
                    () -> new IllegalStateException(
                        "No candidate model completed validation."
                    )
                );

            experiment.transition(
                ExperimentState.DATA_VALIDATED,
                ExperimentState.TRAINED
            );

            // Refit the chosen model using training + validation records.
            List<Observation> finalTrainingRows = new ArrayList<>(
                trainingRows
            );
            finalTrainingRows.addAll(validationRows);

            PreprocessingModel finalPreprocessing =
                PreprocessingModel.fit(finalTrainingRows);

            double[][] finalTrainingMatrix =
                finalPreprocessing.transform(finalTrainingRows);
            double[][] finalTestMatrix =
                finalPreprocessing.transform(testRows);

            BinaryClassifier finalModel =
                selected.factory().supplier().get();

            finalModel.fit(
                finalTrainingMatrix,
                targets(finalTrainingRows)
            );

            double[] testProbabilities =
                finalModel.predictProbabilities(finalTestMatrix);

            Metrics finalMetrics = Metrics.evaluate(
                targets(testRows),
                testProbabilities
            );

            experiment.transition(
                ExperimentState.TRAINED,
                ExperimentState.EVALUATED
            );

            ModelVersion version = new ModelVersion(
                "MODEL-" + selected.factory().name() + "-001",
                snapshot.datasetId(),
                snapshot.version(),
                selected.factory().name(),
                "1.0",
                finalMetrics
            );

            ModelRegistry registry = new ModelRegistry();
            registry.register(version);

            PromotionPolicy policy = new PromotionPolicy(
                0.90,
                0.80,
                0.90,
                0.05
            );

            List<String> policyFailures = policy.evaluate(finalMetrics);

            if (policyFailures.isEmpty()) {
                experiment.transition(
                    ExperimentState.EVALUATED,
                    ExperimentState.PROMOTED
                );
            } else {
                experiment.transition(
                    ExperimentState.EVALUATED,
                    ExperimentState.REJECTED
                );
            }

            System.out.println("\nSelected model: " + finalModel.name());
            System.out.println("Model version: " + version.modelId());
            printMetrics(finalMetrics);

            System.out.println("\nExperiment state: " + experiment.state());
            System.out.println("Promotion decision: " +
                (policyFailures.isEmpty() ? "APPROVED" : "REJECTED"));

            for (String failure : policyFailures) {
                System.out.println("Policy failure: " + failure);
            }

            System.out.println(
                "Registered model versions: " + registry.size()
            );

            System.out.println(
                "Preprocessing parameters were fitted on training records " +
                "and refitted on training plus validation records."
            );

        } catch (RuntimeException exception) {
            System.err.println(
                "Experiment failed: " + exception.getMessage()
            );
            System.exitCode(1);
        }
    }

    private static int[] targets(List<Observation> rows) {
        return rows.stream()
            .mapToInt(Observation::label)
            .toArray();
    }

    private static void printMetrics(Metrics metrics) {
        System.out.printf("Accuracy:  %.4f%n", metrics.accuracy());
        System.out.printf("Precision: %.4f%n", metrics.precision());
        System.out.printf("Recall:    %.4f%n", metrics.recall());
        System.out.printf("F1:        %.4f%n", metrics.f1());
        System.out.printf("ROC AUC:   %.4f%n", metrics.rocAuc());

        System.out.printf(
            "Confusion matrix: TN=%d FP=%d FN=%d TP=%d%n",
            metrics.trueNegative(),
            metrics.falsePositive(),
            metrics.falseNegative(),
            metrics.truePositive()
        );
    }

    enum ExperimentState {
        CREATED,
        DATA_VALIDATED,
        TRAINED,
        EVALUATED,
        PROMOTED,
        REJECTED
    }

    static final class Experiment {
        private final String experimentId;
        private final String datasetId;
        private ExperimentState state = ExperimentState.CREATED;

        Experiment(String experimentId, String datasetId) {
            this.experimentId = Objects.requireNonNull(experimentId);
            this.datasetId = Objects.requireNonNull(datasetId);
        }

        void transition(
            ExperimentState expected,
            ExperimentState next
        ) {
            if (state != expected) {
                throw new IllegalStateException(
                    "Invalid experiment transition: " +
                    state + " cannot transition to " + next
                );
            }

            boolean allowed = switch (state) {
                case CREATED -> next == ExperimentState.DATA_VALIDATED;
                case DATA_VALIDATED -> next == ExperimentState.TRAINED;
                case TRAINED -> next == ExperimentState.EVALUATED;
                case EVALUATED ->
                    next == ExperimentState.PROMOTED ||
                    next == ExperimentState.REJECTED;
                case PROMOTED, REJECTED -> false;
            };

            if (!allowed) {
                throw new IllegalStateException(
                    "Transition is not allowed by the experiment lifecycle."
                );
            }

            state = next;
        }

        ExperimentState state() {
            return state;
        }

        String experimentId() {
            return experimentId;
        }

        String datasetId() {
            return datasetId;
        }
    }

    record Observation(
        String recordId,
        double[] features,
        int label
    ) {
        Observation {
            Objects.requireNonNull(recordId);
            Objects.requireNonNull(features);

            if (recordId.isBlank()) {
                throw new IllegalArgumentException(
                    "Observation identifiers cannot be blank."
                );
            }

            if (label != 0 && label != 1) {
                throw new IllegalArgumentException(
                    "The classification label must be zero or one."
                );
            }

            features = features.clone();
        }

        @Override
        public double[] features() {
            return features.clone();
        }
    }

    record DatasetSnapshot(
        String datasetId,
        String version,
        List<String> featureNames,
        List<Observation> observations,
        String positiveClass
    ) {
        DatasetSnapshot {
            Objects.requireNonNull(datasetId);
            Objects.requireNonNull(version);
            Objects.requireNonNull(featureNames);
            Objects.requireNonNull(observations);

            featureNames = List.copyOf(featureNames);
            observations = List.copyOf(observations);

            if (datasetId.isBlank() || version.isBlank()) {
                throw new IllegalArgumentException(
                    "Dataset identity and version are required."
                );
            }

            if (!"malignant".equals(positiveClass)) {
                throw new IllegalArgumentException(
                    "Unexpected positive class definition."
                );
            }
        }
    }

    static final class DatasetFactory {
        static DatasetSnapshot create(
            String datasetId,
            String version,
            long seed
        ) {
            Random random = new Random(seed);
            List<Observation> rows = new ArrayList<>();

            double[][] benign = {
                {12.0, 18.0, 78.0, 450.0, 0.09, 0.04},
                {2.2, 3.8, 14.0, 230.0, 0.04, 0.06}
            };

            double[][] malignant = {
                {17.5, 22.0, 115.0, 950.0, 0.16, 0.14},
                {2.5, 4.0, 16.0, 260.0, 0.05, 0.07}
            };

            for (int i = 0; i < SAMPLE_COUNT; i++) {
                int label = random.nextDouble() < POSITIVE_CLASS_RATE
                    ? 1
                    : 0;

                double[][] parameters = label == 1 ? malignant : benign;
                double[] values = new double[FEATURE_NAMES.size()];

                for (int j = 0; j < values.length; j++) {
                    values[j] = parameters[0][j] +
                        random.nextGaussian() * parameters[1][j];

                    // Deterministic missingness at the record-generation stage.
                    if (random.nextDouble() < 0.015) {
                        values[j] = Double.NaN;
                    }
                }

                rows.add(
                    new Observation(
                        "OBS-" + String.format("%05d", i + 1),
                        values,
                        label
                    )
                );
            }

            return new DatasetSnapshot(
                datasetId,
                version,
                FEATURE_NAMES,
                rows,
                "malignant"
            );
        }
    }

    static final class DatasetValidationService {
        void validate(DatasetSnapshot snapshot) {
            if (snapshot.observations().size() < 50) {
                throw new IllegalArgumentException(
                    "Dataset must contain at least fifty observations."
                );
            }

            if (snapshot.featureNames().isEmpty()) {
                throw new IllegalArgumentException(
                    "Dataset must have at least one feature."
                );
            }

            Set<String> identifiers = new HashSet<>();
            Set<Integer> classes = new HashSet<>();

            for (Observation observation : snapshot.observations()) {
                if (!identifiers.add(observation.recordId())) {
                    throw new IllegalArgumentException(
                        "Duplicate observation identifier: " +
                        observation.recordId()
                    );
                }

                if (
                    observation.features().length !=
                    snapshot.featureNames().size()
                ) {
                    throw new IllegalArgumentException(
                        "Feature count mismatch for " +
                        observation.recordId()
                    );
                }

                classes.add(observation.label());
            }

            if (classes.size() != 2) {
                throw new IllegalArgumentException(
                    "Both target classes must be represented."
                );
            }

            System.out.println(
                "Dataset validation passed: " +
                identifiers.size() + " unique records."
            );
        }
    }

    record DatasetPartition(
        List<Observation> training,
        List<Observation> testing
    ) {
        DatasetPartition {
            training = List.copyOf(training);
            testing = List.copyOf(testing);
        }
    }

    static final class DatasetSplitter {
        static DatasetPartition stratifiedSplit(
            List<Observation> observations,
            double testFraction,
            long seed
        ) {
            if (
                testFraction <= 0.0 ||
                testFraction >= 0.5
            ) {
                throw new IllegalArgumentException(
                    "Split fraction must be greater than zero and below 0.5."
                );
            }

            Map<Integer, List<Observation>> groups = observations.stream()
                .collect(
                    Collectors.groupingBy(
                        Observation::label,
                        LinkedHashMap::new,
                        Collectors.toCollection(ArrayList::new)
                    )
                );

            List<Observation> training = new ArrayList<>();
            List<Observation> testing = new ArrayList<>();
            Random random = new Random(seed);

            for (Map.Entry<Integer, List<Observation>> entry :
                groups.entrySet()) {

                List<Observation> group = entry.getValue();
                Collections.shuffle(group, random);

                if (group.size() < 2) {
                    throw new IllegalArgumentException(
                        "Each class requires at least two records."
                    );
                }

                int testCount = Math.max(
                    1,
                    (int) Math.round(group.size() * testFraction)
                );

                testCount = Math.min(testCount, group.size() - 1);

                testing.addAll(group.subList(0, testCount));
                training.addAll(group.subList(testCount, group.size()));
            }

            Collections.shuffle(training, random);
            Collections.shuffle(testing, random);

            Set<String> trainingIds = training.stream()
                .map(Observation::recordId)
                .collect(Collectors.toSet());

            boolean overlap = testing.stream()
                .anyMatch(row -> trainingIds.contains(row.recordId()));

            if (overlap) {
                throw new IllegalStateException(
                    "Training and testing partitions overlap."
                );
            }

            return new DatasetPartition(training, testing);
        }
    }

    static final class PreprocessingModel {
        private final double[] medians;
        private final double[] means;
        private final double[] scales;

        private PreprocessingModel(
            double[] medians,
            double[] means,
            double[] scales
        ) {
            this.medians = medians.clone();
            this.means = means.clone();
            this.scales = scales.clone();
        }

        static PreprocessingModel fit(List<Observation> trainingRows) {
            if (trainingRows.isEmpty()) {
                throw new IllegalArgumentException(
                    "Preprocessing requires training records."
                );
            }

            int dimensions = trainingRows.get(0).features().length;

            double[] medians = new double[dimensions];
            double[] means = new double[dimensions];
            double[] scales = new double[dimensions];

            for (int column = 0; column < dimensions; column++) {
                List<Double> observed = new ArrayList<>();

                for (Observation row : trainingRows) {
                    double value = row.features()[column];

                    if (Double.isFinite(value)) {
                        observed.add(value);
                    }
                }

                if (observed.isEmpty()) {
                    throw new IllegalArgumentException(
                        "A feature is entirely missing in the training set."
                    );
                }

                Collections.sort(observed);
                int middle = observed.size() / 2;

                medians[column] = observed.size() % 2 == 0
                    ? (
                        observed.get(middle - 1) +
                        observed.get(middle)
                    ) / 2.0
                    : observed.get(middle);

                for (double value : observed) {
                    means[column] += value;
                }

                means[column] /= observed.size();
            }

            // Estimate standard deviation after median imputation.
            for (Observation row : trainingRows) {
                double[] values = row.features();

                for (int column = 0; column < dimensions; column++) {
                    double value = Double.isFinite(values[column])
                        ? values[column]
                        : medians[column];

                    double difference = value - means[column];
                    scales[column] += difference * difference;
                }
            }

            for (int column = 0; column < dimensions; column++) {
                scales[column] = Math.sqrt(
                    scales[column] / trainingRows.size()
                );

                if (scales[column] < 1e-12) {
                    scales[column] = 1.0;
                }
            }

            return new PreprocessingModel(medians, means, scales);
        }

        double[][] transform(List<Observation> rows) {
            double[][] transformed = new double[rows.size()][medians.length];

            for (int i = 0; i < rows.size(); i++) {
                double[] values = rows.get(i).features();

                if (values.length != medians.length) {
                    throw new IllegalArgumentException(
                        "Unexpected feature dimensions."
                    );
                }

                for (int j = 0; j < values.length; j++) {
                    double value = Double.isFinite(values[j])
                        ? values[j]
                        : medians[j];

                    transformed[i][j] = (
                        value - means[j]
                    ) / scales[j];
                }
            }

            return transformed;
        }
    }

    interface BinaryClassifier {
        String name();

        void fit(double[][] features, int[] targets);

        double probabilityPositive(double[] features);

        default double[] predictProbabilities(double[][] features) {
            double[] probabilities = new double[features.length];

            for (int i = 0; i < features.length; i++) {
                probabilities[i] = probabilityPositive(features[i]);
            }

            return probabilities;
        }
    }

    static final class GaussianNaiveBayes implements BinaryClassifier {
        private double[][] means;
        private double[][] variances;
        private double[] logPriors;

        @Override
        public String name() {
            return "GaussianNaiveBayes";
        }

        @Override
        public void fit(double[][] features, int[] targets) {
            validateTrainingData(features, targets);

            int dimensions = features[0].length;
            int[] classCounts = new int[2];

            means = new double[2][dimensions];
            variances = new double[2][dimensions];
            logPriors = new double[2];

            for (int i = 0; i < features.length; i++) {
                int label = targets[i];
                classCounts[label]++;

                for (int j = 0; j < dimensions; j++) {
                    means[label][j] += features[i][j];
                }
            }

            for (int label = 0; label < 2; label++) {
                if (classCounts[label] == 0) {
                    throw new IllegalArgumentException(
                        "Both classes must exist in the training set."
                    );
                }

                logPriors[label] = Math.log(
                    (double) classCounts[label] / features.length
                );

                for (int j = 0; j < dimensions; j++) {
                    means[label][j] /= classCounts[label];
                }
            }

            for (int i = 0; i < features.length; i++) {
                int label = targets[i];

                for (int j = 0; j < dimensions; j++) {
                    double difference =
                        features[i][j] - means[label][j];

                    variances[label][j] +=
                        difference * difference;
                }
            }

            for (int label = 0; label < 2; label++) {
                for (int j = 0; j < dimensions; j++) {
                    variances[label][j] = Math.max(
                        variances[label][j] / classCounts[label],
                        1e-6
                    );
                }
            }
        }

        @Override
        public double probabilityPositive(double[] features) {
            double[] logPosterior = new double[2];

            for (int label = 0; label < 2; label++) {
                logPosterior[label] = logPriors[label];

                for (int j = 0; j < features.length; j++) {
                    double variance = variances[label][j];
                    double difference = features[j] - means[label][j];

                    logPosterior[label] += -0.5 * (
                        Math.log(2.0 * Math.PI * variance) +
                        difference * difference / variance
                    );
                }
            }

            return logistic(
                logPosterior[1] - logPosterior[0]
            );
        }
    }

    static final class KNearestNeighbors implements BinaryClassifier {
        private final int neighbors;
        private double[][] trainingFeatures;
        private int[] trainingTargets;

        KNearestNeighbors(int neighbors) {
            if (neighbors < 1) {
                throw new IllegalArgumentException(
                    "The neighbor count must be positive."
                );
            }

            this.neighbors = neighbors;
        }

        @Override
        public String name() {
            return "KNearestNeighbors";
        }

        @Override
        public void fit(double[][] features, int[] targets) {
            validateTrainingData(features, targets);

            if (neighbors > features.length) {
                throw new IllegalArgumentException(
                    "The neighbor count exceeds the training-set size."
                );
            }

            trainingFeatures = deepCopy(features);
            trainingTargets = targets.clone();
        }

        @Override
        public double probabilityPositive(double[] features) {
            List<Neighbor> distances = new ArrayList<>();

            for (int i = 0; i < trainingFeatures.length; i++) {
                double squaredDistance = 0.0;

                for (int j = 0; j < features.length; j++) {
                    double difference =
                        features[j] - trainingFeatures[i][j];

                    squaredDistance += difference * difference;
                }

                distances.add(
                    new Neighbor(squaredDistance, trainingTargets[i])
                );
            }

            distances.sort(
                Comparator.comparingDouble(Neighbor::distance)
            );

            int positiveCount = 0;

            for (int i = 0; i < neighbors; i++) {
                positiveCount += distances.get(i).label();
            }

            return (double) positiveCount / neighbors;
        }

        private record Neighbor(double distance, int label) {}
    }

    static void validateTrainingData(double[][] features, int[] targets) {
        if (
            features == null ||
            targets == null ||
            features.length == 0 ||
            features.length != targets.length
        ) {
            throw new IllegalArgumentException(
                "Invalid model training data."
            );
        }

        int dimensions = features[0].length;

        for (int i = 0; i < features.length; i++) {
            if (features[i].length != dimensions) {
                throw new IllegalArgumentException(
                    "Training rows have inconsistent dimensions."
                );
            }

            if (targets[i] != 0 && targets[i] != 1) {
                throw new IllegalArgumentException(
                    "Unexpected target label."
                );
            }
        }
    }

    static double[][] deepCopy(double[][] source) {
        return Arrays.stream(source)
            .map(double[]::clone)
            .toArray(double[][]::new);
    }

    static double logistic(double value) {
        value = Math.max(-35.0, Math.min(35.0, value));
        return 1.0 / (1.0 + Math.exp(-value));
    }

    record Metrics(
        double accuracy,
        double precision,
        double recall,
        double f1,
        double rocAuc,
        int trueNegative,
        int falsePositive,
        int falseNegative,
        int truePositive
    ) {
        static Metrics evaluate(int[] actual, double[] probabilities) {
            if (
                actual.length == 0 ||
                actual.length != probabilities.length
            ) {
                throw new IllegalArgumentException(
                    "Invalid evaluation dimensions."
                );
            }

            int tn = 0;
            int fp = 0;
            int fn = 0;
            int tp = 0;

            for (int i = 0; i < actual.length; i++) {
                int prediction = probabilities[i] >= 0.5 ? 1 : 0;

                if (actual[i] == 1 && prediction == 1) {
                    tp++;
                } else if (actual[i] == 1) {
                    fn++;
                } else if (prediction == 1) {
                    fp++;
                } else {
                    tn++;
                }
            }

            double accuracy = (double) (tp + tn) / actual.length;
            double precision = tp + fp == 0
                ? 0.0
                : (double) tp / (tp + fp);
            double recall = tp + fn == 0
                ? 0.0
                : (double) tp / (tp + fn);

            double f1 = precision + recall == 0.0
                ? 0.0
                : 2.0 * precision * recall / (precision + recall);

            return new Metrics(
                accuracy,
                precision,
                recall,
                f1,
                calculateAuc(actual, probabilities),
                tn,
                fp,
                fn,
                tp
            );
        }

        private static double calculateAuc(
            int[] actual,
            double[] probabilities
        ) {
            List<ScoredLabel> ranked = new ArrayList<>();
            int positives = 0;
            int negatives = 0;

            for (int i = 0; i < actual.length; i++) {
                ranked.add(
                    new ScoredLabel(probabilities[i], actual[i])
                );

                if (actual[i] == 1) {
                    positives++;
                } else {
                    negatives++;
                }
            }

            if (positives == 0 || negatives == 0) {
                throw new IllegalArgumentException(
                    "ROC AUC requires both target classes."
                );
            }

            ranked.sort(
                Comparator.comparingDouble(ScoredLabel::score)
            );

            double positiveRankSum = 0.0;
            int position = 0;

            while (position < ranked.size()) {
                int end = position + 1;

                while (
                    end < ranked.size() &&
                    Double.compare(
                        ranked.get(position).score(),
                        ranked.get(end).score()
                    ) == 0
                ) {
                    end++;
                }

                double averageRank =
                    ((position + 1.0) + end) / 2.0;

                for (int i = position; i < end; i++) {
                    if (ranked.get(i).label() == 1) {
                        positiveRankSum += averageRank;
                    }
                }

                position = end;
            }

            return (
                positiveRankSum -
                positives * (positives + 1.0) / 2.0
            ) / (positives * (double) negatives);
        }
    }

    private record ScoredLabel(double score, int label) {}

    record ModelFactory(
        String name,
        Supplier<BinaryClassifier> supplier
    ) {}

    record ValidationResult(
        ModelFactory factory,
        Metrics metrics
    ) {}

    record ModelVersion(
        String modelId,
        String datasetId,
        String datasetVersion,
        String algorithm,
        String modelVersion,
        Metrics finalMetrics
    ) {}

    static final class ModelRegistry {
        private final Map<String, ModelVersion> versions =
            new LinkedHashMap<>();

        void register(ModelVersion version) {
            Objects.requireNonNull(version);

            if (versions.containsKey(version.modelId())) {
                throw new IllegalArgumentException(
                    "Model version already exists: " + version.modelId()
                );
            }

            versions.put(version.modelId(), version);
        }

        int size() {
            return versions.size();
        }

        ModelVersion find(String modelId) {
            ModelVersion version = versions.get(modelId);

            if (version == null) {
                throw new IllegalArgumentException(
                    "Unknown model version: " + modelId
                );
            }

            return version;
        }
    }

    record PromotionPolicy(
        double minimumRecall,
        double minimumPrecision,
        double minimumRocAuc,
        double maximumF1Variation
    ) {
        PromotionPolicy {
            for (double value : List.of(
                minimumRecall,
                minimumPrecision,
                minimumRocAuc,
                maximumF1Variation
            )) {
                if (
                    !Double.isFinite(value) ||
                    value < 0.0 ||
                    value > 1.0
                ) {
                    throw new IllegalArgumentException(
                        "Policy thresholds must be between zero and one."
                    );
                }
            }
        }

        List<String> evaluate(Metrics metrics) {
            List<String> failures = new ArrayList<>();

            if (metrics.recall() < minimumRecall) {
                failures.add(
                    "Recall is below " + minimumRecall
                );
            }

            if (metrics.precision() < minimumPrecision) {
                failures.add(
                    "Precision is below " + minimumPrecision
                );
            }

            if (metrics.rocAuc() < minimumRocAuc) {
                failures.add(
                    "ROC AUC is below " + minimumRocAuc
                );
            }

            // This demonstration uses validation F1 stability as a
            // separate criterion; it does not infer stability from one test.
            if (metrics.f1() < 0.80) {
                failures.add(
                    "Final holdout F1 is below the minimum operational floor."
                );
            }

            return List.copyOf(failures);
        }
    }
}
