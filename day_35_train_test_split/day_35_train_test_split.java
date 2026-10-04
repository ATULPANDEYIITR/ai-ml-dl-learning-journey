import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Predicate;

public class TrainTestSplitEnterpriseDemo {

    enum DatasetPartition {
        TRAIN,
        VALIDATION,
        TEST
    }

    enum ModelStage {
        CREATED,
        TRAINED,
        VALIDATED,
        FINALIZED
    }

    record Observation(
            int id,
            double linesChanged,
            double filesChanged,
            double reviewComments,
            double previousFailures,
            double experience,
            int target,
            String customerGroup,
            LocalDate date
    ) {}

    record Split(
            List<Observation> train,
            List<Observation> validation,
            List<Observation> test
    ) {}

    record Metrics(
            double accuracy,
            double precision,
            double recall,
            double f1
    ) {}

    interface SplitPolicy {
        Split create(List<Observation> observations);
    }

    static final class StratifiedRandomPolicy implements SplitPolicy {
        private final long seed;

        StratifiedRandomPolicy(long seed) {
            this.seed = seed;
        }

        @Override
        public Split create(List<Observation> observations) {
            Map<Integer, List<Observation>> classes = new HashMap<>();

            for (Observation observation : observations) {
                classes
                        .computeIfAbsent(observation.target(), ignored -> new ArrayList<>())
                        .add(observation);
            }

            Random random = new Random(seed);
            List<Observation> train = new ArrayList<>();
            List<Observation> validation = new ArrayList<>();
            List<Observation> test = new ArrayList<>();

            for (List<Observation> classRows : classes.values()) {
                Collections.shuffle(classRows, random);

                int trainEnd = classRows.size() * 70 / 100;
                int validationEnd = classRows.size() * 85 / 100;

                train.addAll(classRows.subList(0, trainEnd));
                validation.addAll(classRows.subList(trainEnd, validationEnd));
                test.addAll(classRows.subList(validationEnd, classRows.size()));
            }

            Collections.shuffle(train, random);
            Collections.shuffle(validation, random);
            Collections.shuffle(test, random);

            return new Split(
                    List.copyOf(train),
                    List.copyOf(validation),
                    List.copyOf(test)
            );
        }
    }

    static final class ChronologicalPolicy implements SplitPolicy {
        @Override
        public Split create(List<Observation> observations) {
            List<Observation> ordered = new ArrayList<>(observations);
            ordered.sort((a, b) -> a.date().compareTo(b.date()));

            int trainEnd = ordered.size() * 70 / 100;
            int validationEnd = ordered.size() * 85 / 100;

            return new Split(
                    List.copyOf(ordered.subList(0, trainEnd)),
                    List.copyOf(ordered.subList(trainEnd, validationEnd)),
                    List.copyOf(ordered.subList(validationEnd, ordered.size()))
            );
        }
    }

    static final class GroupPolicy implements SplitPolicy {
        private final long seed;

        GroupPolicy(long seed) {
            this.seed = seed;
        }

        @Override
        public Split create(List<Observation> observations) {
            Map<String, List<Observation>> groups = new HashMap<>();

            for (Observation observation : observations) {
                groups
                        .computeIfAbsent(observation.customerGroup(), ignored -> new ArrayList<>())
                        .add(observation);
            }

            List<String> names = new ArrayList<>(groups.keySet());
            Collections.shuffle(names, new Random(seed));

            int trainTarget = observations.size() * 70 / 100;
            int validationTarget = observations.size() * 15 / 100;

            List<Observation> train = new ArrayList<>();
            List<Observation> validation = new ArrayList<>();
            List<Observation> test = new ArrayList<>();

            for (String name : names) {
                List<Observation> group = groups.get(name);

                if (train.size() + group.size() <= trainTarget) {
                    train.addAll(group);
                } else if (validation.size() + group.size() <= validationTarget) {
                    validation.addAll(group);
                } else {
                    test.addAll(group);
                }
            }

            return new Split(
                    List.copyOf(train),
                    List.copyOf(validation),
                    List.copyOf(test)
            );
        }
    }

    static final class StandardScaler {
        private double[] means;
        private double[] standardDeviations;

        void fit(List<double[]> rows) {
            if (rows.isEmpty()) {
                throw new IllegalArgumentException("Cannot fit scaler on empty data.");
            }

            int width = rows.get(0).length;
            means = new double[width];
            standardDeviations = new double[width];

            for (double[] row : rows) {
                if (row.length != width) {
                    throw new IllegalArgumentException("Inconsistent feature width.");
                }

                for (int i = 0; i < width; i++) {
                    means[i] += row[i];
                }
            }

            for (int i = 0; i < width; i++) {
                means[i] /= rows.size();
            }

            for (double[] row : rows) {
                for (int i = 0; i < width; i++) {
                    double difference = row[i] - means[i];
                    standardDeviations[i] += difference * difference;
                }
            }

            for (int i = 0; i < width; i++) {
                standardDeviations[i] =
                        Math.sqrt(standardDeviations[i] / rows.size());

                if (standardDeviations[i] == 0) {
                    standardDeviations[i] = 1;
                }
            }
        }

        List<double[]> transform(List<double[]> rows) {
            if (means == null) {
                throw new IllegalStateException("Scaler has not been fitted.");
            }

            List<double[]> result = new ArrayList<>();

            for (double[] row : rows) {
                if (row.length != means.length) {
                    throw new IllegalArgumentException("Feature width mismatch.");
                }

                double[] normalized = new double[row.length];

                for (int i = 0; i < row.length; i++) {
                    normalized[i] =
                            (row[i] - means[i]) / standardDeviations[i];
                }

                result.add(normalized);
            }

            return result;
        }

        List<double[]> fitTransform(List<double[]> rows) {
            fit(rows);
            return transform(rows);
        }
    }

    static final class CentroidModel {
        private final Map<Integer, double[]> centroids = new HashMap<>();
        private ModelStage stage = ModelStage.CREATED;

        void fit(List<double[]> features, List<Integer> targets) {
            if (features.isEmpty() || features.size() != targets.size()) {
                throw new IllegalArgumentException("Training data is invalid.");
            }

            Map<Integer, List<double[]>> groups = new HashMap<>();

            for (int i = 0; i < features.size(); i++) {
                groups
                        .computeIfAbsent(targets.get(i), ignored -> new ArrayList<>())
                        .add(features.get(i));
            }

            centroids.clear();

            for (Map.Entry<Integer, List<double[]>> entry : groups.entrySet()) {
                List<double[]> rows = entry.getValue();
                double[] centroid = new double[rows.get(0).length];

                for (double[] row : rows) {
                    for (int i = 0; i < row.length; i++) {
                        centroid[i] += row[i];
                    }
                }

                for (int i = 0; i < centroid.length; i++) {
                    centroid[i] /= rows.size();
                }

                centroids.put(entry.getKey(), centroid);
            }

            stage = ModelStage.TRAINED;
        }

        List<Integer> predict(List<double[]> features) {
            if (stage == ModelStage.CREATED) {
                throw new IllegalStateException("Prediction before training is forbidden.");
            }

            List<Integer> predictions = new ArrayList<>();

            for (double[] row : features) {
                int selectedLabel = -1;
                double selectedDistance = Double.POSITIVE_INFINITY;

                for (Map.Entry<Integer, double[]> entry : centroids.entrySet()) {
                    double distance = 0;

                    for (int i = 0; i < row.length; i++) {
                        double difference = row[i] - entry.getValue()[i];
                        distance += difference * difference;
                    }

                    if (distance < selectedDistance) {
                        selectedDistance = distance;
                        selectedLabel = entry.getKey();
                    }
                }

                predictions.add(selectedLabel);
            }

            return predictions;
        }

        void markValidated() {
            if (stage != ModelStage.TRAINED) {
                throw new IllegalStateException("Only a trained model can be validated.");
            }
            stage = ModelStage.VALIDATED;
        }

        void finalizeModel() {
            if (stage != ModelStage.VALIDATED) {
                throw new IllegalStateException(
                        "Test evaluation requires completed validation."
                );
            }
            stage = ModelStage.FINALIZED;
        }
    }

    static final class ExperimentService {
        private final SplitPolicy policy;

        ExperimentService(SplitPolicy policy) {
            this.policy = policy;
        }

        Metrics execute(List<Observation> observations) {
            Split split = policy.create(observations);

            if (split.train().isEmpty()
                    || split.validation().isEmpty()
                    || split.test().isEmpty()) {
                throw new IllegalStateException("Every partition must contain observations.");
            }

            StandardScaler scaler = new StandardScaler();

            List<double[]> trainFeatures =
                    scaler.fitTransform(features(split.train()));

            List<double[]> validationFeatures =
                    scaler.transform(features(split.validation()));

            CentroidModel model = new CentroidModel();

            model.fit(trainFeatures, targets(split.train()));

            Metrics validationMetrics = evaluate(
                    targets(split.validation()),
                    model.predict(validationFeatures)
            );

            model.markValidated();

            // The test set is not used to tune the model. It is evaluated only
            // after the validation decision has been completed.
            List<double[]> testFeatures =
                    scaler.transform(features(split.test()));

            Metrics testMetrics = evaluate(
                    targets(split.test()),
                    model.predict(testFeatures)
            );

            model.finalizeModel();

            System.out.println("Validation F1: " +
                    String.format("%.4f", validationMetrics.f1()));

            return testMetrics;
        }
    }

    static List<double[]> features(List<Observation> rows) {
        return rows.stream()
                .map(row -> new double[]{
                        row.linesChanged(),
                        row.filesChanged(),
                        row.reviewComments(),
                        row.previousFailures(),
                        row.experience()
                })
                .toList();
    }

    static List<Integer> targets(List<Observation> rows) {
        return rows.stream()
                .map(Observation::target)
                .toList();
    }

    static Metrics evaluate(List<Integer> actual, List<Integer> predicted) {
        if (actual.isEmpty() || actual.size() != predicted.size()) {
            throw new IllegalArgumentException("Evaluation data is invalid.");
        }

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        for (int i = 0; i < actual.size(); i++) {
            if (actual.get(i) == 1 && predicted.get(i) == 1) tp++;
            else if (actual.get(i) == 0 && predicted.get(i) == 0) tn++;
            else if (actual.get(i) == 0 && predicted.get(i) == 1) fp++;
            else if (actual.get(i) == 1 && predicted.get(i) == 0) fn++;
        }

        double precision = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        double recall = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        double f1 = precision + recall == 0
                ? 0
                : 2 * precision * recall / (precision + recall);

        return new Metrics(
                (double) (tp + tn) / actual.size(),
                precision,
                recall,
                f1
        );
    }

    static List<Observation> generateDataset() {
        Random random = new Random(2026);
        List<Observation> observations = new ArrayList<>();

        for (int id = 0; id < 240; id++) {
            double lines = 5 + random.nextInt(300);
            double files = 1 + random.nextInt(10);
            double comments = random.nextInt(8);
            double failures = random.nextInt(5);
            double experience = 1 + random.nextInt(10);

            double score =
                    0.025 * lines +
                    0.55 * files +
                    0.75 * comments +
                    0.9 * failures -
                    0.65 * experience;

            observations.add(new Observation(
                    id,
                    lines,
                    files,
                    comments,
                    failures,
                    experience,
                    score > 3 ? 1 : 0,
                    "group-" + (id % 12),
                    LocalDate.of(2026, 1, 1).plusDays(id)
            ));
        }

        return observations;
    }

    static double positiveRate(List<Observation> rows) {
        return rows.stream()
                .filter(row -> row.target() == 1)
                .count() / (double) rows.size();
    }

    static void demonstratePredicateRisk(List<Observation> rows) {
        Predicate<Observation> futureRecord =
                row -> row.date().isAfter(LocalDate.of(2026, 6, 30));

        long futureCount = rows.stream()
                .filter(futureRecord)
                .count();

        System.out.println("Future observations in dataset: " + futureCount);
        System.out.println(
                "A production time-series experiment must not train on future "
                        + "information when predicting earlier deployment periods."
        );
    }

    public static void main(String[] args) {
        System.out.println("ENTERPRISE TRAIN / VALIDATION / TEST MODEL");
        System.out.println("==========================================");

        List<Observation> dataset = generateDataset();

        SplitPolicy stratifiedPolicy =
                new StratifiedRandomPolicy(2026);

        Split stratified = stratifiedPolicy.create(dataset);

        System.out.println("Dataset size: " + dataset.size());
        System.out.println("Train size: " + stratified.train().size());
        System.out.println("Validation size: " + stratified.validation().size());
        System.out.println("Test size: " + stratified.test().size());

        System.out.println(
                "Train positive rate: " +
                        String.format("%.3f", positiveRate(stratified.train()))
        );

        System.out.println(
                "Test positive rate: " +
                        String.format("%.3f", positiveRate(stratified.test()))
        );

        ExperimentService service =
                new ExperimentService(stratifiedPolicy);

        Metrics finalMetrics = service.execute(dataset);

        System.out.println("\nFinal test metrics");
        System.out.println("Accuracy: " + String.format("%.4f", finalMetrics.accuracy()));
        System.out.println("Precision: " + String.format("%.4f", finalMetrics.precision()));
        System.out.println("Recall: " + String.format("%.4f", finalMetrics.recall()));
        System.out.println("F1: " + String.format("%.4f", finalMetrics.f1()));

        SplitPolicy chronologicalPolicy =
                new ChronologicalPolicy();

        Split chronological = chronologicalPolicy.create(dataset);

        System.out.println("\nChronological boundary");
        System.out.println("Training ends: " + chronological.train().getLast().date());
        System.out.println("Validation starts: " + chronological.validation().getFirst().date());
        System.out.println("Test starts: " + chronological.test().getFirst().date());

        SplitPolicy groupPolicy =
                new GroupPolicy(2026);

        Split grouped = groupPolicy.create(dataset);

        Set<String> trainGroups = new HashSet<>(
                grouped.train().stream()
                        .map(Observation::customerGroup)
                        .toList()
        );

        Set<String> testGroups = new HashSet<>(
                grouped.test().stream()
                        .map(Observation::customerGroup)
                        .toList()
        );

        Set<String> overlap = new HashSet<>(trainGroups);
        overlap.retainAll(testGroups);

        System.out.println("\nGroup-aware boundary");
        System.out.println("Train/test group overlap: " + overlap.size());

        demonstratePredicateRisk(dataset);

        System.out.println("\nEnterprise rules");
        System.out.println("Training owns parameter fitting.");
        System.out.println("Validation owns model-selection evidence.");
        System.out.println("Testing owns final generalization evidence.");
        System.out.println("The split policy must match temporal and entity dependencies.");
        System.out.println("State transitions prevent a final test decision from occurring before validation.");
    }
}
