import java.util.*;
import java.util.stream.Collectors;

/**
 * Imbalanced Learning: Java 17
 *
 * Enterprise scenario: Financial transaction fraud detection.
 *
 * Features:
 * - Immutable transaction records
 * - Class imbalance analysis
 * - Stratified splitting
 * - Random oversampling and undersampling
 * - SMOTE
 * - Class weighting
 * - Weighted nearest-centroid classification
 * - Model evaluation
 * - Experiment governance and validation
 *
 * Compile: javac ImbalancedLearning.java
 * Run:     java ImbalancedLearning
 */

public class ImbalancedLearning {

    enum SamplingStrategy {
        BASELINE,
        OVERSAMPLING,
        UNDERSAMPLING,
        SMOTE,
        CLASS_WEIGHTING
    }

    enum ExperimentStatus {
        CREATED,
        VALIDATED,
        TRAINED,
        EVALUATED,
        FAILED
    }

    record Transaction(double[] features, int label) {
        Transaction {
            Objects.requireNonNull(features);

            if (features.length == 0) {
                throw new IllegalArgumentException(
                    "Features cannot be empty."
                );
            }

            if (label != 0 && label != 1) {
                throw new IllegalArgumentException(
                    "Binary labels required."
                );
            }

            features = features.clone();

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Non-finite feature detected."
                    );
                }
            }
        }

        @Override
        public double[] features() {
            return features.clone();
        }
    }

    record Distribution(
        int majority,
        int minority,
        int negative,
        int positive,
        double ratio
    ) {}

    record Metrics(
        int tp,
        int tn,
        int fp,
        int fn
    ) {
        double precision() {
            return tp + fp == 0
                ? 0.0 : (double) tp / (tp + fp);
        }

        double recall() {
            return tp + fn == 0
                ? 0.0 : (double) tp / (tp + fn);
        }

        double specificity() {
            return tn + fp == 0
                ? 0.0 : (double) tn / (tn + fp);
        }

        double f1() {
            double p = precision();
            double r = recall();

            return p + r == 0
                ? 0.0 : 2 * p * r / (p + r);
        }

        double balancedAccuracy() {
            return (recall() + specificity()) / 2;
        }
    }

    record DatasetSplit(
        List<Transaction> training,
        List<Transaction> testing
    ) {}

    record ExperimentResult(
        SamplingStrategy strategy,
        ExperimentStatus status,
        Distribution distribution,
        Metrics metrics
    ) {}

    static class DatasetValidator {

        static void validate(List<Transaction> data) {
            if (data == null || data.isEmpty()) {
                throw new IllegalArgumentException(
                    "Dataset cannot be empty."
                );
            }

            int dimensions = data.get(0).features().length;
            int[] counts = new int[2];

            for (Transaction transaction : data) {
                if (transaction.features().length != dimensions) {
                    throw new IllegalArgumentException(
                        "Inconsistent feature dimensions."
                    );
                }

                counts[transaction.label()]++;
            }

            if (counts[0] == 0 || counts[1] == 0) {
                throw new IllegalArgumentException(
                    "Both classes must exist."
                );
            }
        }

        static Distribution analyze(List<Transaction> data) {
            DatasetValidator.validate(data);

            int negative = 0;
            int positive = 0;

            for (Transaction transaction : data) {
                if (transaction.label() == 0) negative++;
                else positive++;
            }

            return new Distribution(
                Math.max(negative, positive),
                Math.min(negative, positive),
                negative,
                positive,
                (double) Math.max(negative, positive)
                    / Math.min(negative, positive)
            );
        }
    }

    static class DataFactory {

        static List<Transaction> generate(
            int majorityCount,
            int minorityCount
        ) {
            if (majorityCount < 2 || minorityCount < 2) {
                throw new IllegalArgumentException(
                    "Both classes need at least two records."
                );
            }

            Random random = new Random(42);
            List<Transaction> result = new ArrayList<>();

            for (int i = 0; i < majorityCount; i++) {
                result.add(new Transaction(
                    new double[]{
                        random.nextGaussian() * 1.2,
                        random.nextGaussian() * 1.2
                    },
                    0
                ));
            }

            for (int i = 0; i < minorityCount; i++) {
                result.add(new Transaction(
                    new double[]{
                        2 + random.nextGaussian(),
                        2 + random.nextGaussian()
                    },
                    1
                ));
            }

            Collections.shuffle(result, random);

            return List.copyOf(result);
        }
    }

    static class SplitService {

        static DatasetSplit stratifiedSplit(
            List<Transaction> data,
            double testFraction
        ) {
            if (!(testFraction > 0 && testFraction < 1)) {
                throw new IllegalArgumentException(
                    "Invalid test fraction."
                );
            }

            DatasetValidator.validate(data);

            Map<Integer, List<Transaction>> groups =
                data.stream().collect(
                    Collectors.groupingBy(
                        Transaction::label,
                        HashMap::new,
                        Collectors.toCollection(ArrayList::new)
                    )
                );

            Random random = new Random(42);
            List<Transaction> training = new ArrayList<>();
            List<Transaction> testing = new ArrayList<>();

            for (int label : List.of(0, 1)) {
                List<Transaction> group = groups.get(label);

                if (group.size() < 2) {
                    throw new IllegalArgumentException(
                        "Each class requires two records."
                    );
                }

                Collections.shuffle(group, random);

                int testSize = Math.max(
                    1,
                    Math.min(
                        group.size() - 1,
                        (int) (group.size() * testFraction)
                    )
                );

                testing.addAll(group.subList(0, testSize));
                training.addAll(group.subList(testSize, group.size()));
            }

            Collections.shuffle(training, random);
            Collections.shuffle(testing, random);

            return new DatasetSplit(
                List.copyOf(training),
                List.copyOf(testing)
            );
        }
    }

    interface SamplingService {
        List<Transaction> apply(List<Transaction> data);
    }

    static class RandomOverSampler implements SamplingService {

        @Override
        public List<Transaction> apply(List<Transaction> data) {
            DatasetValidator.validate(data);

            Random random = new Random(42);

            List<Transaction> negative = new ArrayList<>();
            List<Transaction> positive = new ArrayList<>();

            for (Transaction transaction : data) {
                if (transaction.label() == 0) negative.add(transaction);
                else positive.add(transaction);
            }

            int target = Math.max(
                negative.size(), positive.size()
            );

            List<Transaction> result = new ArrayList<>(data);

            while (negative.size() < target) {
                negative.add(negative.get(
                    random.nextInt(negative.size())
                ));
                result.add(negative.get(negative.size() - 1));
            }

            while (positive.size() < target) {
                positive.add(positive.get(
                    random.nextInt(positive.size())
                ));
                result.add(positive.get(positive.size() - 1));
            }

            Collections.shuffle(result, random);
            return List.copyOf(result);
        }
    }

    static class RandomUnderSampler implements SamplingService {

        @Override
        public List<Transaction> apply(List<Transaction> data) {
            DatasetValidator.validate(data);

            Random random = new Random(42);

            List<Transaction> negative = data.stream()
                .filter(t -> t.label() == 0)
                .collect(Collectors.toCollection(ArrayList::new));

            List<Transaction> positive = data.stream()
                .filter(t -> t.label() == 1)
                .collect(Collectors.toCollection(ArrayList::new));

            int target = Math.min(
                negative.size(), positive.size()
            );

            Collections.shuffle(negative, random);
            Collections.shuffle(positive, random);

            List<Transaction> result = new ArrayList<>();

            result.addAll(negative.subList(0, target));
            result.addAll(positive.subList(0, target));

            Collections.shuffle(result, random);
            return List.copyOf(result);
        }
    }

    static class SMOTESampler implements SamplingService {

        private final int neighbors;

        SMOTESampler(int neighbors) {
            if (neighbors <= 0) {
                throw new IllegalArgumentException(
                    "Neighbor count must be positive."
                );
            }

            this.neighbors = neighbors;
        }

        private double distance(double[] a, double[] b) {
            double sum = 0;

            for (int i = 0; i < a.length; i++) {
                double difference = a[i] - b[i];
                sum += difference * difference;
            }

            return Math.sqrt(sum);
        }

        @Override
        public List<Transaction> apply(List<Transaction> data) {
            DatasetValidator.validate(data);

            List<Transaction> minority = data.stream()
                .filter(t -> t.label() == 1)
                .toList();

            long majorityCount = data.stream()
                .filter(t -> t.label() == 0)
                .count();

            if (minority.size() < 2) {
                throw new IllegalArgumentException(
                    "SMOTE requires at least two minority records."
                );
            }

            int k = Math.min(
                neighbors, minority.size() - 1
            );

            int required = (int) majorityCount - minority.size();

            Random random = new Random(42);
            List<Transaction> result = new ArrayList<>(data);

            for (int iteration = 0; iteration < required; iteration++) {
                Transaction source = minority.get(
                    random.nextInt(minority.size())
                );

                double[] sourceFeatures = source.features();

                List<Transaction> nearest = minority.stream()
                    .filter(candidate -> candidate != source)
                    .sorted(
                        Comparator.comparingDouble(
                            candidate -> distance(
                                sourceFeatures,
                                candidate.features()
                            )
                        )
                    )
                    .limit(k)
                    .toList();

                Transaction neighbor = nearest.get(
                    random.nextInt(nearest.size())
                );

                double[] neighborFeatures = neighbor.features();
                double factor = random.nextDouble();

                double[] synthetic = new double[sourceFeatures.length];

                for (int j = 0; j < synthetic.length; j++) {
                    synthetic[j] = sourceFeatures[j]
                        + factor * (
                            neighborFeatures[j] - sourceFeatures[j]
                        );
                }

                result.add(new Transaction(synthetic, 1));
            }

            Collections.shuffle(result, random);
            return List.copyOf(result);
        }
    }

    static class ClassWeightPolicy {

        static Map<Integer, Double> calculate(
            List<Transaction> data
        ) {
            DatasetValidator.validate(data);

            long[] counts = new long[2];

            for (Transaction transaction : data) {
                counts[transaction.label()]++;
            }

            double total = data.size();

            return Map.of(
                0, total / (2 * counts[0]),
                1, total / (2 * counts[1])
            );
        }
    }

    static class WeightedCentroidClassifier {

        private double[][] centroids;
        private Map<Integer, Double> weights;
        private boolean trained = false;

        void fit(
            List<Transaction> data,
            Map<Integer, Double> classWeights
        ) {
            DatasetValidator.validate(data);

            int dimensions = data.get(0).features().length;

            centroids = new double[2][dimensions];
            int[] counts = new int[2];

            weights = Map.copyOf(classWeights);

            for (double weight : weights.values()) {
                if (!Double.isFinite(weight) || weight <= 0) {
                    throw new IllegalArgumentException(
                        "Class weights must be positive."
                    );
                }
            }

            for (Transaction transaction : data) {
                int label = transaction.label();
                double[] features = transaction.features();

                counts[label]++;

                for (int j = 0; j < dimensions; j++) {
                    centroids[label][j] += features[j];
                }
            }

            for (int label = 0; label < 2; label++) {
                if (counts[label] == 0) {
                    throw new IllegalStateException(
                        "Missing training class."
                    );
                }

                for (int j = 0; j < dimensions; j++) {
                    centroids[label][j] /= counts[label];
                }
            }

            trained = true;
        }

        private double distance(double[] a, double[] b) {
            double sum = 0;

            for (int i = 0; i < a.length; i++) {
                sum += Math.pow(a[i] - b[i], 2);
            }

            return Math.sqrt(sum);
        }

        int predict(double[] features) {
            if (!trained) {
                throw new IllegalStateException(
                    "Model is not trained."
                );
            }

            if (features.length != centroids[0].length) {
                throw new IllegalArgumentException(
                    "Invalid prediction dimensions."
                );
            }

            double negativeScore =
                distance(features, centroids[0])
                / weights.get(0);

            double positiveScore =
                distance(features, centroids[1])
                / weights.get(1);

            return positiveScore < negativeScore ? 1 : 0;
        }
    }

    static class EvaluationService {

        static Metrics evaluate(
            List<Transaction> testing,
            WeightedCentroidClassifier model
        ) {
            int tp = 0;
            int tn = 0;
            int fp = 0;
            int fn = 0;

            for (Transaction transaction : testing) {
                int prediction = model.predict(
                    transaction.features()
                );

                if (transaction.label() == 1 && prediction == 1) tp++;
                else if (transaction.label() == 0 && prediction == 0) tn++;
                else if (transaction.label() == 0 && prediction == 1) fp++;
                else fn++;
            }

            return new Metrics(tp, tn, fp, fn);
        }
    }

    static class ExperimentService {

        ExperimentResult execute(
            SamplingStrategy strategy,
            List<Transaction> training,
            List<Transaction> testing,
            Map<Integer, Double> weights
        ) {
            ExperimentStatus status = ExperimentStatus.CREATED;

            try {
                DatasetValidator.validate(training);
                status = ExperimentStatus.VALIDATED;

                WeightedCentroidClassifier model =
                    new WeightedCentroidClassifier();

                model.fit(training, weights);
                status = ExperimentStatus.TRAINED;

                Metrics metrics = EvaluationService.evaluate(
                    testing, model
                );

                status = ExperimentStatus.EVALUATED;

                return new ExperimentResult(
                    strategy,
                    status,
                    DatasetValidator.analyze(training),
                    metrics
                );

            } catch (RuntimeException exception) {
                System.err.println(
                    "Experiment failed: " + strategy
                    + " | " + exception.getMessage()
                );

                return new ExperimentResult(
                    strategy,
                    ExperimentStatus.FAILED,
                    null,
                    null
                );
            }
        }
    }

    static void printResult(ExperimentResult result) {
        if (result.status() != ExperimentStatus.EVALUATED) {
            System.out.println(
                result.strategy() + " FAILED"
            );
            return;
        }

        Metrics metrics = result.metrics();

        System.out.printf(
            "%-23s Precision=%.3f Recall=%.3f F1=%.3f "
                + "BalancedAccuracy=%.3f%n",
            result.strategy(),
            metrics.precision(),
            metrics.recall(),
            metrics.f1(),
            metrics.balancedAccuracy()
        );
    }

    public static void main(String[] args) {
        try {
            List<Transaction> original =
                DataFactory.generate(240, 30);

            DatasetSplit split =
                SplitService.stratifiedSplit(original, 0.25);

            List<Transaction> training = split.training();
            List<Transaction> testing = split.testing();

            System.out.println("IMBALANCED LEARNING");
            System.out.println("=".repeat(75));

            System.out.println(
                "Original: "
                + DatasetValidator.analyze(original)
            );

            System.out.println(
                "Training: "
                + DatasetValidator.analyze(training)
            );

            System.out.println(
                "Testing: "
                + DatasetValidator.analyze(testing)
            );

            ExperimentService service =
                new ExperimentService();

            List<ExperimentResult> results = List.of(
                service.execute(
                    SamplingStrategy.BASELINE,
                    training,
                    testing,
                    Map.of(0, 1.0, 1, 1.0)
                ),

                service.execute(
                    SamplingStrategy.OVERSAMPLING,
                    new RandomOverSampler().apply(training),
                    testing,
                    Map.of(0, 1.0, 1, 1.0)
                ),

                service.execute(
                    SamplingStrategy.UNDERSAMPLING,
                    new RandomUnderSampler().apply(training),
                    testing,
                    Map.of(0, 1.0, 1, 1.0)
                ),

                service.execute(
                    SamplingStrategy.SMOTE,
                    new SMOTESampler(5).apply(training),
                    testing,
                    Map.of(0, 1.0, 1, 1.0)
                ),

                service.execute(
                    SamplingStrategy.CLASS_WEIGHTING,
                    training,
                    testing,
                    ClassWeightPolicy.calculate(training)
                )
            );

            System.out.println("\nEXPERIMENT RESULTS");

            results.forEach(ImbalancedLearning::printResult);

            System.out.println("\nVALIDATION");

            if (
                new SMOTESampler(5).apply(training).size()
                < training.size()
            ) {
                throw new IllegalStateException(
                    "SMOTE validation failed."
                );
            }

            if (
                new RandomUnderSampler().apply(training).size()
                > training.size()
            ) {
                throw new IllegalStateException(
                    "Undersampling validation failed."
                );
            }

            System.out.println("All validation tests passed.");

            System.out.println("\nENTERPRISE CONTROLS");
            System.out.println("- Version datasets and model configurations.");
            System.out.println("- Enforce training-only sampling.");
            System.out.println("- Preserve original validation distributions.");
            System.out.println("- Validate generated transaction characteristics.");
            System.out.println("- Apply access control to sensitive records.");
            System.out.println("- Audit experiment decisions and parameters.");
            System.out.println("- Monitor false negatives and false positives.");
            System.out.println("- Revalidate models after distribution changes.");

        } catch (Exception exception) {
            System.err.println(
                "Application failure: " + exception.getMessage()
            );

            System.exit(1);
        }
    }
}
