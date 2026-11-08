import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.stream.Collectors;

/*
 * ML Pipeline Enterprise Workflow
 *
 * Java 17 implementation featuring:
 * - Immutable domain records
 * - Schema validation
 * - Transformer interface
 * - Median imputation
 * - Standard scaling
 * - One-hot encoding
 * - Composable transformation stages
 * - Logistic regression
 * - Training and validation separation
 * - Pipeline state transitions
 * - Evaluation services
 * - Artifact export
 * - Inference
 */

public class MLPipelineEnterprise {

    enum PipelineStatus {
        CREATED,
        VALIDATED,
        FITTING,
        FITTED,
        PREDICTING,
        FAILED
    }

    enum FeatureType {
        NUMERIC,
        CATEGORICAL
    }

    record Customer(
        double age,
        double spending,
        double tenure,
        String contract,
        String region,
        int target
    ) {
        Customer {
            if (target != 0 && target != 1) {
                throw new IllegalArgumentException(
                    "Target must be binary"
                );
            }

            if (!Double.isNaN(age) && age <= 0) {
                throw new IllegalArgumentException("Invalid age");
            }

            if (!Double.isNaN(spending) && spending < 0) {
                throw new IllegalArgumentException(
                    "Invalid spending"
                );
            }

            if (tenure <= 0) {
                throw new IllegalArgumentException("Invalid tenure");
            }

            Objects.requireNonNull(contract);
            Objects.requireNonNull(region);
        }
    }

    record Prediction(
        double probability,
        int classification
    ) {}

    record ModelMetrics(
        double accuracy,
        double precision,
        double recall,
        double f1,
        int truePositive,
        int falsePositive,
        int trueNegative,
        int falseNegative
    ) {}

    record TrainingSplit(
        List<Customer> training,
        List<Customer> validation
    ) {
        TrainingSplit {
            training = List.copyOf(training);
            validation = List.copyOf(validation);
        }
    }

    record FeatureSchema(
        Map<String, FeatureType> columns
    ) {
        FeatureSchema {
            columns = Map.copyOf(columns);

            if (columns.isEmpty()) {
                throw new IllegalArgumentException(
                    "Schema cannot be empty"
                );
            }
        }

        void validate(Customer customer) {
            Objects.requireNonNull(customer);

            if (columns.get("contract") != FeatureType.CATEGORICAL ||
                columns.get("region") != FeatureType.CATEGORICAL) {
                throw new IllegalArgumentException(
                    "Categorical schema mismatch"
                );
            }
        }
    }

    static class PipelineException extends RuntimeException {
        PipelineException(String message) {
            super(message);
        }

        PipelineException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    interface Transformer<I, O> {
        void fit(List<I> data);

        List<O> transform(List<I> data);

        String name();
    }

    static final class ImputedRecord {
        final double age;
        final double spending;
        final double tenure;
        final String contract;
        final String region;
        final int target;

        ImputedRecord(
            double age,
            double spending,
            double tenure,
            String contract,
            String region,
            int target
        ) {
            this.age = age;
            this.spending = spending;
            this.tenure = tenure;
            this.contract = contract;
            this.region = region;
            this.target = target;
        }
    }

    static final class FeatureVector {
        private final Map<String, Double> values;
        private final int target;

        FeatureVector(Map<String, Double> values, int target) {
            this.values = Collections.unmodifiableMap(
                new LinkedHashMap<>(values)
            );

            this.target = target;
        }

        Map<String, Double> values() {
            return values;
        }

        int target() {
            return target;
        }
    }

    static double median(List<Double> input) {
        List<Double> values = input.stream()
            .filter(Double::isFinite)
            .sorted()
            .toList();

        if (values.isEmpty()) {
            throw new PipelineException(
                "Cannot calculate median of empty feature"
            );
        }

        int middle = values.size() / 2;

        if (values.size() % 2 == 0) {
            return (
                values.get(middle - 1) +
                values.get(middle)
            ) / 2.0;
        }

        return values.get(middle);
    }

    static final class MedianImputer
        implements Transformer<Customer, ImputedRecord> {

        private double ageMedian;
        private double spendingMedian;
        private boolean fitted;

        @Override
        public void fit(List<Customer> data) {
            if (data.isEmpty()) {
                throw new PipelineException(
                    "Imputer received empty data"
                );
            }

            ageMedian = median(
                data.stream()
                    .map(Customer::age)
                    .toList()
            );

            spendingMedian = median(
                data.stream()
                    .map(Customer::spending)
                    .toList()
            );

            fitted = true;
        }

        @Override
        public List<ImputedRecord> transform(
            List<Customer> data
        ) {
            if (!fitted) {
                throw new PipelineException(
                    "Imputer has not been fitted"
                );
            }

            return data.stream()
                .map(customer -> new ImputedRecord(
                    Double.isFinite(customer.age())
                        ? customer.age()
                        : ageMedian,
                    Double.isFinite(customer.spending())
                        ? customer.spending()
                        : spendingMedian,
                    customer.tenure(),
                    customer.contract(),
                    customer.region(),
                    customer.target()
                ))
                .toList();
        }

        @Override
        public String name() {
            return "MedianImputer";
        }
    }

    static final class StandardScaler
        implements Transformer<ImputedRecord, ImputedRecord> {

        private final Map<String, Double> means =
            new LinkedHashMap<>();

        private final Map<String, Double> deviations =
            new LinkedHashMap<>();

        private boolean fitted;

        @Override
        public void fit(List<ImputedRecord> data) {
            if (data.isEmpty()) {
                throw new PipelineException(
                    "Scaler received empty data"
                );
            }

            Map<String, Function<ImputedRecord, Double>> extractors =
                Map.of(
                    "age", record -> record.age,
                    "spending", record -> record.spending,
                    "tenure", record -> record.tenure
                );

            for (var entry : extractors.entrySet()) {
                List<Double> values = data.stream()
                    .map(entry.getValue())
                    .toList();

                double mean = values.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElseThrow();

                double variance = values.stream()
                    .mapToDouble(value ->
                        Math.pow(value - mean, 2)
                    )
                    .average()
                    .orElseThrow();

                means.put(entry.getKey(), mean);
                deviations.put(
                    entry.getKey(),
                    Math.sqrt(variance)
                );
            }

            fitted = true;
        }

        private double scale(String feature, double value) {
            double deviation = deviations.get(feature);

            return deviation == 0
                ? 0
                : (value - means.get(feature)) / deviation;
        }

        @Override
        public List<ImputedRecord> transform(
            List<ImputedRecord> data
        ) {
            if (!fitted) {
                throw new PipelineException(
                    "Scaler has not been fitted"
                );
            }

            return data.stream()
                .map(record -> new ImputedRecord(
                    scale("age", record.age),
                    scale("spending", record.spending),
                    scale("tenure", record.tenure),
                    record.contract,
                    record.region,
                    record.target
                ))
                .toList();
        }

        @Override
        public String name() {
            return "StandardScaler";
        }
    }

    static final class OneHotEncoder
        implements Transformer<ImputedRecord, FeatureVector> {

        private final Map<String, List<String>> categories =
            new LinkedHashMap<>();

        private boolean fitted;

        @Override
        public void fit(List<ImputedRecord> data) {
            if (data.isEmpty()) {
                throw new PipelineException(
                    "Encoder received empty data"
                );
            }

            categories.put(
                "contract",
                data.stream()
                    .map(record -> record.contract)
                    .collect(Collectors.toCollection(TreeSet::new))
                    .stream().toList()
            );

            categories.put(
                "region",
                data.stream()
                    .map(record -> record.region)
                    .collect(Collectors.toCollection(TreeSet::new))
                    .stream().toList()
            );

            fitted = true;
        }

        @Override
        public List<FeatureVector> transform(
            List<ImputedRecord> data
        ) {
            if (!fitted) {
                throw new PipelineException(
                    "Encoder has not been fitted"
                );
            }

            return data.stream().map(record -> {
                Map<String, Double> features =
                    new LinkedHashMap<>();

                features.put("age", record.age);
                features.put("spending", record.spending);
                features.put("tenure", record.tenure);

                encode(
                    features,
                    "contract",
                    record.contract
                );

                encode(
                    features,
                    "region",
                    record.region
                );

                return new FeatureVector(
                    features,
                    record.target
                );
            }).toList();
        }

        private void encode(
            Map<String, Double> features,
            String name,
            String value
        ) {
            for (String category : categories.get(name)) {
                features.put(
                    name + "__" + category,
                    value.equals(category) ? 1.0 : 0.0
                );
            }
        }

        @Override
        public String name() {
            return "OneHotEncoder";
        }
    }

    static final class LogisticRegression {

        private List<String> featureNames = List.of();
        private double[] weights;
        private final double learningRate = 0.08;
        private final double regularization = 0.01;
        private final int iterations = 2500;
        private boolean fitted;

        private static double sigmoid(double value) {
            value = Math.max(-500, Math.min(500, value));

            return 1.0 / (1.0 + Math.exp(-value));
        }

        private double[] extract(FeatureVector vector) {
            double[] values = new double[featureNames.size()];

            for (int i = 0; i < featureNames.size(); i++) {
                Double value = vector.values().get(featureNames.get(i));

                if (value == null || !Double.isFinite(value)) {
                    throw new PipelineException(
                        "Invalid model feature: " + featureNames.get(i)
                    );
                }

                values[i] = value;
            }

            return values;
        }

        private double probability(FeatureVector vector) {
            double score = weights[0];
            double[] values = extract(vector);

            for (int i = 0; i < values.length; i++) {
                score += weights[i + 1] * values[i];
            }

            return sigmoid(score);
        }

        void fit(List<FeatureVector> data) {
            if (data.isEmpty()) {
                throw new PipelineException(
                    "Model training data is empty"
                );
            }

            featureNames = data.get(0).values()
                .keySet()
                .stream()
                .sorted()
                .toList();

            Set<Integer> classes = data.stream()
                .map(FeatureVector::target)
                .collect(Collectors.toSet());

            if (classes.size() != 2) {
                throw new PipelineException(
                    "Binary model requires both classes"
                );
            }

            weights = new double[featureNames.size() + 1];

            for (int iteration = 0;
                 iteration < iterations;
                 iteration++) {

                double[] gradients = new double[weights.length];

                for (FeatureVector vector : data) {
                    double[] values = extract(vector);

                    double error =
                        probability(vector) - vector.target();

                    gradients[0] += error;

                    for (int j = 0; j < values.length; j++) {
                        gradients[j + 1] += error * values[j];
                    }
                }

                double scale = 1.0 / data.size();

                weights[0] -=
                    learningRate * gradients[0] * scale;

                for (int j = 1; j < weights.length; j++) {
                    double gradient =
                        gradients[j] * scale +
                        regularization * weights[j];

                    weights[j] -= learningRate * gradient;
                }
            }

            fitted = true;
        }

        List<Prediction> predict(List<FeatureVector> data) {
            if (!fitted) {
                throw new PipelineException(
                    "Model has not been fitted"
                );
            }

            return data.stream()
                .map(vector -> {
                    double probability = probability(vector);

                    return new Prediction(
                        probability,
                        probability >= 0.5 ? 1 : 0
                    );
                })
                .toList();
        }

        void save(Path path) throws IOException {
            if (!fitted) {
                throw new PipelineException(
                    "Cannot save an unfitted model"
                );
            }

            try (BufferedWriter writer = Files.newBufferedWriter(
                path,
                StandardCharsets.UTF_8
            )) {
                writer.write("ML_PIPELINE_V1");
                writer.newLine();

                for (String feature : featureNames) {
                    writer.write("FEATURE=" + feature);
                    writer.newLine();
                }

                for (double weight : weights) {
                    writer.write("WEIGHT=" + weight);
                    writer.newLine();
                }
            }
        }
    }

    static final class Pipeline {

        private final MedianImputer imputer = new MedianImputer();
        private final StandardScaler scaler = new StandardScaler();
        private final OneHotEncoder encoder = new OneHotEncoder();
        private final LogisticRegression estimator =
            new LogisticRegression();

        private PipelineStatus status = PipelineStatus.CREATED;

        private void transition(PipelineStatus next) {
            Map<PipelineStatus, Set<PipelineStatus>> allowed =
                Map.of(
                    PipelineStatus.CREATED,
                    Set.of(
                        PipelineStatus.VALIDATED,
                        PipelineStatus.FAILED
                    ),
                    PipelineStatus.VALIDATED,
                    Set.of(
                        PipelineStatus.FITTING,
                        PipelineStatus.FAILED
                    ),
                    PipelineStatus.FITTING,
                    Set.of(
                        PipelineStatus.FITTED,
                        PipelineStatus.FAILED
                    ),
                    PipelineStatus.FITTED,
                    Set.of(
                        PipelineStatus.PREDICTING,
                        PipelineStatus.FAILED
                    ),
                    PipelineStatus.PREDICTING,
                    Set.of(
                        PipelineStatus.FITTED,
                        PipelineStatus.FAILED
                    ),
                    PipelineStatus.FAILED,
                    Set.of()
                );

            if (!allowed.get(status).contains(next)) {
                throw new PipelineException(
                    "Invalid state transition: " + status + " -> " + next
                );
            }

            System.out.println(
                "[STATE] " + status + " -> " + next
            );

            status = next;
        }

        void fit(List<Customer> data) {
            transition(PipelineStatus.VALIDATED);
            transition(PipelineStatus.FITTING);

            try {
                imputer.fit(data);

                List<ImputedRecord> imputed =
                    imputer.transform(data);

                scaler.fit(imputed);

                List<ImputedRecord> scaled =
                    scaler.transform(imputed);

                encoder.fit(imputed);

                List<FeatureVector> encoded =
                    encoder.transform(scaled);

                estimator.fit(encoded);

                transition(PipelineStatus.FITTED);

            } catch (RuntimeException error) {
                status = PipelineStatus.FAILED;
                throw error;
            }
        }

        List<Prediction> predict(List<Customer> data) {
            transition(PipelineStatus.PREDICTING);

            try {
                List<ImputedRecord> imputed =
                    imputer.transform(data);

                List<ImputedRecord> scaled =
                    scaler.transform(imputed);

                List<FeatureVector> encoded =
                    encoder.transform(scaled);

                List<Prediction> result =
                    estimator.predict(encoded);

                transition(PipelineStatus.FITTED);

                return result;

            } catch (RuntimeException error) {
                status = PipelineStatus.FAILED;
                throw error;
            }
        }

        void save(Path path) throws IOException {
            if (status != PipelineStatus.FITTED) {
                throw new PipelineException(
                    "Only fitted pipelines can be persisted"
                );
            }

            estimator.save(path);
        }

        PipelineStatus status() {
            return status;
        }
    }

    static ModelMetrics evaluate(
        List<Integer> actual,
        List<Prediction> predictions
    ) {
        if (actual.isEmpty() || actual.size() != predictions.size()) {
            throw new PipelineException(
                "Invalid evaluation input"
            );
        }

        int tp = 0;
        int fp = 0;
        int tn = 0;
        int fn = 0;

        for (int i = 0; i < actual.size(); i++) {
            int expected = actual.get(i);
            int predicted = predictions.get(i).classification();

            if (expected == 1 && predicted == 1) tp++;
            else if (expected == 0 && predicted == 1) fp++;
            else if (expected == 0 && predicted == 0) tn++;
            else fn++;
        }

        double accuracy = (double) (tp + tn) / actual.size();

        double precision = tp + fp == 0
            ? 0
            : (double) tp / (tp + fp);

        double recall = tp + fn == 0
            ? 0
            : (double) tp / (tp + fn);

        double f1 = precision + recall == 0
            ? 0
            : 2 * precision * recall / (precision + recall);

        return new ModelMetrics(
            accuracy,
            precision,
            recall,
            f1,
            tp,
            fp,
            tn,
            fn
        );
    }

    static List<Customer> createDataset(int count) {
        Random random = new Random(42);
        List<Customer> customers = new ArrayList<>();

        String[] contracts = {
            "monthly", "annual", "two_year"
        };

        String[] regions = {
            "north", "south", "east", "west"
        };

        for (int i = 0; i < count; i++) {
            int age = 18 + random.nextInt(55);
            double spending = 25 + random.nextDouble() * 100;
            int tenure = 1 + random.nextInt(72);

            String contract =
                contracts[random.nextInt(contracts.length)];

            String region =
                regions[random.nextInt(regions.length)];

            double score =
                -1.0 +
                (spending - 65) * 0.012 -
                tenure * 0.025 +
                (contract.equals("monthly") ? 1.2 : 0) +
                (contract.equals("two_year") ? -1.0 : 0) +
                (age < 25 ? 0.4 : 0);

            int target = random.nextDouble() <
                1.0 / (1.0 + Math.exp(-score)) ? 1 : 0;

            double ageValue = random.nextDouble() < 0.03
                ? Double.NaN
                : age;

            double spendingValue = random.nextDouble() < 0.03
                ? Double.NaN
                : spending;

            customers.add(new Customer(
                ageValue,
                spendingValue,
                tenure,
                contract,
                region,
                target
            ));
        }

        return customers;
    }

    static TrainingSplit split(List<Customer> input) {
        List<Customer> shuffled = new ArrayList<>(input);

        Collections.shuffle(shuffled, new Random(42));

        int validationSize = shuffled.size() / 5;

        return new TrainingSplit(
            shuffled.subList(validationSize, shuffled.size()),
            shuffled.subList(0, validationSize)
        );
    }

    static FeatureSchema createSchema() {
        Map<String, FeatureType> columns = new LinkedHashMap<>();

        columns.put("age", FeatureType.NUMERIC);
        columns.put("spending", FeatureType.NUMERIC);
        columns.put("tenure", FeatureType.NUMERIC);
        columns.put("contract", FeatureType.CATEGORICAL);
        columns.put("region", FeatureType.CATEGORICAL);

        return new FeatureSchema(columns);
    }

    public static void main(String[] args) {
        try {
            List<Customer> dataset = createDataset(600);

            FeatureSchema schema = createSchema();

            dataset.forEach(schema::validate);

            TrainingSplit split = split(dataset);

            System.out.println(
                "Training records: " + split.training().size()
            );

            System.out.println(
                "Validation records: " + split.validation().size()
            );

            Pipeline pipeline = new Pipeline();

            pipeline.fit(split.training());

            List<Prediction> predictions =
                pipeline.predict(split.validation());

            List<Integer> actual = split.validation()
                .stream()
                .map(Customer::target)
                .toList();

            ModelMetrics metrics = evaluate(actual, predictions);

            System.out.printf(
                "\nAccuracy: %.4f%n",
                metrics.accuracy()
            );

            System.out.printf(
                "Precision: %.4f%n",
                metrics.precision()
            );

            System.out.printf(
                "Recall: %.4f%n",
                metrics.recall()
            );

            System.out.printf(
                "F1 score: %.4f%n",
                metrics.f1()
            );

            System.out.println("\nConfusion matrix:");
            System.out.printf(
                "%d %d%n%d %d%n",
                metrics.trueNegative(),
                metrics.falsePositive(),
                metrics.falseNegative(),
                metrics.truePositive()
            );

            List<Customer> newCustomers = List.of(
                new Customer(
                    28, 95, 6, "monthly", "north", 0
                ),
                new Customer(
                    55, 50, 65, "two_year", "south", 0
                )
            );

            System.out.println("\nInference results:");

            pipeline.predict(newCustomers).forEach(result ->
                System.out.printf(
                    "Probability: %.4f | Prediction: %d%n",
                    result.probability(),
                    result.classification()
                )
            );

            Path artifact = Path.of("java_ml_pipeline.model");

            pipeline.save(artifact);

            System.out.println(
                "\nModel artifact: " + artifact.toAbsolutePath()
            );

            if (pipeline.status() != PipelineStatus.FITTED) {
                throw new PipelineException(
                    "Unexpected final pipeline state"
                );
            }

            System.out.println(
                "Enterprise ML pipeline completed successfully."
            );

        } catch (IOException | RuntimeException error) {
            System.err.println(
                "ML pipeline failure: " + error.getMessage()
            );

            System.exitCode = 1;
        }
    }
}
