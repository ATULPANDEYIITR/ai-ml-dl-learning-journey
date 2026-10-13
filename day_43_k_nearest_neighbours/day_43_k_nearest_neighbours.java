import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

/*
 * KNNGovernance.java
 *
 * Enterprise case study: a manufacturing quality team classifies incoming
 * components and estimates inspection costs using k-nearest neighbors.
 *
 * The Java domain model makes prediction behavior explicit through immutable
 * records, enums, validation, and a service boundary.
 *
 * Compile and run:
 *   javac KNNGovernance.java
 *   java KNNGovernance
 *
 * Requires Java 17 or later.
 */

public class KNNGovernance {

    enum Task {
        CLASSIFICATION,
        REGRESSION
    }

    enum DistanceMetric {
        EUCLIDEAN,
        MANHATTAN,
        CHEBYSHEV
    }

    enum Weighting {
        UNIFORM,
        INVERSE_DISTANCE
    }

    enum QualityStatus {
        ACCEPTABLE,
        REVIEW,
        REJECT
    }

    record ComponentSample(
        String componentId,
        List<Double> features,
        QualityStatus status,
        double inspectionCost
    ) {
        ComponentSample {
            if (componentId == null || componentId.isBlank()) {
                throw new IllegalArgumentException("Component ID is required.");
            }
            features = List.copyOf(Objects.requireNonNull(features, "features"));
            validateVector(features, "features");
            Objects.requireNonNull(status, "status");

            if (!Double.isFinite(inspectionCost) || inspectionCost < 0) {
                throw new IllegalArgumentException("Inspection cost must be finite and nonnegative.");
            }
        }
    }

    record Neighbor<T>(int index, double distance, T target) {
        Neighbor {
            if (index < 0 || !Double.isFinite(distance) || distance < 0) {
                throw new IllegalArgumentException("Invalid neighbor.");
            }
            Objects.requireNonNull(target, "target");
        }
    }

    record Prediction<T>(
        T value,
        Map<T, Double> probabilities,
        List<Neighbor<T>> neighbors
    ) {
        Prediction {
            Objects.requireNonNull(value, "value");
            probabilities = Map.copyOf(probabilities);
            neighbors = List.copyOf(neighbors);
        }
    }

    static void validateVector(List<Double> vector, String name) {
        if (vector.isEmpty()) {
            throw new IllegalArgumentException(name + " cannot be empty.");
        }
        for (Double value : vector) {
            if (value == null || !Double.isFinite(value)) {
                throw new IllegalArgumentException(name + " contains a null or nonfinite value.");
            }
        }
    }

    static void validateMatrix(List<List<Double>> X) {
        if (X == null || X.isEmpty()) {
            throw new IllegalArgumentException("Training matrix cannot be empty.");
        }
        int dimensions = X.getFirst().size();
        if (dimensions == 0) {
            throw new IllegalArgumentException("Samples require at least one feature.");
        }

        for (List<Double> row : X) {
            validateVector(row, "sample");
            if (row.size() != dimensions) {
                throw new IllegalArgumentException("Inconsistent feature dimensions.");
            }
        }
    }

    static double distance(
        List<Double> a,
        List<Double> b,
        DistanceMetric metric
    ) {
        validateVector(a, "left operand");
        validateVector(b, "right operand");

        if (a.size() != b.size()) {
            throw new IllegalArgumentException("Feature dimensions do not match.");
        }

        double result = 0.0;
        for (int index = 0; index < a.size(); index++) {
            double difference = Math.abs(a.get(index) - b.get(index));

            switch (metric) {
                case EUCLIDEAN -> result += difference * difference;
                case MANHATTAN -> result += difference;
                case CHEBYSHEV -> result = Math.max(result, difference);
            }
        }

        if (metric == DistanceMetric.EUCLIDEAN) {
            result = Math.sqrt(result);
        }
        if (!Double.isFinite(result)) {
            throw new ArithmeticException("Distance overflow.");
        }
        return result;
    }

    static final class StandardScaler {
        private List<Double> means = List.of();
        private List<Double> scales = List.of();
        private boolean fitted;

        StandardScaler fit(List<List<Double>> X) {
            validateMatrix(X);
            int dimensions = X.getFirst().size();
            List<Double> calculatedMeans = new ArrayList<>();
            List<Double> calculatedScales = new ArrayList<>();

            for (int column = 0; column < dimensions; column++) {
                final int feature = column;
                double mean = X.stream()
                    .mapToDouble(row -> row.get(feature))
                    .average()
                    .orElseThrow();

                double variance = X.stream()
                    .mapToDouble(row -> Math.pow(row.get(feature) - mean, 2))
                    .average()
                    .orElseThrow();

                double deviation = Math.sqrt(variance);
                calculatedMeans.add(mean);
                calculatedScales.add(deviation > 0 ? deviation : 1.0);
            }

            means = List.copyOf(calculatedMeans);
            scales = List.copyOf(calculatedScales);
            fitted = true;
            return this;
        }

        List<Double> transform(List<Double> row) {
            if (!fitted) {
                throw new IllegalStateException("Scaler has not been fitted.");
            }
            validateVector(row, "sample");
            if (row.size() != means.size()) {
                throw new IllegalArgumentException("Feature count differs from scaler.");
            }

            List<Double> transformed = new ArrayList<>();
            for (int index = 0; index < row.size(); index++) {
                transformed.add((row.get(index) - means.get(index)) / scales.get(index));
            }
            return List.copyOf(transformed);
        }

        List<List<Double>> transformMatrix(List<List<Double>> X) {
            validateMatrix(X);
            return X.stream().map(this::transform).toList();
        }
    }

    static final class KNNModel<T> {
        private final int k;
        private final Task task;
        private final DistanceMetric metric;
        private final Weighting weighting;
        private final List<List<Double>> samples;
        private final List<T> targets;
        private final Set<T> classes;
        private final int dimensions;

        KNNModel(
            int k,
            Task task,
            DistanceMetric metric,
            Weighting weighting,
            List<List<Double>> X,
            List<T> y
        ) {
            if (k < 1) {
                throw new IllegalArgumentException("k must be positive.");
            }
            validateMatrix(X);
            if (y == null || y.size() != X.size()) {
                throw new IllegalArgumentException("Features and targets must align.");
            }
            if (k > X.size()) {
                throw new IllegalArgumentException("k exceeds the training sample count.");
            }

            this.k = k;
            this.task = Objects.requireNonNull(task, "task");
            this.metric = Objects.requireNonNull(metric, "metric");
            this.weighting = Objects.requireNonNull(weighting, "weighting");
            this.samples = X.stream().map(List::copyOf).toList();
            this.targets = List.copyOf(y);
            this.dimensions = X.getFirst().size();

            if (targets.stream().anyMatch(Objects::isNull)) {
                throw new IllegalArgumentException("Targets cannot be null.");
            }

            this.classes = new LinkedHashSet<>(targets);

            if (task == Task.REGRESSION) {
                for (T target : targets) {
                    if (!(target instanceof Number number) ||
                        !Double.isFinite(number.doubleValue())) {
                        throw new IllegalArgumentException(
                            "Regression targets must be finite numbers."
                        );
                    }
                }
            }
        }

        List<Neighbor<T>> kneighbors(List<Double> query) {
            validateVector(query, "query");
            if (query.size() != dimensions) {
                throw new IllegalArgumentException("Query dimensions do not match model.");
            }

            List<Neighbor<T>> neighbors = new ArrayList<>();
            for (int index = 0; index < samples.size(); index++) {
                neighbors.add(new Neighbor<>(
                    index,
                    distance(query, samples.get(index), metric),
                    targets.get(index)
                ));
            }

            neighbors.sort(
                Comparator.comparingDouble(Neighbor<T>::distance)
                    .thenComparingInt(Neighbor<T>::index)
            );
            return List.copyOf(neighbors.subList(0, k));
        }

        private double weight(double distance) {
            if (weighting == Weighting.UNIFORM || distance == 0.0) {
                return 1.0;
            }
            return 1.0 / distance;
        }

        Prediction<T> classify(List<Double> query) {
            if (task != Task.CLASSIFICATION) {
                throw new IllegalStateException("This model is configured for regression.");
            }

            List<Neighbor<T>> neighbors = kneighbors(query);
            List<Neighbor<T>> exact = neighbors.stream()
                .filter(neighbor -> neighbor.distance() == 0.0)
                .toList();

            if (weighting == Weighting.INVERSE_DISTANCE && !exact.isEmpty()) {
                neighbors = exact;
            }

            Map<T, Double> totals = new LinkedHashMap<>();
            Map<T, Double> distanceTotals = new LinkedHashMap<>();

            for (T label : classes) {
                totals.put(label, 0.0);
                distanceTotals.put(label, 0.0);
            }

            for (Neighbor<T> neighbor : neighbors) {
                totals.merge(neighbor.target(), weight(neighbor.distance()), Double::sum);
                distanceTotals.merge(
                    neighbor.target(), neighbor.distance(), Double::sum
                );
            }

            double denominator = totals.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

            Map<T, Double> probabilities = new LinkedHashMap<>();
            totals.forEach((label, total) -> probabilities.put(label, total / denominator));

            // Ties are resolved by total weight, then cumulative distance, then label text.
            T winner = classes.stream().min((a, b) -> {
                int byWeight = Double.compare(totals.get(b), totals.get(a));
                if (byWeight != 0) return byWeight;

                int byDistance = Double.compare(
                    distanceTotals.get(a), distanceTotals.get(b)
                );
                if (byDistance != 0) return byDistance;

                return String.valueOf(a).compareTo(String.valueOf(b));
            }).orElseThrow();

            return new Prediction<>(winner, probabilities, neighbors);
        }

        double regress(List<Double> query) {
            if (task != Task.REGRESSION) {
                throw new IllegalStateException("This model is configured for classification.");
            }

            List<Neighbor<T>> neighbors = kneighbors(query);
            List<Neighbor<T>> exact = neighbors.stream()
                .filter(neighbor -> neighbor.distance() == 0.0)
                .toList();

            if (weighting == Weighting.INVERSE_DISTANCE && !exact.isEmpty()) {
                neighbors = exact;
            }

            double numerator = 0.0;
            double denominator = 0.0;

            for (Neighbor<T> neighbor : neighbors) {
                double target = ((Number) neighbor.target()).doubleValue();
                double neighborWeight = weight(neighbor.distance());
                numerator += neighborWeight * target;
                denominator += neighborWeight;
            }

            return numerator / denominator;
        }
    }

    static double accuracy(List<?> actual, List<?> predicted) {
        if (actual.isEmpty() || actual.size() != predicted.size()) {
            throw new IllegalArgumentException("Evaluation vectors must align and be nonempty.");
        }

        long correct = java.util.stream.IntStream.range(0, actual.size())
            .filter(index -> Objects.equals(actual.get(index), predicted.get(index)))
            .count();

        return (double) correct / actual.size();
    }

    static double meanAbsoluteError(List<Double> actual, List<Double> predicted) {
        if (actual.isEmpty() || actual.size() != predicted.size()) {
            throw new IllegalArgumentException("Evaluation vectors must align and be nonempty.");
        }

        return java.util.stream.IntStream.range(0, actual.size())
            .mapToDouble(index -> Math.abs(actual.get(index) - predicted.get(index)))
            .average()
            .orElseThrow();
    }

    public static void main(String[] args) {
        List<ComponentSample> inventory = List.of(
            new ComponentSample("VALVE-201", List.of(0.8, 1.2, 14.0),
                QualityStatus.ACCEPTABLE, 400.0),
            new ComponentSample("VALVE-202", List.of(1.0, 1.5, 18.0),
                QualityStatus.ACCEPTABLE, 450.0),
            new ComponentSample("VALVE-203", List.of(1.8, 2.7, 29.0),
                QualityStatus.REVIEW, 1200.0),
            new ComponentSample("VALVE-204", List.of(2.1, 3.1, 35.0),
                QualityStatus.REVIEW, 1450.0),
            new ComponentSample("VALVE-205", List.of(4.2, 5.7, 60.0),
                QualityStatus.REJECT, 4000.0),
            new ComponentSample("VALVE-206", List.of(4.8, 6.4, 72.0),
                QualityStatus.REJECT, 5100.0),
            new ComponentSample("VALVE-207", List.of(1.4, 2.0, 23.0),
                QualityStatus.ACCEPTABLE, 800.0),
            new ComponentSample("VALVE-208", List.of(3.7, 4.9, 54.0),
                QualityStatus.REJECT, 3500.0)
        );

        List<List<Double>> X = inventory.stream()
            .map(ComponentSample::features)
            .toList();

        List<QualityStatus> y = inventory.stream()
            .map(ComponentSample::status)
            .toList();

        List<Double> costs = inventory.stream()
            .map(ComponentSample::inspectionCost)
            .toList();

        StandardScaler scaler = new StandardScaler().fit(X);
        List<List<Double>> normalized = scaler.transformMatrix(X);

        KNNModel<QualityStatus> classifier = new KNNModel<>(
            3,
            Task.CLASSIFICATION,
            DistanceMetric.EUCLIDEAN,
            Weighting.INVERSE_DISTANCE,
            normalized,
            y
        );

        List<Double> newComponent = scaler.transform(List.of(3.4, 4.6, 48.0));
        Prediction<QualityStatus> prediction = classifier.classify(newComponent);

        System.out.println("Incoming component quality assessment");
        System.out.println("Predicted status: " + prediction.value());
        System.out.println("Weighted class probabilities: " + prediction.probabilities());
        System.out.println("Neighbor records: " + prediction.neighbors());

        KNNModel<Double> costRegressor = new KNNModel<>(
            3,
            Task.REGRESSION,
            DistanceMetric.MANHATTAN,
            Weighting.INVERSE_DISTANCE,
            normalized,
            costs
        );

        System.out.printf(
            "Estimated inspection cost: %.2f%n",
            costRegressor.regress(newComponent)
        );

        List<QualityStatus> inSamplePredictions = normalized.stream()
            .map(sample -> classifier.classify(sample).value())
            .toList();

        System.out.printf(
            "In-sample diagnostic accuracy: %.3f%n",
            accuracy(y, inSamplePredictions)
        );

        double mae = meanAbsoluteError(costs, normalized.stream()
            .map(costRegressor::regress)
            .toList());

        System.out.printf("In-sample cost MAE: %.2f%n", mae);

        try {
            classifier.classify(List.of(1.0, 2.0));
        } catch (IllegalArgumentException exception) {
            System.out.println("Invalid request rejected: " + exception.getMessage());
        }

        try {
            new KNNModel<>(
                50, Task.CLASSIFICATION, DistanceMetric.EUCLIDEAN,
                Weighting.UNIFORM, normalized, y
            );
        } catch (IllegalArgumentException exception) {
            System.out.println("Invalid policy rejected: " + exception.getMessage());
        }

        // Quality predictions inform inspection queues; they do not replace
        // contractual acceptance tests or an authorized human release decision.
    }
}
