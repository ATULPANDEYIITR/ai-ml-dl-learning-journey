import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Enterprise-oriented K-Means clustering with explicit configuration,
 * immutable result records, K-Means++ initialization, and model selection.
 */
public class KMeansEnterprise {

    public enum Initialization {
        RANDOM,
        KMEANS_PLUS_PLUS
    }

    public record Config(
            int k,
            Initialization initialization,
            int maxIterations,
            double tolerance,
            long seed,
            int restarts
    ) {
        public Config {
            if (k < 1) throw new IllegalArgumentException("K must be positive.");
            Objects.requireNonNull(initialization, "Initialization is required.");
            if (maxIterations < 1 || restarts < 1) {
                throw new IllegalArgumentException(
                        "Iteration and restart counts must be positive.");
            }
            if (!Double.isFinite(tolerance) || tolerance < 0) {
                throw new IllegalArgumentException(
                        "Tolerance must be finite and non-negative.");
            }
        }
    }

    public record Result(
            List<List<Double>> centroids,
            List<Integer> labels,
            double inertia,
            int iterations,
            boolean converged
    ) {
        public Result {
            centroids = centroids.stream()
                    .map(List::copyOf)
                    .toList();
            labels = List.copyOf(labels);
            if (!Double.isFinite(inertia) || inertia < 0) {
                throw new IllegalArgumentException("Invalid inertia.");
            }
        }

        public List<Integer> clusterSizes(int k) {
            int[] counts = new int[k];
            labels.forEach(label -> counts[label]++);
            return Arrays.stream(counts).boxed().toList();
        }
    }

    public static final class DatasetException extends IllegalArgumentException {
        public DatasetException(String message) {
            super(message);
        }
    }

    public static final class KMeansService {
        private final Config config;

        public KMeansService(Config config) {
            this.config = Objects.requireNonNull(config);
        }

        public Result fit(List<List<Double>> input) {
            List<List<Double>> points = validate(input);

            if (config.k() > points.size()) {
                throw new DatasetException("K cannot exceed the sample count.");
            }

            Random master = new Random(config.seed());
            Result best = null;

            for (int run = 0; run < config.restarts(); run++) {
                Result candidate = fitOnce(points, new Random(master.nextLong()));

                if (best == null || candidate.inertia() < best.inertia()) {
                    best = candidate;
                }
            }

            return Objects.requireNonNull(best);
        }

        public List<Integer> predict(List<List<Double>> input, Result result) {
            List<List<Double>> points = validate(input);
            int dimension = result.centroids().get(0).size();

            if (points.stream().anyMatch(point -> point.size() != dimension)) {
                throw new DatasetException(
                        "Prediction dimensions do not match training data.");
            }

            return points.stream()
                    .map(point -> nearest(point, result.centroids()))
                    .toList();
        }

        private Result fitOnce(List<List<Double>> points, Random random) {
            List<List<Double>> centers = initialize(points, random);
            List<Integer> labels = new ArrayList<>(
                    Collections.nCopies(points.size(), -1));
            boolean converged = false;
            int iterations = 0;
            int dimension = points.get(0).size();

            for (int iteration = 1; iteration <= config.maxIterations(); iteration++) {
                List<Integer> nextLabels = points.stream()
                        .map(point -> nearest(point, centers))
                        .toList();

                double[][] sums = new double[config.k()][dimension];
                int[] counts = new int[config.k()];

                for (int i = 0; i < points.size(); i++) {
                    int cluster = nextLabels.get(i);
                    counts[cluster]++;

                    for (int feature = 0; feature < dimension; feature++) {
                        sums[cluster][feature] += points.get(i).get(feature);
                    }
                }

                List<List<Double>> nextCenters = new ArrayList<>();

                for (int cluster = 0; cluster < config.k(); cluster++) {
                    if (counts[cluster] == 0) {
                        // Reseed an empty cluster using the most distant sample.
                        List<Double> farthest = points.stream()
                                .max(Comparator.comparingDouble(point ->
                                        minimumDistance(point, centers)))
                                .orElseThrow();
                        nextCenters.add(List.copyOf(farthest));
                    } else {
                        List<Double> center = new ArrayList<>();

                        for (int feature = 0; feature < dimension; feature++) {
                            center.add(sums[cluster][feature] / counts[cluster]);
                        }

                        nextCenters.add(List.copyOf(center));
                    }
                }

                double shift = IntStream.range(0, config.k())
                        .mapToDouble(cluster -> squaredDistance(
                                centers.get(cluster), nextCenters.get(cluster)))
                        .max()
                        .orElse(0.0);

                boolean stable = nextLabels.equals(labels);
                labels = new ArrayList<>(nextLabels);
                centers = nextCenters;
                iterations = iteration;

                if (stable || shift <= config.tolerance() * config.tolerance()) {
                    converged = true;
                    break;
                }
            }

            // Reassign using the final centroids before calculating the objective.
            List<Integer> finalLabels = points.stream()
                    .map(point -> nearest(point, centers))
                    .toList();

            double inertia = IntStream.range(0, points.size())
                    .mapToDouble(i -> squaredDistance(
                            points.get(i), centers.get(finalLabels.get(i))))
                    .sum();

            return new Result(centers, finalLabels, inertia, iterations, converged);
        }

        private List<List<Double>> initialize(
                List<List<Double>> points, Random random) {
            List<List<Double>> centers = new ArrayList<>();

            if (config.initialization() == Initialization.RANDOM) {
                List<Integer> indices = IntStream.range(0, points.size())
                        .boxed().collect(java.util.stream.Collectors.toCollection(
                                ArrayList::new));
                Collections.shuffle(indices, random);

                for (int i = 0; i < config.k(); i++) {
                    centers.add(points.get(indices.get(i)));
                }

                return centers;
            }

            centers.add(points.get(random.nextInt(points.size())));

            while (centers.size() < config.k()) {
                double[] weights = points.stream()
                        .mapToDouble(point -> minimumDistance(point, centers))
                        .toArray();
                double total = Arrays.stream(weights).sum();

                if (total == 0.0) {
                    List<List<Double>> unused = points.stream()
                            .filter(point -> !centers.contains(point))
                            .toList();

                    centers.add(unused.isEmpty()
                            ? points.get(random.nextInt(points.size()))
                            : unused.get(random.nextInt(unused.size())));
                    continue;
                }

                double threshold = random.nextDouble() * total;
                double cumulative = 0.0;
                int selected = points.size() - 1;

                for (int i = 0; i < weights.length; i++) {
                    cumulative += weights[i];
                    if (cumulative >= threshold) {
                        selected = i;
                        break;
                    }
                }

                centers.add(points.get(selected));
            }

            return centers;
        }
    }

    private static List<List<Double>> validate(List<List<Double>> input) {
        if (input == null || input.isEmpty()) {
            throw new DatasetException("Dataset must not be empty.");
        }

        if (input.get(0) == null || input.get(0).isEmpty()) {
            throw new DatasetException("Points must contain features.");
        }

        int dimension = input.get(0).size();
        List<List<Double>> points = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            List<Double> point = input.get(i);

            if (point == null || point.size() != dimension) {
                throw new DatasetException("Inconsistent dimensions at sample " + i);
            }

            for (Double value : point) {
                if (value == null || !Double.isFinite(value)) {
                    throw new DatasetException("Features must be finite numbers.");
                }
            }

            points.add(List.copyOf(point));
        }

        return List.copyOf(points);
    }

    private static double squaredDistance(List<Double> a, List<Double> b) {
        double distance = 0.0;

        for (int i = 0; i < a.size(); i++) {
            double difference = a.get(i) - b.get(i);
            distance += difference * difference;
        }

        return distance;
    }

    private static int nearest(List<Double> point, List<List<Double>> centers) {
        int best = 0;
        double bestDistance = squaredDistance(point, centers.get(0));

        for (int i = 1; i < centers.size(); i++) {
            double distance = squaredDistance(point, centers.get(i));

            if (distance < bestDistance) {
                best = i;
                bestDistance = distance;
            }
        }

        return best;
    }

    private static double minimumDistance(
            List<Double> point, List<List<Double>> centers) {
        return centers.stream()
                .mapToDouble(center -> squaredDistance(point, center))
                .min()
                .orElse(Double.POSITIVE_INFINITY);
    }

    public static void main(String[] args) {
        List<List<Double>> customers = List.of(
                List.of(1.0, 1.2), List.of(1.3, 0.8),
                List.of(0.7, 1.1), List.of(1.2, 1.5),
                List.of(5.0, 5.2), List.of(5.4, 4.8),
                List.of(4.7, 5.1), List.of(5.2, 5.5),
                List.of(9.0, 1.0), List.of(9.4, 1.4),
                List.of(8.6, 0.7), List.of(9.2, 0.6)
        );

        Config config = new Config(
                3, Initialization.KMEANS_PLUS_PLUS, 300, 1e-6, 42L, 10);
        KMeansService service = new KMeansService(config);
        Result result = service.fit(customers);

        System.out.printf("Inertia: %.4f%n", result.inertia());
        System.out.println("Iterations: " + result.iterations());
        System.out.println("Converged: " + result.converged());
        System.out.println("Cluster sizes: " + result.clusterSizes(config.k()));

        for (int i = 0; i < result.centroids().size(); i++) {
            System.out.println("Centroid " + i + ": " + result.centroids().get(i));
        }

        System.out.println("\nElbow candidates:");
        for (int k = 1; k <= 6; k++) {
            Config candidateConfig = new Config(
                    k, Initialization.KMEANS_PLUS_PLUS, 300, 1e-6, 42L, 5);
            Result candidate = new KMeansService(candidateConfig).fit(customers);
            System.out.printf("K=%d, inertia=%.4f%n", k, candidate.inertia());
        }

        if (result.labels().size() != customers.size()) {
            throw new IllegalStateException("Cluster assignment count is invalid.");
        }
    }
}
