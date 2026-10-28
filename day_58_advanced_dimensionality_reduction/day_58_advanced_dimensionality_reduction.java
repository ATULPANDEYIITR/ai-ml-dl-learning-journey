import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

public class DimensionalityReductionLab {

    enum Method {
        PCA,
        TSNE,
        UMAP_STYLE
    }

    enum ExperimentState {
        CREATED,
        VALIDATED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    record Point(double x, double y) {}

    record ExperimentConfig(
        int iterations,
        int neighbors,
        double perplexity,
        double learningRate,
        double minDist,
        long seed
    ) {
        ExperimentConfig {
            if (iterations < 1) {
                throw new IllegalArgumentException(
                    "iterations must be positive"
                );
            }
            if (neighbors < 1) {
                throw new IllegalArgumentException(
                    "neighbors must be positive"
                );
            }
            if (perplexity <= 1.0) {
                throw new IllegalArgumentException(
                    "perplexity must be greater than one"
                );
            }
            if (learningRate <= 0.0) {
                throw new IllegalArgumentException(
                    "learningRate must be positive"
                );
            }
            if (minDist <= 0.0) {
                throw new IllegalArgumentException(
                    "minDist must be positive"
                );
            }
        }
    }

    record ExperimentReport(
        Method method,
        ExperimentState state,
        int observations,
        long durationMillis,
        List<Point> embedding
    ) {}

    interface ReductionAlgorithm {
        List<Point> reduce(
            double[][] data,
            ExperimentConfig config
        );
    }

    static final class ReductionException extends RuntimeException {
        ReductionException(String message) {
            super(message);
        }
    }

    static final class DatasetValidator {
        private DatasetValidator() {}

        static void validate(double[][] data) {
            if (data == null || data.length == 0) {
                throw new ReductionException(
                    "Dataset cannot be empty"
                );
            }

            if (data[0] == null || data[0].length < 2) {
                throw new ReductionException(
                    "Dataset requires at least two features"
                );
            }

            int width = data[0].length;

            for (double[] row : data) {
                if (row == null || row.length != width) {
                    throw new ReductionException(
                        "All rows must have equal dimensions"
                    );
                }

                for (double value : row) {
                    if (!Double.isFinite(value)) {
                        throw new ReductionException(
                            "Feature values must be finite"
                        );
                    }
                }
            }
        }
    }

    static final class StandardScaler {
        static double[][] transform(double[][] input) {
            DatasetValidator.validate(input);

            int n = input.length;
            int d = input[0].length;

            double[] means = new double[d];
            double[] scales = new double[d];

            for (double[] row : input) {
                for (int j = 0; j < d; j++) {
                    means[j] += row[j];
                }
            }

            for (int j = 0; j < d; j++) {
                means[j] /= n;
            }

            for (double[] row : input) {
                for (int j = 0; j < d; j++) {
                    double delta = row[j] - means[j];
                    scales[j] += delta * delta;
                }
            }

            for (int j = 0; j < d; j++) {
                scales[j] = Math.sqrt(
                    scales[j] / Math.max(1, n - 1)
                );

                if (scales[j] < 1e-12) {
                    scales[j] = 1.0;
                }
            }

            double[][] result = new double[n][d];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    result[i][j] =
                        (input[i][j] - means[j]) / scales[j];
                }
            }

            return result;
        }
    }

    static final class PCAAlgorithm implements ReductionAlgorithm {

        @Override
        public List<Point> reduce(
            double[][] data,
            ExperimentConfig config
        ) {
            double[][] scaled = StandardScaler.transform(data);
            int n = scaled.length;
            int d = scaled[0].length;

            double[][] covariance = new double[d][d];

            for (double[] row : scaled) {
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        covariance[i][j] += row[i] * row[j];
                    }
                }
            }

            double divisor = Math.max(1, n - 1);

            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    covariance[i][j] /= divisor;
                }
            }

            List<double[]> components = new ArrayList<>();

            for (int component = 0; component < 2; component++) {
                EigenPair eigenPair =
                    powerIteration(
                        covariance,
                        config.seed() + component
                    );

                components.add(eigenPair.vector());

                double eigenvalue = eigenPair.value();
                double[] vector = eigenPair.vector();

                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        covariance[i][j] -=
                            eigenvalue * vector[i] * vector[j];
                    }
                }
            }

            List<Point> result = new ArrayList<>();

            for (double[] row : scaled) {
                double x = dot(row, components.get(0));
                double y = dot(row, components.get(1));
                result.add(new Point(x, y));
            }

            return result;
        }

        private record EigenPair(
            double value,
            double[] vector
        ) {}

        private static EigenPair powerIteration(
            double[][] matrix,
            long seed
        ) {
            Random random = new Random(seed);
            double[] vector = new double[matrix.length];

            for (int i = 0; i < vector.length; i++) {
                vector[i] = random.nextDouble() * 2 - 1;
            }

            normalize(vector);

            for (int iteration = 0; iteration < 200; iteration++) {
                double[] candidate = multiply(matrix, vector);
                normalize(candidate);

                double change = 0.0;

                for (int i = 0; i < vector.length; i++) {
                    change = Math.max(
                        change,
                        Math.abs(candidate[i] - vector[i])
                    );
                }

                vector = candidate;

                if (change < 1e-10) {
                    break;
                }
            }

            double[] product = multiply(matrix, vector);
            double value = dot(vector, product);

            return new EigenPair(value, vector);
        }

        private static double[] multiply(
            double[][] matrix,
            double[] vector
        ) {
            double[] result = new double[matrix.length];

            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < vector.length; j++) {
                    result[i] +=
                        matrix[i][j] * vector[j];
                }
            }

            return result;
        }

        private static void normalize(double[] vector) {
            double norm = Math.sqrt(dot(vector, vector));

            if (norm < 1e-12) {
                throw new ReductionException(
                    "Degenerate PCA component"
                );
            }

            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }

        private static double dot(double[] a, double[] b) {
            double result = 0.0;

            for (int i = 0; i < a.length; i++) {
                result += a[i] * b[i];
            }

            return result;
        }
    }

    static final class GraphEmbeddingAlgorithm
        implements ReductionAlgorithm {

        private final Method method;

        GraphEmbeddingAlgorithm(Method method) {
            this.method = Objects.requireNonNull(method);
        }

        @Override
        public List<Point> reduce(
            double[][] input,
            ExperimentConfig config
        ) {
            double[][] data =
                StandardScaler.transform(input);

            int n = data.length;

            if (config.neighbors() >= n) {
                throw new ReductionException(
                    "Neighbor count must be smaller than sample count"
                );
            }

            double[][] distances = pairwiseDistances(data);
            List<List<Integer>> graph =
                buildNeighborGraph(distances, config.neighbors());

            Random random = new Random(config.seed());
            List<Point> embedding = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                embedding.add(
                    new Point(
                        random.nextDouble() * 2 - 1,
                        random.nextDouble() * 2 - 1
                    )
                );
            }

            for (int iteration = 0;
                 iteration < config.iterations();
                 iteration++) {

                double[][] gradient =
                    new double[n][2];

                for (int i = 0; i < n; i++) {
                    List<Integer> neighbors =
                        graph.get(i);

                    double sigma = 1.0;

                    if (!neighbors.isEmpty()) {
                        sigma =
                            neighbors.stream()
                                .mapToDouble(
                                    j -> Math.sqrt(
                                        distances[i][j]
                                    )
                                )
                                .average()
                                .orElse(1.0);
                    }

                    sigma = Math.max(sigma, 1e-9);

                    for (int j : neighbors) {
                        Point a = embedding.get(i);
                        Point b = embedding.get(j);

                        double dx = a.x() - b.x();
                        double dy = a.y() - b.y();
                        double distance =
                            Math.sqrt(dx * dx + dy * dy) +
                            1e-9;

                        double sourceDistance =
                            Math.sqrt(distances[i][j]);

                        double weight =
                            Math.exp(
                                -sourceDistance / sigma
                            );

                        double force =
                            weight *
                            (distance - config.minDist());

                        gradient[i][0] +=
                            force * dx / distance;

                        gradient[i][1] +=
                            force * dy / distance;
                    }
                }

                double rate =
                    config.learningRate() /
                    (1.0 + iteration / 100.0);

                for (int i = 0; i < n; i++) {
                    Point current = embedding.get(i);

                    embedding.set(
                        i,
                        new Point(
                            current.x() -
                                rate * 0.01 * gradient[i][0],
                            current.y() -
                                rate * 0.01 * gradient[i][1]
                        )
                    );
                }
            }

            return embedding;
        }

        private static double[][] pairwiseDistances(
            double[][] data
        ) {
            int n = data.length;
            double[][] distances = new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double value = 0.0;

                    for (int k = 0; k < data[i].length; k++) {
                        double delta =
                            data[i][k] - data[j][k];
                        value += delta * delta;
                    }

                    distances[i][j] = value;
                    distances[j][i] = value;
                }
            }

            return distances;
        }

        private static List<List<Integer>> buildNeighborGraph(
            double[][] distances,
            int k
        ) {
            List<List<Integer>> graph =
                new ArrayList<>();

            for (int i = 0; i < distances.length; i++) {
                List<Integer> candidates =
                    new ArrayList<>();

                for (int j = 0; j < distances.length; j++) {
                    if (i != j) {
                        candidates.add(j);
                    }
                }

                candidates.sort(
                    Comparator.comparingDouble(
                        j -> distances[i][j]
                    )
                );

                graph.add(
                    new ArrayList<>(
                        candidates.subList(0, k)
                    )
                );
            }

            return graph;
        }
    }

    static final class ReductionService {
        private final Map<Method, ReductionAlgorithm> algorithms =
            new HashMap<>();

        ReductionService() {
            algorithms.put(Method.PCA, new PCAAlgorithm());
            algorithms.put(
                Method.TSNE,
                new GraphEmbeddingAlgorithm(Method.TSNE)
            );
            algorithms.put(
                Method.UMAP_STYLE,
                new GraphEmbeddingAlgorithm(Method.UMAP_STYLE)
            );
        }

        ExperimentReport execute(
            Method method,
            double[][] data,
            ExperimentConfig config
        ) {
            ExperimentState state =
                ExperimentState.CREATED;

            long start = System.currentTimeMillis();

            try {
                DatasetValidator.validate(data);
                state = ExperimentState.VALIDATED;
                state = ExperimentState.RUNNING;

                ReductionAlgorithm algorithm =
                    algorithms.get(method);

                if (algorithm == null) {
                    throw new ReductionException(
                        "No algorithm registered for method"
                    );
                }

                List<Point> embedding =
                    algorithm.reduce(data, config);

                state = ExperimentState.COMPLETED;

                return new ExperimentReport(
                    method,
                    state,
                    data.length,
                    System.currentTimeMillis() - start,
                    List.copyOf(embedding)
                );
            } catch (RuntimeException error) {
                state = ExperimentState.FAILED;
                throw error;
            }
        }
    }

    static double[][] createDataset(
        int samplesPerCurve,
        long seed
    ) {
        if (samplesPerCurve < 5) {
            throw new IllegalArgumentException(
                "At least five observations per curve are required"
            );
        }

        Random random = new Random(seed);
        double[][] result =
            new double[samplesPerCurve * 2][4];

        for (int curve = 0; curve < 2; curve++) {
            for (int i = 0; i < samplesPerCurve; i++) {
                double t =
                    4.0 * Math.PI * i /
                    (samplesPerCurve - 1);

                double noise =
                    random.nextGaussian() * 0.05;

                double shift =
                    curve == 0 ? 0.0 : Math.PI;

                int index =
                    curve * samplesPerCurve + i;

                result[index] = new double[] {
                    t * Math.cos(t + shift) + noise,
                    t * Math.sin(t + shift) + noise,
                    t / (4.0 * Math.PI) +
                        (curve == 0 ? 0.0 : 1.5) +
                        noise,
                    Math.sin(t / 2.0 + shift) + noise
                };
            }
        }

        return result;
    }

    static void printReport(
        ExperimentReport report
    ) {
        System.out.println(
            "method=" + report.method()
        );
        System.out.println(
            "state=" + report.state()
        );
        System.out.println(
            "observations=" + report.observations()
        );
        System.out.println(
            "durationMs=" + report.durationMillis()
        );

        int limit =
            Math.min(10, report.embedding().size());

        for (int i = 0; i < limit; i++) {
            Point point =
                report.embedding().get(i);

            System.out.printf(
                "%d %.6f %.6f%n",
                i,
                point.x(),
                point.y()
            );
        }

        System.out.println();
    }

    public static void main(String[] args) {
        ExperimentConfig config =
            new ExperimentConfig(
                250,
                8,
                8.0,
                50.0,
                0.1,
                42L
            );

        double[][] dataset =
            createDataset(20, config.seed());

        ReductionService service =
            new ReductionService();

        for (Method method : Method.values()) {
            ExperimentReport report =
                service.execute(
                    method,
                    dataset,
                    config
                );

            printReport(report);
        }
    }
}
