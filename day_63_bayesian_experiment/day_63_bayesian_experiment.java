import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class BayesianOptimizationEnterprise {

    enum Direction {
        MAXIMIZE,
        MINIMIZE
    }

    record Bounds(double low, double high) {
        Bounds {
            if (!Double.isFinite(low) ||
                !Double.isFinite(high) ||
                low >= high) {
                throw new IllegalArgumentException(
                    "Invalid optimization bounds.");
            }
        }
    }

    record Observation(List<Double> point, double value) {
        Observation {
            point = List.copyOf(point);

            if (point.isEmpty() || !Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Invalid observation.");
            }
        }
    }

    record Prediction(double mean, double standardDeviation) {}

    interface Objective {
        double evaluate(List<Double> point);
    }

    static final class InvalidOptimizationStateException
            extends RuntimeException {
        InvalidOptimizationStateException(String message) {
            super(message);
        }
    }

    static final class GaussianProcess {
        private final double lengthScale;
        private final double signalVariance;
        private final double noiseVariance;

        private List<List<Double>> trainingPoints = List.of();
        private double[][] cholesky = new double[0][0];
        private double[] alpha = new double[0];

        GaussianProcess(
                double lengthScale,
                double signalVariance,
                double noiseVariance) {

            if (lengthScale <= 0 ||
                signalVariance <= 0 ||
                noiseVariance < 0) {
                throw new IllegalArgumentException(
                    "Invalid Gaussian Process parameters.");
            }

            this.lengthScale = lengthScale;
            this.signalVariance = signalVariance;
            this.noiseVariance = noiseVariance;
        }

        void fit(List<Observation> observations) {
            if (observations.isEmpty()) {
                throw new InvalidOptimizationStateException(
                    "The surrogate has no training data.");
            }

            trainingPoints = observations.stream()
                .map(Observation::point)
                .toList();

            int n = observations.size();
            double[][] matrix = new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] =
                        kernel(trainingPoints.get(i),
                               trainingPoints.get(j));

                    if (i == j) {
                        matrix[i][j] += noiseVariance + 1e-9;
                    }
                }
            }

            cholesky = cholesky(matrix);

            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = observations.get(i).value();
            }

            alpha = backward(
                cholesky,
                forward(cholesky, y));
        }

        Prediction predict(List<Double> point) {
            if (trainingPoints.isEmpty()) {
                throw new InvalidOptimizationStateException(
                    "The surrogate has not been fitted.");
            }

            double[] k = new double[trainingPoints.size()];

            for (int i = 0; i < k.length; i++) {
                k[i] = kernel(point, trainingPoints.get(i));
            }

            double mean = 0.0;

            for (int i = 0; i < k.length; i++) {
                mean += k[i] * alpha[i];
            }

            double[] solved = forward(cholesky, k);
            double variance = kernel(point, point);

            for (double value : solved) {
                variance -= value * value;
            }

            return new Prediction(
                mean,
                Math.sqrt(Math.max(0.0, variance)));
        }

        private double kernel(List<Double> a, List<Double> b) {
            double squaredDistance = 0.0;

            for (int i = 0; i < a.size(); i++) {
                double delta = a.get(i) - b.get(i);
                squaredDistance += delta * delta;
            }

            return signalVariance *
                Math.exp(
                    -0.5 * squaredDistance /
                    (lengthScale * lengthScale));
        }

        private static double[][] cholesky(double[][] matrix) {
            int n = matrix.length;
            double[][] lower = new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    double value = matrix[i][j];

                    for (int k = 0; k < j; k++) {
                        value -=
                            lower[i][k] * lower[j][k];
                    }

                    if (i == j) {
                        if (value <= 1e-12) {
                            throw new InvalidOptimizationStateException(
                                "Kernel matrix is not positive definite.");
                        }

                        lower[i][j] = Math.sqrt(value);
                    } else {
                        lower[i][j] =
                            value / lower[j][j];
                    }
                }
            }

            return lower;
        }

        private static double[] forward(
                double[][] lower,
                double[] vector) {

            double[] result =
                new double[vector.length];

            for (int i = 0; i < vector.length; i++) {
                double value = vector[i];

                for (int j = 0; j < i; j++) {
                    value -=
                        lower[i][j] * result[j];
                }

                result[i] =
                    value / lower[i][i];
            }

            return result;
        }

        private static double[] backward(
                double[][] lower,
                double[] vector) {

            double[] result =
                new double[vector.length];

            for (int i = vector.length - 1;
                 i >= 0;
                 i--) {

                double value = vector[i];

                for (int j = i + 1;
                     j < vector.length;
                     j++) {

                    value -=
                        lower[j][i] * result[j];
                }

                result[i] =
                    value / lower[i][i];
            }

            return result;
        }
    }

    static final class ExpectedImprovement {
        private static double normalPdf(double z) {
            return Math.exp(-0.5 * z * z) /
                Math.sqrt(2.0 * Math.PI);
        }

        private static double normalCdf(double z) {
            return 0.5 *
                (1.0 + erf(z / Math.sqrt(2.0)));
        }

        private static double erf(double x) {
            double sign = x < 0 ? -1.0 : 1.0;
            double value = Math.abs(x);

            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            double t = 1.0 / (1.0 + p * value);

            double polynomial =
                (((((a5 * t + a4) * t) + a3) * t + a2)
                    * t + a1) * t;

            return sign *
                (1.0 - polynomial *
                    Math.exp(-value * value));
        }

        static double score(
                Prediction prediction,
                double bestValue,
                double exploration) {

            if (prediction.standardDeviation() <= 1e-12) {
                return 0.0;
            }

            double improvement =
                prediction.mean() -
                bestValue -
                exploration;

            double z =
                improvement /
                prediction.standardDeviation();

            return improvement * normalCdf(z) +
                prediction.standardDeviation() *
                normalPdf(z);
        }
    }

    static final class OptimizationService {
        private final List<Bounds> bounds;
        private final Objective objective;
        private final Random random;
        private final GaussianProcess surrogate;
        private final List<Observation> observations =
            new ArrayList<>();

        OptimizationService(
                List<Bounds> bounds,
                Objective objective,
                long seed) {

            if (bounds.isEmpty()) {
                throw new IllegalArgumentException(
                    "Optimization requires at least one dimension.");
            }

            this.bounds = List.copyOf(bounds);
            this.objective = objective;
            this.random = new Random(seed);
            this.surrogate =
                new GaussianProcess(0.8, 1.0, 1e-5);
        }

        Observation optimize(
                int initialPoints,
                int iterations,
                int gridSize) {

            if (initialPoints < 1 ||
                iterations < 1 ||
                gridSize < 2) {
                throw new IllegalArgumentException(
                    "Invalid optimization configuration.");
            }

            for (int i = 0; i < initialPoints; i++) {
                observe(randomPoint());
            }

            for (int i = 0; i < iterations; i++) {
                observe(selectNextPoint(gridSize));
            }

            return observations.stream()
                .max(Comparator.comparingDouble(
                    Observation::value))
                .orElseThrow();
        }

        List<Observation> observations() {
            return List.copyOf(observations);
        }

        void exportCsv(Path path) throws IOException {
            try (BufferedWriter writer =
                     Files.newBufferedWriter(path)) {

                writer.write(
                    "iteration,x1,x2,objective");
                writer.newLine();

                for (int i = 0;
                     i < observations.size();
                     i++) {

                    Observation observation =
                        observations.get(i);

                    writer.write(
                        (i + 1) + "," +
                        observation.point().get(0) + "," +
                        observation.point().get(1) + "," +
                        observation.value());

                    writer.newLine();
                }
            }
        }

        private void observe(List<Double> point) {
            for (int i = 0; i < point.size(); i++) {
                Bounds bound = bounds.get(i);

                if (point.get(i) < bound.low() ||
                    point.get(i) > bound.high()) {
                    throw new InvalidOptimizationStateException(
                        "Candidate lies outside the search domain.");
                }
            }

            double value = objective.evaluate(point);

            if (!Double.isFinite(value)) {
                throw new InvalidOptimizationStateException(
                    "Objective produced a non-finite result.");
            }

            observations.add(
                new Observation(point, value));
        }

        private List<Double> randomPoint() {
            List<Double> point = new ArrayList<>();

            for (Bounds bound : bounds) {
                point.add(
                    bound.low() +
                    random.nextDouble() *
                    (bound.high() - bound.low()));
            }

            return List.copyOf(point);
        }

        private List<Double> selectNextPoint(int gridSize) {
            surrogate.fit(observations);

            double bestValue =
                observations.stream()
                    .mapToDouble(Observation::value)
                    .max()
                    .orElseThrow();

            Set<String> observed =
                new HashSet<>();

            for (Observation observation : observations) {
                observed.add(key(observation.point()));
            }

            List<List<Double>> candidates =
                generateGrid(gridSize);

            List<Double> selected = null;
            double bestScore =
                Double.NEGATIVE_INFINITY;

            for (List<Double> candidate : candidates) {
                if (observed.contains(key(candidate))) {
                    continue;
                }

                Prediction prediction =
                    surrogate.predict(candidate);

                double score =
                    ExpectedImprovement.score(
                        prediction,
                        bestValue,
                        0.01);

                if (score > bestScore) {
                    bestScore = score;
                    selected = candidate;
                }
            }

            if (selected == null) {
                throw new InvalidOptimizationStateException(
                    "No unexplored candidate is available.");
            }

            return selected;
        }

        private List<List<Double>> generateGrid(
                int gridSize) {

            List<List<Double>> result =
                new ArrayList<>();

            if (bounds.size() != 2) {
                throw new UnsupportedOperationException(
                    "This enterprise example uses two dimensions.");
            }

            for (int i = 0; i < gridSize; i++) {
                double x =
                    bounds.get(0).low() +
                    i * (bounds.get(0).high() -
                         bounds.get(0).low()) /
                    (gridSize - 1);

                for (int j = 0; j < gridSize; j++) {
                    double y =
                        bounds.get(1).low() +
                        j * (bounds.get(1).high() -
                             bounds.get(1).low()) /
                        (gridSize - 1);

                    result.add(List.of(x, y));
                }
            }

            return result;
        }

        private static String key(List<Double> point) {
            return point.stream()
                .map(value -> String.format(
                    java.util.Locale.ROOT,
                    "%.9f",
                    value))
                .reduce((a, b) -> a + "," + b)
                .orElse("");
        }
    }

    private static double objective(List<Double> point) {
        double x1 = point.get(0);
        double x2 = point.get(1);

        return
            1.8 * Math.exp(
                -((x1 - 1.5) * (x1 - 1.5) +
                  (x2 - 2.0) * (x2 - 2.0)) / 1.4)
            +
            1.25 * Math.exp(
                -((x1 + 2.0) * (x1 + 2.0) +
                  (x2 + 1.0) * (x2 + 1.0)) / 0.8)
            -
            0.08 * (x1 * x1 + x2 * x2)
            +
            0.015 * Math.sin(17.0 * x1 + 11.0 * x2);
    }

    public static void main(String[] args) {
        try {
            OptimizationService service =
                new OptimizationService(
                    List.of(
                        new Bounds(-4.0, 4.0),
                        new Bounds(-4.0, 4.0)),
                    BayesianOptimizationEnterprise::objective,
                    42L);

            Observation best =
                service.optimize(6, 18, 17);

            System.out.println(
                "Bayesian Optimization");
            System.out.println(
                "=====================");
            System.out.println(
                "Observations: " +
                service.observations().size());
            System.out.printf(
                "Best x: (%.6f, %.6f)%n",
                best.point().get(0),
                best.point().get(1));
            System.out.printf(
                "Best objective: %.6f%n",
                best.value());

            Path history =
                Path.of(
                    "bayesian_optimization_history.csv");

            service.exportCsv(history);

            System.out.println(
                "History written to: " + history);
        } catch (Exception exception) {
            System.err.println(
                "Optimization failed: " +
                exception.getMessage());
            System.exit(1);
        }
    }
}
