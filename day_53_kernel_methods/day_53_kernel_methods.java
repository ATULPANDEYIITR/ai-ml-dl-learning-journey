import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/*
 * Kernel Methods enterprise-oriented model.
 *
 * The domain is a similarity service that evaluates feature vectors using
 * interchangeable kernel policies. The implementation explicitly models:
 * - kernel type and parameters,
 * - similarity evaluation,
 * - Gram matrices,
 * - positive-semidefinite validation,
 * - kernel similarity classification,
 * - kernel ridge regression,
 * - validation and failure states.
 *
 * Java 17 records provide immutable value-oriented domain types while
 * sealed interfaces constrain the supported kernel family.
 */

public class KernelMethodsEnterprise {

    public record FeatureVector(List<Double> values) {

        public FeatureVector {
            if (values == null || values.isEmpty()) {
                throw new ValidationException(
                    "feature vector must not be empty"
                );
            }

            values = List.copyOf(values);

            for (Double value : values) {
                if (
                    value == null ||
                    !Double.isFinite(value)
                ) {
                    throw new ValidationException(
                        "feature vector contains an invalid value"
                    );
                }
            }
        }

        public int dimension() {
            return values.size();
        }

        public double get(int index) {
            return values.get(index);
        }
    }

    public record ClassificationResult(
        int predictedLabel,
        double similarityScore
    ) {}

    public record RegressionResult(
        double prediction
    ) {}

    public enum KernelType {
        LINEAR,
        POLYNOMIAL,
        RBF
    }

    public static final class ValidationException
        extends IllegalArgumentException {

        public ValidationException(String message) {
            super(message);
        }
    }

    public sealed interface Kernel
        permits LinearKernel, PolynomialKernel, RbfKernel {

        KernelType type();

        double similarity(
            FeatureVector first,
            FeatureVector second
        );
    }

    public static final class LinearKernel
        implements Kernel {

        @Override
        public KernelType type() {
            return KernelType.LINEAR;
        }

        @Override
        public double similarity(
            FeatureVector first,
            FeatureVector second
        ) {
            checkDimension(first, second);

            double result = 0.0;

            for (int i = 0; i < first.dimension(); i++) {
                result +=
                    first.get(i) * second.get(i);
            }

            return result;
        }
    }

    public static final class PolynomialKernel
        implements Kernel {

        private final int degree;
        private final double gamma;
        private final double coefficient;

        public PolynomialKernel(
            int degree,
            double gamma,
            double coefficient
        ) {
            if (degree < 1) {
                throw new ValidationException(
                    "polynomial degree must be positive"
                );
            }

            if (
                !Double.isFinite(gamma) ||
                gamma <= 0.0
            ) {
                throw new ValidationException(
                    "polynomial gamma must be positive"
                );
            }

            if (
                !Double.isFinite(coefficient) ||
                coefficient < 0.0
            ) {
                throw new ValidationException(
                    "polynomial coefficient must be non-negative"
                );
            }

            this.degree = degree;
            this.gamma = gamma;
            this.coefficient = coefficient;
        }

        @Override
        public KernelType type() {
            return KernelType.POLYNOMIAL;
        }

        @Override
        public double similarity(
            FeatureVector first,
            FeatureVector second
        ) {
            checkDimension(first, second);

            double dot = 0.0;

            for (int i = 0; i < first.dimension(); i++) {
                dot +=
                    first.get(i) *
                    second.get(i);
            }

            return Math.pow(
                gamma * dot + coefficient,
                degree
            );
        }
    }

    public static final class RbfKernel
        implements Kernel {

        private final double gamma;

        public RbfKernel(double gamma) {
            if (
                !Double.isFinite(gamma) ||
                gamma <= 0.0
            ) {
                throw new ValidationException(
                    "RBF gamma must be positive"
                );
            }

            this.gamma = gamma;
        }

        @Override
        public KernelType type() {
            return KernelType.RBF;
        }

        @Override
        public double similarity(
            FeatureVector first,
            FeatureVector second
        ) {
            checkDimension(first, second);

            double distance = 0.0;

            for (int i = 0; i < first.dimension(); i++) {
                double difference =
                    first.get(i) - second.get(i);

                distance +=
                    difference * difference;
            }

            return Math.exp(-gamma * distance);
        }
    }

    public record TrainingSet(
        List<FeatureVector> samples,
        List<Integer> labels
    ) {

        public TrainingSet {
            if (
                samples == null ||
                labels == null ||
                samples.isEmpty()
            ) {
                throw new ValidationException(
                    "training set must not be empty"
                );
            }

            if (samples.size() != labels.size()) {
                throw new ValidationException(
                    "sample and label counts differ"
                );
            }

            samples = List.copyOf(samples);
            labels = List.copyOf(labels);

            int dimension =
                samples.getFirst().dimension();

            for (FeatureVector sample : samples) {
                if (sample.dimension() != dimension) {
                    throw new ValidationException(
                        "training dimensions differ"
                    );
                }
            }
        }
    }

    public static final class KernelSimilarityService {

        private final Kernel kernel;

        public KernelSimilarityService(Kernel kernel) {
            this.kernel = kernel;
        }

        public ClassificationResult classify(
            TrainingSet training,
            FeatureVector query
        ) {
            Map<Integer, Double> scores =
                new TreeMap<>();

            for (int i = 0;
                 i < training.samples().size();
                 i++) {

                int label =
                    training.labels().get(i);

                double similarity =
                    kernel.similarity(
                        training.samples().get(i),
                        query
                    );

                scores.merge(
                    label,
                    similarity,
                    Double::sum
                );
            }

            Map.Entry<Integer, Double> best =
                scores.entrySet()
                    .stream()
                    .max(
                        Map.Entry.comparingByValue()
                    )
                    .orElseThrow(
                        () -> new IllegalStateException(
                            "no classification score exists"
                        )
                    );

            return new ClassificationResult(
                best.getKey(),
                best.getValue()
            );
        }
    }

    public static final class GramMatrixService {

        public double[][] build(
            List<FeatureVector> samples,
            Kernel kernel
        ) {
            if (samples == null || samples.isEmpty()) {
                throw new ValidationException(
                    "samples must not be empty"
                );
            }

            int n = samples.size();
            double[][] matrix =
                new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    double value =
                        kernel.similarity(
                            samples.get(i),
                            samples.get(j)
                        );

                    matrix[i][j] = value;
                    matrix[j][i] = value;
                }
            }

            return matrix;
        }

        public boolean isPositiveSemidefinite(
            double[][] matrix
        ) {
            int n = matrix.length;
            double[][] lower =
                new double[n][n];

            for (int i = 0; i < n; i++) {
                if (matrix[i].length != n) {
                    throw new ValidationException(
                        "matrix must be square"
                    );
                }

                for (int j = 0; j <= i; j++) {
                    double value = matrix[i][j];

                    for (int k = 0; k < j; k++) {
                        value -=
                            lower[i][k] *
                            lower[j][k];
                    }

                    if (i == j) {
                        if (value < -1e-9) {
                            return false;
                        }

                        lower[i][j] =
                            Math.sqrt(
                                Math.max(0.0, value)
                            );
                    } else {
                        double pivot =
                            lower[j][j];

                        if (Math.abs(pivot) <= 1e-9) {
                            if (Math.abs(value) > 1e-9) {
                                return false;
                            }

                            lower[i][j] = 0.0;
                        } else {
                            lower[i][j] =
                                value / pivot;
                        }
                    }
                }
            }

            return true;
        }
    }

    public static final class KernelRidgeService {

        private final Kernel kernel;
        private final double regularization;

        private List<FeatureVector> samples =
            List.of();

        private double[] coefficients =
            new double[0];

        public KernelRidgeService(
            Kernel kernel,
            double regularization
        ) {
            if (
                !Double.isFinite(regularization) ||
                regularization <= 0.0
            ) {
                throw new ValidationException(
                    "regularization must be positive"
                );
            }

            this.kernel = kernel;
            this.regularization = regularization;
        }

        public void fit(
            List<FeatureVector> samples,
            List<Double> targets
        ) {
            if (
                samples == null ||
                targets == null ||
                samples.isEmpty()
            ) {
                throw new ValidationException(
                    "training data must not be empty"
                );
            }

            if (samples.size() != targets.size()) {
                throw new ValidationException(
                    "sample and target counts differ"
                );
            }

            this.samples = List.copyOf(samples);

            int n = samples.size();

            double[][] matrix =
                new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] =
                        kernel.similarity(
                            samples.get(i),
                            samples.get(j)
                        );
                }

                matrix[i][i] += regularization;
            }

            double[] values =
                new double[n];

            for (int i = 0; i < n; i++) {
                values[i] = targets.get(i);
            }

            coefficients =
                solveLinearSystem(matrix, values);
        }

        public RegressionResult predict(
            FeatureVector query
        ) {
            if (coefficients.length == 0) {
                throw new IllegalStateException(
                    "model has not been fitted"
                );
            }

            double prediction = 0.0;

            for (int i = 0;
                 i < samples.size();
                 i++) {

                prediction +=
                    coefficients[i] *
                    kernel.similarity(
                        samples.get(i),
                        query
                    );
            }

            return new RegressionResult(
                prediction
            );
        }
    }

    private static void checkDimension(
        FeatureVector first,
        FeatureVector second
    ) {
        if (
            first.dimension() !=
            second.dimension()
        ) {
            throw new ValidationException(
                "feature-vector dimensions differ"
            );
        }
    }

    private static double[] solveLinearSystem(
        double[][] matrix,
        double[] rhs
    ) {
        int n = matrix.length;

        if (n == 0 || rhs.length != n) {
            throw new ValidationException(
                "invalid linear system"
            );
        }

        double[][] augmented =
            new double[n][n + 1];

        for (int i = 0; i < n; i++) {
            if (matrix[i].length != n) {
                throw new ValidationException(
                    "linear-system matrix must be square"
                );
            }

            System.arraycopy(
                matrix[i],
                0,
                augmented[i],
                0,
                n
            );

            augmented[i][n] = rhs[i];
        }

        for (int column = 0;
             column < n;
             column++) {

            int pivot = column;

            for (int row = column + 1;
                 row < n;
                 row++) {

                if (
                    Math.abs(
                        augmented[row][column]
                    ) >
                    Math.abs(
                        augmented[pivot][column]
                    )
                ) {
                    pivot = row;
                }
            }

            if (
                Math.abs(
                    augmented[pivot][column]
                ) < 1e-12
            ) {
                throw new IllegalStateException(
                    "singular or unstable system"
                );
            }

            double[] temporary =
                augmented[column];

            augmented[column] =
                augmented[pivot];

            augmented[pivot] =
                temporary;

            for (int row = column + 1;
                 row < n;
                 row++) {

                double factor =
                    augmented[row][column] /
                    augmented[column][column];

                for (int k = column;
                     k <= n;
                     k++) {

                    augmented[row][k] -=
                        factor *
                        augmented[column][k];
                }
            }
        }

        double[] solution =
            new double[n];

        for (int row = n - 1;
             row >= 0;
             row--) {

            double remainder =
                augmented[row][n];

            for (int column = row + 1;
                 column < n;
                 column++) {

                remainder -=
                    augmented[row][column] *
                    solution[column];
            }

            solution[row] =
                remainder /
                augmented[row][row];
        }

        return solution;
    }

    private static FeatureVector vector(
        double... values
    ) {
        return new FeatureVector(
            Arrays.stream(values)
                .boxed()
                .toList()
        );
    }

    public static void main(String[] args) {
        try {
            List<FeatureVector> samples =
                List.of(
                    vector(-1, -1),
                    vector(-1, 1),
                    vector(1, -1),
                    vector(1, 1),
                    vector(-0.8, -1.1),
                    vector(-1.1, 0.8),
                    vector(0.9, -1.2),
                    vector(1.1, 1.0)
                );

            List<Integer> labels =
                List.of(0, 1, 1, 0, 0, 1, 1, 0);

            TrainingSet training =
                new TrainingSet(samples, labels);

            Kernel rbf =
                new RbfKernel(2.0);

            Kernel polynomial =
                new PolynomialKernel(
                    3,
                    0.5,
                    1.0
                );

            GramMatrixService gramService =
                new GramMatrixService();

            double[][] rbfGram =
                gramService.build(
                    samples,
                    rbf
                );

            System.out.println(
                "RBF Gram matrix PSD: " +
                gramService.isPositiveSemidefinite(
                    rbfGram
                )
            );

            double[][] polynomialGram =
                gramService.build(
                    samples,
                    polynomial
                );

            System.out.println(
                "Polynomial Gram matrix PSD: " +
                gramService.isPositiveSemidefinite(
                    polynomialGram
                )
            );

            KernelSimilarityService classifier =
                new KernelSimilarityService(rbf);

            for (
                FeatureVector query :
                List.of(
                    vector(-0.9, -0.9),
                    vector(-0.9, 0.9),
                    vector(0.9, -0.9),
                    vector(0.9, 0.9)
                )
            ) {
                ClassificationResult result =
                    classifier.classify(
                        training,
                        query
                    );

                System.out.printf(
                    "query=%s -> class=%d score=%.4f%n",
                    query.values(),
                    result.predictedLabel(),
                    result.similarityScore()
                );
            }

            KernelRidgeService regression =
                new KernelRidgeService(
                    new RbfKernel(0.7),
                    0.05
                );

            regression.fit(
                List.of(
                    vector(-2),
                    vector(-1),
                    vector(0),
                    vector(1),
                    vector(2)
                ),
                List.of(
                    4.0, 1.0, 0.0, 1.0, 4.0
                )
            );

            for (double x :
                List.of(-1.5, -0.5, 0.5, 1.5)) {

                RegressionResult result =
                    regression.predict(vector(x));

                System.out.printf(
                    "x=%.1f -> prediction=%.4f%n",
                    x,
                    result.prediction()
                );
            }

        } catch (RuntimeException error) {
            System.err.println(
                "Kernel service failure: " +
                error.getMessage()
            );
        }
    }
}
