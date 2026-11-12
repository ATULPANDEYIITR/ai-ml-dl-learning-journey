import java.util.*;
import java.util.concurrent.*;
import java.util.function.DoubleUnaryOperator;

/*
 * Enterprise Activation Function Framework.
 *
 * Java 17 implementation:
 * - Typed activation definitions
 * - Immutable activation configurations
 * - Registry and policy validation
 * - Dense activation layers
 * - Softmax and gradient propagation
 * - Batch inference
 * - Classification loss
 * - Custom exceptions
 * - Numerical validation
 *
 * Compile: javac ActivationFunctions.java
 * Run: java ActivationFunctions
 */

public class ActivationFunctions {

    enum ActivationType {
        SIGMOID,
        TANH,
        RELU,
        LEAKY_RELU,
        GELU,
        SOFTMAX
    }

    static class ActivationException extends RuntimeException {
        public ActivationException(String message) {
            super(message);
        }
    }

    record ActivationConfig(
        ActivationType type,
        double alpha
    ) {
        ActivationConfig {
            Objects.requireNonNull(type, "Activation type required.");

            if (!Double.isFinite(alpha) || alpha < 0) {
                throw new ActivationException(
                    "Alpha must be finite and nonnegative."
                );
            }
        }

        static ActivationConfig of(ActivationType type) {
            return new ActivationConfig(type, 0.01);
        }
    }

    interface ActivationFunction {
        double forward(double x, double alpha);

        double derivative(double x, double alpha);
    }

    static final class NumericalValidation {

        private NumericalValidation() {}

        static double number(double value) {
            if (!Double.isFinite(value)) {
                throw new ActivationException(
                    "Input must be finite."
                );
            }

            return value;
        }

        static double[] vector(double[] values) {
            if (values == null || values.length == 0) {
                throw new ActivationException(
                    "Input vector cannot be empty."
                );
            }

            double[] copy = values.clone();

            for (double value : copy) {
                number(value);
            }

            return copy;
        }

        static void matchingDimensions(
            double[] a,
            double[] b
        ) {
            if (a.length != b.length) {
                throw new ActivationException(
                    "Vector dimensions do not match."
                );
            }
        }
    }

    static final class ActivationRegistry {

        private final Map<ActivationType, ActivationFunction>
            functions = new EnumMap<>(ActivationType.class);

        ActivationRegistry() {
            registerDefaults();
        }

        private void register(
            ActivationType type,
            ActivationFunction function
        ) {
            if (functions.putIfAbsent(type, function) != null) {
                throw new ActivationException(
                    "Duplicate activation registration: " + type
                );
            }
        }

        private void registerDefaults() {

            register(ActivationType.SIGMOID, new ActivationFunction() {

                @Override
                public double forward(double x, double alpha) {
                    NumericalValidation.number(x);

                    if (x >= 0) {
                        return 1.0 / (1.0 + Math.exp(-x));
                    }

                    double z = Math.exp(x);
                    return z / (1.0 + z);
                }

                @Override
                public double derivative(double x, double alpha) {
                    double y = forward(x, alpha);
                    return y * (1.0 - y);
                }
            });

            register(ActivationType.TANH, new ActivationFunction() {

                @Override
                public double forward(double x, double alpha) {
                    return Math.tanh(
                        NumericalValidation.number(x)
                    );
                }

                @Override
                public double derivative(double x, double alpha) {
                    double y = forward(x, alpha);
                    return 1.0 - y * y;
                }
            });

            register(ActivationType.RELU, new ActivationFunction() {

                @Override
                public double forward(double x, double alpha) {
                    return Math.max(
                        0.0,
                        NumericalValidation.number(x)
                    );
                }

                @Override
                public double derivative(double x, double alpha) {
                    return NumericalValidation.number(x) > 0
                        ? 1.0 : 0.0;
                }
            });

            register(
                ActivationType.LEAKY_RELU,
                new ActivationFunction() {

                    @Override
                    public double forward(double x, double alpha) {
                        NumericalValidation.number(x);

                        return x > 0 ? x : alpha * x;
                    }

                    @Override
                    public double derivative(double x, double alpha) {
                        NumericalValidation.number(x);

                        return x > 0 ? 1.0 : alpha;
                    }
                }
            );

            register(ActivationType.GELU, new ActivationFunction() {

                @Override
                public double forward(double x, double alpha) {
                    NumericalValidation.number(x);

                    double c = Math.sqrt(2.0 / Math.PI);

                    double inner = c * (
                        x + 0.044715 * Math.pow(x, 3)
                    );

                    return 0.5 * x * (1.0 + Math.tanh(inner));
                }

                @Override
                public double derivative(double x, double alpha) {
                    NumericalValidation.number(x);

                    double c = Math.sqrt(2.0 / Math.PI);

                    double inner = c * (
                        x + 0.044715 * Math.pow(x, 3)
                    );

                    double t = Math.tanh(inner);

                    double innerDerivative = c * (
                        1.0 + 3.0 * 0.044715 * x * x
                    );

                    return 0.5 * (1.0 + t)
                        + 0.5 * x * (1.0 - t * t)
                        * innerDerivative;
                }
            });
        }

        ActivationFunction get(ActivationType type) {
            ActivationFunction function = functions.get(type);

            if (function == null) {
                throw new ActivationException(
                    "Activation is not available: " + type
                );
            }

            return function;
        }
    }

    static final class Softmax {

        private Softmax() {}

        static double[] forward(double[] logits) {
            double[] values =
                NumericalValidation.vector(logits);

            double maximum = Arrays.stream(values)
                .max()
                .orElseThrow();

            double[] result = new double[values.length];

            double sum = 0;

            for (int i = 0; i < values.length; i++) {
                result[i] = Math.exp(values[i] - maximum);
                sum += result[i];
            }

            if (!Double.isFinite(sum) || sum <= 0) {
                throw new ActivationException(
                    "Softmax normalization failed."
                );
            }

            for (int i = 0; i < result.length; i++) {
                result[i] /= sum;
            }

            return result;
        }

        static double[][] jacobian(double[] probabilities) {
            double[] p =
                NumericalValidation.vector(probabilities);

            validateProbabilities(p);

            int n = p.length;
            double[][] result = new double[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] = p[i] *
                        ((i == j ? 1.0 : 0.0) - p[j]);
                }
            }

            return result;
        }

        static double[] jvp(
            double[] probabilities,
            double[] upstream
        ) {
            double[] p =
                NumericalValidation.vector(probabilities);

            double[] gradient =
                NumericalValidation.vector(upstream);

            NumericalValidation.matchingDimensions(p, gradient);
            validateProbabilities(p);

            double dot = 0;

            for (int i = 0; i < p.length; i++) {
                dot += p[i] * gradient[i];
            }

            double[] result = new double[p.length];

            for (int i = 0; i < p.length; i++) {
                result[i] = p[i] * (gradient[i] - dot);
            }

            return result;
        }

        private static void validateProbabilities(double[] p) {
            double sum = 0;

            for (double value : p) {
                if (value < 0) {
                    throw new ActivationException(
                        "Negative probability."
                    );
                }

                sum += value;
            }

            if (Math.abs(sum - 1.0) > 1e-8) {
                throw new ActivationException(
                    "Probabilities must sum to one."
                );
            }
        }
    }

    static final class DenseActivationLayer {

        private final ActivationConfig config;
        private final ActivationRegistry registry;

        DenseActivationLayer(
            ActivationConfig config,
            ActivationRegistry registry
        ) {
            this.config = Objects.requireNonNull(config);
            this.registry = Objects.requireNonNull(registry);
        }

        double[] forward(double[] inputs) {
            double[] values =
                NumericalValidation.vector(inputs);

            if (config.type() == ActivationType.SOFTMAX) {
                return Softmax.forward(values);
            }

            ActivationFunction function =
                registry.get(config.type());

            double[] output = new double[values.length];

            for (int i = 0; i < values.length; i++) {
                output[i] = function.forward(
                    values[i],
                    config.alpha()
                );
            }

            return output;
        }

        double[] backward(
            double[] inputs,
            double[] upstream
        ) {
            double[] values =
                NumericalValidation.vector(inputs);

            double[] gradients =
                NumericalValidation.vector(upstream);

            NumericalValidation.matchingDimensions(
                values,
                gradients
            );

            if (config.type() == ActivationType.SOFTMAX) {
                return Softmax.jvp(
                    Softmax.forward(values),
                    gradients
                );
            }

            ActivationFunction function =
                registry.get(config.type());

            double[] result = new double[values.length];

            for (int i = 0; i < values.length; i++) {
                result[i] = function.derivative(
                    values[i],
                    config.alpha()
                ) * gradients[i];
            }

            return result;
        }
    }

    static final class InferenceService {

        private final List<DenseActivationLayer> layers;

        InferenceService(List<DenseActivationLayer> layers) {
            if (layers == null || layers.isEmpty()) {
                throw new ActivationException(
                    "Inference requires at least one layer."
                );
            }

            this.layers = List.copyOf(layers);
        }

        double[] predict(double[] inputs) {
            double[] current =
                NumericalValidation.vector(inputs);

            for (DenseActivationLayer layer : layers) {
                current = layer.forward(current);
            }

            return current;
        }

        CompletableFuture<List<double[]>> predictBatch(
            List<double[]> batches,
            Executor executor
        ) {
            if (batches == null || batches.isEmpty()) {
                throw new ActivationException(
                    "Batch cannot be empty."
                );
            }

            List<CompletableFuture<double[]>> tasks =
                batches.stream()
                    .map(batch -> CompletableFuture.supplyAsync(
                        () -> predict(batch), executor
                    ))
                    .toList();

            CompletableFuture<Void> all = CompletableFuture.allOf(
                tasks.toArray(new CompletableFuture[0])
            );

            return all.thenApply(ignored ->
                tasks.stream()
                    .map(CompletableFuture::join)
                    .toList()
            );
        }
    }

    static double crossEntropy(
        double[] probabilities,
        int target
    ) {
        double[] p =
            NumericalValidation.vector(probabilities);

        if (target < 0 || target >= p.length) {
            throw new ActivationException(
                "Target class is outside the output range."
            );
        }

        for (double value : p) {
            if (value < 0 || value > 1) {
                throw new ActivationException(
                    "Invalid probability."
                );
            }
        }

        return -Math.log(Math.max(1e-15, p[target]));
    }

    static void printArray(String label, double[] values) {
        System.out.printf("%-22s", label);

        for (double value : values) {
            System.out.printf("%12.6f", value);
        }

        System.out.println();
    }

    static void runTests(ActivationRegistry registry) {

        ActivationFunction sigmoid =
            registry.get(ActivationType.SIGMOID);

        if (Math.abs(sigmoid.forward(0, 0.01) - 0.5) > 1e-10) {
            throw new AssertionError("Sigmoid test failed.");
        }

        if (registry.get(ActivationType.RELU)
                .forward(-3, 0.01) != 0) {
            throw new AssertionError("ReLU test failed.");
        }

        double[] p = Softmax.forward(
            new double[]{1000, 1001, 1002}
        );

        double sum = Arrays.stream(p).sum();

        if (Math.abs(sum - 1.0) > 1e-10) {
            throw new AssertionError("Softmax test failed.");
        }

        double[] gradient = Softmax.jvp(
            p,
            new double[]{1, 0, -1}
        );

        if (Math.abs(Arrays.stream(gradient).sum()) > 1e-10) {
            throw new AssertionError("Softmax JVP test failed.");
        }

        System.out.println("All numerical tests passed.");
    }

    public static void main(String[] args) {

        ActivationRegistry registry = new ActivationRegistry();

        runTests(registry);

        double[] inputs = {-3, -1, 0, 1, 3};

        System.out.println("\nACTIVATION FUNCTION COMPARISON");

        for (ActivationType type : List.of(
            ActivationType.SIGMOID,
            ActivationType.TANH,
            ActivationType.RELU,
            ActivationType.LEAKY_RELU,
            ActivationType.GELU
        )) {
            DenseActivationLayer layer =
                new DenseActivationLayer(
                    ActivationConfig.of(type),
                    registry
                );

            printArray(type.name(), layer.forward(inputs));
        }

        System.out.println("\nCLASSIFICATION");

        double[] logits = {2.0, 1.0, 0.1};
        double[] probabilities = Softmax.forward(logits);

        printArray("Softmax", probabilities);

        System.out.printf(
            "Cross-entropy: %.6f%n",
            crossEntropy(probabilities, 0)
        );

        System.out.println("\nBACKPROPAGATION");

        DenseActivationLayer reluLayer =
            new DenseActivationLayer(
                ActivationConfig.of(ActivationType.RELU),
                registry
            );

        printArray(
            "ReLU backward",
            reluLayer.backward(
                inputs,
                new double[]{1, 1, 1, 1, 1}
            )
        );

        printArray(
            "Softmax JVP",
            Softmax.jvp(
                probabilities,
                new double[]{1, 0, -1}
            )
        );

        System.out.println("\nENTERPRISE INFERENCE SERVICE");

        List<DenseActivationLayer> pipeline = List.of(
            new DenseActivationLayer(
                ActivationConfig.of(ActivationType.RELU),
                registry
            ),
            new DenseActivationLayer(
                ActivationConfig.of(ActivationType.SOFTMAX),
                registry
            )
        );

        InferenceService service =
            new InferenceService(pipeline);

        printArray(
            "Prediction",
            service.predict(new double[]{-2, 1, 3})
        );

        ExecutorService executor =
            Executors.newFixedThreadPool(3);

        try {
            List<double[]> batches = List.of(
                new double[]{1, 2, 3},
                new double[]{-1, 0, 1},
                new double[]{4, 2, 1}
            );

            List<double[]> results = service.predictBatch(
                batches,
                executor
            ).join();

            System.out.println("\nPARALLEL BATCH RESULTS");

            for (int i = 0; i < results.size(); i++) {
                printArray("Batch " + (i + 1), results.get(i));
            }
        } finally {
            executor.shutdown();
        }
    }
}
