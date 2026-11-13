import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Forward Propagation Enterprise Computation Engine.
 *
 * Features:
 * - Explicit neural network domain types
 * - Dense layers and immutable parameters
 * - Matrix-vector computation
 * - Activation policies
 * - Forward execution services
 * - Batch inference
 * - Execution tracing
 * - Validation and custom exceptions
 * - Model architecture reporting
 *
 * Compile: javac ForwardPropagation.java
 * Run: java ForwardPropagation
 */
public class ForwardPropagation {

    enum Activation {
        LINEAR,
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX
    }

    enum ExecutionStatus {
        ACCEPTED,
        COMPLETED,
        REJECTED,
        FAILED
    }

    static class NetworkException extends RuntimeException {
        NetworkException(String message) {
            super(message);
        }
    }

    static class DimensionException extends NetworkException {
        DimensionException(String message) {
            super(message);
        }
    }

    static class NumericalException extends NetworkException {
        NumericalException(String message) {
            super(message);
        }
    }

    record LayerTrace(
        String layerName,
        double[] weightedSums,
        double[] activatedOutputs
    ) {
        LayerTrace {
            weightedSums = weightedSums.clone();
            activatedOutputs = activatedOutputs.clone();
        }

        @Override
        public double[] weightedSums() {
            return weightedSums.clone();
        }

        @Override
        public double[] activatedOutputs() {
            return activatedOutputs.clone();
        }
    }

    record Prediction(
        String executionId,
        double[] output,
        List<LayerTrace> traces,
        ExecutionStatus status
    ) {
        Prediction {
            output = output.clone();
            traces = List.copyOf(traces);
            Objects.requireNonNull(status);
        }

        @Override
        public double[] output() {
            return output.clone();
        }
    }

    interface ActivationPolicy {
        double[] apply(double[] values);
    }

    static final class Activations {

        private Activations() {}

        static ActivationPolicy create(Activation activation) {
            Objects.requireNonNull(activation);

            return switch (activation) {
                case LINEAR -> values -> values.clone();

                case RELU -> values -> {
                    double[] result = new double[values.length];

                    for (int i = 0; i < values.length; i++) {
                        result[i] = Math.max(0.0, values[i]);
                    }

                    return result;
                };

                case SIGMOID -> values -> {
                    double[] result = new double[values.length];

                    for (int i = 0; i < values.length; i++) {
                        double x = values[i];

                        if (x >= 0) {
                            result[i] = 1.0 / (1.0 + Math.exp(-x));
                        } else {
                            double exponential = Math.exp(x);
                            result[i] = exponential / (1.0 + exponential);
                        }
                    }

                    return result;
                };

                case TANH -> values -> {
                    double[] result = new double[values.length];

                    for (int i = 0; i < values.length; i++) {
                        result[i] = Math.tanh(values[i]);
                    }

                    return result;
                };

                case SOFTMAX -> values -> {
                    double maximum = Double.NEGATIVE_INFINITY;

                    for (double value : values) {
                        maximum = Math.max(maximum, value);
                    }

                    double[] result = new double[values.length];
                    double total = 0.0;

                    for (int i = 0; i < values.length; i++) {
                        result[i] = Math.exp(values[i] - maximum);
                        total += result[i];
                    }

                    if (!Double.isFinite(total) || total <= 0.0) {
                        throw new NumericalException(
                            "Invalid softmax denominator."
                        );
                    }

                    for (int i = 0; i < result.length; i++) {
                        result[i] /= total;
                    }

                    return result;
                };
            };
        }
    }

    static final class DenseLayer {

        private final String name;
        private final double[][] weights;
        private final double[] biases;
        private final Activation activation;
        private final ActivationPolicy activationPolicy;

        private final int inputSize;
        private final int outputSize;

        DenseLayer(
            String name,
            double[][] weights,
            double[] biases,
            Activation activation
        ) {
            this.name = Objects.requireNonNull(name);
            Objects.requireNonNull(weights);
            Objects.requireNonNull(biases);

            if (weights.length == 0) {
                throw new IllegalArgumentException(
                    "Weight matrix cannot be empty."
                );
            }

            this.outputSize = weights.length;
            this.inputSize = weights[0].length;

            if (inputSize == 0) {
                throw new DimensionException(
                    "Weight rows cannot be empty."
                );
            }

            if (biases.length != outputSize) {
                throw new DimensionException(
                    "Bias count must equal output neuron count."
                );
            }

            this.weights = new double[outputSize][inputSize];

            for (int i = 0; i < outputSize; i++) {
                if (weights[i] == null ||
                    weights[i].length != inputSize) {
                    throw new DimensionException(
                        "Weight matrix must be rectangular."
                    );
                }

                for (int j = 0; j < inputSize; j++) {
                    validateFinite(weights[i][j], "Weight");
                    this.weights[i][j] = weights[i][j];
                }
            }

            this.biases = biases.clone();

            for (double bias : this.biases) {
                validateFinite(bias, "Bias");
            }

            this.activation = Objects.requireNonNull(activation);
            this.activationPolicy = Activations.create(activation);
        }

        private static void validateFinite(
            double value,
            String field
        ) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    field + " must be finite."
                );
            }
        }

        LayerTrace forward(double[] input) {
            validateInput(input);

            if (input.length != inputSize) {
                throw new DimensionException(
                    name + " expects " + inputSize + " inputs."
                );
            }

            double[] weightedSums = new double[outputSize];

            for (int neuron = 0; neuron < outputSize; neuron++) {
                double sum = biases[neuron];

                for (int feature = 0; feature < inputSize; feature++) {
                    sum += weights[neuron][feature] * input[feature];
                }

                if (!Double.isFinite(sum)) {
                    throw new NumericalException(
                        "Invalid weighted sum in " + name
                    );
                }

                weightedSums[neuron] = sum;
            }

            double[] outputs = activationPolicy.apply(weightedSums);

            for (double value : outputs) {
                validateFinite(value, "Activation output");
            }

            return new LayerTrace(
                name,
                weightedSums,
                outputs
            );
        }

        int inputSize() {
            return inputSize;
        }

        int outputSize() {
            return outputSize;
        }

        int parameterCount() {
            return inputSize * outputSize + outputSize;
        }

        String name() {
            return name;
        }

        Activation activation() {
            return activation;
        }
    }

    static final class NetworkDefinition {

        private final List<DenseLayer> layers;

        NetworkDefinition(List<DenseLayer> layers) {
            if (layers == null || layers.isEmpty()) {
                throw new IllegalArgumentException(
                    "Network requires at least one layer."
                );
            }

            this.layers = List.copyOf(layers);

            for (int i = 1; i < this.layers.size(); i++) {
                DenseLayer previous = this.layers.get(i - 1);
                DenseLayer current = this.layers.get(i);

                if (previous.outputSize() != current.inputSize()) {
                    throw new DimensionException(
                        "Layer dimensions are incompatible at position "
                        + (i + 1)
                    );
                }
            }
        }

        List<DenseLayer> layers() {
            return layers;
        }

        int inputSize() {
            return layers.getFirst().inputSize();
        }

        int outputSize() {
            return layers.getLast().outputSize();
        }

        int parameterCount() {
            return layers.stream()
                .mapToInt(DenseLayer::parameterCount)
                .sum();
        }
    }

    static final class InferenceService {

        private final NetworkDefinition definition;
        private final Consumer<Prediction> completionListener;

        InferenceService(
            NetworkDefinition definition,
            Consumer<Prediction> completionListener
        ) {
            this.definition = Objects.requireNonNull(definition);
            this.completionListener = Objects.requireNonNull(
                completionListener
            );
        }

        Prediction execute(
            String executionId,
            double[] input
        ) {
            Objects.requireNonNull(executionId);
            validateInput(input);

            if (input.length != definition.inputSize()) {
                throw new DimensionException(
                    "Input feature count does not match network."
                );
            }

            double[] current = input.clone();

            List<LayerTrace> traces = new ArrayList<>();

            for (DenseLayer layer : definition.layers()) {
                LayerTrace trace = layer.forward(current);

                traces.add(trace);
                current = trace.activatedOutputs();
            }

            Prediction prediction = new Prediction(
                executionId,
                current,
                traces,
                ExecutionStatus.COMPLETED
            );

            completionListener.accept(prediction);

            return prediction;
        }

        List<Prediction> executeBatch(
            List<double[]> samples
        ) {
            if (samples == null || samples.isEmpty()) {
                throw new IllegalArgumentException(
                    "Batch must contain samples."
                );
            }

            List<Prediction> results = new ArrayList<>();

            for (int i = 0; i < samples.size(); i++) {
                results.add(
                    execute("BATCH-" + (i + 1), samples.get(i))
                );
            }

            return List.copyOf(results);
        }
    }

    static void validateInput(double[] input) {
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException(
                "Input cannot be empty."
            );
        }

        for (double value : input) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Input contains a non-finite value."
                );
            }
        }
    }

    static NetworkDefinition createNetwork() {
        DenseLayer hidden = new DenseLayer(
            "Hidden Layer",
            new double[][] {
                {0.2, 0.4, -0.3},
                {0.5, -0.1, 0.2},
                {-0.2, 0.3, 0.6},
                {0.1, 0.2, 0.4}
            },
            new double[] {0.1, -0.1, 0.05, 0.0},
            Activation.RELU
        );

        DenseLayer output = new DenseLayer(
            "Output Layer",
            new double[][] {
                {0.3, -0.2, 0.4, 0.1},
                {-0.1, 0.5, 0.2, -0.3}
            },
            new double[] {0.05, -0.05},
            Activation.SOFTMAX
        );

        return new NetworkDefinition(
            List.of(hidden, output)
        );
    }

    static void printArray(String label, double[] values) {
        System.out.print(label + ": [");

        for (int i = 0; i < values.length; i++) {
            System.out.printf("%.6f", values[i]);

            if (i + 1 < values.length) {
                System.out.print(", ");
            }
        }

        System.out.println("]");
    }

    static void runTests(InferenceService service) {
        Prediction result = service.execute(
            "TEST-001",
            new double[] {0.2, 0.5, 0.8}
        );

        double sum = 0.0;

        for (double value : result.output()) {
            sum += value;
        }

        if (Math.abs(sum - 1.0) > 1e-10) {
            throw new NetworkException("Softmax test failed.");
        }

        if (result.traces().size() != 2) {
            throw new NetworkException("Layer execution test failed.");
        }

        boolean dimensionRejected = false;

        try {
            service.execute("INVALID-001", new double[] {1.0});
        } catch (DimensionException exception) {
            dimensionRejected = true;
        }

        if (!dimensionRejected) {
            throw new NetworkException(
                "Dimension validation test failed."
            );
        }

        boolean nonFiniteRejected = false;

        try {
            service.execute(
                "INVALID-002",
                new double[] {
                    Double.NaN,
                    1.0,
                    2.0
                }
            );
        } catch (IllegalArgumentException exception) {
            nonFiniteRejected = true;
        }

        if (!nonFiniteRejected) {
            throw new NetworkException(
                "Numerical input validation failed."
            );
        }

        List<Prediction> batch = service.executeBatch(
            List.of(
                new double[] {0.2, 0.5, 0.8},
                new double[] {0.7, 0.1, 0.4}
            )
        );

        if (batch.size() != 2) {
            throw new NetworkException("Batch test failed.");
        }

        System.out.println("All Java 17 tests passed.");
    }

    public static void main(String[] args) {

        NetworkDefinition definition = createNetwork();

        System.out.println("ENTERPRISE FORWARD PROPAGATION ENGINE");
        System.out.println("--------------------------------------");

        System.out.println(
            "Input features: " + definition.inputSize()
        );

        System.out.println(
            "Output features: " + definition.outputSize()
        );

        System.out.println(
            "Trainable parameters: " + definition.parameterCount()
        );

        for (int i = 0; i < definition.layers().size(); i++) {
            DenseLayer layer = definition.layers().get(i);

            System.out.printf(
                "Layer %d: %s (%d -> %d), %s%n",
                i + 1,
                layer.name(),
                layer.inputSize(),
                layer.outputSize(),
                layer.activation()
            );
        }

        InferenceService service = new InferenceService(
            definition,
            prediction -> System.out.println(
                "Execution completed: " + prediction.executionId()
            )
        );

        double[] input = {0.2, 0.5, 0.8};

        Prediction prediction = service.execute(
            "INFERENCE-001",
            input
        );

        System.out.println("\nFORWARD PROPAGATION TRACE");

        printArray("Input", input);

        for (LayerTrace trace : prediction.traces()) {
            System.out.println("\n" + trace.layerName());

            printArray(
                "Weighted sums",
                trace.weightedSums()
            );

            printArray(
                "Activated outputs",
                trace.activatedOutputs()
            );
        }

        System.out.println("\nFINAL PREDICTION");

        printArray("Output", prediction.output());

        System.out.println("\nBATCH INFERENCE");

        List<Prediction> batch = service.executeBatch(
            List.of(
                new double[] {0.2, 0.5, 0.8},
                new double[] {0.7, 0.1, 0.4},
                new double[] {0.9, 0.3, 0.2}
            )
        );

        for (Prediction item : batch) {
            printArray(
                item.executionId(),
                item.output()
            );
        }

        runTests(service);
    }
}
