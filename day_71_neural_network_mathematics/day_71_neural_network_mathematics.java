import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/*
 * Neural Network Mathematics
 *
 * Enterprise-oriented implementation covering:
 * neurons, weighted sums, biases, activation functions,
 * dense layers, forward propagation, backpropagation,
 * numerical differentiation, training, and validation.
 *
 * Java 17
 * No third-party dependencies.
 */

public class NeuralNetworkMath {

    // ========================================================
    // 1. Domain Exceptions
    // ========================================================

    static class NetworkException extends RuntimeException {
        NetworkException(String message) {
            super(message);
        }
    }

    // ========================================================
    // 2. Activation Functions
    // ========================================================

    enum Activation {
        LINEAR {
            @Override
            double apply(double x) {
                return x;
            }

            @Override
            double derivative(double x) {
                return 1.0;
            }
        },

        SIGMOID {
            @Override
            double apply(double x) {
                if (x >= 0) {
                    return 1.0 / (1.0 + Math.exp(-Math.min(x, 709)));
                }

                double exp = Math.exp(Math.max(x, -709));
                return exp / (1.0 + exp);
            }

            @Override
            double derivative(double x) {
                double output = apply(x);
                return output * (1.0 - output);
            }
        },

        TANH {
            @Override
            double apply(double x) {
                return Math.tanh(x);
            }

            @Override
            double derivative(double x) {
                double output = Math.tanh(x);
                return 1.0 - output * output;
            }
        },

        RELU {
            @Override
            double apply(double x) {
                return Math.max(0.0, x);
            }

            @Override
            double derivative(double x) {
                return x > 0 ? 1.0 : 0.0;
            }
        },

        LEAKY_RELU {
            @Override
            double apply(double x) {
                return x > 0 ? x : 0.01 * x;
            }

            @Override
            double derivative(double x) {
                return x > 0 ? 1.0 : 0.01;
            }
        };

        abstract double apply(double x);

        abstract double derivative(double x);
    }

    // ========================================================
    // 3. Immutable Result Types
    // ========================================================

    record NeuronOutput(
        double weightedSum,
        double activationOutput
    ) {}

    record Prediction(
        List<Double> outputs,
        long sequenceNumber
    ) {
        Prediction {
            outputs = List.copyOf(outputs);
        }
    }

    record TrainingSample(
        List<Double> inputs,
        List<Double> targets
    ) {
        TrainingSample {
            inputs = List.copyOf(inputs);
            targets = List.copyOf(targets);

            validateVector(inputs);
            validateVector(targets);
        }
    }

    record TrainingReport(
        int epochsCompleted,
        double finalLoss,
        List<Double> lossHistory
    ) {
        TrainingReport {
            lossHistory = List.copyOf(lossHistory);
        }
    }

    // ========================================================
    // 4. Validation Service
    // ========================================================

    static void validateNumber(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new NetworkException(
                name + " must be finite."
            );
        }
    }

    static void validateVector(List<Double> values) {
        if (values == null || values.isEmpty()) {
            throw new NetworkException(
                "Vector cannot be null or empty."
            );
        }

        for (Double value : values) {
            if (value == null || !Double.isFinite(value)) {
                throw new NetworkException(
                    "Vector contains an invalid number."
                );
            }
        }
    }

    static void validateDimension(
        List<Double> values,
        int expected
    ) {
        validateVector(values);

        if (values.size() != expected) {
            throw new NetworkException(
                "Expected " + expected +
                " values but received " + values.size()
            );
        }
    }

    // ========================================================
    // 5. Mathematical Operations
    // ========================================================

    static double weightedSum(
        List<Double> inputs,
        List<Double> weights,
        double bias
    ) {
        validateVector(inputs);
        validateDimension(weights, inputs.size());
        validateNumber(bias, "Bias");

        double result = bias;

        for (int i = 0; i < inputs.size(); i++) {
            result += inputs.get(i) * weights.get(i);
        }

        validateNumber(result, "Weighted sum");

        return result;
    }

    static double meanSquaredError(
        List<Double> predictions,
        List<Double> targets
    ) {
        validateDimension(targets, predictions.size());

        double total = 0.0;

        for (int i = 0; i < predictions.size(); i++) {
            double difference =
                predictions.get(i) - targets.get(i);

            total += difference * difference;
        }

        return total / predictions.size();
    }

    // ========================================================
    // 6. Neuron Entity
    // ========================================================

    static final class Neuron {

        private final String id;
        private final Activation activation;

        private List<Double> weights;
        private double bias;

        private long parameterVersion;

        Neuron(
            String id,
            List<Double> weights,
            double bias,
            Activation activation
        ) {
            if (id == null || id.isBlank()) {
                throw new NetworkException(
                    "Neuron identifier is required."
                );
            }

            validateVector(weights);
            validateNumber(bias, "Bias");

            this.id = id;
            this.weights = new ArrayList<>(weights);
            this.bias = bias;
            this.activation = Objects.requireNonNull(activation);
        }

        NeuronOutput forward(List<Double> inputs) {
            validateDimension(inputs, weights.size());

            double z = weightedSum(inputs, weights, bias);

            return new NeuronOutput(
                z,
                activation.apply(z)
            );
        }

        void updateParameters(
            List<Double> newWeights,
            double newBias
        ) {
            validateDimension(newWeights, weights.size());
            validateNumber(newBias, "New bias");

            weights = new ArrayList<>(newWeights);
            bias = newBias;

            parameterVersion++;
        }

        List<Double> weights() {
            return Collections.unmodifiableList(weights);
        }

        double bias() {
            return bias;
        }

        Activation activation() {
            return activation;
        }

        String id() {
            return id;
        }

        long parameterVersion() {
            return parameterVersion;
        }
    }

    // ========================================================
    // 7. Dense Layer
    // ========================================================

    static final class DenseLayer {

        private final List<Neuron> neurons;
        private final int inputSize;

        DenseLayer(List<Neuron> neurons) {
            if (neurons == null || neurons.isEmpty()) {
                throw new NetworkException(
                    "Dense layer cannot be empty."
                );
            }

            this.neurons = List.copyOf(neurons);
            this.inputSize = neurons.get(0).weights().size();

            for (Neuron neuron : neurons) {
                if (neuron.weights().size() != inputSize) {
                    throw new NetworkException(
                        "Inconsistent neuron dimensions."
                    );
                }
            }
        }

        List<Double> forward(List<Double> inputs) {
            validateDimension(inputs, inputSize);

            return neurons.stream()
                .map(neuron -> neuron.forward(inputs).activationOutput())
                .toList();
        }

        int inputSize() {
            return inputSize;
        }

        int outputSize() {
            return neurons.size();
        }

        List<Neuron> neurons() {
            return neurons;
        }
    }

    // ========================================================
    // 8. Training Policy
    // ========================================================

    record TrainingPolicy(
        int maximumEpochs,
        double learningRate,
        double convergenceThreshold,
        int reportingInterval
    ) {
        TrainingPolicy {
            if (maximumEpochs <= 0) {
                throw new NetworkException(
                    "Maximum epochs must be positive."
                );
            }

            validateNumber(learningRate, "Learning rate");
            validateNumber(convergenceThreshold, "Threshold");

            if (
                learningRate <= 0 ||
                convergenceThreshold < 0 ||
                reportingInterval <= 0
            ) {
                throw new NetworkException(
                    "Invalid training policy."
                );
            }
        }
    }

    // ========================================================
    // 9. Neural Network Service
    // ========================================================

    static final class NeuralNetwork {

        private final List<DenseLayer> layers;
        private final int inputSize;
        private final int outputSize;

        private long inferenceCount;

        private List<List<Double>> lastActivations =
            new ArrayList<>();

        private List<List<Double>> lastWeightedSums =
            new ArrayList<>();

        NeuralNetwork(List<DenseLayer> layers) {
            if (layers == null || layers.isEmpty()) {
                throw new NetworkException(
                    "Network requires at least one layer."
                );
            }

            this.layers = List.copyOf(layers);

            for (int i = 1; i < layers.size(); i++) {
                if (
                    layers.get(i - 1).outputSize()
                    != layers.get(i).inputSize()
                ) {
                    throw new NetworkException(
                        "Adjacent layers have incompatible dimensions."
                    );
                }
            }

            inputSize = layers.get(0).inputSize();
            outputSize = layers.get(layers.size() - 1).outputSize();
        }

        // ----------------------------------------------------
        // Forward Propagation
        // ----------------------------------------------------

        List<Double> forward(List<Double> inputs) {
            validateDimension(inputs, inputSize);

            List<Double> current = new ArrayList<>(inputs);

            lastActivations = new ArrayList<>();
            lastWeightedSums = new ArrayList<>();

            lastActivations.add(new ArrayList<>(current));

            for (DenseLayer layer : layers) {
                List<Double> next = new ArrayList<>();
                List<Double> sums = new ArrayList<>();

                for (Neuron neuron : layer.neurons()) {
                    NeuronOutput result = neuron.forward(current);

                    sums.add(result.weightedSum());
                    next.add(result.activationOutput());
                }

                lastWeightedSums.add(sums);
                lastActivations.add(new ArrayList<>(next));

                current = next;
            }

            inferenceCount++;

            return List.copyOf(current);
        }

        Prediction predict(List<Double> inputs) {
            return new Prediction(
                forward(inputs),
                inferenceCount
            );
        }

        // ----------------------------------------------------
        // Backpropagation
        // ----------------------------------------------------

        double trainSample(
            TrainingSample sample,
            double learningRate
        ) {
            validateNumber(learningRate, "Learning rate");

            if (learningRate <= 0) {
                throw new NetworkException(
                    "Learning rate must be positive."
                );
            }

            List<Double> predictions = forward(sample.inputs());

            validateDimension(sample.targets(), outputSize);

            int layerCount = layers.size();

            List<List<Double>> deltas = new ArrayList<>();

            for (int i = 0; i < layerCount; i++) {
                deltas.add(new ArrayList<>());
            }

            int last = layerCount - 1;

            DenseLayer outputLayer = layers.get(last);

            for (int i = 0; i < outputSize; i++) {
                Neuron neuron = outputLayer.neurons().get(i);

                double prediction = predictions.get(i);
                double target = sample.targets().get(i);

                double lossGradient =
                    2.0 * (prediction - target);

                double derivative = neuron.activation().derivative(
                    lastWeightedSums.get(last).get(i)
                );

                deltas.get(last).add(
                    lossGradient * derivative
                );
            }

            // Calculate hidden-layer gradients using original weights.
            for (int layer = layerCount - 2;
                 layer >= 0;
                 layer--) {

                DenseLayer currentLayer = layers.get(layer);
                DenseLayer nextLayer = layers.get(layer + 1);

                for (int neuronIndex = 0;
                     neuronIndex < currentLayer.outputSize();
                     neuronIndex++) {

                    double propagated = 0.0;

                    for (int next = 0;
                         next < nextLayer.outputSize();
                         next++) {

                        propagated +=
                            nextLayer.neurons()
                                .get(next)
                                .weights()
                                .get(neuronIndex)
                            * deltas.get(layer + 1).get(next);
                    }

                    double derivative =
                        currentLayer.neurons()
                            .get(neuronIndex)
                            .activation()
                            .derivative(
                                lastWeightedSums.get(layer)
                                    .get(neuronIndex)
                            );

                    deltas.get(layer).add(
                        propagated * derivative
                    );
                }
            }

            // Apply parameter updates after gradient computation.
            for (int layer = 0; layer < layerCount; layer++) {
                DenseLayer denseLayer = layers.get(layer);

                List<Double> previous =
                    lastActivations.get(layer);

                for (int neuronIndex = 0;
                     neuronIndex < denseLayer.outputSize();
                     neuronIndex++) {

                    Neuron neuron =
                        denseLayer.neurons().get(neuronIndex);

                    List<Double> updatedWeights =
                        new ArrayList<>(neuron.weights());

                    double delta =
                        deltas.get(layer).get(neuronIndex);

                    for (int weight = 0;
                         weight < updatedWeights.size();
                         weight++) {

                        updatedWeights.set(
                            weight,
                            updatedWeights.get(weight)
                            - learningRate * delta * previous.get(weight)
                        );
                    }

                    neuron.updateParameters(
                        updatedWeights,
                        neuron.bias() - learningRate * delta
                    );
                }
            }

            return meanSquaredError(
                predictions,
                sample.targets()
            );
        }

        // ----------------------------------------------------
        // Training Workflow
        // ----------------------------------------------------

        TrainingReport train(
            List<TrainingSample> dataset,
            TrainingPolicy policy
        ) {
            if (dataset == null || dataset.isEmpty()) {
                throw new NetworkException(
                    "Training dataset cannot be empty."
                );
            }

            for (TrainingSample sample : dataset) {
                validateDimension(sample.inputs(), inputSize);
                validateDimension(sample.targets(), outputSize);
            }

            List<Double> history = new ArrayList<>();
            double loss = Double.POSITIVE_INFINITY;

            for (
                int epoch = 0;
                epoch < policy.maximumEpochs();
                epoch++
            ) {
                for (TrainingSample sample : dataset) {
                    trainSample(sample, policy.learningRate());
                }

                double total = 0.0;

                for (TrainingSample sample : dataset) {
                    total += meanSquaredError(
                        forward(sample.inputs()),
                        sample.targets()
                    );
                }

                loss = total / dataset.size();
                history.add(loss);

                if (
                    epoch % policy.reportingInterval() == 0
                ) {
                    System.out.printf(
                        "Epoch %d | Loss %.8f%n",
                        epoch,
                        loss
                    );
                }

                if (loss < policy.convergenceThreshold()) {
                    return new TrainingReport(
                        epoch + 1,
                        loss,
                        history
                    );
                }
            }

            return new TrainingReport(
                policy.maximumEpochs(),
                loss,
                history
            );
        }

        long inferenceCount() {
            return inferenceCount;
        }
    }

    // ========================================================
    // 10. Numerical Differentiation
    // ========================================================

    static double numericalDerivative(
        DoubleUnaryOperator function,
        double x,
        double epsilon
    ) {
        validateNumber(x, "Input");
        validateNumber(epsilon, "Epsilon");

        if (epsilon <= 0) {
            throw new NetworkException(
                "Epsilon must be positive."
            );
        }

        return (
            function.applyAsDouble(x + epsilon)
            - function.applyAsDouble(x - epsilon)
        ) / (2.0 * epsilon);
    }

    // ========================================================
    // 11. Demonstration
    // ========================================================

    static void demonstrateNeuron() {
        System.out.println("\n--- Individual Neuron ---");

        Neuron neuron = new Neuron(
            "neuron-001",
            List.of(0.5, -0.3, 0.8),
            0.2,
            Activation.SIGMOID
        );

        NeuronOutput output = neuron.forward(
            List.of(1.0, 2.0, 0.5)
        );

        System.out.printf(
            "Weighted sum: %.6f%n",
            output.weightedSum()
        );

        System.out.printf(
            "Activation output: %.6f%n",
            output.activationOutput()
        );
    }

    static void demonstrateActivations() {
        System.out.println("\n--- Activation Functions ---");

        for (Activation activation : Activation.values()) {
            double x = 1.5;

            System.out.printf(
                "%-12s output=%.6f derivative=%.6f%n",
                activation,
                activation.apply(x),
                activation.derivative(x)
            );
        }
    }

    static void demonstrateXOR() {
        System.out.println("\n--- XOR Training ---");

        List<DenseLayer> layers = List.of(
            new DenseLayer(List.of(
                new Neuron(
                    "hidden-1",
                    List.of(0.5, -0.4),
                    0.1,
                    Activation.TANH
                ),
                new Neuron(
                    "hidden-2",
                    List.of(-0.3, 0.7),
                    -0.2,
                    Activation.TANH
                ),
                new Neuron(
                    "hidden-3",
                    List.of(0.2, 0.6),
                    0.0,
                    Activation.TANH
                ),
                new Neuron(
                    "hidden-4",
                    List.of(-0.5, -0.2),
                    0.1,
                    Activation.TANH
                )
            )),

            new DenseLayer(List.of(
                new Neuron(
                    "output-1",
                    List.of(0.8, -0.6, 0.3, -0.2),
                    0.15,
                    Activation.SIGMOID
                )
            ))
        );

        NeuralNetwork network = new NeuralNetwork(layers);

        List<TrainingSample> dataset = List.of(
            new TrainingSample(List.of(0.0, 0.0), List.of(0.0)),
            new TrainingSample(List.of(0.0, 1.0), List.of(1.0)),
            new TrainingSample(List.of(1.0, 0.0), List.of(1.0)),
            new TrainingSample(List.of(1.0, 1.0), List.of(0.0))
        );

        TrainingReport report = network.train(
            dataset,
            new TrainingPolicy(
                15000,
                0.5,
                0.001,
                1000
            )
        );

        System.out.printf(
            "Final loss: %.8f%n",
            report.finalLoss()
        );

        for (TrainingSample sample : dataset) {
            Prediction prediction = network.predict(sample.inputs());

            System.out.printf(
                "%s -> %.6f | Class: %d%n",
                sample.inputs(),
                prediction.outputs().get(0),
                prediction.outputs().get(0) >= 0.5 ? 1 : 0
            );
        }

        System.out.println(
            "Total inferences: " + network.inferenceCount()
        );
    }

    static void demonstrateDerivative() {
        System.out.println("\n--- Derivative Verification ---");

        double x = 0.7;

        double analytical =
            Activation.SIGMOID.derivative(x);

        double numerical = numericalDerivative(
            Activation.SIGMOID::apply,
            x,
            1e-6
        );

        System.out.printf(
            "Analytical: %.8f%nNumerical: %.8f%n",
            analytical,
            numerical
        );

        System.out.println(
            "Verification: " +
            (Math.abs(analytical - numerical) < 1e-5
                ? "PASSED"
                : "FAILED")
        );
    }

    static void demonstrateValidation() {
        System.out.println("\n--- Validation ---");

        try {
            weightedSum(
                List.of(1.0, 2.0),
                List.of(0.5),
                0.1
            );
        } catch (NetworkException exception) {
            System.out.println(
                "Expected error: " + exception.getMessage()
            );
        }

        try {
            validateNumber(
                Double.NaN,
                "Test input"
            );
        } catch (NetworkException exception) {
            System.out.println(
                "Expected error: " + exception.getMessage()
            );
        }
    }

    public static void main(String[] args) {
        try {
            demonstrateNeuron();
            demonstrateActivations();
            demonstrateDerivative();
            demonstrateValidation();
            demonstrateXOR();

        } catch (Exception exception) {
            System.err.println(
                "Network execution failed: " +
                exception.getMessage()
            );

            exception.printStackTrace();
            System.exit(1);
        }
    }
}
