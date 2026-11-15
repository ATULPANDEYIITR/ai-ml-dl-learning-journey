import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/*
 * Enterprise-oriented backpropagation and gradient governance.
 *
 * Components:
 * - Immutable training configuration.
 * - Scalar computational graph.
 * - Reverse-mode automatic differentiation.
 * - Gradient verification.
 * - Neural network training.
 * - Training lifecycle management.
 * - Validation and numerical failure detection.
 * - Immutable training reports.
 */

public class BackpropagationEnterprise {

    enum TrainingState {
        CREATED,
        VALIDATING,
        TRAINING,
        COMPLETED,
        FAILED
    }

    record TrainingConfig(
        int inputSize,
        int hiddenSize,
        int epochs,
        double learningRate,
        double gradientLimit
    ) {
        TrainingConfig {
            if (inputSize < 1 || hiddenSize < 1 || epochs < 1) {
                throw new IllegalArgumentException(
                    "Network dimensions and epochs must be positive."
                );
            }

            if (
                !Double.isFinite(learningRate) ||
                learningRate <= 0 ||
                !Double.isFinite(gradientLimit) ||
                gradientLimit <= 0
            ) {
                throw new IllegalArgumentException(
                    "Invalid optimization configuration."
                );
            }
        }
    }

    record Sample(double[] inputs, double target) {
        Sample {
            if (inputs == null || inputs.length == 0) {
                throw new IllegalArgumentException("Empty inputs.");
            }

            inputs = inputs.clone();

            for (double value : inputs) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Non-finite input."
                    );
                }
            }

            if (!Double.isFinite(target)) {
                throw new IllegalArgumentException(
                    "Non-finite target."
                );
            }
        }

        @Override
        public double[] inputs() {
            return inputs.clone();
        }
    }

    record TrainingReport(
        TrainingState state,
        int completedEpochs,
        double finalLoss,
        List<Double> checkpoints,
        String message
    ) {
        TrainingReport {
            checkpoints = List.copyOf(checkpoints);
        }
    }

    static final class GraphNode {
        final double value;
        final String operation;
        final List<GraphNode> parents;

        double gradient;

        GraphNode(double value) {
            this(value, "input", List.of());
        }

        GraphNode(
            double value,
            String operation,
            List<GraphNode> parents
        ) {
            if (!Double.isFinite(value)) {
                throw new ArithmeticException(
                    "Non-finite graph value."
                );
            }

            this.value = value;
            this.operation = operation;
            this.parents = List.copyOf(parents);
        }

        GraphNode add(GraphNode other) {
            GraphNode result = new GraphNode(
                value + other.value,
                "add",
                List.of(this, other)
            );

            return result;
        }

        GraphNode multiply(GraphNode other) {
            return new GraphNode(
                value * other.value,
                "multiply",
                List.of(this, other)
            );
        }

        GraphNode tanh() {
            return new GraphNode(
                Math.tanh(value),
                "tanh",
                List.of(this)
            );
        }

        GraphNode power(double exponent) {
            if (
                !Double.isFinite(exponent) ||
                (value < 0 && exponent != Math.rint(exponent)) ||
                (value == 0 && exponent < 1)
            ) {
                throw new ArithmeticException(
                    "Invalid power operation."
                );
            }

            return new GraphNode(
                Math.pow(value, exponent),
                "power:" + exponent,
                List.of(this)
            );
        }

        void backward() {
            List<GraphNode> topology = new ArrayList<>();
            Set<GraphNode> visited = new HashSet<>();

            buildTopology(this, visited, topology);

            for (GraphNode node : topology) {
                node.gradient = 0;
            }

            gradient = 1;

            Collections.reverse(topology);

            for (GraphNode node : topology) {
                node.propagate();
            }
        }

        private static void buildTopology(
            GraphNode node,
            Set<GraphNode> visited,
            List<GraphNode> topology
        ) {
            if (!visited.add(node)) return;

            for (GraphNode parent : node.parents) {
                buildTopology(parent, visited, topology);
            }

            topology.add(node);
        }

        private void propagate() {
            switch (operation) {
                case "input" -> {
                }

                case "add" -> {
                    parents.get(0).gradient += gradient;
                    parents.get(1).gradient += gradient;
                }

                case "multiply" -> {
                    GraphNode a = parents.get(0);
                    GraphNode b = parents.get(1);

                    a.gradient += b.value * gradient;
                    b.gradient += a.value * gradient;
                }

                case "tanh" -> {
                    GraphNode parent = parents.get(0);

                    parent.gradient +=
                        (1 - value * value) * gradient;
                }

                default -> {
                    if (operation.startsWith("power:")) {
                        double exponent = Double.parseDouble(
                            operation.substring(6)
                        );

                        GraphNode parent = parents.get(0);

                        parent.gradient +=
                            exponent *
                            Math.pow(parent.value, exponent - 1) *
                            gradient;
                    } else {
                        throw new IllegalStateException(
                            "Unknown graph operation: " + operation
                        );
                    }
                }
            }
        }
    }

    static final class Neuron {
        final double[] weights;
        double bias;

        Neuron(int inputs, Random random) {
            weights = new double[inputs];

            for (int i = 0; i < inputs; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }

        double forward(double[] inputs) {
            if (inputs.length != weights.length) {
                throw new IllegalArgumentException(
                    "Neuron input mismatch."
                );
            }

            double sum = bias;

            for (int i = 0; i < inputs.length; i++) {
                sum += weights[i] * inputs[i];
            }

            return Math.tanh(sum);
        }
    }

    static final class Model {
        final Neuron[] hidden;
        final Neuron output;

        Model(TrainingConfig config, long seed) {
            Random random = new Random(seed);

            hidden = new Neuron[config.hiddenSize()];

            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = new Neuron(
                    config.inputSize(),
                    random
                );
            }

            output = new Neuron(
                config.hiddenSize(),
                random
            );
        }

        double predict(double[] inputs) {
            if (inputs.length != hidden[0].weights.length) {
                throw new IllegalArgumentException(
                    "Model input dimensions do not match."
                );
            }

            double[] activations = new double[hidden.length];

            for (int i = 0; i < hidden.length; i++) {
                activations[i] = hidden[i].forward(inputs);
            }

            return output.forward(activations);
        }

        double trainBatch(
            List<Sample> dataset,
            double learningRate,
            double gradientLimit
        ) {
            int hiddenCount = hidden.length;
            int inputCount = hidden[0].weights.length;

            double[][] hiddenWeightGradients =
                new double[hiddenCount][inputCount];

            double[] hiddenBiasGradients =
                new double[hiddenCount];

            double[] outputWeightGradients =
                new double[hiddenCount];

            double outputBiasGradient = 0;
            double totalLoss = 0;

            for (Sample sample : dataset) {
                double[] inputs = sample.inputs();

                if (inputs.length != inputCount) {
                    throw new IllegalArgumentException(
                        "Inconsistent sample dimensions."
                    );
                }

                double[] hiddenValues = new double[hiddenCount];

                for (int i = 0; i < hiddenCount; i++) {
                    hiddenValues[i] = hidden[i].forward(inputs);
                }

                double prediction = output.forward(hiddenValues);

                double error = prediction - sample.target();

                totalLoss += error * error;

                double outputDelta =
                    2 * error / dataset.size();

                outputBiasGradient += outputDelta;

                for (int j = 0; j < hiddenCount; j++) {
                    outputWeightGradients[j] +=
                        outputDelta * hiddenValues[j];

                    double hiddenDelta =
                        outputDelta *
                        output.weights[j] *
                        (1 - hiddenValues[j] * hiddenValues[j]);

                    hiddenBiasGradients[j] += hiddenDelta;

                    for (int k = 0; k < inputCount; k++) {
                        hiddenWeightGradients[j][k] +=
                            hiddenDelta * inputs[k];
                    }
                }
            }

            // Clip gradients to reduce the risk of unstable updates.
            for (int j = 0; j < hiddenCount; j++) {
                hiddenBiasGradients[j] = clip(
                    hiddenBiasGradients[j], gradientLimit
                );

                for (int k = 0; k < inputCount; k++) {
                    hiddenWeightGradients[j][k] = clip(
                        hiddenWeightGradients[j][k],
                        gradientLimit
                    );
                }

                outputWeightGradients[j] = clip(
                    outputWeightGradients[j],
                    gradientLimit
                );
            }

            outputBiasGradient = clip(
                outputBiasGradient,
                gradientLimit
            );

            // Compute candidate parameters first.
            double[][] candidateWeights =
                new double[hiddenCount][inputCount];

            double[] candidateBiases = new double[hiddenCount];

            for (int j = 0; j < hiddenCount; j++) {
                candidateBiases[j] =
                    hidden[j].bias -
                    learningRate * hiddenBiasGradients[j];

                for (int k = 0; k < inputCount; k++) {
                    candidateWeights[j][k] =
                        hidden[j].weights[k] -
                        learningRate * hiddenWeightGradients[j][k];
                }
            }

            double[] candidateOutputWeights =
                new double[hiddenCount];

            for (int j = 0; j < hiddenCount; j++) {
                candidateOutputWeights[j] =
                    output.weights[j] -
                    learningRate * outputWeightGradients[j];
            }

            double candidateOutputBias =
                output.bias - learningRate * outputBiasGradient;

            validateFinite(candidateOutputBias);

            for (double value : candidateOutputWeights) {
                validateFinite(value);
            }

            for (int j = 0; j < hiddenCount; j++) {
                validateFinite(candidateBiases[j]);

                for (double value : candidateWeights[j]) {
                    validateFinite(value);
                }
            }

            // Commit the validated update.
            for (int j = 0; j < hiddenCount; j++) {
                hidden[j].bias = candidateBiases[j];

                System.arraycopy(
                    candidateWeights[j],
                    0,
                    hidden[j].weights,
                    0,
                    inputCount
                );
            }

            System.arraycopy(
                candidateOutputWeights,
                0,
                output.weights,
                0,
                hiddenCount
            );

            output.bias = candidateOutputBias;

            return totalLoss / dataset.size();
        }

        private static double clip(double value, double limit) {
            return Math.max(-limit, Math.min(limit, value));
        }

        private static void validateFinite(double value) {
            if (!Double.isFinite(value)) {
                throw new ArithmeticException(
                    "Invalid parameter update."
                );
            }
        }
    }

    interface TrainingListener {
        void onEpoch(int epoch, double loss);
    }

    static final class TrainingService {
        private TrainingState state = TrainingState.CREATED;

        TrainingReport execute(
            Model model,
            TrainingConfig config,
            List<Sample> dataset,
            TrainingListener listener
        ) {
            List<Double> checkpoints = new ArrayList<>();

            int completedEpochs = 0;
            double loss = Double.NaN;

            try {
                transition(TrainingState.VALIDATING);

                validateDataset(config, dataset);

                transition(TrainingState.TRAINING);

                for (int epoch = 1; epoch <= config.epochs(); epoch++) {
                    loss = model.trainBatch(
                        dataset,
                        config.learningRate(),
                        config.gradientLimit()
                    );

                    if (!Double.isFinite(loss)) {
                        throw new ArithmeticException(
                            "Training produced non-finite loss."
                        );
                    }

                    completedEpochs = epoch;

                    if (epoch % 500 == 0 || epoch == 1) {
                        checkpoints.add(loss);

                        if (listener != null) {
                            listener.onEpoch(epoch, loss);
                        }
                    }
                }

                transition(TrainingState.COMPLETED);

                return new TrainingReport(
                    state,
                    completedEpochs,
                    loss,
                    checkpoints,
                    "Training completed successfully."
                );

            } catch (RuntimeException exception) {
                state = TrainingState.FAILED;

                return new TrainingReport(
                    state,
                    completedEpochs,
                    loss,
                    checkpoints,
                    exception.getMessage()
                );
            }
        }

        private void transition(TrainingState next) {
            boolean valid =
                (state == TrainingState.CREATED &&
                    next == TrainingState.VALIDATING) ||
                (state == TrainingState.VALIDATING &&
                    next == TrainingState.TRAINING) ||
                (state == TrainingState.TRAINING &&
                    next == TrainingState.COMPLETED);

            if (!valid) {
                throw new IllegalStateException(
                    "Invalid training state transition."
                );
            }

            state = next;
        }

        private void validateDataset(
            TrainingConfig config,
            List<Sample> dataset
        ) {
            if (dataset == null || dataset.isEmpty()) {
                throw new IllegalArgumentException(
                    "Training dataset is empty."
                );
            }

            for (Sample sample : dataset) {
                if (sample.inputs().length != config.inputSize()) {
                    throw new IllegalArgumentException(
                        "Training dimensions do not match configuration."
                    );
                }
            }
        }
    }

    static void verifyChainRule() {
        GraphNode x = new GraphNode(2);
        GraphNode three = new GraphNode(3);
        GraphNode two = new GraphNode(2);

        GraphNode result = x
            .multiply(x)
            .add(x.multiply(three))
            .add(two);

        result.backward();

        System.out.println("Chain rule result: " + result.value);
        System.out.println("Gradient: " + x.gradient);

        if (Math.abs(x.gradient - 7.0) > 1e-9) {
            throw new IllegalStateException(
                "Chain rule verification failed."
            );
        }
    }

    static void verifyNumericalGradient() {
        double x = 1.5;
        double epsilon = 1e-6;

        double analytical = 2 * x * Math.exp(x * x);

        double numerical = (
            Math.exp(Math.pow(x + epsilon, 2)) -
            Math.exp(Math.pow(x - epsilon, 2))
        ) / (2 * epsilon);

        System.out.println("Analytical: " + analytical);
        System.out.println("Numerical:  " + numerical);

        if (Math.abs(analytical - numerical) > 1e-3) {
            throw new IllegalStateException(
                "Numerical gradient verification failed."
            );
        }
    }

    public static void main(String[] args) {
        verifyChainRule();
        verifyNumericalGradient();

        TrainingConfig config = new TrainingConfig(
            2,
            4,
            3000,
            0.08,
            5.0
        );

        List<Sample> dataset = List.of(
            new Sample(new double[]{0, 0}, 0),
            new Sample(new double[]{0, 1}, 1),
            new Sample(new double[]{1, 0}, 1),
            new Sample(new double[]{1, 1}, 0)
        );

        Model model = new Model(config, 42);

        TrainingService service = new TrainingService();

        TrainingReport report = service.execute(
            model,
            config,
            dataset,
            (epoch, loss) -> System.out.printf(
                "Epoch %4d | Loss %.6f%n",
                epoch,
                loss
            )
        );

        System.out.println("\nTraining report");
        System.out.println("State: " + report.state());
        System.out.println("Completed epochs: " + report.completedEpochs());
        System.out.println("Final loss: " + report.finalLoss());
        System.out.println("Message: " + report.message());

        if (report.state() != TrainingState.COMPLETED) {
            throw new IllegalStateException(report.message());
        }

        System.out.println("\nInference results");

        for (Sample sample : dataset) {
            System.out.printf(
                "%.0f XOR %.0f = %.5f (target %.0f)%n",
                sample.inputs()[0],
                sample.inputs()[1],
                model.predict(sample.inputs()),
                sample.target()
            );
        }
    }
}
