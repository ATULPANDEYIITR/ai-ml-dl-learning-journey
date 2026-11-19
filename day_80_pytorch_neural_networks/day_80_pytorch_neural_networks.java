import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/*
 * Neural Networks with Java 17.
 *
 * Enterprise-oriented implementation of:
 * - Neural network modules
 * - Dense layers
 * - Activation functions
 * - Forward propagation
 * - Backpropagation
 * - Mini-batch training
 * - Gradient accumulation
 * - Validation
 * - Model inference
 * - Training policy and state management
 *
 * Dependencies: Java 17 standard library.
 *
 * Compile: javac NeuralNetworkDemo.java
 * Run: java NeuralNetworkDemo
 */

public class NeuralNetworkDemo {

    // -----------------------------------------------------
    // 1. Training configuration
    // -----------------------------------------------------

    record TrainingConfig(
        int inputSize,
        int hiddenSize,
        int outputSize,
        int epochs,
        int batchSize,
        double learningRate,
        long seed
    ) {
        TrainingConfig {
            if (
                inputSize <= 0 ||
                hiddenSize <= 0 ||
                outputSize <= 1 ||
                epochs <= 0 ||
                batchSize <= 0
            ) {
                throw new IllegalArgumentException(
                    "Invalid network or training dimensions."
                );
            }

            if (
                !Double.isFinite(learningRate) ||
                learningRate <= 0
            ) {
                throw new IllegalArgumentException(
                    "Learning rate must be positive."
                );
            }
        }
    }

    // -----------------------------------------------------
    // 2. Domain exceptions
    // -----------------------------------------------------

    static class TrainingException extends RuntimeException {
        TrainingException(String message) {
            super(message);
        }
    }

    // -----------------------------------------------------
    // 3. Dataset model
    // -----------------------------------------------------

    record Sample(double[] features, int label) {
        Sample {
            if (features == null || features.length == 0) {
                throw new IllegalArgumentException(
                    "Sample features cannot be empty."
                );
            }

            features = features.clone();

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Sample contains non-finite values."
                    );
                }
            }

            if (label < 0) {
                throw new IllegalArgumentException(
                    "Label cannot be negative."
                );
            }
        }

        @Override
        public double[] features() {
            return features.clone();
        }
    }

    // -----------------------------------------------------
    // 4. Module contract
    // -----------------------------------------------------

    interface Module {
        double[] forward(double[] input);

        double[] backward(double[] gradient);

        void update(double learningRate);

        void zeroGradients();

        int parameterCount();
    }

    // -----------------------------------------------------
    // 5. Dense layer
    // -----------------------------------------------------

    static final class LinearLayer implements Module {

        private final int inputSize;
        private final int outputSize;

        private final double[][] weights;
        private final double[] biases;

        private final double[][] weightGradients;
        private final double[] biasGradients;

        private double[] cachedInput;

        LinearLayer(
            int inputSize,
            int outputSize,
            Random random
        ) {
            if (inputSize <= 0 || outputSize <= 0) {
                throw new IllegalArgumentException(
                    "Layer dimensions must be positive."
                );
            }

            this.inputSize = inputSize;
            this.outputSize = outputSize;

            weights = new double[outputSize][inputSize];
            biases = new double[outputSize];

            weightGradients = new double[outputSize][inputSize];
            biasGradients = new double[outputSize];

            double scale = Math.sqrt(2.0 / inputSize);

            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = random.nextGaussian() * scale;
                }
            }
        }

        @Override
        public double[] forward(double[] input) {
            if (input.length != inputSize) {
                throw new IllegalArgumentException(
                    "Dense layer input dimension mismatch."
                );
            }

            cachedInput = input.clone();

            double[] output = new double[outputSize];

            for (int i = 0; i < outputSize; i++) {
                output[i] = biases[i];

                for (int j = 0; j < inputSize; j++) {
                    output[i] += weights[i][j] * input[j];
                }
            }

            return output;
        }

        @Override
        public double[] backward(double[] gradient) {
            if (cachedInput == null) {
                throw new IllegalStateException(
                    "Forward propagation must precede backpropagation."
                );
            }

            if (gradient.length != outputSize) {
                throw new IllegalArgumentException(
                    "Gradient dimension mismatch."
                );
            }

            double[] inputGradient = new double[inputSize];

            for (int i = 0; i < outputSize; i++) {
                if (!Double.isFinite(gradient[i])) {
                    throw new TrainingException(
                        "Non-finite gradient detected."
                    );
                }

                biasGradients[i] += gradient[i];

                for (int j = 0; j < inputSize; j++) {
                    weightGradients[i][j] +=
                        gradient[i] * cachedInput[j];

                    inputGradient[j] +=
                        weights[i][j] * gradient[i];
                }
            }

            return inputGradient;
        }

        @Override
        public void update(double learningRate) {
            for (int i = 0; i < outputSize; i++) {
                biases[i] -= learningRate * biasGradients[i];

                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] -=
                        learningRate * weightGradients[i][j];
                }
            }
        }

        @Override
        public void zeroGradients() {
            for (int i = 0; i < outputSize; i++) {
                biasGradients[i] = 0.0;

                for (int j = 0; j < inputSize; j++) {
                    weightGradients[i][j] = 0.0;
                }
            }
        }

        @Override
        public int parameterCount() {
            return outputSize * inputSize + outputSize;
        }
    }

    // -----------------------------------------------------
    // 6. Activation module
    // -----------------------------------------------------

    static final class ActivationLayer implements Module {

        private final DoubleUnaryOperator activation;
        private final DoubleUnaryOperator derivative;

        private double[] cachedInput;

        ActivationLayer(
            DoubleUnaryOperator activation,
            DoubleUnaryOperator derivative
        ) {
            this.activation = activation;
            this.derivative = derivative;
        }

        static ActivationLayer relu() {
            return new ActivationLayer(
                value -> Math.max(0.0, value),
                value -> value > 0.0 ? 1.0 : 0.0
            );
        }

        @Override
        public double[] forward(double[] input) {
            cachedInput = input.clone();

            double[] output = new double[input.length];

            for (int i = 0; i < input.length; i++) {
                output[i] = activation.applyAsDouble(input[i]);
            }

            return output;
        }

        @Override
        public double[] backward(double[] gradient) {
            if (cachedInput == null ||
                cachedInput.length != gradient.length) {
                throw new IllegalStateException(
                    "Invalid activation backward state."
                );
            }

            double[] result = new double[gradient.length];

            for (int i = 0; i < gradient.length; i++) {
                result[i] = gradient[i] *
                    derivative.applyAsDouble(cachedInput[i]);
            }

            return result;
        }

        @Override
        public void update(double learningRate) {}

        @Override
        public void zeroGradients() {}

        @Override
        public int parameterCount() {
            return 0;
        }
    }

    // -----------------------------------------------------
    // 7. Loss function
    // -----------------------------------------------------

    record LossResult(
        double loss,
        double[] gradient
    ) {}

    static final class CrossEntropyLoss {

        static double[] softmax(double[] logits) {
            if (logits.length == 0) {
                throw new IllegalArgumentException(
                    "Logits cannot be empty."
                );
            }

            double maximum = Double.NEGATIVE_INFINITY;

            for (double value : logits) {
                if (!Double.isFinite(value)) {
                    throw new TrainingException(
                        "Invalid network logits."
                    );
                }

                maximum = Math.max(maximum, value);
            }

            double[] probabilities = new double[logits.length];

            double sum = 0.0;

            for (int i = 0; i < logits.length; i++) {
                probabilities[i] = Math.exp(logits[i] - maximum);
                sum += probabilities[i];
            }

            for (int i = 0; i < probabilities.length; i++) {
                probabilities[i] /= sum;
            }

            return probabilities;
        }

        static LossResult calculate(
            double[] logits,
            int target
        ) {
            if (target < 0 || target >= logits.length) {
                throw new IllegalArgumentException(
                    "Target class is invalid."
                );
            }

            double[] probabilities = softmax(logits);

            double loss = -Math.log(
                Math.max(probabilities[target], Double.MIN_NORMAL)
            );

            double[] gradient = probabilities.clone();

            gradient[target] -= 1.0;

            return new LossResult(loss, gradient);
        }
    }

    // -----------------------------------------------------
    // 8. Sequential model
    // -----------------------------------------------------

    static final class NeuralNetwork {

        private final List<Module> modules = new ArrayList<>();

        NeuralNetwork(TrainingConfig config) {
            Random random = new Random(config.seed());

            modules.add(new LinearLayer(
                config.inputSize(),
                config.hiddenSize(),
                random
            ));

            modules.add(ActivationLayer.relu());

            modules.add(new LinearLayer(
                config.hiddenSize(),
                config.hiddenSize(),
                random
            ));

            modules.add(ActivationLayer.relu());

            modules.add(new LinearLayer(
                config.hiddenSize(),
                config.outputSize(),
                random
            ));
        }

        double[] forward(double[] input) {
            double[] output = input.clone();

            for (Module module : modules) {
                output = module.forward(output);
            }

            return output;
        }

        void backward(double[] gradient) {
            double[] current = gradient.clone();

            for (int i = modules.size() - 1; i >= 0; i--) {
                current = modules.get(i).backward(current);
            }
        }

        void zeroGradients() {
            modules.forEach(Module::zeroGradients);
        }

        void update(double learningRate) {
            modules.forEach(
                module -> module.update(learningRate)
            );
        }

        int parameterCount() {
            return modules.stream()
                .mapToInt(Module::parameterCount)
                .sum();
        }

        int predict(double[] input) {
            double[] probabilities =
                CrossEntropyLoss.softmax(forward(input));

            int best = 0;

            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > probabilities[best]) {
                    best = i;
                }
            }

            return best;
        }
    }

    // -----------------------------------------------------
    // 9. Training state
    // -----------------------------------------------------

    enum TrainingStatus {
        CREATED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    record EpochMetrics(
        int epoch,
        double trainingLoss,
        double trainingAccuracy,
        double validationLoss,
        double validationAccuracy
    ) {}

    // -----------------------------------------------------
    // 10. Evaluation service
    // -----------------------------------------------------

    static final class EvaluationService {

        static EpochMetrics evaluate(
            NeuralNetwork model,
            List<Sample> dataset,
            int epoch,
            double trainingLoss,
            double trainingAccuracy
        ) {
            if (dataset.isEmpty()) {
                throw new TrainingException(
                    "Validation dataset cannot be empty."
                );
            }

            double totalLoss = 0.0;
            int correct = 0;

            for (Sample sample : dataset) {
                double[] logits = model.forward(sample.features());

                totalLoss += CrossEntropyLoss.calculate(
                    logits,
                    sample.label()
                ).loss();

                if (model.predict(sample.features()) == sample.label()) {
                    correct++;
                }
            }

            return new EpochMetrics(
                epoch,
                trainingLoss,
                trainingAccuracy,
                totalLoss / dataset.size(),
                (double) correct / dataset.size()
            );
        }
    }

    // -----------------------------------------------------
    // 11. Training service
    // -----------------------------------------------------

    static final class TrainingService {

        private final TrainingConfig config;
        private final NeuralNetwork model;

        private TrainingStatus status = TrainingStatus.CREATED;

        TrainingService(
            TrainingConfig config,
            NeuralNetwork model
        ) {
            this.config = config;
            this.model = model;
        }

        TrainingStatus getStatus() {
            return status;
        }

        List<EpochMetrics> train(
            List<Sample> trainingData,
            List<Sample> validationData
        ) {
            if (status != TrainingStatus.CREATED) {
                throw new TrainingException(
                    "Training service cannot be started in its current state."
                );
            }

            if (trainingData.isEmpty() || validationData.isEmpty()) {
                throw new TrainingException(
                    "Training and validation datasets are required."
                );
            }

            for (Sample sample : trainingData) {
                validateSample(sample);
            }

            for (Sample sample : validationData) {
                validateSample(sample);
            }

            status = TrainingStatus.RUNNING;

            List<EpochMetrics> history = new ArrayList<>();

            try {
                Random random = new Random(config.seed() + 1);

                for (
                    int epoch = 1;
                    epoch <= config.epochs();
                    epoch++
                ) {
                    Collections.shuffle(trainingData, random);

                    double totalLoss = 0.0;
                    int correct = 0;

                    for (
                        int start = 0;
                        start < trainingData.size();
                        start += config.batchSize()
                    ) {
                        int end = Math.min(
                            start + config.batchSize(),
                            trainingData.size()
                        );

                        int batchCount = end - start;

                        model.zeroGradients();

                        for (int i = start; i < end; i++) {
                            Sample sample = trainingData.get(i);

                            double[] logits = model.forward(
                                sample.features()
                            );

                            LossResult result =
                                CrossEntropyLoss.calculate(
                                    logits,
                                    sample.label()
                                );

                            totalLoss += result.loss();

                            if (
                                model.predict(sample.features()) ==
                                sample.label()
                            ) {
                                correct++;
                            }

                            model.backward(result.gradient());
                        }

                        double averageLearningRate =
                            config.learningRate() / batchCount;

                        if (!Double.isFinite(averageLearningRate)) {
                            throw new TrainingException(
                                "Invalid optimizer update."
                            );
                        }

                        model.update(averageLearningRate);
                    }

                    double trainingLoss =
                        totalLoss / trainingData.size();

                    double trainingAccuracy =
                        (double) correct / trainingData.size();

                    EpochMetrics metrics =
                        EvaluationService.evaluate(
                            model,
                            validationData,
                            epoch,
                            trainingLoss,
                            trainingAccuracy
                        );

                    history.add(metrics);

                    if (epoch == 1 || epoch % 10 == 0) {
                        System.out.printf(
                            "Epoch %3d | Train Loss %.4f | " +
                            "Train Accuracy %.4f | " +
                            "Validation Loss %.4f | " +
                            "Validation Accuracy %.4f%n",
                            epoch,
                            metrics.trainingLoss(),
                            metrics.trainingAccuracy(),
                            metrics.validationLoss(),
                            metrics.validationAccuracy()
                        );
                    }
                }

                status = TrainingStatus.COMPLETED;

            } catch (RuntimeException exception) {
                status = TrainingStatus.FAILED;
                throw exception;
            }

            return List.copyOf(history);
        }

        private void validateSample(Sample sample) {
            if (sample.features().length != config.inputSize()) {
                throw new TrainingException(
                    "Dataset feature dimension mismatch."
                );
            }

            if (sample.label() >= config.outputSize()) {
                throw new TrainingException(
                    "Dataset label exceeds output classes."
                );
            }
        }
    }

    // -----------------------------------------------------
    // 12. Synthetic classification dataset
    // -----------------------------------------------------

    static List<Sample> generateDataset(
        int samplesPerClass,
        long seed
    ) {
        Random random = new Random(seed);

        List<Sample> samples = new ArrayList<>();

        for (int i = 0; i < samplesPerClass; i++) {
            samples.add(new Sample(
                new double[] {
                    -1.0 + random.nextGaussian() * 0.65,
                    -1.0 + random.nextGaussian() * 0.65
                },
                0
            ));

            samples.add(new Sample(
                new double[] {
                    1.0 + random.nextGaussian() * 0.65,
                    1.0 + random.nextGaussian() * 0.65
                },
                1
            ));
        }

        Collections.shuffle(samples, random);

        return samples;
    }

    // -----------------------------------------------------
    // 13. Main application
    // -----------------------------------------------------

    public static void main(String[] args) {

        try {
            TrainingConfig config = new TrainingConfig(
                2,
                32,
                2,
                100,
                16,
                0.01,
                42
            );

            List<Sample> dataset = generateDataset(
                500,
                config.seed()
            );

            int split = (int) (dataset.size() * 0.8);

            List<Sample> trainingData = new ArrayList<>(
                dataset.subList(0, split)
            );

            List<Sample> validationData = new ArrayList<>(
                dataset.subList(split, dataset.size())
            );

            NeuralNetwork model = new NeuralNetwork(config);

            System.out.println("Java 17 Neural Network");
            System.out.println(
                "Trainable parameters: " + model.parameterCount()
            );

            TrainingService service = new TrainingService(
                config,
                model
            );

            List<EpochMetrics> history = service.train(
                trainingData,
                validationData
            );

            System.out.println("\nTRAINING STATE");
            System.out.println(service.getStatus());

            EpochMetrics finalMetrics =
                history.get(history.size() - 1);

            System.out.println("\nFINAL VALIDATION");
            System.out.printf(
                "Loss: %.4f%nAccuracy: %.4f%n",
                finalMetrics.validationLoss(),
                finalMetrics.validationAccuracy()
            );

            System.out.println("\nINFERENCE");

            double[][] testInputs = {
                {-1.2, -0.8},
                {1.1, 0.9},
                {0.0, 0.0}
            };

            for (double[] input : testInputs) {
                System.out.printf(
                    "Input: [%.2f, %.2f] => Class: %d%n",
                    input[0],
                    input[1],
                    model.predict(input)
                );
            }

            System.out.println("\nExecution completed.");

        } catch (Exception exception) {
            System.err.println(
                "Neural network execution failed: " +
                exception.getMessage()
            );

            System.exit(1);
        }
    }
}
