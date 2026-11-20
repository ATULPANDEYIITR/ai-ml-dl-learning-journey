import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Deep Learning Training Enterprise Model.
 *
 * Demonstrates:
 * - Epoch and mini-batch management
 * - Learning-rate scheduling
 * - Training and validation metrics
 * - Checkpoint persistence and recovery
 * - Early stopping
 * - Explicit state transitions
 * - Domain validation and exception handling
 *
 * Compile: javac DeepLearningTraining.java
 * Run: java DeepLearningTraining
 */
public class DeepLearningTraining {

    enum TrainingState {
        INITIALIZED,
        RUNNING,
        COMPLETED,
        EARLY_STOPPED,
        FAILED
    }

    static class TrainingException extends RuntimeException {
        TrainingException(String message) {
            super(message);
        }

        TrainingException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    record TrainingConfig(
        int epochs,
        int batchSize,
        double learningRate,
        double minimumLearningRate,
        double decay,
        int patience,
        int checkpointInterval,
        double minDelta,
        long seed,
        Path checkpointDirectory
    ) {
        TrainingConfig {
            if (epochs < 1 || batchSize < 1) {
                throw new IllegalArgumentException(
                    "Epochs and batch size must be positive."
                );
            }

            if (!Double.isFinite(learningRate) ||
                !Double.isFinite(minimumLearningRate) ||
                learningRate <= 0 ||
                minimumLearningRate <= 0) {
                throw new IllegalArgumentException("Invalid learning rate.");
            }

            if (!Double.isFinite(decay) || decay <= 0 || decay > 1) {
                throw new IllegalArgumentException("Decay must be in (0, 1].");
            }

            if (patience < 1 || checkpointInterval < 1 ||
                !Double.isFinite(minDelta) || minDelta < 0) {
                throw new IllegalArgumentException("Invalid training policy.");
            }

            if (checkpointDirectory == null) {
                throw new IllegalArgumentException(
                    "Checkpoint directory is required."
                );
            }
        }
    }

    record Metrics(
        int epoch,
        int batches,
        double learningRate,
        double trainingLoss,
        double validationLoss
    ) {}

    record Sample(List<Double> features, double target) {
        Sample {
            if (features == null || features.isEmpty() ||
                !Double.isFinite(target)) {
                throw new IllegalArgumentException("Invalid sample.");
            }

            features = List.copyOf(features);

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Non-finite feature.");
                }
            }
        }
    }

    static final class Dataset {
        private final List<Sample> samples;
        private final int featureCount;

        Dataset(List<Sample> samples) {
            if (samples == null || samples.isEmpty()) {
                throw new IllegalArgumentException("Dataset cannot be empty.");
            }

            this.featureCount = samples.get(0).features().size();

            if (featureCount < 1) {
                throw new IllegalArgumentException("Invalid feature count.");
            }

            for (Sample sample : samples) {
                if (sample.features().size() != featureCount) {
                    throw new IllegalArgumentException(
                        "Inconsistent feature dimensions."
                    );
                }
            }

            this.samples = List.copyOf(samples);
        }

        int size() {
            return samples.size();
        }

        int featureCount() {
            return featureCount;
        }

        Sample get(int index) {
            return samples.get(index);
        }
    }

    static final class LinearModel {
        private double[] weights;
        private double bias;

        LinearModel(int featureCount, long seed) {
            if (featureCount < 1) {
                throw new IllegalArgumentException("Invalid feature count.");
            }

            Random random = new Random(seed);
            weights = new double[featureCount];

            for (int i = 0; i < featureCount; i++) {
                weights[i] = (random.nextDouble() - 0.5) * 0.2;
            }
        }

        private LinearModel(double[] weights, double bias) {
            this.weights = weights.clone();
            this.bias = bias;
        }

        double predict(List<Double> features) {
            if (features.size() != weights.length) {
                throw new TrainingException("Feature dimension mismatch.");
            }

            double result = bias;

            for (int i = 0; i < weights.length; i++) {
                result += weights[i] * features.get(i);
            }

            return result;
        }

        void update(
            double[] gradients,
            double biasGradient,
            double learningRate
        ) {
            if (gradients.length != weights.length) {
                throw new TrainingException("Gradient dimension mismatch.");
            }

            for (int i = 0; i < weights.length; i++) {
                weights[i] -= learningRate * gradients[i];
            }

            bias -= learningRate * biasGradient;

            validate();
        }

        private void validate() {
            for (double weight : weights) {
                if (!Double.isFinite(weight)) {
                    throw new TrainingException("Invalid model weights.");
                }
            }

            if (!Double.isFinite(bias)) {
                throw new TrainingException("Invalid model bias.");
            }
        }

        LinearModel copy() {
            return new LinearModel(weights, bias);
        }

        double[] weights() {
            return weights.clone();
        }

        double bias() {
            return bias;
        }

        void restore(double[] restoredWeights, double restoredBias) {
            if (restoredWeights.length != weights.length ||
                !Double.isFinite(restoredBias)) {
                throw new TrainingException("Invalid model restoration.");
            }

            weights = restoredWeights.clone();
            bias = restoredBias;
            validate();
        }
    }

    interface LearningRatePolicy {
        double rateAt(int epoch);
    }

    static final class ExponentialLearningRatePolicy
        implements LearningRatePolicy {

        private final double initialRate;
        private final double decay;
        private final double minimum;

        ExponentialLearningRatePolicy(
            double initialRate,
            double decay,
            double minimum
        ) {
            this.initialRate = initialRate;
            this.decay = decay;
            this.minimum = minimum;
        }

        @Override
        public double rateAt(int epoch) {
            return Math.max(
                minimum,
                initialRate * Math.pow(decay, epoch - 1)
            );
        }
    }

    record Checkpoint(
        int epoch,
        double learningRate,
        double bestLoss,
        int bestEpoch,
        double[] weights,
        double bias
    ) {
        Checkpoint {
            weights = weights.clone();
        }

        @Override
        public double[] weights() {
            return weights.clone();
        }
    }

    static final class CheckpointRepository {
        private final Path directory;

        CheckpointRepository(Path directory) {
            this.directory = directory.toAbsolutePath().normalize();
        }

        void initialize() throws IOException {
            Files.createDirectories(directory);
        }

        Path save(Checkpoint checkpoint) {
            Path destination = directory.resolve(
                "epoch_" + String.format("%05d", checkpoint.epoch()) + ".txt"
            );

            Path temporary = directory.resolve(
                "." + destination.getFileName() + ".tmp"
            );

            try {
                try (BufferedWriter writer = Files.newBufferedWriter(
                    temporary,
                    StandardCharsets.UTF_8
                )) {
                    writer.write("TRAINING_CHECKPOINT_V1");
                    writer.newLine();
                    writer.write(Integer.toString(checkpoint.epoch()));
                    writer.newLine();
                    writer.write(Double.toString(checkpoint.learningRate()));
                    writer.newLine();
                    writer.write(Double.toString(checkpoint.bestLoss()));
                    writer.newLine();
                    writer.write(Integer.toString(checkpoint.bestEpoch()));
                    writer.newLine();
                    writer.write(Integer.toString(checkpoint.weights().length));
                    writer.newLine();

                    for (double weight : checkpoint.weights()) {
                        writer.write(Double.toString(weight));
                        writer.newLine();
                    }

                    writer.write(Double.toString(checkpoint.bias()));
                    writer.newLine();
                }

                try {
                    return Files.move(
                        temporary,
                        destination,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE
                    );
                } catch (AtomicMoveNotSupportedException exception) {
                    return Files.move(
                        temporary,
                        destination,
                        StandardCopyOption.REPLACE_EXISTING
                    );
                }

            } catch (IOException exception) {
                try {
                    Files.deleteIfExists(temporary);
                } catch (IOException ignored) {
                    // Preserve the original persistence exception.
                }

                throw new TrainingException(
                    "Checkpoint persistence failed.",
                    exception
                );
            }
        }

        Checkpoint load(Path requestedPath, int expectedFeatures) {
            Path checkpointPath =
                requestedPath.toAbsolutePath().normalize();

            if (!checkpointPath.getParent().equals(directory)) {
                throw new TrainingException(
                    "Checkpoint is outside the configured directory."
                );
            }

            try (BufferedReader reader = Files.newBufferedReader(
                checkpointPath,
                StandardCharsets.UTF_8
            )) {
                if (!"TRAINING_CHECKPOINT_V1".equals(reader.readLine())) {
                    throw new TrainingException("Invalid checkpoint signature.");
                }

                int epoch = Integer.parseInt(reader.readLine());
                double learningRate = Double.parseDouble(reader.readLine());
                double bestLoss = Double.parseDouble(reader.readLine());
                int bestEpoch = Integer.parseInt(reader.readLine());
                int dimensions = Integer.parseInt(reader.readLine());

                if (epoch < 0 || bestEpoch < 0 ||
                    dimensions != expectedFeatures ||
                    !Double.isFinite(learningRate) ||
                    learningRate <= 0 ||
                    !Double.isFinite(bestLoss)) {
                    throw new TrainingException("Invalid checkpoint metadata.");
                }

                double[] weights = new double[dimensions];

                for (int i = 0; i < dimensions; i++) {
                    weights[i] = Double.parseDouble(reader.readLine());

                    if (!Double.isFinite(weights[i])) {
                        throw new TrainingException("Invalid checkpoint weight.");
                    }
                }

                double bias = Double.parseDouble(reader.readLine());

                if (!Double.isFinite(bias) || reader.readLine() != null) {
                    throw new TrainingException("Invalid checkpoint contents.");
                }

                return new Checkpoint(
                    epoch,
                    learningRate,
                    bestLoss,
                    bestEpoch,
                    weights,
                    bias
                );

            } catch (IOException | NumberFormatException exception) {
                throw new TrainingException(
                    "Checkpoint recovery failed.",
                    exception
                );
            }
        }
    }

    static final class TrainingService {
        private final TrainingConfig config;
        private final Dataset training;
        private final Dataset validation;
        private final LearningRatePolicy learningRatePolicy;
        private final CheckpointRepository checkpointRepository;

        private LinearModel model;
        private LinearModel bestModel;

        private TrainingState state = TrainingState.INITIALIZED;
        private final List<Metrics> history = new ArrayList<>();

        private double bestLoss = Double.POSITIVE_INFINITY;
        private int bestEpoch = 0;
        private int staleEpochs = 0;
        private int startEpoch = 1;

        TrainingService(
            TrainingConfig config,
            Dataset training,
            Dataset validation
        ) {
            this.config = config;
            this.training = training;
            this.validation = validation;

            if (training.featureCount() != validation.featureCount()) {
                throw new IllegalArgumentException(
                    "Training and validation dimensions differ."
                );
            }

            this.model = new LinearModel(
                training.featureCount(),
                config.seed()
            );

            this.bestModel = model.copy();

            this.learningRatePolicy =
                new ExponentialLearningRatePolicy(
                    config.learningRate(),
                    config.decay(),
                    config.minimumLearningRate()
                );

            this.checkpointRepository =
                new CheckpointRepository(config.checkpointDirectory());
        }

        private double calculateLoss(Dataset dataset) {
            double total = 0;

            for (int i = 0; i < dataset.size(); i++) {
                Sample sample = dataset.get(i);

                double error =
                    model.predict(sample.features()) - sample.target();

                total += error * error;
            }

            return total / dataset.size();
        }

        private void trainBatch(List<Integer> indices, double rate) {
            double[] gradients =
                new double[training.featureCount()];

            double biasGradient = 0;

            for (int index : indices) {
                Sample sample = training.get(index);

                double error =
                    model.predict(sample.features()) - sample.target();

                for (int j = 0; j < gradients.length; j++) {
                    gradients[j] +=
                        2 * error * sample.features().get(j) /
                        indices.size();
                }

                biasGradient += 2 * error / indices.size();
            }

            model.update(gradients, biasGradient, rate);
        }

        void restore(Path checkpointPath) {
            if (state != TrainingState.INITIALIZED) {
                throw new TrainingException("Cannot restore active training.");
            }

            Checkpoint checkpoint = checkpointRepository.load(
                checkpointPath,
                training.featureCount()
            );

            model.restore(checkpoint.weights(), checkpoint.bias());
            bestModel = model.copy();
            bestLoss = checkpoint.bestLoss();
            bestEpoch = checkpoint.bestEpoch();
            startEpoch = checkpoint.epoch() + 1;
        }

        List<Metrics> train() {
            if (state != TrainingState.INITIALIZED) {
                throw new TrainingException("Training already started.");
            }

            state = TrainingState.RUNNING;

            try {
                checkpointRepository.initialize();

                Random random = new Random(config.seed() + startEpoch);

                for (int epoch = startEpoch;
                     epoch <= config.epochs();
                     epoch++) {

                    List<Integer> indices = new ArrayList<>(
                        IntStream.range(0, training.size()).boxed().toList()
                    );

                    Collections.shuffle(indices, random);

                    double rate = learningRatePolicy.rateAt(epoch);
                    int batches = 0;

                    for (int start = 0;
                         start < indices.size();
                         start += config.batchSize()) {

                        int end = Math.min(
                            start + config.batchSize(),
                            indices.size()
                        );

                        trainBatch(indices.subList(start, end), rate);
                        batches++;
                    }

                    double trainLoss = calculateLoss(training);
                    double validationLoss = calculateLoss(validation);

                    if (validationLoss < bestLoss - config.minDelta()) {
                        bestLoss = validationLoss;
                        bestEpoch = epoch;
                        bestModel = model.copy();
                        staleEpochs = 0;
                    } else {
                        staleEpochs++;
                    }

                    Metrics metrics = new Metrics(
                        epoch,
                        batches,
                        rate,
                        trainLoss,
                        validationLoss
                    );

                    history.add(metrics);

                    System.out.printf(
                        "Epoch %03d | batches=%d | lr=%.6f | " +
                        "train=%.6f | validation=%.6f%n",
                        epoch,
                        batches,
                        rate,
                        trainLoss,
                        validationLoss
                    );

                    if (epoch % config.checkpointInterval() == 0 ||
                        epoch == bestEpoch) {

                        checkpointRepository.save(
                            new Checkpoint(
                                epoch,
                                rate,
                                bestLoss,
                                bestEpoch,
                                model.weights(),
                                model.bias()
                            )
                        );
                    }

                    if (staleEpochs >= config.patience()) {
                        state = TrainingState.EARLY_STOPPED;
                        break;
                    }
                }

                model = bestModel.copy();

                if (state == TrainingState.RUNNING) {
                    state = TrainingState.COMPLETED;
                }

                return List.copyOf(history);

            } catch (RuntimeException | IOException exception) {
                state = TrainingState.FAILED;
                throw new TrainingException(
                    "Training service failed.",
                    exception
                );
            }
        }

        TrainingState state() {
            return state;
        }

        double bestLoss() {
            return bestLoss;
        }

        int bestEpoch() {
            return bestEpoch;
        }

        LinearModel model() {
            return model.copy();
        }

        List<Metrics> history() {
            return List.copyOf(history);
        }
    }

    static List<Sample> generateSamples(int count, long seed) {
        Random random = new Random(seed);
        List<Sample> samples = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            double x1 = random.nextDouble() * 4 - 2;
            double x2 = random.nextDouble() * 4 - 2;
            double noise = random.nextGaussian() * 0.05;

            double target = 3 * x1 - 2 * x2 + 1.5 + noise;

            samples.add(new Sample(
                List.of(x1, x2),
                target
            ));
        }

        return samples;
    }

    static void runTests() {
        Sample sample = new Sample(List.of(1.0, 2.0), 3.0);

        if (sample.features().size() != 2) {
            throw new AssertionError("Sample validation failed.");
        }

        LinearModel model = new LinearModel(2, 42);

        double prediction = model.predict(List.of(0.0, 0.0));

        if (!Double.isFinite(prediction)) {
            throw new AssertionError("Prediction must be finite.");
        }

        TrainingConfig config = new TrainingConfig(
            5, 4, 0.05, 0.00001, 0.98, 3, 2,
            1e-8, 42, Path.of("java_test_checkpoints")
        );

        LearningRatePolicy scheduler =
            new ExponentialLearningRatePolicy(
                config.learningRate(),
                config.decay(),
                config.minimumLearningRate()
            );

        if (!(scheduler.rateAt(2) < scheduler.rateAt(1))) {
            throw new AssertionError("Learning-rate decay failed.");
        }

        System.out.println("All assertions passed.");
    }

    public static void main(String[] args) {
        try {
            if (args.length > 0 && args[0].equals("--test")) {
                runTests();
                return;
            }

            long seed = 42;

            List<Sample> samples = generateSamples(200, seed);

            int split = (int) (samples.size() * 0.8);

            Dataset training = new Dataset(
                samples.subList(0, split)
            );

            Dataset validation = new Dataset(
                samples.subList(split, samples.size())
            );

            TrainingConfig config = new TrainingConfig(
                40,
                16,
                0.05,
                0.00001,
                0.98,
                8,
                5,
                1e-8,
                seed,
                Path.of("java_checkpoints")
            );

            TrainingService service = new TrainingService(
                config,
                training,
                validation
            );

            service.train();

            System.out.println("\nTraining report");
            System.out.println("----------------");
            System.out.println("State: " + service.state());
            System.out.println("Best epoch: " + service.bestEpoch());
            System.out.printf(
                "Best validation loss: %.8f%n",
                service.bestLoss()
            );

            System.out.println(
                "Model weights: " +
                java.util.Arrays.toString(service.model().weights())
            );

            System.out.println(
                "Model bias: " + service.model().bias()
            );

        } catch (Exception exception) {
            System.err.println("Training failure: " + exception.getMessage());
            System.exit(1);
        }
    }
}
