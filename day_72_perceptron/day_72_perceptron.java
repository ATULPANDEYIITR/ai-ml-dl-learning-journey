import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/*
 * Perceptron Learning Rule and Linear Separability
 *
 * Java 17 enterprise-oriented implementation.
 *
 * Domain components:
 * - TrainingSample
 * - TrainingPolicy
 * - Reviewable epoch events
 * - PerceptronModel
 * - EvaluationReport
 * - ModelTrainingService
 * - FeatureScaler
 *
 * The model uses online updates and exposes immutable
 * snapshots of training state.
 */

public class Main {

    enum TrainingStatus {
        NOT_TRAINED,
        TRAINING,
        CONVERGED,
        MAXIMUM_EPOCHS_REACHED,
        FAILED
    }

    static class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    static class TrainingException extends RuntimeException {
        TrainingException(String message) {
            super(message);
        }
    }

    record TrainingSample(List<Double> features, int label) {

        TrainingSample {
            Objects.requireNonNull(features, "Features cannot be null.");

            if (features.isEmpty()) {
                throw new ValidationException("Features cannot be empty.");
            }

            features = List.copyOf(features);

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new ValidationException(
                        "Features must be finite."
                    );
                }
            }

            if (label != 0 && label != 1) {
                throw new ValidationException(
                    "Label must be 0 or 1."
                );
            }
        }
    }

    record TrainingPolicy(
        double learningRate,
        int maximumEpochs,
        boolean shuffle,
        long randomSeed
    ) {
        TrainingPolicy {
            if (!Double.isFinite(learningRate) || learningRate <= 0) {
                throw new ValidationException(
                    "Learning rate must be positive."
                );
            }

            if (maximumEpochs < 1) {
                throw new ValidationException(
                    "Maximum epochs must be positive."
                );
            }
        }

        static TrainingPolicy standard() {
            return new TrainingPolicy(0.1, 100, false, 42);
        }
    }

    record EpochSnapshot(
        int epoch,
        int mistakes,
        List<Double> weights,
        double bias
    ) {
        EpochSnapshot {
            weights = List.copyOf(weights);
        }
    }

    record EvaluationReport(
        int truePositive,
        int trueNegative,
        int falsePositive,
        int falseNegative,
        double accuracy
    ) {}

    record ModelSnapshot(
        List<Double> weights,
        double bias,
        TrainingStatus status,
        int epochsTrained
    ) {
        ModelSnapshot {
            weights = List.copyOf(weights);
        }
    }

    static final class PerceptronModel {

        private final TrainingPolicy policy;

        private double[] weights;
        private double bias;

        private TrainingStatus status =
            TrainingStatus.NOT_TRAINED;

        private int epochsTrained;

        private final List<EpochSnapshot> history =
            new ArrayList<>();

        private final List<Consumer<EpochSnapshot>> listeners =
            new CopyOnWriteArrayList<>();

        PerceptronModel(TrainingPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        private void validateSamples(List<TrainingSample> samples) {
            if (samples == null || samples.isEmpty()) {
                throw new ValidationException(
                    "Dataset must not be empty."
                );
            }

            int dimensions = samples.get(0).features().size();

            boolean hasZero = false;
            boolean hasOne = false;

            for (TrainingSample sample : samples) {
                if (sample.features().size() != dimensions) {
                    throw new ValidationException(
                        "Inconsistent feature dimensions."
                    );
                }

                hasZero |= sample.label() == 0;
                hasOne |= sample.label() == 1;
            }

            if (!hasZero || !hasOne) {
                throw new ValidationException(
                    "Both classes must be present."
                );
            }
        }

        double netInput(List<Double> features) {
            if (weights == null) {
                throw new TrainingException("Model is not trained.");
            }

            if (features.size() != weights.length) {
                throw new ValidationException(
                    "Feature dimension mismatch."
                );
            }

            double result = bias;

            for (int i = 0; i < weights.length; i++) {
                double value = features.get(i);

                if (!Double.isFinite(value)) {
                    throw new ValidationException(
                        "Feature must be finite."
                    );
                }

                result += weights[i] * value;
            }

            if (!Double.isFinite(result)) {
                throw new TrainingException(
                    "Numerical overflow during prediction."
                );
            }

            return result;
        }

        int predict(List<Double> features) {
            return netInput(features) >= 0.0 ? 1 : 0;
        }

        void addEpochListener(Consumer<EpochSnapshot> listener) {
            listeners.add(Objects.requireNonNull(listener));
        }

        void removeEpochListener(Consumer<EpochSnapshot> listener) {
            listeners.remove(listener);
        }

        void train(List<TrainingSample> inputSamples) {
            if (status == TrainingStatus.TRAINING) {
                throw new TrainingException(
                    "Concurrent training is not permitted."
                );
            }

            validateSamples(inputSamples);

            List<TrainingSample> samples =
                new ArrayList<>(inputSamples);

            int dimensions = samples.get(0).features().size();

            weights = new double[dimensions];
            bias = 0.0;
            history.clear();
            epochsTrained = 0;
            status = TrainingStatus.TRAINING;

            Random random = new Random(policy.randomSeed());

            try {
                for (
                    int epoch = 1;
                    epoch <= policy.maximumEpochs();
                    epoch++
                ) {
                    if (policy.shuffle()) {
                        Collections.shuffle(samples, random);
                    }

                    int mistakes = 0;

                    for (TrainingSample sample : samples) {
                        int prediction = predict(sample.features());
                        int error = sample.label() - prediction;

                        if (error != 0) {
                            mistakes++;

                            for (int j = 0; j < dimensions; j++) {
                                weights[j] +=
                                    policy.learningRate() *
                                    error *
                                    sample.features().get(j);
                            }

                            bias += policy.learningRate() * error;

                            for (double weight : weights) {
                                if (!Double.isFinite(weight)) {
                                    throw new TrainingException(
                                        "Weight overflow detected."
                                    );
                                }
                            }

                            if (!Double.isFinite(bias)) {
                                throw new TrainingException(
                                    "Bias overflow detected."
                                );
                            }
                        }
                    }

                    epochsTrained = epoch;

                    List<Double> weightSnapshot =
                        IntStream.range(0, dimensions)
                            .mapToObj(i -> weights[i])
                            .toList();

                    EpochSnapshot snapshot = new EpochSnapshot(
                        epoch,
                        mistakes,
                        weightSnapshot,
                        bias
                    );

                    history.add(snapshot);

                    for (Consumer<EpochSnapshot> listener : listeners) {
                        listener.accept(snapshot);
                    }

                    if (mistakes == 0) {
                        status = TrainingStatus.CONVERGED;
                        return;
                    }
                }

                status = TrainingStatus.MAXIMUM_EPOCHS_REACHED;

            } catch (RuntimeException exception) {
                status = TrainingStatus.FAILED;
                throw exception;
            }
        }

        EvaluationReport evaluate(List<TrainingSample> samples) {
            if (status == TrainingStatus.NOT_TRAINED ||
                status == TrainingStatus.FAILED) {
                throw new TrainingException(
                    "Model is unavailable for evaluation."
                );
            }

            int tp = 0;
            int tn = 0;
            int fp = 0;
            int fn = 0;

            for (TrainingSample sample : samples) {
                int prediction = predict(sample.features());

                if (prediction == 1 && sample.label() == 1) tp++;
                else if (prediction == 0 && sample.label() == 0) tn++;
                else if (prediction == 1) fp++;
                else fn++;
            }

            int total = tp + tn + fp + fn;

            if (total == 0) {
                throw new ValidationException(
                    "Evaluation dataset cannot be empty."
                );
            }

            return new EvaluationReport(
                tp,
                tn,
                fp,
                fn,
                (double) (tp + tn) / total
            );
        }

        ModelSnapshot snapshot() {
            if (weights == null) {
                throw new TrainingException("Model is not initialized.");
            }

            List<Double> values = IntStream.range(0, weights.length)
                .mapToObj(i -> weights[i])
                .toList();

            return new ModelSnapshot(
                values,
                bias,
                status,
                epochsTrained
            );
        }

        List<EpochSnapshot> history() {
            return List.copyOf(history);
        }
    }

    static final class FeatureScaler {

        private double[] means;
        private double[] standardDeviations;

        void fit(List<TrainingSample> samples) {
            if (samples == null || samples.isEmpty()) {
                throw new ValidationException(
                    "Scaler requires training samples."
                );
            }

            int dimensions = samples.get(0).features().size();

            means = new double[dimensions];
            standardDeviations = new double[dimensions];

            for (TrainingSample sample : samples) {
                if (sample.features().size() != dimensions) {
                    throw new ValidationException(
                        "Inconsistent feature dimensions."
                    );
                }

                for (int j = 0; j < dimensions; j++) {
                    means[j] += sample.features().get(j);
                }
            }

            for (int j = 0; j < dimensions; j++) {
                means[j] /= samples.size();
            }

            for (TrainingSample sample : samples) {
                for (int j = 0; j < dimensions; j++) {
                    double difference =
                        sample.features().get(j) - means[j];

                    standardDeviations[j] += difference * difference;
                }
            }

            for (int j = 0; j < dimensions; j++) {
                standardDeviations[j] = Math.sqrt(
                    standardDeviations[j] / samples.size()
                );

                if (standardDeviations[j] < 1e-12) {
                    standardDeviations[j] = 1.0;
                }
            }
        }

        List<TrainingSample> transform(
            List<TrainingSample> samples
        ) {
            if (means == null) {
                throw new TrainingException(
                    "Scaler has not been fitted."
                );
            }

            return samples.stream().map(sample -> {
                if (sample.features().size() != means.length) {
                    throw new ValidationException(
                        "Feature dimension mismatch."
                    );
                }

                List<Double> normalized =
                    new ArrayList<>(means.length);

                for (int j = 0; j < means.length; j++) {
                    normalized.add(
                        (sample.features().get(j) - means[j]) /
                        standardDeviations[j]
                    );
                }

                return new TrainingSample(
                    normalized,
                    sample.label()
                );
            }).toList();
        }
    }

    static final class ModelTrainingService {

        private final TrainingPolicy policy;

        ModelTrainingService(TrainingPolicy policy) {
            this.policy = policy;
        }

        PerceptronModel execute(List<TrainingSample> samples) {
            PerceptronModel model = new PerceptronModel(policy);

            model.addEpochListener(snapshot -> {
                System.out.printf(
                    "Epoch %3d | Mistakes: %d | Weights: %s | Bias: %.3f%n",
                    snapshot.epoch(),
                    snapshot.mistakes(),
                    snapshot.weights(),
                    snapshot.bias()
                );
            });

            model.train(samples);

            return model;
        }
    }

    private static List<TrainingSample> andDataset() {
        return List.of(
            new TrainingSample(List.of(0.0, 0.0), 0),
            new TrainingSample(List.of(0.0, 1.0), 0),
            new TrainingSample(List.of(1.0, 0.0), 0),
            new TrainingSample(List.of(1.0, 1.0), 1)
        );
    }

    private static List<TrainingSample> xorDataset() {
        return List.of(
            new TrainingSample(List.of(0.0, 0.0), 0),
            new TrainingSample(List.of(0.0, 1.0), 1),
            new TrainingSample(List.of(1.0, 0.0), 1),
            new TrainingSample(List.of(1.0, 1.0), 0)
        );
    }

    private static void demonstrate(
        String name,
        List<TrainingSample> samples
    ) {
        System.out.println("\n=================================");
        System.out.println(name);
        System.out.println("=================================");

        ModelTrainingService service =
            new ModelTrainingService(TrainingPolicy.standard());

        PerceptronModel model = service.execute(samples);

        ModelSnapshot snapshot = model.snapshot();

        System.out.println("\nModel state");
        System.out.println("Weights: " + snapshot.weights());
        System.out.println("Bias: " + snapshot.bias());
        System.out.println("Status: " + snapshot.status());

        EvaluationReport report = model.evaluate(samples);

        System.out.println("\nEvaluation");
        System.out.printf("Accuracy: %.2f%%%n", report.accuracy() * 100);
        System.out.println("TP: " + report.truePositive());
        System.out.println("TN: " + report.trueNegative());
        System.out.println("FP: " + report.falsePositive());
        System.out.println("FN: " + report.falseNegative());

        for (TrainingSample sample : samples) {
            System.out.println(
                sample.features() +
                " expected=" + sample.label() +
                " predicted=" + model.predict(sample.features())
            );
        }

        if (name.equals("AND gate") &&
            snapshot.status() != TrainingStatus.CONVERGED) {
            throw new TrainingException("AND should converge.");
        }

        if (name.equals("XOR gate") &&
            snapshot.status() == TrainingStatus.CONVERGED) {
            throw new TrainingException("XOR should not converge.");
        }
    }

    public static void main(String[] args) {
        try {
            demonstrate("AND gate", andDataset());
            demonstrate("XOR gate", xorDataset());

            System.out.println(
                "\nAll Java 17 experiments completed successfully."
            );

        } catch (RuntimeException exception) {
            System.err.println(
                "Execution failed: " + exception.getMessage()
            );

            System.exit(1);
        }
    }
}
