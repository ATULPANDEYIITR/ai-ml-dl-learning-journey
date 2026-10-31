import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class ModelSelectionEnterprise {

    enum ModelType {
        LOGISTIC_REGRESSION,
        DECISION_TREE
    }

    enum ReviewState {
        NOT_EVALUATED,
        VALIDATED,
        SELECTED,
        REJECTED
    }

    record Dataset(List<double[]> features, List<Integer> labels) {
        Dataset {
            if (features == null || labels == null ||
                features.isEmpty() || features.size() != labels.size()) {
                throw new IllegalArgumentException("Invalid dataset.");
            }

            int width = features.get(0).length;

            if (width == 0) {
                throw new IllegalArgumentException("Dataset has no features.");
            }

            for (double[] row : features) {
                if (row.length != width) {
                    throw new IllegalArgumentException("Inconsistent feature width.");
                }

                for (double value : row) {
                    if (!Double.isFinite(value)) {
                        throw new IllegalArgumentException("Non-finite feature.");
                    }
                }
            }

            for (int label : labels) {
                if (label != 0 && label != 1) {
                    throw new IllegalArgumentException("Labels must be binary.");
                }
            }
        }

        Dataset subset(List<Integer> indices) {
            List<double[]> selectedFeatures = new ArrayList<>();
            List<Integer> selectedLabels = new ArrayList<>();

            for (int index : indices) {
                selectedFeatures.add(features.get(index));
                selectedLabels.add(labels.get(index));
            }

            return new Dataset(selectedFeatures, selectedLabels);
        }
    }

    record Fold(List<Integer> train, List<Integer> validation) {}

    record Metrics(
        double accuracy,
        double precision,
        double recall,
        double f1,
        double logLoss
    ) {}

    record CVResult(
        ModelType model,
        List<Double> foldF1,
        double meanF1,
        double stdF1,
        double meanAccuracy,
        double meanLogLoss
    ) {}

    record ModelDecision(
        ModelType model,
        double crossValidationF1,
        double repeatedCVF1,
        ReviewState state
    ) {}

    interface Classifier {
        void fit(Dataset dataset);
        int[] predict(List<double[]> features);
        double[] predictProbability(List<double[]> features);
    }

    static class LogisticRegression implements Classifier {
        private final double learningRate;
        private final int epochs;
        private final double l2;

        private double[] weights;
        private double bias;

        LogisticRegression(double learningRate, int epochs, double l2) {
            if (learningRate <= 0 || epochs <= 0 || l2 < 0) {
                throw new IllegalArgumentException("Invalid model configuration.");
            }

            this.learningRate = learningRate;
            this.epochs = epochs;
            this.l2 = l2;
        }

        private double sigmoid(double value) {
            value = Math.max(-60, Math.min(60, value));
            return 1.0 / (1.0 + Math.exp(-value));
        }

        @Override
        public void fit(Dataset dataset) {
            int width = dataset.features().get(0).length;
            weights = new double[width];
            bias = 0;

            for (int epoch = 0; epoch < epochs; epoch++) {
                double[] gradient = new double[width];
                double biasGradient = 0;

                for (int i = 0; i < dataset.features().size(); i++) {
                    double[] row = dataset.features().get(i);
                    double linear = bias;

                    for (int j = 0; j < width; j++) {
                        linear += weights[j] * row[j];
                    }

                    double error = sigmoid(linear) - dataset.labels().get(i);

                    for (int j = 0; j < width; j++) {
                        gradient[j] += error * row[j];
                    }

                    biasGradient += error;
                }

                for (int j = 0; j < width; j++) {
                    gradient[j] =
                        gradient[j] / dataset.features().size()
                        + l2 * weights[j];

                    weights[j] -= learningRate * gradient[j];
                }

                bias -= learningRate
                    * biasGradient
                    / dataset.features().size();
            }
        }

        @Override
        public double[] predictProbability(List<double[]> features) {
            if (weights == null) {
                throw new IllegalStateException("Model has not been fitted.");
            }

            double[] result = new double[features.size()];

            for (int i = 0; i < features.size(); i++) {
                double linear = bias;

                for (int j = 0; j < weights.length; j++) {
                    linear += weights[j] * features.get(i)[j];
                }

                result[i] = sigmoid(linear);
            }

            return result;
        }

        @Override
        public int[] predict(List<double[]> features) {
            double[] probabilities = predictProbability(features);
            int[] result = new int[probabilities.length];

            for (int i = 0; i < probabilities.length; i++) {
                result[i] = probabilities[i] >= 0.5 ? 1 : 0;
            }

            return result;
        }
    }

    static class DecisionTree implements Classifier {
        private static final class Node {
            final boolean leaf;
            final int prediction;
            final int feature;
            final double threshold;
            final Node left;
            final Node right;

            Node(
                boolean leaf,
                int prediction,
                int feature,
                double threshold,
                Node left,
                Node right
            ) {
                this.leaf = leaf;
                this.prediction = prediction;
                this.feature = feature;
                this.threshold = threshold;
                this.left = left;
                this.right = right;
            }
        }

        private final int maxDepth;
        private final int minSamplesSplit;
        private Node root;

        DecisionTree(int maxDepth, int minSamplesSplit) {
            if (maxDepth < 1 || minSamplesSplit < 2) {
                throw new IllegalArgumentException("Invalid tree configuration.");
            }

            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
        }

        private double gini(List<Integer> labels) {
            if (labels.isEmpty()) return 0;

            double positive = 0;

            for (int label : labels) {
                positive += label;
            }

            positive /= labels.size();
            return 2 * positive * (1 - positive);
        }

        private int majority(List<Integer> labels) {
            int positive = 0;

            for (int label : labels) {
                positive += label;
            }

            return positive * 2 >= labels.size() ? 1 : 0;
        }

        private record Split(int feature, double threshold, double gain) {}

        private Split bestSplit(Dataset dataset, List<Integer> indices) {
            double parent = gini(
                indices.stream()
                    .map(dataset.labels()::get)
                    .toList()
            );

            Split best = null;

            int width = dataset.features().get(0).length;

            for (int feature = 0; feature < width; feature++) {
                List<Double> values = indices.stream()
                    .map(index -> dataset.features().get(index)[feature])
                    .distinct()
                    .sorted()
                    .toList();

                for (int i = 0; i + 1 < values.size(); i++) {
                    double threshold =
                        (values.get(i) + values.get(i + 1)) / 2.0;

                    List<Integer> left = new ArrayList<>();
                    List<Integer> right = new ArrayList<>();

                    for (int index : indices) {
                        if (dataset.features().get(index)[feature] <= threshold) {
                            left.add(index);
                        } else {
                            right.add(index);
                        }
                    }

                    if (left.isEmpty() || right.isEmpty()) continue;

                    double weighted =
                        (left.size() * gini(
                            left.stream()
                                .map(dataset.labels()::get)
                                .toList()
                        )
                        + right.size() * gini(
                            right.stream()
                                .map(dataset.labels()::get)
                                .toList()
                        ))
                        / indices.size();

                    double gain = parent - weighted;

                    if (best == null || gain > best.gain()) {
                        best = new Split(feature, threshold, gain);
                    }
                }
            }

            return best;
        }

        private Node build(
            Dataset dataset,
            List<Integer> indices,
            int depth
        ) {
            List<Integer> labels = indices.stream()
                .map(dataset.labels()::get)
                .toList();

            int prediction = majority(labels);
            Set<Integer> classes = new HashSet<>(labels);

            if (
                depth >= maxDepth ||
                indices.size() < minSamplesSplit ||
                classes.size() == 1
            ) {
                return new Node(true, prediction, 0, 0, null, null);
            }

            Split split = bestSplit(dataset, indices);

            if (split == null || split.gain() <= 0) {
                return new Node(true, prediction, 0, 0, null, null);
            }

            List<Integer> left = new ArrayList<>();
            List<Integer> right = new ArrayList<>();

            for (int index : indices) {
                if (dataset.features().get(index)[split.feature()] <= split.threshold()) {
                    left.add(index);
                } else {
                    right.add(index);
                }
            }

            return new Node(
                false,
                prediction,
                split.feature(),
                split.threshold(),
                build(dataset, left, depth + 1),
                build(dataset, right, depth + 1)
            );
        }

        @Override
        public void fit(Dataset dataset) {
            List<Integer> indices = new ArrayList<>();

            for (int i = 0; i < dataset.labels().size(); i++) {
                indices.add(i);
            }

            root = build(dataset, indices, 0);
        }

        private int predictOne(double[] row, Node node) {
            if (node.leaf) {
                return node.prediction;
            }

            return row[node.feature] <= node.threshold
                ? predictOne(row, node.left)
                : predictOne(row, node.right);
        }

        @Override
        public int[] predict(List<double[]> features) {
            if (root == null) {
                throw new IllegalStateException("Tree has not been fitted.");
            }

            int[] result = new int[features.size()];

            for (int i = 0; i < features.size(); i++) {
                result[i] = predictOne(features.get(i), root);
            }

            return result;
        }

        @Override
        public double[] predictProbability(List<double[]> features) {
            int[] predictions = predict(features);
            double[] probabilities = new double[predictions.length];

            for (int i = 0; i < predictions.length; i++) {
                probabilities[i] = predictions[i] == 1 ? 0.85 : 0.15;
            }

            return probabilities;
        }
    }

    static Classifier createModel(ModelType type) {
        return switch (type) {
            case LOGISTIC_REGRESSION ->
                new LogisticRegression(0.08, 600, 0.02);
            case DECISION_TREE ->
                new DecisionTree(4, 6);
        };
    }

    static List<Fold> stratifiedKFold(
        List<Integer> labels,
        int k,
        long seed
    ) {
        if (k < 2 || k > labels.size()) {
            throw new IllegalArgumentException("Invalid number of folds.");
        }

        Map<Integer, List<Integer>> byClass = new HashMap<>();

        for (int i = 0; i < labels.size(); i++) {
            byClass.computeIfAbsent(labels.get(i), key -> new ArrayList<>())
                .add(i);
        }

        Random random = new Random(seed);
        List<List<Integer>> validationFolds = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            validationFolds.add(new ArrayList<>());
        }

        for (List<Integer> indices : byClass.values()) {
            Collections.shuffle(indices, random);

            for (int i = 0; i < indices.size(); i++) {
                validationFolds
                    .get(i % k)
                    .add(indices.get(i));
            }
        }

        List<Fold> folds = new ArrayList<>();

        for (List<Integer> validation : validationFolds) {
            Collections.shuffle(validation, random);

            Set<Integer> validationSet = new HashSet<>(validation);
            List<Integer> training = new ArrayList<>();

            for (int i = 0; i < labels.size(); i++) {
                if (!validationSet.contains(i)) {
                    training.add(i);
                }
            }

            folds.add(new Fold(training, validation));
        }

        return folds;
    }

    static Metrics metrics(
        List<Integer> actual,
        int[] predicted,
        double[] probabilities
    ) {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        double loss = 0;

        for (int i = 0; i < actual.size(); i++) {
            if (actual.get(i) == 1 && predicted[i] == 1) tp++;
            if (actual.get(i) == 0 && predicted[i] == 0) tn++;
            if (actual.get(i) == 0 && predicted[i] == 1) fp++;
            if (actual.get(i) == 1 && predicted[i] == 0) fn++;

            double p = Math.max(
                1e-15,
                Math.min(1 - 1e-15, probabilities[i])
            );

            loss -= actual.get(i) * Math.log(p)
                + (1 - actual.get(i)) * Math.log(1 - p);
        }

        double accuracy =
            (double) (tp + tn) / actual.size();

        double precision =
            tp + fp == 0 ? 0 : (double) tp / (tp + fp);

        double recall =
            tp + fn == 0 ? 0 : (double) tp / (tp + fn);

        double f1 =
            precision + recall == 0
                ? 0
                : 2 * precision * recall / (precision + recall);

        return new Metrics(
            accuracy,
            precision,
            recall,
            f1,
            loss / actual.size()
        );
    }

    static double mean(List<Double> values) {
        return values.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0);
    }

    static double std(List<Double> values) {
        if (values.size() < 2) return 0;

        double average = mean(values);

        double variance = values.stream()
            .mapToDouble(value -> Math.pow(value - average, 2))
            .sum()
            / (values.size() - 1);

        return Math.sqrt(variance);
    }

    static CVResult evaluate(
        ModelType modelType,
        Dataset dataset,
        List<Fold> folds
    ) {
        List<Double> scores = new ArrayList<>();
        List<Double> accuracies = new ArrayList<>();
        List<Double> losses = new ArrayList<>();

        for (Fold fold : folds) {
            Dataset training = dataset.subset(fold.train());
            Dataset validation = dataset.subset(fold.validation());

            Classifier model = createModel(modelType);
            model.fit(training);

            int[] predictions =
                model.predict(validation.features());

            double[] probabilities =
                model.predictProbability(validation.features());

            Metrics metrics =
                metrics(validation.labels(), predictions, probabilities);

            scores.add(metrics.f1());
            accuracies.add(metrics.accuracy());
            losses.add(metrics.logLoss());
        }

        return new CVResult(
            modelType,
            List.copyOf(scores),
            mean(scores),
            std(scores),
            mean(accuracies),
            mean(losses)
        );
    }

    static Dataset generateDataset(int size, long seed) {
        Random random = new Random(seed);
        List<double[]> features = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            double x1 = random.nextGaussian();
            double x2 = random.nextGaussian();
            double x3 = random.nextGaussian();
            double x4 = random.nextGaussian();

            double score =
                1.6 * x1
                - 1.2 * x2
                + 0.7 * x3
                + 0.25 * x4
                + 0.8 * random.nextGaussian();

            features.add(new double[]{x1, x2, x3, x4});
            labels.add(score > 0.35 ? 1 : 0);
        }

        return new Dataset(features, labels);
    }

    static Dataset[] holdoutSplit(
        Dataset dataset,
        double testRatio,
        long seed
    ) {
        List<Integer> indices = new ArrayList<>();

        for (int i = 0; i < dataset.labels().size(); i++) {
            indices.add(i);
        }

        Collections.shuffle(indices, new Random(seed));

        int testSize =
            Math.max(1, (int) Math.round(indices.size() * testRatio));

        return new Dataset[]{
            dataset.subset(indices.subList(testSize, indices.size())),
            dataset.subset(indices.subList(0, testSize))
        };
    }

    public static void main(String[] args) {
        Dataset complete = generateDataset(180, 7);
        Dataset[] split = holdoutSplit(complete, 0.20, 42);

        Dataset training = split[0];
        Dataset holdout = split[1];

        List<Fold> folds =
            stratifiedKFold(training.labels(), 5, 42);

        System.out.println("ML MODEL SELECTION");
        System.out.println("==================");
        System.out.println("Training samples: " + training.labels().size());
        System.out.println("Holdout samples: " + holdout.labels().size());
        System.out.println();

        EnumMap<ModelType, CVResult> results =
            new EnumMap<>(ModelType.class);

        for (ModelType type : ModelType.values()) {
            CVResult result = evaluate(type, training, folds);
            results.put(type, result);

            System.out.println(type);
            System.out.println("Fold F1: " + result.foldF1());
            System.out.printf("Mean F1: %.4f%n", result.meanF1());
            System.out.printf("F1 std: %.4f%n", result.stdF1());
            System.out.printf("Accuracy: %.4f%n", result.meanAccuracy());
            System.out.printf("Log loss: %.4f%n%n", result.meanLogLoss());
        }

        CVResult selected = results.values()
            .stream()
            .min(
                Comparator
                    .comparingDouble(CVResult::meanF1)
                    .reversed()
                    .thenComparingDouble(CVResult::stdF1)
            )
            .orElseThrow();

        List<Double> repeatedF1 = new ArrayList<>();

        for (int repetition = 0; repetition < 3; repetition++) {
            List<Fold> repeatedFolds =
                stratifiedKFold(
                    training.labels(),
                    5,
                    42L + repetition
                );

            repeatedF1.add(
                evaluate(
                    selected.model(),
                    training,
                    repeatedFolds
                ).meanF1()
            );
        }

        double repeatedMean = mean(repeatedF1);

        ModelDecision decision = new ModelDecision(
            selected.model(),
            selected.meanF1(),
            repeatedMean,
            ReviewState.SELECTED
        );

        System.out.println("REPEATED STRATIFIED CV");
        System.out.println("----------------------");
        System.out.println("Model: " + decision.model());
        System.out.println("F1: " + repeatedF1);
        System.out.printf("Repeated mean F1: %.4f%n", repeatedMean);
        System.out.println();

        Classifier finalModel = createModel(decision.model());
        finalModel.fit(training);

        int[] predictions =
            finalModel.predict(holdout.features());

        double[] probabilities =
            finalModel.predictProbability(holdout.features());

        Metrics finalMetrics =
            metrics(holdout.labels(), predictions, probabilities);

        System.out.println("FINAL HOLDOUT");
        System.out.println("-------------");
        System.out.println("Selected model: " + decision.model());
        System.out.printf("Accuracy: %.4f%n", finalMetrics.accuracy());
        System.out.printf("Precision: %.4f%n", finalMetrics.precision());
        System.out.printf("Recall: %.4f%n", finalMetrics.recall());
        System.out.printf("F1: %.4f%n", finalMetrics.f1());
        System.out.printf("Log loss: %.4f%n", finalMetrics.logLoss());
    }
}
