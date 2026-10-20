import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

public class XGBoostEnterpriseModel {

    enum RiskClass {
        LOW,
        HIGH
    }

    record PullRequest(
        int id,
        double changedLines,
        double testFailureRate,
        double historicalRisk,
        double reviewCoverage,
        int label
    ) {
        PullRequest {
            if (id <= 0) throw new IllegalArgumentException("Invalid ID.");
            if (changedLines < 0) throw new IllegalArgumentException("Negative changes.");
            if (testFailureRate < 0 || testFailureRate > 1)
                throw new IllegalArgumentException("Invalid test failure rate.");
            if (historicalRisk < 0 || historicalRisk > 1)
                throw new IllegalArgumentException("Invalid historical risk.");
            if (reviewCoverage < 0 || reviewCoverage > 1)
                throw new IllegalArgumentException("Invalid review coverage.");
            if (label != 0 && label != 1)
                throw new IllegalArgumentException("Label must be 0 or 1.");
        }
    }

    record HyperParameters(
        int maxDepth,
        int estimators,
        double learningRate,
        double lambda,
        double alpha,
        double gamma,
        double minChildWeight
    ) {
        HyperParameters {
            if (maxDepth < 1 || estimators < 1)
                throw new IllegalArgumentException("Invalid tree limits.");
            if (learningRate <= 0 || learningRate > 1)
                throw new IllegalArgumentException("Invalid learning rate.");
            if (lambda < 0 || alpha < 0 || gamma < 0)
                throw new IllegalArgumentException("Invalid regularization.");
            if (minChildWeight <= 0)
                throw new IllegalArgumentException("Invalid child weight.");
        }
    }

    interface Objective {
        double gradient(double prediction, int label);
        double hessian(double prediction);
    }

    static final class LogisticObjective implements Objective {
        @Override
        public double gradient(double prediction, int label) {
            return sigmoid(prediction) - label;
        }

        @Override
        public double hessian(double prediction) {
            double p = sigmoid(prediction);
            return Math.max(p * (1 - p), 1e-6);
        }
    }

    static final class TreeNode {
        final int feature;
        final double threshold;
        final double weight;
        final TreeNode left;
        final TreeNode right;

        TreeNode(double weight) {
            this.feature = -1;
            this.threshold = 0;
            this.weight = weight;
            this.left = null;
            this.right = null;
        }

        TreeNode(
            int feature,
            double threshold,
            TreeNode left,
            TreeNode right
        ) {
            this.feature = feature;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.weight = 0;
        }

        boolean isLeaf() {
            return feature < 0;
        }

        double predict(double[] features) {
            if (isLeaf()) return weight;
            return features[feature] <= threshold
                ? left.predict(features)
                : right.predict(features);
        }
    }

    static final class EnterpriseXGBoost {
        private final HyperParameters parameters;
        private final Objective objective;
        private final List<TreeNode> trees = new ArrayList<>();

        EnterpriseXGBoost(
            HyperParameters parameters,
            Objective objective
        ) {
            this.parameters = Objects.requireNonNull(parameters);
            this.objective = Objects.requireNonNull(objective);
        }

        private double regularizedGradient(double gradient) {
            if (gradient > parameters.alpha())
                return gradient - parameters.alpha();
            if (gradient < -parameters.alpha())
                return gradient + parameters.alpha();
            return 0;
        }

        private double leafWeight(
            double[] gradients,
            double[] hessians
        ) {
            double g = 0;
            double h = 0;

            for (double value : gradients) g += value;
            for (double value : hessians) h += value;

            if (h < parameters.minChildWeight()) return 0;

            return -regularizedGradient(g) /
                (h + parameters.lambda());
        }

        private double score(double gradient, double hessian) {
            double g = regularizedGradient(gradient);
            return (g * g) / (hessian + parameters.lambda());
        }

        private double gain(
            double leftG,
            double leftH,
            double rightG,
            double rightH,
            double parentG,
            double parentH
        ) {
            if (
                leftH < parameters.minChildWeight() ||
                rightH < parameters.minChildWeight()
            ) {
                return Double.NEGATIVE_INFINITY;
            }

            return 0.5 * (
                score(leftG, leftH) +
                score(rightG, rightH) -
                score(parentG, parentH)
            ) - parameters.gamma();
        }

        private TreeNode buildTree(
            List<PullRequest> rows,
            double[] gradients,
            double[] hessians,
            List<Integer> indexes,
            int depth
        ) {
            double[] localG = new double[indexes.size()];
            double[] localH = new double[indexes.size()];

            for (int i = 0; i < indexes.size(); i++) {
                localG[i] = gradients[indexes.get(i)];
                localH[i] = hessians[indexes.get(i)];
            }

            TreeNode leaf = new TreeNode(leafWeight(localG, localH));

            if (
                depth >= parameters.maxDepth() ||
                indexes.size() < 2
            ) {
                return leaf;
            }

            double parentG = indexes.stream()
                .mapToDouble(i -> gradients[i])
                .sum();

            double parentH = indexes.stream()
                .mapToDouble(i -> hessians[i])
                .sum();

            double bestGain = parameters.gamma();
            int bestFeature = -1;
            double bestThreshold = 0;
            List<Integer> bestLeft = List.of();
            List<Integer> bestRight = List.of();

            for (int feature = 0; feature < 4; feature++) {
                final int selectedFeature = feature;

                List<Double> values = indexes.stream()
                    .map(i -> featureValue(rows.get(i), selectedFeature))
                    .distinct()
                    .sorted(Comparator.naturalOrder())
                    .toList();

                for (int i = 0; i + 1 < values.size(); i++) {
                    double threshold = (values.get(i) + values.get(i + 1)) / 2;

                    List<Integer> left = new ArrayList<>();
                    List<Integer> right = new ArrayList<>();

                    for (int index : indexes) {
                        if (
                            featureValue(rows.get(index), feature)
                                <= threshold
                        ) {
                            left.add(index);
                        } else {
                            right.add(index);
                        }
                    }

                    if (left.isEmpty() || right.isEmpty()) continue;

                    double leftG = left.stream()
                        .mapToDouble(i2 -> gradients[i2])
                        .sum();
                    double leftH = left.stream()
                        .mapToDouble(i2 -> hessians[i2])
                        .sum();
                    double rightG = right.stream()
                        .mapToDouble(i2 -> gradients[i2])
                        .sum();
                    double rightH = right.stream()
                        .mapToDouble(i2 -> hessians[i2])
                        .sum();

                    double candidateGain = gain(
                        leftG, leftH, rightG, rightH,
                        parentG, parentH
                    );

                    if (candidateGain > bestGain) {
                        bestGain = candidateGain;
                        bestFeature = feature;
                        bestThreshold = threshold;
                        bestLeft = left;
                        bestRight = right;
                    }
                }
            }

            if (bestFeature < 0) return leaf;

            return new TreeNode(
                bestFeature,
                bestThreshold,
                buildTree(
                    rows, gradients, hessians,
                    bestLeft, depth + 1
                ),
                buildTree(
                    rows, gradients, hessians,
                    bestRight, depth + 1
                )
            );
        }

        void fit(List<PullRequest> rows) {
            if (rows.isEmpty())
                throw new IllegalArgumentException("No training records.");

            double[] margins = new double[rows.size()];
            trees.clear();

            for (int round = 0; round < parameters.estimators(); round++) {
                double[] gradients = new double[rows.size()];
                double[] hessians = new double[rows.size()];

                for (int i = 0; i < rows.size(); i++) {
                    gradients[i] =
                        objective.gradient(margins[i], rows.get(i).label());
                    hessians[i] =
                        objective.hessian(margins[i]);
                }

                List<Integer> indexes =
                    IntStream.range(0, rows.size()).boxed().toList();

                TreeNode tree = buildTree(
                    rows,
                    gradients,
                    hessians,
                    indexes,
                    0
                );

                trees.add(tree);

                for (int i = 0; i < rows.size(); i++) {
                    margins[i] +=
                        parameters.learningRate() *
                        tree.predict(toFeatures(rows.get(i)));
                }
            }
        }

        double probability(PullRequest row) {
            if (trees.isEmpty())
                throw new IllegalStateException("Model is not trained.");

            double margin = 0;
            for (TreeNode tree : trees) {
                margin += parameters.learningRate() *
                    tree.predict(toFeatures(row));
            }

            return sigmoid(margin);
        }

        RiskClass classify(PullRequest row) {
            return probability(row) >= 0.5
                ? RiskClass.HIGH
                : RiskClass.LOW;
        }
    }

    static double featureValue(PullRequest row, int feature) {
        return switch (feature) {
            case 0 -> row.changedLines();
            case 1 -> row.testFailureRate();
            case 2 -> row.historicalRisk();
            case 3 -> row.reviewCoverage();
            default -> throw new IllegalArgumentException("Unknown feature.");
        };
    }

    static double[] toFeatures(PullRequest row) {
        return new double[] {
            row.changedLines(),
            row.testFailureRate(),
            row.historicalRisk(),
            row.reviewCoverage()
        };
    }

    static double sigmoid(double value) {
        if (value >= 0) {
            double z = Math.exp(-value);
            return 1 / (1 + z);
        }
        double z = Math.exp(value);
        return z / (1 + z);
    }

    public static void main(String[] args) {
        List<PullRequest> requests = List.of(
            new PullRequest(101, 10, .00, .10, .90, 0),
            new PullRequest(102, 20, .00, .20, .80, 0),
            new PullRequest(103, 25, .02, .15, .90, 0),
            new PullRequest(104, 40, .10, .40, .70, 1),
            new PullRequest(105, 55, .15, .55, .60, 1),
            new PullRequest(106, 70, .20, .65, .55, 1),
            new PullRequest(107, 80, .25, .80, .40, 1),
            new PullRequest(108, 95, .35, .90, .20, 1)
        );

        HyperParameters parameters = new HyperParameters(
            3,
            40,
            .10,
            2.0,
            .05,
            .05,
            1.0
        );

        EnterpriseXGBoost model =
            new EnterpriseXGBoost(
                parameters,
                new LogisticObjective()
            );

        model.fit(requests);

        for (PullRequest request : requests) {
            System.out.printf(
                "PR-%d probability=%.3f class=%s%n",
                request.id(),
                model.probability(request),
                model.classify(request)
            );
        }
    }
}
