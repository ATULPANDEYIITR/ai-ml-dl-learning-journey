import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/*
 * Enterprise repository-risk forecasting case study.
 *
 * The system estimates merge-risk scores from Pull Request characteristics.
 * Gradient boosting is modeled explicitly through:
 *
 *   baseline = average historical risk
 *   residual = observed risk - current prediction
 *   weak learner = shallow regression tree fitted to residuals
 *   new prediction = old prediction + learningRate * weakLearner
 *
 * Java-specific design:
 * - records provide immutable domain observations
 * - enums make feature names and workflow states explicit
 * - services separate training from domain reporting
 * - immutable record values reduce accidental mutation
 * - collections provide the training and reporting structures
 *
 * Java 17+
 */

public class GradientBoostingEnterpriseDemo {

    enum RiskFeature {
        CHANGED_LINES,
        FILE_COUNT,
        TEST_FAILURES,
        REVIEW_COMMENTS
    }

    record PullRequestRisk(
        int changedLines,
        int fileCount,
        int testFailures,
        int reviewComments,
        double riskScore
    ) {
        PullRequestRisk {
            if (changedLines < 0 || fileCount < 0) {
                throw new IllegalArgumentException(
                    "size features cannot be negative"
                );
            }

            if (testFailures < 0 || reviewComments < 0) {
                throw new IllegalArgumentException(
                    "workflow counts cannot be negative"
                );
            }

            if (!Double.isFinite(riskScore)) {
                throw new IllegalArgumentException(
                    "risk score must be finite"
                );
            }
        }

        double feature(RiskFeature feature) {
            return switch (feature) {
                case CHANGED_LINES -> changedLines;
                case FILE_COUNT -> fileCount;
                case TEST_FAILURES -> testFailures;
                case REVIEW_COMMENTS -> reviewComments;
            };
        }
    }

    record TreeNode(
        double prediction,
        RiskFeature feature,
        double threshold,
        TreeNode left,
        TreeNode right
    ) {
        boolean isLeaf() {
            return feature == null;
        }
    }

    static final class RegressionTree {

        private final int maxDepth;
        private final int minSamplesLeaf;
        private TreeNode root;
        private final EnumMap<RiskFeature, Double> importance =
            new EnumMap<>(RiskFeature.class);

        RegressionTree(int maxDepth, int minSamplesLeaf) {
            if (maxDepth < 0 || minSamplesLeaf < 1) {
                throw new IllegalArgumentException(
                    "invalid tree configuration"
                );
            }

            this.maxDepth = maxDepth;
            this.minSamplesLeaf = minSamplesLeaf;

            for (RiskFeature feature : RiskFeature.values()) {
                importance.put(feature, 0.0);
            }
        }

        void fit(
            List<PullRequestRisk> observations,
            List<Double> residuals
        ) {
            if (observations.isEmpty()
                || observations.size() != residuals.size()) {
                throw new IllegalArgumentException(
                    "invalid tree training data"
                );
            }

            List<Integer> indices = new ArrayList<>();

            for (int index = 0; index < observations.size(); index++) {
                indices.add(index);
            }

            root = build(observations, residuals, indices, 0);
        }

        private TreeNode build(
            List<PullRequestRisk> observations,
            List<Double> residuals,
            List<Integer> indices,
            int depth
        ) {
            double prediction = average(residuals, indices);

            if (
                depth >= maxDepth
                || indices.size() < minSamplesLeaf * 2
            ) {
                return new TreeNode(
                    prediction,
                    null,
                    0.0,
                    null,
                    null
                );
            }

            Split best = findBestSplit(
                observations,
                residuals,
                indices
            );

            if (best == null) {
                return new TreeNode(
                    prediction,
                    null,
                    0.0,
                    null,
                    null
                );
            }

            List<Integer> left = new ArrayList<>();
            List<Integer> right = new ArrayList<>();

            for (int index : indices) {
                if (
                    observations.get(index).feature(best.feature)
                    <= best.threshold
                ) {
                    left.add(index);
                } else {
                    right.add(index);
                }
            }

            if (
                left.size() < minSamplesLeaf
                || right.size() < minSamplesLeaf
            ) {
                return new TreeNode(
                    prediction,
                    null,
                    0.0,
                    null,
                    null
                );
            }

            importance.merge(
                best.feature,
                best.improvement,
                Double::sum
            );

            return new TreeNode(
                prediction,
                best.feature,
                best.threshold,
                build(observations, residuals, left, depth + 1),
                build(observations, residuals, right, depth + 1)
            );
        }

        private Split findBestSplit(
            List<PullRequestRisk> observations,
            List<Double> residuals,
            List<Integer> indices
        ) {
            double parentError = squaredError(residuals, indices);
            Split best = null;

            for (RiskFeature feature : RiskFeature.values()) {
                List<Integer> ordered = new ArrayList<>(indices);

                ordered.sort(
                    (a, b) -> Double.compare(
                        observations.get(a).feature(feature),
                        observations.get(b).feature(feature)
                    )
                );

                for (
                    int position = minSamplesLeaf;
                    position <= ordered.size() - minSamplesLeaf;
                    position++
                ) {
                    double leftValue =
                        observations.get(
                            ordered.get(position - 1)
                        ).feature(feature);

                    double rightValue =
                        observations.get(
                            ordered.get(position)
                        ).feature(feature);

                    if (leftValue == rightValue) {
                        continue;
                    }

                    double threshold =
                        (leftValue + rightValue) / 2.0;

                    List<Integer> left =
                        new ArrayList<>(
                            ordered.subList(0, position)
                        );

                    List<Integer> right =
                        new ArrayList<>(
                            ordered.subList(position, ordered.size())
                        );

                    double improvement =
                        parentError
                        - squaredError(residuals, left)
                        - squaredError(residuals, right);

                    if (
                        improvement > 1e-12
                        && (best == null
                            || improvement > best.improvement)
                    ) {
                        best = new Split(
                            feature,
                            threshold,
                            improvement
                        );
                    }
                }
            }

            return best;
        }

        private static double average(
            List<Double> values,
            List<Integer> indices
        ) {
            double total = 0.0;

            for (int index : indices) {
                total += values.get(index);
            }

            return total / indices.size();
        }

        private static double squaredError(
            List<Double> values,
            List<Integer> indices
        ) {
            double center = average(values, indices);
            double total = 0.0;

            for (int index : indices) {
                double difference = values.get(index) - center;
                total += difference * difference;
            }

            return total;
        }

        double predict(PullRequestRisk observation) {
            Objects.requireNonNull(root);

            TreeNode node = root;

            while (!node.isLeaf()) {
                if (
                    observation.feature(node.feature())
                    <= node.threshold()
                ) {
                    node = node.left();
                } else {
                    node = node.right();
                }
            }

            return node.prediction();
        }

        Map<RiskFeature, Double> importance() {
            return Map.copyOf(importance);
        }

        private record Split(
            RiskFeature feature,
            double threshold,
            double improvement
        ) {}
    }

    static final class GradientBoostingRiskService {

        private final int treeCount;
        private final double learningRate;
        private final int treeDepth;
        private final int minSamplesLeaf;
        private final List<RegressionTree> trees = new ArrayList<>();

        private double baseline;

        GradientBoostingRiskService(
            int treeCount,
            double learningRate,
            int treeDepth,
            int minSamplesLeaf
        ) {
            if (
                treeCount < 1
                || learningRate <= 0
                || treeDepth < 0
                || minSamplesLeaf < 1
            ) {
                throw new IllegalArgumentException(
                    "invalid boosting configuration"
                );
            }

            this.treeCount = treeCount;
            this.learningRate = learningRate;
            this.treeDepth = treeDepth;
            this.minSamplesLeaf = minSamplesLeaf;
        }

        void train(List<PullRequestRisk> data) {
            if (data.isEmpty()) {
                throw new IllegalArgumentException(
                    "training data cannot be empty"
                );
            }

            baseline = data.stream()
                .mapToDouble(PullRequestRisk::riskScore)
                .average()
                .orElseThrow();

            List<Double> predictions = new ArrayList<>();

            for (int index = 0; index < data.size(); index++) {
                predictions.add(baseline);
            }

            trees.clear();

            for (int iteration = 0; iteration < treeCount; iteration++) {
                /*
                 * The residual is the negative gradient for squared error.
                 * A tree learns the remaining systematic error rather than
                 * relearning the complete target from scratch.
                 */
                List<Double> residuals = new ArrayList<>();

                for (int index = 0; index < data.size(); index++) {
                    residuals.add(
                        data.get(index).riskScore()
                        - predictions.get(index)
                    );
                }

                RegressionTree tree = new RegressionTree(
                    treeDepth,
                    minSamplesLeaf
                );

                tree.fit(data, residuals);

                for (int index = 0; index < predictions.size(); index++) {
                    predictions.set(
                        index,
                        predictions.get(index)
                        + learningRate * tree.predict(data.get(index))
                    );
                }

                trees.add(tree);
            }
        }

        double predict(PullRequestRisk observation) {
            if (trees.isEmpty()) {
                throw new IllegalStateException(
                    "risk service has not been trained"
                );
            }

            double prediction = baseline;

            for (RegressionTree tree : trees) {
                prediction +=
                    learningRate * tree.predict(observation);
            }

            return prediction;
        }

        Map<RiskFeature, Double> aggregateImportance() {
            EnumMap<RiskFeature, Double> result =
                new EnumMap<>(RiskFeature.class);

            for (RiskFeature feature : RiskFeature.values()) {
                result.put(feature, 0.0);
            }

            for (RegressionTree tree : trees) {
                for (
                    Map.Entry<RiskFeature, Double> entry :
                    tree.importance().entrySet()
                ) {
                    result.merge(
                        entry.getKey(),
                        entry.getValue(),
                        Double::sum
                    );
                }
            }

            double total = result.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

            if (total > 0) {
                result.replaceAll(
                    (feature, value) -> value / total
                );
            }

            return Map.copyOf(result);
        }
    }

    static List<PullRequestRisk> trainingData() {
        return List.of(
            new PullRequestRisk(12, 2, 0, 1, 0.08),
            new PullRequestRisk(35, 5, 0, 2, 0.18),
            new PullRequestRisk(70, 8, 1, 4, 0.38),
            new PullRequestRisk(110, 13, 2, 7, 0.62),
            new PullRequestRisk(180, 20, 3, 11, 0.87),
            new PullRequestRisk(240, 28, 5, 16, 0.96),
            new PullRequestRisk(20, 3, 0, 0, 0.06),
            new PullRequestRisk(55, 6, 1, 3, 0.29),
            new PullRequestRisk(95, 11, 1, 5, 0.48),
            new PullRequestRisk(150, 17, 2, 8, 0.72),
            new PullRequestRisk(210, 25, 4, 14, 0.93),
            new PullRequestRisk(300, 35, 6, 20, 0.99)
        );
    }

    public static void main(String[] args) {
        List<PullRequestRisk> history = trainingData();

        GradientBoostingRiskService service =
            new GradientBoostingRiskService(
                100,
                0.04,
                2,
                2
            );

        service.train(history);

        List<PullRequestRisk> incoming = List.of(
            new PullRequestRisk(42, 6, 0, 2, 0.0),
            new PullRequestRisk(135, 15, 2, 8, 0.0),
            new PullRequestRisk(260, 31, 5, 17, 0.0)
        );

        System.out.println("Enterprise Pull Request risk estimates");

        for (PullRequestRisk request : incoming) {
            double risk = service.predict(request);

            System.out.printf(
                "changedLines=%d files=%d testFailures=%d "
                + "reviewComments=%d predictedRisk=%.3f%n",
                request.changedLines(),
                request.fileCount(),
                request.testFailures(),
                request.reviewComments(),
                risk
            );
        }

        System.out.println("\nResidual-driven feature importance");

        service.aggregateImportance().entrySet().stream()
            .sorted(
                Map.Entry.<RiskFeature, Double>comparingByValue()
                    .reversed()
            )
            .forEach(
                entry -> System.out.printf(
                    "%-20s %.4f%n",
                    entry.getKey(),
                    entry.getValue()
                )
            );

        /*
         * A production enterprise service should validate feature definitions,
         * monitor drift, retain the training configuration, and prevent target
         * leakage. The model score is an estimate, not a reason to bypass
         * repository policy. Governance rules still determine merge eligibility.
         */
    }
}
