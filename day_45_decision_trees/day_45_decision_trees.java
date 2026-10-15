import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;

/**
 * Enterprise-oriented decision tree for release-risk triage.
 *
 * Java 17 or later:
 *   javac DecisionTreeEnterprise.java
 *   java DecisionTreeEnterprise
 *
 * The model predicts whether a software release should be eligible for
 * automated approval or should receive manual review. It illustrates
 * impurity-based splitting, typed domain data, explicit stopping rules,
 * prediction validation, and an auditable tree structure.
 */
public class DecisionTreeEnterprise {

    enum Decision {
        ELIGIBLE,
        MANUAL_REVIEW
    }

    enum Criterion {
        ENTROPY,
        GINI
    }

    enum Feature {
        COVERAGE,
        FAILED_TESTS,
        SECURITY_FINDINGS
    }

    record ReleaseObservation(
        double coverage,
        int failedTests,
        int securityFindings,
        Decision decision
    ) {
        ReleaseObservation {
            if (!Double.isFinite(coverage) || coverage < 0 || coverage > 100) {
                throw new IllegalArgumentException(
                    "Coverage must be finite and between 0 and 100."
                );
            }
            if (failedTests < 0 || securityFindings < 0) {
                throw new IllegalArgumentException(
                    "Test failures and security findings cannot be negative."
                );
            }
            Objects.requireNonNull(decision, "Decision is required.");
        }

        double value(Feature feature) {
            return switch (feature) {
                case COVERAGE -> coverage;
                case FAILED_TESTS -> failedTests;
                case SECURITY_FINDINGS -> securityFindings;
            };
        }
    }

    record ClassDistribution(int eligible, int manualReview) {
        ClassDistribution {
            if (eligible < 0 || manualReview < 0 ||
                eligible + manualReview == 0) {
                throw new IllegalArgumentException(
                    "Class distribution must contain observations."
                );
            }
        }

        int total() {
            return eligible + manualReview;
        }

        Decision majority() {
            // Ties are resolved toward manual review to avoid optimistic
            // predictions when the node has no clear majority.
            return eligible > manualReview
                ? Decision.ELIGIBLE
                : Decision.MANUAL_REVIEW;
        }

        Map<Decision, Double> frequencies() {
            Map<Decision, Double> result = new LinkedHashMap<>();
            result.put(Decision.ELIGIBLE, eligible / (double) total());
            result.put(Decision.MANUAL_REVIEW, manualReview / (double) total());
            return Map.copyOf(result);
        }
    }

    static final class Split {
        final Feature feature;
        final double threshold;
        final double gain;
        final List<ReleaseObservation> left;
        final List<ReleaseObservation> right;

        Split(
            Feature feature,
            double threshold,
            double gain,
            List<ReleaseObservation> left,
            List<ReleaseObservation> right
        ) {
            this.feature = feature;
            this.threshold = threshold;
            this.gain = gain;
            this.left = List.copyOf(left);
            this.right = List.copyOf(right);
        }
    }

    static final class Node {
        final ClassDistribution distribution;
        final double impurity;
        final int depth;
        Feature feature;
        double threshold;
        double gain;
        Node left;
        Node right;

        Node(ClassDistribution distribution, double impurity, int depth) {
            this.distribution = distribution;
            this.impurity = impurity;
            this.depth = depth;
        }

        boolean isLeaf() {
            return feature == null;
        }
    }

    static final class DecisionTree {
        private final Criterion criterion;
        private final int maxDepth;
        private final int minSamplesLeaf;
        private final double minGain;
        private Node root;
        private final Map<Feature, Double> weightedGain =
            new TreeMap<>(Comparator.comparing(Enum::name));
        private int trainingSize;

        DecisionTree(
            Criterion criterion,
            int maxDepth,
            int minSamplesLeaf,
            double minGain
        ) {
            this.criterion = Objects.requireNonNull(criterion);
            if (maxDepth < 0 || minSamplesLeaf < 1 || minGain < 0) {
                throw new IllegalArgumentException(
                    "Invalid tree regularization parameters."
                );
            }
            this.maxDepth = maxDepth;
            this.minSamplesLeaf = minSamplesLeaf;
            this.minGain = minGain;
        }

        void fit(List<ReleaseObservation> trainingData) {
            if (trainingData == null || trainingData.isEmpty()) {
                throw new IllegalArgumentException(
                    "Training data cannot be empty."
                );
            }

            trainingData.forEach(Objects::requireNonNull);
            trainingSize = trainingData.size();
            weightedGain.clear();
            for (Feature feature : Feature.values()) {
                weightedGain.put(feature, 0.0);
            }

            root = build(List.copyOf(trainingData), 0);
        }

        private ClassDistribution distribution(
            List<ReleaseObservation> observations
        ) {
            int eligible = 0;
            int review = 0;

            for (ReleaseObservation observation : observations) {
                if (observation.decision() == Decision.ELIGIBLE) {
                    eligible++;
                } else {
                    review++;
                }
            }

            return new ClassDistribution(eligible, review);
        }

        private double impurity(ClassDistribution distribution) {
            double total = distribution.total();
            double pEligible = distribution.eligible() / total;
            double pReview = distribution.manualReview() / total;

            if (criterion == Criterion.GINI) {
                return 1.0 - pEligible * pEligible - pReview * pReview;
            }

            double result = 0.0;
            for (double probability : new double[]{pEligible, pReview}) {
                if (probability > 0.0) {
                    result -= probability * (Math.log(probability) / Math.log(2));
                }
            }
            return result;
        }

        private Node build(List<ReleaseObservation> observations, int depth) {
            ClassDistribution distribution = distribution(observations);
            Node node = new Node(distribution, impurity(distribution), depth);

            if (depth >= maxDepth ||
                node.impurity == 0.0 ||
                observations.size() < minSamplesLeaf * 2) {
                return node;
            }

            Optional<Split> candidate = findBestSplit(observations);
            if (candidate.isEmpty() || candidate.get().gain <= minGain) {
                return node;
            }

            Split split = candidate.get();
            node.feature = split.feature;
            node.threshold = split.threshold;
            node.gain = split.gain;

            double weight = observations.size() / (double) trainingSize;
            weightedGain.merge(split.feature, weight * split.gain, Double::sum);

            node.left = build(split.left, depth + 1);
            node.right = build(split.right, depth + 1);
            return node;
        }

        private Optional<Split> findBestSplit(
            List<ReleaseObservation> observations
        ) {
            ClassDistribution parent = distribution(observations);
            double parentImpurity = impurity(parent);
            Split best = null;

            for (Feature feature : Feature.values()) {
                List<Double> values = observations.stream()
                    .map(observation -> observation.value(feature))
                    .distinct()
                    .sorted()
                    .toList();

                for (int i = 0; i + 1 < values.size(); i++) {
                    double lower = values.get(i);
                    double upper = values.get(i + 1);
                    double threshold = lower + (upper - lower) / 2.0;

                    List<ReleaseObservation> left = new ArrayList<>();
                    List<ReleaseObservation> right = new ArrayList<>();

                    for (ReleaseObservation observation : observations) {
                        if (observation.value(feature) <= threshold) {
                            left.add(observation);
                        } else {
                            right.add(observation);
                        }
                    }

                    if (left.size() < minSamplesLeaf ||
                        right.size() < minSamplesLeaf) {
                        continue;
                    }

                    double weightedChildImpurity =
                        left.size() / (double) observations.size()
                            * impurity(distribution(left))
                        + right.size() / (double) observations.size()
                            * impurity(distribution(right));

                    double gain = parentImpurity - weightedChildImpurity;

                    // Feature enumeration and ascending thresholds make the
                    // first equal-scoring split deterministic.
                    if (best == null || gain > best.gain + 1e-15) {
                        best = new Split(
                            feature,
                            threshold,
                            gain,
                            left,
                            right
                        );
                    }
                }
            }

            return Optional.ofNullable(best);
        }

        Decision predict(ReleaseObservation observation) {
            Objects.requireNonNull(observation);
            if (root == null) {
                throw new IllegalStateException("Fit the tree before prediction.");
            }

            Node current = root;
            while (!current.isLeaf()) {
                current = observation.value(current.feature) <= current.threshold
                    ? current.left
                    : current.right;
            }
            return current.distribution.majority();
        }

        Map<Decision, Double> predictFrequencies(
            ReleaseObservation observation
        ) {
            Node current = root;
            if (current == null) {
                throw new IllegalStateException("Fit the tree before prediction.");
            }

            while (!current.isLeaf()) {
                current = observation.value(current.feature) <= current.threshold
                    ? current.left
                    : current.right;
            }
            return current.distribution.frequencies();
        }

        Map<Feature, Double> featureImportance() {
            double total = weightedGain.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

            Map<Feature, Double> result = new LinkedHashMap<>();
            for (Feature feature : Feature.values()) {
                double raw = weightedGain.get(feature);
                result.put(feature, total > 0 ? raw / total : 0.0);
            }
            return Map.copyOf(result);
        }

        String render() {
            if (root == null) {
                throw new IllegalStateException("No fitted tree is available.");
            }
            StringBuilder output = new StringBuilder();
            renderNode(root, "", output);
            return output.toString();
        }

        private void renderNode(Node node, String prefix, StringBuilder output) {
            if (node.isLeaf()) {
                output.append(prefix)
                    .append("predict=").append(node.distribution.majority())
                    .append(", samples=").append(node.distribution.total())
                    .append(", impurity=")
                    .append(String.format("%.4f", node.impurity))
                    .append('\n');
                return;
            }

            output.append(prefix)
                .append("if ").append(node.feature)
                .append(" <= ").append(String.format("%.3f", node.threshold))
                .append(", gain=").append(String.format("%.4f", node.gain))
                .append('\n');

            renderNode(node.left, prefix + "  then ", output);
            renderNode(node.right, prefix + "  else ", output);
        }
    }

    static List<ReleaseObservation> createHistory() {
        List<ReleaseObservation> history = new ArrayList<>();

        for (double coverage : new double[]{44, 59, 68, 77, 85, 93, 98}) {
            for (int failedTests : new int[]{0, 1, 3}) {
                for (int securityFindings : new int[]{0, 1}) {
                    boolean risk = coverage < 75 ||
                        failedTests > 0 ||
                        securityFindings > 0;

                    history.add(new ReleaseObservation(
                        coverage,
                        failedTests,
                        securityFindings,
                        risk ? Decision.MANUAL_REVIEW : Decision.ELIGIBLE
                    ));
                }
            }
        }

        return List.copyOf(history);
    }

    public static void main(String[] args) {
        ClassDistribution balanced = new ClassDistribution(5, 5);
        ClassDistribution pure = new ClassDistribution(10, 0);

        DecisionTree entropyTree = new DecisionTree(
            Criterion.ENTROPY, 4, 2, 1e-12
        );
        DecisionTree giniTree = new DecisionTree(
            Criterion.GINI, 4, 2, 1e-12
        );

        List<ReleaseObservation> history = createHistory();
        entropyTree.fit(history);
        giniTree.fit(history);

        System.out.printf(
            "Balanced-node entropy: %.4f%n",
            entropyTree.impurity(balanced)
        );
        System.out.printf(
            "Pure-node Gini impurity: %.4f%n",
            giniTree.impurity(pure)
        );

        System.out.println("\nLearned enterprise release tree");
        System.out.print(entropyTree.render());

        ReleaseObservation candidate = new ReleaseObservation(
            89, 0, 0, Decision.ELIGIBLE
        );

        System.out.println("Candidate decision: " + entropyTree.predict(candidate));
        System.out.println(
            "Leaf class frequencies: " +
            entropyTree.predictFrequencies(candidate)
        );
        System.out.println(
            "Normalized feature importance: " +
            entropyTree.featureImportance()
        );

        try {
            new ReleaseObservation(140, 0, 0, Decision.ELIGIBLE);
        } catch (IllegalArgumentException exception) {
            System.out.println(
                "Invalid coverage rejected: " + exception.getMessage()
            );
        }
    }
}
