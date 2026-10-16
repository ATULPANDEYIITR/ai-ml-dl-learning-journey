import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.TreeMap;

/**
 * Enterprise-oriented decision tree optimization for equipment-risk triage.
 *
 * Java 17 or later. The program models training constraints, explicit split
 * rules, cost-complexity pruning, model evaluation, and feature importance.
 */
public final class DecisionTreeOptimization {

    enum EquipmentState {
        NORMAL,
        INSPECTION_REQUIRED
    }

    record AssetObservation(double[] measurements, EquipmentState state) {
        AssetObservation {
            Objects.requireNonNull(measurements, "measurements");
            Objects.requireNonNull(state, "state");

            if (measurements.length == 0) {
                throw new IllegalArgumentException("At least one measurement is required.");
            }

            measurements = measurements.clone();

            for (double value : measurements) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Measurements must be finite.");
                }
            }
        }

        @Override
        public double[] measurements() {
            return measurements.clone();
        }
    }

    record TrainingPolicy(
        OptionalInt maximumDepth,
        int minimumSamplesSplit,
        int minimumSamplesLeaf,
        double minimumImpurityDecrease
    ) {
        TrainingPolicy {
            Objects.requireNonNull(maximumDepth, "maximumDepth");

            if (maximumDepth.isPresent() && maximumDepth.getAsInt() < 0) {
                throw new IllegalArgumentException("Maximum depth cannot be negative.");
            }
            if (minimumSamplesSplit < 2) {
                throw new IllegalArgumentException("Minimum split size must be at least two.");
            }
            if (minimumSamplesLeaf < 1) {
                throw new IllegalArgumentException("Minimum leaf size must be positive.");
            }
            if (!Double.isFinite(minimumImpurityDecrease) ||
                minimumImpurityDecrease < 0) {
                throw new IllegalArgumentException(
                    "Minimum impurity decrease must be finite and non-negative."
                );
            }
        }

        static TrainingPolicy controlled() {
            return new TrainingPolicy(OptionalInt.of(4), 8, 4, 0.002);
        }

        static TrainingPolicy unrestricted() {
            return new TrainingPolicy(OptionalInt.empty(), 2, 1, 0.0);
        }
    }

    static final class TreeNode {
        private EquipmentState prediction;
        private final Map<EquipmentState, Integer> classCounts;
        private final int sampleCount;
        private final double impurity;

        private int depth;
        private Integer featureIndex;
        private Double threshold;
        private double gain;
        private TreeNode left;
        private TreeNode right;

        TreeNode(
            EquipmentState prediction,
            Map<EquipmentState, Integer> classCounts,
            int sampleCount,
            double impurity,
            int depth
        ) {
            this.prediction = prediction;
            this.classCounts = Map.copyOf(classCounts);
            this.sampleCount = sampleCount;
            this.impurity = impurity;
            this.depth = depth;
        }

        boolean isLeaf() {
            return featureIndex == null;
        }
    }

    static final class TreeModel {
        private final TrainingPolicy policy;
        private final int featureCount;
        private TreeNode root;
        private double[] featureImportances;

        TreeModel(TrainingPolicy policy, int featureCount) {
            this.policy = policy;
            this.featureCount = featureCount;
            this.featureImportances = new double[featureCount];
        }

        void fit(List<AssetObservation> observations) {
            validateDataset(observations, featureCount);
            root = build(observations, 0);
            refreshFeatureImportances();
        }

        private TreeNode build(List<AssetObservation> observations, int depth) {
            Map<EquipmentState, Integer> counts = counts(observations);
            EquipmentState prediction = majority(counts);
            double impurity = gini(observations);

            TreeNode node = new TreeNode(
                prediction, counts, observations.size(), impurity, depth
            );

            if (impurity == 0.0 ||
                (policy.maximumDepth().isPresent() &&
                 depth >= policy.maximumDepth().getAsInt()) ||
                observations.size() < policy.minimumSamplesSplit()) {
                return node;
            }

            SplitCandidate split = findBestSplit(observations);

            if (split == null ||
                split.gain() <= 0.0 ||
                split.gain() < policy.minimumImpurityDecrease()) {
                return node;
            }

            node.featureIndex = split.featureIndex();
            node.threshold = split.threshold();
            node.gain = split.gain();
            node.left = build(split.left(), depth + 1);
            node.right = build(split.right(), depth + 1);

            return node;
        }

        private record SplitCandidate(
            int featureIndex,
            double threshold,
            double gain,
            List<AssetObservation> left,
            List<AssetObservation> right
        ) {}

        private SplitCandidate findBestSplit(List<AssetObservation> observations) {
            double parentImpurity = gini(observations);
            SplitCandidate best = null;

            for (int feature = 0; feature < featureCount; feature++) {
                final int featureIndex = feature;

                double[] sortedValues = observations.stream()
                    .mapToDouble(sample -> sample.measurements()[featureIndex])
                    .distinct()
                    .sorted()
                    .toArray();

                for (int i = 0; i + 1 < sortedValues.length; i++) {
                    double threshold =
                        sortedValues[i] + (sortedValues[i + 1] - sortedValues[i]) / 2.0;

                    List<AssetObservation> left = new ArrayList<>();
                    List<AssetObservation> right = new ArrayList<>();

                    for (AssetObservation observation : observations) {
                        if (observation.measurements()[feature] <= threshold) {
                            left.add(observation);
                        } else {
                            right.add(observation);
                        }
                    }

                    if (left.size() < policy.minimumSamplesLeaf() ||
                        right.size() < policy.minimumSamplesLeaf()) {
                        continue;
                    }

                    double weightedImpurity =
                        ((double) left.size() / observations.size()) * gini(left) +
                        ((double) right.size() / observations.size()) * gini(right);

                    double gain = parentImpurity - weightedImpurity;

                    if (best == null || gain > best.gain()) {
                        best = new SplitCandidate(
                            feature, threshold, gain, left, right
                        );
                    }
                }
            }

            return best;
        }

        EquipmentState predict(double[] measurements) {
            requireFitted();

            if (measurements == null || measurements.length != featureCount) {
                throw new IllegalArgumentException("Unexpected measurement count.");
            }

            for (double measurement : measurements) {
                if (!Double.isFinite(measurement)) {
                    throw new IllegalArgumentException("Measurements must be finite.");
                }
            }

            TreeNode current = root;
            while (!current.isLeaf()) {
                current = measurements[current.featureIndex] <= current.threshold
                    ? current.left
                    : current.right;
            }

            return current.prediction;
        }

        void prune(double alpha) {
            requireFitted();

            if (!Double.isFinite(alpha) || alpha < 0.0) {
                throw new IllegalArgumentException(
                    "Pruning alpha must be finite and non-negative."
                );
            }

            root = pruneNode(root, alpha);
            refreshDepths(root, 0);
            refreshFeatureImportances();
        }

        private TreeNode pruneNode(TreeNode node, double alpha) {
            if (node.isLeaf()) return node;

            node.left = pruneNode(node.left, alpha);
            node.right = pruneNode(node.right, alpha);

            int leaves = leafCount(node);
            if (leaves <= 1) return node;

            double leafRisk = node.sampleCount * node.impurity;
            double effectiveAlpha =
                (leafRisk - subtreeRisk(node)) / (leaves - 1);

            if (effectiveAlpha <= alpha) {
                node.featureIndex = null;
                node.threshold = null;
                node.left = null;
                node.right = null;
                node.gain = 0.0;
            }

            return node;
        }

        private double subtreeRisk(TreeNode node) {
            if (node.isLeaf()) {
                return node.sampleCount * node.impurity;
            }
            return subtreeRisk(node.left) + subtreeRisk(node.right);
        }

        private int leafCount(TreeNode node) {
            if (node.isLeaf()) return 1;
            return leafCount(node.left) + leafCount(node.right);
        }

        private int nodeCount(TreeNode node) {
            if (node == null) return 0;
            return 1 + nodeCount(node.left) + nodeCount(node.right);
        }

        private int maximumDepth(TreeNode node) {
            if (node.isLeaf()) return node.depth;
            return Math.max(maximumDepth(node.left), maximumDepth(node.right));
        }

        private void refreshDepths(TreeNode node, int depth) {
            node.depth = depth;
            if (!node.isLeaf()) {
                refreshDepths(node.left, depth + 1);
                refreshDepths(node.right, depth + 1);
            }
        }

        private void refreshFeatureImportances() {
            featureImportances = new double[featureCount];
            collectImportance(root);

            double total = Arrays.stream(featureImportances).sum();
            if (total > 0.0) {
                for (int i = 0; i < featureImportances.length; i++) {
                    featureImportances[i] /= total;
                }
            }
        }

        private void collectImportance(TreeNode node) {
            if (node.isLeaf()) return;

            featureImportances[node.featureIndex] += node.sampleCount * node.gain;
            collectImportance(node.left);
            collectImportance(node.right);
        }

        double[] featureImportances() {
            requireFitted();
            return featureImportances.clone();
        }

        int nodeCount() {
            requireFitted();
            return nodeCount(root);
        }

        int leafCount() {
            requireFitted();
            return leafCount(root);
        }

        int depth() {
            requireFitted();
            return maximumDepth(root);
        }

        private void requireFitted() {
            if (root == null) {
                throw new IllegalStateException("The model has not been fitted.");
            }
        }
    }

    private static Map<EquipmentState, Integer> counts(
        List<AssetObservation> observations
    ) {
        Map<EquipmentState, Integer> counts = new TreeMap<>(
            Comparator.comparing(Enum::name)
        );

        for (AssetObservation observation : observations) {
            counts.merge(observation.state(), 1, Integer::sum);
        }

        return counts;
    }

    private static EquipmentState majority(Map<EquipmentState, Integer> counts) {
        return counts.entrySet().stream()
            .max(Comparator.<Map.Entry<EquipmentState, Integer>>comparingInt(
                Map.Entry::getValue
            ).thenComparing(entry -> entry.getKey().name(), Comparator.reverseOrder()))
            .orElseThrow(() -> new IllegalArgumentException("Empty class counts."))
            .getKey();
    }

    private static double gini(List<AssetObservation> observations) {
        if (observations.isEmpty()) return 0.0;

        Map<EquipmentState, Integer> counts = counts(observations);
        double sumSquares = 0.0;

        for (int count : counts.values()) {
            double probability = (double) count / observations.size();
            sumSquares += probability * probability;
        }

        return 1.0 - sumSquares;
    }

    private static void validateDataset(
        List<AssetObservation> observations,
        int expectedFeatureCount
    ) {
        if (observations == null || observations.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty.");
        }

        for (AssetObservation observation : observations) {
            if (observation == null ||
                observation.measurements().length != expectedFeatureCount) {
                throw new IllegalArgumentException("Malformed training observation.");
            }
        }
    }

    private static List<AssetObservation> createDataset(int size, long seed) {
        java.util.Random random = new java.util.Random(seed);
        List<AssetObservation> observations = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            double vibration = 0.2 + random.nextDouble() * 8.8;
            double temperature = 30.0 + random.nextDouble() * 85.0;
            double serviceHours = random.nextDouble() * 20.0;
            double priorFaults = random.nextInt(6);

            double risk =
                0.95 * vibration +
                0.025 * temperature +
                0.035 * serviceHours +
                0.8 * priorFaults +
                random.nextGaussian() * 1.5;

            EquipmentState state = risk > 10.0
                ? EquipmentState.INSPECTION_REQUIRED
                : EquipmentState.NORMAL;

            observations.add(new AssetObservation(
                new double[]{vibration, temperature, serviceHours, priorFaults},
                state
            ));
        }

        return observations;
    }

    private static double accuracy(
        TreeModel model,
        List<AssetObservation> observations
    ) {
        if (observations.isEmpty()) {
            throw new IllegalArgumentException("Evaluation data cannot be empty.");
        }

        long correct = observations.stream()
            .filter(observation ->
                model.predict(observation.measurements()) == observation.state())
            .count();

        return (double) correct / observations.size();
    }

    private static void report(
        String title,
        TreeModel model,
        List<AssetObservation> validation
    ) {
        System.out.printf(
            "%s: validation accuracy=%.3f, nodes=%d, leaves=%d, depth=%d%n",
            title,
            accuracy(model, validation),
            model.nodeCount(),
            model.leafCount(),
            model.depth()
        );
    }

    public static void main(String[] args) {
        List<AssetObservation> data = createDataset(320, 20261016L);

        int boundary = data.size() * 4 / 5;
        List<AssetObservation> training = new ArrayList<>(data.subList(0, boundary));
        List<AssetObservation> validation = new ArrayList<>(data.subList(boundary, data.size()));

        TreeModel unrestricted = new TreeModel(
            TrainingPolicy.unrestricted(), 4
        );
        unrestricted.fit(training);

        TreeModel controlled = new TreeModel(
            TrainingPolicy.controlled(), 4
        );
        controlled.fit(training);

        TreeModel postPruned = new TreeModel(
            new TrainingPolicy(OptionalInt.empty(), 4, 2, 0.0), 4
        );
        postPruned.fit(training);

        System.out.println("Enterprise Equipment Risk Classification");
        System.out.println("Training records: " + training.size());
        System.out.println("Validation records: " + validation.size());

        report("Unrestricted model", unrestricted, validation);
        report("Pre-pruned model", controlled, validation);

        postPruned.prune(0.012);
        report("Post-pruned model", postPruned, validation);

        String[] featureNames = {
            "vibration_mm_s",
            "temperature_c",
            "service_hours",
            "previous_fault_count"
        };

        System.out.println("\nFeature importance:");
        double[] importances = controlled.featureImportances();

        for (int i = 0; i < featureNames.length; i++) {
            System.out.printf(
                "  %-26s %.4f%n",
                featureNames[i],
                importances[i]
            );
        }

        double[] incomingMeasurements = {7.3, 98.0, 13.0, 3.0};
        System.out.println(
            "\nIncoming asset classification: " +
            controlled.predict(incomingMeasurements)
        );

        try {
            controlled.predict(new double[]{7.3, Double.NaN, 13.0, 3.0});
        } catch (IllegalArgumentException error) {
            System.out.println("Rejected invalid telemetry: " + error.getMessage());
        }

        System.out.println(
            "\nOperational policy: select complexity using validation data, " +
            "measure missed failures separately, and review feature importance " +
            "as a property of the fitted model rather than a causal explanation."
        );
    }
}
