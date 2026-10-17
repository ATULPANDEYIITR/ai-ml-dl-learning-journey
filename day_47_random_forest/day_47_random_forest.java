import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

/**
 * Random Forest for enterprise equipment maintenance triage.
 *
 * Compile: javac RandomForestDemo.java
 * Run:     java RandomForestDemo
 *
 * Uses Java 17 and the standard library only.
 */
public class RandomForestDemo {

    enum MaintenanceDecision {
        NORMAL,
        MAINTENANCE_REQUIRED
    }

    record Inspection(
        double vibration,
        double temperature,
        double operatingHours,
        double pressureDeviation,
        MaintenanceDecision decision
    ) {
        Inspection {
            if (!Double.isFinite(vibration)
                    || !Double.isFinite(temperature)
                    || !Double.isFinite(operatingHours)
                    || !Double.isFinite(pressureDeviation)) {
                throw new IllegalArgumentException("Features must be finite.");
            }
            Objects.requireNonNull(decision, "decision");
        }

        double[] features() {
            return new double[] {
                vibration, temperature, operatingHours, pressureDeviation
            };
        }
    }

    record TrainingPolicy(
        int treeCount,
        int maxDepth,
        int minimumLeafSize,
        int featureSubsetSize,
        long seed
    ) {
        TrainingPolicy {
            if (treeCount < 1 || maxDepth < 0
                    || minimumLeafSize < 1 || featureSubsetSize < 1) {
                throw new IllegalArgumentException("Invalid training policy.");
            }
        }
    }

    static final class Node {
        final MaintenanceDecision prediction;
        final int featureIndex;
        final double threshold;
        final Node left;
        final Node right;

        Node(MaintenanceDecision prediction) {
            this(prediction, -1, 0.0, null, null);
        }

        Node(
            MaintenanceDecision prediction,
            int featureIndex,
            double threshold,
            Node left,
            Node right
        ) {
            this.prediction = prediction;
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }

        boolean isLeaf() {
            return featureIndex == -1;
        }
    }

    static final class DecisionTree {
        private final TrainingPolicy policy;
        private final Random random;
        private final Map<Integer, Double> importance = new HashMap<>();
        private Node root;

        DecisionTree(TrainingPolicy policy, long treeSeed) {
            this.policy = policy;
            this.random = new Random(treeSeed);
        }

        void fit(List<Inspection> observations) {
            if (observations.isEmpty()) {
                throw new IllegalArgumentException("Training set is empty.");
            }
            importance.clear();
            root = grow(new ArrayList<>(observations), 0);
        }

        private Node grow(List<Inspection> rows, int depth) {
            List<MaintenanceDecision> labels = rows.stream()
                .map(Inspection::decision)
                .toList();

            MaintenanceDecision prediction = majority(labels);
            Set<MaintenanceDecision> distinct = new HashSet<>(labels);

            if (depth >= policy.maxDepth()
                    || rows.size() < 2 * policy.minimumLeafSize()
                    || distinct.size() == 1) {
                return new Node(prediction);
            }

            int featureCount = rows.get(0).features().length;
            if (policy.featureSubsetSize() > featureCount) {
                throw new IllegalArgumentException(
                    "Feature subset exceeds feature count."
                );
            }

            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < featureCount; i++) candidates.add(i);
            Collections.shuffle(candidates, random);
            candidates = candidates.subList(0, policy.featureSubsetSize());

            double parentGini = gini(labels);
            double bestGain = 0.0;
            int bestFeature = -1;
            double bestThreshold = 0.0;
            List<Inspection> bestLeft = null;
            List<Inspection> bestRight = null;

            for (int feature : candidates) {
                List<Double> values = rows.stream()
                    .map(row -> row.features()[feature])
                    .distinct()
                    .sorted()
                    .toList();

                for (int i = 0; i + 1 < values.size(); i++) {
                    double threshold = values.get(i)
                        + (values.get(i + 1) - values.get(i)) / 2.0;

                    List<Inspection> left = new ArrayList<>();
                    List<Inspection> right = new ArrayList<>();

                    for (Inspection row : rows) {
                        if (row.features()[feature] <= threshold) {
                            left.add(row);
                        } else {
                            right.add(row);
                        }
                    }

                    if (left.size() < policy.minimumLeafSize()
                            || right.size() < policy.minimumLeafSize()) {
                        continue;
                    }

                    double weightedGini =
                        (double) left.size() / rows.size()
                            * gini(left.stream().map(Inspection::decision).toList())
                        + (double) right.size() / rows.size()
                            * gini(right.stream().map(Inspection::decision).toList());

                    double gain = parentGini - weightedGini;
                    if (gain > bestGain + 1e-12) {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = threshold;
                        bestLeft = left;
                        bestRight = right;
                    }
                }
            }

            if (bestFeature == -1) return new Node(prediction);

            importance.merge(
                bestFeature,
                bestGain * rows.size(),
                Double::sum
            );

            return new Node(
                prediction,
                bestFeature,
                bestThreshold,
                grow(bestLeft, depth + 1),
                grow(bestRight, depth + 1)
            );
        }

        MaintenanceDecision predict(double[] features) {
            if (root == null) throw new IllegalStateException("Tree is not trained.");
            if (features.length != 4) {
                throw new IllegalArgumentException("Expected four features.");
            }

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Features must be finite.");
                }
            }

            Node current = root;
            while (!current.isLeaf()) {
                current = features[current.featureIndex] <= current.threshold
                    ? current.left
                    : current.right;
            }
            return current.prediction;
        }

        Map<Integer, Double> importance() {
            return Map.copyOf(importance);
        }
    }

    static final class RandomForestService {
        private final TrainingPolicy policy;
        private final List<DecisionTree> trees = new ArrayList<>();
        private final Map<Integer, Double> importance = new TreeMap<>();
        private double oobAccuracy = Double.NaN;

        RandomForestService(TrainingPolicy policy) {
            this.policy = policy;
        }

        void train(List<Inspection> rows) {
            if (rows.isEmpty()) throw new IllegalArgumentException("Empty dataset.");

            trees.clear();
            importance.clear();
            Map<Integer, List<MaintenanceDecision>> oobVotes = new HashMap<>();
            Random seedGenerator = new Random(policy.seed());

            for (int i = 0; i < policy.treeCount(); i++) {
                long treeSeed = seedGenerator.nextLong();
                Random sampler = new Random(treeSeed);
                List<Inspection> bootstrap = new ArrayList<>();
                Set<Integer> selected = new HashSet<>();

                for (int j = 0; j < rows.size(); j++) {
                    int index = sampler.nextInt(rows.size());
                    bootstrap.add(rows.get(index));
                    selected.add(index);
                }

                DecisionTree tree = new DecisionTree(policy, treeSeed ^ 0x5DEECE66DL);
                tree.fit(bootstrap);
                trees.add(tree);

                tree.importance().forEach(
                    (feature, value) -> importance.merge(feature, value, Double::sum)
                );

                // An observation is evaluated by this tree only when it did
                // not occur in this tree's bootstrap training sample.
                for (int j = 0; j < rows.size(); j++) {
                    if (!selected.contains(j)) {
                        oobVotes.computeIfAbsent(j, ignored -> new ArrayList<>())
                            .add(tree.predict(rows.get(j).features()));
                    }
                }
            }

            double total = importance.values().stream()
                .mapToDouble(Double::doubleValue).sum();

            if (total > 0) {
                importance.replaceAll((feature, value) -> value / total);
            }

            int evaluated = 0;
            int correct = 0;

            for (Map.Entry<Integer, List<MaintenanceDecision>> entry
                    : oobVotes.entrySet()) {
                evaluated++;
                if (majority(entry.getValue()) == rows.get(entry.getKey()).decision()) {
                    correct++;
                }
            }

            oobAccuracy = evaluated == 0
                ? Double.NaN
                : (double) correct / evaluated;
        }

        MaintenanceDecision predict(Inspection inspection) {
            if (trees.isEmpty()) throw new IllegalStateException("Forest is not trained.");

            List<MaintenanceDecision> votes = trees.stream()
                .map(tree -> tree.predict(inspection.features()))
                .toList();

            return majority(votes);
        }

        Map<Integer, Double> featureImportance() {
            return Map.copyOf(importance);
        }

        double outOfBagAccuracy() {
            return oobAccuracy;
        }
    }

    private static MaintenanceDecision majority(
        List<MaintenanceDecision> decisions
    ) {
        int maintenance = 0;
        int normal = 0;

        for (MaintenanceDecision decision : decisions) {
            if (decision == MaintenanceDecision.MAINTENANCE_REQUIRED) {
                maintenance++;
            } else {
                normal++;
            }
        }

        // Deterministic tie policy: choose NORMAL only when its vote count
        // is strictly greater; a tie raises the conservative alert.
        return maintenance >= normal
            ? MaintenanceDecision.MAINTENANCE_REQUIRED
            : MaintenanceDecision.NORMAL;
    }

    private static double gini(List<MaintenanceDecision> labels) {
        if (labels.isEmpty()) return 0.0;

        long maintenance = labels.stream()
            .filter(label -> label == MaintenanceDecision.MAINTENANCE_REQUIRED)
            .count();

        double probability = (double) maintenance / labels.size();
        return 1.0 - probability * probability
            - (1.0 - probability) * (1.0 - probability);
    }

    private static List<Inspection> generateDataset(int count, long seed) {
        Random random = new Random(seed);
        List<Inspection> rows = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            double vibration = random.nextDouble() * 10;
            double temperature = 35 + random.nextDouble() * 75;
            double hours = random.nextDouble() * 12000;
            double pressure = random.nextDouble() * 8;

            double risk =
                0.28 * vibration
                + 0.045 * Math.max(0, temperature - 70)
                + 0.00025 * hours
                + 0.65 * pressure
                + 0.12 * vibration * pressure
                + random.nextGaussian() * 0.8;

            MaintenanceDecision decision = risk > 8
                ? MaintenanceDecision.MAINTENANCE_REQUIRED
                : MaintenanceDecision.NORMAL;

            rows.add(new Inspection(
                vibration, temperature, hours, pressure, decision
            ));
        }

        Collections.shuffle(rows, random);
        return rows;
    }

    public static void main(String[] args) {
        List<Inspection> data = generateDataset(240, 2026);
        int split = (int) (data.size() * 0.8);

        List<Inspection> training = new ArrayList<>(data.subList(0, split));
        List<Inspection> testing = new ArrayList<>(data.subList(split, data.size()));

        TrainingPolicy policy = new TrainingPolicy(51, 6, 2, 2, 71);
        RandomForestService service = new RandomForestService(policy);
        service.train(training);

        long correct = testing.stream()
            .filter(row -> service.predict(row) == row.decision())
            .count();

        System.out.printf("Test accuracy: %.3f%n", (double) correct / testing.size());
        System.out.printf("OOB accuracy: %.3f%n", service.outOfBagAccuracy());

        String[] featureNames = {
            "Vibration", "Temperature", "Operating hours", "Pressure deviation"
        };

        System.out.println("\nFeature importance:");
        service.featureImportance().entrySet().stream()
            .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.printf(
                "%-22s %.4f%n",
                featureNames[entry.getKey()],
                entry.getValue()
            ));

        Inspection highRisk = new Inspection(
            8.2, 97.0, 9800, 6.5,
            MaintenanceDecision.MAINTENANCE_REQUIRED
        );

        System.out.println("\nHigh-risk asset assessment: "
            + service.predict(highRisk));

        try {
            service.predict(new Inspection(
                Double.NaN, 97.0, 9800, 6.5,
                MaintenanceDecision.NORMAL
            ));
        } catch (IllegalArgumentException exception) {
            System.out.println("Invalid measurement rejected: "
                + exception.getMessage());
        }
    }
}
