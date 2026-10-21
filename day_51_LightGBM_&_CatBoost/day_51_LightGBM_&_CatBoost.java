import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/*
 * Java 17 enterprise-style model for efficient gradient boosting.
 *
 * The domain is customer churn prediction. The implementation separates:
 *   - categorical preprocessing
 *   - ordered target statistics
 *   - histogram split discovery
 *   - second-order boosting
 *   - prediction and validation
 *
 * It is deliberately dependency-free.
 */
public class EfficientBoostingDemo {

    record Customer(
        double age,
        double income,
        double tenure,
        String region,
        String plan,
        int churn
    ) {
        Customer {
            if (age < 0 || income < 0 || tenure < 0) {
                throw new IllegalArgumentException("Numeric attributes cannot be negative.");
            }

            if (churn != 0 && churn != 1) {
                throw new IllegalArgumentException("Churn must be binary.");
            }

            Objects.requireNonNull(region, "region");
            Objects.requireNonNull(plan, "plan");
        }
    }

    record TrainingExample(double[] features, int target) {}

    record Prediction(double probability, int label) {}

    interface FeatureEncoder {
        double[] encode(Customer customer);
        List<double[]> fitTransform(List<Customer> customers);
    }

    static final class OrderedCategoricalEncoder implements FeatureEncoder {
        private final double priorWeight;
        private final Map<String, Statistics> regionStatistics = new HashMap<>();
        private final Map<String, Statistics> planStatistics = new HashMap<>();
        private double globalMean;

        private static final class Statistics {
            double sum;
            int count;

            void add(int target) {
                sum += target;
                count++;
            }
        }

        OrderedCategoricalEncoder(double priorWeight) {
            if (priorWeight <= 0) {
                throw new IllegalArgumentException("priorWeight must be positive.");
            }

            this.priorWeight = priorWeight;
        }

        @Override
        public List<double[]> fitTransform(List<Customer> customers) {
            if (customers.isEmpty()) {
                throw new IllegalArgumentException("Training data cannot be empty.");
            }

            globalMean = customers.stream()
                .mapToInt(Customer::churn)
                .average()
                .orElseThrow();

            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < customers.size(); i++) {
                order.add(i);
            }

            Collections.shuffle(order, new Random(42));

            List<double[]> result = new ArrayList<>(
                Collections.nCopies(customers.size(), null)
            );

            /*
             * The current target is deliberately added after the current
             * feature vector is created. This makes the statistic ordered.
             */
            for (int index : order) {
                Customer customer = customers.get(index);

                double regionValue =
                    statistic(regionStatistics, customer.region());

                double planValue =
                    statistic(planStatistics, customer.plan());

                result.set(index, new double[] {
                    customer.age(),
                    customer.income(),
                    customer.tenure(),
                    regionValue,
                    planValue
                });

                regionStatistics
                    .computeIfAbsent(customer.region(), key -> new Statistics())
                    .add(customer.churn());

                planStatistics
                    .computeIfAbsent(customer.plan(), key -> new Statistics())
                    .add(customer.churn());
            }

            return result;
        }

        private double statistic(
            Map<String, Statistics> map,
            String category
        ) {
            Statistics statistics = map.get(category);

            if (statistics == null) {
                return globalMean;
            }

            return (
                statistics.sum + priorWeight * globalMean
            ) / (
                statistics.count + priorWeight
            );
        }

        @Override
        public double[] encode(Customer customer) {
            return new double[] {
                customer.age(),
                customer.income(),
                customer.tenure(),
                statistic(regionStatistics, customer.region()),
                statistic(planStatistics, customer.plan())
            };
        }
    }

    static final class HistogramTree {
        private static final class Node {
            double value;
            int feature = -1;
            double threshold;
            Node left;
            Node right;
        }

        private final int maxDepth;
        private final int maxBins;
        private final int minLeaf;

        private final List<double[]> binEdges = new ArrayList<>();
        private Node root;

        HistogramTree(int maxDepth, int maxBins, int minLeaf) {
            if (maxDepth <= 0 || maxBins < 2 || minLeaf <= 0) {
                throw new IllegalArgumentException("Invalid tree parameters.");
            }

            this.maxDepth = maxDepth;
            this.maxBins = maxBins;
            this.minLeaf = minLeaf;
        }

        void fit(
            List<double[]> features,
            double[] gradients,
            double[] hessians
        ) {
            if (features.isEmpty()) {
                throw new IllegalArgumentException("Cannot fit an empty tree.");
            }

            if (
                features.size() != gradients.length ||
                features.size() != hessians.length
            ) {
                throw new IllegalArgumentException("Training arrays have inconsistent sizes.");
            }

            buildEdges(features);

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < features.size(); i++) {
                indices.add(i);
            }

            root = grow(
                features,
                gradients,
                hessians,
                indices,
                0
            );
        }

        private void buildEdges(List<double[]> features) {
            binEdges.clear();

            for (int feature = 0; feature < features.get(0).length; feature++) {
                List<Double> values = new ArrayList<>();

                for (double[] row : features) {
                    values.add(row[feature]);
                }

                values.sort(Comparator.naturalOrder());

                List<Double> unique = new ArrayList<>();

                for (double value : values) {
                    if (
                        unique.isEmpty() ||
                        Double.compare(unique.get(unique.size() - 1), value) != 0
                    ) {
                        unique.add(value);
                    }
                }

                List<Double> edges = new ArrayList<>();

                if (unique.size() <= maxBins) {
                    edges.addAll(unique);
                } else {
                    for (int i = 1; i < maxBins; i++) {
                        int position =
                            Math.min(
                                unique.size() - 1,
                                Math.max(
                                    0,
                                    i * unique.size() / maxBins
                                )
                            );

                        edges.add(unique.get(position));
                    }
                }

                binEdges.add(
                    edges.stream()
                        .distinct()
                        .mapToDouble(Double::doubleValue)
                        .collect(
                            () -> new ArrayList<Double>(),
                            List::add,
                            List::addAll
                        )
                );
            }
        }

        private int bucket(int feature, double value) {
            List<Double> edges = binEdges.get(feature);
            int bucket = 0;

            while (
                bucket < edges.size() &&
                value > edges.get(bucket)
            ) {
                bucket++;
            }

            return bucket;
        }

        private double gain(
            double[] gradients,
            double[] hessians,
            List<Integer> left,
            List<Integer> right
        ) {
            if (left.size() < minLeaf || right.size() < minLeaf) {
                return Double.NEGATIVE_INFINITY;
            }

            return 0.5 * score(gradients, hessians, left)
                + 0.5 * score(gradients, hessians, right);
        }

        private double score(
            double[] gradients,
            double[] hessians,
            List<Integer> indices
        ) {
            double gradientSum = 0;
            double hessianSum = 0;

            for (int index : indices) {
                gradientSum += gradients[index];
                hessianSum += hessians[index];
            }

            return gradientSum * gradientSum / (hessianSum + 1.0);
        }

        private Node grow(
            List<double[]> features,
            double[] gradients,
            double[] hessians,
            List<Integer> indices,
            int depth
        ) {
            double gradientSum = 0;
            double hessianSum = 0;

            for (int index : indices) {
                gradientSum += gradients[index];
                hessianSum += hessians[index];
            }

            Node node = new Node();
            node.value = -gradientSum / (hessianSum + 1.0);

            if (
                depth >= maxDepth ||
                indices.size() < minLeaf * 2
            ) {
                return node;
            }

            double bestGain = 0;
            int bestFeature = -1;
            double bestThreshold = 0;
            List<Integer> bestLeft = null;
            List<Integer> bestRight = null;

            for (int feature = 0; feature < features.get(0).length; feature++) {
                Map<Integer, List<Integer>> buckets = new HashMap<>();

                for (int index : indices) {
                    int bucket = bucket(feature, features.get(index)[feature]);

                    buckets
                        .computeIfAbsent(bucket, key -> new ArrayList<>())
                        .add(index);
                }

                List<Integer> orderedBuckets =
                    new ArrayList<>(buckets.keySet());

                orderedBuckets.sort(Comparator.naturalOrder());

                if (orderedBuckets.size() < 2) {
                    continue;
                }

                List<Integer> left = new ArrayList<>();

                for (
                    int position = 0;
                    position < orderedBuckets.size() - 1;
                    position++
                ) {
                    left.addAll(
                        buckets.get(orderedBuckets.get(position))
                    );

                    List<Integer> right = new ArrayList<>();

                    for (
                        int next = position + 1;
                        next < orderedBuckets.size();
                        next++
                    ) {
                        right.addAll(
                            buckets.get(orderedBuckets.get(next))
                        );
                    }

                    double currentGain =
                        gain(
                            gradients,
                            hessians,
                            left,
                            right
                        );

                    if (currentGain > bestGain) {
                        bestGain = currentGain;
                        bestFeature = feature;
                        bestLeft = new ArrayList<>(left);
                        bestRight = right;

                        int currentBucket =
                            orderedBuckets.get(position);

                        List<Double> edges =
                            binEdges.get(feature);

                        bestThreshold =
                            currentBucket < edges.size()
                                ? edges.get(currentBucket)
                                : features.get(
                                    indices.get(indices.size() - 1)
                                  )[feature];
                    }
                }
            }

            if (bestFeature == -1) {
                return node;
            }

            node.feature = bestFeature;
            node.threshold = bestThreshold;

            node.left = grow(
                features,
                gradients,
                hessians,
                bestLeft,
                depth + 1
            );

            node.right = grow(
                features,
                gradients,
                hessians,
                bestRight,
                depth + 1
            );

            return node;
        }

        double predict(double[] features) {
            if (root == null) {
                throw new IllegalStateException("Tree has not been fitted.");
            }

            Node current = root;

            while (current.feature != -1) {
                current =
                    features[current.feature] <= current.threshold
                        ? current.left
                        : current.right;
            }

            return current.value;
        }
    }

    static final class BoostingModel {
        private final int rounds;
        private final double learningRate;
        private final int maxDepth;
        private final int maxBins;

        private final List<HistogramTree> trees = new ArrayList<>();
        private double baseScore;

        BoostingModel(
            int rounds,
            double learningRate,
            int maxDepth,
            int maxBins
        ) {
            this.rounds = rounds;
            this.learningRate = learningRate;
            this.maxDepth = maxDepth;
            this.maxBins = maxBins;
        }

        void fit(List<double[]> features, List<Integer> targets) {
            if (
                features.isEmpty() ||
                features.size() != targets.size()
            ) {
                throw new IllegalArgumentException("Invalid training data.");
            }

            double positiveRate =
                targets.stream()
                    .mapToInt(Integer::intValue)
                    .average()
                    .orElseThrow();

            positiveRate =
                Math.max(1e-6, Math.min(1 - 1e-6, positiveRate));

            baseScore =
                Math.log(positiveRate / (1 - positiveRate));

            double[] raw = new double[features.size()];
            java.util.Arrays.fill(raw, baseScore);

            trees.clear();

            for (int round = 0; round < rounds; round++) {
                double[] gradients = new double[features.size()];
                double[] hessians = new double[features.size()];

                for (int i = 0; i < features.size(); i++) {
                    double probability = sigmoid(raw[i]);

                    gradients[i] =
                        probability - targets.get(i);

                    hessians[i] =
                        Math.max(
                            1e-6,
                            probability * (1 - probability)
                        );
                }

                HistogramTree tree =
                    new HistogramTree(
                        maxDepth,
                        maxBins,
                        2
                    );

                tree.fit(
                    features,
                    gradients,
                    hessians
                );

                trees.add(tree);

                for (int i = 0; i < features.size(); i++) {
                    raw[i] +=
                        learningRate * tree.predict(features.get(i));
                }
            }
        }

        List<Prediction> predict(List<double[]> features) {
            List<Prediction> predictions = new ArrayList<>();

            for (double[] row : features) {
                double raw = baseScore;

                for (HistogramTree tree : trees) {
                    raw += learningRate * tree.predict(row);
                }

                double probability = sigmoid(raw);

                predictions.add(
                    new Prediction(
                        probability,
                        probability >= 0.5 ? 1 : 0
                    )
                );
            }

            return predictions;
        }
    }

    private static double sigmoid(double value) {
        value = Math.max(-35, Math.min(35, value));
        return 1.0 / (1.0 + Math.exp(-value));
    }

    private static List<Customer> sampleCustomers() {
        return List.of(
            new Customer(23, 31000, 4, "north", "basic", 1),
            new Customer(25, 42000, 8, "south", "basic", 1),
            new Customer(29, 58000, 18, "west", "pro", 0),
            new Customer(31, 61000, 24, "north", "pro", 0),
            new Customer(35, 72000, 30, "east", "pro", 0),
            new Customer(41, 83000, 44, "west", "enterprise", 0),
            new Customer(48, 91000, 52, "north", "enterprise", 0),
            new Customer(52, 39000, 9, "south", "basic", 1),
            new Customer(27, 36000, 6, "east", "basic", 1),
            new Customer(33, 69000, 27, "south", "pro", 0),
            new Customer(38, 51000, 15, "east", "basic", 1),
            new Customer(45, 87000, 47, "west", "enterprise", 0),
            new Customer(22, 28000, 3, "north", "basic", 1),
            new Customer(57, 99000, 72, "north", "enterprise", 0),
            new Customer(30, 47000, 11, "west", "pro", 0),
            new Customer(36, 45000, 10, "south", "basic", 1),
            new Customer(43, 78000, 36, "east", "pro", 0),
            new Customer(50, 54000, 21, "south", "pro", 1),
            new Customer(28, 62000, 22, "north", "pro", 0),
            new Customer(39, 34000, 7, "east", "basic", 1)
        );
    }

    public static void main(String[] args) {
        List<Customer> customers = sampleCustomers();

        List<Customer> train =
            new ArrayList<>(customers.subList(0, 15));

        List<Customer> test =
            new ArrayList<>(customers.subList(15, customers.size()));

        OrderedCategoricalEncoder encoder =
            new OrderedCategoricalEncoder(5.0);

        List<double[]> trainFeatures =
            encoder.fitTransform(train);

        List<double[]> testFeatures =
            test.stream()
                .map(encoder::encode)
                .toList();

        List<Integer> trainTargets =
            train.stream()
                .map(Customer::churn)
                .toList();

        BoostingModel model =
            new BoostingModel(
                35,
                0.06,
                3,
                6
            );

        model.fit(trainFeatures, trainTargets);

        List<Prediction> predictions =
            model.predict(testFeatures);

        System.out.println(
            "Enterprise histogram boosting with ordered categorical statistics"
        );
        System.out.println(
            "---------------------------------------------------------------"
        );

        for (int i = 0; i < predictions.size(); i++) {
            Prediction prediction = predictions.get(i);

            System.out.printf(
                "customer=%d actual=%d probability=%.4f label=%d%n",
                i + 1,
                test.get(i).churn(),
                prediction.probability(),
                prediction.label()
            );
        }

        System.out.println();
        System.out.println("Governance rules:");
        System.out.println("- Ordered statistics must be fitted only on training data.");
        System.out.println("- Unknown production categories use the global prior.");
        System.out.println("- Histogram resolution controls accuracy versus computation.");
        System.out.println("- Tree depth and leaf constraints control model complexity.");
        System.out.println("- Production deployment requires validation and drift monitoring.");
    }
}
