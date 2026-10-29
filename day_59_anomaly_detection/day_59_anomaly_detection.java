import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

/*
 * Enterprise Repository Telemetry Anomaly Governance
 *
 * The Java implementation models anomaly detection as an enterprise service:
 *
 * Repository -> Telemetry -> Detection Policies -> Review Risk -> Decision
 *
 * It uses explicit domain records, enums, policy interfaces, state
 * transitions, immutable configuration, validation, and service objects.
 */

public class AnomalyGovernance {

    enum DetectorType {
        ISOLATION_FOREST,
        LOCAL_OUTLIER_FACTOR,
        ONE_CLASS_SVM
    }

    enum RiskLevel {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }

    enum EvaluationState {
        RECEIVED,
        VALIDATED,
        SCORED,
        REVIEW_REQUIRED,
        ACCEPTED,
        REJECTED
    }

    record Telemetry(
        long id,
        List<Double> features,
        boolean knownAnomaly,
        String description
    ) {
        Telemetry {
            if (id <= 0) {
                throw new ValidationException("Telemetry ID must be positive.");
            }

            Objects.requireNonNull(features);
            Objects.requireNonNull(description);

            if (features.isEmpty()) {
                throw new ValidationException("Telemetry requires features.");
            }

            for (double value : features) {
                if (!Double.isFinite(value)) {
                    throw new ValidationException(
                        "Telemetry features must be finite."
                    );
                }
            }

            features = List.copyOf(features);
        }
    }

    record DetectorResult(
        DetectorType detector,
        List<Double> scores,
        List<Boolean> predictions,
        double threshold
    ) {}

    record GovernanceDecision(
        long telemetryId,
        EvaluationState state,
        RiskLevel risk,
        List<DetectorType> triggeringDetectors,
        String reason
    ) {}

    static final class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    static final class FeatureScaler {
        private List<Double> means = List.of();
        private List<Double> deviations = List.of();

        void fit(List<Telemetry> telemetry) {
            validateDimensions(telemetry);

            int dimensions = telemetry.getFirst().features().size();

            means = IntStream.range(0, dimensions)
                .mapToObj(column ->
                    telemetry.stream()
                        .mapToDouble(item -> item.features().get(column))
                        .average()
                        .orElseThrow()
                )
                .toList();

            deviations = IntStream.range(0, dimensions)
                .mapToObj(column -> {
                    double mean = means.get(column);

                    double variance =
                        telemetry.stream()
                            .mapToDouble(item -> {
                                double delta =
                                    item.features().get(column) - mean;
                                return delta * delta;
                            })
                            .average()
                            .orElse(0.0);

                    double deviation = Math.sqrt(variance);
                    return deviation > 1e-12 ? deviation : 1.0;
                })
                .toList();
        }

        List<List<Double>> transform(List<Telemetry> telemetry) {
            if (means.isEmpty()) {
                throw new ValidationException("Scaler is not fitted.");
            }

            return telemetry.stream()
                .map(item -> IntStream.range(0, means.size())
                    .mapToObj(i ->
                        (item.features().get(i) - means.get(i))
                        / deviations.get(i)
                    )
                    .toList()
                )
                .toList();
        }

        private void validateDimensions(List<Telemetry> telemetry) {
            if (telemetry.size() < 2) {
                throw new ValidationException(
                    "At least two telemetry observations are required."
                );
            }

            int dimensions = telemetry.getFirst().features().size();

            if (telemetry.stream()
                    .anyMatch(item ->
                        item.features().size() != dimensions)) {
                throw new ValidationException(
                    "Telemetry dimensions are inconsistent."
                );
            }
        }
    }

    interface Detector {
        DetectorType type();

        DetectorResult detect(List<List<Double>> rows);
    }

    static double distance(
        List<Double> a,
        List<Double> b
    ) {
        if (a.size() != b.size()) {
            throw new ValidationException("Dimension mismatch.");
        }

        double sum = 0.0;

        for (int i = 0; i < a.size(); i++) {
            double delta = a.get(i) - b.get(i);
            sum += delta * delta;
        }

        return Math.sqrt(sum);
    }

    static double quantile(List<Double> values, double q) {
        if (values.isEmpty() || q < 0 || q > 1) {
            throw new ValidationException("Invalid quantile.");
        }

        List<Double> sorted = new ArrayList<>(values);
        sorted.sort(Double::compareTo);

        double position = (sorted.size() - 1) * q;
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);

        if (lower == upper) {
            return sorted.get(lower);
        }

        double fraction = position - lower;

        return sorted.get(lower)
            + fraction * (sorted.get(upper) - sorted.get(lower));
    }

    static double harmonic(int n) {
        if (n <= 1) return 0;

        double result = 0;

        for (int i = 1; i <= n; i++) {
            result += 1.0 / i;
        }

        return result;
    }

    static double expectedPathLength(int n) {
        if (n <= 1) return 0;

        return 2 * harmonic(n - 1)
            - (2.0 * (n - 1) / n);
    }

    static final class IsolationForestDetector implements Detector {
        private final int treeCount;
        private final int sampleSize;
        private final double contamination;
        private final Random random;

        private final List<IsolationTree> trees = new ArrayList<>();
        private double normalizer;

        IsolationForestDetector(
            int treeCount,
            int sampleSize,
            double contamination,
            long seed
        ) {
            if (treeCount < 1 ||
                sampleSize < 2 ||
                contamination <= 0 ||
                contamination >= 0.5) {
                throw new ValidationException(
                    "Invalid Isolation Forest configuration."
                );
            }

            this.treeCount = treeCount;
            this.sampleSize = sampleSize;
            this.contamination = contamination;
            this.random = new Random(seed);
        }

        @Override
        public DetectorType type() {
            return DetectorType.ISOLATION_FOREST;
        }

        @Override
        public DetectorResult detect(List<List<Double>> rows) {
            if (rows.size() < 4) {
                throw new ValidationException(
                    "Isolation Forest requires at least four rows."
                );
            }

            int actualSample =
                Math.min(sampleSize, rows.size());

            normalizer = expectedPathLength(actualSample);

            trees.clear();

            for (int i = 0; i < treeCount; i++) {
                List<List<Double>> sample =
                    new ArrayList<>(rows);

                java.util.Collections.shuffle(
                    sample,
                    random
                );

                sample =
                    sample.subList(
                        0,
                        actualSample
                    );

                IsolationTree tree =
                    new IsolationTree(
                        (int) Math.ceil(
                            Math.log(actualSample) / Math.log(2)
                        ),
                        random
                    );

                tree.fit(sample);
                trees.add(tree);
            }

            List<Double> scores =
                rows.stream()
                    .map(row -> {
                        double averagePath =
                            trees.stream()
                                .mapToDouble(tree ->
                                    tree.pathLength(row)
                                )
                                .average()
                                .orElseThrow();

                        return Math.pow(
                            2,
                            -averagePath / normalizer
                        );
                    })
                    .toList();

            double threshold =
                quantile(
                    scores,
                    1.0 - contamination
                );

            List<Boolean> predictions =
                scores.stream()
                    .map(score -> score >= threshold)
                    .toList();

            return new DetectorResult(
                type(),
                scores,
                predictions,
                threshold
            );
        }
    }

    static final class IsolationTree {
        private final int maxDepth;
        private final Random random;
        private Node root;

        private static final class Node {
            int feature = -1;
            double split;
            int size;
            Node left;
            Node right;
        }

        IsolationTree(int maxDepth, Random random) {
            this.maxDepth = maxDepth;
            this.random = random;
        }

        void fit(List<List<Double>> rows) {
            root = build(rows, 0);
        }

        private Node build(
            List<List<Double>> rows,
            int depth
        ) {
            Node node = new Node();
            node.size = rows.size();

            if (rows.size() <= 1 ||
                depth >= maxDepth) {
                return node;
            }

            int feature =
                random.nextInt(rows.getFirst().size());

            double min =
                rows.stream()
                    .mapToDouble(row -> row.get(feature))
                    .min()
                    .orElseThrow();

            double max =
                rows.stream()
                    .mapToDouble(row -> row.get(feature))
                    .max()
                    .orElseThrow();

            if (max - min <= 1e-12) {
                return node;
            }

            double split =
                min + random.nextDouble() * (max - min);

            List<List<Double>> left =
                rows.stream()
                    .filter(row -> row.get(feature) < split)
                    .toList();

            List<List<Double>> right =
                rows.stream()
                    .filter(row -> row.get(feature) >= split)
                    .toList();

            if (left.isEmpty() || right.isEmpty()) {
                return node;
            }

            node.feature = feature;
            node.split = split;
            node.left = build(left, depth + 1);
            node.right = build(right, depth + 1);

            return node;
        }

        double pathLength(List<Double> row) {
            Node current = root;
            int depth = 0;

            while (
                current.feature >= 0 &&
                current.left != null &&
                current.right != null
            ) {
                depth++;

                current =
                    row.get(current.feature) < current.split
                        ? current.left
                        : current.right;
            }

            return depth + expectedPathLength(current.size);
        }
    }

    static final class LocalOutlierFactorDetector
        implements Detector {

        private final int k;
        private List<List<Double>> training = List.of();
        private List<Double> kDistances = List.of();
        private List<Double> localDensity = List.of();

        LocalOutlierFactorDetector(int k) {
            if (k < 2) {
                throw new ValidationException(
                    "LOF requires k >= 2."
                );
            }
            this.k = k;
        }

        @Override
        public DetectorType type() {
            return DetectorType.LOCAL_OUTLIER_FACTOR;
        }

        @Override
        public DetectorResult detect(List<List<Double>> rows) {
            if (rows.size() <= k) {
                throw new ValidationException(
                    "LOF requires more observations than k."
                );
            }

            training = rows;

            List<List<Neighbor>> neighborLists =
                IntStream.range(0, rows.size())
                    .mapToObj(this::trainingNeighbors)
                    .toList();

            kDistances =
                neighborLists.stream()
                    .map(list ->
                        list.getLast().distance)
                    .toList();

            localDensity =
                neighborLists.stream()
                    .map(list -> {
                        double reachability =
                            list.stream()
                                .mapToDouble(item ->
                                    Math.max(
                                        kDistances.get(item.index),
                                        item.distance
                                    )
                                )
                                .sum();

                        return list.size() / Math.max(
                            reachability,
                            1e-12
                        );
                    })
                    .toList();

            List<Double> scores =
                rows.stream()
                    .map(this::score)
                    .toList();

            double threshold = 1.5;

            List<Boolean> predictions =
                scores.stream()
                    .map(value -> value >= threshold)
                    .toList();

            return new DetectorResult(
                type(),
                scores,
                predictions,
                threshold
            );
        }

        private List<Neighbor> trainingNeighbors(int index) {
            return IntStream.range(0, training.size())
                .filter(i -> i != index)
                .mapToObj(i ->
                    new Neighbor(
                        i,
                        distance(
                            training.get(index),
                            training.get(i)
                        )
                    )
                )
                .sorted(Comparator.comparingDouble(
                    Neighbor::distance
                ))
                .limit(k)
                .toList();
        }

        private double score(List<Double> row) {
            List<Neighbor> neighbors =
                IntStream.range(0, training.size())
                    .mapToObj(i ->
                        new Neighbor(
                            i,
                            distance(row, training.get(i))
                        )
                    )
                    .sorted(Comparator.comparingDouble(
                        Neighbor::distance
                    ))
                    .limit(k)
                    .toList();

            double reachability =
                neighbors.stream()
                    .mapToDouble(item ->
                        Math.max(
                            kDistances.get(item.index),
                            item.distance
                        )
                    )
                    .sum();

            double queryDensity =
                neighbors.size() /
                Math.max(reachability, 1e-12);

            double neighborDensity =
                neighbors.stream()
                    .mapToDouble(item ->
                        localDensity.get(item.index)
                    )
                    .average()
                    .orElse(0);

            return neighborDensity /
                Math.max(queryDensity, 1e-12);
        }

        record Neighbor(int index, double distance) {}
    }

    static final class OneClassSVMDetector
        implements Detector {

        private final double gamma;
        private final double nu;
        private List<List<Double>> training = List.of();

        OneClassSVMDetector(double gamma, double nu) {
            if (gamma <= 0 ||
                nu <= 0 ||
                nu > 0.5) {
                throw new ValidationException(
                    "Invalid One-Class SVM configuration."
                );
            }

            this.gamma = gamma;
            this.nu = nu;
        }

        @Override
        public DetectorType type() {
            return DetectorType.ONE_CLASS_SVM;
        }

        @Override
        public DetectorResult detect(List<List<Double>> rows) {
            training = rows;

            List<Double> scores =
                rows.stream()
                    .map(this::score)
                    .toList();

            double threshold =
                quantile(
                    scores,
                    1.0 - nu
                );

            List<Boolean> predictions =
                scores.stream()
                    .map(score -> score >= threshold)
                    .toList();

            return new DetectorResult(
                type(),
                scores,
                predictions,
                threshold
            );
        }

        private double score(List<Double> row) {
            double similarity =
                training.stream()
                    .mapToDouble(train -> {
                        double squaredDistance =
                            IntStream.range(
                                0,
                                row.size()
                            )
                            .mapToDouble(i -> {
                                double delta =
                                    row.get(i) - train.get(i);
                                return delta * delta;
                            })
                            .sum();

                        return Math.exp(
                            -gamma * squaredDistance
                        );
                    })
                    .average()
                    .orElse(0);

            return 1.0 - similarity;
        }
    }

    interface RiskPolicy {
        RiskLevel classify(
            List<DetectorType> triggers
        );
    }

    static final class RepositoryRiskPolicy
        implements RiskPolicy {

        @Override
        public RiskLevel classify(
            List<DetectorType> triggers
        ) {
            return switch (triggers.size()) {
                case 0 -> RiskLevel.LOW;
                case 1 -> RiskLevel.MEDIUM;
                case 2 -> RiskLevel.HIGH;
                default -> RiskLevel.CRITICAL;
            };
        }
    }

    static final class GovernanceService {
        private final List<Detector> detectors;
        private final RiskPolicy riskPolicy;

        GovernanceService(
            List<Detector> detectors,
            RiskPolicy riskPolicy
        ) {
            if (detectors.isEmpty()) {
                throw new ValidationException(
                    "At least one detector is required."
                );
            }

            this.detectors = List.copyOf(detectors);
            this.riskPolicy = Objects.requireNonNull(riskPolicy);
        }

        List<GovernanceDecision> evaluate(
            List<Telemetry> telemetry,
            List<List<Double>> features
        ) {
            Map<DetectorType, DetectorResult> results =
                new EnumMap<>(DetectorType.class);

            for (Detector detector : detectors) {
                results.put(
                    detector.type(),
                    detector.detect(features)
                );
            }

            List<GovernanceDecision> decisions =
                new ArrayList<>();

            for (int i = 0; i < telemetry.size(); i++) {
                List<DetectorType> triggers =
                    results.entrySet().stream()
                        .filter(entry ->
                            entry.getValue()
                                .predictions()
                                .get(i)
                        )
                        .map(Map.Entry::getKey)
                        .toList();

                RiskLevel risk =
                    riskPolicy.classify(triggers);

                EvaluationState state =
                    risk == RiskLevel.LOW
                        ? EvaluationState.ACCEPTED
                        : EvaluationState.REVIEW_REQUIRED;

                decisions.add(
                    new GovernanceDecision(
                        telemetry.get(i).id(),
                        state,
                        risk,
                        triggers,
                        triggers.isEmpty()
                            ? "No detector exceeded its threshold."
                            : "Anomaly evidence requires investigation."
                    )
                );
            }

            return List.copyOf(decisions);
        }
    }

    static List<Telemetry> createTelemetry() {
        Random random = new Random(27);
        List<Telemetry> data = new ArrayList<>();

        for (long id = 1; id <= 150; id++) {
            data.add(
                new Telemetry(
                    id,
                    List.of(
                        Math.max(
                            1,
                            8 + random.nextGaussian() * 3
                        ),
                        Math.max(
                            1,
                            120 + random.nextGaussian() * 45
                        ),
                        Math.max(
                            0,
                            55 + random.nextGaussian() * 20
                        ),
                        Math.max(
                            0.2,
                            18 + random.nextGaussian() * 7
                        ),
                        Math.max(
                            0,
                            0.4 + random.nextGaussian() * 0.8
                        ),
                        Math.max(
                            0,
                            1 + random.nextGaussian() * 0.7
                        )
                    ),
                    false,
                    "normal repository telemetry"
                )
            );
        }

        List<List<Double>> anomalyProfiles = List.of(
            List.of(90.0, 2500.0, 1700.0, 1.0, 14.0, 12.0),
            List.of(2.0, 4.0, 1.0, 260.0, 0.0, 0.0),
            List.of(60.0, 1800.0, 90.0, 2.0, 17.0, 10.0),
            List.of(3.0, 7.0, 2.0, 0.2, 18.0, 1.0),
            List.of(75.0, 3200.0, 15.0, 3.0, 0.0, 13.0)
        );

        long id = 151;

        for (List<Double> profile : anomalyProfiles) {
            data.add(
                new Telemetry(
                    id++,
                    profile,
                    true,
                    "known synthetic anomaly"
                )
            );
        }

        return List.copyOf(data);
    }

    public static void main(String[] args) {
        try {
            List<Telemetry> telemetry =
                createTelemetry();

            FeatureScaler scaler =
                new FeatureScaler();

            scaler.fit(telemetry);

            List<List<Double>> features =
                scaler.transform(telemetry);

            List<Detector> detectors = List.of(
                new IsolationForestDetector(
                    90,
                    64,
                    0.06,
                    42
                ),
                new LocalOutlierFactorDetector(10),
                new OneClassSVMDetector(
                    0.18,
                    0.08
                )
            );

            GovernanceService service =
                new GovernanceService(
                    detectors,
                    new RepositoryRiskPolicy()
                );

            List<GovernanceDecision> decisions =
                service.evaluate(
                    telemetry,
                    features
                );

            System.out.println(
                "ENTERPRISE ANOMALY GOVERNANCE ENGINE"
            );
            System.out.println(
                "=========================================="
            );

            Map<RiskLevel, Long> counts =
                decisions.stream()
                    .collect(
                        java.util.stream.Collectors.groupingBy(
                            GovernanceDecision::risk,
                            () -> new EnumMap<>(RiskLevel.class),
                            java.util.stream.Collectors.counting()
                        )
                    );

            counts.forEach(
                (risk, count) ->
                    System.out.printf(
                        "%-10s %d%n",
                        risk,
                        count
                    )
            );

            System.out.println(
                "\nReview-required observations:"
            );

            decisions.stream()
                .filter(decision ->
                    decision.state()
                        == EvaluationState.REVIEW_REQUIRED
                )
                .sorted(
                    Comparator.comparing(
                        GovernanceDecision::risk
                    ).reversed()
                )
                .limit(12)
                .forEach(decision ->
                    System.out.printf(
                        "event=%d risk=%s detectors=%s%n",
                        decision.telemetryId(),
                        decision.risk(),
                        decision.triggeringDetectors()
                    )
                );

            long invalidStates =
                decisions.stream()
                    .filter(decision ->
                        decision.state() == null ||
                        decision.risk() == null
                    )
                    .count();

            if (invalidStates != 0) {
                throw new ValidationException(
                    "Governance produced invalid state."
                );
            }

            System.out.println(
                "\nDomain validation completed."
            );

            System.out.println(
                "\nSecurity controls:"
            );
            System.out.println(
                "- validate telemetry before scoring"
            );
            System.out.println(
                "- protect repository telemetry from unauthorized access"
            );
            System.out.println(
                "- version detector parameters and feature schemas"
            );
            System.out.println(
                "- separate anomaly evidence from final authorization"
            );
            System.out.println(
                "- retain audit records for governance decisions"
            );
        } catch (ValidationException exception) {
            System.err.println(
                "Validation failure: "
                    + exception.getMessage()
            );
            System.exit(1);
        }
    }
}
