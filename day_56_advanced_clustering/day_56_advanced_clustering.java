import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

public class AdvancedClustering {

    enum Linkage {
        SINGLE,
        COMPLETE,
        AVERAGE
    }

    enum PointState {
        UNVISITED,
        NOISE,
        CLUSTERED
    }

    record Point(int id, double x, double y, String label) {
        Point {
            if (id < 0) {
                throw new ValidationException("Point ID cannot be negative.");
            }
            if (!Double.isFinite(x) || !Double.isFinite(y)) {
                throw new ValidationException("Coordinates must be finite.");
            }
            Objects.requireNonNull(label, "label");
        }

        double distanceTo(Point other) {
            return Math.hypot(x - other.x, y - other.y);
        }
    }

    record Cluster(int id, List<Point> points) {
        Cluster {
            points = List.copyOf(points);
        }
    }

    record ClusteringPolicy(
        double eps,
        int minSamples,
        int targetClusters,
        Linkage linkage
    ) {
        ClusteringPolicy {
            if (!(eps > 0.0) || !Double.isFinite(eps)) {
                throw new ValidationException("eps must be positive.");
            }
            if (minSamples < 1) {
                throw new ValidationException("minSamples must be positive.");
            }
            if (targetClusters < 1) {
                throw new ValidationException(
                    "targetClusters must be positive."
                );
            }
            Objects.requireNonNull(linkage, "linkage");
        }
    }

    static final class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    static final class ClusterService {

        private final ClusteringPolicy policy;

        ClusterService(ClusteringPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        List<Cluster> hierarchical(List<Point> points) {
            validateDataset(points);

            if (policy.targetClusters() > points.size()) {
                throw new ValidationException(
                    "Target clusters cannot exceed observation count."
                );
            }

            List<List<Point>> working = points.stream()
                .map(List::of)
                .map(ArrayList::new)
                .collect(Collectors.toCollection(ArrayList::new));

            while (working.size() > policy.targetClusters()) {
                int bestI = 0;
                int bestJ = 1;
                double bestDistance = Double.POSITIVE_INFINITY;

                for (int i = 0; i < working.size(); i++) {
                    for (int j = i + 1; j < working.size(); j++) {
                        double distance = linkageDistance(
                            working.get(i),
                            working.get(j),
                            policy.linkage()
                        );

                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestI = i;
                            bestJ = j;
                        }
                    }
                }

                working.get(bestI).addAll(working.get(bestJ));
                working.remove(bestJ);
            }

            List<Cluster> result = new ArrayList<>();

            for (int i = 0; i < working.size(); i++) {
                result.add(new Cluster(i, working.get(i)));
            }

            return List.copyOf(result);
        }

        Map<Integer, Integer> dbscan(List<Point> points) {
            validateDataset(points);

            Map<Integer, Point> byId = points.stream()
                .collect(Collectors.toUnmodifiableMap(
                    Point::id,
                    point -> point
                ));

            Map<Integer, List<Integer>> neighborhoods = new HashMap<>();

            for (Point point : points) {
                List<Integer> neighbors = new ArrayList<>();

                for (Point candidate : points) {
                    if (point.distanceTo(candidate) <= policy.eps()) {
                        neighbors.add(candidate.id());
                    }
                }

                neighborhoods.put(point.id(), neighbors);
            }

            Map<Integer, PointState> states = new HashMap<>();
            Map<Integer, Integer> labels = new LinkedHashMap<>();

            for (Point point : points) {
                states.put(point.id(), PointState.UNVISITED);
            }

            int clusterId = 0;

            for (Point point : points) {
                if (states.get(point.id()) != PointState.UNVISITED) {
                    continue;
                }

                List<Integer> neighbors = neighborhoods.get(point.id());

                if (neighbors.size() < policy.minSamples()) {
                    states.put(point.id(), PointState.NOISE);
                    labels.put(point.id(), -1);
                    continue;
                }

                expandCluster(
                    point.id(),
                    clusterId,
                    neighborhoods,
                    states,
                    labels
                );

                clusterId++;
            }

            // Every label references an existing observation. This explicit
            // lookup protects the domain boundary from corrupted identifiers.
            labels.keySet().forEach(byId::get);

            return Map.copyOf(labels);
        }

        private void expandCluster(
            int seedId,
            int clusterId,
            Map<Integer, List<Integer>> neighborhoods,
            Map<Integer, PointState> states,
            Map<Integer, Integer> labels
        ) {
            ArrayDeque<Integer> queue =
                new ArrayDeque<>(neighborhoods.get(seedId));

            states.put(seedId, PointState.CLUSTERED);
            labels.put(seedId, clusterId);

            Set<Integer> queued = new HashSet<>(queue);

            while (!queue.isEmpty()) {
                int candidateId = queue.removeFirst();

                if (states.get(candidateId) == PointState.UNVISITED) {
                    states.put(candidateId, PointState.CLUSTERED);

                    List<Integer> candidateNeighbors =
                        neighborhoods.get(candidateId);

                    if (candidateNeighbors.size() >= policy.minSamples()) {
                        for (int neighborId : candidateNeighbors) {
                            if (queued.add(neighborId)) {
                                queue.addLast(neighborId);
                            }
                        }
                    }
                }

                if (!labels.containsKey(candidateId) ||
                    labels.get(candidateId) == -1) {
                    labels.put(candidateId, clusterId);
                    states.put(candidateId, PointState.CLUSTERED);
                }
            }
        }

        private double linkageDistance(
            List<Point> left,
            List<Point> right,
            Linkage linkage
        ) {
            List<Double> distances = new ArrayList<>();

            for (Point a : left) {
                for (Point b : right) {
                    distances.add(a.distanceTo(b));
                }
            }

            return switch (linkage) {
                case SINGLE -> distances.stream()
                    .mapToDouble(Double::doubleValue)
                    .min()
                    .orElseThrow();

                case COMPLETE -> distances.stream()
                    .mapToDouble(Double::doubleValue)
                    .max()
                    .orElseThrow();

                case AVERAGE -> distances.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElseThrow();
            };
        }

        private void validateDataset(List<Point> points) {
            if (points == null || points.isEmpty()) {
                throw new ValidationException(
                    "At least one observation is required."
                );
            }

            Set<Integer> ids = new HashSet<>();

            for (Point point : points) {
                if (!ids.add(point.id())) {
                    throw new ValidationException(
                        "Duplicate point ID: " + point.id()
                    );
                }
            }
        }

        double silhouette(
            List<Point> points,
            Map<Integer, Integer> labels
        ) {
            validateDataset(points);

            Map<Integer, List<Point>> groups = new HashMap<>();

            for (Point point : points) {
                Integer label = labels.get(point.id());

                if (label != null && label != -1) {
                    groups.computeIfAbsent(label, key -> new ArrayList<>())
                        .add(point);
                }
            }

            if (groups.size() < 2) {
                return 0.0;
            }

            double total = 0.0;
            int count = 0;

            for (Point point : points) {
                Integer label = labels.get(point.id());

                if (label == null || label == -1) {
                    continue;
                }

                List<Point> own = groups.get(label);

                double a = 0.0;

                if (own.size() > 1) {
                    a = own.stream()
                        .filter(other -> other.id() != point.id())
                        .mapToDouble(point::distanceTo)
                        .average()
                        .orElse(0.0);
                }

                double b = Double.POSITIVE_INFINITY;

                for (Map.Entry<Integer, List<Point>> entry : groups.entrySet()) {
                    if (entry.getKey().equals(label)) {
                        continue;
                    }

                    double mean = entry.getValue().stream()
                        .mapToDouble(point::distanceTo)
                        .average()
                        .orElse(Double.POSITIVE_INFINITY);

                    b = Math.min(b, mean);
                }

                double denominator = Math.max(a, b);
                total += denominator == 0.0
                    ? 0.0
                    : (b - a) / denominator;

                count++;
            }

            return count == 0 ? 0.0 : total / count;
        }
    }

    private static List<Point> dataset() {
        return List.of(
            new Point(1, 1.0, 1.0, "service-a"),
            new Point(2, 1.2, 0.9, "service-a"),
            new Point(3, 0.8, 1.1, "service-a"),
            new Point(4, 1.1, 1.3, "service-a"),
            new Point(5, 10.0, 10.0, "service-b"),
            new Point(6, 10.2, 10.1, "service-b"),
            new Point(7, 9.8, 10.3, "service-b"),
            new Point(8, 10.1, 9.7, "service-b"),
            new Point(9, 20.0, 2.0, "isolated"),
            new Point(10, 30.0, 30.0, "isolated")
        );
    }

    private static void printClusters(List<Cluster> clusters) {
        for (Cluster cluster : clusters) {
            List<Integer> ids = cluster.points().stream()
                .map(Point::id)
                .sorted(Comparator.naturalOrder())
                .toList();

            System.out.println(
                "Cluster " + cluster.id() + ": " + ids
            );
        }
    }

    private static void runValidationTest() {
        List<Point> duplicate = List.of(
            new Point(1, 0, 0, "x"),
            new Point(1, 1, 1, "y")
        );

        ClusterService service = new ClusterService(
            new ClusteringPolicy(1.0, 3, 2, Linkage.AVERAGE)
        );

        try {
            service.hierarchical(duplicate);
            throw new IllegalStateException(
                "Duplicate validation did not fail."
            );
        } catch (ValidationException expected) {
            System.out.println("Validation test passed.");
        }
    }

    public static void main(String[] args) {
        runValidationTest();

        ClusteringPolicy policy = new ClusteringPolicy(
            1.0,
            3,
            2,
            Linkage.AVERAGE
        );

        ClusterService service = new ClusterService(policy);
        List<Point> points = dataset();

        List<Cluster> hierarchical = service.hierarchical(points);

        System.out.println(
            "\nEnterprise service grouping using hierarchical clustering:"
        );
        printClusters(hierarchical);

        Map<Integer, Integer> hierarchicalLabels =
            new HashMap<>();

        for (Cluster cluster : hierarchical) {
            for (Point point : cluster.points()) {
                hierarchicalLabels.put(point.id(), cluster.id());
            }
        }

        System.out.printf(
            "Hierarchical silhouette: %.4f%n",
            service.silhouette(points, hierarchicalLabels)
        );

        Map<Integer, Integer> densityLabels =
            service.dbscan(points);

        Map<Integer, List<Integer>> grouped = new LinkedHashMap<>();

        for (Map.Entry<Integer, Integer> entry : densityLabels.entrySet()) {
            grouped.computeIfAbsent(entry.getValue(), key -> new ArrayList<>())
                .add(entry.getKey());
        }

        System.out.println("\nDBSCAN operational groups:");

        grouped.forEach((clusterId, ids) -> {
            ids.sort(Comparator.naturalOrder());

            System.out.println(
                clusterId == -1
                    ? "Noise: " + ids
                    : "Cluster " + clusterId + ": " + ids
            );
        });

        System.out.printf(
            "DBSCAN silhouette: %.4f%n",
            service.silhouette(points, densityLabels)
        );

        System.out.println(
            "\nDesign note: hierarchical clustering provides a "
            + "merge-based structure, while DBSCAN explicitly models "
            + "density and can leave isolated observations as noise."
        );
    }
}
