import java.util.*;
import java.util.stream.Collectors;

/*
 * Object Detection Enterprise Pipeline
 *
 * Java 17
 * Standard library only.
 *
 * Includes:
 * Bounding-box domain modeling
 * IoU and Generalized IoU
 * Confidence filtering
 * Hard NMS and Soft-NMS
 * Pipeline state transitions
 * Detection evaluation
 * Immutable configuration and results
 */

public class ObjectDetectionEnterprise {

    public static class DetectionException extends RuntimeException {
        public DetectionException(String message) {
            super(message);
        }
    }

    public record BoundingBox(
        double x1,
        double y1,
        double x2,
        double y2
    ) {
        public BoundingBox {
            if (!Double.isFinite(x1) ||
                !Double.isFinite(y1) ||
                !Double.isFinite(x2) ||
                !Double.isFinite(y2)) {
                throw new DetectionException(
                    "Coordinates must be finite."
                );
            }

            if (x2 <= x1 || y2 <= y1) {
                throw new DetectionException(
                    "Bounding box dimensions must be positive."
                );
            }
        }

        public double width() {
            return x2 - x1;
        }

        public double height() {
            return y2 - y1;
        }

        public double area() {
            return width() * height();
        }

        public double centerX() {
            return (x1 + x2) / 2.0;
        }

        public double centerY() {
            return (y1 + y2) / 2.0;
        }

        public BoundingBox clip(int imageWidth, int imageHeight) {
            if (imageWidth <= 0 || imageHeight <= 0) {
                throw new DetectionException(
                    "Invalid image dimensions."
                );
            }

            double nx1 = Math.max(0, Math.min(imageWidth, x1));
            double ny1 = Math.max(0, Math.min(imageHeight, y1));
            double nx2 = Math.max(0, Math.min(imageWidth, x2));
            double ny2 = Math.max(0, Math.min(imageHeight, y2));

            if (nx2 <= nx1 || ny2 <= ny1) {
                return null;
            }

            return new BoundingBox(nx1, ny1, nx2, ny2);
        }

        public static BoundingBox fromXYWH(
            double x,
            double y,
            double width,
            double height
        ) {
            return new BoundingBox(
                x, y, x + width, y + height
            );
        }
    }

    public record Detection(
        BoundingBox box,
        double confidence,
        int classId,
        String label,
        String imageId
    ) {
        public Detection {
            Objects.requireNonNull(box, "Bounding box required.");
            Objects.requireNonNull(label, "Label required.");
            Objects.requireNonNull(imageId, "Image ID required.");

            if (!Double.isFinite(confidence) ||
                confidence < 0 || confidence > 1) {
                throw new DetectionException(
                    "Confidence must be between 0 and 1."
                );
            }

            if (classId < 0 || label.isBlank() || imageId.isBlank()) {
                throw new DetectionException(
                    "Invalid detection metadata."
                );
            }
        }

        public Detection withConfidence(double score) {
            return new Detection(
                box, score, classId, label, imageId
            );
        }

        public Detection withBox(BoundingBox updatedBox) {
            return new Detection(
                updatedBox, confidence, classId, label, imageId
            );
        }
    }

    public enum SuppressionMode {
        HARD_NMS,
        SOFT_NMS
    }

    public enum PipelineState {
        CREATED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    public record PipelinePolicy(
        double confidenceThreshold,
        double iouThreshold,
        double sigma,
        int maxDetections,
        boolean classAgnostic,
        SuppressionMode suppressionMode
    ) {
        public PipelinePolicy {
            if (!Double.isFinite(confidenceThreshold) ||
                confidenceThreshold < 0 ||
                confidenceThreshold > 1) {
                throw new DetectionException(
                    "Invalid confidence threshold."
                );
            }

            if (!Double.isFinite(iouThreshold) ||
                iouThreshold < 0 ||
                iouThreshold > 1) {
                throw new DetectionException(
                    "Invalid IoU threshold."
                );
            }

            if (!Double.isFinite(sigma) || sigma <= 0) {
                throw new DetectionException(
                    "Soft-NMS sigma must be positive."
                );
            }

            if (maxDetections < 1) {
                throw new DetectionException(
                    "Maximum detections must be positive."
                );
            }

            Objects.requireNonNull(suppressionMode);
        }
    }

    public record PipelineResult(
        String imageId,
        List<Detection> detections,
        long elapsedNanos,
        PipelineState state
    ) {
        public PipelineResult {
            detections = List.copyOf(detections);
        }
    }

    public record EvaluationResult(
        int truePositives,
        int falsePositives,
        int falseNegatives
    ) {
        public double precision() {
            int total = truePositives + falsePositives;

            return total == 0
                ? 0
                : (double) truePositives / total;
        }

        public double recall() {
            int total = truePositives + falseNegatives;

            return total == 0
                ? 0
                : (double) truePositives / total;
        }

        public double f1() {
            double p = precision();
            double r = recall();

            return p + r == 0 ? 0 : 2 * p * r / (p + r);
        }
    }

    public static final class GeometryService {

        private GeometryService() {}

        public static double iou(
            BoundingBox a,
            BoundingBox b
        ) {
            double ix1 = Math.max(a.x1(), b.x1());
            double iy1 = Math.max(a.y1(), b.y1());
            double ix2 = Math.min(a.x2(), b.x2());
            double iy2 = Math.min(a.y2(), b.y2());

            double intersection =
                Math.max(0, ix2 - ix1) *
                Math.max(0, iy2 - iy1);

            double union =
                a.area() + b.area() - intersection;

            return union > 0 ? intersection / union : 0;
        }

        public static double giou(
            BoundingBox a,
            BoundingBox b
        ) {
            double overlap = iou(a, b);

            double x1 = Math.min(a.x1(), b.x1());
            double y1 = Math.min(a.y1(), b.y1());
            double x2 = Math.max(a.x2(), b.x2());
            double y2 = Math.max(a.y2(), b.y2());

            double enclosing = (x2 - x1) * (y2 - y1);

            double ix1 = Math.max(a.x1(), b.x1());
            double iy1 = Math.max(a.y1(), b.y1());
            double ix2 = Math.min(a.x2(), b.x2());
            double iy2 = Math.min(a.y2(), b.y2());

            double intersection =
                Math.max(0, ix2 - ix1) *
                Math.max(0, iy2 - iy1);

            double union =
                a.area() + b.area() - intersection;

            return enclosing > 0
                ? overlap - (enclosing - union) / enclosing
                : overlap;
        }
    }

    public static final class SuppressionService {

        public List<Detection> confidenceFilter(
            List<Detection> input,
            double threshold
        ) {
            if (!Double.isFinite(threshold) ||
                threshold < 0 || threshold > 1) {
                throw new DetectionException(
                    "Invalid confidence threshold."
                );
            }

            return input.stream()
                .filter(d -> d.confidence() >= threshold)
                .toList();
        }

        public List<Detection> hardNms(
            List<Detection> input,
            PipelinePolicy policy
        ) {
            List<Detection> candidates = new ArrayList<>(input);

            candidates.sort(
                Comparator.comparingDouble(Detection::confidence)
                    .reversed()
            );

            List<Detection> selected = new ArrayList<>();

            while (!candidates.isEmpty() &&
                   selected.size() < policy.maxDetections()) {

                Detection best = candidates.remove(0);
                selected.add(best);

                candidates.removeIf(candidate -> {
                    if (!policy.classAgnostic() &&
                        candidate.classId() != best.classId()) {
                        return false;
                    }

                    return GeometryService.iou(
                        best.box(), candidate.box()
                    ) > policy.iouThreshold();
                });
            }

            return List.copyOf(selected);
        }

        private record ScoredDetection(
            Detection detection,
            double score
        ) {}

        public List<Detection> softNms(
            List<Detection> input,
            PipelinePolicy policy
        ) {
            List<ScoredDetection> candidates = input.stream()
                .map(d -> new ScoredDetection(d, d.confidence()))
                .collect(Collectors.toCollection(ArrayList::new));

            List<Detection> selected = new ArrayList<>();

            while (!candidates.isEmpty() &&
                   selected.size() < policy.maxDetections()) {

                candidates.sort(
                    Comparator.comparingDouble(ScoredDetection::score)
                        .reversed()
                );

                ScoredDetection best = candidates.remove(0);

                if (best.score() < 0.001) {
                    break;
                }

                selected.add(
                    best.detection().withConfidence(best.score())
                );

                List<ScoredDetection> updated = new ArrayList<>();

                for (ScoredDetection candidate : candidates) {
                    double score = candidate.score();

                    boolean sameClass =
                        candidate.detection().classId() ==
                        best.detection().classId();

                    if (sameClass) {
                        double overlap = GeometryService.iou(
                            best.detection().box(),
                            candidate.detection().box()
                        );

                        if (overlap > policy.iouThreshold()) {
                            score *= Math.exp(
                                -(overlap * overlap) / policy.sigma()
                            );
                        }
                    }

                    if (score >= 0.001) {
                        updated.add(
                            new ScoredDetection(
                                candidate.detection(),
                                score
                            )
                        );
                    }
                }

                candidates = updated;
            }

            return List.copyOf(selected);
        }

        public List<Detection> suppress(
            List<Detection> input,
            PipelinePolicy policy
        ) {
            return switch (policy.suppressionMode()) {
                case HARD_NMS -> hardNms(input, policy);
                case SOFT_NMS -> softNms(input, policy);
            };
        }
    }

    public static final class DetectionEvaluationService {

        public EvaluationResult evaluate(
            List<Detection> predictions,
            List<Detection> targets,
            double threshold
        ) {
            if (!Double.isFinite(threshold) ||
                threshold < 0 || threshold > 1) {
                throw new DetectionException(
                    "Invalid evaluation threshold."
                );
            }

            List<Detection> ordered = predictions.stream()
                .sorted(
                    Comparator.comparingDouble(
                        Detection::confidence
                    ).reversed()
                )
                .toList();

            Set<Integer> matched = new HashSet<>();

            int tp = 0;
            int fp = 0;

            for (Detection prediction : ordered) {
                int bestIndex = -1;
                double bestOverlap = threshold;

                for (int i = 0; i < targets.size(); i++) {
                    if (matched.contains(i)) {
                        continue;
                    }

                    Detection target = targets.get(i);

                    if (!prediction.imageId().equals(target.imageId()) ||
                        prediction.classId() != target.classId()) {
                        continue;
                    }

                    double overlap = GeometryService.iou(
                        prediction.box(),
                        target.box()
                    );

                    if (overlap >= bestOverlap) {
                        bestOverlap = overlap;
                        bestIndex = i;
                    }
                }

                if (bestIndex >= 0) {
                    matched.add(bestIndex);
                    tp++;
                } else {
                    fp++;
                }
            }

            return new EvaluationResult(
                tp,
                fp,
                targets.size() - matched.size()
            );
        }
    }

    public static final class DetectionPipeline {

        private final PipelinePolicy policy;
        private final SuppressionService suppression;
        private PipelineState state;

        private long imagesProcessed;
        private long inputCount;
        private long outputCount;

        public DetectionPipeline(PipelinePolicy policy) {
            this.policy = Objects.requireNonNull(policy);
            this.suppression = new SuppressionService();
            this.state = PipelineState.CREATED;
        }

        public synchronized PipelineResult process(
            String imageId,
            List<Detection> input,
            int imageWidth,
            int imageHeight
        ) {
            if (state == PipelineState.FAILED) {
                throw new DetectionException(
                    "Failed pipeline cannot process more images."
                );
            }

            if (imageId == null || imageId.isBlank() ||
                imageWidth <= 0 || imageHeight <= 0) {
                throw new DetectionException(
                    "Invalid image metadata."
                );
            }

            if (input.stream().anyMatch(
                d -> !d.imageId().equals(imageId)
            )) {
                throw new DetectionException(
                    "Mixed image identifiers."
                );
            }

            long start = System.nanoTime();
            state = PipelineState.RUNNING;

            try {
                inputCount += input.size();

                List<Detection> clipped = input.stream()
                    .map(d -> {
                        BoundingBox box = d.box().clip(
                            imageWidth,
                            imageHeight
                        );

                        return box == null ? null : d.withBox(box);
                    })
                    .filter(Objects::nonNull)
                    .toList();

                List<Detection> filtered =
                    suppression.confidenceFilter(
                        clipped,
                        policy.confidenceThreshold()
                    );

                List<Detection> output =
                    suppression.suppress(filtered, policy);

                outputCount += output.size();
                imagesProcessed++;

                state = PipelineState.COMPLETED;

                return new PipelineResult(
                    imageId,
                    output,
                    System.nanoTime() - start,
                    state
                );

            } catch (RuntimeException exception) {
                state = PipelineState.FAILED;
                throw exception;
            }
        }

        public Map<String, Object> report() {
            Map<String, Object> report = new LinkedHashMap<>();

            report.put("state", state);
            report.put("imagesProcessed", imagesProcessed);
            report.put("inputDetections", inputCount);
            report.put("outputDetections", outputCount);

            report.put(
                "reductionPercentage",
                inputCount == 0
                    ? 0.0
                    : 100.0 * (inputCount - outputCount) / inputCount
            );

            return Collections.unmodifiableMap(report);
        }
    }

    private static void runTests() {
        BoundingBox a = new BoundingBox(10, 10, 110, 110);
        BoundingBox b = new BoundingBox(60, 60, 160, 160);

        double expected = 2500.0 / 17500.0;

        if (Math.abs(GeometryService.iou(a, b) - expected) > 1e-9) {
            throw new AssertionError("IoU calculation failed.");
        }

        if (GeometryService.iou(a, a) != 1.0) {
            throw new AssertionError("IoU identity test failed.");
        }

        if (GeometryService.giou(a, a) != 1.0) {
            throw new AssertionError("GIoU identity test failed.");
        }

        try {
            new BoundingBox(10, 10, 10, 20);
            throw new AssertionError("Invalid box accepted.");
        } catch (DetectionException expectedException) {
            // Expected validation failure.
        }

        System.out.println("All tests passed.");
    }

    public static void main(String[] args) {
        runTests();

        PipelinePolicy policy = new PipelinePolicy(
            0.25,
            0.45,
            0.5,
            100,
            false,
            SuppressionMode.HARD_NMS
        );

        DetectionPipeline pipeline = new DetectionPipeline(policy);

        List<Detection> predictions = List.of(
            new Detection(
                new BoundingBox(50, 40, 200, 220),
                0.96, 0, "person", "image-001"
            ),
            new Detection(
                new BoundingBox(55, 45, 198, 218),
                0.89, 0, "person", "image-001"
            ),
            new Detection(
                new BoundingBox(300, 100, 430, 260),
                0.91, 2, "car", "image-001"
            ),
            new Detection(
                new BoundingBox(500, 50, 580, 130),
                0.18, 0, "person", "image-001"
            ),
            new Detection(
                new BoundingBox(610, 100, 710, 220),
                0.82, 0, "person", "image-001"
            )
        );

        PipelineResult result = pipeline.process(
            "image-001",
            predictions,
            640,
            480
        );

        System.out.println("\nFINAL DETECTIONS");

        result.detections().forEach(d ->
            System.out.printf(
                "%s confidence=%.3f box=[%.1f, %.1f, %.1f, %.1f]%n",
                d.label(),
                d.confidence(),
                d.box().x1(),
                d.box().y1(),
                d.box().x2(),
                d.box().y2()
            )
        );

        List<Detection> targets = List.of(
            new Detection(
                new BoundingBox(52, 42, 202, 222),
                1.0, 0, "person", "image-001"
            ),
            new Detection(
                new BoundingBox(305, 105, 435, 265),
                1.0, 2, "car", "image-001"
            )
        );

        DetectionEvaluationService evaluator =
            new DetectionEvaluationService();

        EvaluationResult evaluation = evaluator.evaluate(
            result.detections(),
            targets,
            0.5
        );

        System.out.println("\nEVALUATION");
        System.out.println("TP: " + evaluation.truePositives());
        System.out.println("FP: " + evaluation.falsePositives());
        System.out.println("FN: " + evaluation.falseNegatives());
        System.out.printf("Precision: %.4f%n", evaluation.precision());
        System.out.printf("Recall: %.4f%n", evaluation.recall());
        System.out.printf("F1: %.4f%n", evaluation.f1());

        System.out.println("\nPIPELINE REPORT");
        pipeline.report().forEach(
            (key, value) -> System.out.println(key + ": " + value)
        );
    }
}
