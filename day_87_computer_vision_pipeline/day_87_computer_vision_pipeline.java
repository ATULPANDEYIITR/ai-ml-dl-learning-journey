import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Enterprise Computer Vision Pipeline.
 *
 * Features:
 * - P3 PPM image ingestion
 * - Image validation and preprocessing
 * - Deterministic data augmentation
 * - Feature extraction
 * - Nearest-centroid classification
 * - Enterprise model governance
 * - Batch prediction and model persistence
 *
 * Requires Java 17. No third-party dependencies.
 */
public class ComputerVisionPipeline {

    public static void main(String[] args) {
        try {
            if (args.length > 0 && args[0].equals("--predict")) {
                if (args.length != 3) {
                    throw new ValidationException(
                        "Usage: java ComputerVisionPipeline --predict image.ppm model.csv"
                    );
                }

                runPrediction(Path.of(args[1]), Path.of(args[2]));
                return;
            }

            runDemo();

        } catch (Exception exception) {
            System.err.println(
                "[PIPELINE ERROR] " + exception.getMessage()
            );
            System.exit(1);
        }
    }

    private static void runDemo() throws Exception {
        PipelineConfig config = new PipelineConfig(
            16,
            16,
            42L,
            Normalization.ZERO_ONE
        );

        ImageAugmentor augmentor = new ImageAugmentor(config.seed());

        ImagePreprocessor preprocessor = new ImagePreprocessor();
        FeatureExtractor extractor = new FeatureExtractor();

        ModelRegistry registry = new ModelRegistry();

        ClassificationModel model = new ClassificationModel(
            "vision-baseline-001",
            "1.0.0",
            "RGB-statistical-classifier"
        );

        List<TrainingRecord> trainingRecords = List.of(
            new TrainingRecord(
                syntheticImage(230, 20, 20),
                "red"
            ),
            new TrainingRecord(
                syntheticImage(210, 30, 25),
                "red"
            ),
            new TrainingRecord(
                syntheticImage(20, 220, 20),
                "green"
            ),
            new TrainingRecord(
                syntheticImage(25, 200, 30),
                "green"
            ),
            new TrainingRecord(
                syntheticImage(20, 20, 230),
                "blue"
            ),
            new TrainingRecord(
                syntheticImage(25, 30, 210),
                "blue"
            )
        );

        List<TrainingSample> trainingSamples = new ArrayList<>();

        for (TrainingRecord record : trainingRecords) {
            Image processed = preprocessor.resize(
                record.image(),
                config.width(),
                config.height()
            );

            trainingSamples.add(
                new TrainingSample(
                    extractor.extract(processed),
                    record.label()
                )
            );

            Image augmented = augmentor.horizontalFlip(processed);

            trainingSamples.add(
                new TrainingSample(
                    extractor.extract(augmented),
                    record.label()
                )
            );
        }

        model.train(trainingSamples);

        registry.register(model);
        registry.activate(model.modelId());

        ClassificationService classificationService =
            new ClassificationService(
                registry,
                preprocessor,
                extractor,
                config
            );

        List<Image> testImages = List.of(
            syntheticImage(225, 25, 25),
            syntheticImage(25, 215, 25),
            syntheticImage(25, 25, 220)
        );

        List<String> expected = List.of(
            "red", "green", "blue"
        );

        List<PredictionResult> results =
            classificationService.predictBatch(testImages, 2);

        System.out.println("\nENTERPRISE COMPUTER VISION PIPELINE");

        for (int i = 0; i < results.size(); i++) {
            PredictionResult result = results.get(i);

            System.out.println("\nExpected: " + expected.get(i));
            System.out.println("Predicted: " + result.predictedClass());
            System.out.printf(
                "Confidence: %.4f%n",
                result.confidence()
            );

            if (!result.predictedClass().equals(expected.get(i))) {
                throw new PipelineException("Classification test failed.");
            }
        }

        Path modelPath = Path.of("enterprise_vision_model.csv");

        model.save(modelPath);

        ClassificationModel restored = ClassificationModel.load(modelPath);

        if (!restored.modelId().equals(model.modelId())) {
            throw new PipelineException("Model persistence failed.");
        }

        System.out.println("\nModel governance: PASSED");
        System.out.println("Classification tests: PASSED");
        System.out.println("Model persistence: PASSED");

        Image sample = syntheticImage(180, 90, 40);

        Image rotated = augmentor.rotate90(sample);
        Image resized = preprocessor.resize(sample, 16, 16);

        if (rotated.width() != sample.height() ||
            resized.width() != 16) {
            throw new PipelineException("Transformation test failed.");
        }

        System.out.println("Preprocessing tests: PASSED");
        System.out.println("Output model: " + modelPath.toAbsolutePath());
    }

    private static void runPrediction(
        Path imagePath,
        Path modelPath
    ) throws Exception {

        Image image = PPMCodec.read(imagePath);
        ClassificationModel model = ClassificationModel.load(modelPath);

        ModelRegistry registry = new ModelRegistry();
        registry.register(model);
        registry.activate(model.modelId());

        PipelineConfig config = new PipelineConfig(
            32,
            32,
            42L,
            Normalization.ZERO_ONE
        );

        ClassificationService service = new ClassificationService(
            registry,
            new ImagePreprocessor(),
            new FeatureExtractor(),
            config
        );

        PredictionResult result = service.predict(image);

        System.out.println("Predicted: " + result.predictedClass());
        System.out.printf("Confidence: %.4f%n", result.confidence());
        System.out.println("Probabilities: " + result.probabilities());
    }

    private static Image syntheticImage(
        int red,
        int green,
        int blue
    ) {
        List<Pixel> pixels = new ArrayList<>();

        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                int variation = (x + y) % 12;

                pixels.add(
                    new Pixel(
                        Math.min(255, red + variation),
                        Math.min(255, green + variation),
                        Math.min(255, blue + variation)
                    )
                );
            }
        }

        return new Image(16, 16, 255, pixels);
    }

    enum Normalization {
        ZERO_ONE,
        MINUS_ONE_ONE
    }

    static final class ValidationException extends RuntimeException {
        ValidationException(String message) {
            super(message);
        }
    }

    static final class PipelineException extends RuntimeException {
        PipelineException(String message) {
            super(message);
        }
    }

    record PipelineConfig(
        int width,
        int height,
        long seed,
        Normalization normalization
    ) {
        PipelineConfig {
            if (width <= 0 || width > 4096 ||
                height <= 0 || height > 4096) {
                throw new ValidationException("Invalid target dimensions.");
            }

            Objects.requireNonNull(normalization);
        }
    }

    record Pixel(int red, int green, int blue) {
        Pixel {
            if (red < 0 || red > 65535 ||
                green < 0 || green > 65535 ||
                blue < 0 || blue > 65535) {
                throw new ValidationException("Invalid pixel channel.");
            }
        }

        int channel(int index) {
            return switch (index) {
                case 0 -> red;
                case 1 -> green;
                case 2 -> blue;
                default -> throw new ValidationException(
                    "Invalid channel index."
                );
            };
        }
    }

    record Image(
        int width,
        int height,
        int maxValue,
        List<Pixel> pixels
    ) {
        Image {
            if (width <= 0 || height <= 0 ||
                width > 4096 || height > 4096) {
                throw new ValidationException("Invalid image dimensions.");
            }

            if (maxValue < 1 || maxValue > 65535) {
                throw new ValidationException("Invalid maximum channel value.");
            }

            Objects.requireNonNull(pixels);

            if ((long) width * height != pixels.size()) {
                throw new ValidationException("Incorrect pixel count.");
            }

            if (pixels.stream().anyMatch(
                pixel ->
                    pixel.red() > maxValue ||
                    pixel.green() > maxValue ||
                    pixel.blue() > maxValue
            )) {
                throw new ValidationException("Pixel exceeds maximum value.");
            }

            pixels = List.copyOf(pixels);
        }

        Pixel at(int x, int y) {
            if (x < 0 || x >= width || y < 0 || y >= height) {
                throw new ValidationException("Pixel coordinate out of range.");
            }

            return pixels.get(y * width + x);
        }
    }

    record TrainingRecord(Image image, String label) {
        TrainingRecord {
            Objects.requireNonNull(image);

            if (label == null || label.isBlank()) {
                throw new ValidationException("Empty training label.");
            }
        }
    }

    record TrainingSample(FeatureVector features, String label) {
        TrainingSample {
            Objects.requireNonNull(features);

            if (label == null || label.isBlank()) {
                throw new ValidationException("Empty training label.");
            }
        }
    }

    record FeatureVector(List<Double> values) {
        static final List<String> NAMES = List.of(
            "red_mean",
            "green_mean",
            "blue_mean",
            "red_std",
            "green_std",
            "blue_std",
            "red_ratio",
            "green_ratio",
            "blue_ratio"
        );

        FeatureVector {
            if (values == null || values.size() != NAMES.size()) {
                throw new ValidationException("Invalid feature dimensions.");
            }

            if (values.stream().anyMatch(
                value -> value == null || !Double.isFinite(value)
            )) {
                throw new ValidationException("Non-finite feature value.");
            }

            values = List.copyOf(values);
        }
    }

    record PredictionResult(
        String predictedClass,
        double confidence,
        Map<String, Double> probabilities,
        String modelId
    ) {
        PredictionResult {
            Objects.requireNonNull(predictedClass);

            if (!Double.isFinite(confidence) ||
                confidence < 0 || confidence > 1) {
                throw new ValidationException("Invalid confidence.");
            }

            probabilities = Map.copyOf(probabilities);
        }
    }

    static final class PPMCodec {

        static Image read(Path path) throws IOException {
            List<String> tokens = new ArrayList<>();

            try (BufferedReader reader = Files.newBufferedReader(
                path,
                StandardCharsets.US_ASCII
            )) {
                String line;

                while ((line = reader.readLine()) != null) {
                    int comment = line.indexOf('#');

                    if (comment >= 0) {
                        line = line.substring(0, comment);
                    }

                    for (String token : line.trim().split("\\s+")) {
                        if (!token.isEmpty()) {
                            tokens.add(token);
                        }
                    }
                }
            }

            if (tokens.size() < 4 || !tokens.get(0).equals("P3")) {
                throw new PipelineException("Unsupported PPM format.");
            }

            try {
                int width = Integer.parseInt(tokens.get(1));
                int height = Integer.parseInt(tokens.get(2));
                int maximum = Integer.parseInt(tokens.get(3));

                if (width <= 0 || height <= 0 ||
                    width > 4096 || height > 4096 ||
                    maximum < 1 || maximum > 65535) {
                    throw new ValidationException("Invalid PPM header.");
                }

                long expected = (long) width * height * 3;

                if (expected != tokens.size() - 4) {
                    throw new PipelineException("Incorrect PPM pixel count.");
                }

                List<Pixel> pixels = new ArrayList<>();

                for (int i = 4; i < tokens.size(); i += 3) {
                    int red = Integer.parseInt(tokens.get(i));
                    int green = Integer.parseInt(tokens.get(i + 1));
                    int blue = Integer.parseInt(tokens.get(i + 2));

                    if (red < 0 || red > maximum ||
                        green < 0 || green > maximum ||
                        blue < 0 || blue > maximum) {
                        throw new ValidationException("Invalid PPM channel.");
                    }

                    pixels.add(new Pixel(red, green, blue));
                }

                return new Image(width, height, maximum, pixels);

            } catch (NumberFormatException exception) {
                throw new PipelineException("Malformed PPM numeric value.");
            }
        }

        static void write(Image image, Path path) throws IOException {
            try (BufferedWriter writer = Files.newBufferedWriter(
                path,
                StandardCharsets.US_ASCII
            )) {
                writer.write("P3\n");
                writer.write(image.width() + " " + image.height() + "\n");
                writer.write(image.maxValue() + "\n");

                for (Pixel pixel : image.pixels()) {
                    writer.write(
                        pixel.red() + " " +
                        pixel.green() + " " +
                        pixel.blue() + "\n"
                    );
                }
            }
        }
    }

    static final class ImagePreprocessor {

        Image resize(Image source, int targetWidth, int targetHeight) {
            if (targetWidth <= 0 || targetHeight <= 0 ||
                targetWidth > 4096 || targetHeight > 4096) {
                throw new ValidationException("Invalid resize dimensions.");
            }

            List<Pixel> output = new ArrayList<>(
                targetWidth * targetHeight
            );

            for (int y = 0; y < targetHeight; y++) {
                int sourceY = Math.min(
                    source.height() - 1,
                    y * source.height() / targetHeight
                );

                for (int x = 0; x < targetWidth; x++) {
                    int sourceX = Math.min(
                        source.width() - 1,
                        x * source.width() / targetWidth
                    );

                    output.add(source.at(sourceX, sourceY));
                }
            }

            return new Image(
                targetWidth,
                targetHeight,
                source.maxValue(),
                output
            );
        }

        List<List<Double>> normalize(
            Image image,
            Normalization normalization
        ) {
            return image.pixels().stream()
                .map(pixel -> Arrays.asList(
                    normalizeChannel(pixel.red(), image.maxValue(), normalization),
                    normalizeChannel(pixel.green(), image.maxValue(), normalization),
                    normalizeChannel(pixel.blue(), image.maxValue(), normalization)
                ))
                .toList();
        }

        private double normalizeChannel(
            int channel,
            int maximum,
            Normalization normalization
        ) {
            double value = (double) channel / maximum;

            return normalization == Normalization.ZERO_ONE
                ? value
                : value * 2 - 1;
        }

        List<Double> grayscale(Image image) {
            return image.pixels().stream()
                .map(pixel ->
                    (
                        0.2126 * pixel.red() +
                        0.7152 * pixel.green() +
                        0.0722 * pixel.blue()
                    ) / image.maxValue()
                )
                .toList();
        }
    }

    static final class ImageAugmentor {
        private final Random random;

        ImageAugmentor(long seed) {
            random = new Random(seed);
        }

        Image horizontalFlip(Image image) {
            List<Pixel> output = new ArrayList<>();

            for (int y = 0; y < image.height(); y++) {
                for (int x = image.width() - 1; x >= 0; x--) {
                    output.add(image.at(x, y));
                }
            }

            return new Image(
                image.width(),
                image.height(),
                image.maxValue(),
                output
            );
        }

        Image verticalFlip(Image image) {
            List<Pixel> output = new ArrayList<>();

            for (int y = image.height() - 1; y >= 0; y--) {
                for (int x = 0; x < image.width(); x++) {
                    output.add(image.at(x, y));
                }
            }

            return new Image(
                image.width(),
                image.height(),
                image.maxValue(),
                output
            );
        }

        Image rotate90(Image image) {
            List<Pixel> output = new ArrayList<>();

            for (int x = 0; x < image.width(); x++) {
                for (int y = image.height() - 1; y >= 0; y--) {
                    output.add(image.at(x, y));
                }
            }

            return new Image(
                image.height(),
                image.width(),
                image.maxValue(),
                output
            );
        }

        Image brightness(Image image, double factor) {
            if (!Double.isFinite(factor) || factor < 0 || factor > 3) {
                throw new ValidationException("Invalid brightness factor.");
            }

            List<Pixel> output = image.pixels().stream()
                .map(pixel -> new Pixel(
                    adjust(pixel.red(), factor, image.maxValue()),
                    adjust(pixel.green(), factor, image.maxValue()),
                    adjust(pixel.blue(), factor, image.maxValue())
                ))
                .toList();

            return new Image(
                image.width(),
                image.height(),
                image.maxValue(),
                output
            );
        }

        private int adjust(int value, double factor, int maximum) {
            return (int) Math.max(
                0,
                Math.min(maximum, Math.round(value * factor))
            );
        }

        Image randomTransform(Image image) {
            Image result = image;

            if (random.nextBoolean()) {
                result = horizontalFlip(result);
            }

            if (random.nextBoolean()) {
                result = verticalFlip(result);
            }

            return brightness(
                result,
                0.8 + random.nextDouble() * 0.4
            );
        }
    }

    static final class FeatureExtractor {

        FeatureVector extract(Image image) {
            double[] means = new double[3];
            double[] squares = new double[3];

            int total = image.pixels().size();

            for (Pixel pixel : image.pixels()) {
                for (int i = 0; i < 3; i++) {
                    double value =
                        (double) pixel.channel(i) / image.maxValue();

                    means[i] += value;
                    squares[i] += value * value;
                }
            }

            for (int i = 0; i < 3; i++) {
                means[i] /= total;
                squares[i] = Math.sqrt(
                    Math.max(
                        0,
                        squares[i] / total - means[i] * means[i]
                    )
                );
            }

            double sum = Math.max(
                means[0] + means[1] + means[2],
                1e-12
            );

            return new FeatureVector(List.of(
                means[0],
                means[1],
                means[2],
                squares[0],
                squares[1],
                squares[2],
                means[0] / sum,
                means[1] / sum,
                means[2] / sum
            ));
        }
    }

    interface Classifier {
        void train(List<TrainingSample> samples);

        PredictionResult predict(FeatureVector features, String modelId);
    }

    static final class ClassificationModel implements Classifier {

        private final String modelId;
        private final String version;
        private final String architecture;

        private final Map<String, List<Double>> centroids =
            new TreeMap<>();

        private ModelStatus status = ModelStatus.CREATED;

        ClassificationModel(
            String modelId,
            String version,
            String architecture
        ) {
            if (modelId == null || modelId.isBlank() ||
                version == null || version.isBlank() ||
                architecture == null || architecture.isBlank()) {
                throw new ValidationException("Invalid model metadata.");
            }

            this.modelId = modelId;
            this.version = version;
            this.architecture = architecture;
        }

        String modelId() {
            return modelId;
        }

        ModelStatus status() {
            return status;
        }

        @Override
        public void train(List<TrainingSample> samples) {
            if (status != ModelStatus.CREATED) {
                throw new PipelineException("Invalid training state.");
            }

            if (samples == null || samples.isEmpty()) {
                throw new ValidationException("Training samples are empty.");
            }

            status = ModelStatus.TRAINING;

            try {
                Map<String, List<List<Double>>> grouped =
                    samples.stream().collect(
                        Collectors.groupingBy(
                            TrainingSample::label,
                            TreeMap::new,
                            Collectors.mapping(
                                sample -> sample.features().values(),
                                Collectors.toList()
                            )
                        )
                    );

                for (var entry : grouped.entrySet()) {
                    List<List<Double>> vectors = entry.getValue();

                    List<Double> centroid = new ArrayList<>();

                    for (int i = 0; i < FeatureVector.NAMES.size(); i++) {
                        final int index = i;

                        centroid.add(
                            vectors.stream()
                                .mapToDouble(vector -> vector.get(index))
                                .average()
                                .orElseThrow()
                        );
                    }

                    centroids.put(entry.getKey(), List.copyOf(centroid));
                }

                status = ModelStatus.TRAINED;

            } catch (RuntimeException exception) {
                status = ModelStatus.FAILED;
                throw exception;
            }
        }

        @Override
        public PredictionResult predict(
            FeatureVector features,
            String modelId
        ) {
            if (status != ModelStatus.ACTIVE &&
                status != ModelStatus.TRAINED) {
                throw new PipelineException("Model is not available.");
            }

            if (centroids.isEmpty()) {
                throw new PipelineException("Model has no trained classes.");
            }

            Map<String, Double> distances = new TreeMap<>();

            for (var entry : centroids.entrySet()) {
                double squared = 0;

                for (int i = 0; i < features.values().size(); i++) {
                    double difference =
                        features.values().get(i) -
                        entry.getValue().get(i);

                    squared += difference * difference;
                }

                distances.put(entry.getKey(), Math.sqrt(squared));
            }

            double minimum = Collections.min(distances.values());

            Map<String, Double> weights = new LinkedHashMap<>();

            distances.forEach(
                (label, distance) ->
                    weights.put(label, Math.exp(-(distance - minimum)))
            );

            double denominator = weights.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

            Map<String, Double> probabilities = new TreeMap<>();

            weights.forEach(
                (label, weight) ->
                    probabilities.put(label, weight / denominator)
            );

            var best = probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow();

            return new PredictionResult(
                best.getKey(),
                best.getValue(),
                probabilities,
                modelId
            );
        }

        void activate() {
            if (status != ModelStatus.TRAINED) {
                throw new PipelineException(
                    "Only trained models can be activated."
                );
            }

            status = ModelStatus.ACTIVE;
        }

        void save(Path path) throws IOException {
            if (centroids.isEmpty()) {
                throw new PipelineException("Cannot save untrained model.");
            }

            try (BufferedWriter writer = Files.newBufferedWriter(
                path,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
            )) {
                writer.write(modelId + "\n");
                writer.write(version + "\n");
                writer.write(architecture + "\n");

                for (var entry : centroids.entrySet()) {
                    writer.write(entry.getKey());

                    for (double value : entry.getValue()) {
                        writer.write("," + value);
                    }

                    writer.write("\n");
                }
            }
        }

        static ClassificationModel load(Path path) throws IOException {
            List<String> lines = Files.readAllLines(
                path,
                StandardCharsets.UTF_8
            );

            if (lines.size() < 4) {
                throw new PipelineException("Incomplete model file.");
            }

            ClassificationModel model = new ClassificationModel(
                lines.get(0),
                lines.get(1),
                lines.get(2)
            );

            for (int i = 3; i < lines.size(); i++) {
                String[] parts = lines.get(i).split(",");

                if (parts.length != 10 || parts[0].isBlank()) {
                    throw new PipelineException("Invalid centroid record.");
                }

                List<Double> vector = new ArrayList<>();

                for (int j = 1; j < parts.length; j++) {
                    double value = Double.parseDouble(parts[j]);

                    if (!Double.isFinite(value)) {
                        throw new PipelineException("Invalid model value.");
                    }

                    vector.add(value);
                }

                if (model.centroids.putIfAbsent(
                    parts[0],
                    List.copyOf(vector)
                ) != null) {
                    throw new PipelineException("Duplicate model class.");
                }
            }

            if (model.centroids.isEmpty()) {
                throw new PipelineException("Model has no classes.");
            }

            model.status = ModelStatus.TRAINED;
            return model;
        }
    }

    enum ModelStatus {
        CREATED,
        TRAINING,
        TRAINED,
        ACTIVE,
        FAILED,
        RETIRED
    }

    static final class ModelRegistry {

        private final Map<String, ClassificationModel> models =
            new HashMap<>();

        private String activeModelId;

        void register(ClassificationModel model) {
            Objects.requireNonNull(model);

            if (models.putIfAbsent(model.modelId(), model) != null) {
                throw new PipelineException("Duplicate model identifier.");
            }
        }

        void activate(String modelId) {
            ClassificationModel model = models.get(modelId);

            if (model == null) {
                throw new PipelineException("Model not registered.");
            }

            if (activeModelId != null) {
                ClassificationModel previous = models.get(activeModelId);

                if (previous != null) {
                    previous.status = ModelStatus.TRAINED;
                }
            }

            model.activate();
            activeModelId = modelId;
        }

        ClassificationModel activeModel() {
            if (activeModelId == null) {
                throw new PipelineException("No active model.");
            }

            return models.get(activeModelId);
        }
    }

    static final class ClassificationService {

        private final ModelRegistry registry;
        private final ImagePreprocessor preprocessor;
        private final FeatureExtractor extractor;
        private final PipelineConfig config;

        ClassificationService(
            ModelRegistry registry,
            ImagePreprocessor preprocessor,
            FeatureExtractor extractor,
            PipelineConfig config
        ) {
            this.registry = registry;
            this.preprocessor = preprocessor;
            this.extractor = extractor;
            this.config = config;
        }

        PredictionResult predict(Image image) {
            ClassificationModel model = registry.activeModel();

            Image processed = preprocessor.resize(
                image,
                config.width(),
                config.height()
            );

            FeatureVector features = extractor.extract(processed);

            return model.predict(features, model.modelId());
        }

        List<PredictionResult> predictBatch(
            List<Image> images,
            int concurrency
        ) throws Exception {
            if (images == null ||
                concurrency < 1 ||
                concurrency > 32) {
                throw new ValidationException("Invalid batch parameters.");
            }

            ExecutorService executor = Executors.newFixedThreadPool(
                concurrency
            );

            try {
                List<CompletableFuture<PredictionResult>> futures =
                    images.stream()
                        .map(image ->
                            CompletableFuture.supplyAsync(
                                () -> predict(image),
                                executor
                            )
                        )
                        .toList();

                return futures.stream()
                    .map(CompletableFuture::join)
                    .toList();

            } finally {
                executor.shutdown();

                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            }
        }
    }
}
