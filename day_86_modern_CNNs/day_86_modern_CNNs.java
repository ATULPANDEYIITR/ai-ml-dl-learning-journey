import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Modern CNN Enterprise Experiment Management System.
 *
 * Java 17
 *
 * Provides:
 * - CNN architecture domain modeling
 * - ResNet, DenseNet, EfficientNet
 * - Residual and dense connectivity validation
 * - Compound scaling
 * - MBConv and SE mechanisms
 * - Immutable experiment specifications
 * - Resource and deployment policies
 * - Experiment lifecycle management
 * - Concurrent experiment execution
 * - Metrics and model comparison
 *
 * Run:
 * javac ModernCNN.java
 * java ModernCNN
 */

public class ModernCNN {

    enum ArchitectureFamily {
        RESNET,
        DENSENET,
        EFFICIENTNET
    }

    enum LayerType {
        CONVOLUTION,
        DEPTHWISE,
        POOLING,
        NORMALIZATION,
        ACTIVATION,
        RESIDUAL_ADD,
        CONCATENATION,
        SQUEEZE_EXCITATION,
        DENSE
    }

    enum ExperimentState {
        CREATED,
        VALIDATING,
        RUNNING,
        COMPLETED,
        FAILED,
        CANCELLED
    }

    enum DeploymentTarget {
        EDGE,
        MOBILE,
        SERVER,
        CLOUD
    }

    static class DomainException extends RuntimeException {
        DomainException(String message) {
            super(message);
        }
    }

    static class ResourceViolationException extends RuntimeException {
        ResourceViolationException(String message) {
            super(message);
        }
    }

    record TensorShape(int channels, int height, int width) {

        TensorShape {
            if (channels <= 0 || height <= 0 || width <= 0) {
                throw new DomainException(
                    "Tensor dimensions must be positive."
                );
            }
        }

        long elements() {
            return 1L * channels * height * width;
        }
    }

    record Layer(
        String name,
        LayerType type,
        TensorShape input,
        TensorShape output,
        long parameters,
        long flops
    ) {
        Layer {
            Objects.requireNonNull(name);
            Objects.requireNonNull(type);
            Objects.requireNonNull(input);
            Objects.requireNonNull(output);

            if (name.isBlank() || parameters < 0 || flops < 0) {
                throw new DomainException("Invalid layer definition.");
            }
        }
    }

    record ArchitecturePolicy(
        long maximumParameters,
        long maximumFlops,
        long maximumActivationBytes,
        DeploymentTarget target
    ) {
        ArchitecturePolicy {
            if (
                maximumParameters <= 0 ||
                maximumFlops <= 0 ||
                maximumActivationBytes <= 0
            ) {
                throw new DomainException("Invalid resource policy.");
            }

            Objects.requireNonNull(target);
        }
    }

    record ArchitectureReport(
        String name,
        ArchitectureFamily family,
        TensorShape input,
        TensorShape output,
        int layerCount,
        long parameters,
        long flops,
        long activationBytes
    ) {}

    static final class CNNArchitecture {

        private final String name;
        private final ArchitectureFamily family;
        private final TensorShape input;
        private final List<Layer> layers = new ArrayList<>();
        private TensorShape current;

        CNNArchitecture(
            String name,
            ArchitectureFamily family,
            TensorShape input
        ) {
            this.name = Objects.requireNonNull(name);
            this.family = Objects.requireNonNull(family);
            this.input = Objects.requireNonNull(input);
            this.current = input;
        }

        TensorShape shape() {
            return current;
        }

        void add(Layer layer) {
            if (!layer.input().equals(current)) {
                throw new DomainException(
                    "Invalid tensor connection: " + layer.name()
                );
            }

            layers.add(layer);
            current = layer.output();
        }

        void convolution(
            String name,
            int outputChannels,
            int kernel,
            int stride,
            int groups
        ) {
            if (
                outputChannels <= 0 ||
                kernel <= 0 ||
                stride <= 0 ||
                groups <= 0
            ) {
                throw new DomainException(
                    "Invalid convolution configuration."
                );
            }

            TensorShape source = current;

            if (
                source.channels() % groups != 0 ||
                outputChannels % groups != 0
            ) {
                throw new DomainException(
                    "Grouped convolution channel mismatch."
                );
            }

            int padding = kernel / 2;

            int height = (
                source.height() + 2 * padding - kernel
            ) / stride + 1;

            int width = (
                source.width() + 2 * padding - kernel
            ) / stride + 1;

            TensorShape output = new TensorShape(
                outputChannels,
                height,
                width
            );

            long parameters =
                1L * outputChannels *
                (source.channels() / groups) *
                kernel * kernel;

            long flops =
                2L * output.elements() *
                (source.channels() / groups) *
                kernel * kernel;

            LayerType type = groups == source.channels()
                ? LayerType.DEPTHWISE
                : LayerType.CONVOLUTION;

            add(new Layer(
                name, type, source, output, parameters, flops
            ));
        }

        void normalization(String name) {
            TensorShape source = current;

            add(new Layer(
                name,
                LayerType.NORMALIZATION,
                source,
                source,
                2L * source.channels(),
                4L * source.elements()
            ));
        }

        void activation(String name) {
            TensorShape source = current;

            add(new Layer(
                name,
                LayerType.ACTIVATION,
                source,
                source,
                0,
                source.elements()
            ));
        }

        void pooling(String name, int kernel, int stride) {
            TensorShape source = current;

            int height = (source.height() - kernel) / stride + 1;
            int width = (source.width() - kernel) / stride + 1;

            TensorShape output = new TensorShape(
                source.channels(), height, width
            );

            add(new Layer(
                name,
                LayerType.POOLING,
                source,
                output,
                0,
                output.elements() * kernel * kernel
            ));
        }

        void residualAdd(String name, TensorShape shortcut) {
            if (!current.equals(shortcut)) {
                throw new DomainException(
                    "Residual operands must have matching dimensions."
                );
            }

            add(new Layer(
                name,
                LayerType.RESIDUAL_ADD,
                current,
                current,
                0,
                current.elements()
            ));
        }

        void concatenate(
            String name,
            TensorShape previous,
            TensorShape generated
        ) {
            if (
                previous.height() != generated.height() ||
                previous.width() != generated.width()
            ) {
                throw new DomainException(
                    "Dense concatenation requires equal spatial dimensions."
                );
            }

            TensorShape output = new TensorShape(
                previous.channels() + generated.channels(),
                previous.height(),
                previous.width()
            );

            add(new Layer(
                name,
                LayerType.CONCATENATION,
                previous,
                output,
                0,
                output.elements()
            ));
        }

        void squeezeExcitation(String name, int reduction) {
            if (reduction <= 0) {
                throw new DomainException("Invalid SE reduction.");
            }

            TensorShape source = current;
            int reduced = Math.max(
                1, source.channels() / reduction
            );

            long parameters =
                1L * source.channels() * reduced + reduced +
                1L * reduced * source.channels() + source.channels();

            long flops =
                4L * source.channels() * reduced +
                source.elements();

            add(new Layer(
                name,
                LayerType.SQUEEZE_EXCITATION,
                source,
                source,
                parameters,
                flops
            ));
        }

        void globalAveragePooling(String name) {
            TensorShape source = current;

            TensorShape output = new TensorShape(
                source.channels(), 1, 1
            );

            add(new Layer(
                name,
                LayerType.POOLING,
                source,
                output,
                0,
                source.elements()
            ));
        }

        void classifier(String name, int classes) {
            if (classes <= 0) {
                throw new DomainException("Invalid class count.");
            }

            TensorShape source = current;
            TensorShape output = new TensorShape(classes, 1, 1);

            long parameters =
                1L * source.channels() * classes + classes;

            long flops =
                2L * source.channels() * classes;

            add(new Layer(
                name,
                LayerType.DENSE,
                source,
                output,
                parameters,
                flops
            ));
        }

        long parameters() {
            return layers.stream()
                .mapToLong(Layer::parameters)
                .sum();
        }

        long flops() {
            return layers.stream()
                .mapToLong(Layer::flops)
                .sum();
        }

        long activationBytes() {
            long maximum = input.elements();

            for (Layer layer : layers) {
                maximum = Math.max(
                    maximum,
                    layer.output().elements()
                );
            }

            return maximum * Float.BYTES;
        }

        ArchitectureReport report() {
            return new ArchitectureReport(
                name,
                family,
                input,
                current,
                layers.size(),
                parameters(),
                flops(),
                activationBytes()
            );
        }

        List<Layer> layers() {
            return Collections.unmodifiableList(layers);
        }
    }

    static void residualBlock(
        CNNArchitecture model,
        String name,
        int channels,
        int stride
    ) {
        TensorShape shortcut = model.shape();

        model.convolution(name + ".conv1", channels, 3, stride, 1);
        model.normalization(name + ".bn1");
        model.activation(name + ".relu1");

        model.convolution(name + ".conv2", channels, 3, 1, 1);
        model.normalization(name + ".bn2");

        if (!shortcut.equals(model.shape())) {
            model.convolution(
                name + ".projection", channels, 1, stride, 1
            );
        }

        model.residualAdd(name + ".add", shortcut);
        model.activation(name + ".output");
    }

    static CNNArchitecture buildResNet18(
        int size,
        int classes
    ) {
        CNNArchitecture model = new CNNArchitecture(
            "ResNet-18",
            ArchitectureFamily.RESNET,
            new TensorShape(3, size, size)
        );

        model.convolution("stem", 64, 7, 2, 1);
        model.normalization("stem.bn");
        model.activation("stem.relu");
        model.pooling("stem.maxpool", 3, 2);

        int[] channels = {64, 128, 256, 512};

        for (int stage = 0; stage < channels.length; stage++) {
            for (int block = 0; block < 2; block++) {
                int stride = stage > 0 && block == 0 ? 2 : 1;

                residualBlock(
                    model,
                    "stage" + (stage + 1) +
                        ".block" + (block + 1),
                    channels[stage],
                    stride
                );
            }
        }

        model.globalAveragePooling("global_average");
        model.classifier("classification_head", classes);

        return model;
    }

    static void denseLayer(
        CNNArchitecture model,
        String name,
        int growthRate
    ) {
        TensorShape previous = model.shape();

        model.normalization(name + ".bn1");
        model.convolution(
            name + ".bottleneck", growthRate * 4, 1, 1, 1
        );

        model.normalization(name + ".bn2");
        model.convolution(name + ".growth", growthRate, 3, 1, 1);

        TensorShape generated = model.shape();

        model.concatenate(name + ".concat", previous, generated);
    }

    static CNNArchitecture buildDenseNet121(
        int size,
        int classes
    ) {
        CNNArchitecture model = new CNNArchitecture(
            "DenseNet-121",
            ArchitectureFamily.DENSENET,
            new TensorShape(3, size, size)
        );

        model.convolution("stem", 64, 7, 2, 1);
        model.normalization("stem.bn");
        model.pooling("stem.pool", 3, 2);

        int[] blockSizes = {6, 12, 24, 16};
        int growthRate = 32;

        for (int stage = 0; stage < blockSizes.length; stage++) {
            for (int index = 0; index < blockSizes[stage]; index++) {
                denseLayer(
                    model,
                    "stage" + (stage + 1) + ".layer" + (index + 1),
                    growthRate
                );
            }

            if (stage < 3) {
                int compressed = Math.max(
                    1, model.shape().channels() / 2
                );

                model.convolution(
                    "transition" + stage + ".conv",
                    compressed,
                    1,
                    1,
                    1
                );

                model.pooling(
                    "transition" + stage + ".pool",
                    2,
                    2
                );
            }
        }

        model.globalAveragePooling("global_average");
        model.classifier("classification_head", classes);

        return model;
    }

    record EfficientScale(
        double width,
        double depth,
        int resolution
    ) {}

    static int roundChannels(int channels, double multiplier) {
        double scaled = channels * multiplier;
        int divisor = 8;

        int rounded = Math.max(
            divisor,
            (int) (Math.floor(
                (scaled + divisor / 2.0) / divisor
            ) * divisor)
        );

        if (rounded < 0.9 * scaled) {
            rounded += divisor;
        }

        return rounded;
    }

    static CNNArchitecture buildEfficientNet(
        String variant,
        int classes
    ) {
        Map<String, EfficientScale> scales = Map.of(
            "B0", new EfficientScale(1.0, 1.0, 224),
            "B1", new EfficientScale(1.0, 1.1, 240),
            "B2", new EfficientScale(1.1, 1.2, 260),
            "B3", new EfficientScale(1.2, 1.4, 300),
            "B4", new EfficientScale(1.4, 1.8, 380)
        );

        EfficientScale scale = scales.get(variant);

        if (scale == null) {
            throw new DomainException(
                "Unsupported EfficientNet variant: " + variant
            );
        }

        CNNArchitecture model = new CNNArchitecture(
            "EfficientNet-" + variant,
            ArchitectureFamily.EFFICIENTNET,
            new TensorShape(
                3, scale.resolution(), scale.resolution()
            )
        );

        model.convolution(
            "stem",
            roundChannels(32, scale.width()),
            3,
            2,
            1
        );

        model.normalization("stem.bn");

        int[][] stages = {
            {1, 16, 1, 3, 1},
            {6, 24, 2, 3, 2},
            {6, 40, 2, 5, 2},
            {6, 80, 3, 3, 2},
            {6, 112, 3, 5, 1},
            {6, 192, 4, 5, 2},
            {6, 320, 1, 3, 1}
        };

        for (int stage = 0; stage < stages.length; stage++) {
            int expansion = stages[stage][0];
            int channels = roundChannels(
                stages[stage][1],
                scale.width()
            );

            int repeats = (int) Math.ceil(
                stages[stage][2] * scale.depth()
            );

            int kernel = stages[stage][3];
            int baseStride = stages[stage][4];

            for (int block = 0; block < repeats; block++) {
                TensorShape shortcut = model.shape();
                int expanded = shortcut.channels() * expansion;

                if (expansion != 1) {
                    model.convolution(
                        "s" + stage + ".expand",
                        expanded,
                        1,
                        1,
                        1
                    );
                }

                int stride = block == 0 ? baseStride : 1;

                model.convolution(
                    "s" + stage + ".depthwise",
                    expanded,
                    kernel,
                    stride,
                    expanded
                );

                model.normalization("depthwise.bn");
                model.squeezeExcitation("se", 4);

                model.convolution(
                    "s" + stage + ".project",
                    channels,
                    1,
                    1,
                    1
                );

                if (
                    stride == 1 &&
                    shortcut.channels() == channels &&
                    shortcut.height() == model.shape().height() &&
                    shortcut.width() == model.shape().width()
                ) {
                    model.residualAdd("mbconv.skip", shortcut);
                }
            }
        }

        model.convolution(
            "head",
            roundChannels(1280, scale.width()),
            1,
            1,
            1
        );

        model.globalAveragePooling("global_average");
        model.classifier("classification_head", classes);

        return model;
    }

    interface ExperimentValidator {
        void validate(CNNArchitecture architecture);
    }

    static class ResourcePolicyValidator
        implements ExperimentValidator {

        private final ArchitecturePolicy policy;

        ResourcePolicyValidator(ArchitecturePolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        @Override
        public void validate(CNNArchitecture architecture) {
            if (architecture.parameters() >
                policy.maximumParameters()) {
                throw new ResourceViolationException(
                    "Parameter budget exceeded."
                );
            }

            if (architecture.flops() > policy.maximumFlops()) {
                throw new ResourceViolationException(
                    "FLOP budget exceeded."
                );
            }

            if (
                architecture.activationBytes() >
                policy.maximumActivationBytes()
            ) {
                throw new ResourceViolationException(
                    "Activation memory budget exceeded."
                );
            }

            if (
                policy.target() == DeploymentTarget.MOBILE &&
                architecture.input.height() > 600
            ) {
                throw new ResourceViolationException(
                    "Input resolution exceeds mobile policy."
                );
            }
        }
    }

    static final class Experiment {

        private static final Map<ExperimentState, Set<ExperimentState>>
            TRANSITIONS = Map.of(
                ExperimentState.CREATED,
                Set.of(
                    ExperimentState.VALIDATING,
                    ExperimentState.CANCELLED
                ),
                ExperimentState.VALIDATING,
                Set.of(
                    ExperimentState.RUNNING,
                    ExperimentState.FAILED,
                    ExperimentState.CANCELLED
                ),
                ExperimentState.RUNNING,
                Set.of(
                    ExperimentState.COMPLETED,
                    ExperimentState.FAILED,
                    ExperimentState.CANCELLED
                ),
                ExperimentState.COMPLETED, Set.of(),
                ExperimentState.FAILED, Set.of(),
                ExperimentState.CANCELLED, Set.of()
            );

        private final String id;
        private final CNNArchitecture architecture;
        private final ExperimentValidator validator;

        private ExperimentState state = ExperimentState.CREATED;
        private String failureReason;

        Experiment(
            CNNArchitecture architecture,
            ExperimentValidator validator
        ) {
            this.id = UUID.randomUUID().toString();
            this.architecture = Objects.requireNonNull(architecture);
            this.validator = Objects.requireNonNull(validator);
        }

        synchronized void transition(ExperimentState next) {
            if (!TRANSITIONS.get(state).contains(next)) {
                throw new DomainException(
                    "Invalid experiment state transition: " +
                    state + " -> " + next
                );
            }

            state = next;
        }

        ArchitectureReport execute() {
            try {
                transition(ExperimentState.VALIDATING);
                validator.validate(architecture);

                transition(ExperimentState.RUNNING);

                // The engine evaluates architecture metadata.
                // It does not claim to train model weights.
                ArchitectureReport report = architecture.report();

                transition(ExperimentState.COMPLETED);
                return report;

            } catch (RuntimeException error) {
                failureReason = error.getMessage();

                if (
                    state == ExperimentState.VALIDATING ||
                    state == ExperimentState.RUNNING
                ) {
                    transition(ExperimentState.FAILED);
                }

                throw error;
            }
        }

        String id() {
            return id;
        }

        ExperimentState state() {
            return state;
        }

        String failureReason() {
            return failureReason;
        }
    }

    static class ExperimentService
        implements AutoCloseable {

        private final ExecutorService executor;

        ExperimentService(int concurrency) {
            if (concurrency <= 0) {
                throw new DomainException(
                    "Concurrency must be positive."
                );
            }

            executor = Executors.newFixedThreadPool(concurrency);
        }

        CompletableFuture<ArchitectureReport> submit(
            Experiment experiment
        ) {
            return CompletableFuture.supplyAsync(
                experiment::execute,
                executor
            );
        }

        @Override
        public void close() {
            executor.shutdown();

            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException error) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    static void runTests() {
        if (roundChannels(32, 1.0) != 32) {
            throw new DomainException("Channel rounding test failed.");
        }

        CNNArchitecture resnet = buildResNet18(64, 10);
        CNNArchitecture dense = buildDenseNet121(64, 10);
        CNNArchitecture efficient = buildEfficientNet("B0", 10);

        for (CNNArchitecture architecture :
            List.of(resnet, dense, efficient)) {

            if (architecture.report().output().channels() != 10) {
                throw new DomainException(
                    "Classification output test failed."
                );
            }

            if (architecture.parameters() <= 0 ||
                architecture.flops() <= 0) {
                throw new DomainException(
                    "Invalid architecture metrics."
                );
            }
        }

        try {
            new TensorShape(0, 32, 32);
            throw new DomainException(
                "Invalid tensor was accepted."
            );
        } catch (DomainException expected) {
            // Validation is expected.
        }

        System.out.println("All Java 17 tests passed.");
    }

    public static void main(String[] args) {
        try {
            runTests();

            List<CNNArchitecture> architectures = List.of(
                buildResNet18(224, 1000),
                buildDenseNet121(224, 1000),
                buildEfficientNet("B0", 1000),
                buildEfficientNet("B3", 1000)
            );

            ArchitecturePolicy policy = new ArchitecturePolicy(
                100_000_000L,
                100_000_000_000L,
                512L * 1024 * 1024,
                DeploymentTarget.SERVER
            );

            ResourcePolicyValidator validator =
                new ResourcePolicyValidator(policy);

            List<Experiment> experiments = architectures.stream()
                .map(model -> new Experiment(model, validator))
                .collect(Collectors.toList());

            try (ExperimentService service =
                new ExperimentService(2)) {

                List<CompletableFuture<ArchitectureReport>> futures =
                    experiments.stream()
                        .map(service::submit)
                        .collect(Collectors.toList());

                List<ArchitectureReport> reports = futures.stream()
                    .map(CompletableFuture::join)
                    .collect(Collectors.toList());

                reports.stream()
                    .sorted(Comparator.comparingLong(
                        ArchitectureReport::parameters
                    ))
                    .forEach(report -> System.out.printf(
                        "%-20s Parameters: %12d | GFLOPs: %9.3f | " +
                        "Layers: %4d | Memory: %8.2f MB%n",
                        report.name(),
                        report.parameters(),
                        report.flops() / 1e9,
                        report.layerCount(),
                        report.activationBytes() / (1024.0 * 1024.0)
                    ));

                System.out.println(
                    "\nExperiment timestamp: " + Instant.now()
                );
            }

        } catch (Exception error) {
            System.err.println(
                "Experiment execution failed: " + error.getMessage()
            );
            System.exit(1);
        }
    }
}
