import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/*
 * CNN Architecture Enterprise Governance
 *
 * Java 17
 *
 * Compile:
 * javac CnnArchitectureGovernance.java
 *
 * Run:
 * java CnnArchitectureGovernance
 *
 * Models:
 * LeNet-5, AlexNet, VGG16 and Inception v1.
 */

public class CnnArchitectureGovernance {

    enum Architecture {
        LENET,
        ALEXNET,
        VGG16,
        INCEPTION
    }

    enum DeploymentStatus {
        DRAFT,
        VALIDATED,
        APPROVED,
        DEPLOYED,
        REJECTED
    }

    enum ReviewStatus {
        PENDING,
        PASSED,
        FAILED
    }

    enum LayerType {
        CONVOLUTION,
        POOLING,
        DENSE,
        ACTIVATION,
        CONCATENATION
    }

    static class GovernanceException extends RuntimeException {
        GovernanceException(String message) {
            super(message);
        }
    }

    record TensorShape(int channels, int height, int width) {

        TensorShape {
            if (channels <= 0 || height <= 0 || width <= 0) {
                throw new IllegalArgumentException(
                    "Tensor dimensions must be positive"
                );
            }
        }

        String display() {
            return channels + "x" + height + "x" + width;
        }
    }

    record Layer(
        String name,
        LayerType type,
        int inputChannels,
        int outputChannels,
        int kernel,
        int stride,
        int padding
    ) {
        Layer {
            Objects.requireNonNull(name);
            Objects.requireNonNull(type);

            if (name.isBlank() ||
                inputChannels <= 0 ||
                outputChannels <= 0 ||
                kernel <= 0 ||
                stride <= 0 ||
                padding < 0) {
                throw new IllegalArgumentException("Invalid layer definition");
            }
        }

        long parameterCount() {
            return (long) inputChannels * outputChannels *
                kernel * kernel + outputChannels;
        }
    }

    record ArchitectureDefinition(
        Architecture type,
        String name,
        int year,
        int minimumInput,
        int expectedChannels,
        List<Layer> layers,
        List<String> innovations
    ) {
        ArchitectureDefinition {
            Objects.requireNonNull(type);
            Objects.requireNonNull(name);

            layers = List.copyOf(layers);
            innovations = List.copyOf(innovations);
        }
    }

    record ExperimentConfiguration(
        Architecture architecture,
        TensorShape input,
        int classes,
        int batchSize,
        double learningRate
    ) {
        ExperimentConfiguration {
            Objects.requireNonNull(architecture);
            Objects.requireNonNull(input);

            if (classes < 2 || batchSize < 1 || batchSize > 1024) {
                throw new IllegalArgumentException(
                    "Invalid experiment dimensions"
                );
            }

            if (!Double.isFinite(learningRate) ||
                learningRate <= 0 ||
                learningRate > 1) {
                throw new IllegalArgumentException("Invalid learning rate");
            }
        }
    }

    static class ArchitectureRepository {

        private final Map<Architecture, ArchitectureDefinition> definitions =
            new EnumMap<>(Architecture.class);

        ArchitectureRepository() {
            registerDefinitions();
        }

        private Layer conv(
            String name,
            int input,
            int output,
            int kernel,
            int stride,
            int padding
        ) {
            return new Layer(
                name,
                LayerType.CONVOLUTION,
                input,
                output,
                kernel,
                stride,
                padding
            );
        }

        private void registerDefinitions() {
            definitions.put(
                Architecture.LENET,
                new ArchitectureDefinition(
                    Architecture.LENET,
                    "LeNet-5",
                    1998,
                    32,
                    1,
                    List.of(
                        conv("conv1", 1, 6, 5, 1, 0),
                        conv("conv2", 6, 16, 5, 1, 0),
                        conv("conv3", 16, 120, 5, 1, 0)
                    ),
                    List.of(
                        "Small convolutional filters",
                        "Average pooling",
                        "Tanh activation"
                    )
                )
            );

            definitions.put(
                Architecture.ALEXNET,
                new ArchitectureDefinition(
                    Architecture.ALEXNET,
                    "AlexNet",
                    2012,
                    224,
                    3,
                    List.of(
                        conv("conv1", 3, 64, 11, 4, 2),
                        conv("conv2", 64, 192, 5, 1, 2),
                        conv("conv3", 192, 384, 3, 1, 1),
                        conv("conv4", 384, 256, 3, 1, 1),
                        conv("conv5", 256, 256, 3, 1, 1)
                    ),
                    List.of(
                        "ReLU",
                        "Dropout",
                        "GPU-based training"
                    )
                )
            );

            List<Layer> vggLayers = new ArrayList<>();

            int previous = 3;
            int[][] blocks = {
                {64, 64},
                {128, 128},
                {256, 256, 256},
                {512, 512, 512},
                {512, 512, 512}
            };

            int layerNumber = 1;

            for (int[] block : blocks) {
                for (int output : block) {
                    vggLayers.add(
                        conv(
                            "conv" + layerNumber++,
                            previous,
                            output,
                            3,
                            1,
                            1
                        )
                    );
                    previous = output;
                }
            }

            definitions.put(
                Architecture.VGG16,
                new ArchitectureDefinition(
                    Architecture.VGG16,
                    "VGG16",
                    2014,
                    224,
                    3,
                    vggLayers,
                    List.of(
                        "Thirteen 3x3 convolutions",
                        "Five max-pooling stages",
                        "Uniform network design"
                    )
                )
            );

            definitions.put(
                Architecture.INCEPTION,
                new ArchitectureDefinition(
                    Architecture.INCEPTION,
                    "Inception v1",
                    2014,
                    224,
                    3,
                    List.of(
                        conv("branch1", 192, 64, 1, 1, 0),
                        conv("branch2_reduce", 192, 96, 1, 1, 0),
                        conv("branch2", 96, 128, 3, 1, 1),
                        conv("branch3_reduce", 192, 16, 1, 1, 0),
                        conv("branch3", 16, 32, 5, 1, 2),
                        conv("branch4", 192, 32, 1, 1, 0)
                    ),
                    List.of(
                        "Parallel convolution branches",
                        "1x1 bottleneck convolutions",
                        "Feature concatenation"
                    )
                )
            );
        }

        ArchitectureDefinition get(Architecture architecture) {
            ArchitectureDefinition definition = definitions.get(architecture);

            if (definition == null) {
                throw new GovernanceException(
                    "Architecture is not registered"
                );
            }

            return definition;
        }

        List<ArchitectureDefinition> findAll() {
            return List.copyOf(definitions.values());
        }
    }

    record AnalysisReport(
        String architecture,
        long convolutionParameters,
        long denseParameters,
        long totalParameters,
        TensorShape input,
        int classes
    ) {}

    static class ArchitectureAnalysisService {

        private final ArchitectureRepository repository;

        ArchitectureAnalysisService(ArchitectureRepository repository) {
            this.repository = repository;
        }

        AnalysisReport analyze(ExperimentConfiguration configuration) {
            ArchitectureDefinition definition =
                repository.get(configuration.architecture());

            TensorShape input = configuration.input();

            if (input.channels() != definition.expectedChannels()) {
                throw new GovernanceException("Input channel mismatch");
            }

            if (input.height() < definition.minimumInput() ||
                input.width() < definition.minimumInput()) {
                throw new GovernanceException("Input dimensions too small");
            }

            long convolutionParameters = 0;

            for (Layer layer : definition.layers()) {
                convolutionParameters += layer.parameterCount();
            }

            int classes = configuration.classes();
            long denseParameters = 0;

            switch (configuration.architecture()) {
                case LENET -> {
                    denseParameters += dense(84, classes);
                }
                case ALEXNET -> {
                    denseParameters += dense(256 * 6 * 6, 4096);
                    denseParameters += dense(4096, 4096);
                    denseParameters += dense(4096, classes);
                }
                case VGG16 -> {
                    denseParameters += dense(512 * 7 * 7, 4096);
                    denseParameters += dense(4096, 4096);
                    denseParameters += dense(4096, classes);
                }
                case INCEPTION -> {
                    denseParameters += dense(1024, classes);
                }
            }

            return new AnalysisReport(
                definition.name(),
                convolutionParameters,
                denseParameters,
                convolutionParameters + denseParameters,
                input,
                classes
            );
        }

        private long dense(int input, int output) {
            return (long) input * output + output;
        }
    }

    record QualityGates(
        ReviewStatus accuracyReview,
        ReviewStatus securityReview,
        ReviewStatus reproducibilityReview,
        ReviewStatus resourceReview
    ) {
        boolean passed() {
            return accuracyReview == ReviewStatus.PASSED &&
                   securityReview == ReviewStatus.PASSED &&
                   reproducibilityReview == ReviewStatus.PASSED &&
                   resourceReview == ReviewStatus.PASSED;
        }
    }

    static class DeploymentRecord {

        private final String id;
        private final ExperimentConfiguration configuration;
        private DeploymentStatus status;
        private QualityGates gates;
        private final Instant createdAt;

        DeploymentRecord(ExperimentConfiguration configuration) {
            this.id = UUID.randomUUID().toString();
            this.configuration = configuration;
            this.status = DeploymentStatus.DRAFT;
            this.createdAt = Instant.now();
        }

        String id() {
            return id;
        }

        DeploymentStatus status() {
            return status;
        }

        ExperimentConfiguration configuration() {
            return configuration;
        }

        Instant createdAt() {
            return createdAt;
        }

        void transition(
            DeploymentStatus expected,
            DeploymentStatus next
        ) {
            if (status != expected) {
                throw new GovernanceException(
                    "Invalid state transition: " + status + " -> " + next
                );
            }

            status = next;
        }

        void setGates(QualityGates gates) {
            this.gates = Objects.requireNonNull(gates);
        }

        boolean hasPassedGates() {
            return gates != null && gates.passed();
        }
    }

    static class DeploymentService {

        private final ArchitectureAnalysisService analysisService;
        private final Map<String, DeploymentRecord> records =
            new HashMap<>();

        DeploymentService(ArchitectureAnalysisService analysisService) {
            this.analysisService = analysisService;
        }

        DeploymentRecord register(ExperimentConfiguration configuration) {
            DeploymentRecord record = new DeploymentRecord(configuration);

            // Analysis occurs before the record enters the validated state.
            analysisService.analyze(configuration);

            record.transition(
                DeploymentStatus.DRAFT,
                DeploymentStatus.VALIDATED
            );

            records.put(record.id(), record);

            return record;
        }

        void approve(DeploymentRecord record, QualityGates gates) {
            if (record.status() != DeploymentStatus.VALIDATED) {
                throw new GovernanceException(
                    "Only validated models can be approved"
                );
            }

            record.setGates(gates);

            if (!record.hasPassedGates()) {
                record.transition(
                    DeploymentStatus.VALIDATED,
                    DeploymentStatus.REJECTED
                );

                throw new GovernanceException(
                    "One or more quality gates failed"
                );
            }

            record.transition(
                DeploymentStatus.VALIDATED,
                DeploymentStatus.APPROVED
            );
        }

        void deploy(DeploymentRecord record) {
            if (record.status() != DeploymentStatus.APPROVED ||
                !record.hasPassedGates()) {
                throw new GovernanceException(
                    "Deployment authorization denied"
                );
            }

            record.transition(
                DeploymentStatus.APPROVED,
                DeploymentStatus.DEPLOYED
            );
        }

        DeploymentRecord find(String id) {
            return records.get(id);
        }
    }

    public static void main(String[] args) {

        ArchitectureRepository repository =
            new ArchitectureRepository();

        ArchitectureAnalysisService analyzer =
            new ArchitectureAnalysisService(repository);

        DeploymentService deploymentService =
            new DeploymentService(analyzer);

        System.out.println("CNN ENTERPRISE ARCHITECTURE REGISTRY");
        System.out.println("=====================================");

        for (ArchitectureDefinition definition : repository.findAll()) {
            System.out.printf(
                "%-15s | Year: %d | Layers: %d%n",
                definition.name(),
                definition.year(),
                definition.layers().size()
            );
        }

        System.out.println("\nMODEL PARAMETER ANALYSIS");
        System.out.println("========================");

        for (Architecture architecture : Architecture.values()) {

            int size = architecture == Architecture.LENET ? 32 : 224;
            int channels = architecture == Architecture.LENET ? 1 : 3;

            ExperimentConfiguration configuration =
                new ExperimentConfiguration(
                    architecture,
                    new TensorShape(channels, size, size),
                    10,
                    16,
                    0.001
                );

            AnalysisReport report = analyzer.analyze(configuration);

            System.out.printf(
                "%s | Convolution: %,d | Dense: %,d | Total: %,d%n",
                report.architecture(),
                report.convolutionParameters(),
                report.denseParameters(),
                report.totalParameters()
            );
        }

        System.out.println("\nDEPLOYMENT WORKFLOW");
        System.out.println("===================");

        ExperimentConfiguration configuration =
            new ExperimentConfiguration(
                Architecture.INCEPTION,
                new TensorShape(3, 224, 224),
                10,
                16,
                0.001
            );

        DeploymentRecord record =
            deploymentService.register(configuration);

        System.out.println("Registered: " + record.id());
        System.out.println("State: " + record.status());

        QualityGates gates = new QualityGates(
            ReviewStatus.PASSED,
            ReviewStatus.PASSED,
            ReviewStatus.PASSED,
            ReviewStatus.PASSED
        );

        deploymentService.approve(record, gates);
        deploymentService.deploy(record);

        System.out.println("Final state: " + record.status());

        if (deploymentService.find(record.id()) == null) {
            throw new GovernanceException("Deployment record missing");
        }

        // A failed security review must prevent deployment.
        try {
            ExperimentConfiguration invalidConfiguration =
                new ExperimentConfiguration(
                    Architecture.LENET,
                    new TensorShape(1, 32, 32),
                    10,
                    8,
                    0.01
                );

            DeploymentRecord invalid =
                deploymentService.register(invalidConfiguration);

            QualityGates failedGates = new QualityGates(
                ReviewStatus.PASSED,
                ReviewStatus.FAILED,
                ReviewStatus.PASSED,
                ReviewStatus.PASSED
            );

            deploymentService.approve(invalid, failedGates);
        } catch (GovernanceException exception) {
            System.out.println(
                "Rejected deployment: " + exception.getMessage()
            );
        }

        System.out.println("\nEnterprise validation completed.");
    }
}
