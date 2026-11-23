import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/*
 * CNN Fundamentals: Enterprise-oriented convolution processing.
 *
 * Java 17 implementation.
 *
 * Compile:
 * javac CNNFundamentals.java
 *
 * Run:
 * java CNNFundamentals
 */

public class CNNFundamentals {

    enum Padding {
        VALID,
        SAME
    }

    enum PipelineState {
        IDLE,
        RUNNING,
        COMPLETED,
        FAILED
    }

    static class CNNException extends RuntimeException {
        CNNException(String message) {
            super(message);
        }
    }

    static class ShapeException extends CNNException {
        ShapeException(String message) {
            super(message);
        }
    }

    static class ConfigurationException extends CNNException {
        ConfigurationException(String message) {
            super(message);
        }
    }

    record Dimensions(int channels, int height, int width) {
        Dimensions {
            if (channels <= 0 || height <= 0 || width <= 0) {
                throw new ConfigurationException(
                    "Dimensions must be positive."
                );
            }
        }
    }

    record PaddingInfo(int top, int bottom, int left, int right) {}

    record ConvConfig(int stride, Padding padding) {
        ConvConfig {
            if (stride <= 0) {
                throw new ConfigurationException(
                    "Stride must be positive."
                );
            }

            Objects.requireNonNull(padding);
        }
    }

    static final class Tensor {

        private final int channels;
        private final int height;
        private final int width;
        private final double[][][] data;

        Tensor(int channels, int height, int width) {
            new Dimensions(channels, height, width);

            this.channels = channels;
            this.height = height;
            this.width = width;

            this.data = new double[channels][height][width];
        }

        Tensor(double[][][] values) {
            Objects.requireNonNull(values);

            if (values.length == 0) {
                throw new ShapeException("Tensor has no channels.");
            }

            if (values[0].length == 0 || values[0][0].length == 0) {
                throw new ShapeException("Tensor dimensions cannot be empty.");
            }

            this.channels = values.length;
            this.height = values[0].length;
            this.width = values[0][0].length;

            this.data = new double[channels][height][width];

            for (int c = 0; c < channels; c++) {
                if (values[c].length != height) {
                    throw new ShapeException("Channel heights differ.");
                }

                for (int r = 0; r < height; r++) {
                    if (values[c][r].length != width) {
                        throw new ShapeException("Channel widths differ.");
                    }

                    for (int col = 0; col < width; col++) {
                        double value = values[c][r][col];

                        if (!Double.isFinite(value)) {
                            throw new ShapeException(
                                "Tensor contains a non-finite value."
                            );
                        }

                        data[c][r][col] = value;
                    }
                }
            }
        }

        double get(int channel, int row, int col) {
            return data[channel][row][col];
        }

        void set(int channel, int row, int col, double value) {
            if (!Double.isFinite(value)) {
                throw new ShapeException("Tensor value must be finite.");
            }

            data[channel][row][col] = value;
        }

        Dimensions dimensions() {
            return new Dimensions(channels, height, width);
        }

        Tensor copy() {
            return new Tensor(data);
        }

        int channels() {
            return channels;
        }

        int height() {
            return height;
        }

        int width() {
            return width;
        }
    }


    static final class Filter {

        private final double[][][] weights;
        private final double bias;
        private final int channels;
        private final int kernelHeight;
        private final int kernelWidth;

        Filter(double[][][] weights, double bias) {
            Objects.requireNonNull(weights);

            if (weights.length == 0 || weights[0].length == 0) {
                throw new ShapeException("Filter cannot be empty.");
            }

            if (weights[0][0].length == 0) {
                throw new ShapeException("Kernel cannot be empty.");
            }

            if (!Double.isFinite(bias)) {
                throw new ConfigurationException("Invalid bias.");
            }

            this.channels = weights.length;
            this.kernelHeight = weights[0].length;
            this.kernelWidth = weights[0][0].length;
            this.bias = bias;

            this.weights = new double[channels][kernelHeight][kernelWidth];

            for (int c = 0; c < channels; c++) {
                if (weights[c].length != kernelHeight) {
                    throw new ShapeException("Kernel heights differ.");
                }

                for (int r = 0; r < kernelHeight; r++) {
                    if (weights[c][r].length != kernelWidth) {
                        throw new ShapeException("Kernel widths differ.");
                    }

                    for (int col = 0; col < kernelWidth; col++) {
                        double value = weights[c][r][col];

                        if (!Double.isFinite(value)) {
                            throw new ShapeException(
                                "Kernel contains non-finite weights."
                            );
                        }

                        this.weights[c][r][col] = value;
                    }
                }
            }
        }

        double get(int channel, int row, int col) {
            return weights[channel][row][col];
        }

        int channels() {
            return channels;
        }

        int kernelHeight() {
            return kernelHeight;
        }

        int kernelWidth() {
            return kernelWidth;
        }

        double bias() {
            return bias;
        }

        long parameterCount() {
            return (long) channels * kernelHeight * kernelWidth + 1;
        }
    }


    record ConvolutionResult(
        Tensor output,
        Dimensions inputDimensions,
        PaddingInfo padding,
        ConvConfig config
    ) {}


    static final class ConvolutionService {

        private int outputDimension(
            int input,
            int kernel,
            int stride,
            Padding padding
        ) {
            if (padding == Padding.SAME) {
                return (input + stride - 1) / stride;
            }

            if (kernel > input) {
                throw new ShapeException(
                    "Kernel exceeds VALID input dimensions."
                );
            }

            return (input - kernel) / stride + 1;
        }

        private PaddingInfo calculatePadding(
            int height,
            int width,
            int kernelHeight,
            int kernelWidth,
            int stride,
            Padding padding
        ) {
            if (padding == Padding.VALID) {
                return new PaddingInfo(0, 0, 0, 0);
            }

            int outputHeight = (height + stride - 1) / stride;
            int outputWidth = (width + stride - 1) / stride;

            int totalHeight = Math.max(
                0,
                (outputHeight - 1) * stride + kernelHeight - height
            );

            int totalWidth = Math.max(
                0,
                (outputWidth - 1) * stride + kernelWidth - width
            );

            int top = totalHeight / 2;
            int left = totalWidth / 2;

            return new PaddingInfo(
                top,
                totalHeight - top,
                left,
                totalWidth - left
            );
        }

        ConvolutionResult apply(
            Tensor input,
            List<Filter> filters,
            ConvConfig config
        ) {
            Objects.requireNonNull(input);
            Objects.requireNonNull(filters);
            Objects.requireNonNull(config);

            if (filters.isEmpty()) {
                throw new ConfigurationException(
                    "At least one filter is required."
                );
            }

            Filter reference = filters.get(0);

            for (Filter filter : filters) {
                if (
                    filter.channels() != input.channels() ||
                    filter.channels() != reference.channels() ||
                    filter.kernelHeight() != reference.kernelHeight() ||
                    filter.kernelWidth() != reference.kernelWidth()
                ) {
                    throw new ShapeException(
                        "Filter dimensions are incompatible."
                    );
                }
            }

            int outputHeight = outputDimension(
                input.height(),
                reference.kernelHeight(),
                config.stride(),
                config.padding()
            );

            int outputWidth = outputDimension(
                input.width(),
                reference.kernelWidth(),
                config.stride(),
                config.padding()
            );

            PaddingInfo pad = calculatePadding(
                input.height(),
                input.width(),
                reference.kernelHeight(),
                reference.kernelWidth(),
                config.stride(),
                config.padding()
            );

            Tensor output = new Tensor(
                filters.size(),
                outputHeight,
                outputWidth
            );

            /*
             * One filter produces one output channel.
             * Contributions from all input channels are accumulated.
             */
            for (int f = 0; f < filters.size(); f++) {
                Filter filter = filters.get(f);

                for (int outRow = 0; outRow < outputHeight; outRow++) {
                    for (int outCol = 0; outCol < outputWidth; outCol++) {

                        double sum = filter.bias();

                        for (int c = 0; c < input.channels(); c++) {
                            for (
                                int kr = 0;
                                kr < filter.kernelHeight();
                                kr++
                            ) {
                                for (
                                    int kc = 0;
                                    kc < filter.kernelWidth();
                                    kc++
                                ) {
                                    int inputRow =
                                        outRow * config.stride() +
                                        kr - pad.top();

                                    int inputCol =
                                        outCol * config.stride() +
                                        kc - pad.left();

                                    if (
                                        inputRow >= 0 &&
                                        inputRow < input.height() &&
                                        inputCol >= 0 &&
                                        inputCol < input.width()
                                    ) {
                                        sum += input.get(
                                            c,
                                            inputRow,
                                            inputCol
                                        ) * filter.get(c, kr, kc);
                                    }
                                }
                            }
                        }

                        if (!Double.isFinite(sum)) {
                            throw new CNNException(
                                "Convolution produced a non-finite result."
                            );
                        }

                        output.set(f, outRow, outCol, sum);
                    }
                }
            }

            return new ConvolutionResult(
                output,
                input.dimensions(),
                pad,
                config
            );
        }
    }


    interface Activation {
        double apply(double value);
    }


    static final class ReLU implements Activation {
        @Override
        public double apply(double value) {
            return Math.max(0.0, value);
        }
    }


    static final class Sigmoid implements Activation {
        @Override
        public double apply(double value) {
            if (value >= 0) {
                return 1.0 / (1.0 + Math.exp(-value));
            }

            double exp = Math.exp(value);
            return exp / (1.0 + exp);
        }
    }


    static final class ActivationService {

        Tensor apply(Tensor input, Activation activation) {
            Objects.requireNonNull(input);
            Objects.requireNonNull(activation);

            Tensor output = input.copy();

            for (int c = 0; c < input.channels(); c++) {
                for (int r = 0; r < input.height(); r++) {
                    for (int col = 0; col < input.width(); col++) {
                        output.set(
                            c,
                            r,
                            col,
                            activation.apply(input.get(c, r, col))
                        );
                    }
                }
            }

            return output;
        }
    }


    static final class PoolingService {

        Tensor maxPool(Tensor input, int poolSize, int stride) {
            if (poolSize <= 0 || stride <= 0) {
                throw new ConfigurationException(
                    "Pooling dimensions must be positive."
                );
            }

            if (
                poolSize > input.height() ||
                poolSize > input.width()
            ) {
                throw new ShapeException(
                    "Pooling window exceeds input."
                );
            }

            int outputHeight =
                (input.height() - poolSize) / stride + 1;

            int outputWidth =
                (input.width() - poolSize) / stride + 1;

            Tensor output = new Tensor(
                input.channels(),
                outputHeight,
                outputWidth
            );

            for (int c = 0; c < input.channels(); c++) {
                for (int r = 0; r < outputHeight; r++) {
                    for (int col = 0; col < outputWidth; col++) {

                        double maximum = Double.NEGATIVE_INFINITY;

                        for (int kr = 0; kr < poolSize; kr++) {
                            for (int kc = 0; kc < poolSize; kc++) {
                                maximum = Math.max(
                                    maximum,
                                    input.get(
                                        c,
                                        r * stride + kr,
                                        col * stride + kc
                                    )
                                );
                            }
                        }

                        output.set(c, r, col, maximum);
                    }
                }
            }

            return output;
        }
    }


    record PipelineResult(
        ConvolutionResult convolution,
        Tensor activated,
        Tensor pooled
    ) {}


    static final class InferencePipeline implements AutoCloseable {

        private final ConvolutionService convolutionService;
        private final ActivationService activationService;
        private final PoolingService poolingService;
        private final ExecutorService executor;

        private volatile PipelineState state = PipelineState.IDLE;

        InferencePipeline(int workerCount) {
            if (workerCount <= 0) {
                throw new ConfigurationException(
                    "Worker count must be positive."
                );
            }

            this.convolutionService = new ConvolutionService();
            this.activationService = new ActivationService();
            this.poolingService = new PoolingService();

            this.executor = Executors.newFixedThreadPool(workerCount);
        }

        synchronized CompletableFuture<PipelineResult> execute(
            Tensor input,
            List<Filter> filters,
            ConvConfig config
        ) {
            if (state == PipelineState.RUNNING) {
                throw new CNNException(
                    "An inference operation is already running."
                );
            }

            state = PipelineState.RUNNING;

            return CompletableFuture.supplyAsync(() -> {
                try {
                    ConvolutionResult convolution =
                        convolutionService.apply(input, filters, config);

                    Tensor activated = activationService.apply(
                        convolution.output(),
                        new ReLU()
                    );

                    Tensor pooled = poolingService.maxPool(
                        activated,
                        2,
                        2
                    );

                    state = PipelineState.COMPLETED;

                    return new PipelineResult(
                        convolution,
                        activated,
                        pooled
                    );

                } catch (RuntimeException exception) {
                    state = PipelineState.FAILED;
                    throw exception;
                }
            }, executor);
        }

        PipelineState state() {
            return state;
        }

        synchronized void reset() {
            if (state == PipelineState.RUNNING) {
                throw new CNNException(
                    "Cannot reset an active inference pipeline."
                );
            }

            state = PipelineState.IDLE;
        }

        @Override
        public void close() {
            executor.shutdown();

            try {
                if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException exception) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }


    static long countParameters(List<Filter> filters) {
        return filters.stream()
            .mapToLong(Filter::parameterCount)
            .sum();
    }


    static void printTensor(String name, Tensor tensor) {
        System.out.println("\n" + name);

        for (int c = 0; c < tensor.channels(); c++) {
            System.out.println("Feature map " + (c + 1));

            for (int r = 0; r < tensor.height(); r++) {
                for (int col = 0; col < tensor.width(); col++) {
                    System.out.printf(
                        "%9.2f",
                        tensor.get(c, r, col)
                    );
                }

                System.out.println();
            }
        }
    }


    static void demonstrateBasicConvolution() {
        System.out.println("\n1. BASIC CONVOLUTION");

        Tensor input = new Tensor(new double[][][] {
            {
                {1, 1, 1, 0, 0},
                {1, 1, 1, 0, 0},
                {1, 1, 1, 0, 0},
                {0, 0, 0, 1, 1},
                {0, 0, 0, 1, 1}
            }
        });

        Filter edge = new Filter(new double[][][] {
            {
                {-1, 0, 1},
                {-1, 0, 1},
                {-1, 0, 1}
            }
        }, 0);

        ConvolutionResult result = new ConvolutionService().apply(
            input,
            List.of(edge),
            new ConvConfig(1, Padding.VALID)
        );

        printTensor("Edge feature map", result.output());
    }


    static void demonstrateConfigurations() {
        System.out.println("\n2. STRIDE AND PADDING");

        Tensor input = new Tensor(new double[][][] {
            {
                {1, 2, 3, 4, 5},
                {6, 7, 8, 9, 10},
                {11, 12, 13, 14, 15},
                {16, 17, 18, 19, 20},
                {21, 22, 23, 24, 25}
            }
        });

        Filter kernel = new Filter(new double[][][] {
            {
                {1, 0, -1},
                {1, 0, -1},
                {1, 0, -1}
            }
        }, 0);

        ConvolutionService service = new ConvolutionService();

        for (Padding padding : Padding.values()) {
            for (int stride : List.of(1, 2)) {
                ConvolutionResult result = service.apply(
                    input,
                    List.of(kernel),
                    new ConvConfig(stride, padding)
                );

                System.out.printf(
                    "Padding=%s, stride=%d, output=%dx%d%n",
                    padding,
                    stride,
                    result.output().height(),
                    result.output().width()
                );

                printTensor("Feature map", result.output());
            }
        }
    }


    static List<Filter> createMultiChannelFilters() {
        Filter first = new Filter(new double[][][] {
            {
                {1, 0},
                {0, -1}
            },
            {
                {0, 1},
                {-1, 0}
            },
            {
                {1, 1},
                {1, 1}
            }
        }, 0.5);

        Filter second = new Filter(new double[][][] {
            {
                {-1, 0},
                {0, 1}
            },
            {
                {1, 1},
                {0, 0}
            },
            {
                {0, -1},
                {1, 0}
            }
        }, -1.0);

        return List.of(first, second);
    }


    static void demonstrateEnterprisePipeline() {
        System.out.println("\n3. MULTI-CHANNEL INFERENCE PIPELINE");

        Tensor input = new Tensor(new double[][][] {
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}
            },
            {
                {16, 15, 14, 13},
                {12, 11, 10, 9},
                {8, 7, 6, 5},
                {4, 3, 2, 1}
            },
            {
                {1, 1, 1, 1},
                {1, 2, 2, 1},
                {1, 2, 2, 1},
                {1, 1, 1, 1}
            }
        });

        List<Filter> filters = createMultiChannelFilters();

        System.out.println(
            "Trainable parameters: " + countParameters(filters)
        );

        try (InferencePipeline pipeline = new InferencePipeline(2)) {

            PipelineResult result = pipeline.execute(
                input,
                filters,
                new ConvConfig(1, Padding.SAME)
            ).join();

            System.out.println("Pipeline state: " + pipeline.state());

            printTensor(
                "Convolution output",
                result.convolution().output()
            );

            printTensor("ReLU output", result.activated());

            printTensor("Max pooling output", result.pooled());
        }
    }


    static void demonstrateParameterCalculation() {
        System.out.println("\n4. PARAMETER CALCULATION");

        int inputChannels = 3;
        int filters = 16;
        int kernelHeight = 3;
        int kernelWidth = 3;

        long weights =
            (long) inputChannels *
            filters *
            kernelHeight *
            kernelWidth;

        long biases = filters;

        System.out.println("Weights: " + weights);
        System.out.println("Biases: " + biases);
        System.out.println("Total: " + (weights + biases));
    }


    static void runTests() {
        System.out.println("\n5. VALIDATION TESTS");

        Tensor input = new Tensor(new double[][][] {
            {
                {1, 2},
                {3, 4}
            }
        });

        Filter identity = new Filter(new double[][][] {
            {
                {1, 0},
                {0, 1}
            }
        }, 0);

        ConvolutionResult result = new ConvolutionService().apply(
            input,
            List.of(identity),
            new ConvConfig(1, Padding.VALID)
        );

        if (result.output().get(0, 0, 0) != 5) {
            throw new AssertionError("Identity-kernel test failed.");
        }

        System.out.println("Identity kernel: PASSED");

        boolean rejected = false;

        try {
            new ConvolutionService().apply(
                input,
                List.of(identity),
                new ConvConfig(0, Padding.VALID)
            );
        } catch (ConfigurationException exception) {
            rejected = true;
        }

        if (!rejected) {
            throw new AssertionError("Invalid stride was accepted.");
        }

        System.out.println("Invalid stride: PASSED");

        rejected = false;

        try {
            Filter oversized = new Filter(new double[][][] {
                {
                    {1, 1, 1},
                    {1, 1, 1},
                    {1, 1, 1}
                }
            }, 0);

            new ConvolutionService().apply(
                input,
                List.of(oversized),
                new ConvConfig(1, Padding.VALID)
            );
        } catch (ShapeException exception) {
            rejected = true;
        }

        if (!rejected) {
            throw new AssertionError("Oversized kernel was accepted.");
        }

        System.out.println("Oversized kernel: PASSED");

        rejected = false;

        try {
            new Tensor(new double[][][] {
                {
                    {1, Double.NaN}
                }
            });
        } catch (ShapeException exception) {
            rejected = true;
        }

        if (!rejected) {
            throw new AssertionError("NaN input was accepted.");
        }

        System.out.println("Non-finite values: PASSED");
    }


    public static void main(String[] args) {
        demonstrateBasicConvolution();
        demonstrateConfigurations();
        demonstrateEnterprisePipeline();
        demonstrateParameterCalculation();
        runTests();

        System.out.println(
            "\nCNN fundamentals execution completed successfully."
        );
    }
}
