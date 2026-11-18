import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;

/*
 * PyTorch Fundamentals: Java 17 Tensor and Autograd Engine
 *
 * Enterprise-oriented tensor computation model:
 * - Immutable tensor metadata
 * - Explicit CPU device management
 * - Shape validation
 * - Broadcasting
 * - Matrix multiplication
 * - Reverse-mode automatic differentiation
 * - Gradient lifecycle management
 * - Experiment execution policies
 * - Numerical gradient verification
 *
 * Compile:
 *   javac PyTorchFundamentals.java
 *
 * Run:
 *   java PyTorchFundamentals
 *
 * This is an educational CPU implementation.
 * It is not a substitute for PyTorch or a GPU backend.
 */

public class PyTorchFundamentals {

    // --------------------------------------------------------
    // Domain errors
    // --------------------------------------------------------

    static class TensorException extends RuntimeException {
        TensorException(String message) {
            super(message);
        }
    }

    static class ShapeException extends TensorException {
        ShapeException(String message) {
            super(message);
        }
    }

    static class GradientException extends TensorException {
        GradientException(String message) {
            super(message);
        }
    }

    // --------------------------------------------------------
    // Domain types
    // --------------------------------------------------------

    enum Device {
        CPU
    }

    enum Operation {
        LEAF,
        ADD,
        MULTIPLY,
        SUM
    }

    enum ExperimentStatus {
        CREATED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    // --------------------------------------------------------
    // Shape utilities
    // --------------------------------------------------------

    static int elementCount(int[] shape) {
        if (shape == null || shape.length == 0) {
            throw new ShapeException(
                "Shape must contain at least one dimension."
            );
        }

        int count = 1;

        for (int dimension : shape) {
            if (dimension <= 0) {
                throw new ShapeException(
                    "Tensor dimensions must be positive."
                );
            }

            count = Math.multiplyExact(count, dimension);
        }

        return count;
    }

    static int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];

        int current = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = current;
            current = Math.multiplyExact(
                current,
                shape[i]
            );
        }

        return strides;
    }

    static int[] broadcastShape(int[] left, int[] right) {
        int dimensions = Math.max(
            left.length,
            right.length
        );

        int[] output = new int[dimensions];

        for (int i = 0; i < dimensions; i++) {
            int a = i < left.length
                ? left[left.length - 1 - i]
                : 1;

            int b = i < right.length
                ? right[right.length - 1 - i]
                : 1;

            if (a != b && a != 1 && b != 1) {
                throw new ShapeException(
                    "Incompatible broadcasting dimensions."
                );
            }

            output[dimensions - 1 - i] = Math.max(a, b);
        }

        return output;
    }

    static int broadcastIndex(
        int outputIndex,
        int[] outputShape,
        int[] inputShape
    ) {
        int[] outputStrides =
            calculateStrides(outputShape);

        int[] inputStrides =
            calculateStrides(inputShape);

        int offset =
            outputShape.length - inputShape.length;

        int result = 0;

        for (int axis = 0; axis < outputShape.length; axis++) {
            int coordinate =
                (outputIndex / outputStrides[axis]) %
                outputShape[axis];

            if (axis >= offset) {
                int inputAxis = axis - offset;

                if (inputShape[inputAxis] != 1) {
                    result +=
                        coordinate * inputStrides[inputAxis];
                }
            }
        }

        return result;
    }

    // --------------------------------------------------------
    // Computational graph node
    // --------------------------------------------------------

    static final class Node {
        final double[] values;
        final double[] gradients;
        final int[] shape;
        final List<Node> parents = new ArrayList<>();

        final Device device;
        final boolean requiresGrad;

        Operation operation = Operation.LEAF;

        Runnable backwardFunction = () -> {};

        Node(
            double[] values,
            int[] shape,
            boolean requiresGrad,
            Device device
        ) {
            int expected = elementCount(shape);

            if (values == null || values.length != expected) {
                throw new ShapeException(
                    "Data size does not match tensor shape."
                );
            }

            this.values = values.clone();
            this.shape = shape.clone();
            this.gradients = new double[values.length];

            this.requiresGrad = requiresGrad;
            this.device = Objects.requireNonNull(device);

            for (double value : values) {
                if (!Double.isFinite(value)) {
                    throw new TensorException(
                        "Tensor contains non-finite values."
                    );
                }
            }
        }
    }

    // --------------------------------------------------------
    // Tensor
    // --------------------------------------------------------

    static final class Tensor {

        private final Node node;

        Tensor(
            double[] values,
            int[] shape,
            boolean requiresGrad
        ) {
            this.node = new Node(
                values,
                shape,
                requiresGrad,
                Device.CPU
            );
        }

        private Tensor(Node node) {
            this.node = node;
        }

        static Tensor zeros(int... shape) {
            return new Tensor(
                new double[elementCount(shape)],
                shape,
                false
            );
        }

        static Tensor ones(int... shape) {
            double[] values =
                new double[elementCount(shape)];

            Arrays.fill(values, 1.0);

            return new Tensor(values, shape, false);
        }

        int[] shape() {
            return node.shape.clone();
        }

        int numel() {
            return node.values.length;
        }

        Device device() {
            return node.device;
        }

        boolean requiresGrad() {
            return node.requiresGrad;
        }

        double item() {
            if (numel() != 1) {
                throw new TensorException(
                    "item() requires a scalar tensor."
                );
            }

            return node.values[0];
        }

        double[] values() {
            return node.values.clone();
        }

        double[] gradients() {
            return node.gradients.clone();
        }

        private void ensureSameDevice(Tensor other) {
            if (device() != other.device()) {
                throw new TensorException(
                    "Tensor devices do not match."
                );
            }
        }

        // Element-wise addition with broadcasting.
        Tensor add(Tensor other) {
            ensureSameDevice(other);

            int[] outputShape =
                broadcastShape(shape(), other.shape());

            int size = elementCount(outputShape);

            double[] values = new double[size];

            for (int i = 0; i < size; i++) {
                int leftIndex = broadcastIndex(
                    i,
                    outputShape,
                    node.shape
                );

                int rightIndex = broadcastIndex(
                    i,
                    outputShape,
                    other.node.shape
                );

                values[i] =
                    node.values[leftIndex] +
                    other.node.values[rightIndex];
            }

            Node output = new Node(
                values,
                outputShape,
                requiresGrad() || other.requiresGrad(),
                device()
            );

            output.operation = Operation.ADD;

            output.parents.add(node);
            output.parents.add(other.node);

            output.backwardFunction = () -> {
                if (node.requiresGrad) {
                    for (int i = 0; i < size; i++) {
                        int index = broadcastIndex(
                            i,
                            outputShape,
                            node.shape
                        );

                        node.gradients[index] +=
                            output.gradients[i];
                    }
                }

                if (other.node.requiresGrad) {
                    for (int i = 0; i < size; i++) {
                        int index = broadcastIndex(
                            i,
                            outputShape,
                            other.node.shape
                        );

                        other.node.gradients[index] +=
                            output.gradients[i];
                    }
                }
            };

            return new Tensor(output);
        }

        // Element-wise multiplication.
        Tensor multiply(Tensor other) {
            ensureSameDevice(other);

            int[] outputShape =
                broadcastShape(shape(), other.shape());

            int size = elementCount(outputShape);

            double[] values = new double[size];

            for (int i = 0; i < size; i++) {
                int leftIndex = broadcastIndex(
                    i,
                    outputShape,
                    node.shape
                );

                int rightIndex = broadcastIndex(
                    i,
                    outputShape,
                    other.node.shape
                );

                values[i] =
                    node.values[leftIndex] *
                    other.node.values[rightIndex];
            }

            Node output = new Node(
                values,
                outputShape,
                requiresGrad() || other.requiresGrad(),
                device()
            );

            output.operation = Operation.MULTIPLY;

            output.parents.add(node);
            output.parents.add(other.node);

            output.backwardFunction = () -> {
                for (int i = 0; i < size; i++) {
                    int leftIndex = broadcastIndex(
                        i,
                        outputShape,
                        node.shape
                    );

                    int rightIndex = broadcastIndex(
                        i,
                        outputShape,
                        other.node.shape
                    );

                    if (node.requiresGrad) {
                        node.gradients[leftIndex] +=
                            output.gradients[i] *
                            other.node.values[rightIndex];
                    }

                    if (other.node.requiresGrad) {
                        other.node.gradients[rightIndex] +=
                            output.gradients[i] *
                            node.values[leftIndex];
                    }
                }
            };

            return new Tensor(output);
        }

        Tensor square() {
            return multiply(this);
        }

        Tensor sum() {
            double total = Arrays.stream(node.values).sum();

            Node output = new Node(
                new double[]{total},
                new int[]{1},
                requiresGrad(),
                device()
            );

            output.operation = Operation.SUM;
            output.parents.add(node);

            output.backwardFunction = () -> {
                if (!node.requiresGrad) {
                    return;
                }

                for (int i = 0; i < node.gradients.length; i++) {
                    node.gradients[i] +=
                        output.gradients[0];
                }
            };

            return new Tensor(output);
        }

        // Matrix multiplication.
        Tensor matmul(Tensor other) {
            ensureSameDevice(other);

            if (node.shape.length != 2 ||
                other.node.shape.length != 2) {
                throw new ShapeException(
                    "Matrix multiplication requires 2D tensors."
                );
            }

            int rows = node.shape[0];
            int inner = node.shape[1];

            int otherInner = other.node.shape[0];
            int columns = other.node.shape[1];

            if (inner != otherInner) {
                throw new ShapeException(
                    "Matrix dimensions are incompatible."
                );
            }

            double[] result = new double[rows * columns];

            for (int i = 0; i < rows; i++) {
                for (int k = 0; k < inner; k++) {
                    for (int j = 0; j < columns; j++) {
                        result[i * columns + j] +=
                            node.values[i * inner + k] *
                            other.node.values[k * columns + j];
                    }
                }
            }

            // The result is suitable for forward computation.
            // This implementation intentionally limits graph
            // differentiation to element-wise operations and sum.
            return new Tensor(
                result,
                new int[]{rows, columns},
                false
            );
        }

        // Reverse-mode differentiation.
        void backward() {
            if (numel() != 1) {
                throw new GradientException(
                    "backward requires a scalar output."
                );
            }

            if (!requiresGrad()) {
                throw new GradientException(
                    "Output does not require gradients."
                );
            }

            List<Node> topology = new ArrayList<>();
            Set<Node> visited = new HashSet<>();

            buildTopology(node, visited, topology);

            for (Node current : topology) {
                if (current.requiresGrad) {
                    Arrays.fill(current.gradients, 0.0);
                }
            }

            node.gradients[0] = 1.0;

            for (int i = topology.size() - 1; i >= 0; i--) {
                Node current = topology.get(i);

                if (current.requiresGrad) {
                    current.backwardFunction.run();
                }
            }
        }

        private static void buildTopology(
            Node current,
            Set<Node> visited,
            List<Node> topology
        ) {
            if (!visited.add(current)) {
                return;
            }

            for (Node parent : current.parents) {
                buildTopology(parent, visited, topology);
            }

            topology.add(current);
        }

        void zeroGrad() {
            Arrays.fill(node.gradients, 0.0);
        }

        void print(String label) {
            System.out.println(
                label + " values=" +
                Arrays.toString(node.values) +
                " shape=" +
                Arrays.toString(node.shape) +
                " device=" + node.device
            );
        }
    }

    // --------------------------------------------------------
    // Experiment governance
    // --------------------------------------------------------

    interface ExperimentPolicy {
        void validate(String name, Tensor input);
    }

    static final class DefaultExperimentPolicy
        implements ExperimentPolicy {

        @Override
        public void validate(String name, Tensor input) {
            if (name == null || name.isBlank()) {
                throw new TensorException(
                    "Experiment name cannot be empty."
                );
            }

            if (input == null || input.numel() == 0) {
                throw new TensorException(
                    "Experiment input cannot be empty."
                );
            }

            if (input.numel() > 10_000_000) {
                throw new TensorException(
                    "Input exceeds configured resource limit."
                );
            }
        }
    }

    record ExperimentResult(
        String name,
        ExperimentStatus status,
        long durationNanos,
        String details
    ) {}

    static final class ExperimentService {
        private final ExperimentPolicy policy;

        ExperimentService(ExperimentPolicy policy) {
            this.policy = Objects.requireNonNull(policy);
        }

        ExperimentResult execute(
            String name,
            Tensor input
        ) {
            policy.validate(name, input);

            long start = System.nanoTime();

            try {
                double total = Arrays.stream(
                    input.values()
                ).sum();

                long duration =
                    System.nanoTime() - start;

                return new ExperimentResult(
                    name,
                    ExperimentStatus.COMPLETED,
                    duration,
                    "Input sum = " + total
                );

            } catch (RuntimeException exception) {
                return new ExperimentResult(
                    name,
                    ExperimentStatus.FAILED,
                    System.nanoTime() - start,
                    exception.getMessage()
                );
            }
        }
    }

    // --------------------------------------------------------
    // Numerical gradient validation
    // --------------------------------------------------------

    static double numericalDerivative(
        DoubleUnaryOperator function,
        double x
    ) {
        double epsilon = 1e-6;

        return (
            function.applyAsDouble(x + epsilon) -
            function.applyAsDouble(x - epsilon)
        ) / (2 * epsilon);
    }

    // --------------------------------------------------------
    // Demonstrations
    // --------------------------------------------------------

    static void demonstrateTensors() {
        System.out.println("\n1. TENSOR FUNDAMENTALS");

        Tensor matrix = new Tensor(
            new double[]{1, 2, 3, 4},
            new int[]{2, 2},
            false
        );

        Tensor bias = new Tensor(
            new double[]{10, 20},
            new int[]{1, 2},
            false
        );

        matrix.print("Matrix");
        matrix.add(bias).print("Broadcast addition");

        Tensor right = new Tensor(
            new double[]{5, 6, 7, 8},
            new int[]{2, 2},
            false
        );

        matrix.matmul(right).print("Matrix multiplication");

        Tensor.ones(2, 3).print("Ones");
        Tensor.zeros(2, 3).print("Zeros");
    }

    static void demonstrateAutograd() {
        System.out.println("\n2. AUTOGRAD");

        Tensor x = new Tensor(
            new double[]{3},
            new int[]{1},
            true
        );

        Tensor constantTwo = new Tensor(
            new double[]{2},
            new int[]{1},
            false
        );

        Tensor constantOne = new Tensor(
            new double[]{1},
            new int[]{1},
            false
        );

        Tensor y = x.square()
            .add(x.multiply(constantTwo))
            .add(constantOne);

        y.print("Output");

        y.backward();

        System.out.println(
            "Gradient: " +
            Arrays.toString(x.gradients())
        );

        if (Math.abs(x.gradients()[0] - 8.0) > 1e-8) {
            throw new GradientException(
                "Gradient calculation failed."
            );
        }

        Tensor vector = new Tensor(
            new double[]{2, 3},
            new int[]{2},
            true
        );

        vector.square().sum().backward();

        System.out.println(
            "Vector gradients: " +
            Arrays.toString(vector.gradients())
        );
    }

    static void demonstrateGradientCheck() {
        System.out.println("\n3. GRADIENT CHECKING");

        Tensor x = new Tensor(
            new double[]{2},
            new int[]{1},
            true
        );

        Tensor output = x.square().multiply(
            new Tensor(
                new double[]{3},
                new int[]{1},
                false
            )
        ).sum();

        output.backward();

        double analytical = x.gradients()[0];

        double numerical = numericalDerivative(
            value -> 3 * value * value,
            2
        );

        System.out.println("Analytical: " + analytical);
        System.out.println("Numerical: " + numerical);

        if (Math.abs(analytical - numerical) > 1e-5) {
            throw new GradientException(
                "Gradient verification failed."
            );
        }
    }

    static void demonstrateExperiments() {
        System.out.println("\n4. EXPERIMENT MANAGEMENT");

        ExperimentService service =
            new ExperimentService(
                new DefaultExperimentPolicy()
            );

        Tensor input = new Tensor(
            new double[]{1, 2, 3, 4},
            new int[]{2, 2},
            false
        );

        ExperimentResult result =
            service.execute("tensor-validation", input);

        System.out.println("Name: " + result.name());
        System.out.println("Status: " + result.status());
        System.out.println("Details: " + result.details());

        System.out.println(
            "Execution time: " +
            result.durationNanos() +
            " ns"
        );
    }

    static void demonstrateFailures() {
        System.out.println("\n5. FAILURE HANDLING");

        try {
            new Tensor(
                new double[]{1, 2},
                new int[]{3},
                false
            );
        } catch (TensorException error) {
            System.out.println(
                "Expected shape failure: " +
                error.getMessage()
            );
        }

        try {
            Tensor a = new Tensor(
                new double[]{1, 2, 3, 4},
                new int[]{2, 2},
                false
            );

            Tensor b = new Tensor(
                new double[]{1, 2, 3},
                new int[]{3},
                false
            );

            a.matmul(b);
        } catch (TensorException error) {
            System.out.println(
                "Expected matrix failure: " +
                error.getMessage()
            );
        }
    }

    // --------------------------------------------------------
    // Main
    // --------------------------------------------------------

    public static void main(String[] args) {
        try {
            System.out.println("PYTORCH FUNDAMENTALS");
            System.out.println(
                "Java 17 Tensor and Autograd Model"
            );

            demonstrateTensors();
            demonstrateAutograd();
            demonstrateGradientCheck();
            demonstrateExperiments();
            demonstrateFailures();

            System.out.println(
                "\nAll Java demonstrations completed."
            );

        } catch (RuntimeException error) {
            System.err.println(
                "Execution failed: " + error.getMessage()
            );

            System.exit(1);
        }
    }
}
