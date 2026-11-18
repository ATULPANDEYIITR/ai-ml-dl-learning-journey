#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

/*
 * PyTorch Fundamentals: C++17 Tensor and Autograd Engine
 *
 * Features:
 * - Multidimensional tensor representation
 * - Shape validation
 * - Element-wise operations
 * - Broadcasting
 * - Matrix multiplication
 * - Reverse-mode automatic differentiation
 * - Gradient accumulation
 * - Computational graph traversal
 * - CPU device modeling
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic pytorch.cpp -o pytorch
 *
 * Run:
 *   ./pytorch
 *
 * This implementation is a CPU educational engine.
 * It does not implement CUDA or replace libtorch.
 */

namespace tensorlib {

// ------------------------------------------------------------
// Errors and domain types
// ------------------------------------------------------------

class TensorError : public std::runtime_error {
public:
    explicit TensorError(const std::string& message)
        : std::runtime_error(message) {}
};

class ShapeError : public TensorError {
public:
    explicit ShapeError(const std::string& message)
        : TensorError(message) {}
};

enum class Device {
    CPU
};

enum class Operation {
    Leaf,
    Add,
    Multiply,
    Sum
};

std::string deviceName(Device device) {
    switch (device) {
        case Device::CPU:
            return "CPU";
    }

    throw TensorError("Unknown device.");
}

// ------------------------------------------------------------
// Tensor shape management
// ------------------------------------------------------------

std::size_t elementCount(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw ShapeError("Shape cannot be empty.");
    }

    std::size_t total = 1;

    for (const auto dimension : shape) {
        if (dimension == 0) {
            throw ShapeError("Dimensions must be positive.");
        }

        if (total > static_cast<std::size_t>(-1) / dimension) {
            throw ShapeError("Tensor dimensions overflow.");
        }

        total *= dimension;
    }

    return total;
}

std::vector<std::size_t> calculateStrides(
    const std::vector<std::size_t>& shape
) {
    std::vector<std::size_t> strides(shape.size());

    std::size_t current = 1;

    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = current;
        current *= shape[i];
    }

    return strides;
}

std::vector<std::size_t> broadcastShape(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b
) {
    const std::size_t dimensions =
        std::max(a.size(), b.size());

    std::vector<std::size_t> result(dimensions);

    for (std::size_t i = 0; i < dimensions; ++i) {
        const std::size_t left =
            i < a.size() ? a[a.size() - 1 - i] : 1;

        const std::size_t right =
            i < b.size() ? b[b.size() - 1 - i] : 1;

        if (
            left != right &&
            left != 1 &&
            right != 1
        ) {
            throw ShapeError(
                "Incompatible broadcasting dimensions."
            );
        }

        result[dimensions - 1 - i] =
            std::max(left, right);
    }

    return result;
}

// ------------------------------------------------------------
// Automatic differentiation node
// ------------------------------------------------------------

struct Node {
    std::vector<double> values;
    std::vector<double> gradients;
    std::vector<std::size_t> shape;

    std::vector<std::shared_ptr<Node>> parents;

    std::function<void()> backwardFunction;

    Operation operation = Operation::Leaf;
    Device device = Device::CPU;
    bool requiresGradient = false;

    explicit Node(
        std::vector<double> data,
        std::vector<std::size_t> dimensions,
        bool requiresGradient = false
    )
        : values(std::move(data)),
          shape(std::move(dimensions)),
          requiresGradient(requiresGradient) {

        if (values.size() != elementCount(shape)) {
            throw ShapeError(
                "Data length does not match tensor shape."
            );
        }

        for (double value : values) {
            if (!std::isfinite(value)) {
                throw TensorError(
                    "Tensor contains a non-finite value."
                );
            }
        }

        gradients.assign(values.size(), 0.0);
        backwardFunction = [] {};
    }
};

// ------------------------------------------------------------
// Tensor class
// ------------------------------------------------------------

class Tensor {
private:
    std::shared_ptr<Node> node;

    explicit Tensor(std::shared_ptr<Node> value)
        : node(std::move(value)) {}

    static std::size_t broadcastIndex(
        std::size_t outputIndex,
        const std::vector<std::size_t>& outputShape,
        const std::vector<std::size_t>& inputShape
    ) {
        const auto outputStrides =
            calculateStrides(outputShape);

        const auto inputStrides =
            calculateStrides(inputShape);

        const std::size_t offset =
            outputShape.size() - inputShape.size();

        std::size_t result = 0;

        for (std::size_t axis = 0;
             axis < outputShape.size();
             ++axis) {

            const std::size_t coordinate =
                (outputIndex / outputStrides[axis]) %
                outputShape[axis];

            if (axis >= offset) {
                const std::size_t inputAxis = axis - offset;

                if (inputShape[inputAxis] != 1) {
                    result +=
                        coordinate * inputStrides[inputAxis];
                }
            }
        }

        return result;
    }

    static void accumulate(
        const std::shared_ptr<Node>& target,
        std::size_t index,
        double gradient
    ) {
        if (target->requiresGradient) {
            target->gradients[index] += gradient;
        }
    }

    static void backwardNode(
        const std::shared_ptr<Node>& current
    ) {
        current->backwardFunction();
    }

public:
    Tensor(
        std::vector<double> data,
        std::vector<std::size_t> shape,
        bool requiresGradient = false
    )
        : node(std::make_shared<Node>(
            std::move(data),
            std::move(shape),
            requiresGradient
        )) {}

    static Tensor zeros(
        const std::vector<std::size_t>& shape
    ) {
        return Tensor(
            std::vector<double>(elementCount(shape), 0.0),
            shape
        );
    }

    static Tensor ones(
        const std::vector<std::size_t>& shape
    ) {
        return Tensor(
            std::vector<double>(elementCount(shape), 1.0),
            shape
        );
    }

    const std::vector<double>& values() const {
        return node->values;
    }

    const std::vector<double>& gradients() const {
        return node->gradients;
    }

    const std::vector<std::size_t>& shape() const {
        return node->shape;
    }

    bool requiresGradient() const {
        return node->requiresGradient;
    }

    Device device() const {
        return node->device;
    }

    std::size_t numel() const {
        return node->values.size();
    }

    double item() const {
        if (numel() != 1) {
            throw TensorError(
                "item() requires a scalar tensor."
            );
        }

        return node->values[0];
    }

    // Element-wise addition with broadcasting.
    Tensor add(const Tensor& other) const {
        if (device() != other.device()) {
            throw TensorError("Device mismatch.");
        }

        const auto outputShape =
            broadcastShape(shape(), other.shape());

        const std::size_t size =
            elementCount(outputShape);

        std::vector<double> result(size);

        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t leftIndex =
                broadcastIndex(i, outputShape, shape());

            const std::size_t rightIndex =
                broadcastIndex(i, outputShape, other.shape());

            result[i] =
                node->values[leftIndex] +
                other.node->values[rightIndex];
        }

        auto output = std::make_shared<Node>(
            std::move(result),
            outputShape,
            requiresGradient() || other.requiresGradient()
        );

        output->operation = Operation::Add;
        output->parents = {node, other.node};

        output->backwardFunction = [
            output,
            left = node,
            right = other.node,
            outputShape
        ] {
            if (left->requiresGradient) {
                for (std::size_t i = 0;
                     i < output->values.size();
                     ++i) {

                    accumulate(
                        left,
                        broadcastIndex(
                            i,
                            outputShape,
                            left->shape
                        ),
                        output->gradients[i]
                    );
                }
            }

            if (right->requiresGradient) {
                for (std::size_t i = 0;
                     i < output->values.size();
                     ++i) {

                    accumulate(
                        right,
                        broadcastIndex(
                            i,
                            outputShape,
                            right->shape
                        ),
                        output->gradients[i]
                    );
                }
            }
        };

        return Tensor(output);
    }

    // Element-wise multiplication.
    Tensor multiply(const Tensor& other) const {
        const auto outputShape =
            broadcastShape(shape(), other.shape());

        const std::size_t size =
            elementCount(outputShape);

        std::vector<double> result(size);

        for (std::size_t i = 0; i < size; ++i) {
            const std::size_t leftIndex =
                broadcastIndex(i, outputShape, shape());

            const std::size_t rightIndex =
                broadcastIndex(i, outputShape, other.shape());

            result[i] =
                node->values[leftIndex] *
                other.node->values[rightIndex];
        }

        auto output = std::make_shared<Node>(
            std::move(result),
            outputShape,
            requiresGradient() || other.requiresGradient()
        );

        output->operation = Operation::Multiply;
        output->parents = {node, other.node};

        output->backwardFunction = [
            output,
            left = node,
            right = other.node,
            outputShape
        ] {
            for (std::size_t i = 0;
                 i < output->values.size();
                 ++i) {

                const std::size_t li =
                    broadcastIndex(
                        i,
                        outputShape,
                        left->shape
                    );

                const std::size_t ri =
                    broadcastIndex(
                        i,
                        outputShape,
                        right->shape
                    );

                accumulate(
                    left,
                    li,
                    output->gradients[i] *
                    right->values[ri]
                );

                accumulate(
                    right,
                    ri,
                    output->gradients[i] *
                    left->values[li]
                );
            }
        };

        return Tensor(output);
    }

    Tensor square() const {
        return multiply(*this);
    }

    // Reduction operation.
    Tensor sum() const {
        const double total = std::accumulate(
            node->values.begin(),
            node->values.end(),
            0.0
        );

        auto output = std::make_shared<Node>(
            std::vector<double>{total},
            std::vector<std::size_t>{1},
            requiresGradient()
        );

        output->operation = Operation::Sum;
        output->parents = {node};

        output->backwardFunction = [
            output,
            parent = node
        ] {
            if (!parent->requiresGradient) {
                return;
            }

            for (double& gradient : parent->gradients) {
                gradient += output->gradients[0];
            }
        };

        return Tensor(output);
    }

    // Matrix multiplication for two-dimensional tensors.
    Tensor matmul(const Tensor& other) const {
        if (shape().size() != 2 ||
            other.shape().size() != 2) {
            throw ShapeError(
                "matmul requires two matrices."
            );
        }

        const std::size_t rows = shape()[0];
        const std::size_t inner = shape()[1];

        const std::size_t otherInner = other.shape()[0];
        const std::size_t columns = other.shape()[1];

        if (inner != otherInner) {
            throw ShapeError(
                "Matrix dimensions are incompatible."
            );
        }

        std::vector<double> result(rows * columns, 0.0);

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t k = 0; k < inner; ++k) {
                for (std::size_t j = 0; j < columns; ++j) {
                    result[i * columns + j] +=
                        node->values[i * inner + k] *
                        other.node->values[k * columns + j];
                }
            }
        }

        return Tensor(
            std::move(result),
            {rows, columns},
            requiresGradient() || other.requiresGradient()
        );
    }

    // Reverse-mode automatic differentiation.
    void backward(double seed = 1.0) {
        if (numel() != 1) {
            throw TensorError(
                "backward requires a scalar output."
            );
        }

        if (!requiresGradient()) {
            throw TensorError(
                "Tensor does not require gradients."
            );
        }

        std::vector<std::shared_ptr<Node>> topology;
        std::unordered_set<const Node*> visited;

        std::function<void(const std::shared_ptr<Node>&)> visit =
            [&](const std::shared_ptr<Node>& current) {

                if (!visited.insert(current.get()).second) {
                    return;
                }

                for (const auto& parent : current->parents) {
                    visit(parent);
                }

                topology.push_back(current);
            };

        visit(node);

        // Reset gradients for a fresh backward pass.
        for (const auto& current : topology) {
            if (current->requiresGradient) {
                std::fill(
                    current->gradients.begin(),
                    current->gradients.end(),
                    0.0
                );
            }
        }

        node->gradients[0] = seed;

        for (auto it = topology.rbegin();
             it != topology.rend();
             ++it) {
            backwardNode(*it);
        }
    }

    void zeroGrad() {
        std::fill(
            node->gradients.begin(),
            node->gradients.end(),
            0.0
        );
    }

    void print(const std::string& label) const {
        std::cout << label << " [";

        for (std::size_t i = 0; i < numel(); ++i) {
            if (i != 0) std::cout << ", ";
            std::cout << node->values[i];
        }

        std::cout << "] shape=(";

        for (std::size_t i = 0; i < shape().size(); ++i) {
            if (i != 0) std::cout << ", ";
            std::cout << shape()[i];
        }

        std::cout << ") device=" << deviceName(device())
                  << '\n';
    }
};

// ------------------------------------------------------------
// Numerical gradient validation
// ------------------------------------------------------------

double numericalDerivative(
    const std::function<double(double)>& function,
    double input,
    double epsilon = 1e-6
) {
    return (
        function(input + epsilon) -
        function(input - epsilon)
    ) / (2.0 * epsilon);
}

// ------------------------------------------------------------
// Demonstrations
// ------------------------------------------------------------

void demonstrateTensors() {
    std::cout << "\n1. TENSOR FUNDAMENTALS\n";

    Tensor matrix(
        {1, 2, 3, 4},
        {2, 2}
    );

    Tensor bias(
        {10, 20},
        {1, 2}
    );

    matrix.print("Matrix");

    Tensor result = matrix.add(bias);
    result.print("Broadcast addition");

    Tensor right(
        {5, 6, 7, 8},
        {2, 2}
    );

    matrix.matmul(right).print("Matrix product");

    Tensor::ones({2, 3}).print("Ones");
    Tensor::zeros({2, 2}).print("Zeros");
}

void demonstrateAutograd() {
    std::cout << "\n2. AUTOMATIC DIFFERENTIATION\n";

    Tensor x({3.0}, {1}, true);

    Tensor y = x.square()
        .add(x.multiply(Tensor({2.0}, {1})))
        .add(Tensor({1.0}, {1}));

    y.print("Output");

    y.backward();

    std::cout << "Gradient dy/dx = "
              << x.gradients()[0] << '\n';

    if (std::abs(x.gradients()[0] - 8.0) > 1e-8) {
        throw TensorError("Gradient validation failed.");
    }

    Tensor vector({2.0, 3.0}, {2}, true);
    Tensor loss = vector.square().sum();

    loss.backward();

    vector.print("Input vector");

    std::cout << "Vector gradients: ";

    for (double gradient : vector.gradients()) {
        std::cout << gradient << ' ';
    }

    std::cout << '\n';
}

void demonstrateGradientChecking() {
    std::cout << "\n3. NUMERICAL GRADIENT CHECKING\n";

    Tensor x({2.0}, {1}, true);

    Tensor result = x.square().multiply(
        Tensor({3.0}, {1})
    ).sum();

    result.backward();

    const double analytical = x.gradients()[0];

    const auto function = [](double value) {
        return 3.0 * value * value;
    };

    const double numerical =
        numericalDerivative(function, 2.0);

    std::cout << "Analytical: " << analytical << '\n';
    std::cout << "Numerical: " << numerical << '\n';

    if (std::abs(analytical - numerical) > 1e-5) {
        throw TensorError("Numerical gradient mismatch.");
    }
}

void demonstrateFailureHandling() {
    std::cout << "\n4. FAILURE HANDLING\n";

    try {
        Tensor invalid({1, 2}, {3});
    } catch (const TensorError& error) {
        std::cout << "Shape validation: "
                  << error.what() << '\n';
    }

    try {
        Tensor a({1, 2, 3, 4}, {2, 2});
        Tensor b({1, 2, 3}, {3});

        a.matmul(b);
    } catch (const TensorError& error) {
        std::cout << "Matrix validation: "
                  << error.what() << '\n';
    }

    try {
        Tensor output({1, 2}, {2});
        output.backward();
    } catch (const TensorError& error) {
        std::cout << "Backward validation: "
                  << error.what() << '\n';
    }
}

} // namespace tensorlib

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main() {
    using namespace tensorlib;

    try {
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "PYTORCH FUNDAMENTALS\n";
        std::cout << "C++17 Tensor and Autograd Engine\n";

        demonstrateTensors();
        demonstrateAutograd();
        demonstrateGradientChecking();
        demonstrateFailureHandling();

        std::cout << "\nAll C++ demonstrations completed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Execution error: "
                  << error.what() << '\n';

        return 1;
    }
}
