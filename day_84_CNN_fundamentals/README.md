# CNN Fundamentals: Convolution, Filters, Kernels, Stride and Padding

## 1. Technical Scope

A Convolutional Neural Network (CNN) is a neural network architecture designed to process structured data, particularly images and other spatial signals.

CNNs extract spatial features by applying learnable filters to input tensors. Instead of connecting every input pixel to every neuron, convolutional layers use locally connected operations and shared weights.

The implementations in this repository demonstrate:

- Two-dimensional convolution and cross-correlation.
- Filters and kernels.
- Single-channel and multi-channel inputs.
- Multiple output feature maps.
- Stride and padding.
- Bias parameters.
- ReLU and sigmoid activation.
- Max pooling.
- Output dimension calculation.
- Trainable parameter calculation.
- Input validation and failure handling.
- Structured convolution experiment storage.

The implementations use standard language facilities without requiring external numerical computing libraries.

## 2. CNN Input Representation

An image is represented as a numerical tensor.

For a grayscale image:

- Height represents the number of rows.
- Width represents the number of columns.
- There is one input channel.

For an RGB image:

- Height represents the number of rows.
- Width represents the number of columns.
- There are three input channels: red, green and blue.

A general input tensor has dimensions:

\[
X \in \mathbb{R}^{C_{in}\times H\times W}
\]

Where:

| Symbol | Meaning |
|---|---|
| \(C_{in}\) | Number of input channels |
| \(H\) | Input height |
| \(W\) | Input width |
| \(X\) | Input tensor |

For example, an RGB image of dimensions 32 × 32 has the shape:

\[
3\times32\times32
\]

The three channels are processed together by each convolution filter.

## 3. Convolution and Cross-Correlation

Convolution is the fundamental spatial operation used by CNNs.

A small matrix of learnable weights moves across an input image. At each position, corresponding input and kernel values are multiplied and accumulated.

Deep-learning libraries commonly implement cross-correlation while referring to the operation as convolution.

The distinction is important:

- Mathematical convolution flips the kernel horizontally and vertically.
- Cross-correlation applies the kernel without flipping it.
- CNN implementations conventionally use cross-correlation.

The implementations in Python, JavaScript, C++17 and Java 17 use cross-correlation.

For a two-dimensional input and kernel, the operation is:

\[
Y(i,j)=
\sum_{m=0}^{K_h-1}
\sum_{n=0}^{K_w-1}
X(i+m,j+n)K(m,n)
\]

Where:

- \(X\) is the input matrix.
- \(K\) is the kernel.
- \(Y\) is the resulting feature map.
- \(K_h\) is kernel height.
- \(K_w\) is kernel width.

For a multi-channel input:

\[
Y_f(i,j)=b_f+
\sum_{c=0}^{C_{in}-1}
\sum_{m=0}^{K_h-1}
\sum_{n=0}^{K_w-1}
X(c,i+m,j+n)K_f(c,m,n)
\]

Each filter produces one output feature map.

The contributions from all input channels are accumulated into that feature map.

## 4. Filters and Kernels

A kernel is the spatial weight matrix applied to a local input region.

A filter contains the complete set of kernel weights across all input channels and usually includes a bias.

For a grayscale input, a filter may contain one 3 × 3 kernel.

For an RGB input, a filter contains three 3 × 3 kernels.

The resulting filter shape is:

\[
C_{in}\times K_h\times K_w
\]

A convolution layer generally contains multiple filters.

For example:

- Input channels: 3
- Number of filters: 16
- Kernel dimensions: 3 × 3

The output contains 16 feature maps.

Each filter learns a different spatial representation.

Typical learned features include:

- Horizontal edges.
- Vertical edges.
- Diagonal boundaries.
- Corners.
- Textures.
- Repeated patterns.
- More complex structures in deeper layers.

A filter is not necessarily a predefined edge detector. In a trainable CNN, its weights are optimized through training.

### Parameter calculation

Each filter contains:

\[
C_{in}\times K_h\times K_w
\]

Weight parameters and one bias parameter.

For \(F\) filters:

\[
P=F(C_{in}K_hK_w+1)
\]

Example:

| Parameter | Value |
|---|---:|
| Input channels | 3 |
| Filters | 16 |
| Kernel height | 3 |
| Kernel width | 3 |
| Weights per filter | 27 |
| Biases per filter | 1 |
| Total parameters | 448 |

The calculation is:

\[
P=16(3\times3\times3+1)
\]

\[
P=448
\]

## 5. Stride

Stride specifies how far a kernel moves between consecutive positions.

A stride of 1 moves the kernel one position at a time.

A stride of 2 moves it two positions at a time.

A larger stride reduces the spatial dimensions of the resulting feature map.

For VALID padding:

\[
H_{out}=
\left\lfloor
\frac{H-K_h}{S}
\right\rfloor+1
\]

\[
W_{out}=
\left\lfloor
\frac{W-K_w}{S}
\right\rfloor+1
\]

Where \(S\) represents stride.

Example:

| Input | Kernel | Stride | Output |
|---|---|---|---|
| 32 × 32 | 3 × 3 | 1 | 30 × 30 |
| 32 × 32 | 3 × 3 | 2 | 15 × 15 |
| 32 × 32 | 5 × 5 | 1 | 28 × 28 |
| 32 × 32 | 5 × 5 | 2 | 14 × 14 |

Stride changes spatial resolution without changing the number of filters.

## 6. Padding

Padding adds values around the input boundaries before convolution.

Zero padding is commonly used.

It controls output dimensions and determines how boundary pixels participate in convolution.

### VALID padding

VALID padding does not add any additional values.

Only positions where the entire kernel fits inside the original input are evaluated.

For a 5 × 5 input and a 3 × 3 kernel with stride 1:

\[
H_{out}=5-3+1=3
\]

\[
W_{out}=5-3+1=3
\]

The output shape is 3 × 3.

### SAME padding

SAME padding calculates padding to preserve the output spatial dimensions according to the stride.

The output dimensions are:

\[
H_{out}=\left\lceil\frac{H}{S}\right\rceil
\]

\[
W_{out}=\left\lceil\frac{W}{S}\right\rceil
\]

Total padding for height:

\[
P_h=
\max(0,(H_{out}-1)S+K_h-H)
\]

Total padding for width:

\[
P_w=
\max(0,(W_{out}-1)S+K_w-W)
\]

Padding is distributed across the two sides:

\[
P_{top}=\left\lfloor\frac{P_h}{2}\right\rfloor
\]

\[
P_{bottom}=P_h-P_{top}
\]

The same calculation applies to left and right padding.

When total padding is odd, the additional element is placed on the bottom or right.

### VALID and SAME comparison

| Property | VALID | SAME |
|---|---|---|
| Additional border values | None | Zero padding |
| Boundary coverage | Reduced | Preserved through padding |
| Output dimensions | Depend on kernel and stride | Ceiling of input divided by stride |
| Typical use | Spatial reduction | Controlled spatial dimensions |

For an input of 32 × 32 and a 3 × 3 kernel:

| Stride | VALID | SAME |
|---|---|---|
| 1 | 30 × 30 | 32 × 32 |
| 2 | 15 × 15 | 16 × 16 |

SAME does not always mean identical input and output dimensions. This is only guaranteed when stride is 1.

## 7. Bias and Activation

A convolution filter commonly includes a bias.

The bias is added after the weighted sum:

\[
Z_f(i,j)=Y_f(i,j)+b_f
\]

The bias allows a filter to shift its response independently of the weighted input.

The convolution output is often passed through a nonlinear activation function.

### ReLU

Rectified Linear Unit:

\[
ReLU(x)=\max(0,x)
\]

Negative values become zero. Positive values remain unchanged.

ReLU is computationally inexpensive and widely used in hidden CNN layers.

### Sigmoid

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Sigmoid maps numerical values into the interval:

\[
0<\sigma(x)<1
\]

The implementations use a numerically stable sigmoid calculation to avoid unnecessary exponential overflow.

Activation is separate from convolution. It does not change the kernel weights or the convolution operation itself.

## 8. Max Pooling

Pooling reduces the spatial dimensions of feature maps.

Max pooling selects the maximum value from each local region.

For a 2 × 2 pooling window:

\[
Y(i,j)=
\max
\begin{pmatrix}
X(2i,2j) &
X(2i,2j+1)\\
X(2i+1,2j) &
X(2i+1,2j+1)
\end{pmatrix}
\]

With a 2 × 2 window and stride 2:

| Input | Output |
|---|---|
| 4 × 4 | 2 × 2 |
| 8 × 8 | 4 × 4 |
| 16 × 16 | 8 × 8 |

Pooling operates independently on each feature map.

It does not learn kernel weights and does not perform convolution.

The implementations use valid-window max pooling without padding.

## 9. Complete CNN Processing Workflow

A basic CNN feature extraction pipeline can be represented as:

    Input Image
        |
        v
    Input Tensor
        |
        v
    Convolution Layer
        |
        v
    Filter Bank
        |
        v
    Feature Maps
        |
        v
    Bias Addition
        |
        v
    Activation Function
        |
        v
    Max Pooling
        |
        v
    Reduced Feature Maps

A deeper CNN can repeat convolution, activation and pooling stages before passing the resulting representations to classification or other prediction layers.

The source implementations demonstrate feature extraction rather than a complete trained image classifier.

## 10. Python Implementation

**File:** `cnn_fundamentals.py`

The Python implementation uses standard-library features and explicit tensor structures.

| Component | Implementation |
|---|---|
| Tensor representation | Nested lists |
| Configuration | Frozen dataclass |
| Padding | Enum |
| Filter management | FilterBank class |
| Convolution | `correlate2d()` and `FilterBank.apply()` |
| Activation | `relu()` and `sigmoid()` |
| Pooling | `max_pool2d()` |
| Parameter counting | `count_parameters()` |
| Validation | Custom exceptions and assertions |

The implementation progresses from single-channel convolution to multi-channel filters.

It also reports the input and output tensor shapes, applied padding and trainable parameter count.

The validation suite covers:

- Identity-kernel behavior.
- SAME padding.
- Irregular matrix dimensions.
- Oversized VALID kernels.
- Invalid stride.
- Non-finite values.

No external packages are required.

## 11. JavaScript Implementation

**File:** `cnn_fundamentals.js`

The JavaScript implementation uses a filter-oriented architecture.

Its principal components are:

| Component | Responsibility |
|---|---|
| `Filter` | Stores kernel weights and bias |
| `FilterBank` | Manages multiple filters |
| `CNNPipeline` | Coordinates inference stages |
| `calculatePadding()` | Computes padding distribution |
| `correlate2d()` | Performs two-dimensional cross-correlation |
| `maxPool2D()` | Performs spatial reduction |

### Asynchronous processing

The `applyAsync()` method uses promises to schedule individual filter computations.

`Promise.all()` collects the resulting feature maps.

The `CNNPipeline` class extends `EventTarget` and exposes execution events:

- `started`
- `completed`
- `failed`

Its state records whether execution is idle, running, completed or failed.

The asynchronous structure demonstrates how convolution operations can be organized within an event-driven application.

The implementation uses JavaScript's standard execution environment. Promise-based scheduling does not itself provide CPU parallelism. For actual parallel numerical computation, worker threads or another suitable computational backend would be needed.

## 12. C++17 Implementation

**File:** `cnn_fundamentals.cpp`

The C++ implementation uses contiguous tensor storage.

The `Tensor` structure stores values in a single vector rather than allocating an independent dynamic array for every row.

The flattened index is:

\[
Index=(c\times H\times W)+(r\times W)+col
\]

This provides predictable storage and avoids repeated nested-container allocations.

| Component | Responsibility |
|---|---|
| `Tensor` | Stores multi-channel input and output |
| `Kernel` | Stores filter weights and bias |
| `PaddingInfo` | Represents four-sided padding |
| `ConvResult` | Stores output and convolution metadata |
| `ConvolutionEngine` | Executes convolution |
| `applyReLU()` | Applies activation |
| `applyMaxPooling()` | Performs pooling |
| `countParameters()` | Calculates trainable parameters |

The convolution engine uses direct indexed access to the tensor and kernel storage.

It supports multiple filters and input channels.

The implementation also validates tensor dimensions, finite numerical values, kernel compatibility and stride.

The design provides a useful foundation for later SIMD optimization, memory blocking and parallel filter execution.

## 13. Java 17 Implementation

**File:** `CNNFundamentals.java`

The Java implementation models convolution as a service-oriented inference workflow.

It uses Java 17 language features and standard concurrency facilities.

| Component | Responsibility |
|---|---|
| `Tensor` | Encapsulates numerical input and output |
| `Filter` | Represents immutable kernel configuration |
| `ConvConfig` | Stores stride and padding settings |
| `ConvolutionService` | Performs convolution |
| `Activation` | Defines activation behavior |
| `ReLU` | Implements rectified linear activation |
| `Sigmoid` | Implements sigmoid activation |
| `PoolingService` | Performs max pooling |
| `InferencePipeline` | Coordinates processing stages |

The `Activation` interface separates activation behavior from convolution logic.

This permits activation implementations to be changed without modifying the convolution service.

### Enterprise-oriented processing

The inference pipeline uses:

- Explicit execution states.
- Configuration validation.
- Immutable configuration records.
- Dedicated processing services.
- CompletableFuture.
- ExecutorService.
- Exception propagation.
- Controlled executor shutdown.

The pipeline rejects overlapping execution requests.

Its state is updated when processing succeeds or fails.

The executor is shut down through the `AutoCloseable` lifecycle.

This separation makes it easier to test, maintain and extend individual processing stages.

The example is an educational service model rather than a production inference server. A real deployment would also require request limits, workload scheduling, resource monitoring and model-version management.

## 14. PostgreSQL Implementation

**File:** `cnn_fundamentals.sql`

The database implementation models convolution experiments and their stored numerical results.

It separates experiment configuration, layer definitions, filter metadata, kernel weights, input tensor values and output feature maps.

### Relational domain model

    Experiments
        |
        v
    Convolution Layers
        |
        v
    Filters
        |
        v
    Kernel Weights

    Experiments
        |
        v
    Tensor Values

    Convolution Layers
        |
        v
    Feature Map Values
        |
        v
    Feature Map Statistics
        |
        v
    Experiment Reports

### Main database tables

| Table | Purpose |
|---|---|
| `experiments` | Stores experiment configuration and lifecycle state |
| `convolution_layers` | Stores kernel dimensions, stride, padding and output dimensions |
| `filters` | Stores filter identities and biases |
| `kernel_weights` | Stores individual filter weights |
| `tensor_values` | Stores input tensor elements |
| `feature_map_values` | Stores raw and activated convolution outputs |

### Database integrity

The schema uses:

- Primary keys.
- Foreign keys.
- Unique constraints.
- Positive-dimension checks.
- Finite-value checks.
- Indexes for tensor coordinates and kernel retrieval.
- Triggers for layer and coordinate validation.
- Experiment state-transition validation.

The layer validation trigger verifies that each layer's input dimensions are compatible with the preceding layer's output.

It also verifies that the configured output dimensions agree with the selected padding and stride.

### PostgreSQL functions

`calculate_output_dimension()` calculates the spatial output dimension.

`calculate_padding()` calculates the padding before and after an input dimension.

`check_experiment_eligibility()` determines whether an experiment has the necessary input values, layers and filters.

### Convolution procedure

`execute_convolution()` performs the following operations:

1. Checks experiment eligibility.
2. Changes the experiment state to RUNNING.
3. Retrieves the configured convolution layers.
4. Retrieves each filter.
5. Validates kernel completeness.
6. Calculates convolution values.
7. Applies the configured activation.
8. Stores feature map elements.
9. Changes the experiment state to COMPLETED.

If execution fails, the procedure records a FAILED state and propagates the exception.

The procedure executes inside the caller's transaction. An unhandled error still causes the enclosing transaction to require rollback.

### Reporting

`feature_map_statistics` provides per-filter statistics.

`experiment_report` provides experiment-level information, including layer count, filter count, parameter count and stored output elements.

The script also includes retrieval, eligibility and validation queries.

PostgreSQL is used here as a relational experiment and result-management layer. It is not intended to replace optimized numerical computing libraries or GPU-based convolution engines.

## 15. Computational Complexity

Let:

- \(C_{in}\) represent input channels.
- \(F\) represent output filters.
- \(H_{out}\) represent output height.
- \(W_{out}\) represent output width.
- \(K_h\) represent kernel height.
- \(K_w\) represent kernel width.

The approximate multiplication count for one convolution layer is:

\[
O(FH_{out}W_{out}C_{in}K_hK_w)
\]

The operation count increases with the number of filters, input channels, output positions and kernel elements.

### Memory requirements

Input tensor storage:

\[
O(C_{in}HW)
\]

Filter storage:

\[
O(FC_{in}K_hK_w)
\]

Output tensor storage:

\[
O(FH_{out}W_{out})
\]

The C++ implementation uses contiguous tensor storage to improve memory locality.

The other implementations use nested structures for clarity and maintainability.

## 16. Important Engineering Considerations

### Numerical precision

The examples use double-precision floating-point values.

Floating-point arithmetic introduces rounding differences, so numerical outputs from different hardware and implementations may not always be bit-for-bit identical.

### Kernel size

Larger kernels examine wider spatial regions but increase computation and parameter requirements.

Smaller kernels reduce the number of parameters and are commonly stacked in deeper networks.

### Stride selection

A larger stride reduces output resolution and computation.

It can also discard spatial details.

### Padding selection

VALID reduces the spatial dimensions according to the kernel size.

SAME maintains dimensions according to the stride, using zero padding where required.

### Multi-channel filters

A filter must have one kernel for every input channel.

Each channel contributes to the same output feature map.

### Parameter sharing

The same filter weights are reused at every spatial position.

This reduces parameter requirements compared with a fully connected architecture.

### Activation placement

Activation is applied after convolution and bias addition.

Keeping activation separate from convolution makes the implementation easier to test and extend.

### Input validation

Malformed tensors, inconsistent channels, invalid strides, oversized VALID kernels and non-finite values must be rejected before numerical processing.

### Performance considerations

The reference implementations use direct convolution loops.

They are designed for learning and correctness rather than high-throughput production inference.

Optimized CNN implementations commonly use:

- Vectorized operations.
- SIMD instructions.
- GPU acceleration.
- Parallel filter execution.
- Cache-aware memory layouts.
- Specialized convolution algorithms.
- Reduced-precision arithmetic where supported.

### Database limitations

Storing every tensor element and feature map value as a separate relational record creates substantial database overhead.

The PostgreSQL implementation is useful for learning, experiment auditing and structured reporting.

Large-scale training and inference systems normally keep numerical tensor computation in specialized memory and compute systems, using databases for metadata, configuration, tracking and selected results.

## 17. Execution and Verification

All five source files are independent implementations.

| File | Runtime | External dependencies |
|---|---|---|
| `cnn_fundamentals.py` | Python 3 | None |
| `cnn_fundamentals.js` | Node.js | None |
| `cnn_fundamentals.cpp` | C++17 compiler | None |
| `CNNFundamentals.java` | Java 17 | None |
| `cnn_fundamentals.sql` | PostgreSQL 15+ | None |

The programs demonstrate:

- Basic edge-detection convolution.
- Multiple stride configurations.
- VALID and SAME padding.
- Multi-channel convolution.
- Multiple filters.
- Bias addition.
- Activation functions.
- Max pooling.
- Parameter calculation.
- Invalid-input rejection.

The SQL script creates a sample experiment, executes a convolution, stores the resulting feature map and runs reporting queries.

The numerical implementations use the same core mathematical definitions while adopting different data structures, execution models and software design approaches. 
