# Modern Convolutional Neural Networks

## Architecture Engineering: ResNet, DenseNet and EfficientNet

This project examines three important developments in convolutional neural network (CNN) architecture design:

- Residual learning through ResNet.
- Feature reuse through DenseNet.
- Compound model scaling through EfficientNet.

The implementations demonstrate architecture construction, tensor transformations, computational complexity estimation, connectivity validation, resource evaluation, and experiment management across Python, JavaScript, C++17, Java 17, and PostgreSQL.

The implementations are architecture-analysis and experiment-management systems. They calculate structural properties and estimated computational costs. They do not perform numerical forward propagation through learned weights, gradient computation, backpropagation, or actual model training.

## 1. Foundations of Modern CNNs

A convolutional neural network processes spatially structured data through layers that progressively transform image representations.

A conventional CNN generally contains:

1. Convolutional feature extraction.
2. Normalization.
3. Nonlinear activation.
4. Spatial downsampling.
5. Feature aggregation.
6. Classification.

A tensor representing an image batch can be expressed as:

\[
X \in \mathbb{R}^{N \times C \times H \times W}
\]

Where:

- \(N\): batch size.
- \(C\): number of channels.
- \(H\): image height.
- \(W\): image width.

The implementations model an individual feature tensor using its channel count, height, and width. Batch dimensions are excluded from architecture-level parameter and FLOP calculations.

### Convolution operation

A convolution extracts local spatial features using learnable filters.

For a convolution with input channels \(C_{in}\), output channels \(C_{out}\), kernel size \(K\), and groups \(G\), the number of weights is:

\[
P = C_{out}\frac{C_{in}}{G}K^2
\]

For a convolution with stride \(S\), padding \(P_d\), and input dimension \(D\), the output dimension is:

\[
D_{out} =
\left\lfloor
\frac{D+2P_d-K}{S}
\right\rfloor+1
\]

The approximate number of floating-point operations is:

\[
FLOPs \approx
2H_{out}W_{out}C_{out}
\frac{C_{in}}{G}K^2
\]

This convention counts one multiplication and one addition as two FLOPs. It excludes certain implementation-dependent operations, such as activation functions, memory movement, and some normalization overheads.

### Pooling

Pooling reduces spatial dimensions while retaining important feature information.

Max pooling selects the maximum value within a local window. Average pooling calculates the average.

Global average pooling reduces every spatial feature map to one value:

\[
y_c =
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}x_{cij}
\]

It produces a channel vector that can be supplied to a classification head.

## 2. Residual Networks: ResNet

<text-placeholder></text-placeholder>

ResNet addresses optimization difficulties associated with very deep conventional networks.

As network depth increases, optimization can become difficult because gradients must propagate through many transformations. Deeper networks may also experience degradation in training accuracy, even when additional layers should theoretically provide more representational capacity.

Residual learning changes the function learned by a block.

Instead of directly learning:

\[
H(x)
\]

A residual block learns:

\[
F(x)=H(x)-x
\]

The block produces:

\[
y=F(x)+x
\]

Here:

- \(x\) is the original feature tensor.
- \(F(x)\) is the learned residual transformation.
- \(y\) is the resulting feature tensor.

The shortcut allows information to travel through the block without passing through every transformation.

### Identity shortcut

An identity shortcut is appropriate when the input and output have identical dimensions:

\[
y=F(x)+x
\]

The addition requires:

\[
C_{in}=C_{out}
\]

\[
H_{in}=H_{out}
\]

\[
W_{in}=W_{out}
\]

### Projection shortcut

When channel dimensions or spatial dimensions change, a learned projection aligns the shortcut with the residual transformation.

\[
y=F(x)+W_sx
\]

Here, \(W_s\) generally represents a learned \(1\times1\) convolution, potentially with stride greater than one.

The projection is not required in every residual block. It is used when the dimensions cannot be matched through an identity connection.

### ResNet architecture

| Variant | Residual block | Typical block distribution |
|---|---|---|
| ResNet-18 | Basic | 2, 2, 2, 2 |
| ResNet-34 | Basic | 3, 4, 6, 3 |
| ResNet-50 | Bottleneck | 3, 4, 6, 3 |

The basic block generally uses two \(3\times3\) convolutions.

The bottleneck block uses:

1. A \(1\times1\) convolution to reduce channels.
2. A \(3\times3\) convolution for spatial feature extraction.
3. A \(1\times1\) convolution to expand channels.

This structure makes deeper networks more computationally manageable.

### Python implementation

The Python implementation includes:

- Basic residual blocks.
- Bottleneck residual blocks.
- Identity shortcut detection.
- Projection shortcut accounting.
- ResNet-18, ResNet-34, and ResNet-50 configurations.
- Tensor-shape validation.

The `residual_block()` function checks the dimensions of the transformed tensor and accounts for projection when necessary.

The residual addition is recorded as a separate operation. This distinction makes the computational graph easier to inspect.

### Engineering considerations

Residual connections improve information and gradient pathways, but they do not eliminate all optimization difficulties.

The shortcut must remain dimensionally compatible with the transformed branch. Projection shortcuts introduce additional parameters and computation.

Residual addition also requires both branches to be available at the same spatial resolution.

## 3. Densely Connected Networks: DenseNet

DenseNet introduces a different connectivity strategy.

Instead of connecting only the input and output of an individual residual block, DenseNet makes earlier feature representations available to subsequent layers within a dense block.

For a dense block:

\[
x_l=H_l([x_0,x_1,\ldots,x_{l-1}])
\]

Where:

- \(x_0\) is the initial block input.
- \(x_l\) is the output of the current layer.
- \(H_l\) represents the current layer's transformation.
- \([\cdot]\) represents channel-wise concatenation.

Every layer receives the accumulated features from preceding layers.

### Growth rate

DenseNet introduces the growth rate \(k\).

Each dense layer contributes \(k\) new feature channels.

If the initial channel count is \(C_0\), after \(L\) dense layers:

\[
C_L=C_0+kL
\]

For example, an initial tensor with 64 channels and a growth rate of 32 produces:

\[
C_L=64+(32\times6)=256
\]

after six dense layers.

The increasing channel count is a defining property of dense connectivity.

### Dense bottleneck

DenseNet commonly employs a bottleneck transformation:

\[
1\times1 \rightarrow 3\times3
\]

The \(1\times1\) convolution reduces the computational burden of the following spatial convolution.

The bottleneck commonly produces \(4k\) intermediate channels before generating \(k\) new feature channels.

### Transition layers

Dense blocks increase channel counts. Transition layers control the resulting model size.

A transition layer generally includes:

- Batch normalization.
- Nonlinear activation.
- \(1\times1\) convolution.
- Average pooling.

Compression is controlled by a factor:

\[
C_{transition}=\lfloor\theta C_{input}\rfloor
\]

Where:

\[
0<\theta\leq1
\]

A compression factor of 0.5 approximately halves the channel count.

### DenseNet-121

The implemented DenseNet-121 configuration uses four dense blocks:

| Dense block | Number of layers |
|---|---:|
| Block 1 | 6 |
| Block 2 | 12 |
| Block 3 | 24 |
| Block 4 | 16 |

The growth rate is 32.

### Python implementation

The `dense_layer()` function records:

1. Bottleneck processing.
2. Spatial convolution.
3. Generation of new feature maps.
4. Channel-wise concatenation.

The `transition_layer()` function implements channel compression and spatial downsampling.

The concatenation operation increases the number of channels rather than adding two tensors element by element.

### Engineering considerations

Dense connectivity improves feature reuse and provides direct information pathways between layers.

It can reduce the need to relearn similar features at different depths.

The main engineering challenge is memory consumption. Earlier feature maps remain relevant to later layers, so dense connectivity can increase activation storage and memory traffic.

Concatenation also creates implementation-specific memory allocation and bandwidth costs.

A model with relatively few parameters is not necessarily the model with the lowest memory consumption.

## 4. EfficientNet

EfficientNet focuses on balancing model accuracy and computational efficiency.

Traditional scaling methods commonly increase one dimension:

- Depth scaling adds layers.
- Width scaling increases channels.
- Resolution scaling increases input image dimensions.

EfficientNet introduces compound scaling, which coordinates these dimensions.

### Compound scaling

The scaling relationship can be represented as:

\[
d=\alpha^\phi
\]

\[
w=\beta^\phi
\]

\[
r=\gamma^\phi
\]

Where:

- \(d\) is the depth multiplier.
- \(w\) is the width multiplier.
- \(r\) is the resolution multiplier.
- \(\phi\) is the compound scaling coefficient.
- \(\alpha,\beta,\gamma\) are scaling constants.

The scaling constants are selected subject to a computational resource constraint.

An approximate relationship for convolutional computation is:

\[
FLOPs \propto d\,w^2\,r^2
\]

This relationship explains why increasing all three dimensions requires careful coordination.

### EfficientNet variants

| Variant | Width | Depth | Input resolution |
|---|---:|---:|---:|
| B0 | 1.0 | 1.0 | 224 |
| B1 | 1.0 | 1.1 | 240 |
| B2 | 1.1 | 1.2 | 260 |
| B3 | 1.2 | 1.4 | 300 |
| B4 | 1.4 | 1.8 | 380 |
| B5 | 1.6 | 2.2 | 456 |
| B6 | 1.8 | 2.6 | 528 |
| B7 | 2.0 | 3.1 | 600 |

These configurations illustrate coordinated scaling. They are not guarantees of identical hardware latency or memory consumption across devices.

### MBConv blocks

EfficientNet uses mobile inverted bottleneck convolution blocks.

A typical MBConv structure consists of:

1. Expansion convolution.
2. Depthwise spatial convolution.
3. Squeeze-and-Excitation.
4. Pointwise projection.
5. Optional residual connection.

The expansion stage increases the number of channels before spatial processing.

The depthwise convolution processes each channel independently.

The projection stage reduces the expanded representation to the required output channel count.

### Depthwise separable convolution

A conventional convolution applies a spatial kernel across all input channels.

A depthwise convolution applies separate spatial kernels to individual channels.

Its parameter count is:

\[
P_{depthwise}=C_{in}K^2
\]

A pointwise convolution then mixes information across channels:

\[
P_{pointwise}=C_{in}C_{out}
\]

The combination can substantially reduce parameter and computation requirements compared with a full spatial convolution.

### Squeeze-and-Excitation

Squeeze-and-Excitation recalibrates channel responses.

The squeeze stage uses global average pooling:

\[
z_c=\frac{1}{HW}\sum_{i,j}x_{cij}
\]

The excitation stage uses two learned transformations:

\[
s=\sigma(W_2\delta(W_1z))
\]

Where:

- \(W_1\) reduces channel dimensionality.
- \(\delta\) is a nonlinear activation.
- \(W_2\) restores channel dimensionality.
- \(\sigma\) is a sigmoid function.

The resulting channel weights recalibrate the original feature maps:

\[
\hat{x}_c=s_cx_c
\]

This allows the network to emphasize informative channels.

### Python implementation

The implementation includes:

- EfficientNet B0 through B7 scaling configurations.
- Channel rounding to multiples of eight.
- MBConv block construction.
- Expansion and depthwise convolution.
- Squeeze-and-Excitation parameter estimation.
- Conditional residual connections.
- Classification head construction.

The `round_channels()` function ensures that channel dimensions follow the model's channel-divisibility convention.

## 5. Distinguishing the Three Architectures

The architectures address different engineering problems.

| Property | ResNet | DenseNet | EfficientNet |
|---|---|---|---|
| Main mechanism | Residual addition | Feature concatenation | Compound scaling |
| Connectivity | Identity or projection shortcuts | All preceding dense-block features | Conditional MBConv shortcuts |
| Main objective | Improve deep optimization | Feature reuse | Accuracy and efficiency balance |
| Channel behavior | Usually stage-based | Increases within dense blocks | Scaled by width coefficient |
| Spatial processing | Standard convolutions | Standard convolutions and transitions | Depthwise and pointwise convolutions |
| Attention mechanism | Not intrinsic to standard ResNet | Not intrinsic to standard DenseNet | SE in standard MBConv configurations |
| Main tradeoff | Additional shortcut and block computation | Activation memory and concatenation | Scaling and hardware-dependent efficiency |

These approaches are not mutually exclusive at the level of individual operations. Modern architectures can combine residual connections, attention mechanisms, depthwise convolutions, and other techniques.

The important distinction is the primary architectural strategy.

## 6. Python Architecture Analysis

File: `modern_cnn.py`

The Python program acts as an architecture construction and analysis engine.

### Domain structures

| Component | Responsibility |
|---|---|
| `TensorShape` | Represents feature-map dimensions |
| `Layer` | Stores individual operation metadata |
| `Architecture` | Maintains the computation sequence |
| `LayerType` | Classifies operations |
| `ArchitectureError` | Handles invalid configurations |

### Architecture construction

The program constructs five model configurations:

- ResNet-18.
- ResNet-50.
- DenseNet-121.
- EfficientNet-B0.
- EfficientNet-B3.

The architecture builders use reusable operations rather than embedding shape calculations in individual model definitions.

### Computational analysis

Each operation records:

- Input dimensions.
- Output dimensions.
- Parameter count.
- Estimated FLOPs.
- Layer-specific configuration.

Total parameters are calculated as:

\[
P_{model}=\sum_{l=1}^{L}P_l
\]

Estimated FLOPs are calculated as:

\[
F_{model}=\sum_{l=1}^{L}F_l
\]

The report includes individual layer metadata and aggregate architecture statistics.

### Validation

The validation process checks:

- Positive tensor dimensions.
- Valid convolution groups.
- Positive output dimensions.
- Nonnegative computational costs.
- Sequential tensor consistency.
- Classification output dimensions.
- Valid model variants.
- Valid dense-block configurations.

The script executes assertions before generating the report.

### Output

Execution generates `modern_cnn_report.json`.

The JSON file contains architecture configurations, individual layer records, tensor shapes, parameter totals, and FLOP estimates.

## 7. JavaScript Event-Driven Experiment Management

File: `modern_cnn.js`

The JavaScript implementation provides a complementary experiment-orchestration perspective.

It uses Node.js standard-library features rather than external packages.

### Graph modeling

`CNNGraph` represents the computational graph through an ordered collection of nodes.

Each node contains:

- Operation name.
- Operation type.
- Input tensor.
- Output tensor.
- Parameter count.
- FLOP estimate.

The graph rejects invalid sequential tensor connections.

Residual additions and dense concatenations have dedicated validation methods.

### Experiment lifecycle

The `Experiment` class models an experiment as a stateful object.

The supported lifecycle is:

`CREATED → VALIDATING → RUNNING → COMPLETED`

Failure and cancellation states are separately represented.

An invalid transition is rejected rather than silently changing the experiment state.

### Event-driven behavior

The experiment extends Node.js `EventEmitter`.

It publishes:

- `stateChanged`
- `completed`
- `failed`

This separates experiment execution from monitoring and reporting.

A monitoring service can subscribe to lifecycle events without modifying architecture construction.

### Asynchronous scheduling

`ExperimentScheduler` limits concurrent execution.

The scheduler maintains:

- A pending queue.
- A concurrency limit.
- An active execution count.
- Individual completion and rejection handlers.

`Promise.allSettled()` allows successful and failed experiment results to be collected together.

This prevents one failed experiment from hiding the outcomes of other experiments.

### Resource policies

The JavaScript implementation evaluates:

- Maximum parameters.
- Maximum FLOPs.
- Valid architecture construction.

A resource violation prevents execution from progressing into the running state.

The output is written to `modern_cnn_experiments.json`.

## 8. C++17 Architecture and Resource Engine

File: `modern_cnn.cpp`

The C++ implementation emphasizes explicit tensor modeling, computational accounting, and resource eligibility.

### Domain modeling

| Structure or class | Responsibility |
|---|---|
| `Tensor` | Tensor dimensions and element count |
| `Layer` | Computational operation |
| `Architecture` | Complete model representation |
| `CNNBuilder` | Architecture construction |
| `ResourcePolicy` | Deployment constraints |
| `EligibilityEngine` | Architecture eligibility decisions |
| `CNNException` | Domain validation failures |

### Computational graph construction

`CNNBuilder` provides dedicated methods for:

- Convolution.
- Grouped convolution.
- Depthwise convolution.
- Pooling.
- Normalization.
- Activation.
- Residual addition.
- Feature concatenation.
- Squeeze-and-Excitation.
- Classification.

The builder maintains the current tensor shape.

Every operation receives a source shape and produces a validated output shape.

### Memory estimation

The implementation estimates activation storage using:

\[
M=HWCb
\]

Where \(b\) is the number of bytes per element.

For FP32:

\[
b=4
\]

The current estimate records the largest individual feature tensor.

It does not represent total training memory, simultaneous live tensors, optimizer state, gradients, workspace allocations, or peak GPU memory.

### Resource eligibility

The `EligibilityEngine` evaluates three constraints:

1. Parameter budget.
2. FLOP budget.
3. Activation memory budget.

An architecture that exceeds any configured limit is rejected.

This demonstrates the distinction between constructing a valid architecture and determining whether it is suitable for a deployment environment.

### Complexity considerations

Architecture construction and report generation are approximately linear in the number of recorded layers.

The model stores layer metadata in an STL vector.

Parameter and FLOP aggregation traverse this vector.

The implementation uses `long long` for computational totals to reduce the risk of integer overflow associated with large models.

## 9. Java 17 Enterprise Experiment Management

File: `ModernCNN.java`

The Java implementation models architecture evaluation as an enterprise-oriented experiment workflow.

It uses records, enums, interfaces, immutable report structures, collections, and concurrent execution.

### Domain types

- `TensorShape`: immutable tensor dimensions.
- `Layer`: immutable layer metadata.
- `ArchitecturePolicy`: deployment resource policy.
- `ArchitectureReport`: experiment output.
- `ArchitectureFamily`: architecture classification.
- `ExperimentState`: experiment lifecycle.
- `DeploymentTarget`: deployment environment.

### Policy-based validation

`ExperimentValidator` defines the validation contract.

`ResourcePolicyValidator` implements the contract.

The validation service checks:

- Maximum parameters.
- Maximum FLOPs.
- Maximum activation memory.
- Input-resolution restrictions for mobile deployment.

This separates resource policies from architecture construction.

A different deployment environment can use a different policy without changing the architecture implementation.

### State management

The `Experiment` class enforces explicit state transitions.

Invalid transitions raise a domain exception.

Failed validation changes the experiment state to `FAILED`.

Successful validation allows execution to proceed.

### Concurrency

`ExperimentService` uses a fixed-size `ExecutorService`.

`CompletableFuture` represents asynchronous experiment results.

The configured thread-pool size limits the number of simultaneously executing experiment tasks.

The service performs orderly shutdown and restores the thread interruption state when interrupted.

### Enterprise engineering implications

The architecture registry, experiment lifecycle, policy validation, and report generation are separated.

This supports independent evolution of:

- Architecture definitions.
- Resource governance.
- Experiment scheduling.
- Execution monitoring.
- Evaluation reporting.

## 10. PostgreSQL Research Database

File: `modern_cnn.sql`

The database provides a relational model for architecture research and experiment tracking.

### Entity relationships

The primary relationships are:

`Researchers`
  → `Architectures`
  → `Architecture Configurations`
  → `Architecture Layers`
  → `Layer Connections`

`Datasets`
  → `Experiments`
  → `Training Runs`
  → `Evaluation Metrics`

`Deployment Policies`
  → `Experiments`
  → `Experiment Eligibility`

### Database entities

| Table | Purpose |
|---|---|
| `researchers` | Researcher identities |
| `datasets` | Image dataset metadata |
| `architectures` | CNN architecture registry |
| `architecture_configurations` | Model-specific hyperparameters |
| `architecture_layers` | Layer-level architecture information |
| `layer_connections` | Residual and dense connectivity |
| `deployment_policies` | Deployment resource constraints |
| `experiments` | Experiment definitions and lifecycle |
| `training_runs` | Individual training executions |
| `evaluation_metrics` | Accuracy and other measured metrics |
| `experiment_eligibility` | Policy evaluation results |

### Referential integrity

Primary keys identify individual records.

Foreign keys maintain relationships between researchers, architectures, datasets, experiments, and runs.

Unique constraints prevent duplicate architecture names, experiment codes, configuration versions, and run numbers.

Check constraints enforce valid tensor dimensions, resource limits, learning rates, batch sizes, and experiment states.

### Architecture configurations

PostgreSQL JSONB stores architecture-specific configuration data.

This is useful because the three architecture families have different configuration requirements.

ResNet uses residual block configurations.

DenseNet uses growth rates and compression factors.

EfficientNet uses width, depth, resolution, and expansion parameters.

Structured columns remain available for frequently queried properties.

### Layer connections

`architecture_layers` stores individual computational operations.

`layer_connections` stores graph edges.

The distinction is important because a strictly sequential layer list cannot fully represent residual and dense connectivity.

The connection trigger validates:

- Source layer existence.
- Target layer existence.
- Architecture ownership.
- Topological layer ordering.

The trigger rejects backward connections, preventing simple cyclic dependencies in the recorded architecture graph.

### Automatic architecture metrics

The `refresh_architecture_metrics()` trigger recalculates parameter totals, FLOPs, and layer counts when layer records change.

This reduces the risk of maintaining inconsistent aggregate architecture statistics.

The trigger does not automatically determine whether every architecture is semantically correct. That remains the responsibility of architecture validation.

### Deployment eligibility

The `evaluate_architecture_eligibility()` function checks an architecture against a deployment policy.

The function considers:

- Parameter count.
- FLOP count.
- Activation memory.
- Input resolution.
- Architecture validation status.

It returns an eligibility decision and a rejection reason.

The `register_experiment()` procedure prevents an ineligible architecture from being registered under the selected deployment policy.

### Reporting

The database provides:

- `architecture_comparison`
- `experiment_performance`

It also includes queries using:

- Common table expressions.
- Aggregation.
- Window functions.
- JSONB operators.
- Conditional expressions.
- Lateral function evaluation.

These operations support architecture comparison, model selection, and experiment monitoring.

### Transaction management

The registration workflow uses transactions to maintain data consistency.

The rollback demonstration shows how database constraints reject invalid experiment configurations without retaining partially inserted records.

## 11. Parameter Count, FLOPs and Memory Are Different Metrics

Three metrics are particularly important when evaluating modern CNN architectures.

### Parameter count

Parameter count represents the number of trainable model values.

It influences:

- Model storage.
- Checkpoint size.
- Optimizer memory.
- Potential deployment footprint.

A lower parameter count does not necessarily guarantee faster inference.

### FLOPs

FLOPs estimate arithmetic computation.

They are useful for comparing theoretical computational requirements.

They do not directly determine actual execution time.

Hardware architecture, kernel optimization, memory bandwidth, parallelism, precision, and implementation quality all influence latency.

### Activation memory

Activation memory represents intermediate feature-map storage.

For a single FP32 feature tensor:

\[
M=4CHW
\]

DenseNet can have substantial activation-storage requirements because multiple preceding feature representations remain relevant.

ResNet can also require significant intermediate storage during training because backward propagation needs information from earlier operations.

EfficientNet reduces certain convolutional costs through depthwise processing, but input resolution and feature-map dimensions still influence memory requirements.

### Comparison framework

| Metric | What it measures | Main engineering concern |
|---|---|---|
| Parameters | Model weights | Storage and optimizer state |
| FLOPs | Arithmetic operations | Computational demand |
| Activation memory | Intermediate tensors | Memory capacity and bandwidth |
| Latency | Actual execution time | Hardware and implementation |
| Accuracy | Predictive performance | Generalization |
| Throughput | Samples processed per unit time | Parallel execution |

The supplied implementations calculate the first three structural metrics. Actual latency, throughput, and predictive accuracy require execution against real hardware and datasets.

## 12. Validation and Failure Conditions

The five implementations demonstrate different validation strategies.

| Failure condition | Expected handling |
|---|---|
| Nonpositive tensor dimensions | Reject configuration |
| Invalid convolution groups | Reject operation |
| Invalid output dimensions | Reject architecture |
| Residual dimension mismatch | Reject addition |
| Dense spatial mismatch | Reject concatenation |
| Invalid EfficientNet variant | Reject configuration |
| Invalid resource budget | Reject policy |
| Excessive parameter count | Reject experiment |
| Excessive FLOPs | Reject experiment |
| Excessive activation estimate | Reject deployment |
| Invalid lifecycle transition | Reject transition |
| Failed asynchronous experiment | Record failure |

The validation mechanisms operate at different levels.

Tensor validation protects individual operations.

Architecture validation protects graph consistency.

Resource validation determines deployment suitability.

Experiment-state validation protects workflow integrity.

These controls should not be treated as interchangeable.

## 13. Performance and Security Considerations

### Performance

- Use grouped and depthwise convolution where supported by the architecture.
- Avoid unnecessary tensor copies when implementing dense feature concatenation.
- Apply global average pooling to reduce classification-head dimensionality.
- Use channel rounding consistently when implementing EfficientNet variants.
- Limit concurrent experiments according to available CPU, GPU, and memory resources.
- Store architecture metadata separately from large model checkpoints.
- Index frequently queried architecture and experiment attributes.
- Use database aggregation rather than repeatedly transferring large result sets.

### Numerical precision

FP32 is the assumed storage format for the C++ activation estimate.

Actual deployments may use FP16, BF16, or integer quantization.

Mixed precision and quantization can reduce memory consumption and improve performance, depending on hardware support.

The parameter and FLOP estimates in this project are independent of numerical precision.

### Security

A production experiment-management system should enforce:

- Authentication for researcher accounts.
- Role-based authorization.
- Restricted database privileges.
- Validation of dataset and checkpoint locations.
- Protection of experiment credentials.
- Controlled access to training artifacts.
- Audit records for configuration and policy changes.
- Encryption of sensitive datasets and model artifacts.
- Validation of imported architecture configurations.

The sample database provides relational integrity and controlled workflows. It does not implement a complete identity-management or authorization system.

## 14. Limitations and Interpretation

The implementations are deliberately focused on architecture structure and resource estimation.

They do not implement:

- Actual convolution tensor arithmetic.
- Learned model weights.
- Automatic differentiation.
- Backpropagation.
- Optimizer updates.
- Dataset loading and augmentation.
- Distributed training.
- GPU kernel execution.
- Model checkpoint serialization.
- Accuracy benchmarking.

The architecture builders use simplified computational accounting. Normalization, activation, pooling, squeeze-and-excitation, and concatenation estimates are analytical approximations.

The EfficientNet implementations demonstrate the core compound-scaling and MBConv mechanisms, not every detail of the complete original training recipes. Some architecture-specific choices, including activation functions, stochastic depth, initialization, normalization behavior, and dropout, are not fully modeled.

The database's seeded layer records are illustrative metadata records rather than complete layer-by-layer reconstructions of all registered architectures.

For meaningful performance comparisons, all models should be evaluated using the same dataset, preprocessing pipeline, batch size, numerical precision, hardware, measurement procedure, and evaluation protocol.
