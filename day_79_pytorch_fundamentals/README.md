# PyTorch Fundamentals: Tensors, Devices, Autograd and Tensor Operations

## 1. Technical Scope

PyTorch is a tensor-based computational framework used for numerical computing, automatic differentiation, deep learning, and GPU-accelerated workloads.

Its fundamental computational model combines:

- Multidimensional tensor storage.
- Shape-aware mathematical operations.
- CPU and accelerator device execution.
- Computational graph construction.
- Automatic differentiation.
- Gradient-based optimization.
- Neural network execution.

The five implementations approach these concepts from different engineering perspectives.

| Implementation | Engineering focus |
|---|---|
| Python | Actual PyTorch operations and neural network training |
| JavaScript | Typed-array tensor engine and event-driven execution |
| C++17 | Tensor representation and reverse-mode differentiation |
| Java 17 | Enterprise-oriented tensor and experiment management |
| PostgreSQL | Tensor metadata, computation records, and experiment governance |

The Python implementation uses the actual PyTorch library. The other implementations provide educational computational models rather than reproducing the complete PyTorch runtime.

## 2. Tensor Fundamentals

### 2.1 Definition

A tensor is a multidimensional numerical data structure.

Its dimensions determine its shape, while its data type determines how individual values are represented.

Common tensor categories include:

| Tensor category | Example shape | Description |
|---|---|---|
| Scalar | `[]` | Single numerical value |
| Vector | `[5]` | One-dimensional collection |
| Matrix | `[3, 4]` | Two-dimensional collection |
| Three-dimensional tensor | `[2, 3, 4]` | Three-dimensional collection |
| Batch tensor | `[32, 3, 224, 224]` | Collection of image tensors |

PyTorch supports tensors with different dimensions and data types.

A tensor is described by several properties:

- `shape`: Number of elements along each dimension.
- `ndim`: Number of dimensions.
- `dtype`: Numerical representation.
- `device`: Execution location.
- `requires_grad`: Whether gradient tracking is enabled.
- `grad`: Accumulated gradient, when available.
- `numel()`: Total number of elements.

### 2.2 Tensor Construction

The Python implementation demonstrates:

- `torch.tensor()` for explicit values.
- `torch.zeros()` for zero initialization.
- `torch.ones()` for one initialization.
- `torch.randn()` for normally distributed random values.
- `torch.arange()` for sequential values.

The distinction between initialization methods matters during model development.

For example, random initialization is commonly used for trainable neural network parameters, while zero initialization is useful for temporary buffers and mathematical demonstrations.

### 2.3 Data Types

Important PyTorch data types include:

| Data type | Typical application |
|---|---|
| `torch.float16` | Reduced-precision computation |
| `torch.bfloat16` | Reduced-precision machine learning |
| `torch.float32` | Standard neural network computation |
| `torch.float64` | Numerical precision and gradient verification |
| `torch.int32` | Integer computation |
| `torch.int64` | Indexing and integer identifiers |
| `torch.bool` | Boolean masks |

Floating-point precision affects memory consumption, numerical stability, and computational performance.

For example, a float32 value occupies four bytes, while a float64 value occupies eight bytes.

For a tensor containing one million elements:

| Data type | Approximate storage |
|---|---:|
| float16 | 2 MB |
| float32 | 4 MB |
| float64 | 8 MB |

These values represent element storage only. Actual memory consumption can be higher because of intermediate tensors, gradients, model parameters, and runtime allocations.

## 3. Tensor Shapes, Storage and Memory

### 3.1 Shape Compatibility

Tensor operations depend on compatible dimensions.

For example:

- A matrix with shape `[2, 3]` can multiply a matrix with shape `[3, 4]`.
- The resulting matrix has shape `[2, 4]`.
- Element-wise operations require compatible shapes or valid broadcasting relationships.

Shape validation prevents incorrect computations and accidental memory allocation.

### 3.2 Broadcasting

Broadcasting allows compatible tensors with different shapes to participate in the same operation without explicitly copying all values.

Consider:

- Matrix: `[2, 3]`
- Bias vector: `[3]`

Adding the bias vector to the matrix applies the same three bias values to both rows.

The broadcasting rules compare dimensions from right to left.

Two dimensions are compatible when:

- Their sizes are equal.
- The first dimension has size one.
- The second dimension has size one.

For example:

| Shape A | Shape B | Result |
|---|---|---|
| `[2, 3]` | `[3]` | `[2, 3]` |
| `[4, 1, 3]` | `[1, 5, 3]` | `[4, 5, 3]` |
| `[2, 3]` | `[2, 4]` | Invalid |

Broadcasting does not make every differently shaped tensor compatible.

### 3.3 Tensor Operations

The implementations demonstrate the following mathematical operations.

| Operation | PyTorch expression | Purpose |
|---|---|---|
| Addition | `a + b` | Element-wise addition |
| Subtraction | `a - b` | Element-wise difference |
| Multiplication | `a * b` | Element-wise multiplication |
| Division | `a / b` | Element-wise division |
| Matrix multiplication | `torch.matmul(a, b)` | Linear algebra |
| Transpose | `a.T` | Exchange matrix dimensions |
| Reduction | `a.sum()` | Aggregate values |
| Mean | `a.mean()` | Calculate average |
| Reshape | `a.reshape(...)` | Change tensor dimensions |
| Concatenation | `torch.cat()` | Join tensors along an axis |
| Stacking | `torch.stack()` | Introduce a new dimension |
| Softmax | `torch.softmax()` | Normalize scores into probabilities |

### 3.4 Matrix Multiplication

For matrices:

\[
A \in \mathbb{R}^{m \times n}
\]

\[
B \in \mathbb{R}^{n \times p}
\]

Their product is:

\[
C = AB
\]

where:

\[
C \in \mathbb{R}^{m \times p}
\]

Each output element is calculated as:

\[
C_{ij} =
\sum_{k=1}^{n} A_{ik}B_{kj}
\]

The conventional computational complexity is:

\[
O(mnp)
\]

Matrix multiplication is central to neural network execution because fully connected layers and many attention operations depend on it.

### 3.5 Views, Reshaping and Copies

Tensor transformations have different memory implications.

**View**

`tensor.view()` returns a different interpretation of compatible underlying storage. It generally requires compatible memory layout and strides.

**Reshape**

`tensor.reshape()` attempts to return a view when possible. Otherwise, it can allocate a copy.

**Clone**

`tensor.clone()` creates independent tensor storage while preserving the differentiable relationship when gradient tracking is active.

**Detach**

`tensor.detach()` produces a tensor disconnected from the original autograd graph.

These operations should not be treated as interchangeable.

The Python implementation demonstrates storage sharing, independent copies, contiguity, and transposition.

## 4. Device Management

A tensor's device determines where its operations execute.

The principal device categories are:

| Device | Purpose |
|---|---|
| CPU | General-purpose computation |
| CUDA | NVIDIA GPU computation |

PyTorch identifies devices through objects such as `torch.device("cpu")` and `torch.device("cuda")`.

### 4.1 Device Selection

The Python implementation automatically selects CUDA when available and otherwise selects CPU.

Explicit selection is also supported.

A CUDA device must be available before attempting to allocate or transfer tensors to it.

### 4.2 Moving Tensors

The following operations demonstrate device transfer:

- `tensor.to(device)`
- `tensor.cpu()`
- `tensor.cuda()`

Operands participating in an operation generally need compatible devices.

A CPU tensor and a CUDA tensor cannot ordinarily participate directly in the same mathematical operation.

The required transfer must be explicit.

### 4.3 GPU Memory

GPU memory consumption involves:

- Model parameters.
- Input tensors.
- Intermediate activations.
- Gradient tensors.
- Optimizer state.
- Cached allocations.

PyTorch exposes memory information through:

- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- `torch.cuda.empty_cache()`

Allocated memory and reserved memory describe different aspects of the CUDA allocator.

Reserved memory includes memory managed by PyTorch's caching allocator. It should not automatically be interpreted as memory currently occupied by live tensor values.

### 4.4 GPU Synchronization

CUDA operations are generally asynchronous with respect to the CPU.

A timing measurement that immediately stops after launching a GPU operation can underestimate actual execution time.

The Python performance example uses:

`torch.cuda.synchronize()`

before and after the measured operation.

This ensures that queued CUDA work has completed before the timing measurement is finalized.

## 5. Automatic Differentiation

### 5.1 Definition

Automatic differentiation calculates derivatives by applying the chain rule to a sequence of mathematical operations.

PyTorch implements automatic differentiation through its autograd system.

When gradient tracking is enabled, PyTorch records relevant operations and constructs a computational graph.

The graph represents relationships between input tensors, intermediate results, and output tensors.

### 5.2 Computational Graph

Consider:

\[
y = x^2 + 2x + 1
\]

The computation can be represented as:

- Input: \(x\)
- Square operation: \(x^2\)
- Multiplication: \(2x\)
- Addition: \(x^2 + 2x\)
- Final addition: \(x^2 + 2x + 1\)

For \(x=3\):

\[
y = 9 + 6 + 1 = 16
\]

The derivative is:

\[
\frac{dy}{dx}=2x+2
\]

Therefore:

\[
\frac{dy}{dx}\bigg|_{x=3}=8
\]

The Python implementation verifies this calculation through `backward()`.

### 5.3 Gradient Tracking

A tensor can participate in automatic differentiation when created with:

`requires_grad=True`

Operations involving such tensors can create a computational graph.

The resulting tensor may expose its gradient through:

`tensor.grad`

Not every tensor retains gradients. Intermediate tensors generally do not retain their gradients unless `retain_grad()` is explicitly requested.

Leaf tensors that require gradients are the usual locations where gradients are accumulated.

### 5.4 Backward Propagation

Calling:

`loss.backward()`

initiates reverse-mode automatic differentiation.

The system traverses the computation graph in reverse dependency order.

Each operation applies its local derivative to the gradient received from downstream operations.

The chain rule determines how gradients propagate through the graph.

For a composition:

\[
y=f(g(x))
\]

the derivative is:

\[
\frac{dy}{dx}
=
\frac{df}{dg}
\frac{dg}{dx}
\]

Reverse-mode differentiation is especially useful when a computation has many inputs and relatively few outputs, such as a scalar neural network loss.

### 5.5 Gradient Accumulation

PyTorch accumulates gradients into existing `.grad` values by default.

For example, repeated backward passes can produce:

\[
g_{\text{total}}=g_1+g_2
\]

This behavior is important when implementing training loops.

The usual optimization workflow resets gradients before computing the next batch:

`optimizer.zero_grad(set_to_none=True)`

The alternative `tensor.grad = None` is also demonstrated in the Python script.

The custom C++ and JavaScript implementations explicitly accumulate gradients at parent nodes. Their backward-pass reset behavior is separately implemented and should not be confused with PyTorch's default accumulation behavior.

## 6. Gradient Control and Graph Management

PyTorch provides several mechanisms for controlling gradient tracking.

| Mechanism | Behavior |
|---|---|
| `requires_grad=True` | Enables gradient tracking for a tensor |
| `backward()` | Computes gradients through the graph |
| `torch.no_grad()` | Disables gradient recording within a context |
| `torch.inference_mode()` | Provides a more restrictive inference execution mode |
| `detach()` | Separates a tensor from its previous autograd history |
| `retain_graph=True` | Retains a graph for another backward pass |
| `zero_grad()` | Resets accumulated optimizer gradients |

### 6.1 no_grad

`torch.no_grad()` is useful for calculations that do not require gradient tracking.

For example, evaluating a trained model does not ordinarily require constructing a backward graph.

This can reduce unnecessary memory consumption.

### 6.2 Inference Mode

`torch.inference_mode()` is designed for inference workloads.

It can provide additional performance benefits compared with ordinary gradient-disabled execution.

It imposes stronger restrictions on tensors created inside the context.

### 6.3 Detachment

Detachment is useful when a computation should use a tensor's numerical values without continuing its previous gradient history.

This is relevant to:

- Logging.
- Numerical inspection.
- Certain training workflows.
- Data transfer between differentiable and nondifferentiable processing stages.

Detachment should be applied deliberately because it changes gradient propagation.

## 7. Gradient Verification

Correct forward calculations do not guarantee correct gradients.

Gradient verification is therefore an important numerical validation mechanism.

### 7.1 Analytical Derivatives

For:

\[
f(x)=x^3
\]

the derivative is:

\[
f'(x)=3x^2
\]

At \(x=2\):

\[
f'(2)=12
\]

### 7.2 Numerical Derivatives

A central finite-difference approximation is:

\[
f'(x)\approx
\frac{f(x+\epsilon)-f(x-\epsilon)}
{2\epsilon}
\]

where \(\epsilon\) is a sufficiently small perturbation.

The analytical and numerical values should be close within a suitable tolerance.

### 7.3 PyTorch gradcheck

The Python implementation uses:

`torch.autograd.gradcheck()`

This function numerically checks derivatives against finite-difference estimates.

It is especially useful when implementing custom differentiable operations.

Double precision is generally preferred for gradient checking because finite-difference calculations can be sensitive to numerical errors.

Gradient checking does not prove that a model is correct in every situation. It validates the tested function and inputs within the selected numerical tolerance.

## 8. Neural Network Fundamentals

### 8.1 Model Architecture

The Python implementation defines a regression network using:

- An input layer.
- A hidden fully connected layer.
- ReLU activation.
- An output layer.

Its architecture is:

`Input → Linear → ReLU → Linear → Output`

For a two-feature input:

\[
X \in \mathbb{R}^{N \times 2}
\]

the network generates:

\[
\hat{Y} \in \mathbb{R}^{N \times 1}
\]

### 8.2 Activation Function

ReLU is defined as:

\[
ReLU(x)=\max(0,x)
\]

It introduces nonlinearity into the network.

Without nonlinear activation functions, multiple linear layers composed together would still represent a linear transformation.

### 8.3 Loss Function

The example uses mean squared error:

\[
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
\]

This measures the average squared prediction error.

It is appropriate for the synthetic regression problem used in the script.

### 8.4 Optimization

The training example uses Adam.

The training workflow is:

1. Generate input features.
2. Generate target values.
3. Initialize the neural network.
4. Select a loss function.
5. Configure the optimizer.
6. Divide observations into batches.
7. Reset gradients.
8. Execute the forward pass.
9. Calculate loss.
10. Execute backward propagation.
11. Apply gradient clipping.
12. Update model parameters.
13. Evaluate the trained model.

The implementation also checks for non-finite loss values and verifies that training reduces the initial epoch loss.

### 8.5 Gradient Clipping

Gradient clipping restricts gradient magnitude.

The Python example uses:

`torch.nn.utils.clip_grad_norm_()`

This is useful when unusually large gradients destabilize optimization.

Clipping is a control mechanism, not a substitute for diagnosing numerical instability or selecting an appropriate learning rate.

## 9. Advanced Tensor Operations

The Python implementation includes additional operations frequently used in machine learning workloads.

### 9.1 Softmax

Softmax transforms a vector of scores into normalized probabilities.

For element \(i\):

\[
P_i =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

The probabilities sum to one along the selected dimension.

Softmax is commonly used in classification and attention-related computations.

### 9.2 Gather

`torch.gather()` selects values using an index tensor.

It is useful in:

- Selecting class-specific predictions.
- Sequence processing.
- Indexed tensor calculations.
- Batch-dependent data extraction.

### 9.3 Einsum

`torch.einsum()` expresses tensor contractions using index notation.

The Python example uses:

`torch.einsum("bij,bjk->bik", left, right)`

This represents batch matrix multiplication.

Einsum can express complex multidimensional operations without manually implementing every indexing step.

## 10. Python Implementation

The Python file is the principal practical implementation because it uses actual PyTorch APIs.

### Implemented components

| Component | Implementation |
|---|---|
| Configuration | Frozen dataclass |
| Device selection | CPU/CUDA resolution |
| Validation | Shape, dtype and finite-value checks |
| Tensor creation | PyTorch factory functions |
| Operations | Arithmetic, broadcasting and matrix multiplication |
| Memory | Views, copies and contiguity |
| Autograd | Forward graph and backward execution |
| Verification | Analytical derivatives and gradcheck |
| Neural networks | `nn.Module` and `nn.Sequential` |
| Optimization | Adam and MSE |
| Training | Mini-batch execution |
| Inference | `inference_mode()` |
| Performance | Timing and CUDA synchronization |

The program uses a deterministic random seed for reproducibility.

The training dataset is generated locally, avoiding an external dataset dependency.

CUDA-specific functionality is conditional, so the script can also execute on CPU-only systems.

## 11. JavaScript Implementation

The JavaScript file develops a separate CPU tensor engine using Node.js standard-library capabilities.

It does not call PyTorch or depend on external npm packages.

### 11.1 Typed-Array Storage

Tensor values are stored using `Float64Array`.

This provides:

- Contiguous numerical storage.
- Predictable numeric representation.
- Efficient indexed access.
- Reduced object overhead compared with arrays of boxed numerical objects.

Shape and stride metadata are maintained separately from the underlying numerical storage.

### 11.2 Broadcasting Engine

The implementation calculates output dimensions using `broadcastShape()`.

It then converts output positions into input positions using shape and stride information.

This demonstrates the relationship between logical tensor coordinates and flat numerical storage.

### 11.3 Automatic Differentiation

Each differentiable operation records:

- Parent nodes.
- Operation type.
- Forward values.
- Backward propagation logic.

The backward process uses a topological ordering of the computational graph.

A `Set` prevents repeated graph traversal.

The reverse traversal propagates gradients from the scalar output toward its input nodes.

### 11.4 Event-Driven Execution

`ExperimentRunner` extends Node.js `EventEmitter`.

It produces events for:

- Experiment started.
- Experiment completed.
- Experiment failed.

Execution uses asynchronous scheduling through `setImmediate()`.

The experiment service records execution duration and maintains an in-memory result registry.

The implementation restricts concurrent execution and rejects duplicate experiment names.

This model illustrates event-driven experiment orchestration, not GPU execution.

## 12. C++17 Implementation

The C++17 implementation emphasizes explicit memory representation, domain modeling, and computational graph management.

### 12.1 Node Representation

Each graph node contains:

- Numerical values.
- Gradient values.
- Tensor shape.
- Parent references.
- Operation metadata.
- Device metadata.
- Gradient-tracking state.
- Backward callback.

`std::shared_ptr` manages shared computational graph nodes.

This allows multiple operations to reference common parents without manually managing their lifetime.

### 12.2 Shape and Stride Management

The implementation calculates row-major strides.

For shape:

`[2, 3]`

the corresponding strides are:

`[3, 1]`

A multidimensional index can therefore be mapped to flat storage.

This is the basis for the indexing and broadcasting operations.

### 12.3 Reverse-Mode Differentiation

The C++ engine uses:

- Depth-first graph traversal.
- `std::unordered_set` for visited-node tracking.
- A topological node sequence.
- Reverse iteration for gradient propagation.

For an element-wise multiplication:

\[
z=xy
\]

the local derivatives are:

\[
\frac{\partial z}{\partial x}=y
\]

\[
\frac{\partial z}{\partial y}=x
\]

The implementation applies these derivatives to the incoming gradient.

Broadcasting requires gradients to be accumulated into the corresponding original input positions.

### 12.4 Numerical Verification

The C++ implementation compares analytical derivatives against a central finite-difference approximation.

It uses a numerical tolerance to determine whether the values agree.

### 12.5 Complexity Considerations

For an element-wise operation over \(N\) elements:

- Forward execution: \(O(N)\).
- Backward execution: \(O(N)\).
- Additional numerical storage: \(O(N)\).

For conventional matrix multiplication:

\[
O(mnp)
\]

The implementation uses straightforward loops for clarity. Production tensor engines require more sophisticated memory layouts, vectorization, cache optimization, and accelerator support.

## 13. Java 17 Implementation

The Java implementation approaches tensor computation through explicit domain types and an experiment-management service.

### 13.1 Domain Modeling

The implementation defines:

- `Device`
- `Operation`
- `ExperimentStatus`
- `Node`
- `Tensor`
- `ExperimentPolicy`
- `ExperimentResult`
- `ExperimentService`

The `ExperimentResult` record provides a compact immutable result structure.

The policy interface separates experiment validation from execution.

### 13.2 Tensor Operations

The implementation supports:

- Tensor creation.
- Zero and one initialization.
- Broadcasting.
- Addition.
- Element-wise multiplication.
- Squaring.
- Reduction.
- Matrix multiplication.

The matrix multiplication implementation is intentionally a forward-only operation. Reverse-mode differentiation is implemented for the element-wise operations and reduction.

This distinction prevents the example from incorrectly implying that every implemented forward operation has a corresponding backward implementation.

### 13.3 Experiment Governance

The experiment service validates:

- Experiment names.
- Input tensor availability.
- Input size limits.

It returns structured execution results and distinguishes completed and failed execution states.

This design illustrates how numerical computation can be integrated into a larger application with explicit policies and execution records.

### 13.4 Enterprise Engineering Considerations

A production Java tensor service would also need:

- Resource quotas.
- Device allocation policies.
- Persistent experiment metadata.
- Execution cancellation.
- Distributed scheduling.
- Authorization.
- Monitoring.
- Failure recovery.

These are application-level concerns rather than intrinsic tensor arithmetic operations.

## 14. PostgreSQL Implementation

The PostgreSQL script models the management and audit layer surrounding tensor computation.

A relational database is suitable for storing tensor metadata, experiment ownership, device registrations, execution history, and validation records.

It is not intended to replace numerical tensor storage or PyTorch's computation engine.

### 14.1 Relational Domain Model

The database contains the following entities:

| Entity | Responsibility |
|---|---|
| `researchers` | Experiment ownership and user roles |
| `compute_devices` | Registered CPU and CUDA devices |
| `experiments` | Execution lifecycle and configuration |
| `tensors` | Shape, dtype, storage and gradient metadata |
| `tensor_operations` | Computation graph operation records |
| `tensor_gradients` | Gradient validation and measurement records |
| `execution_metrics` | Runtime and numerical measurements |

The primary and foreign keys preserve relationships between experiments, tensors, operations, and researchers.

### 14.2 Tensor Metadata Integrity

The `tensors` table stores:

- Tensor shape.
- Data type.
- Device.
- Element count.
- Estimated storage.
- Gradient requirement.
- Gradient state.

The shape constraint rejects null or nonpositive dimensions.

The metadata trigger validates:

1. Tensor dimensions.
2. Calculated element count.
3. Device availability.
4. Data-type storage requirements.
5. Device memory capacity.

The calculated element count is:

\[
N=\prod_{i=1}^{d} s_i
\]

where \(s_i\) represents the size of dimension \(i\).

Estimated tensor storage is:

\[
M=N\times B
\]

where \(B\) represents bytes per element.

The metadata validation uses a configured device-capacity check. It does not claim to represent real-time GPU allocation or available runtime memory.

### 14.3 Computation Graph Records

The `tensor_operations` table stores operation metadata.

Each record identifies:

- Experiment.
- Operation type.
- Input tensor.
- Output tensor.
- Execution sequence.
- Execution duration.
- Additional JSON metadata.

The relationship trigger ensures that operation inputs and outputs belong to the same experiment.

The sequence constraint prevents duplicate operation sequence numbers within an experiment.

The table represents recorded operation relationships rather than a complete executable autograd graph.

### 14.4 Gradient Records

The `tensor_gradients` table stores:

- Tensor reference.
- Backward operation reference.
- Gradient norm.
- Finite-gradient status.
- Computation timestamp.

A gradient norm is useful for diagnosing optimization behavior.

A non-finite gradient can indicate numerical instability, invalid input data, or problematic model operations.

The database stores validation evidence, not the full gradient tensor values.

### 14.5 Experiment Lifecycle

The experiment lifecycle is represented by:

`created → running → completed`

Alternative terminal states include:

- `failed`
- `cancelled`

The state-transition trigger prevents invalid transitions such as moving directly from a completed experiment to a running experiment.

The `start_experiment()` procedure locks the target record before changing its state.

This prevents concurrent transactions from independently starting the same experiment through this procedure.

### 14.6 Views and Reporting

The database includes:

- `device_memory_report`
- `gradient_readiness`

These views provide structured operational information.

CTEs aggregate tensor counts, differentiable tensors, gradient records, operation counts, storage estimates, and execution durations.

Additional queries retrieve experiment ownership, operation sequences, device information, and execution metrics.

### 14.7 Transactions and Failure Handling

The SQL script uses:

- A transaction for schema creation and sample data.
- A savepoint for the experiment workflow.
- Row locking during experiment startup.
- Constraints and triggers for integrity enforcement.
- Nested exception handling for expected invalid operations.

The invalid-state and invalid-shape demonstrations verify that database rules reject incorrect operations.

## 15. Important Distinctions

Several concepts are closely related but should not be confused.

| Concept | Meaning |
|---|---|
| Tensor | Multidimensional numerical data |
| Shape | Dimensions of a tensor |
| Stride | Memory displacement between adjacent elements along each dimension |
| Device | Location where tensor computation executes |
| Dtype | Numerical representation of tensor elements |
| Broadcasting | Expansion of compatible logical dimensions |
| View | Alternative interpretation of compatible storage |
| Clone | Independent tensor storage |
| Autograd | Automatic derivative computation |
| Computational graph | Recorded dependency relationships |
| Gradient | Derivative of an output with respect to an input |
| Backpropagation | Reverse propagation of gradients |
| Optimizer | Algorithm that updates model parameters |
| Inference | Model execution without requiring training gradients |

Three distinctions are especially important.

**Tensor operations versus autograd**

A forward operation calculates numerical results. Autograd additionally records the relationships needed for differentiation.

Not every operation must participate in gradient tracking.

**Gradient tracking versus optimization**

Autograd calculates gradients. An optimizer uses those gradients to update parameters.

These are separate responsibilities.

**Device selection versus device execution**

A tensor's device metadata identifies its execution location. Device placement alone does not guarantee faster execution. Performance depends on hardware, operation size, data movement, memory availability, and synchronization.

## 16. Edge Cases and Failure Conditions

The implementations include validation and demonstrate several failure scenarios.

| Failure condition | Expected handling |
|---|---|
| Empty tensor shape | Reject invalid shape |
| Nonpositive dimensions | Reject tensor creation |
| Data and shape mismatch | Raise shape error |
| NaN or infinity | Reject invalid numerical values |
| Incompatible broadcasting | Raise shape error |
| Incompatible matrix dimensions | Reject matrix multiplication |
| Incorrect gradient target | Reject backward execution |
| Missing CUDA support | Reject explicit CUDA selection |
| Device mismatch | Require compatible device placement |
| Non-finite training loss | Stop training |
| Invalid experiment transition | Reject database update |
| Tensor storage exceeding capacity | Reject metadata insertion |
| Duplicate experiment name | Reject execution registration |

These checks demonstrate the importance of validating numerical inputs, structural metadata, and execution state before computation.

## 17. Performance and Security Considerations

### 17.1 Memory Efficiency

Large tensors can consume substantial memory.

Practical controls include:

- Selecting appropriate data types.
- Avoiding unnecessary tensor copies.
- Using views when compatible.
- Releasing unused references.
- Monitoring GPU allocations.
- Processing large datasets in batches.

### 17.2 Numerical Stability

Numerical instability can occur through:

- Extremely large values.
- Very small denominators.
- Overflow.
- Underflow.
- Invalid logarithms.
- Non-finite gradients.
- Poorly conditioned computations.

Finite-value validation and gradient monitoring help identify these problems.

### 17.3 Mixed Precision

Mixed-precision execution uses reduced-precision formats for supported operations.

It can improve memory efficiency and throughput on compatible accelerators.

The Python implementation demonstrates CUDA autocasting.

Production training may also require gradient scaling, depending on the selected precision and hardware.

### 17.4 Reproducibility

The Python implementation sets random seeds for reproducible initialization and synthetic data generation.

Exact reproducibility is not guaranteed across different hardware, software versions, execution backends, or nondeterministic operations.

### 17.5 Data and Model Security

Tensor pipelines may process confidential or proprietary information.

Appropriate controls include:

- Restricting access to model inputs and checkpoints.
- Validating uploaded tensor metadata.
- Limiting allocation sizes.
- Preventing unauthorized device access.
- Protecting experiment records.
- Avoiding sensitive information in diagnostic logs.
- Applying least-privilege database roles.
- Monitoring unusual resource consumption.

The PostgreSQL example models roles and metadata integrity. Production deployments must also implement database grants, application authentication, authorization, and operational auditing.

## 18. Implementation Limitations

The six implementations deliberately have different capabilities.

| Feature | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Actual PyTorch API | Yes | No | No | No | No |
| Tensor storage | PyTorch | Float64Array | STL vectors | Java arrays | Metadata |
| Broadcasting | PyTorch | Yes | Yes | Yes | Metadata validation |
| Matrix multiplication | PyTorch | Yes | Yes | Yes | Operation records |
| Reverse-mode autograd | PyTorch | Yes | Yes | Partial | Gradient records |
| CUDA execution | Conditional | No | No | No | Device metadata |
| Neural network training | Yes | No | No | No | No |
| Experiment management | Script | Event-driven | Demonstrations | Service model | Relational workflow |

The custom JavaScript, C++, and Java engines prioritize conceptual understanding and transparent implementation.

They do not implement PyTorch's complete operation library, optimized kernels, GPU integration, distributed execution, or production-grade autograd semantics.

The JavaScript engine uses double-precision storage. The C++ and Java engines also use double-precision numerical values.

The Java matrix multiplication example is forward-only. Its autograd implementation is limited to supported element-wise operations and reduction.

The PostgreSQL model records computation metadata and operational evidence. It does not execute tensor arithmetic.

## 19. Execution and Verification

| File | Execution requirement |
|---|---|
| Python | Python 3.10+, PyTorch 2.x |
| JavaScript | Node.js 18+ |
| C++17 | C++17-compatible compiler |
| Java 17 | JDK 17 |
| PostgreSQL | PostgreSQL 15+ |

### Verification coverage

The implementations include checks for:

- Tensor shape correctness.
- Broadcasting compatibility.
- Matrix multiplication dimensions.
- Gradient calculations.
- Numerical derivative agreement.
- Finite numerical values.
- Device availability.
- Training loss reduction.
- Experiment execution state.
- Database metadata integrity.

The Python implementation provides the actual PyTorch learning and execution environment, while the other files expose the underlying tensor, differentiation, and experiment-management mechanisms through independent implementations.
