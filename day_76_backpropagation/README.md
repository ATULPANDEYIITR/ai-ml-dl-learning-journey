
# Backpropagation: Chain Rule, Gradients and Computational Graphs

## 1. Technical Scope

This project implements backpropagation through computational graphs using five complementary approaches:

| Implementation | Primary responsibility |
|---|---|
| Python | Scalar automatic differentiation and neural network training |
| JavaScript | Event-driven computational graph and asynchronous training |
| C++17 | Efficient batch-gradient neural network implementation |
| Java 17 | Enterprise-oriented training lifecycle and gradient governance |
| PostgreSQL | Relational representation of computational graphs, experiments and training records |

The central objective is to understand how derivatives are calculated, propagated and accumulated through interconnected mathematical operations.

The implementations cover both automatic differentiation and manually derived neural network gradients.

## 2. Mathematical Foundations

### 2.1 Derivatives

A derivative measures how a function changes with respect to an input variable.

For a function:

\[
y=f(x)
\]

the derivative is:

\[
\frac{dy}{dx}
=
\lim_{h\to 0}
\frac{f(x+h)-f(x)}{h}
\]

For example:

\[
f(x)=x^2+3x+2
\]

Its derivative is:

\[
\frac{df}{dx}=2x+3
\]

At \(x=2\):

\[
f(2)=12
\]

\[
f'(2)=7
\]

The Python, JavaScript, C++17 and Java implementations verify this result using automatic differentiation.

### 2.2 Chain Rule

The chain rule calculates the derivative of a composite function.

Suppose:

\[
y=f(u)
\]

and:

\[
u=g(x)
\]

Then:

\[
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
\]

Consider:

\[
y=(x^2+3x+2)^2
\]

Define:

\[
u=x^2+3x+2
\]

Then:

\[
\frac{dy}{du}=2u
\]

and:

\[
\frac{du}{dx}=2x+3
\]

Therefore:

\[
\boxed{
\frac{dy}{dx}=2(x^2+3x+2)(2x+3)
}
\]

The chain rule allows derivatives to be calculated locally and propagated through a larger computation.

### 2.3 Computational Graphs

A computational graph represents a mathematical expression as a directed graph.

Each node represents an input, parameter or mathematical operation.

Edges represent dependencies between operations.

For:

\[
f(x)=x^2+3x+2
\]

the graph can be represented as:

```text
          x
         / \
        /   \
        *     * <--- 3
        |     |
        |     |
        +-----+
          |
          +
          |
          2
          |
         output
```

The graph stores the relationship between operations rather than treating the expression as one indivisible calculation.

Important properties include:

- Nodes represent computational operations or values.
- Edges represent data dependencies.
- Forward execution calculates intermediate values.
- Reverse execution propagates derivatives.
- Shared nodes accumulate gradients from multiple downstream paths.

A valid computational graph must be a directed acyclic graph (DAG).

Cycles would prevent a valid topological execution order.

## 3. Forward Propagation

Forward propagation calculates the output of a computational graph.

For a sequence of operations:

\[
x \rightarrow u \rightarrow v \rightarrow y
\]

each operation receives its inputs, evaluates its mathematical function and stores its output.

For example:

\[
u=x^2
\]

\[
v=3x
\]

\[
y=u+v+2
\]

At \(x=2\):

| Operation | Output |
|---|---|
| Input \(x\) | 2 |
| Square \(x^2\) | 4 |
| Multiply \(3x\) | 6 |
| Add intermediate values | 10 |
| Add constant 2 | 12 |

The intermediate values are retained because the backward pass requires them to calculate local derivatives.

## 4. Reverse-Mode Automatic Differentiation

Reverse-mode automatic differentiation calculates derivatives by traversing the computational graph in reverse topological order.

For a computational graph with many inputs and one scalar output, this approach is particularly efficient.

Each node maintains:

- Its forward value.
- Its operation.
- Its parent nodes.
- Its accumulated gradient.
- Its local backward rule.

The output node initially receives:

\[
\frac{\partial y}{\partial y}=1
\]

Each operation then propagates its gradient to its parents.

### 4.1 Addition

For:

\[
z=a+b
\]

the derivatives are:

\[
\frac{\partial z}{\partial a}=1
\]

\[
\frac{\partial z}{\partial b}=1
\]

Therefore:

\[
a.grad \mathrel{+}=z.grad
\]

\[
b.grad \mathrel{+}=z.grad
\]

### 4.2 Multiplication

For:

\[
z=ab
\]

the derivatives are:

\[
\frac{\partial z}{\partial a}=b
\]

\[
\frac{\partial z}{\partial b}=a
\]

Therefore:

\[
a.grad \mathrel{+}=b\cdot z.grad
\]

\[
b.grad \mathrel{+}=a\cdot z.grad
\]

The original forward values are required during backward propagation.

### 4.3 Power

For:

\[
z=a^n
\]

the derivative is:

\[
\frac{\partial z}{\partial a} = na^{n-1}
\]

The implementation validates invalid fractional powers and undefined derivatives at zero.

### 4.4 Hyperbolic Tangent

For:

\[
z=\tanh(a)
\]

the derivative is:

\[
\frac{\partial z}{\partial a} = 1-\tanh^2(a)
\]

Since the forward output is already stored, the derivative can be calculated directly from that output.

### 4.5 ReLU

The Rectified Linear Unit is:

\[
ReLU(x)=\max(0,x)
\]

Its derivative is:

\[
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x<0
\end{cases}
\]

At zero, the implementation adopts a derivative of zero.

This is a valid practical subgradient convention.

## 5. Gradient Accumulation

Gradient accumulation is essential when a computational node contributes to multiple downstream operations.

Consider:

\[
y=x^2+x^2+x
\]

Both square operations depend on the same input.

Therefore:

\[
\frac{dy}{dx}=2x+2x+1
\]

At \(x=3\):

\[
\frac{dy}{dx}=13
\]

Each backward operation contributes to the same input gradient.

The implementations use additive accumulation rather than overwriting an existing gradient.

This distinction is fundamental to reverse-mode automatic differentiation.

Before a new backward pass, previously accumulated gradients must be reset when computing a fresh independent gradient evaluation.

## 6. Topological Ordering

A computational graph must process each node after its dependencies have been evaluated.

Topological ordering provides this execution sequence.

The implementations use depth-first traversal and visited-node tracking.

The procedure is:

1. Start at the output node.
2. Visit each parent recursively.
3. Add a node after visiting its parents.
4. Reverse the resulting order for backward propagation.
5. Initialize the output gradient.
6. Execute each node's backward rule.

For a graph containing \(V\) nodes and \(E\) edges:

\[
T(V,E)=O(V+E)
\]

The graph traversal requires:

\[
S(V)=O(V)
\]

additional space for the traversal state and ordering.

The Python and JavaScript implementations use object identity to avoid processing shared nodes repeatedly.

The C++17 implementation uses indexed node tracking and pointers to graph-owned nodes.

## 7. Numerical Gradient Verification

Automatic differentiation should be tested against an independent numerical approximation.

The central finite-difference approximation is:

\[
f'(x) \approx \frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
\]

Here:

- \(\epsilon\) is a small positive perturbation.
- The numerator estimates the output change.
- The denominator measures the input change.

The implementations use numerical gradient checks for selected mathematical functions.

A comparison can be made using:

\[
E=|g_{\text{automatic}}-g_{\text{numerical}}|
\]

A small error indicates that the analytical gradient is consistent with the numerical approximation.

The perturbation size requires care.

An excessively large value introduces approximation error, while an excessively small value can amplify floating-point rounding errors.

Numerical checks should generally use double precision.

Gradient checking is most useful for small deterministic examples rather than large training workloads.

## 8. Neural Network Backpropagation

A neural network is a composition of mathematical operations.

Consider a network with:

- Two input neurons.
- Four hidden neurons.
- One output neuron.

Each neuron computes:

\[
z=\sum_{i=1}^{n}w_ix_i+b
\]

The activation is:

\[
a=\tanh(z)
\]

Here:

- \(x_i\) represents an input.
- \(w_i\) represents a trainable weight.
- \(b\) represents a bias.
- \(z\) represents the weighted sum.
- \(a\) represents the activated output.

The hidden layer produces:

\[
h_j= \tanh\left( \sum_i w_{ji}x_i+b_j \right)
\]

The output layer calculates:

\[
\hat y= \tanh\left( \sum_j v_jh_j+c \right)
\]

### 8.1 Loss Function

The implementations use mean squared error:

\[
L= \frac{1}{N} \sum_{i=1}^{N} (\hat y_i-y_i)^2
\]

Where:

- \(N\) is the number of training samples.
- \(\hat y_i\) is the predicted value.
- \(y_i\) is the target value.

The objective is to reduce the difference between predictions and targets.

### 8.2 Output Gradient

For an individual sample:

\[
L=(\hat y-y)^2
\]

The derivative with respect to the prediction is:

\[
\frac{\partial L}{\partial \hat y} = 2(\hat y-y)
\]

If the output activation is tanh:

\[
\delta_o= 2(\hat y-y)(1-\hat y^2)
\]

For a batch, the contribution is divided by the number of samples.

### 8.3 Hidden Layer Gradient

The hidden neuron receives the gradient through the output weights.

Its error contribution is:

\[
\delta_j= \delta_o v_j(1-h_j^2)
\]

The weight derivatives are:

\[
\frac{\partial L}{\partial w_{ji}} = \delta_jx_i
\]

The bias derivative is:

\[
\frac{\partial L}{\partial b_j} = \delta_j
\]

### 8.4 Gradient Descent

The parameters are updated using:

\[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
\]

Where \(\eta\) represents the learning rate.

The same procedure applies to biases.

The implementations use batch gradient descent, calculating contributions from the complete dataset before applying parameter updates.

## 9. XOR Classification

The XOR function produces:

| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR is not linearly separable.

A single linear neuron cannot correctly represent this decision boundary.

A hidden layer introduces nonlinear transformations that allow the network to approximate the XOR function.

The four samples are used for training.

The network is evaluated on the same four samples after training.

The output values are continuous because tanh produces values in the interval:

\[
(-1,1)
\]

Predictions should be interpreted as numerical approximations rather than exact binary outputs.

## 10. Implementation Architecture

### 10.1 Python

The Python implementation provides a general scalar automatic differentiation engine.

Its `Value` dataclass represents individual computational nodes.

The arithmetic operators construct new graph nodes and attach local derivative rules.

Important implementation decisions:

- `parents` records graph dependencies.
- `operation` identifies the mathematical operation.
- `backward_fn` stores the local gradient calculation.
- `grad` stores the accumulated derivative.
- `backward()` builds a topological ordering before propagation.

The `MLP` class builds a multilayer neural network from individual neurons.

The XOR training example demonstrates how the differentiation engine supports a complete optimization workflow.

The script also performs numerical gradient checks and validates non-finite values.

### 10.2 JavaScript

The JavaScript implementation emphasizes event-driven training.

`TensorNode` represents a graph node.

The graph uses `Set` for visited-node tracking.

The `NeuralNetwork` class extends `EventEmitter` to expose training events.

Two events are provided:

| Event | Purpose |
|---|---|
| `epoch` | Reports an epoch and its loss |
| `trainingComplete` | Reports successful training completion |

The training method is asynchronous and periodically yields execution through `setImmediate`.

This allows other Node.js event-loop work to proceed between training intervals.

The checkpoint stores model parameters and training metadata.

Executable functions and graph objects are deliberately excluded from serialized model data.

### 10.3 C++17

The C++ implementation separates two responsibilities.

`ComputationalGraph` demonstrates reverse-mode differentiation using graph nodes and parent pointers.

`NeuralNetwork` implements an efficient manually differentiated batch-training process.

This distinction is deliberate.

A general automatic differentiation engine provides flexibility, while a purpose-built neural network can calculate gradients directly without allocating an individual graph node for every operation.

The implementation uses:

- STL vectors for model parameters.
- Pointer-based graph relationships.
- `unordered_map` for node indexing.
- `vector<bool>` for traversal tracking.
- `mt19937` for reproducible initialization.
- Exceptions for invalid configurations and training failures.

The network calculates candidate gradients before applying parameter updates.

The approach reduces unnecessary intermediate graph allocations during neural network training.

### 10.4 Java 17

The Java implementation models training as an enterprise service.

The main domain types include:

| Type | Responsibility |
|---|---|
| `TrainingConfig` | Immutable optimization configuration |
| `Sample` | Validated training record |
| `TrainingReport` | Immutable execution result |
| `GraphNode` | Computational graph node |
| `Neuron` | Trainable neuron |
| `Model` | Neural network and optimization operations |
| `TrainingService` | Training lifecycle management |
| `TrainingListener` | Training progress notifications |

Training states are explicitly represented:

```text
CREATED
   |
VALIDATING
   |
TRAINING
   |
COMPLETED
```

A failure transitions the service to `FAILED`.

The service validates input dimensions and configuration before training.

Gradient clipping is applied before parameter updates:

\[
g_{\text{clipped}} = \max(-c,\min(c,g))
\]

Here, \(c\) is the configured clipping threshold.

Candidate parameters are validated before the model state is updated.

This provides a clearer separation between gradient calculation, validation and parameter commitment.

## 11. PostgreSQL Data Architecture

The database represents the computational and experimental lifecycle.

Its relational structure separates graph topology, node values, gradients, tensor metadata and optimization records.

### 11.1 Main Entities

| Table | Responsibility |
|---|---|
| `experiments` | Training experiment configuration |
| `computational_graphs` | Graph identity and version |
| `graph_nodes` | Mathematical operations and values |
| `graph_edges` | Parent-child dependencies |
| `tensor_metadata` | Data types and tensor shapes |
| `training_runs` | Training lifecycle |
| `optimization_steps` | Epoch-level optimization records |
| `gradient_records` | Analytical and numerical gradient comparisons |
| `training_datasets` | Dataset metadata |
| `dataset_samples` | Feature and target records |

### 11.2 Relational Relationships

```text
EXPERIMENTS
    |
    +---- COMPUTATIONAL_GRAPHS
    |           |
    |           +---- GRAPH_NODES
    |           |         |
    |           |         +---- TENSOR_METADATA
    |           |
    |           +---- GRAPH_EDGES
    |
    +---- TRAINING_RUNS
    |           |
    |           +---- OPTIMIZATION_STEPS
    |                       |
    |                       +---- GRADIENT_RECORDS
    |
    +---- TRAINING_DATASETS
                |
                +---- DATASET_SAMPLES
```

### 11.3 Graph Integrity

`graph_edges` uses foreign keys to ensure that parent and child nodes belong to the referenced graph.

A trigger detects cycles before an edge is inserted.

The recursive query follows existing dependencies from the proposed child.

If the proposed parent is already reachable, inserting the edge would introduce a cycle and is rejected.

This provides database-level protection for graph topology.

Operation arity is also validated.

Binary operations require two input dependencies, unary operations require one, and input or parameter nodes require none.

### 11.4 Training State Management

The `training_runs` table records the lifecycle of an experiment.

The trigger prevents invalid state transitions and modifications to terminal runs.

A successful run cannot be marked `COMPLETED` before all configured epochs have been processed.

Failure records must contain a failure reason.

The database therefore maintains a consistent record of training execution.

### 11.5 Gradient Auditing

The `gradient_records` table records:

- Analytical gradient.
- Numerical gradient.
- Absolute error.
- Verification result.
- Associated optimization step.
- Associated computational node.

This supports gradient debugging and historical comparison.

### 11.6 Reporting

The `experiment_training_report` view aggregates training execution statistics.

Additional SQL queries retrieve graph dependencies, validate dataset dimensions, aggregate optimization steps and determine training eligibility.

The database is designed as an experiment metadata and audit store.

It does not implement a high-performance tensor execution engine.

## 12. Important Distinctions

| Concept | Meaning |
|---|---|
| Differentiation | Mathematical calculation of derivatives |
| Chain rule | Rule for differentiating composite functions |
| Computational graph | Representation of mathematical dependencies |
| Forward propagation | Evaluation of graph operations |
| Backpropagation | Reverse propagation of derivatives |
| Automatic differentiation | Systematic calculation of derivatives using operation-level rules |
| Gradient | Derivative of an objective with respect to a variable |
| Gradient accumulation | Addition of derivative contributions from multiple paths |
| Gradient descent | Optimization using parameter updates |
| Gradient checking | Comparison of calculated derivatives with numerical approximations |

Backpropagation is not itself an optimization algorithm.

It calculates gradients.

Gradient descent uses those gradients to modify model parameters.

Similarly, automatic differentiation is not equivalent to numerical differentiation.

Automatic differentiation applies mathematical derivative rules to computational operations, whereas numerical differentiation approximates derivatives through function evaluations.

## 13. Edge Cases and Failure Handling

The implementations address several important failure conditions.

| Condition | Handling |
|---|---|
| Division by zero | Rejected |
| Invalid fractional power | Rejected |
| Logarithm of non-positive value | Rejected |
| Non-finite graph values | Rejected |
| Incorrect input dimensions | Rejected |
| Empty training dataset | Rejected |
| Invalid learning rate | Rejected |
| Invalid training epochs | Rejected |
| Shared graph dependencies | Gradients accumulated |
| Invalid gradient updates | Rejected before parameter commitment |
| Invalid training state transition | Rejected |
| Computational graph cycle | Rejected by database trigger |

Numerical instability can still occur when gradients become excessively large or when an activation saturates.

Gradient clipping helps limit extreme updates but does not eliminate every source of numerical instability.

## 14. Performance Considerations

### Computational Graph

For a graph containing \(V\) nodes and \(E\) edges:

| Operation | Complexity |
|---|---|
| Graph construction | \(O(V+E)\) |
| Topological traversal | \(O(V+E)\) |
| Backward propagation | \(O(V+E)\) |
| Gradient initialization | \(O(V)\) |

The actual cost of individual operations depends on their underlying tensor dimensions.

### Neural Network

For a fully connected network with:

- \(I\) input neurons.
- \(H\) hidden neurons.
- \(O\) output neurons.

The approximate forward computational cost per sample is:

\[
O(IH+HO)
\]

Backward propagation has a comparable asymptotic computational cost.

Batch processing reduces the overhead of repeatedly updating model parameters for individual samples.

### Memory

Automatic differentiation retains intermediate values and graph dependencies until backward propagation is complete.

Memory consumption therefore increases with graph size.

For larger networks, activation checkpointing can reduce memory consumption by recomputing selected intermediate values during backward propagation.

The scalar implementations intentionally prioritize transparency over large-scale computational efficiency.

## 15. Security and Reliability

Production training systems should apply controls beyond mathematical correctness.

Important considerations include:

- Validate dataset dimensions and numerical values.
- Restrict dataset and checkpoint file permissions.
- Avoid loading untrusted executable serialization formats.
- Verify dataset integrity before training.
- Keep training configuration separate from model parameters.
- Record reproducible random seeds.
- Monitor loss and gradient magnitudes.
- Reject non-finite model updates.
- Preserve training failure information.
- Validate graph dependencies before execution.
- Apply database privileges according to operational responsibilities.
- Use transaction boundaries for consistent experiment records.

A successful gradient calculation does not establish that a model is accurate, unbiased or suitable for deployment.

Model evaluation requires independent validation data and appropriate performance metrics.

## 16. Limitations

The implementations deliberately use scalar mathematical operations and small neural networks.

They are educational implementations rather than production tensor frameworks.

Specific limitations include:

- No GPU execution.
- No distributed gradient synchronization.
- No convolutional or recurrent layers.
- No automatic mixed precision.
- No optimizer implementations beyond basic gradient descent.
- No sparse tensor execution.
- No production-scale dataset pipeline.
- No automatic differentiation for arbitrary programming-language operations.
- No database-backed tensor storage or numerical execution.

The Python, JavaScript, C++17 and Java implementations use different engineering approaches while demonstrating the same mathematical foundations.

The PostgreSQL implementation provides relational experiment management and graph-integrity enforcement rather than numerical tensor computation.

## 17. Execution and Expected Behavior

| File | Execution |
|---|---|
| Python | Run with Python 3 |
| JavaScript | Run with Node.js |
| C++17 | Compile with a C++17-compatible compiler |
| Java 17 | Compile and execute `BackpropagationEnterprise` |
| PostgreSQL | Execute the complete SQL script in PostgreSQL |

The programs demonstrate:

1. Derivative calculation using the chain rule.
2. Computational graph construction.
3. Reverse-mode gradient propagation.
4. Gradient accumulation.
5. Numerical gradient verification.
6. Multilayer neural network training.
7. XOR classification.
8. Training-state validation.
9. Gradient clipping and parameter updates.
10. Relational storage and reporting of training experiments.

The shared mathematical foundation is the chain rule. Its implementation through computational graphs makes it possible to calculate gradients efficiently across complex networks of interconnected operations.
