# Activation Functions in Neural Networks

## Technical Scope

Activation functions transform the outputs of neurons before those outputs are passed to subsequent neural network layers.

A typical neuron computes:

\[
z = Wx+b
\]

The activation function transforms this intermediate value:

\[
a=f(z)
\]

Here:

- \(W\) represents weights.
- \(x\) represents input features.
- \(b\) represents the bias.
- \(z\) represents the pre-activation value.
- \(f\) represents the activation function.
- \(a\) represents the activated output.

Without nonlinear activation functions, a network consisting entirely of linear transformations can be reduced to a single linear transformation. Nonlinear activations allow neural networks to represent more complex relationships.

This implementation covers six activation functions:

1. Sigmoid
2. Tanh
3. ReLU
4. Leaky ReLU
5. GELU
6. Softmax

The five source files approach the subject through numerical computation, event-driven execution, systems programming, enterprise-oriented inference, and relational data management.

## 1. Sigmoid

<escape>**Definition**</escape>

The sigmoid function, also called the logistic function, maps a real-valued input into an output strictly between zero and one.

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Its derivative is:

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]

**Characteristics**

- Output range: \((0,1)\).
- Smooth and differentiable.
- Suitable for binary classification output probabilities.
- Provides a probability-like interpretation for individual binary outputs.
- Can suffer from vanishing gradients when inputs have large absolute magnitudes.

| Input | Approximate output |
|---:|---:|
| -3 | 0.047426 |
| -1 | 0.268941 |
| 0 | 0.500000 |
| 1 | 0.731059 |
| 3 | 0.952574 |

**Implementation decision**

The Python, JavaScript, C++, and Java implementations use a numerically stable formulation.

For nonnegative inputs:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

For negative inputs:

\[
\sigma(x)=\frac{e^x}{1+e^x}
\]

This avoids unnecessary exponential overflow for large negative values.

## 2. Tanh

<escape>**Definition**</escape>

The hyperbolic tangent function is:

\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]

Its derivative is:

\[
\tanh'(x)=1-\tanh^2(x)
\]

**Characteristics**

- Output range: \((-1,1)\).
- Zero-centered output.
- Smooth and differentiable.
- Often useful in recurrent architectures and intermediate neural network representations.
- Can suffer from vanishing gradients for strongly saturated inputs.

| Input | Approximate output |
|---:|---:|
| -3 | -0.995055 |
| -1 | -0.761594 |
| 0 | 0 |
| 1 | 0.761594 |
| 3 | 0.995055 |

**Distinction from sigmoid**

Both functions are bounded and saturating.

Sigmoid produces positive outputs between zero and one. Tanh produces negative and positive outputs around zero.

Their mathematical relationship is:

\[
\tanh(x)=2\sigma(2x)-1
\]

Tanh is therefore zero-centered, while sigmoid is not.

## 3. ReLU

<escape>**Definition**</escape>

Rectified Linear Unit is defined as:

\[
ReLU(x)=\max(0,x)
\]

Its derivative is:

\[
ReLU'(x)=
\begin{cases}
0 & x\leq 0\\
1 & x>0
\end{cases}
\]

At zero, the implementation adopts a derivative convention of zero.

**Characteristics**

- Output range: \([0,\infty)\).
- Computationally inexpensive.
- Produces sparse activations when some inputs are negative.
- Helps reduce saturation-related gradient problems on the positive side.
- Can suffer from the dying ReLU problem.

The dying ReLU problem occurs when a neuron repeatedly receives negative pre-activation values and consequently produces zero gradients.

**Example**

Input:

\[
x=[-3,-1,0,1,3]
\]

Output:

\[
ReLU(x)=[0,0,0,1,3]
\]

ReLU is commonly used in hidden layers of feedforward networks and convolutional neural networks.

## 4. Leaky ReLU

<escape>**Definition**</escape>

Leaky ReLU introduces a small negative slope.

\[
f(x)=
\begin{cases}
x & x>0\\
\alpha x & x\leq0
\end{cases}
\]

Its derivative is:

\[
f'(x)=
\begin{cases}
1 & x>0\\
\alpha & x\leq0
\end{cases}
\]

Here, \(\alpha\) is a nonnegative configurable parameter.

The implementations use:

\[
\alpha=0.01
\]

unless another value is supplied.

**Example**

For an input of \(-3\):

\[
f(-3)=0.01(-3)=-0.03
\]

For an input of \(3\):

\[
f(3)=3
\]

**Distinction from ReLU**

| Property | ReLU | Leaky ReLU |
|---|---|---|
| Positive input | Linear | Linear |
| Negative input | Zero | Small negative value |
| Negative-side gradient | Zero | Alpha |
| Dying neuron risk | Present | Reduced |
| Additional parameter | No | Yes |

Leaky ReLU allows gradients to propagate through negative inputs, although the gradient magnitude depends on the selected negative slope.

## 5. GELU

<escape>**Definition**</escape>

Gaussian Error Linear Unit uses the cumulative distribution function of the standard normal distribution.

Its exact formulation is:

\[
GELU(x)=x\Phi(x)
\]

where \(\Phi(x)\) represents the standard normal cumulative distribution function.

A widely used approximation is:

\[
GELU(x)\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)
\right)
\right]
\]

**Characteristics**

- Smooth nonlinear transformation.
- Retains negative-valued outputs for some negative inputs.
- Does not impose a strict zero threshold.
- Uses a smooth gating-like behavior.
- More computationally expensive than ReLU.
- Common in transformer architectures.

**Derivative**

Let:

\[
u=\sqrt{\frac{2}{\pi}}(x+0.044715x^3)
\]

Then:

\[
GELU'(x)=
\frac{1}{2}(1+\tanh(u))
+
\frac{x}{2}(1-\tanh^2(u))
\sqrt{\frac{2}{\pi}}
(1+3(0.044715)x^2)
\]

The implementations use the tanh approximation for both forward and backward calculations.

## 6. Softmax

<escape>**Definition**</escape>

Softmax transforms a vector of real-valued logits into a probability distribution.

For a vector \(z\) containing \(n\) elements:

\[
softmax(z_i)=
\frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
\]

The resulting probabilities satisfy:

\[
0<p_i<1
\]

and:

\[
\sum_{i=1}^{n}p_i=1
\]

**Example**

Given:

\[
z=[2.0,1.0,0.1]
\]

The approximate probability distribution is:

| Class | Logit | Probability |
|---|---:|---:|
| Class 0 | 2.0 | 0.659001 |
| Class 1 | 1.0 | 0.242433 |
| Class 2 | 0.1 | 0.098566 |

The highest probability corresponds to the predicted class.

**Numerical stability**

Direct exponentiation can overflow for large logits.

The implementations use the maximum-subtraction technique:

\[
softmax(z_i)=
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
\]

where:

\[
m=\max(z)
\]

Subtracting the same constant from all logits does not change the resulting probability distribution.

**Softmax derivative**

Softmax has a vector-valued output, so its derivative is a Jacobian matrix.

\[
J_{ij}=p_i(\delta_{ij}-p_j)
\]

Here, \(\delta_{ij}\) is the Kronecker delta.

For backward propagation, constructing the complete Jacobian is unnecessary.

Given an upstream gradient vector \(g\):

\[
J^Tg=p\odot(g-p^Tg)
\]

This allows the Jacobian-vector product to be computed in \(O(n)\) time rather than constructing an \(O(n^2)\) matrix.

**Classification loss**

The implementations also demonstrate categorical cross-entropy:

\[
L=-\log(p_y)
\]

where \(p_y\) is the predicted probability assigned to the correct class.

For numerical stability, the implementations use a small positive probability floor before evaluating the logarithm.

## 7. Comparative Analysis

| Function | Output range | Zero-centered | Main application |
|---|---|---|---|
| Sigmoid | (0,1) | No | Binary classification output |
| Tanh | (-1,1) | Yes | Hidden representations and recurrent networks |
| ReLU | [0,∞) | No | Hidden layers |
| Leaky ReLU | (-∞,∞) | No | Hidden layers with negative-side gradients |
| GELU | Approximately (-0.17,∞) | No | Transformer and deep learning architectures |
| Softmax | (0,1), sum = 1 | No | Multiclass classification output |

These functions are not interchangeable.

The appropriate choice depends on the network architecture, output interpretation, gradient behavior, numerical requirements, and computational constraints.

## 8. Forward Propagation and Backward Propagation

Activation functions participate in two fundamental stages of neural network training.

### Forward propagation

A layer receives an input vector and calculates activated outputs.

\[
z=Wx+b
\]

\[
a=f(z)
\]

The output of one layer becomes the input to the next layer.

A simplified architecture is:

`Input → Linear Transformation → Activation → Linear Transformation → Output`

The implementations demonstrate activation layers independently of weight and bias calculations.

### Backward propagation

During training, the gradient of the loss is propagated backward through the network.

For element-wise activations:

\[
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\odot f'(z)
\]

Here, \(\odot\) denotes element-wise multiplication.

Softmax requires its Jacobian-vector product because its outputs are coupled through the normalization denominator.

The Python, JavaScript, C++, and Java implementations distinguish these two derivative mechanisms.

A production classification implementation can also combine softmax and cross-entropy into a numerically efficient gradient calculation:

\[
\frac{\partial L}{\partial z}=p-y
\]

where \(y\) is the target probability vector.

## 9. Python Implementation

The Python script provides a numerical activation library and an executable testing environment.

### Main components

| Component | Responsibility |
|---|---|
| `ActivationType` | Enumerates supported functions |
| `ActivationConfig` | Stores activation configuration |
| `Activation` | Dispatches scalar operations |
| `DenseActivationLayer` | Implements vector forward and backward operations |
| `softmax` | Computes normalized probabilities |
| `softmax_jacobian` | Constructs the full softmax Jacobian |
| `softmax_jvp` | Calculates efficient gradient propagation |
| `numerical_derivative` | Performs finite-difference checking |
| `ActivationTests` | Tests numerical behavior and validation |

### Numerical validation

The script rejects:

- Empty vectors.
- Non-numeric inputs.
- Boolean values used as numeric inputs.
- NaN values.
- Infinite values.
- Negative Leaky ReLU slopes.
- Mismatched gradient dimensions.
- Invalid softmax probability distributions.

### Gradient checking

The numerical derivative uses the central difference approximation:

\[
f'(x)\approx
\frac{f(x+\epsilon)-f(x-\epsilon)}
{2\epsilon}
\]

This provides an independent numerical comparison with the implemented analytical derivatives.

The test suite checks sigmoid, tanh, ReLU, and GELU away from their nondifferentiable or convention-dependent points.

Run the script with Python 3.10 or later:

`python activation_functions.py`

No external packages are required.

## 10. JavaScript Implementation

The JavaScript implementation adopts an event-driven inference architecture rather than reproducing the Python class structure.

### Activation registry

`ActivationRegistry` stores implementations in a JavaScript `Map`.

Each registered activation exposes two operations:

- `forward`
- `derivative`

The registry validates implementations and rejects duplicate registrations.

### Event-driven execution

`ActivationEngine` extends Node.js `EventEmitter`.

It publishes two events:

| Event | Meaning |
|---|---|
| `executed` | Activation calculation completed |
| `failed` | Activation calculation failed |

This design separates numerical execution from monitoring and reporting.

### Asynchronous execution

The engine supports:

- `executeAsync()` for Promise-based inference.
- `executeBatch()` for asynchronous batch execution.
- `statistics()` for execution and failure counts.

`Promise.all()` coordinates batch operations and propagates failures.

This is an orchestration model. It does not imply automatic CPU parallelism for JavaScript numerical operations.

### Security and reliability

The implementation:

- Validates activation identifiers.
- Rejects invalid numerical inputs.
- Freezes registry implementations.
- Returns immutable result objects.
- Rejects unknown activations.
- Separates failure events from successful execution events.

Run using:

`node activation_functions.js`

The implementation uses only Node.js built-in modules.

## 11. C++17 Implementation

The C++ program emphasizes explicit numerical operations, type safety, and efficient memory management.

### Domain components

| Component | Responsibility |
|---|---|
| `ActivationType` | Strongly typed activation identifiers |
| `ActivationFunction` | Numerical formulas and derivatives |
| `ActivationLayer` | Forward and backward vector operations |
| `NeuralNetwork` | Sequential activation pipeline |
| `crossEntropy` | Classification loss calculation |
| `validateVector` | Numerical input validation |

### Memory and performance

The implementation uses `std::vector<double>` for numerical data.

Important design decisions include:

- Reserving vector capacity where output sizes are known.
- Passing large input vectors by constant reference.
- Using STL algorithms for reductions and maximum calculations.
- Avoiding unnecessary intermediate Jacobian matrices during gradient propagation.
- Using strongly typed enums instead of unvalidated string identifiers.

### Complexity

Let \(n\) represent the number of input values.

| Operation | Time complexity |
|---|---:|
| Scalar activation | O(1) |
| Element-wise activation | O(n) |
| Element-wise derivative | O(n) |
| Softmax | O(n) |
| Full softmax Jacobian | O(n²) |
| Softmax JVP | O(n) |
| Sequential activation pipeline | O(kn) |

Here, \(k\) represents the number of activation layers.

Compile using:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic activation_functions.cpp -o activation_functions`

Run using:

`./activation_functions`

The program does not require third-party libraries.

## 12. Java 17 Implementation

The Java implementation models activation computation as an enterprise-oriented inference service.

### Domain model

The implementation uses:

- `ActivationType` for supported activation identifiers.
- `ActivationConfig` as an immutable record.
- `ActivationFunction` as an interface.
- `ActivationRegistry` for implementation management.
- `DenseActivationLayer` for layer-level execution.
- `Softmax` for vector-specific numerical operations.
- `InferenceService` for sequential and batch inference.

### Immutability and validation

The activation configuration is immutable.

The inference service stores its layer collection through `List.copyOf()`.

Input vectors are copied before processing, reducing accidental modification of caller-owned arrays.

Custom exceptions provide explicit failure information for:

- Missing configurations.
- Invalid dimensions.
- Unknown activation functions.
- Invalid probabilities.
- Out-of-range classification targets.

### Concurrent batch inference

`InferenceService` supports concurrent batch execution through:

- `CompletableFuture`
- `Executor`
- `CompletableFuture.allOf()`

An external executor controls the number of concurrent tasks.

The example uses a fixed-size thread pool and shuts it down after inference.

The service treats individual batch failures as inference failures rather than silently returning incomplete results.

Compile and run:

`javac ActivationFunctions.java`

`java ActivationFunctions`

No third-party dependencies are required.

## 13. PostgreSQL Implementation

The SQL script models activation functions as configurable components of neural network architectures.

It is designed to demonstrate the relationship between numerical activation operations and relational model-management records.

### Database architecture

`activation_lab` contains the following entities:

| Table | Purpose |
|---|---|
| `activation_definitions` | Activation function catalog |
| `neural_models` | Model metadata and lifecycle status |
| `model_layers` | Ordered activation configurations |
| `evaluation_runs` | Inference execution records |
| `inference_outputs` | Sample-level logits and predictions |

### Relationship model

`activation_definitions`  
↓  
`model_layers`  
↓  
`neural_models`  
↓  
`evaluation_runs`  
↓  
`inference_outputs`

The activation catalog provides reusable definitions.

A model can contain multiple layers, and each layer references one activation definition.

An evaluation run records the execution environment and lifecycle status.

Sample-level outputs record the original logits, activation outputs, predictions, actual labels, and loss values.

### Database integrity

The implementation uses:

- Primary keys.
- Foreign keys.
- Unique constraints.
- Check constraints.
- Identity columns.
- Referential integrity.
- Indexes for model and evaluation retrieval.

A trigger prevents Softmax from being configured as a hidden layer.

The classification-output validation also restricts the final activation to sigmoid or softmax.

A second trigger enforces valid evaluation state transitions.

Terminal evaluation states cannot be silently changed back to an earlier state.

### SQL numerical functions

The database implements:

- `sigmoid()`
- `tanh_activation()`
- `relu()`
- `leaky_relu()`
- `gelu()`
- `softmax()`
- `activation_derivative()`
- `softmax_jvp()`

The Softmax implementation subtracts the maximum logit before exponentiation.

The JVP implementation computes the gradient without explicitly constructing the full Jacobian.

### Reporting

The SQL views provide:

- Activation usage by model.
- Layer configuration details.
- Classification accuracy.
- Correct and incorrect predictions.
- Average classification loss.
- Model activation configuration reports.

The `complete_evaluation()` procedure handles the transition from running to completed evaluation status.

The sample data includes valid model configurations and a deliberately incorrect prediction to demonstrate classification reporting.

Execute the script using PostgreSQL:

`psql -U postgres -d your_database -f activation_functions.sql`

The script creates objects inside the `activation_lab` schema.

## 14. Implementation Comparison

| Dimension | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Primary purpose | Numerical library | Event-driven inference | Efficient computation | Enterprise inference | Relational model management |
| Activation registry | Enum dispatcher | Map | Strongly typed enum | Enum and registry | Relational catalog |
| Forward propagation | Yes | Yes | Yes | Yes | Numerical functions |
| Backward propagation | Yes | Yes | Yes | Yes | Derivative functions |
| Softmax | Yes | Yes | Yes | Yes | Yes |
| Softmax JVP | Yes | Yes | Yes | Yes | Yes |
| Batch processing | Vector operations | Promises | Vector operations | CompletableFuture | Relational records |
| Validation | Exceptions | Custom errors | Standard exceptions | Custom exceptions | Constraints and triggers |
| Testing | unittest | Assertions | Runtime tests | Runtime tests | Executable queries |

The implementations deliberately use different engineering approaches.

They are independent programs and database scripts, not interchangeable components of a single executable application.

## 15. Numerical and Engineering Considerations

### Vanishing gradients

Sigmoid and tanh saturate at large positive and negative input values.

Their derivatives approach zero in saturated regions.

This can reduce the magnitude of gradients propagated to earlier layers.

### Dying ReLU

ReLU produces zero output and zero gradient for negative inputs.

A neuron that remains in this region may stop contributing to gradient-based learning.

Leaky ReLU addresses this issue through its configurable negative slope.

### GELU approximation

The implementations use the tanh approximation rather than evaluating the Gaussian cumulative distribution function directly.

This provides a practical balance between numerical behavior and computational cost.

### Softmax stability

Subtracting the maximum logit prevents overflow during exponentiation.

The normalization denominator must remain positive and finite.

### Softmax and multiclass classification

Softmax is a vector-level operation.

Applying sigmoid independently to each class does not produce the same probability normalization.

Softmax is appropriate when classes are mutually exclusive.

Independent sigmoid outputs are appropriate for multilabel classification, where several classes may be active simultaneously.

### Floating-point precision

The implementations use double-precision floating-point arithmetic.

Production neural network systems may use:

- Float32.
- Float16.
- Brain Float16.
- Mixed precision.

Lower precision can improve memory usage and computational throughput, but it introduces additional numerical stability considerations.

### Derivative conventions

ReLU is not mathematically differentiable at zero.

The implementations choose a derivative of zero at that point.

This is an accepted subgradient convention for practical backpropagation.

Leaky ReLU also has a change in slope at zero.

The implementation uses the negative-side slope at the boundary.

## 16. Testing and Failure Conditions

The source files demonstrate validation and numerical behavior across common operating conditions.

| Test condition | Expected behavior |
|---|---|
| Sigmoid input zero | Output 0.5 |
| Tanh input zero | Output 0 |
| ReLU negative input | Output 0 |
| Leaky ReLU negative input | Negative scaled output |
| GELU input zero | Output 0 |
| Softmax large logits | Finite normalized probabilities |
| Empty softmax vector | Validation failure |
| NaN input | Validation failure |
| Infinite input | Validation failure |
| Negative Leaky ReLU alpha | Validation failure |
| Mismatched gradient dimensions | Validation failure |
| Invalid probability distribution | Validation failure |
| Unknown activation | Registry or dispatcher failure |
| Invalid classification target | Classification failure |

The Python implementation includes explicit unit tests and finite-difference gradient checks.

The other implementations include executable assertions or runtime tests.

The PostgreSQL script uses database constraints, triggers, and executable evaluation queries.

## 17. Practical Application Scenarios

| Application | Activation choice | Reason |
|---|---|---|
| Binary classification | Sigmoid | Individual binary probability |
| Multiclass classification | Softmax | Normalized mutually exclusive class probabilities |
| Convolutional hidden layers | ReLU | Simple and efficient nonlinear transformation |
| Deep networks with negative-side gradient concerns | Leaky ReLU | Preserves a negative-side gradient |
| Transformer architectures | GELU | Smooth nonlinear transformation |
| Recurrent hidden-state transformations | Tanh | Bounded, zero-centered output |
| Multilabel classification | Independent sigmoid outputs | Multiple classes can be active simultaneously |

The final activation function should be selected according to the target representation and loss function.

An output activation cannot be selected independently of the learning objective.

## 18. Limitations

These implementations are numerical demonstrations and activation-management examples rather than complete automatic differentiation frameworks.

They do not implement:

- Trainable weight matrices.
- Bias optimization.
- Optimizers such as SGD or Adam.
- Complete neural network training loops.
- Automatic differentiation graphs.
- GPU kernels.
- Distributed training.
- Mixed-precision execution.
- Full model serialization.
- Production model serving infrastructure.

The JavaScript batch executor demonstrates asynchronous orchestration, not GPU acceleration or guaranteed CPU parallelism.

The SQL database records model configurations and inference results. It is not intended to replace a dedicated numerical computing engine.

The GELU implementation uses an approximation, and the derivative calculations use the same approximation to maintain consistency between forward and backward operations.

These boundaries distinguish the implemented activation mechanisms from the broader infrastructure required to train, deploy, and operate production neural networks.
