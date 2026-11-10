# Neural Network Mathematics

## 1. Mathematical Foundations

A neural network is a computational system composed of interconnected processing units called artificial neurons. Each neuron receives numerical inputs, combines them using trainable weights, adds a bias, and transforms the resulting value through an activation function.

The mathematical operation performed by an individual neuron is:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

Where:

| Symbol | Meaning |
|---|---|
| \(x_i\) | Input feature |
| \(w_i\) | Trainable weight |
| \(n\) | Number of inputs |
| \(b\) | Bias |
| \(z\) | Weighted sum or pre-activation value |
| \(f\) | Activation function |
| \(a\) | Neuron output |

The final neuron output is:

\[
a=f(z)
\]

Weights determine the influence of individual inputs. Bias shifts the weighted sum independently of the input values. The activation function introduces a transformation that determines the neuron's output.

The five implementations use these mathematical operations as their central computational model.

## 2. Artificial Neurons

An artificial neuron is the fundamental computational unit of a neural network.

Consider three input features:

\[
x=[1.0,2.0,0.5]
\]

The associated weights are:

\[
w=[0.5,-0.3,0.8]
\]

With bias:

\[
b=0.2
\]

The weighted sum becomes:

\[
z=(1.0)(0.5)+(2.0)(-0.3)+(0.5)(0.8)+0.2
\]

\[
z=0.5-0.6+0.4+0.2
\]

\[
z=0.5
\]

For a sigmoid activation:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Therefore:

\[
a=\sigma(0.5)\approx0.622459
\]

The Python, JavaScript, C++17, and Java implementations execute this calculation independently.

### Neuron components

| Component | Function |
|---|---|
| Inputs | Supply numerical features |
| Weights | Represent learned input importance |
| Bias | Adjusts the activation threshold |
| Weighted sum | Combines inputs and parameters |
| Activation | Transforms the weighted sum |
| Output | Supplies information to subsequent neurons |

A neuron does not independently determine the entire behavior of a network. Its output becomes an input to other neurons in subsequent layers.

## 3. Weights and Bias

### Weights

A weight represents the strength and direction of an input connection.

- Positive weights increase the weighted sum when the corresponding input is positive.
- Negative weights decrease the weighted sum when the corresponding input is positive.
- Larger absolute values produce stronger contributions.
- Zero weights eliminate the contribution of their associated inputs.

For example:

\[
z=0.7x_1-0.4x_2+0.2x_3
\]

Each input contributes independently to the final weighted sum.

Weights are trainable parameters. Training algorithms adjust them to reduce prediction error.

### Bias

Bias is an additional trainable parameter.

\[
z=w^Tx+b
\]

Without bias, a linear neuron must produce zero when every input is zero. A bias allows the activation threshold or decision boundary to shift.

Weights and biases are distinct parameters and receive separate gradient updates during training.

## 4. Activation Functions

Activation functions transform pre-activation values into neuron outputs.

The implementations support five activation functions.

### Linear activation

\[
f(z)=z
\]

Derivative:

\[
f'(z)=1
\]

Characteristics:

- Preserves the weighted sum.
- Useful for regression output layers.
- Does not independently introduce nonlinearity.

Stacking only linear transformations produces another linear transformation. Consequently, a network containing only linear activation functions cannot represent arbitrary nonlinear relationships.

### Sigmoid activation

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Derivative:

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

Characteristics:

- Produces outputs between zero and one.
- Commonly used for binary classification outputs.
- Has a smooth mathematical derivative.
- Can suffer from vanishing gradients when its input has a large positive or negative magnitude.

The implementations use a numerically stable sigmoid calculation to reduce exponential overflow.

### Hyperbolic tangent

\[
f(z)=\tanh(z)
\]

Derivative:

\[
f'(z)=1-\tanh^2(z)
\]

Characteristics:

- Produces outputs between negative one and positive one.
- Is zero-centered.
- Can represent positive and negative intermediate activations.
- Can suffer from vanishing gradients in saturated regions.

The Python, C++, and Java XOR examples use tanh in the hidden layer.

### ReLU

\[
f(z)=\max(0,z)
\]

Derivative:

\[
f'(z)=
\begin{cases}
1 & z>0\\
0 & z\leq0
\end{cases}
\]

Characteristics:

- Computationally inexpensive.
- Preserves positive values.
- Produces zero for negative inputs.
- Can produce inactive neurons when their inputs remain negative.

At zero, the derivative convention used in these implementations is zero.

### Leaky ReLU

\[
f(z)=
\begin{cases}
z & z>0\\
0.01z & z\leq0
\end{cases}
\]

Derivative:

\[
f'(z)=
\begin{cases}
1 & z>0\\
0.01 & z\leq0
\end{cases}
\]

Leaky ReLU retains a small negative-region gradient, reducing the risk of permanently inactive neurons.

### Activation comparison

| Function | Output range | Main characteristic |
|---|---|---|
| Linear | Negative to positive infinity | Identity transformation |
| Sigmoid | 0 to 1 | Probability-like output |
| Tanh | -1 to 1 | Zero-centered output |
| ReLU | 0 to positive infinity | Efficient positive activation |
| Leaky ReLU | Negative to positive infinity | Small negative-region gradient |

The choice of activation depends on the mathematical behavior required by the network architecture.

## 5. Dense Layers

A dense layer connects every neuron in one layer to every neuron in the next layer.

For an input vector:

\[
x\in\mathbb{R}^{n}
\]

A dense layer containing \(m\) neurons uses:

\[
W\in\mathbb{R}^{m\times n}
\]

and:

\[
b\in\mathbb{R}^{m}
\]

Its mathematical operation is:

\[
z=Wx+b
\]

The activation function is applied element by element:

\[
a=f(Wx+b)
\]

For a network with architecture:

\[
2\rightarrow4\rightarrow1
\]

The network contains:

- Two input features.
- Four hidden neurons.
- One output neuron.

The total number of trainable parameters is:

\[
(2\times4+4)+(4\times1+1)
\]

\[
=12+5=17
\]

The general parameter count for a fully connected network is:

\[
P=\sum_{l=1}^{L}
(n_{l-1}n_l+n_l)
\]

Where \(n_l\) represents the number of neurons in layer \(l\).

## 6. Forward Propagation

Forward propagation calculates the output of a network by passing information through its layers.

For the first layer:

\[
a^{(1)}=f(W^{(1)}x+b^{(1)})
\]

For the second layer:

\[
a^{(2)}=f(W^{(2)}a^{(1)}+b^{(2)})
\]

For the final layer:

\[
\hat{y}=f(W^{(L)}a^{(L-1)}+b^{(L)})
\]

The notation \(\hat{y}\) represents the network prediction.

The computational sequence is:

Input features
→ Weighted sums
→ Bias addition
→ Activation functions
→ Hidden-layer outputs
→ Subsequent layer calculations
→ Final prediction

Each layer depends on the output of the preceding layer.

The implementations retain intermediate weighted sums and activation values because these values are required for derivative calculations during backpropagation.

## 7. Loss Functions

A loss function measures the difference between a network's prediction and its expected target.

The implementations use Mean Squared Error (MSE):

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
\]

Where:

- \(\hat{y}_i\) represents a prediction.
- \(y_i\) represents the corresponding target.
- \(n\) represents the number of output values.

For example:

| Prediction | Target | Squared error |
|---:|---:|---:|
| 0.8 | 1.0 | 0.04 |
| 0.3 | 0.0 | 0.09 |
| 0.6 | 0.5 | 0.01 |

Therefore:

\[
MSE=\frac{0.04+0.09+0.01}{3}
\]

\[
MSE=0.046667
\]

MSE is useful for demonstrating gradient-based optimization. For practical classification systems, binary cross-entropy or categorical cross-entropy may be more appropriate.

## 8. Derivatives and Gradient Descent

A derivative describes how a mathematical output changes when its input changes.

Neural network training requires derivatives because weights and biases must be adjusted to reduce the loss function.

For a parameter \(w\), gradient descent updates the parameter using:

\[
w_{\text{new}}=
w_{\text{old}}-\eta\frac{\partial L}{\partial w}
\]

For bias:

\[
b_{\text{new}}=
b_{\text{old}}-\eta\frac{\partial L}{\partial b}
\]

Where:

- \(L\) is the loss.
- \(\eta\) is the learning rate.
- The partial derivative measures the effect of the parameter on the loss.

A learning rate that is too large can cause unstable optimization. A very small learning rate can result in slow convergence.

## 9. Backpropagation

Backpropagation calculates gradients by applying the chain rule from the output layer toward the input layer.

For an output neuron using MSE:

\[
\delta^{(L)}
=
2(\hat{y}-y)f'(z^{(L)})
\]

For a hidden layer:

\[
\delta^{(l)}
=
(W^{(l+1)})^T
\delta^{(l+1)}
\odot f'(z^{(l)})
\]

The symbol \(\odot\) represents element-wise multiplication.

The weight gradient is:

\[
\frac{\partial L}{\partial W^{(l)}}
=
\delta^{(l)}(a^{(l-1)})^T
\]

The bias gradient is:

\[
\frac{\partial L}{\partial b^{(l)}}
=
\delta^{(l)}
\]

The implementations calculate all layer gradients before updating parameters. This is important because calculating hidden-layer gradients using partially updated weights would produce incorrect gradient propagation.

### Training workflow

1. Receive an input and expected output.
2. Execute forward propagation.
3. Calculate prediction error.
4. Calculate output-layer gradients.
5. Propagate gradients through hidden layers.
6. Calculate weight and bias gradients.
7. Update trainable parameters.
8. Repeat across training samples.
9. Measure training loss.
10. Stop when the configured convergence condition is satisfied or the epoch limit is reached.

The Python, C++, and Java programs implement this training workflow.

## 10. XOR Classification

XOR is a nonlinear classification problem.

| Input A | Input B | Expected output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single linear neuron cannot represent the XOR decision boundary.

A network with a nonlinear hidden layer can represent this relationship.

The implementations demonstrate the architecture:

\[
2\rightarrow4\rightarrow1
\]

The hidden layer uses tanh activation, and the output layer uses sigmoid activation.

The output is interpreted using a classification threshold:

\[
\hat{c}=
\begin{cases}
1 & \hat{y}\geq0.5\\
0 & \hat{y}<0.5
\end{cases}
\]

The XOR example demonstrates why nonlinear activation functions are necessary for learning nonlinear relationships.

The demonstration uses online gradient updates, meaning parameters are updated after each training sample.

## 11. Python Implementation

The Python program provides a complete mathematical and training workflow.

| Component | Implementation |
|---|---|
| Activation functions | Enum and activation functions |
| Neuron | Dataclass |
| Weighted sum | `weighted_sum()` |
| Loss | Mean Squared Error |
| Network | `NeuralNetwork` |
| Initialization | Xavier initialization |
| Forward propagation | Layer-by-layer calculation |
| Backpropagation | Explicit delta calculations |
| Optimization | Gradient descent |
| Persistence | JSON serialization |
| Validation | Dimension and finite-number checks |
| Numerical testing | Central finite differences |

The implementation uses `math.fsum()` for more accurate accumulation of weighted contributions and errors.

Model persistence stores the architecture, activation configuration, weights, and biases. The loading operation validates the model structure before restoring parameters.

The JSON file should be treated as untrusted input in production. A production model-loading service should impose file-size limits, access controls, schema validation, and integrity verification.

## 12. JavaScript Implementation

The JavaScript implementation uses an event-driven design.

Its primary components are:

| Component | Responsibility |
|---|---|
| `ActivationRegistry` | Maintains activation implementations |
| `Neuron` | Computes neuron outputs |
| `DenseLayer` | Manages groups of neurons |
| `NeuralNetwork` | Coordinates forward propagation |
| `EventEmitter` | Publishes computational events |
| `predictBatch()` | Processes batches with bounded concurrency |
| `saveModel()` | Persists model configuration |
| `loadModel()` | Reconstructs the network |

The implementation emits events when neurons calculate outputs, parameters change, predictions complete, and batches finish.

This separation is useful for monitoring, logging, and integrating inference operations with application services.

The batch inference implementation limits concurrency to avoid creating an unrestricted number of simultaneous tasks.

Its asynchronous interface provides scheduling and event-loop integration. The mathematical inference itself is synchronous CPU computation. JavaScript promises do not automatically make neural network arithmetic parallel.

For computationally intensive workloads, worker threads or specialized numerical execution systems would be appropriate.

## 13. C++17 Implementation

The C++17 program emphasizes explicit domain modeling and controlled memory usage.

| Component | Responsibility |
|---|---|
| `Activation` | Enumerates activation functions |
| `NeuronResult` | Stores weighted sum and output |
| `ActivationFunctions` | Performs activation calculations |
| `Neuron` | Stores neuron parameters |
| `NeuralNetwork` | Manages architecture and propagation |
| `TrainingMetrics` | Stores training performance |
| `numericalDerivative()` | Validates mathematical derivatives |

The program uses standard library vectors for weights, biases, layer outputs, and gradient values.

Move semantics reduce unnecessary copying when transferring layer matrices and intermediate values.

The neural network uses Xavier initialization and explicitly calculates hidden-layer gradients before modifying weights.

Exception handling rejects invalid dimensions, non-finite values, unsupported configurations, and invalid learning rates.

## 14. Java 17 Implementation

The Java implementation follows an enterprise-oriented domain model.

Its components include:

| Component | Responsibility |
|---|---|
| `Activation` | Defines activation behavior |
| `Neuron` | Represents a trainable computational unit |
| `DenseLayer` | Enforces layer-level dimensions |
| `TrainingSample` | Represents immutable training data |
| `TrainingPolicy` | Defines training constraints |
| `TrainingReport` | Records optimization results |
| `NeuralNetwork` | Coordinates inference and training |
| `Prediction` | Represents an inference result |

Java records provide immutable representations for training samples, predictions, and training reports.

The network constructor validates compatibility between adjacent layers.

The training policy enforces valid epoch counts, learning rates, convergence thresholds, and reporting intervals.

Neuron parameter updates use version counters to make parameter changes observable.

The design separates mathematical calculations, network structure, training policy, and operational results.

The implementation is educational rather than a concurrent production training service. Its mutable network parameters require synchronization or isolated model instances if accessed concurrently.

## 15. PostgreSQL Implementation

The PostgreSQL implementation represents neural network mathematics through a relational data model.

### Database architecture

| Table | Purpose |
|---|---|
| `neural_networks` | Network registration and configuration |
| `network_layers` | Layer positions and activation functions |
| `neurons` | Neuron identifiers and biases |
| `neuron_weights` | Connections and trainable weights |
| `training_runs` | Training execution records |
| `inference_records` | Prediction execution records |

The relationship between entities is:

Neural Network
→ Network Layers
→ Neurons
→ Weight Connections

Neural Network
→ Training Runs

Neural Network
→ Inference Records

### Relational integrity

Primary keys uniquely identify records.

Foreign keys prevent connections to nonexistent neurons and prevent layers from referencing nonexistent networks.

Unique constraints prevent duplicate neuron positions and duplicate source-target connections.

Check constraints enforce valid neuron counts, finite numerical parameters, and valid learning rates.

Indexes support common retrieval patterns involving network identifiers, layer positions, source neurons, target neurons, and execution dates.

### Database mathematical functions

The SQL script defines:

- `activation_value()`
- `activation_derivative()`
- `calculate_weighted_sum()`
- `evaluate_neuron()`
- `validate_network_structure()`
- `network_training_eligible()`

These functions allow mathematical operations to be performed directly against stored network parameters.

### Training workflow

The `training_runs` table records training execution states.

The supported states are:

- Pending
- Running
- Completed
- Failed

The transition trigger prevents completed or failed training runs from being restarted through ordinary updates.

The `start_training_run()` procedure checks the network configuration, learning rate, epoch count, and current training state before creating a new execution record.

The procedure records workflow state rather than performing numerical gradient-based training. The Python, C++, and Java implementations contain the actual educational training algorithms.

### Reporting

The `training_summary` view provides:

- Number of training runs.
- Completed training runs.
- Failed training runs.
- Average final loss.
- Best final loss.

Additional SQL queries examine inference performance, connection counts, and weight distributions.

### Transaction handling

The script uses a database transaction for its schema and sample-data initialization.

A savepoint demonstrates recovery from a rejected weight insertion.

The database constraints provide an additional integrity boundary independent of application-level validation.

## 16. Mathematical and Implementation Differences

The five source files share the same underlying mathematical definitions but address different engineering requirements.

| Area | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Neurons | Dataclass | Event-driven class | Domain class | Domain class | Relational records |
| Weighted sums | Function | Function | Function | Function | Stored function |
| Activation | Enum | Registry | Enum | Enum | PostgreSQL enum |
| Layers | Nested lists | DenseLayer | Matrix | DenseLayer | Layer records |
| Forward propagation | Implemented | Implemented | Implemented | Implemented | Single-neuron evaluation |
| Backpropagation | Implemented | Not implemented | Implemented | Implemented | Not implemented |
| Gradient descent | Implemented | Not implemented | Implemented | Implemented | Execution tracking |
| Persistence | JSON | JSON | Runtime memory | Runtime memory | Relational tables |
| Validation | Exceptions | Custom errors | Exceptions | Custom exceptions | Constraints and functions |

These differences are intentional.

The implementations demonstrate that the same mathematical model can be represented through procedural calculations, object-oriented abstractions, event-driven services, strongly typed domain models, and relational database structures.

## 17. Numerical Stability and Validation

Neural network calculations require careful handling of numerical values.

### Exponential overflow

The sigmoid function contains an exponential operation:

\[
e^{-z}
\]

Large input magnitudes can create overflow or underflow conditions.

The implementations use a stable calculation that treats positive and negative inputs separately.

### Dimension mismatches

For a neuron with \(n\) weights, the input vector must contain exactly \(n\) elements.

A mismatch indicates an invalid mathematical operation and must be rejected before computation.

### Non-finite values

NaN and infinity can propagate through weighted sums, activations, and gradients.

The implementations validate numerical inputs to prevent invalid calculations.

### Gradient verification

The derivative implementations can be tested using central finite differences:

\[
f'(x)\approx
\frac{f(x+\epsilon)-f(x-\epsilon)}
{2\epsilon}
\]

The analytical and numerical derivatives should be close when the function is differentiable at the selected point and the finite-difference interval is sufficiently small.

The sigmoid derivative verification in the four application implementations demonstrates this principle.

At activation-function discontinuities or nondifferentiable points, derivative comparisons require an explicitly selected derivative convention.

## 18. Performance Considerations

For a dense layer with \(n\) input neurons and \(m\) output neurons, the computational cost is:

\[
O(nm)
\]

For a network containing multiple dense layers, forward propagation costs:

\[
O\left(
\sum_{l=1}^{L}n_{l-1}n_l
\right)
\]

Backpropagation has a similar asymptotic computational cost.

The main performance considerations are:

- Store layer parameters in contiguous structures where possible.
- Avoid unnecessary copying of intermediate vectors.
- Validate architecture dimensions before training.
- Use batch processing to improve computational efficiency.
- Avoid excessive logging during large training workloads.
- Use matrix multiplication libraries for larger networks.
- Separate model inference from parameter updates.
- Monitor memory usage when retaining training history.

The C++ implementation uses standard vectors, while the Python implementation favors readable mathematical operations.

The JavaScript implementation bounds asynchronous batch scheduling. The Java implementation uses immutable input records and explicit layer boundaries.

The PostgreSQL implementation uses indexes for metadata retrieval rather than treating relational storage as a high-performance tensor computation engine.

## 19. Security and Production Considerations

A production neural network system requires additional controls beyond mathematical correctness.

Important considerations include:

- Validate external input dimensions and numerical ranges.
- Reject NaN and infinite values.
- Restrict access to model configuration and training records.
- Validate model files before loading.
- Protect stored model parameters against unauthorized modification.
- Apply resource limits to training and inference operations.
- Avoid exposing sensitive training data in application logs.
- Maintain model version information.
- Separate model configuration permissions from inference permissions.
- Record training failures and numerical anomalies.
- Monitor prediction latency and model output distributions.

The examples provide basic mathematical validation and failure handling. They do not implement distributed training, hardware acceleration, production model serving, or comprehensive access-control systems.

## 20. Practical Mathematical Distinctions

| Concept | What it represents |
|---|---|
| Input | Numerical information received by a neuron |
| Weight | Strength of an input connection |
| Bias | Independent trainable offset |
| Weighted sum | Linear combination of inputs and parameters |
| Pre-activation | Value before activation transformation |
| Activation | Nonlinear or linear transformation |
| Neuron output | Result passed to the next computational stage |
| Layer | Collection of neurons operating on a shared input vector |
| Forward propagation | Computation of network predictions |
| Loss | Numerical measure of prediction error |
| Derivative | Rate of change of a mathematical function |
| Gradient | Collection of parameter derivatives |
| Backpropagation | Calculation of gradients through network layers |
| Gradient descent | Parameter optimization using gradients |
| Epoch | One complete pass through the training dataset |
| Learning rate | Step-size parameter controlling updates |

These concepts form the mathematical foundation for understanding how artificial neural networks process information and learn relationships from numerical data.
