# Forward Propagation in Artificial Neural Networks

## 1. Technical Scope

Forward propagation is the computational process through which an artificial neural network transforms input features into an output prediction.

It begins with an input vector, processes the vector through successive neural network layers, and produces an output using learned weights, biases, and activation functions.

This implementation covers the mathematical and computational foundations of forward propagation, including:

- Neural network architecture.
- Input, hidden, and output layers.
- Neurons and their connections.
- Weight matrices and bias vectors.
- Matrix-based computation.
- Activation functions.
- Sequential layer execution.
- Single-sample inference.
- Batch inference.
- Numerical stability.
- Architecture validation.
- Execution tracing.
- Model parameter management.

The five implementations demonstrate different engineering approaches to the same computational process.

| Implementation | Primary perspective |
|---|---|
| Python | Mathematical modeling, simulation, and validation |
| JavaScript | Event-driven inference and asynchronous batch execution |
| C++17 | Explicit matrix computation and efficient state management |
| Java 17 | Enterprise-oriented neural network architecture and execution services |
| PostgreSQL | Relational model storage, architecture validation, and inference processing |

## 2. Neural Network Architecture

A feedforward neural network consists of interconnected computational units called neurons.

Its architecture generally contains three categories of layers.

### 2.1 Input Layer

The input layer receives the original feature values.

For an input containing three features:

\[
X =
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
\]

An example input is:

\[
X =
\begin{bmatrix}
0.2\\
0.5\\
0.8
\end{bmatrix}
\]

Each element represents one numerical feature.

The input layer does not necessarily perform a weighted transformation. It supplies the feature vector to the first computational layer.

### 2.2 Hidden Layers

Hidden layers transform information received from preceding layers.

A hidden layer contains multiple neurons. Each neuron receives values from the previous layer and computes a weighted sum.

For a network with three input features and four hidden neurons:

\[
W_1 =
\begin{bmatrix}
w_{11}&w_{12}&w_{13}\\
w_{21}&w_{22}&w_{23}\\
w_{31}&w_{32}&w_{33}\\
w_{41}&w_{42}&w_{43}
\end{bmatrix}
\]

The matrix contains four rows because the layer has four output neurons and three columns because each neuron receives three inputs.

### 2.3 Output Layer

The output layer transforms the final hidden representation into the network's prediction.

For classification involving two output classes, the output layer can contain two neurons.

A softmax activation converts their weighted sums into normalized probabilities.

For example:

\[
Y =
\begin{bmatrix}
0.35\\
0.65
\end{bmatrix}
\]

The second output represents the higher predicted class probability.

The network does not automatically establish whether a prediction is correct. Evaluating prediction quality requires comparison against known target values.

## 3. Neurons, Weights, and Biases

### 3.1 Neurons

A neuron is a computational unit that receives numerical inputs, applies weights, adds a bias, and evaluates an activation function.

Its basic operation is:

\[
z = \sum_{i=1}^{n}w_i x_i+b
\]

Where:

| Symbol | Meaning |
|---|---|
| \(x_i\) | Input feature |
| \(w_i\) | Weight associated with that feature |
| \(b\) | Bias |
| \(z\) | Weighted sum before activation |
| \(n\) | Number of incoming features |

The neuron produces its output by applying an activation function:

\[
a=f(z)
\]

### 3.2 Weights

Weights determine the strength and direction of the influence that an input feature has on a neuron.

- Positive weights increase the weighted contribution.
- Negative weights decrease the weighted contribution.
- Zero weights eliminate the direct contribution of the corresponding feature.

Weights are parameters learned during training.

Forward propagation uses these parameters but does not update them.

### 3.3 Biases

A bias is an independently learned parameter associated with a neuron.

It shifts the weighted sum before activation.

Without a bias, a neuron computes a transformation constrained by the absence of an independent offset.

A bias allows the activation threshold or response to shift independently of the input values.

Each output neuron has one bias.

For a dense layer with four output neurons, the bias vector contains four elements.

## 4. Matrix-Based Forward Computation

Matrix operations provide an efficient representation of dense neural network computation.

Instead of calculating each neuron independently, the entire layer can be evaluated using matrix-vector multiplication.

The principal equation is:

\[
Z = WA+b
\]

The activation operation follows:

\[
A_{\text{next}}=f(Z)
\]

Where:

- \(W\) represents the weight matrix.
- \(A\) represents the input activation vector.
- \(b\) represents the bias vector.
- \(Z\) represents the pre-activation vector.
- \(f\) represents the activation function.

### 4.1 Dimension Compatibility

Suppose:

\[
W \in \mathbb{R}^{m \times n}
\]

\[
A \in \mathbb{R}^{n \times 1}
\]

\[
b \in \mathbb{R}^{m \times 1}
\]

Then:

\[
WA \in \mathbb{R}^{m \times 1}
\]

Consequently:

\[
Z \in \mathbb{R}^{m \times 1}
\]

The weight matrix must have as many columns as the input vector has elements.

Its number of rows determines the number of output neurons.

This dimension relationship is enforced in all four programming implementations and the database architecture validation.

### 4.2 Worked Example

Consider:

\[
A =
\begin{bmatrix}
1\\
2
\end{bmatrix}
\]

\[
W =
\begin{bmatrix}
0.5&-0.2\\
0.3&0.8
\end{bmatrix}
\]

\[
b =
\begin{bmatrix}
0.1\\
-0.1
\end{bmatrix}
\]

First calculate:

\[
WA =
\begin{bmatrix}
(0.5)(1)+(-0.2)(2)\\
(0.3)(1)+(0.8)(2)
\end{bmatrix}
\]

\[
WA =
\begin{bmatrix}
0.1\\
1.9
\end{bmatrix}
\]

Add the bias:

\[
Z =
\begin{bmatrix}
0.1\\
1.9
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
-0.1
\end{bmatrix}
\]

\[
Z =
\begin{bmatrix}
0.2\\
1.8
\end{bmatrix}
\]

Applying ReLU:

\[
A_{\text{next}} =
\begin{bmatrix}
0.2\\
1.8
\end{bmatrix}
\]

The Python implementation explicitly demonstrates this computation.

## 5. Activation Functions

Activation functions determine how a neuron transforms its weighted sum.

They introduce nonlinearity into the network, allowing it to represent more complex relationships than a simple linear transformation.

### 5.1 Linear Activation

\[
f(x)=x
\]

The output equals the input.

Linear activation is useful when the network needs an unrestricted numerical output, such as a regression prediction.

### 5.2 ReLU

Rectified Linear Unit:

\[
f(x)=\max(0,x)
\]

Behavior:

| Input | Output |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

ReLU is widely used in hidden layers because it is computationally inexpensive.

One limitation is the possibility of neurons becoming inactive when they consistently receive negative weighted sums.

### 5.3 Sigmoid

\[
f(x)=\frac{1}{1+e^{-x}}
\]

The output lies between zero and one.

Typical applications include binary classification output layers.

Large positive inputs approach one, while large negative inputs approach zero.

The implementations use a numerically stable calculation to reduce overflow risk.

### 5.4 Hyperbolic Tangent

\[
f(x)=\tanh(x)
\]

The output lies between negative one and positive one.

Unlike sigmoid, its output is centered around zero.

### 5.5 Softmax

Softmax converts a vector of weighted sums into a probability distribution.

\[
P_i =
\frac{e^{z_i}}
{\sum_{j=1}^{k}e^{z_j}}
\]

Where \(k\) is the number of output classes.

Its outputs satisfy:

\[
\sum_{i=1}^{k}P_i=1
\]

Each output represents a normalized class score.

For numerical stability, the implementations subtract the largest input value before calculating exponentials:

\[
P_i =
\frac{e^{z_i-z_{\max}}}
{\sum_j e^{z_j-z_{\max}}}
\]

This produces the same mathematical result while reducing the risk of exponential overflow.

## 6. Multiple-Layer Forward Propagation

Forward propagation applies the same computational pattern sequentially across the network.

Consider:

- Three input features.
- Four hidden neurons.
- Two output neurons.

The architecture is:

\[
3 \rightarrow 4 \rightarrow 2
\]

The first layer computes:

\[
Z_1=W_1X+b_1
\]

\[
A_1=\operatorname{ReLU}(Z_1)
\]

The second layer computes:

\[
Z_2=W_2A_1+b_2
\]

\[
A_2=\operatorname{Softmax}(Z_2)
\]

The final prediction is:

\[
\hat{Y}=A_2
\]

Plain-text representation:

Input Features

→ First Dense Layer

→ Weighted Sum

→ Bias Addition

→ ReLU Activation

→ Hidden Representation

→ Second Dense Layer

→ Weighted Sum

→ Bias Addition

→ Softmax Activation

→ Final Prediction

The output of one layer becomes the input of the next layer.

This sequential dependency is the central mechanism of forward propagation.

## 7. Forward Propagation and Backpropagation

These are related but distinct neural network operations.

| Aspect | Forward propagation | Backpropagation |
|---|---|---|
| Primary purpose | Calculate predictions | Calculate gradients |
| Direction | Input to output | Output to earlier layers |
| Main operations | Matrix multiplication and activation | Chain rule and gradient computation |
| Parameters | Reads weights and biases | Computes their gradients |
| Result | Predicted outputs | Parameter derivatives |
| Training role | Prediction calculation | Learning signal calculation |
| Inference role | Required | Not normally required |

Forward propagation alone does not train a neural network.

Training requires an additional process involving a loss function, gradient computation, and an optimizer.

The implementations in this project focus on forward computation and inference rather than parameter optimization.

## 8. Python Implementation

The Python implementation provides a self-contained simulation using standard-library functionality.

### Main components

| Component | Responsibility |
|---|---|
| `Activation` | Defines supported activation functions |
| `DenseLayer` | Stores weights, biases, and layer configuration |
| `NeuralNetwork` | Coordinates sequential layer execution |
| `validate_vector` | Validates numerical inputs |
| `activate` | Applies activation policies |
| `softmax` | Calculates normalized probabilities |
| `predict_batch` | Processes multiple independent samples |
| `save` and `load` | Serialize and restore model parameters |

### Important implementation decisions

The weight orientation is:

`weights[output_neuron][input_neuron]`

This makes each row correspond to one output neuron.

`math.fsum` is used for more accurate summation of weighted contributions.

The network validates adjacent layer dimensions before accepting an architecture.

Execution tracing records:

- Layer name.
- Pre-activation weighted sums.
- Activated outputs.

The model serialization mechanism stores architecture and parameters as JSON.

The demonstration also verifies that the restored model produces the same prediction.

### Validation coverage

The script checks:

- Empty vectors.
- Non-finite numerical values.
- Incompatible dimensions.
- Incorrect bias counts.
- Invalid weight matrices.
- Softmax normalization.
- Extreme sigmoid inputs.
- Batch inference.
- Model serialization.

## 9. JavaScript Implementation

The JavaScript implementation uses an event-driven architecture.

### Main components

| Component | Responsibility |
|---|---|
| `DenseLayer` | Encapsulates individual layer parameters |
| `NeuralNetwork` | Coordinates network inference |
| `EventEmitter` | Publishes inference events |
| `predictBatch` | Performs asynchronous batch processing |
| `saveModel` | Persists the model configuration |
| `loadModel` | Restores the network configuration |

### Event-driven inference

The network publishes an `inference` event when a computation completes.

An `inferenceError` event reports failed computations.

This separates numerical execution from application-level monitoring.

### Asynchronous batch processing

The batch inference implementation uses asynchronous workers.

Each worker processes independent samples and stores the result at its original index.

This preserves input ordering.

The implementation limits the number of workers to prevent unbounded task creation.

JavaScript execution within a single event loop does not automatically provide CPU parallelism. The asynchronous implementation improves task orchestration but does not replace worker threads or native numerical libraries for large computational workloads.

### Parameter encapsulation

Weights and biases use private class fields.

External callers cannot directly modify the network's internal parameter arrays.

This reduces accidental state corruption during inference.

## 10. C++17 Implementation

The C++ implementation emphasizes explicit memory ownership, typed domain structures, and predictable computational behavior.

### Main components

| Component | Responsibility |
|---|---|
| `Vector` | Represents numerical feature and activation vectors |
| `Matrix` | Represents layer weight matrices |
| `Activation` | Defines activation types |
| `DenseLayer` | Performs one layer transformation |
| `NeuralNetwork` | Manages sequential layer execution |
| `LayerResult` | Stores weighted sums and activated outputs |
| `Prediction` | Stores the final output and execution trace |

### Engineering decisions

The implementation uses `std::vector<double>` for numerical storage.

Layer parameters are encapsulated inside the `DenseLayer` class.

The network owns its collection of layers.

`std::move` is used when transferring computed intermediate results to avoid unnecessary copies.

The implementation validates matrix dimensions before execution.

It also rejects non-finite inputs and intermediate weighted sums.

The implementation deliberately avoids external numerical libraries so that the underlying computation remains visible.

For production numerical workloads, optimized matrix libraries and hardware acceleration can provide substantially better throughput.

## 11. Java 17 Implementation

The Java implementation models neural network inference as an enterprise-oriented service.

### Domain structure

| Component | Purpose |
|---|---|
| `NetworkDefinition` | Represents a validated architecture |
| `DenseLayer` | Defines layer parameters and computation |
| `ActivationPolicy` | Defines activation behavior |
| `InferenceService` | Coordinates inference execution |
| `Prediction` | Represents an inference result |
| `LayerTrace` | Records intermediate computation |
| `ExecutionStatus` | Represents execution states |

### Immutable model configuration

The network definition uses immutable layer collections.

Layer constructors copy supplied weights and biases.

The prediction and trace records also protect their internal arrays through defensive copying.

These decisions reduce accidental mutation of model state.

### Service-based execution

The inference service accepts an execution identifier and input vector.

It validates the request, executes each layer, and produces a structured prediction.

A completion callback receives the completed result.

This structure separates model definition from inference orchestration.

### Failure management

The implementation distinguishes:

- Invalid input.
- Incompatible dimensions.
- Invalid weighted sums.
- Activation computation failures.
- Batch processing errors.

Custom exceptions make failures more identifiable than generic runtime errors.

The example is intentionally an in-memory service. It does not include authentication, distributed execution, database persistence, or production-grade request auditing.

## 12. PostgreSQL Implementation

The PostgreSQL implementation models neural network architectures and inference execution as relational data.

### 12.1 Database Schema

The `neural_network` schema contains the following primary entities.

| Table | Responsibility |
|---|---|
| `neural_models` | Stores model identity, architecture dimensions, version, and status |
| `dense_layers` | Stores layer ordering and dimensions |
| `layer_parameters` | Stores weights and biases |
| `inference_requests` | Stores input samples and inference states |
| `layer_execution_results` | Stores intermediate layer computations |

### 12.2 Relational Relationships

The database relationships are:

`neural_models`

→ `dense_layers`

→ `layer_parameters`

`neural_models`

→ `inference_requests`

`inference_requests`

→ `layer_execution_results`

Each model can contain multiple ordered dense layers.

Each dense layer has one associated parameter record.

Each inference request belongs to one model.

An inference request can contain multiple layer execution records.

### 12.3 Weight Storage

The implementation stores weight matrices as JSONB.

For a layer containing two output neurons and three input features, the weight matrix contains two rows with three values each.

Biases are stored as a PostgreSQL `DOUBLE PRECISION[]` array.

The JSONB representation allows the database to preserve matrix structure without introducing a separate relational row for every individual weight.

Its disadvantage is that individual matrix elements are less convenient to query and index than normalized numerical records.

For large production models, external model storage or specialized numerical storage is generally more appropriate.

### 12.4 Database Validation

The SQL implementation includes:

- Positive layer dimensions.
- Primary and foreign keys.
- Unique layer ordering.
- Unique model names.
- Bias dimension validation.
- Weight matrix dimension validation.
- Numeric value validation.
- Sequential layer compatibility.
- Input feature validation.
- Output feature validation.
- Model activation validation.

The `validate_model` function checks that each layer receives the number of features produced by its predecessor.

It also verifies that the final layer produces the declared output dimension.

### 12.5 Inference Processing

The `forward_propagation` function executes the network inside PostgreSQL.

Its process is:

1. Retrieve an active model.
2. Validate the input vector.
3. Validate the architecture.
4. Retrieve the first layer's parameters.
5. Calculate weighted sums.
6. Apply the configured activation.
7. Pass the resulting vector to the next layer.
8. Continue until the final layer.
9. Return the prediction and execution trace as JSONB.

The `process_inference` procedure manages inference request states.

The lifecycle is:

`pending` → `running` → `completed`

A processing failure produces:

`pending` → `running` → `failed`

The request retains its error message for reporting.

### 12.6 Reporting and Analysis

The SQL implementation includes:

- Architecture retrieval.
- Intermediate computation retrieval.
- Inference reports.
- Model parameter aggregation.
- Execution status aggregation.
- Architecture validation.
- Model eligibility evaluation.
- Direct forward propagation queries.

Indexes support common retrieval patterns involving model identity, layer ordering, inference status, and request timestamps.

## 13. Computational Complexity

Consider a dense layer with:

- \(n\) input features.
- \(m\) output neurons.

Each output neuron performs approximately \(n\) multiplications and additions.

Therefore, the dominant computational complexity is:

\[
O(mn)
\]

For a network with multiple layers, the total complexity is:

\[
O\left(\sum_{l=1}^{L}n_{l-1}n_l\right)
\]

Where:

- \(L\) is the number of computational layers.
- \(n_{l-1}\) is the input dimension of layer \(l\).
- \(n_l\) is its output dimension.

The number of trainable parameters in a dense layer is:

\[
P=mn+m
\]

The first term represents weights.

The second term represents biases.

For the implemented architecture:

\[
3 \rightarrow 4 \rightarrow 2
\]

The hidden layer contains:

\[
(3 \times 4)+4=16
\]

parameters.

The output layer contains:

\[
(4 \times 2)+2=10
\]

parameters.

Total:

\[
P=16+10=26
\]

The model therefore contains 26 trainable parameters.

### Memory Complexity

The parameter storage requirement is proportional to:

\[
O\left(\sum_{l=1}^{L}n_{l-1}n_l\right)
\]

During single-sample inference, storing intermediate activations requires memory proportional to the layer widths.

Storing a complete execution trace increases memory consumption because pre-activation and post-activation values are retained.

For high-throughput inference, traces should normally be optional.

## 14. Batch Forward Propagation

Batch inference processes multiple independent input samples.

For a batch containing \(B\) samples, the input can be represented as a matrix:

\[
X \in \mathbb{R}^{B \times n}
\]

The layer computation becomes:

\[
Z=XW^T+b
\]

The activation is applied element-wise for element-wise activation functions.

For softmax, normalization is performed independently for each sample across its output classes.

Batch processing improves computational efficiency when matrix operations are implemented using optimized numerical libraries.

The Python, JavaScript, C++, and Java implementations provide batch inference interfaces.

Their basic batch implementations process samples independently. They do not implement hardware-accelerated batched matrix multiplication.

## 15. Numerical Stability and Validation

Numerical correctness is essential because neural network computation repeatedly transforms floating-point values.

Important failure conditions include:

| Failure | Consequence | Implemented control |
|---|---|---|
| Incorrect input dimensions | Invalid matrix multiplication | Dimension validation |
| Missing biases | Incomplete neuron computation | Bias count validation |
| Ragged weight matrix | Inconsistent layer structure | Matrix validation |
| NaN input | Unreliable predictions | Finite-value checks |
| Infinite weighted sum | Invalid activation | Numerical validation |
| Sigmoid overflow | Unstable numerical calculation | Stable sigmoid implementation |
| Softmax overflow | Invalid probability distribution | Maximum subtraction |
| Incompatible adjacent layers | Invalid architecture | Network validation |
| Missing model parameters | Incomplete execution | Database validation |

Floating-point arithmetic can introduce small numerical differences across programming languages and database execution environments.

Exact equality is therefore not generally appropriate for comparing independently implemented numerical results. A tolerance-based comparison is preferable.

## 16. Security and Production Considerations

Although forward propagation is a mathematical operation, production inference systems require additional controls.

### Model integrity

- Validate model dimensions before activation.
- Restrict parameter modification to authorized processes.
- Maintain model version information.
- Protect serialized model files.
- Verify model integrity after loading.
- Avoid accepting arbitrary untrusted model configurations.

### Input security

- Validate the number of input features.
- Reject non-finite values.
- Apply application-specific input range validation.
- Limit request sizes.
- Restrict batch dimensions.
- Record meaningful execution failures.

### Operational controls

- Separate inference execution from model training.
- Monitor latency and throughput.
- Track model versions.
- Measure memory consumption.
- Monitor prediction distributions.
- Record execution failures.
- Use bounded concurrency.
- Apply appropriate access controls to persisted parameters.

### Numerical and model limitations

A mathematically valid forward pass does not guarantee a reliable prediction.

Prediction quality depends on training data, parameter quality, model architecture, and the suitability of the input distribution.

The implementations demonstrate computation rather than statistical model evaluation.

## 17. Practical Applications

Forward propagation is a fundamental operation in many neural network applications.

| Application | Input | Output |
|---|---|---|
| Image classification | Image features or pixel representations | Class probabilities |
| Fraud detection | Transaction features | Fraud probability |
| Credit risk modeling | Financial and behavioral features | Risk score |
| Medical prediction | Clinical measurements | Prediction scores |
| Customer segmentation | Customer attributes | Cluster-related scores |
| Demand forecasting | Historical demand features | Forecast values |
| Text classification | Numerical text representations | Category probabilities |
| Predictive maintenance | Equipment sensor measurements | Failure prediction |

The same mathematical operation supports different applications by changing the input representation, architecture, weights, biases, and output interpretation.

## 18. Implementation Comparison

| Capability | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Dense layers | Yes | Yes | Yes | Yes | Yes |
| Weight matrices | Lists | Arrays | STL vectors | Arrays | JSONB |
| Bias vectors | Lists | Arrays | STL vectors | Arrays | SQL arrays |
| ReLU | Yes | Yes | Yes | Yes | Yes |
| Sigmoid | Yes | Yes | Yes | Yes | Yes |
| Tanh | Yes | Yes | Yes | Yes | Yes |
| Softmax | Yes | Yes | Yes | Yes | Yes |
| Batch inference | Yes | Asynchronous | Yes | Yes | Request processing |
| Execution trace | Yes | Yes | Yes | Yes | Yes |
| Parameter serialization | JSON | JSON | In-memory | In-memory | Relational storage |
| Architecture validation | Yes | Yes | Yes | Yes | Yes |
| Custom failure handling | Yes | Yes | Yes | Yes | Yes |

## 19. Execution Requirements

| Deliverable | Runtime | Dependencies |
|---|---|---|
| Python | Python 3.10+ | Standard library |
| JavaScript | Node.js 18+ | Node.js standard library |
| C++17 | C++17 compiler | Standard library |
| Java | Java 17 | Java standard library |
| PostgreSQL | PostgreSQL 14+ | PostgreSQL server |

The PostgreSQL script creates its own schema, domain types, tables, functions, procedures, triggers, indexes, views, and sample records.

The SQL uses PostgreSQL-specific features including JSONB, PL/pgSQL, identity columns, array operations, and stored procedures.

## 20. Conceptual Knowledge Checklist

- [ ] Understand the difference between input, hidden, and output layers.
- [ ] Understand the role of an individual neuron.
- [ ] Distinguish weights from biases.
- [ ] Understand matrix-vector multiplication.
- [ ] Calculate a weighted sum manually.
- [ ] Apply an activation function.
- [ ] Understand layer-to-layer dimension compatibility.
- [ ] Calculate a complete multilayer forward pass.
- [ ] Distinguish pre-activation values from activated outputs.
- [ ] Understand sigmoid, ReLU, tanh, and softmax.
- [ ] Understand batch inference.
- [ ] Calculate trainable parameter counts.
- [ ] Identify numerical instability.
- [ ] Distinguish forward propagation from backpropagation.
- [ ] Understand why forward propagation alone does not train a model.
- [ ] Understand how inference services manage execution.
- [ ] Understand how neural network parameters can be represented in a relational database.
- [ ] Understand the computational and memory complexity of dense layers.
