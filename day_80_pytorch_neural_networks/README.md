# PyTorch Neural Networks: Modules, Layers, Forward Propagation and Training Loops

## 1. Technical Scope

This project implements the fundamental and advanced mechanisms of feedforward neural networks, with PyTorch serving as the primary framework.

A neural network consists of computational units arranged in layers. Each layer transforms its input into an output. During training, the network adjusts its parameters to reduce a specified loss function.

The project examines five complementary implementation perspectives:

| File | Engineering perspective | Main responsibility |
|---|---|---|
| Python | PyTorch framework | Tensor-based neural network construction and training |
| JavaScript | Event-driven computation | Explicit forward propagation and parameter updates |
| C++17 | Computational engineering | Matrix-based neural network and backpropagation |
| Java 17 | Enterprise architecture | Modular neural network and controlled training lifecycle |
| PostgreSQL | Experiment management | Relational tracking of models, configurations, training runs and metrics |

The implementations use a two-class classification problem with two numerical input features. The dataset is synthetically generated to make the training workflow reproducible.

## 2. Neural Network Fundamentals

### 2.1 Neurons and parameters

A neuron receives input values, multiplies them by weights, adds a bias and applies an activation function.

For a single neuron:

\[
z = \sum_{i=1}^{n} w_i x_i+b
\]

\[
a=f(z)
\]

Where:

- \(x_i\): Input features.
- \(w_i\): Trainable weights.
- \(b\): Trainable bias.
- \(z\): Linear transformation.
- \(f\): Activation function.
- \(a\): Neuron output.

Weights and biases are the principal trainable parameters in a conventional fully connected network.

### 2.2 Layers

A layer groups related computational operations.

| Layer | Function | Trainable parameters |
|---|---|---|
| Linear | Weighted transformation | Yes |
| ReLU | Introduces non-linearity | No |
| Sigmoid | Produces values between zero and one | No |
| Tanh | Produces values between negative one and one | No |
| Batch Normalization | Normalizes intermediate activations | Usually |
| Dropout | Randomly disables activations during training | No |
| Softmax | Converts logits into class probabilities | No |
| Embedding | Maps discrete identifiers to vectors | Yes |
| Convolution | Extracts local spatial patterns | Yes |
| Max Pooling | Reduces spatial dimensions | No |
| Flatten | Converts multidimensional data into a vector | No |

A layer's input and output dimensions must be compatible with the adjacent layers.

For example:

Input dimension: 2

Hidden dimension: 32

Output dimension: 2

The resulting fully connected architecture is:

Input (2)
    |
Linear (2 → 32)
    |
ReLU
    |
Linear (32 → 32)
    |
ReLU
    |
Linear (32 → 2)
    |
Class logits

The Python implementation also incorporates Batch Normalization and Dropout.

### 2.3 Neural network modules

A module is a reusable computational component that can contain parameters, child modules and execution logic.

In PyTorch, `torch.nn.Module` is the foundation of model construction.

A custom module generally defines:

- An initialization method that creates layers.
- A `forward()` method that describes computation.
- Trainable parameters registered with the module.
- Optional state and supporting methods.

PyTorch automatically discovers registered submodules and their parameters.

The distinction between a layer and a module is important. A layer is a particular computational operation, while a module can represent an individual layer, a collection of layers or an entire network.

## 3. Forward Propagation

Forward propagation calculates the network output from an input tensor.

For a network with multiple layers:

\[
a^{(0)}=x
\]

\[
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
\]

\[
a^{(l)}=f^{(l)}(z^{(l)})
\]

The output of one layer becomes the input of the next.

The final classification layer produces logits rather than normalized probabilities.

For two classes, the network produces:

\[
z=[z_0,z_1]
\]

Softmax converts the logits into probabilities:

\[
P(y=i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

The predicted class is the index with the highest probability.

### PyTorch implementation

The Python file uses `nn.Sequential` to organize the computational layers.

The model's `forward()` method delegates computation to the sequential module.

This separates architecture definition from the training process.

The training implementation passes batches of input tensors through the network rather than processing every sample independently.

## 4. Activation Functions

Activation functions introduce non-linearity. Without suitable non-linear transformations, a network composed exclusively of linear layers is mathematically equivalent to a single linear transformation.

### ReLU

\[
f(x)=\max(0,x)
\]

Its derivative is:

\[
f'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq0
\end{cases}
\]

ReLU is computationally inexpensive and commonly used in hidden layers.

### Sigmoid

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Sigmoid is useful for certain binary classification output configurations. It can produce very small gradients when its input has a large positive or negative magnitude.

### Tanh

\[
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]

Tanh produces values in the interval \([-1,1]\).

### GELU

GELU provides a smooth activation function commonly used in transformer architectures.

The current classification implementation uses ReLU because it is appropriate for the selected feedforward architecture.

## 5. Loss Functions

A loss function measures the difference between model predictions and target values.

### Cross-entropy loss

For a classification problem:

\[
L=-\sum_{i=1}^{C}y_i\log(p_i)
\]

Where:

- \(C\): Number of classes.
- \(y_i\): Target indicator.
- \(p_i\): Predicted class probability.

The Python implementation uses `nn.CrossEntropyLoss`.

An important PyTorch distinction is that `CrossEntropyLoss` expects raw logits and integer class labels. Applying Softmax to the logits before passing them to this loss is unnecessary and changes the intended calculation.

For binary classification with two output neurons, the implementation uses two logits and integer targets of zero or one.

Other common losses include:

| Loss | Typical application |
|---|---|
| Cross Entropy | Multiclass classification |
| Binary Cross Entropy | Binary classification |
| Mean Squared Error | Regression |
| Mean Absolute Error | Regression |
| Huber Loss | Regression with reduced sensitivity to large errors |

## 6. Backpropagation and Automatic Differentiation

Backpropagation calculates how the loss changes with respect to network parameters.

The chain rule propagates gradients from the output layer toward the input layer.

For a parameter \(\theta\):

\[
\frac{\partial L}{\partial \theta}
\]

The optimizer uses this gradient to update the parameter.

A basic gradient descent update is:

\[
\theta_{t+1}
=
\theta_t-\eta\frac{\partial L}{\partial\theta}
\]

Where \(\eta\) is the learning rate.

### PyTorch autograd

PyTorch provides automatic differentiation through its autograd engine.

The essential operations are:

| Operation | Purpose |
|---|---|
| `loss.backward()` | Computes gradients |
| `optimizer.zero_grad()` | Clears accumulated gradients |
| `optimizer.step()` | Updates trainable parameters |
| `torch.no_grad()` | Disables gradient tracking |
| `model.parameters()` | Retrieves model parameters |
| `model.state_dict()` | Retrieves model state |

PyTorch normally accumulates gradients in parameter gradient fields. Clearing gradients before each optimization step prevents unintended accumulation across batches.

The Python implementation uses `set_to_none=True` to avoid unnecessary gradient-buffer initialization.

### Manual backpropagation

The JavaScript, C++17 and Java 17 implementations calculate gradients explicitly.

For a dense layer:

\[
Y=WX+b
\]

The gradients are:

\[
\frac{\partial L}{\partial W}
=
\frac{\partial L}{\partial Y}X^T
\]

\[
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial Y}
\]

\[
\frac{\partial L}{\partial X}
=
W^T\frac{\partial L}{\partial Y}
\]

These equations establish the relationship between the forward transformation and backward gradient propagation.

Manual implementations make the computational process explicit but do not provide the extensive automatic differentiation and tensor optimization facilities available in PyTorch.

## 7. Training Loops

A training loop coordinates the repeated process of presenting data, computing predictions, measuring loss, calculating gradients and updating parameters.

The implemented workflow is:

Dataset
   |
Training / Validation Split
   |
Batch Generation
   |
Forward Propagation
   |
Loss Calculation
   |
Backpropagation
   |
Gradient Processing
   |
Optimizer Update
   |
Validation
   |
Metric Recording
   |
Checkpointing

### 7.1 Epochs

An epoch represents one complete pass through the training dataset.

If a dataset contains 1,000 samples and the batch size is 100:

\[
\text{Batches per epoch}=1000/100=10
\]

The network therefore performs ten optimization updates per epoch when all batches are processed.

### 7.2 Batches

A batch is a subset of training samples processed together.

Mini-batch training balances computational efficiency and gradient estimation.

Small batches can produce noisier gradients. Larger batches generally require more memory and can alter optimization behavior.

### 7.3 Training mode and evaluation mode

PyTorch provides two important model states.

| Method | Behavior |
|---|---|
| `model.train()` | Enables training behavior |
| `model.eval()` | Enables evaluation behavior |

These methods affect modules such as Dropout and Batch Normalization.

They do not independently enable or disable gradient computation.

For evaluation and inference, `torch.no_grad()` prevents unnecessary gradient tracking.

### 7.4 Training and validation datasets

The training dataset updates model parameters.

The validation dataset evaluates model behavior without being used for gradient updates.

The separation helps identify overfitting.

The Python implementation uses `random_split` and separate data loaders for the two datasets.

The JavaScript, C++17 and Java 17 programs also separate training and validation records.

## 8. Optimization Algorithms

An optimizer determines how network parameters change after gradients are calculated.

### Stochastic Gradient Descent

\[
\theta_{t+1}
=
\theta_t-\eta g_t
\]

SGD updates parameters using estimated gradients.

### Adam

Adam maintains moving averages of gradients and squared gradients.

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

The bias-corrected estimates are used to calculate adaptive parameter updates.

The Python implementation uses Adam with weight decay.

The PostgreSQL experiment configuration supports SGD, Adam, AdamW and RMSprop.

The manually implemented language examples use gradient descent with mini-batch gradient accumulation rather than claiming to reproduce Adam.

## 9. Regularization and Training Stability

### Dropout

Dropout randomly disables selected activations during training.

This reduces dependence on particular hidden units.

The Python architecture uses a dropout probability of 0.15.

Dropout is disabled automatically when the model enters evaluation mode.

### Weight decay

Weight decay discourages excessively large parameter values.

It is commonly used to improve generalization.

The Python implementation configures weight decay through Adam.

### Gradient clipping

Large gradients can cause unstable parameter updates.

Gradient clipping limits gradient magnitude.

The Python implementation uses:

`torch.nn.utils.clip_grad_norm_`

The JavaScript implementation clips the output gradient before backpropagation. The manual C++17 and Java implementations illustrate gradient accumulation and update scaling, but do not implement mathematically equivalent global parameter-gradient norm clipping.

### Learning-rate scheduling

A learning-rate scheduler modifies the learning rate during training.

The Python implementation uses `ReduceLROnPlateau`.

It reduces the learning rate when validation loss stops improving.

### Early stopping

Early stopping terminates training when validation performance fails to improve over a specified number of epochs.

The Python trainer stores the best model state and restores it after training.

This avoids retaining a later model state that may have worse validation performance.

## 10. Python Implementation: PyTorch Framework

The Python script provides the primary framework-based implementation.

### Architecture

The `NeuralNetwork` class inherits from `nn.Module`.

Its layers are:

- Linear input transformation.
- ReLU activation.
- Batch Normalization.
- Second Linear transformation.
- ReLU activation.
- Dropout.
- Output Linear transformation.

### Training service

The `Trainer` class separates training orchestration from the model definition.

Its responsibilities include:

- Loss function initialization.
- Optimizer initialization.
- Learning-rate scheduling.
- Epoch execution.
- Validation.
- Metric recording.
- Early stopping.
- Best-state restoration.

### Data management

`TensorDataset` and `DataLoader` provide structured dataset access.

The training loader shuffles records, while the validation loader maintains a predictable evaluation order.

### Checkpointing

The Python implementation saves:

- Model parameters.
- Optimizer state.
- Training configuration.
- PyTorch version.

The checkpoint can be restored for further training or inference.

Only load checkpoint files from trusted sources. Loading serialized model files can carry security risks depending on the serialization format and loading configuration.

The implementation uses `weights_only=True` when restoring its checkpoint.

### Limitations

The example is a compact classification experiment rather than a production training platform.

It does not implement distributed training, mixed precision, model serving, experiment scheduling or large-scale data streaming.

## 11. JavaScript Implementation: Event-Driven Neural Computation

The JavaScript file implements a multilayer perceptron using Node.js and standard-library functionality.

It does not mechanically translate the PyTorch script.

Its engineering emphasis is explicit computation and event-driven training.

### Core components

| Component | Responsibility |
|---|---|
| `RandomGenerator` | Deterministic initialization |
| `LinearLayer` | Dense transformation and gradients |
| `ReLULayer` | Non-linear activation |
| `NeuralNetwork` | Sequential module execution |
| `crossEntropyLoss` | Loss and gradient calculation |
| `TrainingEngine` | Training orchestration |
| `EventEmitter` | Training lifecycle notifications |
| `evaluate` | Validation metrics |

### Event-driven execution

The training engine emits events for epoch completion and overall training completion.

This separates the training process from logging and monitoring.

Consumers can subscribe to events without changing the model implementation.

### Asynchronous execution

The trainer yields to the Node.js event loop between epochs using `setImmediate`.

This allows other event-loop work to execute between training iterations.

It does not make the numerical computations parallel or eliminate the CPU cost of training.

### Persistence

The model exports its layer configuration and parameters as JSON.

The checkpoint write uses a temporary file followed by a rename.

This reduces the likelihood of exposing an incomplete checkpoint after an interrupted write.

The example saves checkpoints but does not implement checkpoint restoration.

### Limitations

The JavaScript implementation uses ordinary arrays rather than optimized tensor storage.

It is intended to demonstrate the mathematics and software architecture, not to compete with specialized neural network runtimes.

Its sample-level updates within each batch also differ from the averaged mini-batch updates used by the Python implementation.

## 12. C++17 Implementation: Computational Engineering

The C++17 implementation uses standard-library containers and object-oriented abstractions.

### Design

The abstract `Module` class defines the common interface for network components.

`LinearLayer` implements trainable weights, biases and gradient accumulation.

`ReLULayer` implements the activation operation.

`NeuralNetwork` manages the ordered sequence of modules.

This design separates layer behavior from network orchestration.

### Memory and performance

The implementation uses `std::vector` for dynamic storage.

The matrix representation is a vector of row vectors.

This representation is easy to inspect but does not provide the memory layout and vectorized computation available in dedicated numerical libraries.

For large networks, contiguous matrix storage and optimized matrix multiplication are preferable.

### Gradient accumulation

The dense layers accumulate gradients across the batch before applying an update.

This reduces update frequency compared with the JavaScript implementation.

The implementation is educational and CPU-oriented.

It does not include multithreading, GPU execution, SIMD optimization or automatic differentiation.

The program uses a standard C++17 toolchain and has no external library requirement.

## 13. Java 17 Implementation: Modular Training Architecture

The Java implementation demonstrates a more explicitly structured application design.

### Domain types

The implementation defines:

- `TrainingConfig` as an immutable record.
- `Sample` as an immutable record with defensive copying.
- `TrainingStatus` as an enumeration.
- `EpochMetrics` as an immutable record.
- `TrainingException` as a dedicated runtime exception.

### Service responsibilities

| Component | Responsibility |
|---|---|
| `LinearLayer` | Trainable linear transformation |
| `ActivationLayer` | Activation operations |
| `CrossEntropyLoss` | Classification loss |
| `NeuralNetwork` | Sequential computation |
| `EvaluationService` | Validation |
| `TrainingService` | Training lifecycle |

### Training state management

The training service defines four states:

CREATED
   |
RUNNING
   |
COMPLETED

A failed operation transitions the service to FAILED.

The service prevents repeated execution after completion or failure.

This illustrates explicit lifecycle management in enterprise-oriented software.

### Validation

The implementation checks:

- Network dimensions.
- Learning rate.
- Dataset feature dimensions.
- Target class range.
- Empty datasets.
- Numerical validity.
- Training service state.

The record-based sample representation protects the internal feature array from modification through the public accessor.

### Limitations

The Java implementation does not use a specialized tensor library.

Its array-based matrix operations are intended for relatively small experiments.

It does not implement production checkpoint persistence, optimizer state restoration, distributed training or automatic differentiation.

## 14. PostgreSQL Implementation: Neural Experiment Management

The PostgreSQL script models the operational data surrounding neural network development.

It separates neural computation from experiment metadata management.

### Relational domain model

Researchers
   |
Projects
   |
   +---- Datasets
   |
   +---- Models
            |
            +---- Model Layers
            |
            +---- Training Configurations
                       |
                       +---- Training Runs
                                  |
                                  +---- Epoch Metrics
                                  |
                                  +---- Model Checkpoints

### Main tables

| Table | Purpose |
|---|---|
| `researchers` | Researcher identities and roles |
| `projects` | Experiment ownership |
| `datasets` | Dataset versions and characteristics |
| `models` | Network definitions |
| `model_layers` | Ordered layer configurations |
| `training_configurations` | Optimizers and hyperparameters |
| `training_runs` | Individual training executions |
| `epoch_metrics` | Training and validation measurements |
| `model_checkpoints` | Checkpoint metadata and integrity information |

### Architecture metadata

The `model_layers` table stores:

- Layer sequence.
- Layer type.
- Input dimensions.
- Output dimensions.
- Layer configuration.
- Trainability.

JSONB provides flexibility for layer-specific configuration.

For example, a linear layer can store input and output feature counts, while a dropout layer can store its probability.

### Training state transitions

The `training_runs` table uses a dedicated status enumeration.

The transition trigger prevents invalid state changes.

For example:

- A created run can become running.
- A running run can complete.
- A running run can fail.
- A completed run cannot be restarted.
- A failed run requires a failure reason.

The trigger also maintains start and completion timestamps.

### Epoch metric integrity

The epoch validation trigger checks that:

- The associated run exists.
- The run is currently active.
- The epoch does not exceed its configured limit.
- Epoch numbers are recorded sequentially.

The progress trigger updates the run's current epoch and best validation loss.

### Reporting

The SQL script includes views and queries for:

- Training eligibility.
- Experiment performance.
- Epoch-level metrics.
- Best validation epoch.
- Optimizer comparison.
- Checkpoint verification.
- Failed training runs.
- Data-quality checks.

### Transactions

The demonstration uses transactions for training initiation, metric recording and completion.

The failure example deliberately rolls back its changes.

This demonstrates the difference between a persistent experiment record and a transaction that has not been committed.

### Security

The database example creates a read-only reporting role.

A production system should also consider:

- Restricting write access to training metadata.
- Separating application and administrative accounts.
- Protecting checkpoint storage.
- Validating dataset provenance.
- Auditing configuration changes.
- Avoiding the storage of sensitive training data in unrestricted metadata fields.

The database stores checkpoint locations and hashes, not the actual model tensor contents.

## 15. Important Technical Distinctions

| Concept | Meaning |
|---|---|
| Module | Reusable computational component |
| Layer | Specific transformation or operation |
| Parameter | Trainable model value |
| Forward method | Calculates model outputs |
| Logit | Raw model output before probability normalization |
| Activation | Non-linear transformation |
| Loss | Measures prediction error |
| Gradient | Derivative of loss with respect to a value |
| Backpropagation | Calculates gradients through the network |
| Optimizer | Updates trainable parameters |
| Epoch | One pass through training data |
| Batch | Subset of training records |
| Validation | Measures performance without training updates |
| Checkpoint | Saved model and associated state |
| Inference | Uses a trained model to produce predictions |

A neural network module does not itself imply that the model is training.

Likewise, a successful forward pass does not mean that parameters have been updated.

Parameter updates require gradients and an optimization operation.

## 16. Edge Cases and Failure Conditions

The implementations address several important failure conditions.

| Failure condition | Handling |
|---|---|
| Invalid network dimensions | Configuration validation |
| Incorrect input dimensions | Layer-level validation |
| Empty dataset | Training and evaluation rejection |
| Invalid class label | Loss validation |
| Non-finite inputs | Numerical validation |
| Non-finite loss | Training failure |
| Missing forward pass | Backward-operation rejection |
| Missing gradients | Update validation |
| Invalid training state | Lifecycle enforcement |
| Missing validation data | Training rejection |
| Checkpoint write failure | Temporary-file cleanup |
| Invalid database transition | Trigger exception |
| Duplicate epoch | Unique constraint |
| Invalid metric range | Database check constraint |

These checks are important because numerical and structural errors can otherwise produce incorrect training results without immediately terminating execution.

## 17. Performance Considerations

### Computational complexity

For a fully connected layer with input dimension \(n\) and output dimension \(m\), forward propagation requires approximately:

\[
O(nm)
\]

operations per sample.

For a batch of size \(B\):

\[
O(Bnm)
\]

The parameter count is:

\[
P=nm+m
\]

The additional \(m\) parameters represent biases.

For multiple dense layers, total parameter count is the sum of the parameters in each layer.

### Memory consumption

Training requires storage for:

- Model parameters.
- Intermediate activations.
- Gradients.
- Optimizer state.
- Input batches.

Optimizers such as Adam maintain additional state for their parameter updates.

Memory requirements therefore exceed the storage needed for model weights alone.

### Framework-level optimization

PyTorch supports optimized tensor operations, hardware acceleration and automatic differentiation.

The manual JavaScript, C++17 and Java 17 implementations are intentionally smaller and less optimized.

For large neural networks, using matrix-oriented computational kernels and suitable hardware is essential.

### Database performance

The PostgreSQL implementation uses indexes for:

- Model lookup.
- Training run status.
- Model-to-run relationships.
- Epoch retrieval.
- Validation-loss reporting.
- Checkpoint history.

The unique constraint on `(run_id, epoch_number)` prevents duplicate epoch records.

## 18. Reproducibility and Experimental Reliability

Reproducibility requires more than fixing a random seed.

A reliable experiment should retain:

- Model architecture.
- Dataset identity and version.
- Dataset content hash.
- Random seed.
- Optimizer.
- Learning rate.
- Batch size.
- Epoch count.
- Regularization configuration.
- Execution device.
- Software version.
- Training and validation metrics.
- Checkpoint identity.

The Python script seeds Python and PyTorch random generators.

The PostgreSQL schema stores experiment configuration and dataset fingerprints.

Hardware, software versions, nondeterministic kernels and floating-point differences can still affect exact reproducibility.

## 19. Practical Learning Outcomes

The five implementations establish the relationship between mathematical operations, software abstractions and experiment management.

The Python implementation demonstrates how PyTorch manages modules, parameters, automatic differentiation and training.

The JavaScript implementation makes the computation and event-driven execution explicit.

The C++17 implementation demonstrates memory-oriented computational structures and module interfaces.

The Java 17 implementation illustrates domain modeling, service separation and controlled training states.

The PostgreSQL implementation demonstrates how architectures, datasets, training configurations, metrics and checkpoints can be managed as related experimental records.

Together, these components provide a technically grounded treatment of neural network construction, forward propagation, backpropagation, optimization, validation and training lifecycle management.
