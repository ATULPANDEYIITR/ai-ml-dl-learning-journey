# Batch Normalization: Internal Normalization and Training vs Inference

## 1. Technical Scope

Batch Normalization is a neural-network normalization technique that standardizes intermediate activations using statistics calculated across a mini-batch during training.

It transforms the distribution of feature activations and introduces learnable affine parameters that allow the network to retain useful representational flexibility.

The implementation covers:

- Internal feature normalization.
- Mini-batch mean and variance.
- Numerical stabilization.
- Learnable scale and shift.
- Running statistics.
- Training and inference behavior.
- Backpropagation.
- Model-state persistence.
- Execution validation.
- Statistical reporting and database integrity.

The five implementations provide complementary computational and engineering perspectives.

## 2. Mathematical Foundation

Consider an activation matrix:

\[
X \in \mathbb{R}^{N \times D}
\]

Where:

- \(N\) is the mini-batch size.
- \(D\) is the number of features.
- \(x_{ij}\) is the activation of sample \(i\), feature \(j\).

Batch Normalization independently processes each feature across the batch dimension.

### 2.1 Mini-batch mean

For feature \(j\):

\[
\mu_j =
\frac{1}{N}
\sum_{i=1}^{N}x_{ij}
\]

The mean represents the average activation of that feature in the current mini-batch.

### 2.2 Mini-batch variance

The population variance is:

\[
\sigma_j^2 =
\frac{1}{N}
\sum_{i=1}^{N}
(x_{ij}-\mu_j)^2
\]

Population variance divides by \(N\), rather than \(N-1\).

This is the convention used throughout the five implementations for current mini-batch normalization.

### 2.3 Numerical stabilization

The normalized activation is:

\[
\hat{x}_{ij} =
\frac{x_{ij}-\mu_j}
{\sqrt{\sigma_j^2+\epsilon}}
\]

The parameter \(\epsilon\) is a small positive constant.

It prevents division by zero and reduces numerical instability when feature variance is extremely small.

A constant feature has zero variance. Epsilon allows its normalization to remain mathematically defined.

### 2.4 Learnable affine transformation

Batch Normalization introduces two trainable parameters per feature:

- Gamma (\(\gamma\)): scale.
- Beta (\(\beta\)): shift.

The final output is:

\[
y_{ij} =
\gamma_j\hat{x}_{ij}+\beta_j
\]

Initially:

\[
\gamma_j=1
\]

\[
\beta_j=0
\]

These initial values preserve the normalized representation.

During optimization, gamma and beta are updated through backpropagation.

The affine transformation means that normalization does not permanently restrict the network to zero-mean, unit-variance activations.

## 3. Internal Normalization

Internal normalization operates on intermediate neural-network activations rather than exclusively on raw input features.

A typical dense neural-network layer can be represented as:

\[
Z = XW+b
\]

Where:

- \(X\) is the incoming activation matrix.
- \(W\) is the weight matrix.
- \(b\) is the bias vector.
- \(Z\) contains the resulting pre-activation values.

Batch Normalization can be applied to these intermediate values:

\[
Z \rightarrow BN(Z) \rightarrow f(BN(Z))
\]

Here, \(f\) is the subsequent activation function.

For a convolutional network, normalization is generally performed independently for each channel using the batch and spatial dimensions, subject to the framework's tensor-layout conventions.

The implementations in this repository use two-dimensional matrices and feature-wise normalization. They do not implement convolutional tensor layouts.

### 3.1 Internal normalization and optimization

Normalization changes the numerical scale and distribution of activations presented to subsequent operations.

Its practical effects can include:

- More controlled activation magnitudes.
- Improved numerical conditioning in some networks.
- Reduced sensitivity to certain initialization choices.
- More stable optimization behavior.
- Support for different learning-rate choices.

Batch Normalization does not guarantee faster convergence or eliminate every optimization problem.

Its effectiveness depends on architecture, batch composition, training procedure, and deployment conditions.

## 4. Training Behavior

Training mode uses the current mini-batch statistics.

For each incoming batch:

1. Validate the matrix dimensions.
2. Calculate the feature-wise mean.
3. Calculate the feature-wise population variance.
4. Normalize each activation.
5. Apply gamma and beta.
6. Update running statistics.
7. Preserve intermediate values for backpropagation.

The output for a training batch is determined by that batch's statistics.

Two identical samples can consequently produce different normalized values when evaluated in different training batches.

This is an important distinction between training and inference.

### 4.1 Running mean

Running mean estimates the activation mean over successive training batches.

The implementations use exponential moving averaging:

\[
\mu_{\text{running}}^{(t)}
=
m\mu_{\text{running}}^{(t-1)}
+
(1-m)\mu_{\text{batch}}^{(t)}
\]

Where \(m\) represents momentum.

A momentum value of 0.9 assigns 90% weight to the previous running estimate and 10% to the latest batch statistic.

### 4.2 Running variance

Running variance follows the corresponding update:

\[
v_{\text{running}}^{(t)}
=
mv_{\text{running}}^{(t-1)}
+
(1-m)v_{\text{batch}}^{(t)}
\]

The running variance is used during inference.

An implementation may instead use a bias-corrected estimator or framework-specific variance conventions. Consequently, running statistics from different frameworks should not automatically be assumed interchangeable.

### 4.3 Momentum interpretation

Momentum conventions differ between implementations.

This repository consistently uses:

\[
\text{new} =
m \times \text{old}
+
(1-m) \times \text{current}
\]

A momentum of 0.9 therefore means that the current batch contributes 10% to the updated estimate.

The numerical value must be interpreted according to the actual update equation rather than relying exclusively on a parameter name.

## 5. Inference Behavior

Inference mode uses the stored running mean and variance.

The transformation becomes:

\[
y_{ij} =
\gamma_j
\frac{x_{ij}-\mu_{\text{running},j}}
{\sqrt{v_{\text{running},j}+\epsilon}}
+
\beta_j
\]

Unlike training mode, inference does not calculate normalization statistics from the current request batch.

It also does not update running statistics.

This distinction makes inference independent of the composition of the current prediction batch.

### 5.1 Why running statistics matter

Consider a model trained using batches containing thousands of observations.

At deployment, the model may receive one observation at a time.

Calculating a mean and variance from a single observation would produce zero population variance and a fundamentally different transformation from the one used during training.

Stored running statistics provide a consistent reference distribution.

### 5.2 Training versus inference

| Property | Training | Inference |
|---|---|---|
| Mean | Current batch | Running mean |
| Variance | Current batch | Running variance |
| Normalization | Batch-dependent | Stored-state dependent |
| Gamma | Learned parameter | Learned parameter |
| Beta | Learned parameter | Learned parameter |
| Running mean | Updated | Unchanged |
| Running variance | Updated | Unchanged |
| Backpropagation | Supported | Not normally performed |
| Output | Depends on batch composition | Independent of prediction batch composition |

An important operational rule is that a model should not enter inference before its running statistics have been initialized appropriately.

The Python, JavaScript, C++17, and Java 17 implementations explicitly enforce this condition.

## 6. Backpropagation

Batch Normalization is differentiable and participates in gradient-based optimization.

Let the upstream gradient be:

\[
dY = \frac{\partial L}{\partial Y}
\]

Where \(L\) is the loss function.

### 6.1 Gradient for beta

\[
\frac{\partial L}{\partial \beta_j}
=
\sum_{i=1}^{N}dY_{ij}
\]

### 6.2 Gradient for gamma

\[
\frac{\partial L}{\partial \gamma_j}
=
\sum_{i=1}^{N}
dY_{ij}\hat{x}_{ij}
\]

### 6.3 Gradient for normalized input

The gradient with respect to the input activation is:

\[
\frac{\partial L}{\partial x_{ij}}
=
\frac{\gamma_j}
{N\sqrt{\sigma_j^2+\epsilon}}
\left[
NdY_{ij}
-
\sum_{k=1}^{N}dY_{kj}
-
\hat{x}_{ij}
\sum_{k=1}^{N}
dY_{kj}\hat{x}_{kj}
\right]
\]

This expression accounts for the dependence of the batch mean and variance on all samples in the batch.

Consequently, the gradient for one sample is not independent of the other samples.

The implementations calculate:

- Input gradients.
- Gamma gradients.
- Beta gradients.

The affine parameters are updated using gradient descent.

The examples use an explicitly supplied upstream gradient. They do not implement a complete neural-network loss function, optimizer, or dense-layer gradient pipeline.

## 7. Python Implementation

File: `batch_normalization.py`

The Python implementation provides a complete feature-wise normalization component using standard-library functionality.

### 7.1 Architecture

| Component | Responsibility |
|---|---|
| `Mode` | Separates training and inference |
| `BatchStatistics` | Stores current batch statistics |
| `ForwardCache` | Preserves intermediate forward-pass values |
| `BatchNormalization` | Manages normalization and parameter state |
| `calculate_statistics()` | Computes feature-wise population statistics |
| `forward()` | Executes normalization |
| `backward()` | Calculates gradients |
| `update_parameters()` | Updates gamma and beta |
| `save()` | Persists model state |
| `load()` | Restores model state |

### 7.2 Data structures

Python lists represent the activation matrix and feature vectors.

Dataclasses provide explicit structures for statistics and cached forward-pass information.

Enums prevent accidental mixing of execution-mode strings and mode values.

Exceptions distinguish shape errors from numerical errors.

### 7.3 Persistence

The JSON state contains:

- Feature count.
- Epsilon.
- Momentum.
- Gamma.
- Beta.
- Running mean.
- Running variance.
- Number of training batches.

The inference state can therefore be restored without repeating the original training batches.

Production deployments should additionally validate model provenance, version compatibility, and file authenticity.

## 8. JavaScript Implementation

File: `batch_normalization.js`

The JavaScript implementation emphasizes event-driven processing and asynchronous inference.

### 8.1 Event-driven architecture

`NormalizationEngine` extends Node.js `EventEmitter`.

It emits events when:

- Running statistics are updated.
- A forward pass completes.
- Affine parameters change.
- Model state is saved.

These events provide integration points for monitoring and application-level processing.

### 8.2 Private state

JavaScript private class fields protect internal parameters.

External callers cannot directly modify:

- Running mean.
- Running variance.
- Gamma.
- Beta.
- Feature count.
- Momentum.
- Epsilon.

A read-only state snapshot provides access to diagnostic information.

### 8.3 Online statistical calculation

The JavaScript implementation uses Welford's online variance algorithm.

It updates the mean and second central moment incrementally rather than constructing separate intermediate arrays.

For observations \(x_1,\ldots,x_N\):

\[
\delta_i=x_i-\mu_{i-1}
\]

\[
\mu_i=\mu_{i-1}+\frac{\delta_i}{i}
\]

\[
M_{2,i}=M_{2,i-1}+\delta_i(x_i-\mu_i)
\]

The population variance is:

\[
\sigma^2=\frac{M_2}{N}
\]

This approach improves numerical stability compared with directly subtracting large squared quantities.

### 8.4 Asynchronous inference

`inferAsync()` serializes inference requests using a Promise chain.

It provides deterministic processing order within the engine.

The queue does not provide parallel execution or distributed request coordination. A production inference service would require additional concurrency controls, request limits, and workload isolation.

### 8.5 Persistence

The implementation stores model configuration and parameters in JSON.

The file is written with restricted permissions where supported by the operating system.

Production systems should use authenticated model artifacts and protected storage rather than trusting arbitrary JSON input.

## 9. C++17 Implementation

File: `batch_normalization.cpp`

The C++ implementation focuses on explicit numerical operations, domain modeling, and memory-efficient processing.

### 9.1 Domain structures

| Structure | Purpose |
|---|---|
| `Statistics` | Mean, variance and sample count |
| `ForwardCache` | Intermediate values required for backpropagation |
| `Gradients` | Input, gamma and beta gradients |
| `BatchNormalization` | Complete normalization engine |
| `Mode` | Training and inference state |

### 9.2 Numerical computation

The implementation uses:

- `std::vector` for matrices and feature arrays.
- `std::sqrt` for standard deviation.
- `std::isfinite` for numerical validation.
- `std::max` to prevent tiny negative variance caused by floating-point effects.
- Exceptions for invalid input and failed operations.

Welford's algorithm is used to calculate statistics incrementally.

### 9.3 State management

The running-statistic update uses temporary vectors before committing changes.

This reduces the possibility of partially updated model state when a calculation fails.

Parameter updates follow the same principle.

The implementation also verifies that inference leaves running statistics unchanged.

### 9.4 Performance considerations

For \(N\) samples and \(D\) features:

| Operation | Time complexity |
|---|---|
| Mean calculation | \(O(ND)\) |
| Variance calculation | \(O(ND)\) |
| Normalization | \(O(ND)\) |
| Backpropagation | \(O(ND)\) |
| Parameter update | \(O(D)\) |

The implementation uses \(O(ND+D)\) memory for a forward pass with its cache.

The matrix-based implementation is suitable for demonstrating the numerical algorithm. High-throughput production workloads would benefit from contiguous tensor storage, SIMD operations, parallel execution, and accelerator-based computation.

## 10. Java 17 Enterprise Implementation

File: `BatchNormalizationEnterprise.java`

The Java implementation uses explicit domain types and separates policy validation, statistical computation, model state, and normalization services.

### 10.1 Domain architecture

The primary components are:

- `Configuration`
- `Statistics`
- `ForwardResult`
- `GradientResult`
- `ModelState`
- `StatisticsService`
- `NormalizationPolicy`
- `NormalizationService`

Java records provide immutable configuration and result structures.

The model state contains mutable training parameters and running statistics.

### 10.2 Model lifecycle

The implementation defines three states:

| State | Meaning |
|---|---|
| `UNINITIALIZED` | Running statistics have not been established |
| `TRAINING_READY` | At least one training batch has been processed |
| `INFERENCE_READY` | The model has executed inference using stored statistics |

A retired model state is also defined to represent a model that should no longer process requests.

The service enforces the inference initialization requirement.

### 10.3 Policy separation

`NormalizationPolicy` validates incoming batches and learning rates.

`StatisticsService` calculates population statistics.

`NormalizationService` coordinates forward execution, gradient calculation, and parameter updates.

This separation allows enterprise applications to evolve validation and statistical computation without concentrating all responsibilities in one class.

### 10.4 Immutability and state safety

Execution results are copied into immutable lists.

This prevents callers from modifying cached normalized activations or returned outputs.

Parameter updates are calculated into temporary collections before being committed.

For production environments, the model state should also support versioned snapshots and synchronized access when shared across concurrent training operations.

## 11. PostgreSQL Implementation

File: `batch_normalization.sql`

The database implementation models the operational and statistical state of normalization experiments.

It does not replace the numerical tensor computation performed by the application implementations.

Instead, it stores configuration, training statistics, running statistics, affine parameters, execution history, and audit information.

### 11.1 Relational model

The primary relationships are:

`Experiments`

→ `Normalization Models`

→ `Model Versions`

→ `Batches`

→ `Feature Statistics`

`Normalization Models`

→ `Running Statistics`

`Normalization Models`

→ `Affine Parameters`

`Batches`

→ `Normalization Audit`

### 11.2 Table responsibilities

| Table | Stored information |
|---|---|
| `experiments` | Experiment metadata |
| `normalization_models` | Feature count, epsilon, momentum and lifecycle |
| `model_versions` | Versioned model definitions |
| `batches` | Training and inference executions |
| `feature_statistics` | Current batch mean and variance |
| `running_statistics` | Persistent inference statistics |
| `affine_parameters` | Gamma and beta |
| `normalization_audit` | Execution mode and statistics provenance |

### 11.3 Database integrity

Primary keys identify records.

Foreign keys preserve relationships between models, versions, batches, and statistics.

Check constraints enforce valid feature dimensions, epsilon, momentum, variance, and parameter values.

Indexes support:

- Model execution history.
- Execution-mode filtering.
- Status-based retrieval.

Triggers initialize feature-level running statistics and affine parameters whenever a model is created.

Additional triggers reject feature statistics outside the model's configured feature dimensions.

### 11.4 Training transaction

The `process_training_batch()` function performs several operations:

1. Locks the batch record.
2. Validates its execution mode and status.
3. Locks the associated model.
4. Checks the number of feature statistics.
5. Verifies that all feature indexes exist.
6. Updates running means.
7. Updates running variances.
8. Marks the batch as processed.
9. Updates model lifecycle state.
10. Creates an audit record.

The function executes inside the caller's PostgreSQL transaction.

A failed validation raises an exception and prevents partial transactional updates from being committed.

### 11.5 Inference transaction

The `register_inference()` function validates an inference request and records its use of stored statistics.

It does not modify running mean or running variance.

The distinction is preserved in the audit table through `statistics_source`.

### 11.6 Analytical queries

The SQL script includes:

- Model configuration retrieval.
- Inference-statistic retrieval.
- Aggregated batch statistics.
- Comparison between current batch and running statistics.
- Model lifecycle reporting.
- Execution history.
- Missing-feature-state validation.
- Training and inference statistics-source validation.
- Transactional affine parameter updates.
- Controlled handling of invalid feature indexes.

The reporting views make model state and execution history available without duplicating the underlying statistical records.

## 12. Edge Cases and Failure Handling

### 12.1 Empty mini-batches

An empty batch has no valid mean or variance.

All four application implementations reject empty batches.

### 12.2 Single-sample batches

For a batch containing one sample:

\[
\mu=x
\]

\[
\sigma^2=0
\]

Therefore:

\[
\hat{x}=0
\]

The output becomes beta.

Although mathematically defined, single-sample training batches can provide poor statistical estimates and weak normalization behavior.

### 12.3 Constant features

When all observations of a feature have the same value:

\[
\sigma^2=0
\]

Epsilon prevents division by zero.

The normalized activation becomes zero, subject to floating-point precision.

### 12.4 Non-finite values

NaN and infinite values can corrupt means, variances, gradients, and running statistics.

The application implementations reject non-finite input values and invalid parameter updates.

### 12.5 Incorrect feature dimensions

Every sample must contain exactly the configured number of features.

Dimension mismatches are rejected before normalization.

### 12.6 Inference before training

Inference without valid running statistics is rejected.

Initializing running variance to one is a convenient starting state, not proof that a model has been adequately calibrated.

### 12.7 Inference batch composition

Inference statistics must remain independent of the current prediction batch.

The implementations explicitly test that inference does not modify running statistics.

### 12.8 Distribution shift

If the deployment distribution changes significantly from the training distribution, stored running statistics may become less representative.

Potential responses include controlled recalibration, retraining, or model replacement.

Updating statistics indiscriminately from production inference traffic can introduce undesirable feedback and data-quality problems.

## 13. Numerical and Performance Considerations

### 13.1 Floating-point precision

Mean, variance, and gradient computations are sensitive to numerical precision.

Large activation magnitudes combined with small differences can cause cancellation.

Welford's algorithm reduces this risk during statistical accumulation.

For mixed-precision neural-network training, normalization statistics are commonly accumulated using higher precision than the activation storage format.

### 13.2 Epsilon selection

An excessively small epsilon may not adequately stabilize low-variance features.

An excessively large epsilon changes the normalization scale.

The appropriate value depends on the numerical precision, activation distribution, and architecture.

### 13.3 Mini-batch size

Small batches produce noisier estimates of activation distributions.

Larger batches generally provide more stable estimates, although they increase memory requirements and may not always be practical.

### 13.4 Running-statistic momentum

Higher momentum places more weight on historical statistics.

Lower momentum makes running statistics respond more rapidly to recent batches.

Momentum must be selected with consideration for batch ordering, training duration, and distribution changes.

### 13.5 Distributed training

When training across multiple devices, ordinary Batch Normalization computes statistics independently on each device unless synchronization is implemented.

Synchronized Batch Normalization aggregates statistics across participating devices.

This can improve consistency but introduces communication overhead.

The implementations in this repository do not implement distributed synchronization.

## 14. Security and Production Considerations

Normalization statistics are part of model state and should be treated as model artifacts.

Important controls include:

- Restricting access to serialized parameters.
- Validating imported model configurations.
- Protecting running statistics against unauthorized modification.
- Recording model versions.
- Auditing training and inference executions.
- Preventing malformed batches from corrupting model state.
- Separating training privileges from inference privileges.
- Protecting sensitive input data.
- Monitoring activation distributions.
- Validating model compatibility before deployment.

The PostgreSQL implementation provides relational integrity and execution auditing.

It does not implement authentication, row-level security, encrypted artifact storage, or a complete production authorization system.

The Python and JavaScript persistence examples demonstrate basic serialization rather than a secure model registry.

## 15. Implementation Comparison

| Dimension | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Primary focus | Numerical algorithm | Event-driven processing | Numerical efficiency | Enterprise services | Persistent model state |
| Statistics | Batch population | Welford | Welford | Welford | Stored statistics |
| Training mode | Yes | Yes | Yes | Yes | Transactional workflow |
| Inference mode | Yes | Yes | Yes | Yes | Stored-state retrieval |
| Backpropagation | Yes | Yes | Yes | Yes | Not implemented |
| Gamma and beta | Yes | Yes | Yes | Yes | Yes |
| Running statistics | Yes | Yes | Yes | Yes | Yes |
| Persistence | JSON | JSON | Runtime state | Runtime state | Relational tables |
| Validation | Exceptions | Custom errors | Exceptions | Policy classes | Constraints and triggers |

## 16. Execution

### Python

File: `batch_normalization.py`

Run using Python 3.10 or later:

`python batch_normalization.py`

The script demonstrates training, gradient calculation, parameter updates, inference, persistence, and validation.

### JavaScript

File: `batch_normalization.js`

Run using Node.js 18 or later:

`node batch_normalization.js`

The script demonstrates event notifications, asynchronous inference, model persistence, and validation.

### C++17

File: `batch_normalization.cpp`

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic batch_normalization.cpp -o batch_normalization`

Run:

`./batch_normalization`

On Windows:

`batch_normalization.exe`

### Java 17

File: `BatchNormalizationEnterprise.java`

Compile:

`javac BatchNormalizationEnterprise.java`

Run:

`java BatchNormalizationEnterprise`

### PostgreSQL

File: `batch_normalization.sql`

Execute using:

`psql -U postgres -d your_database -f batch_normalization.sql`

The executing database role needs permission to create the schema, tables, types, functions, triggers, and views.

The script creates the `batchnorm` schema and inserts demonstration records.

It is designed as an executable learning database script rather than a repeatable migration. Execute it in a fresh database or remove the existing schema before running it again.

## 17. Technical Limitations

The implementations intentionally use two-dimensional feature matrices.

They do not implement:

- Three-dimensional convolutional activation layouts.
- Distributed synchronized Batch Normalization.
- GPU acceleration.
- Mixed-precision training.
- Automatic differentiation.
- Full neural-network optimization.
- Distributed model-state synchronization.
- Bias-corrected running-statistic estimation.
- Production model serving infrastructure.
- Automated drift detection and recalibration.

The implementations also use population variance consistently for current batches. Framework-specific running-variance updates can differ.

These distinctions matter when reproducing an existing model or transferring normalization parameters between frameworks.

The mathematical normalization, affine transformation, training/inference separation, gradient calculations, and running-statistic behavior form the central mechanisms implemented across the repository.
