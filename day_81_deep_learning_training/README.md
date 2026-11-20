# Deep Learning Training: Epochs, Batches, Learning Rate, and Checkpoints

## 1. Technical Scope

Deep learning training is an iterative optimization process in which a model learns parameters by minimizing a specified loss function over a training dataset.

The implementation demonstrates four fundamental training mechanisms:

- Epochs: complete passes through the training dataset.
- Batches: subsets of training samples used for individual optimization updates.
- Learning rate: the step size used to update model parameters.
- Checkpoints: persisted model and training states that support recovery and evaluation.

The five implementations use a linear regression model trained through mini-batch stochastic gradient descent. This provides an executable optimization example without requiring external numerical libraries or GPU hardware.

The database implementation provides a separate enterprise-oriented model for recording training experiments, execution history, metrics, and checkpoints.

## 2. Core Training Concepts

### Epochs

An epoch represents one complete pass through the training dataset.

If a dataset contains 1,000 training samples, completing one epoch means processing all 1,000 samples.

For a training configuration with 40 epochs, the model can process the training dataset up to 40 times, unless early stopping or another termination condition ends execution.

Epochs provide a useful unit for:

- Monitoring optimization progress.
- Comparing training and validation losses.
- Scheduling learning-rate changes.
- Determining checkpoint intervals.
- Detecting overfitting.

An epoch does not necessarily represent one parameter update. The number of updates depends on the batch size.

### Batches

A batch is a subset of training samples processed together.

The number of batches in one epoch is:

\[
B = \left\lceil \frac{N}{S} \right\rceil
\]

Where:

- \(B\) = number of batches.
- \(N\) = number of training samples.
- \(S\) = batch size.

For 700 training samples and a batch size of 32:

\[
B = \left\lceil \frac{700}{32} \right\rceil = 22
\]

The final batch contains 28 samples.

The implementations retain this incomplete final batch rather than discarding its samples.

Three common batching strategies are:

| Strategy | Description | Main characteristic |
|---|---|---|
| Batch gradient descent | Processes the entire training dataset for each update | Stable full-dataset gradient |
| Stochastic gradient descent | Processes one sample per update | Frequent, noisy updates |
| Mini-batch gradient descent | Processes a subset of samples per update | Balance between computational efficiency and gradient stability |

Mini-batch training is the primary method implemented in these programs.

### Learning Rate

The learning rate controls the magnitude of parameter updates during optimization.

For a parameter \(\theta\), a gradient descent update is:

\[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
\]

Where:

- \(\theta_t\) = current parameter.
- \(\eta\) = learning rate.
- \(\nabla L\) = gradient of the loss function.

A learning rate that is too large can cause unstable optimization or divergence.

A learning rate that is too small can make convergence unnecessarily slow.

The implementations use exponential learning-rate decay:

\[
\eta_e = \max(\eta_{\min}, \eta_0 \gamma^{e-1})
\]

Where:

- \(\eta_e\) = learning rate for epoch \(e\).
- \(\eta_0\) = initial learning rate.
- \(\gamma\) = decay factor.
- \(\eta_{\min}\) = minimum learning rate.

For example, an initial learning rate of 0.05 and a decay factor of 0.98 gradually reduce the optimization step size.

The minimum learning-rate constraint prevents the scheduled value from decreasing indefinitely.

### Checkpoints

A checkpoint is a persisted representation of a model and its associated training state.

A useful checkpoint can contain:

- Model weights.
- Model bias.
- Completed epoch.
- Current learning rate.
- Best validation loss.
- Best epoch.
- Model or checkpoint format version.

Checkpoints support recovery after:

- Process termination.
- Hardware failure.
- Interrupted training.
- Resource exhaustion.
- Infrastructure restarts.

A checkpoint is not necessarily the final model. The checkpoint with the lowest validation loss may be preferable to the most recently saved checkpoint.

The implementations distinguish periodic checkpointing from saving a model when validation performance improves.

## 3. Training Workflow

The general execution sequence is:

**Dataset preparation**

- Validate feature dimensions.
- Validate numerical values.
- Separate training and validation datasets.
- Initialize model parameters.

**Training initialization**

- Configure epoch count.
- Configure batch size.
- Set initial learning rate.
- Select learning-rate scheduling policy.
- Configure checkpoint intervals.
- Configure early stopping.

**Optimization**

- Shuffle training samples.
- Divide samples into mini-batches.
- Calculate predictions.
- Calculate loss.
- Calculate gradients.
- Update model parameters.

**Evaluation**

- Calculate training loss.
- Calculate validation loss.
- Record epoch metrics.
- Compare validation performance with the best previous result.

**Persistence and termination**

- Save periodic checkpoints.
- Preserve improved model states.
- Apply early stopping.
- Restore the best model.
- Record the final training state.

Plain-text representation:

    Dataset
       |
       v
    Data Validation
       |
       v
    Training / Validation Split
       |
       v
    Model Initialization
       |
       v
    Epoch Loop
       |
       +---- Shuffle Training Samples
       |
       +---- Mini-Batch Creation
       |
       +---- Forward Prediction
       |
       +---- Loss Calculation
       |
       +---- Gradient Calculation
       |
       +---- Parameter Updates
       |
       v
    Validation Evaluation
       |
       +---- Improved Performance?
       |          |
       |          +---- Yes: Save Best Model
       |          |
       |          +---- No: Update Patience Counter
       |
       +---- Periodic Checkpoint
       |
       v
    Continue or Stop
       |
       v
    Restore Best Model
       |
       v
    Training Report

## 4. Optimization Mathematics

### Linear Model

The executable examples use a linear prediction model:

\[
\hat{y} = \sum_{j=1}^{d} w_jx_j+b
\]

Where:

- \(x_j\) = input feature.
- \(w_j\) = trainable weight.
- \(b\) = bias.
- \(d\) = number of input features.
- \(\hat{y}\) = predicted output.

### Mean Squared Error

The loss function is mean squared error:

\[
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
\]

Where \(y_i\) is the actual target.

For a mini-batch containing \(S\) samples, the weight gradient is:

\[
\frac{\partial L}{\partial w_j}
=
\frac{2}{S}
\sum_{i=1}^{S}
(\hat{y}_i-y_i)x_{ij}
\]

The bias gradient is:

\[
\frac{\partial L}{\partial b}
=
\frac{2}{S}
\sum_{i=1}^{S}(\hat{y}_i-y_i)
\]

The model updates its weights and bias after calculating the gradients for the entire mini-batch.

This implementation uses full-precision floating-point arithmetic through the languages' standard numeric types.

It does not implement neural-network layers, automatic differentiation, convolution, attention, or GPU execution.

## 5. Training and Validation

Training loss measures model performance on samples used for parameter optimization.

Validation loss measures performance on a separate dataset that is not used for gradient updates.

These measurements serve different purposes.

| Measurement | Purpose |
|---|---|
| Training loss | Measures optimization performance on training samples |
| Validation loss | Measures generalization performance on validation samples |
| Test loss | Provides final evaluation on an independent test dataset |

A decreasing training loss does not guarantee improving validation performance.

A substantial difference between training and validation loss can indicate overfitting.

The implementations calculate both training and validation losses after each epoch.

The test partition is represented in the PostgreSQL dataset model but is not used by the training algorithms.

A production evaluation process should preserve test data until model selection and training decisions are complete.

## 6. Early Stopping

Early stopping terminates training when validation performance stops improving for a specified number of consecutive epochs.

The implementation maintains:

- Best validation loss.
- Best epoch.
- Number of consecutive non-improving epochs.
- A copy of the best model parameters.

The improvement rule is:

\[
L_{\text{current}} <
L_{\text{best}}-\delta
\]

Where \(\delta\) is the minimum improvement threshold.

If the condition is satisfied, the best model is updated and the patience counter is reset.

Otherwise, the counter increases.

Training stops when:

\[
\text{Stale Epochs} \geq \text{Patience}
\]

Early stopping helps reduce unnecessary computation and can limit overfitting.

The best model is restored after training rather than assuming the parameters from the last epoch are optimal.

## 7. Checkpoint Design and Recovery

### Checkpoint Contents

The implementations persist:

| Field | Purpose |
|---|---|
| Format version | Identifies checkpoint structure |
| Epoch | Identifies completed training progress |
| Weights | Stores learned model parameters |
| Bias | Stores the model intercept |
| Learning rate | Records optimization configuration |
| Best validation loss | Preserves model-selection information |
| Best epoch | Identifies the best recorded validation result |

### Atomic Persistence

Writing a checkpoint directly to its final destination risks leaving a partially written file if the process fails.

The implementations use a temporary file followed by replacement of the destination.

The persistence sequence is:

    Create Temporary File
           |
           v
    Write Checkpoint Data
           |
           v
    Flush and Close
           |
           v
    Replace Destination
           |
           v
    Checkpoint Available

Python additionally flushes file data using `fsync`.

Java uses an atomic move when the filesystem supports it and falls back to a regular replacement.

C++ uses a temporary file and filesystem rename.

JavaScript uses an exclusive temporary file and filesystem rename.

Atomic replacement reduces the risk of exposing incomplete checkpoint files. It does not by itself guarantee durable storage after sudden hardware or filesystem failure.

### Recovery

Recovery restores the model parameters and training metadata from a validated checkpoint.

The implementations verify model dimensions and numerical values before accepting restored state.

A checkpoint containing incompatible dimensions or invalid numerical values must not be used.

A production recovery mechanism should also validate a cryptographic checksum, checkpoint provenance, optimizer state, random-number generator state, and data-loader progress.

The current educational implementations persist model parameters and selected progress metadata, not complete optimizer and execution state.

Consequently, resuming training is not guaranteed to reproduce an uninterrupted run exactly.

## 8. Python Implementation

The Python program uses a modular training architecture.

| Component | Responsibility |
|---|---|
| `TrainingConfig` | Stores and validates training parameters |
| `Dataset` | Validates input features and targets |
| `Model` | Maintains weights, bias, and predictions |
| `LearningRateScheduler` | Calculates epoch-specific learning rates |
| `CheckpointManager` | Saves and restores model state |
| `Trainer` | Controls optimization and evaluation |
| `TrainingMetrics` | Stores epoch-level measurements |

Python-specific implementation decisions include:

- Dataclasses for configuration and metric records.
- Enumerations for training states.
- Lists for features, targets, and gradients.
- Exceptions for configuration and persistence failures.
- JSON checkpoint serialization.
- Temporary-file replacement for checkpoint persistence.
- Assertions for basic validation tests.
- A command-line interface for training and recovery.

The batch gradient implementation accumulates gradients before updating model parameters. This ensures that all samples in a batch contribute to the same parameter-update calculation.

Execution:

- `python deep_learning_training.py`
- `python deep_learning_training.py --tests`
- `python deep_learning_training.py --epochs 60 --batch-size 32`
- `python deep_learning_training.py --resume checkpoints/epoch_00005.json`

The checkpoint supplied for recovery must belong to the configured checkpoint directory.

## 9. JavaScript Implementation

The JavaScript program uses an event-driven architecture suitable for monitoring asynchronous training workflows.

Its main components are:

| Component | Responsibility |
|---|---|
| `Dataset` | Validates structured training data |
| `LinearModel` | Manages model parameters |
| `LearningRateScheduler` | Calculates scheduled learning rates |
| `CheckpointStore` | Performs asynchronous persistence |
| `TrainingEngine` | Coordinates training execution |
| `EventEmitter` | Publishes training lifecycle events |

Important events include:

- `epochCompleted`
- `checkpointSaved`
- `trainingCompleted`
- `trainingFailed`
- `cancellationRequested`
- `restored`

The event-driven approach separates optimization from monitoring and reporting.

An external monitoring component can subscribe to epoch events without changing gradient calculations.

The engine also supports cancellation between batches.

The cancellation operation does not forcibly interrupt a gradient calculation already in progress.

The deterministic synthetic dataset generator provides reproducible input data. The shuffle mechanism uses the JavaScript random-number generator, so complete training reproducibility requires controlling shuffle state as well.

Execution:

- `node deep_learning_training.js`

No third-party npm dependencies are required.

## 10. C++17 Implementation

The C++ program emphasizes explicit domain modeling, efficient data access, and controlled state management.

| Component | Responsibility |
|---|---|
| `TrainingConfig` | Defines training constraints |
| `Sample` | Represents one feature-target observation |
| `Dataset` | Owns and validates training samples |
| `LinearModel` | Manages parameters and updates |
| `Scheduler` | Implements learning-rate decay |
| `CheckpointManager` | Handles checkpoint persistence |
| `Trainer` | Executes mini-batch optimization |

The implementation uses:

- `std::vector` for sample and parameter storage.
- `std::shuffle` for epoch-level randomization.
- `std::mt19937` for deterministic pseudorandom generation.
- `std::filesystem` for checkpoint management.
- `std::numeric_limits` for best-loss initialization.
- Exceptions for invalid configurations and execution failures.

The implementation passes batch indices rather than repeatedly copying the complete dataset.

This reduces unnecessary data movement.

The model validates parameter values after updates to detect numerical instability.

Compilation:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic deep_learning_training.cpp -o deep_learning_training`

Execution:

`./deep_learning_training`

On Windows:

`deep_learning_training.exe`

## 11. Java 17 Implementation

The Java implementation emphasizes enterprise-oriented domain separation and explicit contracts.

| Component | Responsibility |
|---|---|
| `TrainingConfig` | Immutable training configuration |
| `Sample` | Validated feature-target record |
| `Dataset` | Validated dataset representation |
| `LinearModel` | Model parameter management |
| `LearningRatePolicy` | Learning-rate scheduling contract |
| `CheckpointRepository` | Checkpoint persistence and recovery |
| `TrainingService` | Training orchestration |
| `Metrics` | Immutable epoch measurements |
| `TrainingState` | Training lifecycle |

Java records represent immutable configuration and metric objects.

The `LearningRatePolicy` interface separates scheduling behavior from the training service.

The checkpoint repository isolates persistence operations from optimization.

The service maintains explicit training states and prevents repeated training initialization.

Compilation:

`javac DeepLearningTraining.java`

Execution:

`java DeepLearningTraining`

Assertion tests:

`java DeepLearningTraining --test`

## 12. PostgreSQL Database Design

The database implementation represents training as a relational workflow.

### Entity Relationships

    Experiment
        |
        +---- Dataset
        |
        +---- Training Run
                  |
                  +---- Training Epoch
                  |          |
                  |          +---- Training Batch
                  |          |
                  |          +---- Training Metric
                  |
                  +---- Model Checkpoint
                  |
                  +---- Training Audit

### Core Tables

| Table | Purpose |
|---|---|
| `experiment` | Groups related training experiments |
| `dataset` | Stores dataset identity and partition information |
| `training_run` | Stores configuration and lifecycle state |
| `training_epoch` | Records epoch-level execution |
| `training_batch` | Tracks individual mini-batch execution |
| `training_metric` | Stores typed measurements |
| `model_checkpoint` | Records checkpoint metadata |
| `training_audit` | Stores lifecycle and operational events |

### Data Integrity

The database enforces:

- Positive epoch counts.
- Positive batch sizes.
- Valid learning rates.
- Valid learning-rate decay.
- Valid training and validation dataset partitions.
- Non-negative losses.
- Accuracy values between zero and one.
- Unique epoch numbers within a training run.
- Unique batch numbers within an epoch.
- Unique checkpoint names.
- One designated best checkpoint per run.

The state-transition trigger prevents invalid training lifecycle changes.

The epoch registration procedure checks that epochs are registered sequentially and that the run is active.

The checkpoint registration procedure associates persisted model artifacts with their corresponding epoch and validation loss.

### Transactions

Training epoch registration is a transactional operation.

It records:

1. Epoch execution information.
2. Training and validation metrics.
3. Training progress.
4. Best validation performance.
5. Audit information.

If an operation fails, PostgreSQL rolls back the transaction.

This prevents partially registered epoch records from being committed.

### Reporting

The database includes queries for:

- Training run retrieval.
- Epoch history.
- Training and validation loss comparisons.
- Aggregated training metrics.
- Best-epoch identification.
- Learning-rate calculation.
- Checkpoint retrieval.
- Failed-run identification.
- Incomplete batch detection.
- Training completion eligibility.
- Audit history.

Execution:

`psql -U postgres -d training_db -f deep_learning_training.sql`

The SQL script creates the `deep_learning` schema and its sample records.

The checkpoint table stores metadata and storage references. Actual model files remain in external storage rather than being stored as database records.

## 13. Important Distinctions

| Concept | Meaning | Primary effect |
|---|---|---|
| Epoch | Complete pass through training data | Measures training progress |
| Batch | Subset processed together | Determines update frequency |
| Batch size | Number of samples per update | Influences memory and gradient estimates |
| Iteration | One optimization update | Changes model parameters |
| Learning rate | Optimization step size | Controls update magnitude |
| Learning-rate scheduler | Rule for changing learning rate | Controls optimization over time |
| Training loss | Error on training data | Measures optimization |
| Validation loss | Error on validation data | Supports model selection |
| Checkpoint | Persisted training state | Enables recovery |
| Early stopping | Termination based on validation performance | Limits unnecessary training |

An epoch and an iteration are not interchangeable.

For example, 700 training samples and a batch size of 32 produce 22 iterations per epoch.

At 40 epochs, the maximum number of parameter updates is:

\[
40 \times 22 = 880
\]

This assumes training completes every configured epoch.

## 14. Performance Considerations

The computational cost of one epoch for the linear model is approximately:

\[
O(Nd)
\]

Where:

- \(N\) = number of training samples.
- \(d\) = number of features.

For \(E\) epochs, total computational complexity is:

\[
O(ENd)
\]

Batch size changes the number of optimization updates and intermediate memory requirements, but does not remove the need to process the training samples.

| Configuration | Operational implication |
|---|---|
| Small batch size | More frequent updates and greater gradient variability |
| Large batch size | Fewer updates and higher memory requirements |
| High learning rate | Larger updates and increased instability risk |
| Low learning rate | Smaller updates and potentially slower convergence |
| Frequent checkpoints | Lower potential recovery loss but greater storage activity |
| Infrequent checkpoints | Lower persistence overhead but greater recovery work |

The educational implementations use in-memory datasets and dense feature vectors.

Large production datasets may require streaming, parallel data loading, distributed processing, and GPU acceleration.

## 15. Security and Reliability

Training infrastructure must protect both data and model artifacts.

Important controls include:

- Validate dataset dimensions and numerical values.
- Restrict access to training datasets.
- Apply access controls to checkpoint storage.
- Validate checkpoint format before restoration.
- Verify checkpoint checksums.
- Protect sensitive model parameters.
- Use authenticated storage access.
- Avoid exposing secrets in training logs.
- Preserve experiment and checkpoint provenance.
- Maintain audit records for important state changes.
- Separate training, validation, and test data.
- Monitor numerical instability and failed execution.

The supplied implementations validate numerical data and restrict checkpoint paths to their configured directories.

The database stores checkpoint checksums but does not independently calculate or verify the checksum of an external artifact.

The sample checkpoint storage URIs are metadata examples. Production storage requires actual access control, retention policies, and integrity verification.

## 16. Limitations

The implementations intentionally focus on the mechanics of optimization and training-state management.

They do not provide:

- Neural-network architectures with multiple layers.
- Automatic differentiation.
- Backpropagation through deep computational graphs.
- Convolutional neural networks.
- Transformer architectures.
- GPU acceleration.
- Mixed-precision training.
- Distributed gradient synchronization.
- Adam or other advanced optimizer implementations.
- Complete optimizer-state recovery.
- Distributed checkpoint coordination.
- Production experiment orchestration.

The linear model provides a controlled environment for understanding epochs, mini-batches, learning-rate schedules, validation, and checkpoints.

The persistence mechanisms demonstrate file-level recovery, while the PostgreSQL model demonstrates relational experiment tracking.

These are complementary concerns: model artifact persistence and experiment metadata management should be coordinated but should not be treated as the same operation.
