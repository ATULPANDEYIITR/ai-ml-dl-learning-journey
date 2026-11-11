# Perceptron Learning Rule and Linear Separability

## 1. Technical Scope

This project implements a binary perceptron classifier and examines the mathematical conditions under which its learning procedure converges.

The perceptron is one of the earliest computational models of supervised learning. It represents a linear decision function that separates observations into two classes using adjustable weights and a bias.

The implementations demonstrate the same fundamental learning algorithm through different engineering perspectives:

- Python: experimental training, evaluation, feature scaling, and model persistence.
- JavaScript: asynchronous training, event-driven observations, and model serialization.
- C++17: a compact classification and training engine using strongly modeled data structures.
- Java 17: an enterprise-oriented training service with immutable records, policies, and explicit state transitions.
- PostgreSQL: relational storage and transactional execution of training operations.

The central distinction is between a dataset that admits a linear decision boundary and one that cannot be separated by a single linear classifier.

## 2. Mathematical Foundations

### 2.1 Linear decision function

For an input vector containing \(n\) features:

\[
x = [x_1,x_2,\ldots,x_n]
\]

the perceptron calculates:

\[
z = \sum_{i=1}^{n} w_i x_i+b
\]

where:

- \(x_i\): input feature.
- \(w_i\): weight associated with the feature.
- \(b\): bias.
- \(z\): weighted sum or net input.

The bias shifts the decision boundary without requiring an additional externally supplied feature.

### 2.2 Step activation function

The implementations use a binary threshold activation function:

\[
\hat y =
\begin{cases}
1 & z \geq 0\\
0 & z < 0
\end{cases}
\]

The boundary convention is important. A net input of exactly zero belongs to class 1.

The activation function produces a discrete classification rather than a probability.

Consequently, the standard perceptron does not directly provide calibrated confidence scores.

### 2.3 Perceptron learning rule

Training begins with zero-valued weights and bias.

For each observation, the algorithm calculates:

\[
e=y-\hat y
\]

The weight update is:

\[
w_i^{new}=w_i^{old}+\eta e x_i
\]

The bias update is:

\[
b^{new}=b^{old}+\eta e
\]

where \(\eta\) represents the learning rate.

There are three possible outcomes:

| Target | Prediction | Error | Action |
|---|---|---|---|
| 0 | 0 | 0 | No update |
| 1 | 1 | 0 | No update |
| 1 | 0 | 1 | Increase weights according to input |
| 0 | 1 | -1 | Decrease weights according to input |

An observation that is already correctly classified does not change the model.

An incorrectly classified observation modifies the decision function so that its next classification is influenced by the correction.

The algorithm is online: each update immediately affects the predictions for subsequent observations in the same epoch.

## 3. Linear Separability

### 3.1 Definition

A binary dataset is linearly separable when there exists a weight vector and bias that correctly classify every observation using one linear decision boundary.

For labels represented as \(\{-1,+1\}\), the condition is:

\[
y_j(w^Tx_j+b)>0
\]

for every observation \(j\).

For the implementations, labels are represented as \(\{0,1\}\), with the same classification condition expressed through the threshold function.

### 3.2 AND gate

The AND dataset is:

| Input 1 | Input 2 | Target |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

A valid decision function is:

\[
z=x_1+x_2-1.5
\]

This produces:

- Negative output for the first three observations.
- Positive output for the final observation.

The dataset is linearly separable, so the perceptron can converge under the standard learning assumptions.

### 3.3 OR gate

The OR function is:

| Input 1 | Input 2 | Target |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

A valid decision function is:

\[
z=x_1+x_2-0.5
\]

A single linear boundary separates the two classes.

### 3.4 XOR gate

The XOR dataset is:

| Input 1 | Input 2 | Target |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The positive observations occupy opposite corners of the input space.

No single straight line can separate these positive observations from the negative observations.

The XOR problem is therefore not linearly separable.

The implemented perceptrons train XOR for a fixed number of epochs and demonstrate that the standard algorithm does not converge on this dataset.

Failure to converge within a finite training budget is not, by itself, a mathematical proof that an arbitrary dataset is non-separable. The XOR dataset is known to be non-separable from its geometric arrangement.

## 4. The Four Distinct Technical Areas

The implementations maintain a clear distinction between the learning mechanism, the geometry of the data, the resulting model, and the operational training process.

| Area | Meaning | Implementation concern |
|---|---|---|
| Perceptron model | Weighted linear decision function | Weights, bias, activation |
| Learning rule | Mechanism for correcting incorrect classifications | Error-driven updates |
| Linear separability | Existence of a valid linear boundary | Dataset geometry and convergence |
| Training management | Execution and monitoring of learning | Epochs, validation, state, persistence |

Convergence is a property of the training process under specified conditions. Linear separability is a property of the classification problem.

These concepts should not be treated as interchangeable.

## 5. Training Workflow

The implemented training process follows this sequence:

1. Validate the dataset.
2. Verify feature dimensions.
3. Verify that labels are binary.
4. Confirm that both classes are represented.
5. Initialize weights and bias.
6. Iterate through observations.
7. Calculate the net input.
8. Apply the threshold activation.
9. Calculate classification error.
10. Update weights and bias when necessary.
11. Record the number of mistakes.
12. Check the convergence condition.
13. Stop when an epoch has no mistakes or the epoch budget is exhausted.

Plain-text representation:

    Dataset
       |
       v
    Validation
       |
       v
    Initialize Weights and Bias
       |
       v
    Calculate Weighted Sum
       |
       v
    Apply Step Activation
       |
       v
    Compare Prediction with Target
       |
       v
    Calculate Error
       |
       v
    Update Weights and Bias
       |
       v
    Complete Epoch
       |
       v
    Zero Mistakes?
       |
       +---- Yes ----> Converged
       |
       +---- No -----> Next Epoch
                          |
                          v
                    Epoch Limit?
                          |
                          +---- Yes ----> Maximum Epochs Reached

## 6. Python Implementation

The Python implementation provides an experimental model for studying the algorithm.

### Main components

| Component | Responsibility |
|---|---|
| `TrainingConfig` | Immutable training parameters |
| `EpochResult` | Records individual epoch outcomes |
| `Evaluation` | Stores classification metrics |
| `StandardScaler` | Normalizes feature distributions |
| `Perceptron` | Maintains parameters and executes training |
| `validate_dataset` | Enforces data integrity |

### Important implementation decisions

**Online updates**

The implementation changes weights immediately after each incorrect prediction.

This preserves the sequential behavior of the perceptron learning rule.

**Deterministic execution**

The random seed controls optional observation shuffling.

Repeated experiments can therefore use the same initial conditions and ordering.

**Feature scaling**

The standard scaler calculates means and standard deviations from training observations.

The same fitted statistics must be used for validation and test observations.

Fitting a scaler separately on evaluation data can introduce data leakage.

Constant-valued features receive a scale of one to avoid division by zero.

**Model persistence**

The model is serialized into JSON with its weights, bias, dimensions, and configuration.

Loading validates the saved parameters before restoring the model.

The temporary-file replacement approach reduces the chance of leaving a partially written model file.

**Evaluation**

The confusion matrix distinguishes true positives, true negatives, false positives, and false negatives.

Accuracy is calculated as:

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

### Edge cases

- Empty datasets are rejected.
- Inconsistent feature dimensions are rejected.
- Non-finite feature values are rejected.
- Invalid labels are rejected.
- Prediction with the wrong feature count is rejected.
- An untrained model cannot be saved.
- A constant feature does not cause scaler division by zero.

## 7. JavaScript Implementation

The JavaScript implementation emphasizes asynchronous execution and event-driven training observation.

### Main components

| Component | Responsibility |
|---|---|
| `FeatureScaler` | Feature normalization |
| `Perceptron` | Model and training state |
| `EventEmitter` | Publishes training events |
| `validateDataset` | Validates observations |
| `validateConfig` | Validates training parameters |
| `createRandom` | Deterministic random number generation |

### Event-driven training

The implementation publishes three important types of events:

- `epoch`: emitted after each completed epoch.
- `converged`: emitted when an epoch produces no classification mistakes.
- `trainingError`: emitted when training fails.

The `finished` event reports the final training state.

Consumers can subscribe to these events without changing the learning algorithm.

### Asynchronous execution

Training uses an asynchronous method and periodically yields to the Node.js event loop.

This provides opportunities for other scheduled tasks to execute during longer experiments.

The implementation also supports cancellation through an `AbortSignal`.

Cancellation is distinct from convergence. A cancelled training operation must not be reported as a successfully converged model.

### Model serialization

The implementation uses JSON serialization and filesystem operations.

A model file contains:

- Model version.
- Configuration.
- Weight vector.
- Bias.
- Number of features.
- Convergence status.
- Number of completed epochs.

The version field provides a basis for validating serialized model formats.

The filesystem operations use a temporary destination followed by a rename.

Production deployments should also consider file permissions, concurrent writers, destination-directory permissions, and atomic replacement behavior on the target filesystem.

## 8. C++17 Implementation

The C++ implementation emphasizes domain modeling and explicit state management.

### Main components

| Component | Responsibility |
|---|---|
| `Sample` | Feature vector and binary target |
| `TrainingConfig` | Learning configuration |
| `EpochReport` | Training observations |
| `Evaluation` | Classification metrics |
| `Perceptron` | Core classification engine |
| `DatasetFactory` | AND and XOR datasets |
| `Experiment` | Experiment execution and reporting |

### Engineering decisions

The model owns its weight vector and bias.

External callers interact through methods such as `train`, `predict`, and `evaluate`.

The use of `std::vector` supports variable feature dimensions without requiring fixed-size arrays.

`std::mt19937` supplies deterministic pseudorandom shuffling.

The training engine uses exceptions for invalid data, numerical overflow, and invalid operations.

The implementation checks weights and bias for non-finite values after updates.

### Complexity

For:

- \(N\) observations.
- \(D\) features.
- \(E\) training epochs.

The dominant training cost is:

\[
O(E \times N \times D)
\]

The model requires:

\[
O(D)
\]

space for weights.

Recorded epoch history can require:

\[
O(E \times D)
\]

additional memory.

The implementation therefore keeps the core model compact while allowing experiments to inspect its training history.

## 9. Java 17 Implementation

The Java implementation models perceptron training as a service-oriented classification workflow.

### Domain types

| Type | Purpose |
|---|---|
| `TrainingSample` | Validated immutable observation |
| `TrainingPolicy` | Training configuration |
| `EpochSnapshot` | Immutable epoch record |
| `EvaluationReport` | Evaluation results |
| `ModelSnapshot` | Read-only model state |
| `TrainingStatus` | Training lifecycle |
| `PerceptronModel` | Classification and learning |
| `FeatureScaler` | Feature normalization |
| `ModelTrainingService` | Coordinates training |

### Immutable records

Java records represent data that should not be modified after creation.

The `TrainingSample` record validates its inputs and stores an immutable copy of its feature list.

`EpochSnapshot` prevents consumers from modifying historical weight values.

This avoids accidental changes to previously recorded training results.

### State transitions

The model uses explicit training states:

| State | Meaning |
|---|---|
| `NOT_TRAINED` | No initialized model |
| `TRAINING` | Training is executing |
| `CONVERGED` | An epoch completed without mistakes |
| `MAXIMUM_EPOCHS_REACHED` | Training stopped at its configured limit |
| `FAILED` | An exception interrupted training |

The state is more informative than a simple Boolean convergence flag.

### Enterprise-oriented design

The training service separates training orchestration from the mathematical model.

Listeners receive immutable epoch snapshots and can support monitoring or reporting.

The model rejects concurrent training through its state check.

The feature scaler uses statistics calculated from the training dataset.

The implementation uses only Java 17 standard-library capabilities.

## 10. PostgreSQL Implementation

The database implementation represents the training lifecycle as a relational workflow.

### Relational structure

    datasets
       |
       +---- samples
       |
       +---- training_runs
                 |
                 +---- model_weights
                 |
                 +---- epoch_updates
                 |
                 +---- predictions

### Tables

| Table | Purpose |
|---|---|
| `datasets` | Dataset metadata and feature count |
| `samples` | Feature vectors and target labels |
| `training_runs` | Training configuration and lifecycle |
| `model_weights` | Individual model weights |
| `epoch_updates` | Observation-level learning history |
| `predictions` | Stored classification results |

### Data integrity

Primary keys identify database records.

Foreign keys maintain relationships between datasets, observations, and training runs.

Check constraints enforce:

- Positive learning rates.
- Positive epoch limits.
- Binary labels.
- Valid prediction labels.
- Valid prediction errors.
- Positive feature indexes.
- Valid confidence-score ranges.

The sample-validation trigger checks the feature count against the parent dataset.

The epoch-update trigger checks that the target label corresponds to the referenced observation.

The training-state trigger assigns timestamps when a run starts or completes.

### Training procedure

`train_perceptron` implements the online learning rule inside PostgreSQL.

It:

1. Locks the selected training-run row.
2. Checks dataset and model initialization.
3. Verifies that both classes exist.
4. Resets the model parameters.
5. Starts the training run.
6. Processes observations in sample order.
7. Calculates predictions.
8. Calculates classification errors.
9. Updates individual weights.
10. Updates the bias.
11. Records epoch observations.
12. Determines convergence.
13. Updates the final training status.

The row-level lock prevents two transactions from simultaneously training the same run through this procedure.

The calling transaction controls whether the operation is committed or rolled back.

### Evaluation

`evaluate_model` calculates:

- True positives.
- True negatives.
- False positives.
- False negatives.
- Accuracy.

`prediction_report` presents individual predictions alongside actual labels.

`training_summary` presents the lifecycle and configuration of each run.

### Performance considerations

The primary key on `model_weights` supports efficient weight lookup by run and feature index.

The indexes on samples and training runs support dataset retrieval and run filtering.

The epoch-update index supports historical analysis by run and epoch.

The SQL procedure prioritizes transparent relational operations over computational efficiency. It performs database updates for individual observations, which can become expensive for large datasets.

For production-scale training, numerical computation would generally be performed in an application or specialized numerical engine, with PostgreSQL retaining metadata, model versions, experiment records, and evaluation results.

## 11. Learning Rate and Convergence

The learning rate controls the magnitude of parameter corrections.

A smaller learning rate produces smaller parameter updates.

A larger learning rate produces larger updates.

For the classical perceptron convergence guarantee, the learning rate must be positive and the training observations must be linearly separable.

The standard convergence theorem assumes a finite dataset and a fixed ordering or a suitable repeated presentation of its observations.

If a separating hyperplane exists with a positive margin, the perceptron makes only a finite number of mistakes under the standard assumptions.

The learning rate scales the parameter values but does not change the existence of a separating boundary.

For non-separable datasets, the standard perceptron has no general guarantee of convergence.

It may continue updating parameters until the epoch limit is reached.

## 12. Decision Boundary Geometry

For two features:

\[
w_1x_1+w_2x_2+b=0
\]

When \(w_2\neq0\), the boundary can be expressed as:

\[
x_2=-\frac{w_1}{w_2}x_1-\frac{b}{w_2}
\]

The slope is:

\[
m=-\frac{w_1}{w_2}
\]

The vertical intercept is:

\[
c=-\frac{b}{w_2}
\]

When \(w_2=0\) and \(w_1\neq0\), the boundary is vertical.

For higher-dimensional input, the decision boundary is a hyperplane rather than a two-dimensional line.

## 13. Evaluation and Interpretation

A successful training run should not be evaluated solely by the number of epochs.

Relevant measurements include:

| Measurement | Interpretation |
|---|---|
| Epoch count | Number of completed passes |
| Mistake count | Incorrect classifications during an epoch |
| Convergence status | Whether an epoch completed with zero mistakes |
| Accuracy | Fraction of correctly classified observations |
| Confusion matrix | Distribution of classification outcomes |
| Weight vector | Learned feature contributions |
| Bias | Learned decision-boundary offset |

Training accuracy alone does not establish generalization performance.

An independent test dataset should be used when evaluating performance on unseen observations.

For small datasets, accuracy can be misleading because individual observations have a large effect on the reported result.

## 14. Important Distinctions and Limitations

### Perceptron versus logistic regression

A perceptron uses a threshold activation and a mistake-driven update.

Logistic regression models the probability of a binary outcome using a sigmoid function and a differentiable loss function.

Their training objectives and output interpretations are different.

### Perceptron versus multilayer neural networks

A single-layer perceptron creates one linear decision boundary.

A multilayer network can learn nonlinear decision functions through intermediate representations and nonlinear activation functions.

A single perceptron cannot solve XOR in its original two-dimensional feature representation.

### Linear separability versus perfect training accuracy

A dataset is linearly separable when a valid linear decision function exists.

Perfect accuracy on a particular training run does not automatically establish linear separability if the evaluation does not cover all relevant observations or if data transformations have changed the feature representation.

### Convergence versus optimality

The classical perceptron finds a separating solution when the convergence assumptions hold.

It does not necessarily find the maximum-margin separating hyperplane.

It is therefore different from a hard-margin support vector machine.

## 15. Security, Reliability, and Production Considerations

The examples validate input dimensions, binary labels, finite feature values, and training configuration.

For production deployment, additional controls are important:

- Restrict access to model files and database tables.
- Validate serialized model formats before loading.
- Avoid accepting arbitrary filesystem destinations from untrusted clients.
- Use parameterized SQL when integrating application inputs.
- Record model versions and training configurations.
- Maintain separate training, validation, and test datasets.
- Preserve the scaler parameters with the corresponding model.
- Use database transactions for consistent training records.
- Monitor numerical overflow and invalid model states.
- Apply resource limits to dataset size and training duration.
- Track training failures separately from non-convergence.
- Protect model artifacts from unauthorized modification.

The SQL implementation uses database constraints and transactional execution to protect relational integrity. Database privileges and application-level authorization must still be configured separately.

## 16. Running the Implementations

### Python

Requirements: Python 3.10 or newer.

Run:

`python perceptron.py`

The implementation uses Python standard-library modules only.

### JavaScript

Requirements: Node.js 18 or newer.

Run:

`node perceptron.js`

The implementation uses Node.js standard-library modules and does not require npm packages.

### C++17

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic perceptron.cpp -o perceptron`

Run:

`./perceptron`

On Windows:

`perceptron.exe`

### Java 17

Compile:

`javac Main.java`

Run:

`java Main`

### PostgreSQL

Execute the SQL script using a PostgreSQL 15 or newer database.

For example:

`psql -U postgres -d your_database -f perceptron.sql`

The script creates the `perceptron_lab` schema and its training records.

The final training and evaluation queries demonstrate the AND and XOR experiments.

## 17. Expected Experimental Outcomes

| Experiment | Linear separability | Expected behavior |
|---|---|---|
| AND | Yes | Converges |
| OR | Yes | Converges |
| XOR | No | Does not converge within the configured budget |

The implementations provide a common mathematical foundation while exposing different engineering concerns: numerical experimentation, asynchronous execution, strongly typed model management, enterprise service design, and relational persistence.
