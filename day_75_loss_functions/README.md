# Loss Functions: Mathematical Foundations, Implementation and Model Evaluation

## Domain Scope

Loss functions quantify the difference between actual observations and model predictions. They provide an objective measure of prediction error and form a central component of supervised machine learning.

The four loss functions implemented in this project are:

1. Mean Squared Error (MSE)
2. Mean Absolute Error (MAE)
3. Binary Cross-Entropy (BCE)
4. Categorical Cross-Entropy (CCE)

The implementations cover numerical computation, gradient calculation, batch processing, validation, evaluation policies, and persistence of model evaluation results.

The central distinction is between regression and classification objectives.

| Category | Loss function | Typical application |
|---|---|---|
| Regression | MSE | Continuous numerical prediction |
| Regression | MAE | Continuous prediction with greater robustness to outliers |
| Binary classification | BCE | Two-class prediction |
| Multiclass classification | CCE | Prediction among multiple mutually exclusive classes |

## Mathematical Foundations

### Mean Squared Error

MSE calculates the average squared difference between actual and predicted numerical values.

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Where:

- \(n\) represents the number of observations.
- \(y_i\) represents the actual value.
- \(\hat y_i\) represents the predicted value.

Squaring the error makes larger deviations disproportionately expensive.

For example, errors of 2 and 5 contribute squared errors of 4 and 25.

**Properties**

- Nonnegative.
- Zero when every prediction is correct.
- Differentiable everywhere.
- Sensitive to large errors and outliers.
- Expressed in squared units of the target variable.

The derivative with respect to a prediction is:

\[
\frac{\partial MSE}{\partial \hat y_i}
=
\frac{2(\hat y_i-y_i)}{n}
\]

This gradient is useful for gradient-based optimization.

MSE is appropriate when large prediction errors should receive substantial penalties.

### Mean Absolute Error

MAE calculates the average absolute difference between actual and predicted values.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

Unlike MSE, MAE does not square the error.

**Properties**

- Nonnegative.
- Zero for perfect predictions.
- Less sensitive to extreme observations than MSE.
- Expressed in the original units of the target variable.
- Not differentiable at zero error.

Its derivative, where defined, is:

\[
\frac{\partial MAE}{\partial \hat y_i}
=
\frac{\operatorname{sign}(\hat y_i-y_i)}{n}
\]

At zero error, a subgradient of zero is used in the implementations.

MAE is useful for demand forecasting, financial estimates, and other regression applications where the magnitude of an error matters without requiring quadratic penalties.

### Binary Cross-Entropy

Binary cross-entropy evaluates probabilistic predictions for a binary target.

\[
BCE=-\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
\]

Where:

- \(y_i\) is either zero or one.
- \(p_i\) is the predicted probability of the positive class.

For a positive observation, the loss depends on \(-\log(p_i)\).

For a negative observation, the loss depends on \(-\log(1-p_i)\).

Incorrect predictions made with high confidence receive large penalties.

**Properties**

- Used for binary classification.
- Evaluates predicted probabilities rather than numerical class labels.
- Nonnegative.
- Zero in the limiting case of perfectly predicted probabilities.
- Penalizes confident incorrect predictions heavily.

The logit-based formulation is:

\[
L(z,y)=
\max(z,0)-zy+\log(1+e^{-|z|})
\]

Here, \(z\) is the model's raw output before sigmoid activation.

The derivative with respect to the logit is:

\[
\frac{\partial L}{\partial z}
=
\sigma(z)-y
\]

Where:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The gradient of the mean loss includes division by the batch size.

### Categorical Cross-Entropy

Categorical cross-entropy evaluates predictions across multiple mutually exclusive classes.

\[
CCE=-\frac{1}{n}\sum_{i=1}^{n}
\sum_{j=1}^{K}y_{ij}\log(p_{ij})
\]

Where:

- \(K\) represents the number of classes.
- \(y_{ij}\) represents the one-hot target.
- \(p_{ij}\) represents the predicted probability.

For one-hot targets, only the probability associated with the correct class contributes to the loss.

The equivalent sparse-label formulation is:

\[
CCE=-\frac{1}{n}\sum_{i=1}^{n}
\log(p_{i,c_i})
\]

Where \(c_i\) is the correct class index.

The implementations use integer class labels, avoiding the need to construct full one-hot target arrays.

**Properties**

- Used for mutually exclusive multiclass classification.
- Requires a valid probability distribution for each observation.
- Penalizes low probability assigned to the correct class.
- Supports probability-based model evaluation.

For raw class logits, the stable formulation is:

\[
L(z,c)=
\log\left(\sum_{j=1}^{K}e^{z_j}\right)-z_c
\]

The corresponding gradient is:

\[
\frac{\partial L}{\partial z_j}
=
p_j-y_j
\]

For mean batch loss, the gradient is divided by the batch size.

## Comparison of the Four Loss Functions

| Characteristic | MSE | MAE | BCE | CCE |
|---|---|---|---|---|
| Learning task | Regression | Regression | Binary classification | Multiclass classification |
| Input | Numerical values | Numerical values | Binary probabilities | Class probabilities |
| Target | Continuous | Continuous | Binary labels | Class indices |
| Error treatment | Squared | Absolute | Logarithmic | Logarithmic |
| Outlier sensitivity | High | Lower | Depends on probability error | Depends on probability error |
| Differentiability | Everywhere | Not at zero error | Smooth in logits | Smooth in logits |
| Typical output layer | Linear | Linear | Sigmoid | Softmax |

## Numerical Stability

Numerical stability is important because floating-point operations have finite precision.

### Probability Clipping

Direct logarithm calculations become undefined when their arguments are zero.

The implementations use:

\[
p'=\min(\max(p,\epsilon),1-\epsilon)
\]

A small epsilon prevents `log(0)`.

Clipping is a numerical safeguard. It changes the calculated loss at exact probability boundaries and should not replace proper numerical design.

### Logits-Based Cross-Entropy

The preferred approach for training is often to calculate cross-entropy directly from logits.

For binary classification, the stable formulation avoids directly calculating an extreme sigmoid probability before taking its logarithm.

For multiclass classification, the log-sum-exp operation is evaluated using maximum subtraction:

\[
\log\sum_j e^{z_j}
=
m+\log\sum_j e^{z_j-m}
\]

Where:

\[
m=\max_j z_j
\]

This prevents exponential overflow for large positive logits.

### Input Validation

The implementations reject:

- Empty batches.
- Different actual and predicted dimensions.
- Non-finite values.
- Invalid binary labels.
- Probabilities outside the interval from zero to one.
- Invalid class indices.
- Multiclass probability distributions that do not sum to one.
- Inconsistent class dimensions.

These checks prevent invalid metrics from silently entering model evaluation.

## Reduction Strategies

Loss functions can return individual observation losses or combine them into a single value.

| Reduction | Mechanism | Application |
|---|---|---|
| NONE | Returns individual losses | Error analysis |
| SUM | Adds individual losses | Aggregate error accounting |
| MEAN | Calculates average loss | Standard batch optimization |

For example, consider three squared errors:

\[
[1,4,9]
\]

Their sum is 14 and their mean is:

\[
\frac{14}{3}=4.6667
\]

The reduction strategy affects the scale of the loss and its gradient. It must remain consistent when comparing training runs.

## Implementation Architecture

### Python: Numerical and Analytical Evaluation

The Python implementation uses standard-library functionality and provides a complete loss evaluation module.

Its principal components are:

| Component | Responsibility |
|---|---|
| `LossError` | Input and domain validation |
| `Reduction` | Controls loss aggregation |
| `LossResult` | Stores individual and aggregated losses |
| `mean_squared_error()` | Calculates MSE |
| `mean_absolute_error()` | Calculates MAE |
| `binary_cross_entropy()` | Calculates BCE from probabilities |
| `binary_cross_entropy_with_logits()` | Calculates stable BCE from logits |
| `categorical_cross_entropy()` | Calculates sparse-label CCE |
| `categorical_cross_entropy_with_logits()` | Calculates stable multiclass CCE |
| Gradient functions | Calculate derivatives for optimization |

The implementation also includes unit tests using `unittest`.

The examples demonstrate regression, binary classification, multiclass classification, extreme logits, and invalid inputs.

The analytical gradients are calculated with respect to predictions or logits. They are not complete parameter gradients for a neural network. Parameter gradients require the chain rule through the model architecture.

### JavaScript: Event-Driven Loss Monitoring

The JavaScript implementation provides an operational monitoring perspective.

It uses:

- `Map` for grouping evaluation history.
- `EventEmitter` for publishing evaluation events.
- `Promise.all()` for asynchronous batch execution.
- Immutable event records.
- Policy thresholds for evaluation decisions.

The `LossMonitor` class separates mathematical calculations from operational acceptance rules.

An evaluation produces an event containing:

- Evaluation identifier.
- Metric name.
- Loss value.
- Acceptance threshold.
- Acceptance result.
- Timestamp.

Two events are supported:

| Event | Meaning |
|---|---|
| `evaluation` | An evaluation was completed |
| `thresholdExceeded` | The loss exceeded its configured threshold |

The asynchronous batch evaluator uses a bounded number of workers.

It isolates individual batch failures instead of allowing one failed batch to discard all successful results.

The monitoring policy is illustrative. Actual production thresholds must be calibrated to the dataset, model objective, and operational requirements.

### C++17: Loss Calculation Engine

The C++ implementation emphasizes explicit domain modeling, computational control, and validation.

Its principal components are:

- `LossEngine`
- `LossResult`
- `LossType`
- `Reduction`
- `EvaluationPolicy`
- `EvaluationRecord`
- `GovernanceEngine`

The `LossEngine` implements mathematical operations independently of evaluation policy.

The `GovernanceEngine` receives calculated losses and determines whether they satisfy configured thresholds.

This separation prevents the mathematical definition of a loss function from becoming dependent on a particular operational acceptance policy.

The implementation uses STL vectors for batches and a history collection for evaluation records.

### Java 17: Enterprise Evaluation Service

The Java implementation introduces explicit enterprise domain types.

| Type | Purpose |
|---|---|
| `Model` | Identifies a model and its version |
| `LossResult` | Represents a calculated loss |
| `EvaluationPolicy` | Defines metric thresholds |
| `EvaluationRecord` | Stores an evaluation decision |
| `EvaluationStatus` | Represents acceptance state |
| `EvaluationService` | Coordinates evaluation and auditing |

Java records provide compact immutable data structures.

The evaluation service separates:

1. Model identification.
2. Loss calculation.
3. Policy lookup.
4. Threshold comparison.
5. Decision recording.
6. Audit reporting.

The status transition is:

`PENDING → ACCEPTED`

or

`PENDING → REJECTED`

The implementation uses immutable evaluation records and returns a defensive copy of its audit history.

A production service would persist these records in a transactional database and apply access controls to model ownership and evaluation operations.

## PostgreSQL Database Design

The PostgreSQL implementation provides a relational model for loss evaluation and model governance.

### Entity Relationships

The principal relationship is:

`Models`

`  └── Training Runs`

`        └── Predictions`

`        └── Loss Evaluations`

`              ├── Loss Policies`

`              └── Evaluation Events`

### Models

The `models` table identifies the model, version, task type, and responsible owner.

A unique constraint prevents duplicate model names and versions.

### Training Runs

The `training_runs` table represents individual experiments.

Each run records:

- Model association.
- Dataset name.
- Random seed.
- Execution status.
- Start timestamp.
- Completion timestamp.

The run-state trigger prevents invalid transitions and prevents completed or failed runs from being reopened.

### Predictions

The `predictions` table stores observation-level prediction data.

Its structure supports three different tasks.

| Task | Actual data | Predicted data |
|---|---|---|
| Regression | Actual numerical value | Predicted numerical value |
| Binary classification | Binary label | Positive-class probability |
| Multiclass classification | Target class | Probability array |

The task-specific trigger prevents classification fields from being inserted into regression records and vice versa.

Multiclass predictions must contain valid probability distributions.

### Loss Policies

The `loss_policies` table stores acceptance thresholds.

A policy can specify separate maximum values for:

- MSE.
- MAE.
- BCE.
- CCE.

A missing threshold means that the corresponding metric is not evaluated by that policy.

The minimum sample count prevents acceptance decisions based on insufficient observations.

### Loss Evaluations

The `loss_evaluations` table stores calculated metrics and their decisions.

It records the metric, sample count, loss value, threshold, evaluation status, and evaluator identity.

The unique constraint on run, policy, and metric prevents duplicate evaluations for the same combination.

The procedure uses `ON CONFLICT` to update an existing evaluation when a run is evaluated again under the same policy.

### Evaluation Events

The `evaluation_events` table records changes to evaluation status.

Its trigger provides an audit trail for initial decisions and subsequent status changes.

In production, audit tables should also have appropriate permissions to prevent unauthorized modification or deletion.

## Database Functions and Workflow

The database includes four principal mathematical functions:

| Function | Operation |
|---|---|
| `calculate_mse()` | Calculates squared prediction error |
| `calculate_mae()` | Calculates absolute prediction error |
| `calculate_binary_cross_entropy()` | Calculates binary classification loss |
| `calculate_categorical_cross_entropy()` | Calculates multiclass classification loss |

The `evaluate_training_run()` procedure coordinates database-level evaluation.

Its workflow is:

1. Locate and lock the training run.
2. Confirm that the run is completed.
3. Retrieve the active evaluation policy.
4. Check the minimum sample requirement.
5. Validate prediction completeness when required.
6. Select the appropriate loss function.
7. Calculate the metric.
8. Compare the metric with its configured threshold.
9. Insert or update the evaluation.
10. Record the decision through the audit trigger.

The procedure uses a transaction so that failures do not leave partially completed evaluation changes.

### Reporting Views

Three views support operational reporting.

| View | Purpose |
|---|---|
| `v_model_loss_report` | Model-wise loss evaluation |
| `v_policy_performance` | Policy acceptance and rejection statistics |
| `v_run_eligibility` | Evaluation eligibility for each run and policy |

The eligibility view distinguishes a run that has not been evaluated from one that has failed an acceptance condition.

## Important Distinctions

### Loss Versus Metric

A loss function provides an optimization objective or prediction-error measurement.

A performance metric evaluates a particular aspect of model behavior.

Although the same mathematical function can sometimes serve both purposes, their operational uses can differ.

For example, MSE can be optimized during training, while MAE is reported as a separate validation metric.

### Loss Versus Accuracy

Accuracy measures the proportion of correctly classified observations.

Cross-entropy measures the quality of predicted probabilities.

Consider two classifiers predicting the correct class with probabilities of 0.55 and 0.99.

Both can produce the same classification accuracy, but their cross-entropy losses differ.

Therefore, accuracy alone does not describe probability quality.

### Regression Versus Classification Loss

MSE and MAE operate on numerical prediction errors.

BCE and CCE evaluate probabilistic classification outputs.

Using MSE on binary labels is mathematically possible, but it does not provide the same probabilistic objective as binary cross-entropy.

Similarly, categorical cross-entropy is designed for mutually exclusive classes. Multilabel classification generally requires a separate binary loss for each label.

### Loss Versus Acceptance Threshold

A loss value describes observed model error.

An acceptance threshold expresses an operational requirement.

A model can have a mathematically valid loss but still fail a configured acceptance policy.

A threshold is not a universal mathematical property of the loss function. It depends on the application, data distribution, and business requirements.

## Practical Numerical Example

Consider the following regression observations:

| Actual | Predicted | Squared error | Absolute error |
|---:|---:|---:|---:|
| 3.0 | 2.5 | 0.25 | 0.50 |
| -0.5 | 0.0 | 0.25 | 0.50 |
| 2.0 | 2.0 | 0.00 | 0.00 |
| 7.0 | 8.0 | 1.00 | 1.00 |

The resulting values are:

\[
MSE=\frac{1.5}{4}=0.375
\]

\[
MAE=\frac{2.0}{4}=0.5
\]

For binary classification, consider:

| Actual class | Predicted probability |
|---:|---:|
| 1 | 0.95 |
| 0 | 0.10 |
| 1 | 0.80 |
| 0 | 0.20 |

The mean binary cross-entropy is:

\[
BCE=
-\frac{1}{4}
\left[
\log(0.95)+
\log(0.90)+
\log(0.80)+
\log(0.80)
\right]
\]

For multiclass classification, consider three classes and a correct class probability of 0.70 for every observation.

The individual cross-entropy is:

\[
CCE=-\log(0.70)
\]

This demonstrates why cross-entropy evaluates confidence in the correct class rather than simply counting correct predictions.

## Execution

### Python

Save the first implementation as `loss_functions.py`.

Run:

`python loss_functions.py`

The script executes demonstrations and unit tests.

### JavaScript

Save the second implementation as `loss_functions.js`.

Run:

`node loss_functions.js`

The program demonstrates synchronous calculations, asynchronous batch processing, monitoring events, and policy decisions.

### C++17

Save the third implementation as `loss_functions.cpp`.

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic loss_functions.cpp -o loss_functions`

Run:

`./loss_functions`

On Windows:

`loss_functions.exe`

### Java 17

Save the fourth implementation as `LossGovernance.java`.

Compile:

`javac LossGovernance.java`

Run:

`java LossGovernance`

### PostgreSQL

Execute the fifth implementation using PostgreSQL 14 or later.

The script creates the `ml_governance` schema, database objects, sample data, evaluation procedures, and reporting queries.

The script is enclosed in a transaction. A failure before `COMMIT` requires rollback of the transaction.

The database setup should be executed by a role with appropriate schema and database-object creation permissions.

## Performance Considerations

For a batch containing \(n\) observations and \(K\) classes:

| Operation | Time complexity | Additional space |
|---|---:|---:|
| MSE | \(O(n)\) | \(O(n)\) for individual losses |
| MAE | \(O(n)\) | \(O(n)\) for individual losses |
| BCE | \(O(n)\) | \(O(n)\) for individual losses |
| CCE | \(O(nK)\) | \(O(n)\) for individual losses |
| Softmax | \(O(K)\) | \(O(K)\) |
| MSE gradient | \(O(n)\) | \(O(n)\) |
| Binary logit gradient | \(O(n)\) | \(O(n)\) |

For large training workloads:

- Calculate losses in batches rather than loading an entire dataset into memory.
- Use vectorized numerical operations where appropriate.
- Prefer logits-based cross-entropy during model optimization.
- Avoid retaining unnecessary per-observation losses in production.
- Use database indexes for run identifiers and evaluation retrieval.
- Keep evaluation policies versioned when historical reproducibility is required.

The SQL implementation uses arrays for compact demonstration and batch-level calculations. For large datasets, streaming aggregation or set-based processing is generally more scalable than collecting every prediction into arrays.

## Security and Reliability

Production implementations should consider the following controls:

- Validate inputs at the application and database boundaries.
- Restrict database write permissions.
- Separate model owners, evaluators, and administrators.
- Record the identity responsible for evaluation.
- Prevent unauthorized modification of completed experiments.
- Preserve historical policy versions.
- Use transactions for evaluation updates.
- Protect sensitive datasets and prediction records.
- Avoid exposing sample-level information in general reporting.
- Apply retention policies to evaluation history.
- Monitor numerical overflow and non-finite values.
- Use reproducible dataset versions and random seeds.

Loss thresholds should be evaluated on representative validation data rather than selected solely to produce an accepted model.

## Limitations and Engineering Considerations

The implementations intentionally focus on four standard loss functions.

They do not implement:

- Huber loss.
- Log-cosh loss.
- Kullback-Leibler divergence.
- Focal loss.
- Contrastive loss.
- Ranking losses.
- Multilabel-specific training objectives.
- Distributed gradient aggregation.
- Neural network parameter optimization.

The gradient implementations calculate derivatives with respect to predictions or logits. They do not implement automatic differentiation or a complete neural network training system.

The probability-based implementations use clipping to handle exact boundary probabilities. The logits-based implementations are preferable for numerically stable training.

The evaluation policies use illustrative numerical thresholds. Production thresholds require validation against actual datasets, class distributions, prediction requirements, and operational consequences.

The JavaScript and C++ monitoring engines use in-memory histories. The Java service uses an in-memory audit collection. PostgreSQL provides persistent evaluation records and database-level audit events.

The PostgreSQL sample schema assumes that each prediction contains one observation and one target. It does not model distributed training, large-scale feature storage, or complete experiment artifact management.

These boundaries keep the mathematical definitions, numerical implementations, and evaluation governance responsibilities separate.
