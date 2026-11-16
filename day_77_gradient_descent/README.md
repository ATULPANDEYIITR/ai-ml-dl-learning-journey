# Gradient Descent: Batch, Stochastic and Mini-Batch Optimization

## 1. Project Scope

This project implements and compares three fundamental gradient descent optimization strategies used to estimate parameters of a linear regression model:

- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent

The implementations use five different engineering perspectives: Python numerical modeling, JavaScript event-driven optimization, C++ numerical computation, Java enterprise-oriented training services, and PostgreSQL-based experiment management.

The central objective is to minimize a differentiable loss function by repeatedly updating model parameters in the direction opposite to the calculated gradient.

For a linear regression model:

\[
\hat y = w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

where:

- \(x_i\) represents an input feature.
- \(w_i\) represents a model weight.
- \(b\) represents the bias.
- \(\hat y\) represents the predicted value.
- \(y\) represents the actual target.

## 2. Mathematical Foundation

### 2.1 Objective Function

The implementations use the half mean squared error objective:

\[
J(w,b)=\frac{1}{2m}
\sum_{i=1}^{m}(\hat y_i-y_i)^2
\]

The factor \(1/2\) simplifies differentiation.

With L2 regularization:

\[
J(w,b)=
\frac{1}{2m}
\sum_{i=1}^{m}(\hat y_i-y_i)^2
+
\frac{\lambda}{2}\sum_{j=1}^{n}w_j^2
\]

Here, \(\lambda\) controls the strength of regularization. The bias is not penalized.

The weight gradient is:

\[
\frac{\partial J}{\partial w_j}
=
\frac{1}{m}
\sum_{i=1}^{m}
(\hat y_i-y_i)x_{ij}
+\lambda w_j
\]

The bias gradient is:

\[
\frac{\partial J}{\partial b}
=
\frac{1}{m}
\sum_{i=1}^{m}(\hat y_i-y_i)
\]

The parameter update rule is:

\[
w_j^{(t+1)}
=
w_j^{(t)}
-
\alpha
\frac{\partial J}{\partial w_j}
\]

\[
b^{(t+1)}
=
b^{(t)}
-
\alpha
\frac{\partial J}{\partial b}
\]

where \(\alpha\) is the learning rate.

## 3. Three Gradient Descent Strategies

### 3.1 Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the complete training dataset before updating the parameters.

For \(m\) training examples:

\[
g_w=
\frac{1}{m}
\sum_{i=1}^{m}
(\hat y_i-y_i)x_i
\]

One parameter update occurs after processing all \(m\) examples.

**Characteristics**

- Uses the entire training dataset for every update.
- Produces a comparatively stable gradient.
- Requires more computation before each parameter update.
- Can become expensive for large datasets.
- Provides deterministic updates for fixed data and parameters.

### 3.2 Stochastic Gradient Descent

Stochastic Gradient Descent calculates the gradient from one training example at a time.

For sample \(i\):

\[
g_w=(\hat y_i-y_i)x_i
\]

The parameters are updated immediately after processing that sample.

**Characteristics**

- Performs frequent parameter updates.
- Requires less memory for individual updates.
- Introduces variability into the optimization trajectory.
- Can make rapid progress during early training.
- May fluctuate around a minimum instead of settling smoothly.

The learning rate requires particular attention because individual observations can produce substantially different gradients.

### 3.3 Mini-Batch Gradient Descent

Mini-Batch Gradient Descent divides the training dataset into smaller groups.

For a batch containing \(B\) observations:

\[
g_w=
\frac{1}{B}
\sum_{i=1}^{B}
(\hat y_i-y_i)x_i
\]

The model updates its parameters after processing each batch.

The number of batches per epoch is:

\[
K=\left\lceil\frac{m}{B}\right\rceil
\]

The final batch can contain fewer observations than the configured batch size. Its gradients must be divided by its actual number of observations.

**Characteristics**

- Balances update frequency and gradient stability.
- Supports efficient vectorized computation.
- Uses less memory than full-batch processing.
- Works well with modern hardware accelerators.
- Introduces controlled variability into optimization.

## 4. Comparative Analysis

| Property | Batch | Stochastic | Mini-Batch |
|---|---|---|---|
| Samples per update | Entire dataset | One | Configured batch |
| Updates per epoch | 1 | m | Ceiling of m/B |
| Gradient stability | High | Low | Moderate to high |
| Memory requirement | High | Low | Moderate |
| Update frequency | Low | High | Moderate |
| Parallel computation | Possible | Limited per update | Efficient |
| Optimization trajectory | Smooth | Noisy | Moderately noisy |
| Typical application | Smaller datasets | Online learning | General-purpose training |

The three methods optimize the same objective function. Their principal difference is the number of training observations used to estimate each gradient.

## 5. Learning Rate and Convergence

The learning rate determines the magnitude of each parameter update.

An excessively small learning rate can result in slow convergence.

An excessively large learning rate can cause:

- Oscillations around a minimum.
- Unstable parameter updates.
- Increasing loss.
- Numerical overflow.
- Divergence.

A simplified convergence condition for gradient descent on a smooth convex objective with an \(L\)-Lipschitz gradient is:

\[
0<\alpha<\frac{2}{L}
\]

This is a theoretical condition for the relevant smooth convex setting, not a universal learning-rate guarantee for every dataset or optimization problem.

The implementations use configurable learning rates rather than assuming that one value works for all strategies.

### Convergence Monitoring

The training engines maintain loss history and monitor improvement between epochs.

If improvement remains below the configured tolerance for a specified number of consecutive epochs, training stops early.

This is an operational stopping policy. It should not be interpreted as proof that a global optimum has been reached.

## 6. Gradient Clipping and Regularization

### Gradient Clipping

Large gradients can cause excessively large parameter updates.

The implementations use global gradient norm clipping:

\[
\|g\|_2=
\sqrt{\sum_jg_j^2+g_b^2}
\]

The clipping scale is:

\[
s=\min\left(1,\frac{c}{\|g\|_2}\right)
\]

where \(c\) is the maximum allowed gradient norm.

The adjusted gradient is:

\[
g'=sg
\]

Clipping limits update magnitude but does not replace appropriate learning-rate selection.

### L2 Regularization

L2 regularization adds a penalty proportional to the squared magnitude of model weights.

Its purpose is to discourage unnecessarily large weights and control model complexity.

The implementation excludes the bias from regularization.

Regularization strength must be selected carefully. Excessive regularization can produce underfitting.

## 7. Python Implementation

**File:** `gradient_descent.py`

The Python implementation provides a complete numerical training workflow.

### Main Components

| Component | Responsibility |
|---|---|
| `Sample` | Immutable representation of an observation |
| `Dataset` | Synthetic regression dataset generation |
| `TrainingConfig` | Optimization configuration and validation |
| `LinearRegression` | Prediction, loss, gradients and updates |
| `GradientDescentTrainer` | Strategy-specific training orchestration |
| `TrainingResult` | Model parameters and training history |

### Implementation Mechanisms

The optimizer selection uses an enumeration:

- `BATCH`
- `STOCHASTIC`
- `MINI_BATCH`

The trainer constructs the appropriate collection of sample indices for each strategy.

For mini-batch training, the final incomplete batch is retained rather than discarded.

The implementation also includes:

- Feature-dimension validation.
- Finite-value checks.
- Deterministic random shuffling.
- Gradient clipping.
- L2 regularization.
- Early stopping.
- Mean squared error evaluation.
- R-squared calculation.
- CSV export.
- Analytical gradient verification through finite differences.

The numerical gradient verification compares analytical derivatives with approximations calculated using:

\[
\frac{\partial J}{\partial w_j}
\approx
\frac{J(w_j+\epsilon)-J(w_j-\epsilon)}
{2\epsilon}
\]

This provides an independent check of the gradient implementation.

## 8. JavaScript Implementation

**File:** `gradient_descent.js`

The JavaScript implementation uses an event-driven architecture.

### Main Components

| Component | Responsibility |
|---|---|
| `RandomGenerator` | Deterministic random sampling |
| `Dataset` | Regression sample generation |
| `LinearRegression` | Numerical model operations |
| `OptimizerStrategy` | Batch construction |
| `TrainingEngine` | Asynchronous training orchestration |
| `EventEmitter` | Training lifecycle notifications |

### Event-Driven Training

The `TrainingEngine` extends Node.js `EventEmitter`.

It publishes two important events:

- `epoch`: Reports epoch-level optimization metrics.
- `complete`: Reports successful training completion.

This separates the numerical training process from reporting and monitoring.

The training method is asynchronous and periodically yields to the Node.js event loop using `setImmediate`.

This improves responsiveness for event-loop tasks. It does not make the numerical calculations parallel or eliminate CPU blocking during an individual synchronous training operation.

### JavaScript-Specific Decisions

- `Float64Array` stores numerical weights and gradients.
- `Map`-independent strategy selection keeps the optimizer interface simple.
- `Object.freeze` protects configuration and result structures.
- `Promise` supports asynchronous orchestration.
- `EventEmitter` supports event-driven monitoring.
- Node.js `fs/promises` provides asynchronous CSV writing.
- Temporary-file replacement reduces the risk of leaving a partially written report.

The implementation uses only Node.js built-in modules.

## 9. C++17 Implementation

**File:** `gradient_descent.cpp`

The C++ implementation emphasizes numerical computation, explicit state management, and standard-library efficiency.

### Main Components

| Component | Responsibility |
|---|---|
| `Sample` | Observation representation |
| `TrainingConfig` | Optimization settings |
| `Gradients` | Gradient data structure |
| `EpochMetrics` | Training performance record |
| `LinearRegression` | Numerical model |
| `GradientDescentEngine` | Optimization execution |
| `TrainingResult` | Final model and history |

### Implementation Decisions

- `std::vector<double>` stores model parameters.
- `std::mt19937` provides reproducible random shuffling.
- `std::shuffle` supports stochastic ordering.
- `std::inner_product` calculates regularization penalties.
- `std::numeric_limits` provides numerical bounds.
- Exceptions handle invalid configuration and numerical failures.

The engine explicitly constructs batches using STL containers.

The implementation separates dataset generation, numerical calculations, optimization, and evaluation.

This structure makes it possible to change the optimization strategy without rewriting the linear regression model.

## 10. Java 17 Enterprise Implementation

**File:** `GradientDescentEnterprise.java`

The Java implementation models gradient descent as a configurable enterprise training service.

### Domain Types

The implementation defines separate types for:

- `OptimizerType`
- `TrainingStatus`
- `ModelStatus`
- `Sample`
- `TrainingPolicy`
- `Gradients`
- `EpochMetrics`
- `EvaluationMetrics`
- `TrainingReport`

Java records represent immutable data-transfer structures.

### Strategy Pattern

The `BatchStrategy` interface separates optimization strategy selection from the training service.

Three implementations provide different batch-construction behaviors:

- `BatchOptimizer`
- `StochasticOptimizer`
- `MiniBatchOptimizer`

`StrategyFactory` selects the correct strategy through an enum-based registry.

### Training Lifecycle

The training service uses explicit state transitions:

| State | Meaning |
|---|---|
| CREATED | Training request initialized |
| VALIDATED | Configuration and data validated |
| RUNNING | Optimization in progress |
| COMPLETED | Configured training completed |
| EARLY_STOPPED | Training stopped by convergence policy |
| FAILED | Training encountered an error |

The model has its own independent lifecycle:

- INITIALIZED
- TRAINED
- EVALUATED

Separating model state from training-run state prevents unrelated lifecycle information from being combined.

The implementation also includes:

- Custom exceptions.
- Immutable configuration records.
- Strategy-based optimization.
- Stream-based dataset validation.
- Policy-based convergence monitoring.
- Evaluation services.
- CSV audit reporting.

## 11. PostgreSQL Implementation

**File:** `gradient_descent.sql`

The PostgreSQL implementation provides relational storage and execution of gradient descent experiments.

Unlike the other implementations, it uses PL/pgSQL to execute numerical optimization and persist each epoch's results.

### Relational Domain Model

The principal entities are:

| Table | Purpose |
|---|---|
| `datasets` | Dataset metadata |
| `samples` | Feature arrays and target values |
| `optimization_policies` | Optimizer configurations |
| `optimization_runs` | Training execution records |
| `model_parameters` | Parameter snapshots |
| `epoch_metrics` | Epoch-level performance |
| `model_evaluations` | Independent evaluation results |

### Relationship Structure

Dataset → Samples

Optimization Policy → Optimization Runs

Dataset → Optimization Runs

Optimization Run → Model Parameters

Optimization Run → Epoch Metrics

Optimization Run → Model Evaluations

### Database Integrity

The SQL implementation uses:

- Primary keys.
- Foreign keys.
- Unique constraints.
- Check constraints.
- Indexes.
- Validation triggers.
- State-transition triggers.
- Stored functions.
- Stored procedures.
- Reporting views.

The sample-validation trigger ensures that every observation contains the expected number of features.

The sample-count trigger maintains the dataset's recorded sample count when records are inserted, deleted, or reassigned.

### Training Execution

The `train_model` procedure:

1. Locks the optimization run.
2. Validates its lifecycle state.
3. Retrieves the selected optimization policy.
4. Checks training dataset eligibility.
5. Initializes weights and bias.
6. Calculates initial loss.
7. Constructs strategy-specific batches.
8. Calculates gradients.
9. Applies gradient clipping.
10. Updates weights and bias.
11. Calculates epoch loss.
12. Stores parameter snapshots and metrics.
13. Evaluates convergence.
14. Updates the final run state.

The `evaluate_model` procedure evaluates a trained model against a separate testing dataset.

This separation helps prevent the training loss from being confused with independent generalization performance.

### Database Transactions

The script creates metadata, datasets, samples, policies, and training runs inside an explicit transaction.

Each training procedure call executes within its own database transaction.

A failure rolls back changes made by that procedure call. The exception handler attempts to record the failure state before propagating the error. Because the failure is re-raised, the transaction rollback also rolls back that failure-state update. Durable failure auditing therefore requires a separate transaction or an external orchestration mechanism.

## 12. Dataset and Experimental Design

The implementations use a synthetic two-feature linear regression problem.

The underlying relationship is:

\[
y=2.5x_1-1.7x_2+4+\epsilon
\]

where \(\epsilon\) represents random noise.

The known generating parameters are:

| Parameter | Value |
|---|---:|
| Weight 1 | 2.5 |
| Weight 2 | -1.7 |
| Bias | 4.0 |
| Noise standard deviation | 0.5 |

The Python, JavaScript, C++17, and Java 17 implementations generate separate training and testing datasets.

The PostgreSQL implementation stores deterministic training and testing records using relational sample tables.

The same mathematical model is used to make optimization results interpretable across implementations.

## 13. Evaluation Metrics

### Mean Squared Error

\[
MSE=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2
\]

Lower MSE indicates smaller average squared prediction errors.

### Coefficient of Determination

\[
R^2=
1-
\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}
\]

An R-squared value of 1 represents perfect predictions.

A negative value indicates performance worse than predicting the evaluation-set mean under this metric.

R-squared is not restricted to the interval from zero to one.

## 14. Computational Complexity

Let:

- \(m\) = number of training samples.
- \(n\) = number of features.
- \(B\) = mini-batch size.
- \(E\) = number of epochs.

The principal computational costs are:

| Strategy | Updates per epoch | Approximate epoch cost |
|---|---:|---:|
| Batch | 1 | \(O(mn)\) |
| Stochastic | m | \(O(mn)\) |
| Mini-Batch | \(\lceil m/B\rceil\) | \(O(mn)\) |

All three strategies process the complete training dataset once per epoch in these implementations.

Their computational differences arise primarily from gradient accumulation and parameter-update frequency.

The model's parameter storage requires \(O(n)\) memory.

Batch Gradient Descent requires storage for the complete dataset or an equivalent full-dataset representation.

Stochastic Gradient Descent can process observations individually.

Mini-Batch Gradient Descent requires storage for the current batch, approximately \(O(Bn)\) for its feature representation when materialized separately.

In practice, vectorization, memory bandwidth, data-loading costs, hardware acceleration, and batch construction can materially affect execution time.

## 15. Edge Cases and Failure Handling

The implementations account for several operational problems.

| Condition | Handling |
|---|---|
| Empty dataset | Reject training |
| Missing features | Reject sample |
| Inconsistent feature dimensions | Reject dataset |
| Non-finite input | Reject input |
| Non-positive learning rate | Reject configuration |
| Invalid batch size | Reject configuration |
| Negative regularization | Reject configuration |
| Extremely large gradient | Apply clipping |
| Non-finite loss | Stop with numerical failure |
| Empty evaluation data | Reject evaluation |
| Incomplete final mini-batch | Process remaining samples |
| Insufficient convergence | Apply patience-based stopping |
| CSV write failure | Report execution failure |

A gradient clipping threshold can reduce extreme updates, but it cannot guarantee numerical stability for every learning rate or dataset.

Feature scaling is also important. Features with substantially different magnitudes can produce poorly conditioned optimization problems.

## 16. Security and Production Considerations

### Input Validation

All numerical inputs should be validated before training.

Reject:

- NaN values.
- Positive and negative infinity.
- Invalid dimensions.
- Negative regularization values.
- Invalid learning rates.
- Invalid training limits.

### Data Separation

Training and testing records must remain logically separate.

Testing data should not influence parameter updates or training-policy selection.

Validation datasets can be used for model selection and hyperparameter tuning.

### Reproducibility

Random seeds should be recorded with experiment metadata.

Reproducibility also depends on data ordering, implementation details, numerical libraries, hardware, and floating-point behavior.

### Resource Management

For large-scale workloads:

- Use bounded training concurrency.
- Avoid loading entire datasets unnecessarily.
- Monitor memory consumption.
- Limit epoch history retention when appropriate.
- Use checkpointing for long-running experiments.
- Monitor loss and gradient norms.
- Separate numerical computation from audit storage when database workloads become expensive.

### Database Security

The PostgreSQL schema should use least-privilege roles.

Application accounts should not receive unrestricted modification permissions over model history, policies, or training records.

Production deployments should also use controlled procedure execution, connection pooling, transaction monitoring, and durable external failure logging.

## 17. Execution

### Python

Save the first implementation as `gradient_descent.py`.

Run:

`python gradient_descent.py`

### JavaScript

Save the second implementation as `gradient_descent.js`.

Run:

`node gradient_descent.js`

### C++17

Save the third implementation as `gradient_descent.cpp`.

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic gradient_descent.cpp -o gradient_descent`

Execute:

`./gradient_descent`

### Java 17

Save the fourth implementation as `GradientDescentEnterprise.java`.

Compile:

`javac --release 17 GradientDescentEnterprise.java`

Execute:

`java GradientDescentEnterprise`

### PostgreSQL

Execute the fifth implementation using a PostgreSQL 15 or later database.

The script creates the `gradient_lab` schema, inserts sample records, executes all three optimization strategies, evaluates the models, and runs reporting queries.

The database account requires permission to create schemas, types, tables, functions, procedures, triggers, and views.

## 18. Practical Interpretation

The choice between batch, stochastic, and mini-batch gradient descent depends on dataset size, available memory, update frequency, and optimization stability.

Batch Gradient Descent provides a full-dataset gradient at every update.

Stochastic Gradient Descent offers frequent updates and supports incremental processing.

Mini-Batch Gradient Descent provides an operational compromise between the two approaches.

The implementations demonstrate that all three strategies use the same underlying differentiation and parameter-update principles. Their differences arise from how training observations are grouped when estimating gradients.

The resulting experiment records, parameter histories, and evaluation metrics provide a common basis for examining their behavior across programming languages and execution environments.
