# Optimizer Laboratory: Momentum, AdaGrad, RMSProp, Adam, and AdamW

## 1. Technical Scope

This project implements and compares five first-order optimization algorithms used to minimize differentiable objective functions:

- Momentum
- AdaGrad
- RMSProp
- Adam
- AdamW

The implementations examine how optimizers use gradient history, adapt learning rates, maintain internal state, and control parameter updates.

The project contains five independent implementations:

| File | Engineering perspective | Main functionality |
|---|---|---|
| Python | Numerical experimentation | Optimizer implementations, gradient verification, convergence comparison |
| JavaScript | Event-driven execution | Asynchronous training, event monitoring, cancellation, optimizer registry |
| C++17 | Numerical computation | Polymorphic optimizer engine and vector-based training |
| Java 17 | Enterprise architecture | Immutable configurations, domain types, optimizer policies, lifecycle management |
| PostgreSQL | Relational data management | Experiment records, configurations, training jobs, checkpoints, metrics, and audit history |

Each implementation independently models the five optimization algorithms.

## 2. Mathematical Foundations

### 2.1 Optimization Problem

An optimizer minimizes an objective function with respect to a parameter vector.

Let:

- \(\theta_t\) represent the model parameters at iteration \(t\).
- \(J(\theta_t)\) represent the objective function.
- \(g_t\) represent the gradient.
- \(\eta\) represent the learning rate.

The gradient is:

\[
g_t = \nabla_\theta J(\theta_t)
\]

The basic gradient descent update is:

\[
\theta_{t+1} = \theta_t - \eta g_t
\]

The gradient indicates the direction of greatest local increase in the objective. Subtracting it moves the parameters in the direction of local decrease.

The learning rate determines the magnitude of the update.

A learning rate that is too large can cause unstable updates or divergence. A learning rate that is too small can make convergence unnecessarily slow.

### 2.2 Objective Function Used in the Implementations

The implementations use a positive-definite quadratic objective:

\[
J(\theta) =
\frac{1}{2}
\sum_{i=1}^{d}
a_i(\theta_i-y_i)^2
\]

Where:

- \(d\) is the number of parameters.
- \(a_i\) is the positive curvature associated with parameter \(i\).
- \(y_i\) is the target parameter value.
- \(\theta_i\) is the current parameter value.

Its gradient is:

\[
\frac{\partial J}{\partial \theta_i}
=
a_i(\theta_i-y_i)
\]

Different curvature values create different gradient magnitudes across dimensions.

This provides a controlled environment for observing adaptive learning rates and optimization behavior.

The objective is deterministic and has a known minimum at:

\[
\theta_i=y_i
\]

This makes it suitable for testing numerical correctness and convergence.

## 3. Momentum

### 3.1 Concept

Momentum improves ordinary gradient descent by incorporating information from previous gradients.

It maintains a velocity vector that accumulates gradient information over time.

The velocity equation is:

\[
v_t = \beta v_{t-1}+g_t
\]

The parameter update is:

\[
\theta_{t+1}=\theta_t-\eta v_t
\]

Where:

- \(v_t\) is the current velocity.
- \(\beta\) is the momentum coefficient.
- \(g_t\) is the current gradient.
- \(\eta\) is the learning rate.

The usual momentum coefficient is close to 0.9.

### 3.2 Mechanism

Momentum reduces sensitivity to individual gradient fluctuations.

When consecutive gradients point in similar directions, the velocity accumulates their contribution.

When gradients alternate directions, their contributions can partially cancel.

This can improve progress in optimization problems containing elongated valleys or noisy gradients.

### 3.3 Implementation

The five implementations maintain a velocity vector initialized to zero.

Each parameter has an associated velocity value.

The state is updated after every gradient evaluation and retained across training iterations.

Momentum does not maintain a separate estimate of squared gradients.

### 3.4 Limitations

- Performance depends on the momentum coefficient.
- A large learning rate can still cause divergence.
- The velocity introduces additional optimizer state.
- Momentum does not independently adapt the learning rate of each parameter.

## 4. AdaGrad

### 4.1 Concept

AdaGrad adapts the learning rate separately for each parameter by accumulating historical squared gradients.

Its accumulated gradient state is:

\[
G_t=G_{t-1}+g_t^2
\]

The parameter update is:

\[
\theta_{t+1}
=
\theta_t-
\frac{\eta}{\sqrt{G_t}+\epsilon}g_t
\]

The operations involving vectors are element-wise.

The epsilon value prevents division by zero.

### 4.2 Mechanism

Parameters receiving consistently large gradients accumulate larger squared-gradient values.

Their effective learning rates decrease.

Parameters receiving smaller or infrequent gradients retain relatively larger effective learning rates.

This makes AdaGrad particularly relevant to sparse-gradient problems.

### 4.3 Important Distinction

AdaGrad accumulates squared gradients indefinitely.

The accumulator does not discard older observations.

As training progresses, the denominator can become increasingly large, reducing parameter updates.

This is an important limitation when training for many iterations.

### 4.4 Implementation

Each implementation maintains an accumulator with the same dimensions as the parameter vector.

For every update:

1. Calculate the squared gradient.
2. Add it to the accumulated value.
3. Calculate the adjusted learning rate.
4. Update the corresponding parameter.

The accumulator persists throughout the training run.

## 5. RMSProp

### 5.1 Concept

RMSProp addresses the continually increasing accumulator used by AdaGrad.

Instead of accumulating every squared gradient equally, it calculates an exponentially weighted moving average.

The equation is:

\[
v_t=\beta v_{t-1}+(1-\beta)g_t^2
\]

The parameter update is:

\[
\theta_{t+1}
=
\theta_t-
\frac{\eta}{\sqrt{v_t}+\epsilon}g_t
\]

### 5.2 Mechanism

The moving average gives greater influence to recent squared gradients.

Older observations progressively lose influence.

This allows the optimizer to respond to changing gradient magnitudes without allowing its denominator to grow indefinitely.

### 5.3 Difference from AdaGrad

| Property | AdaGrad | RMSProp |
|---|---|---|
| Gradient history | Cumulative | Exponential moving average |
| Old gradients | Retained permanently | Gradually discounted |
| Learning-rate adaptation | Per parameter | Per parameter |
| Long-term accumulator growth | Generally increases | Bounded for bounded gradients |
| Typical application | Sparse-gradient problems | Nonstationary optimization |

### 5.4 Implementation

The implementations maintain a squared-gradient moving average.

The beta coefficient controls the contribution of previous state.

A value close to one gives more influence to historical observations.

RMSProp does not require Adam-style first-moment bias correction in the implementation provided here.

## 6. Adam

### 6.1 Concept

Adam combines momentum-like first-moment estimation with RMSProp-like second-moment estimation.

It maintains two independent state vectors.

The first moment estimates the average gradient:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

The second moment estimates the average squared gradient:

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

### 6.2 Bias Correction

Both moving averages are initially zero.

Their early estimates are therefore biased toward zero.

Adam corrects this initialization bias:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t}
\]

\[
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

The final parameter update is:

\[
\theta_{t+1}
=
\theta_t-
\frac{\eta\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

### 6.3 Mechanism

The first moment incorporates directional information from previous gradients.

The second moment estimates the magnitude of recent gradients.

The combination allows Adam to adjust updates using both gradient direction and gradient scale.

### 6.4 Default Parameters

Common initial settings are:

| Parameter | Common value | Purpose |
|---|---:|---|
| Learning rate | 0.001 | Controls update magnitude |
| Beta 1 | 0.9 | First-moment decay |
| Beta 2 | 0.999 | Second-moment decay |
| Epsilon | 1e-8 | Numerical stability |

The benchmark scripts use different learning rates for individual experiments to demonstrate optimization behavior on the selected quadratic objective.

These experimental values are not universal recommendations.

### 6.5 Implementation Considerations

The implementation must use the correct iteration number when calculating bias correction.

The first and second moments must retain their state between updates.

Resetting either moment at every iteration would change the algorithm.

Adam requires two additional vectors of optimizer state.

## 7. AdamW

### 7.1 Concept

AdamW introduces decoupled weight decay.

It separates weight decay from the adaptive gradient update.

The adaptive Adam update is:

\[
\theta_{t+1}^{Adam}
=
\theta_t-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

The decoupled weight-decay operation is:

\[
\theta_{t+1}
=
\theta_{t+1}^{Adam}
-
\eta\lambda\theta_t
\]

Where:

- \(\lambda\) represents the weight-decay coefficient.
- \(\eta\) represents the learning rate.

Equivalently:

\[
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

### 7.2 Decoupled Weight Decay

Weight decay reduces parameter magnitudes independently of the adaptive gradient scaling.

This distinction is important because directly adding an L2 regularization term to the gradient causes that term to be processed by Adam's adaptive moment calculations.

AdamW applies the decay independently.

### 7.3 Adam and AdamW Compared

| Property | Adam | AdamW |
|---|---|---|
| First moment | Yes | Yes |
| Second moment | Yes | Yes |
| Bias correction | Yes | Yes |
| Adaptive gradient update | Yes | Yes |
| Decoupled weight decay | No | Yes |
| Separate weight-decay coefficient | Not required | Supported |

### 7.4 Implementation

The AdamW implementations reuse the first- and second-moment calculations from Adam.

After obtaining the adaptive update, they subtract the decoupled decay term using the original parameter values.

The original parameters must be used for the decay calculation.

This ensures that the decay operation is based on the parameter state before the current update.

### 7.5 Practical Implications

AdamW is particularly relevant when parameter regularization is required with adaptive optimization.

The decay coefficient should be selected carefully.

Excessive decay can shrink parameters too aggressively and impair model performance.

The quadratic benchmark demonstrates the mathematical update mechanism. It does not establish that AdamW will achieve a lower validation loss on every machine-learning task.

## 8. Comparison of the Five Optimizers

| Characteristic | Momentum | AdaGrad | RMSProp | Adam | AdamW |
|---|---|---|---|---|---|
| First-moment state | Velocity | No | No | Yes | Yes |
| Second-moment state | No | Cumulative | Moving average | Yes | Yes |
| Bias correction | No | No | No | Yes | Yes |
| Adaptive per-parameter scaling | No | Yes | Yes | Yes | Yes |
| Decoupled weight decay | No | No | No | No | Yes |
| Main distinction | Directional accumulation | Historical squared-gradient accumulation | Recent squared-gradient averaging | First and second moments | Adam with decoupled decay |

### 8.1 State Complexity

For \(d\) parameters:

| Optimizer | Additional state |
|---|---|
| Momentum | \(d\) velocity values |
| AdaGrad | \(d\) accumulated squared-gradient values |
| RMSProp | \(d\) moving-average values |
| Adam | \(2d\) moment values |
| AdamW | \(2d\) moment values |

The parameter vector itself requires \(O(d)\) memory.

All five update implementations require \(O(d)\) computation per iteration.

For a training process with \(T\) iterations, the optimizer-update computation is generally:

\[
O(Td)
\]

This excludes the cost of evaluating the objective and calculating its gradient.

## 9. Learning Rate and Numerical Stability

### 9.1 Learning Rate

The learning rate is a major determinant of optimization behavior.

An excessively large learning rate can cause:

- Oscillation.
- Overshooting.
- Increasing loss.
- Numerical overflow.
- Divergence.

An excessively small learning rate can cause:

- Slow convergence.
- Excessive training time.
- Limited progress within the available iteration budget.

The learning rate should be evaluated together with the optimizer and the objective function.

### 9.2 Epsilon

The epsilon parameter is a small positive constant used in adaptive denominators.

For example:

\[
\sqrt{v_t}+\epsilon
\]

It prevents division by zero when the squared-gradient estimate is zero or very small.

Its value can influence numerical behavior, especially when gradient magnitudes are extremely small.

### 9.3 Gradient Clipping

The implementations support global norm clipping.

Given a gradient vector \(g\), its Euclidean norm is:

\[
\|g\|_2=\sqrt{\sum_i g_i^2}
\]

For clipping threshold \(c\):

\[
g' =
\begin{cases}
g, & \|g\|_2\leq c\\
g\frac{c}{\|g\|_2}, & \|g\|_2>c
\end{cases}
\]

This constrains the magnitude of an unusually large gradient while preserving its direction.

Gradient clipping is useful in training scenarios where unstable or exceptionally large gradients occur.

It does not guarantee convergence.

## 10. Python Implementation

### 10.1 Architecture

The Python implementation uses an abstract optimizer base class.

The five concrete optimizers implement their own state-update mechanisms.

The common base class handles:

- Configuration validation.
- Parameter and gradient dimension checks.
- Non-finite gradient rejection.
- Global gradient clipping.
- Shared numerical validation.

The `OptimizerConfig` dataclass centralizes configuration.

The `OptimizerName` enumeration identifies supported algorithms.

### 10.2 Numerical Objective

`QuadraticObjective` provides:

- Objective evaluation.
- Analytical gradient calculation.
- Dimension validation.
- Positive-curvature validation.

The `TrainingResult` dataclass stores the final parameter vector, initial loss, final loss, iteration count, and loss history.

### 10.3 Gradient Verification

The numerical gradient checker uses central finite differences:

\[
\frac{\partial J}{\partial\theta_i}
\approx
\frac{
J(\theta+\epsilon e_i)
-
J(\theta-\epsilon e_i)
}{
2\epsilon
}
\]

Here, \(e_i\) is a unit vector in dimension \(i\).

The numerical result is compared with the analytical gradient.

This detects common implementation mistakes in objective derivatives.

### 10.4 Testing

The test routine verifies:

- Every optimizer produces correctly sized output.
- Invalid learning rates are rejected.
- Non-finite gradients are rejected.
- Parameter and gradient dimensions must match.
- AdamW reduces parameter magnitudes when gradients are zero and weight decay is positive.
- Numerical gradients agree with analytical gradients.

## 11. JavaScript Implementation

### 11.1 Event-Driven Architecture

The JavaScript implementation uses Node.js event-driven capabilities.

Its `EventBus` provides subscriptions for training events.

Supported events include:

- `iteration`
- `completed`
- `error`

A listener can observe the current loss and parameter values without becoming part of the optimizer calculation.

### 11.2 Optimizer Registry

`OptimizerRegistry` uses a JavaScript `Map` to associate algorithm names with optimizer factories.

This separates optimizer selection from the training engine.

An unsupported optimizer name produces a validation error.

### 11.3 Asynchronous Training

`TrainingEngine` provides an asynchronous training workflow.

The engine periodically yields to the Node.js event loop using `setImmediate`.

This allows other event-loop operations to execute during a long optimization process.

The engine supports cancellation through its `cancel()` method.

The cancellation mechanism is checked between iterations.

### 11.4 Failure Handling

The implementation distinguishes between:

- Configuration validation failures.
- Dimension mismatches.
- Non-finite objective values.
- Unsupported optimizer selection.
- Training execution failures.

A `finally` block restores the running state even if training fails.

### 11.5 Parallel Experiment Execution

The benchmark uses `Promise.all` to execute independent optimizer experiments.

Each experiment receives its own optimizer instance and maintains independent internal state.

Parallel execution is an orchestration mechanism. It does not imply that JavaScript numerical calculations automatically execute on multiple CPU cores.

## 12. C++17 Implementation

### 12.1 Object-Oriented Design

The C++ implementation defines an abstract `Optimizer` class.

Concrete implementations override:

- `update()`
- `name()`

The training engine depends on the optimizer interface rather than individual algorithm implementations.

This allows runtime selection through polymorphism.

### 12.2 Memory and State Management

The implementation uses:

- `std::vector<double>` for parameter and gradient storage.
- `std::unique_ptr` for optimizer ownership.
- `std::make_unique` for controlled object creation.
- `std::move` for efficient transfer of vector ownership.
- STL algorithms and containers for result processing.

Each optimizer maintains its own state.

The optimizer state must not be shared between independent training runs.

### 12.3 Configuration Validation

The constructor validates:

- Learning rate.
- Epsilon.
- Weight decay.
- Gradient clipping threshold.

Vector operations validate finite values and compatible dimensions.

Invalid configurations produce `std::invalid_argument`.

Non-finite objective values are rejected during training.

### 12.4 Engineering Trade-Offs

The polymorphic design provides extensibility but introduces virtual dispatch.

For small numerical vectors, this overhead is generally insignificant compared with the mathematical calculations.

For very large workloads, memory allocation, vectorization, cache behavior, and hardware acceleration become more important performance considerations.

## 13. Java 17 Implementation

### 13.1 Enterprise-Oriented Domain Model

The Java implementation separates:

- Optimizer configuration.
- Objective definition.
- Optimizer policies.
- Training service.
- Training status.
- Training results.
- Validation and training exceptions.

Java records represent immutable configuration, objective, and result data.

The optimizer implementations maintain mutable internal state because optimization requires state across iterations.

### 13.2 Optimizer Policy Interface

`OptimizerPolicy` defines the common optimizer contract.

The interface exposes:

- Parameter updates.
- Optimizer type.
- Step count.

The factory creates implementations using the selected optimizer type.

This allows the training service to operate without knowing the internal equations of each algorithm.

### 13.3 Training Lifecycle

The `TrainingService` models explicit lifecycle states:

| State | Meaning |
|---|---|
| INITIALIZED | Training has not started |
| RUNNING | Parameter updates are being executed |
| COMPLETED | Training has finished successfully |
| FAILED | Training has encountered an execution failure |

The service validates its inputs and records the final state.

A synchronized training method prevents simultaneous execution through the same service instance.

Independent experiments should use separate service and optimizer instances.

### 13.4 Immutability and Validation

Records provide immutable configuration objects.

`List.copyOf` protects stored configuration and result collections against external structural modification.

The optimizer state remains internally mutable.

This combination separates externally supplied configuration from operational training state.

## 14. PostgreSQL Implementation

### 14.1 Relational Architecture

The PostgreSQL implementation models the operational lifecycle of optimization experiments.

The relational structure is:

Experiment Owner
    |
    v
Experiment
    |
    v
Optimizer Configuration
    |
    v
Training Run
    |
    +-------------------+
    |                   |
    v                   v
Parameter Checkpoints   Training Metrics
    |
    v
Training Events

### 14.2 Main Tables

| Table | Purpose |
|---|---|
| `experiment_owner` | Identifies the responsible research or engineering group |
| `experiment` | Stores the optimization experiment definition |
| `optimizer_configuration` | Stores algorithm-specific hyperparameters |
| `training_run` | Represents an individual execution |
| `parameter_checkpoint` | Stores parameter snapshots |
| `training_metric` | Stores iteration-level training metrics |
| `training_event` | Records lifecycle events |

### 14.3 Optimizer Configuration

The configuration table stores:

- Optimizer type.
- Learning rate.
- Epsilon.
- Beta 1.
- Beta 2.
- Weight decay.
- Gradient clipping threshold.

Check constraints reject invalid configuration values.

The configuration is associated with an experiment through a foreign key.

The composite foreign key on `training_run` ensures that the selected configuration belongs to the same experiment.

### 14.4 Training State Management

The `training_status` enumeration represents the execution lifecycle.

A training run can move through:

CREATED
   |
   v
QUEUED
   |
   v
RUNNING
   |
   +------------------+
   |         |        |
   v         v        v
COMPLETED  FAILED  CANCELLED

The trigger `enforce_training_transitions` prevents changes from terminal states to different states.

It also sets the start and completion timestamps.

### 14.5 Checkpoint Integrity

The `parameter_checkpoint` table stores the parameter vector as a PostgreSQL `DOUBLE PRECISION[]` array.

The checkpoint validation trigger compares the number of stored parameters with the experiment's declared objective dimensions.

The trigger also rejects non-finite parameter values.

The `record_checkpoint` procedure:

1. Locks the relevant training run.
2. Confirms that the run exists.
3. Verifies that the run is active.
4. Checks iteration progression.
5. Inserts the checkpoint.
6. Records the corresponding metric.
7. Updates the training run.

The procedure executes within the caller's database transaction.

The row-level lock prevents competing checkpoint transactions from independently advancing the same run based on an identical previous state.

### 14.6 Audit Events

Training lifecycle changes automatically generate records in `training_event`.

The event table provides an operational history of:

- Job creation.
- Job execution.
- Completion.
- Failure.
- Cancellation.

The JSONB field stores event-specific details.

This separates lifecycle auditing from the current status stored in `training_run`.

### 14.7 Reporting and Aggregation

The `optimizer_run_report` view combines experiment, owner, configuration, and execution information.

The `optimizer_convergence_report` view aggregates:

- Total training runs.
- Completed runs.
- Failed runs.
- Average initial loss.
- Average final loss.
- Average iteration count.

The reporting queries use joins, aggregate functions, conditional aggregation, CTEs, and window functions.

### 14.8 Performance Considerations

Indexes support:

- Owner and experiment status retrieval.
- Optimizer configuration searches.
- Training status monitoring.
- Recent experiment execution queries.
- Checkpoint history retrieval.
- Metric history retrieval.
- Event history searches.

The GIN index on the JSONB event details supports searches involving structured event attributes.

The uniqueness constraint on run and iteration prevents duplicate checkpoint or metric records for the same iteration.

### 14.9 Database Limitations

PostgreSQL stores the experimental records but does not implement the numerical optimizer calculations.

The Python, JavaScript, C++17, and Java 17 programs perform numerical updates.

The SQL procedures manage execution records, checkpoint consistency, and reporting.

This separation avoids treating a relational database as a numerical computing engine.

## 15. Cross-Language Implementation Comparison

| Engineering concern | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Optimizer abstraction | Abstract base class | Base class | Virtual interface | Interface | Optimizer enum and configuration |
| Parameter storage | Lists | Arrays | STL vectors | Lists | Numeric arrays |
| Configuration | Dataclass | Object | Struct | Record | Relational table |
| State management | Instance attributes | Instance attributes | Class members | Class members | Persistent tables |
| Validation | Exceptions and assertions | Custom errors | Standard exceptions | Custom exceptions | Constraints and triggers |
| Training control | Iteration loop | Async engine | Training engine | Training service | Stored procedures |
| Reporting | Dataclasses and console | Event-driven reports | Result structures | Result records | Views and aggregates |
| Testing | Assertions | Runtime tests | Runtime tests | Runtime tests | Constraints and transactional operations |

## 16. Practical Experimental Considerations

### 16.1 Optimizer Selection

No optimizer is universally optimal.

Selection depends on:

- Objective-function characteristics.
- Gradient noise.
- Parameter dimensionality.
- Dataset size.
- Gradient sparsity.
- Memory constraints.
- Training duration.
- Generalization requirements.

Momentum is useful when gradient direction remains reasonably consistent.

AdaGrad can be useful when gradient sparsity is important.

RMSProp handles changing gradient magnitudes through moving averages.

Adam combines directional and magnitude estimates.

AdamW is relevant when decoupled weight decay is required.

### 16.2 Training Loss and Validation Loss

Training loss measures objective performance on the training data.

Validation loss measures performance on separate validation data.

A decrease in training loss does not guarantee improved generalization.

The implementations use a deterministic quadratic objective rather than a real dataset.

Their numerical results therefore evaluate algorithm mechanics, not predictive performance on an application dataset.

### 16.3 Hyperparameter Evaluation

A meaningful optimizer comparison should control experimental conditions.

Recommended experimental controls include:

- Identical model initialization.
- Identical data splits.
- Identical objective definitions.
- Consistent gradient calculation.
- Clearly specified learning rates.
- Consistent stopping criteria.
- Multiple random seeds where stochastic training is involved.
- Validation metrics.
- Recorded computational time.

Comparing algorithms with arbitrary and substantially different hyperparameters can produce misleading conclusions.

### 16.4 Checkpointing

A checkpoint should preserve enough state to resume training correctly.

Saving only model parameters is insufficient for stateful optimizers.

A complete optimizer checkpoint should include:

| Optimizer | Required state |
|---|---|
| Momentum | Velocity vector and iteration count |
| AdaGrad | Squared-gradient accumulator |
| RMSProp | Squared-gradient moving average |
| Adam | First moment, second moment, iteration count |
| AdamW | First moment, second moment, iteration count, weight-decay configuration |

The SQL schema records model parameter checkpoints and metrics. It does not serialize every internal optimizer state vector required for exact continuation.

A production checkpointing implementation must persist those additional values.

## 17. Security and Reliability

Numerical optimization systems can encounter invalid data, corrupted checkpoints, interrupted execution, and resource exhaustion.

The implementations address several relevant risks:

- Reject non-finite parameters and gradients.
- Validate configuration before training.
- Enforce vector dimensional consistency.
- Separate independent optimizer instances.
- Detect non-finite objective values.
- Record training failures.
- Restrict invalid lifecycle transitions.
- Preserve checkpoint consistency through transactions.
- Validate database relationships through foreign keys.
- Use parameterized database operations when connecting an application to PostgreSQL.

Production systems should also consider:

- Authentication and authorization.
- Database role separation.
- Encryption of sensitive training data.
- Backup and recovery.
- Training resource limits.
- Checkpoint retention.
- Experiment reproducibility.
- Monitoring and alerting.
- Access controls for experiment records.

## 18. Execution and Verification

### Python

Runtime requirement: Python 3.10 or later.

File: `optimizer_lab.py`

Execution:

`python optimizer_lab.py`

The script uses the Python standard library.

### JavaScript

Runtime requirement: Node.js 18 or later.

File: `optimizer_lab.js`

Execution:

`node optimizer_lab.js`

No npm packages are required.

### C++17

Compiler requirement: C++17-compatible compiler.

File: `optimizer_lab.cpp`

Compilation:

`g++ -std=c++17 -O2 -Wall -Wextra optimizer_lab.cpp -o optimizer_lab`

Execution:

`./optimizer_lab`

### Java 17

Runtime requirement: JDK 17.

File: `OptimizerEnterprise.java`

Compilation:

`javac OptimizerEnterprise.java`

Execution:

`java OptimizerEnterprise`

### PostgreSQL

Database requirement: PostgreSQL 15 or later.

File: `optimizer_lab.sql`

Execution:

`psql -d your_database -f optimizer_lab.sql`

The script creates the `optimizer_lab` schema and its database objects.

It should be executed using a role with the necessary schema, table, function, procedure, and trigger permissions.

The script contains its own transaction and sample records.

## 19. Interpretation of Experimental Results

The benchmark reports:

- Initial objective loss.
- Final objective loss.
- Number of iterations.
- Final parameter values.
- Loss history where supported.

The percentage improvement is:

\[
\text{Improvement} =
\frac{J_{\text{initial}}-J_{\text{final}}}
{J_{\text{initial}}}\times100
\]

A higher percentage indicates a greater reduction in the selected objective.

This metric does not measure execution speed, memory efficiency, or generalization.

A rigorous evaluation should consider all of these dimensions separately.

The most important distinction across the five algorithms is how they maintain and use gradient history. Momentum accumulates directional information, AdaGrad accumulates squared gradients, RMSProp discounts older squared gradients, Adam combines first- and second-moment estimates with bias correction, and AdamW separates weight decay from adaptive gradient scaling.
