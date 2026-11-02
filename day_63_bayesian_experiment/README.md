# Bayesian Optimization

## Scope

This project implements Bayesian Optimization as a sequential strategy for optimizing an expensive or black-box objective.

The technical model separates three responsibilities:

| Component | Responsibility | Implementation |
|---|---|---|
| Objective | Produces an observed value for a candidate point | Deterministic black-box function with controlled pseudo-noise |
| Surrogate model | Estimates the objective and uncertainty | Gaussian Process with an RBF kernel |
| Acquisition function | Selects the next experiment | Expected Improvement |

The optimization loop is:

`initial observations → Gaussian Process → acquisition evaluation → next candidate → objective evaluation → updated observations`

The implementations use a two-dimensional bounded search domain so that acquisition optimization can be performed with an explicit candidate grid.

## Bayesian Optimization Model

Bayesian Optimization is useful when evaluating the objective is more expensive than evaluating the optimization logic itself. Instead of testing every possible configuration, the optimizer maintains a probabilistic approximation of the objective and chooses new evaluations using both predicted performance and uncertainty.

For a candidate `x`, the Gaussian Process produces:

- `μ(x)`: predicted mean objective value
- `σ(x)`: predictive standard deviation

The surrogate therefore represents both an expected outcome and uncertainty about that outcome.

The implementation uses the RBF kernel:

`k(x, x') = σ² exp(-||x-x'||² / (2l²))`

where `l` controls the length scale and `σ²` controls the signal variance.

A small diagonal noise term is added to the kernel matrix. This models noisy observations and improves numerical stability when the matrix is close to singular.

## Surrogate Model

The Gaussian Process is fitted from the accumulated observations.

The implementation constructs the kernel matrix `K`, adds observation noise, and computes a Cholesky factorization:

`K = L Lᵀ`

The Cholesky representation avoids explicitly calculating a matrix inverse. Forward and backward substitution are then used to obtain the coefficients required for posterior prediction.

For a candidate point, the implementation calculates the kernel vector between that candidate and all observed points. The posterior mean is obtained from the kernel vector and the fitted coefficients. Predictive variance is reduced near observed locations and remains larger in regions with limited evidence.

The Python, C++, and Java implementations use the same mathematical structure while expressing the numerical operations with language-specific data structures.

## Acquisition Strategy

The acquisition function is Expected Improvement.

For a maximization problem, the implementation evaluates the expected amount by which a candidate can improve upon the best observed objective.

The acquisition function considers two properties:

- exploitation through a high predicted mean
- exploration through a high predictive standard deviation

The exploration parameter is `0.01` in the executable examples.

A candidate with a high predicted value can receive a high acquisition score even when uncertainty is modest. A candidate with substantial uncertainty can also become attractive because the unknown objective might substantially exceed the current best value.

Candidates already evaluated are excluded before acquisition selection.

## Optimization Strategy

The executable optimization process contains two phases.

### Initial exploration

A fixed-seed random generator produces an initial set of points inside the configured bounds.

The default examples use six initial observations.

### Sequential selection

Each subsequent iteration performs:

1. Fit the Gaussian Process to all accepted observations.
2. Determine the current best objective.
3. Generate a bounded candidate grid.
4. Predict the mean and uncertainty for each unexplored candidate.
5. Calculate Expected Improvement.
6. Select the candidate with the highest acquisition value.
7. Evaluate the black-box objective.
8. Add the new observation to the history.

The examples use 18 sequential iterations after initialization.

The grid is deliberately explicit rather than hidden behind an optimization library. This makes the acquisition-selection mechanism observable and reproducible.

## Python Implementation

The Python implementation is a complete command-line program built only from the standard library.

Important components include:

- `Bounds` for domain validation
- `Observation` as an immutable dataclass
- `GaussianProcess` for surrogate fitting and prediction
- `AcquisitionFunction` for Expected Improvement
- `BayesianOptimizer` for sequential optimization
- CSV export for experiment history
- assertions for result validation
- deterministic random initialization

The Python implementation also supports both maximization and minimization through `OptimizationDirection`. Minimization is converted internally into maximization by transforming the observed objective values.

The script writes `bayesian_optimization_history.csv` after optimization.

## JavaScript Implementation

The JavaScript implementation is designed around Node.js event-driven behavior.

`RepositoryLikeExperiment` extends `EventEmitter` and emits events when:

- an observation is recorded
- a new acquisition candidate is selected

Objective evaluation is represented with a Promise and `setImmediate`. This models an expensive or remote evaluation without introducing external dependencies.

The implementation uses:

- JavaScript classes
- `Map`/`Set`-style collection techniques
- Promises
- asynchronous evaluation
- event emission
- filesystem output through `node:fs/promises`
- deterministic pseudo-random initialization
- explicit validation

The asynchronous boundary makes the implementation suitable as a conceptual starting point for experiments where objective evaluation could involve a service call, simulation, benchmark execution, or external measurement.

## C++17 Implementation

The C++ implementation treats Bayesian Optimization as a numerical engineering case study.

The primary classes are:

- `GaussianProcess`
- `Acquisition`
- `GovernanceOptimizer`

The GP uses `std::vector` for matrix and observation storage and performs Cholesky factorization without an external numerical library.

The acquisition calculation uses standard-library mathematical functions including `std::erf`, `std::exp`, and `std::sqrt`.

The optimizer uses `std::mt19937` for reproducible initialization and explicitly generates a two-dimensional candidate grid.

CSV output is implemented with standard C++ file streams.

The implementation checks invalid bounds, invalid hyperparameters, empty observations, non-positive-definite kernel matrices, unavailable candidates, and file-output failures.

## Java 17 Implementation

The Java implementation uses an enterprise-oriented service structure.

The domain model includes:

- `Bounds`
- `Observation`
- `Prediction`
- `Direction`
- `Objective`
- `GaussianProcess`
- `ExpectedImprovement`
- `OptimizationService`

Java records provide immutable domain values for bounds, observations, and predictions.

`OptimizationService` owns the optimization workflow and validates state transitions such as:

`configuration → initial observations → surrogate fitting → acquisition selection → new observation → best result`

`InvalidOptimizationStateException` separates workflow-state failures from ordinary argument validation.

The service exports the completed observation history as CSV and rejects candidates outside configured bounds.

The implementation uses Java collections, streams for result selection, immutable `List.copyOf` values, a domain-specific exception, and explicit service boundaries.

## PostgreSQL Model

The PostgreSQL implementation represents an optimization experiment as relational data rather than executing the numerical Gaussian Process itself.

The database separates:

- experiments
- optimization dimensions
- acquisition strategies
- objective observations
- surrogate predictions
- optimization decisions

The core relationship is:

`experiment → dimensions`
  
`experiment → observations`
  
`experiment → acquisition strategy`
  
`experiment → surrogate predictions`
  
`experiment → optimization decisions`

This separation allows the database to retain both experimental evidence and the decisions produced by an optimization process.

## Database Constraints

The schema uses database-level validation for important invariants.

`experiments` validates:

- positive length scale
- positive signal variance
- non-negative noise variance
- non-negative exploration weight
- positive evaluation limits

`dimensions` requires the lower bound to be smaller than the upper bound.

`observations` requires a JSON array for the candidate vector and requires an objective value for accepted observations.

A trigger validates that the number of values in an observation matches the number of configured dimensions.

Another trigger prevents an experiment from entering `completed` state without at least one accepted observation.

## PostgreSQL Workflow

The sample experiment is named `pricing_policy_surface`.

Its two dimensions are:

| Dimension | Lower bound | Upper bound |
|---|---:|---:|
| `price_adjustment` | -4 | 4 |
| `inventory_pressure` | -4 | 4 |

The experiment uses Expected Improvement with an RBF Gaussian Process configuration.

The sample data contains:

- accepted objective evaluations
- a failed evaluation
- surrogate prediction data
- an acquisition decision
- a subsequent accepted observation
- a completed experiment state

The transaction that records the final observation and completes the experiment demonstrates atomic workflow state changes.

## Relational Queries

The SQL script provides queries for several operational views of the optimization process.

The experiment status query reports accepted and failed evaluations.

`best_observation()` selects the best accepted observation according to the experiment's optimization direction.

The aggregation query calculates:

- evaluation count
- mean objective
- minimum objective
- maximum objective
- population standard deviation

`evaluate_merge_candidate()` is used to validate candidate dimensionality and search-domain membership.

The decision query exposes acquisition-driven candidate selections and their acquisition scores.

The dimension query reports the active optimization domain.

## Distinguishing the Core Concepts

The implementation deliberately keeps the main concepts separate.

### Surrogate model

The surrogate estimates the unknown objective.

It does not choose the next experiment.

The Gaussian Process provides both the predicted mean and predictive uncertainty.

### Acquisition function

The acquisition function converts the surrogate's prediction and uncertainty into a decision score.

Expected Improvement is the decision criterion used here.

It does not directly evaluate the expensive objective.

### Optimization strategy

The optimization strategy controls the sequence of actions.

It determines:

- how the initial design is generated
- when the surrogate is refitted
- which candidates are considered
- how acquisition scores are compared
- which candidate is evaluated next
- when optimization stops

The relationship is therefore:

`observations → surrogate → acquisition → candidate selection → new observation`

Changing the surrogate does not necessarily require changing the objective. Changing Expected Improvement to another acquisition function does not change the stored observations.

## Validation and Failure Conditions

The source implementations explicitly handle several invalid states.

### Search-domain violations

Candidates must remain inside the configured bounds.

### Empty training data

The Gaussian Process cannot produce a meaningful posterior before observations exist.

### Numerical instability

Cholesky decomposition rejects a kernel matrix that is not sufficiently positive definite. A small diagonal jitter is used to improve numerical stability.

### Duplicate candidates

Already observed points are excluded from acquisition selection.

### Invalid objective values

Non-finite objective values are rejected.

### Invalid configuration

Bounds, Gaussian Process parameters, iteration counts, and grid sizes are validated before optimization proceeds.

### No unexplored candidate

The optimization process fails explicitly when the candidate grid contains no available point.

## Performance Characteristics

For `n` observations, Gaussian Process training requires construction and factorization of an `n × n` kernel matrix.

The Cholesky factorization is approximately `O(n³)` in time and `O(n²)` in memory.

Prediction for one candidate requires approximately `O(n²)` work when the triangular solve is included.

The grid-based acquisition search adds a factor based on the number of candidate points.

For a two-dimensional grid with `g` points per dimension, there are approximately `g²` candidates.

This explicit grid is practical for the small educational search space used here. A high-dimensional production optimizer would normally require a more efficient acquisition optimizer because grid size grows exponentially with dimensionality.

## Numerical and Modeling Trade-offs

The examples use fixed hyperparameters rather than estimating kernel parameters during every iteration. This keeps the implementation focused on the relationship between the surrogate and acquisition function.

The RBF kernel assumes smooth variation across nearby points. If the real objective contains discontinuities, sharp regime changes, categorical variables, or strong heteroscedastic noise, a different surrogate design may be more appropriate.

The noise variance represents observation uncertainty. Too little noise can make the kernel matrix numerically unstable when observations are nearly duplicated. Too much noise can make the surrogate underfit the observed objective.

The exploration parameter in Expected Improvement affects the balance between finding known high-performing regions and investigating uncertain regions.

## Reproducibility

All executable implementations use deterministic seeds.

The black-box objective includes a deterministic pseudo-noise component rather than a system-time random noise source. This keeps repeated executions comparable while still producing a non-perfectly smooth objective.

The resulting history is written as CSV so that each evaluation can be inspected independently.

## Security and Production Considerations

The example objective is local and controlled. A production Bayesian Optimization system may invoke external simulations, benchmark jobs, databases, or services.

Those integrations should enforce:

- bounded input validation
- explicit resource limits
- evaluation timeouts
- authentication for remote objective services
- authorization for experiment modification
- immutable or auditable experiment history
- validation of externally returned objective values
- protection against untrusted file paths
- separation between experiment configuration and executable code
- rate limits for expensive objective evaluations

The database model separates optimization decisions from observations so that decisions can be audited without rewriting historical measurements.

## Practical Workflow

A realistic optimization service can use the following lifecycle:

`experiment created`
  
`→ search dimensions validated`
  
`→ initial observations collected`
  
`→ Gaussian Process fitted`
  
`→ acquisition evaluated`
  
`→ candidate selected`
  
`→ objective evaluated`
  
`→ observation persisted`
  
`→ surrogate updated`
  
`→ stopping condition checked`
  
`→ best configuration recorded`

The stopping condition can be based on a fixed evaluation budget, elapsed time, acquisition value, improvement threshold, or resource limit.

The implemented examples use a fixed evaluation count.

## File Responsibilities

| File | Primary role |
|---|---|
| Python | Complete sequential Bayesian Optimization implementation |
| JavaScript | Asynchronous and event-driven optimization model |
| C++17 | Numerical implementation with explicit matrix operations |
| Java 17 | Enterprise-oriented domain and service model |
| PostgreSQL | Persistent relational representation of experiments, observations, predictions, and decisions |
| README.md | Technical description of the implemented Bayesian Optimization mechanisms |

The five source implementations are intentionally different expressions of the same optimization architecture rather than literal translations of one another.
