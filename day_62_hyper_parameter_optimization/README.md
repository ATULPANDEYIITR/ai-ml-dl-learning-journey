# Hyperparameter Optimization: Grid Search, Random Search, and Search Spaces

## Scope

This project models hyperparameter optimization as a controlled search problem over a finite configuration space. The implementations use three connected concepts:

- **Search space**: the allowed values and ranges for model hyperparameters.
- **Grid search**: exhaustive evaluation of the Cartesian product of selected values.
- **Random search**: evaluation of randomly selected configurations under a fixed trial budget.

The domain scenario uses three hyperparameters:

| Hyperparameter | Type | Example values | Optimization role |
|---|---|---|---|
| `learning_rate` | Floating-point | `0.005` to `0.2` | Controls the magnitude of each optimization update |
| `max_depth` | Integer | `2`, `4`, `6`, `8`, `10` | Controls model-tree complexity |
| `estimators` | Integer | `50`, `100`, `200`, `400` | Controls the number of model components |

The central workflow is:

Search Space  
→ Configuration Generation  
→ Trial Evaluation  
→ Validation Score  
→ Trial Recording  
→ Best Configuration  
→ Search Strategy Comparison

The implementations use a deterministic synthetic validation surface rather than a machine-learning library. This keeps the behavior executable without external dependencies while preserving the computational structure of hyperparameter optimization.

## Search-Space Design

A search space defines which configurations are legal candidates. In the source implementations, a configuration is a combination of one value from each dimension.

For six learning rates, five depths, and four estimator counts:

`6 × 5 × 4 = 120`

possible configurations exist.

This Cartesian-product property is the defining computational characteristic of grid search. Adding another dimension or adding values to an existing dimension increases the total number of trials multiplicatively.

A useful distinction is:

| Concept | Meaning |
|---|---|
| Hyperparameter | A value selected before or around model training |
| Search-space dimension | One tunable hyperparameter and its candidate values |
| Configuration | One complete combination of hyperparameter values |
| Trial | One evaluation of one configuration |
| Objective metric | Numerical measure used to compare trials |
| Search strategy | Rule used to select configurations |

The implementations validate that search dimensions are non-empty and contain legal values. Duplicate candidates are rejected because they waste trial budget without adding information.

## Grid Search

Grid search evaluates every configuration in the defined Cartesian product.

For the primary configuration space:

`learning_rate × max_depth × estimators`

each possible combination becomes a separate trial.

### Strengths

Grid search is useful when:

- the search space is small;
- candidate values have clear engineering meaning;
- exhaustive coverage is required;
- reproducibility is important;
- the number of combinations is operationally affordable.

### Limitation

Grid search scales poorly as dimensions grow.

For example:

- 6 values across 3 dimensions → 216 configurations
- 10 values across 5 dimensions → 100,000 configurations
- 20 values across 8 dimensions → 25.6 billion configurations

The problem is not the implementation of the loops. The problem is the multiplicative structure of the search space.

The Python implementation uses `itertools.product`, JavaScript uses nested iteration, C++17 uses explicit STL-based loops, Java uses collections, and PostgreSQL can represent the same Cartesian-product model through relational operations.

## Random Search

Random search samples configurations from the same defined search space but evaluates only a specified number of candidates.

The implementations use a fixed trial budget such as `20`.

If a search space contains 120 possible configurations, random search can inspect 20 rather than all 120.

The practical trade-off is:

| Strategy | Coverage | Budget control | Large-space behavior |
|---|---|---|---|
| Grid | Complete | Poor when space is large | Expensive |
| Random | Partial | Strong | More scalable |

Random search does not guarantee that the global optimum will be evaluated. It converts exhaustive coverage into probabilistic exploration.

The source implementations prevent duplicate random configurations by storing configuration keys in sets. This ensures that a requested number of random trials represents distinct candidates when enough unique configurations exist.

## Python Implementation

The Python source models the optimization process with dataclasses and enums.

`SearchSpace` owns the legal candidate dimensions and validates them before optimization begins. `Configuration` represents an immutable candidate. `TrialResult` records the evaluation result, duration, strategy, and configuration.

`SyntheticModelEvaluator` supplies a deterministic validation-quality surface with small controlled noise. The hidden best region is near:

- learning rate `0.05`;
- maximum depth `6`;
- `200` estimators.

This allows search behavior to be evaluated without downloading datasets or installing an ML framework.

`HyperparameterOptimizer` separates candidate generation from evaluation. Grid candidates are generated exhaustively, while random candidates are sampled without duplication.

The script also performs assertions for:

- expected search-space size;
- grid trial count;
- random trial count;
- score bounds;
- invalid search-space rejection.

The separation between candidate generation and evaluation is important because a production system can replace the synthetic evaluator with an actual training-and-validation function without changing the search-space logic.

## JavaScript Implementation

The JavaScript source focuses on asynchronous and event-driven orchestration.

`SearchSpace` validates the candidate dimensions. `Configuration` provides stable configuration keys. `SeededRandom` provides repeatable sampling without requiring an external package.

`TrialEventBus` emits:

- `trialStarted`;
- `trialCompleted`;
- `trialFailed`.

This event model is appropriate for optimization systems where training jobs may run independently and completion order is not necessarily the same as submission order.

`AsyncEvaluator` uses Promises and delayed execution to model asynchronous validation jobs. `Promise.all` coordinates independent trials.

The implementation therefore emphasizes orchestration rather than simply translating the Python loops.

The event-driven structure can support later integration with:

- job queues;
- worker processes;
- training services;
- progress dashboards;
- trial monitoring;
- failure reporting.

The current implementation intentionally remains dependency-free.

## C++17 Implementation

The C++17 source implements the optimization engine as a technical case study centered on explicit data structures and deterministic execution.

`Hyperparameters` represents a configuration. `Trial` stores the resulting score and estimated duration.

`SearchSpace` owns the dimensions and validates the configuration space.

`GovernanceEngine` performs:

- exhaustive grid generation;
- random candidate selection;
- duplicate prevention;
- synthetic evaluation;
- validation-score calculation.

An `unordered_set` with a custom hash prevents repeated random configurations.

The evaluator uses `std::clamp` to ensure scores remain within the valid `[0, 1]` range. `std::max_element` selects the best trial and `std::sort` ranks random-search results.

The design separates configuration generation from evaluation. This makes the search engine independent of the particular model-training implementation.

The primary performance distinction is explicit:

`Grid complexity = product of dimension sizes`

while:

`Random complexity = requested trial budget`

This difference becomes increasingly important when search spaces contain many dimensions.

## Java 17 Implementation

The Java implementation presents hyperparameter optimization as an enterprise-style service.

The domain model uses:

- records for immutable hyperparameter and trial data;
- enums for strategy and trial state;
- an interface for trial evaluation;
- a custom exception for evaluation failures;
- a service class for orchestration;
- immutable search-space lists.

`OptimizationService` owns candidate generation and trial execution.

`TrialEvaluator` provides an abstraction between the optimization workflow and the actual validation mechanism. `ValidationEvaluator` supplies the executable synthetic scoring implementation.

Trial states distinguish:

- `CREATED`;
- `RUNNING`;
- `COMPLETED`;
- `FAILED`.

The current service records failures as failed trial results rather than allowing one bad candidate to terminate the complete optimization run.

Java streams are used for best-trial selection, average-score calculation, and ranking.

This structure reflects an enterprise optimization service where the search algorithm, domain model, evaluation service, persistence layer, and reporting layer can remain separate.

## PostgreSQL Model

The PostgreSQL implementation treats optimization as a relational experiment-management problem.

The schema contains:

- `experiments`;
- `hyperparameters`;
- `search_space_values`;
- `trials`;
- `hpo_trial_events`.

The `experiments` table records the optimization strategy, trial budget, objective metric, and optimization direction.

`hyperparameters` stores the search dimensions, while `search_space_values` stores their candidate values.

`trials` records the actual configurations evaluated and their validation results.

The `hpo_trial_events` table provides an audit trail for state transitions.

### Database-Level Validation

The schema uses:

- primary keys;
- foreign keys;
- unique constraints;
- check constraints;
- indexes;
- enum types;
- triggers.

A completed trial must have a validation score. A completed trial cannot be moved backward into another state.

A trigger records status transitions, allowing operational history to be inspected after optimization runs.

The partial index on completed trials supports best-score queries without indexing incomplete trial records unnecessarily.

## Trial Lifecycle

The relational workflow is:

`queued → running → completed`

or:

`queued → running → failed`

The database trigger records changes in the trial state.

A completed trial represents an evaluation whose objective metric is available. A failed trial represents an attempted configuration whose evaluation did not produce a usable metric.

This distinction matters in production systems because a missing result is not equivalent to a poor model score.

## Objective Evaluation

The examples use validation accuracy as the objective metric and maximize it.

A real optimization service could instead minimize:

- validation loss;
- mean absolute error;
- latency;
- memory consumption;
- resource cost.

The objective direction is stored in the experiment metadata so the database model is not tied permanently to accuracy maximization.

A robust production system should also record the dataset version, code version, random seed, feature configuration, training duration, and evaluation protocol for reproducibility.

## Edge Cases

The implementations account for several important failure conditions.

### Empty Search Dimensions

A dimension with no candidate values cannot generate configurations and is rejected.

### Invalid Hyperparameters

Positive constraints are applied to learning rate, depth, and estimator count.

### Duplicate Random Trials

Random search can sample the same configuration more than once. The implementations use sets or equivalent structures to prevent duplicate candidates within the requested budget.

### Oversized Random Budget

If the requested random trial count exceeds the number of unique configurations, the implementations cap the number of trials at the search-space size.

### Failed Evaluation

A training or validation failure should not necessarily terminate the entire optimization experiment. The Java and PostgreSQL models preserve failed-trial state so failures remain observable.

### No Successful Trials

Best-trial selection raises an explicit failure when no successful evaluation exists.

## Relationship Between Search Space and Budget

Search-space size and trial budget are separate concepts.

For grid search:

`trial count = search-space size`

For random search:

`trial count = min(requested budget, search-space size)`

This distinction is important when comparing optimization strategies. A grid search can consume the entire computational budget simply because the Cartesian product is large, while random search can enforce a fixed maximum number of evaluations.

## Performance Considerations

The dominant cost in hyperparameter optimization is usually model evaluation, not candidate generation.

If one training run costs `T`, grid search over `N` configurations has approximate evaluation cost:

`N × T`

Random search with budget `B` has approximate evaluation cost:

`B × T`

where `B` is normally much smaller than `N`.

The source implementations also estimate duration from model complexity. This is not a real training-time measurement, but it demonstrates why optimization systems should record resource cost alongside objective quality.

In real workloads, trial scheduling can be extended to:

- parallel workers;
- GPU allocation;
- CPU quotas;
- memory limits;
- timeout enforcement;
- retry policies;
- early termination.

## Search-Space Granularity

The chosen values affect what an optimizer can discover.

A grid containing:

`0.01, 0.05, 0.1`

cannot select `0.03`, even if that value would be optimal.

Increasing grid resolution improves candidate coverage but increases computational cost.

For continuous parameters, practical systems often use a meaningful distribution or logarithmic scale instead of evenly spaced values. Learning rates are a common example because useful values can span several orders of magnitude.

The source project keeps the space finite and explicit so grid and random search can be compared deterministically.

## Security and Reliability

Hyperparameter optimization infrastructure should treat configuration data as untrusted input when configurations originate from external users or automated systems.

Important controls include:

- validate numeric bounds;
- reject NaN and infinite values;
- enforce maximum trial budgets;
- restrict resource-intensive configurations;
- isolate training processes;
- enforce execution timeouts;
- record failures;
- authenticate experiment ownership;
- prevent unauthorized modification of completed results;
- preserve immutable experiment history where auditability matters.

A trial budget is also a resource-control mechanism. Without a hard limit, automated optimization can accidentally submit a much larger number of expensive evaluations than intended.

## Practical Distinctions

| Area | Primary question |
|---|---|
| Search space | Which configurations are legal? |
| Grid search | Can every legal configuration be evaluated? |
| Random search | Which limited subset should be sampled? |
| Trial | What happened when one configuration was evaluated? |
| Objective metric | How is configuration quality measured? |
| Budget | How many evaluations are affordable? |
| Best configuration | Which completed trial has the preferred objective value? |
| Experiment record | How can the optimization run be audited and reproduced? |

The implementations keep these concerns separate so that changing the search strategy does not require redesigning the evaluation model or database representation.
