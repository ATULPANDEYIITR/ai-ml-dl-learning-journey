# Advanced Dimensionality Reduction

## Scope

This project implements a small, executable study of advanced dimensionality reduction with three distinct perspectives:

| Method | Main mechanism | Primary structural objective |
|---|---|---|
| PCA | Global linear projection | Preserve directions of maximum variance |
| t-SNE | Probability-based neighborhood embedding | Preserve local neighborhood relationships |
| UMAP-style embedding | Weighted nearest-neighbor graph optimization | Preserve local manifold connectivity |

The implementations use a synthetic curved manifold so that the difference between global variance preservation and local-neighborhood preservation can be examined directly.

The project deliberately treats t-SNE, UMAP, and general manifold learning as related but distinct concepts. t-SNE converts high-dimensional pairwise similarities into a low-dimensional probability model. UMAP is based on a fuzzy neighborhood graph and a low-dimensional optimization objective. Manifold learning describes the broader assumption that observations may occupy a lower-dimensional nonlinear structure embedded inside a higher-dimensional space.

## Conceptual Model

A useful abstraction is:

`high-dimensional observations → distance structure → local/global representation → low-dimensional embedding`

The important distinction is what information survives this transformation.

PCA attempts to preserve global variance through a linear transformation. It is useful when the dominant structure is approximately linear.

t-SNE constructs conditional neighborhood probabilities in the original space and compares them with a heavy-tailed probability distribution in the embedding. Its optimization strongly emphasizes local relationships.

UMAP constructs a weighted nearest-neighbor graph and seeks a low-dimensional layout that preserves important graph relationships. The graph representation makes locality explicit and provides a different computational structure from exact all-pairs t-SNE.

Manifold learning is the broader family. Methods can be linear or nonlinear, graph-based or distance-based, and can prioritize global geometry, local neighborhoods, or a mixture of both.

## Synthetic Manifold

The source dataset contains two curved trajectories. Each observation contains four numerical features derived from a parameterized curve, with controlled noise.

Conceptually:

`parameter t`
`   ↓`
`curved trajectory`
`   ↓`
`4-dimensional observation`
`   ↓`
`standardization`
`   ↓`
`PCA / t-SNE / UMAP-style graph embedding`
`   ↓`
`2-dimensional coordinates`

The construction is useful because a nonlinear structure can have poor representation under a simple linear projection while still possessing strong local neighborhoods.

## Python Implementation

The Python file is a complete command-line experiment runner.

It provides:

- input validation
- dataclasses for reduction configuration and results
- standardization
- covariance construction
- power-iteration PCA
- exact pairwise distance calculation
- perplexity-based probability construction
- symmetric t-SNE probabilities
- low-dimensional t-SNE optimization
- k-nearest-neighbor graph construction
- weighted graph embedding
- trustworthiness measurement
- embedding stress measurement
- CSV output
- deterministic random seeds
- assertions for result integrity
- command-line configuration

The PCA implementation uses power iteration rather than a numerical linear-algebra dependency. This keeps the source self-contained while making the eigenvector mechanism explicit.

The t-SNE implementation uses exact pairwise distances. For a dataset with `n` observations, its major memory and computational components are quadratic in `n`. This is appropriate for a compact learning implementation but is not a substitute for optimized large-scale implementations.

The UMAP implementation is explicitly a compact UMAP-style approximation. It constructs a weighted k-nearest-neighbor graph and optimizes an embedding using attractive and repulsive forces. It is not intended to reproduce every detail of a reference UMAP implementation.

The script also calculates trustworthiness, which evaluates whether nearby points in the embedding correspond to nearby points in the original space.

Typical execution uses the command-line options:

`python dimensionality_reduction.py --method tsne`

The supported methods are `pca`, `tsne`, and `umap`.

## JavaScript Implementation

The JavaScript implementation uses Node.js standard-library functionality and emphasizes event-driven execution.

Its main components are:

| Component | Responsibility |
|---|---|
| `ValidationError` | Represents invalid dataset or configuration input |
| `standardize` | Normalizes feature scales |
| `pairwiseDistances` | Builds an exact distance matrix |
| `perplexityProbabilities` | Estimates t-SNE neighborhood probabilities |
| `tsne` | Optimizes a two-dimensional embedding |
| `nearestNeighbors` | Builds local neighborhood relationships |
| `umapStyle` | Performs graph-based low-dimensional optimization |
| `ReductionExperiment` | Provides lifecycle events around an experiment |
| `saveCsv` | Persists the resulting embedding |

`ReductionExperiment` emits `started`, `progress`, and `completed` events. This separates the numerical algorithm from operational monitoring.

The implementation uses a deterministic seeded random generator rather than relying on uncontrolled randomness. This is important when comparing algorithm behavior across runs.

Execution can use:

`node dimensionality_reduction.js tsne`

or:

`node dimensionality_reduction.js umap`

The output is written under the `output` directory.

## C++17 Implementation

The C++ implementation uses an explicit numerical-engineering architecture.

The central `ManifoldEmbeddingEngine` supports:

- PCA
- t-SNE-style embedding
- UMAP-style graph embedding

Supporting structures include:

- `ReductionMethod`
- `ReductionConfig`
- `Point2D`
- `Graph`
- `DatasetValidator`

The `Graph` class stores sparse neighborhood relationships rather than requiring every relationship to be represented as a graph edge. This models the local-manifold perspective more directly.

PCA uses covariance estimation, power iteration, and matrix deflation.

The t-SNE path creates a pairwise probability matrix and a Student-t-style low-dimensional distribution. Gradient updates use momentum.

The UMAP-style path first builds a k-nearest-neighbor graph and assigns exponentially decaying edge weights. Optimization then attempts to bring connected observations together.

The program also handles invalid dimensions, empty datasets, non-finite values, invalid neighborhood sizes, and unsupported methods through exceptions.

The implementation uses only C++17 standard-library facilities.

## Java 17 Implementation

The Java implementation models dimensionality reduction as an enterprise-style experiment service rather than as a collection of standalone numerical functions.

The main domain types are:

- `Method`
- `ExperimentState`
- `Point`
- `ExperimentConfig`
- `ExperimentReport`

The `ReductionAlgorithm` interface provides an explicit algorithm boundary. `PCAAlgorithm` implements the linear method, while `GraphEmbeddingAlgorithm` provides the graph-based nonlinear path.

`ReductionService` acts as the orchestration layer. It validates input, resolves the requested algorithm, records execution state, measures duration, and produces an immutable `ExperimentReport`.

The state model is:

`CREATED → VALIDATED → RUNNING → COMPLETED`

A runtime failure moves the operation to `FAILED`.

Java records are used for immutable configuration and result objects. Collections are exposed through immutable copies where appropriate, reducing accidental mutation of experiment results.

The Java implementation therefore emphasizes explicit domain modeling, lifecycle management, validation, and separation of algorithmic concerns.

## PostgreSQL Model

The PostgreSQL implementation represents dimensionality-reduction experiments as persistent relational data.

The primary relationship is:

`Dataset → Observations → Experiments → Embedding Points`

A parallel graph representation is:

`Observation → Neighbor Edge → Observation`

The database includes:

- `dataset`
- `observation`
- `reduction_experiment`
- `embedding_point`
- `nearest_neighbor_edge`
- `status_check`

The database uses PostgreSQL enum types for reduction methods and experiment states.

## Database Constraints

Important rules are enforced at database level rather than relying entirely on application code.

Examples include:

- datasets must contain at least two features
- observations must contain at least two feature values
- experiment iterations must be positive
- learning rates must be positive
- t-SNE requires a valid perplexity
- UMAP requires a positive neighbor count
- UMAP requires a positive minimum distance
- embedding coordinates must be finite
- one observation cannot have multiple embedding rows for the same experiment
- neighbor edges cannot connect an observation to itself

The `validate_experiment_parameters` trigger also verifies that t-SNE perplexity and UMAP neighborhood size are compatible with the dataset size.

This is important because parameter validity depends on more than the individual parameter itself.

For example, a perplexity value can be numerically positive while still being invalid for a dataset containing too few observations.

## Experiment Workflow

The relational workflow is:

`create dataset`
`     ↓`
`insert observations`
`     ↓`
`configure reduction experiment`
`     ↓`
`validate parameters`
`     ↓`
`run experiment`
`     ↓`
`store embedding`
`     ↓`
`run status checks`
`     ↓`
`evaluate structural validity`
`     ↓`
`report results`

The database distinguishes experiment configuration from its resulting embedding. This prevents the parameter set from being confused with the coordinates produced by that configuration.

## t-SNE Mechanics

t-SNE starts by converting distances in the original space into conditional probabilities.

For an observation `i`, nearby observations receive larger probability values under a Gaussian-like kernel. The bandwidth is adjusted so that the resulting distribution has a requested perplexity.

The conditional probabilities are then symmetrized.

In the low-dimensional space, t-SNE uses a heavy-tailed distribution. The heavy tail reduces the tendency of distant embedded observations to exert excessively strong attraction.

The optimization attempts to reduce the divergence between the high-dimensional and low-dimensional probability distributions.

Important practical consequences include:

- results depend on random initialization
- perplexity is linked to dataset size
- the method strongly emphasizes local structure
- distances between well-separated clusters should not automatically be interpreted as quantitative global distances
- different runs can produce rotated, reflected, or differently arranged embeddings
- exact pairwise implementations become expensive as the dataset grows

## UMAP Mechanics

The UMAP-style implementation begins with a nearest-neighbor graph.

Each observation is connected to a selected set of nearby observations. Edge weights represent the strength of local relationships.

The graph is then optimized in a low-dimensional space.

The important structural distinction is that the representation is explicitly graph-based. The graph provides a sparse approximation to the local manifold instead of treating every pair as equally important during the representation stage.

Important parameters include:

| Parameter | Effect |
|---|---|
| `neighbors` | Controls the scale of local structure represented by the graph |
| `min_dist` | Controls how closely connected observations can pack in the embedding |
| `learning_rate` | Controls optimization step size |
| `iterations` | Controls optimization duration |
| `seed` | Controls deterministic initialization |

The implementation in this project is intentionally simplified. It captures the nearest-neighbor graph and low-dimensional optimization perspective without claiming numerical equivalence to a full reference implementation.

## PCA and Manifold Learning

PCA is included as a baseline because dimensionality reduction should not automatically imply nonlinear dimensionality reduction.

PCA searches for orthogonal directions that explain maximum variance. It is computationally simpler and easier to interpret than nonlinear methods.

The comparison can therefore be understood as:

| Question | PCA | t-SNE | UMAP-style |
|---|---|---|---|
| Linear transformation | Yes | No | No |
| Local graph emphasis | Low | Strong | Strong |
| Global variance preservation | Strong | Weak | Moderate and parameter-dependent |
| Exact pairwise representation | Covariance-based | Yes in this implementation | No |
| Random initialization | Yes for power iteration | Yes | Yes |
| Main use | Linear compression | Local visualization | Manifold visualization |

A nonlinear embedding should not automatically be treated as superior. The appropriate method depends on whether the analytical task requires global structure, local neighborhoods, compression, visualization, or downstream modeling.

## Validation and Failure Handling

The implementations reject:

- empty datasets
- inconsistent row dimensions
- fewer than two source features
- non-finite numerical values
- invalid neighbor counts
- invalid perplexity
- non-positive optimization parameters
- unsupported output dimensions
- incompatible dataset and algorithm configurations

These checks prevent numerical failures from being mistaken for meaningful embeddings.

The PostgreSQL layer adds another level of protection by enforcing parameter relationships through constraints and triggers.

## Performance Characteristics

The main computational distinction is between global pairwise calculations and sparse neighborhood graphs.

For `n` observations and `d` features:

| Operation | Typical structure |
|---|---|
| Standardization | `O(nd)` |
| Pairwise distance matrix | `O(n²d)` |
| Exact t-SNE probability construction | approximately `O(n²d)` |
| Exact t-SNE optimization | approximately `O(n²)` per iteration |
| k-nearest-neighbor construction by full distance matrix | `O(n²d)` plus sorting |
| Graph-based embedding | dependent on graph size and iteration count |
| PCA power iteration | approximately `O(d²)` per matrix-vector iteration after covariance construction |

The exact t-SNE implementation is the principal scalability limitation.

For large datasets, approximate nearest-neighbor search, sparse graph representations, Barnes-Hut methods, interpolation methods, optimized numerical libraries, and parallel execution can substantially reduce practical costs.

The source files intentionally keep the algorithms explicit rather than hiding these costs behind external libraries.

## Numerical Considerations

Feature scaling matters because distance-based algorithms are sensitive to feature magnitude.

The source implementations standardize features before distance calculations.

A feature measured in thousands can otherwise dominate another feature measured in fractions even when the smaller feature carries meaningful structural information.

Near-zero scales are protected against division by zero by replacing them with a unit scale.

The algorithms also guard against:

- zero distances
- zero normalization denominators
- underflow during probability construction
- zero-length vectors
- non-finite input values

These controls improve numerical stability but do not eliminate all limitations associated with floating-point optimization.

## Interpretation of Embeddings

An embedding is a representation, not a direct measurement of the original feature space.

A point that is visually close to another point in a t-SNE or UMAP-style embedding should be interpreted according to the method's preservation objective.

For t-SNE in particular, visual cluster separation can be informative about local neighborhoods while misleading if interpreted as a precise global distance map.

The same caution applies to UMAP-style embeddings. Neighborhood relationships are emphasized, and changes in graph parameters can alter the visible organization.

Embedding coordinates themselves have no intrinsic semantic meaning. Rotation, reflection, translation, and scaling can change coordinates without changing the underlying neighborhood relationships.

## Security and Reliability Considerations

The numerical implementations do not execute external commands or dynamically evaluate code.

The PostgreSQL design uses:

- foreign keys
- restrictive deletes where experiment history should remain meaningful
- check constraints
- trigger validation
- transaction boundaries
- controlled relational references

Database-level validation is important when multiple applications may write to the same experiment tables.

For larger production systems, access control should restrict who can create experiments, modify datasets, or replace embedding results. Completed experiment artifacts should generally be treated as immutable analytical records.

## Relationship Between the Five Implementations

The implementations intentionally use different engineering perspectives.

| File | Primary perspective |
|---|---|
| Python | Numerical experimentation and diagnostics |
| JavaScript | Event-driven experiment execution |
| C++17 | Explicit numerical and graph-engineering architecture |
| Java 17 | Enterprise-style domain and lifecycle modeling |
| PostgreSQL | Persistent relational experiment and validation model |

They are complementary rather than line-for-line translations.

The Python implementation provides the richest diagnostic workflow. JavaScript emphasizes asynchronous lifecycle events. C++17 emphasizes explicit numerical data structures and graph representation. Java 17 emphasizes domain boundaries and experiment state. PostgreSQL expresses dataset, experiment, embedding, graph, and validation relationships as durable relational structures.

## Practical Decision Criteria

Use PCA when the primary objective is compact linear representation, variance analysis, or a transparent baseline.

Use t-SNE when local neighborhood visualization is the primary objective and the dataset is sufficiently small or an optimized implementation is available.

Use UMAP when a graph-based nonlinear embedding is appropriate and control over neighborhood scale is important.

Use a broader manifold-learning approach when the problem is fundamentally about recovering structure from observations believed to lie near a lower-dimensional nonlinear manifold.

The central engineering decision is not simply which method produces the most visually separated plot. It is which structural properties need to survive the reduction and which distortions are acceptable for the intended analytical task.
