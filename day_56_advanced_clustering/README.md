# Advanced Clustering

## Scope

This project implements three closely related but technically distinct ideas:

- hierarchical clustering
- DBSCAN
- density-based clustering analysis

The implementations use a two-dimensional observation space so that the distance calculations, cluster formation rules, noise handling, and evaluation logic remain explicit.

The central distinction is structural:

`Hierarchical clustering` builds groups through repeated cluster merges.

`DBSCAN` builds groups through density connectivity.

`Density-based clustering` is the broader family of methods in which local point density determines whether observations belong to clusters, including DBSCAN's explicit treatment of noise.

The five source files use the same conceptual domain while deliberately applying different engineering approaches.

## Conceptual Model

The data model is:

`Dataset → Observations → Distances → Clustering Run → Clusters → Memberships → Evaluation Metrics`

Each observation contains:

| Field | Meaning |
|---|---|
| ID | Stable observation identifier |
| X | First numerical feature |
| Y | Second numerical feature |
| Label | Optional source classification used for validation and demonstration |

The clustering algorithms do not require the source label. It is retained as metadata so the examples can compare discovered structure with known demonstration groups.

## Hierarchical Clustering

Hierarchical clustering does not require the final groups to be known in advance during the merge process.

The Python, JavaScript, C++, and Java implementations use an agglomerative strategy:

1. Start with one cluster per observation.
2. Calculate distances between clusters.
3. Select the closest pair.
4. Merge the pair.
5. Repeat until the requested number of clusters remains.

Three linkage definitions are supported:

| Linkage | Cluster distance |
|---|---|
| Single | Minimum distance between any two observations |
| Complete | Maximum distance between any two observations |
| Average | Mean distance across all cross-cluster observation pairs |

The linkage choice changes the geometry of the resulting hierarchy.

Single linkage can connect groups through narrow chains of observations. Complete linkage favors compact groups. Average linkage provides an intermediate criterion based on all cross-cluster distances.

The implementations stop at a requested cluster count rather than constructing and persisting a complete dendrogram.

## DBSCAN

DBSCAN uses two primary parameters:

| Parameter | Meaning |
|---|---|
| `eps` | Radius used to define a local neighborhood |
| `min_samples` | Minimum number of observations required for a core point |

An observation is a core point when its `eps` neighborhood contains at least `min_samples` observations, including itself.

A DBSCAN workflow can be represented as:

`Core point → density-reachable observations → density-connected cluster`

Observations that cannot become connected to a density-supported cluster receive label `-1` and are treated as noise.

This is an important distinction from hierarchical clustering. DBSCAN does not need a requested number of clusters, and isolated observations do not have to be assigned artificially to a group.

## Density-Based Clustering

Density-based clustering focuses on local concentration rather than only global pairwise similarity.

This makes it useful when:

- clusters have different spatial concentrations
- isolated observations are meaningful
- the number of clusters is not known beforehand
- a dataset contains noise
- cluster membership should depend on local connectivity

DBSCAN has an important limitation: its fixed `eps` threshold can be unsuitable when different regions of the dataset have substantially different densities.

Parameter selection therefore requires examination of neighborhood distances and domain characteristics rather than arbitrary values.

## Python Implementation

The Python program is a complete simulation and evaluation workflow.

### Domain structures

`Point` is an immutable dataclass representing an observation.

`Cluster` stores a cluster identifier and its observations.

`DBSCANResult` separates:

- cluster labels
- core observations
- noise observations

`ClusterMethod` provides explicit algorithm identifiers.

### Hierarchical implementation

`hierarchical_clustering()` performs direct agglomerative merging.

The implementation accepts:

- target cluster count
- single linkage
- complete linkage
- average linkage

The distance calculation is intentionally explicit. This makes the linkage behavior inspectable and avoids hiding the clustering mechanics behind an external library.

### DBSCAN implementation

`dbscan()` first constructs an `eps` neighborhood for every observation.

It then:

- identifies core points
- tracks visited observations
- expands density-connected clusters
- changes previously classified noise into cluster members when density connectivity is established

This last behavior is important because an observation initially encountered as noise may later be discovered as a border point connected to a core cluster.

### Evaluation

The Python implementation calculates a silhouette score for non-noise observations.

The silhouette value compares:

- average distance from an observation to its own cluster
- average distance from that observation to the nearest alternative cluster

Values closer to `1` indicate stronger separation, values around `0` indicate overlapping structure, and negative values indicate that an observation may fit another cluster better.

DBSCAN noise is excluded from the metric.

### Validation and failure handling

The program validates:

- non-empty datasets
- unique observation IDs
- finite coordinates
- valid target cluster counts
- positive `eps`
- positive `min_samples`
- supported linkage methods

The built-in tests also verify invalid input handling and basic clustering behavior.

The program writes a JSON report using standard-library file operations.

## JavaScript Implementation

The JavaScript program models clustering as an event-driven processing workflow.

### Event-driven behavior

`EventBus` publishes events when DBSCAN:

- detects a noise candidate
- creates a new cluster

This separates clustering computation from operational logging.

The approach is useful when a clustering engine must later feed:

- monitoring
- audit logs
- user interfaces
- asynchronous processing pipelines

### Data structures

The implementation uses:

- `Map` for labels and neighborhoods
- `Set` for visited and core observations
- classes for domain objects
- Promises and `fs/promises` for report persistence

The source uses only Node.js standard-library functionality.

### Security

The report filename is sanitized before being passed to the filesystem. This prevents arbitrary path characters from being accepted as an output filename.

The generated file is also written with restrictive permissions.

## C++17 Implementation

The C++ program presents clustering as a reusable technical engine.

### Domain design

The implementation separates:

- `Point`
- `Cluster`
- `DBSCANResult`
- `Linkage`
- `ClusteringEngine`

`ClusteringEngine` owns validation and algorithmic operations.

The use of STL containers provides explicit control over:

- sets of visited observations
- maps of cluster assignments
- vectors of cluster members
- ordered reporting

### Governance of clustering results

The engine refuses:

- empty datasets
- duplicate IDs
- non-finite coordinates
- invalid cluster counts
- invalid density parameters

Exceptions propagate to the application boundary where a controlled failure message is produced.

### Performance

The direct implementation uses repeated pairwise calculations.

For `n` observations, neighborhood construction is `O(n²)`.

The simple agglomerative implementation can reach approximately `O(n³)` behavior because cluster distances are repeatedly recomputed.

This is intentionally transparent rather than optimized.

Production-scale implementations should use appropriate spatial indexing or specialized nearest-neighbor structures when the feature space and distance metric permit them.

## Java 17 Implementation

The Java implementation models clustering as an enterprise service.

### Domain types

The program uses:

- records for immutable observations and clusters
- enums for linkage and observation state
- a policy record for clustering parameters
- a service class for algorithm execution
- a custom validation exception

The `ClusteringPolicy` prevents invalid configuration from entering the service layer.

### Explicit state transitions

DBSCAN observations move through:

`UNVISITED → NOISE`

or:

`UNVISITED → CLUSTERED`

A previously classified noise observation can later transition to `CLUSTERED` when expansion establishes density connectivity.

This explicit state model makes an important DBSCAN rule visible instead of hiding it inside collection operations.

### Enterprise-oriented separation

`ClusterService` owns clustering behavior.

`ClusteringPolicy` owns configuration.

`Point` and `Cluster` represent domain data.

`ValidationException` represents invalid domain state.

This separation allows validation and clustering policy to be tested independently from application output.

## PostgreSQL Model

The PostgreSQL implementation models clustering results as persistent relational data.

The main entities are:

| Table | Purpose |
|---|---|
| `datasets` | Identifies independent datasets |
| `observations` | Stores feature coordinates and metadata |
| `clustering_runs` | Stores algorithm and parameter configuration |
| `clusters` | Stores cluster identities |
| `cluster_memberships` | Associates observations with clusters |
| `pairwise_distances` | Stores reusable pairwise distances |
| `clustering_metrics` | Stores evaluation results |

The relationships are:

`datasets → observations`

`datasets → clustering_runs`

`clustering_runs → clusters`

`clusters → cluster_memberships → observations`

`clustering_runs → clustering_metrics`

## Database Constraints

The schema uses database-level enforcement for important invariants.

Examples include:

- unique dataset names
- unique observation keys within a dataset
- finite coordinates
- positive density parameters
- positive cluster counts
- supported algorithm names
- supported linkage names
- silhouette range validation
- ordered observation pairs for pairwise distances

A trigger validates that a cluster membership cannot connect observations from a different dataset or a cluster belonging to another clustering run.

This prevents an application-level programming error from silently creating inconsistent clustering results.

## Database Queries

The SQL script demonstrates several operational analyses.

### Density inspection

A neighborhood query calculates how many observations fall within the configured DBSCAN `eps` radius.

The result identifies candidate core points according to `min_samples`.

### Cluster reporting

Cluster membership aggregation provides:

- cluster size
- noise count
- algorithm
- run identifier

Noise is kept explicitly separate through `is_noise`.

### Distance diagnostics

The pairwise distance query identifies nearby observations.

This supports inspection of whether an `eps` value is plausibly separating dense regions from sparse observations.

### Transactional integrity

The script contains an intentionally invalid membership operation inside a transaction.

The trigger rejects the inconsistent association, and the transaction is rolled back.

This demonstrates why clustering persistence should enforce invariants at the database boundary rather than trusting every client.

## Algorithm Comparison

| Property | Hierarchical | DBSCAN |
|---|---|---|
| Core mechanism | Repeated cluster merging | Density connectivity |
| Number of clusters required | Final count is specified | Not required |
| Noise handling | Normally assigns every observation | Explicit noise label |
| Main parameters | Target count and linkage | `eps`, `min_samples` |
| Hierarchical structure | Yes | No |
| Handles arbitrary cluster shape | Depends on linkage | Strong for density-connected shapes |
| Sensitivity | Linkage and distance metric | Density parameters |
| Direct implementation cost | High for repeated merges | Quadratic without spatial indexing |

## Edge Cases

The implementations explicitly address several failure conditions.

### Duplicate identifiers

Duplicate IDs are rejected because cluster membership and reporting depend on stable observation identity.

### Invalid coordinates

Non-finite coordinates are rejected. Distance calculations involving `NaN` or infinite values can otherwise produce invalid cluster decisions.

### Invalid DBSCAN parameters

A non-positive `eps` or `min_samples` is rejected.

### Single-cluster evaluation

Silhouette evaluation returns `0` when fewer than two usable clusters exist because there is no meaningful alternative cluster against which to compare the observation.

### DBSCAN noise

Noise is represented as `-1` and excluded from silhouette evaluation.

### Isolated observations

An isolated observation can remain noise under DBSCAN instead of being forcibly assigned to an unrelated cluster.

## Complexity and Scaling

For `n` observations and two-dimensional Euclidean distance:

- Pairwise distance construction is `O(n²)`.
- Naive DBSCAN neighborhood discovery is `O(n²)`.
- The direct hierarchical implementation can reach `O(n³)` because cluster-to-cluster distances are repeatedly recomputed.
- Memory for a full pairwise distance matrix is `O(n²)`.

These implementations favor clarity and reproducibility.

Large production datasets normally require more specialized approaches, such as spatial indexing, optimized nearest-neighbor search, sparse neighborhood representations, or implementations designed around the dataset's dimensionality and distance metric.

High-dimensional data also introduces the curse of dimensionality: Euclidean distances can become less discriminative as dimensionality increases, making density thresholds harder to interpret.

## Practical Parameter Interpretation

For DBSCAN:

`eps` controls the radius of local density.

`min_samples` controls how much local support is required for a core observation.

Increasing `eps` generally increases neighborhood sizes and can merge previously separate dense regions.

Decreasing `eps` can fragment clusters and produce more noise.

Increasing `min_samples` requires stronger local density and can convert marginal observations into noise.

The appropriate values depend on feature scaling, measurement units, noise level, and the expected structure of the dataset.

For hierarchical clustering, the linkage rule should be selected according to the desired definition of inter-cluster similarity rather than treated as an implementation detail.

## Security and Reliability Considerations

Clustering systems can be affected by malformed input, extreme coordinates, duplicate identifiers, invalid parameters, and inconsistent persisted results.

The source files therefore apply:

- input validation
- finite-number checks
- unique identifier enforcement
- explicit exception handling
- database constraints
- trigger-based integrity checks
- transactional rollback
- sanitized filesystem output

The PostgreSQL design also separates clustering configuration from cluster membership so that the parameters used to produce a result remain associated with the stored run.

## Engineering Trade-offs

The implementations intentionally avoid external machine-learning libraries.

This makes the algorithmic mechanisms visible and keeps execution deterministic and portable.

The trade-off is performance. These implementations are not substitutes for optimized clustering engines on very large datasets.

The Python implementation emphasizes inspectability and testing.

The JavaScript implementation emphasizes event-driven processing and asynchronous persistence.

The C++17 implementation emphasizes explicit algorithmic control and STL-based performance-oriented data structures.

The Java 17 implementation emphasizes immutable domain types, policy validation, and explicit state transitions.

The PostgreSQL implementation emphasizes persistence, relational integrity, parameterized runs, reporting, and transactional consistency.

## Relationship Between the Implementations

The same conceptual workflow is expressed differently:

`Observations`
→ `Distance / Neighborhood Analysis`
→ `Clustering Algorithm`
→ `Cluster Assignment`
→ `Noise Handling`
→ `Evaluation`
→ `Persistent or Reported Result`

The algorithms remain distinct within that workflow.

Hierarchical clustering answers a merge-structure question.

DBSCAN answers a density-connectivity question.

Silhouette evaluation measures separation among discovered non-noise groups.

The database records the configuration and result rather than pretending that persistence itself performs the clustering algorithm.

## Execution

Python requires a standard Python 3 installation and runs the program directly.

JavaScript runs under a Node.js runtime using only built-in modules.

C++17 can be compiled with a compiler supporting the C++17 standard.

Java requires Java 17 or later.

PostgreSQL requires a database capable of executing PostgreSQL procedural functions, triggers, constraints, common table expressions, and standard PostgreSQL data types.

All five implementations use the same fundamental concepts while keeping their implementation strategies technically distinct.
