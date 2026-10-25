# K-Means Clustering: Centroids, Inertia, Initialization, and Choosing K

## Project Overview

K-Means is an unsupervised learning algorithm that partitions numerical observations into a specified number of clusters. It represents each cluster by a centroid and assigns each observation to its nearest centroid under squared Euclidean distance.

This project implements K-Means across Python, JavaScript, C++17, Java 17, and PostgreSQL. The implementations cover the clustering algorithm itself, reproducible initialization, iterative centroid updates, empty-cluster recovery, inertia calculation, and experiments for choosing the number of clusters.

The example dataset represents customers using two numerical features. Its observations form three approximate groups, making it useful for inspecting centroid positions, cluster assignments, and the effect of changing K.

## Mathematical Model

Let the dataset contain \(n\) observations, each with \(d\) features:

\[
X = \{x_1,x_2,\ldots,x_n\}, \qquad x_i \in \mathbb{R}^d
\]

K-Means partitions the observations into \(K\) clusters. Each cluster \(C_j\) has a centroid \(\mu_j\), calculated as the arithmetic mean of its assigned observations.

\[
\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j}x_i
\]

The objective is to minimize the within-cluster sum of squared distances, commonly called inertia:

\[
J = \sum_{i=1}^{n}\|x_i-\mu_{z_i}\|_2^2
\]

Here, \(z_i\) is the cluster assigned to observation \(x_i\).

Inertia measures how tightly observations are grouped around their centroids. It is non-negative, and a smaller value indicates a tighter fit for the evaluated partition. It does not independently establish that the chosen number of clusters is appropriate.

## Four Core Concepts

### Centroids

A centroid is the coordinate-wise mean of the observations assigned to a cluster. For example, the centroid of two two-dimensional points, \((2,4)\) and \((4,8)\), is \((3,6)\).

Centroids are recalculated after each assignment step. They are not required to be existing observations, which distinguishes a centroid from a medoid.

Centroid positions depend on feature scales. If spending is measured in thousands while activity is measured in single-digit counts, the feature with the larger numerical scale may dominate the distance calculation. Standardization or another suitable scaling strategy should be considered before fitting the model.

### Inertia

Inertia sums the squared Euclidean distance from each observation to its assigned centroid. It is the objective minimized by standard Lloyd-style K-Means iterations.

The algorithm alternates between two operations:

1. Assignment: associate every observation with its nearest centroid.
2. Update: replace each centroid with the mean of its assigned observations.

With exact assignment and mean updates, these steps do not increase the objective. Practical implementations must handle empty clusters carefully, and a reseeding operation can temporarily change this behavior.

Inertia depends on the dataset, feature units, and K. Raw inertia values should not be compared indiscriminately across datasets with different scales or sample counts.

### Initialization

Initialization supplies the first centroid positions and can strongly affect the final solution because K-Means finds a local optimum rather than guaranteeing the global minimum.

Random initialization selects starting observations randomly. It is straightforward but can place multiple initial centroids in the same dense region.

K-Means++ chooses an initial observation and then selects subsequent centers with probabilities proportional to their squared distance from the nearest existing center. This encourages better spatial coverage and often produces stronger starting configurations.

Multiple restarts fit the model repeatedly with different initial configurations. The implementation retains the result with the lowest inertia. A fixed random seed makes runs reproducible within the same implementation and runtime assumptions.

### Choosing K

K is the requested number of clusters and must be a positive integer no greater than the number of observations.

An elbow analysis fits several candidate values of K and records each resulting inertia. Increasing K generally reduces inertia because additional clusters can represent the data more closely. The useful question is where further reductions become less substantial relative to the added complexity.

The elbow is a heuristic, not a guaranteed statistical optimum. Silhouette scores, domain knowledge, cluster stability, operational requirements, and the cost of maintaining many segments can help distinguish plausible choices.

For identical observations, multiple centroid configurations can represent equivalent partitions. Increasing K does not necessarily produce more meaningful business categories.

## Algorithm Workflow

The implemented workflow is:

    Validate feature dimensions and finite numeric values
                         |
                         v
              Initialize K centroids
                         |
                         v
              Assign nearest centroids
                         |
                         v
               Recalculate centroids
                         |
                         v
             Recover empty clusters
                         |
                         v
          Check label stability or movement
                  /             \
               Stop             Repeat
                 |
                 v
       Reassign using final centroids
                 |
                 v
          Calculate final inertia

Convergence is determined by stable cluster labels or centroid movement below a configured tolerance. Reaching the maximum iteration count stops the procedure even when convergence has not been established.

The final reassignment step ensures that reported labels correspond to the returned centroids.

## Python Implementation

The Python script defines `KMeans`, `ClusteringResult`, and an `Initialization` enumeration.

Its main mechanisms include:

- `validate_points()` checks that the dataset is non-empty, every point has the same feature count, and all values are finite.
- `squared_distance()` calculates squared Euclidean distance without an unnecessary square root.
- `_initialize()` implements random selection and K-Means++.
- `_fit_once()` performs assignment, centroid updates, convergence checks, and empty-cluster recovery.
- `fit()` executes multiple restarts and retains the lowest-inertia result.
- `predict()` assigns new observations to the nearest fitted centroid.
- `choose_k()` evaluates candidate values of K and returns their inertia values.

The result is an immutable dataclass containing centroids, labels, inertia, iteration count, and convergence status. The script also reports cluster sizes and runs assertions against basic output invariants.

The standard library is sufficient. Run the file with Python 3.10 or later. The optional `--k` and `--seed` arguments control the requested cluster count and reproducibility.

## JavaScript Implementation

The JavaScript file uses classes, objects, arrays, and a seeded pseudo-random number generator to represent the same clustering workflow through JavaScript-specific control flow.

The `KMeans` class validates its configuration, initializes centroids, executes iterative clustering, calculates inertia, and predicts labels for new observations.

`seededRandom()` provides deterministic pseudo-random values for repeatable experiments. It is intended for reproducibility, not cryptographic security.

The asynchronous `evaluateCandidates()` function evaluates several values of K and yields between candidates through `setImmediate()`. This supports responsive orchestration in a Node.js service, although the clustering calculations themselves remain synchronous and can block the event loop for large datasets.

Validation rejects empty data, inconsistent feature dimensions, and non-finite values. The module exports its class and helper functions so they can be reused by another Node.js application.

Run the file with a supported Node.js version. No external packages are required.

## C++17 Implementation

The C++ program treats clustering as a numerical processing component with explicit data structures and a private implementation boundary.

- `Point` represents a numerical observation.
- `ClusterResult` stores centroids, labels, inertia, iteration count, and convergence status.
- `KMeans` owns the clustering configuration and provides `fit()` and `predict()` operations.
- `std::mt19937` supplies seeded pseudo-random values.
- STL algorithms perform shuffling, nearest-centroid selection, minimum calculations, and farthest-point selection for empty-cluster recovery.

The program evaluates several K values and reports the resulting inertia. Input validation and exceptions provide clear failure behavior for malformed datasets or invalid configurations.

The class interface separates the public fitting operations from internal initialization and iteration logic. This is useful when integrating the algorithm into a larger native application.

Compile with a C++17-compatible compiler. The implementation uses only the standard library.

## Java 17 Implementation

The Java program uses explicit configuration and result records to make the model's inputs and outputs clear.

The `Config` record validates K, initialization strategy, iteration limits, tolerance, seed, and restart count. The `Result` record creates immutable copies of centroid coordinates and labels, preventing callers from accidentally modifying the returned clustering state.

`KMeansService` provides fitting and prediction operations. Its fitting process performs multiple runs and retains the lowest-inertia result. Helper methods isolate distance calculation, nearest-centroid selection, and input validation.

A custom `DatasetException` distinguishes invalid dataset input from unexpected internal failures. Cluster-size reporting is implemented through a result method, and candidate K values are evaluated through the same service interface.

This structure suits an enterprise application in which configuration validation, immutable outputs, and clear service boundaries are important. Compile and run with Java 17 or later.

## PostgreSQL Data Model

The SQL script models datasets and persisted clustering results rather than attempting to reproduce the entire iterative numerical algorithm inside SQL.

| Table or view | Purpose |
|---|---|
| `clustering.datasets` | Stores dataset identity and expected feature count. |
| `clustering.observations` | Stores numerical feature arrays and external observation keys. |
| `clustering.models` | Records K, initialization, tolerance, random seed, restart count, execution state, and final metrics. |
| `clustering.centroids` | Stores the coordinates of each fitted centroid. |
| `clustering.assignments` | Records each observation's assigned cluster and squared distance. |
| `clustering.iteration_history` | Stores inertia and maximum centroid movement per iteration. |
| `clustering.model_evaluations` | Stores results from candidate-K experiments and optional silhouette scores. |
| `clustering.cluster_report` | Aggregates cluster sizes and within-cluster error for completed models. |

Primary keys and foreign keys protect relationships between observations, models, centroids, and assignments. Unique constraints prevent duplicate observation keys within a dataset and duplicate model names within a dataset.

The observation-validation trigger verifies feature count and rejects non-finite values. The model-completion trigger verifies that a completed model has metrics, an appropriate K, the required number of centroids, and an assignment for every observation in its dataset.

The `squared_euclidean()` function calculates squared distance between equal-length arrays. The reporting view aggregates assignments by cluster. A window-function query compares inertia across recorded iterations, and a common table expression ranks candidate K values by their recorded inertia.

The sample data represents persisted illustrative model outputs. It does not claim that PostgreSQL has fitted those centroids itself. The Python, JavaScript, C++, and Java implementations perform the iterative fitting; SQL provides storage, integrity checks, retrieval, and reporting.

Run the script against a PostgreSQL database with permission to create the `clustering` schema and its objects. The script uses a transaction so the schema setup and sample workflow are committed together.

## Choosing K in Practice

The example evaluates K values from one through six. Inspect the resulting inertia sequence rather than assuming that the lowest inertia identifies the best model.

A useful evaluation process considers:

- The point where reductions in inertia begin to diminish.
- Whether clusters contain enough observations to be useful.
- Whether centroids and assignments are stable across different seeds.
- Whether features have appropriate scales and meaningful distance relationships.
- Whether independent measures, such as silhouette scores, support the proposed partition.
- Whether the resulting groups are interpretable for the intended application.

A small inertia value can be misleading when K is large. At K equal to the number of observations, an implementation can assign one observation to each cluster and achieve zero inertia, but that partition may provide little practical value.

## Complexity and Performance

Let \(n\) be the number of observations, \(d\) the number of features, \(K\) the cluster count, \(I\) the number of iterations, and \(R\) the number of restarts.

A typical run has time complexity approximately:

\[
O(nK d I)
\]

With multiple restarts, the fitting cost becomes approximately \(O(RnKdI)\), excluding additional candidate-K experiments. Storing the observations and assignments requires \(O(nd)\) and \(O(n)\) space, respectively. Centroids require \(O(Kd)\) space.

Distance calculations dominate when the dataset has many observations or features. Candidate-K evaluation multiplies the computational cost because each candidate requires a separate fitting procedure.

For large datasets, practical optimizations include efficient numerical arrays, vectorized distance calculations, minibatch approaches, and careful memory management. Those optimizations are outside the implementations provided here.

## Edge Cases and Limitations

- **Invalid K:** K must be positive and cannot exceed the number of observations.
- **Empty dataset:** Fitting fails explicitly rather than returning meaningless centroids.
- **Inconsistent dimensions:** All observations and prediction inputs must use the same feature count.
- **Non-finite values:** NaN and infinite values are rejected before distance calculations.
- **Duplicate observations:** Duplicate-valued points can cause repeated centroid coordinates, even when K-Means++ is used.
- **Empty clusters:** The implementations reseed empty clusters with a distant observation. Duplicate-heavy data can still lead to empty clusters or repeated centroids.
- **Non-convergence:** An iteration limit may be reached before labels stabilize. The result records convergence status so callers can distinguish stopping from convergence.
- **Feature scaling:** Large numerical scales can dominate Euclidean distance.
- **Outliers:** Squared distances give distant observations disproportionate influence on inertia and centroid positions.
- **Local optima:** Different initializations can produce different clusterings.
- **Cluster geometry:** Standard K-Means works best for compact, roughly spherical groups under Euclidean distance. It can perform poorly for strongly overlapping, elongated, or non-convex clusters.
- **Numerical overflow:** Extremely large finite coordinates can overflow squared-distance calculations. Production systems should apply sensible feature bounds or robust numerical checks.

## Security and Data Integrity

The in-memory implementations validate numerical inputs and reject malformed dimensions before fitting. Their deterministic random seeds support reproducibility but must not be treated as security controls.

The PostgreSQL schema uses constraints, foreign keys, and triggers to prevent inconsistent persisted states. Applications that expose model fitting or prediction through an API should also limit dataset sizes, enforce request timeouts, validate authorization, and avoid logging sensitive feature values unnecessarily.

Database access should use least-privilege roles. SQL statements that accept user-controlled identifiers or values should be parameterized by the calling application. The schema's triggers enforce selected integrity rules but do not replace application authorization or operational monitoring.

## Interpreting the Results

Centroids describe the representative coordinates of the discovered groups. Labels associate observations with those groups, and cluster sizes describe the distribution of assignments. Inertia quantifies within-cluster squared error, while the candidate-K table helps reveal diminishing returns as K increases.

These outputs describe geometric structure in the supplied feature space. They do not independently establish business meaning, causation, or future predictive accuracy. Useful cluster interpretation requires understanding the feature definitions, their scales, the data collection process, and the consequences of using the resulting groups.
