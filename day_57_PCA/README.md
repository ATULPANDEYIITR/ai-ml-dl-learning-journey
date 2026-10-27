# Principal Component Analysis: Covariance, Eigenvectors, and Explained Variance

## Scope

This project implements Principal Component Analysis (PCA) as a numerical and data-analysis workflow centered on three mathematical objects:

- the covariance matrix
- eigenvalues and eigenvectors
- explained variance and cumulative explained variance

The five executable implementations use the same core statistical concepts while presenting different engineering perspectives.

The project uses a repository-operations dataset as a realistic analytical setting. Measurements such as review latency, changed files, status-check duration, and review-comment count can be correlated. PCA provides a way to transform those correlated measurements into orthogonal principal directions.

PCA is an analytical transformation. It does not replace repository governance, review policy, branch protection, or deterministic authorization rules.

## Conceptual Model

The mathematical workflow is:

    Observations
         |
         v
    Mean centering
         |
         v
    Covariance matrix
         |
         v
    Eigenvalue decomposition
         |
         +--------------------+
         |                    |
         v                    v
    Eigenvalues          Eigenvectors
         |                    |
         v                    |
    Explained variance        |
         |                    |
         +---------+----------+
                   |
                   v
          Principal components
                   |
                   v
            Transformed data

For an observation matrix X with n observations and p features, the covariance matrix is formed from centered observations:

    C = X_centered^T X_centered / (n - 1)

The covariance matrix describes how feature pairs vary together.

An eigenvector v and eigenvalue lambda satisfy:

    C v = lambda v

The eigenvector defines a principal direction. Its associated eigenvalue represents the variance captured along that direction.

The eigenvalues are ordered from largest to smallest. The explained-variance ratio for component i is:

    EVR_i = lambda_i / sum(lambda_j)

The cumulative explained variance is the running sum of these ratios.

## Distinct Technical Concepts

### Covariance Matrix

Covariance measures joint variation between two features.

A positive covariance indicates that the variables tend to increase together. A negative covariance indicates opposing movement. Values close to zero indicate limited linear co-movement.

The diagonal entries of the covariance matrix are feature variances.

The matrix is symmetric:

    covariance(feature_i, feature_j)
    =
    covariance(feature_j, feature_i)

The implementations explicitly preserve this property.

The use of `n - 1` produces sample covariance. This distinction matters because the denominator differs from the population covariance denominator `n`.

### Eigenvectors

Eigenvectors describe the directions discovered by PCA.

For a covariance matrix, the first principal direction corresponds to the largest eigenvalue. Subsequent directions correspond to progressively smaller eigenvalues and are orthogonal for a symmetric covariance matrix under the standard PCA formulation.

The sign of an eigenvector is not unique. If `v` is an eigenvector, `-v` represents the same direction. The source implementations normalize the sign using a deterministic pivot convention so repeated runs are easier to inspect.

### Explained Variance

Explained variance quantifies how much total variance each principal component captures.

For example, a first component with an explained-variance ratio of `0.80` accounts for 80% of the variance represented by the covariance matrix.

Cumulative explained variance is useful when selecting a reduced dimensionality. A threshold such as 95% asks for the smallest number of principal components whose cumulative explained variance reaches that threshold.

The threshold is a selection criterion, not a universal mathematical requirement.

## Scale Sensitivity

Covariance-based PCA is sensitive to feature units.

Consider measurements expressed as:

| Feature | Unit |
|---|---|
| Review latency | hours |
| Changed files | files |
| Status-check duration | minutes |
| Review comments | comments |

A feature with a large numerical variance can have a stronger effect on the covariance matrix than a feature with a smaller numerical variance.

This is why PCA workflows must explicitly decide between:

- covariance PCA on the original measurements
- PCA after feature standardization

The Python implementation includes standardization to expose this distinction. The other implementations concentrate on covariance-based PCA and retain the scale-sensitive interpretation.

Standardization should not be applied automatically. It changes the analytical question.

## Domain Relationship

Repository engineering contains several separate mechanisms:

    Pull Request
        |
        v
    Code Review
        |
        v
    Approval Decisions
        |
        v
    Branch Protection
        |
        v
    Merge Eligibility

PCA is not another governance stage in this sequence.

Instead, PCA can analyze operational measurements generated by these activities:

    Repository Activity
        |
        +--> review latency
        +--> changed files
        +--> check duration
        +--> review comments
        |
        v
    PCA
        |
        +--> covariance structure
        +--> principal directions
        +--> explained variance
        +--> reduced representation

A merge decision should remain governed by explicit repository policies. A statistical component should not be treated as an authorization mechanism.

## Python Implementation

The Python implementation is the most complete numerical walkthrough.

It provides:

- input validation
- mean centering
- sample covariance construction
- vector normalization
- matrix-vector multiplication
- power iteration
- eigenvalue extraction
- eigenvector deflation
- eigenvalue ordering
- explained-variance calculation
- cumulative variance calculation
- component selection
- projection into principal-component space
- reconstruction
- reconstruction error
- CSV input support
- feature standardization
- executable assertions
- failure handling

The `PCAResult` dataclass keeps the fitted state together. This separates the fitted statistical model from the functions that calculate it.

Power iteration extracts one dominant eigenpair at a time. Deflation removes the contribution of the discovered eigenvector from the working matrix so later iterations can identify subsequent directions.

The implementation also checks for zero-variance data. A dataset in which every feature is constant has no meaningful principal direction.

## JavaScript Implementation

The JavaScript implementation uses Node.js standard-library capabilities and presents PCA as an asynchronous processing workflow.

The numerical model uses:

- classes for PCA-specific errors
- arrays for matrices and vectors
- explicit matrix-vector multiplication
- `Promise`-based dataset processing
- validation of finite numerical values
- covariance construction
- power iteration
- deflation
- projection
- reconstruction
- explained-variance selection

The asynchronous `processDataset` function separates data-processing orchestration from numerical computation. This structure allows a real application to replace the in-memory dataset with asynchronous ingestion without changing the PCA calculation itself.

The implementation deliberately avoids npm dependencies so the file can run directly in a Node.js environment.

## C++17 Implementation

The C++17 program uses repository-operations telemetry as its technical case study.

Its data dimensions represent:

| Feature | Interpretation |
|---|---|
| Review latency | Time associated with review activity |
| Changed files | Size of the proposed change |
| Status-check duration | Automated verification time |
| Review comments | Review discussion volume |

The implementation uses:

- `std::vector`
- structures
- exception-based validation
- `std::inner_product`
- sorting and ordering
- numerical convergence checks
- explicit covariance construction
- power iteration
- eigenvector deflation

The `RepositoryGovernancePCA` class contains the numerical model, while the reporting functions handle presentation.

The program also demonstrates an important architectural distinction: statistical analysis of repository telemetry is separate from authorization or merge enforcement.

## Java 17 Implementation

The Java implementation combines an enterprise-oriented repository governance model with a PCA service.

The domain model includes:

- `ReviewState`
- `CheckState`
- `StatusCheck`
- `ReviewPolicy`
- `PullRequest`
- `PCAResult`

Java records provide immutable domain values for checks, policies, Pull Requests, and PCA results.

The `MergeEligibilityService` evaluates deterministic repository conditions such as:

- Pull Request state
- required approval count
- required status-check success

The PCA service has a separate responsibility. It analyzes numerical repository telemetry and returns covariance, eigenvalues, eigenvectors, explained variance, and transformed observations.

This separation is deliberate. A statistical analysis must not silently become a policy enforcement mechanism.

## PostgreSQL Data Model

The PostgreSQL implementation represents PCA as an auditable relational workflow.

The main relationships are:

    Dataset
       |
       +--> Observation
       |       |
       |       +--> Observation Value
       |
       +--> Feature
       |
       +--> PCA Run
               |
               +--> Covariance Entry
               |
               +--> Principal Component
               |
               +--> Eigenvector Loading
               |
               +--> Projected Observation

### Core Tables

| Table | Purpose |
|---|---|
| `dataset` | Identifies an analytical dataset |
| `observation` | Stores individual observations |
| `feature` | Defines feature names, units, and ordering |
| `observation_value` | Stores feature measurements |
| `pca_run` | Records an analytical PCA execution |
| `covariance_entry` | Stores covariance-matrix elements |
| `principal_component` | Stores eigenvalues and explained variance |
| `eigenvector_loading` | Stores feature loadings |
| `projected_observation` | Stores component scores |

Foreign keys enforce relationships between datasets, observations, features, and PCA runs.

Check constraints prevent invalid numerical states such as negative eigenvalues, invalid variance ratios, or non-finite stored measurements.

## Database Validation

The PostgreSQL script uses triggers for cross-table rules that cannot be expressed naturally with a simple column constraint.

The observation-value trigger verifies that an observation and feature belong to the same dataset.

The projected-observation trigger verifies that the observation belongs to the dataset associated with the PCA run.

A variance-validation trigger prevents the accumulated explained variance from exceeding the mathematical range.

These checks prevent structurally valid rows from representing an invalid analytical relationship.

## SQL Analytical Queries

The script includes queries for:

- feature statistics
- covariance symmetry validation
- principal-component reporting
- dominant feature identification
- cumulative explained variance
- 95% component selection
- projected observation retrieval
- variance-total validation

The `feature_statistics` view reports descriptive statistics including mean, sample variance, standard deviation, minimum, and maximum.

The `pca_variance_report` view exposes eigenvalues and explained variance for reporting.

The `dominant_feature_by_component` view identifies the largest absolute loading for each component.

## PCA Workflow

The complete analytical workflow implemented across the project is:

    1. Validate observations
    2. Establish feature dimensions
    3. Calculate feature means
    4. Center observations
    5. Calculate sample covariance
    6. Decompose the covariance matrix
    7. Order eigenvalues
    8. Associate eigenvectors with eigenvalues
    9. Calculate explained-variance ratios
    10. Calculate cumulative explained variance
    11. Select the required number of components
    12. Project observations
    13. Optionally reconstruct observations
    14. Evaluate reconstruction error

Each stage has a separate technical purpose.

## Component Selection

Component selection can be expressed as:

    smallest k such that

    sum(EVR_1 ... EVR_k) >= threshold

The implementations support a 95% threshold example.

A high cumulative explained variance does not automatically mean that the reduced representation is appropriate for every task. The threshold is a modeling decision and should be interpreted together with the downstream objective.

## Reconstruction

Projection maps centered observations into principal-component coordinates.

Using fewer components loses information.

Reconstruction reverses the transformation approximately:

    X_reconstructed
        =
    projected_components × eigenvectors
        +
    feature_means

The Python implementation calculates reconstruction RMSE to quantify this loss.

A lower reconstruction error indicates that the retained components preserve more of the original numerical variation, though reconstruction quality alone does not establish that the components are useful for a particular business or engineering objective.

## Numerical Stability

Power iteration is intentionally implemented explicitly in the source files.

Its advantages are:

- simple numerical mechanics
- transparent convergence behavior
- low implementation complexity
- suitability for extracting dominant eigenvectors

Its limitations are important:

- convergence can be slow
- closely spaced eigenvalues can make convergence difficult
- repeated deflation can accumulate numerical error
- it is not the preferred approach for large production matrices

Production-scale PCA normally uses optimized numerical linear-algebra implementations with robust eigensolvers or singular-value decomposition.

## Complexity

For a dataset with `n` observations and `p` features:

- mean calculation is approximately `O(np)`
- covariance construction is approximately `O(np²)`
- matrix-vector multiplication is approximately `O(p²)`
- each power-iteration step is approximately `O(p²)`
- extracting multiple components through repeated deflation increases the eigen-analysis cost

Memory usage for a dense covariance matrix is `O(p²)`.

For high-dimensional datasets, the covariance matrix itself can become the dominant memory structure. Sparse, randomized, incremental, or SVD-based approaches may be more appropriate depending on the data shape and analytical objective.

## Edge Cases

The implementations explicitly address several failure conditions.

### Empty Dataset

A PCA model cannot be fitted without observations.

### Single Observation

Sample covariance requires at least two observations.

### Inconsistent Feature Counts

Every observation must have the same number of features.

### Non-Finite Values

`NaN` and infinite values are rejected because they can propagate through covariance and eigenvalue calculations.

### Zero Variance

If all observations are constant, the total covariance variance is zero and meaningful PCA directions do not exist.

### Invalid Component Count

The requested number of components must be between one and the number of features.

### Non-Convergence

Power iteration has a maximum iteration limit. Failure to converge is reported rather than silently returning an unreliable eigenpair.

## Security and Data Handling

PCA itself is not an access-control mechanism.

When PCA is applied to operational or repository data:

- validate imported numerical values
- restrict access to sensitive datasets
- avoid exposing raw observations unnecessarily
- preserve analytical-run metadata
- separate statistical analysis from authorization
- treat database permissions independently from analytical correctness
- validate uploaded CSV data before processing
- avoid dynamically constructing SQL from untrusted identifiers or values

The PostgreSQL model uses typed columns, constraints, foreign keys, and triggers to prevent several classes of invalid database state.

## Implementation Comparison

| Implementation | Primary perspective | Notable mechanism |
|---|---|---|
| Python | Numerical learning and experimentation | Dataclasses, validation, reconstruction, standardization |
| JavaScript | Event-driven processing | Promises, classes, explicit numerical pipeline |
| C++17 | Systems-oriented numerical engine | STL containers, explicit numerical algorithms, exceptions |
| Java 17 | Enterprise domain model | Records, policies, services, immutable collections |
| PostgreSQL | Persistent analytical workflow | Relational constraints, triggers, views, CTEs, transactions |

## Important Distinctions

### Covariance vs Eigenvectors

The covariance matrix describes pairwise feature variation.

Eigenvectors identify directions in feature space that diagonalize that covariance structure.

### Eigenvectors vs Eigenvalues

Eigenvectors define directions.

Eigenvalues quantify variance associated with those directions.

### Explained Variance vs Component Scores

Explained variance describes how much variance a component represents.

Component scores are the coordinates of individual observations after projection onto the component.

### PCA vs Repository Governance

PCA is a statistical transformation.

Repository governance consists of deterministic rules for reviews, approvals, status checks, protected branches, and merge eligibility.

These mechanisms can interact operationally, but they should not be conflated.

## Practical Interpretation

Suppose the first principal component has large positive loadings for review latency, changed files, and review comments.

That component may represent a broad dimension of change complexity or review workload.

That interpretation is an analytical hypothesis, not a universal semantic label. The meaning of a component depends on the original features, their units, preprocessing decisions, and the population being analyzed.

Eigenvector loadings should therefore be interpreted with the original feature definitions rather than treated as independent business metrics.

## Limitations

The implementations use dense matrices and explicit iterative eigenvalue calculations. They are designed for clarity and technical study rather than large-scale numerical production.

The covariance formulation assumes that linear relationships are useful for describing the feature structure.

PCA also does not establish causality. A strong loading or large explained variance indicates statistical structure, not that one repository activity causes another.

The selected number of components should be justified by the intended use rather than by a variance threshold alone.

## Runtime Requirements

| Implementation | Requirement |
|---|---|
| Python | Python 3 with standard library |
| JavaScript | Node.js runtime |
| C++17 | C++ compiler supporting C++17 |
| Java | Java 17 or later |
| PostgreSQL | PostgreSQL database supporting the SQL features used by the script |

No third-party runtime dependencies are required by the source implementations.

## File Responsibilities

The five source files deliberately have different responsibilities.

The Python file emphasizes transparent numerical experimentation and validation.

The JavaScript file emphasizes asynchronous orchestration and object-oriented numerical processing.

The C++17 file emphasizes a systems-oriented implementation of the numerical engine.

The Java 17 file combines immutable domain modeling with an enterprise-style service boundary and keeps deterministic governance rules separate from PCA analysis.

The PostgreSQL file treats PCA as persistent analytical state, using relational constraints and database-side validation to maintain data integrity.
