# Unsupervised Learning: Clustering, Dimensionality Reduction, and Anomaly Detection

## Scope

This project implements three major areas of unsupervised learning:

- **Clustering** groups observations according to similarity without predefined class labels.
- **Dimensionality reduction** transforms high-dimensional observations into a smaller representation while attempting to preserve important structure.
- **Anomaly detection** identifies observations whose behavior differs substantially from the normal population.

The five source files use the same realistic customer-behavior dataset while applying different engineering perspectives. The implementations are deliberately independent rather than being literal translations of one another.

The central workflow is:

`raw observations → validation → scaling → representation → clustering → anomaly analysis → operational decision`

The repository model is also represented as a separate governance scenario where machine-learning changes require review, approvals, automated checks, and protected-branch enforcement.

## Conceptual Model

Unsupervised learning operates without a target variable that specifies the correct output for every observation.

For an observation represented as a vector:

`x = (x1, x2, ..., xd)`

the algorithms attempt to discover structure directly from the feature space.

| Area | Main question | Typical output |
|---|---|---|
| Clustering | Which observations are similar? | Cluster assignments and cluster representatives |
| Dimensionality reduction | Which lower-dimensional representation preserves useful structure? | Reduced feature coordinates |
| Anomaly detection | Which observations behave unusually? | Anomaly scores or anomaly flags |

These outputs are related but are not interchangeable.

A clustering algorithm partitions observations. A dimensionality-reduction method changes representation. An anomaly detector assigns unusualness. A single application can use all three.

## Dataset

The examples use ten customer observations with three numerical features.

The observations intentionally contain several groups:

- A low-valued group around one unit.
- A high-valued group around eight units.
- A middle group around four units.
- One observation with an unusually large first feature.

This structure provides a compact dataset for demonstrating both regular grouping and unusual behavior.

The observations are identified by keys such as `customer-01` and `customer-10`. The identifiers are not model features. This separation prevents operational identity from accidentally influencing numerical similarity.

## Validation

Validation is performed before model operations.

Important validation rules include:

- The dataset cannot be empty.
- Every observation must have an identifier.
- Observation identifiers must be unique.
- Every observation must contain at least one feature.
- Feature values must be numeric and finite.
- Every observation must have the same dimensionality.
- The number of clusters cannot exceed the number of observations.
- Algorithm configuration values must be valid.

These checks matter because unsupervised algorithms can produce plausible-looking but incorrect results when malformed data is silently accepted.

## Feature Scaling

The Python and JavaScript implementations standardize the input features before distance-based processing.

For a feature value `x`, mean `μ`, and standard deviation `σ`, the standardized value is:

`z = (x - μ) / σ`

A constant feature has zero variance. The implementations map such a feature to zero rather than dividing by zero.

Scaling is particularly important for Euclidean-distance clustering. If one feature ranges from 0 to 1 while another ranges from 0 to 100000, the larger-scale feature can dominate the distance calculation.

Scaling therefore changes the geometry used by the clustering algorithm and should be treated as a modeling decision rather than a cosmetic preprocessing step.

## Clustering

### K-Means

The implementations use K-Means as the primary clustering algorithm.

K-Means maintains `k` centroids. Each observation is assigned to its nearest centroid, after which every centroid is recomputed as the arithmetic mean of its assigned observations.

The iterative workflow is:

`initial centroids → assignment → centroid update → movement test → repeat`

The Python, JavaScript, C++, and Java implementations use deterministic initial centroids derived from the first observations. This makes the learning artifacts reproducible.

A production implementation would normally consider multiple initializations because K-Means can converge to different local solutions depending on initialization.

### Distance

The implementations use Euclidean distance:

`d(x, y) = sqrt(sum((xi - yi)^2))`

For clustering assignment, squared distance can be used because the square-root operation does not change the ordering of distances. The implementations retain the ordinary Euclidean form for clarity.

### Empty Clusters

An assignment step can theoretically leave a centroid with no members.

The implementations retain the previous centroid when this occurs. This prevents the model from silently reducing the requested number of clusters.

More sophisticated production systems can reinitialize an empty centroid using a distant observation or use an initialization method designed to reduce this problem.

### Inertia

The Python implementation calculates clustering inertia:

`inertia = sum(distance(observation, assigned_centroid)^2)`

Lower inertia means observations are closer to their assigned centroids, but lower inertia alone does not prove that the selected number of clusters is meaningful.

Increasing `k` generally reduces within-cluster error, so model selection requires an external criterion such as domain interpretability, stability analysis, silhouette measurements, or another appropriate evaluation strategy.

## Dimensionality Reduction

The Python implementation contains a dependency-free two-component PCA-style transformation.

Principal Component Analysis seeks directions of maximum variance.

The workflow is:

`center features → compute covariance → estimate first direction → deflate covariance → estimate second direction → project observations`

The implementation estimates principal directions using power iteration.

For a covariance matrix `C`, power iteration repeatedly applies:

`v(next) = normalize(Cv)`

The process converges toward a dominant eigenvector when the required numerical conditions are satisfied.

The covariance matrix is then deflated so that a second direction can be estimated.

The resulting two coordinates are reported as the first and second principal components.

This implementation is intentionally small and transparent. It is suitable for understanding the mechanics of the transformation, not as a replacement for optimized numerical linear-algebra libraries on large datasets.

## Why Dimensionality Reduction Is Different from Clustering

Dimensionality reduction does not inherently create groups.

For example:

`10-dimensional observations → 2-dimensional coordinates`

is a representation change.

Clustering instead produces something like:

`observation A → cluster 0`

The two operations can be combined:

`high-dimensional data → dimensionality reduction → clustering`

or:

`high-dimensional data → scaling → clustering`

The appropriate order depends on the objective and the algorithm. Reducing dimensions can remove noise, but it can also discard information that is important for separating clusters.

## Anomaly Detection

The Python and JavaScript implementations use a simple distance-based anomaly detector.

First, a global centroid is calculated. Each observation receives its Euclidean distance from that centroid.

Let the resulting distances be `d1, d2, ..., dn`.

The detector calculates:

`threshold = mean(distance) + multiplier × standard_deviation(distance)`

An observation is classified as anomalous when:

`distance > threshold`

The sample contains an observation with a much larger first feature, making it useful for demonstrating the distinction between normal population structure and unusual observations.

This method is deliberately simple. It should not be interpreted as universally reliable anomaly detection.

## Limitations of Distance-Based Anomaly Detection

A global centroid assumes that unusualness can be represented by distance from one central point.

That assumption can fail when:

- The population contains multiple legitimate clusters.
- Clusters have different densities.
- Features are strongly correlated.
- The data contains heavy-tailed distributions.
- Normal observations naturally occur far from the global centroid.
- Feature scaling is inappropriate.

For clustered data, a point can be far from the global centroid while being perfectly normal within its own cluster.

Production anomaly detection can therefore require cluster-aware distances, density-based methods, isolation methods, robust statistics, or domain-specific thresholds.

## Python Implementation

The Python file provides the most complete dependency-free analytical workflow.

It defines:

- `Point` as an immutable observation representation.
- `Cluster` as a mutable clustering result.
- `Anomaly` as an anomaly result record.
- `KMeans` as an iterative clustering model.
- `PCA2D` as a compact principal-component transformation.
- `detect_anomalies()` as a distance-based detector.
- Validation and exception handling for invalid datasets.

The script standardizes observations before applying the models.

It reports:

- Cluster membership.
- Cluster centroids.
- K-Means inertia.
- Two-dimensional PCA coordinates.
- Detected anomalies.

Assertions provide lightweight executable checks that clustering does not lose observations and that the requested number of clusters is produced.

The implementation uses only the Python standard library.

## JavaScript Implementation

The JavaScript file emphasizes event-driven processing and runtime validation.

Its `Observation` class validates and freezes feature data. `EventBus` provides an event-driven mechanism for reporting model progress.

`KMeansEngine` emits iteration events containing centroid movement and emits a final fitting event.

This design is useful when model computation is part of a larger Node.js service where progress, logging, telemetry, or user-interface updates may be triggered asynchronously.

The file also writes an `unsupervised-learning-audit.json` artifact using the Node.js filesystem API.

The asynchronous `runWorkflow()` function establishes a service-oriented execution boundary while the actual mathematical operations remain synchronous and deterministic.

## C++17 Implementation

The C++ program uses STL containers and explicit structures to represent observations, clusters, status checks, Pull Requests, and branch policies.

The clustering implementation uses:

- `std::vector` for numerical features and collections.
- `std::set` for identifier validation.
- `std::find_if` for observation lookup.
- Exceptions for invalid data and configuration.
- Explicit convergence checking.

The program also contains a governance engine.

This second component models a realistic engineering workflow in which an unsupervised-learning change is proposed through a Pull Request and must satisfy repository-level merge conditions.

The governance engine evaluates:

- Draft status.
- Source and target branches.
- Empty changesets.
- Merge conflicts.
- Required approvals.
- Resolved review conversations.
- Automated status checks.

The example intentionally fails one automated data-quality check, demonstrating that approvals alone do not guarantee merge eligibility.

The C++ design therefore separates analytical computation from repository enforcement.

## Java 17 Implementation

The Java implementation models domain behavior through explicit enterprise-style types.

Important types include:

| Type | Responsibility |
|---|---|
| `Observation` | Immutable numerical observation |
| `StatusCheck` | Automated validation result |
| `Review` | Reviewer decision and eligibility |
| `BranchPolicy` | Repository enforcement rules |
| `PullRequest` | Mutable workflow state |
| `MergeEligibilityService` | Centralized merge-policy evaluation |
| `KMeansService` | Clustering operation |

Java records are used for immutable values such as observations, reviews, checks, and branch policies.

`PullRequest` is deliberately stateful because workflow state changes over time.

The `MergeEligibilityService` keeps repository policy decisions separate from the Pull Request data structure. This makes policy evaluation explicit and reduces the risk of mixing data storage with governance rules.

The model also distinguishes:

- `PENDING`
- `APPROVED`
- `CHANGES_REQUESTED`
- `COMMENTED`
- `DISMISSED`

from Pull Request merge states such as:

- `OPEN`
- `BLOCKED`
- `MERGEABLE`
- `MERGED`
- `CLOSED`

This distinction is important because a review decision is not itself a merge state.

## PostgreSQL Implementation

The PostgreSQL script provides a relational representation of both the analytical and governance domains.

The analytical model contains:

`datasets → observations → models → clusters → cluster_memberships`

and:

`models → anomaly_scores`

The database uses PostgreSQL arrays for numerical feature vectors because the learning artifact needs to retain variable-length feature collections while enforcing dimensionality through a trigger.

The `validate_feature_dimensions()` trigger checks each observation against the declared feature count of its dataset.

This moves an important data-integrity rule into the database rather than depending exclusively on application code.

## Relational Governance Model

The repository model contains:

`repositories → branches → pull_requests → reviews → review_comments`

and:

`pull_requests → status_checks`

with:

`branches → branch_protection_policies`

This structure distinguishes several concepts that should not be collapsed into one table.

A Pull Request represents a proposed change.

A review represents an evaluation by a reviewer.

A review comment represents discussion at a specific location.

A status check represents an automated condition.

A branch-protection policy represents repository-level enforcement.

Merge eligibility is therefore derived from multiple independent conditions.

## Merge Eligibility

The PostgreSQL function `refresh_pull_request_state()` evaluates repository conditions and updates the Pull Request state.

The function considers:

- Draft status.
- Merge conflicts.
- Required approvals.
- Active reviewer eligibility.
- Status checks.
- Unresolved review comments.
- Conversation resolution.

The resulting state is either `BLOCKED` or `MERGEABLE`.

The design demonstrates an important distinction:

`approval ≠ merge eligibility`

An approved Pull Request can still be blocked by failing tests, unresolved conversations, conflicts, or insufficient repository-level requirements.

## Review and Approval Semantics

The relational model stores reviewer decisions in `reviews`.

A review may be:

- `PENDING`
- `APPROVED`
- `CHANGES_REQUESTED`
- `COMMENTED`
- `DISMISSED`

An approval is therefore a specific review state rather than a generic Boolean attached to the Pull Request.

The database counts approvals only from active reviewers.

This provides a foundation for more advanced policies such as:

- Required reviewer groups.
- Code-owner restrictions.
- Reviewer independence requirements.
- Stale-approval invalidation.
- Approval dismissal after significant changes.
- Different approval requirements for different protected branches.

The example keeps the policy compact while preserving these conceptual distinctions.

## Branch Protection

Branch protection is modeled separately from review activity.

The policy contains explicit controls for:

- Required approvals.
- Required status checks.
- Conversation resolution.
- Direct-push restrictions.
- Force-push restrictions.
- Linear-history requirements.

These are repository governance conditions, not properties of an individual review.

For example, a reviewer can approve a Pull Request while the protected branch still rejects the merge because direct pushes are prohibited or required automated checks have not passed.

The policy table therefore represents constraints imposed by the target branch.

## Relationship Between the Four Governance Concepts

The repository workflow can be represented as:

`Pull Request`
`    ↓`
`Code Review`
`    ↓`
`Approval decisions`
`    ↓`
`Automated status checks + branch protection`
`    ↓`
`Merge eligibility`
`    ↓`
`Merge`

Each layer has a distinct purpose.

| Concept | Primary responsibility |
|---|---|
| Pull Request | Propose and track a change between branches |
| Code Review | Evaluate the technical change |
| Approval | Record an eligible reviewer's decision |
| Branch Protection | Enforce repository-level merge conditions |

This separation prevents a common modeling error where all merge conditions are represented as a single `approved` flag.

## Failure Conditions

The implementations explicitly handle several failure classes.

### Data failures

Examples include:

- Empty datasets.
- Duplicate identifiers.
- Missing identifiers.
- Non-finite feature values.
- Inconsistent feature dimensions.
- Invalid cluster counts.

### Modeling failures

Examples include:

- Empty clusters.
- Zero-variance features.
- Invalid convergence settings.
- Excessive cluster count.

### Workflow failures

Examples include:

- Draft Pull Requests.
- Merge conflicts.
- Missing approvals.
- Ineligible reviewers.
- Failing status checks.
- Unresolved review conversations.

These failure classes belong to different layers and should not be handled with a single generic error mechanism.

## Performance Considerations

For K-Means with:

- `n` observations,
- `k` clusters,
- `d` dimensions,
- `i` iterations,

the dominant computational cost is approximately:

`O(n × k × d × i)`

Memory usage for the basic representation is approximately proportional to:

`O(n × d + k × d)`

The example implementations intentionally favor clarity over high-performance numerical kernels.

The C++ implementation provides lower-level control over allocation and data representation.

The Java implementation uses immutable records for stable value objects and collections for model state.

The JavaScript implementation uses arrays, Maps, Sets, and event handlers suitable for Node.js service workflows.

The PostgreSQL design adds indexes where query patterns justify them, particularly for Pull Request state, review state, dataset status, and cluster membership.

## Security Considerations

Unsupervised learning systems can create security and operational risks even when the algorithms themselves are mathematically correct.

Important controls include:

- Validate all externally supplied numerical values.
- Reject non-finite values.
- Prevent malformed feature arrays from entering the model.
- Keep operational identifiers separate from numerical features.
- Treat uploaded datasets as untrusted input.
- Avoid executing arbitrary content embedded in dataset metadata.
- Restrict database modification privileges.
- Use transactions for related workflow updates.
- Record review and status-check decisions.
- Enforce branch-level governance independently of application UI.

The repository governance model demonstrates why application-level approval indicators should not be treated as the sole enforcement mechanism.

## Practical Interpretation

Clusters are useful when the goal is to identify groups such as customer segments, behavioral patterns, or operational profiles.

Dimensionality reduction is useful when the original feature space is difficult to inspect or when a lower-dimensional representation is useful for visualization or downstream processing.

Anomaly detection is useful when the objective is to identify observations that require investigation.

The three outputs should be interpreted together rather than assuming that any one algorithm reveals the complete structure of a dataset.

For example:

`cluster assignment = group membership`

does not mean:

`cluster assignment = business category`

Likewise:

`anomaly flag = unusual according to this detector`

does not necessarily mean:

`anomaly flag = fraudulent or incorrect`

Domain validation remains necessary before operational action.

## Implementation Comparison

| File | Primary perspective | Important mechanism |
|---|---|---|
| Python | Analytical modeling | K-Means, PCA-style projection, anomaly scoring |
| JavaScript | Event-driven processing | Event bus, validation, asynchronous workflow |
| C++17 | Systems-oriented implementation | STL-based clustering and governance engine |
| Java 17 | Enterprise domain model | Records, policies, state transitions, services |
| PostgreSQL | Persistent relational workflow | Constraints, triggers, views, functions, transactions |

The implementations intentionally use different strengths instead of presenting five syntactic translations of the same program.

## End-to-End Architecture

The analytical portion can be viewed as:

`Dataset`
`  │`
`  ├── Validation`
`  │`
`  ├── Standardization`
`  │`
`  ├── Clustering ──────────────→ Cluster membership`
`  │`
`  ├── Dimensionality reduction → Reduced representation`
`  │`
`  └── Anomaly detection ───────→ Anomaly score`

The engineering-governance portion is:

`Model change`
`    │`
`    ↓`
`Pull Request`
`    │`
`    ↓`
`Code Review`
`    │`
`    ├── Review comments`
`    ├── Approval decisions`
`    └── Requested changes`
`    │`
`    ↓`
`Automated status checks`
`    │`
`    ↓`
`Protected branch policy`
`    │`
`    ↓`
`Merge eligibility`
`    │`
`    ↓`
`Merge`

These are related workflows but represent different responsibilities.

## Production Boundaries

The implementations are deliberately self-contained and dependency-light.

A production-scale unsupervised-learning system would require stronger numerical infrastructure, model versioning, reproducibility controls, monitoring, data-quality management, experiment tracking, access control, and systematic model evaluation.

The dependency-free PCA implementation is particularly limited for large numerical workloads. Its purpose is to expose the mechanics of covariance estimation, eigenvector approximation, and projection.

Similarly, the simple distance-based anomaly detector should not be treated as a universal anomaly-detection strategy.

The strongest engineering principle demonstrated by the artifacts is separation of concerns:

`data validation → mathematical processing → model output → operational governance`

Each layer has different correctness conditions and should be tested and governed accordingly.
