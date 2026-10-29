# Anomaly Detection: Isolation Forest, Local Outlier Factor, and One-Class SVM

## Scope

This project models anomaly detection for repository-development telemetry. The four principal repository concepts are treated as distinct operational layers:

| Repository concern | Role |
|---|---|
| Pull Requests | Provide the workflow for proposing a set of changes from a source branch toward a target branch |
| Code Review | Evaluates the proposed changes through review decisions, comments, discussions, and requested changes |
| Approvals | Represent explicit reviewer decisions and determine whether configured review requirements are satisfied |
| Branch Protection | Enforces repository-level conditions such as required reviews, required status checks, direct-push restrictions, and merge restrictions |

Anomaly detection is a separate analytical layer. It identifies unusual repository activity that may warrant investigation. An anomaly score does not itself constitute a code-review decision, an approval, or permission to merge.

The implemented relationship is:

Repository → Branches → Commits → Pull Requests → Reviews → Review Comments / Conversations → Status Checks → Anomaly Observations → Branch Protection → Merge Eligibility

The detectors operate on structured telemetry such as changed-file count, additions, deletions, review latency, failed checks, and security-sensitive file scope.

## Detection Models

### Isolation Forest

Isolation Forest identifies observations that can be separated from other observations with relatively few random partitions.

The implementation creates an ensemble of randomized isolation trees. Each tree chooses a feature and a random split value. An observation that reaches a terminal node quickly is considered easier to isolate.

The normalized anomaly score increases as the average path length decreases.

Conceptually:

`shorter isolation path → greater isolation → higher anomaly score`

The Python, C++17, and Java implementations construct trees directly. Random sampling limits each tree's training workload and makes the ensemble less dependent on the complete dataset.

The implementation uses a contamination parameter to determine a score threshold. The threshold is calculated from the observed score distribution rather than being treated as a universal constant.

Isolation Forest is useful when unusual observations are globally separated from the dominant population. It can detect large changes, extreme review latency, unusually high check failures, or unusual combinations of repository activity.

### Local Outlier Factor

Local Outlier Factor is based on local density rather than global separation.

For each observation, the implementation identifies its nearest neighbors and estimates local reachability density. The LOF score compares the density of the observation with the density of its neighborhood.

A value near `1` indicates density comparable to nearby observations. A larger value indicates that the observation is substantially less dense than its neighbors.

Conceptually:

`neighbor density / observation density → LOF score`

The implementation explicitly calculates:

- nearest neighbors
- k-distance
- reachability distance
- local reachability density
- relative local density

LOF is particularly useful when the repository telemetry contains several legitimate clusters. An observation may be globally common but anomalous relative to the local behavior of its peer group.

### One-Class SVM

The One-Class SVM perspective treats normal observations as a boundary-learning problem.

The implementations use an RBF-style kernel:

`K(x, y) = exp(-gamma * ||x - y||²)`

A query's average similarity to the training observations becomes a normality signal. The anomaly score is derived as the inverse of that similarity.

The compact implementations focus on the kernel-based decision mechanism rather than reproducing a complete quadratic-programming optimizer.

The `gamma` parameter controls how quickly similarity decreases with distance. A larger value creates a more localized similarity function. The `nu` parameter controls the expected fraction of observations treated as potential outliers.

This method is useful when anomalies are better described as observations outside a learned normality region rather than simply as points isolated by random partitions.

## Why the Three Detectors Are Kept Separate

The detectors make different assumptions.

| Detector | Primary signal | Useful anomaly pattern |
|---|---|---|
| Isolation Forest | Isolation depth | Globally unusual activity |
| LOF | Local density | Unusual behavior within a peer cluster |
| One-Class SVM | Kernel similarity / boundary | Activity outside learned normality |

They should not be interpreted as interchangeable scoring functions.

The implementations therefore preserve detector-specific scores and thresholds. A two-of-three consensus can be used as an operational alert policy, but consensus is itself a policy decision rather than part of the mathematical definition of any one detector.

## Repository Domain Model

A Pull Request begins as a proposed change between two branches.

The source branch contains the proposed commits. The target branch identifies where the change is intended to land. The current source commit is important because reviews and status checks can become stale when the source branch changes.

Code Review evaluates the actual changes. Reviewers can submit comments, request changes, or approve a revision. Review comments can be associated with files and lines, while conversations represent broader review discussions that may block merging until resolved.

Approvals are explicit decisions. An approval is tied to a reviewer and the commit that was reviewed. If the source branch changes, an earlier approval may no longer represent approval of the current revision.

Branch Protection is repository governance. It specifies requirements independently of an individual reviewer's opinion. A protected branch can require a minimum number of reviews, successful status checks, resolved conversations, restrictions on direct pushes, restrictions on force pushes, and other controls.

Merge eligibility is the resulting decision after these conditions are evaluated.

## Pull Request Lifecycle

A realistic lifecycle is:

`source branch`
  
`→ commits`
  
`→ Pull Request opened`
  
`→ reviewers requested`
  
`→ automated checks`
  
`→ code review`
  
`→ review comments / conversations`
  
`→ requested changes or approvals`
  
`→ source branch synchronization`
  
`→ possible stale approvals`
  
`→ current status checks`
  
`→ branch-protection evaluation`
  
`→ merge eligibility`
  
`→ merge`

A Draft Pull Request represents work that is not yet intended for normal merge processing. The implementation models draft status explicitly.

Closing a Pull Request ends its active workflow without merging it. Reopening changes its workflow state back to open and requires the current branch and governance conditions to be evaluated again.

A merge conflict is a source-versus-target state problem. It is distinct from a failed review. A review can be positive while the source branch is no longer cleanly mergeable against the target branch.

## Code Review Mechanics

Code Review is modeled around reviewer decisions and discussion state.

A review has:

- a reviewer
- a Pull Request
- a review state
- the commit SHA reviewed
- a submission timestamp

The supported review states are:

- `commented`
- `approved`
- `changes_requested`
- `dismissed`

Inline review comments can identify a file and optional line number. A resolved comment records that its associated discussion has been addressed.

The implementations deliberately retain the distinction between a review and a review comment. A review is a decision-bearing object. A comment is evidence or discussion associated with that review.

Review quality is represented through explicit criteria rather than a single numerical quality score. In an operational repository, review criteria can include correctness, security impact, test coverage, maintainability, compatibility, and scope.

A reviewer should evaluate the changes represented by the current source revision. This is why the examples associate review decisions with commit SHAs.

## Approvals

An approval means that an eligible reviewer has accepted the reviewed revision according to the review workflow.

The database stores the reviewed commit SHA. This allows the system to distinguish a current approval from a stale approval.

For example:

`approved commit = abc123`

followed by:

`current source commit = def456`

means that the approval was recorded for an older revision.

The PostgreSQL eligibility function counts only approvals matching the current source commit. This prevents an old approval from automatically satisfying a policy after the proposed changes have changed.

Approval requirements can specify:

- minimum approvals
- minimum distinct reviewers
- code-owner requirements

Reviewer eligibility is separate from the approval count. A repository can restrict who may merge or who may provide valid approvals.

A Pull Request author is prevented by a database trigger from approving their own Pull Request. This is a governance constraint rather than an anomaly-detection rule.

## Branch Protection

Branch Protection operates at the target branch level.

The PostgreSQL policy model includes:

| Policy | Purpose |
|---|---|
| Required reviews | Requires a minimum number of valid reviews |
| Stale approval dismissal | Prevents old approvals from automatically carrying forward |
| Conversation resolution | Requires review discussions to be resolved |
| Required status checks | Requires configured automation checks to pass |
| Direct-push restriction | Prevents ordinary direct updates to a protected branch |
| Force-push restriction | Protects branch history from destructive rewrites |
| Branch deletion restriction | Prevents accidental removal of the protected branch |
| Linear-history requirement | Supports repositories requiring a linear commit history |
| Administrator bypass | Represents whether an administrator exception exists |

Branch protection is not the same thing as code review.

A reviewer decides whether a change is acceptable from a review perspective. Branch protection defines conditions under which repository policy permits the change to merge.

## Status Checks

Status checks are attached to a Pull Request and a specific source commit.

The sample repository requires:

- `unit-tests`
- `security-scan`
- `integration-tests`

A check can be:

- `pending`
- `passed`
- `failed`
- `cancelled`

The merge-eligibility function requires every configured check to pass against the current source commit.

A passing check for an old commit is not treated as evidence that the current source revision passed.

This distinction is essential when the Pull Request receives new commits after an earlier successful test run.

## Anomaly Detection Features

The example telemetry uses six numerical features:

| Feature | Interpretation |
|---|---|
| Changed files | Breadth of the modification |
| Additions | Number of added lines |
| Deletions | Number of removed lines |
| Review latency | Time associated with the review process |
| Failed checks | Automated validation failures |
| Security-sensitive scope | Amount of security-sensitive change |

These values are synthetic operational measurements. They are not claims about the normal behavior of a particular organization.

Before distance- or kernel-based detection, the implementations standardize each feature. This prevents a feature measured in thousands of lines from dominating a feature measured in hours merely because of its numeric scale.

A zero-variance feature is assigned a scale of `1` so that it does not cause division by zero during standardization.

## Python Implementation

The Python source is a complete standard-library implementation.

Its principal components are:

- `Event` dataclass for telemetry
- `StandardScaler` for feature normalization
- `IsolationForest` and `IsolationTree`
- `LocalOutlierFactor`
- `OneClassSVM`
- confusion-matrix evaluation
- detector disagreement analysis
- two-of-three consensus analysis

The script generates synthetic repository activity, inserts several known anomalous profiles, standardizes the features, fits the three detectors, and evaluates their results.

The implementation also reports precision, recall, F1, and accuracy against the synthetic labels.

The Python code includes assertions as executable regression checks. These validate result cardinality, score finiteness, and metric ranges.

The script uses no third-party packages.

## JavaScript Implementation

The JavaScript implementation treats anomaly detection as an event-driven monitoring service.

`MonitoringEventBus` uses Node.js `EventEmitter` to represent telemetry ingestion.

`RepositoryTelemetryService` receives events, validates them, stores them, and evaluates the configured detectors.

The implementation uses:

- JavaScript classes
- `Map`-like data structures through object-oriented result collections
- `Set`-style logical consensus
- Promises
- `setImmediate`
- Node.js `crypto`
- validation exceptions
- asynchronous ingestion
- detector objects

The event-driven structure makes the implementation different from the Python batch-oriented design.

The JavaScript implementation demonstrates a monitoring flow in which telemetry can arrive asynchronously before detection is performed.

## C++17 Implementation

The C++17 implementation presents anomaly detection as a repository telemetry engine.

Its architectural emphasis is on:

- strong domain types
- STL containers
- `std::unique_ptr`
- deterministic random generation
- explicit tree nodes
- exception-based validation
- structured metrics
- resource-conscious data handling

The Isolation Forest implementation uses dynamically allocated tree nodes managed through `std::unique_ptr`. This avoids manual ownership management for the tree hierarchy.

The LOF implementation stores neighbor relationships explicitly and calculates local reachability density.

The One-Class SVM implementation uses the RBF similarity function directly.

The program evaluates the three detector outputs and calculates a two-of-three consensus.

The implementation is compilable with C++17 and requires no external libraries.

## Java 17 Implementation

The Java implementation models anomaly detection as an enterprise governance service.

Important domain types include:

- `Telemetry`
- `DetectorResult`
- `GovernanceDecision`
- `RiskLevel`
- `EvaluationState`
- `DetectorType`

Java records provide immutable data-transfer objects. The `Detector` interface defines a common detection contract without forcing detector implementations to share their internal algorithms.

`RiskPolicy` separates anomaly evidence from governance classification.

The `RepositoryRiskPolicy` maps the number of detector triggers to operational risk:

| Trigger count | Risk |
|---:|---|
| 0 | LOW |
| 1 | MEDIUM |
| 2 | HIGH |
| 3 | CRITICAL |

The resulting `GovernanceDecision` also records whether the observation requires review.

This separation is important: anomaly detection produces evidence, while policy determines what operational action follows.

## PostgreSQL Implementation

The PostgreSQL model represents the repository domain relationally.

Core entities include:

- repositories
- users
- repository membership
- branches
- commits
- Pull Requests
- Pull Request commits
- review requests
- reviews
- review comments
- status checks
- branch-protection policies
- required status checks
- approval requirements
- conversations
- anomaly observations

Primary keys provide stable identity. Foreign keys maintain relationships between repository objects.

Check constraints validate numeric ranges, review state requirements, commit identifiers, conversation resolution timestamps, and approval requirements.

Indexes are concentrated on queries that are expected to filter frequently:

- Pull Request repository and state
- review state
- status-check state
- unresolved conversations
- anomaly state
- commit time

The database uses PostgreSQL enumerated types for domain states where a constrained vocabulary is useful.

## Database-Level Governance

The `prevent_self_approval` trigger rejects an approval where the reviewer is the Pull Request author.

The `enforce_conversation_resolution` trigger prevents a conversation from being marked resolved without a resolution timestamp.

The `merge_eligibility` function evaluates repository policy against the current Pull Request revision.

It checks:

1. Pull Request existence
2. Pull Request state
3. Draft status
4. target branch policy
5. active change requests
6. unresolved conversations
7. required approvals
8. current-commit approval validity
9. required status checks
10. failed checks
11. pending checks

The function returns a Boolean eligibility result and a reason.

This makes the database capable of explaining why a merge is currently blocked rather than returning only a binary result.

## Stale Approvals

Stale approvals are an important boundary between review and branch governance.

A review can be valid for commit `A` while the Pull Request now points to commit `B`.

The database therefore stores:

`review.commit_sha`

and:

`pull_request.source_head_sha`

The eligibility function counts only approvals whose commit SHA matches the current source head.

The reporting query also identifies approvals whose commit differs from the current source revision.

This prevents an earlier approval from silently approving later changes.

## Merge Eligibility

Merge eligibility is the intersection of independent conditions.

A simplified decision model is:

`Pull Request open`
  
`AND not draft`
  
`AND required approvals satisfied`
  
`AND no active change requests`
  
`AND conversations resolved`
  
`AND required checks passed`
  
`AND checks refer to current source revision`
  
`AND branch policy permits merge`

Only when these conditions hold does the database function return an eligible decision.

The presence of an anomaly score does not automatically block a merge. Anomaly observations are persisted as separate evidence. A repository can choose to require investigation when one or more detectors flag unusual behavior, but that operational policy should be explicitly configured rather than hidden inside an anomaly algorithm.

## Detector Disagreement

Different detectors can disagree because they measure different forms of abnormality.

For example, an observation can be:

- globally unusual but locally similar
- locally unusual but globally plausible
- outside a kernel-defined normality region
- unusual according to multiple detectors

The Python implementation explicitly reports disagreement. The JavaScript and C++ implementations also support detector-level results before consensus is applied.

A two-of-three policy reduces dependence on one detector but can reduce recall when only one algorithm identifies a subtle anomaly.

The correct threshold depends on the operational cost of false positives and false negatives.

## Edge Cases

The implementations validate several important edge conditions:

- empty datasets
- too few observations
- inconsistent feature dimensions
- non-finite numeric values
- zero-variance features
- invalid contamination
- invalid `nu`
- invalid kernel parameters
- invalid neighbor counts
- empty detector training state
- mismatched evaluation arrays
- stale review approvals
- unresolved conversations
- failed status checks
- pending status checks
- draft Pull Requests
- active change requests
- self-approval attempts

These cases are treated as explicit validation or governance failures rather than silently producing an apparently valid result.

## Performance Considerations

The implementations are deliberately dependency-free learning models, so their computational characteristics differ from optimized production libraries.

Isolation Forest reduces work by training trees on sampled subsets. For a tree with sample size `s`, the expected depth is approximately logarithmic in `s`, while an ensemble requires repeated traversal of the trees for each observation.

LOF requires nearest-neighbor distance calculations. The simple implementation uses pairwise distance computation, giving quadratic behavior in the number of observations for neighborhood construction.

The One-Class SVM-style implementation evaluates kernel similarity against the training observations. This also has quadratic behavior when every training observation is compared with every observation being scored.

For production-scale telemetry, approximate nearest-neighbor methods, optimized numerical implementations, batch processing, dimensionality reduction, model persistence, and controlled retraining schedules may be necessary.

The examples intentionally expose the underlying mechanisms rather than hiding them behind external numerical libraries.

## Security Considerations

Repository telemetry can reveal sensitive information even when source code is not stored directly.

Operational controls should include:

- authentication of telemetry producers
- authorization for telemetry readers
- encryption in transit
- encryption at rest
- restricted database roles
- audit logging
- feature-schema validation
- model and threshold versioning
- protection against malicious telemetry injection
- retention limits
- separation of anomaly evidence from merge authorization

An anomaly detector should not become an implicit authorization mechanism. A malicious actor who can influence detector inputs should not be able to bypass branch protection merely by manipulating anomaly scores.

Branch-protection enforcement should remain authoritative.

## Failure Handling

The source implementations favor explicit failure behavior.

Python raises `DetectionError`.

JavaScript uses `ValidationError` and terminates the process for unrecoverable monitoring failures.

C++17 uses `ValidationError` derived from `std::runtime_error`.

Java 17 uses `ValidationException` and explicit domain validation.

PostgreSQL uses constraints, triggers, and exceptions to reject invalid state transitions.

This produces a consistent principle across the five implementations:

`invalid input → explicit failure`

rather than:

`invalid input → silent coercion → unreliable decision`

## Practical Interpretation

The most important distinction is between an unusual event and an invalid repository operation.

An anomaly detector answers a statistical question:

`Is this observation unusual relative to the learned or observed normal population?`

Code review answers a software-engineering question:

`Are the proposed changes correct, safe, maintainable, and appropriate?`

An approval answers a reviewer-decision question:

`Has an eligible reviewer accepted the reviewed revision?`

Branch protection answers a governance question:

`Does the target branch permit this change to merge under its configured rules?`

Merge eligibility combines these repository conditions without making statistical anomaly detection a substitute for governance.

## File Responsibilities

| File | Primary responsibility |
|---|---|
| Python | Batch anomaly-detection simulation with three detectors and evaluation |
| JavaScript | Asynchronous telemetry ingestion and event-driven detection service |
| C++17 | Repository telemetry detection engine with explicit STL-based architecture |
| Java 17 | Enterprise governance model with domain types, policies, and state transitions |
| PostgreSQL | Relational repository model, review workflow, branch protection, anomaly observations, and merge eligibility |

The five implementations use the same conceptual domain while deliberately emphasizing different engineering mechanisms.

## Operational Decision Flow

A practical deployment can be represented as:

`Telemetry`
  
`↓`
  
`Schema validation`
  
`↓`
  
`Feature normalization`
  
`↓`
  
`Isolation Forest ─┐`
  
`LOF ──────────────┼→ Detector evidence`
  
`One-Class SVM ────┘`
  
`↓`
  
`Risk / investigation policy`
  
`↓`
  
`Pull Request + current commit`
  
`↓`
  
`Reviews + approvals + status checks + conversations`
  
`↓`
  
`Branch Protection`
  
`↓`
  
`Merge Eligibility`

The anomaly layer provides evidence. Repository governance remains responsible for the final merge decision.
