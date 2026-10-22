# Support Vector Machines: Margins, Hyperplanes, Support Vectors, and Soft Margins

## Scope

This project develops a linear Support Vector Machine (SVM) around four closely related concepts:

- **Hyperplanes** define the separating decision boundary.
- **Margins** measure how far observations are from that boundary.
- **Support vectors** are the observations closest to the decision boundary and therefore strongly influence the learned classifier.
- **Soft margins** permit selected observations to violate the ideal margin through hinge loss and regularization.

The five implementations use a common technical scenario: evaluating repository changes according to measurable engineering-risk features. The classification model is deliberately kept separate from deterministic repository governance. An SVM can provide a risk classification, but a classification result is not equivalent to an authorization decision.

The implementations use a two-dimensional feature space:

| Feature | Interpretation |
|---|---|
| Security-defect signal | Relative quantity of security defects associated with a change |
| Test-coverage signal | Relative testing coverage associated with a change |

The binary labels are represented as `+1` and `-1`, which is the conventional formulation for the linear SVM objective.

---

## Mathematical Model

A linear SVM represents a hyperplane as

`w · x + b = 0`

where:

- `w` is the normal vector to the hyperplane.
- `x` is a feature vector.
- `b` is the bias or intercept.

The classifier uses the sign of the decision function:

`f(x) = w · x + b`

An observation is classified as `+1` when `f(x) >= 0` and as `-1` otherwise.

For an observation with label `y ∈ {-1, +1}`, the functional margin is:

`y(w · x + b)`

A positive functional margin means that the observation is on the correct side of the hyperplane. A functional margin of at least `1` lies outside the canonical margin boundary used by the standard soft-margin formulation.

The geometric distance from an observation to the hyperplane is related to the norm of `w`. The canonical margin boundaries are:

`w · x + b = +1`

and

`w · x + b = -1`

The distance between those boundaries is:

`2 / ||w||`

Consequently, maximizing the margin is equivalent to controlling the magnitude of the weight vector.

---

## Soft-Margin Objective

Perfect separation is often unrealistic. Real datasets can contain noise, overlapping classes, measurement errors, or observations that naturally fall inside the margin.

The implementations therefore use the soft-margin objective:

`1/2 ||w||² + C Σ max(0, 1 - y(w · x + b))`

The first term is the L2 regularization component.

The second term is the hinge-loss component.

`C` controls the relative penalty assigned to margin violations.

A smaller `C` permits a stronger trade-off between classification violations and a small weight norm. A larger `C` places greater emphasis on reducing training violations, potentially producing a narrower margin and greater sensitivity to difficult observations.

The code does not treat `C` as a universal optimal value. It is exposed as a model parameter and the Python implementation compares several values to show how regularization changes the learned model.

---

## Why Support Vectors Matter

For a hard-margin problem, the observations defining the closest boundary points determine the separating hyperplane. In a soft-margin problem, observations near or inside the margin have non-zero hinge-loss contributions and therefore strongly influence optimization.

The implementations identify observations with functional margins approximately at or below `1` as support-vector candidates.

This distinction is important:

- A correctly classified point far outside the margin has zero hinge loss.
- A point close to the margin can influence the learned boundary.
- A point inside the margin contributes positive hinge loss.
- A misclassified point contributes an even larger hinge-loss penalty.
- A noisy observation can therefore influence the model substantially when `C` is large.

The exact dual optimization interpretation of support vectors depends on the optimization formulation and tolerance used. The examples use the functional-margin criterion as an explicit operational approximation.

---

## Feature Scaling

SVM optimization is sensitive to feature scale because the geometry of the hyperplane is expressed in the feature space.

For example, if one feature ranges from `0` to `1` while another ranges from `0` to `100000`, the second feature can dominate the numerical geometry even when it is not more informative.

All implementations therefore demonstrate feature standardization:

`z = (x - μ) / σ`

The mean and standard deviation are calculated from the training data. New observations are transformed using those same training statistics.

In a production training pipeline, the scaler must be persisted together with the model. Recomputing normalization statistics independently for incoming observations would change the coordinate system and invalidate the learned parameters.

---

## Python Implementation

The Python script provides the most direct end-to-end numerical implementation.

`Sample` is an immutable data object representing a feature vector and a binary SVM label. Validation rejects empty feature vectors, invalid labels, and inconsistent dimensions.

`LinearSVM` contains:

- The regularization parameter `C`.
- The learning rate.
- The number of training epochs.
- The weight vector.
- The bias.
- The decision function.
- Functional-margin calculation.
- Hinge-loss calculation.
- Objective evaluation.
- Support-vector candidate identification.
- Prediction.

Training uses sequential subgradient descent. When the functional margin is at least `1`, the hinge-loss gradient is zero and only the regularization gradient affects the weights. When the observation lies inside the margin, the hinge-loss gradient contributes a label-dependent term.

The script also standardizes the dataset and evaluates several `C` values. This makes the effect of soft-margin regularization visible rather than treating the hyperparameter as an unexplained constant.

The prediction section transforms new raw observations with the original training statistics before evaluating them.

---

## JavaScript Implementation

The JavaScript implementation takes an event-driven perspective.

`LinearSVM` extends Node.js `EventEmitter`. During training it emits epoch information and a final model event. This separates numerical training from consumers that may want progress information, metrics, logging, or monitoring.

The implementation includes:

- `TrainingExample` for validated immutable observations.
- `standardize()` for feature transformation.
- `LinearSVM` for the numerical classifier.
- `RepositoryMergeStream` for asynchronous candidate evaluation.
- Event listeners for training and classification events.

The asynchronous candidate evaluation uses `Promise` and `setTimeout` to model observations arriving independently rather than being processed only as a static batch.

The classifier emits a risk-region recommendation, but the implementation explicitly avoids treating that recommendation as a repository authorization mechanism. This is important in systems where statistical models and deterministic controls coexist.

No external npm package is required.

---

## C++ Implementation

The C++17 program separates numerical modeling from repository-governance evaluation.

`StandardScaler` owns the feature-normalization state.

`LinearSoftMarginSVM` implements:

- Parameter validation.
- Weight initialization.
- Subgradient optimization.
- Decision scores.
- Predictions.
- Functional margins.
- Hinge loss.
- Objective evaluation.
- Margin width.
- Support-vector identification.

The governance layer uses separate structures:

- `PullRequest`
- `Review`
- `StatusCheck`
- `BranchProtection`
- `GovernanceResult`

`MergeEligibilityEngine` checks deterministic repository conditions independently from the SVM.

This creates an explicit architectural boundary:

`Pull Request data → SVM risk classification`

and separately:

`Pull Request + reviews + status checks + branch policy → merge eligibility`

The second path remains authoritative.

The example also demonstrates a policy failure in which only one valid approval is available while two are required. Even if the SVM classification is positive, the deterministic policy blocks the merge.

This distinction prevents a statistical model from silently becoming an authorization mechanism.

---

## Java Implementation

The Java 17 implementation models repository governance as a domain-oriented service.

The domain types include:

| Type | Responsibility |
|---|---|
| `Label` | Represents the two SVM classes |
| `PullRequestState` | Represents lifecycle state |
| `ReviewState` | Represents review decisions |
| `CheckState` | Represents status-check state |
| `PullRequest` | Immutable change proposal |
| `Review` | Reviewer decision and stale state |
| `StatusCheck` | Result of an automated check |
| `BranchProtectionPolicy` | Deterministic merge requirements |
| `MergeDecision` | Eligible state and blocking reasons |
| `StandardScaler` | Training-coordinate transformation |
| `LinearSoftMarginSvm` | Numerical classifier |
| `MergeEligibilityService` | Governance decision service |

Java records are used for immutable domain values. The `Review` record exposes `isCurrentApproval()`, making stale-approval behavior explicit rather than leaving the distinction to callers.

`BranchProtectionPolicy` contains independent controls for:

- Required approvals.
- Required status checks.
- Resolved conversations.
- Direct-push restrictions.
- Force-push restrictions.
- Branch deletion.
- Linear history.
- Administrator bypass policy.

The example evaluates a healthy state, a stale-approval state, and a failed-status-check state.

These scenarios demonstrate why a repository governance model should not collapse review decisions, statistical classification, and branch protection into one boolean value.

---

## PostgreSQL Implementation

The SQL script represents the same technical domain relationally.

The schema contains separate entities for:

- Repositories.
- Branches.
- Contributors.
- Pull Requests.
- Commits.
- Review requests.
- Reviews.
- Review comments.
- Status checks.
- Branch-protection policies.
- Required status checks.
- SVM models.
- SVM observations.

The relationships preserve the distinction between a change proposal and the controls applied to it.

A Pull Request references separate source and target branches. Commits belong to Pull Requests. Reviews belong to Pull Requests and identify reviewers. Review comments belong to individual reviews.

Branch protection is represented independently through `branch_protection_policies` and `required_status_checks`.

The database enforces several rules through PostgreSQL constraints and triggers:

- Labels must be `-1` or `+1`.
- Review states are restricted to valid values.
- Status checks have controlled states.
- Approval requirements cannot be negative.
- Reviewers cannot submit reviews on their own Pull Requests.
- SVM feature vectors must have the same dimensionality as the stored model weight vector.

The `current_approval_counts` view distinguishes current approvals from stale historical approvals.

The `unresolved_review_conversations` view identifies outstanding review comments.

The `svm_support_vector_report` view exposes stored observations identified as support-vector candidates.

The `merge_eligibility_report` view evaluates deterministic branch-policy conditions using:

- Pull Request state.
- Source synchronization.
- Merge conflicts.
- Conversation state.
- Current approvals.
- Required status checks.

The database does not attempt to perform general-purpose SVM training. Instead, it stores model parameters and model observations so that relational governance data can be evaluated alongside the learned model state.

---

## Relationship Between the Implementations

The five source files intentionally use different perspectives.

| Implementation | Primary perspective |
|---|---|
| Python | Numerical SVM training and analysis |
| JavaScript | Event-driven model evaluation |
| C++17 | Governance and merge-eligibility engine |
| Java 17 | Enterprise domain model and policy service |
| PostgreSQL | Persistent relational governance and model state |

The mathematical core remains recognizable across implementations, but the surrounding architecture changes.

This prevents the implementations from becoming five translations of the same program.

---

## Hyperplane and Margin Interpretation

For a two-dimensional model, the hyperplane becomes a line:

`w₁x₁ + w₂x₂ + b = 0`

The two canonical margin boundaries are:

`w₁x₁ + w₂x₂ + b = +1`

`w₁x₁ + w₂x₂ + b = -1`

Conceptually:

`negative region | margin | hyperplane | margin | positive region`

The actual orientation depends on the learned weight vector.

The normal vector `w` points perpendicular to the hyperplane. Increasing `||w||` reduces the geometric distance between the canonical margin boundaries, while decreasing `||w||` widens that distance.

This geometric interpretation is why SVM regularization is directly connected to margin maximization.

---

## Soft-Margin Failure Cases

The examples deliberately include observations that are not perfectly separated.

A soft-margin SVM handles these cases through hinge loss rather than requiring every training observation to satisfy:

`y(w · x + b) >= 1`

An observation violating this condition produces positive loss:

`max(0, 1 - y(w · x + b))`

This provides a controlled way to handle noisy or overlapping data.

A large `C` makes violations expensive. A small `C` places relatively greater emphasis on the regularization term.

Neither extreme is universally correct.

Very large regularization pressure can make a model more sensitive to difficult training observations. Very small `C` can tolerate substantial training violations and may underfit a dataset where stronger separation is appropriate.

---

## Review of the Repository-Governance Analogy

The repository scenario is used to provide concrete operational meaning for the classification problem, but the mechanisms remain distinct.

A statistical model may classify a change according to observed risk features.

Repository governance determines whether a change is actually permitted to merge.

Those are different decisions.

The implemented architecture therefore follows:

`Observed change metrics`
   
`↓`

`Feature scaling`

`↓`

`Linear SVM`

`↓`

`Decision score / risk classification`

while independently:

`Pull Request state`

`+ current reviews`

`+ status checks`

`+ branch protection policy`

`↓`

`Merge eligibility`

The second path can block a change regardless of the SVM result.

---

## Edge Cases and Failure Conditions

The implementations explicitly handle several failure conditions:

- Empty training datasets.
- Empty feature vectors.
- Invalid binary labels.
- Inconsistent feature dimensions.
- Non-finite numerical values.
- Invalid regularization parameters.
- Invalid learning rates.
- Invalid epoch counts.
- Prediction vectors with the wrong dimensionality.
- Zero-norm hyperplanes when calculating geometric margin width.
- Stale approvals.
- Missing required status checks.
- Failed required status checks.
- Pending required status checks.
- Unresolved conversations.
- Unsynchronized source branches.
- Merge conflicts.
- Non-open Pull Requests.
- Attempts to treat model output as a policy bypass.

A production SVM system would also need explicit handling for class imbalance, data drift, model versioning, training/validation separation, reproducible preprocessing, numerical monitoring, and model rollback.

---

## Performance Considerations

The educational SVM implementations use sequential subgradient descent.

For `n` observations, `d` features, and `e` epochs, the basic training work is approximately:

`O(e × n × d)`

Memory for the dense feature matrix is approximately:

`O(n × d)`

The implementations are intentionally transparent rather than optimized for very large datasets.

The PostgreSQL indexes target operational access patterns such as Pull Requests by target branch and state, current approvals, status checks, and support-vector reports.

For large-scale machine-learning workloads, specialized numerical optimization and sparse representations can be materially more efficient, especially when the feature dimension is high.

---

## Security and Governance Considerations

SVM output should not be treated as an authorization primitive.

A model can be wrong because:

- Training data may contain errors.
- The feature distribution may change.
- New types of repository changes may not resemble training observations.
- A boundary observation may be ambiguous.
- A model score is not automatically a calibrated probability.
- An attacker may deliberately manipulate input features.

For that reason, deterministic controls such as required approvals, status checks, conflict detection, and protected branches remain independently enforced.

Model metadata should also be versioned. A production system should know which feature transformation, weight vector, bias, regularization configuration, and training dataset produced a particular prediction.

The SQL model stores the weight vector and feature metadata so that predictions and governance decisions can be associated with a specific model record.

---

## Debugging Strategy

When a prediction appears incorrect, the most useful diagnostic sequence is:

`raw features → scaled features → decision score → functional margin → hinge loss → classification`

The functional margin is particularly useful because it distinguishes a confidently classified observation from one close to or inside the margin.

For a labeled observation:

- `margin > 1` means it is outside the canonical margin.
- `0 < margin < 1` means it is correctly classified but violates the canonical margin.
- `margin = 0` means it lies on the decision hyperplane.
- `margin < 0` means it is misclassified.

Support-vector candidates should be inspected when the learned boundary changes unexpectedly because these observations have disproportionate influence on the soft-margin solution.

The preprocessing statistics must also be checked. Applying a model to unscaled data when it was trained on standardized data can produce apparently inexplicable predictions even when the learned parameters themselves are correct.

---

## Practical Distinctions

The core concepts should not be conflated:

| Concept | Primary question |
|---|---|
| Hyperplane | Where is the decision boundary? |
| Margin | How far is an observation from the boundary in geometric terms? |
| Support vector | Which observations strongly constrain the boundary? |
| Hinge loss | How severely does an observation violate the desired margin? |
| Soft margin | How can the model tolerate violations while controlling complexity? |
| `C` | How strongly should margin violations be penalized relative to regularization? |
| Decision score | On which side of the learned hyperplane is an observation, and how strongly? |

These mechanisms work together but answer different questions.

The implementations preserve those distinctions in their data structures, calculations, database objects, and governance workflows.
