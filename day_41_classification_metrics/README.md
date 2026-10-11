# Classification Metrics: Accuracy, Precision, Recall, F1, Specificity, and Confusion Matrix

## Topic scope

Classification metrics convert prediction outcomes into measurable evidence about model behavior. This implementation set focuses on six closely related concepts:

- **Confusion matrix** records the four fundamental outcomes of binary classification: true positives, false positives, true negatives, and false negatives.
- **Accuracy** measures the proportion of all predictions that are correct.
- **Precision** measures how reliable positive predictions are.
- **Recall**, also called sensitivity or true-positive rate, measures how many actual positive cases are detected.
- **Specificity**, also called true-negative rate, measures how many actual negative cases are correctly rejected.
- **F1 score** is the harmonic mean of precision and recall and gives a single measure when both types of positive-class error matter.

The implementations deliberately distinguish these measures instead of treating them as interchangeable.

## Confusion matrix

For a binary classifier, the matrix is interpreted as follows:

| Actual | Predicted negative | Predicted positive |
|---|---:|---:|
| Negative | True Negative | False Positive |
| Positive | False Negative | True Positive |

The four counts are the foundation of the other metrics.

A **true positive** is a positive case correctly identified as positive. A **false positive** is a negative case incorrectly classified as positive. A **true negative** is a negative case correctly rejected. A **false negative** is a positive case missed by the classifier.

The distinction between false positives and false negatives is operationally important. A security-review classifier, for example, may consider a false negative more costly because a risky change could avoid specialist inspection.

## Metric formulas

Let TP, FP, TN, and FN denote the four confusion-matrix counts.

### Accuracy

`Accuracy = (TP + TN) / (TP + FP + TN + FN)`

Accuracy treats every observation equally. It is informative when class distributions and error costs are reasonably balanced.

Accuracy can be misleading when one class dominates. If 95% of observations are negative, a classifier that predicts negative almost everywhere can obtain approximately 95% accuracy while detecting very few positive cases.

### Precision

`Precision = TP / (TP + FP)`

Precision answers:

> Of the cases predicted as positive, how many were actually positive?

High precision means positive alerts tend to be trustworthy. A low precision classifier generates many false alarms.

### Recall

`Recall = TP / (TP + FN)`

Recall answers:

> Of all actual positive cases, how many did the classifier detect?

Recall becomes particularly important when missing a positive case has a high cost. Increasing recall generally requires accepting more positive predictions, which can increase false positives.

### Specificity

`Specificity = TN / (TN + FP)`

Specificity answers:

> Of all actual negative cases, how many did the classifier correctly identify as negative?

Specificity is useful when false positives are operationally expensive. It complements recall because recall concentrates on the positive class while specificity concentrates on the negative class.

### F1 score

`F1 = 2 × Precision × Recall / (Precision + Recall)`

F1 gives equal harmonic weighting to precision and recall. The harmonic mean is intentionally sensitive to imbalance between the two values. A classifier with precision 0.95 and recall 0.20 does not receive a high F1 simply because one of the two measures is excellent.

The implementations return zero when a denominator is zero. This makes the executable examples deterministic and prevents undefined divisions from terminating the demonstrations.

## Relationship between the measures

The metrics answer different operational questions:

| Metric | Main question | Error emphasis |
|---|---|---|
| Accuracy | How often is the complete prediction correct? | Both classes |
| Precision | Can positive predictions be trusted? | False positives |
| Recall | How many positives were detected? | False negatives |
| Specificity | How reliably are negatives rejected? | False positives |
| F1 | How well are precision and recall balanced? | FP and FN |

A model can therefore have high accuracy and poor recall, high recall and poor precision, or high specificity and poor sensitivity. Reporting one number without its context can hide important model behavior.

## Python implementation

The Python program builds a `BinaryConfusionMatrix` data class containing TP, FP, TN, and FN. Metric functions operate directly on that structure, keeping the mathematical source of each measure visible.

The binary evaluator validates equal-length label arrays and restricts labels to zero and one. The `safe_divide` function prevents undefined denominators from causing runtime failures.

The threshold analysis converts positive-class probability scores into labels. A threshold of `0.50` predicts positive when the score is at least 0.50. Lowering the threshold makes positive classification easier and commonly increases recall while reducing specificity. Raising it can have the opposite effect.

The multiclass implementation creates a square confusion matrix where rows represent actual classes and columns represent predicted classes. It then converts each class into a one-vs-rest binary problem. This permits precision, recall, specificity, and F1 to be calculated separately for each class.

Macro F1 gives every class equal weight. Weighted F1 weights each class by its support. These values can differ substantially when class frequencies are unequal.

The security-review example gives the metrics an operational interpretation. A positive prediction means that a Pull Request should receive specialist security review. Threshold selection therefore affects both the number of risky changes detected and the number of ordinary changes unnecessarily routed to security reviewers.

The Python script also contains executable validation checks. These cover mismatched array lengths, invalid binary labels, invalid probability ranges, and expected metric values.

## JavaScript implementation

The JavaScript file uses objects and classes to represent confusion matrices and evaluation behavior. `ValidationError` provides a domain-specific error type instead of allowing invalid classification data to pass silently.

The `EvaluationEngine` introduces an event-driven perspective. Evaluation produces an `evaluationComplete` event after the thresholded predictions and metrics have been calculated. This reflects how a JavaScript application can process evaluation results asynchronously or notify other components without embedding every action inside the metric calculation itself.

The multiclass implementation uses `Map` for class-to-index resolution and produces one-vs-rest counts for each class. The resulting per-class records contain precision, recall, specificity, F1, and support.

The JavaScript threshold workflow focuses on the distinction between model scores and classification decisions. The score itself does not determine whether a case is positive until a threshold is applied.

The implementation also treats invalid probability values and invalid thresholds as validation failures. This matters because silently accepting values outside the probability range can produce misleading evaluation results.

## C++ repository-governance case study

The C++ program models a repository governance scenario in which a classifier estimates whether a Pull Request requires specialist security review.

The `RepositoryGovernanceEngine` keeps classification policy separate from merge policy. The classification threshold determines whether the model labels a change as requiring review. The merge-eligibility method evaluates repository conditions such as:

- required status checks
- required approvals
- unresolved review conversations
- protected-branch status
- direct-push attempts

This separation is important because classification metrics measure prediction quality, while repository governance determines whether a software change is permitted to enter a protected branch.

The C++ implementation uses a `MulticlassMatrix` class for class-specific confusion analysis. Its `oneVsRest` operation converts one class into a binary problem and treats all other classes as the negative class.

The case study therefore supports both binary and multiclass analysis without pretending that multiclass metrics have exactly the same interpretation as a binary classifier.

The program uses C++17 standard-library structures such as `std::vector`, `std::string`, `std::find`, and exception types. No external library is required.

## Java enterprise model

The Java implementation represents repository governance through explicit domain types rather than only primitive flags.

`ReviewClass` describes the predicted classification of a Pull Request. `PullRequestState` represents lifecycle state. `ReviewDecision` captures reviewer identity, approval, dismissal, and unresolved discussion state.

The `ReviewPolicy` interface separates policy definition from the merge-eligibility service. `ProtectedBranchPolicy` implements concrete rules for required approvals and security-review requirements.

The `MergeEligibilityService` combines classification-related review requirements with repository governance conditions. A draft, closed, or already merged Pull Request cannot become merge eligible. Direct pushes to a protected branch are rejected by the policy evaluation. Required approvals, status checks, and unresolved discussions are also evaluated.

The model explicitly rejects an invalid review state in which a review is simultaneously marked as approved and dismissed. Representing this as a domain validation rule prevents contradictory state from becoming part of the application model.

The Java metric engine uses standard collections and records available in modern Java. The implementation does not require a framework or external dependency.

## SQL relational model

The PostgreSQL script models the domain as related entities rather than storing all evaluation information in one table.

`repositories` identifies repositories. `branches` identifies source and target branches and records whether a branch is protected. `pull_requests` represents proposed changes between branches. `commits` records changes associated with Pull Requests.

`developers` and `reviewers` distinguish general repository identities from users with reviewer capabilities. `reviews` records review decisions, while `review_comments` stores line-specific discussions and whether those discussions are resolved.

`status_checks` records automated validation results. `branch_protection_policies` stores repository-level governance rules such as required approvals, status-check requirements, direct-push restrictions, force-push restrictions, deletion restrictions, and linear-history requirements.

`approval_requirements` provides a separate representation of approval policies, including the possibility of requiring a security review.

`classification_results` stores actual labels, predicted labels, positive probabilities, and the threshold used to turn probabilities into classification decisions. Keeping both probability and threshold makes it possible to analyze how different thresholds would change the resulting predictions.

## SQL constraints and integrity

The database uses primary keys and foreign keys to preserve entity relationships. Unique constraints prevent duplicate repository names, branch names within a repository, reviewer assignments, commit identifiers, and status-check names for the same Pull Request.

Check constraints enforce domain rules such as non-negative approval requirements, positive review-comment line numbers, and probability and threshold values within the `[0, 1]` interval.

The Pull Request state constraint requires a merged timestamp whenever the state is `merged`. This prevents one class of contradictory lifecycle state from entering the database.

Indexes target access patterns used by governance queries. The Pull Request state index supports searches by protected target branch and lifecycle state. Review and status-check indexes support merge-eligibility evaluation. The partial comment index concentrates on unresolved review discussions, which are especially relevant when determining whether a Pull Request can merge.

## SQL metric calculations

The SQL script calculates the confusion matrix with PostgreSQL aggregate `FILTER` expressions. This allows TP, FP, TN, and FN to be derived directly from stored classifications.

The metric query then calculates accuracy, precision, recall, specificity, and F1 from those counts. `NULLIF` prevents division-by-zero errors.

Threshold analysis is performed using a common table expression containing multiple thresholds. The same probability observations are reclassified at each threshold, making the trade-off between precision, recall, specificity, and F1 visible without changing the stored model scores.

This distinction is important: a probability score is a model output, while a classification label is a decision derived from that output and a threshold.

## Multiclass interpretation

A multiclass confusion matrix contains one row and one column for each class. The diagonal contains correct predictions. Off-diagonal cells identify specific class confusions.

For example, if `security` is confused with `feature`, the corresponding matrix cell shows a different failure mode from confusing `security` with `documentation`.

Per-class precision asks whether predictions for that class are reliable. Per-class recall asks whether actual observations of that class are detected. Per-class specificity treats that class as positive and all other classes as negative.

Macro averaging prevents a high-support class from dominating the reported F1. Weighted averaging reflects class frequency and can be more representative of aggregate workload when class distribution matters.

## Threshold effects

Classification metrics are affected by the decision threshold when the model produces continuous scores.

A lower threshold usually produces more positive predictions. This can recover additional true positives and increase recall, but it may also create additional false positives and reduce precision or specificity.

A higher threshold usually produces fewer positive predictions. This can improve precision or specificity when the classifier is reasonably calibrated, but it can increase false negatives and reduce recall.

The appropriate threshold is therefore a decision-policy problem rather than a universal mathematical constant.

## Class imbalance

Class imbalance is a major reason to inspect multiple metrics.

Suppose a monitoring system contains 9,500 ordinary events and only 500 critical events. A classifier that predicts almost every event as ordinary can obtain high accuracy while failing to detect a substantial fraction of critical events.

Recall exposes this failure because it directly measures the fraction of actual positive cases detected.

Precision provides a complementary operational view. A system can obtain high recall by marking many cases positive, but if most of those alerts are incorrect, the investigation workload can become excessive.

Specificity describes how well the classifier avoids incorrectly labeling ordinary cases as positive.

F1 is useful when precision and recall both matter, but it does not replace an analysis of the confusion matrix.

## Pull Request, code review, approvals, and branch protection relationship

These four concepts interact but represent different mechanisms.

A **Pull Request** is the change-proposal workflow. It identifies a source branch, a target branch, a changeset, commits, review activity, checks, and eventual merge or closure. A Pull Request can be opened as a draft, synchronized as new commits are pushed, encounter merge conflicts, be reopened after closure, or be merged after its required conditions are satisfied.

**Code Review** evaluates the proposed changes. Reviewers inspect the changeset, create inline comments, identify defects or risks, request changes, approve appropriate changes, and resolve discussions. Review quality depends on scope, evidence, correctness, and whether important findings have been addressed.

An **Approval** is a specific review decision. It is not synonymous with a complete code review process. A review can contain comments without approval, request changes, be dismissed, or become stale under repository policy. Approval requirements can also specify how many eligible reviewers must approve.

**Branch Protection** is repository governance applied to a protected target branch. It can require reviews, require status checks, prevent direct pushes, prevent force pushes, restrict branch deletion, require conversation resolution, require linear history, and restrict who may bypass particular controls where the repository platform permits such behavior.

The relationship can be expressed as:

`Pull Request → Code Review → Approval decisions → Branch Protection enforcement → Merge eligibility`

The arrows describe dependency rather than equivalence. A Pull Request provides the workflow. Code Review evaluates the proposed change. Approvals represent review decisions. Branch Protection defines conditions that must be satisfied before the protected branch accepts the change.

## Review quality and metric quality

Classification metrics should not be interpreted independently of the data and decision process that produced them.

A high F1 score does not prove that a classifier is appropriate for production. The evaluation set may be unrepresentative, the positive class may be defined incorrectly, or the operational cost of FP and FN errors may be highly asymmetric.

Similarly, a high approval count does not prove that a Pull Request received high-quality review. Approval eligibility, reviewer independence, review scope, unresolved discussions, and the freshness of approvals all affect governance quality.

The technical lesson is that measurements should correspond to the decision being made. Classification metrics measure prediction behavior. Review states measure human evaluation activity. Branch protection expresses enforceable repository policy.

## Common failure modes

### Treating accuracy as sufficient

Accuracy can hide minority-class failures. Always inspect the confusion matrix and class-specific measures when the class distribution is uneven.

### Confusing precision with recall

Precision starts from predicted positives. Recall starts from actual positives. A system that generates few positive predictions may have high precision but poor recall.

### Ignoring specificity

Recall does not describe negative-class performance. Specificity is necessary when the cost of false positives matters.

### Selecting a threshold without an operational objective

A threshold should reflect the relative consequences of FP and FN errors, review capacity, and the desired balance of detection and workload.

### Reporting only a single F1 score for multiclass data

Macro and weighted averages answer different questions. Per-class values reveal which classes are responsible for weak performance.

### Evaluating on contaminated data

If observations used to select a threshold are also used to report final performance, the metric can become optimistically biased. Evaluation data should represent the intended deployment population.

### Treating probability as classification

A probability such as `0.73` is not inherently a positive label. The threshold determines the classification decision.

## Performance considerations

For `n` binary observations, constructing the confusion matrix is `O(n)` time and `O(1)` additional counting space.

A multiclass confusion matrix with `k` classes requires a `k × k` matrix. Processing `n` observations remains `O(n)` when class lookup is constant-time, while storing the matrix requires `O(k²)` space.

The SQL implementation benefits from indexes on columns used for Pull Request, review, comment, status-check, and classification queries. Aggregate metric calculations still require processing the relevant classification rows, but selective predicates can reduce the dataset before aggregation.

The JavaScript, Java, and C++ implementations use in-memory structures. For large evaluation sets, production systems would normally stream or batch observations rather than retaining unnecessary raw records.

## Security and governance considerations

Classification results can influence access to specialist review and therefore should not silently become authoritative security decisions.

A false negative in a security-review classifier can allow a sensitive change to bypass specialist inspection. A false positive increases review workload but may be preferable when the cost of missing a critical change is substantially higher.

Repository governance should therefore remain independently enforceable. A classifier can recommend review, but branch protection should enforce required approvals, successful status checks, and other merge conditions rather than relying solely on a model prediction.

Approval records should preserve reviewer identity and state transitions. Dismissed or stale approvals should not automatically be treated as active approvals. Unresolved review discussions should remain visible to merge-eligibility logic when the repository policy requires their resolution.

## Production interpretation

A robust classification evaluation should retain the confusion matrix alongside aggregate metrics. This makes it possible to reconstruct why a metric changed.

Thresholds should be versioned with the evaluation configuration. If a model keeps the same scores but the threshold changes, its predicted labels and resulting metrics can change without the underlying scoring model changing.

Metrics should also be segmented when the application contains materially different classes, environments, workloads, or decision contexts. A single aggregate score can conceal poor behavior in an important subgroup.

The implementations in this set intentionally keep the classification layer distinct from repository governance. The classification metrics quantify prediction behavior, while review, approval, and branch-protection mechanisms determine how a proposed software change is evaluated and whether it is permitted to merge.
