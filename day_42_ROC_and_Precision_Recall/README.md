# ROC and Precision-Recall Analysis

## Scope

This implementation treats ROC analysis, precision-recall analysis, and threshold optimization as related but distinct parts of binary classification evaluation.

The central distinction is:

- **ROC analysis** describes discrimination across operating thresholds by relating true-positive rate to false-positive rate.
- **Precision-recall analysis** describes the trade-off between retrieving positive cases and keeping retrieved cases genuinely positive.
- **Threshold optimization** chooses one operating point for a business or technical objective. AUC does not choose that threshold.
- **Positive-class prevalence** is especially important for interpreting precision-recall behavior.
- **Repository governance** is used as the application setting for the examples, but statistical predictions do not replace deterministic Pull Request, review, approval, or branch-policy rules.

The five executable implementations approach the topic from different technical perspectives. Python emphasizes an end-to-end analytical workflow, JavaScript models an event-driven threshold service, C++ builds a repository governance case study with an independent review-risk classifier, Java models enterprise domain behavior, and PostgreSQL represents the evaluation and governance data relationally.

## Core classification quantities

For a binary classifier, every threshold produces a confusion matrix containing true positives, false positives, true negatives, and false negatives.

A **true positive** occurs when a genuinely positive case receives a positive prediction. A **false positive** occurs when a negative case is incorrectly classified as positive. A **true negative** is a correctly rejected negative case. A **false negative** is a positive case that the classifier fails to identify.

These four quantities change as the decision threshold changes.

### True-positive rate

True-positive rate is also called recall or sensitivity:

`TPR = TP / (TP + FN)`

It measures the fraction of actual positive cases that are retrieved.

### False-positive rate

False-positive rate is:

`FPR = FP / (FP + TN)`

It measures the fraction of actual negative cases that are incorrectly retrieved.

Specificity is the complementary quantity:

`Specificity = TN / (TN + FP)`

Therefore:

`FPR = 1 - Specificity`

### Precision

Precision is:

`Precision = TP / (TP + FP)`

It answers a different question from recall:

> Among the cases classified as positive, how many are actually positive?

This distinction becomes important when positive predictions trigger an expensive review process.

### F1

F1 is the harmonic mean of precision and recall:

`F1 = 2 * Precision * Recall / (Precision + Recall)`

F1 is useful when precision and recall should both matter, but it does not represent every operational objective. A system with very different costs for false positives and false negatives should not automatically use maximum F1 as its deployment rule.

## ROC curve

A ROC curve plots:

`x = false-positive rate`

against:

`y = true-positive rate`

Each point corresponds to a threshold.

A high threshold normally predicts relatively few observations as positive. As the threshold is lowered, more observations become positive. True positives and false positives can both increase.

The Python implementation computes threshold-induced confusion matrices and converts them into ROC operating points.

The JavaScript implementation exposes the same concept through `rocCurve(dataset)`, but the surrounding implementation is event-driven. This separates the metric calculation from the threshold-selection event that a production application might consume.

The C++ implementation uses the same mathematical structure inside a different domain model. The statistical classifier predicts whether a Pull Request requires additional review attention. The resulting threshold is not allowed to directly authorize a merge.

The Java implementation places metric calculations inside `MetricService`, separating analytical calculations from the Pull Request and branch-policy domain.

The SQL implementation derives threshold-specific confusion matrices directly from stored model scores.

## ROC AUC

AUC is the area under the ROC curve.

The implementations use trapezoidal integration over ROC points:

`AUC = sum((FPR_next - FPR_current) * (TPR_next + TPR_current) / 2)`

A ROC AUC near 1 indicates strong ranking discrimination, while a value near 0.5 is consistent with approximately random ranking in a balanced interpretation.

AUC is a threshold-independent summary of discrimination. It does not tell the production system which threshold to use.

This is a critical distinction:

`AUC measurement != threshold selection`

A classifier can have an excellent AUC while still requiring a carefully chosen operating threshold because the application may have strict false-positive limits, limited human-review capacity, asymmetric error costs, or regulatory requirements.

AUC also does not establish calibration. A score of `0.80` does not become a trustworthy 80 percent probability merely because the classifier has a strong ROC AUC.

## Ranking interpretation

ROC AUC is closely related to ranking quality.

If positive examples tend to receive higher scores than negative examples, the ROC curve tends to move toward the upper-left region.

The Python implementation demonstrates this with a monotonic transformation of scores. Applying a strictly increasing transformation such as the square root preserves ordering and therefore preserves the ranking information used by AUC, even though the numerical score values change.

This is why AUC should not be interpreted as a probability-calibration metric.

A ranking model can distinguish cases well while producing scores that are poorly calibrated.

## Precision-recall curve

The precision-recall curve uses:

`x = recall`

and:

`y = precision`

It emphasizes the behavior of positive predictions.

This is especially important for imbalanced classification.

Suppose only 2 percent of observations are genuinely positive. A classifier that predicts almost everything as negative may have a low false-positive rate, but that does not necessarily make it useful for identifying the rare positive class.

Precision makes the number of false positives visible among predicted positives.

The PR curve therefore answers a question that ROC analysis does not emphasize:

> When the system retrieves positive cases, how reliable are those retrieved cases?

## Positive-class prevalence

For a random classifier, the baseline precision is closely related to positive prevalence.

If 5 percent of the population is positive, random positive retrieval has an expected precision around 5 percent.

The Python implementation explicitly calculates this prevalence.

The SQL implementation also exposes positive prevalence with a database query so that PR results can be interpreted against the actual evaluation population.

This matters because precision is not independent of class prevalence.

A model evaluated on a population with one prevalence can produce a different precision profile when deployed on a population with another prevalence, even if the underlying ranking behavior remains similar.

## Average Precision and PR AUC

The implementations distinguish two commonly encountered summaries.

**Average Precision** is calculated as a step-wise accumulation of precision weighted by changes in recall. It is closely tied to ranking order.

**Trapezoidal PR AUC** applies numerical integration to PR points.

These quantities should not be treated as interchangeable without checking the evaluation convention used by a specific library or reporting standard.

The Python implementation exposes both so their computational distinction is visible.

The JavaScript, C++, and Java implementations emphasize Average Precision as the ranking-oriented summary rather than presenting every possible PR integration convention.

## Threshold optimization

A classifier generally produces a score. The application still needs a rule such as:

`predict positive when score >= threshold`

The threshold determines the operating point.

The Python implementation searches every threshold induced by an observed score. This is exact for the finite validation set because predictions can only change when the threshold crosses an observed score.

The same idea appears in JavaScript, C++, Java, and SQL, but each implementation uses it differently.

Threshold optimization can target different objectives.

### Maximum F1

Maximum F1 selects the threshold that balances precision and recall according to the harmonic mean.

This can be appropriate when the two metrics should have similar importance.

It should not be treated as universally optimal.

### F-beta optimization

The Python implementation supports F-beta.

When `beta > 1`, recall receives greater emphasis.

When `beta < 1`, precision receives greater emphasis.

This makes the objective more expressive than F1 when the application explicitly values one side of the trade-off.

### Constrained optimization

A production system may require:

`FPR <= 0.10`

while maximizing recall.

This is a fundamentally different objective from maximizing F1.

The Python, JavaScript, and SQL implementations demonstrate this type of constraint.

A threshold is accepted only when it satisfies the false-positive ceiling. Among acceptable thresholds, recall is maximized.

### Cost-sensitive optimization

Suppose a false negative costs five times as much as a false positive.

Then a cost function can be expressed as:

`Cost = 1 * FP + 5 * FN`

The threshold that minimizes this cost can be substantially different from the threshold that maximizes F1.

The Python, JavaScript, C++, and SQL implementations demonstrate cost-sensitive selection.

This is usually closer to operational decision-making when classification errors have explicit consequences.

## Why AUC should not select the threshold

AUC summarizes performance across thresholds.

Deployment normally needs one threshold.

Therefore the analytical workflow should be conceptually separated:

`score model -> evaluate ranking -> inspect ROC/PR -> define operational objective -> select threshold -> validate threshold -> deploy`

Choosing a threshold solely because it produces the highest AUC is conceptually incorrect because AUC does not belong to a single threshold.

The threshold should instead be connected to the actual decision problem.

## Python implementation

The Python program is the most complete analytical implementation.

`Observation` stores a binary label and a score. `ConfusionMatrix` calculates precision, recall, specificity, FPR, F1, and related quantities from a thresholded prediction.

`candidate_thresholds()` creates the finite set of score-induced operating points. This avoids repeatedly testing arbitrary thresholds that produce exactly the same prediction partition.

`roc_curve()` calculates ROC operating points, while `roc_auc()` integrates them.

`precision_recall_curve()` constructs PR operating points.

`average_precision()` implements step-wise ranking-oriented average precision rather than assuming that trapezoidal PR integration is the same metric.

`optimize_threshold()` provides a reusable threshold-search mechanism. It accepts an objective function so that F1, F-beta, constrained recall, and cost-sensitive decisions can use the same threshold engine without duplicating search logic.

`CostModel` separates error economics from metric calculation. This is useful because the metric layer should not hard-code a single business cost assumption.

`bootstrap_auc()` provides an empirical uncertainty estimate using bootstrap resampling. Replicates containing only one class are skipped because ROC AUC requires both classes.

The script also validates score ranges, binary labels, empty datasets, and single-class evaluation data.

## JavaScript implementation

The JavaScript implementation focuses on an event-driven service perspective.

`BinaryDataset` owns validation and threshold discovery.

`confusionMatrix()` and `metrics()` provide the low-level operating-point calculations.

`ThresholdMonitor` extends Node.js `EventEmitter`. When a policy selects a threshold, the service emits a `threshold-selected` event.

This is useful for modeling an application in which threshold selection is part of a larger pipeline. A real service could consume such an event to record the chosen policy version, publish a configuration change, or initiate a validation workflow.

The asynchronous `scoreModelAsynchronously()` function demonstrates the separation between prediction availability and deterministic metric calculation. The metrics do not need to become asynchronous merely because the upstream scoring service is asynchronous.

The JavaScript implementation therefore focuses less on reproducing a notebook-style analysis and more on how threshold evaluation can become a component inside an event-driven application.

## C++ case study

The C++ program models a repository governance system.

`MergeGovernanceModel` owns Pull Request state and merge-policy conditions. It tracks:

- source and target branches
- draft state
- review approvals
- requested changes
- status-check results
- synchronization state

A Pull Request cannot become merge eligible simply because a statistical model predicts low risk.

The deterministic governance rules remain separate.

This separation is deliberate.

The program then introduces a review-risk classifier whose score represents the likelihood that a Pull Request requires additional human review.

The same program calculates ROC AUC and Average Precision for that classifier and searches for a cost-sensitive operating point.

The statistical threshold therefore supports review routing rather than replacing repository governance.

This architecture prevents a common design mistake in automated development systems: allowing a probabilistic model to silently override deterministic authorization rules.

The C++ implementation uses `std::vector` for observations, sorting for threshold construction, and explicit structures for confusion-matrix state.

Its AUC implementation uses trapezoidal integration.

The threshold optimizer evaluates false-positive and false-negative costs independently from merge authorization.

## Java enterprise model

The Java program uses explicit domain abstractions.

`PullRequestState` represents the Pull Request lifecycle.

`ReviewDecision` separates approval, requested changes, and ordinary comments.

`ReviewPolicy` contains deterministic merge requirements.

`PullRequest` owns state transitions and prevents invalid operations such as reviewing a closed Pull Request or resolving a nonexistent conversation.

The `synchronizeWithTarget()` method demonstrates an important stale-approval policy. A source branch that incorporates new target-branch changes may invalidate previous approvals because reviewers approved an earlier source state.

The statistical component is isolated inside `MetricService`.

This service validates data, generates candidate thresholds, creates confusion matrices, calculates ROC AUC, calculates Average Precision, and performs objective-driven threshold optimization.

The separation between `ReviewPolicy` and `MetricService` is important: statistical evaluation informs classification decisions, while repository authorization remains a domain-policy concern.

## SQL relational model

The PostgreSQL schema represents the governance domain explicitly.

`repositories` stores repository identity and the default branch.

`branches` identifies source and target branches and whether a branch is protected.

`pull_requests` stores the Pull Request lifecycle and the source and target branches.

`commits` and `pull_request_commits` connect changesets to Pull Requests.

`reviews` stores reviewer decisions tied to a specific commit SHA.

That SHA relationship is significant for approval validity. A review attached to an old source revision should not automatically authorize a new source revision when the policy requires current approvals.

`review_comments` stores inline discussion state and whether a discussion has been resolved.

`status_checks` represents automated validation required by branch protection.

`branch_protection_policies` stores repository governance conditions such as required approvals, status checks, conversation resolution, force-push restrictions, deletion restrictions, and stale-approval behavior.

`approval_requirements` can represent required reviewers or teams.

`model_scores` stores classification scores together with evaluation split and model version.

## SQL threshold analysis

The SQL threshold queries calculate confusion matrices directly from model scores.

The threshold query counts:

- true positives
- false positives
- true negatives
- false negatives

The resulting values are transformed into precision, recall, specificity, FPR, and F1.

The ROC query then converts threshold-specific TPR and FPR values into points and integrates them with the trapezoidal rule.

The PR query exposes precision and recall for each operating threshold.

The F1 query selects a single operating point based on the explicit F1 objective.

The cost-sensitive query selects a threshold using a false-negative cost five times larger than the false-positive cost.

The constrained query selects the threshold with maximum recall while enforcing an FPR ceiling.

This makes the difference between measurement and decision policy explicit at the database layer.

## ROC versus PR interpretation

| Aspect | ROC | Precision-Recall |
|---|---|---|
| Primary axes | FPR and TPR | Recall and precision |
| Main focus | Ranking discrimination across both classes | Quality of positive retrieval |
| Negative-class behavior | Explicit through FPR | Indirect |
| Positive prevalence sensitivity | Less directly reflected in ROC geometry | Strongly reflected |
| Useful for rare positives | Can remain informative but may look optimistic | Often more revealing |
| Threshold selection | Not provided by AUC | Not provided by PR AUC/AP |
| Main warning | High AUC does not imply useful deployment threshold | Precision can change substantially with prevalence |

The two curves answer different questions and should not be treated as competing replacements for one another.

## Threshold selection and validation

A threshold selected on a validation dataset is itself a fitted decision rule.

Using the same data to select the threshold and report final performance can produce optimistic estimates.

A sound evaluation structure separates data roles:

`training data -> model fitting`

`validation data -> threshold and policy selection`

`test data -> final unbiased evaluation`

The SQL schema explicitly records `train`, `validation`, `test`, and `production` as evaluation contexts.

The model version is also stored with each score so that results can be associated with the exact scoring configuration.

## Class imbalance

Class imbalance does not make ROC AUC meaningless, but it can make ROC summaries less representative of the operational problem.

When negatives vastly outnumber positives, even a small FPR can correspond to a large number of false positives.

For example, if 100,000 negative observations exist, an FPR of 1 percent produces 1,000 false positives.

If only 100 positives exist, the same operating point might retrieve 90 positives but create 1,000 false alerts.

Precision makes this consequence visible:

`Precision = 90 / (90 + 1000)`

The PR curve therefore becomes particularly valuable when the positive class is rare and positive predictions consume significant operational capacity.

## Common threshold-selection mistakes

### Treating 0.5 as universally correct

A score threshold of 0.5 has no universal statistical justification.

The appropriate threshold depends on how scores were generated, whether they are calibrated, class prevalence, error costs, operational capacity, and the desired trade-off.

### Selecting the maximum AUC threshold

AUC is not attached to one threshold. It summarizes performance across thresholds.

Threshold selection requires a separate objective.

### Maximizing accuracy on imbalanced data

Accuracy can be dominated by the majority class.

A classifier can obtain high accuracy by rarely predicting the minority class while providing poor positive-case retrieval.

### Ignoring prevalence

Precision depends strongly on the population in which the classifier operates.

A PR result measured on one prevalence should not automatically be expected in another population.

### Optimizing and reporting on the same test set

If the threshold is repeatedly tuned against the test set, the test set is no longer functioning as an unbiased final evaluation.

### Treating probability scores as calibrated probabilities

A ranking score can have strong AUC without being calibrated.

Calibration requires separate analysis.

## Edge cases

A ROC or PR analysis needs meaningful positive and negative examples.

A dataset containing only positive examples cannot provide a standard FPR because there are no negatives.

A dataset containing only negative examples cannot provide meaningful recall because there are no positives.

The implementations therefore reject single-class evaluation datasets.

A threshold that produces no positive predictions has zero predicted-positive count. Precision must be defined carefully rather than producing a division-by-zero error.

A threshold that predicts every observation positive produces recall of 1 when positive observations exist, but its precision is determined by prevalence.

Repeated score values also matter. Multiple observations can share exactly the same score, so threshold candidates should be based on unique observed scores rather than assuming every rank corresponds to a distinct threshold.

## Review-routing interpretation

The repository examples demonstrate an important architectural boundary.

A classifier can predict:

`requires_additional_review = true`

That prediction may determine whether an extra reviewer is requested.

It should not automatically mean:

`merge_allowed = true`

Merge authorization belongs to repository governance.

Pull Request mechanics determine how changes move through the development workflow.

Code Review evaluates the changes.

Approvals represent reviewer decisions.

Branch Protection enforces repository-level merge conditions.

The statistical classifier is an additional decision-support component that may help route attention.

These mechanisms interact, but they are not interchangeable.

## Production considerations

Thresholds should be versioned with the model and policy that produced them.

A threshold selected for one population may become inappropriate after a change in class prevalence, workflow volume, reviewer capacity, or model behavior.

Monitoring should track at least the score distribution, positive prevalence, precision, recall, false-positive rate, false-negative rate, and operational alert volume where labels become available.

Model evaluation should also distinguish between discrimination and calibration.

AUC can remain stable while the score distribution or prevalence changes enough to make a previously selected threshold operationally unsuitable.

Threshold changes should therefore be treated as policy changes rather than harmless configuration edits.

The SQL governance model demonstrates why current review state, current commit SHA, status checks, and unresolved conversations belong in deterministic merge eligibility logic.

## Security and governance implications

Statistical scores should not bypass authorization boundaries.

A model output is data. It is not inherently an authorization decision.

Repository permissions, protected branches, required reviews, and merge controls should remain enforced independently.

If a threshold service is exposed through an API, its inputs should be validated and its model version and threshold policy should be auditable.

Approval records should remain associated with the source revision that was reviewed when stale approvals are part of the repository policy.

Force-push restrictions and protected-branch rules should not be weakened merely because a classifier predicts low risk.

## Performance considerations

For `n` observations, evaluating every unique score threshold with a complete scan has approximately `O(n²)` behavior in the straightforward implementations.

This is acceptable for instructional and moderate-sized datasets but can become expensive for large validation sets.

A production implementation can sort observations once by score and update TP and FP incrementally as the threshold moves.

That changes the dominant computation toward sorting complexity, approximately `O(n log n)`, followed by a linear sweep.

The mathematical objective remains the same, but the implementation becomes substantially more efficient.

The database version can also benefit from indexes on evaluation split and score when large numbers of model predictions are stored.

## Relationship between the six implementations

| Deliverable | Technical perspective |
|---|---|
| Python | Complete analytical workflow, validation, metrics, optimization, cost models, bootstrap uncertainty |
| JavaScript | Event-driven threshold service and asynchronous scoring boundary |
| C++ | Repository governance case study with a separate review-risk classifier |
| Java | Enterprise domain model with explicit Pull Request and review state transitions |
| SQL | Relational governance model and database-side statistical analysis |
| README | Cross-implementation technical interpretation and distinctions |

The implementations intentionally do not use identical structures. Each language is used to expose a different engineering perspective while preserving the mathematical meaning of ROC, PR, and threshold analysis.

## Practical decision framework

The appropriate workflow is to first establish whether the model ranks positive cases above negative cases. ROC AUC and PR metrics provide complementary evidence for that question.

The next question is operational: what should happen when the model emits a particular score?

That requires an explicit objective.

A review-routing system may prefer high recall because missing a risky change is expensive.

A high-volume triage system may impose a strict FPR ceiling because reviewer capacity is limited.

A system with asymmetric error costs can optimize expected cost.

A balanced classification objective can use F1 or another explicitly chosen metric.

The selected threshold should then be evaluated on held-out data before being used in production.

The resulting threshold is a policy decision built on top of model evidence, not a property of AUC itself.
