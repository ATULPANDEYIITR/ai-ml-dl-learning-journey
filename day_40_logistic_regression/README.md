# Logistic Regression: Sigmoid, Logits, Decision Boundaries, and Binary Classification

## Scope

This learning artifact develops logistic regression as a binary classification method and connects its mathematical behavior to practical classification decisions.

The implementations focus on four tightly related mechanisms:

- **Logits** represent the unrestricted real-valued score produced before probability conversion.
- **The sigmoid function** maps a logit to a value between zero and one that can be interpreted as a class probability.
- **The decision boundary** is the region where the classifier changes its predicted class.
- **Binary classification** converts predicted probabilities into discrete class labels according to a threshold.

The implementations deliberately use a different practical setting in each language. Python contains a complete learning-oriented logistic regression implementation. JavaScript models an asynchronous loan-risk service. C++ treats logistic regression as a risk component inside a repository merge-eligibility engine. Java models the same general classification mechanism as part of an enterprise repository governance service. PostgreSQL stores the governance entities, review decisions, status checks, policies, and model scores relationally.

The important distinction is that a probability prediction is not itself a business or governance decision. A classifier estimates a probability, while a surrounding system decides how that probability should be used.

## Mathematical Foundation

For an input vector \(x\), logistic regression calculates a linear score:

`z = w1*x1 + w2*x2 + ... + wn*xn + b`

The value `z` is the **logit**.

Unlike a probability, a logit has no requirement to remain between zero and one. It can be negative, zero, or positive and can become arbitrarily large in magnitude.

The sigmoid function converts the logit into a probability:

`p = 1 / (1 + e^(-z))`

The important points are:

| Logit | Sigmoid probability | Classification meaning at threshold 0.5 |
|---:|---:|---|
| Large negative | Near 0 | Class 0 |
| Negative | Below 0.5 | Class 0 |
| 0 | Exactly 0.5 | Boundary |
| Positive | Above 0.5 | Class 1 |
| Large positive | Near 1 | Class 1 |

The sigmoid is monotonic. Increasing the logit always increases the predicted probability.

The inverse relationship is the logit transformation:

`log(p / (1 - p)) = z`

This explains why logistic regression is called a regression model even though it is commonly used for classification. The model performs linear regression in log-odds space and then transforms the result into probability space.

## Binary Classification

Binary classification has two target classes, conventionally represented as `0` and `1`.

A logistic regression model normally produces a probability rather than directly producing a class:

`P(y = 1 | x) = sigmoid(w.x + b)`

A classification threshold then determines the final label.

At a threshold of `0.5`:

`predict 1 when p >= 0.5`

Since `sigmoid(z) = 0.5` exactly when `z = 0`, the corresponding decision boundary is:

`w.x + b = 0`

This relationship is important because the probability threshold and geometric boundary are mathematically connected. Changing the threshold changes the effective classification boundary even when the trained model parameters remain unchanged.

## Decision Boundaries

With two input features, the boundary has the form:

`w1*x1 + w2*x2 + b = 0`

If `w2` is non-zero, it can be expressed as:

`x2 = -(w1/w2)x1 - b/w2`

This is a straight line in two-dimensional feature space.

With three features, the boundary becomes a plane. With more features, it becomes a hyperplane.

The Python and JavaScript implementations expose the two-feature boundary equation directly. The C++ and Java implementations use logistic regression differently: the probability is incorporated into a larger merge-governance decision system rather than being presented as a geometric plotting exercise.

A key limitation follows from this structure. Standard logistic regression creates a linear decision boundary in the transformed feature space. If the true separation is strongly nonlinear, raw features may not be sufficient without feature engineering or a different model family.

## Binary Cross-Entropy

The standard loss function is binary cross-entropy:

`L = -[y log(p) + (1-y) log(1-p)]`

For a dataset, the individual losses are averaged.

When `y = 1`, a prediction close to one produces a small loss and a prediction close to zero produces a large loss.

When `y = 0`, a prediction close to zero produces a small loss and a prediction close to one produces a large loss.

This makes the loss sensitive to confident mistakes.

The Python implementation also computes binary cross-entropy directly from logits using a numerically stable expression. This avoids relying on a probability that may have rounded extremely close to zero or one before the logarithm is evaluated.

## Gradient Descent

The Python implementation trains logistic regression with batch gradient descent.

For each training record, the model computes:

`z = w.x + b`

`p = sigmoid(z)`

The prediction error is:

`p - y`

The feature gradient is obtained by multiplying this error by the corresponding feature value. The intercept gradient is the error itself.

The parameters are then updated in the opposite direction of the gradient:

`w = w - learning_rate * gradient`

`b = b - learning_rate * bias_gradient`

This implementation makes the relationship between the sigmoid output and the optimization process explicit. The model does not need a separate manually coded derivative of the sigmoid for the final gradient because the derivative of the logistic loss simplifies the expression to `p - y`.

## L2 Regularization

The Python implementation supports L2 regularization.

The penalty is based on the squared feature weights:

`lambda / (2n) * sum(wj^2)`

The intercept is deliberately excluded from this penalty.

Regularization discourages unnecessarily large coefficients. It can improve generalization when the training data permits highly variable parameter values.

The trade-off is that excessive regularization can constrain the model too strongly and produce underfitting.

The implementation includes a comparison of multiple regularization strengths and reports the resulting weight magnitude and test F1 score.

## Feature Scaling

The Python and JavaScript implementations standardize numeric features before training.

For each feature:

`standardized = (value - mean) / standard_deviation`

Scaling is particularly important for gradient-based optimization when features have substantially different numerical ranges.

For example, an input measured in thousands may otherwise dominate another input measured between zero and one simply because of its scale.

The standardizer also handles constant features by assigning them a unit scale instead of attempting division by zero.

The scale parameters are fitted using training data and then reused for test and production records. This avoids leaking information from the test set into the training transformation.

## Python Implementation

The Python program is the most complete standalone implementation of the classifier.

It contains a `LogisticRegression` class with:

- batch gradient descent;
- sigmoid probability prediction;
- direct logit calculation;
- binary cross-entropy;
- numerically stable loss calculation;
- L2 regularization;
- configurable classification threshold;
- prediction and accuracy methods;
- a two-dimensional decision-boundary calculation;
- training history;
- feature standardization.

The synthetic fraud-screening dataset contains transaction amount and account age. The generated target reflects a relationship in which larger transaction amounts increase fraud risk while older accounts reduce it.

This example demonstrates why logistic regression parameters should be interpreted relative to feature scaling. The model operates on standardized values during optimization while predictions are made from new records after applying the same training transformation.

The Python implementation also calculates precision, recall, specificity, F1, and the confusion matrix. Accuracy alone can hide important classification behavior, especially when the two classes are not equally costly.

## Classification Thresholds

A model probability and a classification threshold are separate concepts.

Suppose a model produces:

`P(class=1) = 0.65`

At a threshold of `0.50`, this becomes class `1`.

At a threshold of `0.70`, it becomes class `0`.

The underlying model parameters did not change.

The Python and JavaScript programs compare thresholds such as `0.30`, `0.50`, `0.70`, and `0.90`. This demonstrates the precision-recall trade-off directly.

A lower threshold generally makes positive predictions easier to obtain. This tends to increase recall while potentially increasing false positives.

A higher threshold makes positive predictions harder. This can reduce false positives while potentially increasing false negatives.

The appropriate threshold depends on the cost of different errors. A default-risk screening system, fraud detector, medical screening system, and marketing classifier can legitimately require different thresholds even when they use the same mathematical model.

## Numerical Stability

The sigmoid function contains an exponential:

`e^(-z)`

Directly evaluating this expression for extreme values can produce numerical overflow.

The Python and JavaScript implementations use a branch based on the sign of the logit. For positive logits they evaluate the negative exponential. For negative logits they evaluate the positive exponential instead.

This preserves numerical behavior for values such as `-1000` and `1000`.

The Python loss implementation also uses a logit-based expression instead of calculating a probability and then immediately taking logarithms. This is a standard numerical-stability technique for logistic loss.

The lesson is broader than this particular model: mathematically equivalent expressions are not necessarily equally safe in floating-point arithmetic.

## JavaScript Implementation

The JavaScript program models loan-risk classification as an executable Node.js service.

Its `LogisticRegression` class provides:

- sigmoid conversion;
- logit calculation;
- gradient descent;
- regularization;
- probability prediction;
- classification;
- loss calculation;
- accuracy;
- decision-boundary interpretation;
- model serialization.

The `StandardScaler` class handles feature transformation separately from the classifier.

The `LoanRiskService` introduces an application-layer abstraction around the trained model. Its asynchronous `evaluate()` method demonstrates how a classifier can become part of an event-driven Node.js application rather than existing only as an offline mathematical exercise.

The service validates domain values such as debt ratio and income before prediction. This separates model assumptions from application-level input validation.

The model is also serialized into JSON and restored through `fromSerialized()`. This demonstrates the practical distinction between model training and model serving. A production application commonly needs to load a previously trained parameter set rather than retrain for every prediction.

## C++ Case Study: Repository Merge Risk

The C++ program uses logistic regression inside a repository governance scenario.

A Pull Request contains operational features such as changed-file volume, additions, deletions, commit count, synchronization status, and unresolved review conversations.

The logistic model converts those characteristics into a predicted defect probability.

The important architectural distinction is that this probability is not allowed to override repository policy.

The `MergeEligibilityEngine` separately checks:

- whether the Pull Request is open;
- whether it targets the protected branch;
- whether source and target branches differ;
- whether merge conflicts exist;
- whether the source branch is synchronized;
- whether sufficient eligible reviewers approved;
- whether required status checks passed;
- whether required conversations are resolved;
- whether the predicted defect risk remains below the configured threshold.

This separates **probabilistic classification** from **deterministic governance**.

A model can say that a change appears low risk, but it cannot legitimately turn a missing mandatory security approval into an approval.

The case study therefore demonstrates an important systems principle: predictive models should usually be one component of a decision process rather than silently replacing explicit control rules.

## Java Enterprise Model

The Java program represents repository governance using explicit domain types.

`PullRequestStatus`, `ReviewStatus`, and `CheckStatus` are enums. This avoids representing important workflow states as arbitrary strings throughout the application.

`Review` captures reviewer identity, review state, approval eligibility, and review text.

`StatusCheck` represents automated checks independently of human review.

`BranchProtectionPolicy` represents repository-level governance requirements.

`PullRequest` manages lifecycle behavior such as moving from draft to open, synchronizing with the base branch, recording conflicts, adding reviews, recording status checks, and resolving conversations.

The model deliberately rejects invalid transitions. For example, a review cannot be added to a Pull Request that is not open.

`MergeEligibilityService` then combines deterministic repository rules with the logistic risk probability.

This structure demonstrates why enterprise implementations often separate domain state from policy evaluation. The Pull Request owns workflow state, while the eligibility service evaluates whether the current state satisfies a particular policy.

The Java implementation also uses immutable-style records for value objects and returns copied collections from the Pull Request rather than exposing mutable internal lists directly.

## Pull Requests, Code Review, Approvals, and Branch Protection

These concepts interact but are not interchangeable.

### Pull Requests

A Pull Request is the change-proposal workflow.

It identifies a source branch and a target branch and provides the mechanism through which a changeset is proposed for integration.

A Pull Request can begin as a draft, become open for review, receive additional commits, become synchronized with its base branch, encounter merge conflicts, be closed, be reopened where the hosting platform permits it, or eventually be merged.

The changeset is composed of commits and the resulting file differences between the source and target branches.

Synchronization matters because the state originally reviewed may differ from the current state after the base branch has advanced or additional commits have been pushed.

The C++ and Java models treat branch synchronization as an explicit merge condition rather than treating an old review as automatically sufficient.

### Code Review

Code Review evaluates the proposed changes.

A review can contain general feedback or inline comments associated with particular files and lines. It can also express a formal state such as approval or requested changes.

A useful review evaluates criteria such as:

- correctness of the implementation;
- failure behavior;
- validation;
- security-sensitive changes;
- concurrency implications;
- maintainability;
- tests and regression coverage;
- consistency with system architecture.

Review scope matters. A reviewer should understand what changed and why rather than treating the existence of a passing automated test as proof that the implementation is correct.

A review comment and an approval are not equivalent. A comment can raise an issue without becoming an approval decision.

A requested-change state signals that the reviewer considers modification necessary before the change should be accepted.

The database represents review comments separately from review states so that unresolved discussion can be evaluated independently of the approval count.

### Approvals

An approval is a specific review decision.

The important distinction is between the existence of a review and the eligibility of that review to satisfy a protected-branch approval requirement.

The SQL model therefore stores reviewer eligibility separately from the review itself.

An eligible approval must satisfy the repository's approval rules. A comment from an ineligible observer does not become a required approval merely because its review record exists.

Approval requirements can also become stale after new changes. The database example contains a dismissed approval for a large refactor to demonstrate that historical approval information does not necessarily remain a valid merge authorization after material changes.

Approval dismissal is therefore a governance mechanism, not simply a user-interface event.

### Branch Protection

Branch protection is the repository-level enforcement layer.

It can require approvals, require successful status checks, require conversation resolution, restrict direct pushes, prevent force pushes, prevent branch deletion, and require linear history depending on the platform and configured policy.

Branch protection changes the meaning of merge eligibility. A developer may have write access to a repository while still being unable to directly update a protected production branch.

The C++ and Java implementations treat these restrictions as deterministic policy conditions.

The SQL schema stores the policy as persistent data so the database can report which requirements apply to a particular protected branch.

## Relationship Between the Four Mechanisms

The relationship can be represented as:

`Pull Request -> proposes the changeset`

`Code Review -> evaluates the changeset`

`Approval -> records an eligible review decision`

`Branch Protection -> enforces repository-level merge conditions`

Logistic regression adds another layer:

`Logistic Regression -> estimates probability associated with a classification outcome`

The probability should not be confused with any of the four repository mechanisms.

For example, a Pull Request may have a predicted defect probability of `0.12`, but if the protected branch requires two eligible approvals and only one exists, the Pull Request remains blocked.

Likewise, a Pull Request can have all required approvals but still be blocked because an integration test failed.

## SQL Data Model

The PostgreSQL schema models the governance system relationally.

`repositories` identifies the repository and its default branch.

`branches` identifies branches and records whether they are protected.

`pull_requests` stores source and target branches, author, lifecycle state, change volume, synchronization state, merge conflicts, and conversation resolution.

`commits` associates concrete commits with a Pull Request.

`reviews` stores reviewer decisions such as comments, requested changes, approvals, and dismissed approvals.

`review_comments` stores detailed discussion separately from the high-level review state and tracks whether individual discussions are resolved.

`status_checks` represents automated validation independently from human review.

`branch_protection_policies` stores repository-level restrictions and required approvals.

`approval_policies` represents approval requirements as explicit policy data.

`reviewer_eligibility` determines which developers are permitted to satisfy approval requirements.

`logistic_risk_scores` stores the model version, logit, probability, and classification threshold associated with a Pull Request.

`merge_attempts` provides an audit trail for merge eligibility decisions.

Foreign keys preserve relationships between these entities. Unique constraints prevent duplicate repository branch names, duplicate commit hashes, and duplicate Pull Request numbers within a repository.

## Database-Level Merge Evaluation

The `active_eligible_approvals` view filters approvals using both review state and reviewer eligibility.

This prevents a simple count of all reviews from being mistaken for a valid approval count.

The `unresolved_review_conversations` view counts unresolved comments separately.

The `merge_eligibility` view then combines approvals, required checks, unresolved conversations, synchronization state, merge conflicts, branch protection requirements, and the stored logistic probability.

The final query applies both deterministic policy eligibility and the model risk threshold.

This separation makes the database result auditable. A system administrator can see whether a Pull Request is blocked because of missing approvals, failed checks, unresolved conversations, synchronization state, or predicted risk.

## Review Quality and Failure Modes

A review process can fail even when the workflow technically completes.

A reviewer may approve without examining the changed code deeply enough.

A review may become stale after additional commits.

A discussion may remain unresolved even though an approval exists.

An automated status check can pass while an architectural problem remains undetected.

An eligible reviewer can also approve an unsafe change if the review criteria are weak.

The data model therefore does not treat approval count as a complete measure of code quality. Approval is one governance condition among several.

## Probability Versus Decision

Logistic regression estimates a probability. It does not establish certainty.

A probability of `0.70` does not mean that an event will occur exactly seventy percent of the time for a particular individual Pull Request.

It means that, under the model's assumptions and calibration, the model associates the observed feature pattern with an estimated class probability.

A classification threshold converts that continuous output into a discrete decision.

This distinction is central:

`model output -> probability`

`threshold -> classification`

`business or governance policy -> action`

The C++ and Java implementations make this distinction explicit by evaluating model risk separately from repository policy.

## Common Modeling Errors

### Treating probability as certainty

A value close to one is still a model estimate. It does not prove that the positive class will occur.

### Using accuracy alone

Accuracy can conceal poor performance on an important minority class. Precision, recall, specificity, F1, and the confusion matrix provide a more informative view of binary classification behavior.

### Ignoring class imbalance

If one class dominates the training set, a classifier can obtain high accuracy while performing poorly on the minority class.

Threshold analysis and class-sensitive evaluation become important in such cases.

### Scaling the test set independently

The transformation used for test data should be derived from training data. Independently fitting a scaler to the test set changes the representation and introduces information leakage into evaluation.

### Treating the threshold as part of training

Changing a classification threshold does not necessarily retrain the model. The threshold is a decision-policy parameter applied to model probabilities.

### Assuming a linear boundary fits every problem

Logistic regression creates a linear boundary in the chosen feature space. A nonlinear relationship may require transformed features or a different modeling approach.

## Production Considerations

A production logistic regression service should preserve the exact feature definitions used during training.

Changing a feature's unit, missing-value handling, normalization rule, or category encoding can alter the meaning of the learned coefficients.

The model artifact should be versioned with the preprocessing configuration.

Probability calibration should also be evaluated when downstream decisions depend directly on probability magnitude rather than only class labels.

Thresholds should be treated as explicit policy parameters rather than hidden constants scattered through application code.

Input validation should occur before inference.

Extreme numeric inputs should be handled deliberately.

Model and prediction logging should avoid storing sensitive information unnecessarily.

For systems where predictions influence consequential decisions, the model's performance should be monitored after deployment rather than assumed to remain constant.

## Performance Considerations

For `n` samples and `d` features, one batch gradient-descent epoch has approximately `O(n*d)` computational work.

With `e` epochs, the training cost is approximately:

`O(e*n*d)`

Prediction for one record is approximately `O(d)`.

Memory usage for the core batch implementation is dominated by the feature matrix and model parameters.

The Python implementation uses ordinary lists to make the underlying mechanics visible. Production numerical workloads would commonly use optimized array operations, but the mathematical behavior remains the same.

The C++ and Java examples demonstrate a different performance concern: model inference can be inexpensive compared with the surrounding repository policy evaluation, database access, status-check retrieval, and review-state queries.

## Security and Governance Considerations

A classification model should not be treated as an authorization mechanism by itself.

The repository examples deliberately enforce approval eligibility and branch protection independently of the model probability.

A malicious or compromised model score must not be able to grant repository permissions.

Protected-branch enforcement should remain authoritative at the repository or platform layer.

Reviewer identities should be authenticated by the surrounding platform rather than trusted from arbitrary client input.

Audit records should distinguish who attempted a merge from who approved the change.

Model version information should be retained with predictions so that historical decisions can be interpreted against the model that produced them.

## Debugging Considerations

When a prediction looks incorrect, inspect the complete path:

`raw input -> validation -> preprocessing -> logit -> sigmoid -> probability -> threshold -> decision`

If the probability is unexpected, inspect the logit before inspecting the threshold.

If the logit is unexpected, inspect the feature values and coefficient ordering.

If the logit is correct but the class is unexpected, inspect the threshold.

If the model probability is reasonable but the system decision is unexpected, inspect the surrounding policy layer.

The repository case studies reinforce this separation. A Pull Request can be blocked even when the classifier considers it low risk because deterministic governance requirements have not been satisfied.

## Practical Interpretation

Logistic regression is particularly useful when a binary outcome is influenced by several measurable factors and the relationship can be represented reasonably well through a linear combination of features in log-odds space.

Its main computational path is compact:

`features -> linear score -> logit -> sigmoid -> probability -> threshold -> class`

Its practical value comes from combining this simple mathematical structure with careful feature design, validation, loss optimization, evaluation, threshold selection, and explicit decision policy.

The implementations demonstrate that the same classifier can serve different roles: a standalone educational model, an asynchronous application service, a risk component inside a governance engine, and a database-backed decision workflow. The mathematical core remains logistic regression, while the surrounding engineering determines how safely and meaningfully its output is used.
