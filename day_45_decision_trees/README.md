# Decision Trees: Entropy, Gini Impurity, Information Gain, and Splitting

## Technical scope

A decision tree is a supervised learning model that predicts a class by applying a sequence of feature-based conditions. Each internal node partitions the observations into child groups, and each leaf produces a prediction derived from the observations that reach it.

This project examines classification trees through a software-release risk scenario. Historical release observations contain test coverage, failed-test counts, and unresolved security findings. The target class is either `eligible` or `manual_review`.

The classification labels represent an illustrative dataset, not an authoritative deployment policy. A production release system must enforce its actual security and quality requirements independently of a statistical prediction.

The six files implement complementary views of the same subject:

- Python builds a reusable classifier with numerical and categorical splits, predictions, evaluation, and JSON model-structure export.
- JavaScript emphasizes deterministic model construction, split-decision tracing, validation, and safe serialization in a Node.js environment.
- C++ implements a release-risk case study with an explicit node arena, typed features, validation, and weighted feature importance.
- Java models the classifier with domain-specific records, enums, immutable observation data, and explicit tree-building methods.
- PostgreSQL stores observations, defines impurity calculations, generates candidate thresholds, ranks candidate splits, and enforces data integrity.
- This README connects the mathematical criteria to the behavior and limitations of those implementations.

## The classification problem

Each observation has features and a known class:

| Feature | Meaning | Example |
|---|---|---:|
| `coverage` | Percentage of required automated tests covered | 89 |
| `failed_tests` | Number of failed automated tests | 0 |
| `security_findings` | Number of unresolved security findings | 0 |
| `decision` | Historical classification target | `eligible` |

A learned tree might evaluate a condition such as `coverage <= 74.5`. Observations satisfying the condition go to one child; the remaining observations go to the other. A later node can inspect another feature.

The threshold is learned from the training data. It is not a threshold that should automatically be adopted as an organizational release rule.

The target label must not be confused with a prediction input. In particular, a training observation's historical `decision` is used to evaluate candidate splits, not as a feature during prediction. Including the target as a feature would create data leakage and make reported performance misleading.

## Entropy and class uncertainty

For a node containing \(N\) observations, let \(p_k\) denote the proportion belonging to class \(k\). Shannon entropy is

\[
H(S)=-\sum_{k=1}^{K}p_k\log_2(p_k).
\]

Entropy measures class uncertainty in bits.

A pure node has entropy zero because every observation belongs to the same class. For a binary classification problem, a node containing equal numbers of both classes has entropy one bit. An uneven class distribution has an entropy between zero and one.

For example, consider a node containing six `eligible` observations and two `manual_review` observations. The class probabilities are \(0.75\) and \(0.25\), so its entropy is

\[
H(S)=-(0.75\log_2 0.75+0.25\log_2 0.25)
\approx 0.8113.
\]

Entropy is particularly useful for explaining information gain because both quantities use the same uncertainty measure. A split is valuable when its children have lower weighted entropy than their parent.

The Python implementation calculates entropy in `entropy()`, while the C++ and Java implementations compute it using the same base-two logarithm. PostgreSQL uses `ln(p) / ln(2)` to convert the natural logarithm into a base-two logarithm.

Zero-probability classes are excluded from logarithm evaluation. This prevents the calculation from attempting to evaluate the logarithm of zero.

## Gini impurity

Gini impurity is defined as

\[
G(S)=1-\sum_{k=1}^{K}p_k^2.
\]

It is zero when the node is pure. For two equally represented classes, its value is \(0.5\). For a binary classification problem, its maximum is \(0.5\); with more classes, the maximum depends on the number of classes.

Gini impurity is not entropy expressed in another unit. It is a different function of the class probabilities. Both criteria measure node impurity, but their numerical values and responses to class distributions differ.

Gini impurity is frequently used in CART-style classification trees. It avoids logarithm calculations and is straightforward to evaluate. Entropy offers a direct information-theoretic interpretation. Neither criterion is guaranteed to produce better held-out predictions for every dataset.

The Python, JavaScript, C++, and Java programs allow the criterion to be selected. The SQL script defines `impurity_criterion` as an enumeration and calculates impurity according to the criterion stored for each training run.

## Weighted child impurity

A split must be evaluated across both children, not by inspecting only the child with the lowest impurity.

If a parent contains \(N\) observations and the split produces children \(S_L\) and \(S_R\), the weighted child impurity is

\[
I_{\text{children}}
=
\frac{|S_L|}{N}I(S_L)
+
\frac{|S_R|}{N}I(S_R).
\]

The weights matter because a small pure child should not automatically outweigh a much larger mixed child.

Consider a parent with 100 observations. Suppose a split produces a child with 10 observations and impurity zero, and a second child with 90 observations and impurity 0.4. The weighted child impurity is

\[
\frac{10}{100}(0)+\frac{90}{100}(0.4)=0.36.
\]

The result is 0.36, not zero. The larger child contributes most of the remaining uncertainty.

Each implementation uses observation counts to weight the children. The Python function `weighted_child_impurity()` makes this calculation reusable. The JavaScript function `weightedImpurity()` accepts an impurity function. The C++ and Java tree builders calculate the weighted value during candidate evaluation. PostgreSQL derives child proportions from aggregate counts.

## Information gain and Gini reduction

Information gain is the decrease in entropy caused by a split:

\[
IG(S)=H(S)-\left(
\frac{|S_L|}{|S|}H(S_L)
+
\frac{|S_R|}{|S|}H(S_R)
\right).
\]

A positive information gain indicates that the children have lower weighted entropy than the parent. A gain of zero means the split does not reduce entropy. A negative gain should not occur for a correctly evaluated partition of the same observations, apart from possible numerical rounding effects.

The equivalent criterion based on Gini impurity is

\[
\Delta G=G(S)-\left(
\frac{|S_L|}{|S|}G(S_L)
+
\frac{|S_R|}{|S|}G(S_R)
\right).
\]

Information gain and Gini reduction use different impurity functions. Their scores should not be compared as if they shared the same scale.

A tree-building algorithm usually evaluates many candidate splits and selects the one with the largest positive reduction under its chosen criterion. It then repeats the process independently within each resulting child.

The implementations reject splits that fail the minimum-child-size rule or do not exceed the configured minimum gain. These restrictions prevent the tree from creating branches that are too small or that provide negligible improvement.

## How candidate thresholds are generated

For a numeric feature, a threshold partitions observations according to the rule

`feature <= threshold`.

Suppose observed test-coverage values are 60, 70, 80, and 90. Candidate midpoints are 65, 75, and 85. Every midpoint lies between two adjacent distinct observed values and therefore represents a possible partition of the training observations.

There is usually no benefit to evaluating arbitrary thresholds that leave the same observations on each side. Midpoints between adjacent distinct values provide representative partitions without testing every real number.

The Python implementation generates thresholds only between adjacent values with different labels, reducing work in some cases. It also supports a configurable limit on the number of candidate thresholds. Limiting candidates can reduce training time for high-cardinality features, at the cost of potentially missing the best split.

The JavaScript, C++, and Java implementations use sorted distinct feature values and evaluate adjacent midpoints. PostgreSQL uses the `LEAD()` window function over distinct feature values to generate those midpoints.

For integer-valued features such as failed-test counts, a midpoint such as 1.5 separates the integer values 1 and 2. The midpoint is a valid numerical partition even when no observation has that exact value.

### Numerical versus categorical features

A numerical feature supports threshold comparisons because its values have an ordering. A nominal categorical feature, such as a deployment region, does not necessarily have a meaningful numeric distance.

The Python classifier supports both numerical thresholds and categorical equality splits. A categorical split asks whether a feature equals one category. The remaining categories go to the other child.

The JavaScript, C++, and Java examples concentrate on numerical features so their candidate evaluation and release-risk case studies remain focused. The SQL schema explicitly declares its supplied features numeric.

Ordinal encoding must be treated carefully. Converting categories into integers can imply a false ordering and introduce splits that have no meaningful domain interpretation. Missing values also require an explicit policy, such as validated imputation, a dedicated missing category, or a documented routing rule. The sample implementations reject missing or invalid values rather than silently assigning them to a branch.

## Recursive construction and stopping rules

A tree begins with all training observations at the root. It evaluates candidate splits and assigns observations to children. The same process is repeated within each child until a stopping condition is reached.

The programs implement these stopping conditions:

- **Maximum depth:** Prevents unlimited recursive growth. Depth zero represents the root.
- **Minimum samples per leaf:** Prevents a candidate from creating a child with too few observations.
- **Minimum gain:** Rejects splits whose impurity reduction is too small.
- **Pure node:** Stops splitting when all observations in the node share a class.
- **No valid split:** Produces a leaf when the available features cannot produce an acceptable partition.

A leaf predicts its majority class. In the C++ and Java implementations, a tie is resolved toward `manual_review`, making the example conservative when class counts are equal. The Python and JavaScript implementations use a deterministic alphabetical tie-breaker. This is an implementation choice, not a universal decision-tree rule.

The stopping conditions are forms of regularization. A deeper tree can model more detailed patterns but can also fit noise and rare observations. A shallower tree is easier to inspect but may miss meaningful interactions.

## Python implementation

The Python script is a standalone classifier built with the standard library.

### Data validation and model fitting

`DecisionTreeClassifier` validates feature names, row-label counts, supported value types, and missing or non-finite values. It then recursively constructs `TreeNode` objects.

Each node stores its prediction, class counts, sample count, impurity, depth, split feature, threshold or category, gain, and child references. Internal nodes describe a condition. Leaf nodes describe a prediction.

The classifier uses `fit()` to learn the tree, `predict()` to classify new observations, and `predict_proba()` to return empirical class frequencies for the leaf reached by each observation.

These frequencies are not calibrated probabilities. A leaf with two observations can return a frequency of 1.0 for a class even though the evidence is limited. Reliable probability estimates require suitable validation and, when needed, probability calibration.

### Evaluation and diagnostics

The program creates an illustrative release dataset, separates training and test observations using a deterministic stratified split, and reports held-out accuracy and a confusion matrix.

The confusion matrix distinguishes actual classes from predicted classes. In an operational risk system, false `eligible` predictions can be more costly than false `manual_review` predictions. Accuracy alone does not express that difference. Precision, recall, class-specific costs, and threshold-independent metrics may be required for an actual deployment.

The script compares entropy and Gini criteria and explores different maximum depths. These experiments illustrate why a tree's configuration must be evaluated rather than selected solely because it produces a large training score.

Feature importance accumulates the reduction in impurity at each node, weighted by the proportion of training observations reaching that node. The result is normalized across features. This measure describes how the fitted tree uses features; it does not prove causal influence or guarantee importance stability across datasets.

### Model structure export

The Python script writes the fitted tree structure to `deployment_decision_tree.json`. The output contains node attributes and children as JSON data. It does not serialize executable Python code.

The script reads the file back and checks the root sample count. This validates the basic serialization round trip but does not implement a complete model-loading interface or establish that the model is safe and suitable for production use.

## JavaScript implementation

The Node.js implementation emphasizes reproducible split evaluation and an audit trail.

`candidateThresholds()` sorts unique values and returns adjacent midpoints. `scoreThreshold()` calculates weighted impurity reduction for a candidate. `DecisionTree.fit()` evaluates features and thresholds, selects the best valid split, and recursively builds the children.

The `trace` collection records the depth, feature, threshold, gain, and sample count for every accepted split. This provides a compact explanation of how training decisions were made and helps identify unexpected split behavior during debugging.

The implementation uses `Map` to count classes, then converts node counts into plain objects for serialization. It sorts tied class labels to ensure deterministic majority predictions. The input validator rejects non-finite numbers and mismatched row-label counts.

`toJSON()` serializes model data rather than source code. Deserializing a model must not involve evaluating its contents as JavaScript. Production systems should also validate the model version, feature schema, and expected value ranges before using stored predictions.

The example evaluates its training data as a diagnostic. That score is not an estimate of generalization performance because the same observations were used to fit and evaluate the model. A separate held-out dataset or cross-validation is needed for an unbiased assessment.

## C++ case study

The C++ program represents a release-risk classifier as a typed, in-memory system.

### Data and architecture

`Observation` contains numerical quality signals and the target label. `Feature` enumerates the available input features, while `Criterion` distinguishes entropy from Gini impurity.

`Counts` stores the two class counts and provides the majority-class rule. `SplitCandidate` holds a proposed partition, its threshold, its gain, and the resulting observations. `Node` stores the learned decision and child indices.

The tree stores nodes in a vector. Each node refers to its children by index rather than retaining pointers to vector elements. This avoids pointer invalidation when vector growth reallocates its storage.

### Split evaluation

`bestSplit()` gathers distinct values for each feature and evaluates the midpoint between adjacent values. It creates left and right partitions, enforces the minimum leaf size, calculates the weighted child impurity, and retains the candidate with the greatest gain.

`build()` creates the current node before recursively creating children. The parent is referenced by index, so the recursive calls do not depend on the stability of an address inside the vector.

The implementation validates coverage, test-failure counts, security-finding counts, and target labels. Invalid prediction data raises an exception instead of producing an unexplained classification.

### Importance and operational limitations

Feature importance is based on the weighted gains accumulated during construction. The printed tree exposes thresholds and node impurity so that a reviewer can inspect the model's behavior.

The program is a compact case study rather than a production training library. It does not implement categorical splits, missing-value routing, probability calibration, model version management, or a full model persistence format. Its in-memory training data is also small enough that straightforward candidate evaluation is appropriate.

## Java implementation

The Java program uses domain types to make the release-risk model explicit.

### Domain representation

`ReleaseObservation` is a record whose compact constructor validates the coverage range and nonnegative counts. `Decision`, `Criterion`, and `Feature` are enums, preventing unsupported values from silently entering the model.

`ClassDistribution` maintains the two class counts and derives both the majority class and the empirical class frequencies. It rejects empty or inconsistent distributions.

The `Node` class represents either a leaf or a split. `Split` is an immutable record-like object containing the feature, threshold, gain, and copied child partitions. `DecisionTree` owns the training parameters, root node, and weighted-gain statistics.

### Policy and state behavior

The tree's construction logic is separated from prediction and rendering. `fit()` validates and copies the training observations. `findBestSplit()` ranks candidates. `build()` applies stopping conditions and constructs children. `predict()` follows the conditions from root to leaf.

This separation makes the model's responsibilities clearer and reduces accidental mixing of training logic with inference behavior. The use of enums also makes criterion selection explicit and easier to review.

The example's conservative tie rule is part of its domain behavior. An enterprise system should not assume that a classifier's majority class automatically establishes authorization to release software. Security-critical rules should remain explicit and independently enforced.

## PostgreSQL data model

The SQL script treats training observations and split evaluations as relational data.

### Core tables and integrity

`training_run` stores the selected impurity criterion and tree regularization settings. `training_observation` stores feature values and target labels. Its checks enforce the coverage range and nonnegative integer counts.

`feature_definition` documents the numerical features that can participate in a split. `candidate_split` stores each evaluated threshold, the counts on both sides, the parent impurity, the weighted child impurity, and the gain.

`tree_node` stores node statistics and split metadata, while `node_membership` associates observations with nodes. These structures provide the foundation for storing a complete learned tree and its membership information.

The foreign keys, unique constraints, and checks enforce relational integrity. The script demonstrates an invalid coverage insertion inside a nested exception block. PostgreSQL rejects the row because it violates the coverage check; the exception handler catches the expected failure so the script can continue.

### Impurity function

`class_impurity()` is an immutable PL/pgSQL function. It validates class counts, handles an empty distribution, and calculates either entropy or Gini impurity. The entropy branch avoids evaluating a logarithm for a zero-probability class.

The `run_impurity` view joins each training run to its overall class distribution and computes the parent impurity using the configured criterion.

### Candidate generation and ranking

`candidate_thresholds` uses `LEAD()` to find the next distinct feature value within each run and feature. The midpoint becomes a candidate threshold.

`evaluated_candidate_splits` joins each candidate with the observations for the corresponding run. Conditional aggregates calculate the left and right sample counts and their class counts. These counts feed the impurity calculation and produce the weighted child impurity and gain.

`ranked_candidate_splits` uses `ROW_NUMBER()` to rank valid candidates by descending gain, with feature name and threshold as deterministic tie-breakers. It filters out splits that violate the configured minimum leaf size or minimum gain.

The script compares entropy-based and Gini-based runs over the same source observations. The SQL workflow calculates and ranks root-level candidate splits; it does not recursively construct a complete decision tree through a stored procedure. Recursive tree construction remains in the Python, JavaScript, C++, and Java implementations.

### Indexing and transaction boundaries

The indexes on `(run_id)`, `(run_id, decision)`, and `(run_id, depth)` support common filtering and inspection patterns. The candidate-split index supports ranking candidates by gain within a run.

The script wraps schema creation, sample data, validation, and queries in a transaction. The transaction makes the demonstration reproducible as a unit. Its initial schema drop is destructive to an existing schema named `decision_tree_lab`, so it should not be executed against a database that contains data that must be retained.

## Regularization, performance, and model reliability

A basic threshold-based tree can evaluate many feature-threshold combinations at every node. If a node contains \(n\) observations and a numerical feature has \(m\) candidate thresholds, evaluating that feature may require work proportional to \(n m\), ignoring the cost of sorting and bookkeeping.

Sorting values costs \(O(n\log n)\). A straightforward implementation that rescans every observation for each candidate threshold can perform substantially more work than an optimized implementation that sorts observations once and updates cumulative class counts as the threshold moves.

The implementations favor clarity and traceability over optimized training performance. For large datasets, repeated allocation of child partitions and repeated sorting can become expensive. Efficient implementations can reuse sorted feature indices, maintain incremental class counts, and avoid materializing every candidate's full child dataset.

Tree depth and leaf size affect both runtime and model complexity. A tree that grows too deep may memorize individual training examples. Minimum leaf size and minimum gain constrain this behavior, although neither guarantees generalization.

A proper evaluation separates model fitting, hyperparameter selection, and final testing. Repeatedly choosing parameters based on the same test set gradually turns that set into part of the training process. Cross-validation or a separate validation set should be used to select parameters, leaving the final test set untouched until evaluation.

## Common implementation errors

- **Ignoring child weights:** Comparing the unweighted average of child impurities can rank splits incorrectly because the children may have very different sizes.
- **Testing redundant thresholds:** Thresholds that do not change the partition add computation without providing new candidate splits.
- **Allowing empty or undersized leaves:** Such branches can produce unstable predictions and misleadingly pure nodes.
- **Treating feature importance as causality:** Impurity reduction indicates how a feature contributes to a particular fitted tree, not whether changing that feature causes the target outcome to change.
- **Evaluating on training data only:** A strong training score does not demonstrate that the tree will classify unseen observations accurately.
- **Ignoring class-dependent error costs:** A model can have high accuracy and still make too many dangerous false approvals.
- **Encoding nominal categories as arbitrary numbers:** Threshold comparisons can impose a false order on values that have no meaningful numerical relationship.
- **Silently accepting missing or non-finite values:** Unspecified routing can make predictions inconsistent and hide data-quality problems.

## Interpretation and deployment safeguards

A decision tree is useful because its path can be expressed as conditions on observable features. That interpretability depends on the selected features, training distribution, depth, and split criteria. A short tree can still be misleading when its data is biased, labels are unreliable, or the target changes over time.

The release-risk scenario illustrates the difference between prediction and authorization. A learned leaf may predict `eligible`, but a deployment service should still enforce mandatory security checks, required approvals, test policies, and other non-negotiable release conditions. A statistical classifier should not override an explicit safety or compliance requirement.

Operational use also requires monitoring feature distributions, class frequencies, prediction errors, and changes in the release process. If test coverage definitions or security scanning methods change, historical training observations may no longer represent current conditions. Model retraining should follow a documented validation process rather than occurring automatically after every data change.
