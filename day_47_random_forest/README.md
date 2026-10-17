# Random Forest: Bagging, Bootstrap Sampling, and Random Feature Selection

## Technical focus

A Random Forest is an ensemble learning method that combines multiple decision trees to produce a classification or regression result. Its performance depends on two related sources of variation:

- **Bootstrap sampling** changes the observations used to train each tree.
- **Random feature selection** changes the candidate input variables considered when choosing each split.

Bagging, short for bootstrap aggregating, combines predictions from models trained on bootstrap samples. Random Forest extends this approach by introducing feature-level randomness during tree construction. This can reduce correlation between individual trees and improve generalization.

The implementations use numerical equipment-inspection measurements to classify assets as `normal` or `maintenance_required`. The scenario illustrates nonlinear decision boundaries, interactions between measurements, ensemble voting, out-of-bag evaluation, and feature importance.

## Decision trees as the base learners

A classification tree partitions a feature space through rules of the form:

`feature[j] <= threshold`

Each split creates a left child for observations satisfying the rule and a right child for the remaining observations. Repeated splits produce increasingly specific regions of the feature space.

A leaf predicts a class derived from the training observations that reach it. The tree can represent nonlinear relationships without requiring a predefined linear equation.

### Gini impurity

For a node containing class proportions \(p_1,\ldots,p_K\), Gini impurity is

\[
G = 1-\sum_{k=1}^{K}p_k^2.
\]

A node containing only one class has impurity zero. A node with a more even distribution of classes has higher impurity.

For a candidate split, the weighted child impurity is

\[
G_{\text{children}} =
\frac{n_L}{n}G_L+\frac{n_R}{n}G_R.
\]

The impurity reduction is

\[
\Delta G = G_{\text{parent}}-G_{\text{children}}.
\]

The tree selects the candidate split with the largest positive reduction among the features and thresholds considered at that node. Minimum leaf size, maximum depth, and minimum split size constrain tree growth.

The Python implementation also calculates entropy to illustrate an alternative measure of class uncertainty. Entropy and Gini often produce similar splits, although they are not mathematically identical.

## Bootstrap sampling and bagging

A bootstrap sample is produced by drawing observations from the training set **with replacement**. If the original training set contains \(n\) observations, the standard procedure draws \(n\) times.

An observation may appear several times in one tree's bootstrap dataset. Another observation may not appear at all. These differences are important: duplicate draws affect the effective training weight, while omitted observations can be used for out-of-bag evaluation.

The probability that one particular observation is omitted in all \(n\) draws is

\[
\left(1-\frac{1}{n}\right)^n.
\]

As \(n\) becomes large, this probability approaches \(e^{-1}\), approximately 0.368. Consequently, a typical bootstrap sample contains around 63.2% of the original observations as distinct records, although the exact proportion varies.

Bagging reduces prediction instability by aggregating models trained on these different samples. For classification, Random Forest commonly uses majority voting. For regression, it commonly averages tree predictions.

Bootstrap sampling alone does not make a collection of trees a Random Forest. Random feature selection is the second key mechanism.

## Random feature selection

At each candidate split, a Random Forest considers a randomly selected subset of available features rather than necessarily evaluating every feature.

For classification, a common starting point is

\[
m \approx \sqrt{p},
\]

where \(p\) is the total number of input features and \(m\) is the number considered at a split. Regression forests often begin with a different default, such as a fraction of the available features. The appropriate setting depends on the dataset and implementation.

The code examples use four inspection measurements:

| Feature | Meaning | Possible predictive signal |
|---|---|---|
| `vibration` | Equipment vibration measurement | Mechanical wear or imbalance |
| `temperature` | Operating temperature | Excessive heat |
| `operating_hours` | Accumulated operating time | Exposure to wear |
| `pressure_deviation` | Difference from expected pressure | Abnormal operating conditions |

The simulated failure rule includes an interaction between vibration and pressure deviation. A tree can capture this relationship through successive splits.

Random feature selection has an important consequence: a feature that is highly predictive across the dataset may not be selected at every node. Other features can therefore become useful in different trees. This reduces the tendency for all trees to follow the same dominant split sequence.

The subset size is a model hyperparameter, not a guarantee of improved accuracy. A subset that is too small can omit useful variables repeatedly, while a subset that is too large can increase similarity between trees.

## How the mechanisms work together

The Random Forest training process follows this structure:

- Each tree receives a bootstrap sample drawn from the training observations.
- At each non-leaf node, the tree selects a random subset of candidate features.
- Candidate thresholds are evaluated using impurity reduction.
- Tree growth stops when its depth or sample-size conditions prevent further splitting, or when no useful split remains.
- The forest collects predictions from all trained trees.
- Majority voting produces the final classification.
- Observations omitted from individual bootstrap samples contribute to out-of-bag predictions.

Bootstrap sampling creates diversity in the training data. Feature subsampling creates additional diversity in the decision rules. Ensemble voting aggregates these imperfect but varied models.

Random Forests do not require every tree to be accurate independently. They benefit when individual trees provide useful information and their errors are not perfectly correlated.

## Python implementation

The Python script is a self-contained implementation using the standard library.

`Sample` stores numerical features and a categorical target. `TreeNode` represents either a split or a terminal prediction. `DecisionTree` implements recursive CART-style growth, including candidate threshold evaluation, Gini impurity, feature subsampling, minimum leaf size, and majority-class prediction.

The `bootstrap_sample` function returns both the sampled dataset and the indices omitted from it. The second result is used by `RandomForest` to calculate out-of-bag predictions.

The forest maintains separate random-number generators for bootstrap sampling and individual tree construction. This separates random streams and makes a run reproducible when its configuration and input order are unchanged.

The script also reports normalized impurity-based feature importance. The importance contribution for a split is the reduction in impurity multiplied by the number of observations reaching that node. Contributions are accumulated across trees and normalized by their total.

The synthetic dataset is generated from continuous measurements and a nonlinear risk equation. The model is trained on one portion of the dataset and evaluated on a separate test portion. The test accuracy and out-of-bag accuracy answer different questions:

- Test accuracy evaluates predictions on observations withheld from model fitting and forest construction.
- Out-of-bag accuracy evaluates training observations using only trees whose bootstrap samples omitted those observations.

The out-of-bag estimate is useful during model development, but it does not replace an independent test set or an evaluation designed to reflect the intended deployment environment.

## JavaScript implementation

The JavaScript file uses Node.js and the built-in `EventEmitter` class to expose the training lifecycle.

`RandomForest` emits `treeTrained` events as individual trees are fitted and a `trainingComplete` event after training and out-of-bag processing. This separates model execution from reporting or application-level monitoring. A service can listen to those events without embedding logging or interface behavior in the tree algorithm.

The implementation uses a deterministic pseudo-random generator for repeatable bootstrap and feature-subset selection. Its Fisher-Yates shuffle produces candidate feature orderings without replacement, ensuring that one node does not select the same feature multiple times.

The tree stores split information in ordinary objects and recursively traverses child nodes during prediction. The forest aggregates individual tree predictions with majority voting.

The implementation validates training dimensions, finite feature values, tree configuration, and prediction inputs. Invalid prediction vectors fail explicitly instead of returning a classification based on incomplete data.

The event-driven design is useful when training becomes part of a larger monitoring application. Production implementations should also consider cancellation, progress reporting, model serialization, asynchronous job execution, and resource limits.

## C++ case study

The C++ program models equipment inspection as a classification service. Its `InspectionTree` class owns the recursive tree structure, split calculations, and importance statistics. `InspectionForest` coordinates bootstrap creation, independent tree training, out-of-bag voting, and final predictions.

Each tree is stored using `std::unique_ptr<TreeNode>`, which expresses exclusive ownership of its child nodes and allows the tree to be released automatically when its owner is destroyed. Feature vectors and observation collections use standard containers.

The implementation makes the bootstrap procedure explicit. It performs one random draw per training observation, records the selected source indices, and derives the out-of-bag set from the indices that were never selected.

The tree evaluates thresholds between adjacent distinct feature values. It avoids splits that would create a child smaller than the configured minimum leaf size. This prevents a split from being accepted solely because it isolates a small number of observations.

The forest computes feature importance from accumulated impurity reductions and derives out-of-bag predictions only from eligible trees. Invalid feature dimensions and non-finite measurements are rejected.

The case study is intentionally independent of any particular machine-learning framework. Its purpose is to make ownership, sampling, split selection, and ensemble aggregation visible in a compiled implementation. For large datasets, repeated threshold evaluation can be computationally expensive; sorting feature values once per node or using optimized split-search structures can reduce overhead.

## Java implementation

The Java program models training configuration with the immutable `TrainingPolicy` record and inspection measurements with the `Inspection` record. The `MaintenanceDecision` enum constrains the classification domain to valid outcomes.

`DecisionTree` is responsible for recursive split construction and prediction. `RandomForestService` coordinates tree training, bootstrap sampling, out-of-bag predictions, and normalized feature importance.

The use of records provides compact domain types with constructor-level validation. Collections represent bootstrap datasets, feature rankings, and votes. The forest exposes predictions through a service-style API instead of requiring calling code to manipulate individual trees.

The implementation uses a conservative tie rule: when maintenance and normal predictions receive the same number of votes, it returns `MAINTENANCE_REQUIRED`. This rule is an explicit application decision, not a universal Random Forest requirement. It may be appropriate when missed faults are more costly than unnecessary inspections, but the choice should be validated against operational costs.

Java's standard collections and exception mechanisms make this model suitable for demonstrating domain-level validation and separation of responsibilities. A production service would also require defined data contracts, model-version tracking, monitoring, and a tested process for replacing deployed models.

## PostgreSQL implementation

The SQL script records Random Forest experiments and their evaluation artifacts in a relational model.

### Relational schema

| Table | Purpose |
|---|---|
| `random_forest_runs` | Stores model configuration, seed, training fraction, and lifecycle status |
| `inspection_observations` | Stores numerical measurements, actual labels, and train/test membership |
| `forest_trees` | Records each tree belonging to a training run |
| `bootstrap_membership` | Stores the multiplicity of each observation in a tree's bootstrap sample |
| `tree_feature_importance` | Stores per-tree impurity-reduction statistics |
| `oob_predictions` | Stores predictions made by trees that omitted the corresponding training observation |

Primary keys, foreign keys, uniqueness rules, and check constraints enforce basic relational integrity. The run and tree identifiers establish the relationship between configuration, trained trees, and recorded evaluation results.

The bootstrap table uses `draw_count` instead of repeating identical membership rows. A count greater than one represents repeated draws of the same observation.

### Out-of-bag integrity

The `validate_oob_membership` trigger rejects an out-of-bag prediction if the observation was selected for that tree's bootstrap sample. It also checks that the tree and observation belong to the same run and that the observation belongs to the training partition.

These rules protect the stored evaluation evidence against a common mistake: recording a prediction as out-of-bag when the observation was actually used to train the corresponding tree.

The `oob_observation_idx` index supports lookups and aggregation by observation. The observation partition index supports queries that filter a particular run by its training or testing split.

### Metrics and feature ranking

`random_forest_run_metrics` aggregates dataset sizes and out-of-bag results for each run. `random_forest_feature_ranking` sums the impurity reductions associated with each feature and normalizes them within the run.

The final queries inspect class distribution, run metrics, feature rankings, and majority-vote predictions.

The example records are deliberately small enough to inspect manually. Their purpose is to illustrate storage and integrity rules rather than to provide a statistically meaningful benchmark. A real training pipeline must populate these tables from actual model execution.

The script uses PostgreSQL features, including identity columns, filtered aggregates, window functions, PL/pgSQL triggers, and transactional DDL/DML. Run the script in a database where the executing role has permission to create the required objects.

## Feature importance and interpretation

Impurity-based importance measures how much a feature contributes to the splits selected across the fitted trees. Larger normalized values indicate greater accumulated impurity reduction under the training procedure.

This statistic is useful for inspecting model behavior, but it is not proof of causation. Correlated features can share or displace importance. Features with many possible split points can also receive inflated importance in some settings.

For equipment maintenance, importance should be interpreted alongside domain knowledge, measurement quality, class-specific error rates, and the operational cost of missed failures. Permutation importance and carefully designed feature-ablation experiments can provide complementary evidence.

## Out-of-bag evaluation and data leakage

Out-of-bag evaluation uses the bootstrap mechanism to estimate generalization without reserving a separate validation set for every tree. Each observation receives predictions from trees for which it was not sampled.

This estimate can be misleading if the data contains leakage, duplicated entities across partitions, temporal dependence, or measurements collected after the outcome was determined. For example, random splitting is inappropriate when the deployment task is to predict future equipment failures and later observations are closely related to earlier records from the same asset.

Evaluation design must reflect how predictions will be used. A temporal split or asset-level group split may be more realistic than a random row split.

## Limitations and practical considerations

- **Model complexity:** Training many deep trees increases computation and memory usage. Tree count and depth should be selected based on measured validation performance and resource constraints.
- **Class imbalance:** Overall accuracy may conceal a high rate of missed maintenance events. Precision, recall, a confusion matrix, and the costs of false negatives and false positives should be evaluated.
- **Feature importance:** Impurity importance is descriptive, not causal. Correlated measurements and features with many possible thresholds complicate interpretation.
- **Calibration:** The fraction of trees voting for a class is not automatically a calibrated probability. Probability calibration requires separate validation.
- **Reproducibility:** Fixed random seeds make experiments repeatable under the same implementation and input order. Changes to the algorithm or runtime may still alter results.
- **Operational use:** Maintenance recommendations should be checked against sensor reliability, asset-specific requirements, safety rules, and the cost of an incorrect prediction.

A Random Forest provides a flexible baseline for tabular classification because it combines bootstrap diversity, randomized split candidates, and ensemble voting. Its reliability depends on sound data preparation, appropriate validation, careful interpretation, and a deployment policy that accounts for the consequences of classification errors.
