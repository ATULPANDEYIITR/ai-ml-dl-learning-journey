# Decision Tree Optimization: Depth, Pruning, Overfitting, and Feature Importance

## Technical scope

Decision trees classify observations by repeatedly dividing a dataset according to feature-based rules. They are interpretable because a prediction can be traced through a sequence of threshold comparisons. Their flexibility also creates a central optimization problem: a tree can fit its training data extremely well while making unreliable predictions on new observations.

This project investigates four connected concerns:

- **Depth** controls how many successive decisions a tree can make.
- **Pruning** reduces unnecessary branches, either during training or after a larger tree has been constructed.
- **Overfitting** occurs when a model captures accidental details of its training data rather than stable relationships.
- **Feature importance** estimates how much individual features contribute to the fitted tree's impurity reduction.

The implementations use industrial equipment maintenance as the main application. Measurements such as vibration, operating temperature, service hours, and previous faults are used to classify equipment as normal or requiring inspection.

The project includes standalone implementations in Python, JavaScript, C++17, and Java 17, together with a PostgreSQL experiment registry. The programs use synthetic observations so that their behavior can be examined without access to private operational records.

## How a classification tree makes decisions

A classification tree starts with a root node containing the training observations. For a numeric feature, a candidate split has the form

\[
x_j \leq t
\]

where \(x_j\) is a feature and \(t\) is a threshold. Observations satisfying the condition enter the left child; the remaining observations enter the right child.

Each child repeats the process until a stopping condition is met. A terminal node, or leaf, predicts the majority class among the observations reaching that node. Deterministic tie-breaking makes the implementations reproducible.

### Gini impurity

For a node containing class proportions \(p_1,\ldots,p_K\), Gini impurity is

\[
G = 1-\sum_{k=1}^{K}p_k^2
\]

A node containing only one class has zero impurity. A node containing a mixture of classes has positive impurity.

The quality of a candidate split is measured by the reduction in impurity:

\[
\Delta G =
G_{\text{parent}}
-\frac{n_L}{n}G_L
-\frac{n_R}{n}G_R
\]

Here, \(n\) is the number of observations at the parent, while \(n_L\) and \(n_R\) describe the sizes of its children.

The split search chooses a candidate with a high impurity reduction. A minimum impurity-decrease threshold can reject improvements that are too small to justify another decision.

Gini impurity is a splitting criterion, not an evaluation metric. A tree with low training impurity is not necessarily a model that generalizes well.

### Entropy as an alternative

Entropy measures class uncertainty using logarithms:

\[
H = -\sum_{k=1}^{K}p_k\log_2(p_k)
\]

A split can be evaluated through information gain, defined as the parent entropy minus the weighted entropy of the children.

Gini impurity and entropy often produce similar trees, but they need not choose identical thresholds. The Python implementation supports both criteria. The other implementations use Gini impurity to keep their split-search and pruning mechanisms focused.

## Depth and the bias-variance trade-off

Tree depth is the number of edges between the root and the deepest leaf. A root-only tree has depth zero.

A shallow tree may be unable to represent useful interactions among measurements. For example, elevated vibration may indicate a serious problem only when operating temperature or previous fault history is also high. A tree with insufficient depth may miss that combination.

An unrestricted tree can create many narrow rules, including rules that classify individual training observations correctly by chance. Such a tree may have nearly perfect training accuracy but worse validation accuracy.

The relevant relationship is not that deeper trees are always worse. It is that added complexity must be justified by performance on data not used to fit the model.

| Control | Mechanism | Trade-off |
|---|---|---|
| Maximum depth | Stops splitting after a specified depth | Limits interactions and model size |
| Minimum split size | Prevents small nodes from creating new branches | May leave useful local patterns unresolved |
| Minimum leaf size | Requires enough observations in each child | Reduces fragile decisions based on very few records |
| Minimum impurity decrease | Rejects splits with insufficient improvement | Can omit weak but useful predictive signals |
| Post-pruning | Removes branches after a larger tree is trained | Can simplify an over-complex tree but may remove useful structure |

Depth is a useful first constraint, but it does not fully determine complexity. A shallow tree with many wide branches may still contain more leaves than another tree of the same depth. Leaf count, node count, validation performance, and the cost of incorrect decisions should be considered together.

## Pre-pruning and cost-complexity pruning

### Pre-pruning

Pre-pruning places restrictions on tree construction. The model stops splitting when it reaches the configured maximum depth, when too few observations remain, or when the best available split fails the impurity-improvement requirement.

These checks reduce training time and prevent some unstable branches from being created. They can also stop a tree too early. A locally weak split may be necessary before a later split can expose a useful pattern.

### Post-pruning

Post-pruning begins with a larger fitted tree and replaces selected subtrees with leaves.

Cost-complexity pruning balances empirical classification impurity against tree size:

\[
R_{\alpha}(T)=R(T)+\alpha |L(T)|
\]

In this expression:

- \(T\) is the tree.
- \(R(T)\) is its empirical risk.
- \(|L(T)|\) is the number of leaves.
- \(\alpha\) penalizes model complexity.

A larger penalty favors simpler trees. The cost-complexity method evaluates whether a subtree improves the fit enough to justify its extra leaves.

The code implements a weakest-link style calculation. For an internal node, it compares the risk of replacing the entire subtree with the risk of retaining the subtree. The effective pruning value is the increase in leaf risk divided by the number of leaves removed.

The Python, JavaScript, C++, and Java implementations use sample-count-weighted Gini impurity as their pruning risk. This makes the risk calculation explicit and consistent within each implementation. It is an educational empirical-risk implementation rather than a full replacement for every detail of a mature machine-learning library's pruning path.

A single pruning penalty is not universally optimal. The appropriate value depends on the data, class imbalance, and the operational cost of mistakes. It should be selected on validation data rather than by repeatedly inspecting the final test results.

## Overfitting and reliable evaluation

Training accuracy measures performance on the records used to fit the tree. Validation accuracy measures performance on separate records used to compare model configurations. Test accuracy is reserved for the final assessment after optimization choices have been made.

A gap between training and validation accuracy can indicate overfitting. It is not conclusive on its own: a small validation sample, class imbalance, or a distribution shift can also produce unstable estimates.

Equipment failure classification has an important asymmetry. A false negative means that equipment requiring inspection is classified as normal. A false positive means that equipment is sent for inspection even though the observed label is normal.

In a safety-sensitive setting, accuracy alone can conceal operational risk. Recall for the inspection-required class, false-negative counts, precision, inspection capacity, and the consequences of missed failures should be examined alongside accuracy.

The synthetic datasets are useful for testing implementations and comparing settings. They do not establish how well a model would perform on real equipment, where sensor calibration, maintenance practices, missing readings, and changing operating conditions can affect the data.

## Feature importance and interpretation

The implementations calculate impurity-based feature importance by accumulating the impurity reduction attributed to each feature across the internal nodes where it is used.

For a node \(j\), the unnormalized contribution is proportional to

\[
n_j\Delta G_j
\]

where \(n_j\) is the number of observations reaching the node. Contributions are summed by feature and normalized so that the importances sum to one when the tree makes a positive total impurity reduction.

This weighting means that an improvement near the root can contribute substantially because it affects many observations. A feature used only in a small terminal region may have a smaller aggregate importance.

Feature importance is model-specific. It depends on the training sample, available features, splitting criterion, and pruning configuration. Correlated measurements may substitute for one another, distributing importance in ways that do not reflect their independent predictive value.

A high importance score does not establish causality. In the maintenance scenario, vibration may be highly predictive without proving that vibration alone causes a failure. Operational decisions should combine model evidence with engineering knowledge and data-quality checks.

When comparing feature importance before and after pruning, remember that pruning removes some split contributions. The resulting importance distribution can change even when the underlying dataset has not changed.

## Python implementation

The Python program builds a binary classifier from standard-library data structures. Each observation is represented by an immutable `Sample` containing numeric features and a class label. The mutable `Node` class stores the predicted class, class counts, impurity, depth, split feature, threshold, and child references.

`DecisionTreeClassifier` supports configurable depth, minimum split size, minimum leaf size, minimum impurity decrease, and Gini or entropy criteria.

The split search sorts the distinct values for each feature and examines midpoints between adjacent values. For numeric threshold splits, those midpoints represent the meaningful changes in partition membership without evaluating arbitrary thresholds that divide the data identically.

The implementation includes:

- Dataset validation for empty input, inconsistent feature counts, empty labels, and non-finite numeric values.
- Majority-class prediction with deterministic tie-breaking.
- Training and prediction methods that reject invalid input.
- Pre-pruning through depth, sample-size, and impurity-improvement controls.
- Cost-complexity pruning with refreshed node depths and feature importances.
- A readable tree-rule representation for inspecting actual threshold decisions.
- A seeded synthetic maintenance dataset and stratified train-test partition.
- Comparative reports containing training accuracy, test accuracy, tree depth, node count, leaf count, and feature importance.

The example trains an unrestricted model, a constrained model, and a post-pruned model. These configurations illustrate different complexity controls; the selected settings are not claimed to be universally optimal.

Run the file with Python 3.10 or later:

`python decision_tree_optimization.py`

No external package is required.

## JavaScript implementation

The JavaScript file emphasizes the lifecycle of model selection in Node.js.

The `DecisionTree` class encapsulates its tree-building method in a private method, validates its configuration at construction, and exposes prediction, statistics, and pruning operations. The model rejects non-finite input rather than allowing a `NaN` comparison to route unpredictably through threshold rules.

The experiment uses a deterministic pseudo-random generator to produce synthetic observations. It evaluates several maximum-depth configurations and reports training accuracy, validation accuracy, node count, leaf count, and depth.

Candidate evaluation is asynchronous. It yields control to the Node.js event loop between configurations, showing how a sequence of model evaluations can coexist with other event-loop work. The underlying training algorithm remains synchronous; yielding does not make the split search itself parallel.

The selection rule favors validation accuracy and uses node count to break ties. This makes the preference for simpler models explicit without claiming that the smallest tree must always be best.

The JavaScript post-pruning implementation also recalculates depth and statistics after branches are removed. Its feature importance values are calculated from the current tree rather than cached from an earlier model state.

Run the file with Node.js 18 or later:

`node decision_tree_optimization.js`

No npm package is required.

## C++ case study

The C++ program models an equipment-inspection decision service. Its primary design concern is the relationship between the tree's ownership structure, prediction rules, and optimization operations.

`Observation` stores a feature vector and class label. `Split` represents a candidate partition and owns its child datasets. `TreeNode` stores the node's prediction and split metadata while using `std::unique_ptr` to express exclusive ownership of child nodes.

Exclusive ownership is appropriate because each child belongs to one location in the tree. When pruning removes a subtree, resetting its owning pointer releases the subtree's memory automatically.

The `DecisionTree` class separates training, split selection, prediction, pruning, and inspection. Dataset validation rejects inconsistent feature widths and non-finite measurements. Prediction follows threshold comparisons until it reaches a leaf.

The program constructs an unrestricted tree, a depth-limited tree, and a post-pruned tree. It evaluates their accuracy on a held-out partition and reports their structural complexity. The final inspection decision demonstrates how a fitted tree can be applied to incoming telemetry.

The feature-importance calculation traverses the tree iteratively using a vector of node pointers. This avoids introducing an additional recursive traversal for that particular operation. Tree construction and pruning remain recursive, so extremely deep unrestricted trees can still encounter stack-depth limitations.

Compile and run:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic decision_tree_optimization.cpp -o decision_tree_optimization`

`./decision_tree_optimization`

The synthetic data and the partitioning strategy are intended for an executable case study. A production evaluation should also consider stratification, repeated validation, class imbalance, and the cost of missed failures.

## Java implementation

The Java program uses explicit domain types to make the classification workflow easier to validate and maintain.

`EquipmentState` is an enum rather than an unrestricted string. `AssetObservation` is a record that validates and defensively copies its measurement array. This prevents callers from changing an observation's feature values through the array originally passed to the constructor.

`TrainingPolicy` is an immutable record. Its `OptionalInt` maximum depth distinguishes a configured depth limit from an unrestricted tree without using a special numeric sentinel.

`TreeNode` represents mutable training state, while `TreeModel` provides the operations through which the model is fitted, evaluated, pruned, and inspected. The split search is isolated in a `SplitCandidate` record containing the threshold, impurity reduction, and child observations.

The Java implementation demonstrates several enterprise-oriented design decisions:

- Invalid model policies are rejected when constructed, rather than failing during training.
- Model fitting and prediction have separate preconditions.
- Prediction rejects measurement arrays of the wrong length and non-finite values.
- Pruning updates the tree structure, node depths, and feature-importance values together.
- Model reports expose accuracy and structural complexity as separate properties.
- Class labels use an enum, reducing errors caused by inconsistent text values.

The example uses a deterministic random seed and a held-out validation partition. Its purpose is to demonstrate explicit domain modeling and the management of model state, not to provide a distributed training framework.

Compile and run:

`javac DecisionTreeOptimization.java`

`java DecisionTreeOptimization`

The program requires Java 17 or later.

## PostgreSQL experiment registry

The SQL file models the experimental evidence required to compare decision-tree configurations over time. It is not a SQL implementation of the split-search algorithm. The database records model definitions, optimization policies, dataset versions, training runs, evaluation metrics, and feature-importance results.

### Relational model

| Table | Responsibility |
|---|---|
| `model_registry` | Identifies classification tasks, target columns, and splitting criteria |
| `decision_tree_policies` | Stores maximum depth, sample-size controls, impurity thresholds, and pruning settings |
| `dataset_versions` | Records dataset identity, target, feature count, row count, label distribution, and fingerprint |
| `model_runs` | Tracks execution status, random seed, measured depth, node count, leaf count, timestamps, and failures |
| `evaluation_metrics` | Stores partition-specific accuracy, precision, recall, F1, and error counts |
| `feature_importance` | Stores normalized feature importance and the number of splits using each feature |

Foreign keys preserve relationships between model runs, policies, and datasets. Unique constraints prevent duplicate model names, policy names, dataset versions, and partition metrics for the same run.

Check constraints enforce valid parameter ranges and metric bounds. A completed run must contain its structural measurements and finish timestamp. A failed run must include an error message. These checks prevent several incomplete states from being stored as valid results.

The partial unique index `one_active_model_per_task` permits at most one active registry entry for a task while allowing historical model definitions to remain available.

### Triggers and integrity checks

The `validate_completed_tree_run` trigger verifies completion metadata and rejects runs whose measured depth exceeds the configured maximum. It also validates basic consistency between the node count and leaf count.

The feature-importance trigger rejects totals greater than one beyond a small numerical tolerance. It deliberately permits temporary totals below one while individual feature records are being inserted. A completed import process should still verify that its full importance distribution is normalized.

These triggers enforce basic registry integrity. They do not prove that recorded metrics were honestly computed, that the correct data partition was used, or that the underlying tree implementation is correct. Those guarantees require controlled training and evaluation procedures outside the database.

### Experiment queries

The experiment report view joins model definitions, policies, datasets, runs, and evaluation metrics into a common reporting interface.

The comparison query keeps training and validation accuracy separate. The feature-importance query ranks features within each run using a window function. The selection query ranks completed validation runs by accuracy, then by leaf count and false-negative count.

The example records are illustrative registry fixtures rather than measured results from the executable language implementations. They show the structure of a model-comparison workflow; they must not be interpreted as cross-language benchmark results.

Run the SQL script in a PostgreSQL database using a role permitted to create tables, indexes, functions, triggers, and views. The script drops its named demonstration objects before recreating them, so it should be run only in a database where replacing those objects is acceptable.

## Choosing a model for operational use

Tree optimization should follow the intended decision, not a single numerical score.

For an inspection-prioritization system, false negatives may carry a much higher cost than false positives. A model with slightly lower accuracy may be preferable if it identifies more genuinely dangerous assets without exceeding inspection capacity.

A defensible evaluation should therefore examine:

- **Validation performance:** Compare candidate depths and pruning settings without using the final test set to select them.
- **Complexity:** Record depth, node count, and leaf count so that performance gains can be evaluated against maintenance and explanation costs.
- **Class-specific errors:** Measure false negatives and false positives for the inspection-required class separately.
- **Data quality:** Check measurement units, sensor faults, missing observations, and inconsistent labels before fitting the tree.
- **Stability:** Repeat evaluation across suitable splits or time periods to determine whether selected thresholds and feature rankings are fragile.
- **Deployment drift:** Reassess the model when equipment, operating conditions, sensors, or maintenance practices change.

A tree that performs well on randomly partitioned historical data can still fail when used on future time periods. When the deployment predicts future failures, a time-based validation design may better represent the real prediction task.

## Computational considerations

Let \(n\) denote the number of observations and \(d\) the number of features. The implementations enumerate candidate thresholds derived from distinct values for each feature. At a node containing \(m\) observations, sorting feature values costs approximately \(O(d\,m\log m)\). Repeatedly partitioning the observations and recomputing class counts adds further work.

The straightforward implementations favor clarity over optimized split-search performance. They do not maintain incrementally updated class counts while scanning sorted thresholds, and they rebuild child observation collections at every candidate. Consequently, their actual running time can be substantially higher than that of optimized production tree libraries.

Tree memory usage depends on the observations copied into child partitions and on the number of nodes. A highly unbalanced tree can also become deep enough to stress recursive construction, prediction, or pruning.

For large datasets, more efficient implementations can sort each feature once, maintain cumulative class counts, and avoid repeatedly copying full observations. The resulting engineering work should preserve the same validation, stopping, and evaluation semantics so that performance improvements do not silently change model behavior.

## Limitations and interpretation

These implementations use numeric threshold splits and majority-class leaves. They do not implement categorical split strategies, missing-value routing, probability calibration, class-weighted splitting, or a complete production-grade model-selection pipeline.

The synthetic labels are generated from a rule containing noise. This provides a controlled environment for examining how complexity affects performance, but it can make the learned feature relationships easier to recover than those found in real operational systems.

Impurity-based feature importance can favor features with many distinct candidate split points and can behave unpredictably with correlated predictors. It should not be treated as an independent causal analysis.

The pruning methods use the implementation's weighted Gini empirical risk. Pruning parameters and reported importance values may therefore differ from those produced by other libraries, even when the broad optimization principles are the same.

For consequential inspection decisions, tree outputs should be validated against actual outcomes, reviewed with domain experts, and monitored after deployment. A readable decision path is useful evidence about the model's behavior, but interpretability does not guarantee correctness.
