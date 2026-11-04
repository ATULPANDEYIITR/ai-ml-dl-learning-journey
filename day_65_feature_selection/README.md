# Feature Selection

A practical implementation of feature-selection methods across Python, JavaScript, C++17, Java 17, and PostgreSQL.

The project demonstrates how predictive datasets can be reduced to informative feature subsets while controlling redundancy, model complexity, computational cost, and information leakage.

## 1. Technical Objectives

Feature selection identifies a subset of input variables that contributes useful information to a predictive task.

A dataset containing hundreds of features may include:

- Relevant variables that contain predictive information.
- Irrelevant variables that contribute little to prediction.
- Redundant variables that represent similar information.
- Constant variables that provide no discrimination.
- Sensitive variables that require governance restrictions.
- Noisy variables that can reduce model reliability.

The objective is not simply to minimize the number of columns. It is to retain a useful subset that supports predictive performance, interpretability, computational efficiency, and responsible data processing.

## 2. Feature Selection and Feature Extraction

These techniques address different problems.

| Characteristic | Feature Selection | Feature Extraction |
|---|---|---|
| Main operation | Retains existing variables | Creates transformed variables |
| Original meaning | Usually preserved | May become less interpretable |
| Output | Subset of original features | New feature representation |
| Example | Selecting income and age | Principal Component Analysis |
| Main objective | Remove unnecessary variables | Represent information in fewer dimensions |

Feature selection does not automatically transform the values of retained variables.

## 3. Four Feature Selection Methods

### 3.1 Filter Methods

Filter methods evaluate feature characteristics independently of a predictive model.

They are generally computationally efficient and suitable for initial screening of high-dimensional datasets.

The Python and C++ implementations use variance, mutual information, and Pearson correlation. The JavaScript and Java implementations provide corresponding ranking and screening mechanisms.

**Variance threshold**

Variance measures how much a feature changes across observations.

For a feature \(X\) with \(n\) observations:

\[
Var(X)=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

A feature with zero variance has the same value in every observation.

Such a feature cannot distinguish between observations and is normally removed.

The implementations include a configurable minimum variance threshold.

**Pearson correlation**

Pearson correlation measures the strength and direction of a linear relationship between two numerical variables.

\[
r_{XY}=
\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
\]

Its value ranges from -1 to +1.

| Correlation | Interpretation |
|---|---|
| +1 | Perfect positive linear relationship |
| 0 | No linear relationship |
| -1 | Perfect negative linear relationship |

When two features have a correlation magnitude above the configured threshold, one may be removed to reduce redundancy.

Correlation does not establish causation and does not identify every nonlinear relationship.

**Mutual information**

Mutual information measures how much information one variable provides about another.

\[
I(X;Y)=
\sum_{x,y}p(x,y)
\log_2\left(
\frac{p(x,y)}{p(x)p(y)}
\right)
\]

A higher score indicates a stronger statistical dependency.

Unlike Pearson correlation, mutual information can capture nonlinear statistical dependencies.

The implementations use discretization to estimate mutual information for numerical features. The result depends on the selected binning strategy.

**ANOVA F statistic**

The Python implementation also calculates an ANOVA F statistic for binary classification.

It compares variation between classes with variation within classes.

A larger F statistic indicates stronger class separation under the assumptions of the test.

### 3.2 Wrapper Methods

Wrapper methods evaluate feature subsets by repeatedly training or evaluating a predictive model.

The selection process depends on the model and evaluation objective.

The Python implementation uses sequential forward selection with logistic regression.

The JavaScript, C++17, and Java 17 implementations use nearest-centroid classification as their subset evaluation mechanism.

**Sequential Forward Selection**

The process is:

1. Start with an empty feature subset.
2. Evaluate each remaining feature individually.
3. Select the feature producing the highest evaluation score.
4. Add that feature to the current subset.
5. Evaluate the remaining candidates with the selected subset.
6. Repeat until the feature-count limit is reached.

The selection rule can be represented as:

\[
S_{t+1}=S_t\cup
\left\{
\arg\max_{j\notin S_t}
Score(S_t\cup\{j\})
\right\}
\]

The wrapper implementation records the selected features and their evaluation history.

**Computational implications**

For \(p\) original features and a maximum subset size of \(k\), forward selection performs approximately:

\[
O(pk)
\]

candidate subset evaluations.

The actual computational cost depends on the predictive model and dataset size.

Wrapper methods can be expensive because they repeatedly evaluate different feature combinations.

The Python implementation uses a deterministic holdout split for reproducibility. A production experiment should use cross-validation or nested cross-validation and reserve an independent test set for final evaluation.

### 3.3 Embedded Methods

Embedded methods perform feature selection as part of model training.

They combine predictive modeling with feature-selection decisions.

The Python implementation uses L1-regularized logistic regression.

The regularized objective is:

\[
J(w,b)=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
+
\lambda\sum_{j=1}^{p}|w_j|
\]

Where:

- \(w_j\) represents a model coefficient.
- \(b\) represents the intercept.
- \(p_i\) represents the predicted probability.
- \(\lambda\) controls regularization strength.

L1 regularization encourages some coefficients to become exactly or approximately zero.

Features with sufficiently small coefficients can be excluded.

The implementation standardizes feature values during model fitting to reduce the effect of differences in measurement scales.

The JavaScript, C++17, and Java 17 programs implement a separate coefficient-scoring model with an L1-style shrinkage threshold. These are educational embedded-style selection mechanisms, not full L1 logistic regression implementations.

**Advantages**

- Selection is integrated with predictive modeling.
- It avoids evaluating every possible feature subset.
- Regularization can reduce overfitting.
- Coefficients provide an interpretable importance measure for suitable models.

**Limitations**

- Results depend on the selected estimator.
- Correlated features can produce unstable coefficient assignments.
- Regularization parameters require appropriate validation.
- Coefficient magnitude alone does not establish causal importance.

### 3.4 Recursive Feature Elimination

Recursive Feature Elimination repeatedly trains a model and removes the least important feature.

The workflow is:

1. Begin with all eligible features.
2. Fit a predictive estimator.
3. Calculate feature importance.
4. Identify the least important feature.
5. Remove that feature.
6. Refit the estimator.
7. Continue until the target feature count is reached.

The process is:

\[
F_0=\{f_1,f_2,\ldots,f_p\}
\]

\[
F_{t+1}=F_t-\{f_{\text{least important}}\}
\]

The Python implementation uses absolute logistic-regression coefficients.

The other implementations use standardized class-separation scores as an importance approximation.

Each implementation records the features removed during the elimination process.

**Advantages**

- Evaluates feature importance repeatedly.
- Accounts for changes in model behavior after feature removal.
- Produces an explicit elimination history.
- Supports a predefined final feature count.

**Limitations**

- Repeated model fitting increases computational cost.
- Early removal decisions can affect later subsets.
- Results depend on the importance measure.
- The method can be unstable when features are strongly correlated.

## 4. Comparison of Selection Methods

| Property | Filter | Wrapper | Embedded | RFE |
|---|---|---|---|---|
| Uses a predictive model | No | Yes | Yes | Yes |
| Evaluates feature subsets | Usually no | Yes | During training | Iteratively |
| Computational cost | Generally low | Generally high | Model-dependent | Model-dependent |
| Captures feature interactions | Limited | Model-dependent | Model-dependent | Model-dependent |
| Model dependency | Low | High | High | High |
| Typical output | Ranked features | Best evaluated subset | Nonzero coefficients | Final retained subset |
| Common application | Initial screening | Model-specific optimization | Regularized prediction | Model-driven reduction |

The four approaches are not interchangeable.

A practical pipeline can combine them:

Filter screening → Embedded selection → RFE → Independent evaluation.

Wrapper selection can be used when subset-level model performance justifies its computational expense.

## 5. Dataset Design

All five implementations use a synthetic customer-risk classification scenario.

The target variable represents a binary classification outcome.

| Feature | Description | Expected characteristics |
|---|---|---|
| income | Annual customer income | Predictive |
| age | Customer age | Predictive |
| account_balance | Account balance | Predictive |
| transaction_count | Monthly transaction count | Mostly noise |
| income_duplicate | Noisy copy of income | Highly redundant |
| constant_feature | Constant numerical value | Zero variance |

The generated target is influenced by income, age, and account balance.

A logistic probability function determines the binary target:

\[
P(Y=1|X)=
\frac{1}{1+e^{-z}}
\]

Where \(z\) is a weighted combination of the predictive features.

The synthetic dataset is deterministic for the configured random seed.

This makes repeated execution useful for comparing implementation behavior.

## 6. Python Implementation

The Python script is a complete feature-selection laboratory.

**Main components**

| Component | Responsibility |
|---|---|
| `Dataset` | Validates feature and observation data |
| `SelectionResult` | Stores selection output |
| `SelectionMethod` | Defines selection categories |
| `filter_selection()` | Performs statistical screening |
| `sequential_forward_selection()` | Evaluates feature subsets |
| `fit_logistic()` | Trains logistic regression |
| `embedded_selection()` | Applies L1 coefficient selection |
| `recursive_feature_elimination()` | Iteratively removes features |
| `export_results()` | Exports JSON and CSV reports |
| `run_tests()` | Checks statistical and validation behavior |

The logistic-regression implementation uses batch gradient descent.

It includes:

- Probability calculation using the sigmoid function.
- Gradient-based coefficient updates.
- L1 regularization.
- Feature standardization.
- Deterministic training and validation splitting.

Generated files:

- `feature_selection_output/selection_report.json`
- `feature_selection_output/selected_features.csv`

The script uses only Python standard-library modules.

## 7. JavaScript Implementation

The JavaScript implementation provides an event-driven selection engine.

Its architecture separates dataset validation, selection policy, evaluation, execution, and audit logging.

**Main components**

| Component | Responsibility |
|---|---|
| `Dataset` | Validates structured observations |
| `SelectionPolicy` | Defines selection constraints |
| `FeatureSelectionEngine` | Executes selection algorithms |
| `EventEmitter` | Publishes experiment events |
| `runExperiment()` | Executes methods asynchronously |
| `saveReport()` | Writes experiment results |

The event system generates records for:

- Experiment start.
- Filter completion.
- Wrapper completion.
- Embedded completion.
- RFE completion.
- Experiment failure.
- Experiment completion.

`Map` provides efficient frequency tracking for discrete statistical calculations.

`Set` manages remaining candidate features during wrapper selection.

The implementation uses Node.js filesystem and event modules without external npm dependencies.

Generated report:

`feature_selection_output/feature_selection_report.json`

## 8. C++17 Implementation

The C++17 program focuses on domain modeling, statistical computation, and algorithmic efficiency.

**Main components**

| Component | Responsibility |
|---|---|
| `Dataset` | Owns feature definitions and observations |
| `FeatureScore` | Represents statistical ranking information |
| `SelectionResult` | Stores selected features and scores |
| `GovernanceEngine` | Executes selection strategies |
| `filterSelection()` | Performs filter selection |
| `wrapperSelection()` | Evaluates candidate subsets |
| `embeddedSelection()` | Calculates regularized importance scores |
| `recursiveFeatureElimination()` | Removes weak features |

The implementation uses:

- `vector` for observations and selected features.
- `map` for named scores.
- `set` for candidate management.
- STL sorting and accumulation algorithms.
- Exceptions for invalid inputs and execution failures.

The wrapper evaluator uses nearest-centroid classification.

The program is self-contained and requires no external libraries.

## 9. Java 17 Implementation

The Java program represents feature selection as an enterprise-oriented experiment workflow.

It separates immutable domain records, statistical operations, policy configuration, experiment execution, and audit history.

**Domain types**

| Type | Responsibility |
|---|---|
| `FeatureDefinition` | Describes a feature and its sensitivity |
| `SelectionPolicy` | Defines permitted selection behavior |
| `FeatureMetric` | Stores calculated feature metrics |
| `EliminationEvent` | Represents an RFE decision |
| `SelectionResult` | Stores immutable selection results |
| `AuditEvent` | Records experiment operations |
| `ExperimentStatus` | Represents workflow state |

`SelectionService` owns the four selection mechanisms.

`ExperimentManager` controls experiment state transitions.

The lifecycle is:

`CREATED → VALIDATING → RUNNING → COMPLETED`

A runtime failure changes the state to `FAILED`.

The implementation uses Java records, collections, streams, defensive copying, and custom runtime exceptions.

Feature sensitivity is represented explicitly in the domain model.

The sample policy includes configurable feature-count, variance, correlation, and coefficient thresholds.

## 10. PostgreSQL Implementation

The PostgreSQL script models a relational feature-selection management system.

It stores dataset definitions, feature characteristics, selection policies, experiments, metrics, decisions, RFE history, evaluation results, and audit records.

### Database relationship

`Datasets`

→ `Dataset Features`

→ `Selection Policies`

→ `Selection Experiments`

→ `Feature Metrics`

→ `Feature Selection Decisions`

→ `RFE Elimination History`

→ `Experiment Evaluations`

→ `Selection Audit Log`

### Core tables

| Table | Purpose |
|---|---|
| `datasets` | Dataset-level information |
| `dataset_features` | Feature definitions and properties |
| `selection_policies` | Selection thresholds and governance rules |
| `selection_experiments` | Individual selection executions |
| `feature_metrics` | Statistical and model-based scores |
| `feature_selection_decisions` | Selected, rejected, and eliminated features |
| `rfe_elimination_history` | Iterative removal records |
| `experiment_evaluations` | Predictive performance measurements |
| `selection_audit_log` | Experiment and decision audit events |

### Integrity controls

The schema includes:

- Primary keys and foreign keys.
- Unique feature names within datasets.
- Numerical range constraints.
- Feature-to-experiment dataset consistency.
- Sensitive-feature restrictions.
- Target-feature exclusion.
- Valid experiment state transitions.
- Selection decision audit triggers.

The `validate_experiment_feature()` trigger prevents a feature from being associated with an experiment belonging to another dataset.

It also blocks target columns and sensitive features when the governing policy prohibits their use.

### Eligibility function

`check_experiment_eligibility()` evaluates:

- Experiment existence.
- Experiment completion state.
- Requested feature count.
- Policy maximum.
- Actual selected feature count.

It returns a structured eligibility decision and explanation.

`complete_experiment()` provides controlled completion using the eligibility function.

### Reporting

The SQL script creates three reporting views:

- `v_selected_features`
- `v_experiment_summary`
- `v_feature_method_comparison`

These support feature retrieval, method comparison, performance aggregation, and experiment monitoring.

The transaction ensures that the sample schema, records, and workflow operations are committed together.

The sample evaluation measurements are explicitly illustrative and are not results from actual model training.

## 11. Validation and Edge Cases

The implementations address several common data-quality and execution failures.

| Condition | Handling |
|---|---|
| Empty feature definitions | Reject dataset |
| Insufficient observations | Reject dataset |
| Duplicate feature names | Reject dataset |
| Inconsistent row dimensions | Reject dataset |
| Non-finite values | Reject dataset |
| Missing target class | Reject dataset |
| Constant feature | Exclude during variance screening |
| Highly correlated features | Apply redundancy threshold |
| Invalid feature count | Reject configuration |
| Negative regularization | Reject configuration |
| Empty selected subset | Raise selection error |
| Invalid RFE target | Reject execution |
| Experiment already running | Reject concurrent execution |
| Sensitive feature | Apply governance policy |

The scripts use deterministic seeds to make synthetic data reproducible.

## 12. Data Leakage and Evaluation Integrity

Feature selection must be performed using training data only.

Using validation or test labels during feature selection can introduce information leakage.

A correct evaluation workflow is:

`Raw Dataset`

→ `Training Set`

→ `Feature Selection on Training Data`

→ `Model Training`

→ `Validation`

→ `Parameter Adjustment`

→ `Final Evaluation on Untouched Test Set`

For cross-validation, feature selection must be repeated inside each training fold.

Performing feature selection once on the complete dataset before cross-validation can inflate estimated performance.

The Python implementation separates training and validation observations for wrapper evaluation.

The other educational implementations use full-dataset statistical or nearest-centroid scoring for their demonstrations. Their scores should not be interpreted as unbiased estimates of generalization performance.

A production system should also account for class imbalance, missing-value treatment, scaling, outliers, and distribution changes.

## 13. Computational Complexity

Let:

- \(n\) represent the number of observations.
- \(p\) represent the number of original features.
- \(k\) represent the requested selected feature count.
- \(b\) represent the number of discretization bins.

| Operation | Approximate complexity |
|---|---|
| Mean and variance | \(O(n)\) per feature |
| Pearson correlation | \(O(n)\) per feature pair |
| Mutual information | \(O(n\log n)\) per feature due to sorting |
| Filter ranking | \(O(pn\log n)\), excluding pairwise redundancy checks |
| Correlation pruning | \(O(p^2n)\) worst case |
| Forward selection | \(O(pk \cdot C)\) |
| RFE | \(O((p-k)\cdot C)\) |
| Feature storage | \(O(np)\) |

Here, \(C\) is the computational cost of fitting and evaluating the selected estimator.

The actual cost of wrapper and RFE methods depends heavily on model complexity.

Filter methods are generally preferable for initial reduction when the number of variables is very large.

Wrapper methods are more suitable when model-specific subset performance is important and computational resources permit repeated evaluation.

## 14. Security and Governance Considerations

Feature selection can expose sensitive information when datasets contain personal, financial, or organizational records.

Important controls include:

- Restricting dataset access through role-based permissions.
- Excluding prohibited sensitive variables.
- Recording selection decisions.
- Protecting exported experiment reports.
- Validating input data before model fitting.
- Avoiding unnecessary copies of sensitive datasets.
- Preventing unauthorized modification of selection policies.
- Separating experiment execution permissions from policy administration.

The PostgreSQL schema provides sensitivity metadata, policy restrictions, validation triggers, and audit records.

The metadata controls demonstrate governance enforcement. Production deployments also require database roles, grants, encryption, backup controls, and appropriate retention policies.

## 15. Practical Method Selection

| Scenario | Suitable approach |
|---|---|
| Thousands of candidate variables | Filter methods |
| Removal of constant variables | Variance threshold |
| Removal of redundant variables | Correlation filtering |
| Detection of nonlinear dependency | Mutual information |
| Optimization of a specific predictive model | Wrapper methods |
| Regularized predictive modeling | Embedded methods |
| Iterative model-driven reduction | RFE |
| Strict computation budget | Filter methods |
| Model interpretability | Embedded methods with suitable estimators |
| Detailed elimination history | RFE |

No single selection method is universally optimal.

The appropriate approach depends on dataset dimensions, predictive objective, computational resources, interpretability requirements, data sensitivity, and validation design.

## 16. Execution Requirements

| Deliverable | Runtime | External dependencies |
|---|---|---|
| Python | Python 3.10+ | None |
| JavaScript | Node.js 18+ | None |
| C++17 | C++17 compiler | None |
| Java | Java 17 | None |
| PostgreSQL | PostgreSQL 15+ | None |

The Python and JavaScript scripts create output directories automatically.

The C++17 and Java 17 programs print selection results and execution records to the terminal.

The PostgreSQL script creates its schema, tables, constraints, triggers, functions, procedures, sample records, and reports within a transaction.

## 17. Expected Learning Outcomes

The implementations demonstrate:

- Statistical screening of numerical variables.
- Redundancy detection through correlation.
- Information-based feature ranking.
- Model-driven feature-subset evaluation.
- Regularization-based coefficient selection.
- Iterative feature elimination.
- Feature-selection policy management.
- Reproducible experiment execution.
- Database-backed selection history.
- Auditability and data-governance enforcement.
- Separation of feature selection from final model evaluation.

The central technical distinction is that filter methods evaluate features statistically, wrapper methods evaluate feature subsets using a predictive model, embedded methods select features during model fitting, and RFE repeatedly eliminates features according to model-derived importance.
