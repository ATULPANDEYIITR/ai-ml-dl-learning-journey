# Model Interpretability: Feature Importance, Permutation Importance and Partial Dependence

## 1. Domain and Learning Objectives

Model interpretability examines how a predictive model uses input features and how changes in those features relate to its predictions.

A predictive model can achieve good evaluation metrics without providing sufficient information about its decision-making behavior. Interpretability methods help analysts investigate feature influence, predictive dependence, and relationships between input values and model outputs.

This project implements three distinct interpretation techniques:

1. Feature importance using standardized regression coefficients.
2. Permutation importance using changes in predictive error.
3. Partial dependence using average predictions across selected feature values.

The practical scenario is an enterprise business credit-risk model using:

| Feature | Description | Unit |
|---|---|---|
| Revenue | Annual business revenue | Thousand USD |
| Debt ratio | Debt relative to assets | Ratio |
| Customer retention | Customer retention rate | Ratio |

The target is a continuous business-risk score.

## 2. Core Concepts

### 2.1 Model Interpretability

Interpretability describes the extent to which the behavior of a predictive model can be understood through its structure, parameters, or observed predictions.

Two important perspectives are:

- Global interpretability: examines model behavior across a dataset.
- Local interpretability: examines a prediction or a limited group of observations.

The five source implementations concentrate on global interpretation.

The regression model provides a controlled environment for examining how the three techniques differ.

### 2.2 Feature Importance

Feature importance measures the relative contribution or influence of individual input features according to a particular definition.

Importance is not a single universal mathematical quantity. Its meaning depends on the underlying method.

The Python, JavaScript, C++17, and Java 17 implementations calculate standardized coefficient importance.

For a standardized linear regression model:

\[
\hat{y} = \beta_0 + \sum_{j=1}^{p}\beta_j z_j
\]

where:

\[
z_j = \frac{x_j-\mu_j}{\sigma_j}
\]

The standardized coefficient magnitude is:

\[
I_j = |\beta_j|
\]

Normalized coefficient importance is:

\[
I_j^{normalized}
=
\frac{|\beta_j|}
{\sum_{k=1}^{p}|\beta_k|}
\]

The normalization produces relative magnitudes that sum to one when at least one coefficient is nonzero.

For example, a feature with normalized importance of 0.45 accounts for 45% of the total absolute standardized coefficient magnitude.

This is not a statement that the feature causes 45% of the outcome.

**Important distinction:** coefficient importance is directly interpretable for the fitted linear model, but it does not measure predictive performance degradation when a feature is removed.

Correlated features can distribute or distort coefficient magnitudes.

### 2.3 Standardization

Features may have different measurement scales.

For example:

- Revenue may range from 100 to 1,000.
- Debt ratio may range from 0.05 to 0.95.
- Customer retention may range from 0.40 to 0.99.

Directly comparing unstandardized coefficients can therefore be misleading.

Standardization transforms a feature using its mean and standard deviation:

\[
z = \frac{x-\mu}{\sigma}
\]

The implementations calculate the mean and population standard deviation using training observations.

A constant feature receives a scale of one to avoid division by zero.

Standardization improves coefficient comparability, although it does not eliminate problems caused by multicollinearity or correlated inputs.

## 3. Permutation Importance

### 3.1 Concept

Permutation importance measures how much predictive performance deteriorates when the values of one feature are randomly rearranged.

The original dataset establishes the baseline performance.

Each feature is then independently shuffled while the remaining features stay unchanged.

The model makes new predictions using the modified dataset.

For mean squared error:

\[
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

Permutation importance is:

\[
PI_j =
MSE_{permuted,j}
-
MSE_{baseline}
\]

Interpretation:

| Result | Meaning |
|---|---|
| Large positive value | Shuffling the feature substantially damages predictive performance |
| Near-zero value | Shuffling has little effect on measured performance |
| Negative value | Shuffling improves measured performance |
| High standard deviation | The estimated importance varies considerably across permutations |

A negative result should not automatically be replaced with zero in the raw reporting data.

The implementations preserve the raw importance score and use nonnegative values only for normalized presentation.

### 3.2 Repeated Permutations

One random shuffle may produce an unstable estimate.

The implementations perform multiple independent permutations.

For feature \(j\), the mean importance is:

\[
\overline{PI}_j =
\frac{1}{R}
\sum_{r=1}^{R}PI_{j,r}
\]

The variation between repetitions is also calculated.

The repeat count is configurable.

The Python implementation uses a seeded random generator. The JavaScript implementation includes a deterministic random-number generator. C++17 uses the standard random library. Java 17 uses `java.util.Random`.

A fixed seed supports reproducible demonstrations.

### 3.3 Correlated Features

Permutation importance has an important limitation.

Suppose revenue and total assets contain substantially overlapping information.

Shuffling revenue may have a limited effect because the model can still use total assets.

The importance of revenue may consequently appear smaller than its actual contribution to the information available to the model.

Independent permutation can also create unrealistic combinations of feature values.

For correlated datasets, analysts should interpret permutation results alongside correlation analysis, domain knowledge, and alternative evaluation experiments.

### 3.4 Evaluation Dataset

Permutation importance should generally be calculated on a representative holdout dataset rather than on training data alone.

The source implementations separate training and testing observations.

This reduces the risk of interpreting training-specific predictive behavior as evidence of generalization.

The baseline and permuted predictions use the same observations, target values, model version, and evaluation metric.

## 4. Partial Dependence

### 4.1 Concept

Partial dependence examines how the average model prediction changes when one feature is assigned selected values.

For feature \(X_j\):

\[
PD_j(z) =
\frac{1}{n}
\sum_{i=1}^{n}
f(z,X_{i,-j})
\]

Where:

- \(z\) is a selected feature value.
- \(X_{i,-j}\) represents the remaining features for observation \(i\).
- \(f\) is the predictive model.
- \(n\) is the number of observations.

The method replaces one feature value across all observations and averages the resulting predictions.

### 4.2 Grid Construction

The implementations calculate a grid using the minimum and maximum observed values.

For \(m\) grid points:

\[
z_k =
x_{min}
+
\frac{k}{m-1}
(x_{max}-x_{min})
\]

where:

\[
k = 0,1,\ldots,m-1
\]

Each grid point produces one average prediction.

For example, a debt-ratio partial dependence analysis might evaluate:

| Debt ratio | Average predicted score |
|---:|---:|
| 0.10 | 45 |
| 0.30 | 21 |
| 0.50 | -3 |
| 0.70 | -27 |
| 0.90 | -51 |

The decreasing predictions indicate a negative relationship between debt ratio and the model's predicted score in this illustrative dataset.

### 4.3 Interpretation Limitations

Partial dependence describes model behavior, not a causal relationship.

A significant limitation arises when features are correlated.

Changing one feature while holding all other features unchanged may create combinations that rarely or never occur in real observations.

For example, an extremely high revenue value combined with an unusually low asset value might be implausible.

The resulting average predictions can therefore represent artificial combinations rather than realistic business conditions.

Partial dependence should be examined against observed feature distributions and domain constraints.

The implementation handles constant features by generating a single grid point.

## 5. Differences Between the Three Techniques

| Characteristic | Coefficient importance | Permutation importance | Partial dependence |
|---|---|---|---|
| Main purpose | Compare fitted coefficients | Measure predictive dependence | Examine prediction behavior |
| Main input | Model parameters | Model and evaluation data | Model and evaluation data |
| Main output | Relative coefficient magnitude | Performance degradation | Average predictions |
| Requires repeated experiments | No | Recommended | No |
| Captures feature direction | Coefficient sign is available | Not directly | Through the curve |
| Model compatibility | Linear models with coefficients | Model-agnostic | Model-agnostic |
| Correlation sensitivity | High | Potentially high | Potentially high |
| Causal interpretation | No | No | No |

The methods answer different questions.

- Coefficient importance asks which standardized model coefficients have the largest magnitudes.
- Permutation importance asks which features matter most to predictive performance on the selected evaluation dataset.
- Partial dependence asks how average predictions change across selected feature values.

Their results should not be treated as interchangeable.

## 6. Python Implementation

**File:** `model_interpretability.py`

The Python implementation provides a complete regression and interpretation workflow using standard-library modules.

### Main Components

| Component | Responsibility |
|---|---|
| `Feature` | Defines feature metadata |
| `LinearRegressionModel` | Fits and evaluates standardized linear regression |
| `EvaluationResult` | Stores regression evaluation metrics |
| `ImportanceResult` | Stores feature importance results |
| `PartialDependenceResult` | Stores partial dependence observations |
| `coefficient_importance()` | Calculates coefficient magnitudes |
| `permutation_importance()` | Calculates repeated MSE increases |
| `partial_dependence()` | Generates average predictions |
| `InterpretabilityError` | Handles interpretation failures |

### Regression Training

The model uses ordinary least squares.

The regression parameters are calculated using the normal equations:

\[
(X^TX)\beta = X^Ty
\]

Gaussian elimination with partial pivoting solves the resulting linear system.

The implementation checks for singular matrices and reports possible feature collinearity.

The algorithm avoids an external numerical package.

Normal equations have limitations for large or poorly conditioned datasets. Production systems handling substantial feature matrices should use numerically stable linear algebra implementations, such as QR decomposition or singular value decomposition.

### Dataset Generation

The generated observations follow:

\[
y =
0.035R
-
120D
+
180C
+
\epsilon
\]

Where:

- \(R\) is revenue.
- \(D\) is debt ratio.
- \(C\) is customer retention.
- \(\epsilon\) is random noise.

The dataset contains 240 observations.

A deterministic seed ensures repeatable experiments.

### Evaluation Metrics

The implementation calculates:

- Mean squared error.
- Mean absolute error.
- Root mean squared error.

These metrics establish the predictive baseline used by permutation importance.

### Report Generation

The script writes:

`interpretability_report.json`

The report contains:

- Feature metadata.
- Evaluation metrics.
- Standardized coefficient importance.
- Permutation importance.
- Partial dependence grids.
- Regression coefficients.

JSON is used for portable reporting without loading executable serialized objects.

## 7. JavaScript Implementation

**File:** `model_interpretability.js`

The JavaScript implementation emphasizes event-driven execution, feature registration, policy validation, and auditability.

### Main Components

| Component | Responsibility |
|---|---|
| `FeatureRegistry` | Maintains registered feature metadata |
| `LinearModel` | Provides prediction operations |
| `InterpretationPolicy` | Enforces analysis restrictions |
| `InterpretationService` | Calculates the three interpretation methods |
| `InterpretationEventBus` | Publishes workflow events |
| `InterpretationWorkflow` | Coordinates execution and auditing |
| `DeterministicRandom` | Provides repeatable permutation experiments |

### Feature Registry

A `Map` stores feature definitions.

The registry prevents duplicate feature names and rejects incomplete metadata.

This creates a controlled feature contract between the predictive model and the interpretation service.

### Policy Evaluation

The interpretation policy checks:

- Minimum observation count.
- Maximum feature count.
- Minimum permutation repetitions.
- Maximum permutation repetitions.
- Dataset dimensions.
- Finite numeric values.

The service rejects operations that violate these restrictions.

### Event-Driven Workflow

The event bus publishes:

- `interpretation.started`
- `interpretation.completed`
- `interpretation.failed`

Listeners are executed asynchronously through promises.

`Promise.allSettled()` allows the event system to collect listener outcomes without allowing one listener failure to interrupt other listeners.

The workflow records execution status and timestamps.

### Operational Report

The implementation produces:

`javascript_interpretability_report.json`

The report includes feature metadata, evaluation results, importance measurements, partial dependence, and audit history.

The output file is written with restricted file permissions where supported by the operating system.

## 8. C++17 Implementation

**File:** `model_interpretability.cpp`

The C++17 implementation emphasizes explicit domain modeling, numerical operations, STL containers, and exception-based validation.

### Main Components

| Component | Responsibility |
|---|---|
| `Feature` | Represents feature metadata |
| `Dataset` | Maintains validated observations |
| `LinearModel` | Fits and predicts using regression |
| `InterpretationPolicy` | Defines execution limits |
| `Importance` | Stores importance measurements |
| `Evaluation` | Stores evaluation metrics |
| `DependencePoint` | Stores partial dependence observations |

### Numerical Implementation

The model calculates feature means and standard deviations.

It then constructs the normal equations and solves them using Gaussian elimination with partial pivoting.

The implementation avoids external libraries and uses:

- `vector` for feature matrices and results.
- `unordered_set` for duplicate-name validation.
- `mt19937` for deterministic random permutations.
- `shuffle` for feature rearrangement.
- `accumulate` for aggregation.
- `sort` for importance ranking.

### Exception Handling

Invalid dimensions, non-finite values, singular matrices, invalid feature indices, and invalid repetition counts produce explicit exceptions.

The main function catches execution failures and returns a nonzero process exit code.

### Performance Considerations

For \(n\) observations and \(p\) features:

- Prediction requires approximately \(O(np)\) operations.
- Coefficient importance requires \(O(p)\) operations.
- One permutation experiment for a feature requires approximately \(O(np)\) prediction operations.
- Repeated permutation importance requires approximately \(O(Rnp^2)\) operations.
- Partial dependence with \(m\) grid points requires approximately \(O(mnp)\) operations.

The regression normal-equation solution has approximately \(O(np^2+p^3)\) computational cost.

These choices make the implementation suitable for a controlled educational dataset, not an unrestricted high-dimensional production workload.

## 9. Java 17 Implementation

**File:** `ModelInterpretability.java`

The Java implementation emphasizes enterprise service design, immutable records, domain validation, governance policies, state transitions, and audit events.

### Domain Types

The implementation uses Java records for:

- `FeatureDefinition`
- `EvaluationResult`
- `ImportanceResult`
- `DependencePoint`
- `DependenceResult`
- `InterpretationPolicy`
- `AuditEvent`

Records provide concise data structures with generated accessors and value-oriented behavior.

Collections are exposed as immutable copies where appropriate.

### Enterprise Services

| Service | Responsibility |
|---|---|
| `ModelTrainer` | Trains the regression model |
| `EvaluationService` | Calculates predictive performance |
| `InterpretationService` | Performs interpretation analysis |
| `AuditService` | Stores execution events |
| `InterpretationWorkflow` | Coordinates the analysis lifecycle |

### State Management

An interpretation execution follows:

`CREATED → RUNNING → COMPLETED`

A failed execution follows:

`CREATED → RUNNING → FAILED`

The workflow records each transition with a UUID, timestamp, status, and event description.

### Governance Controls

The Java policy checks:

- Minimum sample requirements.
- Maximum feature count.
- Permitted permutation repetitions.
- Feature metadata consistency.
- Sensitive feature restrictions.

The demonstration policy prohibits sensitive features.

This is a governance control for the example workflow. Actual production access control must also be enforced at the data-access and service authorization layers.

### Concurrency Considerations

The audit service uses `CopyOnWriteArrayList` for thread-safe event storage and listener management.

This structure is suitable for relatively frequent reads and infrequent writes.

For high-volume audit ingestion, a persistent database-backed event store is more appropriate.

## 10. PostgreSQL Implementation

**File:** `model_interpretability.sql`

The PostgreSQL implementation models an enterprise interpretability registry and reporting workflow.

Unlike the source programs, which calculate the regression and interpretation results, the database stores the results, validates their relationships, and supports governance reporting.

### Database Architecture

The principal relationship is:

Model Registry

→ Model Versions

→ Model Features

→ Evaluation Datasets

→ Interpretation Runs

→ Feature Importance Results

→ Partial Dependence Results

→ Governance Approvals

→ Audit Events

### Main Tables

| Table | Purpose |
|---|---|
| `model_registry` | Registers predictive models |
| `model_versions` | Tracks individual model versions |
| `features` | Maintains feature metadata |
| `model_features` | Maps features to model versions |
| `interpretation_policies` | Stores analysis governance rules |
| `evaluation_datasets` | Registers evaluation datasets |
| `interpretation_runs` | Tracks execution status and metrics |
| `feature_importance_results` | Stores importance measurements |
| `partial_dependence_results` | Stores feature-value prediction points |
| `interpretation_approvals` | Stores governance decisions |
| `interpretation_audit` | Records workflow events |

### Referential Integrity

Primary keys uniquely identify records.

Foreign keys ensure that:

- Model versions belong to registered models.
- Features belong to the feature registry.
- Interpretation executions reference existing versions and datasets.
- Importance results reference valid executions and features.
- Partial dependence records reference registered features.
- Approvals reference existing interpretation runs.

Unique constraints prevent duplicate feature positions, duplicate feature-method results, and duplicate reviewer decisions for a run.

### Check Constraints

Database checks enforce:

- Positive observation counts.
- Valid feature positions.
- Nonnegative evaluation metrics.
- Valid normalized importance ranges.
- Valid permutation repetition counts.
- Consistent MSE and RMSE values.

### Triggers

Three important integrity mechanisms are implemented.

**Interpretation run validation**

The `validate_interpretation_run()` function checks:

- Active interpretation policy.
- Dataset availability.
- Sample count.
- Model feature count.
- Test dataset requirement.
- Permutation repetition limits.
- Sensitive feature restrictions.

**Importance validation**

The `validate_importance_feature()` function ensures that an importance result belongs to a feature registered for the model version and that its execution is currently running.

**Approval validation**

The `validate_interpretation_approval()` function prevents approval of incomplete interpretation executions.

### Workflow Functions

The SQL script implements:

| Function | Responsibility |
|---|---|
| `start_interpretation()` | Validates and starts an interpretation run |
| `complete_interpretation()` | Records metrics and completes a run |
| `fail_interpretation()` | Records a failure and its reason |

The execution workflow is transactional.

If a validation operation fails, the transaction can be rolled back rather than leaving partially committed results.

### Reporting

The database provides:

- Model and version retrieval.
- Feature metadata retrieval.
- Coefficient and permutation comparisons.
- Importance aggregation.
- Partial dependence reporting.
- Normalization validation.
- Approval eligibility.
- Failed execution reporting.
- Audit history.
- Operational performance reporting.

Indexes support common access patterns involving model versions, execution states, features, interpretation runs, and audit timestamps.

## 11. Interpretation Workflow

The complete workflow is:

1. Register the predictive model.
2. Register its version.
3. Register the input features.
4. Associate features with the model version.
5. Register the evaluation dataset.
6. Select an active interpretation policy.
7. Validate dataset dimensions and governance restrictions.
8. Establish baseline model performance.
9. Calculate standardized coefficient importance.
10. Calculate repeated permutation importance.
11. Calculate partial dependence for each selected feature.
12. Store interpretation results.
13. Validate execution completion.
14. Submit results for governance review.
15. Record approval decisions and audit events.
16. Produce interpretation reports.

The workflow separates numerical interpretation from database governance.

The Python, JavaScript, C++17, and Java 17 programs perform numerical analysis.

PostgreSQL maintains the corresponding enterprise records and workflow state.

## 12. Model Evaluation and Interpretation Quality

Interpretability results should be examined together with predictive performance.

A model with high predictive error may still produce mathematically valid importance scores, but those scores may describe an unreliable predictive relationship.

The following checks should be performed:

| Validation | Purpose |
|---|---|
| Holdout evaluation | Assess generalization |
| Feature validation | Detect invalid or inconsistent inputs |
| Coefficient stability | Identify unstable regression relationships |
| Permutation repetition | Measure importance variability |
| Correlation assessment | Identify redundant information |
| Partial dependence inspection | Identify unusual prediction patterns |
| Dataset distribution checks | Detect unrealistic feature combinations |
| Model-version verification | Ensure the correct model is interpreted |
| Policy validation | Enforce analysis restrictions |
| Audit review | Preserve execution accountability |

### Normalization Considerations

Coefficient and permutation scores have different units.

Coefficient scores represent standardized model parameters.

Permutation scores represent changes in the evaluation metric.

Although both can be normalized for reporting, their normalized values do not become equivalent measurements.

Normalized scores should only be compared within the same method and experiment.

## 13. Security and Governance Considerations

Interpretability results can expose information about model behavior, input characteristics, and potentially sensitive attributes.

Enterprise implementation should consider:

- Role-based access to model metadata.
- Restricted access to sensitive feature definitions.
- Authorization before executing interpretation workflows.
- Encryption of stored model artifacts and datasets.
- Controlled access to evaluation datasets.
- Audit logging for interpretation and approval operations.
- Validation of imported numerical results.
- Model-version and dataset-version tracking.
- Retention policies for interpretation records.
- Separation of analysts and governance approvers.

A feature marked as sensitive in the database is metadata, not a substitute for access control.

Likewise, a database approval record does not establish that an interpretation method is statistically appropriate.

Production systems should distinguish technical validation, statistical review, and organizational approval.

## 14. Limitations and Practical Implications

### Coefficient Importance

- Requires a model with meaningful coefficients.
- Standardization improves scale comparability but does not resolve multicollinearity.
- Coefficient magnitude does not establish causal influence.
- Nonlinear relationships may not be represented adequately by linear coefficients.

### Permutation Importance

- Requires access to model predictions and evaluation data.
- Repeated permutations increase computational cost.
- Correlated features may share predictive information.
- Independent shuffling can generate unrealistic observations.
- Results depend on the selected metric and evaluation population.

### Partial Dependence

- Requires repeated prediction operations.
- Can produce misleading curves for correlated features.
- Extrapolation and unrealistic feature combinations require attention.
- Average effects may conceal different relationships across subgroups.

### Implementation Limitations

The five implementations use ordinary least-squares regression to maintain a common, reproducible business scenario.

They do not implement every possible interpretation method.

In particular, they do not provide:

- SHAP value calculations.
- LIME explanations.
- Individual conditional expectation plots.
- Accumulated local effects.
- Tree-specific impurity importance.
- Causal feature attribution.
- Confidence intervals derived from formal statistical inference.

The database stores interpretation measurements but does not independently recalculate them from model artifacts.

A production deployment should integrate model execution, dataset lineage, numerical validation, and persistent governance controls through authenticated services.

## 15. Execution

### Python

Requirements: Python 3.10 or later.

Run:

`python model_interpretability.py`

Output:

`interpretability_report.json`

### JavaScript

Requirements: Node.js 18 or later.

Run:

`node model_interpretability.js`

Output:

`javascript_interpretability_report.json`

### C++17

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic model_interpretability.cpp -o interpretability`

Run:

`./interpretability`

On Windows:

`interpretability.exe`

### Java 17

Compile:

`javac ModelInterpretability.java`

Run:

`java ModelInterpretability`

### PostgreSQL

Execute:

`psql -d your_database -f model_interpretability.sql`

The database user requires permission to create schemas, tables, types, functions, triggers, and indexes.

## 16. Technical Coverage Matrix

| Capability | Python | JavaScript | C++17 | Java 17 | PostgreSQL |
|---|---|---|---|---|---|
| Feature metadata | Yes | Yes | Yes | Yes | Yes |
| Dataset validation | Yes | Yes | Yes | Yes | Yes |
| Linear regression | Yes | Yes | Yes | Yes | Stores model metadata |
| Standardized coefficients | Yes | Yes | Yes | Yes | Stores results |
| Coefficient importance | Yes | Yes | Yes | Yes | Reporting |
| Permutation importance | Yes | Yes | Yes | Yes | Stores results |
| Partial dependence | Yes | Yes | Yes | Yes | Stores results |
| MSE, MAE and RMSE | Yes | Yes | Yes | Yes | Stores metrics |
| Repeatability | Seeded | Seeded | Seeded | Seeded | Records seed |
| Execution policies | Validation | Policy object | Policy class | Policy record | Policy tables |
| Event handling | Exceptions | Event bus | Exceptions | Audit workflow | Triggers |
| Audit history | JSON report | JSON report | Console | Audit service | Audit table |
| Governance approvals | No | No | No | Execution audit | Approval workflow |
| Persistent reporting | JSON | JSON | Console | Console | Relational views |

## 17. Key Technical Distinctions

The central distinction is the question each interpretation method answers.

**Feature importance through coefficients** examines the magnitude of fitted standardized model parameters.

**Permutation importance** measures the change in predictive performance when one feature's information is disrupted.

**Partial dependence** examines the average predicted outcome when a feature is evaluated at selected values.

These techniques provide complementary evidence.

A feature can have a large coefficient but limited permutation importance when another feature contains similar predictive information.

A feature can have high permutation importance but a partial dependence curve that is nonlinear or otherwise difficult to summarize through a single importance score.

Reliable interpretation requires the model, evaluation data, statistical assumptions, feature relationships, and governance context to be examined together.
