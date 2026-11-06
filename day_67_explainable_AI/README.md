# Explainable AI: SHAP, LIME, and Local and Global Explanations

## 1. Project Scope

This project implements an explainability workflow for a binary credit-risk classification model.

The central objective is to distinguish three related but technically different activities:

- SHAP assigns feature-level contributions using Shapley values.
- LIME approximates model behavior around an individual observation using a locally weighted surrogate.
- Global explanation aggregates information across observations to characterize model behavior at dataset level.

The implementations use a deterministic logistic classification model with five numerical input features. This makes the explanation calculations reproducible and allows their mathematical properties to be checked.

The five implementations provide complementary perspectives:

| Implementation | Engineering focus |
|---|---|
| Python | Exact SHAP enumeration, LIME regression, global importance and JSON reporting |
| JavaScript | Event-driven explanation service, deterministic perturbation and audit records |
| C++17 | STL-based explanation engine, coalition enumeration and numerical regression |
| Java 17 | Enterprise domain model, immutable records, policy enforcement and audit events |
| PostgreSQL | Relational storage, integrity constraints, explanation workflows and reporting |

## 2. Explainable AI Fundamentals

Explainable Artificial Intelligence refers to methods that make the behavior of predictive systems more understandable to people.

A predictive model produces an output. An explanation attempts to identify the factors associated with that output and communicate their influence.

For a binary classifier, the model may return a probability:

\[
f(x)=P(Y=1\mid X=x)
\]

where:

- \(X\) represents the input features.
- \(Y\) represents the predicted outcome.
- \(x\) represents one particular observation.
- \(f(x)\) represents the model output.

For the credit-risk example, the features are:

| Feature | Meaning |
|---|---|
| income | Annual income |
| debt ratio | Debt relative to income |
| credit history | Length of credit history |
| missed payments | Number of missed payments |
| employment years | Employment duration |

The model calculates a weighted score:

\[
z=b+\sum_{i=1}^{M}w_ix_i
\]

It converts the score into a probability using the logistic function:

\[
f(x)=\frac{1}{1+e^{-z}}
\]

A classification threshold converts the probability into a decision.

The probability and decision are different outputs. A probability of 0.65 does not itself establish that an applicant should be approved. That decision also depends on business rules, risk thresholds and applicable governance requirements.

## 3. Local and Global Explanations

### 3.1 Local explanations

A local explanation describes the behavior of a model for one observation.

For example, a credit-risk application may produce a probability of 0.85 for an individual applicant.

A local explanation identifies the features contributing positively or negatively to the model output.

The explanation should answer questions such as:

- Which features contributed most to the prediction?
- Which features increased the predicted probability?
- Which features decreased it?
- What reference or baseline was used?
- How closely does the explanation represent the actual model output?

SHAP and LIME can both provide local explanations, but they calculate their results differently.

### 3.2 Global explanations

A global explanation characterizes model behavior across multiple observations.

It is useful for:

- Understanding influential features across a population.
- Comparing feature importance between model versions.
- Detecting unexpected dependence on input variables.
- Investigating changes in model behavior.
- Supporting model governance and monitoring.

The Python, JavaScript and C++ implementations calculate global importance by aggregating absolute SHAP values.

For feature \(i\), mean absolute SHAP importance is:

\[
I_i=\frac{1}{N}\sum_{j=1}^{N}
|\phi_i^{(j)}|
\]

where:

- \(N\) is the number of observations.
- \(\phi_i^{(j)}\) is the SHAP value for feature \(i\) in observation \(j\).

Absolute values prevent positive and negative contributions from cancelling each other.

Global importance measures average contribution magnitude. It does not indicate whether a feature consistently increases or decreases predictions.

## 4. SHAP Values

### 4.1 Mathematical foundation

SHAP stands for SHapley Additive exPlanations.

It applies Shapley values from cooperative game theory to feature attribution.

Each feature is treated as a participant in a cooperative game. The model output is the total value, and the explanation distributes the difference between a reference prediction and the actual prediction among the features.

The additive representation is:

\[
f(x)=\phi_0+\sum_{i=1}^{M}\phi_i
\]

where:

- \(\phi_0\) is the baseline prediction.
- \(\phi_i\) is the contribution assigned to feature \(i\).
- \(M\) is the total number of features.

A positive SHAP value increases the model output relative to the baseline. A negative value decreases it.

### 4.2 Shapley value calculation

The Shapley value for feature \(i\) is:

\[
\phi_i=
\sum_{S\subseteq F\setminus\{i\}}
\frac{|S|!(M-|S|-1)!}{M!}
\left[
v(S\cup\{i\})-v(S)
\right]
\]

Here:

- \(F\) is the complete feature set.
- \(S\) is a subset excluding feature \(i\).
- \(v(S)\) is the value of a coalition containing features in \(S\).
- The factorial expression determines the coalition weight.

The calculation considers how much a feature changes the output when introduced into different feature combinations.

### 4.3 Background dataset

The implementations use a background dataset to establish reference observations.

The coalition value is estimated using hybrid observations:

- Features included in the coalition retain their applicant values.
- Features excluded from the coalition use values from background observations.
- The model is evaluated on the resulting hybrid observations.
- Their predictions are averaged.

This is an interventional-style approach to feature attribution.

The selected background dataset affects the explanation. A different reference population can produce different baseline predictions and feature contributions.

### 4.4 Exact enumeration

For five features, the number of possible coalitions is:

\[
2^5=32
\]

The implementations enumerate these coalitions and calculate exact Shapley values for the defined coalition function.

The efficiency property requires:

\[
\phi_0+\sum_i\phi_i=f(x)
\]

The Python and Java implementations explicitly validate this property.

### 4.5 Computational complexity

For \(M\) features and \(B\) background observations, the implemented exact method has approximate complexity:

\[
O(2^M \times M \times B)
\]

Exact enumeration becomes expensive as the number of features grows.

For larger models, practical SHAP implementations can use optimized algorithms, model-specific explainers or sampling-based approximations.

### 4.6 Interpretation limitations

SHAP values describe attribution under a selected value function and reference distribution.

They do not establish:

- Causality.
- A real-world intervention effect.
- A feature's independent causal importance.
- The absence of bias.
- The correctness of the underlying model.

Correlated features can create attribution challenges, especially when the selected coalition-value approach does not preserve their dependencies.

## 5. LIME

### 5.1 Definition

LIME stands for Local Interpretable Model-agnostic Explanations.

It explains one prediction by approximating the original model in a local neighborhood around an observation.

Instead of calculating exact Shapley values, LIME generates perturbed observations and fits an interpretable surrogate model.

The surrogate is designed to approximate the original model near the observation being explained.

### 5.2 LIME workflow

The implementations follow this sequence:

1. Validate the original observation.
2. Generate perturbed feature values.
3. Evaluate the original model on the perturbed observations.
4. Calculate the distance between each perturbation and the original observation.
5. Convert distances into proximity weights.
6. Fit a weighted linear regression model.
7. Return local coefficients and surrogate information.

### 5.3 Proximity kernel

The implementations use an exponential proximity kernel:

\[
\pi_x(z)=
\exp\left(
-\frac{D(x,z)^2}{\sigma^2}
\right)
\]

where:

- \(x\) is the original observation.
- \(z\) is a perturbed observation.
- \(D(x,z)\) is their standardized distance.
- \(\sigma\) is the kernel width.

Observations closer to the original observation receive higher weights.

### 5.4 Weighted regression

The surrogate is represented as:

\[
g(z)=\beta_0+\sum_{i=1}^{M}\beta_i z_i
\]

The weighted regression minimizes:

\[
\sum_{j=1}^{N}
\pi_x(z_j)
\left(f(z_j)-g(z_j)\right)^2
+
\lambda\sum_i\beta_i^2
\]

The regularization term stabilizes the regression when the generated observations produce a poorly conditioned matrix.

The source files implement the calculation using normal equations and Gaussian elimination with partial pivoting.

### 5.5 Important LIME parameters

| Parameter | Purpose |
|---|---|
| Perturbation count | Determines how many artificial observations are generated |
| Kernel width | Controls how quickly observation weights decrease |
| Random seed | Supports reproducibility |
| Regularization | Reduces instability in the surrogate regression |
| Feature scaling | Prevents variables with different numerical ranges from dominating distance calculations |

### 5.6 LIME limitations

LIME explanations depend on:

- The perturbation distribution.
- The distance function.
- The kernel width.
- The random seed.
- The selected feature representation.
- The quality of the local surrogate.

A surrogate can fit its local neighborhood poorly.

Its coefficients are not guaranteed to reconstruct the original model prediction through an additive SHAP-style decomposition.

The JavaScript and Python implementations therefore keep LIME metadata and contributions separate from SHAP baseline and additivity calculations.

## 6. SHAP and LIME: Technical Distinctions

| Dimension | SHAP | LIME |
|---|---|---|
| Theoretical foundation | Cooperative game theory | Local surrogate modeling |
| Main calculation | Shapley values | Weighted regression |
| Explanation scope | Local feature attribution | Local model approximation |
| Feature contribution | Coalition-based | Surrogate coefficient |
| Baseline | Expected model output under reference data | Local surrogate intercept |
| Additivity | Shapley efficiency for the defined value function | Not inherently guaranteed |
| Main computational concern | Coalition enumeration | Perturbation and regression |
| Main sensitivity | Background distribution and value function | Sampling and locality parameters |
| Global analysis | Aggregate local SHAP values | Aggregate local explanations with care |

SHAP and LIME should not be interpreted as interchangeable methods.

A difference between their results does not automatically indicate that either implementation is incorrect. They use different explanation objectives and assumptions.

## 7. Python Implementation

The Python implementation is a self-contained explanation engine.

### Main components

| Component | Responsibility |
|---|---|
| `Feature` | Stores feature limits and defaults |
| `RiskModel` | Validates inputs and calculates model probabilities |
| `Explanation` | Stores explanation results and metadata |
| `ExplainableAI` | Implements SHAP, LIME and global importance |
| `_solve_linear_system` | Performs Gaussian elimination |
| `create_background` | Supplies reference observations |
| `main` | Executes the complete workflow |

### SHAP implementation

The `shap_values` method enumerates all feature coalitions using combinations.

It caches coalition predictions to avoid recalculating the same coalition value for each feature.

This is important because every feature uses overlapping coalition subsets.

### LIME implementation

The `lime_explanation` method generates normally distributed perturbations.

Feature values are clipped to their permitted ranges. This prevents invalid model inputs.

The method then constructs weighted normal equations and solves them using partial pivoting.

### Global analysis

The `global_importance` method calculates mean absolute SHAP values for a supplied dataset.

It sorts the features in descending order of average attribution magnitude.

### Output

The script prints:

- Applicant prediction.
- Local SHAP contributions.
- Local LIME coefficients.
- Global feature importance.
- SHAP additivity validation.

It writes a JSON report to `explainability_report.json`.

### Execution

Requirements:

- Python 3.10 or later.
- Python standard library.

Run with:

`python explainable_ai.py`

No third-party package installation is required.

## 8. JavaScript Implementation

The JavaScript implementation uses an event-oriented service design.

### Main components

| Component | Responsibility |
|---|---|
| `CreditRiskModel` | Model validation and prediction |
| `DeterministicRandom` | Reproducible pseudo-random perturbations |
| `ExplanationService` | SHAP, LIME and global analysis |
| `ExplanationAudit` | Records explanation events |
| `main` | Coordinates execution and reporting |

### Deterministic perturbations

The `DeterministicRandom` class uses a seeded linear congruential generator and Box-Muller transformation.

This makes perturbation generation repeatable for a given seed.

Reproducibility is important when comparing explanations during model validation.

### Map-based coalition calculations

The SHAP implementation stores coalition results in a `Map`.

A sorted feature-name representation acts as the coalition key.

This allows efficient lookup of coalition values while maintaining separation between coalition generation and attribution calculation.

### Audit events

The audit service records:

- Prediction events.
- SHAP execution.
- LIME execution.
- Associated metadata.

The JSON report contains prediction results, explanation results, global importance and audit information.

### Execution

Requirements:

- Node.js 18 or later.

Run with:

`node explainable_ai.js`

The script writes `explainability_report.json`.

For production use, audit records should be sent to a controlled persistent logging service rather than retained only in process memory.

## 9. C++17 Implementation

The C++ implementation emphasizes explicit domain structures and computational control.

### Main components

| Component | Responsibility |
|---|---|
| `Feature` | Defines feature limits and coefficients |
| `Observation` | Holds numerical feature values |
| `Explanation` | Stores attribution results |
| `CreditRiskModel` | Performs model calculations |
| `ExplanationEngine` | Calculates SHAP, LIME and global importance |
| `printContributions` | Produces ranked explanation output |

### Coalition representation

The implementation represents feature subsets using an unsigned integer bitmask.

For five features, the complete coalition set occupies 32 masks.

A bit determines whether a feature belongs to the selected coalition.

This avoids dynamic set allocation during coalition enumeration.

### Numerical regression

LIME uses:

- `std::mt19937` for seeded random number generation.
- `std::normal_distribution` for perturbations.
- STL vectors for matrix storage.
- Gaussian elimination for weighted regression.

### Validation

The implementation checks:

- Feature count.
- Numeric finiteness.
- Feature boundaries.
- Kernel width.
- Perturbation count.
- Singular regression matrices.
- Exact SHAP feature limits.
- Additivity.

### Performance

The bitmask implementation is efficient for small feature sets.

The explicit limit of 20 features prevents uncontrolled memory and computational growth in exact coalition enumeration.

The implementation is educational and does not replace optimized production SHAP libraries for large models.

### Execution

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic explainable_ai.cpp -o explainable_ai`

Run:

`./explainable_ai`

On Windows:

`explainable_ai.exe`

## 10. Java 17 Implementation

The Java implementation models explainability as an enterprise service.

It separates model prediction, explanation computation, governance policy and audit responsibilities.

### Domain types

| Type | Responsibility |
|---|---|
| `FeatureDefinition` | Defines feature constraints |
| `Observation` | Immutable input record |
| `ModelPrediction` | Stores classification result |
| `Explanation` | Stores method-specific output |
| `ExplanationPolicy` | Defines operational constraints |
| `AuditEvent` | Records execution events |
| `PredictiveModel` | Defines prediction behavior |

Java records are used for immutable data structures.

This reduces accidental modification of observations and explanation results.

### Service architecture

The `ExplanationService` receives:

- A predictive model.
- A background dataset.
- An explanation policy.

It checks the dataset and policy before performing calculations.

The SHAP service enforces the maximum exact feature count and validates the additivity property.

The LIME service enforces perturbation-count limits and performs weighted regression.

The global service aggregates absolute SHAP contributions.

### Governance policy

The `ExplanationPolicy` record controls:

- Minimum background observations.
- Maximum exact SHAP feature count.
- SHAP additivity tolerance.
- Minimum LIME perturbations.

These controls make explanation requirements explicit rather than embedding them in individual calculations.

### Auditability

The `AuditService` records prediction and explanation events.

The report contains model output, decision, applicant values, SHAP results, LIME results and global importance.

### Execution

Requirements:

- Java 17 or later.

Compile:

`javac ExplainableAIEnterprise.java`

Run:

`java ExplainableAIEnterprise`

The application creates `explainability_enterprise_report.json`.

## 11. PostgreSQL Data Model

The PostgreSQL implementation represents an explanation-management workflow rather than performing the numerical SHAP or LIME algorithms inside the database.

The model stores predictions, input features, reference datasets, explanation executions, attributions, validation results and audit events.

### Entity relationships

The principal relationship is:

Model Version
→ Feature Definitions
→ Predictions
→ Prediction Features
→ Explanations
→ Feature Attributions
→ Explanation Validations
→ Audit Events

Background datasets provide reference observations for explanation calculations.

Global importance records associate aggregate feature measurements with a model, explanation and evaluation dataset.

### Core tables

| Table | Purpose |
|---|---|
| `model_versions` | Registers predictive model versions |
| `feature_definitions` | Defines feature names, ranges and baseline values |
| `predictions` | Stores model outputs and decisions |
| `prediction_features` | Stores actual input feature values |
| `background_datasets` | Registers reference populations |
| `background_observations` | Stores individual reference records |
| `background_features` | Stores background feature values |
| `explanation_policies` | Stores explanation governance rules |
| `explanations` | Stores explanation executions and states |
| `feature_attributions` | Stores local SHAP or LIME-related attribution values |
| `global_importance` | Stores aggregated SHAP importance |
| `explanation_validations` | Stores validation outcomes |
| `explanation_audit` | Stores operational audit events |

### Integrity enforcement

The schema uses primary keys and foreign keys to maintain relationships.

Unique constraints prevent duplicate feature definitions, background observations and explanation ranks.

Check constraints validate numerical boundaries and explanation states.

Triggers validate:

- Prediction feature ranges.
- Background feature ranges.
- Attribution direction.
- Explanation state transitions.

The composite foreign keys ensure that feature values belong to the same model as their prediction or background observation.

### Explanation state management

The explanation workflow uses these states:

| State | Meaning |
|---|---|
| PENDING | Waiting for processing |
| RUNNING | Explanation computation is active |
| COMPLETED | Explanation computation finished |
| FAILED | Execution failed |
| REJECTED | Execution was rejected by policy |

Terminal states cannot be changed by the transition trigger.

A failed explanation must contain a failure reason.

A completed explanation must contain its completion timestamp and principal prediction values.

### SHAP additivity validation

The `validate_shap_additivity` function calculates:

\[
\hat f(x)=\phi_0+\sum_i\phi_i
\]

It compares the reconstructed output with the stored model output.

The difference is evaluated against the active explanation policy's tolerance.

This is an important quality-control mechanism for additive SHAP explanations.

### Eligibility evaluation

The `check_explanation_eligibility` function evaluates:

- Whether the prediction exists.
- Whether an active policy exists.
- Whether the exact SHAP feature limit is satisfied.
- Whether an eligible background dataset exists.

It returns an eligibility result and an explanatory reason.

### Performance-oriented indexes

Indexes support:

- Prediction retrieval by model and timestamp.
- Decision-based prediction reporting.
- Feature attribution lookup.
- Explanation execution monitoring.
- Global importance ranking.
- Audit retrieval.
- Failed validation searches.

The partial index on active policies ensures that only one active policy exists per model.

### Reporting

The SQL script provides queries for:

- Local explanation retrieval.
- Global feature ranking.
- SHAP and LIME execution performance.
- Failed validation detection.
- SHAP additivity.
- Explanation eligibility.
- Positive and negative contribution aggregation.
- Explanation failure percentages.
- Audit history.
- Prediction feature completeness.
- Pending execution retrieval.

### Transaction handling

The script uses a transaction for schema and sample-data creation.

The final demonstration uses a savepoint and rollback to show how invalid feature insertion is handled without retaining the invalid record.

### Database execution

Requirements:

- PostgreSQL 15 or later.
- Permission to create a schema, tables, functions, triggers and indexes.

Execute the SQL file using `psql`:

`psql -U postgres -d explainability_db -f explainable_ai.sql`

The script creates the `explainability` schema and its sample records.

## 12. Validation and Testing Considerations

Explanation quality requires more than successful execution.

The implementations include input validation, deterministic examples and numerical checks.

Important validation categories include:

| Validation | Purpose |
|---|---|
| Feature schema | Ensures expected features are present |
| Numeric validation | Rejects non-finite values |
| Range validation | Prevents invalid observations |
| SHAP additivity | Checks reconstruction of model output |
| Background validation | Prevents empty reference datasets |
| LIME perturbation validation | Controls sampling volume |
| Kernel validation | Prevents invalid locality parameters |
| Regression validation | Detects singular matrices |
| Policy validation | Controls explanation eligibility |
| Audit validation | Preserves execution evidence |

### Additional production tests

A production implementation should test:

1. Identical input and background data produce identical deterministic SHAP results.
2. SHAP contributions reconstruct the model output within tolerance.
3. Changing the background population is reflected in the baseline and attribution results.
4. Repeated LIME runs with the same seed produce identical coefficients.
5. Increasing perturbation counts is evaluated for surrogate stability.
6. Invalid feature values are rejected before prediction.
7. Global importance uses the complete intended evaluation population.
8. Missing features cannot silently receive arbitrary values.
9. Model version changes are reflected in explanation metadata.
10. Failed explanation executions remain distinguishable from completed explanations.

## 13. Security, Privacy and Governance

Explanations can expose information that is not directly visible in a model's ordinary output.

Feature contributions may reveal sensitive characteristics or indicate which input values influence a decision.

Production deployments should therefore enforce the following controls.

### Data protection

- Minimize personal data stored in explanation records.
- Use pseudonymous subject references.
- Encrypt sensitive data at rest and in transit.
- Restrict access to background datasets.
- Apply retention rules to explanation records.
- Avoid exposing raw feature values through unrestricted reporting interfaces.

### Access control

Separate permissions should exist for:

- Model administrators.
- Explanation-service operators.
- Model validators.
- Auditors.
- Business decision-makers.

An auditor may need access to explanation metadata without requiring unrestricted access to the original personal data.

### Model governance

An explanation record should identify:

- Model name and version.
- Explanation algorithm.
- Algorithm version.
- Reference dataset version.
- Explanation parameters.
- Prediction output.
- Execution status.
- Validation outcome.
- Execution timestamp.

This makes it possible to investigate why an explanation was produced and which model configuration was involved.

### Interpretation governance

An explanation must not be treated as proof of causality.

A feature with a large attribution may be associated with other correlated variables.

A local explanation should not automatically be generalized to the complete population.

Global importance should be accompanied by population definitions and evaluation-period information.

## 14. Practical Applications

The implemented mechanisms apply to several operational environments.

| Application | Explainability requirement |
|---|---|
| Credit scoring | Identify features influencing risk predictions |
| Healthcare | Investigate factors contributing to clinical predictions |
| Cybersecurity | Examine variables influencing threat classifications |
| Fraud detection | Identify influential transaction characteristics |
| Insurance | Examine variables contributing to risk estimates |
| Customer analytics | Understand factors associated with customer outcomes |
| Model governance | Maintain reproducible explanation and validation records |

For high-impact decisions, explanations should be evaluated together with model performance, fairness analysis, data quality, human oversight and applicable legal requirements.

An explanation is one component of responsible model governance, not a substitute for it.
