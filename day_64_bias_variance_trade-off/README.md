# Bias-Variance Tradeoff

## 1. Topic and Scope

The bias-variance tradeoff describes how a predictive model's complexity affects its ability to learn a relationship from training data and generalize to unseen observations.

A model that is too simple may fail to represent the underlying relationship. A model that is excessively flexible may fit noise and accidental patterns in the training data. A useful model balances approximation error and sensitivity to the particular training sample.

The five source implementations approach this problem through numerical experiments, event-driven evaluation, model-governance workflows, and relational storage.

The examples use polynomial regression to predict a nonlinear signal:

`f(x) = sin(2πx) + 0.25x`

Training observations include independent Gaussian noise with standard deviation `0.20`.

Polynomial degree controls model complexity. Degrees 1, 2, 5, and 9 are compared under repeated independent training samples.

The implementations use controlled synthetic data. They demonstrate statistical behavior and model-selection mechanisms rather than claiming that a particular degree is universally appropriate for real datasets.

## 2. Core Concepts

### 2.1 Bias

Bias measures the systematic difference between the average prediction of a learning procedure and the true underlying relationship.

For an input value \(x\), squared bias is:

\[
Bias^2(x) =
\left(
E[\hat f(x)] - f(x)
\right)^2
\]

Here:

- \(f(x)\) is the true signal.
- \(\hat f(x)\) is the fitted model's prediction.
- \(E[\hat f(x)]\) represents the average prediction across repeated training samples.

A model with high bias makes systematic approximation errors. It may not have enough flexibility to represent the shape of the underlying signal.

A low-degree polynomial fitted to a strongly nonlinear relationship is a typical example.

### 2.2 Variance

Variance measures how much predictions change when the training dataset changes.

\[
Var(\hat f(x)) =
E\left[
\left(
\hat f(x)-E[\hat f(x)]
\right)^2
\right]
\]

A high-variance model is sensitive to the particular observations selected for training. Small changes in training data can produce substantially different fitted functions.

High polynomial degrees can produce unstable coefficients and predictions, particularly when training samples are limited or features are poorly conditioned.

### 2.3 Irreducible Noise

Irreducible noise represents the variation in observations that cannot be eliminated by selecting a more complex predictor.

For the experimental signal, the observation noise is Gaussian with standard deviation \(0.20\).

Its variance is:

\[
\sigma^2 = 0.20^2 = 0.04
\]

Reducing model bias or prediction variance does not remove independent observation noise.

The noise level is a property of the data-generating process, not a measure of model complexity.

### 2.4 Expected Prediction Error

Under the usual squared-error assumptions, expected prediction error can be decomposed as:

\[
E[(Y-\hat f(x))^2]
=
Bias^2(x)
+
Var(\hat f(x))
+
\sigma^2
\]

The implementation evaluates this relationship across a fixed grid of input values.

The aggregate estimate is:

\[
ExpectedTestMSE =
MeanBias^2
+
MeanPredictionVariance
+
IrreducibleNoise
\]

This decomposition distinguishes three sources of prediction error:

| Component | Meaning | Typical response |
|---|---|---|
| Bias squared | Systematic model approximation error | Consider a more expressive model |
| Prediction variance | Sensitivity to the selected training sample | Reduce instability or constrain complexity |
| Irreducible noise | Random variation in observations | Improve measurement quality where possible |

The decomposition is an expected-error relationship. Individual finite-sample evaluations can differ from the theoretical expectation.

### 2.5 Model Complexity

Model complexity describes the flexibility of a learning procedure.

In the polynomial examples, degree determines the number of coefficients:

\[
ParameterCount = Degree + 1
\]

A degree-1 polynomial has two parameters. A degree-9 polynomial has ten.

More parameters can represent more complicated relationships, but parameter count alone does not fully measure effective complexity. Regularization, sample size, feature scaling, model structure, and the distribution of observations also influence effective model capacity.

The examples use ridge regularization:

\[
\hat{\beta} =
(X^TX+\lambda I)^{-1}X^Ty
\]

The intercept is not penalized in these implementations.

The regularization parameter \(\lambda\) controls the penalty on polynomial coefficients. A positive value can reduce coefficient instability, but it does not guarantee good generalization.

## 3. Underfitting and Overfitting

### 3.1 Underfitting

Underfitting occurs when a model is too restrictive to capture important structure in the underlying relationship.

Common characteristics include:

- High training error.
- High validation or test error.
- Strong systematic prediction error.
- Insufficient model flexibility.

For the experimental signal, a degree-1 polynomial is generally too simple to capture its nonlinear shape.

Increasing model complexity may reduce this systematic error.

A more flexible model is not automatically preferable. Its improvement must be assessed against data not used for fitting.

### 3.2 Overfitting

Overfitting occurs when a model captures patterns that are specific to its training observations rather than learning a relationship that generalizes.

Common characteristics include:

- Very low training error.
- Higher validation or test error.
- Sensitivity to changes in training observations.
- High prediction variance in unstable model configurations.

A high-degree polynomial can fit local fluctuations and noise. Its training error may continue to decline even when expected error on unseen observations increases.

Overfitting can be controlled through methods such as:

- Regularization.
- Reducing unnecessary model complexity.
- Increasing representative training data.
- Cross-validation.
- Early stopping for iterative learning algorithms.
- Feature selection based on appropriate validation procedures.

These methods address different causes of excessive model flexibility and are not interchangeable.

### 3.3 Generalization

Generalization is the ability of a model to make reliable predictions on observations not used during model fitting.

Training performance alone is insufficient to establish generalization.

A model can achieve low training error while performing poorly on new data.

An independent test set estimates performance on unseen observations. Cross-validation provides repeated estimates using different training and validation partitions.

The source programs use repeated independent training datasets and a common evaluation grid to estimate variation in fitted predictions.

## 4. Experimental Method

### 4.1 Repeated Training Samples

The experiments generate independent datasets from the same underlying signal.

Each dataset contains:

- 36 training observations by default.
- Input values sampled from the interval `[-1, 1]`.
- A nonlinear signal.
- Independent Gaussian observation noise.

Each candidate model is trained repeatedly.

The same evaluation grid is used for every fitted model, allowing predictions to be compared at identical input positions.

### 4.2 Estimating Squared Bias

At each evaluation-grid position, the implementation calculates the average prediction across repeated model fits.

It compares that average against the known noiseless signal.

The squared difference is an estimate of squared bias.

Averaging across the grid produces an aggregate squared-bias estimate.

### 4.3 Estimating Prediction Variance

For each grid position, the implementation calculates the squared deviations of individual fitted-model predictions from their average prediction.

The population variance across repeated fits is used:

\[
\widehat{Var}(\hat f(x)) =
\frac{1}{R}
\sum_{r=1}^{R}
\left(
\hat f_r(x)-\overline{\hat f}(x)
\right)^2
\]

Where:

- \(R\) is the number of repeated training samples.
- \(\hat f_r(x)\) is the prediction from repetition \(r\).
- \(\overline{\hat f}(x)\) is the average prediction.

The resulting values are averaged across evaluation-grid positions.

This measures sensitivity of the fitted procedure to changes in sampled training data.

### 4.4 Training Error

Training MSE compares fitted predictions with the observed training targets:

\[
MSE_{train} =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
\]

Training MSE is useful for detecting model fit, but it is not an unbiased estimate of future prediction error because the same observations were used to fit the model.

### 4.5 Expected Test Error

The programs estimate expected prediction error by adding the measured squared bias, prediction variance, and known observation-noise variance.

This provides a controlled estimate of expected performance against future noisy observations under the experiment's assumptions.

It should not be interpreted as an empirical test-set score from a separately sampled test dataset.

### 4.6 Generalization Gap

The reported generalization gap is:

\[
Gap =
ExpectedTestMSE -
AverageTrainingMSE
\]

It is a descriptive comparison between the estimated expected test error and the observed average training error.

A positive value indicates that the expected test error exceeds the measured training error.

The comparison can also be negative. Training and test quantities have different sampling and noise behavior, so a negative value does not imply that a model has achieved better-than-perfect generalization.

## 5. Python Implementation

The Python script provides a self-contained numerical experiment without third-party packages.

### 5.1 Main Components

| Component | Responsibility |
|---|---|
| `Point` | Immutable observation containing an input and target |
| `Evaluation` | Stores metrics and the risk classification |
| `PolynomialRegressor` | Fits and predicts using polynomial ridge regression |
| `solve_linear_system` | Solves the regularized normal equations |
| `generate_dataset` | Generates independently sampled training observations |
| `run_experiment` | Coordinates repeated model fitting and decomposition |
| `classify_model` | Applies a comparative underfitting and overfitting rule |

### 5.2 Numerical Implementation

Polynomial features are represented as:

\[
[1,x,x^2,\ldots,x^d]
\]

The script constructs the normal-equation matrix and right-hand side directly.

Gaussian elimination with partial pivoting solves the resulting linear system.

A small ridge penalty is applied to non-intercept coefficients to reduce numerical instability.

The implementation validates:

- Polynomial degree.
- Dataset size.
- Noise standard deviation.
- Finite observation values.
- Prediction input range.
- Model fitting state.
- Linear-system dimensions.
- Singular or unstable systems.

The normal-equation approach is suitable for the controlled, low-dimensional experiment. For larger or poorly conditioned problems, QR or singular-value decomposition is generally preferable.

### 5.3 Output

The Python program reports:

- Polynomial degree.
- Average training MSE.
- Estimated squared bias.
- Estimated prediction variance.
- Irreducible noise.
- Expected test MSE.
- Generalization gap.
- Comparative risk classification.

It also prints machine-readable JSON containing the results.

The default random seed makes the experiment reproducible.

## 6. JavaScript Implementation

The JavaScript file provides an event-driven implementation of the experiment.

It is designed for a Node.js runtime and uses only standard-library modules.

### 6.1 Event-Based Architecture

The experiment emits lifecycle events:

| Event | Meaning |
|---|---|
| `experiment:started` | The model-comparison workflow has begun |
| `candidate:started` | Repeated evaluation of a model degree has started |
| `candidate:completed` | Metrics for a candidate have been calculated |
| `experiment:completed` | All candidates have been evaluated and ranked |

This separates evaluation execution from reporting and makes the workflow extensible.

Other components could listen for candidate-completion events to record metrics or update an operational monitoring interface.

### 6.2 Runtime and State Management

`BiasVarianceExperiment` owns the experiment configuration and random source.

`PolynomialRegressor` owns the state of one fitted model.

A model cannot produce predictions until fitting has completed.

The experiment yields periodically to the Node.js event loop using `setImmediate`, preventing the complete evaluation workload from monopolizing a single synchronous execution interval.

### 6.3 Reproducibility and Validation

The custom pseudo-random generator provides deterministic input sampling for a given seed.

Configuration validation rejects:

- Duplicate polynomial degrees.
- Invalid sample counts.
- Excessive model degrees.
- Invalid noise values.
- Invalid regularization values.
- Non-finite predictions.

The event system provides a separation between the numerical evaluation process and downstream reporting or monitoring responsibilities.

The experiment remains computationally intensive because repeated regression fitting and prediction occur for every candidate.

## 7. C++17 Implementation

The C++ program models an experimental evaluation engine for a predictive-service team.

### 7.1 Domain Model

| Type | Responsibility |
|---|---|
| `Point` | Represents an observed input-target pair |
| `ExperimentConfig` | Controls repetitions, sample count, grid size, and seed |
| `Metrics` | Stores the statistical evaluation results |
| `RiskLevel` | Represents underfitting, overfitting, or acceptable generalization |
| `PolynomialRegressor` | Fits and evaluates a polynomial model |
| `DatasetGenerator` | Produces independent synthetic training samples |
| `ExperimentRunner` | Coordinates candidate evaluation |

### 7.2 Memory and Performance

The implementation stores predictions for every candidate, repetition, and evaluation-grid position.

If:

- \(R\) is the number of repetitions.
- \(K\) is the number of candidates.
- \(G\) is the evaluation-grid size.
- \(D\) is the maximum polynomial degree.
- \(N\) is the training sample count.

The approximate fitting cost using normal equations and Gaussian elimination is:

\[
O(RK(ND^2+D^3))
\]

Prediction and decomposition add approximately:

\[
O(RKGD)
\]

Stored prediction values require:

\[
O(RKG)
\]

The implementation imposes upper limits on command-line repetition and sample-count parameters to reduce accidental excessive resource consumption.

### 7.3 Failure Handling

The program handles:

- Invalid command-line input.
- Out-of-range polynomial degree.
- Insufficient training samples.
- Non-finite values.
- Prediction before fitting.
- Singular regression systems.
- Invalid noise parameters.

Exceptions are reported to standard error, and the program exits with a failure status.

The implementation uses value-based structures and standard containers rather than external numerical libraries.

## 8. Java 17 Enterprise Implementation

The Java program treats model evaluation as a controlled enterprise workflow.

It separates numerical evaluation from the policy decision that determines whether a candidate model can be approved.

### 8.1 Domain Types

| Type | Responsibility |
|---|---|
| `ExperimentConfig` | Immutable experiment configuration |
| `Candidate` | Model identity, degree, and regularization |
| `Metrics` | Immutable statistical measurements |
| `PolicyDecision` | Risk assessment and approval result |
| `ModelEvaluation` | Candidate, metrics, decision, and lifecycle state |
| `EvaluationBatch` | Immutable collection of evaluated candidates |

Java records make configuration and evaluation results explicit value types.

This reduces accidental modification of evaluation results after they have been produced.

### 8.2 Interfaces and Services

`RegressionEstimator` defines the fitting and prediction contract.

`GeneralizationPolicy` separates statistical assessment from approval policy.

`EvaluationService` performs repeated model evaluation.

`ModelGovernanceService` coordinates the candidate lifecycle and policy assessment.

`DefaultGeneralizationPolicy` evaluates:

- Expected test error relative to the best candidate.
- Average training error.
- Prediction variance.
- Number of training observations per model parameter.

### 8.3 Lifecycle Management

Each model progresses through explicit states:

`CREATED → EVALUATING → EVALUATED → APPROVED`

A candidate that fails policy checks progresses to:

`CREATED → EVALUATING → EVALUATED → REJECTED`

Invalid transitions are rejected.

Approval is not the same as statistical measurement. The evaluation service calculates metrics, while the policy determines whether the candidate satisfies the organization's acceptance conditions.

A model can therefore be evaluated successfully but rejected from approval.

### 8.4 Enterprise Implications

A production-oriented extension would persist:

- Model identifier and version.
- Dataset identifier and fingerprint.
- Training and validation configuration.
- Evaluation metrics.
- Evaluation software version.
- Policy version.
- Approval decision and rationale.
- Date and identity of the responsible reviewer.

This information allows model decisions to be reproduced and audited.

The demonstration uses in-memory data structures and a simulated dataset. It does not implement a persistent enterprise registry.

## 9. PostgreSQL Implementation

The PostgreSQL script provides a relational workflow for recording experiments and evaluating candidate eligibility.

It stores evaluation results rather than training regression models inside the database.

### 9.1 Relational Structure

The principal relationship is:

`dataset_profiles`
→ `experiment_runs`
→ `evaluation_metrics`
→ `model_decisions`

Candidate definitions are stored separately in `model_candidates`.

`experiment_policies` contains the decision thresholds associated with an experiment.

Each evaluation record connects one experiment with one candidate.

### 9.2 Integrity Constraints

The schema uses:

- Primary keys to identify records.
- Foreign keys to maintain relationships.
- Unique constraints to prevent duplicate experiment-candidate evaluations.
- Check constraints to reject invalid metric values and workflow states.
- Generated columns for derived metrics.
- Triggers to enforce workflow behavior.

The expected test MSE is stored as a generated expression:

`bias_squared + prediction_variance + irreducible_noise`

The generalization gap is also generated rather than manually maintained.

This prevents stored derived values from becoming inconsistent with their component measurements.

### 9.3 Experiment State Management

Experiments have four states:

| State | Meaning |
|---|---|
| `DRAFT` | Experiment has been registered |
| `RUNNING` | Evaluation metrics may be inserted |
| `COMPLETED` | Decisions have been recorded |
| `FAILED` | Evaluation has failed |

The permitted transitions are:

`DRAFT → RUNNING`

`RUNNING → COMPLETED`

`RUNNING → FAILED`

The state trigger rejects invalid transitions.

The evaluation trigger ensures that metrics are inserted only while the associated experiment is running.

It also checks that the reported repetition count matches the dataset configuration.

### 9.4 Eligibility Processing

The `complete_experiment` procedure compares candidate models using the stored policy.

It calculates the best expected test MSE and the best average training MSE.

Candidates are classified using the following comparative rules:

- A model with substantially worse test error and substantially worse training error is classified as underfitting.
- A model with substantially worse test error but lower training error is classified as overfitting.
- A model within the expected test-error threshold and variance limit is classified as generalizing well.

Approval also requires sufficient training observations per model parameter.

The thresholds are stored in `experiment_policies`, allowing separate experiments to use different governance policies.

The `fn_deployment_eligibility` function returns accepted candidates only when the associated experiment has completed.

### 9.5 Database Reporting

The script provides:

- Dataset retrieval.
- Experiment-state retrieval.
- Bias-variance decomposition validation.
- Complexity-based metric aggregation.
- Full candidate evaluation reports.
- Decision counts by risk category.
- Deployment eligibility queries.

Indexes support common access patterns involving experiment state, candidate complexity, evaluation results, and decision status.

The transaction demonstrates that an invalid attempt to restart a completed experiment is rejected without changing its final state.

## 10. Interpreting the Comparative Results

The four model degrees represent different levels of flexibility.

| Model degree | Parameter count | Expected behavior |
|---:|---:|---|
| 1 | 2 | High bias is likely |
| 2 | 3 | Limited flexibility |
| 5 | 6 | Greater ability to represent nonlinear structure |
| 9 | 10 | Higher sensitivity to sample variation is possible |

The exact numeric results depend on the sample-generation process, random seed, sample size, repetition count, and regularization.

The expected pattern is more important than any fixed numerical outcome:

- A simple model can have high bias and high training error.
- A more flexible model can reduce bias.
- Excessive flexibility can increase prediction variance.
- Expected test error may stop improving even when training error continues to decline.
- A model with the lowest training error need not have the lowest expected test error.

The model-selection objective is not to minimize training error. It is to select a model with acceptable predictive performance on unseen data.

The risk classification in the source files is a comparative demonstration rule. It is not a universal statistical test for underfitting or overfitting.

## 11. Relationship Between the Five Implementations

| Implementation | Primary perspective | Main mechanism |
|---|---|---|
| Python | Numerical experimentation | Repeated fitting and bias-variance decomposition |
| JavaScript | Event-driven evaluation | Candidate events and asynchronous execution |
| C++17 | Computational experiment engine | STL-based model evaluation and explicit numerical structures |
| Java 17 | Enterprise governance | Immutable domain types, policies, and lifecycle transitions |
| PostgreSQL | Relational experiment management | Constraints, triggers, procedures, reporting, and eligibility |

All five implementations use the same underlying statistical distinction between bias, variance, and irreducible noise.

Their engineering responsibilities differ.

The numerical programs calculate model behavior. The Java implementation adds an approval workflow. The SQL implementation stores measured results and enforces experiment-level integrity.

## 12. Important Limitations and Practical Considerations

### 12.1 Synthetic Data

The examples use a known nonlinear function with controlled noise.

Real datasets may contain:

- Missing values.
- Outliers.
- Measurement errors.
- Nonlinear relationships that change over time.
- Correlated observations.
- Unequal noise levels.
- Distribution shifts between training and deployment.

Results from the synthetic experiment should not be generalized automatically to these conditions.

### 12.2 Evaluation Data Leakage

Training, validation, and test data must have clearly defined roles.

Repeatedly selecting models using the same validation set can overfit the model-selection process itself.

The test set should not repeatedly influence feature or complexity decisions if it is intended to provide an independent final evaluation.

### 12.3 Distribution Shift

The experimental decomposition assumes that repeated training datasets and future observations follow the same underlying data-generating process.

A deployment environment with a different input or target distribution can produce different prediction behavior.

The measured bias and variance estimates do not establish robustness to distribution shift.

### 12.4 Numerical Conditioning

Polynomial features can become poorly conditioned as degree increases.

Normal equations square the condition number of the design matrix, which can amplify numerical error.

The examples limit the maximum degree and use ridge regularization and partial pivoting. More demanding numerical workloads should use stable QR or singular-value decomposition algorithms.

### 12.5 Policy Thresholds

A model-governance threshold is an operational decision, not a mathematical law.

The comparative ratios and variance limits used in the examples are demonstration settings.

Production thresholds should be justified by the business risk, dataset size, validation design, cost of prediction errors, and model-use context.

### 12.6 Generalization Is Not Guaranteed by Approval

Approval means that a candidate satisfies the configured evaluation policy.

It does not guarantee future predictive accuracy, absence of bias in every subgroup, operational stability, or suitability for every deployment environment.

Monitoring is required after deployment because model performance can deteriorate as data and operating conditions change. 
