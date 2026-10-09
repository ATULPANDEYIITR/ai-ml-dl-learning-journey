# Polynomial Regression: Polynomial Features, Nonlinear Relationships, and Overfitting

## Scope

Polynomial regression models a nonlinear relationship between an input variable and a target by expanding the input into polynomial features and then fitting a linear model to those transformed features.

The central distinction is important:

- The relationship between the original input `x` and prediction `y` can be curved.
- The regression remains linear with respect to its learned coefficients.
- Polynomial degree controls the flexibility of the curve.
- Increasing degree can reduce training error while increasing sensitivity to noise.
- Overfitting occurs when the model captures training-specific variation that does not generalize to unseen observations.

The six deliverables use different technical perspectives. The Python implementation provides a complete numerical implementation and model-selection workflow. The JavaScript implementation emphasizes event-driven processing and asynchronous model selection. The C++ implementation builds a computational case study with explicit matrix operations and numerical controls. The Java implementation models an enterprise-oriented model-governance workflow. The PostgreSQL implementation represents the same domain relationally and enforces model-quality policies at the database layer.

## Mathematical Foundation

A degree-two polynomial regression has the form:

`y = β₀ + β₁x + β₂x² + ε`

A degree-three model extends this to:

`y = β₀ + β₁x + β₂x² + β₃x³ + ε`

The important transformation is the creation of features such as `x`, `x²`, and `x³`.

For a single feature, the design matrix for a degree-three model can be represented conceptually as:

`X = [1, x, x², x³]`

The coefficients are still estimated as a vector. The nonlinear behavior comes from the transformed representation of the input.

The intercept is represented by the constant feature `1`. This allows the fitted curve to move vertically rather than being forced through the origin.

## Polynomial Features

Polynomial feature construction changes the representation of the input before regression.

For degree four, a value such as `x = 3` becomes:

`[1, 3, 9, 27, 81]`

The Python `PolynomialFeatures` class constructs these powers explicitly. The JavaScript, C++, and Java implementations independently construct the same mathematical representation using language-specific matrix and collection mechanisms.

The degree determines the maximum power:

| Degree | Features for one variable |
| --- | --- |
| 1 | `1, x` |
| 2 | `1, x, x²` |
| 3 | `1, x, x², x³` |
| 4 | `1, x, x², x³, x⁴` |
| 10 | `1, x, x², ..., x¹⁰` |

Higher degree does not automatically mean a better model. It increases the number of adjustable coefficients and therefore increases model flexibility.

## Nonlinear Relationships

Polynomial regression is useful when the effect of an input changes with the input's magnitude.

The implementations use energy demand and outdoor temperature as the main domain example. Energy demand does not necessarily change at a constant rate as temperature changes.

At low temperatures, heating requirements can increase demand. At moderate temperatures, demand can fall because less heating or cooling is required. At high temperatures, cooling requirements can increase demand again.

A polynomial can represent this curvature more effectively than a simple straight-line model.

The synthetic relationship used throughout the implementations contains quadratic and cubic terms:

`demand = 420 - 13T + 0.72T² - 0.012T³ + noise`

This is deliberately different from a simple linear relationship because the effect of temperature changes across the observed range.

## Linear Regression Versus Polynomial Regression

A linear model assumes:

`y = β₀ + β₁x`

Its slope is constant.

A polynomial model such as:

`y = β₀ + β₁x + β₂x²`

has a slope that changes with `x`.

For the quadratic equation, the derivative is:

`dy/dx = β₁ + 2β₂x`

Therefore, the relationship can become steeper or flatter as the input changes.

This distinction explains why polynomial regression can model curvature without requiring a fundamentally different coefficient-estimation method.

## Fitting the Polynomial Model

The implementations use the normal-equation form of least squares:

`β = (XᵀX)⁻¹Xᵀy`

The design matrix already contains the polynomial features, so the fitting process operates on the transformed feature matrix.

This approach is useful educationally because it makes the relationship between feature construction and regression coefficients explicit.

The implementations also detect singular or numerically unstable matrices. This matters because high-degree polynomial features can become strongly correlated, especially when raw feature values are large.

Production machine-learning libraries commonly use numerically more stable decomposition methods rather than directly computing a matrix inverse.

## Feature Scaling

Polynomial expansion can create very different feature magnitudes.

For example, if `x = 100`, then:

- `x² = 10,000`
- `x³ = 1,000,000`
- `x⁶ = 1,000,000,000,000`

These magnitudes can make the numerical fitting process unstable.

The Python, JavaScript, C++, and Java implementations therefore support standardization before polynomial expansion:

`z = (x - μ) / σ`

Scaling changes the numerical representation but does not remove the underlying relationship between the input and target.

The important workflow is to fit the scaler using training data and then apply the already-fitted transformation to validation and test data. Computing scaling statistics separately on the test set would leak information from the evaluation set into the modeling process.

## Underfitting

A model underfits when its representation is too simple to capture the relationship present in the data.

A degree-one model fitted to a strongly curved relationship may show:

- relatively high training error
- relatively high test error
- systematic residual patterns
- poor representation of curvature

Increasing the degree can initially improve both training and test performance when the original model is too restrictive.

## Overfitting

Overfitting occurs when additional model flexibility begins fitting noise rather than the underlying relationship.

A high-degree polynomial can pass very close to many training observations while producing unstable predictions between or beyond those observations.

A common pattern is:

`training error ↓`

while:

`test error ↑`

The Python script explicitly compares degrees from one through ten. The JavaScript program compares several degrees after the final model is trained. The C++ case study compares multiple degrees as part of its model evaluation. The Java governance layer explicitly treats a large train-test RMSE gap as a model-quality failure condition.

The SQL model stores separate training and test metrics so this distinction can be evaluated through relational queries.

## Degree Selection

Polynomial degree should not be selected simply because it produces the smallest training error.

The implementations use held-out evaluation and cross-validation.

Cross-validation divides the available training data into multiple folds. Each fold becomes a validation set while the remaining folds are used for fitting.

The resulting validation errors provide a more reliable estimate of how a degree is likely to generalize than training error alone.

The Python implementation uses five-fold cross-validation to compare candidate degrees. The JavaScript implementation performs the same conceptual evaluation while emitting asynchronous events for each candidate. The Java implementation represents cross-validation as a reusable service operation.

The SQL implementation stores fold-level RMSE and MAE in `cross_validation_results`, making it possible to calculate mean and standard deviation of validation performance directly in PostgreSQL.

## Regularization

The implementations support ridge-style regularization.

The objective becomes:

`min ||y - Xβ||² + λΣβⱼ²`

The intercept is excluded from the penalty.

A positive `λ` discourages very large coefficients. This is useful for polynomial models because high-degree terms can otherwise acquire large coefficients in an attempt to fit small training fluctuations.

Regularization does not automatically solve overfitting. The regularization strength itself must be selected using validation procedures.

A value that is too small may provide little protection against variance. A value that is too large can flatten the model and produce underfitting.

## Train-Test Error Gap

The train-test RMSE gap is useful as a diagnostic.

The Java implementation's governance policy includes a maximum allowable gap:

`test RMSE - train RMSE`

A large positive difference indicates that the model performs substantially better on the training observations than on unseen observations.

This is not a universal mathematical proof of overfitting, because data distribution changes, noise, sample size, and sampling methodology can also influence the gap. It is a governance signal rather than a standalone diagnosis.

## Residuals

A residual is:

`residual = actual - predicted`

The Python implementation calculates:

- mean residual
- residual standard deviation
- maximum absolute residual

A residual mean close to zero is useful but insufficient.

A polynomial model can have an average residual near zero while still exhibiting a systematic pattern. For example, residuals that are mostly positive at one temperature range and negative at another indicate that the model may be missing structure.

Residual analysis should therefore be performed against the original feature, predicted value, time where relevant, and other important operational dimensions.

## C++ Case Study

The C++ program treats polynomial regression as a computational forecasting system for energy demand.

Its main components are:

- `MatrixOperations`, which performs transpose, multiplication, identity-matrix creation, and inversion.
- `StandardScaler`, which normalizes the input feature.
- `PolynomialFeatureBuilder`, which creates the powers of the feature.
- `PolynomialRegression`, which combines the transformed design matrix, ridge penalty, coefficient estimation, and prediction.
- `EnergyObservation`, which represents the domain record.
- The main workflow, which compares polynomial degrees, evaluates regularization, generates predictions, and explicitly demonstrates extrapolation.

The implementation uses partial pivoting during matrix inversion. This is a numerical safeguard because polynomial design matrices can become ill-conditioned.

The program also uses C++ standard-library containers and algorithms instead of external numerical libraries, making the case study self-contained.

## JavaScript Implementation

The JavaScript program emphasizes an event-driven model-selection workflow.

`PolynomialRegression` owns model state and prediction behavior. `StandardScaler` manages feature normalization. `PolynomialFeatures` constructs the transformed representation.

`ModelSelectionPipeline` provides event-driven processing through:

- `selectionStarted`
- `candidateEvaluated`
- `selectionCompleted`

The asynchronous loop uses `setImmediate` so model-candidate evaluation can participate in an event-driven Node.js workflow rather than being represented as a purely synchronous sequence.

This is useful for applications where model selection is part of a larger service pipeline that reports progress, records events, or triggers downstream operations.

The JavaScript implementation also includes explicit validation errors rather than silently accepting invalid model configurations.

## Java Enterprise Model

The Java implementation separates numerical computation from governance rules.

`ModelPolicy` represents acceptance criteria such as:

- maximum polynomial degree
- maximum test RMSE
- minimum test R²
- maximum train-test RMSE gap
- required regularization
- required approval count

`ModelGovernanceService` evaluates a model against these rules.

`ModelCandidate` records the result of candidate evaluation, including degree, cross-validation performance, training metrics, testing metrics, and status.

This separation is important in enterprise systems because model fitting and model approval are different responsibilities.

A statistically valid model is not automatically a production-approved model. An organization may impose additional constraints concerning error thresholds, model complexity, validation evidence, and review requirements.

## SQL Data Model

The PostgreSQL implementation models the domain through several related tables.

`regression_datasets` identifies a modeling dataset and records its feature and target meaning.

`observations` stores feature values, target values, and dataset split membership. The split constraint restricts values to `TRAIN`, `TEST`, or `VALIDATION`.

`polynomial_models` stores model degree, regularization, feature scaling, and lifecycle status.

`model_coefficients` stores coefficients by polynomial power. This allows a degree-two model to be represented as powers `0`, `1`, and `2`, while a higher-degree model can store additional terms.

`model_evaluations` stores MAE, RMSE, and R² separately for training, validation, testing, and cross-validation evaluations.

`cross_validation_results` stores individual fold results instead of only storing a final average. This preserves the evidence needed to inspect validation variability.

`model_governance_policies` represents production acceptance rules in database data rather than hard-coding every threshold into an application.

`model_reviews` records reviewer decisions. This separates technical evaluation metrics from human governance decisions.

## Database-Level Governance

The SQL trigger `enforce_model_approval_rules()` demonstrates an important distinction between calculating a model and approving a model.

A model cannot simply change to `APPROVED` because an application requests the state transition.

The trigger checks:

- maximum permitted polynomial degree
- test RMSE
- test R²
- train-test RMSE gap
- number of approvals
- unresolved requested changes

If any required condition fails, PostgreSQL raises an exception and rejects the status transition.

This provides database-level enforcement against accidental or unauthorized approval states.

The transaction demonstrating the high-degree candidate's approval is rolled back because the candidate violates the stored governance conditions.

## Model Reviews and Approval Evidence

The SQL model treats reviewer decisions separately from evaluation metrics.

An evaluation answers questions such as:

`How well did the model perform?`

A review answers questions such as:

`Has an authorized reviewer examined the model evidence?`

These are different concerns.

A model can have strong numerical performance and still require review before production use.

The example quadratic model receives two approvals. The high-degree candidate receives a requested-change review because its test performance is substantially worse than its training performance.

## Extrapolation

Polynomial regression is particularly sensitive to extrapolation.

Suppose the training data covers temperatures from `-5°C` to `40°C`. A prediction at `70°C` is outside the observed range.

The fitted polynomial may produce a mathematically valid value, but the model has not learned from observations in that region.

The Python, JavaScript, C++, and Java implementations explicitly demonstrate predictions beyond the training range.

The SQL implementation represents this concern in `prediction_requests`, assigning `EXTRAPOLATION_WARNING` when a requested feature lies outside the observed feature range.

Extrapolation warnings are especially important for high-degree polynomials because powers can grow rapidly outside the training interval.

## Numerical Stability

Polynomial regression can become numerically difficult as degree increases.

The powers of a feature can differ by many orders of magnitude. The columns of the design matrix can also become highly correlated.

Directly calculating `(XᵀX)⁻¹` can therefore amplify floating-point errors.

The supplied implementations use partial pivoting and feature scaling to reduce some of these problems. These techniques make the demonstrations more robust, but they do not eliminate the fundamental numerical limitations of high-degree polynomial regression.

For production numerical systems, stable matrix decompositions such as QR or singular-value decomposition are generally preferable to explicitly computing a matrix inverse.

## Performance

For a single input variable and degree `d`, polynomial feature construction requires approximately `O(nd)` feature-generation work for `n` observations.

The normal-equation approach creates a matrix with approximately `(d + 1) × (d + 1)` elements. Matrix multiplication and inversion become increasingly expensive as degree increases.

The practical limitation is therefore not only statistical. A high-degree polynomial can also increase computational cost and numerical complexity.

For one feature, the number of polynomial terms is manageable for moderate degrees. With multiple original features, the feature count can grow much faster because interaction terms become possible.

For example, a multivariate polynomial feature expansion can include terms such as:

`x₁x₂`

`x₁²x₂`

`x₁x₂²`

This feature explosion can make polynomial regression expensive and difficult to interpret.

## Security and Reliability Considerations

Polynomial regression is not normally a security mechanism. Model governance therefore needs to protect the surrounding workflow rather than treating the regression equation itself as a security control.

Important production concerns include:

- Validate all input values before they enter polynomial expansion.
- Reject non-finite values.
- Keep scaling statistics associated with the exact model version that used them.
- Prevent unauthorized modification of coefficients.
- Record model evaluation evidence with immutable or auditable timestamps where required.
- Keep approval decisions separate from numerical metrics.
- Detect predictions outside the training feature range.
- Avoid silently accepting a model whose test metrics were calculated using training data.
- Prevent evaluation leakage when constructing train, validation, and test datasets.
- Monitor production residuals after deployment because a previously suitable relationship can change.

The SQL constraints provide database-level validation for many structural rules, while the application implementations provide runtime validation before numerical operations.

## Common Failure Modes

### Treating polynomial regression as inherently nonlinear in its parameters

Polynomial regression is nonlinear in the original feature but linear in the coefficients.

This distinction matters because the fitting problem remains a linear least-squares problem after feature transformation.

### Selecting degree using training error

Training error almost always benefits from additional flexibility. It is therefore insufficient for choosing the final degree.

Cross-validation or a separate validation dataset should be used for model selection.

### Ignoring scaling

Large powers of unscaled inputs can create poor numerical conditioning.

Scaling the input before generating polynomial features can improve numerical behavior.

### Penalizing the intercept without justification

The intercept represents the baseline level of the target. Penalizing it can introduce an unnecessary bias.

The implementations therefore exclude the intercept from ridge regularization.

### Assuming a high R² proves the model is good

R² can be high on training data even when a model generalizes poorly.

The examples therefore use test RMSE, test R², cross-validation RMSE, and train-test error differences together.

### Trusting extrapolated predictions

A polynomial curve can change rapidly outside the observed range.

An operational system should identify extrapolation explicitly instead of presenting such values as equally reliable predictions.

## Practical Interpretation

The most useful interpretation of polynomial regression is not that a high-degree equation is better than a straight line.

The important modeling decision is whether the additional curvature is supported by evidence.

A degree-one model is appropriate when the relationship is approximately linear.

A degree-two or degree-three model can represent moderate curvature.

Higher degrees should require stronger validation evidence because they increase flexibility and numerical sensitivity.

Regularization can make a flexible model more stable, but it does not replace appropriate degree selection or sound validation.

The examples therefore treat degree, regularization, validation performance, test performance, residual behavior, and extrapolation range as connected parts of one modeling decision.

## Relationship Between the Five Implementations

The Python program is the most complete numerical teaching implementation. It builds polynomial features, fits models, evaluates them, performs cross-validation, applies regularization, examines residuals, and demonstrates a realistic nonlinear dataset.

The JavaScript program presents polynomial regression as an event-driven application workflow. Its model-selection pipeline emits events while evaluating candidate degrees, making the computational process suitable for a Node.js service architecture.

The C++ program focuses on computational implementation and numerical mechanics. Matrix operations, scaling, feature construction, and ridge regression are represented explicitly to expose the internal mechanics of the fitting process.

The Java program adds enterprise governance. Numerical metrics are connected to explicit policy rules that determine whether a candidate model can be approved.

The SQL program treats the model lifecycle as persistent relational data. It stores observations, models, coefficients, metrics, validation folds, policies, reviewers, reviews, and prediction requests while enforcing approval constraints at the database layer.

These perspectives are deliberately complementary. Polynomial feature construction is the mathematical foundation, nonlinear behavior is the modeling objective, and overfitting is the central generalization risk. The surrounding implementation layers determine how that model is validated, governed, persisted, and used operationally.
