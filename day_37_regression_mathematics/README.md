# Regression Mathematics: Least Squares, Residuals, MSE, R², and Adjusted R²

## Scope

This repository models ordinary least squares regression as a complete mathematical workflow rather than treating regression as a single prediction function.

The implementations use an operational delivery-time problem in which delivery duration is estimated from processing effort and order size. The same mathematical objects are represented differently in each implementation:

- Python provides a progressive numerical implementation with descriptive functions, ordinary least squares, diagnostics, prediction, validation, and edge-case handling.
- JavaScript implements a regression engine with an event-driven architecture, matrix operations, validation, prediction, and model comparison.
- C++ presents a repository-independent enterprise-style model-governance case study in which regression quality is evaluated against explicit deployment policy thresholds.
- Java represents the domain through records, enums, services, immutable collections, validation rules, exceptions, and a governance decision model.
- PostgreSQL stores observations, fitted coefficients, predictions, residuals, and regression metrics while exposing the underlying mathematical calculations through relational queries and constraints.

The central distinction is between fitting a model, measuring its errors, measuring explanatory power, and evaluating whether additional model complexity is justified.

## Regression Model

The implementations use the multiple linear regression form

`y_i = β₀ + β₁x₁ᵢ + β₂x₂ᵢ + εᵢ`

where:

- `y_i` is the observed response.
- `x₁ᵢ` and `x₂ᵢ` are predictor values.
- `β₀` is the intercept.
- `β₁` and `β₂` are predictor coefficients.
- `εᵢ` is the unexplained error for observation `i`.

For the operational scenario, `y` represents delivery duration, `x₁` represents processing hours, and `x₂` represents item count.

The fitted model replaces unknown coefficients with estimates:

`ŷ_i = β̂₀ + β̂₁x₁ᵢ + β̂₂x₂ᵢ`

The difference between the observed and fitted response is the residual:

`e_i = y_i - ŷ_i`

A fitted model therefore produces two fundamentally different outputs: a prediction for each observation and an observation-specific error describing how far that prediction is from reality.

## Least Squares

Ordinary least squares chooses coefficients that minimize the sum of squared residuals:

`SSE = Σ(y_i - ŷ_i)²`

Squaring the residuals prevents positive and negative errors from cancelling and gives larger errors greater influence.

In matrix notation, the model is

`y = Xβ + ε`

and the ordinary least squares estimator is

`β̂ = (XᵀX)⁻¹Xᵀy`

when the required inverse exists.

The Python, JavaScript, C++, and Java implementations explicitly construct the design matrix with an intercept column and calculate the normal-equation solution. The SQL implementation builds the required cross-products from relational aggregates and solves the small three-parameter system.

The normal equation is mathematically direct but not the preferred numerical method for large or poorly conditioned production systems. Explicitly forming `XᵀX` squares the condition number of the design matrix. QR decomposition or singular value decomposition is normally preferable when numerical stability is important.

## Residuals

A residual is the observed response minus its fitted response:

`e_i = y_i - ŷ_i`

A positive residual means the model predicted too little. A negative residual means the model predicted too much.

The implementations retain residuals at observation level because aggregate model metrics cannot reveal which observations caused the error.

The Python implementation reports residual summaries and identifies observations with large standardized residuals. The JavaScript implementation exposes residuals as part of the immutable fitted-model result. The C++ governance engine evaluates the largest absolute residual against a deployment threshold. Java uses residual collections to identify observations exceeding an operational tolerance. SQL exposes individual residuals and squared residuals through diagnostic queries.

With an intercept, the residuals have an important algebraic property: their sum is approximately zero, subject to floating-point arithmetic. This does not mean that every individual residual is small.

## Sum of Squared Errors

The residual sum of squares is

`SSE = Σe_i²`

SSE is an absolute measure of unexplained squared variation for a fitted dataset.

A lower SSE indicates less squared residual error for the same observations, but SSE alone should not be used to compare models with substantially different response scales or datasets.

The implementations calculate SSE directly from residuals rather than treating it as an abstract statistic. This makes the relationship between the fitted values and the error metric explicit.

## Mean Squared Error

There are two related quantities that should not be confused.

For predictive evaluation, the mean squared prediction error can be written as

`MSE_predictive = SSE / n`

where `n` is the number of observations.

For estimating the regression error variance, the residual degrees of freedom are used:

`MSE_error = SSE / (n - p - 1)`

where `p` is the number of predictors excluding the intercept.

The Python, C++, Java, and SQL implementations expose the residual-degrees-of-freedom version as model-error MSE while also using the `SSE / n` form when deriving RMSE.

MSE is expressed in squared response units. If delivery duration is measured in hours, MSE is measured in hours squared.

## RMSE

Root mean squared error is

`RMSE = √(SSE / n)`

for predictive evaluation.

Taking the square root returns the metric to the original response scale. An RMSE of approximately `1.5` hours can therefore be interpreted in terms of typical prediction-error magnitude more naturally than an MSE of `2.25` hours squared.

RMSE is not interchangeable with the residual standard error. The denominator and statistical interpretation must be considered before comparing the two.

## R²

R² compares the fitted model against an intercept-only baseline.

First calculate total sum of squares:

`SST = Σ(y_i - ȳ)²`

Then

`R² = 1 - SSE/SST`

An R² of `0.80` means the fitted model reduces squared error by approximately 80 percent relative to predicting every observation with the response mean, under the standard regression interpretation.

R² is not the percentage accuracy of the model. It does not mean that individual predictions are 80 percent correct.

R² also does not guarantee that a model is useful for prediction. A model can have a high R² while producing problematic errors for particular observations, relying on influential observations, or violating assumptions relevant to the intended use.

When the response has zero variance, `SST` is zero and the conventional R² formula is undefined. The Python and JavaScript implementations explicitly reject that case.

## Adjusted R²

Adding predictors to an ordinary least squares model cannot reduce the training R². Even a weak predictor can therefore increase or leave R² unchanged.

Adjusted R² introduces a degrees-of-freedom penalty:

`Adjusted R² = 1 - (1 - R²)(n - 1)/(n - p - 1)`

The penalty becomes relevant when comparing models with different predictor counts.

The operational examples compare a reduced model containing processing hours with a larger model containing both processing hours and item count. The full model can achieve a higher R² because it has another degree of explanatory freedom. Adjusted R² asks whether the improvement remains useful after accounting for that additional predictor.

Adjusted R² is not a universal model-selection criterion. It is one diagnostic among several and should be interpreted with prediction error, domain validity, residual behavior, and model purpose.

## Relationship Between SST, SSR, and SSE

For ordinary least squares with an intercept,

`SST = SSR + SSE`

where

`SSR = Σ(ŷ_i - ȳ)²`

and

`SSE = Σ(y_i - ŷ_i)²`.

This decomposition explains the relationship between R² and the fitted model.

Because

`R² = SSR/SST`

and

`R² = 1 - SSE/SST`

the statistic describes how much of the total response variation is represented by the fitted component relative to the total variation.

The Python implementation explicitly calculates all three quantities and reports the numerical decomposition error caused by floating-point arithmetic.

The SQL implementation performs the same decomposition using relational aggregation.

## Python Implementation

The Python program is structured around a small `RegressionResult` data class and a standard-library matrix implementation.

`simple_linear_regression()` demonstrates the closed-form single-predictor equations:

`β̂₁ = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)`

and

`β̂₀ = ȳ - β̂₁x̄`.

The more general `fit_ols()` function constructs an intercept-inclusive design matrix and applies

`β̂ = (XᵀX)⁻¹Xᵀy`.

The matrix implementation uses Gauss-Jordan elimination with partial pivoting. Partial pivoting is important because blindly dividing by a small pivot can amplify floating-point errors.

The program also includes prediction, residual summaries, leverage calculations, standardized residuals, model comparison, variance decomposition, input validation, zero-variance handling, and perfect-collinearity detection.

The leverage calculation uses the diagonal of the hat matrix:

`H = X(XᵀX)⁻¹Xᵀ`.

High leverage describes unusual predictor configurations. It is different from a large residual. An observation can have high leverage without having a large residual and can have a large residual without unusually high leverage.

## JavaScript Implementation

The JavaScript implementation uses a `RegressionEngine`, `RegressionModel`, and `RegressionEventBus`.

The event-driven structure is deliberate. A fitted regression model can be treated as an event-producing analytical component in a Node.js service. Consumers can subscribe to `model:fitted` without embedding monitoring logic inside the numerical solver.

The fitted result contains coefficients, predictions, residuals, SSE, MSE, RMSE, R², adjusted R², observation count, and predictor count.

`RegressionModel.predict()` separates inference from training. New observations are evaluated against already-fitted coefficients rather than silently modifying the model.

The implementation also validates finite numerical input and catches singular design matrices. JavaScript uses IEEE-754 double-precision numbers through the `Number` type, so numerical scaling matters when predictor magnitudes differ substantially.

## C++ Case Study

The C++ program treats regression as part of an operational model-governance system.

`RegressionEngine` performs matrix construction, ordinary least squares estimation, prediction, and metric calculation.

`MatrixOperations` isolates numerical operations such as transpose, multiplication, identity-matrix creation, and Gauss-Jordan inversion. Partial pivoting is used when selecting matrix pivots.

`GovernanceEngine` applies explicit deployment thresholds for R², adjusted R², RMSE, and maximum absolute residual. This distinguishes mathematical fitting from organizational model acceptance.

The case study demonstrates that a model can be numerically fitted successfully but still fail an operational policy.

The program also deliberately constructs a perfectly collinear predictor set. If one predictor is an exact multiple of another, the design matrix does not contain enough independent information to estimate unique coefficients through the inverse of `XᵀX`. The program rejects this condition rather than silently producing misleading coefficients.

The C++ comments also distinguish the educational normal-equation implementation from a production numerical solver. QR or SVD would generally be preferred when stability is more important than demonstrating the underlying normal-equation mathematics.

## Java Implementation

The Java program models the domain with explicit types rather than treating regression as a collection of unrelated arrays.

`Observation` is a Java record representing operational input. Its compact constructor enforces finite values and rejects negative operational quantities.

`RegressionResult` is immutable and exposes fitted coefficients, predictions, residuals, SSE, MSE, RMSE, R², and adjusted R².

`GovernancePolicy` makes acceptance criteria explicit. The model is evaluated by `GovernanceService`, which returns a `GovernanceDecision` containing a model status and a set of typed rejection reasons.

This design separates three concerns:

- `RegressionService` fits the mathematical model.
- `RegressionResult` represents the resulting analytical state.
- `GovernanceService` determines whether that result satisfies business deployment policy.

The use of enums for rejection reasons prevents policy failures from being represented only as unstructured text.

Java streams are used for residual filtering and aggregate calculations where the collection-processing semantics are clear. The matrix solver remains explicit because the numerical algorithm is easier to audit when its operations are visible.

## SQL Data Model

The PostgreSQL implementation represents the regression workflow relationally.

`observations` contains the predictor and response data. Its constraints reject negative operational quantities and non-positive delivery durations.

`regression_models` stores the fitted coefficient vector. The intercept and two predictor coefficients are stored separately because the demonstration uses a fixed two-predictor model.

`regression_predictions` connects each model to each observation and stores both fitted values and residuals. The composite primary key prevents duplicate model-observation results.

`regression_metrics` stores SSE, model-error MSE, RMSE, R², and adjusted R².

The SQL implementation uses a materialized view for the normal-equation cross-products. This makes the quantities in `XᵀX` and `Xᵀy` directly inspectable.

The coefficient calculation uses a small-system determinant approach because the demonstration contains only three coefficients. This is not a recommendation for large regression problems. Matrix inversion and determinant-based calculations become inappropriate as dimensionality and numerical sensitivity increase.

## Database-Level Integrity

Constraints are important because regression quality depends on the quality of the observations entering the model.

The observation table prevents negative processing hours and item counts. A relational system can therefore reject invalid values before the analytical layer calculates coefficients.

Foreign keys ensure that predictions refer to existing observations and fitted models.

The prediction table uses a composite key so that one model cannot accidentally contain duplicate results for the same observation.

The transaction example demonstrates how a model and its metrics can be read consistently as a publication unit.

Database constraints cannot prove that a statistical model is substantively appropriate. They can enforce structural validity while statistical diagnostics remain responsible for evaluating residual behavior, explanatory power, and model performance.

## Model Complexity and Degrees of Freedom

A regression with `p` predictors and an intercept estimates `p + 1` coefficients.

With `n` observations, residual degrees of freedom are

`n - p - 1`.

This quantity appears in the model-error MSE and adjusted R².

A model with too many predictors relative to the number of observations has little residual information available for estimating error. The implementations therefore reject datasets without sufficient degrees of freedom.

The requirement is not simply a programming safeguard. It follows directly from the mathematical structure of the fitted model.

## Residual Analysis

Residual analysis should not stop at calculating SSE.

A useful residual review asks whether errors appear systematically related to predictor values, whether some observations dominate the fit, and whether the scale of error changes across the predictor range.

The Python implementation includes leverage and standardized-residual calculations because a residual alone does not describe influence.

For example, a high-leverage observation may have an ordinary residual but still exert substantial influence on the estimated coefficients. Conversely, an observation far from the fitted regression surface can have a large residual while having relatively ordinary predictor values.

The aggregate values R² and adjusted R² cannot reveal these observation-level patterns.

## Numerical Conditioning

The normal equation is conceptually useful:

`β̂ = (XᵀX)⁻¹Xᵀy`

but it can be numerically fragile.

If predictors are highly correlated, `XᵀX` can become close to singular. Small numerical errors may then produce large changes in coefficient estimates.

The implementations deliberately demonstrate perfect multicollinearity. When the matrix cannot be reliably inverted, the model rejects the dataset.

For production numerical regression, QR decomposition or SVD is normally a stronger approach because it avoids some of the instability introduced by explicitly forming the cross-product matrix.

Feature scaling can also improve numerical conditioning, especially when one predictor is measured in very large units and another is measured in very small units.

## Common Mathematical Mistakes

Confusing residuals with predictions is a common error. A prediction is `ŷ`; a residual is `y - ŷ`.

Confusing SSE with MSE is another common error. SSE is the total squared residual error, while MSE divides a squared-error quantity by a chosen denominator.

Confusing predictive MSE with residual error variance can also produce incorrect statistical interpretations. `SSE/n` and `SSE/(n-p-1)` have different purposes.

Treating R² as prediction accuracy is incorrect. R² measures relative explained variation against an intercept-only baseline.

Assuming a larger R² always means a better model ignores the effect of adding predictors. Adjusted R² explicitly penalizes predictor count.

Assuming adjusted R² alone identifies the best model is also inappropriate. Prediction error, residual diagnostics, data quality, domain constraints, and the intended deployment context remain important.

## Practical Interpretation

For the operational scenario, the fitted coefficients describe how the estimated delivery duration changes with processing effort and order size while holding the other predictor fixed.

The residuals show where the model misses observed delivery durations.

SSE aggregates squared misses.

MSE expresses squared error after applying a denominator.

RMSE returns error to the response's original unit.

R² evaluates improvement over an intercept-only baseline.

Adjusted R² evaluates explanatory performance while accounting for the number of predictors.

These statistics answer different questions. They should therefore be interpreted together rather than collapsed into a single measure of model quality.

## Production Considerations

A production regression service should separate data validation, model fitting, prediction, diagnostics, and deployment policy.

The coefficient solver should use numerically stable linear algebra rather than explicit matrix inversion when model dimensions or conditioning require it.

Training data should be versioned so that a reported coefficient vector can be traced back to the exact observations and feature definitions that produced it.

Residual diagnostics should be retained with model metadata when model decisions need to be audited.

Model metrics should be interpreted in the context of the response scale. A numerically small RMSE is not automatically operationally acceptable if even a small prediction error has significant consequences.

A model should not be approved solely because R² is high. A deployment policy can combine R², adjusted R², error thresholds, data-quality conditions, and domain-specific acceptance rules, as demonstrated by the C++ and Java governance implementations.

The core mathematical workflow remains stable across all implementations:

`observations → design matrix → least-squares coefficients → fitted values → residuals → error metrics → explanatory metrics → diagnostics → decision`
