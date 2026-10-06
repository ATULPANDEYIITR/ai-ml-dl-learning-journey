# Linear Regression: Simple Regression, Multiple Regression, Coefficients, and Intercept

## Scope

This learning artifact develops linear regression from the basic two-variable model to multiple linear regression and then connects the mathematics to executable implementations.

The central model is

`y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε`

where `y` is the dependent variable, `x₁ ... xₚ` are predictors, `β₀` is the intercept, `β₁ ... βₚ` are coefficients, and `ε` represents unexplained variation.

The six deliverables use different perspectives:

| Deliverable | Technical focus |
|---|---|
| Python | Complete statistical implementation, matrix-based OLS, diagnostics, scaling, gradient descent, validation, and evaluation |
| JavaScript | Event-driven regression workflow, matrix OLS, prediction, metrics, gradient descent, and runtime validation |
| C++ | Repository-governance case study using a regression engine to estimate Pull Request review duration |
| Java | Enterprise domain model separating regression estimation from deterministic repository governance rules |
| SQL | Relational repository analytics model, regression-oriented data extraction, integrity constraints, indexes, and merge-eligibility queries |
| README | Mathematical and implementation-level explanation of the complete learning artifact |

The examples use linear relationships because the purpose is to understand the mechanics of linear regression rather than to present a general machine-learning framework.

## Core regression model

### Simple linear regression

Simple regression contains one predictor:

`y = β₀ + β₁x + ε`

The fitted prediction is

`ŷ = b₀ + b₁x`

The slope is

`b₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)`

The intercept is

`b₀ = ȳ - b₁x̄`

The slope describes the estimated change in the response associated with a one-unit increase in the predictor.

The intercept is the model's predicted response when the predictor equals zero. That interpretation is mathematical even when zero is outside the practical range of the observed data. An intercept should therefore not automatically be interpreted as a meaningful real-world baseline.

### Multiple linear regression

Multiple regression introduces several predictors:

`y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε`

The important distinction is that each coefficient represents a partial relationship. For example, `β₂` represents the estimated change in `y` associated with a one-unit change in `x₂` while the other included predictors are held constant.

This makes multiple regression different from fitting several independent simple regressions and combining their results.

## Coefficients and intercept

The intercept is represented separately from the predictor coefficients in every implementation.

For a model

`y = 12 + 3x₁ - 0.5x₂`

the intercept is `12`, the coefficient of `x₁` is `3`, and the coefficient of `x₂` is `-0.5`.

The sign of a coefficient determines the direction of the fitted association while its numerical magnitude depends on the unit of the corresponding predictor.

A coefficient cannot be interpreted safely without considering:

- the units of the predictor
- the units of the response
- the other predictors in the model
- the observed data range
- correlations among predictors
- model specification
- whether the relationship is appropriate for linear modeling

A regression coefficient is not automatically a causal effect. Observational regression can quantify conditional associations without proving that changing a predictor will cause the predicted response to change.

## Ordinary least squares

The Python, JavaScript, C++, and Java implementations use the ordinary least squares normal equation for their explicit matrix demonstrations.

The design matrix contains a leading column of ones:

`X = [1, x₁, x₂, ..., xₚ]`

The leading ones allow the first estimated parameter to be the intercept.

The ordinary least squares estimator is

`β = (XᵀX)⁻¹Xᵀy`

The objective is to minimize the residual sum of squares:

`SSE = Σ(yᵢ - ŷᵢ)²`

The residual for observation `i` is

`eᵢ = yᵢ - ŷᵢ`

The implementations explicitly construct the design matrix, transpose it, compute `XᵀX`, invert that matrix, calculate `Xᵀy`, and obtain the coefficient vector.

This makes the relationship between the mathematical formula and executable code visible.

For production numerical systems, explicitly computing a matrix inverse is generally not the preferred numerical approach. QR decomposition or singular value decomposition is usually more stable, especially when predictors are highly correlated or the design matrix is poorly conditioned.

## Python implementation

The Python program provides the broadest numerical treatment.

The simple regression implementation uses the closed-form scalar slope and intercept formulas. This makes the mechanics of one-predictor regression easy to inspect without hiding the calculation behind a library call.

The multiple regression implementation builds a design matrix and applies the normal equation. The `LinearRegressionModel` class keeps the intercept, coefficients, predictions, residuals, and evaluation metrics together.

The implementation also calculates:

- R-squared
- adjusted R-squared
- mean absolute error
- mean squared error
- root mean squared error
- residual variance
- approximate coefficient standard errors

The prediction method checks that the number of supplied features matches the fitted coefficient vector. This prevents a common operational error where a prediction row has a different structure from the training data.

### Feature scaling

The `StandardScaler` class transforms each predictor into approximately zero mean and unit variance.

Scaling does not change the underlying information in a predictor, but it changes the numerical representation. This is especially useful for gradient descent because predictors with dramatically different magnitudes can cause inefficient optimization.

For example, a predictor measured in square meters may have values around hundreds while another predictor such as a ratio may have values around one. Gradient-based optimization can behave poorly when those scales differ substantially.

### Gradient descent

The Python implementation also estimates coefficients through iterative gradient descent.

The objective is mean squared error. At each iteration, the gradients of the intercept and coefficients are calculated and the parameters are moved in the direction that reduces the objective.

This creates an important distinction:

- ordinary least squares through the normal equation obtains a closed-form solution when the required matrix operations are valid
- gradient descent searches for parameters iteratively

The two approaches are solving the same least-squares objective when configured appropriately, but they have different computational and numerical characteristics.

### Multicollinearity

The Python implementation calculates predictor correlations and variance inflation factors.

Multicollinearity occurs when predictors contain substantial overlapping information. For example, floor area and the number of rooms in a building may be strongly related.

A model can still produce a high R-squared while individual coefficients become unstable. This is why overall predictive fit and coefficient interpretability must be evaluated separately.

The variance inflation factor is represented as

`VIFⱼ = 1 / (1 - Rⱼ²)`

where `Rⱼ²` is obtained by regressing predictor `j` against the remaining predictors.

### Residual analysis

Residuals are the differences between observed and predicted values.

For an OLS model with an intercept, residuals sum to approximately zero. That property does not mean the model is automatically appropriate.

Residual patterns can indicate:

- nonlinear relationships
- changing error variance
- influential observations
- omitted predictors
- incorrect functional form
- unusual observations

The Python program therefore prints residual information instead of treating R-squared as the only measure of model quality.

## JavaScript implementation

The JavaScript program presents regression as a Node.js executable workflow.

Its `LinearRegression` class stores the model state and exposes `fit`, `predict`, `equation`, and `summary` behavior.

The implementation uses JavaScript arrays for vectors and matrices and explicitly performs matrix multiplication and Gauss-Jordan inversion.

The JavaScript implementation adds an event-driven layer through `RegressionWorkflow`.

The workflow emits:

- `trainingStarted`
- `trainingCompleted`
- `trainingFailed`

This reflects a practical application architecture in which model training is an observable operation rather than an isolated mathematical function.

The implementation also uses JavaScript-specific runtime behavior such as `console.table`, `Map`, classes, array operations, and event listener registration.

The regression example predicts business revenue from marketing spend, sales staff, and conversion rate. The predictors are deliberately different from the Python examples so the implementation demonstrates multiple regression without merely translating the same dataset.

The gradient-descent implementation standardizes the predictors before optimization and records the loss at each iteration. The decreasing loss provides a direct view of the optimization process.

## C++ case study

The C++ program models a repository analytics scenario.

The regression target is Pull Request review duration in hours.

The predictors are:

- number of changed files
- number of reviewers
- unresolved review discussions at the start of the review process

This produces a model of the form

`review_hours = β₀ + β₁(changed_files) + β₂(reviewer_count) + β₃(initial_unresolved_discussions)`

The C++ implementation deliberately separates prediction from governance.

`LinearRegressionEngine` estimates a continuous quantity. It does not decide whether a Pull Request can be merged.

`MergeEligibilityEngine` evaluates deterministic repository rules such as:

- whether the target branch is protected
- whether the Pull Request is a draft
- whether source and target branches differ
- whether the source branch is synchronized
- whether merge conflicts exist
- whether required status checks pass
- whether active approvals satisfy the policy
- whether reviewers have requested changes
- whether review discussions remain unresolved

This separation is important because statistical prediction and policy enforcement solve different problems.

The regression model can estimate expected review duration. It should not be used as a substitute for explicit merge governance.

The C++ implementation uses vectors, matrices, structures, enumerations, classes, exception handling, partial pivoting, and a coherent repository-governance domain model.

## Java enterprise model

The Java implementation models the same general domain from an enterprise software perspective while using a different architecture.

The domain contains explicit types for:

- Pull Request state
- review state
- reviewer eligibility
- review records
- status checks
- branch protection policies
- merge decisions

The `PullRequest` class controls its own state transitions rather than exposing unrestricted mutable state.

The model rejects invalid transitions such as merging an already closed or merged Pull Request.

`BranchProtectionPolicy` is represented as an immutable record. Its fields express governance requirements such as required approvals, required status checks, synchronization requirements, conversation resolution, direct-push restrictions, force-push restrictions, and deletion restrictions.

`GovernanceService` evaluates those policy rules and produces a `MergeDecision` containing both eligibility and explicit blockers.

This makes policy conflicts visible rather than hiding them inside a sequence of print statements.

The regression component remains independent from the governance component. This separation is particularly important in enterprise systems because an estimated review duration should not silently change an authorization or merge decision.

## SQL relational model

The PostgreSQL script models repository development activity relationally.

The central entities are:

- `repositories`
- `branches`
- `developers`
- `pull_requests`
- `commits`
- `reviews`
- `review_comments`
- `status_checks`
- `branch_protection_policies`
- `required_status_checks`

Foreign keys connect these entities so that review data cannot exist independently of its Pull Request and reviewer.

### Data integrity

The schema uses database constraints to enforce rules that should not depend solely on application validation.

Examples include:

`changed_files >= 0`

`review_duration_hours >= 0`

`required_approvals >= 0`

unique repository branch names

unique status-check names within a Pull Request

different source and target branches

These constraints make invalid states harder to store.

### Indexing

The script creates indexes for common operational access patterns.

The Pull Request repository and target-branch index supports filtering Pull Requests by repository, protected destination, and state.

The review-state index supports approval and change-request evaluation.

The unresolved-comment index focuses on the subset of comments that remain open, which is directly relevant to conversation-resolution policies.

### Regression-oriented view

`regression_training_data` converts normalized repository records into observations suitable for statistical modeling.

Each row represents a historical Pull Request and includes:

- changed files
- reviewer count
- unresolved discussions
- observed review duration

This is the relational preparation stage before coefficients are estimated.

The database therefore supplies structured observations while the application-level regression engines calculate the statistical model.

### Governance queries

The SQL script also evaluates merge eligibility using relational joins and conditional aggregation.

Approval counts, change requests, unresolved discussions, and required status checks are evaluated independently before being combined into a merge-eligibility result.

This mirrors the architectural distinction between:

`Pull Request data → review data → approval state → branch policy → merge eligibility`

and

`historical Pull Request data → regression features → estimated review duration`

The two paths can use related source data without being confused with one another.

## R-squared and adjusted R-squared

R-squared measures the proportion of observed response variation accounted for by the fitted model relative to a baseline based on the response mean.

It is commonly written as

`R² = 1 - SSE/SST`

where `SSE` is the residual sum of squares and `SST` is the total sum of squares.

An R-squared value closer to one indicates stronger in-sample explanatory fit under this definition.

R-squared should not be treated as a universal measure of model quality. It can increase when additional predictors are added, including predictors that contribute little useful information.

Adjusted R-squared accounts for model complexity:

`Adjusted R² = 1 - ((1 - R²)(n - 1) / (n - p - 1))`

where `n` is the number of observations and `p` is the number of predictors.

The adjustment penalizes unnecessary predictor inclusion more than ordinary R-squared.

## Interpreting coefficient changes

Consider two models:

`Model A: sales = 20 + 2.5(advertising)`

`Model B: sales = 10 + 1.4(advertising) + 8.0(sales_staff)`

The advertising coefficient changes because Model B conditions on sales staff.

That does not necessarily mean one calculation is wrong. It means the models answer different conditional questions.

If advertising and sales staff are correlated, the simple regression may attribute some variation to advertising that the multiple model can allocate between the two predictors.

Coefficient changes after adding predictors can therefore provide information about omitted-variable relationships and predictor overlap.

## Edge cases

### Constant predictor

If every observation has the same predictor value, the simple-regression denominator becomes zero.

There is no variation in `x` from which to estimate a slope.

The Python and JavaScript implementations explicitly reject this condition.

### Singular design matrix

In multiple regression, perfectly redundant predictors can make `XᵀX` singular.

For example, if one predictor is exactly twice another predictor, the model does not have enough independent information to identify both coefficients uniquely.

The implementations detect the near-zero pivot during matrix inversion and reject the model.

### Too few observations

A model with an intercept and `p` predictors contains `p + 1` parameters.

There must be sufficient observations to estimate those parameters and leave residual degrees of freedom for meaningful error estimation.

The implementations validate the observation-to-predictor relationship before fitting.

### Non-finite values

`NaN` and infinite values can corrupt matrix operations and model metrics.

The implementations therefore validate numerical inputs before estimation.

### Extrapolation

A fitted linear equation can produce a numerical prediction for an input far outside the observed predictor range. That does not mean the prediction is reliable.

Linear regression estimates relationships supported by the data. Extrapolation requires stronger assumptions than interpolation within the observed region.

## Practical interpretation

Suppose a model estimates

`review_hours = 1.5 + 0.32(changed_files) + 0.9(reviewers)`

The `0.32` coefficient means that, according to the fitted model, one additional changed file is associated with an estimated increase of `0.32` review hours while the number of reviewers is held constant.

The `0.9` coefficient means that one additional reviewer is associated with an estimated increase of `0.9` review hours while changed files are held constant.

The intercept `1.5` represents the model's estimated review duration when both predictors are zero.

Whether those interpretations are practically meaningful depends on whether zero changed files and zero reviewers are meaningful states in the application domain.

## Model evaluation

The implementations expose several complementary metrics.

### Mean absolute error

`MAE = mean(|y - ŷ|)`

MAE expresses average absolute prediction error in the same units as the target.

### Mean squared error

`MSE = mean((y - ŷ)²)`

MSE gives greater weight to larger errors because the residual is squared.

### Root mean squared error

`RMSE = sqrt(MSE)`

RMSE returns the error scale to the target's units while retaining the stronger penalty on large residuals.

A model should not be selected using a single metric without considering the target's business meaning, data-generating process, and validation design.

## Training and testing

The Python implementation separates observations into training and test subsets.

The model is fitted only on training data. Predictions for the test observations are then evaluated independently.

This distinction matters because training metrics describe how well the model fits observations used to estimate its coefficients. Test metrics provide evidence about performance on observations that were not used for coefficient estimation.

A large gap between training and test performance can indicate overfitting, data shift, leakage, or an unsuitable validation design.

## Numerical considerations

The normal equation is valuable for understanding the mathematics, but explicitly calculating `(XᵀX)⁻¹` can amplify numerical problems.

A nearly singular `XᵀX` matrix can arise from highly correlated predictors.

For production statistical computing, QR decomposition or singular value decomposition generally provides better numerical behavior.

Feature scaling is particularly relevant to gradient descent. The optimization process depends on the scale of the predictors, while the closed-form least-squares solution does not require scaling for mathematical validity.

Scaling does change coefficient units. A coefficient for a standardized predictor represents the expected response change associated with a one-standard-deviation change in that predictor rather than a one-unit change in its original unit.

## Relationship between the implementations

The implementations intentionally do not perform identical demonstrations.

The Python program emphasizes mathematical transparency and statistical diagnostics.

The JavaScript program emphasizes an executable Node.js workflow with event-driven training state and prediction behavior.

The C++ program uses regression inside a repository-governance case study and separates statistical estimation from merge-policy enforcement.

The Java program expresses the same domain through explicit enterprise types, immutable policy records, controlled state transitions, and a service that produces merge decisions.

The SQL script focuses on how relational data is structured and transformed into regression-ready observations while maintaining database-level integrity.

Together, the implementations demonstrate that the regression equation remains mathematically consistent while its surrounding software architecture can differ substantially by language and application context.

## Limitations

Linear regression assumes that a linear specification is a reasonable representation of the relationship being modeled.

Important limitations include:

- nonlinear relationships may be represented poorly
- influential observations can disproportionately affect coefficients
- correlated predictors can make individual coefficients unstable
- heteroscedastic residuals can affect conventional inference
- omitted variables can distort coefficient interpretation
- measurement errors in predictors can affect estimates
- extrapolation can be unreliable
- observational coefficients do not automatically establish causality
- a high R-squared does not prove that the model is correctly specified

A technically correct coefficient calculation does not guarantee that the underlying model is appropriate for the real-world question.

## Production considerations

A production implementation should generally use a mature numerical linear-algebra library rather than manually inverting matrices.

Model inputs should have a defined schema so that training and prediction use identical feature order and units.

Training datasets should preserve provenance, timestamps, and version information so that a fitted model can be reproduced.

Coefficient changes should be monitored because shifts in the underlying data distribution can change both estimates and prediction quality.

Validation should occur on data that represents the actual deployment setting. Random splitting is not always appropriate for temporal or grouped data.

The regression model should remain separate from authorization and governance rules. A statistical prediction can inform capacity planning or operational forecasting without becoming an implicit permission mechanism.

The core relationship remains simple:

`data → design matrix → coefficient estimation → prediction → residual analysis`

The quality of the resulting model depends on the data, assumptions, numerical method, feature design, and evaluation process as much as on the regression equation itself.
