# Mathematical Revision: Probability, Statistics, Linear Algebra, Calculus, and Optimization

## 1. Topic Introduction

Mathematical revision for quantitative computing requires more than memorizing formulas. Probability describes uncertainty, statistics extracts information from observations, linear algebra provides the language for vectors and matrices, calculus describes change and accumulation, and optimization provides methods for selecting the best feasible solution.

These subjects are strongly interconnected.

A statistical model can use linear algebra to represent data, calculus to derive gradients, probability to model uncertainty, and optimization to estimate parameters. The three implementations in this repository demonstrate these relationships computationally:

- The Python implementation is a broad mathematical laboratory containing formulas, numerical methods, simulations, statistics, linear algebra, calculus, optimization, and validation.
- The JavaScript implementation translates the major ideas into executable application-oriented numerical code.
- The C++ implementation develops an industry-style quantitative portfolio case study combining statistics, covariance matrices, probability distributions, numerical calculus, linear algebra, and constrained optimization.

The examples are designed to progress from elementary operations to computationally meaningful mathematical systems.

---

## 2. Mathematical Foundations

### 2.1 Variables

A variable represents a quantity whose value can change.

Examples include:

- `x` as an independent variable
- `y` as a dependent variable
- `n` as a sample size
- `p` as a probability
- `w` as a portfolio weight
- `r` as a rate of return

Mathematical programs represent these variables using numerical types.

Python and JavaScript make numerical experimentation concise, while C++ provides explicit control over data representation and program structure.

### 2.2 Constants

A constant has a fixed value within a particular problem.

Important mathematical constants include:

- π
- e
- 0
- 1
- the imaginary unit in complex arithmetic

The implementations use constants such as π through the standard mathematical libraries.

### 2.3 Functions

A function maps input values to output values.

For example:

`f(x) = 2x + 1`

takes `x` as input and returns `2x + 1`.

A function may have:

- one variable
- several variables
- scalar output
- vector output
- discrete inputs
- continuous inputs

Functions are central to calculus and optimization because derivatives and objective functions operate on them.

---

# 3. Probability

## 3.1 Definition

Probability is the mathematical framework for quantifying uncertainty.

For an event `A`:

`0 ≤ P(A) ≤ 1`

where:

- `P(A) = 0` means the event is impossible under the model.
- `P(A) = 1` means the event is certain under the model.

---

## 3.2 Sample Space

The sample space, usually denoted by `Ω`, contains all possible outcomes of an experiment.

For a fair six-sided die:

`Ω = {1, 2, 3, 4, 5, 6}`

An event is a subset of the sample space.

The event of obtaining an even number is:

`A = {2, 4, 6}`

Therefore:

`P(A) = 3/6 = 1/2`

---

## 3.3 Complement Rule

The complement of event `A` is the event that `A` does not occur.

`P(Aᶜ) = 1 - P(A)`

This is useful when calculating the probability of "at least one" or "not occurring" events.

---

## 3.4 Union Rule

For two events:

`P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`

The intersection must be subtracted because it is otherwise counted twice.

If two events are mutually exclusive:

`P(A ∩ B) = 0`

so:

`P(A ∪ B) = P(A) + P(B)`

---

## 3.5 Conditional Probability

Conditional probability measures the probability of `A` given that `B` has occurred.

`P(A|B) = P(A ∩ B) / P(B)`

provided that:

`P(B) > 0`

Conditional probability is important in:

- medical testing
- fraud detection
- reliability analysis
- machine learning
- risk management
- classification
- Bayesian inference

---

## 3.6 Independence

Two events are independent if learning that one occurred does not change the probability of the other.

The defining relationship is:

`P(A ∩ B) = P(A)P(B)`

Equivalently:

`P(A|B) = P(A)`

when the relevant probabilities are defined.

Independence is a mathematical assumption and should not be confused with events merely appearing unrelated.

---

## 3.7 Bayes' Theorem

Bayes' theorem is:

`P(B|A) = P(A|B)P(B) / P(A)`

It is useful for updating probabilities after receiving evidence.

The Python and JavaScript implementations demonstrate a diagnostic-testing example.

A crucial statistical lesson is that:

`P(disease | positive)`

is not generally equal to:

`P(positive | disease)`

Confusing these probabilities is known as the inverse-probability or prosecutor's-fallacy type of error depending on the context.

---

## 3.8 Law of Total Probability

If `B₁, B₂, ..., Bₙ` form a partition of the sample space:

`P(A) = Σ P(A|Bᵢ)P(Bᵢ)`

This is useful when a probability is easier to calculate by separating the population into groups.

The Python implementation demonstrates a multi-factory defect-rate example.

---

# 4. Random Variables

A random variable assigns a numerical value to an outcome.

Random variables can be:

- discrete
- continuous

A discrete random variable can take countable values.

A continuous random variable can take values over an interval or more general continuous domain.

---

## 4.1 Expected Value

For a discrete random variable:

`E[X] = Σ xP(X=x)`

The expected value is a weighted average.

It does not necessarily represent an outcome that can actually occur.

For a fair die:

`E[X] = (1+2+3+4+5+6)/6 = 3.5`

---

## 4.2 Variance

Variance measures dispersion around the expected value.

`Var(X) = E[(X-E[X])²]`

An equivalent computational form is:

`Var(X) = E[X²] - E[X]²`

The standard deviation is:

`σ = √Var(X)`

---

## 4.3 Binomial Distribution

The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials with constant success probability `p`.

Its probability mass function is:

`P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ`

where:

- `n` is the number of trials.
- `k` is the number of successes.
- `p` is the probability of success.

Its mean is:

`E[X] = np`

Its variance is:

`Var(X) = np(1-p)`

Both the Python and JavaScript implementations calculate binomial probabilities directly.

---

# 5. Monte Carlo Simulation

Monte Carlo methods use repeated random sampling to approximate mathematical quantities.

The Python implementation estimates π by randomly generating points in a square containing a circle.

For a unit circle inside a square from `-1` to `1`:

`Area of circle / Area of square = π/4`

Therefore:

`π ≈ 4 × inside / total`

Monte Carlo methods are useful when:

- exact solutions are difficult
- integration is high-dimensional
- simulation is more practical than analytical derivation
- uncertainty needs to be propagated through a model

Monte Carlo estimates contain sampling error. Increasing the number of simulations generally improves the estimate, but the convergence rate can be slow.

---

# 6. Statistics

Statistics concerns the collection, description, analysis, interpretation, and modeling of data.

It is useful to distinguish:

- population
- sample
- parameter
- statistic

A population represents the complete group being studied.

A sample is a subset of observations.

A population parameter describes the population.

A sample statistic is calculated from observed sample data.

---

# 7. Descriptive Statistics

## 7.1 Mean

For observations `x₁, x₂, ..., xₙ`:

`x̄ = Σxᵢ/n`

The mean is sensitive to extreme observations.

---

## 7.2 Median

The median is the central observation after sorting the data.

For an odd number of observations, it is the middle value.

For an even number, it is commonly the average of the two central observations.

The median is more resistant to extreme values than the mean.

---

## 7.3 Mode

The mode is the most frequently occurring value.

A dataset may have:

- one mode
- multiple modes
- no unique mode

The Python implementation uses `statistics.multimode()` to handle multiple modes.

---

## 7.4 Range

The range is:

`maximum - minimum`

It is simple but strongly affected by extreme observations.

---

## 7.5 Variance and Standard Deviation

Population variance:

`σ² = Σ(xᵢ-μ)²/N`

Sample variance:

`s² = Σ(xᵢ-x̄)²/(n-1)`

The denominator `n-1` is used for the standard unbiased estimator of population variance under the conventional independent-sample framework.

Standard deviation is the square root of variance.

---

# 8. Percentiles

A percentile identifies a position within an ordered dataset.

The implementations use linear interpolation between neighboring observations.

The 50th percentile corresponds to the median under this definition.

Percentiles are commonly used for:

- salary analysis
- performance benchmarks
- latency measurements
- exam results
- financial distributions
- risk analysis

Different software packages can use different percentile definitions, so percentile results should be interpreted with awareness of the selected convention.

---

# 9. Z-Scores

A z-score standardizes an observation:

`z = (x-μ)/σ`

It describes the number of standard deviations an observation lies from the mean.

For sample-based standardization, the sample mean and sample standard deviation are often used.

Z-scores are useful for:

- comparing measurements on different scales
- detecting unusually distant observations
- transforming normal variables
- statistical inference

A large absolute z-score does not automatically prove that an observation is erroneous.

---

# 10. Covariance

Covariance describes how two variables vary together.

Sample covariance:

`Cov(X,Y) = Σ(xᵢ-x̄)(yᵢ-ȳ)/(n-1)`

Positive covariance indicates that the variables tend to move in the same direction.

Negative covariance indicates that they tend to move in opposite directions.

Covariance depends on the units of measurement, so it is difficult to compare directly across differently scaled variables.

---

# 11. Correlation

Pearson correlation standardizes covariance:

`r = Cov(X,Y)/(sₓsᵧ)`

For the conventional Pearson coefficient:

`-1 ≤ r ≤ 1`

A value close to:

- `1` indicates strong positive linear association.
- `-1` indicates strong negative linear association.
- `0` indicates little linear association.

Correlation does not establish causation.

A correlation of zero also does not imply complete statistical independence.

---

# 12. Linear Regression

A simple linear regression model is:

`y = β₀ + β₁x + ε`

where:

- `β₀` is the intercept.
- `β₁` is the slope.
- `ε` represents unexplained variation.

The least-squares slope is:

`β₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²`

The intercept is:

`β₀ = ȳ - β₁x̄`

The Python, JavaScript, and C++ implementations all demonstrate regression.

The C++ implementation places regression inside a larger quantitative-analysis architecture.

---

# 13. R²

The coefficient of determination is:

`R² = 1 - SSE/SST`

where:

- `SSE` is the residual sum of squares.
- `SST` is the total sum of squares.

For ordinary linear regression with an intercept, R² measures the proportion of observed response variation explained by the fitted linear model under the model's definition.

A high R² does not prove causality, correct model specification, or good out-of-sample performance.

---

# 14. Linear Algebra

Linear algebra studies vectors, matrices, linear transformations, systems of equations, eigenvalues, eigenvectors, vector spaces, and related structures.

It is fundamental to:

- statistics
- machine learning
- optimization
- computer graphics
- physics
- engineering
- signal processing
- quantitative finance
- scientific computing

---

# 15. Vectors

A vector can be represented as:

`v = [v₁, v₂, ..., vₙ]`

The implementations demonstrate:

- vector addition
- vector subtraction
- scalar multiplication
- dot product
- Euclidean norm
- cosine similarity

---

## 15.1 Dot Product

For vectors `a` and `b`:

`a·b = Σaᵢbᵢ`

The dot product appears in:

- projections
- geometry
- regression
- optimization
- machine learning
- portfolio expected returns

---

## 15.2 Norm

The Euclidean norm is:

`||x||₂ = √(Σxᵢ²)`

It represents the geometric length of a vector.

Other norms exist, including:

`||x||₁ = Σ|xᵢ|`

and:

`||x||∞ = max|xᵢ|`

Different norms produce different geometric and optimization behavior.

---

## 15.3 Cosine Similarity

Cosine similarity is:

`cos(θ) = (a·b)/(||a||||b||)`

It measures angular similarity rather than raw magnitude.

It is commonly used in vector-based information retrieval and other high-dimensional applications.

It is undefined for zero vectors.

---

# 16. Matrices

A matrix is a rectangular arrangement of numbers.

For an `m × n` matrix:

- `m` is the number of rows.
- `n` is the number of columns.

Matrix multiplication is defined when the inner dimensions agree.

If:

`A` is `m × n`

and:

`B` is `n × p`

then:

`AB` is `m × p`.

The entry is:

`(AB)ᵢⱼ = ΣAᵢₖBₖⱼ`

The implementations explicitly validate dimensions before performing matrix operations.

---

# 17. Matrix Transpose

The transpose exchanges rows and columns.

If:

`A = [[1,2],[3,4]]`

then:

`Aᵀ = [[1,3],[2,4]]`

The transpose is important in:

- least squares
- covariance calculations
- quadratic forms
- optimization
- orthogonal projections

---

# 18. Determinants

For:

`A = [[a,b],[c,d]]`

the determinant is:

`det(A) = ad-bc`

A nonzero determinant indicates that a square matrix is invertible.

A zero determinant indicates singularity.

The Python and JavaScript educational implementations use recursive expansion for clarity. This is not the preferred method for large numerical matrices because its computational cost grows rapidly.

Practical numerical software generally uses factorization methods such as LU decomposition rather than recursively expanding a determinant.

---

# 19. Linear Systems

A linear system can be written:

`Ax=b`

where:

- `A` is the coefficient matrix.
- `x` is the unknown vector.
- `b` is the constants vector.

Gaussian elimination transforms the system into a form that can be solved using back substitution.

The implementations use pivot selection to improve numerical robustness.

---

## 19.1 Singular and Nearly Singular Systems

A singular matrix does not have a unique inverse.

A nearly singular matrix can also cause numerical instability because small changes in inputs may produce large changes in the computed solution.

A determinant near zero can indicate poor conditioning, but determinant magnitude alone is not a complete measure of numerical conditioning.

---

# 20. Eigenvalues and Eigenvectors

An eigenvector satisfies:

`Av = λv`

where:

- `A` is a matrix.
- `v` is an eigenvector.
- `λ` is the corresponding eigenvalue.

The C++ case study uses power iteration to estimate the dominant eigenvalue and eigenvector of a covariance matrix.

This is relevant to principal-component-style analysis because dominant eigenvectors identify important directions of variation.

Power iteration is simple and useful when a dominant eigenvalue is sufficiently separated from the others.

---

# 21. Calculus

Calculus studies change, accumulation, limits, derivatives, integrals, and related mathematical structures.

The central computational ideas used in the implementations are:

- limits
- first derivatives
- second derivatives
- numerical integration
- Taylor series
- gradients
- Hessians

---

# 22. Limits

A derivative is defined through a limit:

`f'(x) = lim(h→0)[f(x+h)-f(x)]/h`

A limit describes what a function approaches as its input approaches a particular value.

Numerical limits approximate this process by evaluating the function at progressively smaller distances from the target.

Numerical approximations can be affected by floating-point precision and the choice of step size.

---

# 23. Derivatives

The derivative measures local rate of change.

For:

`f(x) = x³ - 2x + 1`

the exact derivative is:

`f'(x) = 3x² - 2`

At `x=2`:

`f'(2) = 10`

The Python and JavaScript implementations approximate this numerically using finite differences.

---

## 23.1 Forward Difference

`f'(x) ≈ [f(x+h)-f(x)]/h`

It is simple but has first-order truncation error.

---

## 23.2 Central Difference

`f'(x) ≈ [f(x+h)-f(x-h)]/(2h)`

It generally provides better accuracy for smooth functions at comparable step sizes.

A very small `h` is not automatically better because floating-point cancellation can become significant.

---

# 24. Second Derivative

A common numerical approximation is:

`f''(x) ≈ [f(x+h)-2f(x)+f(x-h)]/h²`

The second derivative measures curvature.

For:

`f(x)=x³-2x+1`

the second derivative is:

`f''(x)=6x`

---

# 25. Integration

Integration measures accumulation and area under a curve.

The definite integral is:

`∫ₐᵇ f(x)dx`

If `F'(x)=f(x)`, the fundamental theorem gives:

`∫ₐᵇf(x)dx = F(b)-F(a)`

The Python and JavaScript implementations demonstrate numerical integration through:

- trapezoidal rule
- Simpson's rule

---

## 25.1 Trapezoidal Rule

The interval is divided into small segments and each segment is approximated by a trapezoid.

For sufficiently smooth functions, reducing the step size generally improves accuracy until numerical effects become relevant.

---

## 25.2 Simpson's Rule

Simpson's rule approximates sections using quadratic interpolation.

It requires an even number of subintervals in the implementation.

It is often more accurate than the basic trapezoidal rule for sufficiently smooth functions at similar resolutions.

---

# 26. Taylor Series

A Taylor series approximates a function around a point.

For the exponential function:

`eˣ = 1 + x + x²/2! + x³/3! + ...`

The implementations calculate exponential values using a finite number of terms.

Taylor approximations illustrate the relationship between:

- derivatives
- local approximations
- numerical computation
- truncation error

The number of terms required depends on the function, expansion point, and desired accuracy.

---

# 27. Optimization

Optimization searches for values that minimize or maximize an objective function subject to possible constraints.

An optimization problem commonly contains:

- decision variables
- objective function
- constraints
- feasible region
- optimum

Examples include:

- minimizing cost
- maximizing profit
- minimizing prediction error
- allocating resources
- portfolio construction
- engineering design

---

# 28. Local and Global Optima

A local minimum is optimal relative to nearby points.

A global minimum is optimal over the entire feasible domain.

A differentiable unconstrained local optimum often satisfies:

`∇f(x*) = 0`

This is a necessary stationarity condition under appropriate assumptions, not a guarantee that the point is a global minimum.

For example, saddle points can also have zero gradient.

---

# 29. Gradient

For a function of multiple variables:

`f(x₁,x₂,...,xₙ)`

the gradient is:

`∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`

The gradient points in the direction of steepest local increase under the Euclidean inner product.

Therefore negative gradient is a natural direction for minimization.

---

# 30. Gradient Descent

Gradient descent uses:

`xₖ₊₁ = xₖ - α∇f(xₖ)`

where `α` is the learning rate or step size.

The implementations demonstrate numerical gradient descent.

Important parameters include:

- initial point
- learning rate
- stopping tolerance
- maximum iterations

A learning rate that is too large can cause divergence or oscillation.

A learning rate that is too small can produce slow convergence.

---

# 31. Newton's Method

For one-dimensional root finding:

`xₖ₊₁ = xₖ - f(xₖ)/f'(xₖ)`

Newton's method can converge very rapidly near a sufficiently regular simple root.

Its limitations include:

- derivative requirements
- possible division by values near zero
- sensitivity to starting points
- possible divergence
- convergence to an unintended root

The Python, JavaScript, and C++ implementations demonstrate root finding.

---

# 32. Golden-Section Search

Golden-section search minimizes a unimodal one-dimensional function over an interval without requiring derivatives.

It repeatedly reduces the search interval while reusing function evaluations.

It is useful when:

- the objective is one-dimensional
- derivative information is unavailable
- the objective is reasonably unimodal on the search interval

---

# 33. Constrained Optimization

Real optimization problems frequently contain constraints.

A general form is:

`minimize f(x)`

subject to:

`gᵢ(x) ≤ 0`

and possibly:

`hⱼ(x) = 0`

A feasible solution satisfies every constraint.

The Python and JavaScript implementations include a resource-allocation example.

The C++ case study uses a portfolio constraint:

`wᵢ ≥ 0`

and:

`Σwᵢ = 1`

This represents a long-only portfolio whose entire capital is allocated.

---

# 34. Convexity

A function `f` is convex when:

`f(θx+(1-θ)y) ≤ θf(x)+(1-θ)f(y)`

for suitable `x`, `y`, and `θ ∈ [0,1]`.

Convexity is important because a local minimum of a convex function is also a global minimum.

For twice-differentiable functions, a positive semidefinite Hessian is a standard sufficient condition for convexity over an appropriate convex domain.

Convex optimization has powerful theoretical and computational properties, but not every practical optimization problem is convex.

---

# 35. Hessian

The Hessian contains second partial derivatives:

`Hᵢⱼ = ∂²f/(∂xᵢ∂xⱼ)`

It describes local curvature.

For a twice-differentiable unconstrained problem, Hessian information can help distinguish:

- local minima
- local maxima
- saddle points

For example, a positive-definite Hessian at a stationary point gives a strict local minimum under standard regularity assumptions.

The Python implementation numerically estimates Hessians.

---

# 36. Discrete Optimization

Not every optimization problem has continuous variables.

The Python and JavaScript implementations include the 0/1 knapsack problem.

Each item has:

- weight
- value

The goal is to maximize total value subject to a capacity constraint.

Dynamic programming uses subproblems defined by:

- number of available items
- remaining capacity

The standard dynamic-programming formulation has time complexity:

`O(nC)`

where:

- `n` is the number of items.
- `C` is the integer capacity.

This is pseudo-polynomial in the numerical value of the capacity.

---

# 37. Normal Distribution

The normal probability density function is:

`f(x) = 1/(σ√(2π)) exp(-(x-μ)²/(2σ²))`

where:

- `μ` is the mean.
- `σ` is the standard deviation.

The standard normal distribution has:

`μ = 0`

and:

`σ = 1`

The cumulative distribution function gives:

`P(X ≤ x)`

The implementations use the error function where available to calculate the CDF.

---

# 38. Central Limit Theorem

The central limit theorem explains why standardized sample means often approach a normal distribution under broad conditions as sample size increases.

A common approximation is:

`X̄ ≈ N(μ, σ²/n)`

under appropriate assumptions.

The theorem does not say that the original population must be normal in every application. It describes the distribution of appropriately standardized sums or means under specified regularity conditions.

The Python implementation simulates sample means from a deliberately non-normal population.

---

# 39. Confidence Intervals

A confidence interval is an interval produced by a statistical procedure designed to have a specified long-run coverage property under its assumptions.

For a large-sample approximate mean interval:

`x̄ ± z* SE`

where:

`SE = s/√n`

and `z*` is an appropriate critical value.

For an approximate 95% normal critical value:

`z* ≈ 1.96`

The exact method should depend on sample size, assumptions, variance estimation, distributional properties, and the intended inferential framework.

---

# 40. Least Squares and Linear Algebra

Linear regression can be represented using matrices.

The model is:

`y ≈ Xβ`

The least-squares objective is:

`||Xβ-y||²`

The normal equations are:

`XᵀXβ = Xᵀy`

When `XᵀX` is invertible:

`β = (XᵀX)⁻¹Xᵀy`

The Python implementation explains this relationship, while the C++ implementation uses matrix operations as part of a quantitative system.

In numerical production software, explicitly calculating a matrix inverse is often avoided. QR decomposition or singular value decomposition can provide better numerical behavior.

---

# 41. C++ Industry-Style Case Study

## Problem

The C++ program models a simplified quantitative portfolio-analysis system.

It takes deterministic synthetic return observations for five assets:

- Technology
- Healthcare
- Energy
- Consumer
- Utilities

The system performs the following sequence:

1. Store asset return observations.
2. Calculate expected asset returns.
3. Construct the covariance matrix.
4. Calculate portfolio expected return.
5. Calculate portfolio variance.
6. Calculate portfolio standard deviation.
7. Calculate a Sharpe-style risk-adjusted measure.
8. Estimate a dominant covariance factor using power iteration.
9. Optimize portfolio weights.
10. Enforce non-negative weights summing to one.
11. Use numerical calculus.
12. Estimate a probability using a normal model.
13. Validate important results.

This creates a direct relationship between the mathematical subjects.

---

# 42. C++ Case-Study Data Model

The `Asset` structure contains:

- an asset name
- a vector of return observations

The `Portfolio` class contains:

- assets
- portfolio weights
- weight validation
- expected returns
- covariance matrix
- portfolio variance
- standard deviation
- Sharpe-style ratio

This separation keeps data representation and mathematical operations organized.

---

# 43. Portfolio Expected Return

If portfolio weights are:

`w = [w₁,w₂,...,wₙ]`

and expected asset returns are:

`μ = [μ₁,μ₂,...,μₙ]`

then:

`E[Rₚ] = wᵀμ`

This is a dot product.

That is one of the simplest examples of linear algebra directly representing a practical quantitative formula.

---

# 44. Portfolio Variance

With covariance matrix `Σ`:

`Var(Rₚ) = wᵀΣw`

This is a quadratic form.

The C++ implementation computes:

1. `Σw`
2. `wᵀ(Σw)`

using matrix-vector and vector dot-product operations.

This is computationally important because portfolio risk depends not only on individual asset variances but also on covariance between assets.

---

# 45. Covariance Matrix

For `n` assets, the covariance matrix is an `n × n` matrix.

The diagonal contains individual variances:

`Σᵢᵢ = Var(Xᵢ)`

The off-diagonal elements contain covariances:

`Σᵢⱼ = Cov(Xᵢ,Xⱼ)`

For a valid covariance matrix, the matrix is theoretically symmetric and positive semidefinite.

Numerical calculations can introduce small asymmetries or negative eigenvalues due to floating-point effects or data-processing issues, so production implementations often perform validation and, where appropriate, numerical regularization.

---

# 46. Portfolio Optimization Objective

The C++ case study minimizes:

`λwᵀΣw - μᵀw`

where:

- `w` is the portfolio-weight vector.
- `Σ` is the covariance matrix.
- `μ` is the expected-return vector.
- `λ` is risk aversion.

A larger `λ` places greater emphasis on variance.

A smaller `λ` gives relatively greater importance to expected return.

This is a simplified mean-variance optimization formulation.

---

# 47. Portfolio Constraints

The case study imposes:

`wᵢ ≥ 0`

and:

`Σwᵢ = 1`

The first constraint prevents short positions.

The second ensures that the portfolio is fully allocated.

These constraints define a simplex.

The C++ program uses projection onto the simplex after each gradient step.

This is an example of projected gradient descent.

---

# 48. Simplex Projection

Given an unconstrained vector, the projection algorithm finds the closest vector satisfying:

`wᵢ ≥ 0`

and:

`Σwᵢ = 1`

The implementation sorts the values and calculates an appropriate threshold.

The sorting operation makes the projection approximately:

`O(n log n)`

for `n` portfolio variables.

This is substantially more appropriate than repeatedly searching arbitrary feasible combinations.

---

# 49. Power Iteration and Principal Directions

The C++ case study applies power iteration to the covariance matrix.

Starting from a nonzero vector:

`v₀`

the algorithm repeatedly calculates:

`vₖ₊₁ = Avₖ / ||Avₖ||`

Under suitable spectral conditions, this converges toward the dominant eigenvector.

The dominant eigenvector identifies an important direction in the covariance structure.

This is conceptually related to principal component analysis.

---

# 50. Probability and Portfolio Risk

The C++ implementation also combines the portfolio mean and standard deviation with a normal approximation.

For a return threshold `t`:

`P(R < t) ≈ Φ((t-μ)/σ)`

where `Φ` is the standard normal CDF.

This is a mathematical model, not an empirical guarantee.

Financial return distributions may exhibit:

- skewness
- heavy tails
- volatility clustering
- autocorrelation
- regime changes
- nonlinear dependence

A normal model therefore has limitations when used for real financial risk assessment.

---

# 51. Numerical Methods

The implementations use numerical methods because many mathematical problems cannot or should not be solved through direct symbolic manipulation in software.

Important numerical methods demonstrated include:

- finite differences
- trapezoidal integration
- Simpson integration
- Taylor approximation
- Gaussian elimination
- power iteration
- gradient descent
- projected gradient descent
- golden-section search
- Newton's method
- bisection

---

# 52. Numerical Error

Two important error concepts are:

Absolute error:

`|approximation-exact|`

Relative error:

`|approximation-exact|/|exact|`

Relative error becomes problematic when the exact value is zero, so it requires special handling.

The implementations explicitly handle the zero-reference case.

---

# 53. Floating-Point Arithmetic

Computers generally represent real numbers approximately.

Consequently:

`0.1 + 0.2`

may not equal exactly:

`0.3`

in binary floating-point representation.

This means direct equality tests such as:

`x == expected`

can be inappropriate for numerical results.

A tolerance-based test is often preferable:

`|x-expected| < tolerance`

The exact tolerance should depend on:

- scale
- algorithm
- conditioning
- expected numerical error
- required accuracy

---

# 54. Catastrophic Cancellation

Subtracting nearly equal floating-point numbers can lose significant digits.

For example:

`sqrt(x+1) - sqrt(x)`

can be rewritten algebraically as:

`1/(sqrt(x+1)+sqrt(x))`

The Python, JavaScript, and C++ implementations demonstrate this reformulation.

Algebraic transformations can therefore improve numerical stability without changing mathematical meaning.

---

# 55. Conditioning

Conditioning describes how sensitive a mathematical problem is to small changes in its inputs.

An ill-conditioned problem may amplify small input errors.

Conditioning is a property of the mathematical problem.

Stability is a property of the numerical algorithm used to solve it.

These concepts should not be confused.

A stable algorithm cannot completely remove the intrinsic sensitivity of an ill-conditioned problem.

---

# 56. Common Mathematical Mistakes

## Probability

Common errors include:

- confusing `P(A|B)` with `P(B|A)`
- assuming independence without justification
- adding probabilities of overlapping events without subtracting the intersection
- ignoring base rates
- treating a probability model as certainty

## Statistics

Common errors include:

- treating correlation as causation
- confusing population and sample variance
- interpreting a sample statistic as an exact population parameter
- ignoring outliers
- overinterpreting small samples
- treating statistical significance as practical importance

## Linear Algebra

Common errors include:

- multiplying incompatible matrices
- confusing matrix multiplication with elementwise multiplication
- assuming every square matrix is invertible
- ignoring conditioning
- treating numerical zero as exact zero

## Calculus

Common errors include:

- using a derivative formula outside its assumptions
- choosing an unsuitable numerical step size
- assuming differentiability everywhere
- confusing local and global behavior
- ignoring boundary conditions

## Optimization

Common errors include:

- selecting an excessively large learning rate
- assuming every stationary point is a minimum
- ignoring constraints
- optimizing an incorrectly defined objective
- failing to validate convergence
- trusting a numerical result without checking sensitivity

---

# 57. Edge Cases Demonstrated

The implementations explicitly address several failure conditions.

Examples include:

- empty datasets
- invalid probabilities
- zero standard deviation
- mismatched vector dimensions
- incompatible matrix dimensions
- singular matrices
- zero derivatives
- invalid integration interval counts
- invalid optimization parameters
- invalid portfolio weights
- zero-risk Sharpe-ratio cases
- invalid bisection intervals
- negative capacities
- non-positive item weights

Edge-case handling is part of mathematical correctness in software.

---

# 58. Python Implementation

The Python script is structured as a broad mathematical laboratory.

It demonstrates:

- arithmetic
- functions
- probability
- conditional probability
- Bayes' theorem
- binomial probability
- expected value
- variance
- Monte Carlo simulation
- descriptive statistics
- percentiles
- z-scores
- covariance
- correlation
- linear regression
- vectors
- matrices
- determinants
- Gaussian elimination
- eigenvalue estimation
- limits
- numerical derivatives
- second derivatives
- numerical integration
- Taylor series
- gradients
- gradient descent
- Newton's method
- golden-section search
- constrained optimization
- numerical error
- numerical stability
- confidence intervals
- normal distributions
- the central limit theorem
- least-squares concepts
- logistic functions
- discrete optimization
- financial calculations
- validation

The script is deliberately standard-library-only so that the mathematical mechanisms remain visible.

---

# 59. JavaScript Implementation

The JavaScript implementation provides the same mathematical domain through a different programming environment.

It demonstrates:

- JavaScript numeric behavior
- floating-point limitations
- probability calculations
- random simulation
- statistical functions
- regression
- vector operations
- matrix operations
- Gaussian elimination
- numerical differentiation
- numerical integration
- Taylor approximation
- normal distributions
- gradient descent
- optimization
- dynamic programming
- numerical stability
- financial calculations
- root finding
- validation

JavaScript is particularly relevant when mathematical logic needs to be integrated into:

- browser applications
- dashboards
- interactive calculators
- client-side data analysis
- web-based simulations
- visualization systems

The file does not require an external npm dependency.

---

# 60. C++ Implementation

The C++ implementation differs from the Python and JavaScript files by concentrating on one integrated quantitative case study.

Its architecture is:

`Asset Data → Statistics → Covariance → Linear Algebra → Risk Model → Optimization → Validation`

The program demonstrates how mathematical components can become reusable software components.

C++ is useful for computationally intensive mathematical systems because it provides:

- explicit memory and data structures
- predictable performance
- strong static typing
- standard-library containers
- efficient numerical loops
- low-level control when required

The program is compatible with C++17-style compilation.

---

# 61. C++ Design Decisions

## Data Structures

`std::vector<double>` is used for numerical vectors.

A vector of vectors represents dense matrices.

This keeps the case study dependency-free.

## Classes

The `Portfolio` class encapsulates:

- assets
- weights
- expected returns
- covariance calculations
- portfolio risk
- risk-adjusted return calculations

## Validation

The program checks:

- dimensions
- sample sizes
- portfolio weight sums
- numerical singularity
- optimization parameters
- root-finding conditions

## Error Handling

Invalid mathematical states generate exceptions such as:

- `std::invalid_argument`
- `std::runtime_error`

The `main()` function catches standard exceptions and reports failures.

---

# 62. Complexity Considerations

Important computational complexities include:

| Operation | Typical Complexity |
|---|---:|
| Vector addition | O(n) |
| Dot product | O(n) |
| Dense matrix multiplication | O(n³) |
| Gaussian elimination | O(n³) |
| Covariance matrix construction | O(m²T) |
| Power iteration | O(km²) |
| Numerical gradient | O(d) objective evaluations |
| Simplex projection | O(d log d) |
| Gradient descent | Depends on objective cost |
| Trapezoidal integration | O(n) |
| Simpson integration | O(n) |
| 0/1 knapsack DP | O(nC) |

Here:

- `n` is vector or matrix dimension where applicable.
- `m` is the number of assets.
- `T` is the number of observations.
- `k` is the number of power iterations.
- `d` is the number of optimization variables.
- `C` is integer knapsack capacity.

Complexity analysis should always be interpreted together with memory requirements and numerical properties.

---

# 63. Performance Considerations

For small educational datasets, straightforward implementations are appropriate because they make mathematical operations visible.

For larger systems, performance considerations change.

Potential improvements include:

- optimized matrix multiplication
- cache-friendly memory layouts
- BLAS-style numerical kernels
- QR decomposition
- Cholesky decomposition
- singular value decomposition
- sparse matrix representations
- vectorization
- parallel computation
- automatic differentiation
- specialized optimization algorithms

Performance improvements should not sacrifice numerical correctness without deliberate evaluation.

---

# 64. Numerical Stability Considerations

Production numerical systems should consider:

- scaling
- conditioning
- overflow
- underflow
- cancellation
- tolerance selection
- stable factorization methods
- reproducibility
- convergence criteria

For example, solving:

`Ax=b`

through explicit matrix inversion is generally less attractive numerically than using a suitable factorization.

Similarly, statistical calculations involving very large values can benefit from numerically stable algorithms rather than naive formulas.

---

# 65. Security Considerations

Mathematical algorithms are not inherently secure merely because they are mathematical.

When numerical systems are used in applications, security considerations can include:

- validating external inputs
- preventing malformed numerical data
- limiting resource-intensive requests
- handling NaN and infinity values
- avoiding uncontrolled iteration
- protecting financial or personal datasets
- preserving auditability
- validating model configuration
- separating user-controlled input from executable code

A mathematical calculation can be correct while the surrounding application remains insecure.

---

# 66. Production Considerations

A production mathematical system should distinguish between:

1. Mathematical correctness
2. Numerical accuracy
3. Software correctness
4. Input validation
5. Performance
6. Reproducibility
7. Observability
8. Domain assumptions

A result can be mathematically correct under an unsuitable model.

For example, a normal-distribution probability calculation may be implemented perfectly but still provide a poor real-world approximation if the underlying data are strongly heavy-tailed.

Model assumptions therefore require validation separately from code correctness.

---

# 67. Important Comparisons

## Mean vs Median

Mean:

- uses every observation
- sensitive to outliers
- useful for many mathematical models

Median:

- based on ordering
- more resistant to extreme observations
- useful for skewed data

## Variance vs Standard Deviation

Variance:

- squared units
- convenient mathematically

Standard deviation:

- original measurement units
- easier to interpret

## Covariance vs Correlation

Covariance:

- depends on measurement units
- indicates direction of joint variation

Correlation:

- standardized
- dimensionless
- bounded between -1 and 1

## Derivative vs Gradient

Derivative:

- commonly refers to one-dimensional change

Gradient:

- vector of partial derivatives for multivariable functions

## Local vs Global Minimum

Local:

- compares nearby points

Global:

- compares the complete feasible domain

## Continuous vs Discrete Optimization

Continuous:

- variables can take values over intervals

Discrete:

- variables are selected from separate possible values

## Analytical vs Numerical Methods

Analytical:

- derives an exact symbolic expression when possible

Numerical:

- computes an approximation

Analytical solutions can reveal structure clearly, while numerical methods are often necessary for complicated real-world systems.

---

# 68. Integrated Mathematical Relationships

The most important feature of these topics is their interdependence.

A quantitative system may proceed as follows:

1. Probability defines uncertainty.
2. Statistics estimates quantities from observations.
3. Linear algebra organizes observations into vectors and matrices.
4. Calculus measures how the model changes.
5. Optimization selects parameters or decisions.
6. Numerical analysis determines how reliably the calculations can be performed.
7. Software engineering turns the mathematics into a reproducible implementation.

The C++ portfolio case study demonstrates this chain directly.

---

# 69. Core Formula Reference

## Probability

`P(Aᶜ) = 1 - P(A)`

`P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`

`P(A|B) = P(A ∩ B)/P(B)`

`P(A ∩ B) = P(A|B)P(B)`

`P(B|A) = P(A|B)P(B)/P(A)`

## Binomial Distribution

`P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ`

`E[X] = np`

`Var(X) = np(1-p)`

## Statistics

`x̄ = Σxᵢ/n`

`σ² = Σ(xᵢ-μ)²/N`

`s² = Σ(xᵢ-x̄)²/(n-1)`

`z = (x-μ)/σ`

`r = Cov(X,Y)/(sₓsᵧ)`

## Linear Algebra

`Ax=b`

`(AB)ᵢⱼ = ΣAᵢₖBₖⱼ`

`||x||₂ = √(Σxᵢ²)`

`Av = λv`

`det([[a,b],[c,d]]) = ad-bc`

## Calculus

`f'(x) = lim(h→0)[f(x+h)-f(x)]/h`

`(fg)' = f'g + fg'`

`(f(g(x)))' = f'(g(x))g'(x)`

`∫ₐᵇf(x)dx = F(b)-F(a)`

## Optimization

`∇f(x*) = 0`

`xₖ₊₁ = xₖ - α∇f(xₖ)`

`xₖ₊₁ = xₖ - f(xₖ)/f'(xₖ)`

`Var(Rₚ) = wᵀΣw`

`E[Rₚ] = wᵀμ`

---

# 70. Implementation Coverage

| Mathematical Area | Python | JavaScript | C++ |
|---|---|---|---|
| Arithmetic | Yes | Yes | Yes |
| Functions | Yes | Yes | Yes |
| Probability | Yes | Yes | Yes |
| Bayes' theorem | Yes | Yes | Indirectly |
| Binomial distribution | Yes | Yes | No direct implementation |
| Monte Carlo | Yes | Yes | No direct simulation |
| Descriptive statistics | Yes | Yes | Yes |
| Covariance | Yes | Yes | Yes |
| Correlation | Yes | Yes | Yes |
| Regression | Yes | Yes | Yes |
| Vectors | Yes | Yes | Yes |
| Matrices | Yes | Yes | Yes |
| Gaussian elimination | Yes | Yes | Yes |
| Eigenvalue estimation | Yes | No | Yes |
| Limits | Yes | Yes | No direct example |
| Derivatives | Yes | Yes | Yes |
| Integration | Yes | Yes | No direct integration |
| Taylor series | Yes | Yes | No direct Taylor implementation |
| Gradient descent | Yes | Yes | Yes |
| Newton's method | Yes | Yes | Yes |
| Golden-section search | Yes | Yes | No direct implementation |
| Constrained optimization | Yes | Yes | Yes |
| Dynamic programming | Yes | Yes | No direct implementation |
| Normal distribution | Yes | Yes | Yes |
| Numerical stability | Yes | Yes | Yes |
| Financial mathematics | Yes | Yes | Yes |
| Portfolio optimization | Conceptual | Conceptual | Detailed case study |

---

# 71. Mathematical Interpretation of the Three Implementations

The Python implementation emphasizes breadth and experimentation.

The JavaScript implementation emphasizes executable numerical logic in an application-oriented language.

The C++ implementation emphasizes architecture and a realistic quantitative workflow.

Together, they demonstrate that the same mathematical ideas can be represented at different software abstraction levels without changing their underlying mathematical definitions.
