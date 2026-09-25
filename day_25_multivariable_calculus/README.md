# Multivariable Calculus: Gradients, Jacobians, Hessians, and Directional Derivatives

## 1. Topic Introduction

Multivariable calculus extends ordinary calculus from functions of one variable to functions involving several variables.

A single-variable function may have the form

`f(x) = x²`

whereas a multivariable scalar function may have the form

`f(x, y) = x² + 3xy + 2y²`.

Multivariable calculus provides tools for understanding how such functions change when several inputs can vary independently.

The principal objects studied in this implementation are:

- partial derivatives
- gradients
- directional derivatives
- Jacobians
- Hessians
- first-order linearization
- second-order Taylor approximation
- the multivariable chain rule
- critical-point classification
- gradient-based optimization

These concepts are closely related but serve different purposes.

For a scalar-valued function

`f: R^n -> R`

the gradient contains its first-order derivatives, while the Hessian contains its second-order derivatives.

For a vector-valued function

`F: R^n -> R^m`

the Jacobian contains all first-order partial derivatives.

---

## 2. Scalar-Valued and Vector-Valued Functions

### Scalar-valued functions

A scalar-valued function maps a vector of inputs to a single number.

`f: R^n -> R`

Example:

`f(x, y) = x² + 3xy + 2y²`

An input such as `(2, 1)` produces one output:

`f(2, 1) = 4 + 6 + 2 = 12`.

Scalar fields occur in:

- cost functions
- energy functions
- temperature fields
- pressure fields
- probability density functions
- loss functions in optimization
- physical potential functions

### Vector-valued functions

A vector-valued function produces multiple outputs.

`F: R^n -> R^m`

The C++ and Python implementations use

`F(x,y) = [x² + y, xy, sin(x) + cos(y)]`.

For an input `(x,y)`, the result is a three-dimensional vector.

Vector-valued mappings are common in:

- physical simulations
- coordinate transformations
- robotics
- computer graphics
- engineering systems
- machine-learning models
- differential equations

The distinction is important because the derivative of a scalar-valued function and the derivative of a vector-valued function are represented differently.

---

## 3. Partial Derivatives

A partial derivative measures the rate of change with respect to one variable while treating the other variables as locally fixed.

For

`f(x,y) = x² + 3xy + 2y²`

the partial derivative with respect to `x` is

`∂f/∂x = 2x + 3y`.

The partial derivative with respect to `y` is

`∂f/∂y = 3x + 4y`.

At `(2,1)`:

`∂f/∂x = 7`

and

`∂f/∂y = 10`.

These values describe how the function changes when moving along the coordinate directions.

A common conceptual mistake is to interpret a partial derivative as the complete rate of change of the function. It is only the rate associated with one input variable while the other variables are held fixed.

---

## 4. Gradient

The gradient collects all first-order partial derivatives of a scalar-valued function.

For a function of two variables,

`f(x,y)`

the gradient is

`∇f(x,y) = [∂f/∂x, ∂f/∂y]^T`.

For the polynomial surface used throughout the implementations,

`f(x,y) = x² + 3xy + 2y²`

the gradient is

`∇f(x,y) = [2x + 3y, 3x + 4y]^T`.

At `(2,1)`:

`∇f(2,1) = [7,10]^T`.

### Geometric meaning

The gradient points in the direction of greatest instantaneous increase of a differentiable scalar field.

Its magnitude

`||∇f||`

is the maximum directional derivative when directions are restricted to unit vectors.

The negative gradient points in the direction of greatest instantaneous decrease.

This relationship is fundamental to gradient descent.

---

## 5. Directional Derivatives

A partial derivative examines a coordinate direction. A directional derivative permits an arbitrary direction.

Let `u` be a unit vector.

The directional derivative is

`D_u f(x) = ∇f(x) · u`.

The dot product is essential.

If

`∇f = [a,b]`

and

`u = [u1,u2]`

then

`D_u f = au1 + bu2`.

### Why normalization matters

Suppose the intended direction is

`v = [3,4]`.

Its length is

`||v|| = 5`.

The corresponding unit vector is

`u = [3/5,4/5]`.

The directional derivative in the direction of `v` is therefore

`∇f · [3/5,4/5]`.

The implementations explicitly normalize direction vectors.

A frequent error is to use an arbitrary non-unit vector directly while calling the result a unit directional derivative. The unnormalized expression `∇f · v` is still mathematically meaningful, but it represents a rate associated with the specified vector magnitude rather than the derivative per unit distance in that direction.

---

## 6. Maximum Directional Derivative

For a differentiable function,

`D_u f = ∇f · u`

with

`||u|| = 1`.

By the Cauchy-Schwarz inequality,

`∇f · u <= ||∇f|| ||u||`.

Since `||u|| = 1`,

`D_u f <= ||∇f||`.

Equality occurs when `u` points in the same direction as the gradient.

Therefore:

- maximum directional derivative = `||∇f||`
- direction of maximum increase = `∇f / ||∇f||`
- direction of maximum decrease = `-∇f / ||∇f||`

A direction perpendicular to the gradient has directional derivative zero.

This is why gradients are normal to level curves and level surfaces under appropriate differentiability conditions.

---

## 7. Level Curves and the Gradient

A level curve is defined by

`f(x,y) = c`.

If a point moves along the level curve, the function value remains constant.

Therefore its instantaneous change is zero.

If `u` is a tangent direction,

`D_u f = ∇f · u = 0`.

Thus the gradient is perpendicular to the tangent direction.

For a sufficiently regular level curve, the gradient is therefore normal to the curve.

This provides an important geometric interpretation of the gradient beyond simply treating it as a collection of partial derivatives.

---

## 8. Jacobian Matrix

For a vector-valued function

`F: R^n -> R^m`

the Jacobian is an `m x n` matrix.

Its entries are

`J_ij = ∂F_i / ∂x_j`.

For

`F1 = x² + y`

`F2 = xy`

`F3 = sin(x) + cos(y)`

the Jacobian is

`J = [[2x, 1], [y, x], [cos(x), -sin(y)]]`.

There are three rows because there are three outputs.

There are two columns because there are two inputs.

### Interpretation

For a small input displacement `Δx`,

`F(x + Δx) ≈ F(x) + J(x)Δx`.

The Jacobian is therefore the linear transformation that best approximates the vector-valued mapping locally.

This makes the Jacobian the multivariable analogue of the derivative of a scalar one-variable function.

---

## 9. Jacobian Dimensions

For

`F: R^n -> R^m`

the Jacobian has dimensions

`m x n`.

Examples:

| Function | Jacobian dimensions |
|---|---:|
| `R -> R` | `1 x 1` |
| `R² -> R` | `1 x 2` |
| `R² -> R³` | `3 x 2` |
| `R³ -> R²` | `2 x 3` |
| `Rⁿ -> Rᵐ` | `m x n` |

A common implementation mistake is reversing the row and column interpretation.

Rows correspond to output components.

Columns correspond to input variables.

---

## 10. Gradient and Jacobian Relationship

For a scalar-valued function,

`f: R^n -> R`

the Jacobian is technically a `1 x n` row containing the partial derivatives.

The gradient is conventionally represented as an `n x 1` column vector.

Thus the two contain the same first-order derivative information but use different orientations.

For example,

`f: R² -> R`

has Jacobian

`J = [∂f/∂x, ∂f/∂y]`

while its gradient is

`∇f = [∂f/∂x, ∂f/∂y]^T`.

This distinction matters in matrix calculations and software implementations.

---

## 11. Hessian Matrix

The Hessian contains all second-order partial derivatives of a scalar-valued function.

For

`f(x,y)`

the Hessian is

`H = [[f_xx, f_xy], [f_yx, f_yy]]`.

For

`f(x,y) = x² + 3xy + 2y²`

the second derivatives are

`f_xx = 2`

`f_xy = 3`

`f_yx = 3`

`f_yy = 4`.

Therefore,

`H = [[2,3],[3,4]]`.

The Hessian measures local curvature.

The gradient describes first-order slope information.

The Hessian describes how that slope changes.

---

## 12. Mixed Partial Derivatives

The Hessian includes mixed derivatives such as

`f_xy`

and

`f_yx`.

Under suitable regularity conditions, particularly when the relevant second partial derivatives are continuous in a neighborhood, these mixed derivatives are equal.

Thus the Hessian is symmetric:

`H = H^T`.

The numerical implementations estimate the mixed derivative using a central finite-difference formula.

In unusual cases involving insufficient smoothness or pathological behavior, equality of mixed partial derivatives should not be assumed without the relevant mathematical conditions.

---

## 13. Curvature and the Hessian

The Hessian appears naturally in the second-order Taylor approximation.

For a displacement `Δx`,

`f(x + Δx) ≈ f(x) + ∇f(x)^TΔx + 1/2 Δx^T H(x) Δx`.

The three terms have different meanings:

1. `f(x)` is the current function value.
2. `∇f(x)^TΔx` is the first-order change.
3. `1/2 Δx^T H(x) Δx` is the second-order curvature correction.

The Hessian therefore provides information that the gradient alone cannot provide.

---

## 14. Second-Derivative Test in Two Dimensions

At a critical point where

`∇f = 0`

consider

`D = f_xx f_yy - (f_xy)^2`.

For a sufficiently smooth function:

- `D > 0` and `f_xx > 0` indicates a local minimum.
- `D > 0` and `f_xx < 0` indicates a local maximum.
- `D < 0` indicates a saddle point.
- `D = 0` makes the test inconclusive.

The word "inconclusive" is important.

A zero determinant does not automatically mean that there is no local minimum or maximum. It means this particular second-derivative test does not establish the classification.

---

## 15. Positive and Negative Definiteness

For higher-dimensional optimization, Hessian classification is more naturally expressed through definiteness.

A symmetric Hessian is:

- positive definite when `v^T H v > 0` for every nonzero `v`
- negative definite when `v^T H v < 0` for every nonzero `v`
- indefinite when the quadratic form takes both positive and negative values
- positive semidefinite when it is never negative
- negative semidefinite when it is never positive

At a critical point:

- positive definite Hessian indicates a strict local minimum
- negative definite Hessian indicates a strict local maximum
- indefinite Hessian indicates a saddle point

Semidefinite cases generally require additional analysis.

---

## 16. Numerical Differentiation

The implementations use central finite differences to verify analytical derivatives.

For a one-variable function,

`f'(x) ≈ [f(x+h) - f(x-h)] / (2h)`.

For a partial derivative,

`∂f/∂x ≈ [f(x+h,y) - f(x-h,y)] / (2h)`.

The central difference is generally more accurate than the simple forward difference for smooth functions at an appropriate step size.

Numerical differentiation is useful for:

- validating analytical formulas
- testing scientific software
- approximating derivatives of black-box functions
- prototyping numerical algorithms

It also has limitations.

If `h` is too large, truncation error becomes significant.

If `h` is too small, floating-point cancellation can become significant.

Consequently, numerical differentiation involves a trade-off in choosing the step size.

---

## 17. Analytical Versus Numerical Derivatives

Analytical differentiation derives exact formulas such as

`∂f/∂x = 2x + 3y`.

Numerical differentiation evaluates nearby function values to estimate the derivative.

| Aspect | Analytical | Numerical |
|---|---|---|
| Formula | Explicit | Estimated |
| Accuracy | Exact symbolically | Approximate |
| Runtime | Often efficient after derivation | Requires additional function evaluations |
| Implementation | Requires derivative logic | Can work with black-box functions |
| Floating-point sensitivity | Still present during evaluation | More sensitive to step size |
| Validation | Can be tested | Useful for checking analytical code |

The Python, JavaScript, and C++ programs intentionally use both approaches.

---

## 18. First-Order Linearization

Near a point `x0`, a differentiable scalar function can be approximated by

`f(x0 + Δx) ≈ f(x0) + ∇f(x0)^TΔx`.

This is the multivariable version of the tangent-line approximation.

For a vector-valued function,

`F(x0 + Δx) ≈ F(x0) + J(x0)Δx`.

The two formulas are structurally parallel:

- scalar function uses the gradient
- vector-valued function uses the Jacobian

---

## 19. Second-Order Taylor Approximation

For a scalar function,

`f(x0 + Δx) ≈ f(x0) + ∇f(x0)^TΔx + 1/2 Δx^T H(x0)Δx`.

The Python, JavaScript, and C++ implementations calculate first-order and second-order estimates and compare them with the exact function value.

For sufficiently small displacements, the second-order approximation can capture curvature that the linear approximation misses.

This is important in:

- numerical optimization
- mechanics
- control systems
- scientific computing
- uncertainty analysis
- local model construction

---

## 20. Multivariable Chain Rule

Suppose

`F: R^n -> R^m`

and a parameterized input is

`x = x(t)`.

Then

`dF/dt = J_F(x(t)) x'(t)`.

For a scalar function `f(x,y)` with

`x = x(t)`

and

`y = y(t)`,

the chain rule becomes

`df/dt = f_x dx/dt + f_y dy/dt`.

In vector notation,

`df/dt = ∇f · x'(t)`.

The Python and JavaScript implementations demonstrate this using

`x(t) = t²`

and

`y(t) = sin(t)`.

The C++ case study focuses on the Jacobian and optimization aspects of the same mathematical framework.

---

## 21. Gradient Descent

Gradient descent is an iterative optimization method.

For an objective `f(x)`, the basic update is

`x_(k+1) = x_k - α∇f(x_k)`.

Here:

- `x_k` is the current point
- `α` is the learning rate or step size
- `∇f(x_k)` is the gradient

The negative gradient is used because it points toward local decrease.

The implementations use a quadratic objective for which gradient descent converges toward the minimum when the step size is appropriately selected.

---

## 22. Learning Rate and Convergence

The learning rate controls the size of each optimization step.

If it is too small:

- convergence can be slow
- many iterations may be required

If it is too large:

- the algorithm can overshoot
- oscillations can occur
- the objective can diverge

For more complicated objectives, local curvature matters.

The Hessian can provide information about curvature and is central to more advanced second-order optimization methods.

The simple implementation deliberately uses gradient descent so that the relationship between the gradient and optimization remains explicit.

---

## 23. Python Implementation

The Python program is organized as a standalone educational numerical-calculus environment.

Important components include:

- vector arithmetic
- scalar functions
- vector-valued functions
- analytical gradients
- numerical gradients
- directional derivatives
- analytical Jacobians
- numerical Jacobians
- analytical Hessians
- numerical Hessians
- Taylor approximations
- critical-point classification
- gradient descent
- the multivariable chain rule
- verification tests

Python is particularly convenient for expressing mathematical algorithms because vector and matrix operations can be implemented directly with lists without requiring an external numerical package.

The script also separates mathematical operations into functions. This makes each concept independently testable.

---

## 24. Python Gradient Demonstration

The function `surface(x, y)` represents

`f(x,y) = x² + 3xy + 2y²`.

The function `analytical_gradient(x, y)` implements

`∇f = [2x + 3y, 3x + 4y]`.

The numerical function `numerical_gradient` independently estimates the same derivatives.

Comparing the two implementations is useful because it tests the mathematical derivative against an independent computational approximation.

---

## 25. Python Directional-Derivative Demonstration

The Python function `directional_derivative_from_gradient` first normalizes the direction and then calculates

`∇f · u`.

The program tests multiple directions, including:

- the positive x direction
- the positive y direction
- the diagonal direction
- the negative diagonal direction

This demonstrates that the directional derivative depends on orientation, not merely on the coordinate partial derivatives separately.

---

## 26. Python Jacobian Demonstration

The vector-valued function returns three outputs for two inputs.

Its Jacobian therefore has shape `3 x 2`.

The program implements both:

- an analytical Jacobian
- a numerical Jacobian

The comparison illustrates how the Jacobian is constructed row by row from the derivatives of each output component.

---

## 27. Python Hessian Demonstration

The polynomial surface has the constant Hessian

`[[2,3],[3,4]]`.

The numerical Hessian implementation independently estimates its entries.

The program also demonstrates the two-dimensional second-derivative classification test.

Because the determinant is

`2(4) - 3² = 8 - 9 = -1`

the Hessian is indefinite and the associated stationary behavior is saddle-like.

This example also demonstrates why inspecting only one second derivative is insufficient for classifying a multivariable critical point.

---

## 28. JavaScript Implementation

The JavaScript implementation mirrors the mathematical framework while emphasizing language-level numerical programming.

It demonstrates:

- array-based vector representation
- matrix representation
- scalar functions
- vector-valued functions
- numerical derivatives
- analytical derivatives
- directional derivatives
- Jacobians
- Hessians
- Taylor approximation
- chain rule
- gradient descent
- error handling
- executable assertions

The file is compatible with a modern Node.js runtime and uses only standard JavaScript functionality.

No npm package is required.

---

## 29. JavaScript-Specific Considerations

JavaScript represents numbers using IEEE 754 double-precision floating-point arithmetic.

Therefore, calculations involving very small finite-difference steps can encounter floating-point effects.

The implementation uses ordinary JavaScript arrays for vectors and matrices.

This is suitable for educational examples and small calculations.

For large numerical workloads, specialized numerical libraries or typed-array-based implementations may be appropriate, but the underlying calculus remains the same.

The important mathematical distinction is independent of the programming language.

---

## 30. C++ Case Study

The C++ program models an engineering-style operating-cost problem.

The scalar objective is

`C(x,y) = 4x² + 2y² + xy - 12x - 8y + 30`.

Here:

- `x` represents processing intensity
- `y` represents cooling intensity

The objective maps two control variables to one scalar cost.

Therefore,

`C: R² -> R`.

The program uses the model to demonstrate the complete derivative hierarchy.

---

## 31. C++ System Architecture

The C++ case study uses a `ProcessModel` class.

Its responsibilities include:

- evaluating the scalar cost
- computing the analytical gradient
- providing the analytical Hessian
- generating vector-valued system outputs
- computing the analytical Jacobian

Numerical differentiation functions are kept outside the model class.

This separation distinguishes the mathematical model from numerical verification machinery.

It also makes the architecture easier to extend.

---

## 32. C++ Gradient

For

`C(x,y) = 4x² + 2y² + xy - 12x - 8y + 30`

the gradient is

`∇C = [8x + y - 12, 4y + x - 8]^T`.

The gradient represents the sensitivity of operating cost to changes in the two control variables.

At a stationary point,

`∇C = 0`.

Therefore the stationary point satisfies

`8x + y - 12 = 0`

and

`x + 4y - 8 = 0`.

Solving these equations gives the stationary operating point.

The Hessian confirms the nature of that point.

---

## 33. C++ Hessian

The Hessian is

`H = [[8,1],[1,4]]`.

Its determinant is

`8(4) - 1(1) = 31`.

Because the determinant is positive and `f_xx = 8 > 0`, the two-dimensional second-derivative test identifies the stationary point as a local minimum.

In this quadratic model, the Hessian is constant throughout the domain.

This makes the model useful for illustrating the relationship between curvature and optimization.

---

## 34. C++ Jacobian

The case study also defines the vector-valued output

`F(x,y) = [x² + y, xy, sin(x) + cos(y)]`.

Its Jacobian is

`J = [[2x,1],[y,x],[cos(x),-sin(y)]]`.

The C++ program compares the analytical Jacobian against a numerical finite-difference Jacobian.

This demonstrates that a system with several outputs requires a matrix of first derivatives rather than a single gradient vector.

---

## 35. C++ Directional Derivatives

For a chosen direction `v`, the implementation computes

`u = v / ||v||`

and then

`D_u C = ∇C · u`.

This can be interpreted as the instantaneous change in cost per unit movement along the selected operating direction.

The program compares the analytical result with a numerical central-difference approximation.

---

## 36. C++ Taylor Approximation

The C++ program evaluates a base point, applies a small displacement, and compares:

1. the exact objective value
2. the first-order Taylor approximation
3. the second-order Taylor approximation

The first-order model uses the gradient.

The second-order model uses both the gradient and Hessian.

This demonstrates why curvature becomes relevant when a linear approximation is not sufficiently accurate.

---

## 37. C++ Optimization

The gradient-descent implementation follows

`x_(k+1) = x_k - α∇C(x_k)`.

The program records the objective value after each iteration.

The recorded history makes it possible to inspect whether the optimization process is decreasing the objective.

The case study uses a fixed learning rate for clarity.

Production optimization systems often require more sophisticated techniques such as adaptive step sizes, line searches, trust-region methods, momentum, quasi-Newton methods, or constrained optimization.

Those methods build on the same derivative concepts demonstrated here.

---

## 38. Important Distinctions

### Gradient versus directional derivative

The gradient is a vector.

The directional derivative is a scalar.

The gradient contains all first-order directional information.

A directional derivative extracts the rate associated with one direction.

### Gradient versus Jacobian

The gradient conventionally applies to scalar-valued functions.

The Jacobian applies to vector-valued mappings.

A scalar function's Jacobian contains essentially the same first-order information as its gradient but with a different orientation convention.

### Jacobian versus Hessian

The Jacobian contains first-order derivatives of a vector-valued function.

The Hessian contains second-order derivatives of a scalar-valued function.

### Gradient versus Hessian

The gradient describes slope.

The Hessian describes how slope changes.

The gradient is first-order information.

The Hessian is second-order information.

---

## 39. Common Mistakes

### Mistake 1: Forgetting direction normalization

Using

`∇f · v`

when the problem explicitly defines `v` as a non-unit direction can produce a result scaled by `||v||`.

### Mistake 2: Confusing Jacobian dimensions

For `F: R² -> R³`, the Jacobian is `3 x 2`, not `2 x 3`.

### Mistake 3: Treating the Hessian as the gradient

The gradient contains first derivatives.

The Hessian contains second derivatives.

### Mistake 4: Assuming a zero Hessian determinant proves a saddle point

It does not.

A zero determinant makes the standard two-dimensional second-derivative test inconclusive.

### Mistake 5: Ignoring mixed derivatives

A Hessian is not constructed only from `f_xx` and `f_yy`.

The mixed derivatives `f_xy` and `f_yx` are also required.

### Mistake 6: Using finite differences without considering step size

Very large or very small finite-difference steps can reduce numerical accuracy.

### Mistake 7: Assuming gradient descent always converges

Convergence depends on the objective function, step size, starting point, smoothness, curvature, and other algorithmic conditions.

---

## 40. Edge Cases

Important edge cases include:

- zero direction vectors
- mismatched vector dimensions
- singular or degenerate Hessians
- non-smooth functions
- discontinuous derivatives
- extremely large values
- floating-point cancellation
- unstable finite-difference step sizes
- optimization divergence
- stationary points that are not minima
- functions with multiple local extrema

The implementations explicitly reject zero vectors during normalization and detect dimension mismatches.

---

## 41. Differentiability Considerations

A function can have partial derivatives without being fully differentiable.

The existence of individual partial derivatives at a point does not, by itself, guarantee that a multivariable linear approximation exists.

Differentiability is stronger than merely having partial derivatives.

When the first partial derivatives exist in a neighborhood and satisfy appropriate continuity conditions, differentiability can often be established.

This distinction is important when applying gradient-based reasoning rigorously.

---

## 42. Limitations of Numerical Differentiation

Finite differences approximate derivatives rather than computing symbolic derivatives.

Potential sources of error include:

- truncation error
- round-off error
- cancellation
- noisy function evaluations
- inappropriate step size
- discontinuities
- non-smooth behavior

For expensive functions, numerical differentiation may also require many additional function evaluations.

For an `n`-variable scalar function, estimating all first partial derivatives with central differences generally requires roughly `2n` additional evaluations.

For Hessians, the number of evaluations can grow substantially with dimension.

---

## 43. Performance Considerations

For a function with `n` variables:

- a gradient contains `n` components
- a Hessian contains `n²` entries before exploiting symmetry
- a dense Jacobian for `R^n -> R^m` contains `mn` entries

A dense Hessian therefore has quadratic storage requirements in the number of variables.

In large-scale numerical optimization, exploiting sparsity can dramatically reduce memory and computation.

The same principle applies to Jacobians.

Sparse derivative structures are common in:

- large simulations
- optimization systems
- computational physics
- graph-based models
- scientific computing

---

## 44. Security and Reliability Considerations

Pure calculus calculations do not normally introduce conventional security risks.

Software implementing numerical calculus can still encounter reliability issues.

Important defensive practices include:

- validating dimensions
- rejecting invalid direction vectors
- checking numerical results
- controlling iteration counts
- detecting non-finite values
- avoiding uncontrolled resource consumption
- testing derivative implementations independently

For production numerical services, input validation and resource limits become especially important when users can provide arbitrary functions, dimensions, or optimization parameters.

---

## 45. Production Implementation Considerations

A production-grade derivative engine may need:

- vectorized operations
- sparse matrices
- automatic differentiation
- symbolic differentiation
- numerical stability controls
- configurable tolerances
- convergence criteria
- logging
- deterministic testing
- overflow and underflow detection
- parallel computation
- memory-aware data structures

The educational implementations intentionally avoid external dependencies so that the mathematical mechanisms remain visible.

---

## 46. Relationship to Automatic Differentiation

Automatic differentiation computes derivatives by systematically applying the chain rule to a computational representation of a function.

It differs from:

- symbolic differentiation, which manipulates mathematical expressions
- finite differences, which approximate derivatives numerically

Automatic differentiation can produce highly accurate derivative information without requiring the developer to manually derive every expression.

The fundamental objects remain the same:

- gradients
- Jacobians
- Hessians
- directional derivatives

The implementations in this project manually encode derivatives so the mathematical structure is explicit.

---

## 47. Relationship to Optimization

Gradients and Hessians are central to optimization.

First-order methods use gradients.

Second-order methods use curvature information.

Examples include:

- gradient descent
- Newton's method
- quasi-Newton methods
- trust-region methods

Newton's method for a scalar objective conceptually uses

`x_(k+1) = x_k - H(x_k)^(-1) ∇f(x_k)`.

This demonstrates why the Hessian is more than an academic extension of the gradient: it provides local curvature information that can influence optimization steps.

---

## 48. Real-World Applications

### Engineering

Gradients can identify how a system responds to changes in design parameters.

### Physics

Gradients of scalar potential fields are used to determine vector fields and forces under appropriate physical definitions.

### Machine learning

Gradients drive parameter optimization for differentiable models.

### Robotics

Jacobians relate joint-space velocities to end-effector velocities.

### Computer graphics

Jacobians appear in coordinate transformations, deformation models, and sensitivity calculations.

### Economics and finance

Gradients can describe sensitivity of objectives to multiple economic variables.

### Scientific computing

Hessians and Jacobians are important in nonlinear equation solving and numerical optimization.

### Control systems

Derivatives describe local system sensitivity and support linearization.

### Operations research

Gradients and Hessians support constrained and unconstrained optimization.

---

## 49. Conceptual Hierarchy

The four central objects can be viewed as a derivative hierarchy.

### Gradient

Input:

`f: R^n -> R`

Output:

a vector of first derivatives.

Primary role:

first-order sensitivity of a scalar field.

### Directional derivative

Input:

a scalar function plus a direction.

Output:

one scalar rate of change.

Primary role:

measure change along a specified direction.

### Jacobian

Input:

`F: R^n -> R^m`

Output:

an `m x n` matrix.

Primary role:

first-order sensitivity of multiple outputs to multiple inputs.

### Hessian

Input:

`f: R^n -> R`

Output:

an `n x n` matrix.

Primary role:

second-order curvature of a scalar field.

---

## 50. Implementation Comparison

| Concept | Python | JavaScript | C++ |
|---|---|---|---|
| Scalar functions | Yes | Yes | Yes |
| Vector operations | Yes | Yes | Yes |
| Numerical gradients | Yes | Yes | Yes |
| Directional derivatives | Yes | Yes | Yes |
| Jacobians | Yes | Yes | Yes |
| Hessians | Yes | Yes | Yes |
| Taylor approximation | Yes | Yes | Yes |
| Chain rule | Yes | Yes | Indirectly represented |
| Gradient descent | Yes | Yes | Yes |
| Validation tests | Yes | Yes | Yes |
| Case-study architecture | General study script | Executable numerical file | Engineering model class |

Python emphasizes readable mathematical experimentation.

JavaScript demonstrates the same ideas in a language commonly used for application and web development.

C++ demonstrates a structured, performance-oriented implementation with explicit types, classes, exception handling, and a realistic engineering case study.

---

## 51. Complexity Considerations

For a dense vector of dimension `n`:

- vector addition is `O(n)`
- dot product is `O(n)`
- normalization is `O(n)`

For a dense `m x n` Jacobian:

- storing the matrix requires `O(mn)` space
- matrix-vector multiplication requires `O(mn)` arithmetic operations

For an `n x n` dense Hessian:

- storage is `O(n²)`
- dense matrix-vector multiplication is `O(n²)`

These costs become important as dimensionality increases.

---

## 52. Mathematical Formulas Implemented

Gradient:

`∇f(x) = [∂f/∂x1, ..., ∂f/∂xn]^T`

Directional derivative:

`D_u f(x) = ∇f(x) · u`

with

`||u|| = 1`.

Jacobian:

`J_ij = ∂F_i/∂x_j`

Hessian:

`H_ij = ∂²f/(∂x_i∂x_j)`

First-order approximation:

`f(x+Δx) ≈ f(x) + ∇f(x)^TΔx`

Second-order approximation:

`f(x+Δx) ≈ f(x) + ∇f(x)^TΔx + 1/2 Δx^T H(x)Δx`

Gradient descent:

`x_(k+1) = x_k - α∇f(x_k)`

Multivariable chain rule:

`dF/dt = J_F(x(t))x'(t)`

---

## 53. Practical Interpretation

The most useful way to distinguish the concepts is to ask what question is being answered.

If the question is:

"How does the scalar quantity change with every input?"

Use the gradient.

If the question is:

"How does the scalar quantity change in this particular direction?"

Use a directional derivative.

If the question is:

"How do all outputs change when all inputs change?"

Use the Jacobian.

If the question is:

"How does the gradient itself change?"

Use the Hessian.

This distinction provides a practical mental model for deciding which derivative object is appropriate.

---

## 54. Validation Strategy Used by the Implementations

The implementations intentionally contain both analytical and numerical calculations.

The validation process is:

1. derive the derivative analytically
2. implement the analytical expression
3. independently estimate the derivative numerically
4. compare the results
5. execute mathematical classification tests
6. test invalid inputs and edge cases

This approach is useful because derivative code can be mathematically wrong while still being syntactically valid.

Independent numerical checks can expose implementation mistakes.

---

## 55. Testing Strategy

The programs verify:

- gradient components
- Hessian components
- directional derivatives
- Jacobian components
- critical-point classifications
- invalid zero-direction handling
- dimension validation

Testing numerical mathematics requires tolerances rather than exact equality in many cases.

For example, an expected value of `5` and a computed value of `5.00000001` should normally be treated as equivalent within an appropriate tolerance.

The correct tolerance depends on the numerical method, scale, precision, and conditioning of the problem.

---

## 56. Final Conceptual Distinction

The mathematical structure can be expressed compactly:

`f: R^n -> R`

has a gradient

`∇f`

and Hessian

`H_f`.

A direction `u` extracts a scalar rate

`D_u f = ∇f · u`.

A vector-valued mapping

`F: R^n -> R^m`

has a Jacobian

`J_F`.

The Jacobian is the first-order linear approximation of `F`, while the Hessian captures second-order curvature for scalar-valued functions.

Together, these objects form a central part of multivariable differential calculus and provide the mathematical foundation for sensitivity analysis, local approximation, numerical optimization, scientific computing, engineering models, and many computational methods.
