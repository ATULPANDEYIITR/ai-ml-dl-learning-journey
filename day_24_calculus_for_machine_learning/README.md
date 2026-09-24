# Calculus for Machine Learning

## Topic scope

This project studies the calculus concepts that form the mathematical foundation of many machine-learning algorithms:

- Functions
- Limits
- Continuity
- Derivatives
- Partial derivatives
- Gradients
- Directional derivatives
- Jacobians
- Hessians
- Chain rule
- Taylor approximation
- Numerical differentiation
- Automatic differentiation
- Loss functions
- Gradient descent
- Logistic regression
- Activation functions
- Softmax
- Cross-entropy
- Backpropagation concepts
- Numerical stability
- Optimization and computational complexity

The three implementations approach the same mathematical subject from different perspectives.

The Python program is a broad educational implementation containing scalar calculus, multivariable calculus, optimization, automatic differentiation, loss functions, logistic regression, softmax, and neural-network concepts.

The JavaScript program emphasizes functional programming, executable numerical demonstrations, object-oriented model construction, browser-oriented integration patterns, and automatic differentiation.

The C++ program develops an industry-style binary-classification case study with explicit data structures, validation, gradient computation, numerical verification, training, evaluation, complexity analysis, and numerical-stability considerations.

---

## Introduction

Calculus studies continuous change. Machine learning repeatedly asks a closely related question:

> How does a model's output or error change when one of its inputs or parameters changes?

A derivative answers this question locally for one variable. Partial derivatives extend the idea to functions with several variables. A gradient collects partial derivatives into a vector. The chain rule explains how derivatives propagate through a sequence of operations.

These concepts become directly useful when a machine-learning system contains a parameterized function

`y_hat = f(x; theta)`

and an objective function

`L(theta)`

must be minimized.

Here:

- `x` represents input data.
- `theta` represents model parameters.
- `y_hat` represents a prediction.
- `L` represents a loss or objective.
- `dL/dtheta` describes how the objective changes with respect to the parameters.

Gradient-based optimization uses this information to update the parameters.

---

## Functions

A function maps elements from a domain to elements in a range.

For a scalar function:

`f: R -> R`

the function accepts one real number and produces one real number.

For example:

`f(x) = x^2 + 2x + 1`

At `x = 3`:

`f(3) = 9 + 6 + 1 = 16`

Machine-learning models are also functions. A simple linear model is:

`y_hat = wx + b`

where:

- `w` is a weight.
- `x` is an input.
- `b` is a bias.
- `y_hat` is the predicted output.

For multiple features:

`y_hat = w_1x_1 + w_2x_2 + ... + w_dx_d + b`

This relationship is fundamental to linear regression, logistic regression, neural networks, and many other models.

### Function composition

If:

`g(x) = x + 3`

and:

`f(x) = x^2`

then:

`(f o g)(x) = f(g(x))`

which gives:

`(x + 3)^2`

Function composition is particularly important in neural networks because a network is a composition of transformations.

A simplified network can be written as:

`h = f_1(x)`

`z = f_2(h)`

`y_hat = f_3(z)`

The chain rule makes it possible to calculate how the final output changes with respect to an earlier input or parameter.

---

## Domains and ranges

A function is defined only where its mathematical expression is valid.

For example:

`f(x) = log(x)`

is defined over the real numbers only when:

`x > 0`

Similarly:

`f(x) = 1/x`

is undefined at:

`x = 0`

These restrictions matter in numerical machine-learning systems. Operations such as logarithms, divisions, square roots, exponentials, and inverse trigonometric functions have domain or numerical constraints.

Loss functions frequently require explicit numerical protection. Binary cross-entropy contains logarithms, so probabilities are commonly clipped away from exactly zero and one when calculations are performed directly in floating-point arithmetic.

---

## Limits

The limit describes the value that a function approaches as its input approaches a particular point.

The notation

`lim(x -> a) f(x)`

asks what happens to `f(x)` as `x` gets arbitrarily close to `a`.

For:

`f(x) = x^2`

the limit as `x` approaches `2` is:

`4`

A limit does not necessarily require the function to be defined at the point itself.

### One-sided limits

A left-hand limit approaches a point from smaller values:

`lim(x -> a-) f(x)`

A right-hand limit approaches the point from larger values:

`lim(x -> a+) f(x)`

A two-sided limit exists when the relevant left and right limits agree.

The Python and JavaScript implementations numerically inspect values close to a point. These calculations illustrate the concept but do not constitute a mathematical proof of a limit.

---

## Continuity

A function is continuous at `a` when:

1. `f(a)` exists.
2. `lim(x -> a) f(x)` exists.
3. The limit equals the function value.

In symbolic form:

`lim(x -> a) f(x) = f(a)`

Polynomials are continuous over the real numbers. Functions involving division, logarithms, roots, or piecewise definitions require more careful domain analysis.

Continuity and differentiability are different properties.

A function can be continuous but not differentiable at a point. The standard ReLU activation function demonstrates this:

`ReLU(x) = max(0, x)`

It is continuous at zero but has no classical derivative at zero because the left-hand slope is zero and the right-hand slope is one.

---

## Derivatives

The derivative is defined using a limit:

`f'(x) = lim(h -> 0) [f(x+h) - f(x)] / h`

It measures the instantaneous rate of change of a function.

For:

`f(x) = x^2`

the derivative is:

`f'(x) = 2x`

At `x = 3`:

`f'(3) = 6`

The derivative has several useful interpretations:

- Local rate of change
- Slope of the tangent line
- Local sensitivity
- First-order approximation coefficient

The sensitivity interpretation is especially useful in machine learning.

If a loss changes rapidly when a parameter changes, the magnitude of the corresponding derivative is large. If the derivative is close to zero, a small parameter change has little first-order effect on the loss.

---

## Basic differentiation rules

### Constant rule

For a constant `c`:

`d(c)/dx = 0`

### Power rule

For:

`f(x) = x^n`

the derivative is:

`f'(x) = nx^(n-1)`

Examples:

`d(x^2)/dx = 2x`

`d(x^3)/dx = 3x^2`

`d(x^4)/dx = 4x^3`

### Sum rule

`d(f + g)/dx = f' + g'`

### Product rule

`d(fg)/dx = f'g + fg'`

### Quotient rule

`d(f/g)/dx = (f'g - fg') / g^2`

where `g` is nonzero.

### Common elementary derivatives

`d(exp(x))/dx = exp(x)`

`d(log(x))/dx = 1/x`

`d(sin(x))/dx = cos(x)`

`d(cos(x))/dx = -sin(x)`

These rules are used repeatedly when deriving machine-learning gradients.

---

## Numerical differentiation

A numerical derivative approximates a derivative using function evaluations.

A forward difference is:

`f'(x) ≈ [f(x+h) - f(x)] / h`

A backward difference is:

`f'(x) ≈ [f(x) - f(x-h)] / h`

A central difference is:

`f'(x) ≈ [f(x+h) - f(x-h)] / (2h)`

The central difference is often more accurate for smooth functions because it uses information on both sides of the point.

The implementations demonstrate that decreasing `h` does not guarantee continuously improving accuracy.

Two competing effects exist:

- Truncation error tends to decrease as the step becomes smaller.
- Floating-point round-off can become more important as the step becomes extremely small.

Therefore, numerical differentiation requires a practical step-size choice.

---

## Higher-order derivatives

The second derivative is the derivative of the first derivative:

`f''(x) = d²f/dx²`

For:

`f(x) = x^4`

the first derivative is:

`f'(x) = 4x^3`

and the second derivative is:

`f''(x) = 12x^2`

The second derivative describes local curvature.

In optimization, curvature helps characterize whether a stationary point behaves like a local minimum, local maximum, or another type of stationary point.

---

## Partial derivatives

A multivariable function can depend on several variables.

For example:

`f(x,y) = x² + 3xy + y²`

The partial derivative with respect to `x` is:

`∂f/∂x = 2x + 3y`

The partial derivative with respect to `y` is:

`∂f/∂y = 3x + 2y`

When calculating `∂f/∂x`, `y` is treated as constant. When calculating `∂f/∂y`, `x` is treated as constant.

This concept maps directly onto machine-learning parameters.

If a model has parameters:

`theta = [theta_1, theta_2, ..., theta_d]`

and loss:

`L(theta)`

then the derivative with respect to parameter `theta_i` is:

`∂L/∂theta_i`

The complete collection of these partial derivatives forms the gradient.

---

## Gradient

For a scalar-valued function of several variables:

`f(x_1, x_2, ..., x_d)`

the gradient is:

`∇f = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_d]`

The gradient is a vector.

It points in the direction of greatest local increase of the function.

Therefore, the negative gradient:

`-∇f`

points in the direction of greatest local decrease.

This is the mathematical basis of gradient descent.

---

## Directional derivatives

A directional derivative measures the rate of change of a function in a particular direction.

For a unit vector `u`:

`D_u f(x) = ∇f(x) · u`

The direction must be normalized when this formula is interpreted as the rate of change per unit distance.

The Python and JavaScript implementations calculate directional derivatives numerically and normalize the direction vector first.

A zero direction vector cannot be normalized and is therefore rejected.

---

## Jacobian

The Jacobian generalizes first-order derivatives to vector-valued functions.

Suppose:

`F(x,y) = [f_1(x,y), f_2(x,y)]`

Then the Jacobian is:

`J = [[∂f_1/∂x, ∂f_1/∂y],
      [∂f_2/∂x, ∂f_2/∂y]]`

The Jacobian is central to:

- Multivariable transformations
- Neural-network layer derivatives
- Coordinate transformations
- Sensitivity analysis
- Automatic differentiation
- Optimization
- Numerical methods

For a scalar output, the Jacobian reduces to a gradient represented in a compatible orientation.

---

## Hessian

For a scalar-valued function with several variables, the Hessian contains all second-order partial derivatives.

For two variables:

`H = [[∂²f/∂x², ∂²f/∂x∂y],
     [∂²f/∂y∂x, ∂²f/∂y²]]`

For sufficiently smooth functions, the mixed partial derivatives commonly agree:

`∂²f/∂x∂y = ∂²f/∂y∂x`

The Hessian provides information about curvature.

For a quadratic function such as:

`f(x,y) = x² + 3y² + xy`

the Hessian is constant because all second derivatives are constant.

Large machine-learning models may have millions or billions of parameters. A full Hessian would contain an enormous number of entries, so explicitly constructing it is often impractical.

This explains the importance of first-order optimization methods in large-scale machine learning.

---

## Chain rule

The chain rule describes how derivatives propagate through composed functions.

If:

`y = f(g(x))`

then:

`dy/dx = f'(g(x))g'(x)`

For:

`y = sin(x²)`

the inner function is:

`g(x) = x²`

and the outer function is:

`f(g) = sin(g)`

Therefore:

`dy/dx = cos(x²) * 2x`

Neural networks are compositions of functions. A simplified neural network can be represented as:

`h = activation(Wx + b)`

`y_hat = output(h)`

The derivative of the final loss with respect to an early parameter is obtained by multiplying local derivative factors along the computational path.

This repeated application of the chain rule is the mathematical foundation of backpropagation.

---

## Taylor approximation

A first-order Taylor approximation around `a` is:

`f(x) ≈ f(a) + f'(a)(x-a)`

A second-order approximation is:

`f(x) ≈ f(a) + f'(a)(x-a) + 1/2 f''(a)(x-a)^2`

The first-order approximation describes a local tangent-line model.

The second-order approximation incorporates curvature.

Taylor expansions help explain why gradients are useful in optimization. Near a point, a sufficiently smooth function can often be approximated locally by a linear or quadratic expression.

---

## Loss functions

Machine-learning models require an objective that quantifies prediction error.

### Mean squared error

For predictions `y_hat_i` and targets `y_i`:

`MSE = (1/n) Σ(y_hat_i - y_i)^2`

The derivative with respect to prediction `y_hat_i` is:

`∂MSE/∂y_hat_i = 2(y_hat_i - y_i)/n`

The Python and JavaScript programs calculate both the loss and its derivative with respect to predictions.

The C++ case study uses a different objective because its model is a binary classifier.

---

## Binary cross-entropy

For a binary target `y` and predicted probability `p`:

`L = -[y log(p) + (1-y) log(1-p)]`

For multiple observations, the mean loss is normally used.

The logarithm creates important numerical edge cases.

When `p = 0` and `y = 1`, the expression contains:

`log(0)`

which is undefined.

Direct implementations therefore commonly clip probabilities to a small positive range before applying logarithms.

Production numerical libraries can use specialized stable formulations that avoid unnecessary intermediate overflow or underflow.

---

## Sigmoid function

The sigmoid function is:

`sigmoid(z) = 1 / (1 + exp(-z))`

It maps real-valued inputs into the interval `(0,1)`.

Its derivative can be written in terms of its output:

`sigmoid'(z) = sigmoid(z)(1-sigmoid(z))`

This identity is especially convenient in neural-network implementations.

For logistic regression:

`z = w^T x + b`

`p = sigmoid(z)`

The probability `p` is interpreted as the estimated probability of the positive class.

---

## Logistic regression gradient

For one observation:

`z = w^T x + b`

`p = sigmoid(z)`

with binary cross-entropy loss:

`L = -[y log(p) + (1-y) log(1-p)]`

The combination of sigmoid and binary cross-entropy produces the important simplification:

`∂L/∂z = p-y`

Using the chain rule:

`∂L/∂w = (p-y)x`

and:

`∂L/∂b = p-y`

For a dataset, the gradients are averaged across the observations.

This derivation is implemented explicitly in all three programming languages, although the C++ implementation is organized as the main technical case study.

---

## Gradient descent

Gradient descent updates parameters in the negative gradient direction.

The basic update is:

`theta_new = theta_old - alpha * ∇L(theta_old)`

where `alpha` is the learning rate.

The learning rate determines the size of each update.

### Learning rate too small

Updates may be very small, resulting in slow optimization.

### Learning rate too large

Updates may overshoot useful regions, oscillate, or diverge.

### Appropriate learning rate

Updates can move efficiently toward a region with lower objective value.

The exact behavior depends on the shape of the objective function, parameter scaling, initialization, and optimization method.

---

## Stationary points

A stationary point is a point where the gradient is zero:

`∇f = 0`

A stationary point is not automatically a minimum.

It may be:

- A local minimum
- A local maximum
- A saddle point
- Part of a flat region

Second-order information can help distinguish these cases, although practical high-dimensional optimization is more complicated than a simple one-dimensional second-derivative test.

---

## Activation functions

The JavaScript and Python implementations demonstrate ReLU and sigmoid.

### ReLU

`ReLU(x) = max(0,x)`

For nonzero inputs:

`ReLU'(x) = 0` when `x < 0`

`ReLU'(x) = 1` when `x > 0`

At zero, the classical derivative does not exist.

Software implementations must choose a convention for the derivative at the non-differentiable point. A common simple convention is zero.

### Sigmoid

`sigmoid(x) = 1/(1+exp(-x))`

`sigmoid'(x) = sigmoid(x)(1-sigmoid(x))`

Sigmoid is differentiable everywhere, but its derivative becomes small for very large positive or negative inputs. This saturation can affect gradient propagation.

---

## Softmax

For logits `z_1, ..., z_K`, softmax produces:

`softmax(z_i) = exp(z_i) / Σ exp(z_j)`

The resulting values are positive and sum to one.

Softmax is commonly used to convert multiclass logits into a probability distribution.

### Numerical stability

Directly computing `exp(z_i)` can overflow when logits are large.

The stable implementation subtracts the largest logit:

`softmax(z_i) = exp(z_i - max(z)) / Σ exp(z_j - max(z))`

The subtraction does not change the resulting probabilities because the same multiplicative factor is applied to every exponential and cancels during normalization.

The Python and JavaScript implementations explicitly demonstrate this technique.

---

## Multiclass cross-entropy

If the correct class is `k` and the model assigns it probability `p_k`, the loss is:

`L = -log(p_k)`

Assigning a very small probability to the correct class produces a large penalty.

The derivative of softmax combined with cross-entropy has useful simplifications analogous to sigmoid combined with binary cross-entropy.

---

## Automatic differentiation

Automatic differentiation is different from symbolic differentiation and finite differences.

### Symbolic differentiation

Symbolic differentiation manipulates mathematical expressions to produce another expression representing the derivative.

### Numerical differentiation

Numerical differentiation estimates derivatives from function evaluations.

For example:

`f'(x) ≈ [f(x+h)-f(x-h)]/(2h)`

### Automatic differentiation

Automatic differentiation applies the chain rule to the actual sequence of arithmetic operations used to calculate a result.

The Python and JavaScript programs implement a small forward-mode automatic differentiation system using dual numbers.

A dual number can be represented conceptually as:

`a + bε`

where:

`ε² = 0`

The coefficient `a` stores the ordinary value and `b` stores derivative information.

For example, if `x` has derivative seed `1`, then evaluating `x^3` through dual-number arithmetic produces both:

`x^3`

and:

`3x²`

without using finite differences.

Automatic differentiation is central to modern machine-learning frameworks because neural networks are naturally represented as computational graphs.

---

## Forward mode versus reverse mode

Forward-mode automatic differentiation propagates derivative information from inputs toward outputs.

It is particularly attractive when there are relatively few independent inputs and relatively many outputs.

Reverse-mode automatic differentiation propagates derivative information backward from outputs toward inputs.

It is particularly useful when a computation has many parameters but a small number of scalar outputs, such as a neural-network training loss.

Backpropagation is a form of reverse-mode automatic differentiation.

The small dual-number implementations in this project demonstrate the forward-mode idea rather than implementing a complete reverse-mode computational graph engine.

---

## Python implementation

The Python program is organized as a broad educational laboratory.

### Functions

The program defines ordinary Python functions for:

- Polynomials
- Linear models
- Sigmoid
- Loss functions
- Numerical derivatives
- Gradients
- Hessians
- Optimization

Python functions are first-class objects, which makes it convenient to pass a mathematical function into numerical differentiation routines.

For example, a derivative function can receive a callable representing `f(x)` and evaluate it at `x+h` and `x-h`.

### Numerical differentiation

The Python implementation contains forward, backward, and central differences.

This makes it possible to compare the approximations and observe how step size influences numerical accuracy.

### Multivariable calculus

The Python program calculates partial derivatives and gradients using coordinate perturbations.

For a point:

`[x_1, x_2, ..., x_d]`

one coordinate is changed at a time while the other coordinates remain fixed.

### Hessian

The Python implementation calculates second-order partial derivatives numerically using nested finite differences.

This is educational rather than a preferred production strategy for large models.

### Automatic differentiation

The `DualNumber` class demonstrates forward-mode automatic differentiation.

Arithmetic operations implement the corresponding derivative rules. The implementation includes addition, subtraction, multiplication, division, powers, sine, exponential, and logarithm.

### Machine-learning examples

The Python implementation also includes:

- Mean squared error
- Binary cross-entropy
- Logistic regression
- Gradient descent
- ReLU
- Sigmoid
- Softmax
- Multiclass cross-entropy
- A small neural-network forward pass

The final test section checks several numerical relationships.

---

## JavaScript implementation

The JavaScript program emphasizes how calculus can be expressed through first-class functions, classes, arrays, and executable application logic.

### Functional programming

JavaScript allows mathematical functions to be assigned to variables and passed as arguments.

This makes numerical derivative functions reusable.

The `compose` function demonstrates function composition directly.

### Arrays as vectors

JavaScript arrays provide a convenient representation for vectors.

Operations such as dot products, normalization, gradients, and matrix-like structures can therefore be implemented without external packages.

### Object-oriented models

The `LogisticRegression`, `DualNumber`, and `TinyNeuralNetwork` classes demonstrate how mathematical objects can be represented as reusable application components.

### Browser relevance

JavaScript is particularly useful when calculus is connected to an interactive interface.

A browser application can connect numerical calculations to:

- Input controls
- Sliders
- Charts
- Model visualizations
- Interactive demonstrations
- Educational interfaces

The provided DOM example checks whether a browser environment is available so that the same file remains executable in Node.js.

---

## C++ case study

The C++ program models a small binary-classification system using logistic regression.

The scenario uses two numerical features to classify observations into two classes.

The implementation intentionally uses a small synthetic dataset so that the mathematical calculations remain inspectable.

### Problem definition

Given feature vectors:

`x = [x_1, x_2]`

the model calculates:

`z = w_1x_1 + w_2x_2 + b`

and:

`p = sigmoid(z)`

The probability is converted into a binary class using a threshold.

The training objective is binary cross-entropy.

### Major components

The case study contains:

- `Dataset`
- `LogisticRegression`
- `TrainingConfig`
- `TrainingResult`
- `Evaluation`
- Numerical derivative functions
- Gradient functions
- Hessian calculation
- Training engine
- Dataset validation
- Evaluation logic
- Numerical-stability handling
- Tests

### Dataset validation

The `Dataset` structure checks:

- The dataset is not empty.
- The number of feature rows matches the number of targets.
- Feature vectors are not empty.
- All feature vectors have equal dimensions.
- Targets are binary.

These checks prevent invalid input from silently entering the optimization process.

### Model validation

The model checks feature dimensions before calculating logits.

It also validates classification thresholds and learning rates.

This reflects an important production principle: mathematical correctness depends partly on valid input contracts.

---

## C++ training algorithm

The training loop follows the sequence:

1. Calculate predictions.
2. Calculate binary cross-entropy.
3. Calculate the analytical gradient.
4. Update the weights.
5. Update the bias.
6. Repeat for the requested number of epochs.

The update is:

`w := w - alpha * gradient_w`

and:

`b := b - alpha * gradient_b`

The program reports the loss and parameters periodically.

After training, it evaluates predictions against the training targets.

---

## Gradient verification in C++

The C++ implementation includes an independent finite-difference gradient check.

The analytical gradient is compared with a numerical approximation:

`dL/dtheta_i ≈ [L(theta_i+h) - L(theta_i-h)]/(2h)`

This is an important debugging technique.

When implementing an optimizer, a gradient implementation can be wrong even when the surrounding training code is syntactically correct.

A numerical gradient check provides an independent calculation that can reveal sign errors, missing factors, incorrect averaging, and indexing mistakes.

Finite-difference checks are generally debugging tools rather than the main training mechanism because they require repeated objective evaluations.

---

## Computational complexity

Suppose:

- `n` = number of training samples
- `d` = number of features

For logistic regression, calculating the gradient requires approximately:

`O(n*d)`

operations per epoch.

The model parameter storage is:

`O(d)`

and the dataset requires approximately:

`O(n*d)`

storage.

A numerical gradient for `d` parameters requires multiple additional loss evaluations. Consequently, numerical gradients become inefficient as the number of parameters grows.

A full Hessian contains:

`d²`

entries.

This quadratic growth makes explicit Hessian storage and computation expensive for high-dimensional machine-learning models.

---

## Numerical stability

Floating-point arithmetic is finite and approximate.

Calculus formulas that are valid over real numbers can still cause computational problems when implemented directly.

Important examples include:

- `log(0)`
- Division by zero
- Overflow in `exp(x)`
- Underflow for very negative exponentials
- Subtraction of nearly equal floating-point values
- Extremely small finite-difference steps
- Extremely large model parameters

### Sigmoid stability

The naive sigmoid expression contains:

`exp(-x)`

For a very negative `x`, `-x` is very large and the exponential can overflow.

The implementations use a piecewise formulation that avoids unnecessary overflow.

### Cross-entropy stability

Probabilities are clipped before direct logarithms.

### Softmax stability

The largest logit is subtracted before exponentiation.

These techniques do not change the mathematical result in their intended domains but improve numerical behavior.

---

## Edge cases

Important calculus and machine-learning edge cases include:

### Undefined function value

A function may be undefined at a point even though a limit exists.

### Nonexistent limit

Different one-sided limits can prevent a two-sided limit from existing.

### Continuous but non-differentiable function

ReLU is continuous at zero but not classically differentiable there.

### Zero direction

A directional derivative requires a direction vector. A zero vector cannot be normalized.

### Empty datasets

Loss and gradient functions require at least one observation.

### Dimension mismatch

Vectors and model parameters must have compatible dimensions.

### Invalid binary targets

Binary cross-entropy expects targets of zero or one.

### Invalid probabilities

Probabilities used directly in logarithms must be within an appropriate numerical range.

### Learning rate

A non-positive learning rate is invalid for the gradient-descent implementations.

### Floating-point precision

Very small numerical differentiation steps can increase round-off error.

---

## Common mistakes

### Confusing derivative with function value

`f'(x)` is not generally equal to `f(x)`.

### Forgetting the chain rule

For a composition such as:

`sin(x²)`

the derivative is not simply:

`cos(x)`

It is:

`cos(x²) * 2x`

### Treating a partial derivative as a total derivative

A partial derivative changes one variable while holding the others fixed.

### Forgetting the factor from an average

If a loss is averaged over `n` observations, its gradient usually contains a factor of `1/n`.

### Reversing the gradient-descent direction

Gradient descent uses:

`theta_new = theta - alpha * gradient`

Using addition moves in the local direction of increasing objective value.

### Using an unstable softmax

Directly exponentiating very large logits can overflow.

### Taking logarithms of zero

Directly calculating `log(0)` produces an invalid numerical operation.

### Assuming a zero gradient always means a minimum

Stationary points can have different types.

### Treating finite differences as exact derivatives

Finite differences are approximations and depend on floating-point behavior and step size.

### Using a full Hessian without considering scale

A Hessian grows quadratically with parameter count and can become impractical.

---

## Important distinctions

### Derivative versus partial derivative

A derivative generally refers to a one-variable function.

A partial derivative changes one variable of a multivariable function while treating other variables as fixed.

### Gradient versus Jacobian

A gradient usually describes the first derivatives of a scalar-valued function.

A Jacobian describes first derivatives of a vector-valued function.

### Jacobian versus Hessian

A Jacobian contains first-order derivatives.

A Hessian contains second-order derivatives of a scalar-valued function.

### Numerical differentiation versus automatic differentiation

Numerical differentiation estimates derivatives from function evaluations.

Automatic differentiation propagates derivative information through computational operations.

### Automatic differentiation versus symbolic differentiation

Automatic differentiation operates on the computation itself.

Symbolic differentiation manipulates mathematical expressions.

### Gradient descent versus Newton-style methods

Gradient descent uses first-order information.

Newton-style methods use second-order curvature information.

Second-order methods can converge rapidly in suitable conditions but can require substantially more computation and memory.

---

## Python, JavaScript, and C++ comparison

| Aspect | Python | JavaScript | C++ |
|---|---|---|---|
| Educational experimentation | Strong | Strong | Strong |
| Concise mathematical code | Strong | Strong | Moderate |
| Functional programming | Strong | Strong | Strong |
| Browser integration | Limited | Strong | Limited |
| Explicit memory control | Limited | Limited | Strong |
| Numerical performance | Good for pure Python demonstrations | Good for moderate numerical demonstrations | Strong |
| Systems-oriented modeling | Moderate | Moderate | Strong |
| Standard-library-only implementation | Practical | Practical | Practical |
| Interactive web applications | Indirect | Direct | Usually indirect |
| Low-level optimization control | Limited | Limited | Strong |

These differences are implementation characteristics rather than statements about the mathematical validity of any language.

The mathematical concepts remain the same. The language changes how those concepts are represented, executed, tested, and integrated into applications.

---

## Performance considerations

For a scalar numerical derivative, a central finite difference requires two function evaluations per derivative estimate.

For a function with `d` parameters, a straightforward finite-difference gradient can require approximately `2d` function evaluations.

An analytical gradient can calculate all parameter derivatives in one structured pass when the model permits it.

This difference becomes critical for machine-learning systems with large parameter counts.

The C++ case study explicitly separates:

- Mathematical objective calculation
- Analytical gradient calculation
- Parameter update
- Evaluation

This separation makes it easier to profile and optimize individual stages.

---

## Security considerations

Calculus itself is mathematical, but numerical implementations can appear inside systems that process external inputs.

Relevant engineering concerns include:

- Validate dimensions before arithmetic.
- Validate numeric ranges where required.
- Reject invalid labels.
- Avoid division by zero.
- Avoid undefined logarithms.
- Guard against exponential overflow.
- Avoid unbounded iteration counts in externally controlled training systems.
- Monitor numerical values for `NaN` and infinity in production systems.
- Treat model parameters and input data as untrusted when they originate from external sources.
- Avoid using unrestricted dynamic expression evaluation to represent mathematical formulas.

The supplied implementations use explicit functions instead of evaluating arbitrary strings as code.

---

## Implementation considerations

A production machine-learning system normally uses optimized numerical kernels rather than hand-written scalar loops.

Real systems may also require:

- Vectorized operations
- Parallel execution
- GPU acceleration
- Automatic differentiation frameworks
- Mixed precision
- Batch processing
- Distributed training
- Optimizers beyond basic gradient descent
- Regularization
- Model checkpointing
- Monitoring
- Reproducibility controls

The educational implementations deliberately avoid external numerical dependencies so that the mathematical mechanisms remain visible.

---

## Gradient descent and convexity

Logistic regression with standard binary cross-entropy and a linear logit has a convex objective under common formulations without non-convex additions.

This is different from a general deep neural network, whose optimization landscape is generally non-convex.

For a convex objective, local minima have stronger global implications than they do for an arbitrary non-convex objective.

The basic gradient-descent examples in this project use simple functions so that the relationship between derivative, direction, learning rate, and convergence can be observed directly.

---

## Regularization connection

A machine-learning objective can contain both a data-fitting term and a regularization term.

For example, L2 regularization can add:

`lambda * ||w||²`

to an objective.

Its gradient with respect to `w` is:

`2lambda*w`

This demonstrates another important calculus principle: if an objective is the sum of several terms, its gradient is the sum of their gradients.

Regularization therefore changes the gradient used during optimization.

---

## Backpropagation connection

A neural network can be viewed as a computational graph.

For example:

`z1 = W1x + b1`

`h1 = activation(z1)`

`z2 = W2h1 + b2`

`y_hat = output(z2)`

`L = loss(y_hat,y)`

Backpropagation calculates:

`∂L/∂W2`

`∂L/∂b2`

`∂L/∂W1`

`∂L/∂b1`

by repeatedly applying the chain rule.

The further a parameter is from the loss, the more intermediate derivative factors participate in its gradient.

This explains why activation derivatives, numerical stability, and computational-graph structure matter in deep learning.

---

## Practical applications

The calculus concepts implemented in this project appear in many machine-learning tasks.

### Linear regression

Gradient-based optimization can minimize mean squared error.

### Logistic regression

Gradients optimize binary classification parameters.

### Neural networks

Backpropagation uses the chain rule to calculate parameter gradients.

### Deep learning

Large computational graphs require efficient derivative propagation.

### Hyperparameter optimization

Objective functions can be analyzed with respect to parameters and hyperparameters, although not all practical hyperparameters are differentiable.

### Reinforcement learning

Many policy and value optimization algorithms rely on gradients.

### Computer vision

Neural networks optimize losses over image representations using derivatives.

### Natural language processing

Large neural architectures rely on gradient-based optimization.

### Scientific machine learning

Differentiable physical or numerical models can incorporate calculus directly into optimization.

---

## Testing strategy

The Python, JavaScript, and C++ implementations include tests for important mathematical identities.

Examples include:

- `f(3) = 9` for a quadratic.
- The derivative of `x^3` at `x=2` is approximately `12`.
- The gradient of `x² + 3y²` at `(2,4)` is approximately `[4,24]`.
- `sigmoid(0) = 0.5`.
- Softmax probabilities sum to one.
- Mean squared error produces the expected value.
- Dual-number differentiation produces the expected derivative.
- Invalid dimensions are rejected.
- Invalid learning rates are rejected.

The C++ program also compares analytical gradients with finite-difference estimates.

---

## Why analytical gradients matter

Finite differences are valuable for verification but are not usually the preferred method for training large models.

Suppose a model has one million parameters.

A straightforward numerical gradient would require perturbing each parameter and recalculating the objective.

This is computationally expensive.

An analytical gradient derived using calculus can calculate the complete gradient in a structured pass.

Automatic differentiation provides a practical way to obtain derivatives of complicated programs without manually deriving every final expression.

This is one of the key connections between elementary calculus and modern machine-learning infrastructure.

---

## Conceptual progression

The topic can be understood as a sequence of increasingly general ideas:

`Function`

becomes

`Limit`

which provides the foundation for

`Derivative`

which generalizes to

`Partial derivatives`

which form

`Gradient`

and extend to

`Jacobian` and `Hessian`.

The

`Chain rule`

allows derivatives to pass through compositions.

The resulting gradients can be used by

`Gradient descent`

to optimize

`Loss functions`

for machine-learning models.

For neural networks, repeated chain-rule applications become

`Backpropagation`.

This progression connects introductory calculus directly to practical machine-learning optimization.

---

## Mathematical notation used in the implementations

| Symbol | Meaning |
|---|---|
| `x` | Scalar input |
| `y` | Target value |
| `y_hat` | Model prediction |
| `w` | Model weight vector |
| `b` | Bias |
| `theta` | General parameter vector |
| `L` | Loss or objective |
| `f(x)` | Function evaluated at `x` |
| `f'(x)` | First derivative |
| `f''(x)` | Second derivative |
| `∂f/∂x` | Partial derivative |
| `∇f` | Gradient |
| `J` | Jacobian |
| `H` | Hessian |
| `alpha` | Learning rate |
| `h` | Numerical finite-difference step |
| `p` | Predicted probability |

---

## Relationship between calculus and optimization

Optimization asks for parameter values that minimize or maximize an objective.

Calculus supplies local information about the objective.

For minimization:

`gradient = ∇L(theta)`

and the basic update is:

`theta_new = theta - alpha * gradient`

The gradient therefore provides a local direction of decrease.

The update is only a local approximation. Global behavior depends on the objective's geometry, learning rate, initialization, constraints, parameterization, and optimization algorithm.

---

## Limitations of the implementations

These programs are educational implementations rather than high-performance machine-learning frameworks.

The numerical derivative routines use finite differences and therefore inherit floating-point limitations.

The automatic-differentiation example implements only a small subset of mathematical operations.

The logistic-regression examples use small in-memory datasets.

The C++ program uses scalar loops rather than vectorized BLAS-style numerical kernels.

The neural-network examples focus on forward computation and derivative concepts rather than implementing a complete production backpropagation engine.

These limitations are intentional because the mathematical operations are kept visible.

---

## Best practices demonstrated

The implementations follow several useful numerical-programming principles:

- Validate inputs early.
- Keep mathematical functions small and explicit.
- Separate model, objective, gradient, training, and evaluation logic.
- Use central differences when numerical derivative estimation is appropriate.
- Use analytical gradients for repeated optimization.
- Verify analytical gradients numerically during development.
- Protect logarithms from zero.
- Protect exponentials from unnecessary overflow.
- Validate vector dimensions.
- Reject invalid learning rates.
- Treat non-differentiable points explicitly.
- Keep numerical tolerances explicit in tests.
- Avoid confusing numerical approximation with mathematical equality.
- Consider computational complexity before selecting a differentiation strategy.

---

## File structure

A typical project layout for these deliverables is:

`calculus_for_ml.py`

`calculus_for_ml.js`

`calculus_for_ml.cpp`

`README.md`

The Python file can be executed with a standard Python 3 installation.

The JavaScript file can be executed with a modern JavaScript runtime such as Node.js.

The C++ file requires a compiler supporting C++17 or later.

---

## Core formulas

### Linear model

`y_hat = wx + b`

### Derivative definition

`f'(x) = lim(h -> 0) [f(x+h)-f(x)]/h`

### Central finite difference

`f'(x) ≈ [f(x+h)-f(x-h)]/(2h)`

### Gradient

`∇f = [∂f/∂x_1, ..., ∂f/∂x_d]`

### Directional derivative

`D_u f = ∇f · u`

for a unit direction vector `u`.

### Gradient descent

`theta_new = theta_old - alpha∇L(theta_old)`

### Sigmoid

`sigmoid(z) = 1/(1+exp(-z))`

### Binary cross-entropy

`L = -[y log(p)+(1-y)log(1-p)]`

### Logistic regression

`p = sigmoid(w^Tx+b)`

### Softmax

`p_i = exp(z_i)/Σ_j exp(z_j)`

### First-order Taylor approximation

`f(x) ≈ f(a)+f'(a)(x-a)`

### Second-order Taylor approximation

`f(x) ≈ f(a)+f'(a)(x-a)+1/2 f''(a)(x-a)^2`

---

## Scope of the three implementations

The Python implementation provides the broadest mathematical coverage and is designed as a standalone study program.

The JavaScript implementation emphasizes functional programming, executable numerical experiments, reusable classes, browser integration patterns, and the relationship between mathematical functions and application code.

The C++ implementation emphasizes system-oriented structure, explicit data validation, numerical gradient verification, model training, evaluation, complexity analysis, and production-style error handling.

Together, the implementations demonstrate that the same calculus principles can be represented at different levels of abstraction while preserving the underlying mathematics.
