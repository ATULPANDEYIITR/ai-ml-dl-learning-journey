# Kernel Methods

This project implements kernel methods as a unified study of similarity functions, implicit feature spaces, polynomial kernels, and radial basis function (RBF) kernels.

The five executable source files use different computational perspectives:

| File | Technical perspective | Main implementation |
|---|---|---|
| Python | Mathematical model and experimentation | Kernel functions, Gram matrices, classification, kernel ridge regression |
| JavaScript | Event-driven similarity service | Validation, asynchronous evaluation, classification, regression |
| C++17 | Numerical kernel engine | Polymorphic kernels, Gram matrices, PSD validation, classification, regression |
| Java 17 | Enterprise domain model | Immutable records, sealed kernel hierarchy, services, validation, model workflows |
| PostgreSQL | Relational similarity platform | Kernel catalog, feature storage, SQL similarity functions, evaluations, reporting |

## Mathematical foundation

A kernel function measures similarity between two observations without requiring the application to explicitly construct the feature representation associated with that similarity.

For vectors \(x\) and \(y\), a kernel is commonly written as

\[
K(x,y)=\langle \phi(x),\phi(y)\rangle
\]

where \(\phi\) maps the original input into a potentially higher-dimensional feature space.

The important computational idea is that the algorithm can use \(K(x,y)\) directly instead of constructing \(\phi(x)\).

This is the kernel trick.

The distinction matters when the transformed feature space is very large or conceptually infinite. The algorithm needs inner products between transformed observations, not necessarily the transformed observations themselves.

A finite collection of observations produces a Gram matrix:

\[
G_{ij}=K(x_i,x_j)
\]

For a valid positive-semidefinite kernel, the Gram matrix should be positive semidefinite. This means that for every coefficient vector \(c\),

\[
c^T Gc\geq0
\]

The implementations use numerical Cholesky-style checks to test this property.

## Similarity spaces

A similarity space represents observations through pairwise relationships rather than requiring the application to reason only about their original coordinates.

For example, the samples

- `(-1, -1)`
- `(-1, 1)`
- `(1, -1)`
- `(1, 1)`

have ordinary Euclidean coordinates, but a kernel supplies another representation of their relationships.

A similarity matrix can be visualized conceptually as:

| | \(x_1\) | \(x_2\) | \(x_3\) |
|---|---:|---:|---:|
| \(x_1\) | \(K(x_1,x_1)\) | \(K(x_1,x_2)\) | \(K(x_1,x_3)\) |
| \(x_2\) | \(K(x_2,x_1)\) | \(K(x_2,x_2)\) | \(K(x_2,x_3)\) |
| \(x_3\) | \(K(x_3,x_1)\) | \(K(x_3,x_2)\) | \(K(x_3,x_3)\) |

This matrix is the computational interface through which many kernel algorithms operate.

## Linear kernel

The linear kernel is the ordinary inner product:

\[
K(x,y)=x^Ty
\]

It does not create a nonlinear representation. It is useful as a baseline because it exposes the difference between ordinary linear similarity and nonlinear kernels.

The Python, C++, JavaScript, Java, and PostgreSQL implementations all support linear similarity where appropriate, while the principal demonstrations emphasize nonlinear kernels.

## Polynomial kernel

The polynomial kernel is implemented as

\[
K(x,y)=
(\gamma x^Ty+c)^d
\]

where:

- \(\gamma\) controls the scale of the inner product,
- \(c\) is the additive coefficient,
- \(d\) is the polynomial degree.

A degree greater than one permits interactions between input dimensions.

For example, a quadratic transformation can represent terms related to

\[
x_1^2,\quad x_2^2,\quad x_1x_2
\]

without explicitly constructing those additional features.

The implementation validates the degree, gamma, coefficient, vector dimensions, and finite numeric values before evaluating the kernel.

A large degree can produce very large numerical values. Kernel parameters therefore affect not only model behavior but also numerical stability.

## RBF kernel

The RBF kernel is implemented as

\[
K(x,y)=
e^{-\gamma\lVert x-y\rVert^2}
\]

The similarity is one when two vectors are identical and decreases as their squared Euclidean distance increases.

The parameter \(\gamma\) controls how quickly similarity decays.

A large gamma produces a narrow similarity region. A small gamma produces a broader similarity region.

The RBF kernel is especially useful for the nonlinear classification example because nearby observations can have strong similarity even when a straight-line decision boundary cannot separate the original data effectively.

## Kernel trick

The kernel trick changes the computation rather than merely adding a feature-engineering step.

An explicit nonlinear transformation might require constructing

\[
\phi(x)
\]

and then calculating

\[
\phi(x)^T\phi(y)
\]

A kernel method instead calculates

\[
K(x,y)
\]

directly.

For a polynomial kernel, the transformed feature space can contain many interaction terms. For an RBF kernel, the corresponding feature space can be extremely high-dimensional.

The source implementations therefore store the original vectors and evaluate pairwise kernel values when needed.

This avoids explicitly materializing the transformed representation.

## Python implementation

The Python script treats kernels as callable objects and builds reusable numerical operations around them.

`linear_kernel`, `polynomial_kernel`, and `rbf_kernel` provide separate mathematical policies. `gram_matrix` evaluates pairwise similarities and exploits kernel symmetry by calculating only one triangle of the matrix.

`cholesky_psd_check` tests whether a Gram matrix is numerically positive semidefinite. The tolerance prevents insignificant floating-point errors from being interpreted as mathematical violations.

The `KernelKNN` class demonstrates similarity-based classification. It aggregates kernel similarities by class rather than relying on ordinary Euclidean distance.

The `KernelRidge` class demonstrates a more direct kernel-method formulation. Training solves

\[
(K+\lambda I)\alpha=y
\]

and prediction evaluates

\[
f(x)=\sum_i\alpha_iK(x_i,x)
\]

The implementation includes Gaussian elimination with partial pivoting because the example is intentionally self-contained and uses small matrices.

For large numerical workloads, this direct implementation is not a substitute for specialized numerical linear-algebra libraries.

The XOR-like dataset provides a nonlinear case where the RBF kernel can represent local similarity relationships that are not captured by a simple linear similarity rule.

## JavaScript implementation

The JavaScript file models kernel evaluation as an event-driven service.

Validation occurs at the boundaries of vector, kernel, and model operations. `Map` is used for accumulating class-level similarity scores, which makes label aggregation explicit.

The `KernelSimilarityClassifier` and `KernelRidgeModel` classes provide two different uses of the same kernel abstraction.

The `evaluateAsync` function schedules each major evaluation with `setImmediate`. This makes the execution model explicitly event-driven rather than presenting the work as a single uninterrupted procedural calculation.

No external npm packages are required.

The implementation uses `console.table` for Gram-matrix inspection and returns structured classification and regression results instead of mixing model logic with output formatting.

## C++17 implementation

The C++ program models kernels through a polymorphic `Kernel` interface.

`LinearKernel`, `PolynomialKernel`, and `RBFKernel` implement distinct similarity policies. This design permits the Gram-matrix, classifier, and regression components to operate against the abstract kernel interface.

The `gramMatrix` function calculates only the upper triangular matrix and mirrors it because the supported kernels are symmetric.

The `SimilarityClassifier` aggregates similarity scores by class using `std::map`.

The `KernelRidge` class uses the Gram matrix with diagonal regularization. Regularization changes

\[
K
\]

to

\[
K+\lambda I
\]

which improves numerical stability and controls the fitted model.

The program uses partial-pivot Gaussian elimination. This is suitable for the small case study but has cubic time complexity in the number of training samples.

For a training set with \(n\) observations, constructing a full Gram matrix requires \(O(n^2)\) kernel evaluations and \(O(n^2)\) storage. Solving a dense kernel-ridge system is approximately \(O(n^3)\).

## Java 17 implementation

The Java implementation gives the kernel domain explicit enterprise-oriented types.

`FeatureVector` is an immutable record. The `Kernel` interface is sealed, restricting implementations to the supported kernel classes.

`TrainingSet` represents the invariant that every feature vector has the same dimensionality and that every training sample has a corresponding label.

`KernelSimilarityService` owns classification behavior while `GramMatrixService` owns matrix construction and positive-semidefinite validation.

`KernelRidgeService` explicitly models the fitted and unfitted states. Attempting prediction before fitting produces a failure instead of silently producing an invalid result.

The separation between domain values and services makes policy boundaries clearer. Kernel definitions govern similarity. Classification and regression services consume those similarity rules.

## PostgreSQL implementation

The SQL script treats kernel methods as a relational computational workflow.

The `kernel_definition` table stores kernel type and parameters. `similarity_dataset` defines the expected dimensionality, while `sample` stores the feature arrays and optional classification or regression targets.

A trigger validates feature dimensionality against the dataset definition. This is important because the database should not accept a vector that the mathematical operation cannot compare correctly with other vectors in the dataset.

The SQL functions implement:

- dot products,
- squared Euclidean distance,
- linear kernel evaluation,
- polynomial kernel evaluation,
- RBF kernel evaluation.

`kernel_similarity` selects the appropriate calculation from the stored kernel definition.

`kernel_evaluation` stores pairwise similarity values. This is useful when repeated analysis requires the same kernel evaluations because calculating all pairwise similarities repeatedly can be expensive.

The `kernel_similarity_report` view exposes the stored similarity relationships in a human-readable relational form.

The `populate_kernel_evaluations` function uses `ON CONFLICT` to make repeated population idempotent for an existing kernel/sample pair.

## Classification workflow

The nonlinear classification workflow follows this structure:

```text
Training vectors
      |
      v
Kernel definition
      |
      v
Pairwise similarity
      |
      v
Gram matrix / similarity scores
      |
      v
Class-level aggregation
      |
      v
Predicted class
```

The RBF classifier does not first convert every vector into an explicitly stored nonlinear feature vector.

Instead, it asks how strongly each training vector resembles the query under the selected kernel.

This distinction is central to kernel methods.

## Kernel ridge regression

Kernel ridge regression uses the same similarity concept for a continuous target.

The training equation is

\[
(K+\lambda I)\alpha=y
\]

where:

- \(K\) is the Gram matrix,
- \(\lambda\) is the regularization parameter,
- \(I\) is the identity matrix,
- \(y\) contains the training targets,
- \(\alpha\) contains the learned coefficients.

Prediction uses the kernel between the new observation and each training observation:

\[
f(x)=K(x,X)^T\alpha
\]

The implementations use a small quadratic dataset where targets follow the pattern

\[
y=x^2
\]

The purpose is to demonstrate how a nonlinear similarity function can support regression without constructing an explicit nonlinear polynomial feature table.

## Polynomial and RBF distinctions

The polynomial and RBF kernels should not be treated as interchangeable.

| Property | Polynomial | RBF |
|---|---|---|
| Main control | Degree and gamma | Gamma |
| Similarity basis | Inner product | Distance |
| Nonlinearity | Polynomial interactions | Local radial similarity |
| Feature interpretation | Related to finite polynomial interactions | Usually not represented explicitly |
| Sensitivity | Degree can strongly change scale | Gamma strongly controls locality |
| Typical behavior | Global interaction structure | Local similarity structure |

The polynomial kernel can model interactions whose structure is related to polynomial expansions.

The RBF kernel emphasizes proximity in the original metric space and can create highly nonlinear decision behavior.

## Kernel parameter effects

Kernel parameters should be treated as model-design choices rather than arbitrary constants.

For the RBF kernel:

\[
K(x,y)=e^{-\gamma\lVert x-y\rVert^2}
\]

If \(\gamma\) is too large, similarity can become concentrated around individual observations. The resulting model can become overly sensitive to local training examples.

If \(\gamma\) is too small, many observations may appear similarly related. The model can lose useful local structure.

For a polynomial kernel, increasing the degree increases the complexity of the implicit interaction structure and can also cause numerical magnification.

Regularization in kernel ridge regression provides another control mechanism by modifying the Gram matrix before solving for the coefficients.

## Positive-semidefinite requirement

Kernel validity is tied to the Gram matrix.

A symmetric matrix \(K\) is positive semidefinite when

\[
c^TKc\geq0
\]

for every coefficient vector \(c\).

The implementations use a Cholesky-style decomposition as a numerical diagnostic.

The check is not a symbolic proof that a kernel is valid for every possible input. It evaluates the supplied finite Gram matrix and therefore detects numerical or implementation problems for that sample set.

Floating-point arithmetic can produce tiny negative values where exact mathematics would produce zero, so a tolerance is required.

## Validation and failure handling

The implementations reject:

- empty feature vectors,
- mismatched feature dimensions,
- non-finite feature values,
- invalid polynomial degrees,
- non-positive kernel parameters,
- invalid regularization values,
- empty training datasets,
- mismatched sample and target counts,
- singular or numerically unstable linear systems,
- prediction requests before model fitting.

These checks are part of the computational model rather than cosmetic input handling.

A kernel cannot meaningfully compare vectors with different dimensions, and a kernel-ridge solver cannot safely proceed when its system matrix is numerically singular.

## Database integrity

The PostgreSQL implementation moves important validation into the database layer.

The dataset declares an expected feature dimension. The `sample_dimension_validation` trigger verifies that each inserted feature array has exactly that dimension.

This prevents inconsistent vectors from entering the similarity system.

Kernel definitions use PostgreSQL `CHECK` constraints to reject invalid polynomial degrees, gamma values, coefficients, and regularization parameters.

The schema therefore combines application-level mathematical operations with database-level integrity rules.

## Performance considerations

For \(n\) training samples, a dense Gram matrix requires

\[
O(n^2)
\]

kernel evaluations and

\[
O(n^2)
\]

memory.

Kernel ridge regression on a dense Gram matrix requires approximately

\[
O(n^3)
\]

time for direct linear-system solution.

This makes kernel methods increasingly expensive as the number of training observations grows.

The implementations intentionally use small datasets so the mathematical operations remain visible.

For larger systems, the main scaling problem is often the Gram matrix itself rather than the individual kernel formula.

The RBF kernel also requires distance computation, which is \(O(d)\) for \(d\)-dimensional observations. A complete dense Gram matrix therefore has approximately \(O(n^2d)\) arithmetic work before the later model computation.

## Security and numerical considerations

Kernel implementations should reject non-finite values such as NaN and infinity because they can propagate through matrix calculations and make downstream results unreliable.

Parameters should be validated before use. In a service environment, kernel parameters should not be accepted blindly from untrusted clients.

The PostgreSQL implementation uses typed parameters and database functions rather than constructing SQL statements through string concatenation. Kernel configuration is represented as relational data, which avoids treating mathematical parameters as SQL syntax.

Numerical stability is also a security-adjacent reliability concern. Extremely large polynomial degrees or poorly scaled input values can cause overflow or unstable calculations.

Regularization helps in kernel ridge regression, but it does not eliminate all numerical problems.

## Relationship between the five implementations

The implementations intentionally solve the same mathematical domain from different architectural perspectives.

Python emphasizes experimentation and direct mathematical expression.

JavaScript emphasizes asynchronous execution and service-style evaluation.

C++ emphasizes explicit numerical and polymorphic design.

Java emphasizes immutable domain modeling and explicit enterprise service boundaries.

PostgreSQL emphasizes persistent kernel configuration, relational integrity, SQL computation, and repeatable similarity reporting.

The underlying relationship remains:

\[
\text{input vectors} \rightarrow \text{kernel} \rightarrow \text{similarity} \rightarrow \text{Gram matrix or scores} \rightarrow \text{learning task}
\]

The kernel is the central abstraction. The surrounding architecture determines how that abstraction is validated, executed, stored, and consumed.
````
