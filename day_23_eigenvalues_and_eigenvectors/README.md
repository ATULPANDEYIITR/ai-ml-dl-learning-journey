# Eigenvalues & Eigenvectors

## Topic

**Eigenvalues & Eigenvectors: eigen decomposition, characteristic equations, and applications in machine learning**

This repository studies eigenvalues and eigenvectors from the basic definition through computational methods and applications in machine learning, data analysis, Markov chains, and graph analysis.

The implementations use three programming languages:

- Python for a broad mathematical study implementation.
- JavaScript for executable numerical demonstrations using standard JavaScript.
- C++ for an industry-style spectral analytics case study involving sensor observations, PCA, graph structure, and transition behavior.

The central equation is

`Av = λv`

where:

- `A` is a square matrix.
- `v` is a non-zero eigenvector.
- `λ` is the corresponding eigenvalue.
- `Av` is the transformed vector.
- `λv` is the same vector scaled by `λ`.

The defining property is that an eigenvector is a direction that a linear transformation does not rotate away from itself. The transformation may stretch it, shrink it, or reverse its direction.

---

## Fundamental linear algebra

### Vectors

A vector is an ordered collection of numbers.

For example:

`v = [2, 3]`

can be interpreted as a point, a direction, a feature representation, or a state depending on the application.

A vector can be multiplied by a scalar:

`3v = [6, 9]`

The direction is unchanged for a positive scalar, while a negative scalar reverses the direction.

The Euclidean norm of a vector is:

`||v|| = sqrt(v₁² + v₂² + ... + vₙ²)`

Normalization converts a non-zero vector into a unit vector:

`v_normalized = v / ||v||`

The eigenvector implementations normalize vectors so that their numerical representation is easier to compare.

### Matrices

A matrix is a rectangular arrangement of numbers.

A square matrix has the same number of rows and columns. Eigenvalues are defined for square matrices because the equation `Av = λv` requires `A` to map a vector space into itself.

For example:

`A = [[4, 1], [2, 3]]`

represents a transformation from two-dimensional vectors to two-dimensional vectors.

### Matrix-vector multiplication

For

`A = [[a, b], [c, d]]`

and

`v = [x, y]`

the result is:

`Av = [ax + by, cx + dy]`

The Python, JavaScript, and C++ implementations all explicitly implement matrix-vector multiplication so the relationship between the mathematical notation and executable computation remains visible.

---

## Linear transformations

A matrix can represent a linear transformation.

A transformation can:

- stretch space,
- compress space,
- rotate space,
- reflect space,
- shear space,
- combine several of these operations.

Most vectors change both their magnitude and direction after multiplication by a matrix.

Eigenvectors are special because the transformation preserves their direction.

The defining relationship is:

`Av = λv`

The scalar `λ` is the eigenvalue.

If:

`λ > 1`

the vector is stretched.

If:

`0 < λ < 1`

the vector is compressed.

If:

`λ = 1`

the vector is unchanged.

If:

`λ = 0`

the vector is mapped to the zero vector.

If:

`λ < 0`

the vector is reversed as well as scaled.

---

## Characteristic equation

An eigenvalue must satisfy:

`Av = λv`

Rearranging:

`Av - λv = 0`

Factoring the vector:

`(A - λI)v = 0`

where `I` is the identity matrix.

A non-zero solution `v` exists only when:

`det(A - λI) = 0`

This equation is the **characteristic equation**.

The determinant condition is important because a singular matrix has a non-trivial null space. The eigenvalue is therefore obtained by finding the values of `λ` that make `A - λI` singular.

The polynomial:

`det(A - λI)`

is called the characteristic polynomial.

---

## The 2 × 2 characteristic polynomial

For:

`A = [[a, b], [c, d]]`

the characteristic polynomial is:

`λ² - (a + d)λ + (ad - bc)`

Two important matrix quantities appear:

### Trace

The trace is the sum of diagonal elements:

`tr(A) = a + d`

The trace equals the sum of eigenvalues, counting algebraic multiplicity.

### Determinant

For a 2 × 2 matrix:

`det(A) = ad - bc`

The determinant equals the product of the eigenvalues, counting algebraic multiplicity.

Therefore:

`λ₁ + λ₂ = tr(A)`

and:

`λ₁λ₂ = det(A)`

These relationships provide useful validation checks.

---

## Eigenvectors

After finding an eigenvalue `λ`, substitute it into:

`(A - λI)v = 0`

The resulting homogeneous system produces the corresponding eigenvector or eigenvectors.

Eigenvectors are not unique in magnitude.

If `v` is an eigenvector, then for any non-zero scalar `c`:

`cv`

is also an eigenvector for the same eigenvalue.

For example, if:

`v = [1, 2]`

is an eigenvector, then:

`[2, 4]`

and:

`[-3, -6]`

represent the same eigenvector direction.

The implementations generally normalize eigenvectors to unit length.

---

## Eigenpair validation

A numerical eigenvalue and eigenvector should not be accepted merely because they look plausible.

An eigenpair can be checked through the residual:

`r = Av - λv`

A correct eigenpair should have a small residual norm:

`||Av - λv|| ≈ 0`

The Python, JavaScript, and C++ implementations calculate this residual.

This is particularly important for numerical algorithms because floating-point calculations rarely produce exact symbolic zeros.

A residual of approximately `10^-12` can be an excellent result for a particular numerical computation, while a residual of `0.1` would indicate a serious mismatch.

The appropriate tolerance depends on:

- matrix magnitude,
- matrix condition,
- numerical algorithm,
- floating-point precision,
- problem size.

---

## Algebraic multiplicity

An eigenvalue can appear more than once as a root of the characteristic polynomial.

The number of times an eigenvalue appears as a root is its **algebraic multiplicity**.

For example:

`(λ - 2)² = 0`

has eigenvalue `2` with algebraic multiplicity two.

Repeated eigenvalues require additional care because algebraic multiplicity does not tell us how many independent eigenvectors exist.

---

## Geometric multiplicity

The **geometric multiplicity** of an eigenvalue is the dimension of its eigenspace.

The eigenspace associated with `λ` is:

`null(A - λI)`

The geometric multiplicity cannot exceed the algebraic multiplicity.

A matrix can have a repeated eigenvalue but fewer independent eigenvectors than the matrix dimension.

This distinction is central to diagonalization.

---

## Diagonalization

A matrix is diagonalizable if it can be represented as:

`A = PDP⁻¹`

where:

- `P` contains linearly independent eigenvectors as columns.
- `D` contains the corresponding eigenvalues on its diagonal.
- `P⁻¹` is the inverse of `P`.

For example:

`D = [[λ₁, 0], [0, λ₂]]`

The transformation represented by `A` becomes a simple scaling operation in the eigenvector coordinate system.

This is one of the main reasons eigen decomposition is useful.

---

## Why diagonalization matters

Consider:

`A = PDP⁻¹`

Then:

`A² = PDP⁻¹PDP⁻¹`

Since:

`P⁻¹P = I`

we obtain:

`A² = PD²P⁻¹`

Similarly:

`Aᵏ = PDᵏP⁻¹`

The diagonal matrix is especially easy to exponentiate:

`Dᵏ = diag(λ₁ᵏ, λ₂ᵏ, ..., λₙᵏ)`

This can make repeated application of a transformation much easier to analyze.

The Python and JavaScript implementations demonstrate matrix powers and the C++ implementation demonstrates the numerical components needed to obtain eigenpairs.

---

## When diagonalization fails

Not every square matrix is diagonalizable.

A matrix is diagonalizable when it has enough linearly independent eigenvectors to form a basis.

A classic example is:

`A = [[2, 1], [0, 2]]`

Its only eigenvalue is `2`, repeated twice.

Solving:

`(A - 2I)v = 0`

produces only one independent eigenvector direction.

Consequently, two independent eigenvectors cannot be placed into `P`, and the matrix is not diagonalizable.

The important distinction is:

- Repeated eigenvalue does not automatically mean non-diagonalizable.
- Distinct eigenvalues guarantee linearly independent eigenvectors.
- Repeated eigenvalues require examination of the eigenspaces.

---

## Symmetric matrices

A real matrix is symmetric when:

`A = Aᵀ`

For example:

`[[4, 1], [1, 4]]`

is symmetric.

Real symmetric matrices have especially useful spectral properties.

Their eigenvalues are real, and they possess an orthonormal basis of eigenvectors.

This is a consequence of the **spectral theorem**.

For a real symmetric matrix:

`A = QΛQᵀ`

where:

- `Q` contains orthonormal eigenvectors.
- `Λ` is diagonal.
- `QᵀQ = I`.
- `Q⁻¹ = Qᵀ`.

This structure is particularly important in machine learning because covariance matrices are symmetric.

---

## Positive semidefinite matrices

A real symmetric matrix `A` is positive semidefinite when:

`xᵀAx >= 0`

for every vector `x`.

Covariance matrices are positive semidefinite.

A positive semidefinite matrix has non-negative eigenvalues.

This is important for PCA because covariance eigenvalues represent variance along principal directions.

A negative variance would not be physically meaningful, so the mathematical structure of covariance matrices naturally leads to non-negative eigenvalues.

---

## Eigen decomposition

The term **eigen decomposition** usually refers to representing a diagonalizable matrix as:

`A = PΛP⁻¹`

For symmetric matrices, the preferred form is:

`A = QΛQᵀ`

where `Q` is orthogonal.

The columns of `P` or `Q` are eigenvectors.

The diagonal entries of `Λ` are eigenvalues.

This decomposition separates a transformation into:

1. Change into an eigenvector coordinate system.
2. Scale each eigenvector coordinate by its eigenvalue.
3. Transform back into the original coordinate system.

---

## Eigenvalue ordering

Eigenvalues are not intrinsically required to be listed in a particular order.

For machine learning applications, eigenpairs are often sorted by descending eigenvalue magnitude or descending eigenvalue when working with covariance matrices.

For PCA, the eigenvalues of a covariance matrix are normally ordered from largest to smallest.

The corresponding eigenvectors must be reordered at the same time.

It is incorrect to sort eigenvalues without applying the same permutation to their eigenvectors.

---

## Sign ambiguity

If:

`v`

is an eigenvector, then:

`-v`

is also an eigenvector for the same eigenvalue.

Therefore two correct PCA implementations may produce:

`[0.70, 0.71]`

and:

`[-0.70, -0.71]`

for the same principal direction.

The direction represented by the one-dimensional subspace is the same.

This is a common source of confusion when comparing numerical outputs.

---

## Complex eigenvalues

A real matrix can have complex eigenvalues.

The 90-degree rotation matrix:

`[[0, -1], [1, 0]]`

has eigenvalues:

`+i`

and:

`-i`

There are no non-zero real eigenvectors for this rotation because every non-zero real vector changes direction.

Complex eigenvalues are not an error. They indicate that the transformation's invariant directions require complex vector space.

The Python implementation uses Python's built-in complex-number support to demonstrate this case.

The JavaScript and C++ implementations deliberately use a more limited real-valued computational path for the main case studies.

---

## Python implementation

The Python file is designed as a comprehensive study program.

It contains implementations for:

- vector operations,
- matrix operations,
- matrix multiplication,
- matrix powers,
- determinants,
- traces,
- characteristic polynomials,
- 2 × 2 eigenvalues,
- eigenvectors,
- eigenpair validation,
- matrix inversion,
- diagonalization,
- eigen-based matrix powers,
- symmetric-matrix checks,
- covariance matrices,
- Rayleigh quotients,
- power iteration,
- deflation,
- multiple eigenpair estimation,
- PCA,
- projection,
- reconstruction,
- reconstruction error,
- Markov chains,
- graph Laplacians,
- complex eigenvalues,
- edge cases.

The script intentionally implements many operations rather than hiding the mathematics behind an external numerical package.

This makes the relationship between the formulas and the algorithms visible.

### Basic eigenpair example

The matrix used in several demonstrations is:

`A = [[4, 1], [2, 3]]`

Its characteristic polynomial is:

`λ² - 7λ + 10`

which factors as:

`(λ - 5)(λ - 2)`

Therefore the eigenvalues are:

`5` and `2`.

The corresponding eigenvectors can be obtained by solving:

`(A - 5I)v = 0`

and:

`(A - 2I)v = 0`.

The program calculates the vectors numerically and verifies them with residual norms.

---

## Python matrix inversion

The Python implementation includes a Gauss-Jordan inversion routine.

The method augments:

`A`

with:

`I`

to form:

`[A | I]`

and performs row operations until the left side becomes:

`I`

The right side then becomes:

`A⁻¹`

The implementation is intended for small educational matrices.

Large production numerical workloads should normally use specialized linear algebra algorithms rather than a simple educational recursive or Gauss-Jordan implementation.

---

## Python power iteration

Power iteration repeatedly applies a matrix to a vector:

`v_next = Av`

and normalizes the result.

The method tends to converge toward the eigenvector associated with the eigenvalue having the largest magnitude when suitable convergence conditions hold.

The estimated eigenvalue can then be calculated using the Rayleigh quotient:

`R(v) = (vᵀAv)/(vᵀv)`

For a true eigenvector:

`R(v) = λ`

The method is attractive because it does not require explicitly constructing a characteristic polynomial.

### Convergence considerations

Power iteration works particularly well when:

- the dominant eigenvalue is unique in magnitude,
- the starting vector has a non-zero component along the dominant eigenvector,
- the matrix is numerically well behaved.

Convergence can be slow when the two largest eigenvalues in magnitude are close.

For example, if:

`|λ₂ / λ₁|`

is close to one, many iterations may be required.

If the dominant eigenvalue is not unique in magnitude, basic power iteration may fail to converge to a single eigenvector.

---

## Rayleigh quotient

The Rayleigh quotient is:

`R(v) = (vᵀAv)/(vᵀv)`

for non-zero `v`.

For a symmetric matrix, the Rayleigh quotient has strong relationships with the smallest and largest eigenvalues.

If `v` is an eigenvector associated with `λ`, then:

`R(v) = λ`

This makes it useful for estimating eigenvalues from approximate eigenvectors.

The implementations use it to estimate the eigenvalue after power iteration normalizes the vector.

---

## Deflation

After finding one eigenpair:

`Av₁ = λ₁v₁`

a simplified symmetric-matrix deflation step is:

`A_new = A - λ₁v₁v₁ᵀ`

when `v₁` is normalized.

This removes the contribution of the discovered eigenpair and allows another power iteration to search for another direction.

Deflation is useful for understanding the concept of extracting multiple eigenpairs.

Production eigensolvers generally use more sophisticated methods because naive deflation can accumulate numerical errors.

---

# Principal component analysis

## PCA motivation

Principal Component Analysis, or PCA, is one of the most important applications of eigenvalues and eigenvectors in machine learning and data analysis.

Suppose a dataset contains several correlated features.

PCA searches for new orthogonal directions that capture the variation in the data.

The first principal component is the direction of maximum variance.

The second principal component captures the largest remaining variance subject to being orthogonal to the first.

The process continues for additional components.

---

## PCA preprocessing

PCA commonly begins by centering each feature.

For feature `j`, calculate its mean:

`μ_j = (1/n) Σ x_ij`

Then subtract the mean:

`x'_ij = x_ij - μ_j`

This moves the dataset so that each feature has mean zero.

The Python, JavaScript, and C++ implementations center the observations before calculating covariance.

---

## Covariance matrix

For centered data matrix `X`, the sample covariance matrix is:

`C = XᵀX / (n - 1)`

The covariance matrix is symmetric.

For two features:

`C = [[var(X₁), cov(X₁,X₂)], [cov(X₂,X₁), var(X₂)]]`

The diagonal contains feature variances.

The off-diagonal entries describe covariance between features.

---

## PCA and eigenvectors

The eigenvectors of the covariance matrix define the principal directions.

The eigenvalues tell us how much variance lies along those directions.

If the eigenvalues are:

`λ₁ >= λ₂ >= ... >= λ_d`

then the first principal component corresponds to the eigenvector associated with `λ₁`.

The proportion of variance explained by component `i` is:

`λ_i / Σ λ_j`

This is called the **explained variance ratio**.

---

## Dimensionality reduction

Suppose the original feature vector has dimension `d`.

PCA can retain only the first `k` components where:

`k < d`

The data can then be represented using:

`k`

coordinates instead of:

`d`

coordinates.

The compressed representation is:

`z = Q_kᵀ(x - μ)`

where:

- `μ` is the mean vector.
- `Q_k` contains the first `k` principal eigenvectors.
- `z` is the reduced representation.

The C++ case study performs one-component compression for two-dimensional observations.

---

## Reconstruction

A reduced representation can be approximately reconstructed:

`x_hat = μ + Q_kz`

When fewer components are retained, reconstruction generally loses information.

The reconstruction error measures how much information was discarded.

The C++ and Python implementations calculate mean squared reconstruction error.

For PCA, the discarded variance is related to the eigenvalues of the omitted components.

---

## PCA and feature scaling

Centering alone does not make all features comparable.

If one feature is measured in units with a much larger numerical scale, it can dominate the covariance matrix.

Depending on the application, PCA may therefore use standardized features.

Standardization commonly uses:

`z = (x - μ) / σ`

where `σ` is the feature standard deviation.

This changes the analysis from covariance-based PCA to a correlation-based interpretation when applied consistently.

Whether scaling is appropriate depends on the meaning and units of the features.

---

## PCA versus arbitrary eigen decomposition

PCA is not simply "finding eigenvalues of any matrix."

The important matrix is usually the covariance matrix or, equivalently in many computational settings, a data-derived matrix such as `XᵀX`.

The covariance matrix has the required statistical interpretation.

Its eigenvectors describe directions in feature space.

Its eigenvalues quantify variance along those directions.

---

# Machine learning relevance

Eigenvalues and eigenvectors appear in many machine learning techniques.

## PCA

PCA uses covariance eigenvectors to determine principal directions.

## Spectral clustering

Spectral clustering uses eigenvectors of graph-derived matrices to transform data into a representation where clustering structure can become easier to identify.

## Kernel methods

Kernel matrices can be analyzed spectrally. Their eigenvalues and eigenvectors provide information about the geometry and effective dimensionality of the represented data.

## Covariance analysis

Eigenvalues can identify directions of high or low variance.

## Optimization

The eigenvalues of Hessian matrices describe curvature directions in many optimization problems.

Large positive eigenvalues indicate strong curvature along corresponding eigenvector directions.

Small eigenvalues indicate flatter directions.

Negative eigenvalues can indicate directions of negative curvature.

## Graph machine learning

Graph adjacency matrices and graph Laplacians have important spectral structures.

Eigenvectors can encode graph connectivity, smoothness, community structure, and diffusion behavior.

## Markov models

Transition matrices have eigenvalues and eigenvectors that describe long-term behavior and convergence properties.

An eigenvalue of `1` is especially important because stationary distributions correspond to invariant states.

---

# Graph Laplacians

For a graph with adjacency matrix `A`, define the degree matrix `D`.

The graph Laplacian is:

`L = D - A`

For an undirected graph, `L` is symmetric.

The vector of all ones satisfies:

`L1 = 0`

Therefore zero is an eigenvalue.

For an undirected graph, the multiplicity of zero as a Laplacian eigenvalue is related to the number of connected components.

This gives spectral methods a way to extract structural information from graphs.

The C++ case study constructs a sensor-network graph and computes its Laplacian.

---

# Markov chains

A Markov transition matrix describes transitions between states.

Under the row-vector convention:

`π_next = πP`

A stationary distribution satisfies:

`πP = π`

Rewriting:

`πP = 1π`

shows the relationship with eigenvalue `1`.

The stationary distribution is therefore an eigenvector-like invariant object associated with eigenvalue one, with the additional requirement that its entries form a probability distribution.

The Python, JavaScript, and C++ implementations demonstrate iterative computation of a stationary distribution.

---

# JavaScript implementation

The JavaScript implementation provides a numerical implementation without external npm packages.

It demonstrates:

- matrix construction,
- vector operations,
- matrix multiplication,
- matrix powers,
- determinants,
- traces,
- 2 × 2 characteristic equations,
- real eigenvalues,
- complex eigenvalue representation,
- eigenvectors,
- residual validation,
- matrix inversion,
- diagonalization,
- Rayleigh quotients,
- power iteration,
- covariance matrices,
- PCA,
- projection,
- reconstruction,
- reconstruction error,
- Markov chains,
- graph Laplacians,
- repeated eigenvalue cases.

JavaScript uses IEEE 754 double-precision numbers for its standard `Number` type.

This has important consequences.

Numerical calculations can contain floating-point error, and exact symbolic equality should generally not be assumed.

For example, a calculation that mathematically equals zero may produce a small value such as:

`2.220446049250313e-16`

Numerical comparisons should therefore normally use a tolerance.

---

## JavaScript complex values

Standard JavaScript does not provide a built-in primitive complex-number type equivalent to Python's built-in `complex`.

The JavaScript implementation represents a complex value using an object containing:

`{ real, imaginary }`

This allows the 2 × 2 eigenvalue routine to demonstrate complex roots without an external library.

The main PCA and power-iteration paths intentionally remain real-valued.

---

# C++ sensor analytics case study

The C++ implementation models a small sensor analytics system.

The scenario contains:

- sensor observations,
- correlated measurements,
- a sensor network topology,
- graph analysis,
- PCA,
- dimensionality reduction,
- reconstruction,
- a state-transition model.

The objective is to demonstrate how eigenvalues and eigenvectors can become part of a larger computational system rather than existing only as isolated mathematical exercises.

---

## Sensor observations

The case study uses observations such as:

`[2.0, 1.1]`

`[3.0, 1.9]`

`[4.0, 3.0]`

and additional measurements following the same general pattern.

The two columns represent two correlated measurements.

The correlation makes the data approximately lie along a dominant direction in two-dimensional feature space.

---

## Covariance computation

The program calculates feature means and centers the observations.

It then calculates:

`C = X_centeredᵀ X_centered / (n - 1)`

The resulting covariance matrix is symmetric.

The largest eigenvalue and its corresponding eigenvector are found using power iteration.

---

## Dominant principal component

The dominant eigenvector is the first principal component.

The dominant eigenvalue represents the variance associated with that direction.

The explained variance ratio is calculated as:

`λ_max / trace(C)`

for the two-dimensional case.

This connects the numerical eigenvalue computation directly to a machine-learning interpretation.

---

## Compression

Each centered observation is projected onto the principal component:

`score = (x - μ)ᵀv`

The original two-dimensional observation is therefore represented by one scalar.

This is a simple example of dimensionality reduction.

---

## Reconstruction

The compressed observation is reconstructed using:

`x_hat = μ + score v`

The reconstruction is not necessarily identical to the original observation.

The difference represents information that was discarded by retaining only one component.

The program calculates mean squared reconstruction error to quantify this difference.

---

## Eigenpair residual in the C++ case study

After power iteration, the C++ program calculates:

`||Cv - λv||`

A small residual provides evidence that the estimated eigenvalue and eigenvector satisfy the eigenvalue equation numerically.

This is preferable to relying only on a displayed eigenvalue.

---

# Architectural structure of the C++ implementation

The program is organized into reusable components.

### Vector operations

The implementation includes:

- dot product,
- norm,
- normalization,
- addition,
- subtraction,
- scalar multiplication.

### Matrix operations

It includes:

- identity construction,
- transpose,
- matrix-vector multiplication,
- matrix multiplication,
- matrix subtraction,
- outer products.

### Eigenvalue utilities

It includes:

- trace,
- determinant for 2 × 2 matrices,
- characteristic-equation roots,
- eigenvector construction,
- eigenpair residuals.

### Numerical eigenvalue estimation

Power iteration estimates the dominant eigenpair without explicitly constructing the characteristic polynomial.

### Statistical analysis

The PCA layer contains:

- feature means,
- centering,
- covariance calculation,
- dominant eigenpair estimation,
- projection,
- reconstruction,
- reconstruction error.

### Graph analysis

The graph component calculates:

`L = D - A`

for a sensor-network adjacency matrix.

### State-transition analysis

The transition component calculates an approximate stationary distribution through repeated multiplication.

### System class

`SensorAnalyticsSystem` combines the mathematical components into an application-level structure.

This separates low-level numerical operations from the higher-level use case.

---

# Complexity considerations

## Matrix-vector multiplication

For an `n × n` dense matrix and an `n`-element vector:

`O(n²)`

operations are required.

## Dense matrix multiplication

Multiplying two dense `n × n` matrices using the standard triple-loop algorithm requires:

`O(n³)`

time.

## Determinant by recursive expansion

The educational determinant implementation uses recursive cofactor expansion.

This becomes extremely expensive as matrix size increases.

It is appropriate for small teaching examples, not large numerical workloads.

## Matrix inversion

The Gauss-Jordan implementation has approximately cubic time complexity for dense square matrices:

`O(n³)`

## Power iteration

Each iteration requires approximately one matrix-vector multiplication:

`O(n²)`

for a dense matrix.

If `k` iterations are required, the approximate cost is:

`O(kn²)`

For sparse matrices, matrix-vector multiplication can be much cheaper when the sparse structure is exploited.

## PCA through covariance eigen decomposition

For a dataset with `n` observations and `d` features, explicitly forming the covariance matrix costs approximately:

`O(nd²)`

The eigen decomposition of the resulting `d × d` dense matrix can become expensive as `d` increases.

For high-dimensional data, singular value decomposition is often preferred because it avoids some of the numerical and computational problems associated with explicitly forming `XᵀX`.

---

# Eigen decomposition versus singular value decomposition

PCA can be formulated using eigen decomposition of a covariance matrix.

It can also be formulated using Singular Value Decomposition:

`X = UΣVᵀ`

For centered data, the columns of `V` correspond to principal directions.

The relationship is connected to:

`XᵀX = VΣ²Vᵀ`

This means that the eigenvalues of `XᵀX` correspond to squared singular values.

SVD is often preferred for numerical PCA because explicitly constructing `XᵀX` can worsen the condition number and amplify numerical effects.

The current implementations use eigen decomposition to make the conceptual connection between PCA and eigenvectors explicit.

---

# Numerical stability

The mathematical formulas are exact, but computer arithmetic is finite.

Important numerical issues include:

- floating-point rounding,
- cancellation,
- overflow,
- underflow,
- ill-conditioned matrices,
- nearly repeated eigenvalues,
- small residuals that are not exactly zero,
- unstable eigenvector directions.

A production numerical implementation should use appropriate tolerance rules rather than exact floating-point comparisons.

---

## Nearly repeated eigenvalues

When eigenvalues are close to each other, eigenvectors can become sensitive to small perturbations.

This does not necessarily mean the algorithm is incorrect.

It means the mathematical problem itself may be sensitive.

This is one reason robust numerical eigensolvers are important for production applications.

---

## Eigenvector sign ambiguity

If a numerical solver produces:

`v`

another correct solver may produce:

`-v`

The two results represent the same one-dimensional eigenspace.

Tests comparing eigenvectors should account for this.

Comparing the corresponding eigenpair residual is often more meaningful than requiring an exact sign.

---

## Scale sensitivity

If the matrix entries are extremely large or extremely small, numerical computations can become difficult.

Scaling, normalization, stable algorithms, and appropriate floating-point types can help.

---

# Common mistakes

## Confusing eigenvalues with eigenvectors

An eigenvalue is a scalar.

An eigenvector is a non-zero vector.

They appear together in:

`Av = λv`

but they are different mathematical objects.

## Using the zero vector

The zero vector technically satisfies:

`A0 = λ0`

for every `λ`.

It is therefore excluded from the definition of an eigenvector.

Eigenvectors must be non-zero.

## Forgetting the identity matrix

The characteristic equation is:

`det(A - λI) = 0`

not simply:

`det(A - λ) = 0`

The identity matrix is required because `λ` must be subtracted from every diagonal entry.

## Sorting eigenvalues without eigenvectors

Eigenvalues and eigenvectors form pairs.

If the eigenvalues are reordered, the eigenvectors must be reordered correspondingly.

## Assuming repeated eigenvalues imply multiple eigenvectors

Algebraic multiplicity and geometric multiplicity are different.

A repeated eigenvalue can have only one independent eigenvector.

## Assuming all real matrices have real eigenvalues

They do not.

A real rotation matrix can have complex eigenvalues.

## Treating numerical zero as exact zero

Floating-point calculations may produce very small residuals instead of exact zero.

Tolerance-based comparisons are required.

## Ignoring data centering in PCA

PCA normally requires centering.

Without centering, the resulting directions can reflect the location of the data relative to the origin rather than variation around the mean.

## Ignoring feature scale

A feature measured in very large units can dominate covariance-based PCA.

Whether to standardize depends on the application and meaning of the variables.

---

# Limitations of these implementations

These files are designed for education and small computational examples.

They intentionally do not attempt to replace mature numerical linear algebra libraries.

Important limitations include:

- direct recursive determinant calculation is unsuitable for large matrices;
- the eigenvalue solver is primarily designed for 2 × 2 demonstrations;
- the JavaScript complex-number representation is minimal;
- the C++ case study uses real-valued arithmetic;
- basic power iteration only estimates a dominant eigenpair;
- simple deflation can accumulate numerical errors;
- PCA is demonstrated with a small dataset;
- sparse matrix representations are not implemented;
- large-scale distributed eigensolvers are outside the scope of the implementation.

These limitations are deliberate because exposing the underlying mathematics is the main purpose of the implementations.

---

# Production implementation considerations

A production numerical system should consider:

- numerical stability,
- matrix conditioning,
- sparse versus dense storage,
- memory consumption,
- parallel computation,
- hardware acceleration,
- deterministic behavior,
- tolerance selection,
- input validation,
- overflow and underflow,
- reproducibility,
- monitoring,
- numerical regression tests,
- appropriate eigensolver selection.

For large scientific and machine-learning workloads, specialized algorithms such as QR-based eigensolvers, Lanczos methods, Arnoldi methods, or SVD-based approaches may be more appropriate than educational implementations.

The choice depends on matrix properties, sparsity, symmetry, dimension, and the number of eigenpairs required.

---

# Security considerations

Eigenvalue calculations are primarily mathematical and numerical rather than security mechanisms.

When integrated into an application, security concerns can still arise from the surrounding system.

Important considerations include:

- validating matrix dimensions,
- rejecting malformed input,
- preventing uncontrolled memory allocation,
- limiting computational workload,
- handling extremely large numerical values,
- detecting invalid floating-point values,
- preventing denial-of-service conditions caused by intentionally expensive inputs.

For an externally accessible numerical service, matrix size and iteration limits should be controlled.

The C++ case study uses validation and exception handling to prevent invalid matrix dimensions and invalid probability matrices from silently propagating through the computation.

---

# Implementation comparison

| Aspect | Python | JavaScript | C++ |
|---|---|---|---|
| Main purpose | Comprehensive study implementation | Numerical and application-oriented demonstrations | Industry-style spectral analytics case study |
| Matrix operations | Implemented directly | Implemented directly | Implemented directly |
| Eigenvalue demonstration | 2 × 2 analytical method | 2 × 2 analytical method | 2 × 2 analytical method |
| Eigenvector calculation | Included | Included for real eigenvalues | Included for real eigenvalues |
| Complex eigenvalues | Demonstrated | Represented explicitly | Real-valued case study |
| Power iteration | Included | Included | Included |
| PCA | Detailed | Included | Integrated into system |
| Markov chains | Included | Included | Included |
| Graph Laplacian | Included | Included | Integrated into system |
| Error handling | Exceptions | Exceptions | Exceptions |
| Primary educational emphasis | Mathematical breadth | Language-specific numerical behavior | Architecture and system integration |

---

# Relationship between the three implementations

The Python implementation emphasizes breadth and mathematical experimentation.

The JavaScript implementation demonstrates how the same numerical concepts can be expressed using JavaScript's array and object model while explicitly handling the absence of a native complex-number primitive.

The C++ implementation emphasizes system design.

It combines:

`data -> covariance -> eigenpair -> PCA -> compression -> reconstruction`

with:

`graph -> Laplacian`

and:

`transition matrix -> stationary distribution`

This illustrates how eigenvalue methods can become part of a larger analytical pipeline.

---

# Important mathematical relationships

The following identities are central to the topic.

Eigenvalue equation:

`Av = λv`

Characteristic equation:

`det(A - λI) = 0`

Trace relationship:

`tr(A) = Σ λ_i`

Determinant relationship:

`det(A) = Π λ_i`

Diagonalization:

`A = PDP⁻¹`

Symmetric spectral decomposition:

`A = QΛQᵀ`

Matrix powers:

`Aᵏ = PDᵏP⁻¹`

Rayleigh quotient:

`R(v) = (vᵀAv)/(vᵀv)`

Covariance matrix:

`C = XᵀX/(n - 1)`

PCA principal direction:

`Cv = λv`

Graph Laplacian:

`L = D - A`

Stationary Markov distribution:

`πP = π`

---

# Interpreting eigenvalues

Eigenvalues can describe how a transformation behaves along its eigenvector directions.

For a linear dynamical process:

`x_{t+1} = Ax_t`

eigenvalues can determine growth or decay rates.

If the magnitude of an eigenvalue is greater than one, the corresponding component can grow under repeated application.

If it is less than one, that component tends to decay.

If its magnitude is one, that component may remain bounded or persist depending on the associated structure.

For discrete-time systems, eigenvalue magnitudes therefore provide important information about stability.

---

# Eigenvalues and system behavior

Consider:

`x_k = A^k x_0`

If `A` is diagonalizable:

`x_k = PD^kP⁻¹x_0`

The long-term behavior is strongly influenced by the largest eigenvalue magnitudes.

This is one of the reasons eigen decomposition appears in:

- dynamical systems,
- differential equations,
- Markov chains,
- network analysis,
- signal processing,
- control systems,
- numerical optimization.

The same mathematical structure appears in different applications because repeated linear transformations naturally lead to powers of matrices.

---

# Why PCA works geometrically

Imagine a cloud of data points.

If the cloud is elongated diagonally, the direction of the elongation represents a direction of high variance.

The covariance matrix encodes relationships among the feature dimensions.

Its largest eigenvalue identifies the variance magnitude of the strongest direction.

Its corresponding eigenvector identifies the direction itself.

The second eigenvector gives an orthogonal direction.

For a two-dimensional dataset, these two eigenvectors form a rotated coordinate system aligned with the major and minor axes of the data's variance structure.

---

# PCA information loss

Suppose a dataset has eigenvalues:

`λ₁ >= λ₂ >= λ₃`

If only the first component is retained, the second and third directions are discarded.

The retained variance is:

`λ₁`

and the discarded variance is:

`λ₂ + λ₃`

The total variance is:

`λ₁ + λ₂ + λ₃`

Therefore:

`explained variance ratio = λ₁ / (λ₁ + λ₂ + λ₃)`

This gives a quantitative basis for selecting the number of components.

---

# Covariance eigenvectors versus covariance eigenvalues

The two play different roles.

The eigenvector answers:

**Which direction?**

The eigenvalue answers:

**How much variance exists in that direction?**

For PCA:

- eigenvectors define principal axes;
- eigenvalues quantify variance along those axes.

Both are necessary for interpreting the decomposition.

---

# Eigen decomposition versus ordinary matrix multiplication

Ordinary matrix multiplication describes what a transformation does directly.

Eigen decomposition changes the representation so the transformation can be understood through independent eigen-directions when enough eigenvectors exist.

This can simplify:

- matrix powers,
- theoretical analysis,
- stability analysis,
- dimensionality reduction,
- graph analysis,
- dynamical-system interpretation.

The decomposition does not change the underlying linear transformation. It changes the coordinate representation used to analyze it.

---

# Practical workflow

A robust eigenvalue workflow is:

1. Validate the matrix dimensions.
2. Determine whether the matrix has useful structural properties such as symmetry or sparsity.
3. Select an appropriate numerical method.
4. Calculate eigenvalues and eigenvectors.
5. Pair each eigenvector with its eigenvalue.
6. Validate eigenpairs using residuals.
7. Check multiplicity and diagonalizability when relevant.
8. Sort eigenpairs consistently when an application requires ordering.
9. Interpret the eigenvalues according to the application.
10. Monitor numerical stability and conditioning.

For PCA:

1. Validate observations.
2. Center the features.
3. Decide whether scaling is appropriate.
4. Calculate covariance or use an SVD-based formulation.
5. Compute principal directions.
6. Sort components by explained variance.
7. Project observations.
8. Evaluate reconstruction or retained variance.
9. Interpret the retained components in the context of the original features.

---

# Running the Python implementation

Save the Python content as:

`eigenvalues_eigenvectors.py`

Run:

`python eigenvalues_eigenvectors.py`

The program executes the mathematical demonstrations sequentially.

---

# Running the JavaScript implementation

Save the JavaScript content as:

`eigenvalues_eigenvectors.js`

Run with Node.js:

`node eigenvalues_eigenvectors.js`

The program prints matrix calculations, eigenpairs, PCA results, power-iteration output, Markov-chain calculations, and graph-Laplacian demonstrations.

---

# Compiling the C++ implementation

Save the C++ content as:

`eigenvalues_eigenvectors.cpp`

Compile using C++17:

`g++ -std=c++17 -O2 eigenvalues_eigenvectors.cpp -o spectral_engine`

Run the executable:

`./spectral_engine`

On Windows with a suitable C++ compiler, the generated executable can be run as:

`spectral_engine.exe`

---

# Expected computational themes

The numerical output may vary slightly between implementations because:

- floating-point arithmetic is approximate;
- eigenvector signs can differ;
- iteration stopping criteria can differ;
- different programming languages may perform intermediate calculations differently.

A mathematically equivalent result does not necessarily have identical printed decimal digits.

The most useful numerical checks are structural relationships and residuals.

For an eigenpair:

`||Av - λv||`

should be small.

For PCA:

`sum(explained_variance_ratio)`

should be approximately one when all components are included.

For a transition matrix:

`sum(π_i)`

should be approximately one.

For a stationary distribution:

`πP`

should be approximately equal to `π`.

For a graph Laplacian:

`L1`

should be zero for an undirected graph using the standard Laplacian definition.

---

# Core conceptual model

The entire topic can be connected through one idea:

A matrix represents a linear transformation.

Most vectors change direction under that transformation.

Eigenvectors identify special directions that remain on their own one-dimensional subspaces.

Eigenvalues specify the scaling applied along those directions.

When enough independent eigenvectors exist, the matrix can be expressed in an eigenvector coordinate system.

For symmetric matrices, the structure becomes especially clean:

`A = QΛQᵀ`

Machine learning uses this structure directly in PCA.

The covariance matrix describes relationships among features.

Its eigenvectors provide principal directions.

Its eigenvalues quantify variance along those directions.

The same spectral ideas extend to graph Laplacians, Markov chains, dynamical systems, optimization, and many other areas in computational mathematics.
