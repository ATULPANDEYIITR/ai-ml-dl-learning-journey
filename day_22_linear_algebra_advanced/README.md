# Advanced Linear Algebra

## Introduction

Linear algebra is the mathematical framework for describing vectors, matrices, linear transformations, systems of equations, geometric relationships, optimization problems, data transformations, and many computational models.

At an elementary level, linear algebra can appear to be a collection of procedures for multiplying matrices or solving simultaneous equations. At an advanced level, it becomes a unified language for understanding structure.

The central objects are vectors and linear transformations. A matrix is one representation of a linear transformation after bases have been selected. Concepts such as rank, nullity, eigenvalues, orthogonality, singular values, and positive definiteness describe structural properties of these transformations.

This project implements advanced linear algebra concepts in three programming languages:

- Python provides a broad educational numerical implementation.
- JavaScript demonstrates the same mathematical ideas in a runtime commonly used for web and application development.
- C++ develops a more explicit industry-style technical case study involving a sensor analytics system.

All three implementations use standard language facilities without external numerical libraries.

## Fundamental mathematical objects

### Scalars

A scalar is a single numerical quantity.

Examples include:

- `5`
- `-2.5`
- `0`
- `3.14159`

Scalars can multiply vectors and matrices.

If

    v = [2, 3]

then

    4v = [8, 12]

Scalar multiplication changes magnitude and can also reverse direction when the scalar is negative.

### Vectors

A vector is an ordered collection of scalars.

For example,

    v = [3, 4]

is a vector in two-dimensional real space.

Vectors can represent:

- positions
- velocities
- forces
- sensor measurements
- features in a machine-learning dataset
- coefficients
- directions
- states of a system

The Python, JavaScript, and C++ implementations all represent vectors as collections of numerical values.

### Vector addition

Vectors of equal dimension can be added component by component.

    [a, b] + [c, d] = [a+c, b+d]

Vector addition is fundamental because many physical and computational systems combine independent contributions.

### Inner product

For real vectors,

    x · y = x1y1 + x2y2 + ... + xnyn

The inner product measures an important relationship between vectors.

If

    x · y = 0

the vectors are orthogonal.

Orthogonality is central to projections, least squares, QR decomposition, Fourier methods, numerical optimization, and many geometric algorithms.

### Norm

The Euclidean norm is

    ||x||₂ = sqrt(x · x)

For

    x = [3, 4]

the norm is

    ||x||₂ = 5

A norm measures vector magnitude.

The Python implementation also supports the L1 and general Lp forms through its vector norm function.

## Matrix fundamentals

A matrix is a rectangular arrangement of numbers.

For example,

    A = [ 1  2
          3  4 ]

has two rows and two columns.

The dimensions are therefore 2 × 2.

Matrices can represent:

- systems of equations
- coordinate transformations
- rotations
- scaling
- projections
- graph relationships
- covariance structures
- linear operators
- discretized physical systems

## Matrix addition

Matrices can be added only when they have equal dimensions.

If

    A = [a b
         c d]

and

    B = [e f
         g h]

then

    A+B = [a+e b+f
           c+g d+h]

The Python, JavaScript, and C++ matrix implementations validate dimensions before performing such operations.

## Matrix multiplication

Matrix multiplication is not component-by-component multiplication.

For matrices A and B,

    C = AB

has entries

    Cij = Σ Aik Bkj

The number of columns of A must equal the number of rows of B.

If A is m × n and B is n × p, then AB is m × p.

The classical dense algorithm requires approximately

    O(mnp)

arithmetic operations.

The implementations use the standard triple-loop algorithm and explicitly validate incompatible dimensions.

## Matrix-vector multiplication

A matrix transforms a vector.

    y = Ax

This is one of the most important interpretations of a matrix.

For example,

    A = [2 0
         0 3]

and

    x = [1
         2]

produce

    Ax = [2
          6]

The matrix scales one coordinate by 2 and the other by 3.

This interpretation connects matrix arithmetic to linear transformations.

## Transpose

The transpose exchanges rows and columns.

If

    A = [1 2 3
         4 5 6]

then

    Aᵀ = [1 4
          2 5
          3 6]

Important identities include

    (AB)ᵀ = BᵀAᵀ

and

    (Aᵀ)ᵀ = A

A matrix satisfying

    Aᵀ = A

is symmetric.

Symmetric matrices have especially useful spectral properties.

## Trace

For a square matrix,

    tr(A) = A11 + A22 + ... + Ann

The trace is also equal to the sum of the eigenvalues, counting algebraic multiplicity.

It satisfies

    tr(AB) = tr(BA)

when the products are defined.

## Linear combinations, span, and basis

A linear combination of vectors v1, v2, ..., vk is

    c1v1 + c2v2 + ... + ckvk

where the coefficients are scalars.

The span of a set of vectors is the collection of every vector that can be produced from their linear combinations.

A basis is a linearly independent spanning set.

For R², two linearly independent vectors form a basis.

For R³, three linearly independent vectors form a basis.

The number of vectors in a basis is the dimension of the vector space.

## Linear independence

Vectors are linearly dependent if some non-trivial combination produces zero.

For vectors v1, ..., vk,

    c1v1 + ... + ckvk = 0

has a solution in which at least one coefficient is non-zero.

If the only solution is

    c1 = c2 = ... = ck = 0

the vectors are linearly independent.

Linear independence determines whether a set can serve as a basis.

## Rank

The rank of a matrix is the dimension of its column space.

It is also equal to:

- the dimension of the row space
- the number of pivot columns
- the maximum number of linearly independent columns
- the maximum number of linearly independent rows

The Python and JavaScript implementations calculate rank from reduced row-echelon form. The C++ implementation uses the same conceptual method.

For an n × n matrix:

    rank(A) = n

means the matrix is full rank.

For a square matrix, full rank is equivalent to invertibility.

## Null space

The null space of A is

    N(A) = {x : Ax = 0}

It contains all vectors that the transformation represented by A maps to zero.

The dimension of the null space is called the nullity.

The rank-nullity theorem states

    rank(A) + nullity(A) = number of columns of A

This theorem is one of the central structural results of linear algebra.

## Gaussian elimination

Gaussian elimination transforms a system into a form that is easier to solve.

Elementary row operations are:

1. Swap two rows.
2. Multiply a row by a non-zero scalar.
3. Add a multiple of one row to another row.

These operations preserve the solution set of a linear system.

The implementations use partial pivoting.

Partial pivoting chooses a large available pivot instead of blindly using the current diagonal element. This improves numerical behavior when floating-point arithmetic is used.

## Reduced row-echelon form

A matrix is in reduced row-echelon form when:

- every non-zero row begins with a leading 1
- each pivot is the only non-zero element in its column
- pivot positions move to the right as rows increase
- zero rows appear below non-zero rows

RREF is particularly useful for:

- solving systems
- finding rank
- identifying free variables
- finding null-space bases

## Consistent and inconsistent systems

A linear system may have:

- no solution
- exactly one solution
- infinitely many solutions

A row such as

    [0 0 0 | 5]

represents

    0 = 5

and therefore indicates inconsistency.

A free variable indicates that the solution is not unique.

The implementations explicitly detect inconsistent systems and systems without unique solutions.

## Determinants

The determinant is a scalar associated with a square matrix.

For a 2 × 2 matrix,

    A = [a b
         c d]

the determinant is

    det(A) = ad - bc

A square matrix is invertible exactly when

    det(A) != 0

The determinant also describes signed volume scaling under the corresponding linear transformation.

If the absolute determinant is greater than one, volumes are scaled upward in magnitude. If it is between zero and one, volumes are contracted. A negative determinant also indicates orientation reversal.

## Matrix inverse

The inverse A⁻¹ satisfies

    AA⁻¹ = A⁻¹A = I

where I is the identity matrix.

The inverse exists only for a square full-rank matrix.

Although the equation

    x = A⁻¹b

is mathematically useful, explicitly computing an inverse is often unnecessary in numerical software.

Solving

    Ax = b

directly is usually more efficient and numerically preferable.

The C++ case study demonstrates this distinction through direct solution and Conjugate Gradient.

## Vector spaces and subspaces

A subspace must satisfy:

- it contains the zero vector
- it is closed under addition
- it is closed under scalar multiplication

Important subspaces associated with a matrix include:

- column space
- row space
- null space
- left null space

The four fundamental subspaces form a major structural framework for matrix analysis.

## Orthogonality

Two vectors are orthogonal if

    x · y = 0

An orthonormal set has vectors that are both:

- mutually orthogonal
- unit length

Orthonormal bases are computationally valuable because coordinates can often be calculated using simple inner products.

## Projection

The projection of v onto a non-zero vector u is

    proj_u(v) =
        (v · u)/(u · u) u

The projection is the component of v lying along u.

The residual

    v - proj_u(v)

is orthogonal to u.

Projection is fundamental to least squares and geometric approximation.

## Gram-Schmidt process

Gram-Schmidt transforms linearly independent vectors into an orthonormal basis spanning the same subspace.

For the first vector,

    q1 = v1 / ||v1||

For the second,

    u2 = v2 - (v2 · q1)q1

and then

    q2 = u2 / ||u2||

The process continues similarly for additional vectors.

The implementation includes a classical Gram-Schmidt method.

Classical Gram-Schmidt is mathematically correct but can suffer from numerical loss of orthogonality for ill-conditioned data.

Modified Gram-Schmidt generally provides better numerical behavior.

Householder transformations are commonly preferred in high-quality QR implementations.

## QR decomposition

QR decomposition expresses a matrix as

    A = QR

where Q has orthonormal columns and R is upper triangular.

QR is important for:

- least squares
- eigenvalue algorithms
- orthogonalization
- numerical linear algebra

For an m × n matrix with m >= n, a thin QR decomposition has:

- Q of size m × n
- R of size n × n

The Python, JavaScript, and C++ implementations demonstrate QR through Gram-Schmidt.

Production numerical software generally uses more stable Householder-based QR implementations.

## Least squares

When a system has more equations than unknowns, an exact solution may not exist.

Instead, least squares finds x minimizing

    ||Ax - b||₂²

The residual is

    r = b - Ax

and the optimal solution satisfies the normal equations

    AᵀAx = Aᵀb

The Python, JavaScript, and C++ implementations demonstrate this method.

For numerical work, QR or SVD is generally preferable to normal equations because forming AᵀA can significantly worsen conditioning.

## Geometric interpretation of least squares

The least-squares prediction Ax is the orthogonal projection of b onto the column space of A.

At the solution,

    Aᵀ(b - Ax) = 0

Therefore the residual is orthogonal to every column of A.

This provides the geometric explanation behind the normal equations.

## Eigenvalues and eigenvectors

A non-zero vector v is an eigenvector of A if

    Av = λv

where λ is the corresponding eigenvalue.

The transformation changes the eigenvector's magnitude and possibly direction only through scalar multiplication.

Eigenvalues satisfy

    det(A - λI) = 0

The corresponding vectors lie in eigenspaces.

Eigenvalues appear throughout:

- differential equations
- dynamical systems
- quantum mechanics
- graph analysis
- stability analysis
- optimization
- principal component analysis
- numerical algorithms

## Power iteration

Power iteration repeatedly applies A to a vector:

    x(k+1) = Ax(k)

followed by normalization.

Under suitable conditions, the vector converges toward a dominant eigenvector.

The associated eigenvalue can be estimated with the Rayleigh quotient:

    λ ≈ (xᵀAx)/(xᵀx)

Power iteration is simple and useful, but it has limitations.

Convergence depends on the spectral gap. If the largest and second-largest eigenvalues in magnitude are close, convergence can be slow.

It also primarily identifies the dominant eigenvalue rather than the full spectrum.

## Symmetric matrices

A real symmetric matrix satisfies

    A = Aᵀ

The spectral theorem states that a real symmetric matrix can be orthogonally diagonalized:

    A = QΛQᵀ

where Q is orthogonal and Λ is diagonal.

The columns of Q are orthonormal eigenvectors.

This is one of the most useful structures in numerical linear algebra.

The Python implementation includes a Jacobi-style eigenvalue method for symmetric matrices.

## Diagonalization

A matrix is diagonalizable if there exists an invertible P such that

    A = PΛP⁻¹

where Λ is diagonal.

When a matrix has a complete set of linearly independent eigenvectors, those eigenvectors can form the columns of P.

Diagonalization can simplify repeated matrix powers:

    A^k = PΛ^kP⁻¹

because powers of a diagonal matrix are computed element by element.

Not every matrix is diagonalizable.

A matrix can have repeated eigenvalues without having enough independent eigenvectors.

## Singular Value Decomposition

The Singular Value Decomposition is

    A = UΣVᵀ

where:

- U contains left singular vectors
- V contains right singular vectors
- Σ contains non-negative singular values

Singular values are the square roots of eigenvalues of

    AᵀA

SVD applies to rectangular matrices and is therefore more general than ordinary eigenvalue decomposition.

Important applications include:

- dimensionality reduction
- least squares
- pseudoinverses
- noise filtering
- low-rank approximation
- numerical rank determination
- recommender systems
- image compression

The Python implementation builds an educational SVD through the eigenstructure of AᵀA.

This makes the mathematical relationship clear, but it is not intended to replace optimized production SVD implementations.

## Moore-Penrose pseudoinverse

The pseudoinverse A⁺ generalizes matrix inversion.

For a suitable SVD,

    A = UΣVᵀ

the pseudoinverse is

    A⁺ = VΣ⁺Uᵀ

where each non-zero singular value σ is replaced by

    1/σ

The pseudoinverse is useful for:

- underdetermined systems
- overdetermined systems
- least squares
- rank-deficient problems

Small singular values require care because their reciprocals can become extremely large and amplify noise.

## Condition number

The condition number describes sensitivity to perturbations.

A large condition number indicates that small input changes can potentially produce large output changes.

For an invertible matrix under a chosen norm,

    κ(A) = ||A|| ||A⁻¹||

The Python implementation demonstrates a Frobenius-norm estimate.

Conditioning is different from algorithmic stability.

A problem may be intrinsically ill-conditioned even when an algorithm is numerically stable.

## Numerical stability

Floating-point arithmetic cannot represent every real number exactly.

For this reason,

    0.1 + 0.2

may not be represented internally as exactly `0.3`.

Numerical linear algebra therefore avoids relying on exact equality.

The implementations use tolerance-based comparisons such as:

    |a - b| < ε

Partial pivoting is another example of a numerical-stability technique.

## Positive-definite matrices

A real symmetric matrix A is positive definite if

    xᵀAx > 0

for every non-zero vector x.

Positive-definite matrices have important properties:

- all eigenvalues are positive
- the matrix is invertible
- Cholesky factorization exists
- quadratic forms are strictly positive
- Conjugate Gradient can be applied

The Python implementation checks positive definiteness through leading principal minors for symmetric matrices.

## Quadratic forms

A quadratic form has the structure

    xᵀAx

when A is an appropriate square matrix.

Quadratic forms occur in:

- energy models
- optimization
- statistics
- covariance analysis
- geometry
- control systems

For a positive-definite A, the quadratic form behaves like a generalized squared length.

## Change of basis

A vector can have different coordinate representations depending on the chosen basis.

If B contains basis vectors as columns,

    Bc = x

then c contains the coordinates of x in the B basis.

The Python implementation solves this system directly.

This distinction is important because a vector and its coordinate representation are conceptually different objects.

## Affine transformations

Linear transformations preserve the origin.

Translations do not.

Homogeneous coordinates allow translation to be incorporated into matrix multiplication.

A two-dimensional affine transformation can be represented as

    [ a b tx
      c d ty
      0 0  1 ]

A point `[x, y]` is extended to

    [x, y, 1]

and multiplied by the matrix.

This approach is widely used in:

- computer graphics
- robotics
- CAD
- geometric modeling
- computer vision

The Python implementation demonstrates this representation.

## Principal Component Analysis

PCA identifies directions of maximum variance.

A common PCA procedure is:

1. Center the data.
2. Compute the covariance matrix.
3. Compute covariance eigenvectors.
4. Sort eigenvectors by descending eigenvalue.
5. Project the centered observations onto selected principal directions.

The covariance matrix is

    C = 1/(n-1) XᵀX

after centering X.

The eigenvectors of C are the principal directions.

The corresponding eigenvalues represent variance along those directions.

PCA is useful for:

- dimensionality reduction
- exploratory data analysis
- visualization
- feature compression
- noise reduction

The Python implementation performs the full educational workflow, including covariance and projection. The JavaScript and C++ implementations demonstrate the covariance stage within their respective environments.

## Graph Laplacian

For a graph with adjacency matrix A and degree matrix D,

    L = D - A

is the graph Laplacian.

For an undirected graph, L is symmetric.

One important property is

    L1 = 0

where 1 is the all-ones vector.

This occurs because each diagonal degree equals the sum of the corresponding adjacency row.

Graph Laplacians are used in:

- network analysis
- clustering
- spectral graph theory
- consensus systems
- image segmentation
- diffusion models
- sensor networks

The C++ case study models a network of communicating sensors.

## Conjugate Gradient

Conjugate Gradient solves systems

    Ax = b

when A is symmetric positive definite.

It does not calculate A⁻¹ explicitly.

Starting from an initial approximation, it constructs mutually conjugate search directions.

The method is particularly important for large sparse systems because matrix-vector multiplication can be much cheaper than dense matrix factorization.

The C++ implementation includes tolerance-based termination and validates the matrix symmetry condition.

The Python implementation also checks positive definiteness.

## Python implementation

The Python implementation is designed as a broad educational numerical laboratory.

It contains:

- vector operations
- a matrix class
- matrix arithmetic
- Gaussian elimination
- RREF
- rank
- determinant
- inverse
- linear-system solving
- null-space calculation
- column-space extraction
- projections
- Gram-Schmidt
- QR decomposition
- least squares
- power iteration
- Rayleigh quotient
- symmetric eigenvalue computation
- SVD
- pseudoinverse
- quadratic forms
- positive-definiteness testing
- condition estimation
- basis-coordinate conversion
- homogeneous transformations
- PCA
- graph Laplacian construction
- Conjugate Gradient
- edge-case handling
- automated assertions

The implementation is intentionally explicit. Mathematical operations are visible instead of being hidden behind a numerical library.

### Python matrix representation

The Python `Matrix` class stores values as nested lists.

For example, an m × n matrix is represented conceptually as:

    [
        [a11, a12, ..., a1n],
        [a21, a22, ..., a2n],
        ...
    ]

This makes the implementation easy to inspect but is not necessarily the most efficient representation for very large numerical workloads.

### Python elimination

Both row-echelon and reduced row-echelon transformations use pivot selection.

The implementation chooses the row containing the largest absolute candidate in the current pivot column.

This is partial pivoting.

### Python SVD

The educational SVD implementation uses the relationship

    AᵀA v = σ²v

to obtain right singular vectors and singular values.

This makes the theoretical relationship between eigenvalue decomposition and SVD visible.

Production SVD implementations require substantially more sophisticated numerical algorithms.

### Python PCA

The PCA implementation centers the dataset, builds the covariance matrix, obtains covariance eigenvectors, sorts them by eigenvalue, and projects observations.

This connects abstract eigenvalue theory to a practical data-analysis procedure.

## JavaScript implementation

The JavaScript implementation provides a compact numerical environment using standard ECMAScript.

It demonstrates:

- vector operations
- a reusable Matrix class
- matrix multiplication
- transposition
- RREF
- rank
- determinant
- inverse
- system solving
- projection
- Gram-Schmidt
- QR decomposition
- least squares
- power iteration
- quadratic forms
- covariance computation
- graph Laplacians
- Conjugate Gradient
- validation and error handling

The implementation is executable with a normal JavaScript runtime such as Node.js.

### JavaScript numerical behavior

JavaScript's ordinary `Number` type uses IEEE-754 double-precision floating-point arithmetic.

Consequently:

- exact equality is inappropriate for many numerical comparisons
- tolerance-based comparisons are required
- very large values can overflow
- very small values can underflow
- ill-conditioned systems can amplify errors

For performance-sensitive numerical software, typed arrays such as `Float64Array` can provide a more compact representation than ordinary nested arrays.

The example deliberately keeps the implementation readable rather than optimizing storage layout.

## C++ case study

The C++ program models a sensor analytics system.

A sensor network provides numerical measurements. The system must transform, calibrate, estimate, analyze, and solve equations involving those measurements.

The architecture contains:

- a vector abstraction based on `std::vector<double>`
- a reusable dense Matrix class
- elimination routines
- determinant calculation
- inversion
- linear-system solving
- Gram-Schmidt orthogonalization
- QR decomposition
- least squares
- covariance analysis
- power iteration
- quadratic forms
- graph Laplacian construction
- Conjugate Gradient
- validation and failure handling

## C++ case-study stages

### Sensor measurement vectors

Sensor values are represented as vectors.

A measurement can contain quantities such as:

    temperature
    pressure

Vector addition is used to represent calibration offsets.

### Calibration matrix

The calibration matrix models linear interactions between sensor channels.

The multiplication

    corrected = A * measurement

applies the transformation.

This demonstrates how a matrix can represent a physical or computational transformation.

### Calibration equations

The program solves a small system

    Ax = b

to determine unknown calibration parameters.

The system is solved through elimination rather than by explicitly calculating A⁻¹.

### Rank analysis

The program constructs a matrix containing dependent rows.

Its rank is smaller than the number of rows and columns because one row contains no additional independent information.

This demonstrates why rank is related to information content and identifiability.

### QR decomposition

The sensor design matrix is decomposed as

    A = QR

The program reconstructs Q R and prints the result.

The reconstruction provides a practical verification that the decomposition represents the original matrix.

### Least-squares calibration

Experimental measurements contain noise, so an exact line may not pass through every point.

The program constructs the design matrix

    [1 x]

for each observation and solves for:

    intercept
    gain

using least squares.

This is a direct example of linear algebra applied to parameter estimation.

### Covariance analysis

The sensor observations are converted into a covariance matrix.

The covariance matrix describes how feature variations occur together.

Large positive off-diagonal values indicate positive linear association, while negative values indicate negative linear association.

Covariance itself does not establish causation.

### Dominant eigenvalue

Power iteration identifies a dominant eigenvalue and eigenvector.

This can represent a dominant direction in a dynamical system or an important structural mode.

### Quadratic energy model

The program evaluates

    xᵀAx

for a symmetric matrix.

This provides a mathematical model of an energy-like quantity.

Positive-definite matrices are especially useful because the value is positive for every non-zero state vector.

### Sensor-network Laplacian

The sensor network is represented by an adjacency matrix.

The degree of each sensor is the number of direct connections.

The Laplacian is constructed as

    L = D - A

Applying L to the constant vector produces zero.

This property is fundamental to graph-based modeling.

### Conjugate Gradient

The final computational stage solves a symmetric positive-definite system with Conjugate Gradient.

The program also calculates the solution through direct elimination so the results can be compared.

The important architectural point is that solving a system does not require explicitly constructing an inverse.

## Important distinctions

### Matrix versus linear transformation

A linear transformation is a mathematical mapping.

A matrix is a representation of that mapping relative to selected bases.

The same abstract transformation can therefore have different matrix representations in different coordinate systems.

### Rank versus determinant

Rank applies to rectangular as well as square matrices.

The determinant is defined only for square matrices.

For a square matrix:

    det(A) != 0

is equivalent to full rank and invertibility.

### Eigenvalues versus singular values

Eigenvalues describe vectors that remain on their own span under a transformation.

Singular values describe stretching magnitudes associated with orthogonal input and output directions.

Eigenvalues can be negative or complex.

Singular values are always non-negative real numbers.

### Inverse versus pseudoinverse

An inverse exists only for an invertible square matrix.

A pseudoinverse can exist for rectangular and rank-deficient matrices.

The pseudoinverse therefore generalizes many inverse-like operations.

### Least squares versus exact solution

An exact solution satisfies

    Ax = b

exactly.

A least-squares solution minimizes

    ||Ax - b||²

when an exact solution is unavailable or when the system is overdetermined.

### QR versus normal equations

Normal equations are conceptually straightforward:

    AᵀAx = Aᵀb

But forming AᵀA can worsen conditioning.

QR avoids this particular squaring of the condition number and is generally more stable.

### QR versus SVD

QR is usually cheaper than SVD and is highly useful for solving least-squares problems.

SVD provides more detailed information about numerical rank and is often preferred for ill-conditioned or rank-deficient problems.

### Dense versus sparse matrices

The implementations use dense storage.

A dense n × n matrix requires O(n²) memory.

If most entries are zero, sparse representations can reduce memory and computational costs dramatically.

Sparse matrices are especially important in:

- graph problems
- finite-element systems
- large optimization problems
- scientific simulations
- discretized differential equations

## Edge cases

Important edge cases demonstrated by the implementations include:

- zero vectors
- singular matrices
- inconsistent systems
- non-unique systems
- incompatible matrix dimensions
- dependent columns
- non-square matrices
- numerical values close to zero
- floating-point comparison
- zero singular values
- degenerate iterative search directions

A robust implementation must distinguish genuine mathematical zero from a floating-point value that is merely very small.

## Common mistakes

### Treating matrix multiplication as element-wise multiplication

Matrix multiplication follows row-column inner products.

Element-wise multiplication is a different operation.

### Ignoring dimensions

The dimensions of matrices determine whether multiplication is legal.

For

    A(m × n)B(p × q)

the multiplication is valid only when

    n = p

### Dividing by a tiny pivot

A tiny pivot can produce very large intermediate values.

Partial pivoting helps reduce this problem.

### Explicitly computing inverses unnecessarily

If the goal is to solve

    Ax = b

direct solution methods are normally preferable to computing A⁻¹ first.

### Assuming every matrix is diagonalizable

Some matrices do not have enough independent eigenvectors for diagonalization.

### Assuming covariance implies causation

A covariance relationship is a statistical property, not a causal explanation.

### Using exact equality with floating-point results

Numerical algorithms should generally use tolerances.

### Using normal equations blindly

Normal equations can magnify conditioning problems.

QR or SVD is preferable when numerical stability is important.

### Assuming power iteration finds every eigenvalue

Basic power iteration normally targets a dominant eigenvalue.

Finding the complete spectrum requires more sophisticated methods.

## Limitations of the educational implementations

These implementations are intended to make mathematical mechanisms explicit.

They are not replacements for highly optimized numerical libraries.

Important limitations include:

- dense matrix storage
- relatively simple memory management
- educational elimination algorithms
- limited numerical robustness
- no SIMD optimization
- no multithreading
- no sparse matrix storage
- no distributed computation
- simplified SVD implementation
- simplified eigenvalue algorithms
- classical Gram-Schmidt
- limited matrix decomposition support

Production numerical software commonly relies on highly optimized kernels, carefully analyzed algorithms, specialized data structures, and extensive numerical testing.

## Performance considerations

For dense n × n matrices:

- matrix storage is O(n²)
- matrix multiplication is O(n³)
- Gaussian elimination is O(n³)
- dense inversion is O(n³)

For an m × n matrix multiplied by an n × p matrix:

    O(mnp)

operations are required by the classical algorithm.

QR decomposition is approximately O(mn²) for m >= n.

SVD is computationally expensive because it involves substantial matrix transformation and spectral computation.

Conjugate Gradient can be much more efficient for large sparse systems because its main operation is matrix-vector multiplication.

## Security and reliability considerations

Linear algebra is not normally viewed as a security mechanism, but numerical systems still require defensive engineering.

Input validation should prevent:

- malformed dimensions
- unexpected empty arrays
- non-finite values
- invalid matrix structures
- division by values effectively equal to zero

Applications should consider:

- NaN propagation
- positive and negative infinity
- floating-point overflow
- floating-point underflow
- maliciously large input sizes
- memory exhaustion
- ill-conditioned matrices
- unstable numerical transformations

A mathematically valid algorithm can still be unsafe operationally if input sizes are uncontrolled.

## Real-world applications

Advanced linear algebra forms the computational foundation of many fields.

### Machine learning

Examples include:

- linear regression
- PCA
- covariance analysis
- embeddings
- optimization
- neural-network transformations

### Computer graphics

Matrices represent:

- translation through homogeneous coordinates
- rotation
- scaling
- camera transformations
- projection
- coordinate-system changes

### Computer vision

Linear algebra supports:

- image transformations
- camera calibration
- geometric reconstruction
- dimensionality reduction
- least-squares estimation

### Robotics

Robotics uses matrices and vectors for:

- robot configurations
- coordinate transformations
- kinematics
- optimization
- state estimation

### Graph analytics

Graph Laplacians support:

- spectral clustering
- network analysis
- diffusion
- consensus models
- community detection

### Scientific computing

Linear systems arise from discretized models of:

- heat transfer
- fluid flow
- structural mechanics
- electromagnetics
- differential equations

### Statistics

Covariance matrices, least squares, quadratic forms, eigenvalues, and decompositions are fundamental statistical tools.

## Implementation comparison

| Concept | Python | JavaScript | C++ |
|---|---|---|---|
| Vector operations | Implemented | Implemented | Implemented |
| Matrix abstraction | `Matrix` class | `Matrix` class | `Matrix` class |
| RREF | Implemented | Implemented | Implemented |
| Rank | Implemented | Implemented | Implemented |
| Determinant | Implemented | Implemented | Implemented |
| Inverse | Implemented | Implemented | Implemented |
| Linear-system solving | Implemented | Implemented | Implemented |
| Gram-Schmidt | Implemented | Implemented | Implemented |
| QR | Implemented | Implemented | Implemented |
| Least squares | Implemented | Implemented | Implemented |
| Eigenvalue iteration | Implemented | Implemented | Implemented |
| SVD | Educational implementation | Not fully implemented | Not required for case study |
| PCA | Full educational workflow | Covariance stage | Covariance stage |
| Graph Laplacian | Implemented | Implemented | Implemented |
| Conjugate Gradient | Implemented | Implemented | Implemented |
| Failure validation | Implemented | Implemented | Implemented |
| Automated checks | Assertions | Test function | Assertions |

## Why the three languages differ

Python is particularly effective for educational numerical experiments because its syntax makes mathematical algorithms relatively compact.

JavaScript is useful when numerical algorithms must be integrated into browser applications, interactive tools, visualization systems, or JavaScript-based application runtimes.

C++ provides explicit control over data representation, memory, execution characteristics, and system architecture. This makes it useful for high-performance numerical systems and technical software where computational cost is important.

The underlying mathematics remains the same even though the implementation mechanisms differ.

## Verification principles

The implementations include several forms of verification.

Matrix inversion is checked through multiplication by the original matrix.

Linear-system solutions can be compared by substituting them back into the original equations.

QR decomposition is verified by reconstructing

    QR

and comparing it conceptually with the original matrix.

Orthogonality is tested through inner products.

Conjugate Gradient results are compared with direct solutions in the C++ case study.

These checks demonstrate an important principle of numerical software: an implementation should verify mathematical invariants where practical rather than assuming that a calculation succeeded merely because it produced numbers.

## Conceptual relationships

Several major concepts in the project are directly connected.

A linear system leads to row reduction.

Row reduction reveals pivots.

Pivots determine rank.

Rank determines whether columns are independent.

Independence determines whether a basis can be formed.

Orthogonality provides a particularly useful type of basis.

Orthogonal bases lead naturally to projections and QR decomposition.

Least squares is a projection problem.

Covariance matrices are symmetric.

Symmetric matrices have orthogonal eigenvectors.

PCA uses covariance eigenvectors.

SVD generalizes spectral analysis to rectangular matrices.

Singular values reveal numerical rank and conditioning.

Positive-definite matrices lead to stable quadratic forms and enable Conjugate Gradient methods.

Graph Laplacians are structured matrices whose spectral properties reveal graph structure.

These relationships demonstrate why advanced linear algebra is best understood as an interconnected system of ideas rather than as a collection of unrelated formulas.
