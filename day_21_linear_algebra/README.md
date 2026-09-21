# Linear algebra: Scalars, vectors, matrices, tensors, and matrix operations

## Topic introduction

Linear algebra is the branch of mathematics concerned with quantities that can be represented as scalars, vectors, matrices, and higher-dimensional arrays called tensors. It provides a common language for describing geometric transformations, systems of equations, optimization, statistics, computer graphics, machine learning, scientific computing, engineering, finance, signal processing, and many other computational fields.

The central idea is that numerical objects can be represented structurally and manipulated according to well-defined algebraic rules.

A scalar is a single number.

A vector is an ordered collection of numbers.

A matrix is a rectangular arrangement of numbers.

A tensor is a general multidimensional numerical structure. In common computational terminology, a scalar can be viewed as a rank-0 tensor, a vector as a rank-1 tensor, a matrix as a rank-2 tensor, and a higher-order tensor as an object with three or more dimensions.

The three implementations in this repository approach these ideas from different perspectives:

- Python provides a broad educational implementation with explicit algorithms and demonstrations.
- JavaScript shows how linear algebra can operate inside general application code, including asynchronous processing and object-oriented data structures.
- C++ develops a more structured industry-style case study with reusable classes, explicit validation, memory-oriented tensor storage, and a numerical scoring pipeline.

---

## Fundamental terminology

### Scalar

A scalar represents one numerical quantity.

Examples include:

- `5`
- `-3.2`
- `0`
- `1000.75`

A scalar has no vector dimension or matrix row and column structure.

Scalars are frequently used to:

- represent measurements,
- scale vectors,
- scale matrices,
- represent coefficients,
- represent probabilities,
- represent physical quantities.

If a vector is

`v = [2, 4, 6]`

then multiplying it by the scalar `3` produces

`3v = [6, 12, 18]`.

The Python implementation demonstrates scalar arithmetic and scalar multiplication of vectors and matrices. The JavaScript and C++ implementations use the same idea in executable numerical operations.

---

## Vectors

A vector is an ordered collection of numerical components.

For example:

`v = [2, -1, 3]`

is a three-dimensional vector.

The number of components is its dimension.

A two-component vector belongs to a two-dimensional vector space such as `R²`, while a three-component vector can belong to `R³`.

Vectors can represent:

- coordinates,
- velocities,
- forces,
- feature values,
- colors,
- financial measurements,
- embeddings,
- sensor readings,
- states in a computational system.

### Vector addition

Two vectors can be added only when they have the same dimension.

For

`a = [1, 2, 3]`

and

`b = [4, 5, 6]`

the result is

`a + b = [5, 7, 9]`.

The operation is component-wise.

### Vector subtraction

Similarly,

`a - b`

is calculated component by component.

The Python, JavaScript, and C++ implementations explicitly validate dimensions before performing these operations.

### Scalar multiplication

For scalar `k` and vector `v`:

`kv = [kv₁, kv₂, ..., kvₙ]`.

For example:

`2[1, 3, -2] = [2, 6, -4]`.

### Dot product

The dot product of two equal-dimensional vectors is a scalar.

For

`a = [a₁, a₂, ..., aₙ]`

and

`b = [b₁, b₂, ..., bₙ]`,

the dot product is

`a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`.

For example:

`[1, 2, 3] · [4, 5, 6] = 1(4) + 2(5) + 3(6) = 32`.

The dot product is fundamental because it connects vector algebra with geometry and matrix multiplication.

### Orthogonality

Two vectors are orthogonal when their dot product is zero.

For example:

`[1, 0] · [0, 1] = 0`.

Orthogonal vectors form an important foundation for coordinate systems, projections, least-squares methods, signal processing, and numerical algorithms.

### Vector norm

A norm measures the size or length of a vector.

The Euclidean or L2 norm is:

`||v||₂ = sqrt(v₁² + v₂² + ... + vₙ²)`.

For:

`v = [3, 4]`

the Euclidean norm is:

`||v||₂ = 5`.

The Python implementation also demonstrates the L1 norm.

### Distance

The Euclidean distance between vectors `a` and `b` is:

`||a - b||₂`.

This is important in classification, clustering, nearest-neighbor methods, geometry, and similarity calculations.

### Angle between vectors

For nonzero vectors:

`cos(θ) = (a · b) / (||a|| ||b||)`.

The implementations clamp the calculated cosine into the interval `[-1, 1]` before applying the inverse cosine. This protects the calculation against small floating-point errors.

---

## Matrices

A matrix is a rectangular collection of numbers organized into rows and columns.

For example:

`A = [[1, 2, 3], [4, 5, 6]]`

has:

- 2 rows,
- 3 columns,
- shape `2 × 3`.

The matrix shape is essential because many matrix operations are valid only for particular dimension combinations.

### Matrix addition

Two matrices can be added only if they have identical shapes.

For matrices `A` and `B`:

`(A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ`.

### Matrix subtraction

Matrix subtraction is also element-wise:

`(A - B)ᵢⱼ = Aᵢⱼ - Bᵢⱼ`.

### Scalar multiplication

A scalar multiplies every element of a matrix.

For scalar `k`:

`kA`.

If:

`A = [[1, 2], [3, 4]]`

then:

`2A = [[2, 4], [6, 8]]`.

---

## Matrix multiplication

Matrix multiplication is different from element-wise multiplication.

Suppose:

`A` has shape `m × n`

and:

`B` has shape `n × p`.

Then:

`AB`

has shape:

`m × p`.

The inner dimensions must match.

For example:

`A` with shape `2 × 3`

can be multiplied by:

`B` with shape `3 × 2`.

The result has shape `2 × 2`.

Each result element is the dot product of one row of `A` and one column of `B`.

For example:

`Cᵢⱼ = Σ AᵢₖBₖⱼ`.

This is one of the most important rules in introductory linear algebra.

### Why dimension checking matters

Attempting to multiply a `2 × 3` matrix by a `2 × 2` matrix is invalid because the inner dimensions `3` and `2` do not match.

All three implementations explicitly reject incompatible matrix dimensions.

---

## Matrix-vector multiplication

A matrix can transform a vector.

If `A` has shape `m × n` and `x` has dimension `n`, then:

`Ax`

produces a vector with dimension `m`.

This operation is central to:

- linear transformations,
- systems of equations,
- machine learning,
- computer graphics,
- signal processing,
- optimization,
- scientific simulations.

The implementations contain reusable matrix-vector multiplication functions.

---

## Transpose

The transpose of a matrix exchanges rows and columns.

If:

`A = [[1, 2, 3], [4, 5, 6]]`

then:

`Aᵀ = [[1, 4], [2, 5], [3, 6]]`.

A matrix with shape `m × n` has a transpose with shape `n × m`.

The transpose is used extensively in:

- dot-product calculations,
- least-squares methods,
- covariance matrices,
- optimization,
- orthogonal transformations,
- machine-learning formulas.

An important identity is:

`(AB)ᵀ = BᵀAᵀ`.

The order reverses.

---

## Identity matrix

The identity matrix behaves like the number `1` in ordinary multiplication.

For a three-dimensional identity matrix:

`I₃ = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]`.

For compatible matrix `A`:

`AI = IA = A`.

Identity matrices are essential when defining matrix inverses, transformations, and many algebraic algorithms.

---

## Diagonal matrices

A diagonal matrix has nonzero values only on its main diagonal.

For example:

`D = [[2, 0, 0], [0, 5, 0], [0, 0, 10]]`.

Diagonal matrices are computationally useful because multiplication by them can often be performed more efficiently than general dense matrix multiplication.

---

## Determinant

The determinant is a scalar associated with a square matrix.

For a two-by-two matrix:

`A = [[a, b], [c, d]]`

the determinant is:

`det(A) = ad - bc`.

The determinant provides important structural information.

A matrix with:

`det(A) = 0`

is singular.

A singular square matrix does not have an ordinary inverse.

The determinant is also related to the scaling of oriented volume under a linear transformation.

The Python, JavaScript, and C++ implementations calculate determinants through elimination rather than recursively expanding minors. This gives an algorithmic complexity of approximately `O(n³)` for dense `n × n` matrices.

---

## Matrix inverse

The inverse of a square matrix `A` is written:

`A⁻¹`.

When the inverse exists:

`AA⁻¹ = A⁻¹A = I`.

The inverse exists only for nonsingular square matrices.

The implementations calculate the inverse using an augmented matrix:

`[A | I]`

and row reduction:

`[A | I] → [I | A⁻¹]`.

This is useful for learning the mechanism, although production numerical software commonly solves systems directly rather than explicitly calculating an inverse when an inverse is not required.

---

## Linear systems

A system of linear equations can be represented as:

`Ax = b`.

For example:

`2x + y = 7`

`x - y = 1`.

The coefficient matrix is:

`A = [[2, 1], [1, -1]]`.

The unknown vector is:

`x = [x, y]`.

The constants form:

`b = [7, 1]`.

Gaussian elimination transforms the augmented matrix into a form from which the solution can be determined.

The Python and C++ implementations include complete linear-system solvers for square systems with unique solutions.

---

## Gaussian elimination

Gaussian elimination repeatedly performs elementary row operations.

The fundamental row operations are:

1. Swap two rows.
2. Multiply a row by a nonzero scalar.
3. Add a multiple of one row to another row.

These operations preserve the solution set of a linear system when applied appropriately to an augmented system.

### Partial pivoting

The implementations use partial pivoting.

Instead of blindly selecting the current row as the pivot, the algorithm searches the remaining rows for a large-magnitude value in the current column and swaps that row into the pivot position.

This improves numerical robustness compared with naive elimination.

It does not eliminate every possible numerical problem, but it is an important practical improvement.

---

## Reduced row-echelon form

Reduced row-echelon form, or RREF, is a canonical form produced by systematic row operations.

The implementation uses RREF for:

- solving systems,
- calculating rank,
- finding inverses.

A matrix in RREF has pivot columns normalized and eliminated above and below their pivots.

---

## Rank

The rank of a matrix is the number of linearly independent rows or, equivalently, the number of linearly independent columns.

It can also be identified as the number of pivots in an appropriate row-reduced form.

For:

`A = [[1, 2], [2, 4]]`

the second row is twice the first, so the rows are not independent.

The rank is therefore `1`.

Rank is useful for understanding:

- whether a system has enough independent information,
- whether vectors are linearly independent,
- whether a matrix is singular,
- dimensionality,
- data redundancy.

---

## Linear independence

Vectors are linearly independent if no vector in the collection can be constructed as a linear combination of the others.

For example:

`[1, 0]`

and:

`[0, 1]`

are linearly independent.

But:

`[1, 2]`

and:

`[2, 4]`

are dependent because the second vector is twice the first.

The Python implementation converts vectors into matrix columns and compares the matrix rank with the number of vectors.

---

## Span and basis

The span of a collection of vectors is the set of all linear combinations of those vectors.

The vectors:

`[1, 0]`

and:

`[0, 1]`

span `R²`.

A basis is a linearly independent set of vectors that spans the relevant vector space.

The standard basis of `R³` is:

`[1, 0, 0]`

`[0, 1, 0]`

`[0, 0, 1]`.

Basis selection is important because the same vector can have different coordinate representations under different bases.

---

## Matrices as transformations

A matrix is not merely a table of numbers. It can represent a transformation.

For example, the matrix:

`[[-1, 0], [0, 1]]`

reflects a two-dimensional vector across the y-axis.

A scaling matrix such as:

`[[2, 0], [0, 3]]`

doubles the x-coordinate and triples the y-coordinate.

A two-dimensional rotation can also be represented by a matrix.

For an angle `θ`, a standard rotation matrix is:

`[[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]`.

Matrix multiplication allows transformations to be composed.

If transformation `A` is applied first and transformation `B` is applied second, the combined transformation is:

`BA`.

The order matters because matrix multiplication is generally not commutative.

In general:

`AB ≠ BA`.

---

## Eigenvalues and eigenvectors

An eigenvector of matrix `A` is a nonzero vector `v` satisfying:

`Av = λv`.

The scalar `λ` is the corresponding eigenvalue.

The defining characteristic is that applying the matrix does not move the eigenvector to a completely different direction. Instead, the vector remains on its original line while its magnitude and possibly orientation change according to the eigenvalue.

For a two-by-two matrix, the characteristic polynomial can be constructed using the trace and determinant:

`λ² - trace(A)λ + det(A) = 0`.

The Python implementation contains a focused two-by-two real-eigenvalue calculation.

The implementation intentionally rejects matrices whose eigenvalues are complex because the demonstration is designed around real-valued arithmetic.

---

## Tensors

A tensor generalizes the idea of scalar, vector, and matrix structures.

A common computational classification is:

- rank 0: scalar,
- rank 1: vector,
- rank 2: matrix,
- rank 3: three-dimensional tensor,
- rank 4 and above: higher-order tensors.

The term rank can be used differently in different mathematical contexts, so tensor rank should not be confused with matrix rank.

### Shape

The shape describes the size of every tensor dimension.

For example:

`[[1, 2], [3, 4]]`

has shape:

`(2, 2)`.

A three-dimensional structure such as two matrices each having two rows and two columns has shape:

`(2, 2, 2)`.

### Tensor indexing

For a three-dimensional tensor, an element can be addressed by three coordinates:

`T[d][r][c]`.

The C++ implementation stores a 3D tensor in one contiguous one-dimensional array and converts multidimensional coordinates into a linear memory index.

This illustrates an important implementation distinction:

Mathematical dimensionality does not require physically separate memory blocks for each dimension.

---

## Element-wise tensor operations

Two tensors with the same shape can be added element by element.

For tensors `A` and `B`:

`Cᵢ... = Aᵢ... + Bᵢ...`.

The Python and JavaScript tensor classes demonstrate this concept.

Scalar multiplication also operates element-wise:

`Cᵢ... = kAᵢ...`.

More advanced tensor systems commonly support operations such as:

- reshaping,
- slicing,
- broadcasting,
- contraction,
- permutation,
- reduction,
- matrix multiplication across selected dimensions.

The educational implementations intentionally focus on the fundamental structure rather than attempting to recreate a full scientific tensor library.

---

## Broadcasting

Broadcasting is a mechanism used by many numerical systems in which arrays with compatible but different shapes can participate in element-wise operations.

For example, a row vector can sometimes be added to every row of a matrix.

Broadcasting rules vary between libraries, so shape compatibility must be understood rather than assumed.

The implementations in this repository deliberately require exact tensor shape equality for their element-wise tensor addition. This keeps the behavior explicit and avoids silently introducing library-specific broadcasting semantics.

---

## Python implementation

The Python program is the broadest educational implementation.

It contains explicit functions for:

- scalar operations,
- vector addition,
- vector subtraction,
- scalar-vector multiplication,
- dot products,
- norms,
- distances,
- vector angles,
- vector projection,
- matrix validation,
- matrix addition,
- matrix subtraction,
- scalar-matrix multiplication,
- matrix multiplication,
- transpose,
- identity matrices,
- diagonal matrices,
- determinants,
- RREF,
- rank,
- linear-system solving,
- matrix inversion,
- linear independence,
- matrix-vector transformations,
- eigenvalue calculation,
- tensor storage,
- tensor addition,
- tensor scaling,
- a small machine-learning-style matrix pipeline.

The implementation avoids external numerical packages so that the underlying algorithms remain visible.

This is useful educationally because a high-level numerical library can make linear algebra extremely concise while hiding the algorithms that actually perform the work.

### Python matrix multiplication

The Python implementation computes each result element as a row-column dot product.

For matrices with shapes `m × n` and `n × p`, the result has shape `m × p`.

The computational complexity for dense square matrices is approximately `O(n³)`.

### Python Gaussian elimination

The `rref` implementation uses partial pivoting and a configurable numerical tolerance.

The tolerance is necessary because floating-point values are approximations.

For example, a theoretically zero value may appear as a very small number such as `1e-14` after several arithmetic operations.

### Python tensor class

The `Tensor` class recursively infers the shape of nested lists.

This allows the same basic structure to represent:

- scalars,
- vectors,
- matrices,
- higher-order arrays.

The class verifies that nested dimensions are rectangular.

---

## JavaScript implementation

The JavaScript implementation demonstrates the same mathematical structures in a general application-oriented environment.

It contains:

- scalar examples,
- vector operations,
- norms,
- distances,
- angles,
- projections,
- matrix addition,
- matrix subtraction,
- scalar multiplication,
- transpose,
- matrix multiplication,
- matrix-vector multiplication,
- determinants,
- RREF,
- matrix inverse,
- asynchronous feature loading,
- a dense matrix computation,
- a tensor class,
- validation,
- error handling,
- a simple performance benchmark.

### JavaScript-specific considerations

JavaScript's ordinary `Number` type is a floating-point numeric representation.

This means numerical comparisons should generally use tolerances when exact mathematical equality is not guaranteed.

The implementation uses `nearlyEqual` for such cases.

The JavaScript file also demonstrates asynchronous execution through a Promise and `async`/`await`.

The mathematical operation itself remains synchronous, but real applications frequently obtain numerical data from:

- network requests,
- databases,
- files,
- browser APIs,
- user input,
- other asynchronous services.

The example shows how a matrix calculation can fit into that type of application flow.

---

## C++ case study

The C++ implementation is organized as an industry-style numerical case study.

It defines:

- a vector utility class,
- a matrix class,
- a three-dimensional tensor storage class,
- a linear-system solver,
- a risk-scoring engine.

### Problem being modeled

The case study models a numerical scoring pipeline.

A raw feature vector is transformed through a matrix:

`z = Wx`.

A bias is added:

`a = Wx + b`.

A final weight vector is used to produce a scalar:

`score = wᵀa`.

This pattern is mathematically simple but representative of a much larger class of numerical systems.

### Matrix transformation

The `RiskScoringEngine` stores a transformation matrix, bias vector, and calibration weights.

The engine validates dimensions when it is constructed.

This is important because dimension errors are structural errors that should normally be caught as early as possible.

### C++ data structures

The matrix is represented by:

`std::vector<std::vector<double>>`.

The tensor uses:

`std::vector<double>`.

The tensor representation is particularly useful for illustrating contiguous storage.

A 3D coordinate `(d, r, c)` is converted into a one-dimensional index.

The formula is:

`d * rows * columns + r * columns + c`.

This is similar to the row-major layout used by many computational systems.

### C++ matrix multiplication

The C++ matrix multiplication uses the loop ordering `i-k-j`.

This allows a value from the left matrix to be reused across several operations before moving to another left-hand element.

Memory access patterns matter for performance because processors operate with caches and memory hierarchies.

For production numerical workloads, specialized libraries usually outperform simple educational implementations because they can use optimized kernels, vector instructions, multithreading, cache-aware algorithms, and specialized hardware.

---

## Important distinctions

### Scalar versus vector

A scalar is one number.

A vector is an ordered collection of numbers.

A scalar does not have multiple components, while a vector has a defined dimension.

### Vector versus matrix

A vector has one dimension of indexing.

A matrix has two dimensions of indexing.

A vector can be represented computationally as a one-dimensional array, while a matrix is normally represented as rows and columns.

### Matrix versus tensor

A matrix is a rank-2 numerical structure.

A tensor is a more general multidimensional structure.

Every matrix can be viewed as a particular tensor, but not every tensor is a matrix.

### Matrix multiplication versus element-wise multiplication

Matrix multiplication follows row-column dot-product rules and has strict dimension requirements.

Element-wise multiplication multiplies corresponding entries.

These are different operations.

Confusing them is one of the most common beginner mistakes.

### Transpose versus inverse

The transpose changes rows into columns.

The inverse is a matrix that reverses multiplication.

They are mathematically different operations.

For certain special matrices, such as orthogonal matrices:

`A⁻¹ = Aᵀ`.

That equality is a special property, not a general rule.

---

## Edge cases

### Zero vector

The norm of the zero vector is valid:

`||0|| = 0`.

The angle between the zero vector and another vector is not defined because the formula divides by the product of the norms.

The implementations explicitly reject this case.

### Singular matrix

A singular matrix has determinant zero and no ordinary inverse.

The implementations detect singular matrices when attempting inversion.

### Dimension mismatch

Vector addition requires equal dimensions.

Matrix addition requires identical shapes.

Matrix multiplication requires matching inner dimensions.

Matrix-vector multiplication requires the vector dimension to equal the matrix column count.

### Empty structures

The implementations reject empty vectors, matrices, and tensor dimensions.

This makes the educational APIs explicit and prevents ambiguous shapes.

### Floating-point equality

A calculation that should mathematically produce zero may produce a tiny value such as:

`0.00000000001`.

For this reason, numerical software commonly uses a tolerance rather than checking only exact equality.

---

## Common mistakes

### Treating matrix multiplication as element-wise multiplication

Given two matrices, multiplying corresponding cells is not ordinary matrix multiplication.

Always check the operation being requested.

### Ignoring matrix dimensions

Before multiplying:

`A × B`

check that:

`columns(A) = rows(B)`.

### Assuming multiplication is commutative

For scalars:

`ab = ba`.

For matrices, generally:

`AB ≠ BA`.

### Dividing by a vector

A vector does not have a normal scalar-style division operation.

Many operations that appear similar to division require a different mathematical definition.

### Inverting a singular matrix

A matrix with determinant zero does not have an ordinary inverse.

### Using exact floating-point comparisons

Numerical computations accumulate rounding errors.

Use a tolerance when appropriate.

### Explicitly calculating an inverse unnecessarily

In numerical applications, if the goal is to solve:

`Ax = b`,

directly solving the system is usually preferable to calculating `A⁻¹` and then multiplying by `b`.

The inverse is useful when the inverse itself is required, but it is not automatically the best computational route for solving every system.

---

## Limitations of the educational implementations

These implementations are designed to expose mathematical mechanisms.

They are not replacements for specialized numerical libraries.

The main limitations include:

- dense storage only,
- no sparse matrix representation,
- no optimized BLAS implementation,
- no advanced decomposition framework,
- no automatic differentiation,
- limited eigenvalue support,
- no complex-number tensor implementation,
- limited tensor broadcasting,
- no GPU acceleration,
- no advanced numerical conditioning analysis,
- no parallel matrix kernels.

The Python and JavaScript implementations intentionally use basic language constructs so that the algorithms remain understandable.

The C++ implementation adds more explicit data-structure and memory considerations but still remains an educational implementation.

---

## Performance considerations

For a vector with `n` components:

- vector addition is `O(n)`,
- dot product is `O(n)`,
- norm calculation is `O(n)`.

For an `m × n` matrix:

- matrix addition is `O(mn)`,
- matrix-vector multiplication is `O(mn)`.

For dense `n × n` matrices:

- standard matrix multiplication is `O(n³)`,
- Gaussian elimination is approximately `O(n³)`,
- elimination-based inversion is approximately `O(n³)`.

Tensor element-wise operations are generally proportional to the number of tensor elements.

Memory complexity is also important.

An `n × n` dense matrix requires `O(n²)` storage.

A large dense matrix can therefore become memory-intensive even before its arithmetic operations become expensive.

For large workloads, optimized numerical libraries commonly improve performance through:

- cache-aware memory layouts,
- SIMD/vector instructions,
- multithreading,
- optimized matrix kernels,
- blocking and tiling,
- GPU acceleration,
- sparse representations where applicable.

---

## Numerical stability

Mathematically equivalent algorithms can have very different numerical behavior.

Important issues include:

- floating-point rounding,
- cancellation,
- overflow,
- underflow,
- poorly conditioned matrices,
- nearly singular systems,
- accumulation of numerical error.

Partial pivoting, used by the implementations, is one technique for improving the practical behavior of Gaussian elimination.

For serious numerical workloads, specialized decomposition algorithms such as QR or singular value decomposition may be more appropriate depending on the problem.

---

## Security and reliability considerations

Linear algebra code can become part of systems that process sensitive or security-relevant data.

Important engineering practices include:

- validate input dimensions,
- reject malformed numerical values,
- detect non-finite values when appropriate,
- avoid uncontrolled memory allocation,
- handle singular systems explicitly,
- use appropriate numerical tolerances,
- avoid silently accepting invalid shapes,
- test boundary conditions,
- avoid trusting externally supplied matrix dimensions,
- monitor numerical ranges for overflow or underflow.

The C++ implementation throws exceptions for invalid dimensions, out-of-range tensor indexing, singular inverses, and inconsistent systems.

In production systems, these failures should be integrated into the application's error-handling and observability strategy.

---

## Practical applications

Linear algebra is foundational to many technical systems.

### Computer graphics

Vectors represent positions, directions, normals, and colors.

Matrices represent:

- rotation,
- scaling,
- translation through homogeneous coordinates,
- camera transformations,
- projection.

### Machine learning

Vectors represent observations and feature representations.

Matrices represent:

- model parameters,
- transformations,
- batches of observations,
- covariance structures.

Dense neural-network layers frequently contain operations of the form:

`Wx + b`.

### Computer vision

Images can be represented as matrices or higher-order tensors.

Color images commonly contain multiple channels, producing multidimensional structures.

Matrix transformations are used for:

- image geometry,
- coordinate transformations,
- feature extraction,
- camera models.

### Data science

Matrices naturally represent tabular numerical data.

Rows can represent observations while columns represent features.

Linear algebra supports:

- covariance,
- dimensionality reduction,
- regression,
- least squares,
- principal component analysis.

### Engineering

Vectors and matrices represent:

- forces,
- states,
- measurements,
- system equations,
- transformations,
- physical models.

### Finance

Matrices can represent:

- covariance structures,
- factor exposures,
- portfolio relationships,
- optimization constraints,
- scenario data.

### Scientific computing

Numerical simulations often reduce continuous models to large systems of equations represented by matrices and vectors.

---

## Relationship between the three implementations

The three files are mathematically aligned but intentionally emphasize different programming concerns.

### Python

The Python program emphasizes algorithm visibility and conceptual breadth.

It is particularly useful for examining:

- the formulas,
- row reduction,
- tensor shape inference,
- matrix transformations,
- mathematical relationships.

### JavaScript

The JavaScript program emphasizes integration with application programming.

It demonstrates:

- object-oriented tensor representation,
- asynchronous data acquisition,
- numerical validation,
- matrix processing inside application logic,
- runtime performance measurement.

### C++

The C++ program emphasizes structured implementation and system-level considerations.

It demonstrates:

- reusable classes,
- explicit type structure,
- exception-based validation,
- contiguous tensor storage,
- cache-aware loop ordering,
- numerical case-study architecture,
- an industry-style scoring pipeline.

---

## Conceptual progression

The topic can be understood as a sequence of increasingly structured numerical objects.

A scalar contains one value.

A vector contains multiple ordered values.

A matrix organizes values into two dimensions and can represent transformations.

A tensor generalizes the concept to arbitrary dimensions.

Operations then build on these structures:

`scalar`

→ scalar-vector scaling

→ vector addition and dot products

→ matrix-vector multiplication

→ matrix-matrix multiplication

→ transformations

→ systems of equations

→ determinants and inverses

→ rank and linear independence

→ higher-dimensional tensor operations.

This progression explains why linear algebra is so widely reusable across computing disciplines: many apparently different technical problems reduce to combinations of the same numerical structures and operations.
