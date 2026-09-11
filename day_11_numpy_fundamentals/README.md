# NumPy fundamentals

## Introduction

NumPy is a foundational Python library for numerical computing. Its central data structure is the `ndarray`, a multidimensional array designed for efficient storage and numerical operations.

This study script develops NumPy fundamentals from basic array creation to multidimensional indexing, slicing, reshaping, Boolean filtering, vectorized computation, and broadcasting. It also introduces related concepts that are important when NumPy is used for scientific computing, statistics, data analysis, simulations, and numerical applications.

The central idea is that NumPy operations are generally expressed in terms of whole arrays rather than individual Python values. This provides concise numerical expressions and often allows computation to be performed efficiently in optimized compiled implementations.

## NumPy arrays

A NumPy array is an `ndarray`. Unlike a normal Python list, a conventional NumPy numerical array is homogeneous, meaning its elements generally share one data type.

For example, an array can contain 64-bit floating-point numbers:

    np.array([1.0, 2.0, 3.0], dtype=np.float64)

The homogeneous representation allows NumPy to store numerical values compactly and perform operations efficiently.

An array has several important properties:

- `ndim` gives the number of dimensions, or axes.
- `shape` gives the length of every axis.
- `size` gives the total number of elements.
- `dtype` specifies the element data type.
- `itemsize` gives the number of bytes used by one element.
- `nbytes` gives the total memory occupied by the array data.

These properties are among the first things to inspect when debugging numerical code.

## Dimensions and axes

The number of dimensions is represented by `ndim`.

A scalar-like array has zero dimensions:

    np.array(42)

A one-dimensional array has one axis:

    np.array([10, 20, 30])

A two-dimensional array has two axes:

    np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

A three-dimensional array contains three axes and can be represented using a shape such as `(2, 3, 4)`.

For an array with shape `(2, 3, 4)`:

- there are 3 dimensions,
- the first axis has length 2,
- the second axis has length 3,
- the third axis has length 4,
- the total number of elements is 24.

The word "axis" is important because many NumPy operations accept an `axis` argument.

For a two-dimensional array:

- `axis=0` refers to the first axis and normally corresponds to movement down rows.
- `axis=1` refers to the second axis and normally corresponds to movement across columns.

For example, summing a matrix with `axis=0` produces one result per column. Summing with `axis=1` produces one result per row.

## Shape

Shape describes the size of an array along every axis.

For example:

    array.shape == (3, 4)

means the array has three rows and four columns when interpreted as a matrix.

The number of elements is the product of the dimensions:

    3 * 4 = 12

Shape is not the same as size.

An array can have:

    shape = (2, 3)
    size = 6

Shape describes structure, while size describes total element count.

Shape is central to NumPy because many operations depend on whether two arrays have compatible dimensions.

## Creating arrays

The script demonstrates several ways to construct arrays.

### `np.array`

A Python list can be converted into an array:

    np.array([10, 20, 30])

Nested lists can create multidimensional arrays:

    np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

For multidimensional construction, the nested structure should normally be rectangular when a regular multidimensional array is intended.

### `np.zeros`

Creates an array filled with zeros.

    np.zeros((2, 3))

The resulting shape is `(2, 3)`.

### `np.ones`

Creates an array filled with ones.

    np.ones((2, 3))

### `np.full`

Creates an array filled with a specified value.

    np.full((2, 3), 7)

### `np.empty`

Creates an array without initializing its values.

    np.empty((2, 3))

The contents of an `empty` array should not be assumed to have any particular value. It is useful when memory needs to be allocated before values are filled explicitly.

### `np.eye`

Creates an identity-like two-dimensional array.

    np.eye(3)

The main diagonal contains ones and the remaining elements contain zeros.

### `np.arange`

`np.arange` produces regularly spaced values using a start, stop, and step.

    np.arange(0, 10, 2)

The stop value is normally excluded.

### `np.linspace`

`np.linspace` creates a specified number of evenly spaced values between two endpoints.

    np.linspace(0, 1, 6)

This is useful when the desired number of samples is more important than the step size.

## Random arrays

Modern NumPy random generation uses random-number generator objects such as those returned by `np.random.default_rng`.

The script uses:

    rng = np.random.default_rng(42)

A fixed seed makes numerical demonstrations reproducible.

Examples include:

- `rng.random` for uniform floating-point values,
- `rng.integers` for discrete integer sampling,
- `rng.normal` for normally distributed values,
- `rng.uniform` for uniformly distributed values.

A numerical random generator should not be confused with a cryptographic random source. NumPy's random facilities are designed primarily for simulation and numerical work.

## Data types

NumPy arrays have an explicit `dtype`.

Common numerical types include:

- `int32`
- `int64`
- `float32`
- `float64`
- `complex128`
- `bool`

The choice of dtype affects:

- memory consumption,
- numerical precision,
- representable range,
- arithmetic behavior,
- interoperability with other systems,
- performance.

For example, `float32` uses less memory than `float64` but provides less precision.

### Integer overflow

Fixed-width integer types have finite ranges.

For example, an `int8` can represent values only within its defined range. Arithmetic that exceeds that range can produce wraparound behavior rather than automatically expanding to an unlimited-size integer.

Therefore dtype selection matters when values may become large.

### Type conversion

`astype` converts an array to another dtype.

For example:

    values.astype(np.int64)

Converting floating-point values to integers does not perform ordinary mathematical rounding. The fractional component is discarded toward zero.

Use an explicit rounding operation such as `np.round` when rounding is actually required.

## Indexing

NumPy uses zero-based indexing.

For a one-dimensional array:

    values[0]

selects the first element.

Negative indexing is supported:

    values[-1]

selects the final element.

For a two-dimensional array:

    matrix[row, column]

selects an individual element.

For example:

    matrix[1, 2]

selects the element in row index 1 and column index 2.

The comma-separated multidimensional syntax is clearer than repeatedly applying nested indexing and generalizes naturally to arrays with more dimensions.

## Slicing

NumPy supports Python-style slicing.

The general structure is:

    start:stop:step

The stop index is excluded.

Examples include:

    values[2:7]
    values[:5]
    values[5:]
    values[::2]
    values[::-1]

For a two-dimensional array, slicing can be applied independently to different axes:

    matrix[1:4, 1:4]

selects a rectangular region.

Other useful patterns include:

    matrix[:, 0]

for the first column, and:

    matrix[0, :]

for the first row.

## Views and copies

One of the most important NumPy memory concepts is the distinction between a view and a copy.

A basic slice commonly produces a view into the original array.

For example:

    view = original[1:4]

Changing `view` can change `original`.

This behavior is useful because a view can operate on existing memory without creating another complete data buffer.

When independent data is required, use:

    copy = original[1:4].copy()

A copy has separate storage and can be modified without changing the original array.

The script uses `np.shares_memory` to demonstrate whether two arrays share memory.

## Fancy indexing

Integer-array indexing allows multiple arbitrary positions to be selected.

For example:

    values[[0, 2, 4]]

selects elements at indexes 0, 2, and 4.

This is often called fancy indexing or advanced integer indexing.

It differs conceptually from a continuous slice because the selected positions do not have to form a contiguous range.

## Boolean masking

Boolean indexing selects elements according to a Boolean condition.

For example:

    scores[scores >= 60]

selects all scores that are at least 60.

The condition creates a Boolean mask containing `True` and `False` values.

Boolean masks are especially important in numerical analysis because they allow filtering without writing explicit Python loops.

Multiple conditions require element-wise logical operators.

Use:

    (values >= 10) & (values < 20)

for element-wise logical AND.

Use:

    (values < 10) | (values > 20)

for element-wise logical OR.

Use `~condition` for element-wise logical negation.

Parentheses around individual comparisons are important because of Python's operator precedence.

Python's scalar `and` and `or` operators should not be substituted for NumPy's element-wise `&` and `|`.

## Conditional replacement

Boolean indexing can also modify selected elements.

For example:

    values[values < 50] = 50

replaces every value below 50 with 50.

This is useful for thresholding, data cleaning, validation, and constrained numerical transformations.

## `np.where`

`np.where` provides vectorized conditional selection.

For example:

    np.where(scores >= 60, "Pass", "Fail")

produces one result for every element.

It can also be used to calculate different numerical values depending on a condition.

## `np.select`

When more than two categories are required, `np.select` can express multiple vectorized conditions.

The script uses it to select different simplified rates based on income thresholds.

This approach is useful for rule-based numerical transformations.

## Arithmetic

NumPy arithmetic on arrays is normally element-wise.

If:

    a = np.array([1, 2, 3])
    b = np.array([10, 20, 30])

then:

    a + b

produces:

    [11, 22, 33]

Similarly:

    a - b
    a * b
    a / b
    a ** 2

operate element by element.

This differs from Python list addition, where `+` concatenates lists.

## Scalar broadcasting

A scalar can be applied to every element of an array:

    array + 10

NumPy treats the scalar as compatible with every array position.

This is the simplest example of broadcasting.

## Element-wise multiplication versus matrix multiplication

This distinction is fundamental.

The `*` operator performs element-wise multiplication.

The `@` operator performs matrix multiplication when applied to compatible two-dimensional arrays.

For matrices `A` and `B`:

    A * B

multiplies corresponding elements.

    A @ B

performs the mathematical matrix product.

Confusing these two operations is a common source of numerical errors.

## Universal functions

NumPy provides vectorized mathematical functions called universal functions, or ufuncs.

The script demonstrates:

- `np.sqrt`
- `np.exp`
- `np.log`
- `np.log10`
- `np.sin`
- `np.cos`
- `np.tan`
- `np.abs`
- `np.floor`
- `np.ceil`

These functions operate element-wise on arrays.

For example:

    np.sqrt([1, 4, 9])

conceptually applies the square-root operation to every element.

## Trigonometric functions

NumPy's standard trigonometric functions use radians.

For degree-based data:

    np.deg2rad(degrees)

converts degrees to radians.

The reverse conversion is:

    np.rad2deg(radians)

The distinction matters because passing degrees directly into a trigonometric function when radians are expected produces incorrect results.

## Aggregations and reductions

Aggregations reduce multiple values to one or fewer values.

Important operations include:

- `np.sum`
- `np.mean`
- `np.min`
- `np.max`
- `np.median`
- `np.std`
- `np.var`

For example:

    np.mean(values)

returns the arithmetic mean.

For multidimensional arrays, `axis` controls which dimension is reduced.

For a matrix:

    matrix.sum(axis=0)

calculates column totals.

    matrix.sum(axis=1)

calculates row totals.

## `keepdims`

Reduction operations can preserve the reduced dimension using:

    keepdims=True

For example:

    matrix.sum(axis=1, keepdims=True)

produces a column-shaped result rather than removing the axis completely.

This is often useful because the resulting shape can then broadcast naturally against the original array.

## Missing values and NaN

`NaN` represents a special floating-point state commonly associated with missing or undefined numerical values.

For example:

    np.isnan(values)

identifies NaN positions.

Ordinary reductions can propagate NaN:

    np.mean(values)

NaN-aware functions can ignore NaN values where appropriate:

    np.nanmean(values)

Other useful checks include:

    np.isinf(values)
    np.isfinite(values)

These distinctions are important when numerical data may contain missing, undefined, or infinite values.

## Reshaping

`reshape` changes the shape of an array while preserving its number of elements.

For example, an array with 12 elements can be reshaped into:

    (3, 4)
    (2, 6)
    (12, 1)
    (1, 12)

because all of those shapes contain 12 elements.

A reshape operation is invalid when the requested shape requires a different number of elements.

### Inferring a dimension

One dimension can be specified as `-1`:

    values.reshape(3, -1)

NumPy calculates the missing dimension automatically.

Only one dimension should be inferred this way because NumPy needs enough information to determine the missing size.

## Flattening

The script compares `flatten` and `ravel`.

`flatten()` returns a copy.

`ravel()` returns a flattened view when possible, although a copy can be produced when necessary.

The distinction matters when memory consumption and mutation behavior are important.

## Transpose

For a two-dimensional array:

    matrix.T

swaps rows and columns.

A shape of:

    (2, 3)

becomes:

    (3, 2)

For higher-dimensional arrays, transpose and axis-reordering operations become more complex because there are multiple axes whose ordering can be changed.

The script also demonstrates `swapaxes` and `moveaxis`.

## Broadcasting

Broadcasting allows NumPy to perform element-wise operations on arrays with different but compatible shapes.

NumPy compares dimensions from the rightmost side.

Two dimensions are compatible when:

1. They are equal, or
2. One of them is 1.

Missing leading dimensions are treated as though they have length 1.

For example:

    matrix.shape == (3, 4)
    vector.shape == (4,)

are compatible.

The vector can be applied to each row.

Similarly:

    matrix.shape == (3, 4)
    column.shape == (3, 1)

are compatible.

The column can be applied across each column position.

## Broadcasting examples

Suppose:

    matrix.shape == (2, 3)
    vector.shape == (3,)

The dimensions can be viewed conceptually as:

    (2, 3)
    (1, 3)

The leading dimension of the vector is effectively expanded to match the matrix.

For:

    matrix.shape == (2, 3)
    column.shape == (2, 1)

the comparison is:

    (2, 3)
    (2, 1)

The dimension of length 1 can expand to length 3.

An array with shape `(2,)` cannot generally be added to an array with shape `(2, 3)` because comparing from the right gives dimensions 3 and 2, which are neither equal nor 1.

## `np.newaxis`

A one-dimensional array with shape `(3,)` has no explicit row or column orientation.

`np.newaxis` can introduce one:

    values[np.newaxis, :]

produces shape `(1, 3)`.

    values[:, np.newaxis]

produces shape `(3, 1)`.

This technique is particularly useful when explicit broadcasting behavior is desired.

## Outer operations

An outer operation combines every element of one vector with every element of another vector.

For vectors of lengths 3 and 4, the result has shape `(3, 4)`.

The script demonstrates `np.outer` as well as explicit broadcasting using singleton dimensions.

This pattern is useful in mathematical grids, pairwise calculations, kernels, distance calculations, and numerical models.

## Stacking

Arrays can be combined in different ways.

`np.concatenate` joins arrays along an existing axis.

`np.stack` creates a new axis.

For two one-dimensional arrays of shape `(3,)`:

    concatenate -> (6,)
    stack       -> (2, 3)

`np.vstack` and `np.hstack` provide convenient forms of vertical and horizontal combination for compatible arrays.

The correct choice depends on whether the operation is intended to extend an existing dimension or introduce a new dimension.

## Splitting

`np.split` divides an array into equal-sized sections when possible.

`np.array_split` can divide an array into sections of different sizes.

This distinction matters when the total number of elements cannot be evenly divided by the requested number of sections.

## Sorting

NumPy provides several sorting-related operations.

`np.sort` returns sorted values.

`array.sort()` sorts an array in place.

`np.argsort` returns indexes that describe the sorting order.

For example, the indexes from `argsort` can be used to reorder related arrays consistently.

## Searching and unique values

`np.searchsorted` determines insertion positions in sorted arrays.

It assumes the input is ordered according to the relevant sorting rules.

`np.unique` identifies unique values.

With:

    np.unique(values, return_counts=True)

the frequency of each unique value can also be obtained.

## Locating minimum and maximum positions

`np.argmin` returns the index of the minimum value.

`np.argmax` returns the index of the maximum value.

They return positions, not the values themselves.

For example:

    index = np.argmax(values)
    maximum = values[index]

This distinction is important when the location of an extreme value is needed.

## Clipping

`np.clip` limits values to a specified interval.

For example:

    np.clip(scores, 0, 100)

ensures that every result is between 0 and 100.

Values below the lower limit become the lower limit, and values above the upper limit become the upper limit.

Clipping is useful for enforcing numerical bounds, but it should not be used blindly. It can hide data-quality problems when extreme values should instead be investigated.

## Cumulative operations

The script demonstrates cumulative operations including:

- `np.cumsum`
- `np.cumprod`
- `np.maximum.accumulate`
- `np.minimum.accumulate`

For example:

    np.cumsum(values)

produces a running total.

Cumulative products are useful for compounding processes such as repeated growth factors.

## Differences

`np.diff` calculates differences between neighboring values.

For:

    [100, 103, 101]

the result is:

    [3, -2]

This is useful for changes, increments, and simple time-series transformations.

## Vectorization

Vectorization means expressing numerical operations using array operations rather than manually iterating over every element in Python.

For example:

    result = values * values + 2 * values + 1

is a vectorized expression.

The equivalent explicit Python loop performs the calculation element by element.

NumPy operations often provide substantial performance advantages for large numerical arrays because much of the computation is implemented outside the Python interpreter.

The exact performance depends on array size, operation complexity, dtype, memory layout, hardware, and memory bandwidth.

Vectorization is not automatically faster for every tiny operation. Measurement is important.

## Memory efficiency

NumPy arrays store homogeneous numerical data in compact memory.

For example, an `int64` array with 100,000 elements requires approximately:

    100000 * 8 bytes

for its raw element storage.

Python lists have additional per-element object and reference overhead.

For numerical workloads, NumPy's contiguous typed storage can provide substantial memory and computational advantages.

The actual memory characteristics of an application depend on dtype, array layout, views, copies, and temporary arrays.

## In-place operations

Operations such as:

    values *= 10
    values += 5

modify an existing array.

In-place operations can reduce allocations and memory traffic.

The trade-off is that the original data is changed.

They should therefore be used only when mutation is intended.

## Memory layout

NumPy arrays have memory-layout characteristics that can affect performance and interoperability.

C-contiguous arrays commonly store the last axis as the fastest-changing dimension.

Fortran-contiguous arrays use a different ordering.

The script demonstrates:

- `C_CONTIGUOUS`
- `F_CONTIGUOUS`
- `np.asfortranarray`

Memory order becomes especially relevant when arrays interact with compiled numerical libraries or native interfaces.

## `np.array` versus `np.asarray`

Both functions can convert data into NumPy arrays, but their copy behavior can differ.

`np.array` commonly creates a new array when given an existing ndarray.

`np.asarray` is designed to avoid an unnecessary copy when the input is already an appropriate ndarray.

Avoiding unnecessary copies can significantly reduce memory usage for large arrays.

## Safe division

Division by zero deserves explicit treatment when the data can contain zero denominators.

The script uses `np.divide` with `out` and `where`.

This allows the program to specify the output for positions where the denominator is zero instead of relying on uncontrolled divide-by-zero behavior.

This technique is useful in ratios, percentages, normalization, and financial calculations.

## Normalization

The script demonstrates standardization using:

    (x - mean) / standard_deviation

For a matrix where rows represent observations and columns represent features, feature means and standard deviations can be calculated using:

    mean = data.mean(axis=0)
    std = data.std(axis=0)

The resulting one-dimensional arrays broadcast across all observations.

This pattern is common in numerical data preprocessing.

## Min-max scaling

Min-max scaling uses:

    (x - minimum) / (maximum - minimum)

to map values into the interval `[0, 1]`.

The denominator becomes zero for a constant feature. A robust implementation must explicitly handle this case rather than allowing division by zero.

The script maps constant columns to zero.

## Matrix multiplication and linear algebra

NumPy includes linear-algebra functionality through `np.linalg`.

The script demonstrates:

- determinant calculation with `np.linalg.det`,
- matrix inversion with `np.linalg.inv`,
- linear-system solution with `np.linalg.solve`.

For a system:

    A x = b

`np.linalg.solve(A, b)` directly solves the system.

When the goal is to solve a linear system, explicitly calculating `A` inverse and multiplying it by `b` is generally less appropriate than using a dedicated solver.

## Floating-point precision

Floating-point numbers are approximations of real numbers.

Consequently, an expression such as:

    0.1 + 0.2

does not necessarily produce a binary floating-point representation exactly equal to the representation of `0.3`.

Direct equality checks can therefore be misleading for numerical calculations.

`np.isclose` provides tolerance-based comparison:

    np.isclose(calculated, expected)

The appropriate tolerance depends on the scale and numerical characteristics of the problem.

## Boolean ambiguity

A NumPy array containing multiple Boolean values cannot normally be interpreted as one Python Boolean.

For example, using a multi-element Boolean array directly in an `if` statement raises an error.

When a single Boolean result is required, use:

    np.any(condition)

or:

    np.all(condition)

`np.any` is true when at least one element is true.

`np.all` is true only when every element is true.

## Empty arrays

Empty arrays are valid NumPy arrays.

However, operations such as means and some other reductions may be mathematically undefined for empty input.

Production code should define how empty input is handled rather than assuming that every reduction has a meaningful result.

## Singleton dimensions

A dimension with length one is called a singleton dimension.

For example:

    shape == (1, 3)

contains a singleton first dimension.

`np.squeeze` removes singleton dimensions.

`np.expand_dims` adds one.

Care is required with `squeeze` because removing dimensions can change the rank of an array in ways that later code may not expect.

## Structured arrays

Standard numerical NumPy arrays are homogeneous.

Structured arrays are a specialized feature that allow each element to contain named fields with potentially different data types.

For example, a structured dtype can contain:

- an integer identifier,
- an integer age,
- a floating-point score.

Structured arrays can be useful in specialized numerical workflows, although many applications involving complex tabular data use higher-level data structures.

## Shape validation

Numerical code should validate array structure at system boundaries.

The script includes functions that check:

- whether an input is a NumPy ndarray,
- whether it has the expected number of dimensions,
- whether matrix multiplication dimensions are compatible.

Explicit validation makes errors easier to understand and prevents shape problems from propagating deeper into a system.

## Practical applications

The techniques demonstrated in the script correspond to common numerical tasks.

### Data analysis

NumPy arrays can represent numerical observations and features. Axis-based reductions can calculate totals, means, standard deviations, and other statistics.

### Scientific computing

Multidimensional arrays can represent measurements, grids, simulations, physical quantities, and numerical models.

### Statistics

Array operations can calculate descriptive statistics and transform observations efficiently.

### Machine learning preprocessing

Broadcasting supports normalization and standardization of feature matrices.

### Finance

Arrays can represent prices, returns, portfolios, cash flows, and scenario calculations.

### Simulation

Random arrays and vectorized operations can support Monte Carlo and other numerical simulations.

### Engineering

Multidimensional numerical arrays can represent sensor measurements, matrices, grids, and engineering calculations.

### Linear algebra

NumPy supports matrix multiplication, systems of equations, matrix decomposition, and other numerical linear-algebra operations.

## Important distinctions

### Shape versus size

Shape describes the dimensions.

Size describes the total number of elements.

For shape `(2, 3)`:

    ndim = 2
    size = 6

### `*` versus `@`

`*` performs element-wise multiplication.

`@` performs matrix multiplication.

### View versus copy

A view can share memory with another array.

A copy owns independent data.

### `flatten()` versus `ravel()`

`flatten()` returns a copy.

`ravel()` returns a view when possible and may return a copy when necessary.

### `concatenate()` versus `stack()`

`concatenate` joins along an existing axis.

`stack` introduces a new axis.

### `split()` versus `array_split()`

`split` requires equal divisions for integer-based section counts.

`array_split` allows uneven divisions.

### `np.mean()` versus `np.nanmean()`

`np.mean` normally propagates NaN.

`np.nanmean` ignores NaN values when calculating the mean.

### `argmax()` versus `max()`

`max` returns the maximum value.

`argmax` returns its index.

## Common mistakes

Common NumPy errors include:

- confusing `shape` and `size`,
- forgetting zero-based indexing,
- assuming slices always create copies,
- using `and` instead of `&` for element-wise conditions,
- using `or` instead of `|`,
- omitting parentheses around Boolean conditions,
- confusing `*` and `@`,
- misunderstanding the shape `(n,)`,
- ignoring broadcasting rules,
- ignoring dtype and overflow,
- comparing floating-point values with exact equality,
- allowing NaN or infinity to enter calculations unexpectedly,
- creating unnecessary copies of large arrays,
- unintentionally modifying an array through a view,
- assuming vectorization is always faster,
- using `np.linalg.inv` when a direct linear-system solver is more appropriate,
- assuming `np.empty` produces zero-initialized values.

## Performance considerations

NumPy is designed to make large numerical operations efficient, but performance depends on implementation details.

### Prefer vectorized operations

When a numerical calculation can be clearly expressed using NumPy operations, vectorization can avoid Python-level iteration.

### Avoid unnecessary copies

Copies consume memory and require data movement.

Views can reduce memory use when their mutation behavior is acceptable.

### Choose dtypes deliberately

A smaller dtype can reduce memory consumption, but it can also reduce precision or numerical range.

### Consider memory bandwidth

Large vectorized calculations may become limited by memory movement rather than arithmetic speed.

### Avoid excessive temporary arrays

A complex expression may create intermediate arrays. For very large data, this can increase peak memory consumption.

### Measure real workloads

Performance should be evaluated using representative input sizes and realistic operations. A benchmark involving tiny arrays may produce conclusions that do not apply to production workloads.

## Numerical considerations

Numerical software must account for the limitations of finite-precision arithmetic.

Important issues include:

- floating-point rounding,
- overflow,
- underflow,
- cancellation,
- accumulated numerical error,
- ill-conditioned mathematical problems,
- NaN propagation,
- infinity,
- inappropriate dtype selection.

Tolerance-based comparisons such as `np.isclose` are often more appropriate than direct equality for calculated floating-point results.

## Security considerations

NumPy is a numerical computing library, not an application security framework.

Security-sensitive applications should validate external numerical input before processing it.

Particular care is required with serialized numerical data and workflows that may deserialize Python objects. Untrusted serialized data should not be loaded blindly when the serialization mechanism permits object reconstruction.

NumPy's numerical random generators should also not be treated as cryptographic random-number generators.

Security boundaries should be implemented at the application and data-processing layers rather than assumed from NumPy itself.

## Debugging considerations

Shape-related problems are among the most common NumPy errors.

A useful debugging procedure is:

1. Inspect `array.shape`.
2. Inspect `array.ndim`.
3. Inspect `array.dtype`.
4. Identify what each axis represents.
5. Compare shapes from the rightmost dimension.
6. Determine whether broadcasting rules are satisfied.
7. Use `reshape`, `expand_dims`, or `np.newaxis` when a particular orientation is intended.
8. Check for NaN and infinity when numerical results appear unexpected.
9. Determine whether an array is a view or copy when mutation behaves unexpectedly.
10. Test numerical assumptions with small arrays before applying them to large datasets.

Printing the shape of every major intermediate result can be particularly useful when developing unfamiliar numerical code.

## Implementation considerations

A reliable NumPy implementation should make array assumptions explicit.

Important assumptions can include:

- expected number of dimensions,
- expected axis meanings,
- accepted dtype,
- expected numerical range,
- whether NaN is permitted,
- whether infinity is permitted,
- whether input mutation is allowed,
- whether output must own independent memory,
- whether the operation requires C-contiguous or Fortran-contiguous storage.

Explicit assumptions make numerical code easier to test and maintain.

## Production design considerations

For production numerical systems, correctness should take priority over compact syntax.

Input validation should occur at boundaries.

Array shapes should be documented when they carry semantic meaning.

Dtypes should be selected deliberately.

Mutation should be controlled.

Potentially undefined operations such as division by zero should have explicit behavior.

Large arrays should be evaluated for memory requirements.

Numerical comparisons should account for floating-point precision.

Random simulations should use controlled generators when reproducibility matters.

Testing should cover normal cases, boundary cases, empty inputs, unusual shapes, dtype behavior, missing values, and numerical tolerances.

## Testing

The script includes direct assertions for core behavior.

Tests verify:

- array dimensions,
- expected shapes,
- element counts,
- indexing,
- slicing,
- broadcasting,
- mathematical results,
- reshaping.

For floating-point calculations, approximate comparisons are used where exact binary equality is inappropriate.

For exact integer arrays, exact array comparison is appropriate.

## Script coverage

The Python study script demonstrates the following NumPy concepts:

- NumPy arrays
- ndarray structure
- array creation
- nested arrays
- dimensions
- axes
- shape
- size
- dtype
- item size
- memory size
- zeros
- ones
- full arrays
- empty arrays
- identity matrices
- arange
- linspace
- random number generation
- indexing
- negative indexing
- multidimensional indexing
- slicing
- slice views
- explicit copies
- integer-array indexing
- Boolean masks
- conditional replacement
- `where`
- `select`
- element-wise arithmetic
- scalar operations
- comparisons
- matrix multiplication
- universal functions
- trigonometry
- aggregation
- axis-based reduction
- `keepdims`
- NaN-aware statistics
- reshape
- inferred dimensions
- flatten
- ravel
- transpose
- swapaxes
- moveaxis
- broadcasting
- `newaxis`
- `expand_dims`
- `squeeze`
- concatenate
- stack
- vertical and horizontal stacking
- splitting
- sorting
- argsort
- searchsorted
- unique values
- argmin
- argmax
- nonzero indexes
- random simulation
- memory sharing
- views and copies
- memory layout
- C-contiguous arrays
- Fortran-contiguous arrays
- normalization
- standardization
- min-max scaling
- safe division
- clipping
- cumulative operations
- differences
- numerical precision
- matrix operations
- linear-system solving
- structured arrays
- shape validation
- numerical edge cases
- Boolean ambiguity
- performance comparison
- vectorization
- in-place operations
- read-only arrays
- practical numerical pipelines
- testing and assertions

## Core syntax reference

Array creation:

    np.array(data)
    np.zeros(shape)
    np.ones(shape)
    np.full(shape, value)
    np.empty(shape)
    np.arange(start, stop, step)
    np.linspace(start, stop, count)

Inspection:

    array.ndim
    array.shape
    array.size
    array.dtype
    array.itemsize
    array.nbytes

Indexing:

    array[index]
    array[row, column]
    array[:, column]
    array[row, :]
    array[start:stop:step]

Boolean selection:

    array[array > value]
    array[(array > lower) & (array < upper)]

Reshaping:

    array.reshape(new_shape)
    array.reshape(rows, -1)

Axis manipulation:

    array.T
    np.expand_dims(array, axis)
    np.squeeze(array)
    np.swapaxes(array, axis1, axis2)
    np.moveaxis(array, source, destination)

Aggregation:

    np.sum(array)
    np.mean(array)
    np.median(array)
    np.min(array)
    np.max(array)
    np.std(array)
    np.var(array)

Element-wise mathematics:

    np.sqrt(array)
    np.exp(array)
    np.log(array)
    np.sin(array)
    np.cos(array)
    np.abs(array)
    np.floor(array)
    np.ceil(array)

Array combination:

    np.concatenate(...)
    np.stack(...)
    np.vstack(...)
    np.hstack(...)

Filtering and selection:

    np.where(condition)
    np.select(...)
    np.nonzero(...)
    np.flatnonzero(...)

Ordering:

    np.sort(array)
    array.sort()
    np.argsort(array)
    np.searchsorted(array, value)
    np.unique(array)

Numerical checks:

    np.isnan(array)
    np.isinf(array)
    np.isfinite(array)
    np.isclose(a, b)

Matrix operations:

    A @ B
    np.linalg.det(A)
    np.linalg.inv(A)
    np.linalg.solve(A, b)

The central NumPy skill demonstrated throughout the script is the ability to reason about an array's shape, axis structure, dtype, memory behavior, and compatibility with other arrays before performing numerical operations.
