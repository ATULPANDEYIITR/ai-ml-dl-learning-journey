"""
NumPy Fundamentals
==================

A self-contained study script covering NumPy arrays, dimensions, shapes,
indexing, slicing, broadcasting, and closely related fundamentals.

Requirements:
    Python 3.9+
    NumPy

Run:
    python numpy_fundamentals.py
"""

from __future__ import annotations

import math
import sys
import time
from typing import Iterable

import numpy as np


# ============================================================================
# SECTION 1: SETUP AND HELPER FUNCTIONS
# ============================================================================

def section(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    """Print a readable subsection heading."""
    print("\n" + "-" * 68)
    print(title)
    print("-" * 68)


def show(name: str, value) -> None:
    """Display a value with its Python type when useful."""
    print(f"{name}:")
    print(value)
    if isinstance(value, np.ndarray):
        print(f"  type      = {type(value).__name__}")
        print(f"  dtype     = {value.dtype}")
        print(f"  ndim      = {value.ndim}")
        print(f"  shape     = {value.shape}")
        print(f"  size      = {value.size}")
        print(f"  itemsize  = {value.itemsize} bytes")
        print(f"  nbytes    = {value.nbytes} bytes")


def assert_equal_arrays(actual: np.ndarray, expected: np.ndarray) -> None:
    """Small testing helper for exact array comparisons."""
    if not np.array_equal(actual, expected):
        raise AssertionError(
            f"Arrays differ.\nActual:\n{actual}\nExpected:\n{expected}"
        )


# ============================================================================
# SECTION 2: WHAT NUMPY IS
# ============================================================================

section("1. NumPy fundamentals")

print(
    """
NumPy is Python's fundamental numerical-computing library.

Its central object is the ndarray, a multidimensional homogeneous array.
Unlike a normal Python list, a NumPy array normally stores values of one
data type in a compact memory representation and supports vectorized
operations.

The most important properties demonstrated in this script are:

    ndim   -> number of dimensions, also called axes
    shape  -> size along every axis
    size   -> total number of elements
    dtype  -> data type of the elements
    indexing -> selecting elements
    slicing  -> selecting ranges or sub-arrays
    broadcasting -> aligning compatible shapes for arithmetic

The examples progress from one-dimensional arrays to multidimensional
arrays and then to more advanced array operations.
"""
)


# ============================================================================
# SECTION 3: CREATING ARRAYS
# ============================================================================

section("2. Creating NumPy arrays")

subsection("2.1 From a Python list")

python_list = [10, 20, 30, 40, 50]
array_from_list = np.array(python_list)

show("Python list", python_list)
show("NumPy array", array_from_list)

subsection("2.2 From nested lists")

matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)

show("Two-dimensional array", matrix)

print(
    """
A nested Python list becomes a multidimensional NumPy array when the
nested structure is rectangular.

For example:

    [[1, 2, 3],
     [4, 5, 6]]

has two rows and three columns, so its shape is (2, 3).
"""
)

subsection("2.3 Explicit dtype")

integers_32 = np.array([1, 2, 3], dtype=np.int32)
floating_point = np.array([1, 2, 3], dtype=np.float64)

show("32-bit integers", integers_32)
show("64-bit floating-point values", floating_point)

subsection("2.4 Common constructor functions")

zeros = np.zeros((2, 3))
ones = np.ones((2, 3))
full = np.full((2, 3), 7)
empty = np.empty((2, 3))
identity = np.eye(3)

show("np.zeros((2, 3))", zeros)
show("np.ones((2, 3))", ones)
show("np.full((2, 3), 7)", full)
show("np.empty((2, 3))", empty)
show("np.eye(3)", identity)

print(
    """
np.empty creates an array without initializing its values. Its contents
should never be assumed to be zero. Initialize it explicitly if the
initial values matter.
"""
)

subsection("2.5 arange")

sequence = np.arange(0, 10, 2)
show("np.arange(0, 10, 2)", sequence)

print(
    """
np.arange(start, stop, step) behaves similarly to Python's range.

The stop value is normally excluded:

    np.arange(0, 5) -> [0, 1, 2, 3, 4]

For floating-point steps, np.linspace is often safer when a specific
number of evenly spaced values is required.
"""
)

subsection("2.6 linspace")

linear_space = np.linspace(0, 1, 6)
show("np.linspace(0, 1, 6)", linear_space)

subsection("2.7 Random arrays")

rng = np.random.default_rng(42)

random_uniform = rng.random((2, 3))
random_integers = rng.integers(1, 10, size=(2, 3))

show("Random uniform values", random_uniform)
show("Random integers from 1 through 9", random_integers)

print(
    """
default_rng creates a modern NumPy random-number generator.

Using a fixed seed such as 42 makes the demonstration reproducible.
In production simulations, choose seeds according to the reproducibility
and randomness requirements of the application.
"""
)


# ============================================================================
# SECTION 4: DIMENSIONS, AXES, SHAPE, SIZE
# ============================================================================

section("3. Dimensions, axes, shape, and size")

scalar_array = np.array(42)
vector = np.array([10, 20, 30, 40])
matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)
cube = np.arange(24).reshape(2, 3, 4)

show("0-dimensional array", scalar_array)
show("1-dimensional array", vector)
show("2-dimensional array", matrix)
show("3-dimensional array", cube)

print(
    """
Dimension terminology:

0D:
    A scalar-like array containing one value.

1D:
    A vector-like array. Shape (n).

2D:
    A matrix-like array. Shape (rows, columns).

3D and higher:
    Arrays containing multiple axes. Their interpretation depends on
    the application.

Important distinction:
    ndim tells how many axes exist.
    shape tells the length of each axis.
    size tells the total number of elements.

For a shape (2, 3, 4):
    ndim = 3
    size = 2 * 3 * 4 = 24
"""
)

subsection("3.1 Understanding axes")

axis_example = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)

show("Axis example", axis_example)

print(
    """
For a 2D array:

    axis=0 moves vertically through rows.
    axis=1 moves horizontally through columns.

Reducing with axis=0 combines values down each column.
Reducing with axis=1 combines values across each row.
"""
)

print("Sum over axis=0:", axis_example.sum(axis=0))
print("Sum over axis=1:", axis_example.sum(axis=1))

# Keep dimensions instead of removing the reduced axis.
print("Sum axis=0 with keepdims=True:")
print(axis_example.sum(axis=0, keepdims=True))

print("Sum axis=1 with keepdims=True:")
print(axis_example.sum(axis=1, keepdims=True))


# ============================================================================
# SECTION 5: DTYPE AND TYPE CONVERSION
# ============================================================================

section("4. Data types")

subsection("4.1 Common NumPy data types")

dtype_examples = {
    "int32": np.array([1, 2, 3], dtype=np.int32),
    "int64": np.array([1, 2, 3], dtype=np.int64),
    "float32": np.array([1, 2, 3], dtype=np.float32),
    "float64": np.array([1, 2, 3], dtype=np.float64),
    "bool": np.array([True, False, True], dtype=bool),
    "complex128": np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
}

for name, arr in dtype_examples.items():
    print(f"{name:12} -> dtype={arr.dtype}, values={arr}")

subsection("4.2 Type conversion")

original = np.array([1.2, 2.7, 3.9])
converted = original.astype(np.int64)

show("Original floating-point array", original)
show("Converted integer array", converted)

print(
    """
astype creates an array with the requested dtype. Converting floating
values to integers truncates the fractional component toward zero.

For example:
    2.9 -> 2
   -2.9 -> -2

This is not the same as mathematical rounding.
"""
)

rounded = np.round(original)
show("Rounded values", rounded)

subsection("4.3 Integer overflow")

small_integer = np.array([127], dtype=np.int8)
overflow_result = small_integer + np.int8(1)

print("int8 maximum:", np.iinfo(np.int8).max)
print("127 stored as int8:", small_integer)
print("127 + 1 with int8 arithmetic:", overflow_result)

print(
    """
Fixed-width integer types have finite ranges. NumPy integer arithmetic
can wrap around when the result cannot be represented by the dtype.

Use an appropriately sized dtype when numerical range is important.
"""
)


# ============================================================================
# SECTION 6: ARRAY INSPECTION
# ============================================================================

section("5. Inspecting arrays")

inspection_array = np.arange(12).reshape(3, 4)

print("Array:")
print(inspection_array)
print("ndim :", inspection_array.ndim)
print("shape:", inspection_array.shape)
print("size :", inspection_array.size)
print("dtype:", inspection_array.dtype)
print("itemsize:", inspection_array.itemsize)
print("nbytes:", inspection_array.nbytes)

print(
    """
Useful inspection attributes:

    ndim
    shape
    size
    dtype
    itemsize
    nbytes

These properties are essential when debugging shape-related problems.
"""


# ============================================================================
# SECTION 7: INDEXING
# ============================================================================

section("6. Indexing")

subsection("6.1 One-dimensional indexing")

numbers = np.array([10, 20, 30, 40, 50])

print("numbers:", numbers)
print("numbers[0]:", numbers[0])
print("numbers[2]:", numbers[2])
print("numbers[-1]:", numbers[-1])
print("numbers[-2]:", numbers[-2])

print(
    """
NumPy uses zero-based indexing, just like Python lists.

Positive indexes:
    0 -> first element
    1 -> second element

Negative indexes:
    -1 -> last element
    -2 -> second-to-last element
"""
)

subsection("6.2 Two-dimensional indexing")

table = np.array(
    [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ]
)

print("table:")
print(table)
print("table[0, 0]:", table[0, 0])
print("table[1, 2]:", table[1, 2])
print("table[-1, -1]:", table[-1, -1])

print(
    """
For a 2D array:

    array[row, column]

is preferred over nested indexing:

    array[row][column]

Both can work, but the comma-based form expresses multidimensional
indexing directly and is usually clearer and more general.
"""
)

subsection("6.3 Selecting rows and columns")

print("First row:", table[0, :])
print("Second row:", table[1, :])
print("First column:", table[:, 0])
print("Third column:", table[:, 2])

subsection("6.4 Three-dimensional indexing")

three_d = np.arange(24).reshape(2, 3, 4)

print("3D array:")
print(three_d)
print("three_d[0, 1, 2]:", three_d[0, 1, 2])
print("First block:")
print(three_d[0])


# ============================================================================
# SECTION 8: SLICING
# ============================================================================

section("7. Slicing")

subsection("7.1 One-dimensional slicing")

values = np.arange(10)

print("values:", values)
print("values[2:7]:", values[2:7])
print("values[:5]:", values[:5])
print("values[5:]:", values[5:])
print("values[::2]:", values[::2])
print("values[1::2]:", values[1::2])
print("values[::-1]:", values[::-1])

print(
    """
The general slice form is:

    start:stop:step

The stop index is excluded.

Examples:

    [2:7]  -> indexes 2 through 6
    [:5]   -> beginning through index 4
    [5:]   -> index 5 through the end
    [::2]  -> every second element
    [::-1] -> reverse order
"""
)

subsection("7.2 Two-dimensional slicing")

grid = np.arange(25).reshape(5, 5)

print("grid:")
print(grid)

print("Rows 1 through 3:")
print(grid[1:4, :])

print("Columns 2 through 4:")
print(grid[:, 2:5])

print("Middle 3x3 region:")
print(grid[1:4, 1:4])

print("Every second row and every second column:")
print(grid[::2, ::2])

subsection("7.3 Slices are generally views")

original = np.array([10, 20, 30, 40, 50])
view = original[1:4]

print("Original before modification:", original)
print("Slice view:", view)

view[0] = 999

print("Original after modifying slice:", original)
print("Slice after modification:", view)

print(
    """
Basic NumPy slices usually produce views rather than independent copies.
Changing the view can therefore change the original array.

Use .copy() when an independent array is required.
"""
)

original = np.array([10, 20, 30, 40, 50])
copied_slice = original[1:4].copy()
copied_slice[0] = 999

print("Original after modifying copied slice:", original)
print("Copied slice:", copied_slice)


# ============================================================================
# SECTION 9: FANCY INDEXING AND BOOLEAN MASKING
# ============================================================================

section("8. Advanced indexing")

subsection("8.1 Integer-array indexing")

scores = np.array([55, 72, 91, 64, 88, 43])

selected = scores[[0, 2, 4]]

print("scores:", scores)
print("Selected indexes [0, 2, 4]:", selected)

subsection("8.2 Boolean indexing")

passed = scores >= 60

print("scores:", scores)
print("scores >= 60:", passed)
print("Passing scores:", scores[passed])

print(
    """
Boolean indexing creates a Boolean mask and uses it to select elements.

This is one of NumPy's most useful techniques for data analysis.
"""
)

subsection("8.3 Multiple conditions")

high_scores = scores[(scores >= 70) & (scores < 90)]

print("Scores from 70 inclusive to below 90:", high_scores)

print(
    """
For NumPy Boolean conditions:

    (condition_a) & (condition_b)   -> logical AND
    (condition_a) | (condition_b)   -> logical OR
    ~(condition_a)                  -> logical NOT

Use parentheses around individual conditions.

Do not replace & with Python's 'and' when working element-wise with
NumPy arrays.
"""
)

subsection("8.4 Conditional replacement")

adjusted_scores = scores.copy()
adjusted_scores[adjusted_scores < 50] = 50

print("Original scores:", scores)
print("Scores with minimum 50:", adjusted_scores)

subsection("8.5 np.where")

status = np.where(scores >= 60, "Pass", "Fail")

print("Status:", status)

bonus_scores = np.where(scores >= 80, scores + 5, scores)

print("Bonus-adjusted scores:", bonus_scores)


# ============================================================================
# SECTION 10: BASIC ARRAY ARITHMETIC
# ============================================================================

section("9. Array arithmetic")

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print("a:", a)
print("b:", b)
print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a ** 2:", a ** 2)

print(
    """
NumPy arithmetic is element-wise by default.

Therefore:

    [1, 2, 3] + [10, 20, 30]

produces:

    [11, 22, 33]

This differs from Python list concatenation.
"""
)

python_a = [1, 2, 3]
python_b = [10, 20, 30]

print("Python list +:", python_a + python_b)

subsection("9.1 Scalar operations")

temperatures = np.array([20.0, 25.0, 30.0])

print("Temperatures:", temperatures)
print("Temperature + 5:", temperatures + 5)
print("Temperature * 1.8:", temperatures * 1.8)
print("Temperature / 2:", temperatures / 2)

subsection("9.2 Comparisons")

print("temperatures > 24:", temperatures > 24)
print("temperatures == 25:", temperatures == 25)

subsection("9.3 Matrix multiplication")

matrix_a = np.array(
    [
        [1, 2],
        [3, 4],
    ]
)

matrix_b = np.array(
    [
        [5, 6],
        [7, 8],
    ]
)

print("Element-wise multiplication:")
print(matrix_a * matrix_b)

print("Matrix multiplication with @:")
print(matrix_a @ matrix_b)

print(
    """
Important distinction:

    A * B
        element-wise multiplication

    A @ B
        matrix multiplication

For 2D arrays, np.matmul(A, B) and A @ B perform matrix multiplication.
"""
)


# ============================================================================
# SECTION 11: UNIVERSAL FUNCTIONS
# ============================================================================

section("10. Universal functions and vectorization")

angles = np.array([0, np.pi / 6, np.pi / 4, np.pi / 2])

print("Angles in radians:", angles)
print("sin:", np.sin(angles))
print("cos:", np.cos(angles))
print("tan:", np.tan(angles))

positive_values = np.array([1, 4, 9, 16])

print("sqrt:", np.sqrt(positive_values))
print("log:", np.log(positive_values))
print("log10:", np.log10(positive_values))
print("exp:", np.exp(np.array([0, 1, 2])))

print(
    """
NumPy provides vectorized mathematical functions called universal
functions, commonly called ufuncs.

Examples include:

    np.sqrt
    np.exp
    np.log
    np.sin
    np.cos
    np.abs
    np.floor
    np.ceil

These operate on arrays element by element without requiring a Python
for-loop for the numerical operation.
"""
)

subsection("10.1 abs, floor, ceil")

decimal_values = np.array([-2.7, -1.2, 0.4, 3.8])

print("Values:", decimal_values)
print("abs:", np.abs(decimal_values))
print("floor:", np.floor(decimal_values))
print("ceil:", np.ceil(decimal_values))


# ============================================================================
# SECTION 12: AGGREGATIONS
# ============================================================================

section("11. Aggregations and reductions")

data = np.array([4, 8, 15, 16, 23, 42])

print("data:", data)
print("sum:", np.sum(data))
print("mean:", np.mean(data))
print("minimum:", np.min(data))
print("maximum:", np.max(data))
print("median:", np.median(data))
print("standard deviation:", np.std(data))
print("variance:", np.var(data))

subsection("11.1 Axis-based reductions")

sales = np.array(
    [
        [100, 120, 130],
        [90, 110, 140],
        [80, 105, 125],
    ]
)

print("Sales by row:")
print(sales)

print("Total sales:", sales.sum())
print("Column totals:", sales.sum(axis=0))
print("Row totals:", sales.sum(axis=1))
print("Column means:", sales.mean(axis=0))
print("Row means:", sales.mean(axis=1))

subsection("11.2 NaN-aware reductions")

values_with_nan = np.array([10.0, 20.0, np.nan, 40.0])

print("Values containing NaN:", values_with_nan)
print("np.mean:", np.mean(values_with_nan))
print("np.nanmean:", np.nanmean(values_with_nan))
print("np.isnan:", np.isnan(values_with_nan))

print(
    """
NaN means "Not a Number" and is commonly used to represent missing or
undefined floating-point data.

Ordinary reductions such as np.mean propagate NaN.
NaN-aware functions such as np.nanmean ignore NaN values when possible.

The two approaches have different semantics and should not be treated as
interchangeable.
"""
)


# ============================================================================
# SECTION 13: RESHAPING
# ============================================================================

section("12. Reshaping arrays")

one_to_twelve = np.arange(1, 13)

print("Original:", one_to_twelve)

reshaped = one_to_twelve.reshape(3, 4)

print("reshape(3, 4):")
print(reshaped)

print("reshape(2, 6):")
print(one_to_twelve.reshape(2, 6))

print("reshape(12, 1):")
print(one_to_twelve.reshape(12, 1))

print(
    """
reshape changes the shape while preserving the number of elements.

Therefore:

    old.size == new.size

must hold.

A 12-element array can become:

    (3, 4)
    (2, 6)
    (12, 1)
    (1, 12)

but not:

    (5, 3)

because 5 * 3 is 15.
"""
)

subsection("12.1 Inferring one dimension with -1")

inferred = one_to_twelve.reshape(3, -1)

print("reshape(3, -1):")
print(inferred)

print(
    """
One dimension may be -1, which tells NumPy to infer that dimension.

For 12 elements:

    reshape(3, -1) -> (3, 4)
"""
)

subsection("12.2 Flatten versus ravel")

matrix = np.arange(12).reshape(3, 4)

flattened = matrix.flatten()
raveled = matrix.ravel()

print("Original:")
print(matrix)
print("flatten:")
print(flattened)
print("ravel:")
print(raveled)

flattened[0] = 999
raveled[1] = 888

print("Original after modifying flattened copy and raveled result:")
print(matrix)

print(
    """
flatten() returns a copy.

ravel() returns a flattened view when possible, although it may return
a copy when the memory layout does not allow a view.

This difference matters when memory usage and mutation behavior are
important.
"""
)


# ============================================================================
# SECTION 14: TRANSPOSE AND AXIS REARRANGEMENT
# ============================================================================

section("13. Transpose and axis manipulation")

matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)

print("Original shape:", matrix.shape)
print(matrix)

transposed = matrix.T

print("Transposed shape:", transposed.shape)
print(transposed)

print(
    """
For a 2D array, .T swaps rows and columns.

Shape:
    (2, 3) -> (3, 2)

For higher-dimensional arrays, axis ordering becomes more important.
"""
)

subsection("13.1 swapaxes")

cube = np.arange(24).reshape(2, 3, 4)

swapped = np.swapaxes(cube, 0, 2)

print("Original shape:", cube.shape)
print("After swapaxes(0, 2):", swapped.shape)

subsection("13.2 moveaxis")

moved = np.moveaxis(cube, 0, -1)

print("Original shape:", cube.shape)
print("After moveaxis(0, -1):", moved.shape)


# ============================================================================
# SECTION 15: BROADCASTING
# ============================================================================

section("14. Broadcasting")

print(
    """
Broadcasting is NumPy's mechanism for performing element-wise operations
between arrays with different but compatible shapes.

NumPy compares dimensions from the rightmost side.

Two dimensions are compatible when:

    1. They are equal, or
    2. One of them is 1.

Missing leading dimensions are treated as if they were 1.

Examples:

    (3, 4) and (4)      -> compatible
    (3, 4) and (1, 4)   -> compatible
    (3, 4) and (3, 1)   -> compatible
    (3, 4) and (3)      -> not compatible
"""
)

subsection("14.1 Scalar broadcasting")

matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)

print("matrix:")
print(matrix)
print("matrix + 10:")
print(matrix + 10)

subsection("14.2 One-dimensional vector with a matrix")

row_vector = np.array([10, 20, 30])

print("matrix:")
print(matrix)
print("row_vector:", row_vector)
print("matrix + row_vector:")
print(matrix + row_vector)

subsection("14.3 Column vector broadcasting")

column_vector = np.array(
    [
        [10],
        [20],
    ]
)

print("column_vector:")
print(column_vector)

print("matrix + column_vector:")
print(matrix + column_vector)

print(
    """
The shapes are:

    matrix          -> (2, 3)
    row_vector      -> (3,)
    column_vector   -> (2, 1)

For matrix + row_vector:

    (2, 3)
    (1, 3)

The row vector is conceptually repeated for each row.

For matrix + column_vector:

    (2, 3)
    (2, 1)

The column vector is conceptually repeated for each column.
"""
)

subsection("14.4 Broadcasting failure")

incompatible = np.array([10, 20])

try:
    result = matrix + incompatible
    print(result)
except ValueError as error:
    print("Broadcasting error:", error)

print(
    """
A shape mismatch is not necessarily an error in the code. It may mean
the intended operation has not been expressed with the correct shape.

reshape can make the intended orientation explicit.
"""
)


# ============================================================================
# SECTION 16: EXPLICIT BROADCASTING WITH NEWAXIS
# ============================================================================

section("15. np.newaxis and explicit dimensions")

values = np.array([1, 2, 3])

row = values[np.newaxis, :]
column = values[:, np.newaxis]

print("values shape:", values.shape)
print("row shape:", row.shape)
print(row)
print("column shape:", column.shape)
print(column)

print(
    """
A 1D array has shape (3), not (1, 3) or (3, 1).

Use np.newaxis or None to add an axis:

    values[np.newaxis, :] -> shape (1, 3)
    values[:, np.newaxis] -> shape (3, 1)

This is particularly useful for broadcasting.
"""
)

outer_sum = column + row

print("Outer sum:")
print(outer_sum)


# ============================================================================
# SECTION 17: STACKING AND CONCATENATION
# ============================================================================

section("16. Combining arrays")

first = np.array([1, 2, 3])
second = np.array([4, 5, 6])

print("concatenate:")
print(np.concatenate([first, second]))

print("stack:")
print(np.stack([first, second]))

print(
    """
concatenate joins existing axes.

stack creates a new axis.

For two shape-(3,) arrays:

    concatenate -> shape (6,)
    stack       -> shape (2, 3)
"""
)

subsection("16.1 Vertical and horizontal stacking")

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("vstack:")
print(np.vstack([a, b]))

print("hstack:")
print(np.hstack([a, b]))

print("column_stack:")
print(np.column_stack([first, second]))

subsection("16.2 Concatenation along a chosen axis")

print("axis=0:")
print(np.concatenate([a, b], axis=0))

print("axis=1:")
print(np.concatenate([a, b], axis=1))


# ============================================================================
# SECTION 18: SPLITTING ARRAYS
# ============================================================================

section("17. Splitting arrays")

values = np.arange(12)

print("values:", values)

parts = np.split(values, 3)

for index, part in enumerate(parts, start=1):
    print(f"Part {index}:", part)

print("array_split with uneven division:")
uneven_parts = np.array_split(np.arange(10), 3)

for index, part in enumerate(uneven_parts, start=1):
    print(f"Part {index}:", part)

print(
    """
np.split requires equal-sized divisions when an integer number of
sections is supplied.

np.array_split permits unequal-sized divisions.
"""
)


# ============================================================================
# SECTION 19: SORTING, SEARCHING, UNIQUE VALUES
# ============================================================================

section("18. Sorting and searching")

unsorted = np.array([40, 10, 30, 20, 10])

print("Original:", unsorted)
print("np.sort:", np.sort(unsorted))

in_place_sort = unsorted.copy()
in_place_sort.sort()

print("ndarray.sort result:", in_place_sort)
print("Original remains unchanged:", unsorted)

subsection("18.1 Argsort")

order = np.argsort(unsorted)

print("Sorting indexes:", order)
print("Values in sorted order:", unsorted[order])

subsection("18.2 Search")

search_values = np.array([10, 20, 30, 40, 50])

print("searchsorted for 25:", np.searchsorted(search_values, 25))
print("searchsorted for 30:", np.searchsorted(search_values, 30))

subsection("18.3 Unique values")

repeated = np.array([1, 2, 2, 3, 3, 3, 4])

unique_values, counts = np.unique(repeated, return_counts=True)

print("Unique values:", unique_values)
print("Counts:", counts)

print(
    """
searchsorted assumes the input array is sorted according to the ordering
being used.

np.unique returns sorted unique values for ordinary numeric arrays.
With return_counts=True, it also reports the frequency of each value.
"""
)


# ============================================================================
# SECTION 20: WHERE, NONZERO, ARGMIN, ARGMAX
# ============================================================================

section("19. Locating values")

measurements = np.array([12, 5, 18, 9, 21, 7])

print("Measurements:", measurements)
print("argmin:", np.argmin(measurements))
print("argmax:", np.argmax(measurements))
print("minimum value:", measurements.min())
print("maximum value:", measurements.max())

print("Indexes where value > 10:", np.where(measurements > 10)[0])
print("Nonzero indexes:", np.nonzero(measurements)[0])

print(
    """
argmin and argmax return indexes, not values.

Use:

    measurements[argmin]

when the corresponding value is required.
"""
)

max_index = np.argmax(measurements)
print("Maximum measurement:", measurements[max_index])


# ============================================================================
# SECTION 21: RANDOM SAMPLING AND SHAPES
# ============================================================================

section("20. Random arrays for simulations")

simulation_rng = np.random.default_rng(123)

normal_sample = simulation_rng.normal(
    loc=100,
    scale=15,
    size=1000,
)

print("Sample shape:", normal_sample.shape)
print("Sample mean:", normal_sample.mean())
print("Sample standard deviation:", normal_sample.std())
print("Sample minimum:", normal_sample.min())
print("Sample maximum:", normal_sample.max())

uniform_sample = simulation_rng.uniform(0, 1, size=(3, 4))
print("Uniform sample:")
print(uniform_sample)

print(
    """
Random generation is useful for simulations, testing, sampling, and
statistical experiments.

The generated distribution depends on the function used. For example:

    normal()  -> normal/Gaussian distribution
    uniform() -> uniform distribution
    integers() -> discrete integer sampling
"""
)


# ============================================================================
# SECTION 22: MEMORY, COPY, AND VIEW SEMANTICS
# ============================================================================

section("21. Copies, views, and memory sharing")

base = np.arange(10)

view = base[2:7]
copy = base[2:7].copy()

print("Base:", base)
print("View:", view)
print("Copy:", copy)

print("shares_memory(base, view):", np.shares_memory(base, view))
print("shares_memory(base, copy):", np.shares_memory(base, copy))

view[0] = 1000

print("Base after modifying view:", base)
print("Copy after modifying view:", copy)

print(
    """
Memory behavior matters for both correctness and performance.

A view can avoid copying large amounts of data, which can be efficient.
The trade-off is that changes may affect the original array.

A copy owns independent data and is safer when independent mutation is
required, but it consumes additional memory and time.
"""
)


# ============================================================================
# SECTION 23: CONTIGUOUS MEMORY AND ORDER
# ============================================================================

section("22. Memory layout")

layout_array = np.arange(12).reshape(3, 4)

print("Array:")
print(layout_array)
print("C-contiguous:", layout_array.flags["C_CONTIGUOUS"])
print("F-contiguous:", layout_array.flags["F_CONTIGUOUS"])

fortran_array = np.asfortranarray(layout_array)

print("Fortran-contiguous array:")
print(fortran_array)
print("C-contiguous:", fortran_array.flags["C_CONTIGUOUS"])
print("F-contiguous:", fortran_array.flags["F_CONTIGUOUS"])

print(
    """
NumPy commonly stores multidimensional arrays in C order, where the
last axis varies fastest in memory.

Fortran order stores data with the first axis varying fastest.

Memory layout can affect interoperability and performance, especially
when arrays are passed to compiled numerical libraries.
"""
)


# ============================================================================
# SECTION 24: BROADCASTING WITH REALISTIC DATA
# ============================================================================

section("23. Practical broadcasting example")

# Rows represent products; columns represent monthly sales.
monthly_sales = np.array(
    [
        [120, 130, 150, 160],
        [80, 95, 100, 110],
        [200, 220, 210, 250],
    ],
    dtype=float,
)

# A percentage adjustment for each month.
monthly_adjustments = np.array([1.00, 1.05, 0.98, 1.10])

adjusted_sales = monthly_sales * monthly_adjustments

print("Monthly sales:")
print(monthly_sales)

print("Monthly adjustment factors:")
print(monthly_adjustments)

print("Adjusted sales:")
print(adjusted_sales)

print(
    """
The adjustment vector has shape (4,), while the sales matrix has shape
(3, 4). Broadcasting applies each month's adjustment to every product.
"""
)

subsection("23.1 Row-specific broadcasting")

product_adjustments = np.array(
    [
        [1.10],
        [0.95],
        [1.20],
    ]
)

product_adjusted_sales = monthly_sales * product_adjustments

print("Product adjustment factors:")
print(product_adjustments)

print("Product-adjusted sales:")
print(product_adjusted_sales)


# ============================================================================
# SECTION 25: NORMALIZATION
# ============================================================================

section("24. Broadcasting for normalization")

features = np.array(
    [
        [10.0, 100.0, 1000.0],
        [20.0, 200.0, 2000.0],
        [30.0, 300.0, 3000.0],
        [40.0, 400.0, 4000.0],
    ]
)

column_means = features.mean(axis=0)
column_stds = features.std(axis=0)

standardized = (features - column_means) / column_stds

print("Original features:")
print(features)

print("Column means:", column_means)
print("Column standard deviations:", column_stds)

print("Standardized features:")
print(standardized)

print("Standardized means:", standardized.mean(axis=0))
print("Standardized standard deviations:", standardized.std(axis=0))

print(
    """
This is a common numerical-data transformation:

    standardized = (x - mean) / standard_deviation

The mean and standard deviation have shape (3,), which broadcast across
the four rows.
"""
)


# ============================================================================
# SECTION 26: NUMPY AND PYTHON LOOPS
# ============================================================================

section("25. Vectorization versus Python loops")

values = np.arange(1, 1_000_001, dtype=np.float64)

start = time.perf_counter()

loop_result = np.empty_like(values)

for index, value in enumerate(values):
    loop_result[index] = value * value + 2 * value + 1

loop_time = time.perf_counter() - start

start = time.perf_counter()

vectorized_result = values * values + 2 * values + 1

vectorized_time = time.perf_counter() - start

print(f"Loop calculation time:       {loop_time:.6f} seconds")
print(f"Vectorized calculation time: {vectorized_time:.6f} seconds")

assert_equal_arrays(loop_result, vectorized_result)

print(
    """
Vectorized NumPy expressions often outperform explicit Python loops
because the numerical work is implemented in optimized compiled code.

The exact speed difference depends on the operation, array size,
hardware, dtype, memory layout, and other factors.

Vectorization is not automatically faster for every tiny operation.
For very small arrays, function-call and setup overhead can dominate.
"""
)


# ============================================================================
# SECTION 27: SHAPE ERRORS AND DEBUGGING
# ============================================================================

section("26. Debugging shape errors")

def safe_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Add two arrays while reporting useful shape information on failure."""
    try:
        return left + right
    except ValueError as error:
        raise ValueError(
            f"Cannot broadcast arrays with shapes "
            f"{left.shape} and {right.shape}. "
            f"Check dimensions from the rightmost axis."
        ) from error


left = np.ones((2, 3))
right = np.ones((3,))

print("Compatible shapes:")
print(left.shape, "+", right.shape)
print(safe_add(left, right))

wrong = np.ones((2,))

print("Incompatible shapes:")
print(left.shape, "+", wrong.shape)

try:
    safe_add(left, wrong)
except ValueError as error:
    print(error)

print(
    """
A practical debugging sequence for NumPy shape errors is:

    1. Print or inspect .shape.
    2. Identify the intended axis meaning.
    3. Compare dimensions from the right.
    4. Check whether dimensions are equal or one is 1.
    5. Use reshape, expand_dims, or newaxis when the intended orientation
       needs to be made explicit.
"""
)


# ============================================================================
# SECTION 28: EXPAND_DIMS AND SQUEEZE
# ============================================================================

section("27. expand_dims and squeeze")

vector = np.array([10, 20, 30])

expanded_axis0 = np.expand_dims(vector, axis=0)
expanded_axis1 = np.expand_dims(vector, axis=1)

print("Original shape:", vector.shape)
print("Expanded axis 0 shape:", expanded_axis0.shape)
print(expanded_axis0)
print("Expanded axis 1 shape:", expanded_axis1.shape)
print(expanded_axis1)

single_dimension = np.array([[[10], [20], [30]]])

print("Before squeeze:", single_dimension.shape)

squeezed = np.squeeze(single_dimension)

print("After squeeze:", squeezed.shape)
print(squeezed)

print(
    """
expand_dims inserts a dimension of length 1.

squeeze removes dimensions whose length is exactly 1.

Be careful with squeeze in production code because removing every
singleton dimension can change the expected rank of an array.
"""
)


# ============================================================================
# SECTION 29: DIMENSION VALIDATION
# ============================================================================

section("28. Shape validation")

def require_2d_matrix(array: np.ndarray) -> None:
    """Validate that an input is a two-dimensional NumPy array."""
    if not isinstance(array, np.ndarray):
        raise TypeError("Expected a NumPy ndarray.")

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, but received ndim={array.ndim} "
            f"and shape={array.shape}."
        )


def matrix_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication after explicit shape validation.

    For A with shape (m, n) and B with shape (n, p), the result has
    shape (m, p).
    """
    require_2d_matrix(left)
    require_2d_matrix(right)

    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Matrix dimensions are incompatible: "
            f"{left.shape} cannot be multiplied by {right.shape}."
        )

    return left @ right


valid_a = np.ones((2, 3))
valid_b = np.ones((3, 4))

result = matrix_multiply(valid_a, valid_b)

print("A shape:", valid_a.shape)
print("B shape:", valid_b.shape)
print("Result shape:", result.shape)

try:
    matrix_multiply(np.ones((2, 3)), np.ones((2, 4)))
except ValueError as error:
    print("Validation error:", error)


# ============================================================================
# SECTION 30: CONDITIONAL LOGIC WITHOUT PYTHON LOOPS
# ============================================================================

section("29. Vectorized conditional logic")

income = np.array([25000, 45000, 75000, 120000, 200000], dtype=float)

tax_rate = np.select(
    [
        income <= 30000,
        income <= 60000,
        income <= 100000,
        income > 100000,
    ],
    [
        0.05,
        0.10,
        0.20,
        0.30,
    ],
    default=np.nan,
)

estimated_tax = income * tax_rate

print("Income:", income)
print("Selected tax rates:", tax_rate)
print("Estimated tax:", estimated_tax)

print(
    """
np.select supports multiple vectorized conditions.

This is useful when a calculation depends on several categories or
thresholds.

The calculation above is intentionally simplified and is not a tax
law implementation.
"""
)


# ============================================================================
# SECTION 31: NAN, INF, FINITE VALUES
# ============================================================================

section("30. NaN, infinity, and finite-value handling")

special_values = np.array(
    [
        1.0,
        np.nan,
        np.inf,
        -np.inf,
        5.0,
    ]
)

print("Special values:", special_values)
print("isnan:", np.isnan(special_values))
print("isinf:", np.isinf(special_values))
print("isfinite:", np.isfinite(special_values))

finite_values = special_values[np.isfinite(special_values)]

print("Finite values:", finite_values)

print(
    """
NaN, positive infinity, and negative infinity are distinct floating-point
states.

Useful checks include:

    np.isnan(x)
    np.isinf(x)
    np.isfinite(x)

These checks are important before statistical calculations or operations
that require finite numerical values.
"""
)


# ============================================================================
# SECTION 32: CLIPPING
# ============================================================================

section("31. Clipping values")

raw_scores = np.array([-10, 20, 55, 80, 110])

clipped_scores = np.clip(raw_scores, 0, 100)

print("Raw scores:", raw_scores)
print("Clipped to 0-100:", clipped_scores)

print(
    """
np.clip limits values to a specified interval.

Values below the minimum become the minimum.
Values above the maximum become the maximum.
Values inside the interval remain unchanged.
"""
)


# ============================================================================
# SECTION 33: REPRODUCIBLE RANDOM SIMULATION
# ============================================================================

section("32. Reproducible simulation")

rng_a = np.random.default_rng(2026)
rng_b = np.random.default_rng(2026)

sample_a = rng_a.integers(0, 100, size=10)
sample_b = rng_b.integers(0, 100, size=10)

print("Sample A:", sample_a)
print("Sample B:", sample_b)
print("Samples equal:", np.array_equal(sample_a, sample_b))

assert_equal_arrays(sample_a, sample_b)

print(
    """
Independent generators initialized with the same seed produce the same
sequence, which is useful for reproducible experiments and tests.

Do not use a predictable seed when an application requires cryptographic
randomness. NumPy's generator is designed for numerical simulation,
not for cryptographic security.
"""
)


# ============================================================================
# SECTION 34: STRUCTURED ARRAY EXAMPLE
# ============================================================================

section("33. Structured arrays")

print(
    """
NumPy arrays are normally homogeneous, meaning all elements share one
dtype.

Structured arrays are a specialized feature that allow records with
named fields and potentially different field types.

They are useful in some low-level numerical workflows, although many
tabular-data applications use dedicated data-frame structures instead.
"""
)

person_dtype = np.dtype(
    [
        ("id", np.int32),
        ("age", np.int16),
        ("score", np.float64),
    ]
)

people = np.array(
    [
        (1, 25, 87.5),
        (2, 31, 92.0),
        (3, 22, 78.5),
    ],
    dtype=person_dtype,
)

print("Structured array:")
print(people)
print("IDs:", people["id"])
print("Scores:", people["score"])


# ============================================================================
# SECTION 35: MATHEMATICAL EXAMPLE
# ============================================================================

section("34. Practical mathematical example: distance")

points = np.array(
    [
        [0.0, 0.0],
        [3.0, 4.0],
        [6.0, 8.0],
        [1.0, 1.0],
    ]
)

origin = np.array([0.0, 0.0])

distances = np.sqrt(np.sum((points - origin) ** 2, axis=1))

print("Points:")
print(points)
print("Distances from origin:")
print(distances)

print(
    """
For points with coordinates (x, y), Euclidean distance from the origin
is:

    sqrt(x^2 + y^2)

The operation is vectorized over every row.
"""
)


# ============================================================================
# SECTION 36: NORMALIZING DATA WITH MIN-MAX SCALING
# ============================================================================

section("35. Practical data example: min-max scaling")

raw_features = np.array(
    [
        [10, 100],
        [20, 150],
        [30, 200],
        [40, 300],
    ],
    dtype=float,
)

feature_min = raw_features.min(axis=0)
feature_max = raw_features.max(axis=0)

min_max_scaled = (
    (raw_features - feature_min)
    / (feature_max - feature_min)
)

print("Raw features:")
print(raw_features)
print("Feature minimums:", feature_min)
print("Feature maximums:", feature_max)
print("Min-max scaled:")
print(min_max_scaled)

print(
    """
Min-max scaling maps each feature to a range commonly defined as [0, 1]:

    x_scaled = (x - min) / (max - min)

A constant feature would cause a zero denominator. Production code
should explicitly handle that case.
"""
)


# ============================================================================
# SECTION 37: ROBUST MIN-MAX SCALING
# ============================================================================

section("36. Robust min-max scaling implementation")

def min_max_scale(array: np.ndarray) -> np.ndarray:
    """
    Scale each column of a 2D array to [0, 1].

    Constant columns are mapped to 0 rather than producing division by zero.
    """
    data = np.asarray(array, dtype=float)

    if data.ndim != 2:
        raise ValueError("min_max_scale expects a 2D array.")

    minimum = np.min(data, axis=0)
    maximum = np.max(data, axis=0)
    span = maximum - minimum

    result = np.zeros_like(data)

    non_constant = span != 0

    result[:, non_constant] = (
        (data[:, non_constant] - minimum[non_constant])
        / span[non_constant]
    )

    return result


constant_feature_data = np.array(
    [
        [10, 100],
        [20, 100],
        [30, 100],
    ],
    dtype=float,
)

print("Input:")
print(constant_feature_data)

print("Scaled:")
print(min_max_scale(constant_feature_data))


# ============================================================================
# SECTION 38: SAFE DIVISION
# ============================================================================

section("37. Safe element-wise division")

numerators = np.array([10.0, 20.0, 30.0, 40.0])
denominators = np.array([2.0, 0.0, 5.0, 0.0])

safe_result = np.divide(
    numerators,
    denominators,
    out=np.full_like(numerators, np.nan),
    where=denominators != 0,
)

print("Numerators:", numerators)
print("Denominators:", denominators)
print("Safe division:", safe_result)

print(
    """
np.divide supports an out parameter and a where mask.

This makes it possible to define what should happen where division is
not valid instead of accepting an uncontrolled divide-by-zero result.
"""
)


# ============================================================================
# SECTION 39: MATRIX OPERATIONS
# ============================================================================

section("38. Core matrix operations")

A = np.array(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)

print("A:")
print(A)

print("Transpose:")
print(A.T)

print("Trace:", np.trace(A))
print("Determinant:", np.linalg.det(A))
print("Inverse:")
print(np.linalg.inv(A))

identity_check = A @ np.linalg.inv(A)

print("A @ inverse(A):")
print(identity_check)

print(
    """
np.linalg contains linear-algebra functionality.

For example:

    np.linalg.det(A)       -> determinant
    np.linalg.inv(A)       -> inverse
    np.linalg.solve(A, b)  -> solution of Ax=b

For solving linear systems, np.linalg.solve is generally preferred to
explicitly calculating an inverse.
"""
)

b = np.array([5.0, 11.0])

solution = np.linalg.solve(A, b)

print("Solution of Ax=b:", solution)
print("Check A @ solution:", A @ solution)


# ============================================================================
# SECTION 40: EDGE CASES
# ============================================================================

section("39. Important edge cases")

subsection("39.1 Empty arrays")

empty_array = np.array([])

print("Empty array:", empty_array)
print("shape:", empty_array.shape)
print("size:", empty_array.size)

try:
    print("Mean of empty array:", np.mean(empty_array))
except Warning as warning:
    print("Warning:", warning)

print(
    """
Empty arrays are valid NumPy arrays, but many reductions have undefined
or warning-producing results on empty input.
"""
)

subsection("39.2 Singleton dimensions")

singleton = np.array([[42]])

print("singleton:", singleton)
print("shape:", singleton.shape)
print("scalar element:", singleton[0, 0])

subsection("39.3 Boolean ambiguity")

boolean_array = np.array([True, False, True])

try:
    if boolean_array:
        print("This will not normally execute.")
except ValueError as error:
    print("Boolean ambiguity:", error)

print(
    """
A NumPy array containing multiple Boolean values cannot normally be used
as a single Python truth value.

Use:

    np.any(array)
    np.all(array)

when a single Boolean result is intended.
"""
)

print("np.any:", np.any(boolean_array))
print("np.all:", np.all(boolean_array))


# ============================================================================
# SECTION 41: COMMON MISTAKES
# ============================================================================

section("40. Common mistakes")

print(
    """
Mistake 1: Confusing shape with size.

    shape=(2, 3)
    size=6

Mistake 2: Forgetting that indexing starts at zero.

Mistake 3: Assuming every slice creates a copy.
    Basic slices commonly create views.

Mistake 4: Using Python 'and' and 'or' for element-wise array conditions.
    Use '&' and '|', with parentheses.

Mistake 5: Confusing * with matrix multiplication.
    * is element-wise.
    @ is matrix multiplication.

Mistake 6: Ignoring dtype.
    Integer overflow, precision, and memory usage can depend on dtype.

Mistake 7: Assuming a 1D vector has a row or column orientation.
    Shape (n,) is neither (1, n) nor (n, 1).

Mistake 8: Ignoring broadcasting rules.
    A visually plausible operation may still fail because shapes are
    incompatible.

Mistake 9: Silently accepting NaN or infinity.
    Validate numerical data when finite values are required.

Mistake 10: Making unnecessary copies of large arrays.
    Copies consume memory and time.

Mistake 11: Modifying a view unintentionally.
    Use .copy() when independent ownership is required.

Mistake 12: Using an explicit Python loop when a clear vectorized
    operation exists.

Mistake 13: Using floating-point equality carelessly.

For example:
"""
)

floating_result = 0.1 + 0.2

print("0.1 + 0.2:", floating_result)
print("Exact equality with 0.3:", floating_result == 0.3)
print("np.isclose:", np.isclose(floating_result, 0.3))

print(
    """
Floating-point values are approximations. np.isclose is often more
appropriate when comparing calculated floating-point values.
"""
)


# ============================================================================
# SECTION 42: PERFORMANCE AND MEMORY COMPARISON
# ============================================================================

section("41. Performance and memory considerations")

large_python_list = list(range(100_000))
large_numpy_array = np.arange(100_000, dtype=np.int64)

print("Python list length:", len(large_python_list))
print("NumPy array size:", large_numpy_array.size)
print("NumPy array nbytes:", large_numpy_array.nbytes)

print(
    """
A Python list stores references to Python objects and has substantial
per-element object overhead.

A NumPy numeric array stores values in a compact typed memory buffer.
This allows efficient numerical operations and predictable memory usage.

Memory savings depend on dtype and the type of values stored in the
Python data structure, so exact comparisons require measurement for the
specific workload.
"""
)

subsection("41.1 Avoiding unnecessary temporary arrays")

values = np.arange(1_000_000, dtype=np.float64)

# This expression is clear and vectorized.
result = values * 2 + 5

print("Vectorized result sample:", result[:5])

print(
    """
Vectorization improves clarity and often improves speed, but complex
expressions can create temporary arrays.

For very large workloads, memory traffic can become the bottleneck.
Techniques such as in-place operations, carefully selected dtypes, and
chunked processing may reduce memory pressure.

In-place operations must be used carefully because they mutate the
original array.
"""
)


# ============================================================================
# SECTION 43: IN-PLACE OPERATIONS
# ============================================================================

section("42. In-place operations")

values = np.array([1.0, 2.0, 3.0, 4.0])

values *= 10
print("After *= 10:", values)

values += 5
print("After += 5:", values)

print(
    """
In-place operators such as +=, -=, *=, and /= can reduce allocation
when the dtype permits the operation.

They also mutate the array, so they are inappropriate when the original
data must remain unchanged.
"""
)


# ============================================================================
# SECTION 44: READ-ONLY ARRAYS
# ============================================================================

section("43. Read-only array protection")

protected = np.array([10, 20, 30])

protected.setflags(write=False)

print("Protected array:", protected)
print("Writable:", protected.flags.writeable)

try:
    protected[0] = 999
except ValueError as error:
    print("Write-protection error:", error)

print(
    """
An ndarray can be marked read-only.

This can help prevent accidental mutation when an array is intended to
be shared among components.

Read-only flags are not a substitute for a complete application-level
security model.
"""
)


# ============================================================================
# SECTION 45: SAFE ARRAY CONVERSION
# ============================================================================

section("44. np.asarray and np.array")

python_values = [1, 2, 3]

array_a = np.array(python_values)
array_b = np.asarray(python_values)

print("np.array:", array_a)
print("np.asarray:", array_b)

existing = np.array([4, 5, 6])

array_copy = np.array(existing)
array_view_or_same = np.asarray(existing)

print("np.shares_memory(existing, array_copy):",
      np.shares_memory(existing, array_copy))
print("np.shares_memory(existing, array_view_or_same):",
      np.shares_memory(existing, array_view_or_same))

print(
    """
np.array generally creates a new array when converting an existing
ndarray unless copy behavior is changed.

np.asarray is designed to avoid an unnecessary copy when the input is
already an ndarray with a suitable representation.

This distinction can matter in performance-sensitive code.
"""
)


# ============================================================================
# SECTION 46: ADVANCED BROADCASTING EXAMPLE
# ============================================================================

section("45. Advanced broadcasting example")

# Four observations, three features.
observations = np.array(
    [
        [10, 100, 1000],
        [20, 200, 2000],
        [30, 300, 3000],
        [40, 400, 4000],
    ],
    dtype=float,
)

# Feature-specific weights.
weights = np.array([0.1, 0.01, 0.001])

weighted_features = observations * weights

print("Observations:")
print(observations)
print("Weights:", weights)
print("Weighted features:")
print(weighted_features)

weighted_sum = weighted_features.sum(axis=1)

print("Weighted sum per observation:", weighted_sum)

print(
    """
The weights have shape (3,), which broadcasts across all four rows.

This pattern appears in scoring systems, weighted metrics, feature
engineering, financial calculations, and many numerical algorithms.
"""
)


# ============================================================================
# SECTION 47: OUTER OPERATIONS
# ============================================================================

section("46. Outer operations")

x = np.array([1, 2, 3])
y = np.array([10, 20, 30, 40])

outer_product = np.outer(x, y)

print("x:", x)
print("y:", y)
print("Outer product:")
print(outer_product)

outer_addition = x[:, np.newaxis] + y

print("Outer addition:")
print(outer_addition)

print(
    """
np.outer is a convenient way to calculate an outer product.

The same structure can often be expressed explicitly with broadcasting:

    x[:, None] * y[None, :]

The resulting shape is (len(x), len(y)).
"""
)


# ============================================================================
# SECTION 48: TILING AND REPEATING
# ============================================================================

section("47. repeat and tile")

base = np.array([1, 2, 3])

print("repeat:", np.repeat(base, 2))
print("tile:", np.tile(base, 2))

matrix = np.array([[1, 2], [3, 4]])

print("Tile matrix:")
print(np.tile(matrix, (2, 3)))

print(
    """
np.repeat repeats individual elements or values according to its axis
rules.

np.tile repeats the entire array pattern.

These operations explicitly create repeated data, unlike broadcasting,
which generally avoids materializing conceptual repetitions.
"""
)


# ============================================================================
# SECTION 49: SEARCH AND FILTERING PRACTICAL EXAMPLE
# ============================================================================

section("48. Practical filtering example")

transaction_amounts = np.array(
    [250, 1200, 450, 3200, 75, 1800, 600],
    dtype=float,
)

large_transactions = transaction_amounts[
    transaction_amounts >= 1000
]

print("Transaction amounts:", transaction_amounts)
print("Transactions >= 1000:", large_transactions)

large_transaction_indexes = np.flatnonzero(
    transaction_amounts >= 1000
)

print("Indexes of large transactions:", large_transaction_indexes)

print(
    """
Boolean masks are suitable for filtering.

np.flatnonzero returns the indexes of nonzero or True elements as a
one-dimensional array.
"""
)


# ============================================================================
# SECTION 50: INDEXING MULTIDIMENSIONAL DATA WITH MASKS
# ============================================================================

section("49. Boolean masking in multidimensional arrays")

temperature_grid = np.array(
    [
        [18, 22, 27],
        [25, 31, 29],
        [16, 19, 33],
    ]
)

hot_mask = temperature_grid >= 30

print("Temperature grid:")
print(temperature_grid)
print("Hot mask:")
print(hot_mask)
print("Hot values:", temperature_grid[hot_mask])

temperature_grid_clipped = temperature_grid.copy()
temperature_grid_clipped[temperature_grid_clipped > 30] = 30

print("Capped temperature grid:")
print(temperature_grid_clipped)


# ============================================================================
# SECTION 51: MULTIDIMENSIONAL INTEGER INDEXING
# ============================================================================

section("50. Selecting coordinates")

grid = np.arange(25).reshape(5, 5)

rows = np.array([0, 1, 3])
columns = np.array([4, 2, 1])

selected_coordinates = grid[rows, columns]

print("Grid:")
print(grid)
print("Rows:", rows)
print("Columns:", columns)
print("Selected coordinates:", selected_coordinates)

print(
    """
When integer arrays are supplied for multiple dimensions, NumPy pairs
corresponding indexes.

For:

    rows    = [0, 1, 3]
    columns = [4, 2, 1]

the selected positions are:

    (0, 4)
    (1, 2)
    (3, 1)
"""
)


# ============================================================================
# SECTION 52: BASIC STATISTICAL CALCULATIONS
# ============================================================================

section("51. Basic statistical calculations")

observations = np.array(
    [12, 15, 14, 18, 21, 17, 16, 20],
    dtype=float,
)

mean = np.mean(observations)
median = np.median(observations)
variance_population = np.var(observations)
std_population = np.std(observations)

print("Observations:", observations)
print("Mean:", mean)
print("Median:", median)
print("Population variance:", variance_population)
print("Population standard deviation:", std_population)

sample_variance = np.var(observations, ddof=1)
sample_std = np.std(observations, ddof=1)

print("Sample variance:", sample_variance)
print("Sample standard deviation:", sample_std)

print(
    """
The ddof parameter changes the divisor used by variance and standard
deviation.

For example:

    ddof=0 -> population-style calculation
    ddof=1 -> sample-style calculation

Choosing the correct definition depends on the statistical context.
"""
)


# ============================================================================
# SECTION 53: CUMULATIVE OPERATIONS
# ============================================================================

section("52. Cumulative operations")

daily_sales = np.array([100, 150, 80, 200, 120])

print("Daily sales:", daily_sales)
print("Cumulative sales:", np.cumsum(daily_sales))
print("Cumulative maximum:", np.maximum.accumulate(daily_sales))

daily_returns = np.array([0.02, -0.01, 0.03, 0.015])

growth_factor = np.cumprod(1 + daily_returns)

print("Daily returns:", daily_returns)
print("Cumulative growth factors:", growth_factor)

print(
    """
Cumulative operations process values progressively along an axis.

Examples:

    np.cumsum
    np.cumprod
    np.maximum.accumulate
    np.minimum.accumulate

These are useful for running totals, cumulative products, and running
extreme values.
"""
)


# ============================================================================
# SECTION 54: DIFFERENCES
# ============================================================================

section("53. Differences between consecutive values")

prices = np.array([100, 103, 101, 108, 110])

print("Prices:", prices)
print("First differences:", np.diff(prices))

print(
    """
np.diff computes differences between neighboring values.

For:

    [100, 103, 101]

the result is:

    [3, -2]
"""
)


# ============================================================================
# SECTION 55: DEGREE/RADIAN CONVERSION
# ============================================================================

section("54. Numerical example with angle conversion")

degrees = np.array([0, 30, 45, 90, 180])

radians = np.deg2rad(degrees)

print("Degrees:", degrees)
print("Radians:", radians)
print("Sine:", np.sin(radians))
print("Converted back to degrees:", np.rad2deg(radians))

print(
    """
NumPy trigonometric functions expect radians.

np.deg2rad converts degrees to radians.
np.rad2deg converts radians to degrees.
"""
)


# ============================================================================
# SECTION 56: NUMERICAL PRECISION
# ============================================================================

section("55. Numerical precision")

float32_values = np.array(
    [1 / 3, 1 / 7, 1 / 11],
    dtype=np.float32,
)

float64_values = np.array(
    [1 / 3, 1 / 7, 1 / 11],
    dtype=np.float64,
)

print("float32:", float32_values)
print("float64:", float64_values)

print(
    """
float32 uses less memory than float64 but provides fewer significant
digits.

The correct dtype depends on the required precision, memory budget,
hardware, and numerical stability of the algorithm.

Using float64 by default is often reasonable for general scientific
computing, while lower precision can be valuable in memory- or
throughput-sensitive workloads.
"""
)


# ============================================================================
# SECTION 57: A SMALL ARRAY-BASED DATA PIPELINE
# ============================================================================

section("56. Complete NumPy data-processing example")

# Each row is a student.
# Columns represent mathematics, statistics, and programming scores.
student_scores = np.array(
    [
        [82, 75, 91],
        [65, 70, 68],
        [90, 88, 95],
        [55, 62, 58],
        [74, 81, 79],
    ],
    dtype=float,
)

subject_means = student_scores.mean(axis=0)
student_means = student_scores.mean(axis=1)

passed_students = student_means >= 60

normalized_scores = (
    student_scores - subject_means
) / student_scores.std(axis=0)

print("Student scores:")
print(student_scores)

print("Subject means:")
print(subject_means)

print("Student means:")
print(student_means)

print("Pass mask:")
print(passed_students)

print("Passing students:")
print(student_scores[passed_students])

print("Subject-standardized scores:")
print(normalized_scores)

print(
    """
This small pipeline combines several core concepts:

    1. A 2D array represents observations and features.
    2. axis=0 calculates statistics per feature.
    3. axis=1 calculates statistics per observation.
    4. Broadcasting subtracts feature means from every row.
    5. Boolean indexing filters passing observations.
    6. Vectorized arithmetic performs transformations without explicit
       Python loops.
"""
)


# ============================================================================
# SECTION 58: TESTING CORE BEHAVIOR
# ============================================================================

section("57. Built-in tests")

subsection("57.1 Shape tests")

test_array = np.arange(12).reshape(3, 4)

assert test_array.ndim == 2
assert test_array.shape == (3, 4)
assert test_array.size == 12

subsection("57.2 Indexing tests")

assert test_array[0, 0] == 0
assert test_array[-1, -1] == 11

subsection("57.3 Slicing tests")

assert_equal_arrays(
    test_array[0],
    np.array([0, 1, 2, 3]),
)

assert_equal_arrays(
    test_array[:, 1],
    np.array([1, 5, 9]),
)

subsection("57.4 Broadcasting tests")

broadcast_test = np.ones((2, 3)) + np.array([1, 2, 3])

assert_equal_arrays(
    broadcast_test,
    np.array(
        [
            [2, 3, 4],
            [2, 3, 4],
        ]
    ),
)

subsection("57.5 Mathematical tests")

assert np.isclose(np.sqrt(9), 3)
assert np.isclose(np.sin(np.pi / 2), 1)
assert np.isclose(np.mean([1, 2, 3]), 2)

print("All core tests passed.")


# ============================================================================
# SECTION 59: MINI PROJECT
# ============================================================================

section("58. Mini project: monthly business analysis")

monthly_revenue = np.array(
    [
        [12000, 14000, 13500, 15000, 16000, 17500],
        [9000, 9500, 11000, 10500, 12500, 13000],
        [15000, 15200, 16000, 17000, 18000, 19000],
    ],
    dtype=float,
)

product_names = np.array(
    [
        "Product A",
        "Product B",
        "Product C",
    ]
)

month_names = np.array(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
    ]
)

product_totals = monthly_revenue.sum(axis=1)
monthly_totals = monthly_revenue.sum(axis=0)
best_product_index = np.argmax(product_totals)
best_month_index = np.argmax(monthly_totals)

growth = np.divide(
    monthly_totals[1:] - monthly_totals[:-1],
    monthly_totals[:-1],
    out=np.zeros_like(monthly_totals[1:]),
    where=monthly_totals[:-1] != 0,
) * 100

print("Revenue matrix:")
print(monthly_revenue)

print("Product totals:")
for name, total in zip(product_names, product_totals):
    print(f"  {name}: {total:,.2f}")

print("Monthly totals:")
for month, total in zip(month_names, monthly_totals):
    print(f"  {month}: {total:,.2f}")

print(
    "Best product:",
    product_names[best_product_index],
    f"({product_totals[best_product_index]:,.2f})",
)

print(
    "Best month:",
    month_names[best_month_index],
    f"({monthly_totals[best_month_index]:,.2f})",
)

print("Month-over-month growth percentages:")
for month, percentage in zip(month_names[1:], growth):
    print(f"  {month}: {percentage:.2f}%")

print(
    """
The example uses a 2D array as a compact representation of business
data and demonstrates:

    - axis-based aggregation
    - argmax
    - vectorized percentage calculations
    - broadcasting-compatible shapes
    - Boolean-safe division
    - array indexing
"""
)


# ============================================================================
# SECTION 60: PRODUCTION CONSIDERATIONS
# ============================================================================

section("59. Production considerations")

print(
    """
When NumPy is used in production numerical software, several concerns
deserve explicit attention.

Correctness
-----------
Validate shape, dtype, units, missing values, and expected numerical
ranges at system boundaries.

Performance
-----------
Prefer vectorized operations where they improve the workload. Measure
real workloads rather than assuming that every vectorized expression is
faster.

Memory
------
Large arrays can consume substantial RAM. Watch .nbytes, avoid unnecessary
copies, choose dtypes deliberately, and process data in chunks when a
complete dataset cannot comfortably fit in memory.

Numerical stability
-------------------
Floating-point arithmetic is approximate. Be cautious with equality
comparisons, subtraction of nearly equal numbers, overflow, underflow,
and poorly conditioned mathematical problems.

Views and mutation
------------------
Know whether an operation returns a view or copy. Unexpected mutation
is a common source of difficult bugs.

Input validation
----------------
Do not trust external numerical input. Validate dimensions, finite
values, ranges, and expected dtypes before performing sensitive
calculations.

Security
--------
NumPy is a numerical library rather than a security boundary. Avoid
loading untrusted serialized array data blindly, especially formats or
workflows capable of object deserialization. Validate data sources and
use safe serialization practices.

Reproducibility
---------------
Use controlled random-number generators and explicit seeds when
reproducibility is required.

Interoperability
----------------
Pay attention to dtype, byte order, shape, memory order, and ownership
when passing arrays between Python, native code, databases, files, or
other numerical systems.
"""
)


# ============================================================================
# SECTION 61: CONCEPTUAL REFERENCE TABLE
# ============================================================================

section("60. Core NumPy reference")

reference = {
    "Create array": "np.array(...)",
    "Zeros": "np.zeros(shape)",
    "Ones": "np.ones(shape)",
    "Range": "np.arange(start, stop, step)",
    "Even spacing": "np.linspace(start, stop, count)",
    "Dimensions": "array.ndim",
    "Shape": "array.shape",
    "Element count": "array.size",
    "Data type": "array.dtype",
    "Index": "array[index]",
    "2D index": "array[row, column]",
    "Slice": "array[start:stop:step]",
    "Boolean filter": "array[mask]",
    "Reshape": "array.reshape(...)",
    "Transpose": "array.T",
    "Flatten copy": "array.flatten()",
    "Flatten view when possible": "array.ravel()",
    "Sum": "np.sum(array)",
    "Mean": "np.mean(array)",
    "Standard deviation": "np.std(array)",
    "Minimum": "np.min(array)",
    "Maximum": "np.max(array)",
    "Unique values": "np.unique(array)",
    "Locations": "np.where(condition)",
    "Sort copy": "np.sort(array)",
    "Sorting indexes": "np.argsort(array)",
    "Concatenate": "np.concatenate(...)",
    "Stack": "np.stack(...)",
    "Add axis": "np.expand_dims(...)",
    "Remove singleton axes": "np.squeeze(...)",
    "Element-wise multiplication": "A * B",
    "Matrix multiplication": "A @ B",
    "Square root": "np.sqrt(array)",
    "Exponential": "np.exp(array)",
    "Natural logarithm": "np.log(array)",
    "Degrees to radians": "np.deg2rad(array)",
    "Finite check": "np.isfinite(array)",
    "NaN check": "np.isnan(array)",
    "Clip values": "np.clip(array, minimum, maximum)",
}

for concept, syntax in reference.items():
    print(f"{concept:32} -> {syntax}")


# ============================================================================
# SECTION 62: FINAL VALIDATION
# ============================================================================

section("61. Final validation")

final_demo = np.arange(1, 13).reshape(3, 4)

print("Final demonstration array:")
print(final_demo)

print("Shape:", final_demo.shape)
print("Dimensions:", final_demo.ndim)
print("Total elements:", final_demo.size)

assert final_demo.shape == (3, 4)
assert final_demo.ndim == 2
assert final_demo.size == 12

# Verify indexing.
assert final_demo[0, 0] == 1
assert final_demo[-1, -1] == 12

# Verify slicing.
assert_equal_arrays(
    final_demo[:, 0],
    np.array([1, 5, 9]),
)

# Verify broadcasting.
broadcast_result = final_demo + np.array([10, 20, 30, 40])

assert_equal_arrays(
    broadcast_result,
    np.array(
        [
            [11, 22, 33, 44],
            [15, 26, 37, 48],
            [19, 30, 41, 52],
        ]
    ),
)

# Verify reshape.
assert_equal_arrays(
    final_demo.reshape(4, 3),
    np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
    ),
)

print("Final validation tests passed.")
print(
    """
NumPy fundamentals demonstrated successfully:

    arrays
    dimensions
    axes
    shapes
    sizes
    dtypes
    array construction
    indexing
    slicing
    views and copies
    Boolean masking
    fancy indexing
    arithmetic
    universal functions
    aggregations
    reshaping
    transpose
    broadcasting
    newaxis
    stacking
    splitting
    sorting
    searching
    filtering
    missing-value handling
    numerical precision
    vectorization
    memory behavior
    matrix operations
    validation
    testing
    practical numerical pipelines
"""
)


if __name__ == "__main__":
    print("\nNumPy fundamentals study script completed successfully.")
