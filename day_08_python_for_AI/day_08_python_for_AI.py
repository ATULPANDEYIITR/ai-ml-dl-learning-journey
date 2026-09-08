"""
Python for AI
Topic: Variables, Data Types, Operators, Conditions, Loops, and Functions

This self-contained script teaches essential Python foundations used in
Artificial Intelligence, Machine Learning, Data Analysis, Automation,
and scientific programming.

Run with:
    python python_for_ai_basics.py

The examples are organized progressively from beginner concepts to
more practical and advanced patterns.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable


# =============================================================================
# 1. VARIABLES
# =============================================================================

print("\n" + "=" * 80)
print("1. VARIABLES")
print("=" * 80)

# A variable is a name that refers to an object in memory.
student_name = "Atul"
age = 33
model_accuracy = 0.94

print("Student:", student_name)
print("Age:", age)
print("Model accuracy:", model_accuracy)

# Python is dynamically typed.
# A variable can refer to objects of different types during execution.
value = 10
print("\nInitial value:", value, "| Type:", type(value).__name__)

value = "ten"
print("Reassigned value:", value, "| Type:", type(value).__name__)

# Multiple assignment.
learning_rate, batch_size, epochs = 0.001, 32, 10

print("\nLearning rate:", learning_rate)
print("Batch size:", batch_size)
print("Epochs:", epochs)

# Assigning one value to multiple names.
feature_enabled = logging_enabled = True

print("\nFeature enabled:", feature_enabled)
print("Logging enabled:", logging_enabled)

# Variable names should be descriptive.
number_of_training_examples = 1000
number_of_classes = 5

print("\nTraining examples:", number_of_training_examples)
print("Classes:", number_of_classes)

# Python naming conventions:
# snake_case -> variables and functions
# PascalCase -> classes
# UPPER_CASE -> constants by convention

MAX_EPOCHS = 100

print("Maximum epochs:", MAX_EPOCHS)


# =============================================================================
# 2. OBJECT IDENTITY, MUTABILITY, AND ASSIGNMENT
# =============================================================================

print("\n" + "=" * 80)
print("2. OBJECT IDENTITY, MUTABILITY, AND ASSIGNMENT")
print("=" * 80)

# Assignment does not necessarily create a copy.
training_data = [1, 2, 3]
another_reference = training_data

another_reference.append(4)

print("training_data:", training_data)
print("another_reference:", another_reference)

# Both names refer to the same list object.
print("Same object:", training_data is another_reference)

# Immutable objects cannot be modified in place.
number = 10
another_number = number

number = 20

print("\nnumber:", number)
print("another_number:", another_number)

# Copying a list creates a new list.
original_features = [1.0, 2.0, 3.0]
copied_features = original_features.copy()

copied_features.append(4.0)

print("\nOriginal features:", original_features)
print("Copied features:", copied_features)


# =============================================================================
# 3. BASIC DATA TYPES
# =============================================================================

print("\n" + "=" * 80)
print("3. BASIC DATA TYPES")
print("=" * 80)

# ---------------------------------------------------------------------------
# 3.1 INTEGER
# ---------------------------------------------------------------------------

number_of_layers = 12
number_of_samples = -50

print("\nInteger examples:")
print(number_of_layers, type(number_of_layers).__name__)
print(number_of_samples, type(number_of_samples).__name__)

# Python integers can represent arbitrarily large integers,
# limited mainly by available memory.
large_number = 10 ** 100
print("Large integer:", large_number)


# ---------------------------------------------------------------------------
# 3.2 FLOAT
# ---------------------------------------------------------------------------

learning_rate = 0.001
loss = 0.245
negative_value = -3.14

print("\nFloat examples:")
print(learning_rate, type(learning_rate).__name__)
print(loss, type(loss).__name__)
print(negative_value, type(negative_value).__name__)

# Floating-point arithmetic can contain precision limitations.
precision_example = 0.1 + 0.2
print("\n0.1 + 0.2 =", precision_example)
print("0.1 + 0.2 == 0.3:", precision_example == 0.3)

# Use math.isclose when comparing floating-point values.
print(
    "Approximately equal:",
    math.isclose(precision_example, 0.3)
)


# ---------------------------------------------------------------------------
# 3.3 COMPLEX NUMBERS
# ---------------------------------------------------------------------------

complex_value = 3 + 4j

print("\nComplex number:", complex_value)
print("Real part:", complex_value.real)
print("Imaginary part:", complex_value.imag)


# ---------------------------------------------------------------------------
# 3.4 BOOLEAN
# ---------------------------------------------------------------------------

is_model_trained = True
is_data_available = False

print("\nBoolean examples:")
print(is_model_trained)
print(is_data_available)

# Booleans are subclasses of integers in Python.
print("True as integer:", int(True))
print("False as integer:", int(False))


# ---------------------------------------------------------------------------
# 3.5 STRING
# ---------------------------------------------------------------------------

model_name = "Neural Network"
dataset_name = 'Training Dataset'

print("\nString examples:")
print(model_name)
print(dataset_name)

# Strings are immutable.
message = "Python"
print("Length:", len(message))
print("First character:", message[0])
print("Last character:", message[-1])

# Slicing syntax:
# sequence[start:stop:step]
print("First three characters:", message[:3])
print("Every second character:", message[::2])
print("Reversed:", message[::-1])

# String formatting.
accuracy = 0.95342

print("\nFormatted output:")
print(f"Model accuracy: {accuracy:.2%}")
print(f"Accuracy as decimal: {accuracy:.4f}")


# ---------------------------------------------------------------------------
# 3.6 NONE
# ---------------------------------------------------------------------------

# None represents the absence of a value.
prediction = None

print("\nPrediction:", prediction)
print("Prediction type:", type(prediction).__name__)

# Use "is None", not "== None".
if prediction is None:
    print("Prediction has not been generated yet.")


# =============================================================================
# 4. COLLECTION DATA TYPES
# =============================================================================

print("\n" + "=" * 80)
print("4. COLLECTION DATA TYPES")
print("=" * 80)

# ---------------------------------------------------------------------------
# 4.1 LIST
# ---------------------------------------------------------------------------

features = [1.2, 3.4, 5.6]

print("\nList:", features)
print("First feature:", features[0])

# Lists are mutable.
features.append(7.8)
features[0] = 9.9

print("Modified list:", features)

# Common list operations.
features.extend([10.0, 11.0])
removed_feature = features.pop()

print("Extended list:", features)
print("Removed feature:", removed_feature)


# ---------------------------------------------------------------------------
# 4.2 TUPLE
# ---------------------------------------------------------------------------

image_dimensions = (1920, 1080)

print("\nTuple:", image_dimensions)
print("Width:", image_dimensions[0])
print("Height:", image_dimensions[1])

# Tuple unpacking.
width, height = image_dimensions
print("Unpacked:", width, height)

# Tuples are immutable.


# ---------------------------------------------------------------------------
# 4.3 DICTIONARY
# ---------------------------------------------------------------------------

model = {
    "name": "Classifier",
    "accuracy": 0.94,
    "epochs": 10,
}

print("\nDictionary:", model)
print("Model name:", model["name"])

model["accuracy"] = 0.96
model["optimizer"] = "gradient_descent"

print("Updated dictionary:", model)

# get() avoids KeyError when a key may not exist.
print("Missing value:", model.get("learning_rate"))
print("Default value:", model.get("learning_rate", 0.001))

print("\nDictionary keys:", list(model.keys()))
print("Dictionary values:", list(model.values()))

for key, value in model.items():
    print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# 4.4 SET
# ---------------------------------------------------------------------------

predicted_classes = [1, 2, 2, 3, 3, 3, 4]
unique_classes = set(predicted_classes)

print("\nOriginal values:", predicted_classes)
print("Unique values:", unique_classes)

set_a = {1, 2, 3}
set_b = {3, 4, 5}

print("Union:", set_a | set_b)
print("Intersection:", set_a & set_b)
print("Difference:", set_a - set_b)


# =============================================================================
# 5. TYPE CONVERSION
# =============================================================================

print("\n" + "=" * 80)
print("5. TYPE CONVERSION")
print("=" * 80)

integer_value = int("42")
float_value = float("3.14")
string_value = str(100)
boolean_value = bool(1)

print(integer_value, type(integer_value).__name__)
print(float_value, type(float_value).__name__)
print(string_value, type(string_value).__name__)
print(boolean_value, type(boolean_value).__name__)

# Invalid conversion raises ValueError.
invalid_values = ["42", "not_a_number", "3.14"]

for item in invalid_values:
    try:
        converted = int(item)
        print(f"Converted {item!r} to {converted}")
    except ValueError as error:
        print(f"Cannot convert {item!r} to int: {error}")


# =============================================================================
# 6. OPERATORS
# =============================================================================

print("\n" + "=" * 80)
print("6. OPERATORS")
print("=" * 80)

# ---------------------------------------------------------------------------
# 6.1 ARITHMETIC OPERATORS
# ---------------------------------------------------------------------------

a = 10
b = 3

print("\nArithmetic operators:")
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a // b =", a // b)
print("a % b =", a % b)
print("a ** b =", a ** b)


# ---------------------------------------------------------------------------
# 6.2 COMPARISON OPERATORS
# ---------------------------------------------------------------------------

accuracy_a = 0.91
accuracy_b = 0.89

print("\nComparison operators:")
print("accuracy_a == accuracy_b:", accuracy_a == accuracy_b)
print("accuracy_a != accuracy_b:", accuracy_a != accuracy_b)
print("accuracy_a > accuracy_b:", accuracy_a > accuracy_b)
print("accuracy_a < accuracy_b:", accuracy_a < accuracy_b)
print("accuracy_a >= accuracy_b:", accuracy_a >= accuracy_b)
print("accuracy_a <= accuracy_b:", accuracy_a <= accuracy_b)


# ---------------------------------------------------------------------------
# 6.3 LOGICAL OPERATORS
# ---------------------------------------------------------------------------

data_ready = True
model_ready = True
gpu_available = False

print("\nLogical operators:")
print("AND:", data_ready and model_ready)
print("OR:", model_ready or gpu_available)
print("NOT:", not gpu_available)

# Short-circuit behavior.
def expensive_check() -> bool:
    print("Expensive check executed")
    return True


print("\nShort-circuit demonstration:")

result = False and expensive_check()
print("False and function:", result)

result = True or expensive_check()
print("True or function:", result)


# ---------------------------------------------------------------------------
# 6.4 MEMBERSHIP OPERATORS
# ---------------------------------------------------------------------------

labels = ["cat", "dog", "bird"]

print("\nMembership operators:")
print("'cat' in labels:", "cat" in labels)
print("'fish' not in labels:", "fish" not in labels)


# ---------------------------------------------------------------------------
# 6.5 IDENTITY OPERATORS
# ---------------------------------------------------------------------------

first_list = [1, 2, 3]
second_list = [1, 2, 3]
third_list = first_list

print("\nEquality:", first_list == second_list)
print("Identity:", first_list is second_list)
print("Same reference:", first_list is third_list)


# ---------------------------------------------------------------------------
# 6.6 BITWISE OPERATORS
# ---------------------------------------------------------------------------

x = 6   # Binary: 110
y = 3   # Binary: 011

print("\nBitwise operators:")
print("x & y =", x & y)
print("x | y =", x | y)
print("x ^ y =", x ^ y)
print("~x =", ~x)
print("x << 1 =", x << 1)
print("x >> 1 =", x >> 1)


# =============================================================================
# 7. OPERATOR PRECEDENCE
# =============================================================================

print("\n" + "=" * 80)
print("7. OPERATOR PRECEDENCE")
print("=" * 80)

result_without_parentheses = 2 + 3 * 4
result_with_parentheses = (2 + 3) * 4

print("2 + 3 * 4 =", result_without_parentheses)
print("(2 + 3) * 4 =", result_with_parentheses)

# Use parentheses when clarity is important.
loss = 10
regularization = 2
samples = 4

average_loss = (loss + regularization) / samples
print("Average loss:", average_loss)


# =============================================================================
# 8. CONDITIONAL STATEMENTS
# =============================================================================

print("\n" + "=" * 80)
print("8. CONDITIONAL STATEMENTS")
print("=" * 80)

model_accuracy = 0.93

if model_accuracy >= 0.95:
    print("Excellent model performance.")
elif model_accuracy >= 0.90:
    print("Good model performance.")
elif model_accuracy >= 0.75:
    print("Acceptable model performance.")
else:
    print("Model needs improvement.")


# Nested conditions.
dataset_size = 50000
gpu_available = True

if dataset_size > 10000:
    if gpu_available:
        print("Use GPU-based training.")
    else:
        print("Large dataset detected; CPU training may be slow.")
else:
    print("Small dataset detected.")


# Combining conditions.
accuracy = 0.92
loss = 0.08

if accuracy >= 0.90 and loss <= 0.10:
    print("Model meets both quality conditions.")


# Truthiness and falsiness.
values_to_test = [
    0,
    1,
    0.0,
    "",
    "text",
    [],
    [1],
    {},
    {"a": 1},
    None,
]

print("\nTruthiness:")

for value in values_to_test:
    print(f"{value!r:15} -> {bool(value)}")


# Ternary conditional expression.
score = 75
result = "Pass" if score >= 50 else "Fail"

print("\nTernary result:", result)


# =============================================================================
# 9. MATCH-CASE PATTERN MATCHING
# =============================================================================

print("\n" + "=" * 80)
print("9. MATCH-CASE PATTERN MATCHING")
print("=" * 80)

# Requires Python 3.10 or later.
algorithm = "classification"

match algorithm:
    case "classification":
        print("Predicting discrete categories.")
    case "regression":
        print("Predicting continuous numerical values.")
    case "clustering":
        print("Finding groups in unlabeled data.")
    case _:
        print("Unknown algorithm category.")


# =============================================================================
# 10. LOOPS
# =============================================================================

print("\n" + "=" * 80)
print("10. LOOPS")
print("=" * 80)

# ---------------------------------------------------------------------------
# 10.1 FOR LOOP
# ---------------------------------------------------------------------------

numbers = [1, 2, 3, 4, 5]

print("\nFor loop:")

for number in numbers:
    print(number)


# Iterating with range.
print("\nRange examples:")

for epoch in range(3):
    print("Epoch:", epoch)

for value in range(2, 10, 2):
    print("Even value:", value)


# enumerate() provides index and value.
labels = ["cat", "dog", "bird"]

print("\nEnumerate:")

for index, label in enumerate(labels):
    print(f"Index {index}: {label}")


# zip() iterates over multiple sequences.
feature_names = ["age", "salary", "experience"]
feature_values = [30, 50000, 5]

print("\nZip:")

for name, value in zip(feature_names, feature_values):
    print(f"{name}: {value}")


# Dictionary iteration.
metrics = {
    "accuracy": 0.95,
    "precision": 0.91,
    "recall": 0.89,
}

print("\nDictionary iteration:")

for metric, value in metrics.items():
    print(f"{metric}: {value:.2%}")


# ---------------------------------------------------------------------------
# 10.2 WHILE LOOP
# ---------------------------------------------------------------------------

print("\nWhile loop:")

epoch = 0

while epoch < 3:
    print("Training epoch:", epoch)
    epoch += 1


# Guard against accidental infinite loops.
counter = 0
maximum_iterations = 5

while counter < maximum_iterations:
    counter += 1

print("Loop completed safely.")


# ---------------------------------------------------------------------------
# 10.3 BREAK
# ---------------------------------------------------------------------------

print("\nBreak demonstration:")

for value in range(10):
    if value == 5:
        print("Stopping at:", value)
        break

    print("Processing:", value)


# ---------------------------------------------------------------------------
# 10.4 CONTINUE
# ---------------------------------------------------------------------------

print("\nContinue demonstration:")

for value in range(6):
    if value % 2 == 0:
        continue

    print("Odd value:", value)


# ---------------------------------------------------------------------------
# 10.5 PASS
# ---------------------------------------------------------------------------

for value in range(3):
    if value == 1:
        pass  # Explicitly performs no operation.
    else:
        print("Value:", value)


# ---------------------------------------------------------------------------
# 10.6 LOOP ELSE
# ---------------------------------------------------------------------------

print("\nLoop else demonstration:")

target = 7
values = [1, 3, 5, 7, 9]

for value in values:
    if value == target:
        print("Target found:", target)
        break
else:
    print("Target not found.")


# =============================================================================
# 11. NESTED LOOPS
# =============================================================================

print("\n" + "=" * 80)
print("11. NESTED LOOPS")
print("=" * 80)

matrix = [
    [1, 2, 3],
    [4, 5, 6],
]

for row_index, row in enumerate(matrix):
    for column_index, value in enumerate(row):
        print(
            f"matrix[{row_index}][{column_index}] = {value}"
        )


# =============================================================================
# 12. LOOP COMPLEXITY
# =============================================================================

print("\n" + "=" * 80)
print("12. LOOP COMPLEXITY")
print("=" * 80)

# A single loop over n items is generally O(n).
def sum_values(values: list[int]) -> int:
    total = 0

    for value in values:
        total += value

    return total


# A nested loop over two n-sized sequences is generally O(n^2).
def generate_pairs(values: list[int]) -> list[tuple[int, int]]:
    pairs = []

    for first in values:
        for second in values:
            pairs.append((first, second))

    return pairs


sample_values = [1, 2, 3]

print("Sum:", sum_values(sample_values))
print("Pairs:", generate_pairs(sample_values))


# =============================================================================
# 13. LIST COMPREHENSIONS
# =============================================================================

print("\n" + "=" * 80)
print("13. LIST COMPREHENSIONS")
print("=" * 80)

numbers = [1, 2, 3, 4, 5]

squared_numbers = [number ** 2 for number in numbers]
print("Squares:", squared_numbers)

even_numbers = [
    number
    for number in numbers
    if number % 2 == 0
]

print("Even numbers:", even_numbers)

# Equivalent traditional loop.
traditional_squares = []

for number in numbers:
    traditional_squares.append(number ** 2)

print("Traditional squares:", traditional_squares)


# =============================================================================
# 14. DICTIONARY AND SET COMPREHENSIONS
# =============================================================================

print("\n" + "=" * 80)
print("14. DICTIONARY AND SET COMPREHENSIONS")
print("=" * 80)

square_mapping = {
    number: number ** 2
    for number in range(5)
}

print("Dictionary comprehension:", square_mapping)

remainders = {
    number % 3
    for number in range(10)
}

print("Set comprehension:", remainders)


# =============================================================================
# 15. FUNCTIONS
# =============================================================================

print("\n" + "=" * 80)
print("15. FUNCTIONS")
print("=" * 80)


# ---------------------------------------------------------------------------
# 15.1 BASIC FUNCTION
# ---------------------------------------------------------------------------

def greet() -> None:
    print("Hello from a function.")


greet()


# ---------------------------------------------------------------------------
# 15.2 FUNCTION PARAMETERS
# ---------------------------------------------------------------------------

def greet_user(name: str) -> None:
    print(f"Hello, {name}!")


greet_user("Atul")


# ---------------------------------------------------------------------------
# 15.3 RETURN VALUES
# ---------------------------------------------------------------------------

def add_numbers(first: float, second: float) -> float:
    return first + second


result = add_numbers(10, 20)
print("Addition result:", result)


# ---------------------------------------------------------------------------
# 15.4 DEFAULT PARAMETERS
# ---------------------------------------------------------------------------

def train_model(
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> str:
    return (
        f"Training for {epochs} epochs "
        f"with learning rate {learning_rate}"
    )


print(train_model())
print(train_model(epochs=20))
print(train_model(learning_rate=0.01, epochs=5))


# ---------------------------------------------------------------------------
# 15.5 POSITIONAL AND KEYWORD ARGUMENTS
# ---------------------------------------------------------------------------

def create_model(
    name: str,
    layers: int,
    activation: str = "relu",
) -> dict[str, Any]:
    return {
        "name": name,
        "layers": layers,
        "activation": activation,
    }


print(
    create_model(
        "Classifier",
        3,
        "sigmoid",
    )
)

print(
    create_model(
        name="Regressor",
        layers=5,
    )
)


# =============================================================================
# 16. VARIABLE-LENGTH ARGUMENTS
# =============================================================================

print("\n" + "=" * 80)
print("16. VARIABLE-LENGTH ARGUMENTS")
print("=" * 80)


# *args collects additional positional arguments.
def calculate_average(*values: float) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


print("Average:", calculate_average(1, 2, 3, 4, 5))
print("Empty average:", calculate_average())


# **kwargs collects additional keyword arguments.
def describe_model(**configuration: Any) -> None:
    for key, value in configuration.items():
        print(f"{key}: {value}")


describe_model(
    name="Neural Network",
    layers=5,
    optimizer="gradient_descent",
)


# Combining normal parameters, *args, and **kwargs.
def flexible_function(
    required: str,
    *values: int,
    **options: Any,
) -> None:
    print("Required:", required)
    print("Values:", values)
    print("Options:", options)


flexible_function(
    "model",
    1,
    2,
    3,
    learning_rate=0.001,
    epochs=10,
)


# =============================================================================
# 17. FUNCTION TYPE HINTS
# =============================================================================

print("\n" + "=" * 80)
print("17. FUNCTION TYPE HINTS")
print("=" * 80)

# Type hints improve readability and support static analysis tools.
# Python does not automatically enforce them at runtime.

def calculate_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list.")

    return sum(values) / len(values)


print("Mean:", calculate_mean([1.0, 2.0, 3.0]))


# =============================================================================
# 18. FUNCTION DOCUMENTATION
# =============================================================================

print("\n" + "=" * 80)
print("18. FUNCTION DOCUMENTATION")
print("=" * 80)


def normalize_value(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Normalize a numerical value to the range 0 through 1.

    Args:
        value: The value to normalize.
        minimum: The minimum value in the original range.
        maximum: The maximum value in the original range.

    Returns:
        The normalized value.

    Raises:
        ValueError: If maximum equals minimum.
    """

    if maximum == minimum:
        raise ValueError(
            "Maximum and minimum must be different."
        )

    return (value - minimum) / (maximum - minimum)


print(
    "Normalized:",
    normalize_value(50, 0, 100)
)


# =============================================================================
# 19. LOCAL AND GLOBAL SCOPE
# =============================================================================

print("\n" + "=" * 80)
print("19. LOCAL AND GLOBAL SCOPE")
print("=" * 80)

global_threshold = 0.5


def check_threshold(value: float) -> bool:
    # global_threshold is accessible for reading.
    local_result = value >= global_threshold
    return local_result


print(check_threshold(0.7))


# Modifying global state should be avoided when possible because it
# makes functions harder to test and reason about.

counter = 0


def increment_counter() -> int:
    global counter
    counter += 1
    return counter


print("Counter:", increment_counter())
print("Counter:", increment_counter())


# Preferred design: pass data in and return new data.
def increment(value: int) -> int:
    return value + 1


print("Pure increment:", increment(10))


# =============================================================================
# 20. LAMBDA FUNCTIONS
# =============================================================================

print("\n" + "=" * 80)
print("20. LAMBDA FUNCTIONS")
print("=" * 80)

# Lambda functions are anonymous expressions.
square = lambda number: number ** 2

print("Square:", square(5))

values = [5, 1, 4, 2, 3]

sorted_values = sorted(
    values,
    key=lambda value: value,
)

print("Sorted values:", sorted_values)


# =============================================================================
# 21. HIGHER-ORDER FUNCTIONS
# =============================================================================

print("\n" + "=" * 80)
print("21. HIGHER-ORDER FUNCTIONS")
print("=" * 80)

# A higher-order function accepts another function or returns one.

def apply_transformation(
    values: list[float],
    transformation: Callable[[float], float],
) -> list[float]:
    return [
        transformation(value)
        for value in values
    ]


def square_value(value: float) -> float:
    return value ** 2


def cube_value(value: float) -> float:
    return value ** 3


values = [1.0, 2.0, 3.0]

print(
    "Squared:",
    apply_transformation(
        values,
        square_value,
    )
)

print(
    "Cubed:",
    apply_transformation(
        values,
        cube_value,
    )
)


# =============================================================================
# 22. RECURSION
# =============================================================================

print("\n" + "=" * 80)
print("22. RECURSION")
print("=" * 80)


def factorial(number: int) -> int:
    """
    Calculate factorial recursively.

    Factorial definition:
        n! = n * (n - 1)!
        0! = 1
    """

    if number < 0:
        raise ValueError(
            "Factorial is not defined for negative integers."
        )

    if number in (0, 1):
        return 1

    return number * factorial(number - 1)


print("5! =", factorial(5))


# Iterative factorial avoids recursion-depth limitations.
def iterative_factorial(number: int) -> int:
    if number < 0:
        raise ValueError(
            "Factorial is not defined for negative integers."
        )

    result = 1

    for value in range(2, number + 1):
        result *= value

    return result


print("Iterative 5! =", iterative_factorial(5))


# =============================================================================
# 23. EXCEPTION HANDLING
# =============================================================================

print("\n" + "=" * 80)
print("23. EXCEPTION HANDLING")
print("=" * 80)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    try:
        return numerator / denominator

    except ZeroDivisionError:
        print("Error: division by zero.")
        return None


print("10 / 2 =", safe_divide(10, 2))
print("10 / 0 =", safe_divide(10, 0))


# Handling multiple exception types.
def convert_to_integer(value: Any) -> int | None:
    try:
        return int(value)

    except (ValueError, TypeError) as error:
        print(
            f"Conversion failed for {value!r}: {error}"
        )
        return None


print(convert_to_integer("42"))
print(convert_to_integer("invalid"))
print(convert_to_integer(None))


# else executes when no exception occurs.
# finally executes regardless of success or failure.
def demonstrate_exception_flow(value: str) -> None:
    try:
        converted = int(value)

    except ValueError:
        print("Conversion failed.")

    else:
        print("Conversion succeeded:", converted)

    finally:
        print("Operation completed.")


demonstrate_exception_flow("10")
demonstrate_exception_flow("not_a_number")


# =============================================================================
# 24. ASSERTIONS
# =============================================================================

print("\n" + "=" * 80)
print("24. ASSERTIONS")
print("=" * 80)


def calculate_probability(
    successful_events: int,
    total_events: int,
) -> float:
    assert total_events > 0, (
        "Total events must be greater than zero."
    )

    assert successful_events >= 0, (
        "Successful events cannot be negative."
    )

    assert successful_events <= total_events, (
        "Successful events cannot exceed total events."
    )

    return successful_events / total_events


print(
    "Probability:",
    calculate_probability(8, 10)
)


# Assertions are useful for detecting programmer assumptions.
# They should not be the only validation mechanism for untrusted input.


# =============================================================================
# 25. PRACTICAL AI-STYLE EXAMPLE: DATA VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("25. PRACTICAL EXAMPLE: DATA VALIDATION")
print("=" * 80)


def validate_numeric_dataset(
    values: list[Any],
) -> list[float]:
    """
    Validate and convert a dataset to floating-point values.

    Rules:
    - Input must be a list.
    - Boolean values are rejected because bool is a subclass of int.
    - Non-numeric values are rejected.
    - Empty datasets are rejected.
    """

    if not isinstance(values, list):
        raise TypeError(
            "Dataset must be provided as a list."
        )

    if not values:
        raise ValueError(
            "Dataset cannot be empty."
        )

    validated_values: list[float] = []

    for value in values:
        if isinstance(value, bool):
            raise TypeError(
                "Boolean values are not valid numerical samples."
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Invalid data type: {type(value).__name__}"
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                "Dataset contains NaN or infinity."
            )

        validated_values.append(numeric_value)

    return validated_values


valid_dataset = [1, 2.5, 3, 4.75]

print(
    "Validated dataset:",
    validate_numeric_dataset(valid_dataset)
)


# =============================================================================
# 26. PRACTICAL AI-STYLE EXAMPLE: FEATURE NORMALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("26. PRACTICAL EXAMPLE: FEATURE NORMALIZATION")
print("=" * 80)


def min_max_normalize(
    values: list[float],
) -> list[float]:
    """
    Scale values to the range [0, 1].

    Edge cases:
    - Empty input raises ValueError.
    - Constant input returns zeros because there is no range.
    """

    if not values:
        raise ValueError(
            "Cannot normalize an empty dataset."
        )

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [0.0 for _ in values]

    normalized = []

    for value in values:
        normalized_value = (
            (value - minimum)
            / (maximum - minimum)
        )

        normalized.append(normalized_value)

    return normalized


raw_features = [10.0, 20.0, 30.0, 40.0]

print("Raw features:", raw_features)
print(
    "Normalized features:",
    min_max_normalize(raw_features)
)

constant_features = [5.0, 5.0, 5.0]

print(
    "Constant features:",
    min_max_normalize(constant_features)
)


# =============================================================================
# 27. PRACTICAL AI-STYLE EXAMPLE: CLASSIFICATION RULE
# =============================================================================

print("\n" + "=" * 80)
print("27. PRACTICAL EXAMPLE: CLASSIFICATION RULE")
print("=" * 80)


def classify_score(score: float) -> str:
    """
    Convert a numerical score into a category.

    Valid range: 0 through 100.
    """

    if not 0 <= score <= 100:
        raise ValueError(
            "Score must be between 0 and 100."
        )

    if score >= 90:
        return "excellent"

    if score >= 75:
        return "good"

    if score >= 50:
        return "acceptable"

    return "poor"


test_scores = [95, 82, 60, 40]

for score in test_scores:
    category = classify_score(score)
    print(f"Score {score}: {category}")


# =============================================================================
# 28. PRACTICAL AI-STYLE EXAMPLE: THRESHOLD CLASSIFIER
# =============================================================================

print("\n" + "=" * 80)
print("28. PRACTICAL EXAMPLE: THRESHOLD CLASSIFIER")
print("=" * 80)


def binary_classifier(
    probability: float,
    threshold: float = 0.5,
) -> int:
    """
    Convert a probability into a binary class.

    Returns:
        1 when probability >= threshold.
        0 otherwise.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if probability >= threshold:
        return 1

    return 0


probabilities = [0.1, 0.49, 0.5, 0.7, 0.95]

for probability in probabilities:
    prediction = binary_classifier(
        probability,
        threshold=0.5,
    )

    print(
        f"Probability {probability:.2f} "
        f"-> Class {prediction}"
    )


# =============================================================================
# 29. PRACTICAL AI-STYLE EXAMPLE: TRAINING SIMULATION
# =============================================================================

print("\n" + "=" * 80)
print("29. PRACTICAL EXAMPLE: TRAINING SIMULATION")
print("=" * 80)


def simulate_training(
    initial_loss: float,
    epochs: int,
    improvement_rate: float,
) -> list[float]:
    """
    Simulate a training process.

    This is not a real machine-learning algorithm.
    It demonstrates variables, conditions, loops, functions,
    validation, and numerical processing.
    """

    if initial_loss <= 0:
        raise ValueError(
            "Initial loss must be greater than zero."
        )

    if epochs <= 0:
        raise ValueError(
            "Epochs must be greater than zero."
        )

    if not 0 < improvement_rate < 1:
        raise ValueError(
            "Improvement rate must be between 0 and 1."
        )

    loss = initial_loss
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        noise = random.uniform(-0.02, 0.02)

        loss *= improvement_rate + noise

        # Prevent negative loss due to simulation noise.
        loss = max(loss, 0.0)

        history.append(loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss:.6f}"
        )

        # Early stopping condition.
        if loss < 0.01:
            print("Early stopping: target loss reached.")
            break

    return history


random.seed(42)

loss_history = simulate_training(
    initial_loss=1.0,
    epochs=10,
    improvement_rate=0.8,
)

print("Recorded losses:", loss_history)


# =============================================================================
# 30. PRACTICAL AI-STYLE EXAMPLE: ACCURACY CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("30. PRACTICAL EXAMPLE: ACCURACY CALCULATION")
print("=" * 80)


def calculate_accuracy(
    predictions: list[Any],
    actual_labels: list[Any],
) -> float:
    """
    Calculate classification accuracy.

    Accuracy = correct predictions / total predictions
    """

    if not predictions:
        raise ValueError(
            "Predictions cannot be empty."
        )

    if len(predictions) != len(actual_labels):
        raise ValueError(
            "Predictions and actual labels must have equal length."
        )

    correct_predictions = 0

    for predicted, actual in zip(
        predictions,
        actual_labels,
    ):
        if predicted == actual:
            correct_predictions += 1

    return correct_predictions / len(predictions)


predictions = ["cat", "dog", "cat", "bird"]
actual_labels = ["cat", "dog", "bird", "bird"]

accuracy = calculate_accuracy(
    predictions,
    actual_labels,
)

print(f"Accuracy: {accuracy:.2%}")


# =============================================================================
# 31. PRACTICAL AI-STYLE EXAMPLE: CONFUSION MATRIX COUNTS
# =============================================================================

print("\n" + "=" * 80)
print("31. PRACTICAL EXAMPLE: BINARY CLASSIFICATION COUNTS")
print("=" * 80)


def binary_classification_counts(
    predictions: list[int],
    actual_labels: list[int],
) -> dict[str, int]:
    """
    Calculate:
    - True Positive
    - True Negative
    - False Positive
    - False Negative
    """

    if len(predictions) != len(actual_labels):
        raise ValueError(
            "Prediction and label lengths must match."
        )

    counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }

    for predicted, actual in zip(
        predictions,
        actual_labels,
    ):
        if predicted not in (0, 1):
            raise ValueError(
                "Predictions must contain only 0 or 1."
            )

        if actual not in (0, 1):
            raise ValueError(
                "Actual labels must contain only 0 or 1."
            )

        if predicted == 1 and actual == 1:
            counts["true_positive"] += 1

        elif predicted == 0 and actual == 0:
            counts["true_negative"] += 1

        elif predicted == 1 and actual == 0:
            counts["false_positive"] += 1

        else:
            counts["false_negative"] += 1

    return counts


predictions = [1, 0, 1, 1, 0, 0]
actual_labels = [1, 0, 0, 1, 1, 0]

counts = binary_classification_counts(
    predictions,
    actual_labels,
)

print("Classification counts:", counts)


# =============================================================================
# 32. FUNCTION DESIGN: PURE VS IMPURE FUNCTIONS
# =============================================================================

print("\n" + "=" * 80)
print("32. PURE VS IMPURE FUNCTIONS")
print("=" * 80)


# Pure function:
# Same input -> same output.
# Does not modify external state.
def multiply_by_two(value: float) -> float:
    return value * 2


# Impure function:
# Modifies external state.
training_log: list[str] = []


def log_training_event(event: str) -> None:
    training_log.append(event)


print("Pure result:", multiply_by_two(10))

log_training_event("Training started")
log_training_event("Epoch completed")

print("External log:", training_log)


# =============================================================================
# 33. MUTABLE DEFAULT ARGUMENT PITFALL
# =============================================================================

print("\n" + "=" * 80)
print("33. MUTABLE DEFAULT ARGUMENT PITFALL")
print("=" * 80)

# Incorrect pattern:
#
# def add_item(item, values=[]):
#     values.append(item)
#     return values
#
# The same list can be reused between calls.


# Correct pattern:
def add_item(
    item: Any,
    values: list[Any] | None = None,
) -> list[Any]:
    if values is None:
        values = []

    values.append(item)
    return values


print(add_item("first"))
print(add_item("second"))


# =============================================================================
# 34. SHALLOW VS DEEP COPYING
# =============================================================================

print("\n" + "=" * 80)
print("34. SHALLOW VS DEEP COPYING")
print("=" * 80)

# A shallow copy duplicates the outer container but may share nested objects.
original_matrix = [
    [1, 2],
    [3, 4],
]

shallow_copy = original_matrix.copy()

shallow_copy[0][0] = 999

print("Original after shallow nested modification:", original_matrix)
print("Shallow copy:", shallow_copy)

# Deep copying nested structures requires the copy module.
import copy

original_matrix = [
    [1, 2],
    [3, 4],
]

deep_copy = copy.deepcopy(original_matrix)

deep_copy[0][0] = 999

print("Original after deep copy modification:", original_matrix)
print("Deep copy:", deep_copy)


# =============================================================================
# 35. PERFORMANCE: LOOP VS COMPREHENSION
# =============================================================================

print("\n" + "=" * 80)
print("35. PERFORMANCE CONSIDERATIONS")
print("=" * 80)

large_values = list(range(10))

# Traditional loop.
loop_result = []

for value in large_values:
    loop_result.append(value * value)

# List comprehension.
comprehension_result = [
    value * value
    for value in large_values
]

print("Loop result:", loop_result)
print(
    "Comprehension result:",
    comprehension_result
)

# For numerical AI workloads involving very large arrays,
# specialized numerical libraries are usually much faster than
# Python-level loops because they execute optimized low-level operations.


# =============================================================================
# 36. DEBUGGING WITH PRINT AND VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("36. DEBUGGING CONSIDERATIONS")
print("=" * 80)


def debug_normalize(
    values: list[float],
) -> list[float]:
    if not values:
        raise ValueError("Values cannot be empty.")

    minimum = min(values)
    maximum = max(values)

    print("DEBUG minimum:", minimum)
    print("DEBUG maximum:", maximum)

    if minimum == maximum:
        return [0.0] * len(values)

    result = []

    for index, value in enumerate(values):
        normalized = (
            (value - minimum)
            / (maximum - minimum)
        )

        print(
            f"DEBUG index={index}, "
            f"value={value}, "
            f"normalized={normalized}"
        )

        result.append(normalized)

    return result


print(
    "Debug normalized:",
    debug_normalize([10, 20, 30])
)


# =============================================================================
# 37. SECURITY AND INPUT VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("37. SECURITY AND INPUT VALIDATION")
print("=" * 80)

# Never use eval() on untrusted user input.
# eval() can execute arbitrary Python expressions.


def safe_positive_integer(
    value: Any,
) -> int:
    """
    Validate a positive integer.

    Explicit validation is preferable to executing input dynamically.
    """

    if isinstance(value, bool):
        raise TypeError(
            "Boolean values are not accepted."
        )

    if not isinstance(value, int):
        raise TypeError(
            "Value must be an integer."
        )

    if value <= 0:
        raise ValueError(
            "Value must be positive."
        )

    return value


print(
    "Validated integer:",
    safe_positive_integer(10)
)


# =============================================================================
# 38. COMBINED PROJECT: SIMPLE MODEL PIPELINE
# =============================================================================

print("\n" + "=" * 80)
print("38. COMBINED PROJECT: SIMPLE MODEL PIPELINE")
print("=" * 80)


def validate_probabilities(
    probabilities: list[float],
) -> list[float]:
    """Validate probability values."""

    if not probabilities:
        raise ValueError(
            "Probability list cannot be empty."
        )

    validated: list[float] = []

    for probability in probabilities:
        if isinstance(probability, bool):
            raise TypeError(
                "Boolean values are not probabilities."
            )

        if not isinstance(
            probability,
            (int, float),
        ):
            raise TypeError(
                "Probability must be numeric."
            )

        probability = float(probability)

        if not math.isfinite(probability):
            raise ValueError(
                "Probability must be finite."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        validated.append(probability)

    return validated


def classify_probabilities(
    probabilities: list[float],
    threshold: float,
) -> list[int]:
    """Convert probabilities into binary predictions."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    predictions = []

    for probability in probabilities:
        prediction = (
            1
            if probability >= threshold
            else 0
        )

        predictions.append(prediction)

    return predictions


def evaluate_binary_predictions(
    predictions: list[int],
    actual_labels: list[int],
) -> dict[str, float]:
    """
    Calculate basic binary classification metrics.
    """

    if len(predictions) != len(actual_labels):
        raise ValueError(
            "Prediction and label lengths must match."
        )

    if not predictions:
        raise ValueError(
            "Predictions cannot be empty."
        )

    correct = 0

    for predicted, actual in zip(
        predictions,
        actual_labels,
    ):
        if predicted == actual:
            correct += 1

    accuracy = correct / len(predictions)

    counts = binary_classification_counts(
        predictions,
        actual_labels,
    )

    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


raw_probabilities = [
    0.95,
    0.12,
    0.72,
    0.48,
    0.81,
    0.35,
]

actual_labels = [
    1,
    0,
    1,
    0,
    1,
    0,
]

validated_probabilities = validate_probabilities(
    raw_probabilities
)

threshold = 0.50

predictions = classify_probabilities(
    validated_probabilities,
    threshold,
)

metrics = evaluate_binary_predictions(
    predictions,
    actual_labels,
)

print("Probabilities:", validated_probabilities)
print("Threshold:", threshold)
print("Predictions:", predictions)

for metric_name, metric_value in metrics.items():
    print(
        f"{metric_name.title()}: "
        f"{metric_value:.2%}"
    )


# =============================================================================
# 39. COMMON MISTAKES DEMONSTRATION
# =============================================================================

print("\n" + "=" * 80)
print("39. COMMON MISTAKES")
print("=" * 80)

# Mistake 1: Using = instead of == in a condition.
# "=" performs assignment.
# "==" compares values.

first_value = 10
second_value = 10

if first_value == second_value:
    print("Values are equal.")


# Mistake 2: Floating-point equality.
value = 0.1 + 0.2

if math.isclose(value, 0.3):
    print("Floating-point values are approximately equal.")


# Mistake 3: Forgetting that strings are immutable.
text = "model"

# text[0] = "M"  # This would raise TypeError.

corrected_text = "M" + text[1:]
print("Corrected string:", corrected_text)


# Mistake 4: Modifying a list while iterating over it.
values = [1, 2, 3, 4, 5]

filtered_values = [
    value
    for value in values
    if value % 2 != 0
]

print("Safely filtered values:", filtered_values)


# =============================================================================
# 40. FINAL INTEGRATED EXAMPLE
# =============================================================================

print("\n" + "=" * 80)
print("40. FINAL INTEGRATED EXAMPLE")
print("=" * 80)


def analyze_scores(
    scores: list[float],
    passing_score: float = 50.0,
) -> dict[str, Any]:
    """
    Analyze numerical scores using variables, data types, operators,
    conditions, loops, functions, validation, and dictionaries.
    """

    if not scores:
        raise ValueError(
            "Score list cannot be empty."
        )

    validated_scores: list[float] = []

    for score in scores:
        if not isinstance(
            score,
            (int, float),
        ) or isinstance(score, bool):
            raise TypeError(
                "Every score must be numeric."
            )

        score = float(score)

        if not math.isfinite(score):
            raise ValueError(
                "Scores must be finite."
            )

        validated_scores.append(score)

    total = sum(validated_scores)
    average = total / len(validated_scores)

    passed = 0
    failed = 0

    for score in validated_scores:
        if score >= passing_score:
            passed += 1
        else:
            failed += 1

    return {
        "scores": validated_scores,
        "count": len(validated_scores),
        "minimum": min(validated_scores),
        "maximum": max(validated_scores),
        "average": average,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(validated_scores),
    }


scores = [78, 92, 45, 61, 88, 39, 72]

analysis = analyze_scores(
    scores,
    passing_score=50,
)

print("Score analysis:")

for key, value in analysis.items():
    if key == "pass_rate":
        print(f"{key}: {value:.2%}")
    else:
        print(f"{key}: {value}")


print("\n" + "=" * 80)
print("END OF PYTHON FOR AI FUNDAMENTALS")
print("=" * 80)
