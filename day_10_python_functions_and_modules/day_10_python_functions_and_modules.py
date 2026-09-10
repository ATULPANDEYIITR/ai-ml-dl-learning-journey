"""
Python Functions, Lambda, Scope, Modules, Packages, and Exceptions
==================================================================

A self-contained study script covering Python functions and modules from
absolute beginner through advanced concepts.

The file is intentionally executable. Most demonstrations are implemented as
functions and classes, and the main() function runs a structured progression.

Covered areas
-------------
1. Functions and why they are useful
2. Defining and calling functions
3. Parameters and arguments
4. Positional and keyword arguments
5. Default parameters
6. Return values
7. Multiple return values
8. Variable-length arguments: *args and **kwargs
9. Argument unpacking
10. Positional-only and keyword-only parameters
11. Type hints
12. Docstrings
13. First-class functions
14. Higher-order functions
15. Lambda expressions
16. map(), filter(), and reduce()
17. Closures
18. Decorators
19. Recursion
20. Scope and the LEGB rule
21. global and nonlocal
22. Mutable default argument pitfalls
23. Function annotations
24. Modules and imports
25. Import styles
26. __name__ and __main__
27. Module namespaces
28. Packages and package structure
29. Exception terminology
30. try, except, else, finally
31. Catching specific exceptions
32. Multiple exceptions
33. Raising exceptions
34. Custom exceptions
35. Exception chaining
36. Context managers
37. Assertions
38. Validation
39. Logging
40. Testing functions
41. Function design and best practices
42. Performance and recursion limits
43. Security considerations
44. Practical application combining functions, modules, validation,
    exceptions, decorators, and higher-order functions

The script uses only the Python standard library.
"""

from __future__ import annotations

import functools
import math
import operator
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence


# ============================================================================
# 1. BASIC FUNCTIONS
# ============================================================================

def section(title: str) -> None:
    """Print a visual section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    """Print a smaller heading."""
    print(f"\n--- {title} ---")


def greet() -> None:
    """A function with no parameters and no explicit return value."""
    print("Hello from a function!")


def greet_person(name: str) -> None:
    """Accept one parameter and use it inside the function."""
    print(f"Hello, {name}!")


def add_numbers(first: float, second: float) -> float:
    """Return the sum of two numbers."""
    return first + second


def multiply_numbers(first: float, second: float) -> float:
    """Return the product of two numbers."""
    return first * second


def calculate_rectangle_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative.")
    return length * width


def calculate_rectangle_perimeter(length: float, width: float) -> float:
    """Calculate the perimeter of a rectangle."""
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative.")
    return 2 * (length + width)


# ============================================================================
# 2. PARAMETERS AND ARGUMENTS
# ============================================================================

def introduce(name: str, age: int, city: str = "Unknown") -> str:
    """
    Demonstrate required and default parameters.

    name and age are required.
    city has a default value.
    """
    return f"{name} is {age} years old and lives in {city}."


def power(base: float, exponent: float = 2) -> float:
    """Return base raised to exponent."""
    return base ** exponent


def divide(dividend: float, divisor: float) -> float:
    """Divide two numbers with explicit validation."""
    if divisor == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return dividend / divisor


def describe_person(name: str, age: int, *, city: str, occupation: str) -> str:
    """
    Demonstrate keyword-only parameters.

    The * means city and occupation must be passed by keyword.
    """
    return f"{name}, age {age}, city={city}, occupation={occupation}"


def create_point(x: float, y: float, /, *, label: str = "point") -> dict[str, Any]:
    """
    Demonstrate both positional-only and keyword-only parameters.

    x and y occur before / and therefore are positional-only.
    label occurs after * and therefore is keyword-only.
    """
    return {"x": x, "y": y, "label": label}


# ============================================================================
# 3. MULTIPLE RETURN VALUES AND UNPACKING
# ============================================================================

def calculate_statistics(numbers: Sequence[float]) -> tuple[float, float, float]:
    """Return minimum, maximum, and arithmetic mean."""
    if not numbers:
        raise ValueError("At least one number is required.")

    minimum = min(numbers)
    maximum = max(numbers)
    average = statistics.mean(numbers)

    return minimum, maximum, average


def quotient_and_remainder(dividend: int, divisor: int) -> tuple[int, int]:
    """Return quotient and remainder."""
    if divisor == 0:
        raise ZeroDivisionError("Divisor cannot be zero.")
    return divmod(dividend, divisor)


# ============================================================================
# 4. *ARGS AND **KWARGS
# ============================================================================

def sum_all(*numbers: float) -> float:
    """
    Accept any number of positional arguments.

    Inside the function, numbers is a tuple.
    """
    return sum(numbers)


def build_profile(**details: Any) -> dict[str, Any]:
    """
    Accept any number of keyword arguments.

    Inside the function, details is a dictionary.
    """
    return details


def summarize_order(customer: str, *items: str, **metadata: Any) -> dict[str, Any]:
    """
    Combine a normal parameter, *args, and **kwargs.

    The order is important:
    normal parameters -> *args -> **kwargs
    """
    return {
        "customer": customer,
        "items": list(items),
        "metadata": metadata,
    }


# ============================================================================
# 5. ARGUMENT UNPACKING
# ============================================================================

def rectangle(length: float, width: float) -> float:
    """Return rectangle area."""
    return length * width


def format_student(name: str, age: int, course: str) -> str:
    """Format student information."""
    return f"{name} | {age} | {course}"


# ============================================================================
# 6. TYPE HINTS AND DOCSTRINGS
# ============================================================================

def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    """
    Calculate a weighted average.

    Parameters
    ----------
    values:
        Numeric observations.
    weights:
        Weight associated with each observation.

    Returns
    -------
    float
        Weighted average.

    Raises
    ------
    ValueError
        If lengths differ, weights sum to zero, or input is empty.
    """
    if not values:
        raise ValueError("Values cannot be empty.")

    if len(values) != len(weights):
        raise ValueError("Values and weights must have equal lengths.")

    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("Weights cannot sum to zero.")

    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


# ============================================================================
# 7. FIRST-CLASS FUNCTIONS
# ============================================================================

def square(number: float) -> float:
    """Return the square of a number."""
    return number * number


def cube(number: float) -> float:
    """Return the cube of a number."""
    return number ** 3


def apply_operation(
    operation: Callable[[float], float],
    value: float,
) -> float:
    """
    Accept a function as an argument.

    Functions are objects in Python, so they can be passed to other functions.
    """
    return operation(value)


def choose_operation(name: str) -> Callable[[float], float]:
    """Return a function based on a textual operation name."""
    operations: dict[str, Callable[[float], float]] = {
        "square": square,
        "cube": cube,
        "absolute": abs,
        "sqrt": math.sqrt,
    }

    try:
        return operations[name]
    except KeyError as error:
        raise ValueError(f"Unknown operation: {name}") from error


# ============================================================================
# 8. LAMBDA EXPRESSIONS
# ============================================================================

def demonstrate_lambdas() -> None:
    """Show practical uses of lambda expressions."""

    # A lambda is a small anonymous function.
    add_ten = lambda number: number + 10

    numbers = [5, 1, 9, 2, 7]

    # sorted() accepts a key function.
    sorted_numbers = sorted(numbers, key=lambda number: -number)

    people = [
        {"name": "A", "age": 32},
        {"name": "B", "age": 21},
        {"name": "C", "age": 27},
    ]

    people_by_age = sorted(people, key=lambda person: person["age"])

    print("Lambda result:", add_ten(5))
    print("Descending:", sorted_numbers)
    print("Sorted people:", people_by_age)


# ============================================================================
# 9. MAP, FILTER, REDUCE
# ============================================================================

def demonstrate_functional_tools() -> None:
    """Demonstrate map(), filter(), and functools.reduce()."""

    numbers = [1, 2, 3, 4, 5, 6]

    squared = list(map(lambda number: number ** 2, numbers))

    even_numbers = list(filter(lambda number: number % 2 == 0, numbers))

    product = functools.reduce(operator.mul, numbers, 1)

    print("Original:", numbers)
    print("Squared:", squared)
    print("Even:", even_numbers)
    print("Product:", product)


# ============================================================================
# 10. CLOSURES
# ============================================================================

def make_multiplier(multiplier: float) -> Callable[[float], float]:
    """
    Return a function that remembers multiplier.

    The returned function is a closure because it retains access to a variable
    from the enclosing function after the enclosing function has returned.
    """

    def multiply(value: float) -> float:
        return value * multiplier

    return multiply


def make_counter(start: int = 0) -> Callable[[], int]:
    """Return a closure containing private mutable state."""
    count = start

    def increment() -> int:
        nonlocal count
        count += 1
        return count

    return increment


# ============================================================================
# 11. DECORATORS
# ============================================================================

def log_calls(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that prints function calls.

    functools.wraps preserves useful metadata such as __name__ and __doc__.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"[LOG] Calling {function.__name__}")
        result = function(*args, **kwargs)
        print(f"[LOG] {function.__name__} returned {result!r}")
        return result

    return wrapper


@log_calls
def decorated_add(first: int, second: int) -> int:
    """Add two integers."""
    return first + second


def repeat(times: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Parameterized decorator.

    A decorator factory first receives configuration and then receives the
    function that will be decorated.
    """

    if times < 0:
        raise ValueError("times cannot be negative.")

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = None

            for _ in range(times):
                result = function(*args, **kwargs)

            return result

        return wrapper

    return decorator


@repeat(3)
def say_once() -> str:
    """Return a simple string."""
    print("Executed")
    return "done"


# ============================================================================
# 12. RECURSION
# ============================================================================

def factorial_recursive(number: int) -> int:
    """
    Calculate factorial recursively.

    Base case:
        0! = 1

    Recursive case:
        n! = n * (n - 1)!
    """
    if not isinstance(number, int):
        raise TypeError("Factorial requires an integer.")

    if number < 0:
        raise ValueError("Factorial is undefined for negative integers.")

    if number in (0, 1):
        return 1

    return number * factorial_recursive(number - 1)


def fibonacci_recursive(number: int) -> int:
    """
    Recursive Fibonacci implementation.

    This version is intentionally simple for teaching. It has exponential
    time complexity and should not be used for large values.
    """
    if number < 0:
        raise ValueError("Fibonacci index cannot be negative.")

    if number <= 1:
        return number

    return fibonacci_recursive(number - 1) + fibonacci_recursive(number - 2)


def fibonacci_iterative(number: int) -> int:
    """Efficient iterative Fibonacci implementation."""
    if number < 0:
        raise ValueError("Fibonacci index cannot be negative.")

    previous = 0
    current = 1

    for _ in range(number):
        previous, current = current, previous + current

    return previous


# ============================================================================
# 13. SCOPE AND LEGB
# ============================================================================

scope_global_value = "global"


def demonstrate_scope() -> str:
    """
    Demonstrate local and global name resolution.

    LEGB means:
        Local
        Enclosing
        Global
        Built-in
    """

    enclosing_value = "enclosing"

    def nested() -> str:
        local_value = "local"

        # Python searches for names from the innermost applicable scope.
        return f"{local_value}, {enclosing_value}, {scope_global_value}, {len([])}"

    return nested()


global_counter = 0


def increment_global_counter() -> int:
    """
    Modify a module-level variable using global.

    global should be used carefully because shared mutable state can make
    programs harder to understand and test.
    """
    global global_counter
    global_counter += 1
    return global_counter


def make_stateful_function() -> Callable[[], int]:
    """Use nonlocal to modify a variable in an enclosing function."""
    state = 0

    def increment() -> int:
        nonlocal state
        state += 1
        return state

    return increment


# ============================================================================
# 14. MUTABLE DEFAULT ARGUMENT PITFALL
# ============================================================================

def unsafe_collect(value: Any, collection: list[Any] = []) -> list[Any]:
    """
    Demonstrate the mutable default argument trap.

    The default list is created once when the function is defined, not every
    time the function is called.
    """
    collection.append(value)
    return collection


def safe_collect(value: Any, collection: list[Any] | None = None) -> list[Any]:
    """
    Correct pattern for an optional mutable argument.

    None is used as a sentinel, and a new list is created per call.
    """
    if collection is None:
        collection = []

    collection.append(value)
    return collection


# ============================================================================
# 15. FUNCTION FACTORIES AND CALLBACKS
# ============================================================================

def run_pipeline(
    value: float,
    transformations: Iterable[Callable[[float], float]],
) -> float:
    """Apply a sequence of functions to a value."""
    result = value

    for transformation in transformations:
        result = transformation(result)

    return result


def notify_user(message: str) -> None:
    """Example callback."""
    print(f"Notification: {message}")


def process_data(
    values: Sequence[int],
    callback: Callable[[str], None],
) -> int:
    """
    Process data and notify the caller through a callback.

    Callbacks are useful when the processing function should not need to know
    exactly what the caller wants to do after processing.
    """
    total = sum(values)
    callback(f"Processed {len(values)} values. Total={total}")
    return total


# ============================================================================
# 16. GENERATORS AND FUNCTION-BASED ITERATION
# ============================================================================

def countdown(start: int) -> Iterator[int]:
    """
    Yield values lazily.

    yield turns this function into a generator function. Values are produced
    one at a time rather than creating the entire sequence in memory.
    """
    current = start

    while current >= 0:
        yield current
        current -= 1


def running_total(numbers: Iterable[float]) -> Iterator[float]:
    """Yield cumulative totals lazily."""
    total = 0.0

    for number in numbers:
        total += number
        yield total


# ============================================================================
# 17. EXCEPTIONS: BASIC EXAMPLES
# ============================================================================

def safe_integer_conversion(value: str) -> int | None:
    """Convert text to integer and return None for invalid input."""
    try:
        return int(value)
    except ValueError:
        return None


def safe_division(dividend: float, divisor: float) -> float | None:
    """Return None when division by zero occurs."""
    try:
        return dividend / divisor
    except ZeroDivisionError:
        return None


def parse_positive_integer(value: str) -> int:
    """
    Validate and convert a positive integer.

    This function demonstrates raising exceptions as part of a function's
    contract.
    """
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{value!r} is not a valid integer.") from error

    if number <= 0:
        raise ValueError("The number must be positive.")

    return number


# ============================================================================
# 18. EXCEPTION HIERARCHY
# ============================================================================

def exception_hierarchy_demo(value: Any) -> str:
    """
    Demonstrate specific exception handling.

    Exception classes form an inheritance hierarchy. Catching Exception is
    broad and should not replace handling known failure conditions precisely.
    """
    try:
        return str(int(value))
    except TypeError:
        return "TypeError: an incompatible type was supplied."
    except ValueError:
        return "ValueError: conversion failed."


def demonstrate_multiple_exceptions(value: str, divisor: str) -> str:
    """Handle multiple related exceptions in one handler."""
    try:
        result = int(value) / int(divisor)
        return str(result)
    except (ValueError, ZeroDivisionError) as error:
        return f"Input error: {error}"


# ============================================================================
# 19. TRY, EXCEPT, ELSE, FINALLY
# ============================================================================

def try_except_else_finally(value: str) -> str:
    """
    Demonstrate all four blocks.

    try:
        Code that may fail.

    except:
        Recovery or handling for a known exception.

    else:
        Runs only when the try block succeeds.

    finally:
        Runs whether an exception occurred or not.
    """
    result = ""

    try:
        number = int(value)
    except ValueError:
        result = "Conversion failed."
    else:
        result = f"Conversion succeeded: {number}"
    finally:
        print("Cleanup/finally block executed.")

    return result


# ============================================================================
# 20. CUSTOM EXCEPTIONS
# ============================================================================

class ApplicationError(Exception):
    """Base exception for application-specific errors."""


class ValidationError(ApplicationError):
    """Raised when user-provided data violates validation rules."""


class InsufficientFundsError(ApplicationError):
    """Raised when an account lacks enough money for a withdrawal."""

    def __init__(self, requested: float, available: float) -> None:
        self.requested = requested
        self.available = available

        super().__init__(
            f"Requested {requested:.2f}, but only {available:.2f} is available."
        )


@dataclass
class BankAccount:
    """Small example combining validation and custom exceptions."""

    owner: str
    balance: float = 0.0

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValidationError("Account owner cannot be empty.")

        if self.balance < 0:
            raise ValidationError("Initial balance cannot be negative.")

    def deposit(self, amount: float) -> float:
        """Deposit a positive amount."""
        if amount <= 0:
            raise ValidationError("Deposit amount must be positive.")

        self.balance += amount
        return self.balance

    def withdraw(self, amount: float) -> float:
        """Withdraw money while enforcing business rules."""
        if amount <= 0:
            raise ValidationError("Withdrawal amount must be positive.")

        if amount > self.balance:
            raise InsufficientFundsError(amount, self.balance)

        self.balance -= amount
        return self.balance


# ============================================================================
# 21. EXCEPTION CHAINING
# ============================================================================

def load_integer_from_text(value: str) -> int:
    """
    Demonstrate explicit exception chaining.

    The original ValueError is retained as the cause of the higher-level
    ValueError.
    """
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"Application could not interpret {value!r} as an integer."
        ) from error


# ============================================================================
# 22. ASSERTIONS
# ============================================================================

def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate a percentage.

    The assertion documents an internal assumption. Assertions are not a
    substitute for validating untrusted user input because Python can be run
    with optimization that removes assert statements.
    """
    assert whole != 0, "Internal assumption failed: whole cannot be zero."
    return (part / whole) * 100


# ============================================================================
# 23. CONTEXT MANAGERS
# ============================================================================

@contextmanager
def managed_operation(operation_name: str) -> Iterator[None]:
    """
    Demonstrate a context manager implemented with contextlib.contextmanager.

    finally ensures cleanup code executes even if the body raises.
    """
    print(f"Starting: {operation_name}")

    try:
        yield
    finally:
        print(f"Finished/cleaned up: {operation_name}")


# ============================================================================
# 24. MODULE CONCEPTS
# ============================================================================

def demonstrate_imports() -> None:
    """
    Demonstrate concepts associated with modules.

    This script itself is a module when saved as a .py file.

    Important concepts:
    - import module
    - import module as alias
    - from module import name
    - module namespaces
    - __name__
    - __main__

    Imports of standard-library modules were placed at the top of this file
    according to common Python style.
    """

    import datetime
    import decimal as decimal_module
    from fractions import Fraction

    current_date = datetime.date.today()
    exact_fraction = Fraction(1, 3)
    precise_decimal = decimal_module.Decimal("0.1") + decimal_module.Decimal("0.2")

    print("Date:", current_date)
    print("Fraction:", exact_fraction)
    print("Decimal:", precise_decimal)
    print("math.pi:", math.pi)


def demonstrate_module_namespace() -> None:
    """Show that a module exposes names through a namespace."""
    print("math module name:", math.__name__)
    print("math module file:", math.__file__)
    print("sqrt through namespace:", math.sqrt(81))


# ============================================================================
# 25. SIMULATING A SMALL MODULE API
# ============================================================================

class TextUtilities:
    """
    A class used only to group demonstration functions.

    In a real project these related functions could instead live in a
    separate text_utils.py module.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize whitespace and surrounding spaces."""
        return " ".join(text.split())

    @staticmethod
    def word_count(text: str) -> int:
        """Count whitespace-separated words."""
        normalized = TextUtilities.normalize(text)
        return len(normalized.split()) if normalized else 0


# ============================================================================
# 26. PACKAGE STRUCTURE EXAMPLE
# ============================================================================

PACKAGE_STRUCTURE_EXAMPLE = """
A practical package could be organized as:

project/
    main.py
    calculator/
        __init__.py
        arithmetic.py
        validation.py
        exceptions.py

A package is a directory containing related Python modules. Modern Python can
also support namespace packages without __init__.py, but __init__.py remains
common and useful for explicit package structure and package initialization.

Example imports:

    from calculator.arithmetic import add
    from calculator.validation import validate_number
    from calculator.exceptions import ValidationError

The exact import path depends on where the package is located relative to the
Python import path.
"""


# ============================================================================
# 27. IMPORT GUARD
# ============================================================================

def run_as_script_message() -> None:
    """
    Demonstrate the purpose of if __name__ == "__main__".

    When this file is executed directly, __name__ normally equals "__main__".
    When imported by another module, __name__ is the module's import name.
    """
    print("__name__ =", __name__)


# ============================================================================
# 28. ADVANCED DECORATOR: TIMING
# ============================================================================

def timed(function: Callable[..., Any]) -> Callable[..., Any]:
    """Measure execution time of a function."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()

        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[TIMING] {function.__name__}: {elapsed:.8f} seconds")

    return wrapper


@timed
def sum_squares(limit: int) -> int:
    """Compute the sum of squares from zero through limit."""
    if limit < 0:
        raise ValueError("limit cannot be negative.")

    return sum(number * number for number in range(limit + 1))


# ============================================================================
# 29. ADVANCED DECORATOR: SIMPLE MEMOIZATION
# ============================================================================

def memoize(function: Callable[..., Any]) -> Callable[..., Any]:
    """
    Cache results based on arguments.

    This simplified cache requires arguments to be hashable and does not
    implement eviction, expiration, or concurrency control. It is educational,
    not a general production cache.
    """
    cache: dict[tuple[Any, ...], Any] = {}

    @functools.wraps(function)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = function(*args)

        return cache[args]

    return wrapper


@memoize
def fibonacci_memoized(number: int) -> int:
    """Fibonacci using a simple memoization decorator."""
    if number < 0:
        raise ValueError("Fibonacci index cannot be negative.")

    if number <= 1:
        return number

    return fibonacci_memoized(number - 1) + fibonacci_memoized(number - 2)


# ============================================================================
# 30. BUILT-IN LRU CACHE
# ============================================================================

@functools.lru_cache(maxsize=128)
def fibonacci_cached(number: int) -> int:
    """
    Fibonacci using functools.lru_cache.

    functools.lru_cache is preferable to implementing a basic cache when its
    semantics match the application because it handles cache management more
    robustly.
    """
    if number < 0:
        raise ValueError("Fibonacci index cannot be negative.")

    if number <= 1:
        return number

    return fibonacci_cached(number - 1) + fibonacci_cached(number - 2)


# ============================================================================
# 31. FUNCTION COMPOSITION
# ============================================================================

def compose(
    first: Callable[[Any], Any],
    second: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """
    Return a function equivalent to second(first(value)).

    Composition allows small functions to be combined into larger operations.
    """

    @functools.wraps(first)
    def composed(value: Any) -> Any:
        return second(first(value))

    return composed


# ============================================================================
# 32. VALIDATION AND ERROR DESIGN
# ============================================================================

def validate_percentage(value: float) -> float:
    """Validate a percentage in the inclusive range 0 to 100."""
    if not isinstance(value, (int, float)):
        raise TypeError("Percentage must be numeric.")

    if not math.isfinite(value):
        raise ValueError("Percentage must be finite.")

    if not 0 <= value <= 100:
        raise ValueError("Percentage must be between 0 and 100.")

    return float(value)


def calculate_discounted_price(
    price: float,
    discount_percentage: float,
) -> float:
    """Calculate a discounted price after validating inputs."""
    if price < 0:
        raise ValidationError("Price cannot be negative.")

    discount = validate_percentage(discount_percentage)

    return price * (1 - discount / 100)


# ============================================================================
# 33. PRACTICAL APPLICATION: FINANCIAL CALCULATOR FUNCTIONS
# ============================================================================

def simple_interest(
    principal: float,
    annual_rate_percentage: float,
    years: float,
) -> float:
    """Calculate simple interest."""
    if principal < 0:
        raise ValidationError("Principal cannot be negative.")

    if years < 0:
        raise ValidationError("Years cannot be negative.")

    rate = validate_percentage(annual_rate_percentage) / 100

    return principal * rate * years


def compound_interest(
    principal: float,
    annual_rate_percentage: float,
    years: float,
    compounds_per_year: int = 12,
) -> float:
    """
    Calculate compound interest earned.

    Formula:
        A = P(1 + r/n)^(nt)
        Interest = A - P
    """
    if principal < 0:
        raise ValidationError("Principal cannot be negative.")

    if years < 0:
        raise ValidationError("Years cannot be negative.")

    if compounds_per_year <= 0:
        raise ValidationError("Compounding frequency must be positive.")

    rate = validate_percentage(annual_rate_percentage) / 100

    amount = principal * (
        1 + rate / compounds_per_year
    ) ** (compounds_per_year * years)

    return amount - principal


def loan_emi(
    principal: float,
    annual_rate_percentage: float,
    years: int,
) -> float:
    """
    Calculate monthly loan EMI.

    For a zero-interest loan, EMI is principal divided by the number of months.
    """
    if principal <= 0:
        raise ValidationError("Principal must be positive.")

    if years <= 0:
        raise ValidationError("Loan duration must be positive.")

    annual_rate = validate_percentage(annual_rate_percentage)

    months = years * 12
    monthly_rate = annual_rate / 100 / 12

    if monthly_rate == 0:
        return principal / months

    return (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )


# ============================================================================
# 34. PRACTICAL APPLICATION: STATISTICS
# ============================================================================

def descriptive_statistics(values: Sequence[float]) -> dict[str, float]:
    """Return several descriptive statistics."""
    if not values:
        raise ValueError("At least one observation is required.")

    return {
        "count": float(len(values)),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def z_score(value: float, mean: float, standard_deviation: float) -> float:
    """Calculate a z-score with explicit zero-standard-deviation handling."""
    if standard_deviation == 0:
        raise ValidationError("Standard deviation cannot be zero.")

    return (value - mean) / standard_deviation


# ============================================================================
# 35. PRACTICAL APPLICATION: COMMAND DISPATCH
# ============================================================================

def command_add(a: float, b: float) -> float:
    """Add two values."""
    return a + b


def command_subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def command_multiply(a: float, b: float) -> float:
    """Multiply two values."""
    return a * b


def command_divide(a: float, b: float) -> float:
    """Divide a by b."""
    return divide(a, b)


COMMANDS: dict[str, Callable[[float, float], float]] = {
    "add": command_add,
    "subtract": command_subtract,
    "multiply": command_multiply,
    "divide": command_divide,
}


def execute_command(command: str, a: float, b: float) -> float:
    """
    Execute an operation through a dictionary of function objects.

    This avoids a long chain of if/elif statements and creates an explicit
    dispatch table.
    """
    try:
        operation = COMMANDS[command]
    except KeyError as error:
        raise ValidationError(f"Unsupported command: {command}") from error

    return operation(a, b)


# ============================================================================
# 36. ERROR HANDLING WITH USER-FACING AND INTERNAL ERRORS
# ============================================================================

def parse_command(expression: str) -> float:
    """
    Parse a deliberately limited calculator expression.

    Supported syntax:
        number operator number

    Supported operators:
        + - * /

    The implementation does not use eval(). Avoiding eval for untrusted input
    is an important security practice.
    """
    parts = expression.split()

    if len(parts) != 3:
        raise ValidationError(
            "Expected format: number operator number"
        )

    left_text, operator_text, right_text = parts

    try:
        left = float(left_text)
        right = float(right_text)
    except ValueError as error:
        raise ValidationError("Both operands must be numeric.") from error

    operation_map: dict[str, Callable[[float, float], float]] = {
        "+": command_add,
        "-": command_subtract,
        "*": command_multiply,
        "/": command_divide,
    }

    try:
        operation = operation_map[operator_text]
    except KeyError as error:
        raise ValidationError(
            f"Unsupported operator: {operator_text!r}"
        ) from error

    return operation(left, right)


# ============================================================================
# 37. TESTING FUNCTIONS
# ============================================================================

def assert_equal(actual: Any, expected: Any, description: str) -> None:
    """
    Minimal educational test helper.

    Production projects should normally use unittest, pytest, or another
    suitable testing framework rather than maintaining a home-grown framework.
    """
    if actual != expected:
        raise AssertionError(
            f"{description}: expected {expected!r}, got {actual!r}"
        )


def run_function_tests() -> None:
    """Run deterministic tests for selected functions."""

    assert_equal(add_numbers(2, 3), 5, "addition")
    assert_equal(power(3), 9, "default exponent")
    assert_equal(power(2, 3), 8, "explicit exponent")
    assert_equal(quotient_and_remainder(17, 5), (3, 2), "division")
    assert_equal(factorial_recursive(5), 120, "factorial")
    assert_equal(fibonacci_iterative(10), 55, "fibonacci")
    assert_equal(fibonacci_memoized(20), 6765, "memoized fibonacci")
    assert_equal(execute_command("add", 4, 5), 9, "command dispatch")
    assert_equal(parse_command("10 + 5"), 15.0, "command parser")

    try:
        divide(10, 0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("divide() should reject zero divisor.")

    try:
        parse_positive_integer("-1")
    except ValueError:
        pass
    else:
        raise AssertionError("Negative integer should be rejected.")

    try:
        execute_command("unknown", 1, 2)
    except ValidationError:
        pass
    else:
        raise AssertionError("Unknown command should raise ValidationError.")

    print("All function tests passed.")


# ============================================================================
# 38. DEBUGGING EXAMPLES
# ============================================================================

def debugging_example(value: Any) -> float:
    """
    A function with explicit validation that makes debugging easier.

    Good debugging starts by narrowing the failing assumption rather than
    catching every possible exception and hiding the original problem.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric value, got {type(value).__name__}")

    if not math.isfinite(value):
        raise ValueError("Value must be finite.")

    return math.sqrt(value)


def demonstrate_traceback() -> None:
    """
    Show how a traceback can be captured.

    traceback.format_exc() converts the active exception traceback into text.
    """
    try:
        debugging_example("not a number")
    except (TypeError, ValueError):
        captured = traceback.format_exc()
        print("Captured traceback:")
        print(captured)


# ============================================================================
# 39. PERFORMANCE COMPARISON
# ============================================================================

def benchmark_fibonacci() -> None:
    """
    Compare intentionally different Fibonacci implementations.

    The recursive version becomes slow quickly.
    The iterative version is linear.
    The cached version avoids repeated work.
    """
    target = 30

    start = time.perf_counter()
    recursive_result = fibonacci_recursive(target)
    recursive_time = time.perf_counter() - start

    start = time.perf_counter()
    iterative_result = fibonacci_iterative(target)
    iterative_time = time.perf_counter() - start

    fibonacci_cached.cache_clear()

    start = time.perf_counter()
    cached_result = fibonacci_cached(target)
    cached_time = time.perf_counter() - start

    print("Results:", recursive_result, iterative_result, cached_result)
    print(f"Recursive: {recursive_time:.8f}s")
    print(f"Iterative: {iterative_time:.8f}s")
    print(f"Cached:    {cached_time:.8f}s")


# ============================================================================
# 40. EXCEPTION PERFORMANCE AND DESIGN
# ============================================================================

def parse_with_explicit_check(value: str) -> int | None:
    """
    Example using a simple explicit check before conversion.

    This is intentionally limited and illustrates that validation strategy
    should match the actual input contract.
    """
    if value.strip().lstrip("+-").isdigit():
        return int(value)

    return None


def parse_with_exception(value: str) -> int | None:
    """Use Python's conversion and handle its documented failure."""
    try:
        return int(value)
    except ValueError:
        return None


# ============================================================================
# 41. SECURITY-RELEVANT FUNCTION DESIGN
# ============================================================================

def safe_filename_component(value: str) -> str:
    """
    Perform basic filename-component validation.

    This does not make arbitrary filesystem operations safe by itself.
    Production file handling requires a complete path and authorization model.
    """
    forbidden = {"/", "\\", "..", "\x00"}

    if not value:
        raise ValidationError("Filename component cannot be empty.")

    if any(token in value for token in forbidden):
        raise ValidationError("Filename component contains unsafe characters.")

    return value


def safe_integer_range(value: str, minimum: int, maximum: int) -> int:
    """
    Parse an integer and enforce a strict range.

    Bounded input is useful for avoiding accidental resource exhaustion.
    """
    number = parse_positive_integer(value)

    if number < minimum or number > maximum:
        raise ValidationError(
            f"Value must be between {minimum} and {maximum}."
        )

    return number


# ============================================================================
# 42. CLASS METHOD USING FUNCTIONS AS BEHAVIOR
# ============================================================================

@dataclass
class CalculationPipeline:
    """
    Store a collection of transformation functions.

    This demonstrates how functions can become configurable behavior.
    """

    transformations: list[Callable[[float], float]]

    def run(self, value: float) -> float:
        """Apply transformations in order."""
        return run_pipeline(value, self.transformations)

    def add(self, transformation: Callable[[float], float]) -> None:
        """Append a transformation."""
        self.transformations.append(transformation)


# ============================================================================
# 43. EXCEPTION-RESILIENT BATCH PROCESSING
# ============================================================================

@dataclass
class BatchResult:
    """Represent successful and failed items from a batch operation."""

    successes: list[Any]
    failures: list[tuple[Any, str]]


def process_batch(
    values: Iterable[str],
    processor: Callable[[str], Any],
) -> BatchResult:
    """
    Process independent records without allowing one bad record to terminate
    the whole batch.

    Only expected processing exceptions should normally be caught here.
    Catching BaseException would also catch KeyboardInterrupt and SystemExit,
    which should generally be allowed to propagate.
    """
    successes: list[Any] = []
    failures: list[tuple[Any, str]] = []

    for value in values:
        try:
            result = processor(value)
        except (ValueError, ValidationError, TypeError) as error:
            failures.append((value, str(error)))
        else:
            successes.append(result)

    return BatchResult(successes, failures)


def batch_integer_processor(value: str) -> int:
    """Processor used by the batch demonstration."""
    return parse_positive_integer(value)


# ============================================================================
# 44. EXCEPTION GROUPS
# ============================================================================

def demonstrate_exception_group() -> None:
    """
    Demonstrate ExceptionGroup and except* introduced in Python 3.11.

    ExceptionGroup allows multiple independent exceptions to be reported
    together. It is especially relevant to concurrent or batch processing.
    """
    if sys.version_info < (3, 11):
        print("ExceptionGroup demonstration requires Python 3.11+.")
        return

    errors = [
        ValueError("Invalid number"),
        TypeError("Wrong data type"),
        ValueError("Another invalid number"),
    ]

    try:
        raise ExceptionGroup("Batch validation errors", errors)
    except* ValueError as value_errors:
        print("Handled ValueError group:", value_errors.exceptions)
    except* TypeError as type_errors:
        print("Handled TypeError group:", type_errors.exceptions)


# ============================================================================
# 45. PRACTICAL MODULE IMPORT DISCUSSION DATA
# ============================================================================

MODULE_CONCEPTS = {
    "module": "A Python file containing definitions and executable statements.",
    "package": "A structured collection of related Python modules.",
    "namespace": "A mapping between names and objects.",
    "import": "The mechanism used to make module/package names available.",
    "standard_library": "Modules distributed with Python.",
    "third_party": "Packages installed separately from Python itself.",
    "absolute_import": "An import resolved from the top-level import path.",
    "relative_import": "An import expressed relative to the current package.",
}


# ============================================================================
# 46. RELATIVE IMPORT EXAMPLE AS DOCUMENTATION
# ============================================================================

RELATIVE_IMPORT_EXAMPLE = """
Inside a package:

    project/
        app/
            __init__.py
            main.py
            utilities.py
            services/
                __init__.py
                reports.py

Inside reports.py, an import can conceptually be:

    from ..utilities import some_function

A single dot refers to the current package level.
Two dots move to the parent package level.

Relative imports are meaningful inside packages and should be designed
carefully so that modules have clear responsibilities.
"""


# ============================================================================
# 47. CIRCULAR IMPORT EXPLANATION
# ============================================================================

CIRCULAR_IMPORT_EXAMPLE = """
A circular import occurs when:

    module_a imports module_b
    module_b imports module_a

Circular imports can produce partially initialized modules and confusing
ImportError or AttributeError failures.

Better design usually means separating shared definitions into a third module,
reducing unnecessary module-level dependencies, or restructuring responsibilities.
"""


# ============================================================================
# 48. IMPORT SIDE EFFECTS
# ============================================================================

def import_side_effects_explanation() -> str:
    """
    Return an explanation of why module-level side effects should be controlled.

    Importing a module executes its top-level statements once for that process.
    Expensive work, network calls, database writes, or destructive operations at
    import time make modules difficult to reuse and test.
    """
    return (
        "Keep reusable definitions at module level and put executable program "
        "startup behind an __main__ guard."
    )


# ============================================================================
# 49. API DESIGN WITH SENTINELS
# ============================================================================

_MISSING = object()


def find_setting(
    settings: dict[str, Any],
    key: str,
    default: Any = _MISSING,
) -> Any:
    """
    Distinguish between a missing default and an explicit None default.

    A unique sentinel object solves an ambiguity that None alone cannot solve.
    """
    if key in settings:
        return settings[key]

    if default is _MISSING:
        raise KeyError(key)

    return default


# ============================================================================
# 50. API DESIGN WITH CALLBACK ERROR HANDLING
# ============================================================================

def retry_operation(
    operation: Callable[[], Any],
    attempts: int = 3,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """
    Retry an operation a bounded number of times.

    Important production considerations:
    - Retry only transient failures.
    - Use backoff and jitter for real network systems.
    - Avoid retrying authentication or validation errors blindly.
    - Set timeouts for external operations.
    - Bound the number of attempts.
    """
    if attempts <= 0:
        raise ValidationError("attempts must be positive.")

    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            return operation()
        except retryable_exceptions as error:
            last_error = error

    assert last_error is not None
    raise last_error


# ============================================================================
# 51. FUNCTION OVERLOADING CONCEPT
# ============================================================================

def format_value(value: int | float | str) -> str:
    """
    Demonstrate Python's common approach to accepting multiple types.

    Python does not use traditional compile-time function overloading in the
    same way as languages such as Java or C++. Defining another function with
    the same name replaces the previous definition.

    Runtime dispatch can instead be implemented through:
    - default arguments
    - *args/**kwargs
    - isinstance()
    - singledispatch
    - explicit dispatch dictionaries
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return f"{value:.2f}"

    raise TypeError(f"Unsupported type: {type(value).__name__}")


# ============================================================================
# 52. SINGLEDISPATCH
# ============================================================================

@functools.singledispatch
def describe_value(value: Any) -> str:
    """Default implementation for describe_value."""
    return f"Object of type {type(value).__name__}: {value!r}"


@describe_value.register
def _(value: int) -> str:
    """Specialized implementation for integers."""
    return f"Integer: {value}"


@describe_value.register
def _(value: float) -> str:
    """Specialized implementation for floating-point values."""
    return f"Float: {value}"


@describe_value.register
def _(value: str) -> str:
    """Specialized implementation for strings."""
    return f"String of length {len(value)}: {value!r}"


# ============================================================================
# 53. ASYNC FUNCTION CONCEPT
# ============================================================================

async def asynchronous_example(value: int) -> int:
    """
    Minimal async function.

    Defining an async function does not execute it immediately. Calling it
    returns a coroutine object, which is normally run by an event loop with
    await or asyncio.run().
    """
    return value * 2


# ============================================================================
# 54. PURE FUNCTION CONCEPT
# ============================================================================

def pure_add(a: int, b: int) -> int:
    """
    Pure function.

    It depends only on its arguments and does not mutate external state.
    Pure functions are usually easy to test, reason about, and reuse.
    """
    return a + b


mutable_external_state: list[str] = []


def impure_append(value: str) -> None:
    """
    Impure function.

    It changes state outside its local scope, which may be appropriate in some
    applications but creates additional reasoning and testing considerations.
    """
    mutable_external_state.append(value)


# ============================================================================
# 55. PRACTICAL APPLICATION: VALIDATED REPORT GENERATOR
# ============================================================================

@dataclass(frozen=True)
class StudentRecord:
    """Immutable data record for a report."""

    name: str
    score: float


def validate_student_record(name: str, score: float) -> StudentRecord:
    """Validate and construct a student record."""
    clean_name = name.strip()

    if not clean_name:
        raise ValidationError("Student name cannot be empty.")

    validate_percentage(score)

    return StudentRecord(clean_name, float(score))


def classify_score(score: float) -> str:
    """Classify a score into a grade category."""
    validate_percentage(score)

    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"

    return "F"


def build_student_report(records: Sequence[StudentRecord]) -> dict[str, Any]:
    """Build an aggregate report from validated records."""
    if not records:
        raise ValidationError("At least one student record is required.")

    scores = [record.score for record in records]

    grade_counts: dict[str, int] = {}

    for score in scores:
        grade = classify_score(score)
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    return {
        "student_count": len(records),
        "average": statistics.mean(scores),
        "highest": max(scores),
        "lowest": min(scores),
        "grade_counts": grade_counts,
    }


# ============================================================================
# 56. COMMON MISTAKE DEMONSTRATIONS
# ============================================================================

def demonstrate_common_mistakes() -> None:
    """Show common function-related mistakes and their correct alternatives."""

    subsection("Mutable default arguments")

    first_unsafe = unsafe_collect("A")
    second_unsafe = unsafe_collect("B")

    print("Unsafe first call:", first_unsafe)
    print("Unsafe second call:", second_unsafe)

    first_safe = safe_collect("A")
    second_safe = safe_collect("B")

    print("Safe first call:", first_safe)
    print("Safe second call:", second_safe)

    subsection("Shadowing built-ins")

    # Avoid naming variables list, str, sum, or input in real code.
    # The following is safe because the variable name is deliberately different.
    values = [1, 2, 3]
    print("Use the built-in sum normally:", sum(values))

    subsection("Late binding in closures")

    functions = []

    for number in range(3):
        # Default argument captures the current value immediately.
        functions.append(lambda number=number: number)

    print("Captured values:", [function() for function in functions])


# ============================================================================
# 57. EDGE CASES
# ============================================================================

def demonstrate_edge_cases() -> None:
    """Exercise important boundary conditions."""

    subsection("Empty sequence")

    try:
        calculate_statistics([])
    except ValueError as error:
        print("Expected:", error)

    subsection("Zero division")

    try:
        divide(10, 0)
    except ZeroDivisionError as error:
        print("Expected:", error)

    subsection("Negative factorial")

    try:
        factorial_recursive(-1)
    except ValueError as error:
        print("Expected:", error)

    subsection("Invalid percentage")

    for value in (-1, 101, float("nan")):
        try:
            validate_percentage(value)
        except (TypeError, ValueError) as error:
            print(f"Rejected {value!r}: {error}")

    subsection("Missing dictionary setting")

    try:
        find_setting({}, "missing")
    except KeyError as error:
        print("Expected missing setting:", error)

    print("Explicit None default:", find_setting({}, "missing", None))


# ============================================================================
# 58. BANK ACCOUNT EXCEPTION DEMONSTRATION
# ============================================================================

def demonstrate_custom_exceptions() -> None:
    """Demonstrate custom exception handling."""

    account = BankAccount("Example User", 1000)

    print("Initial balance:", account.balance)
    print("After deposit:", account.deposit(250))

    try:
        account.withdraw(5000)
    except InsufficientFundsError as error:
        print("Business-rule error:", error)
        print("Requested:", error.requested)
        print("Available:", error.available)

    try:
        account.deposit(-10)
    except ValidationError as error:
        print("Validation error:", error)


# ============================================================================
# 59. PACKAGE AND MODULE INFORMATION
# ============================================================================

def print_module_information() -> None:
    """Print useful runtime information about this module."""
    print("Current module name:", __name__)
    print("Python version:", sys.version.split()[0])
    print("Module concepts:")
    for key, value in MODULE_CONCEPTS.items():
        print(f"  {key}: {value}")


# ============================================================================
# 60. FULL DEMONSTRATION
# ============================================================================

def demonstrate_basics() -> None:
    """Run beginner-level function demonstrations."""

    section("1. Function fundamentals")

    greet()
    greet_person("Alex")

    print("2 + 3 =", add_numbers(2, 3))
    print("4 * 5 =", multiply_numbers(4, 5))

    subsection("Parameters, defaults, positional and keyword arguments")

    print(introduce("Alex", 30))
    print(introduce("Alex", 30, "Lucknow"))
    print(introduce(name="Alex", age=30, city="Lucknow"))

    print("3² =", power(3))
    print("2³ =", power(2, 3))

    print(describe_person(
        "Alex",
        30,
        city="Lucknow",
        occupation="Developer",
    ))

    print(create_point(10, 20, label="origin-like point"))

    subsection("Return values and unpacking")

    minimum, maximum, average = calculate_statistics([10, 20, 30, 40])

    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("Average:", average)

    quotient, remainder = quotient_and_remainder(17, 5)
    print("Quotient:", quotient)
    print("Remainder:", remainder)

    subsection("Variable-length arguments")

    print("sum_all:", sum_all(1, 2, 3, 4, 5))
    print("profile:", build_profile(
        name="Alex",
        age=30,
        role="Developer",
    ))

    print(summarize_order(
        "Alex",
        "Laptop",
        "Mouse",
        "Keyboard",
        priority="high",
        payment="card",
    ))

    subsection("Argument unpacking")

    dimensions = (10, 5)
    print("Unpacked tuple:", rectangle(*dimensions))

    student = {
        "name": "Alex",
        "age": 30,
        "course": "Python",
    }

    print("Unpacked dictionary:", format_student(**student))

    subsection("Type hints and validation")

    print("Weighted average:", weighted_average(
        [80, 90, 100],
        [1, 2, 1],
    ))


def demonstrate_intermediate_functions() -> None:
    """Run intermediate function demonstrations."""

    section("2. Functions as objects, lambdas, closures, and decorators")

    subsection("First-class functions")

    print("square:", apply_operation(square, 5))
    print("cube:", apply_operation(cube, 5))

    operation = choose_operation("sqrt")
    print("Selected operation:", operation(81))

    subsection("Lambda expressions")

    demonstrate_lambdas()

    subsection("map, filter, reduce")

    demonstrate_functional_tools()

    subsection("Closures")

    double = make_multiplier(2)
    triple = make_multiplier(3)

    print("Double 10:", double(10))
    print("Triple 10:", triple(10))

    counter = make_counter(10)

    print("Counter:", counter())
    print("Counter:", counter())
    print("Counter:", counter())

    subsection("Decorators")

    print("Decorated addition:", decorated_add(3, 4))
    print("Parameterized decorator:")
    say_once()

    subsection("Function composition")

    pipeline = compose(
        lambda number: number + 5,
        lambda number: number * 2,
    )

    print("compose(10):", pipeline(10))

    calculation_pipeline = CalculationPipeline([
        lambda number: number + 5,
        lambda number: number * 2,
        math.sqrt,
    ])

    print("Calculation pipeline:", calculation_pipeline.run(11))


def demonstrate_scope_and_recursion() -> None:
    """Run scope and recursion demonstrations."""

    section("3. Scope, LEGB, global, nonlocal, and recursion")

    print("Scope lookup:", demonstrate_scope())

    print("Global counter:", increment_global_counter())
    print("Global counter:", increment_global_counter())

    stateful = make_stateful_function()

    print("Nonlocal state:", stateful())
    print("Nonlocal state:", stateful())
    print("Nonlocal state:", stateful())

    subsection("Recursion")

    print("5! =", factorial_recursive(5))
    print("Fibonacci recursive(10):", fibonacci_recursive(10))
    print("Fibonacci iterative(10):", fibonacci_iterative(10))
    print("Fibonacci memoized(30):", fibonacci_memoized(30))


def demonstrate_modules_and_packages() -> None:
    """Run module and package demonstrations."""

    section("4. Modules and packages")

    demonstrate_imports()
    demonstrate_module_namespace()
    print_module_information()

    print("\nConceptual package structure:")
    print(PACKAGE_STRUCTURE_EXAMPLE)

    print("Relative import example:")
    print(RELATIVE_IMPORT_EXAMPLE)

    print("Circular import guidance:")
    print(CIRCULAR_IMPORT_EXAMPLE)

    print("Import side effects:")
    print(import_side_effects_explanation())


def demonstrate_exceptions() -> None:
    """Run exception demonstrations."""

    section("5. Exceptions and error handling")

    subsection("Basic exception handling")

    print("Conversion '123':", safe_integer_conversion("123"))
    print("Conversion 'abc':", safe_integer_conversion("abc"))

    print("10 / 2:", safe_division(10, 2))
    print("10 / 0:", safe_division(10, 0))

    subsection("Raising exceptions")

    try:
        parse_positive_integer("abc")
    except ValueError as error:
        print("Expected validation error:", error)

    try:
        parse_positive_integer("-10")
    except ValueError as error:
        print("Expected range error:", error)

    subsection("Specific exception types")

    print(exception_hierarchy_demo("42"))
    print(exception_hierarchy_demo("not-a-number"))
    print(exception_hierarchy_demo(None))

    subsection("Multiple exceptions")

    print(demonstrate_multiple_exceptions("10", "2"))
    print(demonstrate_multiple_exceptions("10", "0"))
    print(demonstrate_multiple_exceptions("ten", "2"))

    subsection("try / except / else / finally")

    print(try_except_else_finally("100"))
    print(try_except_else_finally("invalid"))

    subsection("Exception chaining")

    try:
        load_integer_from_text("abc")
    except ValueError as error:
        print("High-level error:", error)
        print("Original cause:", repr(error.__cause__))

    subsection("Custom exceptions")

    demonstrate_custom_exceptions()

    subsection("Context manager")

    with managed_operation("example operation"):
        print("Inside managed operation.")

    subsection("ExceptionGroup")

    demonstrate_exception_group()


def demonstrate_practical_examples() -> None:
    """Run practical applications."""

    section("6. Practical function design")

    subsection("Financial calculations")

    print("Simple interest:", simple_interest(10000, 8, 2))
    print(
        "Compound interest:",
        compound_interest(10000, 8, 2),
    )
    print(
        "Monthly EMI:",
        loan_emi(500000, 8.5, 5),
    )

    subsection("Statistics")

    data = [10, 12, 15, 20, 25, 30]

    print("Descriptive statistics:", descriptive_statistics(data))

    mean = statistics.mean(data)
    standard_deviation = statistics.stdev(data)

    print(
        "Z-score for 20:",
        z_score(20, mean, standard_deviation),
    )

    subsection("Command dispatch")

    for command, a, b in [
        ("add", 10, 5),
        ("subtract", 10, 5),
        ("multiply", 10, 5),
        ("divide", 10, 5),
    ]:
        print(command, "=", execute_command(command, a, b))

    subsection("Limited expression parser")

    for expression in [
        "10 + 5",
        "10 - 3",
        "6 * 7",
        "20 / 4",
    ]:
        print(expression, "=", parse_command(expression))

    subsection("Validated report")

    raw_records = [
        ("Alice", 92),
        ("Bob", 81),
        ("Carol", 74),
        ("David", 58),
    ]

    records = [
        validate_student_record(name, score)
        for name, score in raw_records
    ]

    print("Report:", build_student_report(records))

    subsection("Batch processing")

    batch = process_batch(
        ["10", "20", "-3", "abc", "50"],
        batch_integer_processor,
    )

    print("Successful values:", batch.successes)
    print("Failed values:", batch.failures)


def demonstrate_advanced_topics() -> None:
    """Run advanced demonstrations."""

    section("7. Advanced function concepts")

    subsection("Generators")

    print("Countdown:", list(countdown(5)))
    print(
        "Running totals:",
        list(running_total([1, 2, 3, 4])),
    )

    subsection("Timing decorator")

    print("Sum of squares:", sum_squares(10000))

    subsection("Memoization")

    print("Memoized Fibonacci:", fibonacci_memoized(35))

    subsection("functools.lru_cache")

    fibonacci_cached.cache_clear()
    print("Cached Fibonacci:", fibonacci_cached(35))
    print("Cache information:", fibonacci_cached.cache_info())

    subsection("singledispatch")

    print(describe_value(42))
    print(describe_value(3.14))
    print(describe_value("hello"))
    print(describe_value([1, 2, 3]))

    subsection("Sentinel values")

    print("Existing setting:", find_setting({"mode": "advanced"}, "mode"))
    print("Explicit default:", find_setting({}, "missing", "fallback"))

    subsection("Retry abstraction")

    attempts = {"count": 0}

    def unstable_operation() -> str:
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise TimeoutError("Temporary failure.")

        return "Operation succeeded."

    print(
        retry_operation(
            unstable_operation,
            attempts=3,
            retryable_exceptions=(TimeoutError,),
        )
    )

    subsection("Security-aware parser")

    try:
        parse_command("10 + 5")
        print("Safe expression accepted.")
    except ValidationError as error:
        print(error)

    try:
        parse_command("10 ** 5")
    except ValidationError as error:
        print("Rejected unsupported expression:", error)

    subsection("Safe filename validation")

    print(safe_filename_component("report_2026.txt"))

    try:
        safe_filename_component("../secret.txt")
    except ValidationError as error:
        print("Rejected unsafe filename:", error)


def demonstrate_common_issues() -> None:
    """Run edge cases, common mistakes, debugging, and tests."""

    section("8. Edge cases, debugging, performance, and testing")

    demonstrate_common_mistakes()
    demonstrate_edge_cases()
    demonstrate_traceback()

    subsection("Performance comparison")

    benchmark_fibonacci()

    subsection("Parser strategies")

    print(
        "Explicit check:",
        parse_with_explicit_check("123"),
    )
    print(
        "Exception-based conversion:",
        parse_with_exception("123"),
    )

    print(
        "Explicit check invalid:",
        parse_with_explicit_check("abc"),
    )
    print(
        "Exception-based invalid:",
        parse_with_exception("abc"),
    )

    subsection("Testing")

    run_function_tests()

    subsection("Assertions")

    print("Percentage:", calculate_percentage(25, 200))

    subsection("Pure and impure functions")

    print("Pure result:", pure_add(2, 3))
    impure_append("changed external state")
    print("External state:", mutable_external_state)


# ============================================================================
# 61. EDUCATIONAL REFERENCE INFORMATION
# ============================================================================

FUNCTION_DESIGN_PRINCIPLES = [
    "Give each function one clear responsibility.",
    "Use meaningful names that describe behavior.",
    "Keep functions reasonably small and cohesive.",
    "Prefer explicit parameters over hidden global state.",
    "Validate inputs when the function has an enforceable contract.",
    "Raise specific exceptions for invalid conditions.",
    "Document non-obvious behavior and important assumptions.",
    "Use type hints when they improve readability and tooling.",
    "Avoid mutable default arguments unless intentional.",
    "Preserve decorator metadata with functools.wraps.",
    "Avoid catching exceptions that the function cannot meaningfully handle.",
    "Keep reusable module definitions separate from application startup.",
    "Avoid expensive or destructive import-time side effects.",
    "Keep module responsibilities cohesive.",
    "Avoid circular dependencies between modules.",
    "Test normal paths, boundary cases, and expected failures.",
    "Measure performance before optimizing.",
]


def print_design_principles() -> None:
    """Print function and module design principles."""
    section("9. Function and module design principles")

    for number, principle in enumerate(FUNCTION_DESIGN_PRINCIPLES, start=1):
        print(f"{number}. {principle}")


# ============================================================================
# 62. MAIN PROGRAM
# ============================================================================

def main() -> None:
    """
    Execute the complete study sequence.

    Keeping execution inside main() makes the file easier to import without
    automatically running every demonstration.
    """

    demonstrate_basics()
    demonstrate_intermediate_functions()
    demonstrate_scope_and_recursion()
    demonstrate_modules_and_packages()
    demonstrate_exceptions()
    demonstrate_practical_examples()
    demonstrate_advanced_topics()
    demonstrate_common_issues()
    print_design_principles()

    section("10. Study-file reference")

    print("Python version:", sys.version.split()[0])
    print("Module name:", __name__)
    print("This file is designed to be read from top to bottom.")
    print("All examples use the Python standard library.")


if __name__ == "__main__":
    main()
