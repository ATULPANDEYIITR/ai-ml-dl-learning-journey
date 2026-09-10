# Python functions and modules

## Topic introduction

Python functions and modules are fundamental mechanisms for organizing programs into reusable, testable, and maintainable units.

A function is a named block of behavior that can receive input, perform operations, and optionally return a result. Functions reduce duplication and provide boundaries around individual responsibilities.

A module is a Python file containing definitions such as functions, classes, constants, and executable statements. Modules allow a larger program to be divided into related units.

Packages provide a higher-level organizational structure by grouping related modules into directories.

Exception handling provides a structured mechanism for responding to abnormal conditions without allowing every error to terminate a program unexpectedly.

The accompanying Python script progresses from basic function definitions to advanced topics including closures, decorators, recursion, generators, module namespaces, packages, custom exceptions, exception chaining, context managers, caching, dispatch techniques, validation, testing, performance, and security-aware function design.

## Functions

### Function definition

A function is created with the `def` statement.

The general structure is:

    def function_name(parameters):
        statements
        return result

The function name should describe the operation performed by the function.

For example, the script defines functions such as `add_numbers()`, `multiply_numbers()`, and `calculate_rectangle_area()`.

A function is not normally executed when Python encounters its definition. The definition creates a function object. The function executes when it is called.

### Calling a function

A function is called by writing its name followed by parentheses.

For example:

    add_numbers(2, 3)

The values supplied during a call are arguments. The names declared in the function definition are parameters.

The distinction is useful:

- A parameter is part of the function definition.
- An argument is a value supplied during a function call.

### Parameters

Parameters allow functions to receive information from their callers.

The script demonstrates:

- Required parameters
- Default parameters
- Positional parameters
- Keyword parameters
- Positional-only parameters
- Keyword-only parameters
- Variable-length positional parameters
- Variable-length keyword parameters

A function can combine several of these mechanisms when there is a clear reason to do so.

## Positional arguments

Positional arguments are matched according to their position.

For example, a function accepting `first` and `second` receives the first supplied value as `first` and the second supplied value as `second`.

Positional calls are concise, but they can become less readable when many parameters have similar types or meanings.

## Keyword arguments

Keyword arguments explicitly identify the parameter.

For example:

    introduce(name="Alex", age=30, city="Lucknow")

Keyword arguments improve readability and allow arguments to be supplied in a different order when the function permits it.

## Default parameters

A default parameter has a value that is used when the caller does not provide an argument.

The script uses:

    def power(base, exponent=2):

Calling `power(3)` therefore calculates 3 raised to the second power.

Defaults should represent sensible behavior. They should not hide important business decisions from callers.

## Positional-only parameters

Python supports positional-only parameters using `/`.

For example:

    def create_point(x, y, /, *, label="point"):

The parameters before `/` must be supplied positionally.

This is useful when a library wants to prevent callers from depending on parameter names or when positional use is part of the intended API.

## Keyword-only parameters

Parameters after `*` are keyword-only.

For example:

    def describe_person(name, age, *, city, occupation):

The caller must provide `city` and `occupation` using their parameter names.

Keyword-only parameters are particularly useful for optional configuration because they make calls more self-documenting.

## Return values

The `return` statement sends a value back to the caller.

For example, `add_numbers()` returns the result of an addition instead of printing it.

Returning a value generally makes a function more reusable than printing inside the function.

A caller can store, compare, transform, or pass the returned value to another function.

A function that reaches the end without a return statement returns `None`.

## Multiple return values

Python functions can return multiple values by returning a tuple.

The script demonstrates:

    minimum, maximum, average = calculate_statistics(numbers)

The function technically returns one tuple object, and tuple unpacking assigns its individual elements to separate variables.

This is useful when a function naturally produces several related results.

The number of values on the left must normally match the number of unpacked values.

## Variable-length arguments

### `*args`

The `*args` syntax collects additional positional arguments into a tuple.

The script uses:

    def sum_all(*numbers):

A call such as:

    sum_all(1, 2, 3, 4, 5)

makes `numbers` a tuple containing all five values.

The name `args` is conventional but not required. The important part is the asterisk.

### `**kwargs`

The `**kwargs` syntax collects additional keyword arguments into a dictionary.

For example:

    build_profile(name="Alex", age=30, role="Developer")

makes the `details` parameter a dictionary containing the supplied key-value pairs.

Again, the name `kwargs` is conventional. The two asterisks determine the behavior.

### Combining parameter types

A function can have normal parameters, `*args`, and `**kwargs`.

The conceptual order is:

    normal parameters -> *args -> **kwargs

This should be used when the flexible interface is genuinely useful. Excessive use of `*args` and `**kwargs` can make APIs harder to understand and validate.

## Argument unpacking

Argument unpacking performs the reverse operation.

A sequence can be unpacked with `*`:

    rectangle(*dimensions)

A dictionary can be unpacked with `**`:

    format_student(**student)

The dictionary keys must correspond to acceptable parameter names.

## Type hints

The script uses annotations such as:

    def add_numbers(first: float, second: float) -> float:

The annotations communicate the intended input and output types.

Type hints are primarily information for developers and development tools. Python normally does not automatically enforce them at runtime.

Type hints can improve:

- Readability
- Editor support
- Static analysis
- API documentation
- Maintenance of larger codebases

They should describe the actual contract rather than merely making code appear more formal.

## Docstrings

A docstring is a string placed immediately inside a function, class, or module definition.

The script uses docstrings to describe:

- Purpose
- Parameters
- Return values
- Exceptions
- Important behavior
- Design limitations

Docstrings are accessible through the object's `__doc__` attribute and can be inspected by documentation and development tools.

## Functions are objects

Python functions are first-class objects.

A function can be:

- Assigned to a variable
- Stored in a list or dictionary
- Passed as an argument
- Returned from another function
- Used as a dictionary value

The script demonstrates this through `apply_operation()` and `choose_operation()`.

For example, a dictionary can map operation names to function objects:

    {
        "square": square,
        "cube": cube,
        "sqrt": math.sqrt
    }

This creates a simple dispatch mechanism.

## Higher-order functions

A higher-order function is a function that receives another function, returns a function, or both.

Examples in the script include:

- `apply_operation()`
- `run_pipeline()`
- `choose_operation()`
- `compose()`
- `retry_operation()`

Higher-order functions are useful when behavior itself needs to become configurable.

They are common in callbacks, functional programming, event processing, data transformations, decorators, and framework APIs.

## Lambda expressions

A lambda creates a small anonymous function.

For example:

    lambda number: number + 10

A lambda can contain one expression and returns the result of that expression.

Lambdas are especially useful for short operations passed directly to functions such as `sorted()`, `map()`, and `filter()`.

The script uses lambdas for sorting and simple transformations.

A named `def` function is generally clearer when the operation is complex, reused, or requires documentation.

## `map()`

`map()` applies a function to every item in an iterable.

The script demonstrates squaring numbers with a lambda.

The conceptual operation is:

    input values -> transformation -> transformed values

In modern Python, `map()` returns an iterator. Converting it to `list()` materializes the results.

A list comprehension is often more readable for straightforward transformations, while `map()` can be useful when an existing function naturally expresses the transformation.

## `filter()`

`filter()` keeps elements for which a predicate function returns a truthy value.

The script uses it to select even numbers.

The conceptual operation is:

    input values -> condition -> retained values

As with `map()`, `filter()` returns an iterator.

## `functools.reduce()`

`reduce()` repeatedly combines elements using a binary operation.

The script uses `operator.mul` to calculate a product.

Reduction is appropriate when many values must be accumulated into one result, although ordinary loops or specialized built-ins can sometimes be clearer.

## Closures

A closure is an inner function that retains access to variables from its enclosing scope.

The script's `make_multiplier()` function creates multiplier functions.

For example, a call can create a function that remembers a multiplier of `2`. The enclosing function has already returned, yet the returned function can still access that value.

Closures are useful for:

- Function factories
- Encapsulated state
- Callbacks
- Decorator construction
- Configurable behavior

## `nonlocal`

The `nonlocal` statement allows a nested function to modify a variable in an enclosing function's scope.

The counter example uses:

    nonlocal count

Without `nonlocal`, assignment would normally create a new local variable inside the nested function.

`nonlocal` does not refer to a module-level variable. Module-level variables require `global` when assignment must occur inside a function.

## Scope and LEGB

Python resolves names according to the LEGB rule:

- Local
- Enclosing
- Global
- Built-in

### Local

The local scope belongs to the currently executing function.

### Enclosing

The enclosing scope belongs to an outer function surrounding a nested function.

### Global

The global scope belongs to the module.

### Built-in

The built-in namespace contains names such as `len`, `sum`, `print`, and `Exception`.

The script demonstrates this lookup order through nested functions.

## `global`

The `global` statement tells Python that assignments to a name inside a function should affect the module-level variable.

The script uses a global counter to demonstrate the mechanism.

Although `global` is valid Python, extensive global mutable state generally makes software harder to test and reason about.

Prefer passing values into functions and returning results when practical.

## `nonlocal` versus `global`

The distinction is important.

`nonlocal` modifies a variable in an enclosing function.

`global` modifies a variable in the module namespace.

Neither should be used merely because it is convenient. Both introduce state relationships that can increase coupling.

## Mutable default arguments

One of the most important Python function pitfalls is the mutable default argument.

The following pattern is problematic when persistent state is not intended:

    def collect(value, collection=[]):

The default list is created when the function is defined, not each time the function is called.

The script demonstrates this behavior with `unsafe_collect()`.

The standard safe pattern is:

    def collect(value, collection=None):
        if collection is None:
            collection = []

This creates a new list for each call that does not explicitly supply a collection.

A mutable default can be intentional in unusual cases, but it should never be used accidentally.

## Late binding in closures

Closures capture variables, not necessarily the value one might intuitively expect at function-creation time.

The script demonstrates a common loop-related issue and solves it by using a default argument:

    lambda number=number: number

The default argument captures the current value for that particular function.

Understanding this behavior is important when generating callbacks inside loops.

## Decorators

A decorator is a callable that receives a function and returns a modified or replacement callable.

The basic pattern is:

    def decorator(function):
        def wrapper(*args, **kwargs):
            ...
            return function(*args, **kwargs)
        return wrapper

The script's `log_calls()` decorator adds logging before and after a function executes.

Decorators are commonly used for:

- Logging
- Timing
- Authorization
- Caching
- Validation
- Instrumentation
- Retry policies
- Framework registration

## `functools.wraps`

When a decorator replaces a function with a wrapper, metadata such as the function name and docstring can otherwise be lost.

The script uses `functools.wraps()` to preserve important metadata.

This is a best practice for decorators that wrap ordinary Python functions.

## Parameterized decorators

Sometimes the decorator itself needs configuration.

The `repeat()` example is a decorator factory.

The process has three levels:

1. Receive configuration.
2. Receive the target function.
3. Return the wrapper.

This pattern is useful for configurable logging, retry counts, permissions, caching settings, and other policies.

## Recursion

Recursion occurs when a function calls itself.

A recursive function needs a base case that stops the recursion.

The factorial implementation demonstrates:

    n! = n × (n - 1)!

with `0! = 1` and `1! = 1` as base cases.

The recursive Fibonacci example demonstrates a major performance problem. The straightforward recursive algorithm repeatedly calculates the same values and therefore has exponential time complexity.

## Recursion versus iteration

The script compares recursive, iterative, and cached Fibonacci implementations.

For Fibonacci:

- Naive recursion performs extensive repeated work.
- Iteration uses linear time and constant additional state.
- Memoization avoids repeated subproblems.

Python also has a recursion depth limit to prevent uncontrolled recursion from consuming the call stack indefinitely.

Recursion is useful when the problem naturally has recursive structure, such as tree traversal and divide-and-conquer algorithms. It is not automatically superior to iteration.

## Generators

A function containing `yield` becomes a generator function.

The script uses `countdown()` and `running_total()`.

Generators produce values lazily rather than constructing the entire result sequence in memory.

This can provide substantial memory benefits for large data streams.

A generator is particularly useful for:

- Large datasets
- Streaming processing
- Pipelines
- File processing
- Infinite or very large sequences

## Function composition

Function composition combines smaller functions into a larger operation.

The script's `compose()` function creates a function equivalent to applying one function and then another.

Small composable functions can make data-processing pipelines easier to construct and test.

## Pure and impure functions

A pure function depends only on its arguments and does not modify external state.

For example:

    pure_add(2, 3)

always produces the same result for the same arguments.

An impure function can interact with or modify external state.

Neither category is universally better. Pure functions are often easier to test and reason about, while side effects are necessary for many real applications such as writing files, updating databases, sending messages, and changing application state.

A good design makes important side effects explicit.

## Modules

A module is normally a `.py` file containing Python code.

A module can contain:

- Functions
- Classes
- Constants
- Variables
- Imports
- Module-level executable statements

The script itself becomes a module when saved as a Python file.

For example, if it is saved as `functions_modules.py`, another file could import it using the appropriate module path.

## Importing modules

The common import form is:

    import math

Names are then accessed through the module namespace:

    math.sqrt(81)

An alias can be used:

    import decimal as decimal_module

Specific names can also be imported:

    from fractions import Fraction

Import styles should be selected for readability and namespace clarity.

## Module namespaces

A module creates a namespace containing its defined names.

The script examines attributes such as:

    math.__name__
    math.__file__

Using the module namespace explicitly helps prevent name collisions.

For example:

    math.sqrt()

is clearer about where `sqrt()` originates than importing many unrelated names into the local namespace.

## `__name__`

Every imported Python module has a `__name__` attribute.

When a module is executed directly, its value is normally:

    "__main__"

When imported, its value is normally the module's import name.

This behavior enables the common execution guard.

## The `__main__` guard

The script ends with:

    if __name__ == "__main__":
        main()

This means `main()` executes when the file is run directly.

If the file is imported by another module, the main demonstration does not automatically execute.

This is important for reusable modules because importing a module should generally expose definitions without unexpectedly starting the entire application.

## Module import execution

Python executes a module's top-level statements when it is imported.

The imported module is then cached in the process's module system, so repeated imports do not normally execute the module from scratch each time.

This is why expensive work, destructive operations, network requests, and database writes should generally not be performed merely because a reusable module is imported.

## Module design

A well-designed module should have a clear responsibility.

For example, a financial application could be separated into:

    finance/
        arithmetic.py
        loans.py
        validation.py
        exceptions.py

The exact structure depends on project size and responsibilities.

Splitting code into too many tiny modules can create unnecessary complexity. Keeping everything in one giant module can make dependencies and maintenance difficult.

The goal is cohesive organization.

## Packages

A package groups related modules into a directory structure.

A conceptual application can look like:

    project/
        main.py
        calculator/
            __init__.py
            arithmetic.py
            validation.py
            exceptions.py

The package can then expose functionality through imports such as:

    from calculator.arithmetic import add

The `__init__.py` file is commonly used to make package structure explicit and can contain package initialization code or selected public exports.

Modern Python also supports namespace packages that do not require `__init__.py`, but traditional packages with explicit initialization remain common.

## Relative imports

Relative imports express relationships inside a package.

For example:

    from ..utilities import some_function

A single dot represents the current package level, while additional dots move upward through the package hierarchy.

Relative imports should be used when they make package relationships clear. Excessive or complicated relative imports can make package structure difficult to understand.

## Circular imports

A circular import occurs when modules depend on each other directly or indirectly.

For example:

    module_a -> module_b -> module_a

Circular imports can lead to partially initialized modules and errors during import.

A common solution is to separate shared functionality into a third module.

Another solution may involve redesigning module responsibilities so that dependencies flow in a clearer direction.

## Import side effects

Top-level module statements execute during import.

A module that performs expensive calculations, modifies databases, starts services, or accesses external systems during import is difficult to reuse safely.

Reusable definitions should normally be separated from application startup logic.

The `__main__` guard is one mechanism for achieving this separation.

## Exceptions

An exception represents an abnormal condition detected during program execution.

Examples include:

- `ValueError`
- `TypeError`
- `ZeroDivisionError`
- `KeyError`
- `IndexError`
- `FileNotFoundError`
- `TimeoutError`
- `ImportError`

Exceptions allow errors to travel through the call stack until an appropriate handler is found.

## `try` and `except`

The basic structure is:

    try:
        operation
    except SomeException:
        recovery

Only code that may reasonably raise the expected exception should generally be placed in the `try` block.

A very large `try` block can accidentally associate unrelated failures with the same handler.

## Specific exception handling

The script deliberately catches specific exceptions such as:

    except ValueError:

and:

    except ZeroDivisionError:

Specific handling is preferable because different failures may require different responses.

Catching everything with:

    except Exception:

can be appropriate at a carefully chosen application boundary, such as logging an unexpected failure before terminating or returning an error response.

It is usually poor practice to use broad exception handling to hide programming errors.

## Multiple exceptions

Several exception types can be handled by one handler:

    except (ValueError, ZeroDivisionError):

This is appropriate when the application response is genuinely the same for both conditions.

## `else`

The `else` block associated with `try` runs only when the `try` block completes without an exception.

This can make successful-path logic clearer.

## `finally`

The `finally` block runs regardless of whether an exception occurred.

It is commonly used for cleanup.

Examples include:

- Releasing resources
- Closing connections
- Restoring state
- Removing temporary resources

Context managers are often preferable for standard resource-management patterns.

## Raising exceptions

A function can deliberately raise an exception:

    raise ValueError("Invalid value.")

Raising an exception communicates that the function's contract has been violated or that an operation cannot continue normally.

A good exception message should explain what went wrong without exposing unnecessary sensitive information.

## Exception chaining

The script uses:

    raise ValueError("Higher-level explanation") from error

This preserves the original exception as the cause of the new exception.

Exception chaining is useful when a low-level error needs to be translated into a higher-level application concept without losing diagnostic information.

For example, a database parsing error may be converted into an application-specific validation error while retaining the original cause for debugging.

## Custom exceptions

Custom exceptions allow an application to represent domain-specific failures.

The script defines:

- `ApplicationError`
- `ValidationError`
- `InsufficientFundsError`

A custom exception hierarchy can make error handling more precise.

For example, callers can catch `ApplicationError` to handle all application-level errors while still catching `InsufficientFundsError` separately when detailed recovery is required.

## Exception classes and attributes

`InsufficientFundsError` stores:

- Requested amount
- Available amount

This allows programmatic access to useful information instead of requiring callers to parse an error message.

Structured exception information is often better than embedding all details in text.

## Exception boundaries

Exceptions should be handled at the layer that has enough information to make a meaningful decision.

A low-level function may detect a failure but should not necessarily decide how a user interface displays it.

A higher-level application layer may translate the exception into:

- A user-facing message
- An HTTP response
- A log entry
- A retry
- A rejected transaction

This separation keeps reusable functions independent of presentation concerns.

## Batch processing

The script includes `process_batch()` to demonstrate handling failures for individual records while allowing other records to continue.

This is appropriate when records are independent.

The function stores successful results separately from failures.

A batch system must still define which exceptions are recoverable. Catching every possible exception can hide serious programming errors.

## `ExceptionGroup`

Python 3.11 introduced `ExceptionGroup` and `except*`.

An exception group can contain multiple independent exceptions.

This is particularly useful in concurrent or batch-oriented programs where several operations can fail independently.

The script checks the Python version before demonstrating this feature.

## Assertions

Assertions document internal assumptions.

The script uses an assertion for an internal invariant.

Assertions are not appropriate for validating untrusted external input because Python can run with optimization that removes assertion checks.

User input, API input, files, database data, and other external data should be validated explicitly.

## Context managers

Context managers define setup and cleanup behavior around a block of code.

The `managed_operation()` example is implemented using `contextlib.contextmanager`.

The `with` statement ensures the cleanup section executes even when an exception occurs.

Python's file-handling syntax is a major real-world example:

    with open(path) as file:
        ...

The context manager closes the file appropriately when the block ends.

## Validation

Validation ensures that input satisfies a defined contract.

The script validates:

- Numeric types
- Numeric ranges
- Positive values
- Percentages
- Financial parameters
- Account balances
- Filename components
- Command names

Validation should happen close to the boundary where untrusted or uncontrolled data enters a system.

## Type errors versus value errors

`TypeError` generally means the object has an inappropriate type for an operation.

`ValueError` generally means the type is acceptable but the value is inappropriate.

For example, a function expecting a numeric value might raise `TypeError` for a string and `ValueError` for a negative number when only positive values are permitted.

This distinction helps callers handle failures correctly.

## Function API design

A function is an API when other code depends on its interface.

A good function interface should make clear:

- What arguments are required
- What arguments are optional
- What types are expected
- What value is returned
- Which exceptions can occur
- Which side effects occur
- Which assumptions are required

Changing parameter names can affect keyword callers. Changing return types can affect consumers. Removing exceptions or introducing new ones can also affect callers.

API design therefore requires attention to compatibility.

## Sentinel values

Sometimes `None` is a valid value and cannot safely mean "no default supplied."

The script uses a unique object as a sentinel:

    _MISSING = object()

This allows the function to distinguish:

- No default supplied
- Default explicitly supplied as `None`

Sentinels are useful in flexible APIs.

## Function overloading

Python does not provide traditional function overloading in which multiple functions with the same name coexist based on parameter types.

Defining a second function with the same name replaces the previous definition.

Python can instead use:

- Default parameters
- `*args`
- `**kwargs`
- Explicit type checks
- Dispatch dictionaries
- `functools.singledispatch`

The script demonstrates `singledispatch`.

## `functools.singledispatch`

`singledispatch` selects an implementation based on the type of the first argument.

The script registers implementations for:

- `int`
- `float`
- `str`

and provides a default implementation for other types.

This is useful when type-based behavior is appropriate without manually writing a large conditional chain.

## Caching and memoization

Memoization stores results so that repeated calls with the same inputs do not have to recompute them.

The script demonstrates a basic custom memoization decorator and `functools.lru_cache`.

Caching can significantly improve performance when:

- Computations are expensive
- Inputs repeat
- Results are stable
- Cache size is manageable

Caching can be harmful when data changes frequently, results are large, memory is limited, or stale values are unacceptable.

## `functools.lru_cache`

`lru_cache` provides a standard implementation of a least-recently-used cache.

The script demonstrates:

    fibonacci_cached.cache_clear()
    fibonacci_cached.cache_info()

Production caching decisions require consideration of memory usage, invalidation, concurrency, object lifetime, and data freshness.

## Performance considerations

Functions introduce some calling overhead, but algorithmic complexity usually matters much more than individual function-call overhead.

The Fibonacci examples illustrate this clearly.

A naive recursive Fibonacci implementation has exponential time complexity.

An iterative Fibonacci implementation has linear time complexity.

Memoization can reduce repeated computation dramatically.

Optimization should normally follow measurement rather than intuition.

The script uses `time.perf_counter()` for simple timing measurements.

## Generators and memory

A list stores all its elements in memory.

A generator produces elements lazily.

For a large sequence, replacing:

    list_of_results = [...]

with a generator can reduce memory consumption when the caller can process results incrementally.

The trade-off is that generators are consumed as they are iterated and generally cannot be indexed like lists.

## Decorator performance

Each decorator wrapper introduces another function call layer.

Most decorators have negligible impact compared with expensive I/O or computation, but decorators can matter in extremely hot code paths.

Decorators should therefore add meaningful behavior rather than unnecessary layers.

## Exception performance

Exceptions are designed primarily for exceptional control flow, not ordinary branching.

The script compares conversion strategies to illustrate the design question.

The correct approach depends on the operation and input contract.

For operations such as integer conversion, Python's conversion function naturally communicates invalid input through `ValueError`, so handling that exception can be simpler and more reliable than trying to reproduce the conversion rules manually.

Performance should be measured for genuinely performance-sensitive code rather than optimized prematurely.

## Security considerations

Functions and modules can introduce security risks when they process untrusted data.

### Avoid unsafe `eval()`

The script's calculator parser deliberately does not use `eval()`.

Executing arbitrary input with `eval()` can allow an attacker to execute Python expressions and potentially perform harmful operations.

A safe parser should explicitly define which syntax and operations are allowed.

### Validate ranges

Unbounded user input can cause excessive computation or memory use.

The script includes bounded integer validation to demonstrate the principle.

### Validate paths

Filename and path input can be abused through path traversal patterns such as parent-directory references.

The script demonstrates basic filename-component validation.

Real filesystem security requires stronger controls, including trusted base directories, proper path resolution, authorization, and careful handling of symbolic links where relevant.

### Do not hide errors

Suppressing all exceptions can conceal security failures and operational problems.

Errors should be handled deliberately and logged appropriately without exposing secrets or sensitive internal information.

### Avoid sensitive exception messages

Error messages returned to external users should not reveal:

- Passwords
- Tokens
- Credentials
- Internal file paths
- Database connection details
- Stack traces containing sensitive information

Detailed diagnostics should be restricted to appropriate internal logs.

## Logging and debugging

The script demonstrates logging behavior through a decorator.

In production systems, Python's `logging` module is normally preferable to uncontrolled `print()` statements.

Good diagnostic information should identify:

- What operation failed
- Relevant non-sensitive context
- The exception type
- The original cause
- Request or transaction identifiers where appropriate

Logging should not expose secrets or personal information unnecessarily.

## Testing functions

Functions are natural units of testing because they can have explicit inputs and outputs.

The script includes a small educational `assert_equal()` helper and a collection of deterministic tests.

Tests cover:

- Correct results
- Boundary conditions
- Expected exceptions
- Invalid commands
- Validation failures

Production projects normally use a dedicated testing framework rather than a home-grown test helper.

A robust test suite should include:

- Normal cases
- Boundary cases
- Invalid input
- Exception behavior
- Regression cases
- Integration behavior where appropriate

## Common mistakes

### Forgetting to return

Printing a result is not equivalent to returning it.

A caller cannot meaningfully reuse a printed value.

Prefer:

    return result

when the function's purpose is to provide a value.

### Returning inconsistent types

A function that returns a number in one branch and a dictionary in another can be difficult to use.

Consistent return contracts improve reliability.

### Using global state unnecessarily

Global mutable state can create hidden dependencies between function calls.

Prefer explicit parameters and return values where possible.

### Mutable default arguments

Use `None` or another deliberate sentinel for optional mutable values unless persistent shared state is specifically intended.

### Overusing lambdas

Short lambdas are useful, but complicated lambda expressions reduce readability.

Use named functions when behavior deserves a name or documentation.

### Catching every exception

A broad handler can hide programming errors.

Catch the narrowest useful exception type at the appropriate layer.

### Raising generic exceptions

Domain-specific exception types make error handling more precise.

### Swallowing exceptions

Code such as an empty exception handler can silently discard important failures.

If an exception is intentionally ignored, the reason should be clear and the consequences understood.

### Circular imports

Circular module dependencies often indicate that responsibilities are poorly separated.

### Import-time side effects

Reusable modules should not unexpectedly perform expensive or destructive operations simply because another module imports them.

### Losing decorator metadata

Use `functools.wraps()` when writing decorators around ordinary functions.

## Real-world applications

Functions and modules appear in virtually every substantial Python application.

### Data analysis

Functions can clean, transform, validate, aggregate, and summarize datasets.

Modules can separate data ingestion, transformation, statistical calculations, and reporting.

### Web applications

Functions implement request handlers, validation logic, business rules, authentication checks, and database operations.

Modules and packages separate application layers.

### Financial systems

Functions can calculate:

- Interest
- EMI
- Investment returns
- Loan schedules
- Risk metrics
- Financial ratios

Exceptions can represent invalid transactions and business-rule violations.

### Automation

Functions can encapsulate repetitive operations and provide reusable workflows.

Modules can separate configuration, processing, validation, and output.

### Scientific computing

Functions can represent mathematical models, transformations, simulations, and numerical algorithms.

Modules can organize related models and numerical utilities.

### Software libraries

Public functions form APIs used by other programs.

Clear parameters, return values, exceptions, documentation, and module organization are essential for library quality.

## Practical function design principles

The script includes a set of design principles that can be applied to production code:

- Give each function a clear responsibility.
- Use meaningful names.
- Keep interfaces explicit.
- Avoid unnecessary global state.
- Validate inputs at meaningful boundaries.
- Raise specific exceptions.
- Document non-obvious behavior.
- Use type hints where useful.
- Avoid mutable default arguments.
- Preserve decorator metadata.
- Avoid hiding unexpected exceptions.
- Keep modules cohesive.
- Avoid unnecessary circular dependencies.
- Separate reusable definitions from startup behavior.
- Test both successful and failing paths.
- Measure performance before optimizing.

## Relationship between functions, modules, packages, and exceptions

These concepts operate at different levels.

A function provides a unit of behavior.

A module provides a unit of source-code organization.

A package provides a unit of larger-scale code organization.

An exception provides a mechanism for communicating abnormal conditions across function and module boundaries.

A practical Python application often has this structure conceptually:

    application
        |
        +-- package
        |     |
        |     +-- module
        |           |
        |           +-- functions
        |           +-- classes
        |           +-- constants
        |
        +-- exception hierarchy
        |
        +-- application entry point

Functions can raise exceptions. Modules can define exception classes. Packages can expose selected public functions and classes. The main application can decide how failures should ultimately be presented or handled.

## Implementation considerations

Reusable functions should minimize hidden assumptions.

A function that depends on external state should make that dependency visible when practical.

For example, a function that requires a tax rate can receive the rate as a parameter rather than reading a hidden global variable.

Module boundaries should reflect meaningful responsibilities rather than arbitrary file sizes.

Exception types should reflect the domain in which they are used.

Performance improvements should be based on measurements.

Security validation should occur before untrusted values reach dangerous operations.

## Advanced concepts represented in the script

The Python script includes several concepts beyond ordinary function definitions:

- Closures
- Function factories
- Higher-order functions
- Decorators
- Parameterized decorators
- Recursion
- Generators
- Function composition
- Memoization
- `functools.lru_cache`
- `functools.singledispatch`
- Callback functions
- Sentinel objects
- Context managers
- Exception chaining
- Custom exception hierarchies
- `ExceptionGroup`
- `except*`
- Command dispatch
- Batch error handling
- Security-aware parsing
- Positional-only parameters
- Keyword-only parameters
- Type annotations
- Module namespaces
- Package structures
- Import guards

These mechanisms demonstrate how relatively small Python language features can be combined to create reusable application architecture.

## Scope, modules, and namespaces compared

Scope determines where a name can be resolved.

A namespace is the mapping containing names and their associated objects.

A module is an organizational unit that owns a namespace.

These concepts are related but not identical.

For example, a module can contain a global namespace, while a function invocation creates a local namespace.

Understanding this distinction makes the behavior of imports, nested functions, closures, and name lookup much easier to reason about.

## Exceptions compared with return values

Returning a value is normally appropriate for expected successful results.

Exceptions are appropriate when the operation cannot satisfy its normal contract or when an abnormal condition needs to propagate.

For example:

    calculate_discounted_price(...)

can return a calculated price under valid conditions.

It can raise `ValidationError` when the supplied price or discount is invalid.

The choice depends on the API contract. Expected alternatives can sometimes be represented through return values, while exceptional conditions can be communicated through exceptions.

## Modules compared with functions

Functions organize behavior.

Modules organize related definitions and behavior.

A function is usually small enough to describe one operation. A module can contain many related functions and classes.

A module should not become merely a random collection of unrelated utilities. Cohesion is an important consideration.

## Packages compared with modules

A module is generally a single Python source file.

A package is a larger namespace structure containing related modules and potentially subpackages.

A small program may only need a few modules. Larger systems benefit from package-level organization.

## Limitations of the examples

The script is intentionally self-contained and therefore uses simplified examples.

The custom cache is educational and does not implement production cache invalidation, expiration, persistence, or distributed synchronization.

The retry function demonstrates the concept but does not implement production backoff, jitter, cancellation, or sophisticated transient-error classification.

The filename validation example is not a complete filesystem security solution.

The command parser intentionally supports a tiny explicit grammar and is not a general mathematical expression parser.

The timing examples are educational measurements rather than rigorous benchmarks.

The testing helper demonstrates assertions but is not intended to replace a dedicated testing framework.

These limitations are deliberate because the focus is understanding the underlying Python mechanisms and design principles.
