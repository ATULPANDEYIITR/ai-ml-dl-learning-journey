"""
Python Data Structures
======================

A self-contained progression from absolute beginner to advanced practice covering:

1. Lists
2. Tuples
3. Sets and frozensets
4. Dictionaries
5. Comprehensions
6. Nested data structures
7. Mutability, identity, copying, aliasing
8. Slicing and unpacking
9. Sorting and custom keys
10. Hashability
11. Complexity and performance
12. Iteration and mutation pitfalls
13. Practical data-processing patterns
14. Validation and defensive programming
15. Testing
16. Advanced combinations of data structures
17. Production-oriented design considerations

Run this file directly:

    python data_structures.py

The script intentionally uses mostly the Python standard library.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import reduce
from itertools import chain, groupby
import copy
import math
import random
import sys
import timeit
from typing import Any, Iterable


# ============================================================================
# 0. UTILITY FUNCTIONS
# ============================================================================

def section(title: str) -> None:
    """Print a visually distinct section heading."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    """Print a subsection heading."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def show(label: str, value: Any) -> None:
    """Print a labeled value."""
    print(f"{label}: {value!r}")


def safe_run(description: str, function) -> None:
    """Run an example and display expected exceptions without stopping the script."""
    try:
        result = function()
        print(f"{description}: {result!r}")
    except Exception as exc:
        print(f"{description}: {type(exc).__name__}: {exc}")


# ============================================================================
# 1. FOUNDATIONS: WHAT IS A DATA STRUCTURE?
# ============================================================================

def foundations_demo() -> None:
    section("1. DATA STRUCTURE FOUNDATIONS")

    print(
        """
A data structure is a way of organizing values so that programs can store,
access, update, search, transform, and communicate data efficiently.

Python's core collection types have different design goals:

    list       ordered, mutable sequence
    tuple      ordered, immutable sequence
    set        unordered collection of unique hashable values
    dict       mapping from hashable keys to values

The right structure depends on questions such as:

    - Does order matter?
    - Are duplicate values meaningful?
    - Should the collection be mutable?
    - Do we need fast membership testing?
    - Do we need lookup by a key?
    - What operations dominate the workload?
"""
    )

    integer_value = 42
    text_value = "Python"
    boolean_value = True
    none_value = None

    show("integer", integer_value)
    show("string", text_value)
    show("boolean", boolean_value)
    show("None", none_value)

    numbers = [10, 20, 30]
    coordinates = (10, 20)
    unique_numbers = {10, 20, 30}
    person = {"name": "Asha", "age": 28}

    show("list", numbers)
    show("tuple", coordinates)
    show("set", unique_numbers)
    show("dictionary", person)

    print(
        "\nA collection can itself contain other collections. "
        "This becomes important when representing real-world records."
    )


# ============================================================================
# 2. LISTS
# ============================================================================

def lists_demo() -> None:
    section("2. LISTS")

    subsection("2.1 Creating lists")

    empty_list = []
    numbers = [10, 20, 30, 40]
    mixed = [10, "Python", 3.14, True, None]

    show("empty list", empty_list)
    show("numbers", numbers)
    show("mixed list", mixed)

    # list() can construct a list from any iterable.
    from_range = list(range(5))
    from_string = list("ABC")

    show("list(range(5))", from_range)
    show("list('ABC')", from_string)

    subsection("2.2 Indexing")

    values = ["zero", "one", "two", "three", "four"]

    # Positive indexes start at zero.
    show("values[0]", values[0])
    show("values[2]", values[2])

    # Negative indexes count from the end.
    show("values[-1]", values[-1])
    show("values[-2]", values[-2])

    safe_run("out-of-range indexing", lambda: values[100])

    subsection("2.3 Slicing")

    numbers = [0, 1, 2, 3, 4, 5, 6]

    # Slice syntax is sequence[start:stop:step].
    show("numbers[1:5]", numbers[1:5])
    show("numbers[:4]", numbers[:4])
    show("numbers[3:]", numbers[3:])
    show("numbers[::2]", numbers[::2])
    show("numbers[::-1]", numbers[::-1])

    subsection("2.4 Adding elements")

    values = [1, 2]

    values.append(3)
    show("after append(3)", values)

    values.extend([4, 5])
    show("after extend([4, 5])", values)

    values.insert(0, 0)
    show("after insert(0, 0)", values)

    subsection("2.5 Removing elements")

    values = ["a", "b", "c", "d", "b"]

    removed_last = values.pop()
    show("pop returned", removed_last)
    show("after pop()", values)

    removed_first = values.pop(0)
    show("pop(0) returned", removed_first)
    show("after pop(0)", values)

    values.remove("b")
    show("after remove('b')", values)

    # remove() raises ValueError if the requested value does not exist.
    safe_run("remove missing value", lambda: values.remove("missing"))

    subsection("2.6 Searching")

    values = ["apple", "banana", "cherry", "banana"]

    show("'banana' in values", "banana" in values)
    show("'orange' in values", "orange" in values)
    show("index of banana", values.index("banana"))
    show("count of banana", values.count("banana"))

    subsection("2.7 Updating elements")

    values = [10, 20, 30]
    values[1] = 99
    show("after values[1] = 99", values)

    values[0:2] = [100, 200, 300]
    show("after slice assignment", values)

    # Slice assignment can change the list's length.
    values[1:3] = [7]
    show("after shrinking slice", values)

    subsection("2.8 Sorting")

    numbers = [5, 1, 8, 2, 3]

    # sorted() returns a new list.
    sorted_numbers = sorted(numbers)
    show("original", numbers)
    show("sorted()", sorted_numbers)

    # list.sort() modifies the existing list.
    numbers.sort()
    show("after sort()", numbers)

    numbers.sort(reverse=True)
    show("descending", numbers)

    subsection("2.9 Custom sorting keys")

    students = [
        {"name": "Ravi", "score": 72},
        {"name": "Anita", "score": 91},
        {"name": "Kabir", "score": 84},
    ]

    # key receives each element and determines its comparison value.
    by_score = sorted(students, key=lambda student: student["score"], reverse=True)
    show("students sorted by score", by_score)

    by_name = sorted(students, key=lambda student: student["name"])
    show("students sorted by name", by_name)

    subsection("2.10 Reversing")

    values = [1, 2, 3, 4]
    values.reverse()
    show("reverse()", values)

    # reversed() creates an iterator rather than modifying the list.
    values = [1, 2, 3]
    show("list(reversed(values))", list(reversed(values)))
    show("values unchanged", values)

    subsection("2.11 Copying lists")

    original = [1, 2, 3]
    copied = original.copy()

    copied.append(4)

    show("original", original)
    show("shallow copied list", copied)

    subsection("2.12 Aliasing")

    original = [1, 2, 3]
    alias = original

    # Both variables refer to the same list object.
    alias.append(4)

    show("original after alias mutation", original)
    show("alias", alias)
    show("original is alias", original is alias)

    subsection("2.13 List concatenation and repetition")

    a = [1, 2]
    b = [3, 4]

    show("a + b", a + b)
    show("a * 3", a * 3)

    subsection("2.14 List as a stack")

    stack: list[str] = []

    stack.append("first")
    stack.append("second")
    stack.append("third")

    show("stack", stack)
    show("pop", stack.pop())
    show("stack after pop", stack)

    subsection("2.15 List as a queue: important performance distinction")

    queue = ["A", "B", "C"]

    # This works logically, but pop(0) is O(n) because remaining elements
    # must shift left.
    first = queue.pop(0)

    show("removed first element", first)
    show("remaining queue", queue)

    # collections.deque is normally preferable for frequent operations
    # at both ends. It is demonstrated later.

    subsection("2.16 Nested lists")

    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    show("matrix", matrix)
    show("matrix[1][2]", matrix[1][2])

    diagonal = [matrix[i][i] for i in range(len(matrix))]
    show("diagonal", diagonal)


# ============================================================================
# 3. TUPLES
# ============================================================================

def tuples_demo() -> None:
    section("3. TUPLES")

    subsection("3.1 Creating tuples")

    empty = ()
    coordinates = (10, 20)
    mixed = ("Python", 3.14, True)

    show("empty tuple", empty)
    show("coordinates", coordinates)
    show("mixed tuple", mixed)

    # A one-element tuple requires a trailing comma.
    one_element = (42,)
    not_a_tuple = (42)

    show("one-element tuple", one_element)
    show("(42) type", type(not_a_tuple).__name__)

    subsection("3.2 Tuple indexing and slicing")

    values = ("a", "b", "c", "d")

    show("values[0]", values[0])
    show("values[-1]", values[-1])
    show("values[1:3]", values[1:3])

    subsection("3.3 Tuple immutability")

    values = (10, 20, 30)

    safe_run("attempt to change tuple", lambda: values.__setitem__(0, 99))

    subsection("3.4 Tuple unpacking")

    point = (10, 20)

    x, y = point

    show("x", x)
    show("y", y)

    subsection("3.5 Extended unpacking")

    values = [1, 2, 3, 4, 5]

    first, *middle, last = values

    show("first", first)
    show("middle", middle)
    show("last", last)

    subsection("3.6 Swapping variables")

    left = "L"
    right = "R"

    left, right = right, left

    show("left", left)
    show("right", right)

    subsection("3.7 Tuple methods")

    values = (1, 2, 2, 3, 2)

    show("count(2)", values.count(2))
    show("index(3)", values.index(3))

    subsection("3.8 Tuples as dictionary keys")

    locations = {
        (28.6139, 77.2090): "Delhi",
        (26.8467, 80.9462): "Lucknow",
    }

    show("location lookup", locations[(28.6139, 77.2090)])

    print(
        "\nA tuple can be a dictionary key when all of its elements are hashable. "
        "This is one practical consequence of immutability and hashability."
    )

    subsection("3.9 Mutable objects inside tuples")

    # A tuple cannot replace an element, but an object stored inside it may
    # itself be mutable.
    data = ([1, 2], [3, 4])

    data[0].append(99)

    show("tuple containing lists", data)

    print(
        "\nTuple immutability is shallow: the tuple's references cannot be "
        "replaced, but referenced mutable objects can still change."
    )

    subsection("3.10 Named tuples conceptually")

    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    point = Point(10, 20)

    show("namedtuple", point)
    show("point.x", point.x)
    show("point.y", point.y)


# ============================================================================
# 4. SETS
# ============================================================================

def sets_demo() -> None:
    section("4. SETS")

    subsection("4.1 Creating sets")

    empty_set = set()
    values = {1, 2, 3, 4}
    duplicates_removed = set([1, 2, 2, 3, 3, 3])

    show("empty set", empty_set)
    show("set literal", values)
    show("duplicates removed", duplicates_removed)

    # {} creates an empty dictionary, not an empty set.
    empty_dictionary = {}
    show("{} type", type(empty_dictionary).__name__)

    subsection("4.2 Uniqueness")

    names = ["Asha", "Ravi", "Asha", "Kabir", "Ravi"]
    unique_names = set(names)

    show("original names", names)
    show("unique names", unique_names)

    subsection("4.3 Membership testing")

    languages = {"Python", "Java", "C++"}

    show("'Python' in languages", "Python" in languages)
    show("'Rust' in languages", "Rust" in languages)

    subsection("4.4 Adding and removing")

    values = {1, 2, 3}

    values.add(4)
    show("after add(4)", values)

    values.update([5, 6])
    show("after update", values)

    values.remove(6)
    show("after remove(6)", values)

    # discard() does not raise an exception if the value is absent.
    values.discard(999)
    show("after discard(999)", values)

    safe_run("remove missing set value", lambda: values.remove(999))

    subsection("4.5 Set operations")

    developers = {"Python", "SQL", "Git"}
    analysts = {"SQL", "Excel", "Python"}

    show("union", developers | analysts)
    show("intersection", developers & analysts)
    show("developers only", developers - analysts)
    show("symmetric difference", developers ^ analysts)

    # Equivalent method forms are useful when operands are not convenient
    # to express with operators.
    show("union()", developers.union(analysts))
    show("intersection()", developers.intersection(analysts))

    subsection("4.6 Subsets and supersets")

    small = {1, 2}
    large = {1, 2, 3, 4}

    show("small <= large", small <= large)
    show("large >= small", large >= small)
    show("small.issubset(large)", small.issubset(large))
    show("large.issuperset(small)", large.issuperset(small))

    subsection("4.7 Disjoint sets")

    a = {1, 2}
    b = {3, 4}

    show("a.isdisjoint(b)", a.isdisjoint(b))

    subsection("4.8 Frozenset")

    immutable_set = frozenset({1, 2, 3})

    show("frozenset", immutable_set)

    # A frozenset is hashable and can therefore be used as a dictionary key
    # or as an element of another set.
    grouped = {
        immutable_set: "immutable group"
    }

    show("frozenset as dict key", grouped[immutable_set])

    subsection("4.9 Set limitations")

    print(
        """
Set elements must be hashable. Lists and dictionaries are mutable and
therefore cannot normally be set elements.

For example:
    { [1, 2] }       -> TypeError
    { {"a": 1} }     -> TypeError

Use tuples or frozensets when an immutable, hashable representation is
appropriate.
"""
    )

    safe_run(
        "list as set element",
        lambda: { [1, 2] },
    )


# ============================================================================
# 5. DICTIONARIES
# ============================================================================

def dictionaries_demo() -> None:
    section("5. DICTIONARIES")

    subsection("5.1 Creating dictionaries")

    empty = {}
    person = {
        "name": "Asha",
        "age": 28,
        "city": "Lucknow",
    }

    show("empty dictionary", empty)
    show("person", person)

    subsection("5.2 Dictionary keys and values")

    print(
        """
A dictionary stores key-value pairs.

Keys:
    - must be hashable
    - must be unique within the dictionary

Values:
    - may be mutable or immutable
    - may contain arbitrary Python objects
"""
    )

    subsection("5.3 Accessing values")

    person = {"name": "Asha", "age": 28}

    show("person['name']", person["name"])
    show("person.get('name')", person.get("name"))
    show("missing key with get()", person.get("country"))
    show(
        "missing key with get(default)",
        person.get("country", "Unknown"),
    )

    safe_run("missing key with []", lambda: person["country"])

    subsection("5.4 Adding and updating")

    person["country"] = "India"
    person["age"] = 29

    show("updated person", person)

    person.update({"occupation": "Engineer", "age": 30})
    show("after update()", person)

    subsection("5.5 Removing dictionary entries")

    person = {
        "name": "Asha",
        "age": 30,
        "city": "Lucknow",
    }

    removed_age = person.pop("age")
    show("pop returned", removed_age)
    show("dictionary", person)

    removed_item = person.popitem()
    show("popitem returned", removed_item)
    show("dictionary", person)

    subsection("5.6 Membership")

    person = {"name": "Asha", "age": 30}

    # "in" checks keys, not values.
    show("'name' in person", "name" in person)
    show("'Asha' in person", "Asha" in person)
    show("'Asha' in person.values()", "Asha" in person.values())

    subsection("5.7 keys(), values(), items()")

    person = {
        "name": "Asha",
        "age": 30,
        "city": "Lucknow",
    }

    show("keys", list(person.keys()))
    show("values", list(person.values()))
    show("items", list(person.items()))

    subsection("5.8 Iterating over dictionaries")

    for key in person:
        print(f"key -> {key}")

    for key, value in person.items():
        print(f"{key} -> {value}")

    subsection("5.9 setdefault()")

    inventory: dict[str, list[str]] = {}

    inventory.setdefault("fruits", []).append("apple")
    inventory.setdefault("fruits", []).append("banana")
    inventory.setdefault("vegetables", []).append("carrot")

    show("inventory", inventory)

    subsection("5.10 Dictionary unpacking")

    defaults = {"timeout": 30, "retries": 3}
    overrides = {"retries": 5, "debug": True}

    combined = {**defaults, **overrides}

    show("combined", combined)

    # Later mappings override earlier mappings when keys collide.
    subsection("5.11 Dictionary union operators")

    left = {"a": 1, "b": 2}
    right = {"b": 99, "c": 3}

    show("left | right", left | right)

    mutable_copy = left.copy()
    mutable_copy |= right
    show("left copy after |=", mutable_copy)

    subsection("5.12 Dictionary comprehensions")

    squares = {number: number * number for number in range(6)}

    show("squares", squares)

    even_squares = {
        number: number * number
        for number in range(10)
        if number % 2 == 0
    }

    show("even squares", even_squares)

    subsection("5.13 Sorting dictionaries")

    scores = {
        "Ravi": 72,
        "Anita": 91,
        "Kabir": 84,
    }

    sorted_by_score = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    show("sorted score pairs", sorted_by_score)

    subsection("5.14 Building a dictionary from pairs")

    pairs = [
        ("name", "Asha"),
        ("age", 30),
        ("city", "Lucknow"),
    ]

    show("dict(pairs)", dict(pairs))

    subsection("5.15 Dictionary key collisions")

    # These keys are equal according to Python's equality rules.
    collision_example = {
        1: "integer",
        True: "boolean",
    }

    show("collision_example", collision_example)

    print(
        "\nBecause 1 == True and they have compatible hashes, the second "
        "entry replaces the first value for the same dictionary key."
    )

    subsection("5.16 Dictionary with tuple keys")

    distances = {
        ("Delhi", "Agra"): 233,
        ("Agra", "Lucknow"): 330,
    }

    show("Delhi -> Agra", distances[("Delhi", "Agra")])


# ============================================================================
# 6. COMPREHENSIONS
# ============================================================================

def comprehensions_demo() -> None:
    section("6. COMPREHENSIONS")

    subsection("6.1 List comprehension")

    squares = [number * number for number in range(10)]
    show("squares", squares)

    subsection("6.2 Filtering")

    even_numbers = [
        number
        for number in range(20)
        if number % 2 == 0
    ]

    show("even numbers", even_numbers)

    subsection("6.3 Conditional expression inside comprehension")

    labels = [
        "even" if number % 2 == 0 else "odd"
        for number in range(8)
    ]

    show("labels", labels)

    subsection("6.4 Transforming strings")

    names = ["alice", "BOB", "Charlie"]

    normalized = [
        name.strip().lower()
        for name in names
    ]

    show("normalized names", normalized)

    subsection("6.5 Set comprehension")

    unique_lengths = {
        len(word)
        for word in ["cat", "dog", "elephant", "horse", "cat"]
    }

    show("unique lengths", unique_lengths)

    subsection("6.6 Dictionary comprehension")

    cubes = {
        number: number ** 3
        for number in range(6)
    }

    show("cubes", cubes)

    subsection("6.7 Conditional dictionary comprehension")

    even_cubes = {
        number: number ** 3
        for number in range(10)
        if number % 2 == 0
    }

    show("even cubes", even_cubes)

    subsection("6.8 Nested loops in comprehensions")

    pairs = [
        (x, y)
        for x in range(3)
        for y in range(3)
    ]

    show("Cartesian product", pairs)

    subsection("6.9 Nested list comprehension")

    matrix = [
        [row * 3 + column for column in range(3)]
        for row in range(3)
    ]

    show("matrix", matrix)

    subsection("6.10 Flattening nested lists")

    nested = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    flattened = [
        value
        for group in nested
        for value in group
    ]

    show("flattened", flattened)

    subsection("6.11 Equivalent explicit loop")

    squares_comprehension = [
        number * number
        for number in range(5)
    ]

    squares_loop: list[int] = []

    for number in range(5):
        squares_loop.append(number * number)

    show("comprehension result", squares_comprehension)
    show("loop result", squares_loop)
    show("results equal", squares_comprehension == squares_loop)

    subsection("6.12 Comprehension readability")

    print(
        """
Comprehensions are useful when the transformation is short and clear.

Prefer:

    squares = [n * n for n in numbers]

over a complicated one-line expression containing many conditions,
nested functions, side effects, or difficult-to-read branches.

A normal loop is often clearer when the logic becomes complex.
"""
    )

    subsection("6.13 Generator expressions")

    generator = (number * number for number in range(5))

    show("generator type", type(generator).__name__)
    show("generator consumed as list", list(generator))

    # Generator expressions are lazy: values are produced as requested.
    another_generator = (number * number for number in range(1_000_000))

    first_three = [next(another_generator) for _ in range(3)]

    show("first three generated values", first_three)


# ============================================================================
# 7. NESTED DATA STRUCTURES
# ============================================================================

def nested_structures_demo() -> None:
    section("7. NESTED DATA STRUCTURES")

    subsection("7.1 List of dictionaries")

    employees = [
        {
            "id": 101,
            "name": "Asha",
            "department": "Engineering",
            "skills": ["Python", "SQL"],
        },
        {
            "id": 102,
            "name": "Ravi",
            "department": "Analytics",
            "skills": ["SQL", "Excel"],
        },
        {
            "id": 103,
            "name": "Kabir",
            "department": "Engineering",
            "skills": ["Python", "FastAPI"],
        },
    ]

    show("employees", employees)

    first_employee_name = employees[0]["name"]
    show("first employee name", first_employee_name)

    first_employee_skills = employees[0]["skills"]
    show("first employee skills", first_employee_skills)

    subsection("7.2 Filtering nested records")

    engineering_employees = [
        employee
        for employee in employees
        if employee["department"] == "Engineering"
    ]

    show("engineering employees", engineering_employees)

    subsection("7.3 Extracting nested values")

    all_skills = [
        skill
        for employee in employees
        for skill in employee["skills"]
    ]

    show("all skills", all_skills)

    unique_skills = {
        skill
        for employee in employees
        for skill in employee["skills"]
    }

    show("unique skills", unique_skills)

    subsection("7.4 Dictionary indexed by ID")

    employees_by_id = {
        employee["id"]: employee
        for employee in employees
    }

    show("employee 102", employees_by_id[102])

    subsection("7.5 Nested dictionaries")

    company = {
        "name": "Example Corp",
        "offices": {
            "Delhi": {
                "employees": 120,
                "manager": "Ravi",
            },
            "Lucknow": {
                "employees": 80,
                "manager": "Asha",
            },
        },
    }

    show("Delhi employee count", company["offices"]["Delhi"]["employees"])

    subsection("7.6 Deep traversal")

    total_employees = sum(
        office["employees"]
        for office in company["offices"].values()
    )

    show("total employees", total_employees)

    subsection("7.7 Matrix represented as nested lists")

    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    transposed = [
        [matrix[row][column] for row in range(len(matrix))]
        for column in range(len(matrix[0]))
    ]

    show("matrix", matrix)
    show("transposed", transposed)

    subsection("7.8 Nested mixed structures")

    organization = {
        "engineering": {
            "teams": [
                {
                    "name": "Platform",
                    "members": ["Asha", "Kabir"],
                },
                {
                    "name": "API",
                    "members": ["Ravi"],
                },
            ]
        }
    }

    platform_members = (
        organization["engineering"]["teams"][0]["members"]
    )

    show("platform members", platform_members)

    subsection("7.9 Recursive traversal")

    nested = [
        1,
        [2, 3],
        [4, [5, 6]],
        {"ignored": "mapping"},
    ]

    def flatten_lists(value: Any) -> list[Any]:
        """Recursively flatten only list containers."""
        result: list[Any] = []

        if isinstance(value, list):
            for item in value:
                result.extend(flatten_lists(item))
        else:
            result.append(value)

        return result

    show("flattened nested lists", flatten_lists(nested))

    subsection("7.10 Safe nested dictionary access")

    data = {
        "user": {
            "profile": {
                "name": "Asha"
            }
        }
    }

    # Direct indexing is appropriate when missing data should be an error.
    show("known nested value", data["user"]["profile"]["name"])

    # get() can provide controlled fallback behavior.
    missing_city = (
        data.get("user", {})
        .get("profile", {})
        .get("city", "Unknown")
    )

    show("missing nested city", missing_city)


# ============================================================================
# 8. MUTABILITY, IDENTITY, ALIASING, AND COPYING
# ============================================================================

def mutability_demo() -> None:
    section("8. MUTABILITY, IDENTITY, ALIASING, AND COPYING")

    subsection("8.1 Mutable versus immutable objects")

    mutable_list = [1, 2, 3]
    immutable_tuple = (1, 2, 3)

    mutable_list.append(4)

    show("mutable list after append", mutable_list)
    show("immutable tuple", immutable_tuple)

    print(
        """
Common mutable built-in objects:
    list
    set
    dict
    bytearray

Common immutable built-in objects:
    int
    float
    bool
    str
    tuple
    frozenset
    bytes
"""
    )

    subsection("8.2 == versus is")

    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a

    show("a == b", a == b)
    show("a is b", a is b)
    show("a is c", a is c)

    print(
        """
== asks whether objects are equal in value.

is asks whether two references point to the same object.

Use `is` primarily for identity checks such as:
    value is None

Do not normally replace value equality with identity.
"""
    )

    subsection("8.3 Shallow copies")

    original = [
        ["A", "B"],
        ["C", "D"],
    ]

    shallow = original.copy()

    shallow[0].append("CHANGED")

    show("original after nested mutation", original)
    show("shallow copy", shallow)

    print(
        "\nA shallow copy creates a new outer list but retains references "
        "to the same nested objects."
    )

    subsection("8.4 Deep copies")

    original = [
        ["A", "B"],
        ["C", "D"],
    ]

    deep = copy.deepcopy(original)

    deep[0].append("CHANGED")

    show("original", original)
    show("deep copy", deep)

    subsection("8.5 List multiplication aliasing trap")

    bad_matrix = [[0] * 3] * 3
    bad_matrix[0][0] = 99

    show("bad matrix", bad_matrix)

    print(
        "\nAll rows reference the same inner list in the bad construction."
    )

    good_matrix = [[0] * 3 for _ in range(3)]
    good_matrix[0][0] = 99

    show("good matrix", good_matrix)

    subsection("8.6 Mutable default argument trap")

    def bad_append(item: Any, bucket: list[Any] = []) -> list[Any]:
        # The default list is created once when the function is defined.
        bucket.append(item)
        return bucket

    show("bad_append first call", bad_append("A"))
    show("bad_append second call", bad_append("B"))

    def good_append(item: Any, bucket: list[Any] | None = None) -> list[Any]:
        # None lets each call create its own list.
        if bucket is None:
            bucket = []
        bucket.append(item)
        return bucket

    show("good_append first call", good_append("A"))
    show("good_append second call", good_append("B"))


# ============================================================================
# 9. HASHABILITY
# ============================================================================

def hashability_demo() -> None:
    section("9. HASHABILITY")

    print(
        """
Hashability matters because dictionaries and sets use hash tables.

An object that is hashable has:
    - a hash value that remains stable during its lifetime
    - equality behavior compatible with hashing

Typical hashable objects:
    int, float, bool, str, bytes, tuple of hashable elements,
    frozenset

Typical unhashable objects:
    list, dict, set
"""
    )

    hashable_values = [
        10,
        "Python",
        (1, 2, 3),
        frozenset({1, 2}),
    ]

    for value in hashable_values:
        print(f"hash({value!r}) = {hash(value)}")

    unhashable_values = [
        [1, 2],
        {"a": 1},
        {1, 2},
    ]

    for value in unhashable_values:
        safe_run(
            f"hash({value!r})",
            lambda value=value: hash(value),
        )

    subsection("9.1 Tuple hashability depends on its elements")

    safe_run(
        "hash((1, 2, 3))",
        lambda: hash((1, 2, 3)),
    )

    safe_run(
        "hash(([1], 2))",
        lambda: hash(([1], 2)),
    )

    subsection("9.2 Why mutable keys are problematic")

    print(
        """
A dictionary relies on a key's hash to locate the key.

If a key could change its hash after insertion, the dictionary could
lose the ability to locate it correctly. Python therefore restricts
dictionary keys to hashable objects.
"""
    )


# ============================================================================
# 10. ITERATION AND UNPACKING
# ============================================================================

def iteration_demo() -> None:
    section("10. ITERATION AND UNPACKING")

    subsection("10.1 Enumerate")

    names = ["Asha", "Ravi", "Kabir"]

    for index, name in enumerate(names, start=1):
        print(index, name)

    subsection("10.2 Zip")

    names = ["Asha", "Ravi", "Kabir"]
    scores = [91, 72, 84]

    paired = list(zip(names, scores))

    show("zip result", paired)

    for name, score in zip(names, scores):
        print(f"{name}: {score}")

    subsection("10.3 Zip stops at the shortest input")

    a = [1, 2, 3]
    b = ["x", "y"]

    show("zip(a, b)", list(zip(a, b)))

    # strict=True raises ValueError if lengths differ.
    safe_run(
        "strict zip",
        lambda: list(zip(a, b, strict=True)),
    )

    subsection("10.4 Unpacking dictionary items")

    person = {
        "name": "Asha",
        "age": 30,
    }

    for key, value in person.items():
        print(f"{key} = {value}")

    subsection("10.5 Star unpacking")

    values = [10, 20, 30, 40, 50]

    first, second, *rest = values
    show("first", first)
    show("second", second)
    show("rest", rest)

    first, *middle, second = values
    show("first", first)
    show("middle", middle)
    show("second", second)

    subsection("10.6 Parallel assignment")

    names = ["Asha", "Ravi", "Kabir"]

    first, second, third = names

    show("first", first)
    show("second", second)
    show("third", third)


# ============================================================================
# 11. MUTATING COLLECTIONS DURING ITERATION
# ============================================================================

def mutation_during_iteration_demo() -> None:
    section("11. MUTATING COLLECTIONS DURING ITERATION")

    subsection("11.1 Dangerous list mutation")

    numbers = [1, 2, 3, 4, 5, 6]

    # Removing while iterating can skip elements because indexes shift.
    for number in numbers:
        if number % 2 == 0:
            numbers.remove(number)

    show("result of dangerous mutation", numbers)

    subsection("11.2 Safe filtering")

    numbers = [1, 2, 3, 4, 5, 6]

    numbers = [
        number
        for number in numbers
        if number % 2 != 0
    ]

    show("safe filtered result", numbers)

    subsection("11.3 Safe dictionary mutation")

    data = {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    keys_to_delete = [
        key
        for key, value in data.items()
        if value % 2 == 0
    ]

    for key in keys_to_delete:
        del data[key]

    show("dictionary after safe deletion", data)

    subsection("11.4 Snapshot iteration")

    data = {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    for key in list(data):
        if key == "b":
            del data[key]

    show("after iterating over snapshot", data)


# ============================================================================
# 12. ADVANCED COLLECTION TYPES
# ============================================================================

def advanced_collections_demo() -> None:
    section("12. ADVANCED COLLECTION TYPES")

    subsection("12.1 deque")

    queue = deque(["A", "B", "C"])

    queue.append("D")
    show("deque after append", queue)

    show("popleft", queue.popleft())
    show("deque", queue)

    queue.appendleft("START")
    show("after appendleft", queue)

    queue.rotate(1)
    show("after rotate", queue)

    subsection("12.2 Counter")

    words = ["python", "sql", "python", "excel", "python", "sql"]

    counts = Counter(words)

    show("Counter", counts)
    show("most common", counts.most_common())

    subsection("12.3 defaultdict")

    groups: defaultdict[str, list[str]] = defaultdict(list)

    records = [
        ("Engineering", "Asha"),
        ("Analytics", "Ravi"),
        ("Engineering", "Kabir"),
    ]

    for department, name in records:
        groups[department].append(name)

    show("defaultdict grouping", dict(groups))

    subsection("12.4 defaultdict(int) for counting")

    counts = defaultdict(int)

    for letter in "banana":
        counts[letter] += 1

    show("letter counts", dict(counts))

    subsection("12.5 namedtuple")

    from collections import namedtuple

    Employee = namedtuple("Employee", ["name", "department"])
    employee = Employee("Asha", "Engineering")

    show("employee", employee)
    show("employee.name", employee.name)

    subsection("12.6 Dataclass for structured records")

    @dataclass(frozen=True)
    class Product:
        name: str
        price: float
        category: str

    product = Product("Keyboard", 2500.0, "Electronics")

    show("product", product)

    print(
        "\nA dataclass is often clearer than deeply nested dictionaries "
        "when a stable domain model has known fields."
    )


# ============================================================================
# 13. COMMON COLLECTION PATTERNS
# ============================================================================

def common_patterns_demo() -> None:
    section("13. COMMON DATA-STRUCTURE PATTERNS")

    subsection("13.1 Deduplicate while preserving order")

    values = ["A", "B", "A", "C", "B", "D"]

    seen: set[str] = set()
    unique_in_order: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            unique_in_order.append(value)

    show("original", values)
    show("unique in order", unique_in_order)

    subsection("13.2 Frequency counting")

    numbers = [1, 2, 2, 3, 3, 3, 4]

    frequency = Counter(numbers)

    show("frequency", frequency)

    subsection("13.3 Grouping records")

    employees = [
        {"name": "Asha", "department": "Engineering"},
        {"name": "Ravi", "department": "Analytics"},
        {"name": "Kabir", "department": "Engineering"},
    ]

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for employee in employees:
        grouped[employee["department"]].append(employee)

    show("grouped", dict(grouped))

    subsection("13.4 Inverting a dictionary")

    original = {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    inverted = {
        value: key
        for key, value in original.items()
    }

    show("inverted", inverted)

    print(
        "\nThis inversion is safe only when original values are unique "
        "and hashable."
    )

    subsection("13.5 Handling duplicate values while grouping")

    original = {
        "alice": "engineering",
        "bob": "sales",
        "charlie": "engineering",
    }

    inverted_grouped: defaultdict[str, list[str]] = defaultdict(list)

    for name, department in original.items():
        inverted_grouped[department].append(name)

    show("grouped inversion", dict(inverted_grouped))

    subsection("13.6 Building an index")

    products = [
        {"id": 101, "name": "Laptop"},
        {"id": 102, "name": "Phone"},
        {"id": 103, "name": "Monitor"},
    ]

    product_index = {
        product["id"]: product
        for product in products
    }

    requested_id = 102
    show(
        f"product lookup for ID {requested_id}",
        product_index.get(requested_id),
    )

    subsection("13.7 Partitioning values")

    numbers = list(range(10))

    even = [n for n in numbers if n % 2 == 0]
    odd = [n for n in numbers if n % 2 != 0]

    show("even", even)
    show("odd", odd)

    subsection("13.8 Flattening one level")

    nested = [[1, 2], [3, 4], [5, 6]]

    flattened = list(chain.from_iterable(nested))

    show("flattened", flattened)

    subsection("13.9 Running totals")

    values = [10, 20, 30, 40]

    running_totals: list[int] = []
    total = 0

    for value in values:
        total += value
        running_totals.append(total)

    show("running totals", running_totals)

    subsection("13.10 Sorting records with multiple criteria")

    employees = [
        {"name": "Asha", "department": "Engineering", "score": 91},
        {"name": "Ravi", "department": "Engineering", "score": 91},
        {"name": "Kabir", "department": "Analytics", "score": 84},
    ]

    sorted_employees = sorted(
        employees,
        key=lambda employee: (
            employee["department"],
            -employee["score"],
            employee["name"],
        ),
    )

    show("multi-key sort", sorted_employees)


# ============================================================================
# 14. MERGING AND COPYING NESTED STRUCTURES
# ============================================================================

def merging_and_copying_demo() -> None:
    section("14. MERGING AND COPYING NESTED STRUCTURES")

    subsection("14.1 Dictionary merge")

    base_config = {
        "host": "localhost",
        "port": 8000,
        "debug": False,
    }

    production_config = {
        "debug": False,
        "port": 443,
    }

    merged = base_config | production_config

    show("merged config", merged)

    subsection("14.2 List concatenation versus extend")

    first = [1, 2]
    second = [3, 4]

    combined = first + second

    show("first after +", first)
    show("combined", combined)

    first.extend(second)

    show("first after extend", first)

    print(
        """
`+` creates a new list.

`extend()` modifies the existing list.

This distinction matters when references to the original list exist.
"""
    )

    subsection("14.3 Shallow dictionary copy")

    original = {
        "name": "Asha",
        "skills": ["Python", "SQL"],
    }

    copied = original.copy()
    copied["skills"].append("FastAPI")

    show("original", original)
    show("copied", copied)

    subsection("14.4 Deep dictionary copy")

    original = {
        "name": "Asha",
        "skills": ["Python", "SQL"],
    }

    copied = copy.deepcopy(original)
    copied["skills"].append("FastAPI")

    show("original", original)
    show("deep copy", copied)


# ============================================================================
# 15. VALIDATION AND EDGE CASES
# ============================================================================

def validation_and_edge_cases_demo() -> None:
    section("15. VALIDATION AND EDGE CASES")

    subsection("15.1 Empty collections")

    empty_list: list[int] = []
    empty_tuple: tuple[int, ...] = ()
    empty_set: set[int] = set()
    empty_dict: dict[str, int] = {}

    show("empty list truth value", bool(empty_list))
    show("empty tuple truth value", bool(empty_tuple))
    show("empty set truth value", bool(empty_set))
    show("empty dict truth value", bool(empty_dict))

    print(
        "\nEmpty built-in collections are falsy. Non-empty collections are truthy."
    )

    subsection("15.2 Safe first-element access")

    values: list[int] = []

    first = values[0] if values else None

    show("safe first element", first)

    subsection("15.3 get() versus []")

    settings = {
        "timeout": 30,
    }

    show("existing key", settings.get("timeout"))
    show("missing key", settings.get("retries"))
    show("missing key with fallback", settings.get("retries", 3))

    subsection("15.4 Mutable values in dictionaries")

    data = {
        "numbers": [1, 2, 3],
    }

    data["numbers"].append(4)

    show("mutated nested value", data)

    subsection("15.5 Duplicate dictionary keys")

    duplicate_keys = {
        "name": "First",
        "name": "Second",
    }

    show("duplicate key result", duplicate_keys)

    subsection("15.6 Numeric equality can affect keys")

    numeric_keys = {
        1: "one",
        1.0: "one-point-zero",
    }

    show("numeric key collision", numeric_keys)

    subsection("15.7 NaN as an unusual dictionary/set value")

    nan = float("nan")

    print(
        """
NaN has unusual comparison semantics:
    nan != nan

This can produce surprising behavior in sets, dictionaries, sorting,
deduplication, and numerical processing. Avoid using NaN as an identifier.
"""
    )

    show("nan == nan", nan == nan)

    nan_set = {nan}
    show("nan in same set object", nan in nan_set)

    subsection("15.8 None versus missing")

    record = {
        "middle_name": None,
    }

    show("'middle_name' in record", "middle_name" in record)
    show("record.get('middle_name')", record.get("middle_name"))
    show("'email' in record", "email" in record)

    print(
        "\nA key mapped to None is different from a key that does not exist."
    )


# ============================================================================
# 16. PERFORMANCE AND BIG-O
# ============================================================================

def performance_demo() -> None:
    section("16. PERFORMANCE AND BIG-O")

    print(
        """
Approximate average-case characteristics for Python's built-in structures:

Operation                         list       set/dict
------------------------------------------------------------
Index by position                 O(1)       not applicable
Membership                        O(n)       O(1) average
Append to list                    O(1)*      not applicable
Insert at front                   O(n)       not applicable
Remove from front                 O(n)       not applicable
Dictionary key lookup             -          O(1) average
Set membership                    -          O(1) average

* List append is amortized O(1). A resize can occasionally require
  allocating a larger backing array and copying references.

Important:
    O(1) is an average-case hash-table expectation for sets/dicts,
    not a mathematical guarantee for every possible adversarial situation.
"""
    )

    subsection("16.1 Membership benchmark")

    setup = """
values = list(range(10000))
value_set = set(values)
target = 9999
"""

    list_time = timeit.timeit(
        "target in values",
        setup=setup,
        number=10000,
    )

    set_time = timeit.timeit(
        "target in value_set",
        setup=setup,
        number=10000,
    )

    print(f"List membership benchmark: {list_time:.6f} seconds")
    print(f"Set membership benchmark:  {set_time:.6f} seconds")

    subsection("16.2 Why data structure choice matters")

    print(
        """
If a program repeatedly asks:

    "Have I seen this identifier before?"

a set is generally preferable to repeatedly searching a list.

If a program repeatedly asks:

    "What record belongs to this ID?"

a dictionary indexed by ID is generally preferable to scanning a list.

If a program needs:
    - ordered positional access
    - frequent append operations
    - duplicates

a list is often appropriate.

If the data should represent a fixed record or coordinate:
a tuple can be appropriate.
"""
    )


# ============================================================================
# 17. ALGORITHMIC EXAMPLES USING DATA STRUCTURES
# ============================================================================

def algorithms_demo() -> None:
    section("17. ALGORITHMIC EXAMPLES")

    subsection("17.1 Two-sum using a dictionary")

    def two_sum(numbers: list[int], target: int) -> tuple[int, int] | None:
        """
        Return indexes of two values whose sum equals target.

        The dictionary stores values already seen and their indexes.
        Average time complexity: O(n).
        Additional space: O(n).
        """
        seen: dict[int, int] = {}

        for index, number in enumerate(numbers):
            complement = target - number

            if complement in seen:
                return seen[complement], index

            seen[number] = index

        return None

    show(
        "two_sum([2, 7, 11, 15], 9)",
        two_sum([2, 7, 11, 15], 9),
    )

    show(
        "two_sum([3, 3], 6)",
        two_sum([3, 3], 6),
    )

    subsection("17.2 Remove duplicates while preserving order")

    def deduplicate(values: Iterable[Any]) -> list[Any]:
        """Deduplicate hashable values while preserving first occurrence."""
        seen: set[Any] = set()
        result: list[Any] = []

        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)

        return result

    show(
        "deduplicate",
        deduplicate(["a", "b", "a", "c", "b"]),
    )

    subsection("17.3 First non-repeating character")

    def first_non_repeating_character(text: str) -> str | None:
        counts = Counter(text)

        for character in text:
            if counts[character] == 1:
                return character

        return None

    show(
        "first non-repeating in 'swiss'",
        first_non_repeating_character("swiss"),
    )

    subsection("17.4 Group anagrams")

    def group_anagrams(words: Iterable[str]) -> dict[tuple[str, ...], list[str]]:
        """
        Group words by their sorted-character signature.

        Example:
            eat, tea, ate -> same signature ('a', 'e', 't')
        """
        groups: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)

        for word in words:
            signature = tuple(sorted(word))
            groups[signature].append(word)

        return dict(groups)

    anagrams = group_anagrams(
        ["eat", "tea", "tan", "ate", "nat", "bat"]
    )

    show("grouped anagrams", anagrams)

    subsection("17.5 Stack-based bracket validation")

    def valid_brackets(expression: str) -> bool:
        opening = {"(", "[", "{"}
        matching = {
            ")": "(",
            "]": "[",
            "}": "{",
        }

        stack: list[str] = []

        for character in expression:
            if character in opening:
                stack.append(character)
            elif character in matching:
                if not stack or stack.pop() != matching[character]:
                    return False

        return not stack

    show("valid brackets", valid_brackets("{[()]}"))
    show("invalid brackets", valid_brackets("{[(])}"))

    subsection("17.6 Breadth-first search with deque")

    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E"],
        "D": [],
        "E": [],
    }

    def breadth_first_search(
        graph: dict[str, list[str]],
        start: str,
    ) -> list[str]:
        queue: deque[str] = deque([start])
        visited: set[str] = {start}
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    show("BFS order", breadth_first_search(graph, "A"))


# ============================================================================
# 18. NESTED DATA PROCESSING CASE STUDY
# ============================================================================

def case_study_demo() -> None:
    section("18. PRACTICAL CASE STUDY: SALES DATA")

    sales = [
        {
            "order_id": 1001,
            "customer": "Asha",
            "city": "Lucknow",
            "items": [
                {"product": "Laptop", "quantity": 1, "price": 70000},
                {"product": "Mouse", "quantity": 2, "price": 1200},
            ],
        },
        {
            "order_id": 1002,
            "customer": "Ravi",
            "city": "Delhi",
            "items": [
                {"product": "Keyboard", "quantity": 1, "price": 2500},
                {"product": "Mouse", "quantity": 1, "price": 1200},
            ],
        },
        {
            "order_id": 1003,
            "customer": "Kabir",
            "city": "Lucknow",
            "items": [
                {"product": "Monitor", "quantity": 2, "price": 15000},
            ],
        },
    ]

    subsection("18.1 Calculate order totals")

    def order_total(order: dict[str, Any]) -> float:
        return sum(
            item["quantity"] * item["price"]
            for item in order["items"]
        )

    totals = {
        order["order_id"]: order_total(order)
        for order in sales
    }

    show("order totals", totals)

    subsection("18.2 Calculate total revenue")

    total_revenue = sum(totals.values())

    show("total revenue", total_revenue)

    subsection("18.3 Find highest-value order")

    highest_order = max(
        sales,
        key=order_total,
    )

    show("highest-value order", highest_order)
    show("highest order value", order_total(highest_order))

    subsection("18.4 Product quantities")

    product_quantities: defaultdict[str, int] = defaultdict(int)

    for order in sales:
        for item in order["items"]:
            product_quantities[item["product"]] += item["quantity"]

    show("product quantities", dict(product_quantities))

    subsection("18.5 Revenue by city")

    revenue_by_city: defaultdict[str, float] = defaultdict(float)

    for order in sales:
        revenue_by_city[order["city"]] += order_total(order)

    show("revenue by city", dict(revenue_by_city))

    subsection("18.6 Customers in a city")

    lucknow_customers = {
        order["customer"]
        for order in sales
        if order["city"] == "Lucknow"
    }

    show("Lucknow customers", lucknow_customers)

    subsection("18.7 Flatten all sold products")

    products_sold = [
        item["product"]
        for order in sales
        for item in order["items"]
    ]

    show("products sold", products_sold)


# ============================================================================
# 19. VALIDATION AND TYPE-AWARE DESIGN
# ============================================================================

def validation_demo() -> None:
    section("19. VALIDATION AND TYPE-AWARE DESIGN")

    subsection("19.1 Validating a list of numbers")

    def average(values: Iterable[float]) -> float:
        values_list = list(values)

        if not values_list:
            raise ValueError("average() requires at least one value")

        if not all(isinstance(value, (int, float)) for value in values_list):
            raise TypeError("all values must be numbers")

        return sum(values_list) / len(values_list)

    show("average", average([10, 20, 30]))

    safe_run("average of empty input", lambda: average([]))
    safe_run("average of invalid input", lambda: average([10, "20"]))

    subsection("19.2 Validating dictionary records")

    def validate_employee(record: dict[str, Any]) -> None:
        required_fields = {"id", "name", "department"}

        missing = required_fields - record.keys()

        if missing:
            raise ValueError(
                f"missing required fields: {sorted(missing)}"
            )

        if not isinstance(record["id"], int):
            raise TypeError("id must be an integer")

        if not isinstance(record["name"], str):
            raise TypeError("name must be a string")

        if not isinstance(record["department"], str):
            raise TypeError("department must be a string")

    valid_employee = {
        "id": 101,
        "name": "Asha",
        "department": "Engineering",
    }

    validate_employee(valid_employee)
    print("valid employee accepted")

    safe_run(
        "invalid employee",
        lambda: validate_employee({"id": "101", "name": "Asha"}),
    )


# ============================================================================
# 20. TESTING DATA STRUCTURE OPERATIONS
# ============================================================================

def testing_demo() -> None:
    section("20. TESTING DATA STRUCTURE OPERATIONS")

    subsection("20.1 Assertions")

    values = [1, 2, 3]

    assert values[0] == 1
    assert len(values) == 3
    assert 2 in values

    print("Basic list assertions passed.")

    subsection("20.2 Testing a custom function")

    def unique_sorted(values: Iterable[int]) -> list[int]:
        return sorted(set(values))

    test_cases = [
        ([3, 1, 2, 1], [1, 2, 3]),
        ([], []),
        ([5, 5, 5], [5]),
        ([-1, 2, -1], [-1, 2]),
    ]

    for input_values, expected in test_cases:
        actual = unique_sorted(input_values)
        assert actual == expected, (
            f"Expected {expected}, got {actual}"
        )

    print("All unique_sorted test cases passed.")

    subsection("20.3 Invariant-based thinking")

    data = [5, 1, 3, 2, 4]

    sorted_data = sorted(data)

    assert len(sorted_data) == len(data)
    assert sorted(sorted_data) == sorted(data)
    assert all(
        sorted_data[index] <= sorted_data[index + 1]
        for index in range(len(sorted_data) - 1)
    )

    print("Sorting invariants passed.")


# ============================================================================
# 21. ADVANCED COMBINATIONS
# ============================================================================

def advanced_combinations_demo() -> None:
    section("21. ADVANCED DATA-STRUCTURE COMBINATIONS")

    subsection("21.1 List -> set -> list pipeline")

    values = [5, 3, 5, 1, 3, 2]

    unique_sorted = sorted(set(values))

    show("unique sorted", unique_sorted)

    subsection("21.2 List of tuples -> dictionary")

    pairs = [
        ("Python", 1),
        ("SQL", 2),
        ("Git", 3),
    ]

    lookup = dict(pairs)

    show("lookup", lookup)

    subsection("21.3 Dictionary -> list of sorted tuples")

    scores = {
        "Asha": 91,
        "Ravi": 72,
        "Kabir": 84,
    }

    ranking = sorted(
        scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )

    show("ranking", ranking)

    subsection("21.4 Dictionary of sets")

    permissions = {
        "admin": {"read", "write", "delete"},
        "editor": {"read", "write"},
        "viewer": {"read"},
    }

    show("admin permissions", permissions["admin"])

    subsection("21.5 Set of tuples")

    edges = {
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
    }

    show("graph edges", edges)

    subsection("21.6 Tuple containing a frozenset")

    immutable_record = (
        "engineering",
        frozenset({"Python", "SQL"}),
    )

    show("immutable record", immutable_record)

    subsection("21.7 Dictionary whose values are lists")

    students_by_class: dict[str, list[str]] = {
        "A": ["Asha", "Ravi"],
        "B": ["Kabir", "Neha"],
    }

    students_by_class["A"].append("Vikram")

    show("students by class", students_by_class)


# ============================================================================
# 22. DATA STRUCTURE SELECTION
# ============================================================================

def selection_demo() -> None:
    section("22. CHOOSING THE RIGHT DATA STRUCTURE")

    print(
        """
Use a LIST when:
    - order matters
    - duplicates matter
    - positional indexing is useful
    - frequent appends are required

Use a TUPLE when:
    - the sequence represents a fixed grouping
    - immutability communicates intent
    - the object may need to be hashable
    - a coordinate or fixed record is appropriate

Use a SET when:
    - uniqueness matters
    - membership testing is frequent
    - mathematical set operations are useful

Use a FROZENSET when:
    - set semantics are needed
    - the set should be immutable and hashable

Use a DICTIONARY when:
    - values need to be retrieved by meaningful keys
    - records have named attributes
    - indexing data by an identifier is useful

Use DEQUE when:
    - efficient insertion/removal at both ends is required
    - queue or double-ended queue semantics are needed

Use COUNTER when:
    - frequencies are central to the problem

Use DEFAULTDICT when:
    - grouping or accumulating into automatically created containers
      is the dominant pattern

Use a DATACLASS when:
    - a domain object has stable named fields
    - explicit structure and type information improve maintainability
"""
    )

    decisions = [
        ("Unique user IDs", "set"),
        ("User ID -> user record", "dict"),
        ("Ordered transaction history", "list"),
        ("GPS coordinate pair", "tuple"),
        ("Task queue", "deque"),
        ("Word frequencies", "Counter"),
        ("Department -> employees", "defaultdict(list)"),
        ("Immutable collection of permissions", "frozenset"),
    ]

    for problem, structure in decisions:
        print(f"{problem:35} -> {structure}")


# ============================================================================
# 23. COMMON MISTAKES
# ============================================================================

def common_mistakes_demo() -> None:
    section("23. COMMON MISTAKES")

    mistakes = [
        (
            "Using {} when an empty set is intended",
            "{} creates an empty dictionary; use set() for an empty set.",
        ),
        (
            "Confusing == with is",
            "Use == for equality and is for identity.",
        ),
        (
            "Using list membership for huge repeated lookups",
            "Consider a set when only membership matters.",
        ),
        (
            "Using pop(0) repeatedly for a queue",
            "Use collections.deque and popleft().",
        ),
        (
            "Modifying a list while iterating over it",
            "Build a filtered list or iterate over a snapshot.",
        ),
        (
            "Expecting tuple immutability to freeze nested objects",
            "Tuple immutability does not recursively freeze contained objects.",
        ),
        (
            "Using a list as a dictionary key",
            "Use an immutable hashable representation such as a tuple.",
        ),
        (
            "Using complicated comprehensions",
            "Prefer a normal loop when comprehension logic becomes difficult to read.",
        ),
        (
            "Assuming dictionary values are unique",
            "Dictionary keys are unique; values may repeat freely.",
        ),
        (
            "Assuming shallow copy duplicates nested objects",
            "Use deepcopy when independent nested object graphs are required.",
        ),
        (
            "Assuming sets preserve a meaningful sequence",
            "Use a list when stable positional order is part of the data model.",
        ),
        (
            "Forgetting that dict membership checks keys",
            "Use values() when testing membership among values.",
        ),
    ]

    for mistake, correction in mistakes:
        print(f"\nMistake: {mistake}\nCorrection: {correction}")


# ============================================================================
# 24. SECURITY AND ROBUSTNESS CONSIDERATIONS
# ============================================================================

def security_demo() -> None:
    section("24. SECURITY AND ROBUSTNESS")

    print(
        """
Data structures are not inherently a security boundary. They are
containers used inside larger systems.

Important considerations:

1. Validate untrusted input.
   Never assume that external JSON, form data, CSV-derived dictionaries,
   API payloads, or user-provided collections have the expected shape.

2. Control resource consumption.
   Extremely large lists, deeply nested dictionaries, or huge sets can
   consume substantial memory.

3. Avoid unsafe assumptions about keys.
   Validate required fields before indexing nested structures.

4. Protect sensitive values.
   A dictionary is not a secure secret store merely because it is in memory.

5. Be careful with recursive structures.
   Unexpectedly deep nesting can cause recursion errors or excessive
   processing.

6. Do not use `eval()` to process textual representations of collections.
   Use safe parsers such as JSON parsing for JSON data.

7. Consider algorithmic complexity.
   Repeated linear scans over attacker-controlled input can create
   performance problems.

8. Avoid exposing internal mutable objects unintentionally.
   Return copies or immutable representations when callers should not
   mutate internal state.
"""
    )

    subsection("24.1 Defensive copying")

    def get_permissions() -> list[str]:
        internal_permissions = ["read", "write"]

        # Returning a new list prevents the caller from modifying this
        # function's internal list.
        return internal_permissions.copy()

    permissions = get_permissions()
    permissions.append("delete")

    show("caller-modified permissions", permissions)

    subsection("24.2 Safe parsing concept")

    import json

    text = '{"name": "Asha", "skills": ["Python", "SQL"]}'

    parsed = json.loads(text)

    show("JSON parsed into Python structures", parsed)


# ============================================================================
# 25. MEMORY AND REPRESENTATION
# ============================================================================

def memory_demo() -> None:
    section("25. MEMORY AND REPRESENTATION")

    subsection("25.1 Object identity")

    first = [1, 2, 3]
    second = first
    third = first.copy()

    print(f"id(first):  {id(first)}")
    print(f"id(second): {id(second)}")
    print(f"id(third):  {id(third)}")

    show("first is second", first is second)
    show("first is third", first is third)

    subsection("25.2 Approximate object size")

    print(
        f"sys.getsizeof([1, 2, 3]): {sys.getsizeof([1, 2, 3])} bytes"
    )
    print(
        f"sys.getsizeof((1, 2, 3)): {sys.getsizeof((1, 2, 3))} bytes"
    )
    print(
        f"sys.getsizeof({1, 2, 3}): {sys.getsizeof({1, 2, 3})} bytes"
    )
    print(
        f"sys.getsizeof({'a': 1}): {sys.getsizeof({'a': 1})} bytes"
    )

    print(
        """
sys.getsizeof() measures the immediate object's memory footprint.
It does not recursively include the complete memory used by every
object referenced from a nested collection.

For deep memory analysis, specialized measurement techniques are required.
"""
    )


# ============================================================================
# 26. ITERATORS AND GENERATORS
# ============================================================================

def iterators_demo() -> None:
    section("26. ITERATORS AND LAZY COLLECTION PROCESSING")

    subsection("26.1 Iterating over a list")

    values = [10, 20, 30]

    iterator = iter(values)

    show("next(iterator)", next(iterator))
    show("next(iterator)", next(iterator))
    show("next(iterator)", next(iterator))

    safe_run("next after exhaustion", lambda: next(iterator))

    subsection("26.2 Generator function")

    def generate_squares(limit: int):
        for number in range(limit):
            # yield produces one value at a time.
            yield number * number

    generator = generate_squares(5)

    show("generator type", type(generator).__name__)
    show("generator values", list(generator))

    subsection("26.3 Generator versus list")

    list_values = [number * number for number in range(10)]
    generator_values = (
        number * number
        for number in range(10)
    )

    show("list", list_values)
    show("generator consumed", list(generator_values))

    print(
        """
Lists materialize their values immediately.

Generators are lazy and can be more memory-efficient when processing
large streams of data, because values do not all need to exist at once.
"""
    )


# ============================================================================
# 27. FUNCTIONAL OPERATIONS
# ============================================================================

def functional_operations_demo() -> None:
    section("27. MAP, FILTER, REDUCE, AND DATA STRUCTURES")

    values = [1, 2, 3, 4, 5]

    mapped = list(map(lambda value: value * 2, values))
    filtered = list(filter(lambda value: value % 2 == 0, values))
    reduced = reduce(lambda left, right: left + right, values)

    show("map", mapped)
    show("filter", filtered)
    show("reduce", reduced)

    print(
        """
In many Python programs, comprehensions are more readable than map()
and filter() when the operation is simple.

For example:

    [x * 2 for x in values]
    [x for x in values if x % 2 == 0]

reduce() is useful for certain associative aggregation operations, but
built-in functions such as sum(), min(), max(), any(), and all() are
usually clearer when they express the intended operation directly.
"""
    )

    show("sum", sum(values))
    show("any", any(value > 4 for value in values))
    show("all", all(value > 0 for value in values))


# ============================================================================
# 28. GROUPBY AND SORTING REQUIREMENT
# ============================================================================

def groupby_demo() -> None:
    section("28. GROUPBY AND SORTED DATA")

    records = [
        ("Engineering", "Asha"),
        ("Engineering", "Kabir"),
        ("Analytics", "Ravi"),
        ("Analytics", "Neha"),
    ]

    # itertools.groupby groups consecutive items, so the input must first
    # be sorted by the grouping key.
    records_sorted = sorted(records, key=lambda record: record[0])

    grouped = {}

    for department, group in groupby(
        records_sorted,
        key=lambda record: record[0],
    ):
        grouped[department] = [
            name
            for _, name in group
        ]

    show("groupby result", grouped)

    print(
        """
Important distinction:

defaultdict(list) groups arbitrary input in one pass.

itertools.groupby() groups consecutive equal keys and therefore normally
requires sorting first when equivalent keys may be separated in the input.
"""
    )


# ============================================================================
# 29. DATA NORMALIZATION
# ============================================================================

def normalization_demo() -> None:
    section("29. DATA NORMALIZATION WITH COLLECTIONS")

    raw_names = [
        " Alice ",
        "BOB",
        "alice",
        " Bob ",
        "",
        "  ",
    ]

    normalized_names = [
        name.strip().casefold()
        for name in raw_names
        if name.strip()
    ]

    show("normalized names", normalized_names)

    unique_names = sorted(set(normalized_names))

    show("unique normalized names", unique_names)

    name_frequency = Counter(normalized_names)

    show("name frequency", name_frequency)

    print(
        """
A common processing pipeline is:

    raw sequence
        -> validate
        -> normalize
        -> filter invalid values
        -> deduplicate or aggregate
        -> index/group
        -> sort or present

Each stage can use a different data structure depending on its purpose.
"""
    )


# ============================================================================
# 30. MINI PROJECT: INVENTORY INDEX
# ============================================================================

def inventory_project_demo() -> None:
    section("30. MINI PROJECT: INVENTORY MANAGEMENT")

    inventory = [
        {
            "sku": "KB001",
            "name": "Keyboard",
            "category": "Electronics",
            "price": 2500,
            "stock": 15,
            "tags": {"office", "input"},
        },
        {
            "sku": "MS001",
            "name": "Mouse",
            "category": "Electronics",
            "price": 1200,
            "stock": 30,
            "tags": {"office", "input"},
        },
        {
            "sku": "CH001",
            "name": "Chair",
            "category": "Furniture",
            "price": 8000,
            "stock": 5,
            "tags": {"office", "ergonomic"},
        },
    ]

    subsection("30.1 Build SKU index")

    by_sku = {
        item["sku"]: item
        for item in inventory
    }

    show("KB001", by_sku["KB001"])

    subsection("30.2 Low-stock items")

    low_stock = [
        item
        for item in inventory
        if item["stock"] < 10
    ]

    show("low-stock items", low_stock)

    subsection("30.3 Category index")

    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in inventory:
        by_category[item["category"]].append(item)

    show("category index", dict(by_category))

    subsection("30.4 All tags")

    all_tags = {
        tag
        for item in inventory
        for tag in item["tags"]
    }

    show("all tags", all_tags)

    subsection("30.5 Search by tag")

    requested_tag = "input"

    matching_items = [
        item["name"]
        for item in inventory
        if requested_tag in item["tags"]
    ]

    show(f"items tagged {requested_tag!r}", matching_items)

    subsection("30.6 Inventory value")

    total_inventory_value = sum(
        item["price"] * item["stock"]
        for item in inventory
    )

    show("inventory value", total_inventory_value)

    subsection("30.7 Sort by stock value")

    ranked = sorted(
        inventory,
        key=lambda item: item["price"] * item["stock"],
        reverse=True,
    )

    show("inventory ranked by stock value", ranked)


# ============================================================================
# 31. ADVANCED EDGE CASES
# ============================================================================

def advanced_edge_cases_demo() -> None:
    section("31. ADVANCED EDGE CASES")

    subsection("31.1 Modifying a list through a reference")

    inner = [1, 2]
    outer = [inner, inner]

    outer[0].append(3)

    show("outer with shared references", outer)
    show("outer[0] is outer[1]", outer[0] is outer[1])

    subsection("31.2 Self-referential list")

    self_reference: list[Any] = []
    self_reference.append(self_reference)

    print("Self-referential list created.")
    print(f"self_reference is self_reference[0]: {self_reference is self_reference[0]}")
    print(
        "Python's repr handles the recursive reference with an abbreviated "
        "representation."
    )
    show("self-reference", self_reference)

    subsection("31.3 Dictionary insertion order")

    data = {}

    data["first"] = 1
    data["second"] = 2
    data["third"] = 3

    show("dictionary iteration order", list(data))

    print(
        """
Modern Python specifies dictionary insertion order as part of the language
semantics. This does not mean dictionaries should be treated as sorted
structures. If sorted order is required, explicitly sort the items.
"""
    )

    subsection("31.4 Set order")

    values = {3, 1, 2}

    show("set", values)

    print(
        """
Do not rely on a set's displayed iteration order as a business-level
ordering rule. Use a list or an explicitly sorted sequence when order
matters.
"""
    )


# ============================================================================
# 32. PRACTICAL DESIGN PRINCIPLES
# ============================================================================

def design_principles_demo() -> None:
    section("32. DESIGN PRINCIPLES")

    principles = [
        "Choose a structure based on required operations, not habit.",
        "Use meaningful keys when data has natural identifiers.",
        "Keep data immutable when immutability communicates a useful invariant.",
        "Avoid unnecessarily deep nesting.",
        "Prefer explicit domain models when dictionaries become difficult to reason about.",
        "Use comprehensions for clear transformations, not for compressed complexity.",
        "Use sets for uniqueness and membership, not as ordered sequences.",
        "Use dictionaries for indexing rather than repeatedly scanning large lists.",
        "Use deque for efficient queue operations.",
        "Make ownership and mutability clear when returning collections from functions.",
        "Validate external data before performing deep indexing.",
        "Consider time and space complexity for large workloads.",
        "Test edge cases such as empty collections, duplicates, missing keys, and nested mutation.",
    ]

    for number, principle in enumerate(principles, start=1):
        print(f"{number:2}. {principle}")


# ============================================================================
# 33. INTEGRATED FINAL EXERCISE
# ============================================================================

def integrated_exercise_demo() -> None:
    section("33. INTEGRATED EXAMPLE: STUDENT ANALYTICS")

    students = [
        {
            "id": 1,
            "name": "Asha",
            "city": "Lucknow",
            "courses": {
                "Python": 92,
                "SQL": 88,
                "Statistics": 81,
            },
        },
        {
            "id": 2,
            "name": "Ravi",
            "city": "Delhi",
            "courses": {
                "Python": 78,
                "SQL": 91,
                "Statistics": 85,
            },
        },
        {
            "id": 3,
            "name": "Kabir",
            "city": "Lucknow",
            "courses": {
                "Python": 95,
                "SQL": 89,
                "Statistics": 93,
            },
        },
    ]

    subsection("33.1 Student index")

    students_by_id = {
        student["id"]: student
        for student in students
    }

    show("student 2", students_by_id[2])

    subsection("33.2 Average score per student")

    averages = {
        student["name"]: sum(student["courses"].values())
        / len(student["courses"])
        for student in students
    }

    show("averages", averages)

    subsection("33.3 Rank students")

    ranking = sorted(
        averages.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    show("ranking", ranking)

    subsection("33.4 Highest score by course")

    course_scores: defaultdict[str, list[int]] = defaultdict(list)

    for student in students:
        for course, score in student["courses"].items():
            course_scores[course].append(score)

    highest_by_course = {
        course: max(scores)
        for course, scores in course_scores.items()
    }

    show("highest by course", highest_by_course)

    subsection("33.5 Course enrollment")

    course_enrollment: defaultdict[str, int] = defaultdict(int)

    for student in students:
        for course in student["courses"]:
            course_enrollment[course] += 1

    show("course enrollment", dict(course_enrollment))

    subsection("33.6 Cities represented")

    cities = {
        student["city"]
        for student in students
    }

    show("cities", cities)

    subsection("33.7 Students above threshold")

    threshold = 90

    high_performers = [
        student["name"]
        for student in students
        if averages[student["name"]] >= threshold
    ]

    show("high performers", high_performers)

    subsection("33.8 All distinct courses")

    all_courses = {
        course
        for student in students
        for course in student["courses"]
    }

    show("all courses", all_courses)

    print(
        """
This single example combines:

    list
    dictionary
    nested dictionary
    set
    defaultdict
    comprehensions
    iteration
    aggregation
    sorting
    indexing

The data structure is not merely a storage detail. It determines how
natural and efficient later operations become.
"""
    )


# ============================================================================
# 34. SELF-CHECK QUESTIONS
# ============================================================================

def self_check_demo() -> None:
    section("34. SELF-CHECK QUESTIONS")

    questions = [
        "Why does list membership generally take O(n)?",
        "Why is set membership generally O(1) on average?",
        "Why does {} create a dictionary instead of a set?",
        "What makes a tuple suitable as a dictionary key in some cases?",
        "Why can a tuple containing a list not be hashed?",
        "What is the difference between append() and extend()?",
        "What is the difference between remove(), pop(), and discard()?",
        "Why is pop(0) inefficient for a large queue?",
        "Why does dict.get() behave differently from dict[key] for missing keys?",
        "What does dictionary membership test?",
        "What is the difference between shallow and deep copying?",
        "Why can [[0] * 3] * 3 create surprising matrix behavior?",
        "When is a comprehension less appropriate than a normal loop?",
        "Why should groupby() normally receive sorted data?",
        "Why might defaultdict(list) be useful for grouping?",
        "Why are mutable dictionary keys prohibited?",
        "What does a frozenset provide that a set does not?",
        "Why should order requirements not be encoded using a set?",
        "When is a dataclass preferable to a deeply nested dictionary?",
        "How can a dictionary turn a repeated lookup problem from O(n) per lookup "
        "into approximately O(1) average lookup?",
    ]

    for number, question in enumerate(questions, start=1):
        print(f"{number:2}. {question}")


# ============================================================================
# 35. MAIN
# ============================================================================

def main() -> None:
    """Run the complete educational demonstration."""
    foundations_demo()
    lists_demo()
    tuples_demo()
    sets_demo()
    dictionaries_demo()
    comprehensions_demo()
    nested_structures_demo()
    mutability_demo()
    hashability_demo()
    iteration_demo()
    mutation_during_iteration_demo()
    advanced_collections_demo()
    common_patterns_demo()
    merging_and_copying_demo()
    validation_and_edge_cases_demo()
    performance_demo()
    algorithms_demo()
    case_study_demo()
    validation_demo()
    testing_demo()
    advanced_combinations_demo()
    selection_demo()
    common_mistakes_demo()
    security_demo()
    memory_demo()
    iterators_demo()
    functional_operations_demo()
    groupby_demo()
    normalization_demo()
    inventory_project_demo()
    advanced_edge_cases_demo()
    design_principles_demo()
    integrated_exercise_demo()
    self_check_demo()

    section("END OF DATA STRUCTURES STUDY SCRIPT")

    print(
        """
The script demonstrated Python's principal built-in data structures,
their operations, relationships, trade-offs, edge cases, performance
characteristics, and practical combinations.

A useful way to retain the material is to ask three questions whenever
you design a collection:

    1. What operations will be performed most frequently?
    2. What invariants must the data preserve?
    3. What are the time, memory, mutability, and correctness implications
       of the chosen representation?
"""
    )


if __name__ == "__main__":
    main()
