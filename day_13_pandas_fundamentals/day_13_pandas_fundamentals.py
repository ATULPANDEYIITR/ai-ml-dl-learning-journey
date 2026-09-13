"""
Pandas Fundamentals: Series, DataFrames, Indexing, Filtering, and Sorting

A self-contained study script covering Pandas from absolute beginner to
advanced beginner/intermediate level.

Topics covered:
    1. Pandas and tabular data
    2. Series fundamentals
    3. DataFrame fundamentals
    4. Creating Series and DataFrames
    5. Indexes and labels
    6. Selecting columns and rows
    7. loc and iloc
    8. Filtering with Boolean conditions
    9. Combining conditions
    10. Handling missing values
    11. Adding, modifying, and deleting columns
    12. Sorting rows and indexes
    13. Index management
    14. Data types
    15. Basic descriptive statistics
    16. String and datetime operations
    17. Duplicate handling
    18. Conditional transformations
    19. Copying and avoiding chained assignment
    20. MultiIndex fundamentals
    21. Performance considerations
    22. Common mistakes
    23. Validation and testing examples
    24. A realistic end-to-end data-analysis example

Requirements:
    pip install pandas

The script uses only the Python standard library and Pandas.
"""

from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly separated learning section."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def subsection(title: str) -> None:
    """Print a smaller subsection heading."""
    print(f"\n--- {title} ---")


def show(name: str, value) -> None:
    """Print a named object in a readable format."""
    print(f"\n{name}:")
    print(value)


def demonstrate_exception(description: str, function) -> None:
    """Execute a demonstration and display the expected exception."""
    try:
        function()
    except Exception as exc:
        print(f"{description}")
        print(f"  Exception: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 1. Introduction to Pandas
# ---------------------------------------------------------------------------

section("1. Pandas fundamentals")

print(
    """
Pandas is a Python library for working with labeled and tabular data.

Its two central data structures are:

    Series
        A one-dimensional labeled array.

    DataFrame
        A two-dimensional labeled table containing rows and columns.

A DataFrame can be viewed conceptually as a collection of aligned Series.
Each column is a Series, while the DataFrame supplies the row index and
organizes those columns together.
"""
)

print("Pandas version:", pd.__version__)


# ---------------------------------------------------------------------------
# 2. Series
# ---------------------------------------------------------------------------

section("2. Series fundamentals")

subsection("Creating a Series from a list")

scores = pd.Series([78, 85, 91, 67, 88])
show("scores", scores)

print(
    """
The values are stored as data and each value receives an index. If no index
is supplied, Pandas creates a default integer index beginning at zero.
"""
)

subsection("Creating a Series with custom labels")

student_scores = pd.Series(
    [78, 85, 91, 67],
    index=["Asha", "Bharat", "Charu", "Deepak"],
    name="score",
)
show("student_scores", student_scores)

print("Values:", student_scores.values)
print("Index:", student_scores.index)
print("Name:", student_scores.name)
print("Data type:", student_scores.dtype)

subsection("Series from a dictionary")

population = pd.Series(
    {
        "Delhi": 33.8,
        "Mumbai": 21.7,
        "Bengaluru": 14.4,
        "Lucknow": 4.0,
    },
    name="population_millions",
)
show("population", population)

print(
    """
Dictionary keys become the index labels and dictionary values become the
Series values.
"""
)

subsection("Scalar value and repeated index")

constant_series = pd.Series(10, index=["A", "B", "C"])
show("constant_series", constant_series)


# ---------------------------------------------------------------------------
# 3. Series indexing
# ---------------------------------------------------------------------------

section("3. Series indexing and selection")

sales = pd.Series(
    [1200, 1800, 950, 2100, 1750],
    index=["Mon", "Tue", "Wed", "Thu", "Fri"],
    name="sales",
)

show("sales", sales)

print("Label-based selection:", sales["Tue"])
print("Position-based selection:", sales.iloc[1])
print("First three values:")
print(sales.iloc[:3])

print("Monday through Thursday using labels:")
print(sales.loc["Mon":"Thu"])

print("Selected days:")
print(sales.loc[["Mon", "Wed", "Fri"]])

print(
    """
Important distinction:

    series.loc[label]
        Selects using labels.

    series.iloc[position]
        Selects using integer positions.

With .loc, a label-based slice normally includes the endpoint.
With .iloc, the stop position follows normal Python slicing rules and is
excluded.
"""
)

subsection("Boolean filtering on a Series")

high_sales = sales[sales > 1500]
show("sales greater than 1500", high_sales)

sales_between = sales[(sales >= 1000) & (sales <= 2000)]
show("sales between 1000 and 2000", sales_between)


# ---------------------------------------------------------------------------
# 4. Series operations
# ---------------------------------------------------------------------------

section("4. Series operations")

numbers = pd.Series([10, 20, 30, 40, 50], name="numbers")

show("numbers + 5", numbers + 5)
show("numbers * 2", numbers * 2)
show("numbers squared", numbers ** 2)

print("Mean:", numbers.mean())
print("Median:", numbers.median())
print("Minimum:", numbers.min())
print("Maximum:", numbers.max())
print("Sum:", numbers.sum())
print("Standard deviation:", numbers.std())

print("Unique values:", pd.Series([1, 2, 2, 3, 3, 3]).unique())
print("Value counts:")
print(pd.Series(["A", "B", "A", "C", "B", "A"]).value_counts())


# ---------------------------------------------------------------------------
# 5. Alignment
# ---------------------------------------------------------------------------

section("5. Pandas label alignment")

left = pd.Series([100, 200, 300], index=["A", "B", "C"])
right = pd.Series([10, 20, 30], index=["B", "C", "D"])

show("left", left)
show("right", right)
show("left + right", left + right)

print(
    """
Pandas aligns Series by index labels rather than simply adding values by
physical position.

A and D have no matching label, so their results are missing values.
"""
)


# ---------------------------------------------------------------------------
# 6. DataFrame creation
# ---------------------------------------------------------------------------

section("6. DataFrame fundamentals")

subsection("Creating a DataFrame from a dictionary")

students = pd.DataFrame(
    {
        "name": ["Asha", "Bharat", "Charu", "Deepak", "Esha"],
        "age": [21, 22, 20, 23, 21],
        "score": [88, 76, 95, 67, 84],
        "city": ["Delhi", "Mumbai", "Delhi", "Lucknow", "Mumbai"],
    }
)

show("students", students)

print("Shape:", students.shape)
print("Number of rows:", students.shape[0])
print("Number of columns:", students.shape[1])
print("Column names:", students.columns.tolist())
print("Index:", students.index.tolist())
print("Data types:")
print(students.dtypes)

subsection("DataFrame inspection")

print("First rows:")
print(students.head())

print("\nLast rows:")
print(students.tail())

print("\nRandom sample:")
print(students.sample(2, random_state=42))

print("\nInformation:")
students.info()

print("\nDescriptive statistics:")
print(students.describe())

print("\nDescriptive statistics for all columns:")
print(students.describe(include="all"))


# ---------------------------------------------------------------------------
# 7. Selecting columns
# ---------------------------------------------------------------------------

section("7. Selecting DataFrame columns")

print("One column:")
print(students["name"])

print("\nMultiple columns:")
print(students[["name", "score"]])

print(
    """
A single column selection normally returns a Series.
A list of column names returns a DataFrame.
"""
)

subsection("Attribute-style column access")

print("students.score:")
print(students.score)

print(
    """
Attribute access can be convenient, but bracket notation is safer and more
general because it works with column names containing spaces, punctuation,
or names that conflict with DataFrame attributes.
"""
)


# ---------------------------------------------------------------------------
# 8. Adding and modifying columns
# ---------------------------------------------------------------------------

section("8. Adding and modifying columns")

students["passed"] = students["score"] >= 50
show("students with passed", students)

students["score_percentage"] = students["score"] / 100
show("students with score_percentage", students)

students["age_next_year"] = students["age"] + 1
show("students with age_next_year", students)

students["performance"] = "Average"
students.loc[students["score"] >= 85, "performance"] = "Excellent"
students.loc[
    (students["score"] >= 70) & (students["score"] < 85),
    "performance",
] = "Good"

show("students with performance", students)

subsection("Using np-like vectorized logic through Pandas")

students["result"] = students["score"].ge(50).map(
    {True: "Pass", False: "Fail"}
)

show("students with result", students)


# ---------------------------------------------------------------------------
# 9. Removing columns
# ---------------------------------------------------------------------------

section("9. Removing columns")

temporary = students.copy()

temporary = temporary.drop(
    columns=["score_percentage", "age_next_year", "result"]
)

show("after dropping columns", temporary)

print(
    """
drop() returns a new object by default.

Using inplace=True modifies the existing DataFrame, but explicit assignment
is often easier to reason about and chain safely.
"""
)


# ---------------------------------------------------------------------------
# 10. Row selection with loc and iloc
# ---------------------------------------------------------------------------

section("10. DataFrame row selection")

subsection("iloc: integer-position based selection")

print("First row:")
print(students.iloc[0])

print("\nFirst three rows:")
print(students.iloc[:3])

print("\nRows 1 and 3:")
print(students.iloc[[1, 3]])

print("\nRows 1-3 and columns 0-2:")
print(students.iloc[1:4, 0:3])

subsection("loc: label-based selection")

print("Rows 0 through 2:")
print(students.loc[0:2])

print("\nRows 0 and 3:")
print(students.loc[[0, 3]])

print("\nRows 0 through 2, selected columns:")
print(students.loc[0:2, ["name", "score"]])


# ---------------------------------------------------------------------------
# 11. Custom DataFrame indexes
# ---------------------------------------------------------------------------

section("11. Custom DataFrame indexes")

indexed_students = students.set_index("name")

show("indexed_students", indexed_students)

print("Select Asha by label:")
print(indexed_students.loc["Asha"])

print("\nSelect Asha and Charu:")
print(indexed_students.loc[["Asha", "Charu"]])

print("\nSelect score for Asha:")
print(indexed_students.loc["Asha", "score"])

print(
    """
An index is not necessarily a database primary key, but a meaningful unique
index can make label-based selection convenient.

An index can contain duplicates, although duplicate labels can make selection
results less intuitive.
"""
)


# ---------------------------------------------------------------------------
# 12. Resetting an index
# ---------------------------------------------------------------------------

section("12. reset_index()")

reset_students = indexed_students.reset_index()
show("reset_students", reset_students)

reset_students_without_old_index = indexed_students.reset_index(drop=True)
show("reset_index(drop=True)", reset_students_without_old_index)


# ---------------------------------------------------------------------------
# 13. Filtering
# ---------------------------------------------------------------------------

section("13. Boolean filtering")

subsection("Single condition")

score_above_80 = students[students["score"] > 80]
show("score > 80", score_above_80)

subsection("Equality condition")

delhi_students = students[students["city"] == "Delhi"]
show("city == Delhi", delhi_students)

subsection("Multiple conditions")

delhi_high_scorers = students[
    (students["city"] == "Delhi") & (students["score"] >= 90)
]
show("Delhi students with score >= 90", delhi_high_scorers)

subsection("OR conditions")

delhi_or_mumbai = students[
    (students["city"] == "Delhi") | (students["city"] == "Mumbai")
]
show("Delhi or Mumbai", delhi_or_mumbai)

subsection("NOT condition")

not_delhi = students[students["city"] != "Delhi"]
show("not Delhi", not_delhi)

print(
    """
Pandas requires parentheses around individual Boolean comparisons when
combining them with &, |, and ~.

Use:
    (condition_a) & (condition_b)

Not:
    condition_a and condition_b

Python's 'and' and 'or' operate on single Boolean objects, while Pandas
Boolean expressions normally operate element by element.
"""
)

subsection("isin()")

selected_cities = students[
    students["city"].isin(["Delhi", "Lucknow"])
]
show("Delhi or Lucknow using isin()", selected_cities)

subsection("between()")

score_range = students[students["score"].between(70, 90)]
show("score from 70 through 90", score_range)

subsection("String filtering")

names_starting_with_a = students[
    students["name"].str.startswith("A")
]
show("names starting with A", names_starting_with_a)

cities_containing_del = students[
    students["city"].str.contains("del", case=False, na=False)
]
show("cities containing 'del'", cities_containing_del)


# ---------------------------------------------------------------------------
# 14. query()
# ---------------------------------------------------------------------------

section("14. Filtering with query()")

query_result = students.query("score >= 80 and age <= 22")
show("score >= 80 and age <= 22", query_result)

target_score = 80
query_with_variable = students.query("score >= @target_score")
show("query using Python variable", query_with_variable)

print(
    """
query() can make complex filtering expressions more readable. It is useful
when column names are simple identifiers. Bracket-based filtering remains
more explicit and is often easier for beginners to debug.
"""
)


# ---------------------------------------------------------------------------
# 15. Sorting
# ---------------------------------------------------------------------------

section("15. Sorting DataFrames")

subsection("Sort by one column")

sorted_by_score = students.sort_values("score")
show("ascending score", sorted_by_score)

subsection("Descending sort")

sorted_by_score_desc = students.sort_values("score", ascending=False)
show("descending score", sorted_by_score_desc)

subsection("Sort by multiple columns")

sorted_multiple = students.sort_values(
    by=["city", "score"],
    ascending=[True, False],
)
show("city ascending, score descending", sorted_multiple)

subsection("Missing values and sorting")

values_with_missing = pd.Series(
    [30, None, 10, 20],
    index=["A", "B", "C", "D"],
)

print(values_with_missing.sort_values())
print("\nMissing values first:")
print(values_with_missing.sort_values(na_position="first"))

subsection("Sorting by index")

indexed_example = pd.DataFrame(
    {"score": [80, 95, 70]},
    index=["Charlie", "Alice", "Bob"],
)

show("original", indexed_example)
show("sorted by index", indexed_example.sort_index())


# ---------------------------------------------------------------------------
# 16. Ranking
# ---------------------------------------------------------------------------

section("16. Ranking")

ranking_data = students[["name", "score"]].copy()
ranking_data["rank"] = ranking_data["score"].rank(
    ascending=False,
    method="dense",
)

show("ranking", ranking_data.sort_values("rank"))


# ---------------------------------------------------------------------------
# 17. Missing values
# ---------------------------------------------------------------------------

section("17. Missing data")

missing_data = pd.DataFrame(
    {
        "name": ["Asha", "Bharat", "Charu", "Deepak"],
        "age": [21, None, 20, 23],
        "score": [88, 76, None, 67],
        "city": ["Delhi", "Mumbai", None, "Lucknow"],
    }
)

show("missing_data", missing_data)

print("Missing-value mask:")
print(missing_data.isna())

print("\nNumber of missing values per column:")
print(missing_data.isna().sum())

print("\nRows containing at least one missing value:")
print(missing_data[missing_data.isna().any(axis=1)])

print("\nRows with no missing values:")
print(missing_data.dropna())

subsection("Filling missing values")

filled = missing_data.copy()
filled["age"] = filled["age"].fillna(filled["age"].median())
filled["score"] = filled["score"].fillna(filled["score"].mean())
filled["city"] = filled["city"].fillna("Unknown")

show("filled data", filled)

print(
    """
Missing data requires a domain decision.

Common strategies include:
    - drop rows
    - drop columns
    - fill with a constant
    - fill with mean or median
    - forward fill
    - backward fill
    - model-based imputation

The statistically appropriate method depends on why values are missing and
what the data represents.
"""
)


# ---------------------------------------------------------------------------
# 18. Duplicate rows
# ---------------------------------------------------------------------------

section("18. Duplicate detection")

duplicates = pd.DataFrame(
    {
        "name": ["Asha", "Bharat", "Asha", "Charu"],
        "score": [88, 76, 88, 95],
    }
)

show("duplicates", duplicates)

print("Duplicate mask:")
print(duplicates.duplicated())

print("\nDuplicated rows:")
print(duplicates[duplicates.duplicated()])

deduplicated = duplicates.drop_duplicates()
show("after drop_duplicates()", deduplicated)


# ---------------------------------------------------------------------------
# 19. Data types
# ---------------------------------------------------------------------------

section("19. Data types")

typed_data = pd.DataFrame(
    {
        "integer_column": [1, 2, 3],
        "float_column": [1.5, 2.5, 3.5],
        "text_column": ["A", "B", "C"],
        "boolean_column": [True, False, True],
    }
)

show("typed_data", typed_data)
print("Data types:")
print(typed_data.dtypes)

subsection("Explicit type conversion")

converted = typed_data.copy()
converted["integer_column"] = converted["integer_column"].astype("float64")
show("integer_column converted to float", converted)

subsection("Numeric conversion with invalid values")

raw_numbers = pd.Series(["100", "250", "invalid", "400"])
safe_numbers = pd.to_numeric(raw_numbers, errors="coerce")

show("raw numbers", raw_numbers)
show("safe numeric conversion", safe_numbers)

print(
    """
errors='coerce' converts invalid numeric values into NaN instead of raising
an exception. This is useful for messy external data, but invalid records
should normally be investigated rather than silently ignored.
"""
)


# ---------------------------------------------------------------------------
# 20. DataFrame arithmetic and vectorization
# ---------------------------------------------------------------------------

section("20. Vectorized operations")

financial_data = pd.DataFrame(
    {
        "revenue": [100000, 150000, 125000, 200000],
        "cost": [70000, 90000, 80000, 130000],
    }
)

financial_data["profit"] = (
    financial_data["revenue"] - financial_data["cost"]
)

financial_data["margin"] = (
    financial_data["profit"] / financial_data["revenue"]
)

show("vectorized financial calculations", financial_data)

print(
    """
Vectorized operations apply calculations to whole Series or DataFrame
columns without manually writing a Python loop for every row.

For many standard operations, vectorization is clearer and faster than
row-by-row Python code.
"""
)


# ---------------------------------------------------------------------------
# 21. apply, map, and where
# ---------------------------------------------------------------------------

section("21. map(), apply(), and where()")

subsection("Series.map()")

grades = pd.Series(["A", "B", "A", "C", "B"], name="grade")

grade_points = grades.map(
    {
        "A": 4,
        "B": 3,
        "C": 2,
        "D": 1,
    }
)

show("grades", grades)
show("grade points", grade_points)

subsection("Series.apply()")

numbers = pd.Series([1, 2, 3, 4, 5])

squared = numbers.apply(lambda value: value ** 2)
show("squared with apply()", squared)

subsection("where()")

scores_for_status = pd.Series([35, 55, 72, 90])

passed_scores = scores_for_status.where(
    scores_for_status >= 50,
    other=0,
)

show("original scores", scores_for_status)
show("scores below 50 replaced by 0", passed_scores)

print(
    """
map() is particularly useful for element-wise value mapping on a Series.

apply() can execute a Python function for each element or, for DataFrames,
along an axis. It is flexible but should not automatically be preferred over
vectorized Pandas operations.

where() preserves values that satisfy a condition and replaces values that
do not.
"""
)


# ---------------------------------------------------------------------------
# 22. String operations
# ---------------------------------------------------------------------------

section("22. String operations")

people = pd.DataFrame(
    {
        "name": ["  Asha Sharma ", "Bharat Kumar", "CHARU SINGH", None],
        "email": [
            "asha@example.com",
            "bharat@example.com",
            "charu@example.com",
            None,
        ],
    }
)

people["clean_name"] = people["name"].str.strip().str.title()
people["email_domain"] = people["email"].str.split("@").str[-1]

show("cleaned people", people)

print("Names containing 'sharma':")
print(
    people[
        people["clean_name"].str.contains(
            "sharma",
            case=False,
            na=False,
        )
    ]
)


# ---------------------------------------------------------------------------
# 23. Date and time operations
# ---------------------------------------------------------------------------

section("23. Datetime fundamentals")

dates = pd.DataFrame(
    {
        "event": ["A", "B", "C"],
        "date": ["2026-01-15", "2026-03-20", "2026-06-10"],
    }
)

dates["date"] = pd.to_datetime(dates["date"])

dates["year"] = dates["date"].dt.year
dates["month"] = dates["date"].dt.month
dates["day"] = dates["date"].dt.day
dates["weekday"] = dates["date"].dt.day_name()

show("datetime data", dates)

print("Latest date:", dates["date"].max())
print("Earliest date:", dates["date"].min())

reference_date = pd.Timestamp("2026-07-01")
dates["days_from_reference"] = (
    reference_date - dates["date"]
).dt.days

show("date differences", dates)


# ---------------------------------------------------------------------------
# 24. Renaming
# ---------------------------------------------------------------------------

section("24. Renaming columns and index labels")

rename_example = students[["name", "score", "city"]].copy()

renamed = rename_example.rename(
    columns={
        "name": "student_name",
        "score": "exam_score",
        "city": "location",
    }
)

show("renamed columns", renamed)

renamed_index = renamed.rename(
    index={0: "student_0", 1: "student_1"}
)

show("renamed index labels", renamed_index)


# ---------------------------------------------------------------------------
# 25. Reordering columns
# ---------------------------------------------------------------------------

section("25. Reordering columns")

desired_order = [
    "name",
    "city",
    "age",
    "score",
    "performance",
    "passed",
    "score_percentage",
    "age_next_year",
    "result",
]

reordered = students[desired_order]
show("reordered columns", reordered)


# ---------------------------------------------------------------------------
# 26. Transpose
# ---------------------------------------------------------------------------

section("26. Transpose")

small_table = students[["name", "age", "score"]].head(3)
show("original", small_table)
show("transposed", small_table.T)

print(
    """
Transpose exchanges rows and columns. It is useful for some reporting and
matrix-oriented operations, but it can also change the apparent data types
and structure, so it should be used deliberately.
"""
)


# ---------------------------------------------------------------------------
# 27. Copying and view-related considerations
# ---------------------------------------------------------------------------

section("27. Copying DataFrames safely")

original = students[["name", "score"]].copy()
working_copy = original.copy()

working_copy["score"] = working_copy["score"] + 5

show("original", original)
show("working_copy", working_copy)

print(
    """
Use .copy() when you intentionally want an independent object.

A common source of confusion is chained indexing, for example:

    dataframe[dataframe["score"] > 80]["score"] = 100

This style can create ambiguous assignment behavior.

Prefer .loc for explicit assignment:

    dataframe.loc[dataframe["score"] > 80, "score"] = 100
"""
)


# ---------------------------------------------------------------------------
# 28. Explicit assignment with loc
# ---------------------------------------------------------------------------

section("28. Safe conditional assignment")

assignment_example = students[["name", "score"]].copy()

assignment_example.loc[
    assignment_example["score"] >= 85,
    "category",
] = "High"

assignment_example.loc[
    assignment_example["score"] < 85,
    "category",
] = "Standard"

show("conditional categories", assignment_example)


# ---------------------------------------------------------------------------
# 29. Index uniqueness
# ---------------------------------------------------------------------------

section("29. Index properties")

unique_index = pd.DataFrame(
    {"value": [10, 20, 30]},
    index=["A", "B", "C"],
)

duplicate_index = pd.DataFrame(
    {"value": [10, 20, 30]},
    index=["A", "A", "B"],
)

print("Unique index:")
print(unique_index.index.is_unique)

print("\nDuplicate index:")
print(duplicate_index.index.is_unique)

print("\nSelection from duplicate index:")
print(duplicate_index.loc["A"])


# ---------------------------------------------------------------------------
# 30. MultiIndex fundamentals
# ---------------------------------------------------------------------------

section("30. MultiIndex fundamentals")

multi = pd.DataFrame(
    {
        "sales": [100, 120, 90, 150],
        "profit": [20, 25, 15, 35],
    },
    index=pd.MultiIndex.from_tuples(
        [
            ("North", "Q1"),
            ("North", "Q2"),
            ("South", "Q1"),
            ("South", "Q2"),
        ],
        names=["region", "quarter"],
    ),
)

show("MultiIndex DataFrame", multi)

print("Select North:")
print(multi.loc["North"])

print("\nSelect North Q2:")
print(multi.loc[("North", "Q2")])

print(
    """
A MultiIndex represents multiple levels of labels.

It can model hierarchical structures such as:

    region -> quarter
    country -> state
    department -> employee

MultiIndex is powerful, but for simple analysis a normal flat DataFrame can
be easier to understand and maintain.
"""
)


# ---------------------------------------------------------------------------
# 31. Selecting columns with conditions
# ---------------------------------------------------------------------------

section("31. Conditional column selection")

numeric_columns = students.select_dtypes(include="number")
show("numeric columns", numeric_columns)

object_columns = students.select_dtypes(include="object")
show("object columns", object_columns)


# ---------------------------------------------------------------------------
# 32. Basic aggregation
# ---------------------------------------------------------------------------

section("32. Aggregation fundamentals")

sales_data = pd.DataFrame(
    {
        "region": ["North", "North", "South", "South", "West"],
        "salesperson": ["A", "B", "A", "C", "B"],
        "sales": [100, 150, 120, 180, 200],
    }
)

show("sales_data", sales_data)

print("Total sales:")
print(sales_data["sales"].sum())

print("\nMean sales:")
print(sales_data["sales"].mean())

print("\nSales by region:")
print(sales_data.groupby("region")["sales"].sum())


# ---------------------------------------------------------------------------
# 33. Grouping and sorting
# ---------------------------------------------------------------------------

section("33. GroupBy with sorting")

region_summary = (
    sales_data
    .groupby("region", as_index=False)
    .agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        transaction_count=("sales", "count"),
    )
    .sort_values("total_sales", ascending=False)
)

show("region_summary", region_summary)


# ---------------------------------------------------------------------------
# 34. Common indexing mistakes
# ---------------------------------------------------------------------------

section("34. Common indexing mistakes")

demonstrate_exception(
    "Attempting to select a missing column:",
    lambda: students["does_not_exist"],
)

demonstrate_exception(
    "Attempting to select a missing row label with loc:",
    lambda: students.loc[999],
)

print(
    """
KeyError commonly indicates that a requested label or column does not exist.

When debugging indexing:

    1. Check dataframe.columns.
    2. Check dataframe.index.
    3. Check spelling and capitalization.
    4. Check data types of labels.
    5. Decide whether you intended label-based or position-based selection.
"""
)


# ---------------------------------------------------------------------------
# 35. loc versus iloc comparison
# ---------------------------------------------------------------------------

section("35. loc versus iloc")

comparison = pd.DataFrame(
    {
        "name": ["Asha", "Bharat", "Charu"],
        "score": [88, 76, 95],
    },
    index=[10, 20, 30],
)

show("comparison", comparison)

print("loc[20]:")
print(comparison.loc[20])

print("\niloc[1]:")
print(comparison.iloc[1])

print(
    """
The DataFrame has labels 10, 20, and 30.

    loc[20]
        Means row whose label is 20.

    iloc[1]
        Means second row by integer position.

The two expressions happen to return the same row here, but for different
reasons.
"""
)


# ---------------------------------------------------------------------------
# 36. Advanced filtering edge cases
# ---------------------------------------------------------------------------

section("36. Filtering edge cases")

edge_case_data = pd.DataFrame(
    {
        "name": ["A", "B", "C", "D"],
        "score": [90, None, 70, 90],
        "city": ["Delhi", "Delhi", None, "Mumbai"],
    }
)

show("edge_case_data", edge_case_data)

print("score == 90:")
print(edge_case_data[edge_case_data["score"] == 90])

print("\ncity.isna():")
print(edge_case_data[edge_case_data["city"].isna()])

print(
    """
Missing values do not behave like ordinary values.

For missing numeric values, comparisons such as NaN > 50 do not evaluate
to True. Use isna() or notna() when the explicit question is whether data
is missing.
"""
)


# ---------------------------------------------------------------------------
# 37. Sorting edge cases
# ---------------------------------------------------------------------------

section("37. Sorting edge cases")

sort_edge = pd.DataFrame(
    {
        "name": ["A", "B", "C", "D"],
        "score": [80, None, 80, 70],
    }
)

print("Original:")
print(sort_edge)

print("\nDefault sorting:")
print(sort_edge.sort_values("score"))

print("\nMissing values first:")
print(sort_edge.sort_values("score", na_position="first"))

print(
    """
When multiple rows have equal sorting keys, secondary keys can be supplied
to make the desired ordering explicit.
"""
)

stable_sort = sort_edge.sort_values(
    by=["score", "name"],
    na_position="last",
)

show("score and name sorted", stable_sort)


# ---------------------------------------------------------------------------
# 38. DataFrame comparisons
# ---------------------------------------------------------------------------

section("38. DataFrame comparisons")

comparison_data = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [3, 2, 1],
    }
)

show("comparison_data", comparison_data)

print("a greater than b:")
print(comparison_data["a"] > comparison_data["b"])

print("\nRows where a > b:")
print(comparison_data[comparison_data["a"] > comparison_data["b"]])


# ---------------------------------------------------------------------------
# 39. Statistical methods
# ---------------------------------------------------------------------------

section("39. Statistical fundamentals")

statistics = pd.Series(
    [10, 20, 20, 30, 40, 50],
    name="measurements",
)

statistical_methods = {
    "count": statistics.count(),
    "sum": statistics.sum(),
    "mean": statistics.mean(),
    "median": statistics.median(),
    "min": statistics.min(),
    "max": statistics.max(),
    "variance": statistics.var(),
    "standard_deviation": statistics.std(),
    "25_percentile": statistics.quantile(0.25),
    "75_percentile": statistics.quantile(0.75),
}

for method_name, result in statistical_methods.items():
    print(f"{method_name}: {result}")


# ---------------------------------------------------------------------------
# 40. Reading and writing files conceptually
# ---------------------------------------------------------------------------

section("40. File input and output")

print(
    """
Pandas can read and write many common formats.

Typical examples include:

    pd.read_csv(...)
    pd.read_excel(...)
    pd.read_json(...)

and:

    dataframe.to_csv(...)
    dataframe.to_excel(...)
    dataframe.to_json(...)

This study script intentionally avoids requiring an external input file so
that it remains self-contained.
"""
)


# ---------------------------------------------------------------------------
# 41. CSV round-trip using an in-memory buffer
# ---------------------------------------------------------------------------

section("41. CSV round-trip without an external file")

from io import StringIO

csv_text = """name,score,city
Asha,88,Delhi
Bharat,76,Mumbai
Charu,95,Delhi
"""

csv_dataframe = pd.read_csv(StringIO(csv_text))
show("CSV parsed into DataFrame", csv_dataframe)

csv_output = csv_dataframe.to_csv(index=False)
print("\nDataFrame converted back to CSV:")
print(csv_output)


# ---------------------------------------------------------------------------
# 42. Memory and data types
# ---------------------------------------------------------------------------

section("42. Memory considerations")

memory_example = pd.DataFrame(
    {
        "category": [
            "Retail",
            "Retail",
            "Technology",
            "Technology",
            "Retail",
        ],
        "amount": [100, 200, 150, 300, 125],
    }
)

print("Memory usage before category conversion:")
print(memory_example.memory_usage(deep=True))

memory_optimized = memory_example.copy()
memory_optimized["category"] = memory_optimized["category"].astype("category")

print("\nMemory usage after category conversion:")
print(memory_optimized.memory_usage(deep=True))

print(
    """
Categorical dtype can reduce memory use when a column contains a relatively
small number of repeated labels.

It is not automatically better for every string column. The usefulness of
categorical data depends on cardinality, operations, and workload.
"""
)


# ---------------------------------------------------------------------------
# 43. Performance comparison: vectorization versus Python loop
# ---------------------------------------------------------------------------

section("43. Performance-oriented design")

performance_data = pd.DataFrame(
    {"value": range(1, 10001)}
)

subsection("Vectorized calculation")

performance_data["double_vectorized"] = performance_data["value"] * 2

subsection("List comprehension")

performance_data["double_comprehension"] = [
    value * 2 for value in performance_data["value"]
]

print(
    "Both calculations produce equivalent results:",
    performance_data["double_vectorized"].equals(
        performance_data["double_comprehension"]
    ),
)

print(
    """
For straightforward numerical operations, vectorized Pandas expressions are
normally preferred because they express the operation directly and can take
advantage of efficient underlying implementations.

Python loops are still appropriate when the transformation cannot reasonably
be expressed using vectorized operations.
"""
)


# ---------------------------------------------------------------------------
# 44. Data validation
# ---------------------------------------------------------------------------

section("44. Data validation")

validation_data = pd.DataFrame(
    {
        "name": ["Asha", "Bharat", "Charu"],
        "score": [88, 76, 95],
    }
)

assert not validation_data.empty
assert validation_data["name"].notna().all()
assert validation_data["score"].between(0, 100).all()

print("Validation checks passed.")

invalid_validation_data = pd.DataFrame(
    {
        "name": ["Asha", "Bharat"],
        "score": [88, 140],
    }
)

valid_scores = invalid_validation_data["score"].between(0, 100)

print("\nInvalid score rows:")
print(invalid_validation_data[~valid_scores])


# ---------------------------------------------------------------------------
# 45. End-to-end realistic example
# ---------------------------------------------------------------------------

section("45. End-to-end student performance analysis")

raw_students = pd.DataFrame(
    {
        "student_id": [101, 102, 103, 104, 105, 106, 106],
        "name": [
            " Asha Sharma ",
            "Bharat Kumar",
            "Charu Singh",
            "Deepak Verma",
            "Esha Gupta",
            "Farhan Ali",
            "Farhan Ali",
        ],
        "age": [21, 22, 20, 23, None, 22, 22],
        "math": [88, 76, 95, 67, 84, 91, 91],
        "science": [91, 72, 93, 70, None, 87, 87],
        "city": [
            "Delhi",
            "Mumbai",
            "Delhi",
            "Lucknow",
            "Mumbai",
            "Delhi",
            "Delhi",
        ],
    }
)

subsection("Step 1: Inspect raw data")

show("raw_students", raw_students)

subsection("Step 2: Remove exact duplicate records")

cleaned = raw_students.drop_duplicates()
show("after duplicate removal", cleaned)

subsection("Step 3: Clean names")

cleaned["name"] = cleaned["name"].str.strip().str.title()

subsection("Step 4: Handle missing age")

cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())

subsection("Step 5: Calculate average academic score")

cleaned["average_score"] = cleaned[["math", "science"]].mean(axis=1)

subsection("Step 6: Determine pass/fail")

cleaned["passed"] = cleaned["average_score"] >= 50

subsection("Step 7: Classify performance")

cleaned["performance"] = "Needs Improvement"

cleaned.loc[
    cleaned["average_score"] >= 85,
    "performance",
] = "Excellent"

cleaned.loc[
    (cleaned["average_score"] >= 70)
    & (cleaned["average_score"] < 85),
    "performance",
] = "Good"

cleaned.loc[
    (cleaned["average_score"] >= 50)
    & (cleaned["average_score"] < 70),
    "performance",
] = "Pass"

subsection("Step 8: Sort by performance score")

cleaned = cleaned.sort_values(
    by="average_score",
    ascending=False,
)

show("cleaned and sorted data", cleaned)

subsection("Step 9: Find high performers")

high_performers = cleaned[
    cleaned["average_score"] >= 85
]

show("high performers", high_performers)

subsection("Step 10: City-level analysis")

city_summary = (
    cleaned
    .groupby("city", as_index=False)
    .agg(
        students=("student_id", "count"),
        average_score=("average_score", "mean"),
        highest_score=("average_score", "max"),
    )
    .sort_values("average_score", ascending=False)
)

show("city summary", city_summary)

subsection("Step 11: Final validation")

assert cleaned["student_id"].is_unique
assert cleaned["average_score"].between(0, 100).all()
assert cleaned["passed"].notna().all()

print("Final validation passed.")


# ---------------------------------------------------------------------------
# 46. More advanced DataFrame selection patterns
# ---------------------------------------------------------------------------

section("46. Advanced selection patterns")

analysis_columns = [
    "student_id",
    "name",
    "city",
    "average_score",
    "performance",
]

analysis_view = cleaned.loc[:, analysis_columns]
show("selected analysis columns", analysis_view)

top_three = (
    cleaned
    .nlargest(3, "average_score")
    .loc[:, analysis_columns]
)

show("top three students", top_three)

bottom_two = (
    cleaned
    .nsmallest(2, "average_score")
    .loc[:, analysis_columns]
)

show("bottom two students", bottom_two)


# ---------------------------------------------------------------------------
# 47. Duplicate identifiers versus duplicate rows
# ---------------------------------------------------------------------------

section("47. Duplicate identifiers versus duplicate rows")

identifier_data = pd.DataFrame(
    {
        "student_id": [1, 2, 2, 3],
        "name": ["A", "B", "B Changed", "C"],
    }
)

print("Exact duplicate rows:")
print(identifier_data[identifier_data.duplicated()])

print("\nDuplicate student IDs:")
print(
    identifier_data[
        identifier_data["student_id"].duplicated(keep=False)
    ]
)

print(
    """
drop_duplicates() detects duplicate rows by default.

If the business rule is that student_id must be unique, check that specific
column rather than assuming duplicate rows and duplicate identifiers are the
same problem.
"""
)


# ---------------------------------------------------------------------------
# 48. Sorting with missing values and multiple keys
# ---------------------------------------------------------------------------

section("48. Complex sorting")

complex_sort = pd.DataFrame(
    {
        "department": [
            "IT",
            "IT",
            "Finance",
            "Finance",
            "IT",
        ],
        "employee": ["A", "B", "C", "D", "E"],
        "score": [88, None, 95, 95, 88],
    }
)

complex_sort = complex_sort.sort_values(
    by=["department", "score", "employee"],
    ascending=[True, False, True],
    na_position="last",
)

show("complex sorted data", complex_sort)


# ---------------------------------------------------------------------------
# 49. Index sorting and index reset
# ---------------------------------------------------------------------------

section("49. Index sorting workflow")

unsorted = pd.DataFrame(
    {"value": [30, 10, 20]},
    index=["C", "A", "B"],
)

show("unsorted index", unsorted)
show("sort_index()", unsorted.sort_index())

reset_and_sorted = (
    unsorted
    .sort_index()
    .reset_index()
)

show("sorted index then reset", reset_and_sorted)


# ---------------------------------------------------------------------------
# 50. Useful DataFrame methods
# ---------------------------------------------------------------------------

section("50. Useful DataFrame methods")

method_example = pd.DataFrame(
    {
        "name": ["A", "B", "C", "D"],
        "score": [80, 90, 70, 90],
        "city": ["Delhi", "Mumbai", "Delhi", "Lucknow"],
    }
)

print("shape:", method_example.shape)
print("columns:", method_example.columns.tolist())
print("index:", method_example.index.tolist())
print("empty:", method_example.empty)
print("duplicated rows:", method_example.duplicated().sum())
print("missing values:")
print(method_example.isna().sum())
print("unique cities:", method_example["city"].unique())
print("number of unique cities:", method_example["city"].nunique())


# ---------------------------------------------------------------------------
# 51. Common mistakes
# ---------------------------------------------------------------------------

section("51. Common mistakes")

mistakes = [
    "Confusing loc with iloc.",
    "Forgetting parentheses around conditions combined with & or |.",
    "Using 'and' or 'or' for element-wise Series conditions.",
    "Using == None instead of isna() for missing-value checks.",
    "Assuming a Series aligns by position when it actually aligns by labels.",
    "Assigning through chained indexing.",
    "Ignoring unexpected data types imported from external files.",
    "Dropping missing values without understanding why they are missing.",
    "Treating duplicate rows and duplicate identifiers as the same issue.",
    "Sorting without considering missing values or secondary sort keys.",
    "Using apply() where a vectorized expression is simpler.",
    "Changing an index without understanding how later selections will work.",
]

for number, mistake in enumerate(mistakes, start=1):
    print(f"{number}. {mistake}")


# ---------------------------------------------------------------------------
# 52. Best-practice checklist implemented as assertions
# ---------------------------------------------------------------------------

section("52. Practical best-practice checks")

best_practice_data = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "score": [90, 80, 70],
    }
)

assert isinstance(best_practice_data, pd.DataFrame)
assert best_practice_data["id"].is_unique
assert best_practice_data["score"].between(0, 100).all()
assert best_practice_data["name"].notna().all()

print(
    """
The checks above demonstrate a simple principle:

    Inspect -> clean -> validate -> transform -> filter -> sort -> analyze.

The exact order may change for a specific dataset, but validation should not
be treated as an optional final step.
"""
)


# ---------------------------------------------------------------------------
# 53. Production considerations
# ---------------------------------------------------------------------------

section("53. Production-oriented considerations")

print(
    """
For real-world Pandas workloads, consider:

Data correctness
    Validate columns, data types, ranges, uniqueness, missing values, and
    business rules before trusting analytical results.

Reproducibility
    Keep transformations explicit and deterministic. Avoid relying on hidden
    state or accidental index behavior.

Memory
    Pandas generally operates in memory. Large datasets can require careful
    dtype selection, chunked reading, filtering early, and avoiding
    unnecessary copies.

Performance
    Prefer vectorized operations for standard transformations. Select only
    required columns, avoid repeated expensive operations, and profile actual
    bottlenecks before optimizing.

Index design
    Use meaningful indexes when label-based access benefits from them, but do
    not create complicated indexes merely because Pandas supports them.

Missing data
    Preserve information about missingness when it has analytical meaning.
    Imputation should reflect the data-generating process.

Sorting
    Sort explicitly when ordering matters. Do not assume that the current
    row order represents a meaningful business order.

Data types
    Inspect dtypes after importing external data. Numeric-looking strings,
    mixed values, dates, and categorical data may need explicit conversion.

Security
    Treat external files as untrusted input. Validate content before using
    imported data in downstream systems. Avoid executing arbitrary code based
    on file contents.

Testing
    Test important transformations with representative and edge-case data.
    Assertions are useful for basic invariants, while a full test suite is
    appropriate for production pipelines.
"""
)


# ---------------------------------------------------------------------------
# 54. Performance-aware column selection
# ---------------------------------------------------------------------------

section("54. Selecting only required columns")

large_table_simulation = pd.DataFrame(
    {
        "id": range(1, 101),
        "name": [f"Student {i}" for i in range(1, 101)],
        "score": [i % 101 for i in range(1, 101)],
        "unused_text": ["unused"] * 100,
    }
)

small_view = large_table_simulation[["id", "score"]]

print("Original columns:", large_table_simulation.columns.tolist())
print("Selected columns:", small_view.columns.tolist())

print(
    """
Selecting only required columns can reduce memory usage and make subsequent
operations easier to understand, especially in larger analytical workflows.
"""
)


# ---------------------------------------------------------------------------
# 55. Stable reproducible sampling
# ---------------------------------------------------------------------------

section("55. Reproducible sampling")

sample_one = students.sample(3, random_state=42)
sample_two = students.sample(3, random_state=42)

print("Samples are identical:")
print(sample_one.equals(sample_two))

show("sample", sample_one)


# ---------------------------------------------------------------------------
# 56. Column order and deterministic output
# ---------------------------------------------------------------------------

section("56. Deterministic analytical output")

final_columns = [
    "student_id",
    "name",
    "city",
    "average_score",
    "performance",
    "passed",
]

final_report = (
    cleaned.loc[:, final_columns]
    .sort_values(
        by=["average_score", "name"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

show("final report", final_report)

assert final_report.index.equals(
    pd.RangeIndex(start=0, stop=len(final_report))
)


# ---------------------------------------------------------------------------
# 57. Compact reference examples
# ---------------------------------------------------------------------------

section("57. Pandas quick-reference examples")

reference = pd.DataFrame(
    {
        "name": ["A", "B", "C"],
        "score": [80, 95, 70],
        "city": ["Delhi", "Mumbai", "Delhi"],
    }
)

print(
    """
Create:
    pd.Series([1, 2, 3])
    pd.DataFrame({"a": [1, 2], "b": [3, 4]})

Inspect:
    df.head()
    df.tail()
    df.shape
    df.columns
    df.index
    df.dtypes
    df.info()
    df.describe()

Select:
    df["score"]
    df[["name", "score"]]
    df.loc[0]
    df.loc[:, ["name", "score"]]
    df.iloc[0]
    df.iloc[:, 0:2]

Filter:
    df[df["score"] > 80]
    df[(df["score"] > 80) & (df["city"] == "Delhi")]
    df[df["city"].isin(["Delhi", "Mumbai"])]

Sort:
    df.sort_values("score")
    df.sort_values("score", ascending=False)
    df.sort_values(["city", "score"])
    df.sort_index()

Modify:
    df["new_column"] = ...
    df.loc[condition, "column"] = value

Missing data:
    df.isna()
    df.notna()
    df.dropna()
    df.fillna(value)

Index:
    df.set_index("name")
    df.reset_index()

Duplicates:
    df.duplicated()
    df.drop_duplicates()

Statistics:
    df["score"].mean()
    df["score"].median()
    df["score"].min()
    df["score"].max()
    df["score"].sum()

Grouping:
    df.groupby("city")["score"].mean()
"""
)

show("reference DataFrame", reference)


# ---------------------------------------------------------------------------
# 58. Final execution check
# ---------------------------------------------------------------------------

section("58. Execution check")

print("Python version:", sys.version.split()[0])
print("Pandas version:", pd.__version__)
print("Execution timestamp:", datetime.now().isoformat(timespec="seconds"))

print(
    """
The script completed successfully.

The central Pandas workflow demonstrated throughout this file is:

    create or load data
        ->
    inspect structure and data types
        ->
    select relevant rows and columns
        ->
    clean missing or invalid data
        ->
    validate assumptions
        ->
    transform columns
        ->
    filter records
        ->
    sort and organize results
        ->
    aggregate or analyze
        ->
    produce a deterministic result
"""
)


if __name__ == "__main__":
    print("\nPandas fundamentals study script finished.")
