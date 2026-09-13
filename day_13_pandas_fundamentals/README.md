# Pandas fundamentals: Series, DataFrames, indexing, filtering, sorting

## Introduction

Pandas is a Python library designed for working with structured and labeled data. It is widely used for data cleaning, exploratory analysis, statistical analysis, reporting, feature preparation, and many forms of tabular data processing.

The two central Pandas data structures are:

- **Series**: a one-dimensional labeled array.
- **DataFrame**: a two-dimensional labeled table containing rows and columns.

A useful way to understand a DataFrame is to view each column as a Series sharing the same row index. This relationship allows Pandas to perform label-aware selection, alignment, arithmetic, filtering, transformation, and aggregation.

The accompanying Python script develops these concepts progressively. It begins with basic Series and DataFrame creation and moves into indexing, Boolean filtering, sorting, missing-value handling, data types, conditional transformations, MultiIndex structures, validation, and performance-oriented practices.

## Series

A Series contains values and an associated index.

A simple Series can be created from a list. When no index is provided, Pandas creates an integer index beginning at zero.

A Series can also have explicit labels. For example, student names can serve as labels for examination scores.

The main components of a Series are:

- values
- index
- data type
- optional name

The index is important because Pandas operations generally understand labels rather than treating data as an unstructured sequence.

### Series indexing

Series values can be selected by label or by position.

`loc` performs label-based selection.

`iloc` performs integer-position-based selection.

For example, selecting the label `"Tue"` and selecting position `1` can refer to the same value when the Series uses the default sequential index, but the two operations have different meanings.

A label-based slice with `loc` normally includes both endpoint labels. Integer-position slicing with `iloc` follows normal Python slicing behavior, where the stop position is excluded.

### Series filtering

A Series can be filtered using a Boolean condition. An expression such as `series > 1500` produces a Boolean Series. Passing that Boolean result back into the Series selects the matching elements.

This mechanism is the foundation of DataFrame filtering.

## DataFrames

A DataFrame represents tabular data with rows and columns.

A DataFrame can be created from:

- dictionaries
- lists of dictionaries
- two-dimensional arrays
- other DataFrames
- external files and data sources

The script primarily uses dictionaries and in-memory data so that every example remains self-contained.

Important DataFrame properties include:

- `shape`
- `columns`
- `index`
- `dtypes`
- `empty`

Useful inspection methods include:

- `head()`
- `tail()`
- `sample()`
- `info()`
- `describe()`

Inspection should normally happen before substantial transformation. Understanding the structure and types of a dataset reduces the likelihood of applying an inappropriate operation.

## DataFrame columns

A single column can be selected with bracket notation.

Selecting one column normally produces a Series:

`df["score"]`

Selecting multiple columns produces another DataFrame:

`df[["name", "score"]]`

Bracket notation is generally preferable to attribute-style access because it works with column names containing spaces or other characters and avoids conflicts with existing DataFrame attributes or methods.

A new column can be created through assignment:

`df["passed"] = df["score"] >= 50`

The expression on the right is evaluated element by element and produces a Boolean Series aligned with the DataFrame index.

## Adding and modifying columns

Pandas supports vectorized column operations.

For example, a score column can be transformed directly:

`df["score_percentage"] = df["score"] / 100`

No explicit Python loop is required.

Columns can also be created from multiple existing columns. A financial DataFrame containing revenue and cost can calculate profit using:

`profit = revenue - cost`

and profit margin using:

`margin = profit / revenue`

Vectorized expressions are normally clearer and more efficient than explicit row-by-row Python loops for standard numerical operations.

## Removing columns

Columns can be removed with `drop()`.

A typical operation is:

`df = df.drop(columns=["column_name"])`

This creates a modified DataFrame through assignment. Explicit assignment often makes data transformations easier to reason about than relying heavily on in-place modifications.

## Indexes

An index identifies rows within a Pandas object.

The default DataFrame index is usually a `RangeIndex`, containing sequential integers. An index can also contain meaningful labels such as:

- student names
- dates
- transaction identifiers
- product codes
- geographic categories

An index is not automatically equivalent to a database primary key. It may contain duplicate labels, although duplicate indexes can make selection less intuitive.

### Setting an index

`set_index()` converts an existing column into the index.

For example:

`df.set_index("name")`

allows label-based selection such as:

`df.loc["Asha"]`

### Resetting an index

`reset_index()` converts index information back into ordinary columns.

`reset_index(drop=True)` discards the existing index and creates a fresh default integer index.

Resetting an index is often useful after sorting, filtering, concatenation, or other transformations when a clean sequential index is desired.

## loc and iloc

The distinction between `loc` and `iloc` is one of the most important concepts in Pandas.

### loc

`loc` is label based.

Examples include:

`df.loc[20]`

`df.loc[[10, 30]]`

`df.loc[:, ["name", "score"]]`

The number `20` in `df.loc[20]` means the row whose label is `20`.

### iloc

`iloc` is position based.

Examples include:

`df.iloc[0]`

`df.iloc[1:4]`

`df.iloc[:, 0:2]`

The number `0` in `df.iloc[0]` means the first row position, regardless of the row's label.

This distinction becomes particularly important when a DataFrame has a custom index.

## Boolean filtering

Filtering selects rows according to conditions.

A simple condition is:

`df["score"] > 80`

This creates a Boolean mask.

Applying the mask:

`df[df["score"] > 80]`

returns rows where the condition is true.

### Combining conditions

Multiple conditions use element-wise Boolean operators:

- `&` for AND
- `|` for OR
- `~` for NOT

Each comparison should be surrounded by parentheses.

Correct form:

`(df["score"] >= 70) & (df["city"] == "Delhi")`

Using Python's `and` and `or` for Series conditions is incorrect because those operators expect individual Boolean values rather than element-wise Boolean arrays.

### isin()

`isin()` is useful when a column should match any value from a collection.

For example:

`df[df["city"].isin(["Delhi", "Mumbai"])]`

is clearer than writing a long sequence of equality comparisons.

### between()

`between()` provides a convenient way to select numeric or comparable values within a range.

For example:

`df[df["score"].between(70, 90)]`

selects scores from 70 through 90.

### String filtering

The `.str` accessor provides string-oriented operations.

Examples include:

- `str.startswith()`
- `str.endswith()`
- `str.contains()`
- `str.lower()`
- `str.upper()`
- `str.strip()`
- `str.title()`

Missing string values should be considered when using these operations. Parameters such as `na=False` can be useful when a Boolean filtering expression must handle missing values safely.

## query()

Pandas provides `query()` as another filtering interface.

A condition such as:

`df.query("score >= 80 and age <= 22")`

can be easier to read when the filtering expression becomes complex.

Python variables can be referenced inside a query with the `@` notation.

Bracket-based Boolean filtering remains useful because it is explicit and works naturally with arbitrary Series expressions.

## Sorting

Rows can be sorted with `sort_values()`.

Ascending order is the default:

`df.sort_values("score")`

Descending order can be requested:

`df.sort_values("score", ascending=False)`

Multiple columns can be supplied:

`df.sort_values(["city", "score"])`

Different directions can be specified for each sorting key:

`df.sort_values(["city", "score"], ascending=[True, False])`

This is useful when the primary grouping should be alphabetical while the secondary measure should be ranked from highest to lowest.

### Missing values during sorting

Sorting with missing values requires an explicit understanding of where missing records should appear.

`na_position="first"` places missing values first.

`na_position="last"` places them last.

The choice should reflect the analytical meaning of the output rather than being treated as a cosmetic decision.

### Sorting by index

`sort_index()` sorts rows according to index labels rather than column values.

This is useful when the index itself has a meaningful ordering.

## Filtering and sorting together

Filtering and sorting are often combined in analytical workflows.

For example, a high-performance subset can first be selected using a Boolean condition and then ordered by score:

`df[df["score"] >= 80].sort_values("score", ascending=False)`

This produces a focused result without modifying the original DataFrame.

## Missing values

Real datasets frequently contain missing information.

Pandas commonly represents missing numerical values using `NaN` or related missing-value representations.

Important methods include:

- `isna()`
- `notna()`
- `dropna()`
- `fillna()`

`isna()` creates a Boolean structure showing where values are missing.

`isna().sum()` can be used to count missing values by column.

`dropna()` removes rows or columns containing missing values according to its parameters.

`fillna()` replaces missing values.

### Missing-value comparisons

Missing values should not be treated as ordinary values.

For example, checking whether a value is missing should use:

`series.isna()`

rather than relying on equality comparisons with `None`.

The correct treatment of missing data depends on the data-generating process. Possible strategies include deletion, constant replacement, mean or median imputation, forward filling, backward filling, or more advanced imputation methods.

An automatic replacement can introduce bias if the reason for missingness is ignored.

## Duplicate data

Pandas provides:

- `duplicated()`
- `drop_duplicates()`

`duplicated()` identifies duplicate rows.

`drop_duplicates()` removes duplicate rows.

A critical distinction is the difference between duplicate rows and duplicate identifiers.

Two records may have different information while sharing the same identifier. If an identifier is supposed to be unique, uniqueness should be checked specifically for that column.

For example:

`df["student_id"].is_unique`

tests whether every student identifier is unique.

## Data types

Data types determine how Pandas interprets values and which operations are appropriate.

Common types include:

- integer
- floating-point
- Boolean
- string/object
- datetime
- categorical

The `dtypes` property shows column types.

`astype()` can be used for explicit conversions.

For messy numerical data, `pd.to_numeric()` can convert strings into numeric values.

With `errors="coerce"`, invalid values become missing rather than raising an exception.

This is useful for data cleaning but should be used carefully. Invalid input can represent a real data-quality problem that should be investigated rather than silently converted.

## String cleaning

Text data frequently requires normalization.

Common operations include:

- removing surrounding whitespace
- standardizing capitalization
- searching for patterns
- extracting substrings
- splitting text

For example:

`df["name"].str.strip().str.title()`

removes surrounding whitespace and applies title-style capitalization.

String operations can be chained when each transformation returns another Series.

## Datetime handling

Dates should generally be converted to a proper datetime dtype rather than being retained as arbitrary strings.

`pd.to_datetime()` performs datetime conversion.

Once a column contains datetime values, the `.dt` accessor can expose components such as:

- year
- month
- day
- weekday
- hour
- minute

Date arithmetic produces timedeltas. These can be converted into numeric durations such as a number of days.

Proper datetime types are essential for reliable date filtering, sorting, grouping, and interval calculations.

## Vectorized operations

Vectorization is a major reason Pandas is useful for tabular data.

Instead of manually iterating through every row, an operation can be applied to an entire Series.

For example:

`df["profit"] = df["revenue"] - df["cost"]`

performs the calculation across the complete column.

Vectorized operations are usually preferable for straightforward transformations because they are concise and can take advantage of efficient underlying implementations.

Python loops remain valid when a transformation is inherently procedural or cannot reasonably be expressed through existing vectorized operations.

## map(), apply(), and where()

These methods solve related but different problems.

### map()

`Series.map()` is particularly useful for mapping existing values to other values.

For example, letter grades can be mapped to numerical grade points.

### apply()

`Series.apply()` can execute a Python function element by element.

It is flexible, but it should not be used automatically for every transformation. A vectorized Pandas expression is often preferable when one exists.

### where()

`where()` preserves values satisfying a condition and replaces values that do not.

It is useful for conditional replacement while retaining the original Series structure.

## Conditional assignment

Conditional updates should generally use `.loc`.

A safe pattern is:

`df.loc[df["score"] >= 85, "category"] = "High"`

This explicitly states:

- which rows should be modified
- which column should be modified
- what value should be assigned

This is clearer than modifying the result of chained indexing.

## Chained indexing

A common source of confusion is code that selects data in multiple indexing operations and then tries to modify the intermediate result.

For example, conceptually:

`df[df["score"] > 80]["score"] = 100`

can lead to ambiguous assignment behavior.

The preferred approach is explicit `.loc` assignment:

`df.loc[df["score"] > 80, "score"] = 100`

The explicit form clearly identifies the original DataFrame, target rows, and target column.

## Data alignment

Pandas performs label-based alignment during many Series and DataFrame operations.

Suppose one Series contains labels `A`, `B`, and `C`, while another contains `B`, `C`, and `D`.

Adding the two Series aligns matching labels rather than simply adding values at identical physical positions.

The unmatched labels produce missing results.

This behavior is extremely powerful because it prevents many accidental positional calculations, but it can also surprise users who expect ordinary array-style behavior.

## Statistical operations

Pandas provides many statistical methods directly.

Common methods include:

- `count()`
- `sum()`
- `mean()`
- `median()`
- `min()`
- `max()`
- `var()`
- `std()`
- `quantile()`

These methods commonly handle missing values according to Pandas' default statistical rules.

The choice of statistic should depend on the data. For example, the median is often less sensitive to extreme values than the mean.

## Ranking

`rank()` assigns relative positions to observations.

Different ranking methods determine how ties are handled.

The script demonstrates the `dense` method, where tied observations receive the same rank and subsequent ranks increase without gaps.

Ranking is useful for leaderboards, performance analysis, prioritization, and comparative reporting.

## Grouping and aggregation

Although the primary focus is Series, DataFrames, indexing, filtering, and sorting, grouping provides an important connection to practical DataFrame analysis.

`groupby()` partitions records according to one or more keys.

For example:

`df.groupby("region")["sales"].sum()`

calculates total sales for each region.

Multiple metrics can be produced with `agg()`.

The script demonstrates grouped calculations such as:

- total sales
- average sales
- transaction count
- highest score

A common workflow is:

1. group records
2. calculate aggregate measures
3. return a DataFrame
4. sort the resulting summary

## MultiIndex

A MultiIndex contains multiple levels of labels.

For example, data can be indexed by:

- region and quarter
- country and state
- department and employee

A MultiIndex can represent hierarchical structures naturally.

Selection can use a tuple representing multiple levels.

Although powerful, MultiIndex can make code more complex. A simple flat DataFrame is often preferable when hierarchical indexing does not provide a meaningful analytical advantage.

## Index uniqueness

An index can be unique or duplicated.

`df.index.is_unique` checks whether all index labels are unique.

Duplicate index labels are allowed, but selecting one label may return multiple rows.

This behavior differs from the assumption that a label identifies exactly one record.

If unique record identification is required, the data should be validated accordingly.

## Selecting numeric and non-numeric columns

`select_dtypes()` can select columns based on their data types.

For example:

`df.select_dtypes(include="number")`

selects numeric columns.

This is useful when performing numerical analysis without manually listing every numeric column.

## Reordering columns

Columns can be reordered by supplying the desired column sequence.

This is useful when preparing analytical reports or exporting a final table.

Column ordering does not normally affect the underlying meaning of the data, but deterministic column order improves readability and downstream usability.

## Transposing a DataFrame

The `.T` property transposes a DataFrame.

Rows become columns and columns become rows.

Transposition can be useful for matrix-like reporting, but it may alter how data types are represented when heterogeneous columns are combined into a new orientation.

It should therefore be used intentionally.

## Copying DataFrames

`copy()` creates an independent DataFrame object for many common transformation workflows.

For example:

`working_copy = original.copy()`

allows changes to the working object without intending to modify the original.

Explicit copies are particularly useful when preparing separate analytical views or transformation stages.

## CSV input and output

Pandas supports CSV data through:

`pd.read_csv()`

and:

`DataFrame.to_csv()`

CSV is common for exchanging tabular data because it is simple and widely supported.

The script demonstrates a CSV round-trip using an in-memory text buffer, avoiding dependence on an external file.

When working with real files, important considerations include:

- encoding
- delimiter
- missing-value representation
- column types
- malformed records
- date parsing
- large-file memory usage

## Memory considerations

Pandas DataFrames normally operate in memory.

For large datasets, memory use can become a significant limitation.

Possible techniques include:

- selecting only required columns
- using appropriate data types
- converting repeated low-cardinality strings to categorical data
- processing large input files in chunks
- avoiding unnecessary DataFrame copies
- filtering data early when appropriate

Categorical data can reduce memory usage when a column contains many repeated values relative to its total number of records.

It is not universally beneficial. High-cardinality columns may receive little or no advantage, and some operations may have different performance characteristics.

## Performance considerations

A practical Pandas workflow should distinguish between readable code and premature optimization.

For common arithmetic and logical transformations, vectorized operations are generally preferred.

Selecting only required columns can reduce both memory usage and processing requirements.

Repeatedly creating large intermediate DataFrames can increase memory pressure.

When performance matters, actual workload measurement is more useful than assuming a particular implementation is faster.

The script demonstrates equivalent vectorized and list-comprehension calculations while emphasizing vectorized operations for standard tabular arithmetic.

## Sampling and reproducibility

`sample()` can select random rows from a DataFrame.

A `random_state` value makes the sampling operation reproducible.

Reproducibility is important for testing, demonstrations, debugging, and analytical workflows where the same sample needs to be regenerated.

## Data validation

Cleaning data without validation can produce incorrect results that appear plausible.

Validation should check assumptions relevant to the dataset.

Examples include:

- required columns exist
- required fields are not missing
- numeric values are within expected ranges
- identifiers are unique
- categories belong to an accepted set
- calculated values remain within logical bounds

The script uses assertions to demonstrate simple invariants such as valid scores and unique identifiers.

For production systems, validation should normally be more comprehensive than a few assertions.

## Common mistakes

### Confusing loc and iloc

`loc` uses labels.

`iloc` uses integer positions.

The difference becomes obvious when the DataFrame index is not `0, 1, 2, ...`.

### Using and or or with Series

Python's `and` and `or` are not the correct operators for ordinary element-wise Pandas filtering.

Use:

- `&`
- `|`
- `~`

with appropriate parentheses.

### Incorrect missing-value checks

Use `isna()` and `notna()` for explicit missing-value detection.

### Ignoring data types

Imported data may contain numbers represented as strings, dates represented as text, or columns containing mixed values.

Always inspect `dtypes` before performing important transformations.

### Assuming duplicate rows mean duplicate identifiers

Two records can share an identifier without being identical rows.

Check the business rule directly.

### Unnecessary use of apply()

`apply()` is flexible, but a vectorized operation is often clearer and faster for simple transformations.

### Unsafe chained assignment

Use `.loc` when conditionally modifying DataFrame values.

### Sorting without considering missing values

Missing-value placement can affect reports and rankings. Specify `na_position` when the output requirement is explicit.

## Practical analysis workflow

A robust basic workflow for many Pandas tasks is:

1. Create or load the data.
2. Inspect the shape, columns, index, and data types.
3. Identify missing, invalid, or duplicate records.
4. Clean and standardize relevant values.
5. Validate important assumptions.
6. Create calculated columns.
7. Filter relevant rows.
8. Select relevant columns.
9. Sort the result when ordering matters.
10. Aggregate or analyze the prepared data.
11. Produce a deterministic final result.

The exact sequence can change depending on the task. Validation should remain an explicit part of the workflow rather than being treated as an optional final check.

## Real-world applications

Pandas fundamentals apply to many practical data-processing tasks.

### Business analysis

DataFrames can represent:

- revenue
- costs
- transactions
- customers
- products
- regions
- performance indicators

Filtering and sorting can identify high-value records, while grouping can produce regional or product-level summaries.

### Financial analysis

Pandas can organize:

- prices
- returns
- transactions
- portfolios
- financial ratios
- cash flows

Datetime operations are particularly important for time-based financial data.

### Education analytics

Student DataFrames can contain:

- identifiers
- demographics
- examination scores
- attendance
- subjects
- locations

Filtering can identify high performers, missing records, or students requiring attention.

### Data cleaning

Pandas is widely suited to preprocessing structured datasets before statistical analysis or machine-learning workflows.

Cleaning tasks can include:

- duplicate removal
- type conversion
- missing-value handling
- string normalization
- date conversion
- validation
- filtering invalid records

### Reporting

Pandas can transform raw tables into deterministic reports by selecting required columns, calculating metrics, grouping records, and sorting the final output.

## Security and data-quality considerations

Pandas itself does not make external data trustworthy.

Files and datasets obtained from external sources should be treated as untrusted input.

Important practices include:

- validate imported columns and types
- check expected ranges
- detect malformed records
- avoid blindly trusting identifiers
- inspect unexpected object or string values
- preserve evidence of data-quality problems
- avoid silently coercing invalid records without review
- avoid executing arbitrary content derived from data

Data quality is closely related to analytical security because incorrect or manipulated input can produce incorrect business decisions even when the Pandas code itself is technically correct.

## Implementation considerations

A good Pandas implementation should favor:

- explicit transformations
- meaningful column names
- clear indexing
- predictable output order
- appropriate data types
- validation of assumptions
- vectorized operations where appropriate
- deliberate handling of missing data
- explicit sorting when order matters
- controlled use of copies
- deterministic analytical results

The Python script demonstrates these principles through executable examples rather than treating Pandas as a collection of isolated syntax rules.

## Important distinctions

| Concept | Meaning |
|---|---|
| Series | One-dimensional labeled data structure |
| DataFrame | Two-dimensional labeled table |
| Index | Labels identifying rows |
| loc | Label-based selection |
| iloc | Integer-position-based selection |
| Boolean mask | True/False structure used for filtering |
| isin | Membership-based filtering |
| between | Range-based filtering |
| sort_values | Sort rows according to column values |
| sort_index | Sort according to index labels |
| isna | Detect missing values |
| fillna | Replace missing values |
| dropna | Remove missing records |
| duplicated | Detect duplicate records |
| drop_duplicates | Remove duplicate records |
| groupby | Partition data for grouped analysis |
| MultiIndex | Hierarchical index with multiple levels |
| map | Map Series values through a mapping or function |
| apply | Apply a function to Series or DataFrame data |
| where | Preserve values meeting a condition and replace others |

## Key indexing principle

The most important indexing distinction is:

`loc` asks:

**Which label do I want?**

`iloc` asks:

**Which position do I want?**

This distinction should be understood before working extensively with Pandas because many selection and assignment operations depend on it.
