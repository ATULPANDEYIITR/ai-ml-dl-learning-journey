# Data manipulation with pandas

## Introduction

Data manipulation is the process of transforming raw tabular data into a structure that can be analyzed, compared, aggregated, validated, and used for decision-making.

This study script focuses on six closely related areas of pandas data manipulation:

- `GroupBy`
- `merge`
- `join`
- `concat`
- pivot tables
- missing-data handling

The examples also cover related operations such as `pivot`, `melt`, `stack`, `unstack`, `transform`, `apply`, time-based grouping, validation, duplicate detection, index alignment, and analytical data-quality checks.

The examples use a sales dataset and progressively build more realistic analytical workflows.

## Requirements

The script requires Python and pandas.

Install pandas with:

    pip install pandas

The script does not require external datasets. All demonstration data is created inside the Python file.

## Fundamental pandas data manipulation

A pandas `DataFrame` represents tabular data consisting of rows and labeled columns. Each column is a pandas `Series`.

Typical data-manipulation operations include:

- selecting columns
- filtering rows
- creating calculated columns
- grouping records
- aggregating values
- combining datasets
- reshaping datasets
- detecting and handling missing values
- validating relationships between tables

For example, a calculated revenue column can be created from quantity and unit price:

    sales["revenue"] = sales["quantity"] * sales["unit_price"]

This is a vectorized pandas operation. It applies the calculation to the complete column without requiring an explicit Python loop.

## GroupBy

### Definition

`groupby()` implements the split-apply-combine pattern.

The data is first split into groups according to one or more columns. An operation is then applied to each group, and the results are combined into an output structure.

A simple example is:

    sales.groupby("region")["net_revenue"].sum()

This produces total revenue for each region.

The important idea is that `groupby()` does not itself calculate a summary. It creates a grouped representation that can then be aggregated, transformed, filtered, or otherwise processed.

### Grouping by one column

A single grouping key creates one group for each distinct value.

Examples include:

    sales.groupby("region")["net_revenue"].sum()

    sales.groupby("category")["net_revenue"].mean()

    sales.groupby("region")["order_id"].count()

These operations answer questions such as:

- What is total revenue by region?
- What is average revenue by product category?
- How many orders belong to each region?

### Grouping by multiple columns

Multiple grouping keys create combinations of values.

For example:

    sales.groupby(["region", "category"])["net_revenue"].sum()

The result represents revenue for each region-category combination.

This is useful when one-dimensional grouping is insufficient.

### `as_index`

By default, grouping keys are often placed in the result index.

Using:

    sales.groupby("region", as_index=False)["net_revenue"].sum()

keeps `region` as a regular column.

This is convenient when the result will be passed into later DataFrame operations.

`reset_index()` provides another way to convert index levels into columns:

    result = (
        sales.groupby(["region", "category"])["net_revenue"]
        .sum()
        .reset_index()
    )

## GroupBy aggregation

Aggregation reduces each group to one or more summary values.

Common aggregations include:

- `sum`
- `mean`
- `median`
- `min`
- `max`
- `count`
- `size`
- `nunique`
- `std`
- `var`

Multiple aggregations can be requested together.

The script demonstrates:

    sales.groupby("region")["net_revenue"].agg(
        ["sum", "mean", "min", "max", "count"]
    )

Named aggregation provides clearer output column names:

    sales.groupby("region").agg(
        total_revenue=("net_revenue", "sum"),
        average_order_value=("net_revenue", "mean"),
        largest_order=("net_revenue", "max"),
        order_count=("order_id", "count"),
        customer_count=("customer", "nunique"),
    )

Named aggregation is particularly useful in production analytical pipelines because the resulting column names directly describe the metric.

## `count` versus `size`

`count()` counts non-missing observations in the selected column.

`size()` counts rows in the group.

This distinction matters when missing values exist.

For example:

    frame.groupby("region").size()

counts every row.

By contrast:

    frame.groupby("region")["revenue"].count()

counts only rows where `revenue` is not missing.

## Custom aggregation

A custom Python function can be supplied when built-in aggregations do not express the required calculation.

The script defines a revenue range:

    def revenue_range(series):
        return series.max() - series.min()

It can then be used in an aggregation.

Custom functions are useful, but built-in pandas operations are generally preferable when they express the same calculation because they are usually clearer and can be more efficient.

## GroupBy transform

`transform()` differs fundamentally from ordinary aggregation.

An aggregation usually reduces the number of rows.

A transformation returns a result aligned with the original rows.

For example:

    sales["region_total"] = (
        sales.groupby("region")["net_revenue"]
        .transform("sum")
    )

Every order receives the total revenue of its region.

This makes `transform()` useful for:

- percentage-of-group calculations
- group-level normalization
- group-level comparisons
- rankings and thresholds
- attaching group statistics to original observations

The script calculates:

    sales["share_of_region"] = (
        sales["net_revenue"] / sales["region_total"]
    )

Each order can therefore be interpreted relative to the total revenue of its region.

## Group-wise ranking

A rank can be calculated independently inside each group.

The script uses:

    sales.groupby("region")["net_revenue"].rank(
        method="dense",
        ascending=False
    )

This ranks orders within their respective regions instead of ranking all orders together.

Group-wise ranking is useful for identifying:

- top customers
- highest-value transactions
- best-performing products
- regional leaders
- top-N records within categories

## GroupBy shift

`shift()` accesses previous or subsequent observations.

When combined with grouping, it can compare observations within each entity.

The script calculates each customer's previous order:

    sales.groupby("customer")["net_revenue"].shift(1)

This supports calculations such as:

- change from previous transaction
- previous purchase amount
- period-over-period comparison
- customer activity sequences

The first observation in each group has no previous observation, so its shifted value is missing.

## GroupBy cumulative operations

Cumulative functions maintain a running result within each group.

For example:

    sales.groupby("customer")["net_revenue"].cumsum()

produces cumulative customer revenue.

Other useful cumulative operations include cumulative maximum, minimum, and product.

## GroupBy filtering

`filter()` can retain or remove entire groups based on a condition.

The script demonstrates keeping regions whose total revenue meets a threshold.

This is different from ordinary row filtering. A group-level filter decides whether the complete group remains in the output.

## GroupBy apply

`apply()` provides flexible group-wise processing.

The script uses it to obtain the top two orders for each customer.

`apply()` is powerful, but it can be slower and more complex than specialized pandas operations. When a task can be expressed with `agg`, `transform`, `filter`, `rank`, `nlargest`, or other vectorized operations, those alternatives are usually preferable.

## Grouping categorical data

Pandas supports categorical columns through the `category` dtype.

Categorical data is useful when a column contains a relatively small set of repeated labels, such as:

- region
- product category
- customer segment
- status

The script demonstrates:

    sales["category"] = sales["category"].astype("category")

For categorical grouping, `observed=True` can be used when only categories actually present in the data should be returned.

## Grouping with missing keys

Missing group keys require explicit consideration.

The default GroupBy behavior does not normally create a separate missing-key group.

The script demonstrates:

    frame.groupby("department", dropna=False)["revenue"].sum()

This explicitly retains missing grouping keys.

Whether missing keys should form a group depends on the analytical question.

## Merge

### Definition

`merge()` combines DataFrames by matching key columns.

It is conceptually similar to relational database joins.

A typical example is an order table containing `customer_id` and a customer table containing customer information.

The order table can be enriched with customer attributes by matching the identifier.

## Inner merge

An inner merge retains records with matching keys in both datasets.

    left.merge(right, on="key", how="inner")

If a key exists only on the left or only on the right, that record does not appear in the result.

Inner merges are appropriate when unmatched records should be excluded.

## Left merge

A left merge retains every record from the left DataFrame.

    left.merge(right, on="key", how="left")

Matching information from the right DataFrame is added where available.

If no right-side record exists, the newly added columns contain missing values.

Left merges are common when enriching a transaction table with lookup information.

## Right merge

A right merge preserves all records from the right DataFrame.

    left.merge(right, on="key", how="right")

It is logically equivalent to reversing the two DataFrames and performing a left merge.

## Outer merge

An outer merge retains keys from both sides.

    left.merge(right, on="key", how="outer")

Records without a match receive missing values for columns originating from the other DataFrame.

Outer merges are useful for reconciliation and data-quality audits.

## Merge indicator

The `indicator=True` option creates a column identifying the source of each resulting row.

The script uses:

    merged = left.merge(
        right,
        on="key",
        how="outer",
        indicator=True
    )

Typical indicator values describe whether a row originated from the left dataset, right dataset, or both.

This makes it easy to identify unmatched records.

## Merge validation

One of the most important production techniques demonstrated in the script is merge validation.

The following statement asserts that the right-side key is unique for a many-to-one relationship:

    sales.merge(
        customers,
        on="customer",
        how="left",
        validate="many_to_one"
    )

Validation can detect accidental many-to-many relationships.

Useful relationship expectations include:

- `one_to_one`
- `one_to_many`
- `many_to_one`
- `many_to_many`

When a lookup table is supposed to contain one record per customer, `many_to_one` is often the appropriate validation.

## Why many-to-many merges are dangerous

Suppose the left table contains two rows for customer A and the right table contains three rows for customer A.

A many-to-many merge can produce six rows for that customer.

This is not necessarily a pandas error. It is a mathematically valid Cartesian combination of matching records.

The problem is analytical: revenue, quantities, or transaction counts can become artificially inflated.

The script explicitly demonstrates this row multiplication.

## Merge keys with different names

If the key columns have different names, use `left_on` and `right_on`.

For example:

    left.merge(
        right,
        left_on="customer",
        right_on="customer_name",
        how="left"
    )

This avoids renaming columns solely for the merge.

## Multiple merge keys

A merge can use several columns:

    left.merge(
        right,
        on=["region", "category"],
        how="left"
    )

The combination of the keys determines the match.

This is important when neither column alone uniquely identifies the desired relationship.

## Overlapping column names

If both DataFrames contain columns with the same name apart from the merge key, pandas adds suffixes.

The script explicitly controls them with:

    suffixes=("_old", "_new")

Explicit suffixes make the meaning of overlapping attributes easier to understand.

## Cross merge

A cross merge creates a Cartesian product.

For two datasets with two rows each, the result contains four combinations.

This can be useful for generating all combinations of dimensions, but it can also become extremely large.

Cross merges should therefore be used deliberately.

## `merge_asof`

`merge_asof()` is designed for ordered approximate-time matching.

It can associate an observation with the nearest earlier observation in another time-indexed dataset.

The script uses trade and quote timestamps as an example.

This is useful in areas such as:

- financial market data
- event streams
- sensor measurements
- time-dependent logs

Both datasets must be appropriately sorted according to the matching requirements.

## Join

`join()` provides convenient DataFrame combination, especially when the matching key is an index.

For example:

    sales_indexed.join(
        customer_indexed,
        how="left"
    )

The main conceptual difference is that `join()` is often naturally expressed around indexes, while `merge()` is designed for general key-based relational operations.

Neither operation is universally better. The appropriate choice depends on the structure of the data.

## Concatenate

`concat()` combines objects along an axis.

### Vertical concatenation

When DataFrames represent the same kind of records, they can be stacked vertically:

    pd.concat(
        [january_data, february_data],
        ignore_index=True
    )

This is useful when separate files or batches contain the same columns.

### Horizontal concatenation

DataFrames can also be combined column-wise:

    pd.concat(
        [left, right],
        axis=1
    )

Horizontal concatenation follows index alignment.

It does not automatically match rows using an arbitrary business key. For key-based matching, `merge()` is usually more appropriate.

## Concatenating different schemas

If DataFrames contain different columns, pandas forms the union of the columns by default.

Values missing from a particular input become missing in the result.

The script also demonstrates:

    join="inner"

which keeps only columns shared by all participating objects.

## Concatenation with keys

`keys` can preserve the source of each concatenated DataFrame.

For example:

    pd.concat(
        [north_data, south_data],
        keys=["North", "South"]
    )

This creates a hierarchical index containing source information.

## Pivot tables

A pivot table summarizes data across dimensions.

The script demonstrates:

    pd.pivot_table(
        sales,
        values="net_revenue",
        index="region",
        columns="category",
        aggfunc="sum",
        fill_value=0
    )

This produces a matrix where:

- rows represent regions
- columns represent categories
- cells contain aggregated revenue

Pivot tables are especially useful for analytical reporting.

## Pivot table aggregation

`pivot_table()` supports different aggregation functions.

Common choices include:

- `sum`
- `mean`
- `count`
- `min`
- `max`
- custom functions

Multiple aggregations can also be requested.

The script demonstrates a pivot containing sum, mean, and count.

## Pivot table totals

The `margins=True` option creates totals.

For example:

    pd.pivot_table(
        data,
        values="revenue",
        index="region",
        columns="category",
        aggfunc="sum",
        margins=True
    )

This adds row and column totals to the analytical matrix.

## `pivot()` versus `pivot_table()`

These methods are related but not interchangeable.

`pivot()` reshapes data without aggregation and requires each index-column combination to be unique.

For example:

    data.pivot(
        index="region",
        columns="category",
        values="revenue"
    )

If duplicate combinations exist, `pivot()` raises an error.

`pivot_table()` can handle duplicate combinations because it applies an aggregation function.

Therefore:

- use `pivot()` when combinations are unique
- use `pivot_table()` when aggregation is required

## Melt

`melt()` converts wide-form data into long-form data.

For example, a table containing:

- region
- January
- February
- March

can become a table containing:

- region
- month
- revenue

The script uses:

    wide_data.melt(
        id_vars="region",
        var_name="month",
        value_name="revenue"
    )

Long-form data is often convenient for analysis because one variable occupies one column and each observation occupies one row.

## Stack and unstack

`stack()` and `unstack()` reshape data involving indexes.

`unstack()` can move an index level into columns.

`stack()` can move columns into an index level.

The script demonstrates converting a multi-level grouped result into a matrix and then restoring it.

These operations become particularly important when working with MultiIndex structures.

## MultiIndex

A MultiIndex contains multiple levels of row or column labels.

For example:

    sales.groupby(["region", "category"])["net_revenue"].sum()

can produce an index containing both region and category.

MultiIndex structures are powerful for hierarchical analysis but can be less convenient for downstream systems that expect ordinary columns.

`reset_index()` can convert index levels into normal columns.

## Flattening MultiIndex columns

Pivot tables with several values or aggregation functions can produce MultiIndex columns.

These can be flattened into ordinary strings.

The script demonstrates:

    frame.columns = [
        "_".join(str(part) for part in column).strip("_")
        for column in frame.columns.to_flat_index()
    ]

Flattening is useful when exporting data or passing it to systems that do not handle hierarchical column labels naturally.

## Missing data

Missing data occurs when an observation has no available value.

Pandas can represent missing values using several internal representations, including `NaN`, `None`, and `pd.NA`, depending on the data type.

Missing values should not automatically be interpreted as zero.

A missing revenue value and a revenue of zero have different meanings.

## Detecting missing values

The main detection methods are:

    frame.isna()

and:

    frame.notna()

To count missing values per column:

    frame.isna().sum()

To calculate missing percentages:

    frame.isna().mean() * 100

The script builds a complete missing-data audit containing counts, percentages, and data types.

## Removing missing data

`dropna()` removes rows or columns containing missing values according to specified rules.

For example:

    frame.dropna()

removes rows containing at least one missing value.

Using:

    frame.dropna(subset=["customer", "age"])

requires only selected columns to be non-missing.

The `thresh` argument can require a minimum number of non-missing values.

Removing missing data is appropriate only when the resulting loss of observations is acceptable.

## Filling missing values

`fillna()` replaces missing values.

A constant can be used:

    frame["city"] = frame["city"].fillna("Unknown")

For numerical variables, a statistical value can be used:

    frame["income"] = frame["income"].fillna(
        frame["income"].median()
    )

The choice should be based on the meaning and distribution of the variable.

## Mean versus median

Mean imputation can be strongly affected by extreme values.

Median imputation is more robust to outliers.

For example, if most incomes are between 50,000 and 100,000 but one observation is 1,000,000, the mean can become substantially larger than the typical observation.

Median is often preferable for heavily skewed numerical variables, although neither method is universally correct.

## Forward fill and backward fill

Forward fill carries the last known value forward.

    series.ffill()

Backward fill uses the next available value.

    series.bfill()

These techniques can be appropriate for ordered time-series data when the interpretation supports carrying information across missing observations.

They should not be used automatically for arbitrary datasets.

## Limiting propagation

Forward or backward filling can be limited.

For example:

    series.ffill(limit=1)

allows at most one consecutive missing value to be filled from the previous observation.

This prevents a known value from being propagated indefinitely across long missing intervals.

## Interpolation

Interpolation estimates missing numerical values between known observations.

The script demonstrates linear interpolation:

    series.interpolate(method="linear")

For a sequence such as:

    10, missing, missing, 40

linear interpolation estimates intermediate values along the straight line between 10 and 40.

Interpolation is meaningful only when the underlying variable has an appropriate ordered or continuous interpretation.

## Group-wise imputation

A single global median may be inappropriate when different groups have very different distributions.

The script uses group-wise median imputation:

    frame.groupby("department")["salary"].transform(
        lambda group: group.fillna(group.median())
    )

This preserves differences between departments.

Group-wise imputation should still be evaluated carefully because the grouping variable may itself contain missing values and because imputation can affect downstream statistical analysis.

## Missingness as information

Sometimes the fact that a value is missing is itself meaningful.

For example, missing income information may correlate with a customer's behavior.

The script creates a missingness indicator:

    frame["income_missing"] = (
        frame["income"].isna().astype(int)
    )

This produces a binary feature where:

- `1` means the value is missing
- `0` means the value is present

This technique is often useful in analytical and predictive workflows.

## Standardizing missing-value representations

Real datasets may contain placeholders such as:

- `-1`
- `"NA"`
- `"N/A"`
- `"Unknown"`
- empty strings

These are not necessarily recognized as missing automatically.

The script converts selected sentinel values to `pd.NA`.

This is important because incorrect representations can cause missing values to be counted as real categories or numbers.

## Missing values and arithmetic

Pandas reduction functions often ignore missing values by default.

For example:

    series.sum()

typically skips missing values.

Setting:

    series.sum(skipna=False)

causes the result to reflect the presence of missing data.

The appropriate behavior depends on the analytical meaning of the calculation.

## Missing values introduced by merges

A left merge often creates missing values when a lookup record is unavailable.

For example, an order may contain a customer identifier that does not exist in the customer dimension.

The merged customer attributes will be missing.

This should not automatically be replaced with an arbitrary value. First determine whether the missing lookup indicates:

- a legitimate unknown entity
- an incomplete dimension table
- an invalid foreign key
- a data integration problem

## Zero versus missing

Zero and missing are fundamentally different.

Zero means the quantity is known to be zero.

Missing means the value is unavailable, unknown, or not recorded.

Replacing missing values with zero without domain justification can materially distort:

- averages
- totals
- rates
- distributions
- financial metrics

## Merge auditing

A reliable merge workflow should often include:

- checking key existence
- checking key uniqueness
- validating relationship cardinality
- comparing row counts
- inspecting unmatched records
- examining missing values introduced by the merge

The script demonstrates the use of `indicator=True` to identify unmatched keys.

This is particularly useful when building data pipelines from multiple source systems.

## Duplicate keys

Duplicate keys in a lookup table are one of the most important causes of accidental row multiplication.

Before performing a many-to-one enrichment merge, inspect the key:

    lookup["customer_id"].duplicated().sum()

Duplicate records should be investigated before proceeding.

If duplicates are expected, the relationship should be modeled explicitly rather than assuming that the lookup table contains a single record per key.

## Grain of a dataset

The grain of a dataset describes what one row represents.

Examples include:

- one row per order
- one row per customer
- one row per product
- one row per order line
- one row per customer-month
- one row per region-category pair

Understanding grain is essential before using `merge`, `groupby`, or `concat`.

If one table is at customer level and another is at transaction level, merging them may be correct.

If two tables are both transaction-level but contain different transaction records, joining them may unintentionally multiply observations.

## Relational interpretation

Pandas merges correspond closely to relational database concepts.

An inner merge is similar to an inner join.

A left merge preserves the left relation.

An outer merge preserves keys from both relations.

A cross merge creates a Cartesian product.

This connection is useful because many data-engineering and analytical systems use relational concepts even when pandas is the immediate tool.

## Anti-join analysis

Pandas does not require a special anti-join function for common audit cases.

An anti-join can be constructed using a left merge with an indicator and filtering for left-only rows.

This is useful for questions such as:

- Which orders have unknown customers?
- Which products are absent from the master table?
- Which expected records did not arrive?
- Which IDs exist in one system but not another?

The script implements this pattern explicitly.

## Semi-join analysis

A semi-join identifies records from one table that have at least one match in another table.

An inner merge followed by appropriate deduplication can express this pattern.

Semi-joins are useful when the matching table is being used as a filter rather than as a source of additional attributes.

## Null merge keys

Missing keys require special attention.

A missing identifier usually indicates that the relational relationship is incomplete.

Even when pandas can perform an operation involving missing keys, analysts should determine whether matching records through missing identifiers is logically valid for the particular dataset.

Identifiers should generally be validated before relying on them as relational keys.

## Pre-aggregation before merging

Pre-aggregation can reduce the size of a dataset before enrichment.

For example, transaction-level data can first be aggregated to customer level:

    customer_totals = (
        orders.groupby("customer_id", as_index=False)
        .agg(total_revenue=("revenue", "sum"))
    )

The resulting customer-level data can then be merged with a customer dimension.

This can:

- reduce computational work
- reduce memory usage
- clarify data grain
- reduce opportunities for accidental row multiplication

Pre-aggregation must only be used when the analytical question allows the detail to be discarded.

## Time-based GroupBy operations

Datetime columns can be grouped by derived components such as:

- month
- quarter
- year
- day of week
- hour

The script uses:

    sales["month"] = sales["date"].dt.to_period("M")

and then groups by month.

This is useful for sales trends, operational monitoring, and financial analysis.

## Resampling

Time-series data can be resampled after setting a datetime index.

For example:

    frame.set_index("date")["revenue"].resample("W").sum()

can calculate weekly revenue.

Resampling differs from ordinary grouping because it is specifically designed around time intervals and frequency rules.

## Group-wise rolling calculations

Rolling calculations can be performed within groups.

The script calculates a rolling customer-level average.

This is useful for:

- moving averages
- recent customer spending
- rolling operational metrics
- trend analysis

The grouping ensures that observations from different customers are not mixed.

## Data type consistency

Data manipulation becomes unreliable when columns contain inconsistent types.

For example:

    ["10", "20", "unknown", "40"]

contains numeric-looking strings and a non-numeric value.

The script uses:

    pd.to_numeric(
        series,
        errors="coerce"
    )

to convert valid values and represent invalid values as missing.

Type conversion should be performed deliberately because coercion can create missing values.

## Category normalization

Categorical values may differ because of:

- capitalization
- leading or trailing whitespace
- spelling differences
- inconsistent abbreviations

The script normalizes region names using string operations.

For example:

    frame["region"].str.strip().str.title()

Standardization should occur before grouping or joining when textual categories are expected to represent the same logical value.

## Index alignment

Pandas operations frequently align objects by index labels rather than physical row position.

For example, adding two Series with the same labels in different orders still matches values by label.

This is powerful but requires awareness of index state.

Unexpected indexes can lead to missing values or unexpected alignment.

When positional behavior is intended, indexes should be reset or otherwise handled explicitly.

## Performance considerations

Data manipulation performance depends on:

- number of rows
- number of columns
- data types
- number of groups
- cardinality of keys
- size of intermediate results
- operation type
- memory availability

Important practices include:

- select only required columns
- avoid unnecessary copies
- prefer vectorized operations
- prefer built-in GroupBy aggregations
- avoid unnecessary `apply()`
- use appropriate categorical types for repeated low-cardinality strings
- validate merges before expensive downstream processing
- pre-aggregate when the analytical grain permits it

The script includes a simple comparison between Python-level iteration and vectorized arithmetic.

## Vectorization

Vectorization means expressing operations over complete arrays or Series rather than manually iterating over individual values in Python.

For example:

    frame["revenue"] = frame["quantity"] * frame["unit_price"]

is preferable to a Python loop for ordinary column arithmetic.

Vectorized operations typically reduce Python interpreter overhead and are a fundamental pandas performance pattern.

## Memory considerations

Repeated string values can consume significant memory.

The `category` dtype can reduce memory consumption for columns with relatively few unique values compared with the total number of rows.

The script compares memory usage before and after converting repeated categorical values to the categorical dtype.

Categorical conversion is most useful when the number of unique values is relatively small.

## Safe modification of filtered data

When a filtered DataFrame is intended to become an independent object, the script uses:

    filtered = frame.loc[condition].copy()

This makes the intended ownership of the resulting DataFrame explicit.

It also helps avoid ambiguous chained-assignment behavior.

## Data-quality assertions

Analytical pipelines should verify assumptions rather than silently accepting invalid data.

The script checks conditions such as:

- required identifiers are present
- quantities are positive
- prices are non-negative
- keys are unique where expected
- row counts remain stable after many-to-one enrichment

Assertions are especially useful during development and testing.

## Production merge validation

A production-oriented merge should answer four questions:

1. What is the key?
2. What is the expected relationship between the two datasets?
3. How many rows should exist after the merge?
4. What should happen when a key has no match?

For example, if every order can have only one customer record, a many-to-one relationship should be enforced.

The script uses:

    validate="many_to_one"

to make that assumption executable.

## Reconciliation

A merge should sometimes be reconciled using row counts and key counts.

For a many-to-one enrichment, the number of rows in the result should normally remain equal to the number of rows in the original transaction table.

Unexpected growth indicates that the lookup key is not unique or that the relationship assumption is incorrect.

Unexpected reduction can indicate an inappropriate inner merge or another filtering effect.

## Common mistakes

### Using an inner merge when all source records must be preserved

An inner merge removes unmatched records.

If every transaction must remain in the result, a left merge is generally more appropriate.

### Ignoring duplicate lookup keys

Duplicate lookup keys can create many-to-many row multiplication.

Validate lookup keys before enrichment.

### Confusing `count()` and `size()`

`count()` excludes missing values in the selected column.

`size()` counts rows.

### Using `pivot()` with duplicate combinations

`pivot()` requires unique index-column combinations.

Use `pivot_table()` when duplicate combinations need aggregation.

### Treating horizontal `concat()` as a key-based join

Horizontal concatenation aligns indexes.

It does not perform an arbitrary key match.

Use `merge()` when records should be matched by business keys.

### Replacing every missing value with zero

Zero is not equivalent to missing.

Imputation should reflect the domain meaning of the variable.

### Ignoring data grain

Combining datasets without understanding what one row represents is a major source of incorrect analytical results.

### Using `apply()` unnecessarily

`apply()` is flexible but can introduce Python-level overhead.

Use vectorized methods or built-in aggregation functions when they express the required calculation.

### Failing to inspect merge-induced missing values

A successful merge does not necessarily mean that every record matched.

Audit unmatched keys and missing attributes.

### Assuming column order defines identity

Pandas commonly uses labels and indexes for alignment.

Explicitly manage indexes when positional behavior is required.

## Important distinctions

| Operation | Main purpose |
|---|---|
| `groupby()` | Split data into groups for aggregation, transformation, filtering, or custom processing |
| `merge()` | Match records using one or more keys |
| `join()` | Combine data, commonly using indexes |
| `concat()` | Stack or align multiple pandas objects |
| `pivot()` | Reshape unique combinations without aggregation |
| `pivot_table()` | Reshape and aggregate data |
| `melt()` | Convert wide data into long form |
| `stack()` | Move columns into index levels |
| `unstack()` | Move index levels into columns |
| `fillna()` | Replace missing values |
| `dropna()` | Remove missing observations |
| `transform()` | Calculate group-level results while preserving original row count |

## GroupBy versus pivot table

Both `groupby()` and `pivot_table()` can produce grouped summaries.

GroupBy is generally more flexible for analytical pipelines because it integrates naturally with filtering, transformation, ranking, cumulative operations, and custom processing.

A pivot table is particularly convenient when the desired result is a two-dimensional matrix with row dimensions and column dimensions.

For example, a GroupBy result might contain:

    region    category    revenue
    North     Laptop      100000
    North     Phone        50000
    South     Laptop       80000

A pivot table can reshape this into a matrix with regions as rows and categories as columns.

## Merge versus join

Both operations combine DataFrames, but their natural use cases differ.

`merge()` is the general-purpose key-based relational operation.

`join()` is especially convenient when the matching information is already represented through indexes.

The choice should be based on clarity and the structure of the data rather than on a belief that one operation is universally superior.

## Merge versus concat

`merge()` asks:

Which records correspond to each other based on a key?

`concat()` asks:

How should these separate objects be stacked or aligned?

For example, combining January and February transaction files with identical schemas is usually a concatenation problem.

Adding customer attributes to those transactions is a merge problem.

## Practical analytical workflow

The script builds an end-to-end workflow:

1. Start with transaction data.
2. Calculate revenue.
3. Validate primary keys.
4. Merge customer attributes.
5. Merge product attributes.
6. Validate merge cardinality.
7. Check row counts.
8. Aggregate by region and category.
9. Calculate group-level metrics with `transform`.
10. Create a pivot table for reporting.
11. Audit missing and unmatched values.

This workflow demonstrates how individual pandas operations become components of a larger data-processing pipeline.

## Security and data-integrity considerations

Data manipulation is primarily an analytical concern, but production data pipelines also require attention to data integrity.

Important practices include:

- validate identifiers before joins
- avoid trusting external lookup data without checking uniqueness
- preserve raw data separately from transformed data
- explicitly handle unexpected missing values
- validate data types
- avoid silently coercing invalid data
- reconcile row counts after critical transformations
- document assumptions about keys and grain
- prevent accidental duplication of financial or transactional records

In financial, customer, or operational datasets, an incorrect merge can be more damaging than a visible processing failure because the resulting data may appear valid while containing inflated or duplicated metrics.

## Implementation considerations

A robust pandas pipeline should make its assumptions visible in the code.

For example:

    validate="many_to_one"

communicates the expected relationship between datasets.

Similarly:

    indicator=True

makes unmatched records auditable.

Using named aggregations communicates the meaning of each metric:

    .agg(
        total_revenue=("revenue", "sum"),
        order_count=("order_id", "count")
    )

These patterns make analytical code easier to inspect and test.

## Testing considerations

Important transformations should be tested against known invariants.

Examples include:

- primary keys remain unique
- quantities remain non-negative
- revenue calculations reconcile
- many-to-one merges do not increase row count
- expected columns exist
- required identifiers are not missing
- totals before and after an enrichment remain consistent

The script demonstrates assertions for these conditions.

Testing data manipulation is especially important because many errors do not produce Python exceptions. A technically valid DataFrame can still contain analytically incorrect results.

## Edge cases covered by the script

The Python script explicitly demonstrates:

- empty DataFrames
- all-missing Series
- zero versus missing values
- missing GroupBy keys
- duplicate merge keys
- many-to-many row multiplication
- unmatched foreign keys
- missing values created by joins
- different column schemas during concatenation
- duplicate combinations in `pivot()`
- missing values introduced through type conversion
- categorical groups that are not observed
- index alignment
- time-based grouping
- limited forward filling
- group-wise imputation

These cases are important because real datasets rarely satisfy the assumptions of idealized examples.

## Real-world applications

The techniques in the script are applicable to:

- sales analysis
- customer analytics
- financial reporting
- inventory analysis
- business intelligence
- marketing analysis
- operational reporting
- transaction processing
- data cleaning
- ETL pipelines
- customer segmentation
- performance measurement
- time-series analysis
- reconciliation between systems

For example, a retail organization could use `merge()` to enrich orders with customer and product information, `groupby()` to calculate regional performance, `transform()` to calculate each order's contribution to regional revenue, and `pivot_table()` to produce a reporting matrix.

Missing-data analysis can then identify incomplete customer or product records, while merge validation protects the analysis against accidental row multiplication.

## Core principles demonstrated

The script establishes several practical rules for pandas data manipulation:

- Use `groupby()` for split-apply-combine analysis.
- Use `transform()` when group-level information must remain aligned with individual rows.
- Use `merge()` for key-based relational matching.
- Use `join()` when index-based combination is natural.
- Use `concat()` for stacking or axis-based alignment.
- Use `pivot_table()` for aggregated matrix-style reporting.
- Use `pivot()` only when the relevant combinations are unique.
- Use `melt()` to convert wide analytical structures into long form.
- Validate lookup keys before performing enrichment merges.
- Use merge validation to enforce expected relationship cardinality.
- Audit unmatched records after important merges.
- Understand the grain of every dataset before combining it.
- Treat missing values according to their actual meaning.
- Distinguish missing values from legitimate zero values.
- Prefer vectorized operations for large datasets.
- Use built-in pandas aggregations when they express the required operation.
- Preserve data integrity through row-count and metric reconciliation.
- Make data-quality assumptions explicit in implementation code.
