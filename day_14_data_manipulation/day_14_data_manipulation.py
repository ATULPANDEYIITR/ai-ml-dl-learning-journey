"""
DATA MANIPULATION WITH PANDAS
=============================

A standalone study script covering:

- GroupBy
- merge
- join
- concatenate
- pivot tables
- missing data

The examples progress from beginner concepts to advanced data-manipulation
patterns, including validation, multi-level indexes, aggregation, reshaping,
missing-value strategies, edge cases, debugging, performance, and practical
data-analysis workflows.

Dependency:
    pandas

Install:
    pip install pandas
"""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd


# ============================================================================
# 1. BASIC SETUP
# ============================================================================

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.2f}")


def section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def show(title: str, value) -> None:
    """Print a titled object."""
    print(f"\n--- {title} ---")
    print(value)


# ============================================================================
# 2. FUNDAMENTAL DATASET
# ============================================================================

section("1. Creating the dataset")

sales = pd.DataFrame(
    {
        "order_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "date": pd.to_datetime(
            [
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
                "2026-01-08",
                "2026-01-09",
                "2026-01-10",
                "2026-01-12",
                "2026-01-13",
            ]
        ),
        "customer": [
            "Aarav",
            "Diya",
            "Aarav",
            "Kabir",
            "Diya",
            "Meera",
            "Kabir",
            "Aarav",
        ],
        "region": [
            "North",
            "South",
            "North",
            "West",
            "South",
            "East",
            "West",
            "North",
        ],
        "category": [
            "Laptop",
            "Phone",
            "Phone",
            "Laptop",
            "Tablet",
            "Laptop",
            "Phone",
            "Tablet",
        ],
        "quantity": [1, 2, 1, 2, 3, 1, 2, 2],
        "unit_price": [75000, 30000, 30000, 68000, 25000, 72000, 32000, 22000],
        "salesperson": [
            "Ravi",
            "Neha",
            "Ravi",
            "Arjun",
            "Neha",
            "Sara",
            "Arjun",
            "Ravi",
        ],
    }
)

sales["revenue"] = sales["quantity"] * sales["unit_price"]

show("Original sales DataFrame", sales)
show("Data types", sales.dtypes)
show("Shape", sales.shape)


# ============================================================================
# 3. SELECTING COLUMNS AND CREATING DERIVED DATA
# ============================================================================

section("2. Basic data manipulation")

show("Single column", sales["revenue"])
show("Multiple columns", sales[["customer", "category", "revenue"]])

sales["discount_rate"] = [0.05, 0.10, 0.00, 0.15, 0.05, 0.10, 0.00, 0.05]
sales["discount"] = sales["revenue"] * sales["discount_rate"]
sales["net_revenue"] = sales["revenue"] - sales["discount"]

show(
    "Derived columns",
    sales[
        [
            "order_id",
            "revenue",
            "discount_rate",
            "discount",
            "net_revenue",
        ]
    ],
)


# ============================================================================
# 4. FILTERING BEFORE GROUPING
# ============================================================================

section("3. Filtering")

high_value_orders = sales.loc[sales["net_revenue"] >= 60000]
show("Orders with net revenue >= 60,000", high_value_orders)

north_orders = sales.loc[sales["region"].eq("North")]
show("North region orders", north_orders)

laptop_or_phone = sales.loc[sales["category"].isin(["Laptop", "Phone"])]
show("Laptop or Phone orders", laptop_or_phone)

recent_orders = sales.loc[sales["date"] >= "2026-01-09"]
show("Orders from January 9 onward", recent_orders)


# ============================================================================
# 5. GROUPBY FUNDAMENTALS
# ============================================================================

section("4. GroupBy fundamentals")

"""
GroupBy follows a split-apply-combine pattern:

1. Split rows into groups.
2. Apply an aggregation or transformation.
3. Combine the results.

Example:
    sales.groupby("region")["net_revenue"].sum()

This calculates total net revenue independently for each region.
"""

region_groups = sales.groupby("region")

show("GroupBy object", region_groups)
show("Total revenue by region", region_groups["net_revenue"].sum())
show("Average revenue by region", region_groups["net_revenue"].mean())
show("Number of orders by region", region_groups["order_id"].count())


# ============================================================================
# 6. GROUPBY SIZE, COUNT, NUNIQUE
# ============================================================================

section("5. Group sizes, counts, and unique values")

show("Number of rows per region", sales.groupby("region").size())
show("Number of orders per region", sales.groupby("region")["order_id"].count())
show(
    "Unique customers per region",
    sales.groupby("region")["customer"].nunique(),
)

show(
    "Unique categories per region",
    sales.groupby("region")["category"].nunique(),
)


# ============================================================================
# 7. GROUPBY WITH MULTIPLE COLUMNS
# ============================================================================

section("6. GroupBy using multiple grouping keys")

region_category = (
    sales.groupby(["region", "category"])["net_revenue"]
    .sum()
    .sort_values(ascending=False)
)

show("Revenue by region and category", region_category)

region_customer = (
    sales.groupby(["region", "customer"])["net_revenue"]
    .sum()
)

show("Revenue by region and customer", region_customer)


# ============================================================================
# 8. AS_INDEX AND RESET_INDEX
# ============================================================================

section("7. as_index and reset_index")

grouped_index = sales.groupby("region", as_index=True)["net_revenue"].sum()
show("Default GroupBy result with region as index", grouped_index)

grouped_columns = sales.groupby("region", as_index=False)["net_revenue"].sum()
show("GroupBy result with region retained as a column", grouped_columns)

reset_result = (
    sales.groupby(["region", "category"])["net_revenue"]
    .sum()
    .reset_index()
)

show("Multi-column grouping converted back to regular columns", reset_result)


# ============================================================================
# 9. AGGREGATION
# ============================================================================

section("8. Multiple aggregations")

basic_aggregations = sales.groupby("region")["net_revenue"].agg(
    ["sum", "mean", "min", "max", "count"]
)

show("Multiple built-in aggregations", basic_aggregations)


# ============================================================================
# 10. NAMED AGGREGATION
# ============================================================================

section("9. Named aggregation")

region_summary = sales.groupby("region").agg(
    total_revenue=("net_revenue", "sum"),
    average_order_value=("net_revenue", "mean"),
    largest_order=("net_revenue", "max"),
    order_count=("order_id", "count"),
    customer_count=("customer", "nunique"),
)

show("Professional summary using named aggregation", region_summary)


# ============================================================================
# 11. CUSTOM AGGREGATION
# ============================================================================

section("10. Custom aggregation functions")

def revenue_range(series: pd.Series) -> float:
    """Difference between maximum and minimum values."""
    return series.max() - series.min()


custom_summary = sales.groupby("category").agg(
    total_revenue=("net_revenue", "sum"),
    average_revenue=("net_revenue", "mean"),
    revenue_range=("net_revenue", revenue_range),
)

show("Custom aggregation", custom_summary)


# ============================================================================
# 12. AGGREGATION WITH DIFFERENT FUNCTIONS PER COLUMN
# ============================================================================

section("11. Different aggregations for different columns")

mixed_summary = sales.groupby("category").agg(
    quantity=("quantity", "sum"),
    average_unit_price=("unit_price", "mean"),
    total_revenue=("net_revenue", "sum"),
    orders=("order_id", "count"),
)

show("Different aggregations by column", mixed_summary)


# ============================================================================
# 13. GROUPBY SORTING
# ============================================================================

section("12. Sorting grouped results")

sorted_regions = (
    sales.groupby("region", as_index=False)["net_revenue"]
    .sum()
    .sort_values("net_revenue", ascending=False)
)

show("Regions ranked by revenue", sorted_regions)


# ============================================================================
# 14. GROUPBY FILTER
# ============================================================================

section("13. Filtering complete groups")

large_region_groups = sales.groupby("region").filter(
    lambda group: group["net_revenue"].sum() >= 100000
)

show(
    "Rows belonging to regions whose total net revenue is at least 100,000",
    large_region_groups,
)


# ============================================================================
# 15. GROUPBY TRANSFORM
# ============================================================================

section("14. GroupBy transform")

"""
aggregate() reduces a group to one row per group.

transform() preserves the original number of rows.

This makes transform useful when a group-level statistic must be attached
back to every original observation.
"""

sales["region_total"] = sales.groupby("region")["net_revenue"].transform("sum")
sales["region_average"] = sales.groupby("region")["net_revenue"].transform("mean")

sales["share_of_region"] = (
    sales["net_revenue"] / sales["region_total"]
)

show(
    "Group totals and each order's share of regional revenue",
    sales[
        [
            "order_id",
            "region",
            "net_revenue",
            "region_total",
            "region_average",
            "share_of_region",
        ]
    ],
)


# ============================================================================
# 16. GROUPBY RANKING
# ============================================================================

section("15. Ranking within groups")

sales["regional_rank"] = (
    sales.groupby("region")["net_revenue"]
    .rank(method="dense", ascending=False)
)

show(
    "Revenue ranking inside each region",
    sales[
        ["order_id", "region", "net_revenue", "regional_rank"]
    ].sort_values(["region", "regional_rank"]),
)


# ============================================================================
# 17. GROUPBY SHIFT
# ============================================================================

section("16. GroupBy shift")

sales = sales.sort_values(["customer", "date"]).reset_index(drop=True)

sales["previous_customer_order"] = (
    sales.groupby("customer")["net_revenue"].shift(1)
)

sales["change_from_previous_order"] = (
    sales["net_revenue"] - sales["previous_customer_order"]
)

show(
    "Previous order and change within each customer",
    sales[
        [
            "customer",
            "date",
            "net_revenue",
            "previous_customer_order",
            "change_from_previous_order",
        ]
    ],
)


# ============================================================================
# 18. GROUPBY CUMULATIVE OPERATIONS
# ============================================================================

section("17. GroupBy cumulative calculations")

sales["customer_cumulative_revenue"] = (
    sales.groupby("customer")["net_revenue"].cumsum()
)

show(
    "Cumulative revenue per customer",
    sales[
        [
            "customer",
            "date",
            "net_revenue",
            "customer_cumulative_revenue",
        ]
    ],
)


# ============================================================================
# 19. GROUPBY APPLY
# ============================================================================

section("18. GroupBy apply")

def top_orders(group: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Return the top n orders from a group."""
    return group.nlargest(n, "net_revenue")


top_customer_orders = (
    sales.groupby("customer", group_keys=False)
    .apply(top_orders, n=2, include_groups=True)
)

show("Top two orders for each customer", top_customer_orders)


# ============================================================================
# 20. GROUPBY WITH BOOLEAN CONDITIONS
# ============================================================================

section("19. Conditional group statistics")

sales["high_value"] = sales["net_revenue"] >= 60000

high_value_rate = (
    sales.groupby("region")["high_value"]
    .mean()
    .mul(100)
)

show("Percentage of high-value orders by region", high_value_rate)


# ============================================================================
# 21. GROUPBY WITH CATEGORICAL DATA
# ============================================================================

section("20. Categorical grouping")

sales["category"] = sales["category"].astype("category")

categorical_summary = (
    sales.groupby("category", observed=True)["net_revenue"]
    .sum()
)

show("Grouping a categorical column", categorical_summary)


# ============================================================================
# 22. MERGE FUNDAMENTALS
# ============================================================================

section("21. Creating lookup DataFrames for merge")

customers = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir", "Meera", "Nisha"],
        "segment": ["Premium", "Standard", "Premium", "Standard", "Premium"],
        "city": ["Delhi", "Mumbai", "Pune", "Kolkata", "Bengaluru"],
    }
)

products = pd.DataFrame(
    {
        "category": ["Laptop", "Phone", "Tablet", "Monitor"],
        "department": ["Computers", "Mobile", "Computers", "Computers"],
        "tax_rate": [0.18, 0.18, 0.18, 0.18],
    }
)

show("Customers", customers)
show("Products", products)


# ============================================================================
# 23. INNER MERGE
# ============================================================================

section("22. Inner merge")

inner_customer_merge = sales.merge(
    customers,
    on="customer",
    how="inner",
)

show("Inner merge", inner_customer_merge)


# ============================================================================
# 24. LEFT MERGE
# ============================================================================

section("23. Left merge")

left_customer_merge = sales.merge(
    customers,
    on="customer",
    how="left",
)

show("Left merge", left_customer_merge)


# ============================================================================
# 25. RIGHT MERGE
# ============================================================================

section("24. Right merge")

right_customer_merge = sales.merge(
    customers,
    on="customer",
    how="right",
)

show("Right merge", right_customer_merge)


# ============================================================================
# 26. OUTER MERGE
# ============================================================================

section("25. Outer merge")

outer_customer_merge = sales.merge(
    customers,
    on="customer",
    how="outer",
    indicator=True,
)

show("Outer merge with indicator", outer_customer_merge)


# ============================================================================
# 27. MERGE INDICATOR
# ============================================================================

section("26. Detecting unmatched records")

unmatched_customers = outer_customer_merge.loc[
    outer_customer_merge["_merge"].eq("right_only")
]

show("Customers without sales", unmatched_customers)


# ============================================================================
# 28. MERGE VALIDATION
# ============================================================================

section("27. Merge validation")

validated_merge = sales.merge(
    customers,
    on="customer",
    how="left",
    validate="many_to_one",
)

show("Validated many-to-one merge", validated_merge)


# ============================================================================
# 29. WHY VALIDATION MATTERS
# ============================================================================

section("28. Detecting accidental many-to-many relationships")

bad_customers = pd.DataFrame(
    {
        "customer": ["Aarav", "Aarav", "Diya"],
        "segment": ["Premium", "VIP", "Standard"],
    }
)

try:
    sales.merge(
        bad_customers,
        on="customer",
        how="left",
        validate="many_to_one",
    )
except pd.errors.MergeError as error:
    print("Expected merge validation error:")
    print(error)


# ============================================================================
# 30. MERGING ON DIFFERENT COLUMN NAMES
# ============================================================================

section("29. Merge using left_on and right_on")

customer_codes = pd.DataFrame(
    {
        "customer_name": ["Aarav", "Diya", "Kabir", "Meera"],
        "customer_code": ["C01", "C02", "C03", "C04"],
    }
)

sales_with_codes = sales.merge(
    customer_codes,
    left_on="customer",
    right_on="customer_name",
    how="left",
)

show("Merge with differently named key columns", sales_with_codes)


# ============================================================================
# 31. MERGE MULTIPLE KEYS
# ============================================================================

section("30. Merge using multiple keys")

regional_targets = pd.DataFrame(
    {
        "region": ["North", "South", "West", "East"],
        "category": ["Laptop", "Phone", "Phone", "Laptop"],
        "target": [150000, 80000, 70000, 100000],
    }
)

multi_key_merge = sales.merge(
    regional_targets,
    on=["region", "category"],
    how="left",
)

show("Merge on region and category", multi_key_merge)


# ============================================================================
# 32. SUFFIXES
# ============================================================================

section("31. Handling overlapping column names")

customer_details = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir", "Meera"],
        "city": ["Delhi", "Mumbai", "Pune", "Kolkata"],
        "status": ["Active", "Active", "Inactive", "Active"],
    }
)

customer_status_update = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir", "Meera"],
        "status": ["VIP", "Regular", "Suspended", "VIP"],
    }
)

suffix_merge = customer_details.merge(
    customer_status_update,
    on="customer",
    how="left",
    suffixes=("_old", "_new"),
)

show("Overlapping columns with explicit suffixes", suffix_merge)


# ============================================================================
# 33. CROSS MERGE
# ============================================================================

section("32. Cross merge")

countries = pd.DataFrame({"country": ["India", "Japan"]})
channels = pd.DataFrame({"channel": ["Online", "Retail"]})

cross_merge = countries.merge(channels, how="cross")

show("Cartesian product using cross merge", cross_merge)


# ============================================================================
# 34. ASOF MERGE
# ============================================================================

section("33. merge_asof for nearest-time matching")

quotes = pd.DataFrame(
    {
        "time": pd.to_datetime(
            [
                "2026-01-01 09:00",
                "2026-01-01 09:05",
                "2026-01-01 09:10",
            ]
        ),
        "price": [100, 102, 101],
    }
)

trades = pd.DataFrame(
    {
        "time": pd.to_datetime(
            [
                "2026-01-01 09:02",
                "2026-01-01 09:07",
                "2026-01-01 09:11",
            ]
        ),
        "quantity": [10, 20, 15],
    }
)

asof_result = pd.merge_asof(
    trades.sort_values("time"),
    quotes.sort_values("time"),
    on="time",
)

show("Nearest earlier quote for each trade", asof_result)


# ============================================================================
# 35. JOIN FUNDAMENTALS
# ============================================================================

section("34. DataFrame join")

"""
join() is especially convenient when the matching key is already an index.

merge() is more general and often clearer when ordinary columns are keys.
"""

customer_indexed = customers.set_index("customer")

sales_indexed = sales.set_index("customer")

joined = sales_indexed.join(
    customer_indexed,
    how="left",
    rsuffix="_customer",
)

show("DataFrame.join using index", joined.reset_index())


# ============================================================================
# 36. JOIN TYPES
# ============================================================================

section("35. Join types")

left_join = sales_indexed.join(customer_indexed, how="left")
inner_join = sales_indexed.join(customer_indexed, how="inner")
outer_join = sales_indexed.join(customer_indexed, how="outer")

show("Left join", left_join.reset_index())
show("Inner join", inner_join.reset_index())
show("Outer join", outer_join.reset_index())


# ============================================================================
# 37. JOIN WITH MULTIPLE DATAFRAMES
# ============================================================================

section("36. Joining several indexed DataFrames")

segment_indexed = customers.set_index("customer")[["segment"]]
city_indexed = customers.set_index("customer")[["city"]]

multiple_join = sales_indexed.join(
    [segment_indexed, city_indexed],
    how="left",
)

show("Multiple DataFrames joined by index", multiple_join.reset_index())


# ============================================================================
# 38. CONCATENATE ROWS
# ============================================================================

section("37. Concatenating rows")

january_part_a = sales.iloc[:4].copy()
january_part_b = sales.iloc[4:].copy()

vertical_concat = pd.concat(
    [january_part_a, january_part_b],
    axis=0,
    ignore_index=True,
)

show("Vertical concatenation", vertical_concat)


# ============================================================================
# 39. CONCATENATE COLUMNS
# ============================================================================

section("38. Concatenating columns")

left_info = sales[["order_id", "customer"]].reset_index(drop=True)
right_info = sales[["category", "net_revenue"]].reset_index(drop=True)

horizontal_concat = pd.concat(
    [left_info, right_info],
    axis=1,
)

show("Horizontal concatenation", horizontal_concat)


# ============================================================================
# 40. CONCAT WITH KEYS
# ============================================================================

section("39. Concatenation with source keys")

north_part = sales.loc[sales["region"] == "North"].copy()
south_part = sales.loc[sales["region"] == "South"].copy()

keyed_concat = pd.concat(
    [north_part, south_part],
    keys=["North Data", "South Data"],
)

show("Concatenation with hierarchical source keys", keyed_concat)


# ============================================================================
# 41. CONCAT WITH DIFFERENT COLUMNS
# ============================================================================

section("40. Concatenating DataFrames with different columns")

part_one = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya"],
        "revenue": [50000, 60000],
    }
)

part_two = pd.DataFrame(
    {
        "customer": ["Kabir", "Meera"],
        "region": ["West", "East"],
    }
)

different_columns_concat = pd.concat(
    [part_one, part_two],
    ignore_index=True,
)

show("Different columns produce missing values", different_columns_concat)


# ============================================================================
# 42. CONCAT JOIN OPTIONS
# ============================================================================

section("41. Concatenation with inner column intersection")

inner_column_concat = pd.concat(
    [part_one, part_two],
    ignore_index=True,
    join="inner",
)

show("Only columns common to every DataFrame", inner_column_concat)


# ============================================================================
# 43. PIVOT TABLE FUNDAMENTALS
# ============================================================================

section("42. Pivot table fundamentals")

pivot_revenue = pd.pivot_table(
    sales,
    values="net_revenue",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
)

show("Revenue by region and category", pivot_revenue)


# ============================================================================
# 44. PIVOT TABLE WITH MULTIPLE VALUES
# ============================================================================

section("43. Pivot table with multiple measures")

pivot_multiple = pd.pivot_table(
    sales,
    values=["quantity", "net_revenue"],
    index="region",
    columns="category",
    aggfunc={
        "quantity": "sum",
        "net_revenue": "sum",
    },
    fill_value=0,
)

show("Multiple measures in a pivot table", pivot_multiple)


# ============================================================================
# 45. PIVOT TABLE MULTIPLE AGGREGATIONS
# ============================================================================

section("44. Multiple aggregation functions in a pivot table")

pivot_aggregations = pd.pivot_table(
    sales,
    values="net_revenue",
    index="region",
    columns="category",
    aggfunc=["sum", "mean", "count"],
    fill_value=0,
)

show("Sum, mean, and count by region and category", pivot_aggregations)


# ============================================================================
# 46. PIVOT TABLE MARGINS
# ============================================================================

section("45. Pivot table totals")

pivot_with_totals = pd.pivot_table(
    sales,
    values="net_revenue",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total",
)

show("Pivot table with row and column totals", pivot_with_totals)


# ============================================================================
# 47. PIVOT TABLE DROPNA
# ============================================================================

section("46. Pivot table missing combinations")

pivot_without_fill = pd.pivot_table(
    sales,
    values="net_revenue",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=None,
)

show("Pivot table where absent combinations remain missing", pivot_without_fill)


# ============================================================================
# 48. PIVOT METHOD
# ============================================================================

section("47. pivot versus pivot_table")

"""
pivot() reshapes data without performing aggregation.

It requires the index-column combination to be unique.

pivot_table() can aggregate duplicate combinations.
"""

unique_records = pd.DataFrame(
    {
        "region": ["North", "North", "South", "South"],
        "category": ["Laptop", "Phone", "Laptop", "Phone"],
        "revenue": [100, 200, 300, 400],
    }
)

simple_pivot = unique_records.pivot(
    index="region",
    columns="category",
    values="revenue",
)

show("pivot() with unique combinations", simple_pivot)

duplicate_records = pd.DataFrame(
    {
        "region": ["North", "North", "North"],
        "category": ["Laptop", "Laptop", "Phone"],
        "revenue": [100, 150, 200],
    }
)

try:
    duplicate_records.pivot(
        index="region",
        columns="category",
        values="revenue",
    )
except ValueError as error:
    print("Expected pivot() error with duplicate keys:")
    print(error)

duplicate_pivot_table = duplicate_records.pivot_table(
    index="region",
    columns="category",
    values="revenue",
    aggfunc="sum",
)

show("pivot_table() handles duplicate combinations by aggregation", duplicate_pivot_table)


# ============================================================================
# 49. MELT
# ============================================================================

section("48. Melt: converting wide data to long data")

wide_data = pd.DataFrame(
    {
        "region": ["North", "South"],
        "January": [100, 200],
        "February": [120, 220],
        "March": [150, 250],
    }
)

long_data = wide_data.melt(
    id_vars="region",
    var_name="month",
    value_name="revenue",
)

show("Wide data", wide_data)
show("Long data using melt()", long_data)


# ============================================================================
# 50. STACK AND UNSTACK
# ============================================================================

section("49. Stack and unstack")

multi_region = (
    sales.groupby(["region", "category"])["net_revenue"]
    .sum()
)

show("MultiIndex grouped result", multi_region)

unstacked = multi_region.unstack(fill_value=0)
restacked = unstacked.stack()

show("Unstacked result", unstacked)
show("Restacked result", restacked)


# ============================================================================
# 51. MISSING DATA FUNDAMENTALS
# ============================================================================

section("50. Creating a DataFrame containing missing data")

missing = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir", "Meera", None],
        "age": [28, None, 35, None, 31],
        "income": [80000, 65000, None, 90000, None],
        "city": ["Delhi", "Mumbai", None, "Kolkata", "Delhi"],
        "score": [90, 85, None, 95, None],
    }
)

show("Data with missing values", missing)


# ============================================================================
# 52. ISNA AND NOTNA
# ============================================================================

section("51. Detecting missing values")

show("Boolean mask of missing values", missing.isna())
show("Missing values by column", missing.isna().sum())
show("Non-missing values by column", missing.notna().sum())
show(
    "Percentage missing by column",
    missing.isna().mean().mul(100).round(2),
)


# ============================================================================
# 53. TOTAL MISSING VALUES
# ============================================================================

section("52. Total missing-value count")

total_missing = missing.isna().sum().sum()

print(f"Total missing cells: {total_missing}")


# ============================================================================
# 54. DROPNA
# ============================================================================

section("53. Removing missing observations")

show(
    "Drop rows containing at least one missing value",
    missing.dropna(),
)

show(
    "Drop rows where all values are missing",
    missing.dropna(how="all"),
)

show(
    "Drop rows requiring customer and age to be present",
    missing.dropna(subset=["customer", "age"]),
)


# ============================================================================
# 55. FILLNA WITH CONSTANTS
# ============================================================================

section("54. Filling missing values with constants")

filled_constant = missing.copy()

filled_constant["city"] = filled_constant["city"].fillna("Unknown")
filled_constant["score"] = filled_constant["score"].fillna(0)

show("Constant replacement", filled_constant)


# ============================================================================
# 56. FILLNA WITH STATISTICS
# ============================================================================

section("55. Filling missing numeric values using statistics")

filled_statistics = missing.copy()

filled_statistics["age"] = filled_statistics["age"].fillna(
    filled_statistics["age"].median()
)

filled_statistics["income"] = filled_statistics["income"].fillna(
    filled_statistics["income"].median()
)

show("Median-based numeric imputation", filled_statistics)


# ============================================================================
# 57. MEAN VS MEDIAN
# ============================================================================

section("56. Mean versus median for imputation")

income_example = pd.Series([50000, 55000, 60000, 65000, 1000000, None])

print(f"Mean income:   {income_example.mean():,.2f}")
print(f"Median income: {income_example.median():,.2f}")

"""
The mean is sensitive to extreme values.

The median is generally more robust when the distribution contains outliers.
The correct choice depends on the meaning and distribution of the data.
"""


# ============================================================================
# 58. FORWARD FILL AND BACKWARD FILL
# ============================================================================

section("57. Forward-fill and backward-fill")

time_series = pd.Series(
    [100, None, None, 110, None, 120],
    index=pd.date_range("2026-01-01", periods=6),
)

show("Original time series", time_series)
show("Forward fill", time_series.ffill())
show("Backward fill", time_series.bfill())


# ============================================================================
# 59. LIMITING FILL OPERATIONS
# ============================================================================

section("58. Limiting the number of propagated missing values")

limited_forward_fill = time_series.ffill(limit=1)

show(
    "Forward fill with limit=1",
    limited_forward_fill,
)


# ============================================================================
# 60. INTERPOLATION
# ============================================================================

section("59. Interpolation")

numeric_series = pd.Series([10, None, None, 40, 50])

linear_interpolation = numeric_series.interpolate(method="linear")

show("Original series", numeric_series)
show("Linear interpolation", linear_interpolation)


# ============================================================================
# 61. GROUP-WISE MISSING VALUE IMPUTATION
# ============================================================================

section("60. Group-wise imputation")

employee_data = pd.DataFrame(
    {
        "department": [
            "Sales",
            "Sales",
            "Sales",
            "Engineering",
            "Engineering",
            "Engineering",
        ],
        "salary": [50000, None, 60000, 90000, None, 110000],
    }
)

employee_data["salary_imputed"] = (
    employee_data.groupby("department")["salary"]
    .transform(lambda group: group.fillna(group.median()))
)

show("Original employee data", employee_data)


# ============================================================================
# 62. MISSING DATA AND GROUPBY
# ============================================================================

section("61. Missing group keys")

missing_group_key = pd.DataFrame(
    {
        "department": ["Sales", "Sales", None, "Engineering", None],
        "revenue": [100, 200, 300, 400, 500],
    }
)

default_grouping = (
    missing_group_key.groupby("department")["revenue"].sum()
)

include_missing_group = (
    missing_group_key.groupby("department", dropna=False)["revenue"].sum()
)

show("Default grouping of missing keys", default_grouping)
show("Grouping while retaining missing keys", include_missing_group)


# ============================================================================
# 63. MISSING DATA AND MERGE
# ============================================================================

section("62. Missing values created by merges")

known_regions = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir"],
        "region": ["North", "South", "West"],
    }
)

customer_sales = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Meera", "Nisha"],
        "revenue": [100, 200, 300, 400],
    }
)

merged_missing = customer_sales.merge(
    known_regions,
    on="customer",
    how="left",
)

show("Left merge producing missing region values", merged_missing)


# ============================================================================
# 64. FILLING MERGE-INDUCED MISSING VALUES
# ============================================================================

section("63. Handling unmatched lookup records")

merged_missing["region"] = merged_missing["region"].fillna("Unknown")

show("Missing region replaced after merge", merged_missing)


# ============================================================================
# 65. MISSING VALUE SENTINELS
# ============================================================================

section("64. Different representations of missing data")

sentinel_data = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir", "Meera"],
        "score": [90, -1, 85, -1],
        "city": ["Delhi", "NA", "Mumbai", "Unknown"],
    }
)

sentinel_data["score"] = sentinel_data["score"].replace(-1, pd.NA)
sentinel_data["city"] = sentinel_data["city"].replace(
    ["NA", "Unknown"],
    pd.NA,
)

show("Standardized missing-value representation", sentinel_data)


# ============================================================================
# 66. NONE, NAN, AND PD.NA
# ============================================================================

section("65. None, NaN, and pandas.NA")

missing_types = pd.DataFrame(
    {
        "python_none": [None],
        "numpy_nan": [float("nan")],
        "pandas_na": [pd.NA],
    }
)

show("Different missing-value markers", missing_types)
show("Missing-value detection", missing_types.isna())


# ============================================================================
# 67. DUPLICATES
# ============================================================================

section("66. Duplicates and their relationship to joins")

duplicate_customers = pd.DataFrame(
    {
        "customer": ["Aarav", "Aarav", "Diya"],
        "segment": ["Premium", "VIP", "Standard"],
    }
)

show(
    "Duplicate lookup keys",
    duplicate_customers,
)

show(
    "Duplicate key counts",
    duplicate_customers["customer"].value_counts(),
)

"""
Duplicate keys are important because a many-to-many merge can multiply rows.

For example, if the left table has two rows for a key and the right table has
three rows for the same key, the merged result can contain six rows for that
key.
"""


# ============================================================================
# 68. EXPLICIT MANY-TO-MANY EXAMPLE
# ============================================================================

section("67. Many-to-many merge row multiplication")

left_many = pd.DataFrame(
    {
        "customer": ["Aarav", "Aarav"],
        "order": [1, 2],
    }
)

right_many = pd.DataFrame(
    {
        "customer": ["Aarav", "Aarav", "Aarav"],
        "tag": ["Premium", "Technology", "Delhi"],
    }
)

many_to_many = left_many.merge(
    right_many,
    on="customer",
    how="inner",
)

show("Many-to-many result", many_to_many)
print("Rows generated for Aarav:", len(many_to_many))


# ============================================================================
# 69. RELATIONAL ALGEBRA CONNECTIONS
# ============================================================================

section("68. Relational concepts behind merge and join")

"""
Important conceptual mappings:

- Inner merge  -> intersection of matching keys.
- Left merge   -> preserve every left-side record.
- Right merge  -> preserve every right-side record.
- Outer merge  -> preserve keys from both sides.
- Cross merge  -> Cartesian product.
- Concatenation -> append/stack datasets rather than match records.

Understanding these relationships helps prevent accidental row loss.
"""


# ============================================================================
# 70. DATA INTEGRITY CHECKS
# ============================================================================

section("69. Data-integrity checks before merging")

required_columns = {"customer", "region", "net_revenue"}

missing_required_columns = required_columns.difference(sales.columns)

if missing_required_columns:
    raise ValueError(
        f"Required columns are missing: {missing_required_columns}"
    )

duplicate_order_ids = sales["order_id"].duplicated().sum()

print(f"Duplicate order IDs: {duplicate_order_ids}")

if duplicate_order_ids:
    print("Warning: order_id is not unique.")


# ============================================================================
# 71. VALIDATING A DIMENSION TABLE
# ============================================================================

section("70. Validating unique lookup keys")

customer_key_counts = customers["customer"].value_counts()

invalid_customer_keys = customer_key_counts.loc[
    customer_key_counts > 1
]

if invalid_customer_keys.empty:
    print("Customer lookup table has unique customer keys.")
else:
    print("Duplicate customer keys found:")
    print(invalid_customer_keys)


# ============================================================================
# 72. PRACTICAL SALES PIPELINE
# ============================================================================

section("71. Practical end-to-end data-manipulation pipeline")

raw_orders = pd.DataFrame(
    {
        "order_id": [1, 2, 3, 4, 5, 6],
        "customer_id": ["C1", "C2", "C1", "C3", "C4", "C2"],
        "product_id": ["P1", "P2", "P1", "P3", "P2", "P4"],
        "quantity": [2, 1, 3, 1, 4, 2],
        "unit_price": [500, 1000, 500, 1500, 1000, 700],
    }
)

customer_dimension = pd.DataFrame(
    {
        "customer_id": ["C1", "C2", "C3", "C4"],
        "region": ["North", "South", "West", "East"],
        "segment": ["Premium", "Standard", "Premium", "Standard"],
    }
)

product_dimension = pd.DataFrame(
    {
        "product_id": ["P1", "P2", "P3", "P4"],
        "category": ["Accessories", "Electronics", "Electronics", "Accessories"],
    }
)

raw_orders["gross_revenue"] = (
    raw_orders["quantity"] * raw_orders["unit_price"]
)

enriched_orders = (
    raw_orders
    .merge(
        customer_dimension,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    .merge(
        product_dimension,
        on="product_id",
        how="left",
        validate="many_to_one",
    )
)

show("Enriched order dataset", enriched_orders)

regional_performance = enriched_orders.groupby(
    "region",
    as_index=False,
).agg(
    orders=("order_id", "count"),
    units=("quantity", "sum"),
    revenue=("gross_revenue", "sum"),
    customers=("customer_id", "nunique"),
)

show("Regional performance", regional_performance)


# ============================================================================
# 73. PRACTICAL PIVOT FROM PIPELINE
# ============================================================================

section("72. Regional category analysis")

regional_category_matrix = pd.pivot_table(
    enriched_orders,
    values="gross_revenue",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
)

show(
    "Revenue matrix by region and category",
    regional_category_matrix,
)


# ============================================================================
# 74. LONG-FORM ANALYTICS DATASET
# ============================================================================

section("73. Converting analytical wide data into long format")

quarterly = pd.DataFrame(
    {
        "region": ["North", "South", "West"],
        "Q1": [100000, 120000, 90000],
        "Q2": [110000, 130000, 95000],
        "Q3": [125000, 140000, 105000],
        "Q4": [135000, 150000, 115000],
    }
)

quarterly_long = quarterly.melt(
    id_vars="region",
    var_name="quarter",
    value_name="revenue",
)

show("Quarterly wide dataset", quarterly)
show("Quarterly long dataset", quarterly_long)


# ============================================================================
# 75. RECONSTRUCTING WIDE DATA
# ============================================================================

section("74. Reshaping long data back into wide form")

quarterly_wide_again = quarterly_long.pivot(
    index="region",
    columns="quarter",
    values="revenue",
)

show("Reconstructed wide dataset", quarterly_wide_again)


# ============================================================================
# 76. MULTI-LEVEL PIVOT
# ============================================================================

section("75. Pivot table with multiple index dimensions")

multi_index_pivot = pd.pivot_table(
    enriched_orders,
    values="gross_revenue",
    index=["region", "segment"],
    columns="category",
    aggfunc="sum",
    fill_value=0,
)

show("Multi-level pivot table", multi_index_pivot)


# ============================================================================
# 77. FLATTENING MULTI-INDEX COLUMNS
# ============================================================================

section("76. Flattening MultiIndex columns")

multi_measure_pivot = pd.pivot_table(
    enriched_orders,
    values=["quantity", "gross_revenue"],
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
)

show("MultiIndex columns before flattening", multi_measure_pivot)

flattened = multi_measure_pivot.copy()

flattened.columns = [
    "_".join(str(part) for part in column).strip("_")
    for column in flattened.columns.to_flat_index()
]

show("Flattened columns", flattened.reset_index())


# ============================================================================
# 78. CONDITIONAL AGGREGATION
# ============================================================================

section("77. Conditional aggregation")

conditional_summary = enriched_orders.groupby("region").agg(
    total_revenue=("gross_revenue", "sum"),
    premium_revenue=(
        "gross_revenue",
        lambda series: series[
            enriched_orders.loc[series.index, "segment"].eq("Premium")
        ].sum(),
    ),
)

show("Total and premium revenue by region", conditional_summary)


# ============================================================================
# 79. GROUPBY PERCENTAGES
# ============================================================================

section("78. Percentage of total revenue")

total_revenue = enriched_orders["gross_revenue"].sum()

region_percentage = (
    enriched_orders.groupby("region")["gross_revenue"]
    .sum()
    .div(total_revenue)
    .mul(100)
    .sort_values(ascending=False)
)

show("Each region's percentage of total revenue", region_percentage)


# ============================================================================
# 80. GROUPBY WITH TIME
# ============================================================================

section("79. Grouping by date components")

dated_sales = sales.copy()

dated_sales["month"] = dated_sales["date"].dt.to_period("M")
dated_sales["day_of_week"] = dated_sales["date"].dt.day_name()

monthly_sales = (
    dated_sales.groupby("month")["net_revenue"]
    .sum()
)

weekday_sales = (
    dated_sales.groupby("day_of_week")["net_revenue"]
    .sum()
    .sort_values(ascending=False)
)

show("Monthly revenue", monthly_sales)
show("Revenue by day of week", weekday_sales)


# ============================================================================
# 81. GROUPBY RESAMPLE
# ============================================================================

section("80. Time-based resampling after setting a datetime index")

daily_sales = (
    dated_sales.set_index("date")["net_revenue"]
    .resample("D")
    .sum()
)

weekly_sales = (
    dated_sales.set_index("date")["net_revenue"]
    .resample("W")
    .sum()
)

show("Daily revenue", daily_sales)
show("Weekly revenue", weekly_sales)


# ============================================================================
# 82. GROUPBY ROLLING
# ============================================================================

section("81. Group-wise rolling calculations")

customer_time = (
    dated_sales.sort_values(["customer", "date"])
    .set_index("date")
)

customer_time["three_order_rolling_average"] = (
    customer_time.groupby("customer")["net_revenue"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

show(
    "Rolling average within each customer",
    customer_time[
        [
            "customer",
            "net_revenue",
            "three_order_rolling_average",
        ]
    ],
)


# ============================================================================
# 83. MISSING VALUES IN NUMERIC CALCULATIONS
# ============================================================================

section("82. Missing values and arithmetic")

numeric_missing = pd.Series([10, 20, None, 40])

show("Original series", numeric_missing)
show("Series sum", numeric_missing.sum())
show("Series mean", numeric_missing.mean())

"""
Many pandas reduction operations ignore missing values by default.

The skipna parameter can change this behavior.
"""

show(
    "Sum with skipna=False",
    numeric_missing.sum(skipna=False),
)

show(
    "Mean with skipna=False",
    numeric_missing.mean(skipna=False),
)


# ============================================================================
# 84. DROPNA THRESHOLD
# ============================================================================

section("83. Keeping rows with enough non-missing values")

survey = pd.DataFrame(
    {
        "name": ["A", "B", "C"],
        "age": [25, None, 35],
        "income": [50000, None, 70000],
        "score": [80, 90, None],
    }
)

threshold_result = survey.dropna(
    thresh=3
)

show(
    "Rows with at least three non-missing values",
    threshold_result,
)


# ============================================================================
# 85. MISSINGNESS AS A FEATURE
# ============================================================================

section("84. Creating missingness indicators")

missing_feature = missing.copy()

missing_feature["income_missing"] = (
    missing_feature["income"].isna().astype(int)
)

show(
    "Missingness indicator",
    missing_feature[
        ["customer", "income", "income_missing"]
    ],
)


# ============================================================================
# 86. COALESCE-LIKE LOGIC
# ============================================================================

section("85. Selecting the first available value")

primary_city = pd.Series(["Delhi", None, "Pune", None])
secondary_city = pd.Series(["Mumbai", "Kolkata", None, "Chennai"])

coalesced_city = primary_city.combine_first(secondary_city)

show("Primary values", primary_city)
show("Secondary values", secondary_city)
show("First available value", coalesced_city)


# ============================================================================
# 87. MERGE ORDER CONSIDERATIONS
# ============================================================================

section("86. Merge ordering")

left_order = pd.DataFrame(
    {
        "id": [3, 1, 2],
        "value_left": ["C", "A", "B"],
    }
)

right_order = pd.DataFrame(
    {
        "id": [2, 3, 1],
        "value_right": ["two", "three", "one"],
    }
)

merge_preserve = left_order.merge(
    right_order,
    on="id",
    how="left",
    sort=False,
)

merge_sorted = left_order.merge(
    right_order,
    on="id",
    how="left",
    sort=True,
)

show("Left merge with sort=False", merge_preserve)
show("Left merge with sort=True", merge_sorted)


# ============================================================================
# 88. COLUMN NAME CLEANING
# ============================================================================

section("87. Cleaning columns before data manipulation")

messy_columns = pd.DataFrame(
    {
        "Customer Name ": ["Aarav", "Diya"],
        " Net Revenue": [50000, 60000],
        "Region": ["North", "South"],
    }
)

messy_columns.columns = (
    messy_columns.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

show("Normalized column names", messy_columns)


# ============================================================================
# 89. TYPE CONSISTENCY
# ============================================================================

section("88. Data types and safe conversion")

mixed_quantity = pd.DataFrame(
    {
        "quantity": ["10", "20", "unknown", "40"],
    }
)

mixed_quantity["quantity_numeric"] = pd.to_numeric(
    mixed_quantity["quantity"],
    errors="coerce",
)

show("Original mixed values", mixed_quantity)


# ============================================================================
# 90. MISSING VALUES CREATED BY TYPE CONVERSION
# ============================================================================

section("89. Invalid values becoming missing")

print(
    "Invalid strings become NaN when errors='coerce':"
)
print(mixed_quantity["quantity_numeric"])


# ============================================================================
# 91. CATEGORICAL NORMALIZATION
# ============================================================================

section("90. Normalizing categorical values")

regions_messy = pd.DataFrame(
    {
        "region": [
            "North",
            "north",
            " NORTH ",
            "South",
            "south ",
        ]
    }
)

regions_messy["region_normalized"] = (
    regions_messy["region"]
    .str.strip()
    .str.title()
)

show("Normalized categories", regions_messy)


# ============================================================================
# 92. MEMORY CONSIDERATIONS
# ============================================================================

section("91. Memory usage")

memory_example = pd.DataFrame(
    {
        "customer": ["Aarav", "Diya", "Kabir", "Aarav"] * 1000,
        "region": ["North", "South", "West", "North"] * 1000,
    }
)

memory_before = memory_example.memory_usage(deep=True).sum()

memory_example["customer"] = memory_example["customer"].astype("category")
memory_example["region"] = memory_example["region"].astype("category")

memory_after = memory_example.memory_usage(deep=True).sum()

print(f"Memory before categorical conversion: {memory_before:,} bytes")
print(f"Memory after categorical conversion:  {memory_after:,} bytes")


# ============================================================================
# 93. PERFORMANCE COMPARISON
# ============================================================================

section("92. Vectorized operation versus Python-level loop")

performance_data = pd.DataFrame(
    {
        "value": range(100_000),
    }
)


def loop_square(values: list[int]) -> list[int]:
    """Square values using a Python loop."""
    return [value * value for value in values]


start = time.perf_counter()
loop_result = loop_square(performance_data["value"].tolist())
loop_time = time.perf_counter() - start

start = time.perf_counter()
vectorized_result = performance_data["value"] ** 2
vectorized_time = time.perf_counter() - start

print(f"Python-loop time: {loop_time:.6f} seconds")
print(f"Pandas vectorized time: {vectorized_time:.6f} seconds")
print(
    "Results equal:",
    loop_result == vectorized_result.tolist(),
)

"""
Vectorization generally reduces Python-level iteration overhead.

Performance depends on the operation, data type, hardware, and pandas
version. The principle is to prefer native vectorized operations where they
express the intended transformation clearly.
"""


# ============================================================================
# 94. GROUPBY PERFORMANCE DESIGN
# ============================================================================

section("93. Performance considerations for GroupBy")

"""
Useful principles:

- Group only by columns that are required.
- Select required columns before expensive operations.
- Prefer built-in aggregations such as sum, mean, min, max, count.
- Avoid Python-level apply() when a vectorized operation or aggregation works.
- Convert low-cardinality repeated strings to categorical when appropriate.
- Avoid unnecessary copies of very large DataFrames.
"""

performance_grouped = (
    performance_data.assign(
        group=performance_data["value"] % 10
    )
    .groupby("group")["value"]
    .agg(["count", "sum", "mean"])
)

show("Efficient built-in GroupBy aggregations", performance_grouped)


# ============================================================================
# 95. COPY AND SETTINGWITHCOPY CONSIDERATIONS
# ============================================================================

section("94. Safe modification of filtered data")

filtered = sales.loc[sales["net_revenue"] > 50000].copy()

filtered["priority"] = "High"

show(
    "Explicit copy before modifying a filtered DataFrame",
    filtered[["order_id", "net_revenue", "priority"]],
)

"""
Using .copy() makes the intent explicit when creating an independent object
from a filtered DataFrame.

This helps avoid ambiguous chained-assignment behavior.
"""


# ============================================================================
# 96. ASSERTIONS
# ============================================================================

section("95. Assertions for data quality")

assert sales["order_id"].notna().all(), "order_id contains missing values"
assert (sales["quantity"] > 0).all(), "quantity must be positive"
assert (sales["unit_price"] >= 0).all(), "unit_price cannot be negative"

print("Basic sales data-quality assertions passed.")


# ============================================================================
# 97. RECONCILIATION AFTER MERGE
# ============================================================================

section("96. Reconciliation before and after a merge")

before_rows = len(raw_orders)

reconciled = raw_orders.merge(
    customer_dimension,
    on="customer_id",
    how="left",
    validate="many_to_one",
)

after_rows = len(reconciled)

print(f"Rows before merge: {before_rows}")
print(f"Rows after merge:  {after_rows}")

if before_rows != after_rows:
    raise ValueError(
        "Unexpected row-count change during a many-to-one enrichment merge."
    )

print("Row-count reconciliation passed.")


# ============================================================================
# 98. UNMATCHED KEY ANALYSIS
# ============================================================================

section("97. Finding unmatched foreign keys")

orders_with_unknown_customer = pd.DataFrame(
    {
        "order_id": [1, 2, 3],
        "customer_id": ["C1", "C2", "C99"],
    }
)

customer_lookup = pd.DataFrame(
    {
        "customer_id": ["C1", "C2"],
        "region": ["North", "South"],
    }
)

audit_merge = orders_with_unknown_customer.merge(
    customer_lookup,
    on="customer_id",
    how="left",
    indicator=True,
)

show("Merge audit", audit_merge)

unmatched = audit_merge.loc[
    audit_merge["_merge"].eq("left_only")
]

show("Unmatched customer IDs", unmatched)


# ============================================================================
# 99. DUPLICATE DETECTION BEFORE MERGE
# ============================================================================

section("98. Detecting duplicate lookup keys before merge")

lookup = pd.DataFrame(
    {
        "customer_id": ["C1", "C2", "C2", "C3"],
        "region": ["North", "South", "West", "East"],
    }
)

duplicates = lookup.loc[
    lookup["customer_id"].duplicated(keep=False)
].sort_values("customer_id")

show("Duplicate lookup records", duplicates)


# ============================================================================
# 100. INDEX ALIGNMENT
# ============================================================================

section("99. Pandas alignment and why it matters")

first = pd.Series(
    [100, 200, 300],
    index=["A", "B", "C"],
)

second = pd.Series(
    [10, 20, 30],
    index=["C", "A", "B"],
)

aligned_addition = first + second

show("First Series", first)
show("Second Series", second)
show("Addition by index rather than physical position", aligned_addition)

"""
Pandas generally aligns labeled objects by their indexes.

This is powerful for joins and analytical calculations but can produce
unexpected missing values if indexes are not what the programmer intended.
"""


# ============================================================================
# 101. JOIN VERSUS MERGE
# ============================================================================

section("100. Conceptual comparison: merge versus join")

comparison = pd.DataFrame(
    {
        "Operation": [
            "merge",
            "join",
            "concat",
            "groupby",
            "pivot_table",
        ],
        "Primary purpose": [
            "Match records using keys",
            "Combine data, often using indexes",
            "Stack or align datasets",
            "Split data into groups and aggregate/transform",
            "Summarize and reshape data",
        ],
        "Typical question": [
            "Which customer information belongs to this order?",
            "Which indexed information belongs to this row?",
            "How do I combine these monthly files?",
            "What is revenue by region?",
            "What is revenue by region and category?",
        ],
    }
)

show("Core data-manipulation operations", comparison)


# ============================================================================
# 102. GROUPBY VERSUS PIVOT_TABLE
# ============================================================================

section("101. GroupBy versus pivot_table")

groupby_result = (
    sales.groupby(["region", "category"])["net_revenue"]
    .sum()
    .reset_index()
)

pivot_result = pd.pivot_table(
    sales,
    values="net_revenue",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
)

show("GroupBy result", groupby_result)
show("Pivot-table result", pivot_result)

"""
GroupBy is often preferable for programmatic pipelines and flexible
multi-step transformations.

pivot_table is particularly convenient when the desired result is a
two-dimensional analytical matrix.
"""


# ============================================================================
# 103. JOIN VERSUS CONCAT
# ============================================================================

section("102. Join versus concatenation")

north = pd.DataFrame(
    {
        "id": [1, 2],
        "north_value": [10, 20],
    }
)

south = pd.DataFrame(
    {
        "id": [1, 2],
        "south_value": [30, 40],
    }
)

joined_regions = north.merge(
    south,
    on="id",
    how="inner",
)

concatenated_regions = pd.concat(
    [north, south],
    ignore_index=True,
)

show("Matching records with merge", joined_regions)
show("Stacking records with concat", concatenated_regions)


# ============================================================================
# 104. PRACTICAL MISSING-DATA DECISION LOGIC
# ============================================================================

section("103. Missing-data decision framework")

def missing_data_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact missing-data audit."""
    report = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "non_missing_count": frame.notna().sum(),
            "missing_percentage": frame.isna().mean().mul(100),
            "dtype": frame.dtypes.astype(str),
        }
    )

    return report.sort_values(
        "missing_percentage",
        ascending=False,
    )


show(
    "Missing-data audit",
    missing_data_report(missing),
)

"""
There is no universal rule for missing values.

Possible strategies include:

- Remove rows when the missing observations are few and non-critical.
- Remove columns when the variable is mostly missing and has little value.
- Fill categorical values with an explicit category such as "Unknown".
- Use mean or median for suitable numerical variables.
- Use group-wise statistics when groups have materially different behavior.
- Use forward/backward fill for appropriate ordered time-series data.
- Interpolate when the underlying variable reasonably changes continuously.
- Preserve missingness when missing itself has analytical meaning.
"""


# ============================================================================
# 105. EDGE CASE: EMPTY DATAFRAME
# ============================================================================

section("104. Empty DataFrame behavior")

empty = pd.DataFrame(
    columns=["region", "revenue"]
)

show("Empty DataFrame", empty)

empty_grouped = empty.groupby("region")["revenue"].sum()

show("GroupBy on empty data", empty_grouped)


# ============================================================================
# 106. EDGE CASE: ALL VALUES MISSING
# ============================================================================

section("105. All values missing")

all_missing = pd.Series([None, None, None])

print("Mean:", all_missing.mean())
print("Median:", all_missing.median())
print("Sum:", all_missing.sum())
print("Sum with skipna=False:", all_missing.sum(skipna=False))


# ============================================================================
# 107. EDGE CASE: ZERO VALUES
# ============================================================================

section("106. Zero is not the same as missing")

zero_data = pd.Series([0, 10, None, 20])

print("Missing mask:")
print(zero_data.isna())

print("Zero-value mask:")
print(zero_data.eq(0))

"""
Zero is a valid numeric value.

Replacing zero with missing values without domain justification can corrupt
analysis.
"""


# ============================================================================
# 108. EDGE CASE: EMPTY GROUPS AND CATEGORICAL DATA
# ============================================================================

section("107. Observed categorical groups")

categorical_example = pd.DataFrame(
    {
        "region": pd.Categorical(
            ["North", "South"],
            categories=["North", "South", "East", "West"],
        ),
        "revenue": [100, 200],
    }
)

observed_only = (
    categorical_example.groupby(
        "region",
        observed=True,
    )["revenue"]
    .sum()
)

show("Only observed categorical groups", observed_only)


# ============================================================================
# 109. PRACTICAL CUSTOMER METRICS
# ============================================================================

section("108. Customer-level analytical metrics")

customer_metrics = (
    enriched_orders.groupby("customer_id")
    .agg(
        total_orders=("order_id", "count"),
        total_units=("quantity", "sum"),
        total_revenue=("gross_revenue", "sum"),
        average_order_value=("gross_revenue", "mean"),
        products_purchased=("product_id", "nunique"),
    )
    .sort_values("total_revenue", ascending=False)
)

show("Customer metrics", customer_metrics)


# ============================================================================
# 110. PRACTICAL CATEGORY PERFORMANCE
# ============================================================================

section("109. Category performance")

category_metrics = (
    enriched_orders.groupby("category")
    .agg(
        orders=("order_id", "count"),
        units=("quantity", "sum"),
        revenue=("gross_revenue", "sum"),
        average_price=("unit_price", "mean"),
    )
    .sort_values("revenue", ascending=False)
)

show("Category metrics", category_metrics)


# ============================================================================
# 111. PRACTICAL REGION-CATEGORY CONTRIBUTION
# ============================================================================

section("110. Region-category contribution")

region_category_metrics = (
    enriched_orders.groupby(
        ["region", "category"]
    )
    .agg(
        revenue=("gross_revenue", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "count"),
    )
    .reset_index()
)

region_category_metrics["region_revenue"] = (
    region_category_metrics.groupby("region")["revenue"]
    .transform("sum")
)

region_category_metrics["category_share_within_region"] = (
    region_category_metrics["revenue"]
    / region_category_metrics["region_revenue"]
)

show(
    "Category contribution within each region",
    region_category_metrics,
)


# ============================================================================
# 112. TOP-N PER GROUP
# ============================================================================

section("111. Top-N records within every group")

top_products_per_region = (
    region_category_metrics
    .sort_values(
        ["region", "revenue"],
        ascending=[True, False],
    )
    .groupby("region", group_keys=False)
    .head(1)
)

show(
    "Highest-revenue category in each region",
    top_products_per_region,
)


# ============================================================================
# 113. ANTI-JOIN STYLE ANALYSIS
# ============================================================================

section("112. Finding records with no matching lookup")

left_records = pd.DataFrame(
    {
        "customer_id": ["C1", "C2", "C3", "C4"],
    }
)

right_records = pd.DataFrame(
    {
        "customer_id": ["C1", "C2"],
    }
)

anti_join_style = (
    left_records
    .merge(
        right_records,
        on="customer_id",
        how="left",
        indicator=True,
    )
    .loc[lambda frame: frame["_merge"].eq("left_only")]
    .drop(columns="_merge")
)

show(
    "Left-side records without a match",
    anti_join_style,
)


# ============================================================================
# 114. SEMI-JOIN STYLE ANALYSIS
# ============================================================================

section("113. Finding records that do have a matching lookup")

semi_join_style = (
    left_records
    .merge(
        right_records,
        on="customer_id",
        how="inner",
    )
    .drop_duplicates("customer_id")
)

show(
    "Left-side records with at least one match",
    semi_join_style,
)


# ============================================================================
# 115. NULL KEY BEHAVIOR
# ============================================================================

section("114. Missing merge keys require careful interpretation")

null_left = pd.DataFrame(
    {
        "key": ["A", None, "B"],
        "left_value": [1, 2, 3],
    }
)

null_right = pd.DataFrame(
    {
        "key": ["A", None, "C"],
        "right_value": [10, 20, 30],
    }
)

null_merge = null_left.merge(
    null_right,
    on="key",
    how="inner",
)

show("Merge involving missing keys", null_merge)

"""
Pandas merge behavior around missing keys should be tested explicitly when
null keys are possible. Missing identifiers often represent a data-quality
problem rather than a legitimate relational key.
"""


# ============================================================================
# 116. PRE-AGGREGATION BEFORE MERGE
# ============================================================================

section("115. Reducing data before enrichment")

large_order_table = pd.DataFrame(
    {
        "customer_id": ["C1", "C1", "C2", "C2", "C3"],
        "revenue": [100, 200, 300, 400, 500],
    }
)

customer_totals = (
    large_order_table.groupby("customer_id", as_index=False)
    .agg(total_revenue=("revenue", "sum"))
)

customer_segments = pd.DataFrame(
    {
        "customer_id": ["C1", "C2", "C3"],
        "segment": ["Premium", "Standard", "Premium"],
    }
)

preaggregated_merge = customer_totals.merge(
    customer_segments,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

show("Pre-aggregated customer metrics before enrichment", preaggregated_merge)

"""
Pre-aggregation can substantially reduce the number of rows that need to
participate in later joins and can make the intended grain of the analysis
clearer.
"""


# ============================================================================
# 117. GRAIN OF DATA
# ============================================================================

section("116. Understanding data grain")

"""
Data grain means what one row represents.

Examples:

- One row per order.
- One row per order line.
- One row per customer.
- One row per customer-month.
- One row per region-category combination.

A large fraction of merge and aggregation errors come from combining tables
with incompatible grains without first deciding what the resulting row
should represent.
"""

grain_example = enriched_orders.groupby(
    ["customer_id", "category"],
    as_index=False,
).agg(
    orders=("order_id", "count"),
    revenue=("gross_revenue", "sum"),
)

show("One row per customer-category combination", grain_example)


# ============================================================================
# 118. PIPELINE FUNCTION
# ============================================================================

section("117. Building reusable transformation functions")

def prepare_sales_analysis(
    orders: pd.DataFrame,
    customer_lookup: pd.DataFrame,
    product_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and enrich an order-level dataset.

    Expected grain:
        one row per order.

    Lookup tables are expected to have unique keys.
    """
    required_order_columns = {
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
    }

    missing_columns = required_order_columns.difference(orders.columns)

    if missing_columns:
        raise ValueError(
            f"Orders missing required columns: {sorted(missing_columns)}"
        )

    if orders["order_id"].duplicated().any():
        raise ValueError("order_id must be unique in this example.")

    if customer_lookup["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique in customer lookup.")

    if product_lookup["product_id"].duplicated().any():
        raise ValueError("product_id must be unique in product lookup.")

    result = orders.copy()

    result["revenue"] = (
        result["quantity"] * result["unit_price"]
    )

    result = result.merge(
        customer_lookup,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    return result


prepared = prepare_sales_analysis(
    raw_orders,
    customer_dimension,
    product_dimension,
)

show("Reusable analytical pipeline output", prepared)


# ============================================================================
# 119. TESTING A TRANSFORMATION
# ============================================================================

section("118. Testing expected results")

expected_total_revenue = raw_orders["quantity"].mul(
    raw_orders["unit_price"]
).sum()

actual_total_revenue = prepared["revenue"].sum()

assert actual_total_revenue == expected_total_revenue

assert len(prepared) == len(raw_orders)

print("Revenue reconciliation test passed.")
print("Row-count preservation test passed.")


# ============================================================================
# 120. COMPLETE ANALYTICS WORKFLOW
# ============================================================================

section("119. Complete analytical workflow")

workflow_orders = prepared.copy()

workflow_orders["customer_revenue"] = (
    workflow_orders.groupby("customer_id")["revenue"]
    .transform("sum")
)

workflow_orders["region_revenue"] = (
    workflow_orders.groupby("region")["revenue"]
    .transform("sum")
)

workflow_orders["share_of_customer"] = (
    workflow_orders["revenue"]
    / workflow_orders["customer_revenue"]
)

workflow_orders["share_of_region"] = (
    workflow_orders["revenue"]
    / workflow_orders["region_revenue"]
)

workflow_summary = (
    workflow_orders.groupby(
        ["region", "category"],
        as_index=False,
    )
    .agg(
        orders=("order_id", "count"),
        units=("quantity", "sum"),
        revenue=("revenue", "sum"),
        customers=("customer_id", "nunique"),
    )
    .sort_values("revenue", ascending=False)
)

workflow_pivot = pd.pivot_table(
    workflow_summary,
    values="revenue",
    index="region",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
)

show("Order-level enriched workflow data", workflow_orders)
show("Region-category summary", workflow_summary)
show("Final analytical pivot", workflow_pivot)


# ============================================================================
# 121. BEST-PRACTICE CHECKLIST AS EXECUTABLE RULES
# ============================================================================

section("120. Practical best-practice checks")

def validate_dataframe(
    frame: pd.DataFrame,
    required_columns: set[str],
    key_column: str | None = None,
) -> dict[str, object]:
    """Return basic structural and quality checks."""
    missing_columns = required_columns.difference(frame.columns)

    duplicate_keys = None
    if key_column is not None and key_column in frame.columns:
        duplicate_keys = int(frame[key_column].duplicated().sum())

    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_required_columns": sorted(missing_columns),
        "total_missing_cells": int(frame.isna().sum().sum()),
        "duplicate_key_count": duplicate_keys,
    }


quality_report = validate_dataframe(
    prepared,
    required_columns={
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
        "revenue",
    },
    key_column="order_id",
)

show("Quality report", quality_report)


# ============================================================================
# 122. FINAL CONCEPTUAL RULES
# ============================================================================

section("121. Core rules demonstrated by the script")

rules = [
    "Use groupby when the analysis requires split-apply-combine operations.",
    "Use transform when a group-level result must retain the original row count.",
    "Use merge when records must be matched through keys.",
    "Use join when index-based combination is convenient.",
    "Use concat when datasets should be stacked or aligned rather than matched.",
    "Use pivot_table when a grouped result should become a matrix.",
    "Use melt when wide analytical data should become long analytical data.",
    "Treat missing values according to their business and statistical meaning.",
    "Validate lookup-key uniqueness before one-to-many or many-to-one merges.",
    "Use merge(validate=...) to detect unexpected relationship cardinality.",
    "Use merge(indicator=True) to audit unmatched records.",
    "Check row counts before and after important merges.",
    "Understand the grain of every DataFrame before combining it.",
    "Prefer built-in vectorized operations over Python-level loops for large data.",
    "Use explicit copies when modifying filtered DataFrames.",
    "Do not confuse zero with missing.",
    "Do not silently discard missing values without understanding their meaning.",
]

for number, rule in enumerate(rules, start=1):
    print(f"{number:02d}. {rule}")


# ============================================================================
# 123. SCRIPT COMPLETION
# ============================================================================

section("122. End of data-manipulation study script")

print(
    "The examples above form a complete progression from basic pandas "
    "manipulation to grouped analytics, relational operations, reshaping, "
    "missing-data handling, validation, performance, and production-oriented "
    "data-quality practices."
)
