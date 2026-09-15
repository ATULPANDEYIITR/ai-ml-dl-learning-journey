"""
Data Visualization: Histograms, Scatter Plots, Box Plots, Line Plots, and Distributions

This standalone study script progresses from fundamental statistical visualization
concepts to reusable plotting functions, distribution analysis, outlier detection,
correlation analysis, time-series visualization, uncertainty, and production-oriented
plotting practices.

Required package:
    matplotlib

Optional package:
    numpy

Install if necessary:
    python -m pip install matplotlib numpy

The script uses Matplotlib's non-interactive "Agg" backend so it can execute in
headless environments. Generated charts are saved to a local "visualization_output"
directory instead of requiring a graphical desktop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, multimode, stdev
from typing import Iterable, Sequence
import math
import random
import sys

import matplotlib

# A non-interactive backend makes this educational script executable on servers,
# notebooks without display support, CI systems, and containers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def save_figure(figure: plt.Figure, filename: str) -> Path:
    """
    Save a figure consistently.

    Tight bounding boxes reduce accidental clipping of labels. A moderate DPI
    produces readable raster output without making files unnecessarily large.
    """
    path = OUTPUT_DIR / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# 1. Fundamental concepts
# ---------------------------------------------------------------------------

def print_fundamentals() -> None:
    print("=" * 78)
    print("DATA VISUALIZATION FUNDAMENTALS")
    print("=" * 78)

    concepts = {
        "Variable": "A measurable characteristic such as age, price, or temperature.",
        "Observation": "One recorded value or row in a dataset.",
        "Distribution": "The pattern of values and their frequencies or probabilities.",
        "Frequency": "The number of observations satisfying a condition.",
        "Histogram": "A plot that groups numeric observations into bins.",
        "Scatter plot": "A plot showing the relationship between two numeric variables.",
        "Box plot": "A compact representation of median, quartiles, spread, and potential outliers.",
        "Line plot": "A plot connecting ordered observations, commonly used for time series.",
        "Mean": "Arithmetic average; sensitive to extreme observations.",
        "Median": "Middle value after sorting; generally more robust to outliers.",
        "Quartile": "A percentile-based location dividing ordered data into four parts.",
        "IQR": "Interquartile range, Q3 - Q1.",
        "Outlier": "An observation unusually distant from the main body of a dataset.",
        "Correlation": "A measure of linear association between two numeric variables.",
        "Bin": "An interval used to group numeric observations in a histogram.",
    }

    for term, definition in concepts.items():
        print(f"{term:15} : {definition}")


# ---------------------------------------------------------------------------
# 2. Small descriptive-statistics toolkit
# ---------------------------------------------------------------------------

def quantile(values: Sequence[float], probability: float) -> float:
    """
    Compute a linearly interpolated quantile.

    This implementation uses the common position:
        index = p * (n - 1)

    It avoids relying on an external statistics package for the central
    educational calculations.
    """
    if not values:
        raise ValueError("Cannot calculate a quantile from an empty sequence.")
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def describe(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    """Return several useful descriptive statistics."""
    if not values:
        raise ValueError("At least one observation is required.")

    numeric_values = [float(value) for value in values]
    q1 = quantile(numeric_values, 0.25)
    q2 = quantile(numeric_values, 0.50)
    q3 = quantile(numeric_values, 0.75)

    result: dict[str, float | int | list[float]] = {
        "count": len(numeric_values),
        "minimum": min(numeric_values),
        "q1": q1,
        "median": q2,
        "mean": mean(numeric_values),
        "q3": q3,
        "maximum": max(numeric_values),
        "iqr": q3 - q1,
    }

    if len(numeric_values) >= 2:
        result["sample_std"] = stdev(numeric_values)
    else:
        result["sample_std"] = 0.0

    return result


def detect_iqr_outliers(values: Sequence[float]) -> list[float]:
    """
    Detect potential outliers using Tukey's 1.5 * IQR rule.

    This is a visualization-oriented rule, not proof that an observation is
    erroneous. A valid extreme observation can legitimately be an outlier.
    """
    if not values:
        return []

    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    return [
        float(value)
        for value in values
        if value < lower_fence or value > upper_fence
    ]


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Pearson's linear correlation coefficient."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of observations.")
    if len(x) < 2:
        raise ValueError("At least two paired observations are required.")

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )

    x_sum = sum((value - x_mean) ** 2 for value in x)
    y_sum = sum((value - y_mean) ** 2 for value in y)

    denominator = math.sqrt(x_sum * y_sum)

    if denominator == 0:
        raise ValueError("Correlation is undefined when one variable is constant.")

    return numerator / denominator


# ---------------------------------------------------------------------------
# 3. Histogram fundamentals
# ---------------------------------------------------------------------------

def demonstrate_histogram() -> None:
    """
    Histograms answer questions such as:
      - Where are values concentrated?
      - Is the distribution symmetric?
      - Is it skewed?
      - Are there multiple modes?
      - How broad is the observed range?
    """
    data = [
        48, 52, 53, 55, 56, 57, 58, 59, 60, 60,
        61, 62, 62, 63, 64, 65, 65, 66, 67, 68,
        69, 70, 70, 71, 72, 73, 74, 75, 77, 82,
    ]

    figure, axis = plt.subplots(figsize=(9, 5))

    # A histogram counts observations inside adjacent intervals.
    axis.hist(
        data,
        bins=8,
        edgecolor="black",
        alpha=0.75,
    )

    axis.set_title("Histogram of Example Measurements")
    axis.set_xlabel("Measurement")
    axis.set_ylabel("Frequency")
    save_figure(figure, "01_histogram_basic.png")

    print("\nHistogram:")
    print("  Saved: 01_histogram_basic.png")
    print("  Range:", min(data), "to", max(data))
    print("  Modes:", multimode(data))


def demonstrate_histogram_bin_choices() -> None:
    """Show why bin selection affects the visual interpretation."""
    data = [
        random.gauss(68, 8)
        for _ in range(500)
    ]

    for bin_count in (5, 10, 20, 40):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(data, bins=bin_count, edgecolor="black", alpha=0.75)
        axis.set_title(f"Same Data with {bin_count} Histogram Bins")
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")
        save_figure(figure, f"02_histogram_{bin_count}_bins.png")

    print("\nHistogram bin sensitivity:")
    print("  Few bins can hide important structure.")
    print("  Too many bins can emphasize random noise.")
    print("  Bin choice should reflect sample size and analytical purpose.")


def demonstrate_density_histogram() -> None:
    """
    Density=True changes the y-axis from counts to probability density.

    The total area under the histogram approximately equals one. This makes
    distributions with different sample sizes easier to compare.
    """
    group_a = [random.gauss(60, 5) for _ in range(500)]
    group_b = [random.gauss(72, 8) for _ in range(500)]

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.hist(
        group_a,
        bins=25,
        density=True,
        alpha=0.55,
        label="Group A",
    )
    axis.hist(
        group_b,
        bins=25,
        density=True,
        alpha=0.55,
        label="Group B",
    )

    axis.set_title("Density Histograms for Two Distributions")
    axis.set_xlabel("Value")
    axis.set_ylabel("Density")
    axis.legend()

    save_figure(figure, "03_density_histogram_comparison.png")


# ---------------------------------------------------------------------------
# 4. Distribution shapes
# ---------------------------------------------------------------------------

def demonstrate_distributions() -> None:
    """
    Compare common distribution shapes.

    Normal distribution:
        Approximately symmetric and bell-shaped.

    Right-skewed distribution:
        Long tail toward larger values.

    Bimodal distribution:
        Two distinct concentration regions.
    """
    normal_data = [random.gauss(50, 8) for _ in range(800)]

    right_skewed_data = [
        random.expovariate(1 / 15) + 20
        for _ in range(800)
    ]

    bimodal_data = (
        [random.gauss(35, 4) for _ in range(400)]
        + [random.gauss(65, 5) for _ in range(400)]
    )

    datasets = [
        ("Normal-like distribution", normal_data),
        ("Right-skewed distribution", right_skewed_data),
        ("Bimodal distribution", bimodal_data),
    ]

    for index, (title, data) in enumerate(datasets, start=1):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(data, bins=30, edgecolor="black", alpha=0.75)
        axis.axvline(mean(data), linestyle="--", linewidth=2, label="Mean")
        axis.axvline(median(data), linestyle=":", linewidth=2, label="Median")
        axis.set_title(title)
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")
        axis.legend()
        save_figure(figure, f"04_distribution_shape_{index}.png")

        print(f"\n{title}")
        print(f"  Mean   = {mean(data):.2f}")
        print(f"  Median = {median(data):.2f}")


# ---------------------------------------------------------------------------
# 5. Scatter plots
# ---------------------------------------------------------------------------

def demonstrate_scatter_plot() -> None:
    """
    Scatter plots visualize paired observations.

    They are especially useful for:
      - correlation,
      - clusters,
      - nonlinear patterns,
      - heteroscedasticity,
      - outliers,
      - possible causal hypotheses.

    A scatter plot alone does not prove causation.
    """
    study_hours = [1, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 7, 8, 9]
    exam_scores = [48, 52, 56, 60, 63, 66, 71, 73, 78, 82, 88, 91]

    correlation = pearson_correlation(study_hours, exam_scores)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(study_hours, exam_scores, s=55, alpha=0.8)

    axis.set_title(f"Study Hours vs Exam Score (r = {correlation:.3f})")
    axis.set_xlabel("Study hours")
    axis.set_ylabel("Exam score")
    axis.grid(alpha=0.2)

    save_figure(figure, "05_scatter_basic.png")

    print("\nScatter plot:")
    print(f"  Pearson correlation: {correlation:.3f}")
    print("  A strong correlation describes association, not causation.")


def demonstrate_scatter_edge_cases() -> None:
    """Demonstrate why correlation must be interpreted with the visual pattern."""
    x = list(range(1, 11))

    linear = [2 * value + random.gauss(0, 1) for value in x]
    curved = [(value - 5.5) ** 2 + random.gauss(0, 1) for value in x]
    clustered_x = [1, 1.5, 2, 2.5, 8, 8.5, 9, 9.5]
    clustered_y = [10, 11, 9, 10, 30, 29, 31, 30]

    datasets = [
        ("Approximately linear", x, linear),
        ("Nonlinear relationship", x, curved),
        ("Separated clusters", clustered_x, clustered_y),
    ]

    for index, (title, x_values, y_values) in enumerate(datasets, start=1):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(x_values, y_values, s=55)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(alpha=0.2)
        save_figure(figure, f"06_scatter_case_{index}.png")


# ---------------------------------------------------------------------------
# 6. Box plots
# ---------------------------------------------------------------------------

def demonstrate_box_plot() -> None:
    """
    A standard box plot typically displays:

        lower whisker
        Q1
        median
        Q3
        upper whisker
        potential outlier points

    The exact whisker rule matters. Matplotlib's default uses 1.5 IQR
    as the distance beyond which observations are displayed as fliers.
    """
    department_a = [42, 44, 45, 46, 47, 48, 49, 50, 52, 54, 55]
    department_b = [35, 38, 40, 42, 44, 46, 47, 48, 50, 52, 75]
    department_c = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.boxplot(
        [department_a, department_b, department_c],
        labels=["Department A", "Department B", "Department C"],
        showmeans=True,
    )

    axis.set_title("Box Plot Comparison")
    axis.set_ylabel("Measurement")

    save_figure(figure, "07_box_plot.png")

    print("\nBox plot outlier example:")
    print("  Department B contains an intentionally high observation:", 75)
    print("  Potential outliers should be investigated rather than automatically removed.")


def demonstrate_box_plot_group_comparison() -> None:
    """Compare several distributions compactly."""
    groups = {
        "North": [random.gauss(70, 5) for _ in range(100)],
        "South": [random.gauss(75, 7) for _ in range(100)],
        "East": [random.gauss(68, 4) for _ in range(100)],
        "West": [random.gauss(82, 10) for _ in range(100)],
    }

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(
        list(groups.values()),
        labels=list(groups.keys()),
        showmeans=True,
    )
    axis.set_title("Distribution Comparison Across Regions")
    axis.set_xlabel("Region")
    axis.set_ylabel("Observed value")

    save_figure(figure, "08_box_plot_groups.png")


# ---------------------------------------------------------------------------
# 7. Line plots
# ---------------------------------------------------------------------------

def demonstrate_line_plot() -> None:
    """
    Line plots are appropriate when observations have meaningful order.

    Connecting arbitrary categories with lines can incorrectly imply
    continuity. Time series are the classic use case.
    """
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    revenue = [120, 128, 133, 142, 150, 161, 158, 170, 176, 184, 195, 208]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        months,
        revenue,
        marker="o",
        linewidth=2,
        label="Revenue",
    )
    axis.set_title("Monthly Revenue")
    axis.set_xlabel("Month")
    axis.set_ylabel("Revenue")
    axis.grid(alpha=0.2)
    axis.legend()

    save_figure(figure, "09_line_plot.png")


def demonstrate_multiple_line_series() -> None:
    """Compare several ordered series using a shared x-axis."""
    months = list(range(1, 13))

    actual = [100, 105, 110, 114, 118, 125, 121, 130, 137, 142, 149, 158]
    target = [102, 106, 111, 116, 120, 124, 128, 132, 136, 140, 145, 150]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(months, actual, marker="o", label="Actual")
    axis.plot(months, target, marker="s", linestyle="--", label="Target")

    axis.set_title("Actual vs Target Over Time")
    axis.set_xlabel("Month number")
    axis.set_ylabel("Metric")
    axis.legend()
    axis.grid(alpha=0.2)

    save_figure(figure, "10_line_comparison.png")


# ---------------------------------------------------------------------------
# 8. Confidence and uncertainty visualization
# ---------------------------------------------------------------------------

def demonstrate_uncertainty_band() -> None:
    """
    An uncertainty band communicates a central estimate together with
    variability. Here we use a simulated mean +/- standard deviation band.

    Standard deviation is not automatically a confidence interval. The band
    is explicitly labeled as variability to avoid statistical ambiguity.
    """
    x_values = list(range(1, 21))
    observations = []

    for x_value in x_values:
        observations.append(
            [100 + 2 * x_value + random.gauss(0, 5) for _ in range(30)]
        )

    central = [mean(values) for values in observations]
    spread = [
        stdev(values)
        for values in observations
    ]

    lower = [m - s for m, s in zip(central, spread)]
    upper = [m + s for m, s in zip(central, spread)]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(x_values, central, marker="o", label="Mean")
    axis.fill_between(
        x_values,
        lower,
        upper,
        alpha=0.2,
        label="Mean ± 1 standard deviation",
    )

    axis.set_title("Trend with Variability Band")
    axis.set_xlabel("Time")
    axis.set_ylabel("Measurement")
    axis.legend()
    axis.grid(alpha=0.2)

    save_figure(figure, "11_uncertainty_band.png")


# ---------------------------------------------------------------------------
# 9. Combining visualizations responsibly
# ---------------------------------------------------------------------------

def demonstrate_distribution_dashboard() -> None:
    """
    A compact analytical dashboard can show complementary views:

      Histogram -> distribution shape
      Box plot  -> robust summary and outliers
      Scatter   -> relationship
      Line      -> temporal behavior

    Each plot answers a different analytical question.
    """
    values = [random.gauss(70, 8) for _ in range(150)]
    x_values = list(range(1, 151))
    trend_values = [65 + 0.08 * x + random.gauss(0, 5) for x in x_values]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(values, bins=20, edgecolor="black")
    axes[0, 0].set_title("Histogram")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].boxplot(values, vert=True)
    axes[0, 1].set_title("Box Plot")
    axes[0, 1].set_ylabel("Value")

    axes[1, 0].scatter(x_values, trend_values, s=12, alpha=0.6)
    axes[1, 0].set_title("Scatter Plot")
    axes[1, 0].set_xlabel("Observation")
    axes[1, 0].set_ylabel("Value")

    axes[1, 1].plot(x_values, trend_values, linewidth=1.5)
    axes[1, 1].set_title("Line Plot")
    axes[1, 1].set_xlabel("Observation")
    axes[1, 1].set_ylabel("Value")

    save_figure(figure, "12_distribution_dashboard.png")


# ---------------------------------------------------------------------------
# 10. Reusable visualization functions
# ---------------------------------------------------------------------------

def plot_histogram(
    values: Sequence[float],
    title: str,
    filename: str,
    bins: int | Sequence[float] = 20,
) -> Path:
    """Reusable histogram function with input validation."""
    if not values:
        raise ValueError("Histogram data cannot be empty.")

    if isinstance(bins, int) and bins <= 0:
        raise ValueError("bins must be a positive integer.")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=bins, edgecolor="black", alpha=0.75)
    axis.set_title(title)
    axis.set_xlabel("Value")
    axis.set_ylabel("Frequency")

    return save_figure(figure, filename)


def plot_scatter(
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    filename: str,
) -> Path:
    """Reusable scatter plot with paired-data validation."""
    if len(x_values) != len(y_values):
        raise ValueError("Scatter plot x and y arrays must have equal length.")
    if not x_values:
        raise ValueError("Scatter plot data cannot be empty.")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(x_values, y_values, s=45, alpha=0.75)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.grid(alpha=0.2)

    return save_figure(figure, filename)


def plot_boxplot(
    groups: dict[str, Sequence[float]],
    title: str,
    filename: str,
) -> Path:
    """Reusable grouped box plot."""
    if not groups:
        raise ValueError("At least one group is required.")

    for group_name, values in groups.items():
        if not values:
            raise ValueError(f"Group {group_name!r} cannot be empty.")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(
        list(groups.values()),
        labels=list(groups.keys()),
        showmeans=True,
    )
    axis.set_title(title)
    axis.set_ylabel("Value")

    return save_figure(figure, filename)


def plot_line(
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    filename: str,
) -> Path:
    """Reusable ordered-data line plot."""
    if len(x_values) != len(y_values):
        raise ValueError("Line plot x and y arrays must have equal length.")
    if not x_values:
        raise ValueError("Line plot data cannot be empty.")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x_values, y_values, marker="o")
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.grid(alpha=0.2)

    return save_figure(figure, filename)


# ---------------------------------------------------------------------------
# 11. Data cleaning and validation
# ---------------------------------------------------------------------------

def clean_numeric_data(values: Iterable[object]) -> list[float]:
    """
    Convert numeric-like values into floats while rejecting invalid values.

    Missing values such as None, NaN, and infinity are excluded. In a real
    production pipeline, silently dropping data may be inappropriate, so the
    cleaning policy should be documented and audited.
    """
    cleaned: list[float] = []

    for raw_value in values:
        if raw_value is None:
            continue

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric_value):
            cleaned.append(numeric_value)

    return cleaned


def demonstrate_cleaning() -> None:
    raw_values: list[object] = [
        "10",
        12,
        15.5,
        None,
        "invalid",
        float("nan"),
        float("inf"),
        20,
    ]

    cleaned = clean_numeric_data(raw_values)

    print("\nData cleaning:")
    print("  Raw values:", raw_values)
    print("  Clean values:", cleaned)

    plot_histogram(
        cleaned,
        "Histogram After Numeric Validation",
        "13_cleaned_data_histogram.png",
        bins=5,
    )


# ---------------------------------------------------------------------------
# 12. Performance considerations
# ---------------------------------------------------------------------------

def benchmark_basic_operations() -> None:
    """
    Visualization performance is influenced by:

      - number of observations,
      - number of graphical objects,
      - image resolution,
      - transparency,
      - marker complexity,
      - interactive redraw frequency.

    Aggregating millions of points before rendering can be more effective than
    drawing every raw point.
    """
    large_dataset = [
        random.gauss(100, 15)
        for _ in range(100_000)
    ]

    # A histogram compresses many observations into a small number of bins.
    # Rendering 100,000 individual bars or markers would be substantially less
    # efficient than rendering approximately 50 histogram rectangles.
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(large_dataset, bins=50, edgecolor="black")
    axis.set_title("Large Dataset Aggregated into 50 Histogram Bins")
    axis.set_xlabel("Value")
    axis.set_ylabel("Frequency")

    save_figure(figure, "14_large_dataset_histogram.png")

    print("\nPerformance principle:")
    print("  100,000 observations were represented using 50 histogram bins.")
    print("  Aggregation can reduce rendering complexity substantially.")


# ---------------------------------------------------------------------------
# 13. Common mistakes demonstration
# ---------------------------------------------------------------------------

def demonstrate_common_mistakes() -> None:
    """
    This section intentionally discusses problematic visualization patterns.

    Common mistakes include:
      1. misleading axis truncation,
      2. arbitrary histogram bins,
      3. confusing counts with density,
      4. treating correlation as causation,
      5. connecting unordered categories with lines,
      6. hiding outliers,
      7. overloading a chart with decoration,
      8. using 3D effects that obscure values,
      9. using dual axes without a strong analytical reason,
      10. omitting units and labels.
    """
    categories = ["A", "B", "C", "D"]
    values = [98, 99, 100, 101]

    # Full scale provides an honest visual baseline for this simple comparison.
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(categories, values)
    axis.set_ylim(0, 110)
    axis.set_title("Bar Comparison with Honest Baseline")
    axis.set_xlabel("Category")
    axis.set_ylabel("Measured value")
    save_figure(figure, "15_honest_axis.png")

    print("\nVisualization warning:")
    print("  Axis limits can dramatically change visual perception.")
    print("  Choose limits that preserve interpretability and do not exaggerate differences.")


# ---------------------------------------------------------------------------
# 14. Advanced analytical example
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A small domain model used to keep paired observations together."""
    time: int
    temperature: float
    energy_usage: float


def create_energy_dataset(size: int = 120) -> list[Observation]:
    """
    Simulate a realistic operational dataset.

    Energy consumption depends partly on temperature, with random variation.
    This is a simulation, not evidence of a real physical relationship.
    """
    observations: list[Observation] = []

    for time_index in range(size):
        temperature = 18 + 0.05 * time_index + random.gauss(0, 2)
        energy_usage = (
            200
            + 3.5 * abs(temperature - 22)
            + random.gauss(0, 8)
        )

        observations.append(
            Observation(
                time=time_index,
                temperature=temperature,
                energy_usage=energy_usage,
            )
        )

    return observations


def analyze_energy_dataset() -> None:
    observations = create_energy_dataset()

    times = [item.time for item in observations]
    temperatures = [item.temperature for item in observations]
    energy = [item.energy_usage for item in observations]

    correlation = pearson_correlation(temperatures, energy)
    statistics = describe(energy)
    outliers = detect_iqr_outliers(energy)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(
        temperatures,
        energy,
        s=35,
        alpha=0.65,
    )
    axis.set_title(
        f"Temperature vs Energy Usage (r = {correlation:.3f})"
    )
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Energy usage")
    axis.grid(alpha=0.2)

    save_figure(figure, "16_energy_scatter.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, energy, linewidth=1.4)
    axis.set_title("Energy Usage Over Time")
    axis.set_xlabel("Time index")
    axis.set_ylabel("Energy usage")
    axis.grid(alpha=0.2)

    save_figure(figure, "17_energy_time_series.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(energy, showmeans=True)
    axis.set_title("Energy Usage Distribution")
    axis.set_ylabel("Energy usage")

    save_figure(figure, "18_energy_boxplot.png")

    print("\nAdvanced dataset analysis:")
    print(f"  Observations: {len(observations)}")
    print(f"  Mean energy: {statistics['mean']:.2f}")
    print(f"  Median energy: {statistics['median']:.2f}")
    print(f"  IQR: {statistics['iqr']:.2f}")
    print(f"  Potential IQR outliers: {len(outliers)}")
    print(f"  Temperature-energy correlation: {correlation:.3f}")


# ---------------------------------------------------------------------------
# 15. Automated validation and tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Simple assertions that validate the educational calculations."""
    values = [1, 2, 3, 4, 5]

    assert quantile(values, 0.0) == 1
    assert quantile(values, 0.5) == 3
    assert quantile(values, 1.0) == 5

    statistics = describe(values)
    assert statistics["mean"] == 3.0
    assert statistics["median"] == 3.0
    assert statistics["iqr"] == 2.0

    correlation = pearson_correlation(
        [1, 2, 3, 4],
        [2, 4, 6, 8],
    )
    assert math.isclose(correlation, 1.0)

    assert clean_numeric_data(
        ["1", None, "bad", 2, float("nan")]
    ) == [1.0, 2.0]

    try:
        pearson_correlation([1, 2], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected mismatched lengths to raise ValueError.")

    try:
        quantile([], 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected empty quantile input to raise ValueError.")

    print("\nTests: all assertions passed.")


# ---------------------------------------------------------------------------
# 16. Educational execution
# ---------------------------------------------------------------------------

def main() -> None:
    print_fundamentals()

    print("\n" + "=" * 78)
    print("RUNNING VISUALIZATION EXAMPLES")
    print("=" * 78)

    demonstrate_histogram()
    demonstrate_histogram_bin_choices()
    demonstrate_density_histogram()
    demonstrate_distributions()
    demonstrate_scatter_plot()
    demonstrate_scatter_edge_cases()
    demonstrate_box_plot()
    demonstrate_box_plot_group_comparison()
    demonstrate_line_plot()
    demonstrate_multiple_line_series()
    demonstrate_uncertainty_band()
    demonstrate_distribution_dashboard()
    demonstrate_cleaning()
    benchmark_basic_operations()
    demonstrate_common_mistakes()
    analyze_energy_dataset()
    run_tests()

    print("\n" + "=" * 78)
    print("OUTPUT")
    print("=" * 78)
    print(f"Charts have been saved to: {OUTPUT_DIR.resolve()}")
    print("The generated files illustrate histograms, scatter plots, box plots,")
    print("line plots, distributions, uncertainty, outliers, and comparisons.")


if __name__ == "__main__":
    main()
