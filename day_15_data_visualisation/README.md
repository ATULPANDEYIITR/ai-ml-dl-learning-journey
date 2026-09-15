# Data visualization: Histograms, scatter plots, box plots, line plots, and distributions

## Topic introduction

Data visualization converts numerical observations into visual structures that make patterns easier to inspect. A good visualization is not simply a chart that looks attractive. It is a representation designed around a specific analytical question.

This project studies five closely related areas of statistical visualization:

- Histograms
- Scatter plots
- Box plots
- Line plots
- Distributions

The three implementations approach the subject from different technical perspectives. The Python program uses Matplotlib to produce a broad collection of statistical visualizations. The JavaScript program implements the underlying statistical calculations and demonstrates browser-oriented rendering through SVG and Canvas. The C++ program develops an industry-style production analytics case study and generates a self-contained HTML dashboard containing SVG visualizations.

The examples use simulated data where appropriate. Simulated relationships should not be interpreted as empirical evidence about real-world systems.

## Fundamental terminology

### Observation

An observation is one recorded value or one row in a dataset.

For example, if five machines report temperatures of 21, 22, 24, 23, and 25 degrees, each measurement is an observation.

### Variable

A variable is a characteristic measured across observations.

Examples include:

- temperature
- revenue
- age
- production output
- exam score
- defect rate
- time

A variable may be numeric or categorical. The visualizations in this project primarily concern numeric variables.

### Distribution

A distribution describes how values are arranged across their possible range.

A distribution can reveal:

- central tendency
- variability
- skewness
- concentration
- gaps
- multiple modes
- extreme observations

The distribution is a central concept because histograms and box plots provide different visual summaries of the same underlying phenomenon.

### Frequency

Frequency is the number of observations belonging to a particular value or interval.

For example, if 20 observations fall between 50 and 60, that interval has a frequency of 20.

### Bin

A bin is an interval used by a histogram to group continuous numeric observations.

If a dataset ranges from 0 to 100 and is divided into ten equal intervals, each interval has a width of 10.

The number and width of bins can materially affect the appearance of a histogram.

### Mean

The arithmetic mean is calculated as:

`mean = sum of observations / number of observations`

The mean uses every observation and is sensitive to extreme values.

### Median

The median is the middle value after sorting observations.

When there are an even number of observations, the median is normally calculated from the two central values.

The median is generally less sensitive to extreme observations than the mean.

### Quartiles

Quartiles divide an ordered dataset into four sections.

The important values for a box plot are:

- Q1: 25th percentile
- Q2: 50th percentile, or median
- Q3: 75th percentile

The exact quantile calculation can vary between statistical software packages. The Python, JavaScript, and C++ implementations use a linearly interpolated position based on `p * (n - 1)`.

### Interquartile range

The interquartile range is:

`IQR = Q3 - Q1`

It represents the width of the middle 50 percent of the data.

The IQR is relatively robust against extreme values and is central to the common box-plot outlier rule.

## Core visualization principles

A visualization should answer a recognizable question.

Examples include:

- How are values distributed?
- Are two variables related?
- Which group has greater variability?
- How has a metric changed over time?
- Are unusual observations present?
- Are two distributions different?

Chart selection should follow the structure of the data rather than personal preference.

A visualization should also provide enough context to be interpreted correctly. Units, labels, meaningful scales, titles, and legends are important when they materially improve interpretation.

## Histograms

A histogram represents the distribution of a numeric variable by dividing its range into intervals and counting observations within each interval.

The Python implementation uses Matplotlib's `hist()` function.

A basic histogram demonstrates:

- the range of the observations
- concentration of values
- approximate shape of the distribution
- possible skewness
- possible multiple modes
- potential gaps

A histogram is not equivalent to a bar chart. Histograms normally represent continuous numeric intervals, while bar charts commonly represent categorical groups. Histogram bars therefore represent adjacent numeric intervals and normally touch.

### Histogram bin selection

Bin selection is an important analytical decision.

Too few bins can hide structure. Two distinct groups may appear as one broad distribution.

Too many bins can make random variation appear to be meaningful structure.

The appropriate choice depends on:

- sample size
- measurement precision
- distribution shape
- analytical objective
- comparison requirements

The Python implementation deliberately generates histograms using 5, 10, 20, and 40 bins to demonstrate how the same dataset can appear visually different under different bin counts.

### Frequency versus density

A histogram using ordinary counts displays the number of observations in each bin.

A density histogram normalizes the vertical scale so that the total area is approximately one.

Density histograms are useful when comparing distributions with different sample sizes.

The JavaScript and C++ implementations focus on explicit histogram aggregation, while the Python implementation also demonstrates density histograms using Matplotlib.

### Distribution shape

Common distribution shapes include:

- approximately symmetric distributions
- right-skewed distributions
- left-skewed distributions
- bimodal distributions
- multimodal distributions
- approximately uniform distributions
- distributions with heavy tails

A normal-like distribution is approximately symmetric around its center.

A right-skewed distribution has a longer tail toward larger values.

A bimodal distribution has two prominent concentration regions. Such a pattern can indicate that observations from different underlying populations have been combined.

### Mean and median in skewed distributions

For a symmetric distribution, the mean and median may be close.

In a right-skewed distribution, a small number of high values can pull the mean upward.

This makes comparison of mean and median useful when examining distribution shape.

Neither statistic alone provides a complete description of the distribution.

## Scatter plots

A scatter plot represents paired observations using an x-coordinate and a y-coordinate.

It is useful for examining relationships between two numeric variables.

Examples include:

- study hours versus exam score
- advertising expenditure versus sales
- temperature versus energy consumption
- production hours versus output
- age versus income

The Python implementation calculates Pearson correlation for the paired observations and displays it in the chart title.

The JavaScript implementation explicitly validates that x and y arrays have equal lengths before calculating or rendering a scatter plot.

The C++ case study uses a scatter plot to investigate the relationship between machine temperature and defect rate.

### Correlation

Pearson correlation measures the strength and direction of a linear relationship.

Its value ranges from approximately `-1` to `+1`.

Interpretation is contextual, but broadly:

- values near `+1` indicate strong positive linear association
- values near `-1` indicate strong negative linear association
- values near `0` indicate weak linear association

Correlation does not establish causation.

A high correlation can arise because:

- one variable influences another
- both variables respond to another variable
- the data were selected in a particular way
- the relationship is coincidental
- the observations are not independent

### Nonlinear relationships

Pearson correlation is specifically about linear association.

A strong nonlinear relationship can have a relatively small Pearson correlation.

This is why a scatter plot should be examined alongside the numerical correlation value.

The Python implementation includes a nonlinear example to demonstrate this distinction.

### Clusters

Clusters occur when observations form distinct groups.

Clusters may represent:

- different customer segments
- geographic groups
- machine types
- experimental groups
- different operating regimes

A single correlation value can hide such structure.

A scatter plot makes clusters visually visible.

### Outliers in scatter plots

An outlier may substantially influence correlation and regression results.

A potentially unusual observation should be investigated for:

- measurement error
- data-entry error
- unusual but valid behavior
- a different population
- a genuine exceptional event

Removing an observation solely because it looks inconvenient is poor analytical practice.

## Box plots

A box plot provides a compact distribution summary.

The main components are:

- minimum or lower whisker
- Q1
- median
- Q3
- maximum or upper whisker
- potential outliers

The box itself extends from Q1 to Q3.

The line inside the box represents the median.

The height or width of the box represents the IQR.

### The 1.5 IQR rule

A common outlier rule defines:

`lower fence = Q1 - 1.5 × IQR`

`upper fence = Q3 + 1.5 × IQR`

Observations outside those fences are often displayed as potential outliers.

This rule is descriptive. It does not prove that an observation is invalid.

A valid observation can be statistically unusual.

### Why box plots are useful

Box plots are particularly useful when comparing several groups.

For example, four departments can be compared using:

- median level
- middle-50-percent spread
- potential outliers
- overall distribution position

They require much less visual space than four separate histograms.

### Limitations of box plots

Box plots hide substantial distribution detail.

Two distributions can have similar medians, quartiles, and whiskers while having very different internal shapes.

A histogram or another distribution-oriented visualization may therefore be preferable when multimodality or detailed shape matters.

## Line plots

Line plots connect observations in a meaningful order.

Time series are the most common example.

The Python implementation demonstrates monthly revenue and actual-versus-target series.

Line plots can show:

- trends
- seasonality
- abrupt changes
- growth
- decline
- repeated cycles
- deviations from a target

### Ordering matters

A line implies continuity.

Connecting unrelated categories with a line can create a false impression of a continuous process.

For example, connecting arbitrary product names with a line does not normally make analytical sense.

Line plots are appropriate when the x-axis represents an ordered dimension such as:

- time
- sequence
- distance
- experimental progression

### Multiple line series

Multiple lines can compare several related series.

The chart should clearly distinguish the series using labels or other visual encodings.

Too many simultaneous lines can create visual clutter. When many series must be compared, alternative approaches such as small multiples or aggregation may be more appropriate.

## Distributions and descriptive statistics

The Python program provides a reusable `describe()` function that calculates:

- count
- minimum
- Q1
- median
- mean
- Q3
- maximum
- IQR
- sample standard deviation

This creates a bridge between numerical statistics and graphical representation.

A visualization is more useful when the analyst understands the statistics underlying the graphical elements.

For example:

- the histogram shows distribution shape
- the box plot emphasizes quartiles and potential outliers
- the line plot emphasizes order
- the scatter plot emphasizes paired relationships

These are complementary rather than interchangeable representations.

## Python implementation

The Python script uses Matplotlib and standard-library functionality.

The program is intentionally organized into reusable functions rather than placing all plotting logic inside one large execution block.

Important functions include:

- `quantile()`
- `describe()`
- `detect_iqr_outliers()`
- `pearson_correlation()`
- `plot_histogram()`
- `plot_scatter()`
- `plot_boxplot()`
- `plot_line()`

The program also demonstrates simulated datasets for:

- approximately normal data
- right-skewed data
- bimodal data
- study-hour and exam-score relationships
- regional group comparisons
- time-series trends
- operational energy data

### Python output

The script creates a `visualization_output` directory and saves the generated charts as PNG files.

The use of the Matplotlib `Agg` backend means the script does not require an interactive graphical desktop. This makes the implementation suitable for automated execution, servers, containers, and CI environments.

## JavaScript implementation

The JavaScript file takes a different approach.

Instead of depending on an external visualization library, it implements the statistical calculations directly and demonstrates visualization-oriented rendering through SVG and browser Canvas.

This provides a clearer view of the underlying mechanics of data visualization.

Important functions include:

- `mean()`
- `median()`
- `quantile()`
- `standardDeviation()`
- `describe()`
- `iqrOutliers()`
- `pearsonCorrelation()`
- `createHistogram()`
- `calculateBoxPlot()`
- `renderHistogramSvg()`
- `renderScatterSvg()`
- `renderLineSvg()`

### Why SVG is useful

SVG represents visual elements as structured XML objects.

A histogram can therefore be represented by multiple `<rect>` elements.

A scatter plot can use multiple `<circle>` elements.

A line plot can use a `<polyline>`.

This approach makes the mapping between data and visual objects explicit.

### Why Canvas is useful

Canvas uses a drawing surface controlled through JavaScript.

It is useful for:

- custom visualizations
- browser dashboards
- animation
- high-frequency drawing
- interactive applications

The JavaScript implementation includes a Canvas example that converts numeric values into bar heights.

The choice between SVG and Canvas depends on application requirements. SVG is often convenient when individual visual elements need to remain addressable as document objects. Canvas can be effective when large numbers of drawing operations or custom rendering are required.

## C++ case study

The C++ program models a manufacturing operations analytics system.

Each `ProductionRecord` contains:

- day
- production hours
- machine temperature
- defect rate
- units produced

The program creates 120 simulated daily records and validates each record before adding it to the dataset.

One intentionally unusual defect-rate observation is inserted to demonstrate the behavior of outlier detection.

### Problem being modeled

The analytical team wants to understand:

1. What does the defect-rate distribution look like?
2. Are there potential defect-rate outliers?
3. Is machine temperature associated with defect rate?
4. How does daily production output change across observations?
5. Is production output associated with production hours?

Different visualization types answer different questions.

### System architecture

The C++ implementation follows this structure:

`ProductionRecord`

↓

`Data extraction`

↓

`Statistical analysis`

↓

`Visualization-ready structures`

↓

`SVG generation`

↓

`HTML dashboard`

This separation prevents visualization logic from becoming tightly coupled to the raw data model.

### Data validation

The program validates:

- day values
- production hours
- finite temperature values
- defect-rate bounds
- non-negative production output

Invalid values generate exceptions.

Validation is particularly important in production analytics because incorrect input can produce visually convincing but statistically meaningless charts.

### Histogram implementation

The C++ `createHistogram()` function:

1. Finds the minimum and maximum.
2. Calculates the bin width.
3. Creates the requested bins.
4. Assigns each observation to a bin.
5. Handles the maximum-value boundary explicitly.
6. Handles constant-valued datasets separately.

The maximum-value boundary requires care because the mathematical calculation can produce an index equal to the number of bins.

### Box plot implementation

The C++ `calculateBoxPlot()` function computes:

- Q1
- median
- Q3
- IQR
- lower fence
- upper fence
- lower whisker
- upper whisker
- outliers

It uses the same 1.5 IQR rule used in the other implementations.

### Scatter plot implementation

The C++ program extracts temperature and defect-rate vectors from the production records.

Pearson correlation is then calculated.

The scatter plot shows the individual paired observations.

The numerical correlation is useful, but the scatter plot remains important because correlation alone cannot reveal every structure in the data.

### Line plot implementation

The daily production output is represented as an ordered sequence.

The line plot makes changes across observation order visible.

This is different from a histogram, which intentionally removes observation order and concentrates on the distribution.

### Dashboard

The C++ program writes `production_visualization_dashboard.html`.

The dashboard contains:

- defect-rate histogram
- temperature-versus-defect scatter plot
- daily production line plot
- descriptive statistics
- IQR information
- potential outlier count

The charts are rendered as SVG strings inside an HTML document.

No external visualization library is required.

## Important distinctions

| Visualization | Primary question | Typical input | Main strength |
|---|---|---|---|
| Histogram | How are numeric values distributed? | One numeric variable | Distribution shape |
| Scatter plot | How are two numeric variables related? | Two paired numeric variables | Relationship structure |
| Box plot | How do distributions compare compactly? | One or more numeric groups | Quartiles and spread |
| Line plot | How does an ordered metric change? | Ordered x and numeric y | Trends and temporal behavior |
| Distribution statistics | What numerical properties describe the data? | Numeric variable | Quantitative summary |

## Histogram versus box plot

A histogram provides much more information about distribution shape.

A box plot provides a compact summary.

A histogram can reveal:

- bimodality
- gaps
- skewness
- concentration patterns

A box plot emphasizes:

- median
- quartiles
- IQR
- potential outliers

For detailed distribution analysis, a histogram is generally more informative.

For compact comparison across many groups, box plots are often more efficient.

## Histogram versus line plot

A histogram ignores the original observation order.

A line plot depends on order.

If daily temperatures are plotted over time, the sequence is analytically meaningful.

If those same temperatures are plotted as a histogram, their order is discarded and their distribution becomes the focus.

## Scatter plot versus line plot

A scatter plot shows paired relationships without necessarily implying continuity.

A line plot connects observations and therefore implies meaningful ordering.

The same numeric values can sometimes appear in both forms, but the interpretation is different.

## Mean versus median

The mean incorporates every value and can be strongly influenced by extreme observations.

The median depends on the ordering of values and is generally more robust to outliers.

For skewed distributions, reporting both can be useful.

## Correlation versus causation

A scatter plot may show that two variables move together.

Pearson correlation quantifies linear association.

Neither establishes that one variable causes the other.

A causal conclusion requires an appropriate research or experimental design and consideration of confounding variables.

## Edge cases

The implementations explicitly address several important edge cases.

### Empty data

An empty dataset cannot produce a meaningful histogram, box plot, mean, median, or quantile.

The implementations raise validation errors rather than producing misleading output.

### Constant data

If all observations have the same value, a standard histogram calculation can produce a zero-width range.

The JavaScript and C++ implementations explicitly handle this situation by constructing a small artificial visual interval around the constant value.

### Mismatched scatter data

Each x observation must correspond to one y observation.

The JavaScript and C++ implementations reject unequal-length vectors.

### Constant variable in correlation

Pearson correlation is undefined when either variable has zero variance.

The implementations detect this condition rather than dividing by zero.

### Invalid numeric values

The JavaScript implementation filters non-finite values through `Number.isFinite()`.

The Python cleaning function rejects values such as NaN and infinity.

In a real production pipeline, invalid records should usually be logged and investigated rather than silently discarded.

### Extreme outliers

An extreme observation can substantially affect:

- mean
- standard deviation
- correlation
- regression
- axis scaling

An outlier should be investigated before deciding whether it represents an error or a valid rare event.

## Common mistakes

### Using inappropriate bins

Very large or very small bin counts can distort visual interpretation.

Bin selection should be treated as an analytical choice.

### Treating a histogram as a bar chart

Histograms represent numeric intervals. Bar charts generally represent categories.

### Connecting unrelated categories

A line implies meaningful ordering and continuity.

Connecting arbitrary categories can communicate a relationship that does not exist.

### Hiding outliers

Suppressing unusual observations can conceal important information.

Outliers may represent:

- data-quality problems
- rare events
- fraud
- system failures
- exceptional performance
- new operating conditions

### Assuming correlation proves causation

Correlation is an association measure.

A third variable may explain the relationship between two observed variables.

### Using truncated axes irresponsibly

Axis limits strongly influence visual perception.

The Python implementation includes an example showing a comparison using an honest baseline.

Axis truncation is not always wrong. It can be appropriate when carefully labeled and analytically justified, but it should not be used to exaggerate differences.

### Overloading a chart

Too many colors, labels, lines, markers, annotations, and decorative elements can make a chart harder to interpret.

Every visual element should serve an analytical purpose.

## Limitations

### Histograms

Histograms depend on binning choices.

They can hide fine-scale patterns or create apparent structure from random variation.

### Scatter plots

Scatter plots can become difficult to interpret with extremely large datasets because points overlap.

Aggregation, transparency, sampling, hexbin-style approaches, or density representations can help.

### Box plots

Box plots hide many details of the underlying distribution.

Two very different distributions can have similar box-plot summaries.

### Line plots

Line plots can imply continuity where the underlying process is not continuous.

They are also vulnerable to visual clutter when many series are plotted simultaneously.

### Pearson correlation

Pearson correlation measures linear association.

It can fail to represent nonlinear relationships accurately and can be affected by outliers.

### Simulated data

The examples in this project use generated datasets.

They demonstrate techniques rather than establishing conclusions about real populations.

## Performance considerations

Visualization performance depends on more than the number of observations.

Important factors include:

- number of graphical elements
- rendering backend
- output resolution
- marker size
- transparency
- interactive redraw frequency
- browser rendering
- data aggregation

A histogram is naturally efficient for large datasets because thousands or millions of observations can be represented by a relatively small number of bins.

For example, the Python and JavaScript implementations demonstrate representing a large dataset using a limited number of histogram bins.

Scatter plots can become expensive when every observation is represented by a separate graphical object.

Large datasets may require:

- aggregation
- sampling
- density estimation
- rasterization
- level-of-detail techniques
- server-side preprocessing

The correct optimization depends on whether preserving every observation visually is necessary.

## Computational complexity

The main analytical operations have different computational costs.

For a dataset of size `n`:

- Mean: `O(n)`
- Standard deviation: `O(n)`
- Pearson correlation: `O(n)`
- Histogram construction: approximately `O(n + b)`, where `b` is the number of bins
- Quantile calculation in these implementations: `O(n log n)` because the data are sorted
- Box-plot calculation: approximately `O(n log n)` because quantiles are calculated using sorting

The C++ implementation explicitly documents these complexities.

A production implementation could use selection algorithms or specialized statistical libraries to calculate quantiles more efficiently when appropriate.

## Security considerations

Data visualization systems can process sensitive information.

Potential security concerns include:

- exposing personally identifiable information
- exposing confidential business metrics
- embedding sensitive data in generated HTML
- unsafe HTML or SVG construction
- untrusted labels or metadata
- leaking raw records through downloadable charts

The JavaScript implementation includes XML escaping for SVG text.

The C++ implementation also escapes text inserted into generated SVG and HTML contexts.

In a production web application, escaping should be treated as part of a broader output-encoding strategy. Untrusted content should never be inserted into HTML or SVG without appropriate sanitization or encoding.

Access controls should also be applied to the underlying data, not merely to the visualization interface.

## Implementation considerations

A robust visualization pipeline can be separated into several stages:

`data acquisition`

→ `validation`

→ `cleaning`

→ `transformation`

→ `statistical analysis`

→ `visual encoding`

→ `rendering`

→ `quality verification`

This separation helps identify whether an unexpected result originates from the data, statistical transformation, or visualization layer.

The three implementations reflect this principle at different levels.

Python emphasizes analytical and statistical visualization.

JavaScript emphasizes browser-oriented rendering and explicit visual primitives.

C++ emphasizes system design, validation, statistical computation, and generation of a deployable dashboard artifact.

## Best practices

Use a visualization whose visual structure matches the analytical question.

Label axes with meaningful names and units.

Choose histogram bins deliberately.

Use scatter plots to inspect relationships before relying only on correlation coefficients.

Investigate outliers instead of automatically deleting them.

Use line plots when ordering is meaningful.

Use box plots when compact group comparison is valuable.

Report numerical statistics when they improve interpretation.

Avoid unnecessary decorative elements.

Avoid implying causation from correlation.

Validate data before visualization.

Keep preprocessing separate from rendering logic.

Use consistent definitions of statistics across an analytical workflow.

Document important methodological choices such as quantile definitions, outlier rules, aggregation procedures, and missing-value handling.

## Real-world applications

These visualization methods appear across many domains.

### Business analytics

Histograms can show transaction sizes.

Scatter plots can compare advertising expenditure and revenue.

Box plots can compare regional sales distributions.

Line plots can show monthly revenue.

### Finance

Histograms can show return distributions.

Scatter plots can compare risk and return.

Box plots can compare volatility across assets.

Line plots can display portfolio value over time.

### Manufacturing

Histograms can analyze defect rates.

Scatter plots can investigate machine settings versus defects.

Box plots can compare production lines.

Line plots can monitor daily output.

### Healthcare

Histograms can show patient measurements.

Scatter plots can examine relationships between clinical variables.

Box plots can compare treatment groups.

Line plots can track measurements over time.

### Education

Histograms can show examination-score distributions.

Scatter plots can examine study time versus scores.

Box plots can compare classes.

Line plots can track performance across assessments.

### Data science and machine learning

Histograms can reveal feature distributions.

Scatter plots can expose feature relationships.

Box plots can identify unusual feature values.

Line plots can monitor training and validation metrics over epochs.

Distribution analysis is also important for identifying data drift between training and production datasets.

## Practical interpretation workflow

A useful analytical workflow is to begin with the data structure rather than immediately selecting a chart.

For one numeric variable, start by examining its distribution.

A histogram can reveal the shape, while descriptive statistics provide numerical context.

A box plot can then provide a compact representation of central tendency and spread.

For two numeric variables, inspect a scatter plot.

If a linear relationship appears plausible, calculate a correlation coefficient as a numerical complement.

For ordered observations, especially time series, use a line plot.

When comparing groups, consider whether separate distributions, grouped box plots, or normalized histograms provide the clearest comparison.

The most informative analysis often uses several complementary views rather than relying on one chart.

## Files and execution

The Python implementation requires Matplotlib and can optionally use NumPy, although the included implementation primarily relies on the Python standard library and Matplotlib.

The Python program saves generated figures under `visualization_output`.

The JavaScript implementation requires no external package and can run in Node.js or a browser environment. In a browser, its Canvas demonstration uses the DOM and Canvas APIs.

The C++ implementation requires a modern compiler supporting C++17. It produces `production_visualization_dashboard.html`, which contains the generated SVG visualizations and statistical results.

## Technical correspondence between implementations

The Python program provides the broadest collection of ready-to-use statistical visualization examples.

The JavaScript program demonstrates how the same analytical concepts can be implemented closer to the browser rendering layer. It explicitly constructs histogram bins, box-plot statistics, SVG elements, and Canvas drawing operations.

The C++ program demonstrates how visualization can form one stage of a larger analytical system. It introduces domain modeling, validation, statistical computation, exception handling, complexity considerations, SVG generation, and HTML dashboard construction.

The implementations therefore represent three different levels of abstraction:

- Python: statistical analysis and high-level visualization
- JavaScript: application and browser rendering mechanics
- C++: lower-level analytical system design and visualization generation

The underlying statistical ideas remain the same even though the implementation mechanisms differ.
