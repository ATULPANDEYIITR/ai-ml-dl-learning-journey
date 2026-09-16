# Statistics for AI

## Topic

This project studies fundamental descriptive statistics used throughout data analysis, machine learning, artificial intelligence, experimentation, and model monitoring.

The implementations cover:

- Population
- Sample
- Parameter and statistic
- Mean
- Median
- Mode
- Range
- Deviation from the mean
- Variance
- Standard deviation
- Population variance
- Sample variance
- Bessel's correction
- Outlier sensitivity
- Z-score standardization
- Streaming statistics
- Welford's algorithm
- Numerical precision
- Statistical edge cases
- AI model metric analysis
- Grouped model monitoring
- Validation and testing

The three implementations deliberately use different programming environments to show how the same statistical ideas can be represented with Python, JavaScript, and modern C++.

## Statistical foundation

Statistics provides methods for describing, analyzing, and reasoning about data.

For AI systems, statistical methods are used when examining datasets, understanding features, detecting unusual observations, monitoring model behavior, evaluating experiments, and determining whether measurements are stable or highly variable.

Descriptive statistics are concerned primarily with describing observed data. They do not automatically establish causality, prove that a model is correct, or completely characterize a probability distribution.

## Population and sample

A **population** is the complete collection of observations relevant to a particular question.

For example, if a company wants to analyze every transaction made during a particular day and has records for all transactions, those transactions constitute the population for that analysis.

A **sample** is a subset of a population.

If a company has one million transactions but examines 10,000 selected transactions to estimate characteristics of all one million transactions, the 10,000 observations constitute a sample.

The distinction affects interpretation.

A population quantity is called a **parameter**. A quantity calculated from a sample is called a **statistic**.

For example:

- Population mean: parameter
- Sample mean: statistic
- Population variance: parameter
- Sample variance: statistic

In machine learning, a dataset is frequently treated as observed data from a broader data-generating process. The exact population being represented depends on the problem definition.

## Mean

The arithmetic mean is calculated as:

`mean = sum of observations / number of observations`

For observations `x₁, x₂, ..., xₙ`, the population mean is commonly written as:

`μ = Σxᵢ / N`

For a sample, the sample mean is commonly written as:

`x̄ = Σxᵢ / n`

The computational operation is the same. The notation and interpretation differ because one quantity describes a population while the other describes a sample.

The mean uses every observation. This makes it useful for describing the central location of numerical data, but it also makes it sensitive to extreme observations.

For example, consider:

`30,000, 32,000, 35,000, 36,000, 2,000,000`

The very large final observation substantially increases the mean.

This behavior is not inherently incorrect. It simply means that the mean represents a quantity that is sensitive to extreme values.

## Median

The median is the middle observation after sorting the data.

For an odd number of observations, there is one middle value.

For an even number of observations, the median is the arithmetic mean of the two middle values.

For example:

`1, 2, 3, 4, 5`

has median `3`.

For:

`1, 2, 3, 4`

the median is:

`(2 + 3) / 2 = 2.5`

The median is generally less sensitive to isolated extreme observations than the mean.

This makes the median useful for quantities such as income, transaction values, response times, and other measurements that may contain highly unusual observations.

The median requires ordering the observations in the implementations. A conventional sorting-based implementation has time complexity of approximately `O(n log n)`.

## Mode

The mode is the most frequently occurring observation.

A dataset can have:

- No repeated mode
- One mode
- Multiple modes

For:

`1, 2, 2, 3`

the mode is `2`.

For:

`1, 1, 2, 2, 3`

there are two modes: `1` and `2`.

If every observation occurs exactly once, the implementations return an empty collection to indicate that there is no repeated mode.

Mode is particularly useful for categorical data, where mean and median may not have meaningful interpretations.

## Range

The range is:

`maximum - minimum`

For:

`10, 20, 30, 40`

the range is:

`40 - 10 = 30`

Range is simple but highly sensitive to extreme observations.

It also ignores most of the information contained between the minimum and maximum.

For these reasons, range is normally considered alongside other measures rather than used as the only description of variability.

## Deviations from the mean

A deviation measures the distance of an observation from the mean:

`deviation = xᵢ - mean`

Some observations have positive deviations and others have negative deviations.

An important property of the arithmetic mean is that deviations from the mean sum to zero, apart from floating-point rounding effects in numerical computation.

This means that simply averaging signed deviations does not produce a useful measure of spread.

For example, positive and negative deviations cancel one another.

Variance solves this problem by squaring deviations.

## Variance

Variance measures the average squared distance from the mean.

For a population:

`σ² = Σ(xᵢ - μ)² / N`

The squared deviation has two important consequences:

1. Positive and negative deviations cannot cancel.
2. Large deviations receive disproportionately greater influence because they are squared.

If a value is twice as far from the mean, its squared deviation is four times as large.

Variance therefore measures spread but is expressed in squared units.

If the original variable is measured in milliseconds, variance is measured in squared milliseconds.

## Population variance

Population variance is used when the available observations represent the complete population of interest.

The Python implementation calculates it with `population_variance`.

The JavaScript implementation uses `populationVariance`.

The C++ implementation uses `populationVariance`.

For a population of five values:

`1, 2, 3, 4, 5`

the population mean is `3`.

The squared deviations are:

`4, 1, 0, 1, 4`

Their sum is `10`.

Therefore:

`population variance = 10 / 5 = 2`

## Sample variance

Sample variance is used when observations are treated as a sample from a larger population.

The commonly used unbiased estimator is:

`s² = Σ(xᵢ - x̄)² / (n - 1)`

The denominator is `n - 1`, not `n`.

The correction is known as **Bessel's correction**.

For:

`1, 2, 3, 4, 5`

the sum of squared deviations is `10`.

Population variance:

`10 / 5 = 2`

Sample variance:

`10 / 4 = 2.5`

The distinction is important because using the population formula when estimating population variability from a sample can systematically underestimate the variance under the standard sampling assumptions.

A conventional sample variance cannot be calculated from only one observation because its denominator would be zero.

## Standard deviation

Standard deviation is the square root of variance.

For a population:

`σ = √σ²`

For a sample:

`s = √s²`

Standard deviation has the same measurement unit as the original observations.

If latency is measured in milliseconds, standard deviation is also measured in milliseconds.

This makes standard deviation easier to interpret than variance in many practical applications.

A small standard deviation indicates that observations tend to be concentrated around the mean.

A larger standard deviation indicates greater dispersion.

Standard deviation should not be interpreted without considering the scale, distribution, measurement process, and context of the data.

## Mean versus median

Mean and median answer related but different descriptive questions.

| Property | Mean | Median |
|---|---|---|
| Uses every observation | Yes | No, directly |
| Requires ordering | No | Yes |
| Sensitive to extreme values | High | Lower |
| Useful for symmetric numerical data | Yes | Yes |
| Useful for heavily skewed data | With caution | Often useful |
| Represents arithmetic balance point | Yes | No |

Consider:

`48, 49, 50, 51, 500`

The mean is strongly affected by `500`.

The median remains `50`.

Neither statistic is universally superior. The appropriate measure depends on the data-generating process and analytical objective.

## Mean and variability must be considered together

A mean alone does not describe the spread of a dataset.

Consider:

`49, 50, 51, 50, 50`

and:

`20, 35, 50, 65, 80`

Both have mean `50`.

Their standard deviations are very different.

The first dataset is tightly concentrated around the mean.

The second dataset is substantially more dispersed.

This is why AI data analysis commonly examines central tendency and variability together.

## Outliers

An outlier is an observation that is unusually distant from other observations.

Outliers may arise from:

- Measurement errors
- Data-entry errors
- Sensor failures
- Fraudulent behavior
- Rare legitimate events
- System failures
- Genuine extreme observations

An outlier should not automatically be deleted.

Removing an observation simply because it is unusual can destroy useful information.

The correct interpretation depends on the origin of the observation and the purpose of the analysis.

Mean and standard deviation are sensitive to extreme observations.

Median is generally more resistant to isolated extremes.

This distinction is important when preparing AI datasets.

## Z-scores

A z-score expresses an observation relative to the mean in standard-deviation units.

The formula is:

`z = (x - mean) / standard deviation`

A z-score of `0` represents the mean.

A positive z-score represents an observation above the mean.

A negative z-score represents an observation below the mean.

For example, if the mean latency is `100 ms` and the standard deviation is `10 ms`, an observation of `120 ms` has:

`z = (120 - 100) / 10 = 2`

The implementations use z-scores to demonstrate one possible method for identifying unusually high observations.

A z-score threshold such as `2` or `3` should not automatically be treated as a universal anomaly rule. Its usefulness depends on distributional assumptions, operational requirements, false-positive costs, and the purpose of the monitoring system.

## Standardization in AI

Feature standardization often transforms observations using a formula based on the mean and standard deviation.

A common transformation is:

`z = (x - mean) / standard deviation`

After standardization, the transformed values have mean approximately zero and standard deviation approximately one when the same population-style calculations are used consistently.

Standardization can be useful for algorithms whose behavior depends on numerical scale.

Examples include certain optimization procedures, distance-based methods, and models sensitive to feature magnitudes.

It is not automatically required for every machine-learning algorithm.

Tree-based models, for example, often do not require conventional feature standardization for their basic operation.

## Constant data

Consider:

`7, 7, 7, 7`

The mean is `7`.

Every deviation from the mean is zero.

Therefore:

`variance = 0`

and:

`standard deviation = 0`

A z-score requires division by standard deviation. Therefore z-score standardization is undefined for constant data.

The Python and JavaScript implementations explicitly reject this situation.

## Empty datasets

An empty dataset contains no observations.

Mean, median, variance, and standard deviation cannot be calculated conventionally from an empty dataset.

All three implementations validate input and report an error instead of silently returning an arbitrary number.

Input validation is important because statistical functions are mathematical operations with domain requirements.

## Single-observation samples

For a dataset containing one observation:

`42`

the population variance is zero because the observed population has no variation.

Sample variance is different. The conventional estimator uses:

`n - 1`

and with `n = 1`, the denominator is zero.

The implementations therefore reject conventional sample variance for a single observation.

## Python implementation

The Python implementation is designed as a study-oriented statistical toolkit.

Important functions include:

- `population_mean`
- `sample_mean`
- `median`
- `modes`
- `data_range`
- `population_variance`
- `population_standard_deviation`
- `sample_variance`
- `sample_standard_deviation`
- `z_scores`
- `describe`

The `DescriptiveStatistics` dataclass groups related results into a reusable object.

Python's `Counter` is used for mode calculation because frequency counting is a natural mapping from observations to occurrence counts.

The `OnlineStatistics` class demonstrates Welford's algorithm and allows statistics to be updated one observation at a time.

The script also contains executable assertions that verify important mathematical results.

## JavaScript implementation

The JavaScript implementation emphasizes both statistical calculations and language-specific data-processing behavior.

Important functions include:

- `mean`
- `median`
- `modes`
- `populationVariance`
- `sampleVariance`
- `populationStandardDeviation`
- `sampleStandardDeviation`
- `zScores`
- `describe`

A JavaScript `Map` is used for frequency counting.

The implementation also demonstrates an important JavaScript-specific issue with numeric sorting.

Calling `sort()` without a comparator can perform lexicographic ordering because JavaScript converts values to strings during the default sort operation.

For numerical statistics, the appropriate pattern is represented by:

`values.sort((a, b) => a - b)`

The JavaScript implementation also uses:

- Classes
- Getters
- Async functions
- Async generators
- `for await...of`
- Error handling
- Object iteration
- Assertions

The asynchronous metric stream models the event-driven nature of JavaScript applications.

## Asynchronous model monitoring

The JavaScript implementation includes a simulated asynchronous metric stream.

`simulatedMetricStream` produces values over time.

`analyzeMetricStream` consumes those observations asynchronously and updates an `OnlineStatistics` instance.

This structure corresponds conceptually to applications in which model metrics arrive from:

- Web services
- Browser applications
- Event streams
- Telemetry systems
- Monitoring APIs
- Message-processing systems

The example is intentionally dependency-free. It demonstrates the statistical mechanism without requiring a particular production messaging system.

## C++ case study

The C++ implementation models an AI inference monitoring system.

The hypothetical service records inference latency in milliseconds.

The system calculates:

- Count
- Minimum
- Maximum
- Range
- Mean
- Median
- Mode
- Population variance
- Population standard deviation
- Sample variance
- Sample standard deviation

It also supports streaming statistics and high-latency anomaly identification.

The scenario demonstrates how descriptive statistics can become part of a larger operational system rather than existing only as isolated mathematical formulas.

## C++ system architecture

The case study is divided into several logical components.

### Validation

`validateData` verifies that the dataset is non-empty and that every observation is finite.

The monitoring service separately rejects negative latency values because negative elapsed time is not meaningful in the modeled domain.

### Central tendency

`mean` calculates arithmetic mean.

`median` copies and sorts the input so that the caller's original ordering is preserved.

`modes` calculates frequency counts and returns all values tied for the highest frequency.

### Variability

`populationVariance` implements the population formula.

`sampleVariance` implements the `n - 1` sample estimator.

The standard-deviation functions apply `std::sqrt` to the corresponding variance.

### Report structure

`StatisticsReport` stores multiple descriptive measurements in a single strongly typed structure.

The C++ `std::optional` type represents sample statistics that are unavailable for a one-observation dataset.

### Streaming statistics

`OnlineStatistics` implements Welford's algorithm.

Instead of storing every observation, it maintains:

- Observation count
- Running mean
- `M2`, the accumulated squared-deviation quantity

Each update requires constant time.

The state requires constant memory.

This is valuable for high-volume inference systems where retaining every measurement only for the purpose of calculating a mean or variance would be wasteful.

### Anomaly detection

`findHighLatencyAnomalies` calculates a z-score for every latency measurement.

The example uses an operational threshold to demonstrate how statistics can participate in monitoring.

The threshold is deliberately treated as a configuration choice rather than a universal statistical law.

### Model monitoring

`ModelMonitoringService` stores streaming statistics separately for model versions.

The case study records latency for:

- `model-v1`
- `model-v2`

The service then reports the count, mean, variance, and standard deviation for each version.

This illustrates how the same statistical methods can support model comparison and production monitoring without claiming that the measurements alone establish model quality.

## Algorithmic complexity

The main operations have different computational characteristics.

| Operation | Typical complexity | Reason |
|---|---:|---|
| Mean | `O(n)` | Every observation is processed |
| Population variance | `O(n)` | Every observation contributes |
| Sample variance | `O(n)` | Every observation contributes |
| Standard deviation | `O(n)` | Depends on variance |
| Median using sorting | `O(n log n)` | Data is sorted |
| Mode with ordered map | `O(n log n)` | Frequency map insertion |
| Streaming mean | `O(1)` per observation | Incremental update |
| Welford variance | `O(1)` per observation | Incremental update |
| Welford memory | `O(1)` | Fixed statistical state |

For a large static dataset, vectorized or specialized statistical libraries may provide better performance than hand-written loops.

For a continuously arriving stream, an online algorithm can be substantially more memory-efficient than repeatedly recalculating statistics from all historical observations.

## Welford's algorithm

A naive variance implementation may use an identity based on:

`E[X²] - E[X]²`

This form can suffer from numerical cancellation when values are very large and their variance is comparatively small.

Welford's algorithm updates the mean and accumulated squared deviations incrementally.

For each new observation:

`delta = x - mean`

Then the mean is updated, followed by a second deviation:

`delta_after_update = x - new_mean`

The accumulated quantity is updated using:

`M2 = M2 + delta * delta_after_update`

At the end:

`population variance = M2 / n`

and:

`sample variance = M2 / (n - 1)`

The algorithm is useful when processing large datasets or streaming observations.

## Numerical precision

Python, JavaScript, and C++ commonly use floating-point arithmetic for statistical calculations.

Binary floating-point representation cannot exactly represent many decimal fractions.

For example, JavaScript demonstrates the familiar result that:

`0.1 + 0.2`

is not represented as mathematical decimal `0.3` exactly.

Statistical software therefore operates with numerical approximations.

For ordinary descriptive analysis, double-precision floating point is usually sufficient.

High-precision scientific, financial, or numerical applications may require additional care concerning:

- Floating-point cancellation
- Overflow
- Underflow
- Accumulation error
- Input scale
- Stable algorithms
- Appropriate numeric types

The use of Welford's algorithm is one example of choosing a numerically stable formulation.

## Important distinctions

### Variance versus standard deviation

Variance uses squared units.

Standard deviation returns to the original measurement unit.

If latency is measured in milliseconds:

- Variance is measured in squared milliseconds.
- Standard deviation is measured in milliseconds.

### Population versus sample variance

Population variance divides by `N`.

Sample variance commonly divides by `n - 1` when estimating population variance from a sample.

The correct choice depends on what the observations represent.

### Mean versus median

Mean uses all numerical observations and is sensitive to extremes.

Median depends on ordered position and is generally more robust to isolated extremes.

### Descriptive statistics versus inference

Descriptive statistics summarize observed data.

Statistical inference attempts to draw conclusions about a broader population using data and assumptions.

Mean, median, mode, variance, and standard deviation by themselves do not constitute a complete inferential analysis.

## AI applications

These statistics have many practical uses in AI systems.

### Dataset exploration

Mean, median, and standard deviation help identify the scale and variability of numerical features.

For example, a dataset containing customer transaction values can be inspected for central tendency and unusual spread before model development.

### Feature preprocessing

Mean and standard deviation can be used to standardize numerical features.

This is common when numerical feature scales can affect an algorithm's optimization or distance calculations.

### Data-quality analysis

Unexpected means or unusually high standard deviations can indicate:

- Sensor problems
- Data ingestion failures
- Unit mismatches
- Distribution changes
- Incorrect preprocessing
- Genuine changes in behavior

### Model monitoring

Prediction errors, response latency, confidence scores, or resource consumption can be monitored statistically.

A change in mean may indicate a shift in average behavior.

A change in standard deviation may indicate a change in consistency.

### Anomaly investigation

Z-scores provide one simple way of measuring how far an observation is from a mean in standard-deviation units.

They are useful for teaching and for some operational systems, but anomaly detection should consider the distribution and domain context.

### Experiment analysis

Average model latency, error, accuracy-related measurements, or resource consumption can be summarized using descriptive statistics.

When comparing experiments, the mean should not be examined without considering sample size and variability.

## Limitations

Mean, median, mode, variance, and standard deviation do not fully describe a dataset.

Two datasets can share the same mean and variance while having different distributions.

These statistics do not directly describe:

- Skewness
- Kurtosis
- Quantiles
- Tail behavior
- Multimodality
- Correlation
- Causality
- Temporal dependence
- Feature interactions

A dataset can also contain systematic data-quality problems that are invisible from these statistics alone.

Statistical summaries should therefore be interpreted together with the data collection process and the purpose of the analysis.

## Common mistakes

### Dividing sample variance by `n`

When estimating population variance from a sample under the standard unbiased-estimation framework, the conventional denominator is `n - 1`.

### Confusing variance with standard deviation

Variance is in squared units.

Standard deviation is in the original units.

### Treating mean as universally representative

Strongly skewed data or data with extreme observations can make the mean difficult to interpret as a typical observation.

### Assuming every outlier is an error

An unusual observation may represent a genuine event.

Deleting it without investigation can remove important information.

### Forgetting empty-data validation

Statistical functions require appropriate input domains.

An empty dataset cannot provide a conventional mean or variance.

### Dividing by zero during standardization

Constant data has standard deviation zero.

A z-score transformation therefore requires explicit handling of this case.

### Ignoring sample versus population interpretation

The same numerical observations can produce different interpretations depending on whether they represent the complete target population or a sample from it.

### Using statistics without understanding units

Variance changes the units by squaring them.

Standard deviation returns to the original unit.

### Assuming a z-score proves an anomaly

A large z-score is evidence of distance from the mean under the chosen calculation. It does not by itself prove that an observation is erroneous, fraudulent, unsafe, or abnormal in a domain-specific sense.

## Security and reliability considerations

Statistics code can become part of production data pipelines, so ordinary software engineering concerns remain important.

Input should be validated before numerical processing.

Non-finite values such as NaN and infinity should be handled intentionally.

Domain-specific constraints should also be checked. The C++ monitoring service rejects negative latency because it violates the modeled domain.

Production systems should also consider:

- Malformed external input
- Unexpected data types
- Extreme numeric values
- Missing observations
- Duplicate events
- Incorrect units
- Timestamp inconsistencies
- Adversarial or manipulated measurements
- Monitoring-data integrity

Statistical calculations are only as trustworthy as the data supplied to them.

## Implementation considerations

The Python implementation emphasizes readability and mathematical transparency.

The JavaScript implementation emphasizes application-level data handling, asynchronous processing, and JavaScript-specific behavior.

The C++ implementation emphasizes:

- Strong typing
- Explicit ownership and value semantics
- Standard-library containers
- Exception handling
- Structured reports
- Incremental algorithms
- Computational complexity
- Production-style architecture

The underlying mathematics remains the same, while the programming language changes the implementation techniques.

## Testing

All three implementations contain executable tests or assertions.

The tests verify key identities and expected values, including:

- Mean of `[1, 2, 3, 4, 5]`
- Median of odd and even datasets
- Population variance
- Sample variance
- Multiple modes
- Zero variance for constant data
- Streaming mean
- Streaming variance
- Invalid-input behavior

Testing statistical code is especially important because a small denominator or formula error can produce plausible-looking but incorrect numerical results.

## Practical interpretation of the C++ case study

The C++ program treats inference latency as an operational metric.

Suppose a service records:

`105, 110, 98, 102, 108, 115, 101, 99, 104, 109, 103, 107, 111, 100, 106, 250`

The `250 ms` observation is much larger than the other measurements.

The program calculates the mean and standard deviation, then uses z-scores to identify observations that exceed the configured threshold.

The same data is also processed through Welford's streaming algorithm.

The streaming calculation demonstrates that a service does not need to retain every historical latency value merely to maintain running mean and variance.

The model-monitoring component maintains separate statistics for different model versions, illustrating how descriptive statistics can be incorporated into production-oriented monitoring architecture.

## Relationship to AI workflows

A simplified statistical workflow can be expressed as:

`collect data → validate data → describe data → investigate variability → identify unusual observations → transform where appropriate → train or evaluate models → monitor production behavior`

The statistical calculations in this project primarily support the descriptive and monitoring stages.

They should not be treated as substitutes for model evaluation, hypothesis testing, uncertainty estimation, distribution analysis, or domain-specific validation.

## Mathematical reference

For observations `x₁, x₂, ..., xₙ`:

### Population mean

`μ = Σxᵢ / N`

### Sample mean

`x̄ = Σxᵢ / n`

### Population variance

`σ² = Σ(xᵢ - μ)² / N`

### Sample variance

`s² = Σ(xᵢ - x̄)² / (n - 1)`

### Population standard deviation

`σ = √σ²`

### Sample standard deviation

`s = √s²`

### Z-score

`z = (x - mean) / standard deviation`

These formulas form the mathematical foundation of the three implementations.
