# Exploratory Data Analysis: Univariate, Bivariate, and Multivariate Analysis

## Scope

Exploratory Data Analysis (EDA) is the systematic examination of a dataset before formal statistical modeling, machine learning, forecasting, or operational decision-making.

This project focuses on three analytical levels:

- **Univariate analysis** examines one variable at a time. It establishes distribution, central tendency, dispersion, missingness, frequency, and unusual observations.
- **Bivariate analysis** examines two variables together. It can reveal association, covariance, correlation, pairwise regression relationships, or differences across groups.
- **Multivariate analysis** examines several variables jointly. It is useful for understanding correlation structure, interactions among measurements, categorical combinations, and the behavior of a response after several predictors are considered simultaneously.

The implementations use a common business scenario involving sales transactions. The dataset contains marketing expenditure, discount levels, units sold, customer ratings, delivery time, revenue, geographic region, sales channel, product, and return status.

The three analytical levels are deliberately separated because they answer different questions. A variable can look reasonable by itself while exhibiting an unexpected relationship with another variable, and a pairwise relationship can change when additional variables are introduced.

---

## Analytical Scenario

The dataset represents transactions from a commerce operation.

The numerical variables are:

| Variable | Meaning | Analytical role |
|---|---|---|
| `marketingSpend` | Marketing expenditure associated with a transaction | Predictor and univariate continuous variable |
| `discountPct` | Percentage discount applied to the transaction | Pricing-related predictor |
| `unitsSold` | Number of units sold | Sales-volume measure |
| `customerRating` | Customer rating on a 1–5 scale | Customer-experience measure |
| `deliveryDays` | Delivery duration | Operational performance measure |
| `revenue` | Transaction revenue | Primary numerical response |

The categorical variables are:

| Variable | Meaning |
|---|---|
| `region` | Geographic business region |
| `channel` | Online, Retail, or Partner sales channel |
| `product` | Product category |
| `returned` | Whether the transaction resulted in a return |

The generated data intentionally contains relationships that make EDA meaningful. Marketing expenditure is associated with units sold and revenue, discounting affects sales volume, delivery time influences customer rating, and slower delivery can contribute to return probability.

These relationships are suitable for demonstrating analytical techniques but should not be interpreted as evidence of causal effects.

---

## Univariate Analysis

Univariate analysis treats each variable independently.

The central question is:

> What does this variable look like by itself?

For a numerical variable, useful properties include:

- Number of observed values
- Number of missing values
- Minimum and maximum
- Mean
- Median
- Sample standard deviation
- First quartile
- Third quartile
- Interquartile range
- Distribution shape
- Potential outliers

For categorical variables, frequency and percentage distributions are more appropriate than numerical measures such as a mean.

### Central tendency

The **mean** is calculated as:

`mean = sum(x_i) / n`

The mean uses every observation and is sensitive to extreme values.

The **median** is the middle observation after sorting. It is less sensitive to extreme values and is often useful when a distribution is skewed.

Comparing mean and median can reveal distribution asymmetry. A large difference between them may indicate skewness or influential observations.

### Dispersion

Sample variance measures squared deviations from the sample mean:

`variance = sum((x_i - mean)^2) / (n - 1)`

The sample standard deviation is the square root of the variance.

Standard deviation expresses dispersion in the original measurement units. For example, revenue standard deviation is measured in revenue units rather than squared revenue units.

### Quartiles and IQR

The first quartile, `Q1`, represents the lower portion of the ordered observations, while `Q3` represents the upper portion.

The interquartile range is:

`IQR = Q3 - Q1`

The IQR describes the spread of the central portion of the data and is less sensitive to extreme observations than the range.

### Distribution inspection

The Python implementation creates a terminal histogram for revenue. The JavaScript implementation produces histogram bins as structured data. These representations make concentration and sparse regions visible without requiring a plotting library.

A histogram can expose:

- Concentration around a central region
- Right or left skew
- Multiple modes
- Long tails
- Sparse extreme observations
- Potential data-quality anomalies

The histogram should be interpreted together with numerical summaries rather than treated as a standalone diagnostic.

### Categorical distributions

The implementations calculate frequencies for region, channel, product, and return status.

For example, a frequency table for sales channel answers a different question from revenue statistics:

- frequency analysis asks how many observations belong to each channel;
- revenue analysis asks how the numerical outcome behaves.

A category with a large number of transactions can have a lower average revenue than a smaller category. Frequency and magnitude therefore need to remain analytically distinct.

---

## Missing-Value Analysis

Missing values are explicitly represented as `None` in Python, `null` in JavaScript, and `std::optional<double>` in C++.

This distinction is important because replacing missing values with zero changes the meaning of the observation.

For example:

- missing revenue is not necessarily zero revenue;
- missing customer rating is not a one-star rating;
- missing marketing expenditure is not zero marketing expenditure.

The programs count missing observations before performing numerical analysis.

Pairwise calculations also use complete pairs. If marketing expenditure exists but revenue is missing, that row cannot contribute to a marketing-versus-revenue correlation.

This is a form of pairwise complete-case analysis. In real analytical work, the missingness mechanism should also be investigated because observations may be missing systematically rather than randomly.

---

## Outlier Analysis

The project uses the IQR rule and z-score screening.

For the IQR method:

`lower fence = Q1 - 1.5 × IQR`

`upper fence = Q3 + 1.5 × IQR`

An observation outside these fences is flagged as a potential outlier.

The z-score is:

`z = (x - mean) / standard deviation`

The implementations use an absolute z-score threshold of 3 for a simple screening rule.

These methods answer different diagnostic questions.

The IQR method is resistant to extreme observations and is particularly useful for skewed business measurements. The z-score method relates an observation to the mean and standard deviation and therefore depends more strongly on the distributional assumptions behind those statistics.

An outlier is not automatically an error.

A transaction with unusually high revenue may represent:

- A large enterprise order
- A legitimate promotion
- A special contract
- A data-entry error
- A duplicated transaction
- An unusual but valid business event

The correct response is investigation. Automatic deletion can remove important information and distort subsequent analysis.

---

## Bivariate Analysis

Bivariate analysis considers two variables simultaneously.

The question changes from:

> What does this variable look like?

to:

> How do these two variables behave together?

The project examines marketing expenditure and revenue as a primary numerical pair.

It also performs grouped comparisons such as revenue by channel and customer rating by region.

### Covariance

Sample covariance is:

`cov(X,Y) = sum((X_i - X_mean)(Y_i - Y_mean)) / (n - 1)`

Positive covariance indicates that the variables tend to move in the same direction relative to their means.

Negative covariance indicates that they tend to move in opposite directions.

The magnitude of covariance depends on the units of measurement, so covariance values are difficult to compare across unrelated variable pairs.

### Pearson correlation

Pearson correlation standardizes covariance:

`r = cov(X,Y) / (s_X × s_Y)`

Its range is from `-1` to `+1`.

A positive value represents a positive linear association. A negative value represents a negative linear association. Values close to zero indicate weak linear association.

Correlation does not establish causation.

A strong marketing-spend/revenue correlation could be associated with several mechanisms. Larger transactions may receive more marketing resources, successful regions may spend more, or another variable may affect both.

### Simple linear regression

The Python, JavaScript, and C++ implementations fit a simple model:

`revenue = intercept + slope × marketingSpend`

The slope describes the fitted change in revenue for one unit of the predictor under the model.

The intercept is the predicted response when the predictor equals zero. If zero is outside the meaningful range of the data, the intercept may have little practical interpretation even though it is mathematically required by the model.

The implementations also calculate:

`R² = 1 - SSE / SST`

where `SSE` is the residual sum of squares and `SST` is the total sum of squares.

R² describes the proportion of observed variation explained by the fitted linear relationship in this sample. It does not establish causal influence or guarantee predictive performance on future observations.

### Grouped bivariate analysis

The project also analyzes:

- Revenue grouped by sales channel
- Customer rating grouped by region

These analyses are bivariate because they relate one grouping variable to one numerical measure.

A categorical grouping variable should not automatically be treated as a continuous numerical variable. `Online`, `Retail`, and `Partner` do not possess a natural numeric ordering merely because software can encode them as numbers.

---

## Multivariate Analysis

Multivariate analysis considers several variables simultaneously.

The central motivation is that a pairwise relationship can be incomplete.

Suppose marketing expenditure and revenue are correlated. Revenue may also depend on product price, discount level, transaction volume, and delivery-related factors. Looking only at marketing expenditure and revenue does not account for those additional variables.

The project therefore includes a numerical correlation matrix and a multiple regression model.

### Correlation matrix

The correlation matrix contains pairwise Pearson correlations for:

- Marketing expenditure
- Discount percentage
- Units sold
- Customer rating
- Delivery days
- Revenue

The matrix allows relationships to be inspected across many variables.

A correlation matrix can expose:

- Strongly associated predictors
- Potential redundant measurements
- Variables associated with the response
- Negative operational relationships
- Candidate multicollinearity concerns

Correlation matrices do not establish causal structure.

### Multicollinearity

If two predictors are strongly correlated, their individual regression coefficients can become difficult to interpret.

For example, if marketing expenditure and units sold are highly related, a model that includes both may have unstable coefficient estimates compared with a model where the predictors are less correlated.

This is important because a predictor can appear weak in a multivariate model even when it has a strong pairwise relationship with the target.

The change can occur because another predictor explains overlapping variation.

### Multiple regression

The C++ implementation models revenue using:

`revenue = intercept + b1(marketingSpend) + b2(discountPct) + b3(deliveryDays)`

The Python implementation uses the same conceptual structure but implements a small matrix solver directly.

The design matrix contains an intercept column followed by the predictors.

The normal-equation form is:

`XᵀXβ = Xᵀy`

The implementation solves this system using Gaussian elimination with partial pivoting instead of explicitly calculating a matrix inverse.

Avoiding explicit inversion is preferable because directly computing an inverse introduces unnecessary numerical work and can be less stable.

### Complete-case behavior

Multiple regression requires complete observations for all predictors and the target in this implementation.

A row with a missing delivery time cannot contribute to the fitted model because the design matrix would not contain a complete feature vector.

In a production analysis, the missing-data strategy should be chosen deliberately. Possible approaches include appropriate imputation, missingness indicators, model-specific handling, or exclusion when justified by the analytical design.

---

## Categorical Multivariate Analysis

The project also uses cross-tabulation.

The C++ and JavaScript implementations examine relationships such as:

`region × channel`

This produces counts for combinations of categories.

A cross-tabulation can reveal whether the operational mix differs by region.

For example, a region could contain many Online transactions but relatively few Partner transactions. Such structural differences matter when comparing revenue or customer metrics because the groups may have different transaction compositions.

Cross-tabulation is not equivalent to a numerical correlation matrix. Nominal categories require different analytical treatment.

---

## Python Implementation

The Python program is a standard-library-only EDA engine.

The dataset is represented as dictionaries containing both categorical and numerical fields. The program includes a realistic synthetic-data generator with deliberately introduced missing values and a high-value transaction.

### Univariate implementation

`numeric_values()` extracts valid numerical observations while excluding missing values.

`NumericSummary` stores descriptive statistics.

`summarize_numeric()` calculates:

- observed count;
- missing count;
- minimum;
- maximum;
- mean;
- median;
- standard deviation;
- variance;
- quartiles;
- IQR.

`frequency_table()` provides categorical frequency analysis.

`histogram()` performs terminal-oriented binning so distribution shape can be inspected without external visualization packages.

### Bivariate implementation

`paired_numeric_values()` performs complete-pair selection.

`covariance()` implements sample covariance.

`pearson_correlation()` implements the Pearson coefficient directly.

`simple_linear_regression()` calculates the fitted intercept, slope, correlation, R², and residual standard error.

This makes the numerical mechanics visible instead of delegating the complete calculation to a statistical library.

### Multivariate implementation

`correlation_matrix()` constructs a pairwise numerical correlation matrix.

`crosstab()` creates a categorical contingency structure.

`multiple_linear_regression()` implements ordinary least squares for three predictors. It constructs `XᵀX` and `Xᵀy` and solves the resulting linear system using Gaussian elimination with partial pivoting.

The implementation also validates business constraints such as rating ranges, discount ranges, non-negative delivery time, and non-negative revenue.

### File handling

The Python program exports the generated dataset to `eda_sales_dataset.csv`.

A complementary `load_csv()` function demonstrates how a future dataset could be loaded while converting recognized numerical columns.

The CSV loader is intentionally conservative rather than attempting to infer arbitrary schemas automatically.

---

## JavaScript Implementation

The JavaScript implementation treats EDA as an event-driven analytical pipeline.

It is designed for Node.js rather than a browser because the workflow produces a machine-readable JSON report on disk.

### Dataset and statistical model

`SeededRandom` provides deterministic pseudo-random data generation. Deterministic generation is useful for EDA demonstrations because repeated executions produce comparable analytical results.

`Map` is used for frequency and grouping operations. This is appropriate for categorical aggregation because category values become keys without requiring an artificial numeric encoding.

### Event-driven workflow

`EDAEngine` extends Node.js `EventEmitter`.

The engine emits events such as:

- `started`
- `stageComplete`
- `completed`

The workflow executes:

`inspection → univariate → bivariate → multivariate → outliers → validation`

The use of `setImmediate()` introduces an asynchronous yield between analytical stages. This demonstrates how a Node.js data-processing pipeline can cooperate with the event loop rather than assuming every analytical operation must execute as one uninterrupted synchronous block.

### JavaScript-specific analysis

The implementation uses:

- `Map` for categorical aggregation;
- `Object.fromEntries()` for structured analytical results;
- array transformations for numerical extraction;
- event emitters for pipeline orchestration;
- asynchronous file output through `fs/promises`;
- JSON serialization for machine-readable reports.

The final `eda-report.json` file contains the analytical results rather than merely printing them to the terminal.

---

## C++ Case Study

The C++ implementation models an analytical engine for a commerce organization.

`SaleRecord` represents a transaction.

Optional numerical fields use `std::optional<double>`. This prevents a missing measurement from being confused with a valid zero.

`EDAEngine` owns the dataset and provides analytical operations.

### Numerical implementation

The engine implements:

- mean;
- median;
- quantiles;
- sample variance;
- sample standard deviation;
- covariance;
- Pearson correlation;
- simple linear regression.

The functions validate insufficient observations and zero-variance conditions.

### Multivariate regression engine

The multiple-regression case study uses three predictors:

- `marketingSpend`
- `discountPct`
- `deliveryDays`

with `revenue` as the target.

The implementation builds the normal equations and solves them using Gaussian elimination with partial pivoting.

The design intentionally solves the system instead of calculating an explicit inverse.

This is a useful distinction because the mathematical expression may be written as `(XᵀX)^-1Xᵀy`, but a numerical implementation does not need to construct that inverse explicitly.

### C++ data structures

The case study uses:

- `std::vector` for observations and numerical samples;
- `std::map` for categorical aggregation;
- `std::optional` for missing numerical values;
- `std::string` for categorical dimensions;
- structured result types for regression outputs.

These structures keep missingness, categorical dimensions, and numerical observations explicit.

### Validation

The engine checks business constraints independently of statistical calculations.

A customer rating outside `[1,5]`, a discount outside `[0,100]`, negative delivery time, or negative revenue is treated as a data-quality violation.

Statistical unusualness and business invalidity are intentionally separated. An observation can be statistically extreme while still satisfying all business rules.

---

## Relationship Between the Three Analytical Levels

The three levels form a progression in analytical scope.

| Level | Main question | Typical techniques | Example in this project |
|---|---|---|---|
| Univariate | What does one variable look like? | Mean, median, quartiles, histogram, frequency | Distribution of revenue |
| Bivariate | How are two variables related? | Covariance, correlation, regression, grouped means | Marketing spend versus revenue |
| Multivariate | How do several variables behave jointly? | Correlation matrix, multiple regression, cross-tabulation | Revenue with marketing spend, discounts, and delivery time |

They should not be treated as interchangeable.

A univariate distribution cannot reveal whether two variables move together.

A bivariate relationship cannot reveal every effect associated with additional variables.

A multivariate model cannot compensate for poor variable definitions or defective source data.

EDA therefore works best as a layered process rather than a collection of isolated statistics.

---

## Correlation Versus Causation

A major analytical failure mode is interpreting correlation as proof of causality.

Suppose marketing spend and revenue show a strong positive correlation.

That observation is compatible with several explanations:

- Marketing expenditure could contribute to sales.
- High-revenue products could receive larger marketing budgets.
- Large customers could simultaneously generate high revenue and receive more marketing attention.
- A third variable could influence both marketing allocation and revenue.
- The relationship could vary by region or product.

EDA identifies patterns worth investigating. It does not, by itself, establish the causal mechanism.

Causal conclusions require an appropriate research design, assumptions, controls, or experimental evidence.

---

## Edge Cases

### Constant variables

Pearson correlation becomes undefined when one variable has zero variance.

For example, if every customer rating were exactly `5`, there would be no variation from which to calculate a correlation with another variable.

The implementations detect this condition rather than returning a misleading numerical result.

### Small samples

Variance, covariance, and regression require enough observations to produce meaningful estimates.

The implementations raise errors or reject insufficient data rather than silently calculating invalid statistics.

### Missing paired observations

A row with missing marketing expenditure but valid revenue cannot contribute to a pairwise marketing-revenue correlation.

The implementations select complete pairs for pairwise numerical analysis.

### Outliers

Extreme observations can substantially change means, standard deviations, correlations, and regression coefficients.

The programs flag potential outliers but do not automatically remove them.

### Singular multivariate systems

Multiple regression can fail when predictors do not provide sufficient independent information.

Perfect or near-perfect linear dependence between predictors can make `XᵀX` singular or numerically unstable.

The C++ and Python implementations detect problematic pivots instead of continuing with an invalid solution.

---

## Common EDA Failure Modes

### Treating missing values as zero

A missing observation and a zero measurement have different meanings.

Replacing missing values with zero can shift the mean, reduce variance, alter correlations, and bias regression coefficients.

### Removing every outlier

An extreme observation can be a valid business event.

Deleting it without investigation may remove exactly the cases that matter operationally.

### Using correlation as causation

Correlation describes association, generally linear association for Pearson's coefficient. It does not establish a causal mechanism.

### Ignoring categorical structure

A numerical summary of the entire dataset can conceal major differences between regions, channels, or product groups.

Grouped analysis and cross-tabulation can reveal these structural differences.

### Inspecting only the target variable

Analyzing revenue alone does not reveal whether unusual revenue values are associated with unusual marketing expenditure, discounts, products, channels, or transaction volumes.

### Ignoring measurement units

Covariance depends on units, while correlation is standardized. Comparing covariance values across unrelated measurements without considering scale can lead to incorrect conclusions.

### Overinterpreting R²

A high R² indicates that the fitted model explains substantial variation in the observed sample under the model specification. It does not prove that the model is causal, correctly specified, or reliable on future data.

---

## Performance Considerations

For `n` observations:

- Mean is `O(n)`.
- Variance is `O(n)`.
- Covariance is `O(n)`.
- Pearson correlation is `O(n)`.
- Frequency aggregation is approximately `O(n)` expected for hash-based structures, while ordered maps can be `O(n log k)` for `k` categories.
- Sorting-based median and quantiles are approximately `O(n log n)` in these implementations.
- A correlation matrix with `p` numerical variables requires approximately `O(p²n)` pairwise work.
- Multiple linear regression using a `p × p` normal-equation system includes matrix construction work and an `O(p³)` linear-system solve.

For small educational datasets, these direct implementations are appropriate.

For very large datasets, memory usage, numerical stability, streaming algorithms, vectorized computation, distributed aggregation, and specialized linear algebra implementations become important.

---

## Numerical Stability

Simple descriptive statistics are relatively straightforward, but regression and correlation calculations can become numerically sensitive when variables have very different scales or predictors are highly correlated.

The C++ case study uses partial pivoting during Gaussian elimination.

The Python implementation similarly checks for near-zero pivots in its matrix solution.

In production numerical analysis, additional considerations include:

- feature scaling;
- centered calculations;
- stable covariance algorithms;
- QR decomposition;
- singular-value decomposition;
- regularization for ill-conditioned regression problems.

The educational implementations expose the mechanics without pretending that a small hand-built linear algebra routine is a replacement for a production numerical library.

---

## Data Quality and EDA

EDA is partly statistical analysis and partly data-quality investigation.

The project separates statistical unusualness from business-rule validity.

A value can be:

- valid and typical;
- valid but unusual;
- invalid but statistically ordinary;
- missing;
- duplicated;
- structurally inconsistent.

For example, a revenue value of `22000` may be a statistical outlier but still be a legitimate enterprise transaction.

Conversely, a revenue value of `500` may look statistically ordinary while being invalid because the transaction record was duplicated or mapped to the wrong product.

This distinction is essential in real analytical systems.

---

## Security and Operational Considerations

EDA code often operates on sensitive operational datasets.

Production implementations should therefore consider:

- access control for source data;
- protection of exported CSV and JSON files;
- removal or masking of personally identifiable information;
- validation of imported files;
- safe handling of malformed records;
- auditability of transformations;
- reproducibility through controlled data versions;
- separation of raw data from derived analytical outputs.

The demonstration dataset contains synthetic transaction information and does not require credentials or external services.

---

## Production Considerations

A production EDA pipeline would normally separate several concerns that are intentionally combined in this educational case study.

The ingestion layer would validate schemas and data types.

The quality layer would track missingness, duplicates, invalid ranges, referential consistency, and unexpected categories.

The statistical layer would calculate descriptive and relationship measures.

The visualization layer would expose distributions and relationships through plots.

The reporting layer would preserve analytical results with metadata such as dataset version, analysis timestamp, sample size, filtering conditions, and statistical assumptions.

This separation makes an EDA process auditable and reproducible.

A production pipeline should also distinguish exploratory findings from confirmed business rules. An observed pattern should not automatically become an operational policy.

---

## Reproducibility

All three implementations use deterministic synthetic-data generation with a fixed seed.

This means the same source configuration can reproduce the same analytical dataset and approximately the same results.

Reproducibility is important because EDA often influences downstream decisions. A finding that cannot be reconstructed from the same data and analytical procedure is difficult to audit.

The Python program writes `eda_sales_dataset.csv`.

The JavaScript program writes `eda-report.json`.

The C++ program prints its analytical case study directly to standard output.

---

## Practical Interpretation

The appropriate interpretation sequence is:

**data quality → univariate structure → bivariate relationships → multivariate structure → domain validation → modeling or decision-making**

The sequence matters.

If a numerical field contains invalid values, its univariate statistics can already be misleading.

If two variables show a relationship, grouped or multivariate analysis may reveal that the relationship differs across operational segments.

If a multivariate model produces a strong fit, its assumptions, missing-data handling, predictor relationships, residual behavior, and generalization must still be examined.

EDA therefore acts as a diagnostic foundation rather than merely a collection of descriptive statistics.
