# Feature Scaling: Standardization, Normalization, Robust Scaling, and Transformations

## Scope

Feature scaling is the preprocessing process of changing numerical feature representations so that their magnitudes, distributions, or ranges are suitable for a downstream statistical or machine-learning procedure.

The central distinction in this repository is between **rescaling** and **distribution transformation**.

Standardization, min-max normalization, robust scaling, and max-absolute scaling primarily change the numerical scale of a feature. Logarithmic, square-root, signed-log, and power-style transformations can also change the shape of its distribution.

The implementations use a realistic customer-financial dataset containing:

- income measured in currency units
- monthly transaction counts
- average transaction value
- account age in months

These variables have substantially different numerical ranges. The dataset also contains a high-income observation so that sensitivity to outliers can be observed directly.

---

## Why Feature Scaling Matters

Suppose a distance calculation compares two customers using income and transaction count.

A difference of 20,000 currency units and a difference of 10 transactions are not numerically comparable. In an ordinary Euclidean distance, the income difference contributes approximately 400 million to the squared distance, while the transaction difference contributes only 100.

The resulting distance can therefore be dominated by the measurement unit rather than by the intended analytical importance of the features.

Scaling can also affect optimization. Gradient-based algorithms use derivatives with respect to feature values. When one feature is several orders of magnitude larger than another, its gradient can have a substantially different magnitude. This can force an optimizer to use a very small learning rate or produce inefficient parameter updates.

Feature scaling is especially relevant to methods involving:

- Euclidean or other distance calculations
- nearest-neighbor methods
- clustering
- principal component analysis
- support-vector methods with scale-sensitive kernels
- gradient-based optimization
- regularized linear models where coefficient magnitude interacts with feature scale

Scaling is not automatically necessary for every algorithm. Tree-based methods based on threshold splits generally do not require conventional feature scaling because their decision logic is not based on Euclidean distance or coefficient magnitudes in the same way.

---

## Standardization

Standardization uses the z-score transformation:

`z = (x - μ) / σ`

where `μ` is the training mean and `σ` is the training standard deviation.

The resulting training feature generally has a mean close to zero and a standard deviation close to one.

The Python `StandardScaler`, JavaScript `StandardScaler`, and C++ `StandardScaler` all calculate these parameters during `fit()` and reuse them during `transform()`.

This separation is important. The scaler does not calculate a new mean for every incoming production batch. A production observation must be interpreted using the same statistical coordinate system established by training.

### Properties

Standardization:

- centers features around zero
- puts features on comparable variance scales
- does not constrain values to a fixed interval
- is sensitive to extreme observations because the mean and standard deviation are sensitive to them
- preserves ordering within a feature
- can produce negative and positive values
- is commonly useful before distance calculations and gradient-based optimization

A standardized feature can still contain extreme values. Standardization does not remove outliers. It merely expresses them in units of the fitted standard deviation.

### Constant features

A constant feature has zero variance. Direct standardization would require division by zero.

All three implementations explicitly replace a zero scale denominator with `1`. Since every value equals the fitted mean, the resulting transformed value is zero. This is a safe numerical treatment, although a production feature-selection stage may choose to remove a constant feature because it contains no discriminating information.

---

## Min-Max Normalization

Min-max normalization uses the training minimum and maximum:

`x_scaled = lower + ((x - min) / (max - min)) * (upper - lower)`

With the common `[0, 1]` configuration:

`x_scaled = (x - min) / (max - min)`

For training observations, the smallest observed value maps to zero and the largest maps to one.

### Important behavior on new data

A frequent mistake is assuming that min-max scaling guarantees every future observation will be between zero and one.

It does not.

If a future observation is larger than the training maximum, its transformed value is greater than one. If it is below the training minimum, its transformed value is negative.

This is usually preferable to silently changing the scaling parameters during inference. Clipping can be implemented as a separate business policy when required, but clipping changes the information represented by extreme observations.

Min-max normalization is useful when a bounded numerical representation is important, including situations involving models or systems that benefit from a known training-domain range.

---

## Robust Scaling

Robust scaling replaces mean and standard deviation with statistics that are less sensitive to isolated extreme observations.

The implementation uses:

`scaled = (x - median) / IQR`

where:

`IQR = Q3 - Q1`

and `Q1` and `Q3` are the lower and upper quartiles.

The median represents the center, while the interquartile range represents the spread of the middle portion of the data.

### Why robust scaling differs from standardization

Consider ordinary customer incomes:

`30000, 32000, 35000, 37000, 40000, 42000`

and then add:

`2000000`

The extreme observation dramatically changes the mean and standard deviation. Consequently, standardized values for the ordinary observations also change.

Median and IQR are much less affected by one isolated extreme value. Robust scaling therefore keeps the central observations on a more stable scale.

Robust scaling does not eliminate outliers. It simply prevents an isolated extreme observation from controlling the fitted location and scale as strongly.

---

## Transformations

Scaling and transformation solve related but different problems.

A scaler can change:

- location
- spread
- numerical range

A transformation can also change:

- skewness
- tail behavior
- distribution shape
- the relationship between large and small observations

The implementations demonstrate several transformations.

### Logarithmic transformation

For positive values:

`y = log(x)`

Large values are compressed more strongly than small values.

For example, the numerical difference between 100 and 1,000 becomes much smaller after logarithmic transformation than it was on the original scale.

Log transformations are useful for strongly right-skewed positive measurements such as transaction amounts, revenue, population-like measurements, or exposure quantities.

Ordinary logarithms cannot accept zero or negative values. The Python, JavaScript, and C++ implementations explicitly reject invalid inputs instead of silently producing invalid numerical results.

### Signed logarithm

The signed-log transformation uses:

`sign(x) * log(1 + |x|)`

It supports negative values, zero, and positive values.

This is useful for signed changes, gains and losses, deviations, or other variables where negative observations have genuine meaning.

Zero remains zero because `log(1) = 0`.

### Square-root transformation

The square-root transformation:

`sqrt(x)`

is defined for non-negative values.

It compresses large values while generally being less aggressive than a logarithm. It can be useful for count-like measurements where zero is valid.

### Yeo-Johnson-style transformation

The Python and JavaScript implementations demonstrate a Yeo-Johnson-style power transformation with a supplied lambda. The C++ pipeline focuses on a concrete log-transformation workflow rather than duplicating every transformation.

The important property is that the Yeo-Johnson family can handle zero and negative observations while providing a tunable power transformation.

A production implementation that estimates the optimal lambda should document the estimation procedure and fit that parameter only on the training partition.

---

## Quantile and Rank Transformation

The Python and JavaScript implementations include a rank-to-uniform transformation.

Instead of preserving the original numerical distances, observations are ordered and mapped according to their ranks.

This is fundamentally different from standardization.

If one value is twice another value, rank transformation does not attempt to preserve that ratio. It focuses on ordering and empirical distribution position.

Rank-based methods can be useful when the original distribution is extremely skewed or when a downstream method benefits from a more uniform representation.

Ties require a defined policy. The implementations use average ranks for tied observations.

---

## Transformation Versus Scaling

| Operation | Main mechanism | Outlier sensitivity | Changes distribution shape | Fixed training range |
| --- | --- | --- | --- | --- |
| Standardization | Mean and standard deviation | High | No | No |
| Min-max normalization | Minimum and maximum | High | No | Training values only |
| Robust scaling | Median and IQR | Lower | No | No |
| Max-absolute scaling | Maximum absolute magnitude | High | No | No |
| Log transformation | Logarithmic compression | Reduces large-value influence | Yes | No |
| Square-root transformation | Root compression | Reduces large-value influence | Yes | No |
| Signed log | Signed logarithmic compression | Reduces large magnitude influence | Yes | No |
| Rank transformation | Empirical ordering | Resistant to magnitude extremes | Yes | No |

The distinction matters when designing preprocessing. A skewed feature may first need a transformation and then a scaler.

---

## Training, Validation, Test, and Production Data

The most important operational rule demonstrated by the implementations is:

**Fit preprocessing parameters using training data only.**

Consider:

`training = [20000, 30000, 40000, 50000]`

and test observations:

`[55000, 200000]`

The correct process is:

`fit(training) -> transform(training) -> transform(test)`

The incorrect process is:

`fit(training + test) -> transform(test)`

The second process allows information from the evaluation set to influence preprocessing. Even though the model has not directly used the test labels, the feature representation has already been informed by test observations.

This is data leakage.

The same rule applies to:

- means
- standard deviations
- minima
- maxima
- medians
- quantiles
- transformation parameters
- power-transformation parameters
- learned categorical encodings
- imputation statistics
- other learned preprocessing parameters

The preprocessing configuration and its fitted parameters should be treated as part of the trained model artifact.

---

## Python Implementation

The Python program is a broad executable implementation of the topic.

Its principal classes are:

- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `MaxAbsScaler`
- `FeaturePipeline`

`FeatureScaler` provides the common fitted-state behavior. Each concrete scaler implements its own statistical definition rather than inheriting generic scaling behavior that would blur their differences.

The Python implementation also contains:

- numerical validation
- quantile calculation
- inverse transformations
- distance comparisons
- log and signed-log transformations
- square-root transformation
- Yeo-Johnson-style transformation
- rank-to-uniform conversion
- outlier experiments
- constant-feature handling
- training/test leakage demonstration
- gradient-descent sensitivity demonstration
- composable transformation-plus-scaling pipelines

The `FeaturePipeline` applies column-specific transformations before passing the resulting matrix to a scaler. The example applies `log1p` to a skewed transaction-value feature and then standardizes the resulting representation.

The gradient-descent example demonstrates why scale affects optimization numerically. The raw income feature is measured in tens of thousands, while transaction counts are small integers. The scaled representation permits a much larger learning rate because the input dimensions are numerically better balanced.

---

## JavaScript Implementation

The JavaScript implementation presents the same subject from a Node.js perspective but uses JavaScript-specific design patterns rather than being a direct syntactic translation.

`StandardScaler`, `MinMaxScaler`, and `RobustScaler` use JavaScript classes with private methods such as `#requireFitted()` and `#validateFeatureCount()`.

The `FeaturePipeline` uses a `Map` to associate feature indexes with transformation functions. This demonstrates a useful JavaScript pattern for composing runtime-configurable preprocessing behavior.

The `EventBus` and `PreprocessingService` model an asynchronous preprocessing service. Events such as `fit:start`, `fit:complete`, `transform:start`, and `transform:complete` represent operational boundaries that could correspond to logging, monitoring, queue processing, or service instrumentation.

The asynchronous example uses `Promise` boundaries without adding an external dependency. The scaling calculation remains local and synchronous, while the service interface represents how preprocessing can participate in an asynchronous Node.js application.

The JavaScript implementation also demonstrates:

- explicit numerical validation
- private class methods
- inverse transformations
- outlier comparisons
- leakage caused by fitting on combined training and test observations
- logarithmic, square-root, signed-log, and power-style transformations
- rank transformation
- transformation pipelines
- asynchronous preprocessing events

---

## C++ Case Study

The C++ program models a **financial customer preprocessing service**.

The service receives measurements from several systems where different features use different units. Before an anomaly detector calculates Euclidean distance, the data must be represented on an appropriate numerical scale.

The architecture is organized around a polymorphic `FeatureScaler` interface.

Concrete implementations are:

- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`

Each scaler owns the statistical parameters it learned from training data.

### Data structures

The case study uses:

- `std::vector<double>` for numerical feature vectors
- `std::vector<std::vector<double>>` for matrices
- `std::unique_ptr<FeatureScaler>` for runtime scaler selection
- `std::map` for feature-specific transformation functions
- `PreprocessingPolicy` for explicit preprocessing requirements

This design separates the mathematical operation from the case-study pipeline.

### Pipeline architecture

`FeaturePipeline` accepts feature-specific transformations and a scaler.

The example applies a logarithmic transformation to average transaction value before standardization.

Conceptually:

`raw data -> log transformation -> standardization -> anomaly/distance processing`

The order matters. Applying a transformation after fitting a scaler would produce a different representation from fitting the scaler on transformed training data.

### Preprocessing contract

`PreprocessingContract` represents an operational governance layer.

It validates:

- expected scaler type
- whether test data influenced fitting
- whether the required transformation order was followed

This is useful in production because preprocessing is not merely mathematical code. A model can produce incorrect predictions even when the implementation is syntactically correct if its preprocessing configuration differs between training and inference.

---

## Distance and Scale

The case study explicitly calculates Euclidean distance before and after scaling.

For two feature vectors:

`d(x, y) = sqrt(sum((xi - yi)^2))`

A feature with a large numerical unit can dominate this expression.

Scaling changes the coordinate system so that numerical magnitude is less likely to be confused with analytical importance.

This does not mean every feature should always receive identical variance. Scaling is a modeling decision. A deliberately engineered feature weighting can be meaningful when it reflects domain requirements.

The important point is that unintentional unit differences should not silently determine model behavior.

---

## Outliers and Scaling Choice

An outlier is not automatically an error.

An unusually large income may represent:

- a legitimate high-value customer
- a data-entry mistake
- a unit conversion problem
- a fraud event
- a rare but valid business case

Scaling should not be used as an automatic replacement for data-quality investigation.

Standardization can make a legitimate extreme observation very influential because the mean and standard deviation incorporate it.

Robust scaling reduces this influence on the fitted center and spread, but it does not identify whether the observation is valid.

A production pipeline should therefore distinguish:

`data validation -> anomaly investigation -> transformation -> scaling`

rather than treating all unusual observations as preprocessing problems.

---

## Common Failure Modes

### Fitting on all data

Calculating scaling parameters before splitting into training and test partitions causes leakage.

The test partition must remain unseen when learned preprocessing parameters are calculated.

### Re-fitting during inference

A service that recalculates means, standard deviations, or min/max values for every incoming batch changes the model's coordinate system over time.

This can create prediction drift that is caused by preprocessing rather than by the underlying model.

### Assuming min-max always produces values between zero and one

Only observations within the fitted training extrema are guaranteed to map into the configured interval.

Future observations can exceed those limits.

### Applying logarithms to zero or negative values

Ordinary `log(x)` requires `x > 0`.

A signed-log or another appropriate transformation should be used when negative and zero values have genuine meaning.

### Treating scaling as outlier removal

A scaler changes representation. It does not determine whether a record is valid.

### Ignoring constant features

Zero-variance features cause division-by-zero problems in direct standardization or IQR scaling. The implementations explicitly guard against this condition.

### Inconsistent preprocessing order

If training applies `log -> standardize` but production applies `standardize -> log`, the deployed model receives a representation different from the representation used during training.

---

## Performance Characteristics

For a matrix with `n` observations and `p` features, most scaling operations require approximately `O(n × p)` work once the required statistics are known.

Standardization requires calculating feature means and standard deviations.

Min-max normalization requires feature minima and maxima.

Robust scaling requires quantiles. Exact quantile calculation normally requires sorting feature values, producing approximately `O(n log n)` work per feature in the straightforward implementation.

The educational implementations intentionally calculate these statistics directly and keep the algorithms visible.

Production systems processing very large datasets may use:

- streaming statistics
- distributed aggregation
- approximate quantiles
- batch preprocessing
- persisted preprocessing artifacts

Memory usage for a dense matrix is approximately `O(n × p)`. Sparse production data may require specialized sparse-aware implementations because converting sparse matrices to dense representations can cause substantial memory growth.

---

## Security and Data Integrity Considerations

Feature scaling is not primarily a security mechanism, but preprocessing can create integrity and operational risks.

Input validation should reject:

- NaN values
- positive infinity
- negative infinity
- malformed feature counts
- unexpected data types
- invalid domains for mathematical transformations

The implementations perform finite-value validation before processing.

Preprocessing artifacts should also be versioned with the model they belong to. A model trained with one set of means and standard deviations should not silently consume a different preprocessing artifact.

If preprocessing parameters are persisted in a service, access controls and integrity checks should prevent unauthorized modification. Altering a scaling parameter can substantially change model inputs without changing the model weights themselves.

---

## Debugging Considerations

A production debugging process should inspect the fitted preprocessing parameters alongside model behavior.

Useful diagnostics include:

- training feature means
- training feature standard deviations
- training minima and maxima
- training medians
- training IQRs
- transformation order
- number of fitted features
- count and distribution of rejected values
- frequency of inference values outside training extrema
- differences between training and production feature distributions

Inverse transformation is particularly useful when a scaled value needs to be interpreted in its original business units.

The Python, JavaScript, and C++ implementations demonstrate inverse transformation for scaler-based representations.

---

## Choosing a Scaling Strategy

### Use standardization when

The downstream algorithm benefits from centered features and comparable variance, and the feature distributions are not dominated by problematic extreme observations.

### Use min-max normalization when

A bounded numerical representation is desirable and the handling of future values outside the training extrema has been explicitly considered.

### Use robust scaling when

The feature contains legitimate or persistent extreme observations and median/IQR provide a more stable representation of its central distribution.

### Use a transformation before scaling when

The feature is strongly skewed and changing its distribution shape is useful before applying a conventional scaler.

For example:

`transaction amount -> log1p -> standardization`

can be more suitable than:

`transaction amount -> standardization`

when transaction amounts have a long right tail.

### Do not scale merely because a feature is large

A large numerical magnitude is a reason to investigate whether scale is relevant to the downstream method. It is not by itself evidence that the feature contains a problem.

---

## Practical Relationship Between the Techniques

The four main operations can be viewed as different layers of preprocessing:

**Standardization** establishes a coordinate system based on mean and standard deviation.

**Min-max normalization** establishes a coordinate system based on observed extrema and a configured output interval.

**Robust scaling** establishes a coordinate system based on median and interquartile spread.

**Transformations** change the mathematical representation of the distribution itself.

These operations can be composed, but composition should be deliberate.

A valid pipeline can therefore look like:

`raw feature -> domain validation -> distribution transformation -> scaler -> model`

The fitted statistics and transformation parameters belong to the preprocessing artifact and must be applied consistently at inference time.

---

## Repository Execution

The Python file can be executed directly with:

`python feature_scaling.py`

The JavaScript file can be executed with:

`node feature_scaling.js`

The C++ case study can be compiled with:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic feature_scaling.cpp -o feature_scaling`

and then executed with:

`./feature_scaling`

On Windows, the compiled executable can be run as:

`.\feature_scaling.exe`

No third-party package is required by any of the three implementations.

---

## Technical Limitations

The implementations are intentionally educational and expose the underlying calculations rather than relying on machine-learning libraries.

The quantile implementation uses linear interpolation. Different libraries can use different percentile conventions, so a production system should document the selected definition.

The Python and JavaScript power-transformation examples demonstrate the mechanism but do not automatically estimate an optimal transformation parameter from data.

The C++ program uses dense matrices. Large-scale sparse datasets require different storage and computation strategies.

The implementations use batch fitting. Streaming or distributed systems may require online statistics, mergeable estimators, or approximate quantile algorithms.

These limitations do not change the core preprocessing principles: learned transformations must be fitted consistently, training data must remain isolated from evaluation data, and the transformation applied during inference must match the transformation used during model training.
