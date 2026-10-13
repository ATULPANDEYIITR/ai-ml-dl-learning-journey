# k-Nearest Neighbors: Distance, Classification, and Regression

## Technical scope

The k-Nearest Neighbors (k-NN) algorithm predicts an outcome from the training observations most similar to a query. It is a non-parametric, instance-based learning method: the fitted model retains historical observations rather than estimating a fixed set of coefficients.

The implementation set covers three connected but distinct responsibilities:

- **Distance measurement** determines which observations are similar to the query.
- **Neighborhood selection** determines how many observations influence the result and how ties are handled.
- **Prediction** converts the selected observations into a class, class-vote distribution, or continuous estimate.

The Python and JavaScript implementations provide general-purpose estimators. The C++ program applies k-NN to municipal water-pump maintenance. The Java program models manufacturing quality assessment. The PostgreSQL script represents historical measurements and prediction requests relationally.

## Mathematical foundations

Let the training dataset contain feature vectors \(x_i \in \mathbb{R}^d\) and their targets \(y_i\). For a query vector \(q\), the algorithm computes a distance \(d(q,x_i)\) for every eligible training sample and selects the \(k\) smallest distances.

The selected neighborhood is

\[
N_k(q)=\operatorname{arg\,min}_{S,\ |S|=k}
\sum_{x_i\in S} d(q,x_i),
\]

where \(S\) is a subset of the training observations. In practice, sorting by distance and retaining the first \(k\) records implements this selection. A deterministic secondary ordering is useful when several observations have equal distances.

The algorithm is called a lazy learner because most of the computational work happens at prediction time. Training generally validates and stores the observations. Each prediction compares the query with the retained training records.

## Distance metrics

### Euclidean distance

Euclidean distance measures straight-line separation:

\[
d_2(a,b)=
\sqrt{\sum_{j=1}^{d}(a_j-b_j)^2}.
\]

It is appropriate when feature differences have meaningful magnitudes and the geometry is reasonably represented by straight-line distance. Squared differences make large deviations contribute disproportionately.

The Python, JavaScript, C++, Java, and SQL examples use Euclidean distance as the principal metric. The relational implementation computes the same distance over standardized feature columns.

### Manhattan distance

Manhattan distance sums absolute coordinate differences:

\[
d_1(a,b)=\sum_{j=1}^{d}|a_j-b_j|.
\]

It is useful when movement or deviation across each feature is additive. Compared with Euclidean distance, it does not square large coordinate differences before aggregation.

The Java regression example uses Manhattan distance to estimate inspection cost, illustrating that the best metric for a continuous target need not be the default metric for classification.

### Minkowski distance

Minkowski distance generalizes the \(L_p\) family:

\[
d_p(a,b)=
\left(\sum_{j=1}^{d}|a_j-b_j|^p\right)^{1/p},
\quad p\geq 1.
\]

Setting \(p=1\) produces Manhattan distance, while \(p=2\) produces Euclidean distance. Larger values increasingly emphasize the largest coordinate difference. The Python and JavaScript implementations validate the parameter rather than accepting invalid exponents.

### Chebyshev distance

Chebyshev distance measures the largest absolute coordinate difference:

\[
d_\infty(a,b)=\max_j |a_j-b_j|.
\]

This metric is appropriate when the maximum deviation across dimensions determines similarity or compliance. The C++ and Java implementations include it as an alternative to aggregate-distance metrics.

### Cosine distance

Cosine distance compares vector direction rather than absolute magnitude:

\[
d_{\cos}(a,b)=
1-\frac{a\cdot b}{\|a\|\|b\|}.
\]

It can be useful when direction or relative feature patterns matter more than overall magnitude. It is undefined when either vector has zero norm, so the Python and JavaScript implementations reject that case. Cosine distance should not be substituted indiscriminately for Euclidean distance: two vectors with the same direction but very different magnitudes have zero cosine distance.

## Feature scaling

Distance calculations can be dominated by features with larger numeric ranges. Consider a model with age in years and income in currency units. A difference of 20,000 currency units can overwhelm a difference of several years, even when both features are important.

Standardization transforms a feature using its training-set mean and standard deviation:

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}.
\]

The implementations use population standard deviation. A constant feature receives a scale of one, avoiding division by zero.

Scaling is not automatically beneficial for every dataset. It can reduce unwanted unit effects, but it can also alter meaningful domain relationships. Feature selection, weighting, and scaling decisions should be based on the application and validation results.

The scaler must be fitted using training data only. Fitting it on the complete dataset allows information from validation or test observations to influence preprocessing. The Python cross-validation function and JavaScript cross-validation workflow fit a separate scaler inside each fold.

## Neighborhood selection

The value of \(k\) controls the local scope of a prediction.

A small neighborhood is sensitive to individual observations, noise, and mislabeled records. For classification, a small \(k\) can create irregular decision boundaries. For regression, it can produce estimates strongly influenced by a single training target.

A larger neighborhood usually smooths local variation, but it can combine observations from different classes or regions of a nonlinear relationship. This can increase bias and blur local structure.

The value of \(k\) must not exceed the training sample count. All implementations reject invalid neighborhood sizes. The Python and JavaScript programs also evaluate candidate values through cross-validation.

Cross-validation estimates how the model behaves on observations excluded from each fitting step. The candidate with the highest validation accuracy is not guaranteed to be best on future data. Selection should consider variability across folds, class imbalance, operational costs, and the consequences of false predictions.

## Classification

For classification, the neighborhood contributes votes to candidate classes. With uniform weighting, each neighbor contributes one vote:

\[
\hat y(q)=\operatorname{mode}
\{y_i:x_i\in N_k(q)\}.
\]

The Python, JavaScript, C++, and Java implementations use deterministic tie-breaking so that equal votes do not produce unpredictable results. Their secondary rules consider cumulative neighbor distance and a stable label ordering.

### Distance-weighted classification

Uniform voting treats the first and the kth neighbor equally. Inverse-distance voting gives closer observations greater influence:

\[
w_i=\frac{1}{d(q,x_i)}.
\]

The predicted class is the class with the greatest total weight. The Python, JavaScript, and Java implementations expose normalized vote weights as a class distribution.

These normalized values are useful as local vote fractions, but they are not automatically calibrated probabilities. A model can assign a vote fraction of 0.9 without being correct 90% of the time. Calibration requires separate evaluation against observed outcomes.

### Exact matches

Inverse-distance weighting becomes numerically problematic when a training observation exactly matches the query. Dividing by zero would produce an infinite weight.

The implementations handle exact matches explicitly. If weighted prediction finds zero-distance neighbors, those observations determine the prediction and nonzero-distance neighbors are excluded. Multiple exact matches contribute equal finite weights.

This rule is especially important when duplicate measurements carry conflicting labels. Such conflicts should be investigated because they may indicate measurement collisions, inconsistent labeling, or genuinely ambiguous cases.

## Regression

For regression, the target is continuous rather than categorical. A weighted neighborhood mean produces the prediction:

\[
\hat y(q)=
\frac{\sum_{x_i\in N_k(q)}w_i y_i}
{\sum_{x_i\in N_k(q)}w_i}.
\]

Uniform weights yield an arithmetic mean. Inverse-distance weights make the estimate more sensitive to nearby observations.

The Python, JavaScript, C++, and Java programs estimate quantities such as maintenance or inspection costs from historical cases. The PostgreSQL regression view performs the weighted aggregation directly over the selected neighbors.

Regression predictions should be evaluated with metrics appropriate to the target. Mean Absolute Error (MAE) measures average absolute prediction error:

\[
\operatorname{MAE}=
\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|.
\]

Root Mean Squared Error (RMSE) penalizes large errors more strongly:

\[
\operatorname{RMSE}=
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}.
\]

The Python implementation calculates both measures, while the JavaScript and Java examples calculate MAE.

## Python implementation

The Python script defines a `KNearestNeighbors` estimator with separate classification and regression modes. Its `kneighbors` method returns records containing the original training index, distance, and target. Retaining the index makes equal-distance ordering deterministic and supports tracing a prediction back to its source observation.

The implementation separates distance functions from prediction logic. Euclidean, Manhattan, Minkowski, Chebyshev, and cosine distance can be inspected independently, while the estimator accepts a metric name. Validation rejects inconsistent dimensions, nonfinite feature values, invalid \(k\), unsupported weighting strategies, and nonnumeric regression targets.

`StandardScaler` preserves the training means and scales for later query transformation. `predict_proba` reports normalized local class-vote weights, while `predict` applies the explicit classification tie-break. The regression path computes a weighted mean.

The script includes a seeded train/test split, confusion matrix, accuracy, MAE, RMSE, and cross-validation. Its edge-case checks verify basic metric calculations, exact-match handling, regression averaging, invalid dimensions, and invalid neighborhood sizes.

The training/test split is intentionally simple and does not stratify class labels. Small or imbalanced datasets may require a stratified split so that each evaluation partition contains representative class distributions.

## JavaScript implementation

The JavaScript file emphasizes an event-driven workflow and runtime validation. `KNNClassifier` exposes `fit`, `kneighbors`, `predictOne`, `predict`, `predictProbabilities`, and `addTrainingExample`. The last method demonstrates how a labeled observation can be appended to an in-memory reference set after validation.

The classifier stores copied feature arrays so callers cannot modify the training observations accidentally through the original arrays. It also returns fresh neighborhood objects, preventing prediction code from depending on mutable caller-owned input structures.

`OnlineStandardScaler` is a reusable preprocessing object, while `KNNRegressor` implements continuous-target estimation separately from class voting. This separation makes it harder to confuse categorical aggregation with numeric averaging.

The `crossValidate` function constructs deterministic folds and fits scaling within each fold. `evaluateModelAsync` illustrates how prediction can be composed with asynchronous application code. Its use of a promise does not make the brute-force distance search itself parallel; CPU-intensive workloads require a separate execution strategy.

The tests exercise invalid input, exact matches, distance functions, and regression behavior. The program reports errors through a top-level asynchronous failure handler and sets a nonzero process exit code if evaluation fails.

## C++ case study: municipal pump maintenance

The C++ program models a water utility that needs to prioritize pump inspections. Each `EquipmentRecord` stores an asset identifier, a feature vector, a categorical risk label, and an estimated historical maintenance cost.

`StandardScaler` learns feature means and scales from historical measurements. `KNNClassifier` then identifies the nearest historical assets and predicts a risk category. Its probability method reports normalized weighted votes. `KNNRegressor` estimates a continuous maintenance cost from the same normalized measurement space.

The classifier uses `std::partial_sort` to place the nearest \(k\) observations at the beginning of the candidate range. For a training set of size \(n\), this avoids fully ordering all observations when only a small neighborhood is required. The distance calculations still examine all training records.

The model validates vector dimensions, finite feature values, label alignment, and neighborhood size. Exception handling at the program boundary reports invalid requests rather than allowing malformed input to produce a prediction.

The operational interpretation is deliberately limited: a high predicted risk can prioritize an inspection, but it does not authorize an automatic equipment shutdown. False positives can trigger unnecessary work, while false negatives can delay maintenance. A deployment should evaluate these errors against equipment criticality, safety requirements, and the cost of missed failures.

The printed training accuracy is diagnostic only. Because the same records were used to fit the model and evaluate the predictions, it is not a reliable estimate of performance on new assets.

## Java implementation: manufacturing quality assessment

The Java program uses domain-specific records and enums to represent manufacturing components, quality states, distances, weighting modes, and prediction results.

`ComponentSample` is immutable at its collection boundary and validates component identifiers, measurements, labels, and inspection costs. `Neighbor` and `Prediction` capture the output of the search and classification stages as explicit data structures.

`KNNModel<T>` provides a common neighborhood representation while exposing separate `classify` and `regress` operations. The task enum prevents accidental use of a regression model as a classifier or vice versa. Regression targets are checked for finite numeric values before the model accepts them.

The `StandardScaler` fits means and scales from the historical measurements. The classification model uses Euclidean distance and inverse-distance voting to assign an incoming component to an acceptable, review, or reject category. The regression model uses Manhattan distance to estimate inspection cost, illustrating that classification and regression can use different distance policies.

Java's typed records and enums make the intended domain behavior explicit. They do not remove the need for application-level authorization, persistent audit records, or external validation of the predictions.

The program calculates in-sample metrics for demonstration. A production evaluation should separate training and test records, preserve the training-only scaler, and measure class-specific errors. For manufacturing quality, recall for defective or rejected components can be more important than aggregate accuracy.

## PostgreSQL implementation

The SQL script models raw component measurements, scaler parameters, standardized observations, training partitions, and prediction requests.

`quality_components` preserves the source measurements and historical targets. Its primary key and unique asset code prevent duplicate component identifiers, while check constraints reject negative deviations and inspection costs. The quality status uses a PostgreSQL enum so that unsupported categorical labels cannot enter the table.

`scaler_parameters` stores the means and population standard deviations used to standardize each feature. `standardized_observations` stores the transformed values together with a scaler version. This separation supports traceability between raw measurements and the values used for distance calculations.

`training_observations` identifies which records are eligible to influence a prediction. A production workflow should assign training, validation, and test partitions before model selection and prevent validation or test records from entering the neighbor search.

### Distance ranking and classification

The classification view calculates Euclidean distances between each prediction request and each eligible training observation. A `ROW_NUMBER()` window function ranks candidates within each request, using component identifiers as a stable secondary ordering.

The query retains the first \(k\) candidates and aggregates their weights by quality class. The winning class is selected by vote weight, cumulative distance, and class ordering. Normalized vote weight provides an interpretable local vote fraction.

The regression view uses the same distance-ranking principle but aggregates historical inspection costs into a weighted mean. Separating the views keeps class selection distinct from continuous-target estimation.

### Indexes and query performance

The indexes on quality labels, training partitions, and request timestamps support filtering and request retrieval. They do not create a spatial index over the feature vectors, so the neighborhood queries still perform broad candidate comparisons.

For a training set of \(n\) records and \(d\) features, brute-force distance calculation requires approximately \(O(nd)\) work per query. Selecting the nearest neighbors by sorting requires \(O(n\log n)\) comparison work; partial-selection algorithms can reduce ranking work when \(k\) is small.

The SQL window queries also rank candidate distances for each request. As the training dataset grows, materialized features, carefully managed training partitions, approximate-neighbor structures, or an application-level search index may be necessary.

### Transactional behavior and integrity

The DDL, sample data, scaler fitting, transformation, prediction queries, and validation reports run inside a transaction. The final `COMMIT` makes the demonstration's changes persistent together.

Foreign keys maintain the relationship between component measurements, transformed observations, and training membership. Numeric checks reject invalid values at the database boundary. These constraints complement, rather than replace, application validation.

The SQL script includes a leave-one-out diagnostic query. It excludes each query component from its own candidate set before calculating a neighborhood prediction. This avoids the most direct form of self-neighbor leakage during the diagnostic calculation, although leave-one-out accuracy on a small demonstration dataset remains statistically uncertain.

## Model evaluation and common errors

**Data leakage:** Scaling before splitting or using test labels to select \(k\) contaminates evaluation. Learn preprocessing parameters and choose hyperparameters using training data and validation folds only.

**Unscaled features:** A feature with a large numeric range can dominate the neighborhood even if its predictive value is limited. Inspect feature distributions and compare validated scaling strategies.

**Poor neighborhood size:** A small \(k\) can be sensitive to noise. A large \(k\) can combine dissimilar observations. Select \(k\) using validation performance and application-specific error costs.

**Class imbalance:** Ordinary majority voting can favor a frequent class. Inspect the confusion matrix and class-specific precision and recall. Class weighting or carefully designed sampling may be necessary.

**Mislabeled duplicates:** Identical feature vectors with different labels can create contradictory predictions. Investigate data quality and the meaning of the labels before adjusting distance weights.

**Misinterpreted vote fractions:** A local vote fraction is not necessarily a calibrated probability. Validate calibration separately if downstream decisions require reliable probability estimates.

**Unseen feature ranges:** k-NN generally extrapolates poorly. A query far outside the training distribution can still receive a prediction because the algorithm always selects neighbors. Track nearest-neighbor distances and define an operational threshold for flagging out-of-distribution requests.

## Computational and production considerations

Brute-force k-NN is simple, interpretable, and effective for modest datasets, but prediction cost increases with the number of retained observations and features. Its memory requirement is approximately \(O(nd)\) for \(n\) observations with \(d\) features, excluding stored targets and indexing overhead.

Higher-dimensional data creates an additional problem: distance differences can become less informative, reducing the usefulness of nearest-neighbor rankings. Feature selection, domain-informed weighting, dimensionality reduction, or a different model family may be appropriate after validation.

A production system should version the training dataset, preprocessing parameters, metric, \(k\), weighting rule, and label definitions. Every prediction should be traceable to the model version and the relevant neighbor records. Operational decisions should use explicit thresholds, documented review procedures, and monitoring for changes in input distributions or prediction errors.

The strongest use of k-NN is not merely obtaining a label. It is obtaining a prediction together with the historical observations that influenced it, the distances that made them relevant, and an evaluation process that establishes when those local comparisons are trustworthy.
