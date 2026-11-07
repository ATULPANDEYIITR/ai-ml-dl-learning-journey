# Imbalanced Learning

## Project Overview

This project implements imbalanced classification workflows across Python, JavaScript, C++17, Java 17, and PostgreSQL.

The central application scenario is financial fraud detection, where legitimate transactions substantially outnumber fraudulent transactions.

The implementations examine four connected areas:

- Class imbalance and its effect on classification.
- Resampling through random oversampling and undersampling.
- Synthetic Minority Over-sampling Technique (SMOTE).
- Cost-sensitive learning through class weighting.

The five implementations use executable examples, validation rules, evaluation metrics, and reproducible experiment configurations.

## 1. Class Imbalance

Class imbalance occurs when the number of observations in one target class is substantially greater than the number in another.

For binary classification:

- Class 0 represents legitimate transactions.
- Class 1 represents fraudulent transactions.
- The majority class contains legitimate transactions.
- The minority class contains fraudulent transactions.

For example, a dataset containing 9,000 legitimate transactions and 1,000 fraudulent transactions has a 9:1 imbalance ratio.

The minority class represents only 10% of the observations.

### Imbalance ratio

The imbalance ratio is calculated as:

\[
IR = \frac{N_{\text{majority}}}{N_{\text{minority}}}
\]

An imbalance ratio of 1 indicates equal class frequencies. A larger value indicates greater imbalance.

Class imbalance does not automatically make a dataset unsuitable for classification. Its effect depends on the classification objective, feature distributions, decision costs, and evaluation method.

### Why accuracy can be misleading

Consider a dataset containing:

| Class | Observations |
|---|---:|
| Legitimate | 9,000 |
| Fraudulent | 1,000 |
| Total | 10,000 |

A classifier that predicts every transaction as legitimate achieves 90% accuracy.

Its fraudulent transaction recall is zero.

This demonstrates why accuracy alone is insufficient for imbalanced classification.

Important evaluation measures include:

| Metric | Interpretation |
|---|---|
| Precision | Fraction of positive predictions that are correct |
| Recall | Fraction of actual positive observations detected |
| Specificity | Fraction of actual negative observations correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Balanced accuracy | Average of positive recall and negative recall |
| PR-AUC | Area under the precision-recall curve |
| ROC-AUC | Area under the receiver operating characteristic curve |

Precision and recall are particularly important when the minority class represents a critical event.

PR-AUC is often informative when positive observations are rare.

## 2. Random Resampling

Resampling modifies the training dataset's class distribution before model fitting.

It does not change the original test dataset.

Two basic techniques are implemented.

### Random oversampling

Random oversampling increases the number of minority-class observations by duplicating existing records.

Suppose the training dataset contains:

| Class | Original | After oversampling |
|---|---:|---:|
| Majority | 800 | 800 |
| Minority | 100 | 800 |

The minority class receives 700 additional observations through duplication.

**Mechanism**

1. Separate the observations by class.
2. Identify the majority-class count.
3. Randomly select minority-class observations.
4. Duplicate observations until the target count is reached.
5. Combine and shuffle the resulting dataset.

**Advantages**

- Simple implementation.
- Preserves existing observations.
- Does not discard majority-class information.
- Suitable for establishing a resampling baseline.

**Limitations**

- Repeated records can encourage overfitting.
- Minority-class noise is also duplicated.
- The training dataset becomes larger.
- It does not introduce genuinely new feature combinations.

Random oversampling is useful when retaining all majority-class information is important.

### Random undersampling

Random undersampling reduces the majority-class observations to match the minority-class count.

For example:

| Class | Original | After undersampling |
|---|---:|---:|
| Majority | 800 | 100 |
| Minority | 100 | 100 |

**Mechanism**

1. Separate observations by class.
2. Determine the minority-class count.
3. Randomly select the same number of majority observations.
4. Combine and shuffle the selected records.

**Advantages**

- Smaller training datasets.
- Lower computational requirements.
- Removes majority-class redundancy.
- Can improve training efficiency.

**Limitations**

- Potentially useful majority-class information is discarded.
- Results can depend heavily on random selection.
- Important subgroups may disappear.

Undersampling is appropriate when the majority class contains substantial redundancy or computational constraints are important.

## 3. SMOTE

SMOTE stands for Synthetic Minority Over-sampling Technique.

Unlike random oversampling, SMOTE generates new minority-class observations through interpolation.

It uses neighboring minority-class observations to construct synthetic feature vectors.

### Mathematical mechanism

Let:

- \(x_i\) represent a selected minority observation.
- \(x_j\) represent one of its minority-class nearest neighbors.
- \(\lambda\) be a random number between zero and one.

A synthetic observation is generated using:

\[
x_{\text{synthetic}}
=
x_i+\lambda(x_j-x_i)
\]

For example, consider two observations:

\[
x_i=(2,4)
\]

\[
x_j=(6,8)
\]

For an interpolation factor of \(0.5\):

\[
x_{\text{synthetic}}=(4,6)
\]

The generated observation belongs to the minority class.

### SMOTE workflow

The implementations follow this sequence:

1. Identify minority-class observations.
2. Select one minority observation.
3. Calculate distances to other minority observations.
4. Identify the nearest \(k\) neighbors.
5. Randomly select one neighbor.
6. Generate an interpolation factor.
7. Calculate a synthetic feature vector.
8. Repeat until the required number of observations is generated.

The number of neighbors is configurable.

When fewer than the requested number of neighbors are available, the implementation reduces the neighborhood size.

At least two minority observations are required.

### Distance calculation

The implementations use Euclidean distance:

\[
d(x,y)=
\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

Feature scaling is important because features with larger numerical ranges can dominate distance calculations.

For production systems, numerical preprocessing should be fitted using training data only.

### Advantages

- Introduces new minority feature combinations.
- Reduces direct duplication.
- Can improve minority-class decision boundaries.
- Provides more training examples without discarding majority observations.

### Limitations

SMOTE does not guarantee that synthetic observations represent valid real-world events.

Potential problems include:

- Interpolation between observations from different minority subgroups.
- Synthetic samples near overlapping class boundaries.
- Generation of unrealistic feature combinations.
- Sensitivity to noisy minority observations.
- Increased computational cost from nearest-neighbor calculations.

For financial transactions, a synthetic transaction amount and transaction velocity may be mathematically valid but operationally unrealistic.

Domain constraints should therefore be checked after synthetic generation.

## 4. Class Weighting

Class weighting addresses imbalance without changing the number of observations.

It assigns different importance to observations belonging to different classes.

A frequently used balanced weighting formula is:

\[
w_c=\frac{N}{K \times N_c}
\]

Where:

- \(N\) is the total number of training observations.
- \(K\) is the number of classes.
- \(N_c\) is the number of observations in class \(c\).
- \(w_c\) is the weight assigned to class \(c\).

For a binary dataset containing 900 majority observations and 100 minority observations:

\[
w_0=\frac{1000}{2\times900}
\]

\[
w_1=\frac{1000}{2\times100}
\]

The resulting weights are approximately:

| Class | Weight |
|---|---:|
| Majority | 0.556 |
| Minority | 5.000 |

Minority-class errors therefore receive greater importance during model optimization.

### Weighted loss

For binary classification, weighted binary cross-entropy can be expressed as:

\[
L =
-\frac{1}{N}
\sum_{i=1}^{N}
w_{y_i}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
\]

Here, \(p_i\) represents the predicted positive-class probability.

Class weighting modifies the contribution of each observation to the training objective.

It does not generate additional observations.

### Practical characteristics

| Property | Class weighting |
|---|---|
| Dataset size | Unchanged |
| Duplicate observations | None introduced |
| Synthetic observations | None generated |
| Training objective | Modified |
| Minority importance | Increased |
| Implementation | Model-dependent |

Class weighting is particularly useful when retaining the original training observations is important.

Excessive weights can also increase sensitivity to mislabeled minority observations.

## 5. Distinction Between the Four Approaches

| Technique | Main operation | Dataset size | Main trade-off |
|---|---|---|---|
| Class imbalance analysis | Measures class distribution | Unchanged | Does not directly correct imbalance |
| Random oversampling | Duplicates minority observations | Increases | Repeated observations |
| Random undersampling | Removes majority observations | Decreases | Information loss |
| SMOTE | Interpolates minority observations | Increases | Synthetic data quality |
| Class weighting | Changes training importance | Unchanged | Weight selection |

These methods address related problems through different mechanisms.

They should be evaluated independently before combining them.

Applying multiple techniques without experimentation can overcorrect the class distribution or increase model bias.

## 6. Data Splitting and Leakage Prevention

A reliable imbalanced-learning workflow separates training, validation, and testing before applying resampling.

The recommended process is:

Dataset
|
+-- Training partition
|   |
|   +-- Fit preprocessing
|   +-- Calculate class weights
|   +-- Apply selected resampling method
|   +-- Train classifier
|
+-- Validation partition
|   |
|   +-- Tune hyperparameters
|   +-- Tune classification threshold
|
+-- Testing partition
    |
    +-- Evaluate final model
    +-- Calculate performance metrics

### Why splitting must happen first

Suppose SMOTE is applied before splitting.

A generated observation may be derived from minority observations that later appear in both training and testing partitions.

The model can then receive information related to the test observations during training.

This produces data leakage and an overly optimistic performance estimate.

The correct sequence is:

1. Split the original dataset.
2. Fit preprocessing using training observations.
3. Resample only the training observations.
4. Train the model.
5. Tune decisions using validation observations.
6. Evaluate once against the untouched test observations.

For repeated transactions belonging to the same customer, group-aware splitting is important.

A customer should not appear in both training and testing partitions when that would reveal customer-specific information.

Time-dependent datasets should also use chronological validation when future prediction is the intended task.

## 7. Python Implementation

File: `imbalanced_learning.py`

The Python implementation uses only standard-library modules.

### Core components

| Component | Responsibility |
|---|---|
| `Sample` | Immutable observation representation |
| `Strategy` | Sampling strategy enumeration |
| `Metrics` | Classification metric calculations |
| `generate_dataset()` | Creates reproducible synthetic observations |
| `stratified_split()` | Preserves class proportions across partitions |
| `random_oversample()` | Duplicates minority observations |
| `random_undersample()` | Reduces majority observations |
| `smote()` | Generates interpolated minority observations |
| `LogisticRegression` | Trains a binary classification model |
| `calculate_class_weights()` | Calculates balanced class weights |
| `run_experiment()` | Executes model evaluation |

### Logistic regression

The implementation trains logistic regression through batch gradient descent.

The sigmoid function is:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The prediction is:

\[
P(y=1|x)=\sigma(w^Tx+b)
\]

Class weights are incorporated into the gradient calculations.

L2 regularization helps control excessively large coefficients.

The implementation also checks feature dimensions, numerical validity, training parameters, and classification thresholds.

### Execution

Run the script using Python 3:

`python imbalanced_learning.py`

The program displays class distributions and compares the five experimental strategies using the same untouched testing dataset.

## 8. JavaScript Implementation

File: `imbalanced_learning.js`

The JavaScript implementation uses Node.js and focuses on event-driven experiment execution.

### Core components

| Component | Responsibility |
|---|---|
| `Dataset` | Validates and manages structured observations |
| `RandomGenerator` | Provides reproducible sampling |
| `stratifiedSplit()` | Separates observations by class |
| `randomOversample()` | Implements duplicate-based balancing |
| `randomUndersample()` | Implements majority reduction |
| `smote()` | Performs nearest-neighbor interpolation |
| `CostSensitiveClassifier` | Implements weighted logistic regression |
| `evaluate()` | Calculates classification metrics |
| `ExperimentRunner` | Executes and records experiments |

### Experiment event management

The `ExperimentRunner` maintains an event history.

Its events include:

- `experiment_started`
- `experiment_completed`
- `experiment_failed`

Each event records a unique identifier, timestamp, experiment name, and relevant details.

This structure demonstrates how an experiment service can maintain operational traceability.

The asynchronous execution interface allows experiments to participate in larger Node.js workflows without introducing third-party dependencies.

### Execution

`node imbalanced_learning.js`

The program executes the five strategies and reports their classification performance.

## 9. C++17 Implementation

File: `imbalanced_learning.cpp`

The C++ implementation uses a fraud-detection domain model and standard template library containers.

### Core components

| Component | Responsibility |
|---|---|
| `Record` | Represents a transaction observation |
| `Distribution` | Stores class-frequency information |
| `Metrics` | Maintains confusion-matrix counts |
| `Dataset` | Stores transaction observations |
| `stratifiedSplit()` | Separates training and testing records |
| `randomOversample()` | Implements random duplication |
| `randomUndersample()` | Implements random reduction |
| `applySMOTE()` | Implements synthetic interpolation |
| `WeightedCentroidClassifier` | Performs weighted centroid classification |
| `evaluate()` | Measures classification performance |

### Weighted centroid classifier

The C++ implementation uses a nearest-centroid classifier rather than translating the Python logistic regression model.

It calculates one centroid for each class.

The prediction is based on the distance between a new observation and the two class centroids.

Class weights modify the effective distance:

\[
S_c(x)=\frac{d(x,\mu_c)}{w_c}
\]

Where:

- \(x\) is the transaction being classified.
- \(\mu_c\) is the centroid for class \(c\).
- \(w_c\) is the corresponding class weight.

The class with the lower adjusted distance is selected.

This provides a compact example of how class weighting can influence a classification decision outside a gradient-based learning algorithm.

### Execution

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra imbalanced_learning.cpp -o imbalanced_learning`

Run:

`./imbalanced_learning`

On Windows:

`imbalanced_learning.exe`

## 10. Java 17 Implementation

File: `ImbalancedLearning.java`

The Java implementation models an enterprise classification workflow using explicit domain types and service classes.

### Core components

| Component | Responsibility |
|---|---|
| `Transaction` | Immutable transaction representation |
| `Distribution` | Class-distribution information |
| `Metrics` | Evaluation results |
| `ExperimentStatus` | Experiment lifecycle |
| `SamplingStrategy` | Sampling method identification |
| `DatasetValidator` | Dataset integrity checks |
| `SplitService` | Stratified data partitioning |
| `SamplingService` | Common sampling interface |
| `SMOTESampler` | Synthetic minority generation |
| `ClassWeightPolicy` | Class-weight calculation |
| `WeightedCentroidClassifier` | Classification |
| `EvaluationService` | Performance measurement |
| `ExperimentService` | Experiment execution and error handling |

### Enterprise-oriented design

The implementation uses records for immutable domain data and interfaces for sampling services.

Experiment states are represented explicitly:

`CREATED → VALIDATED → TRAINED → EVALUATED`

A failed experiment transitions to `FAILED`.

The experiment service handles runtime failures independently, allowing other experiments to continue.

The immutable transaction representation prevents callers from directly modifying the internal feature array.

The service architecture also separates dataset validation, sampling, model fitting, and evaluation.

### Execution

Compile:

`javac ImbalancedLearning.java`

Run:

`java ImbalancedLearning`

No third-party dependencies are required.

## 11. PostgreSQL Implementation

File: `imbalanced_learning.sql`

The PostgreSQL implementation provides a relational model for managing imbalanced-learning experiments.

The schema separates original observations, synthetic observations, sampling configurations, model experiments, predictions, and evaluation results.

### Relational architecture

Datasets
|
+-- Observations
|   |
|   +-- Original observations
|   +-- Synthetic observations
|
+-- Sampling strategies
|   |
|   +-- Resampling runs
|
+-- Class weights
|
+-- Model experiments
    |
    +-- Predictions
    |
    +-- Evaluation metrics

### Database objects

| Object | Purpose |
|---|---|
| `datasets` | Stores dataset identity and partition role |
| `observations` | Stores transaction features and labels |
| `sampling_strategies` | Defines sampling methods |
| `resampling_runs` | Tracks resampling operations |
| `class_weights` | Stores class-specific weights |
| `model_experiments` | Registers model experiments |
| `predictions` | Stores actual and predicted labels |
| `evaluation_metrics` | Stores calculated performance |

### Integrity enforcement

The schema uses primary keys, foreign keys, unique constraints, and check constraints.

The observation trigger enforces training-only synthetic record insertion.

The experiment validation trigger verifies that:

- The training dataset has the training role.
- The testing dataset has the testing role.
- The two partitions do not share customer references.

These controls help prevent common dataset-management errors.

### Stored functions and procedures

`get_class_distribution()` calculates class frequencies and the imbalance ratio.

`calculate_class_weights()` calculates balanced class weights and stores them for a training dataset.

`calculate_experiment_metrics()` calculates confusion-matrix statistics and performance measures from stored predictions.

### Reporting

The database includes:

- `dataset_distribution_report`
- `experiment_performance_report`

Additional queries demonstrate class distribution analysis, synthetic record retrieval, aggregation, evaluation, and experiment eligibility.

The SQL script uses transaction handling and a savepoint to demonstrate recovery from an expected validation failure.

### Execution

Run the complete script in a PostgreSQL 15+ database using a SQL client.

The schema is created under `imbalanced_learning`.

The script uses `generate_series()` to create reproducible sample transaction records.

## 12. Evaluation and Model Selection

Different sampling methods can produce different classification outcomes.

A comparison should use the same original test partition.

Consider the following decision criteria:

| Business objective | Important measure |
|---|---|
| Detect as many fraudulent transactions as possible | Recall |
| Reduce incorrect fraud alerts | Precision |
| Balance precision and recall | F1-score |
| Maintain performance across both classes | Balanced accuracy |
| Evaluate rare positive predictions | PR-AUC |
| Account for monetary consequences | Expected business cost |

### Confusion matrix

A binary classification confusion matrix contains four outcomes:

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | True Negative | False Positive |
| Positive | False Negative | True Positive |

In fraud detection:

- True Positive: Fraud correctly identified.
- True Negative: Legitimate transaction correctly accepted.
- False Positive: Legitimate transaction incorrectly flagged.
- False Negative: Fraudulent transaction incorrectly accepted.

False negatives may produce direct financial losses.

False positives may create customer inconvenience, investigation costs, and operational workload.

The preferred model therefore depends on the relative costs of these outcomes.

### Classification threshold

A probability threshold determines how a predicted probability becomes a class label.

\[
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1|x)\geq t\\
0 & \text{otherwise}
\end{cases}
\]

The default threshold in the implementations is 0.5.

A lower threshold can increase recall while reducing precision.

A threshold should be selected using validation data and an explicit business-cost objective.

## 13. Computational Complexity

Let:

- \(N\) be the number of observations.
- \(M\) be the number of features.
- \(K\) be the number of nearest neighbors.
- \(E\) be the number of training epochs.

| Operation | Approximate complexity |
|---|---|
| Class distribution | \(O(N)\) |
| Random oversampling | \(O(N)\) expected |
| Random undersampling | \(O(N)\) |
| One SMOTE neighbor search | \(O(N_m^2M)\) |
| SMOTE generation | \(O(GN_mM)\) with sorting per generated sample |
| Logistic regression training | \(O(ENM)\) |
| Centroid calculation | \(O(NM)\) |
| Centroid prediction | \(O(M)\) |

Here, \(N_m\) represents the minority-class count and \(G\) represents the number of generated observations.

The provided SMOTE implementations use straightforward nearest-neighbor calculations.

For larger datasets, optimized nearest-neighbor indexes or approximate search methods can reduce computation.

Memory usage also increases during oversampling and synthetic data generation because additional records are retained.

## 14. Validation and Edge Cases

The implementations explicitly address several failure conditions.

| Condition | Required handling |
|---|---|
| Empty dataset | Reject input |
| Missing class | Reject binary classification experiment |
| Single minority observation | Reject SMOTE |
| Invalid feature values | Reject non-finite values |
| Inconsistent feature dimensions | Reject dataset |
| Invalid class weights | Reject training configuration |
| Invalid prediction threshold | Reject prediction request |
| Insufficient neighbors | Reduce SMOTE neighborhood size |
| Empty metric denominator | Return zero for undefined precision or recall |
| Training/testing customer overlap | Reject experiment registration |
| Synthetic test observations | Reject database insertion |

A production implementation should also validate feature ranges, categorical values, business rules, duplicate observations, and generated record plausibility.

## 15. Production Considerations

### Data quality

- Validate missing values before model training.
- Normalize numerical features before distance-based sampling.
- Validate class labels and their provenance.
- Detect duplicate observations.
- Monitor changes in minority-class prevalence.
- Check synthetic records against domain rules.

### Experiment governance

- Record dataset versions.
- Record random seeds.
- Store sampling parameters.
- Maintain model versions.
- Preserve original class distributions.
- Record evaluation metrics.
- Maintain experiment execution logs.

### Security and privacy

Financial transaction data may contain sensitive customer information.

Production systems should implement:

- Role-based database access.
- Encryption at rest and in transit.
- Restricted access to raw transaction records.
- Audit logging.
- Data retention policies.
- Appropriate masking of customer identifiers.
- Controlled access to synthetic datasets.

Synthetic observations should not automatically be treated as anonymous or free of privacy risk.

### Model monitoring

Class imbalance can change over time.

A fraud detection system may experience an increase in fraudulent activity, changes in transaction patterns, or changes in customer behavior.

Operational monitoring should track:

- Class prevalence.
- Minority-class recall.
- Minority-class precision.
- False-positive rate.
- False-negative rate.
- Balanced accuracy.
- Prediction probability distribution.
- Feature distribution changes.
- Model drift.
- Investigation workload.

Retraining should be based on validated performance deterioration and changes in the underlying data distribution.

## 16. Important Implementation Limitations

The five programs are educational implementations rather than interchangeable production ML frameworks.

The Python and JavaScript implementations use logistic regression with class-weighted gradient calculations.

The C++17 and Java 17 implementations use weighted nearest-centroid classification to demonstrate a different model architecture.

The SQL implementation manages experiment data and calculates evaluation metrics. It does not independently train a machine-learning model or implement the SMOTE nearest-neighbor algorithm inside PostgreSQL.

The synthetic datasets are deliberately small to keep execution practical and results reproducible.

The basic SMOTE implementations support numerical binary-classification features. They do not implement categorical SMOTE, boundary-aware SMOTE, or specialized methods for high-dimensional datasets.

The centroid classifier's distance adjustment is an educational class-weighting mechanism, not a replacement for a calibrated probabilistic loss function.

For real-world deployment, preprocessing, resampling, model training, threshold selection, calibration, and evaluation must be performed within a leakage-safe experimental workflow.
