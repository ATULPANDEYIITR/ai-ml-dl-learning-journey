# Classical Machine Learning Project

## Project Scope

This project implements a reproducible binary classification workflow using classical machine learning techniques. It covers dataset acquisition, exploratory data analysis, deterministic feature engineering, data validation, preprocessing, model training, model comparison, evaluation, and deployment-quality checks.

The primary Python implementation uses the Wisconsin Diagnostic Breast Cancer dataset distributed with scikit-learn. The positive class is defined as `malignant = 1`; the benign class is `0`.

The other implementations examine complementary parts of the same engineering discipline:

| Implementation | Main responsibility |
|---|---|
| Python | Complete dataset-to-model experiment |
| JavaScript | Event-driven model evaluation and deployment gate |
| C++17 | CSV ingestion, training-only preprocessing, and logistic regression |
| Java 17 | Experiment lifecycle, candidate selection, model registry, and promotion policy |
| PostgreSQL | Dataset lineage, experiment records, evaluations, and deployment decisions |

The dataset is a public educational benchmark. Model outputs are not clinical diagnoses and must not be used to make medical decisions.

## End-to-End Workflow

The project separates data preparation, model selection, final evaluation, and deployment eligibility.

```text
Dataset acquisition
       |
       v
Dataset validation
       |
       v
Exploratory data analysis
       |
       v
Feature engineering
       |
       v
Stratified train / validation / test split
       |
       +-------------------------------+
       |                               |
       v                               v
Training partition              Final test partition
       |                         Reserved from selection
       v
Fold-specific preprocessing
       |
       v
Model training and cross-validation
       |
       v
Candidate selection
       |
       v
Final model fitting
       |
       v
Holdout evaluation
       |
       v
Deployment quality policy
       |
       v
Eligible or blocked
```

A test partition must not influence feature selection, preprocessing parameter estimation, hyperparameter tuning, or model selection. The project keeps learned preprocessing inside the training workflow to prevent information from the evaluation partition from influencing the fitted model.

## Dataset and Target

### Dataset characteristics

The Python implementation loads scikit-learn's Wisconsin Diagnostic Breast Cancer dataset through `load_breast_cancer(as_frame=True)`.

The original dataset contains 569 observations and 30 numerical measurements. The target supplied by scikit-learn assigns `0` to malignant observations and `1` to benign observations. The Python implementation reverses the interpretation into a binary target where malignant observations have label `1`.

This convention makes recall directly represent the proportion of malignant observations correctly identified.

The dataset contains measurements derived from digitized fine-needle aspirate images. Features describe properties such as radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry.

### Feature engineering

Three deterministic features are added to the original numerical measurements:

| Engineered feature | Definition | Purpose |
|---|---|---|
| `worst_to_mean_radius_ratio` | Worst radius divided by mean radius | Represents relative radius variation |
| `mean_area_perimeter_ratio` | Mean area divided by squared mean perimeter | Represents a scale-adjusted geometric relationship |
| `texture_radius_interaction` | Mean texture multiplied by mean radius | Represents a joint effect of two measurements |

These transformations operate on individual observations. They do not learn parameters from the complete dataset.

Division by zero is converted into a missing value, which is handled by the fitted imputation stage. Engineered features should be retained only when their definitions are meaningful for the data domain.

## Exploratory Data Analysis

EDA is used to inspect dataset structure and identify potential data-quality or modelling issues before training.

The Python implementation generates:

| Artifact | Contents |
|---|---|
| `dataset_profile.json` | Dataset dimensions, target distribution, missing-value counts, descriptive statistics, and strongest feature correlations |
| `feature_correlations.csv` | Highest absolute pairwise feature correlations |
| `dataset.csv` | Feature table and positive-class target used by the other implementations |

Descriptive statistics include count, mean, standard deviation, minimum, quartiles, and maximum. The class distribution identifies the number of benign and malignant observations.

Pairwise correlation can reveal redundant measurements. Highly correlated predictors may affect coefficient interpretation in linear models. Correlation alone does not establish causal relationships or prove that a feature should be removed.

The profile is descriptive. It is not used to fit imputation values or scaling parameters.

## Data Splitting and Leakage Prevention

The Python implementation uses a stratified holdout split with a default test fraction of 20%.

Stratification preserves approximate class proportions across the training and test partitions. This matters when one target class is less frequent than another.

The remaining training partition is evaluated with shuffled stratified cross-validation. Each fold independently fits the preprocessing and model stages through scikit-learn's `Pipeline` and `cross_validate`.

The Python implementation uses the following sequence:

1. Load the complete dataset and calculate deterministic row-level features.
2. Create a stratified training and test split.
3. Fit each candidate pipeline through cross-validation using training observations.
4. Select a candidate using mean cross-validation F1, with mean cross-validation ROC AUC as the tie-breaker.
5. Fit the selected candidate on the complete training partition.
6. Evaluate the fitted candidate on the reserved test partition.

The holdout metrics are reported for every candidate, but the champion is selected using training-only cross-validation results. This avoids choosing a model based on its performance on the final test observations.

## Preprocessing

Preprocessing converts raw values into a representation suitable for a selected model.

### Missing-value imputation

The Python implementation uses `SimpleImputer(strategy="median")`. Median imputation is less sensitive to extreme values than mean imputation.

The C++ and Java implementations independently calculate feature medians from training observations. A feature that is entirely missing in the training partition is rejected because no valid training-derived median can be estimated.

### Standardization

Standardization transforms a numerical feature using:

\[
z = \frac{x-\mu}{\sigma}
\]

Here, \(\mu\) is the training mean and \(\sigma\) is the training standard deviation.

Standardization is useful for logistic regression because features with different numerical scales can otherwise affect optimization. Tree-based models do not generally require standardized inputs, so the Python implementation uses separate preprocessing configurations for its linear and tree-based candidates.

A constant feature receives a scale of one in the C++ and Java implementations. This prevents division by zero without creating artificial variation.

### Training-only fitting

Imputation medians, means, and standard deviations are learned from training data. The same fitted parameters are then applied to validation and test observations.

Refitting these values on the complete dataset would expose the model-development process to information from observations that are meant to simulate unseen data.

## Model Implementations

### Python model comparison

The Python implementation compares three classification candidates and one simple baseline.

| Model | Main mechanism | Preprocessing |
|---|---|---|
| Dummy classifier | Predicts the most frequent training class | Median imputation |
| Logistic regression | Estimates a linear decision function using logistic loss | Median imputation and standardization |
| Random forest | Aggregates predictions from multiple decision trees | Median imputation |
| Histogram gradient boosting | Builds sequential trees that improve earlier predictions | Median imputation |

Logistic regression uses class weighting and an increased iteration limit. Random forest uses class-balanced bootstrap sampling and a minimum leaf size of two. Histogram gradient boosting uses a limited tree depth structure and L2 regularization.

The dummy classifier provides a baseline. A model should be compared with a simple reference rather than evaluated only through its own metric values.

### C++17 logistic regression

The C++ implementation provides a separate logistic regression training engine.

Its main mechanisms are:

- CSV input with quoted-field handling.
- Binary target and numerical-feature validation.
- Stratified holdout splitting.
- Median imputation fitted on training observations.
- Standardization fitted on training observations.
- Batch gradient descent.
- Logistic probability estimation.
- L2 weight regularization.
- Holdout metric calculation.
- ROC AUC calculation through ranked positive and negative scores.

The gradient update uses the average prediction error for each coefficient and adds the L2 penalty to the coefficient gradient. The intercept is not penalized.

The implementation uses a fixed learning rate and maximum epoch count. A loss-change condition stops optimization when improvement becomes sufficiently small. Learning rate, regularization, convergence tolerance, and iteration count are practical model parameters that need careful validation for a different dataset.

When no input file is supplied, the program creates a deterministic synthetic binary dataset. This makes the C++ program independently executable. Passing the Python-generated `dataset.csv` uses the same curated input table.

## Model Evaluation

The project uses metrics that describe different aspects of classification quality.

| Metric | Interpretation |
|---|---|
| Accuracy | Fraction of all observations classified correctly |
| Precision | Fraction of positive predictions that are actually positive |
| Recall | Fraction of actual positive observations detected |
| F1 | Harmonic mean of precision and recall |
| ROC AUC | Ranking quality across classification thresholds |

For malignant classification, recall measures how many actual malignant observations are detected. Precision measures how many observations predicted as malignant are actually malignant.

Accuracy alone can conceal poor performance on a minority class. A classifier can achieve a high accuracy score while failing to identify a substantial portion of positive observations.

The confusion matrix separates predictions into four outcomes:

| Outcome | Meaning |
|---|---|
| True negative | Benign observation predicted as benign |
| False positive | Benign observation predicted as malignant |
| False negative | Malignant observation predicted as benign |
| True positive | Malignant observation predicted as malignant |

The standard probability threshold is 0.5 in these examples. In a real application, a threshold must be selected using domain requirements and validation data. Changing the threshold alters precision and recall.

ROC AUC assesses ranking rather than performance at one fixed threshold. A high ROC AUC does not guarantee that the chosen operating threshold satisfies the intended use case.

## JavaScript Deployment-Gate Service

The JavaScript implementation consumes the `metrics.json` report emitted by Python. It does not retrain the models. It evaluates the reported metrics against an explicit quality policy.

The default policy checks:

| Criterion | Required condition |
|---|---|
| Positive class | Must be `malignant` |
| Holdout recall | At least 0.90 |
| Holdout precision | At least 0.80 |
| Holdout ROC AUC | At least 0.95 |
| Mean cross-validation F1 | At least 0.90 |
| Cross-validation F1 standard deviation | At most 0.05 |

A model is marked `ELIGIBLE` only when every criterion passes. Otherwise, it is marked `BLOCKED`, and the failed criteria are retained as reasons.

`DeploymentPolicy` owns threshold validation and policy evaluation. `ExperimentGate` coordinates the evaluation process and extends `EventEmitter` to publish events for experiment loading, individual model decisions, and experiment completion.

The service uses asynchronous processing for model runs. `Promise.all` allows multiple independent evaluations to be processed concurrently. Input reports are size-limited and parsed as JSON without executing their contents.

Eligibility is a policy result, not proof that a model is suitable for a particular real-world application. Policy thresholds must be justified by the intended use case, data quality, and consequences of classification errors.

## Java 17 Enterprise Experiment Service

The Java implementation emphasizes experiment lifecycle management and model registration.

It creates a versioned synthetic dataset with six numerical features and a malignant positive class. Small amounts of missingness are introduced to exercise imputation.

### Experiment lifecycle

The explicit experiment states are:

| State | Meaning |
|---|---|
| `CREATED` | Experiment record exists |
| `DATA_VALIDATED` | Dataset validation has passed |
| `TRAINED` | Candidate training has completed |
| `EVALUATED` | Final selected model has been evaluated |
| `PROMOTED` | The model meets the configured policy |
| `REJECTED` | The model fails the configured policy |

Transitions are checked by the `Experiment` class. Terminal states cannot be reused as intermediate states.

### Candidate selection

The Java service compares Gaussian Naive Bayes and K-nearest neighbors.

Gaussian Naive Bayes estimates per-class feature distributions and class priors. It calculates log posterior scores and converts their difference into a positive-class probability.

K-nearest neighbors compares a new observation with stored standardized training observations. It returns the fraction of the nearest neighbors belonging to the positive class.

The candidates are evaluated on a validation partition. The model with the highest validation F1 is selected, with ROC AUC as the tie-breaker.

The chosen algorithm is then refitted using training and validation observations. Its final holdout evaluation uses the untouched test partition.

### Enterprise domain types

The Java implementation uses records for immutable dataset, observation, metric, model version, and result structures.

`DatasetValidationService` checks minimum dataset size, feature dimensions, unique observation identifiers, and the presence of both target classes.

`PreprocessingModel` owns training-derived imputation and scaling parameters. Its `transform` method applies those parameters without refitting them.

`ModelRegistry` records model versions and prevents duplicate version identifiers. `PromotionPolicy` evaluates the final model metrics and returns explicit failure reasons.

The registry is in-memory. A persistent production registry would require durable storage, access control, artifact checksums, audit history, and concurrent-update handling.

## PostgreSQL Experiment Registry

The PostgreSQL script models experiment metadata and governance in a relational structure.

It stores experiment records rather than the complete numerical matrix used to train a model.

### Relational entities

| Table | Responsibility |
|---|---|
| `dataset_versions` | Dataset identity, version, dimensions, source, and lifecycle state |
| `feature_catalog` | Feature metadata, order, missing counts, and engineering expressions |
| `data_quality_checks` | Named validation outcomes and their structured details |
| `experiments` | Algorithm configuration, random seed, dataset association, and lifecycle state |
| `data_splits` | Training, validation, and test partition counts and fingerprints |
| `preprocessing_steps` | Transformation configuration and fitting scope |
| `model_runs` | Model name, version, training state, and artifact location |
| `model_evaluations` | Holdout and cross-validation performance |
| `deployment_policies` | Required metrics and quality constraints |
| `deployment_decisions` | Eligibility outcome and recorded blocking reasons |

### Data integrity

Primary keys identify records. Foreign keys connect experiment records to dataset versions, model evaluations to model runs, and deployment decisions to policies.

Unique constraints prevent duplicate dataset versions, feature names within a dataset, experiment keys, and model versions within an experiment.

Check constraints enforce valid numerical ranges, dataset dimensions, experiment split proportions, lifecycle values, and evaluation structures.

The experiment transition trigger rejects illegal state changes. For example, an experiment already marked as completed cannot be returned to the running state.

### Evaluation and eligibility

`model_evaluations` distinguishes `holdout_test` results from `cross_validation` results.

Holdout records retain confusion-matrix counts. Cross-validation records retain the mean and standard deviation of F1. The constraints prevent these two evaluation types from being mixed.

The `evaluate_deployment` function checks:

- Whether the dataset is validated.
- Whether its quality checks exist and have passed.
- Whether the experiment and model run completed.
- Whether holdout recall, precision, and ROC AUC satisfy policy thresholds.
- Whether cross-validation F1 and its standard deviation satisfy policy thresholds.
- Whether learned preprocessing uses training-only fitted parameters.

The function inserts a durable decision record and returns its status and reasons. Repeated evaluations create separate decision records, preserving an evaluation history.

### Query patterns

The script includes queries for:

- Dataset and model-version retrieval.
- Data-quality inspection.
- Grouped model-performance reporting.
- Weighted candidate ranking using a common table expression.
- Transactional deployment-policy evaluation.
- Eligibility history.
- Experiment and model lifecycle counts.
- Feature and feature-engineering lineage.

Indexes support common joins and retrieval paths, particularly dataset-to-experiment lookup, experiment-to-model lookup, evaluation retrieval by model and split, and quality-check filtering by dataset.

## Running the Implementations

### Python

Install the required packages:

`pip install numpy pandas scikit-learn joblib`

Run the complete experiment:

`python classical_ml_project.py`

Use custom output and split settings:

`python classical_ml_project.py --output-dir results --test-size 0.2 --folds 5 --random-state 42`

Generated artifacts include:

| File | Purpose |
|---|---|
| `dataset.csv` | Numerical feature table and target |
| `dataset_profile.json` | EDA statistics and class distribution |
| `feature_correlations.csv` | Correlation report |
| `model_comparison.csv` | Cross-validation and test metrics |
| `metrics.json` | Structured model evaluation report |
| `champion_predictions.csv` | Actual labels, predicted labels, probabilities, and correctness |
| `champion_model.joblib` | Serialized selected pipeline |

Joblib uses Python object serialization. Loading an untrusted serialized model can execute malicious code. Only load artifacts from trusted sources.

### JavaScript

Run with the Python evaluation report:

`node ml_deployment_gate.js classical_ml_results/metrics.json`

The service writes `deployment_gate_report.json` by default.

Without an input file, the service evaluates its built-in demonstration report:

`node ml_deployment_gate.js`

An optional second argument sets the output report path.

### C++17

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic ml_engine.cpp -o ml_engine`

Run with its synthetic demonstration dataset:

`./ml_engine`

Run against the Python-exported dataset:

`./ml_engine classical_ml_results/dataset.csv`

The CSV must contain numerical feature columns and a binary column named `target`. Empty feature fields are treated as missing values.

### Java 17

Compile:

`javac ClassicalMLEnterprise.java`

Run:

`java ClassicalMLEnterprise`

The Java implementation generates its own deterministic synthetic dataset and requires no external libraries.

### PostgreSQL

Execute the complete SQL script against a database where the executing role has permission to create schemas, tables, functions, triggers, and indexes.

For example, using `psql`:

`psql -d your_database -v ON_ERROR_STOP=1 -f classical_ml.sql`

The script drops and recreates the `classical_ml` schema. Do not execute it against a schema containing data that must be retained.

## Limitations and Engineering Considerations

### Dataset limitations

The Python implementation uses a small public benchmark with a limited number of observations and numerical features. Its feature distributions and target relationships may not represent data collected in another environment.

The C++ and Java default examples use generated synthetic observations. Synthetic separation is controlled by the chosen generation distributions and is not evidence of real-world model performance.

The SQL script stores experiment metadata and illustrative results. It does not contain the complete underlying observation matrix or execute model training inside PostgreSQL.

### Evaluation limitations

A single train/test split can produce a result that depends on the random seed. Cross-validation provides a more stable view of training-data performance, but it does not remove dataset bias or guarantee future performance.

F1 and ROC AUC are not interchangeable. Threshold selection should be based on validation data, while final test evaluation should remain a separate assessment.

Model choice should account for the cost of false positives and false negatives, data quality, expected operating conditions, and the impact of a wrong prediction.

### Operational controls

A dependable experiment record should preserve:

- Dataset version and input fingerprint.
- Target definition and positive-class meaning.
- Feature names, order, and engineering expressions.
- Split seed and partition fingerprints.
- Preprocessing configurations and fitted parameters.
- Algorithm version and hyperparameters.
- Cross-validation and holdout evaluation results.
- Model artifact identity and integrity checks.
- Deployment policy version and decision history.

Model artifacts should be stored with controlled access. Input data and report files should be validated before processing. Training and evaluation records should be retained to make results auditable and reproducible.

The implementations provide different views of the same classical machine learning workflow: numerical experimentation, policy evaluation, efficient model computation, enterprise lifecycle management, and relational experiment governance.
