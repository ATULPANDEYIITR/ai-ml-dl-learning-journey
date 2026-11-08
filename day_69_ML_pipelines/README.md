# ML Pipelines: Transformers, Preprocessing and Model Workflows

## 1. Introduction

A machine learning pipeline is an ordered sequence of data-processing and model-training operations. It connects raw input data to predictions through reusable, fitted transformations and an estimator.

The central purpose of a pipeline is to ensure that the same preprocessing operations are applied consistently during training, evaluation, and inference.

This project implements a customer churn classification workflow using numerical and categorical customer attributes.

The implementations cover:

- Data validation and feature schemas.
- Missing-value handling.
- Numerical standardization.
- Categorical encoding.
- Transformer composition.
- Logistic regression.
- Training, validation, and testing.
- Model evaluation.
- Pipeline state management.
- Artifact persistence.
- Inference workflows.
- Relational storage of pipeline configurations and execution records.

## 2. Core Architecture

The implemented workflow follows this sequence:

Raw Dataset
    |
    v
Schema Validation
    |
    v
Training / Validation / Test Split
    |
    v
Missing-Value Imputation
    |
    v
Numerical Scaling
    |
    v
Categorical Encoding
    |
    v
Feature Representation
    |
    v
Model Training
    |
    v
Model Evaluation
    |
    v
Pipeline Artifact
    |
    v
Inference

Each stage has a distinct responsibility. A transformer modifies or derives features, while an estimator learns a predictive relationship from the resulting features.

## 3. Transformers

A transformer is a component that learns parameters from input data and applies a transformation to that data.

Most fitted transformers expose two important operations:

| Operation | Responsibility |
|---|---|
| `fit()` | Learn transformation parameters |
| `transform()` | Apply learned parameters |
| `fit_transform()` | Fit and transform training data |

For example, a numerical scaler calculates means and standard deviations from its training data. It subsequently applies those same values to validation, test, and inference records.

A transformer should not independently calculate new statistics from inference data.

### 3.1 Median imputation

Missing numerical values can prevent many estimators from processing a dataset.

Median imputation replaces missing numerical observations with the median of the corresponding training feature.

For observations:

`[20, 25, 30, missing, 40]`

The replacement value is:

`30`

The median is less affected by extreme observations than the mean.

The Python, JavaScript, C++17, and Java implementations include numerical median imputation.

The transformation also handles missing inference values using the previously fitted median.

An entirely missing feature cannot provide a meaningful median and must be rejected or handled through an explicitly defined alternative policy.

### 3.2 Standardization

Numerical features may have significantly different scales.

For example:

| Feature | Typical range |
|---|---:|
| Age | 18–75 |
| Monthly spending | 20–150 |
| Tenure | 1–72 months |

Standardization converts numerical features using:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) represents the original observation.
- \(\mu\) represents the training mean.
- \(\sigma\) represents the training standard deviation.
- \(z\) represents the standardized value.

A feature with zero variance requires special handling because division by zero is undefined. The implementations return zero for such standardized features.

Standardization is particularly useful for optimization-based models such as logistic regression.

### 3.3 One-hot encoding

Categorical features cannot generally be passed directly to numerical estimators.

Consider the contract feature:

| Original value | Encoded representation |
|---|---|
| monthly | 1, 0, 0 |
| annual | 0, 1, 0 |
| two_year | 0, 0, 1 |

One-hot encoding creates separate binary features for the known categories.

The implementations use a consistent category mapping between training and inference.

Unknown categories are handled without creating additional model columns. Their encoded group contains zeros.

For production systems, category mappings must be versioned alongside the model.

## 4. Transformer Composition

A preprocessing workflow normally contains multiple transformers.

For example:

`MedianImputer → StandardScaler → OneHotEncoder`

Each transformation has a defined input and output.

| Transformer | Input | Output |
|---|---|---|
| MedianImputer | Incomplete records | Complete records |
| StandardScaler | Numerical features | Standardized numerical features |
| OneHotEncoder | Categorical features | Binary indicator features |

The sequence matters.

Imputation must occur before scaling because numerical statistics cannot be calculated reliably from missing values.

The transformations must also preserve feature ordering and naming so that the estimator receives the expected inputs.

### Column-specific preprocessing

Real datasets commonly contain mixed feature types.

A numerical preprocessing workflow can use median imputation and standardization, while a categorical workflow uses most-frequent imputation and one-hot encoding.

The Python implementation uses `ColumnTransformer` to apply different preprocessing branches to different columns.

This avoids applying numerical scaling to categorical variables or categorical encoding to continuous numerical features.

## 5. Complete Model Pipelines

A complete model pipeline combines preprocessing and prediction into one executable object.

In the Python implementation, the architecture uses:

| Component | Implementation |
|---|---|
| Numerical preprocessing | `Pipeline` |
| Categorical preprocessing | `Pipeline` |
| Column selection | `ColumnTransformer` |
| Model | `LogisticRegression` |
| Complete workflow | `Pipeline` |

The resulting estimator can perform prediction directly on raw input records.

The caller does not need to manually repeat imputation, scaling, or encoding.

This reduces inconsistencies between training and production inference.

## 6. Training, Validation and Testing

A dataset should be divided before fitting transformations.

The implementations use separate training and evaluation records.

| Partition | Purpose |
|---|---|
| Training | Fit transformations and train the estimator |
| Validation | Evaluate candidate configurations and tune parameters |
| Testing | Estimate final performance on unseen records |

For binary classification, stratified splitting helps preserve class proportions across partitions.

### Data leakage prevention

Data leakage occurs when information unavailable during legitimate model training influences training or evaluation.

An example is calculating a standardization mean using the complete dataset before creating training and testing partitions.

This allows information from evaluation observations to influence training transformations.

The correct sequence is:

1. Split the original dataset.
2. Fit preprocessing on training records.
3. Transform training records.
4. Transform validation and test records using fitted parameters.
5. Train and evaluate the model.

The Python implementation uses a scikit-learn pipeline inside `GridSearchCV`. Consequently, each cross-validation training fold fits its own preprocessing transformations.

This is an important protection against preprocessing leakage.

## 7. Logistic Regression

Logistic regression is a supervised classification algorithm.

For binary classification, it estimates the probability of the positive class.

\[
P(y=1|x)=\frac{1}{1+e^{-z}}
\]

Where:

\[
z=w_0+w_1x_1+\cdots+w_nx_n
\]

The parameters are learned by minimizing a loss function.

The JavaScript, C++17, and Java implementations use batch gradient descent with L2 regularization.

The parameter update is based on:

\[
w_j \leftarrow w_j -
\alpha\left(
\frac{1}{m}\sum_{i=1}^{m}
(\hat y_i-y_i)x_{ij}
+\lambda w_j
\right)
\]

Where:

- \(\alpha\) is the learning rate.
- \(m\) is the number of training observations.
- \(\lambda\) controls regularization.
- \(w_j\) is a model parameter.

The intercept is not regularized in these implementations.

The C++ and Java implementations use typed feature structures and explicit estimator classes rather than relying on external machine learning libraries.

## 8. Model Evaluation

A trained model must be evaluated using records that were not used to fit its parameters.

The implementations calculate classification metrics.

### 8.1 Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

Accuracy measures the proportion of correct predictions.

It can be misleading when class distributions are imbalanced.

### 8.2 Precision

\[
Precision=\frac{TP}{TP+FP}
\]

Precision measures how many predicted positive observations are actually positive.

### 8.3 Recall

\[
Recall=\frac{TP}{TP+FN}
\]

Recall measures how many actual positive observations are correctly identified.

### 8.4 F1 score

\[
F1=2\frac{Precision\times Recall}
{Precision+Recall}
\]

F1 balances precision and recall.

### 8.5 ROC AUC

ROC AUC evaluates the ranking quality of predicted probabilities across classification thresholds.

It is useful when a single threshold does not adequately describe classification performance.

The Python implementation uses scikit-learn's metric functions. The JavaScript implementation calculates ranking-based AUC, while C++17 and Java demonstrate confusion-matrix-based evaluation.

The reported metrics are illustrative outputs from independently implemented training workflows. They should not be treated as directly comparable benchmark results because the dataset generation and training implementations differ.

## 9. Cross-Validation and Hyperparameter Search

A single training/validation split can produce an unstable performance estimate.

Cross-validation evaluates a model over multiple training and validation folds.

The Python implementation uses:

- `StratifiedKFold`
- `GridSearchCV`
- F1 scoring
- Logistic regression regularization parameter `C`
- Automatic refitting of the selected estimator

The search evaluates:

| Parameter | Candidate values |
|---|---|
| `classifier__C` | 0.1, 1.0, 10.0 |
| `classifier__solver` | lbfgs |

The `classifier__` prefix identifies a parameter belonging to the classifier stage of the complete pipeline.

The best estimator is refitted on the complete training partition before final testing.

The validation and test partitions must remain separate. Repeatedly selecting configurations based on test performance compromises the independence of the final evaluation.

## 10. Pipeline State Management

A production pipeline should have explicit execution states.

The JavaScript and Java implementations represent these states directly.

| State | Meaning |
|---|---|
| Created | Pipeline configuration exists |
| Validated | Inputs and configuration passed validation |
| Fitting | Transformers and estimator are being trained |
| Fitted | Training completed successfully |
| Predicting | Inference is being performed |
| Failed | An unrecoverable execution error occurred |

A valid workflow is:

`Created → Validated → Fitting → Fitted → Predicting → Fitted`

A failed pipeline cannot be treated as a successfully trained model.

State transitions help identify invalid operations, prevent prediction before training, and improve operational diagnostics.

## 11. Implementation by Language

### Python

The Python implementation emphasizes the scikit-learn ecosystem.

| Component | Implementation |
|---|---|
| Dataset | pandas DataFrame |
| Numerical processing | SimpleImputer and StandardScaler |
| Categorical processing | OneHotEncoder |
| Transformation composition | ColumnTransformer |
| Model workflow | Pipeline |
| Parameter optimization | GridSearchCV |
| Evaluation | scikit-learn metrics |
| Persistence | joblib |
| Validation | Assertions and explicit exceptions |

The complete fitted preprocessing and classification pipeline is stored as one artifact.

This ensures that inference uses the same fitted transformations as training.

Joblib uses Python serialization mechanisms. Model artifacts must only be loaded from trusted sources because malicious serialized files can execute code during deserialization.

### JavaScript

The JavaScript implementation emphasizes asynchronous persistence and event-driven execution.

Its main components are:

- `SchemaValidator`
- `MedianImputer`
- `StandardScaler`
- `OneHotEncoder`
- `LogisticRegressionModel`
- `Pipeline`

The pipeline extends `EventEmitter` and emits events when transformers finish, execution states change, or errors occur.

Promises and asynchronous file operations support non-blocking artifact persistence.

The implementation uses native JavaScript collections, including `Map` and `Set`, for learned preprocessing metadata.

### C++17

The C++ implementation emphasizes explicit domain modeling and computational control.

Its main components are:

- `Customer`
- `FeatureRow`
- `Prediction`
- `Metrics`
- `Transformer`
- `MedianImputer`
- `StandardScaler`
- `CategoricalEncoder`
- `LogisticRegression`
- `Pipeline`

The abstract transformer interface separates transformation behavior from pipeline orchestration.

STL containers and algorithms support feature storage, median calculation, category management, and dataset splitting.

The implementation exports model parameters to a file. The exported model parameters alone are not a complete deployable pipeline because preprocessing metadata must also be persisted for independent inference.

### Java 17

The Java implementation emphasizes enterprise-oriented type safety and service separation.

Its domain types include:

- `Customer`
- `FeatureSchema`
- `FeatureVector`
- `Prediction`
- `ModelMetrics`
- `TrainingSplit`
- `PipelineStatus`

The generic `Transformer<I, O>` interface establishes typed transformation boundaries.

Immutable records and collections reduce unintended modifications to training and evaluation data.

The pipeline manages state transitions and delegates transformation and evaluation responsibilities to dedicated components.

The exported model file contains estimator information. A complete production artifact would also require the fitted imputation values, scaling statistics, category vocabulary, and feature schema.

### PostgreSQL

The PostgreSQL implementation provides relational storage for machine learning workflow management.

It separates pipeline definitions, execution records, transformer configurations, datasets, and artifacts.

The schema includes:

- Organization and user registries.
- Dataset versions and schema metadata.
- Pipeline definitions and versions.
- Transformer definitions.
- Ordered pipeline stages.
- Model configurations.
- Dataset partitions.
- Pipeline execution records.
- Individual stage execution records.
- Evaluation metrics.
- Model artifact metadata.
- Inference request records.

Database constraints enforce valid record structures and permitted values.

Triggers enforce important execution-state rules, while functions and procedures handle pipeline eligibility and execution initialization.

The database stores artifact references and checksums. Actual model binaries and datasets remain in external storage.

## 12. PostgreSQL Workflow Relationships

The relational design follows:

`Organization → Users`

`Organization → Datasets → Dataset Partitions`

`Organization → Pipelines → Pipeline Versions`

`Pipeline Versions → Pipeline Stages → Transformer Definitions`

`Pipeline Versions → Model Configurations → Model Definitions`

`Pipeline Versions → Pipeline Runs → Run Stages`

`Pipeline Runs → Evaluation Metrics`

`Pipeline Runs → Model Artifacts`

`Model Artifacts → Inference Requests`

A pipeline version captures a specific dataset reference, preprocessing configuration, model configuration, and partition definition.

A pipeline run records one execution of that version.

This distinction allows multiple executions to use the same version without overwriting earlier execution history.

## 13. Validation and Failure Handling

Machine learning pipelines encounter failures at multiple levels.

| Failure | Appropriate handling |
|---|---|
| Missing required feature | Reject input |
| Entirely missing numerical column | Reject or apply an explicit alternative policy |
| Invalid target labels | Reject training data |
| Unknown categorical value | Apply the defined unknown-category policy |
| Zero-variance numerical feature | Avoid division by zero |
| Empty training partition | Reject execution |
| Missing target class | Reject binary model training |
| Invalid pipeline state | Reject operation |
| Failed transformer | Mark pipeline execution as failed |
| Artifact write failure | Report the persistence error |
| Missing fitted model | Reject prediction |

Input validation should happen before expensive training operations.

Execution errors should include the responsible stage and enough information for diagnosis without exposing sensitive input records.

## 14. Performance Considerations

Pipeline performance depends on dataset size, feature dimensionality, transformer complexity, and estimator choice.

| Operation | Typical complexity |
|---|---|
| Median calculation | \(O(n\log n)\) using sorting |
| Mean and variance calculation | \(O(n)\) |
| One-hot encoding | \(O(nk)\), where \(k\) is the number of known categories |
| Logistic regression training | \(O(I \times n \times d)\) |
| Logistic regression inference | \(O(n \times d)\) |

Where:

- \(n\) represents the number of observations.
- \(d\) represents the transformed feature count.
- \(I\) represents the number of optimization iterations.

High-cardinality categorical features can significantly increase feature dimensionality.

Sparse representations reduce memory consumption when most encoded values are zero.

The Python implementation uses sparse one-hot encoding through scikit-learn.

The educational JavaScript, C++17, and Java implementations use in-memory feature representations and are intended for moderate-sized datasets.

Production workloads may require batching, parallel processing, streaming transformations, and memory-aware feature storage.

## 15. Security and Reproducibility

Production pipeline management requires controls beyond model accuracy.

Important controls include:

- Versioned dataset identifiers.
- Dataset checksums.
- Explicit feature schemas.
- Reproducible random seeds.
- Immutable pipeline versions.
- Access-controlled model artifacts.
- Verified artifact checksums.
- Secure storage of credentials.
- Audit records for training and inference.
- Controlled model deployment.
- Monitoring for feature and prediction drift.

A checksum establishes whether an artifact has changed. It does not independently establish who created it or whether it is trustworthy.

Artifact provenance and access permissions must therefore be managed separately.

Training, validation, and testing records must remain appropriately separated throughout the workflow.

## 16. Installation and Execution

### Python

Requirements:

- Python 3.10 or newer.
- NumPy.
- pandas.
- scikit-learn.
- joblib.

Install dependencies using:

`python -m pip install numpy pandas scikit-learn joblib`

Execute:

`python ml_pipeline.py`

The program generates a dataset, fits the complete pipeline, performs cross-validation, evaluates the selected model, saves artifacts, reloads the model, and performs inference.

### JavaScript

Requirements:

- Node.js 18 or newer.

Execute:

`node ml_pipeline.js`

The program requires no third-party packages.

The generated pipeline artifact is stored in `js_ml_artifacts`.

### C++17

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic ml_pipeline.cpp -o ml_pipeline`

Execute:

`./ml_pipeline`

The program exports estimator parameters to `cpp_ml_pipeline.model`.

### Java 17

Compile:

`javac MLPipelineEnterprise.java`

Execute:

`java MLPipelineEnterprise`

The program exports estimator parameters to `java_ml_pipeline.model`.

### PostgreSQL

Requirements:

- PostgreSQL 15 or newer.
- Permission to create schemas, types, tables, functions, and triggers.

Execute:

`psql -U postgres -d postgres -f ml_pipeline.sql`

The script creates the `ml_platform` schema, inserts demonstration records, initializes a pipeline run, records stage execution and metrics, registers artifacts, and executes reporting queries.

The script uses a transaction so that schema and sample-data initialization are handled together.

## 17. Implementation Limitations

The implementations demonstrate the mechanisms of preprocessing and model workflows rather than providing identical production machine learning frameworks.

| Implementation | Main limitation |
|---|---|
| Python | Depends on external scientific Python packages |
| JavaScript | Uses a basic batch gradient-descent estimator |
| C++17 | Uses in-memory records and a simplified numerical model |
| Java 17 | Uses in-memory typed transformations and a simplified estimator |
| PostgreSQL | Stores workflow metadata rather than executing model training |

The JavaScript AUC calculation uses a simplified rank-sum approach and does not average tied ranks. The SQL database stores evaluation results but does not independently verify their statistical correctness.

The C++17 and Java model exports are educational parameter exports, not complete serialized inference pipelines.

The Python implementation provides the most complete fitted preprocessing and model persistence workflow.

A production deployment would additionally require versioned transformer metadata, artifact compatibility checks, access control, monitoring, automated retraining policies, and controlled model promotion.

## 18. Practical Applications

The same pipeline architecture applies to:

| Application | Preprocessing | Model workflow |
|---|---|---|
| Customer churn | Missing-value handling and categorical encoding | Classification |
| Credit risk | Numerical scaling and feature validation | Classification |
| Sales forecasting | Time-based feature construction | Regression |
| Fraud detection | Imputation and feature engineering | Classification |
| Customer segmentation | Scaling and dimensionality reduction | Clustering |
| Predictive maintenance | Sensor preprocessing and aggregation | Classification or regression |

The key architectural requirement remains consistent: transformations must be fitted using the appropriate training data and reused consistently during subsequent evaluation and inference.
