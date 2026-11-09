# Production-Style Classical Machine Learning Pipeline

## Project Description

This project implements a production-oriented classical machine learning workflow for customer churn prediction.

It covers the engineering activities required to move a supervised learning model from raw data to controlled production deployment.

The project uses customer attributes such as tenure, monthly charges, support calls, contract type, payment method, and internet service to predict whether a customer will discontinue a service.

The five implementations approach the same engineering problem from different perspectives:

| Implementation | Primary responsibility |
|---|---|
| Python | Statistical model development, training, evaluation, prediction, and monitoring |
| JavaScript | Event-driven pipeline orchestration, retries, quality gates, and model registry |
| C++17 | Efficient model evaluation and release-governance engine |
| Java 17 | Enterprise workflow, approval policies, domain modeling, and deployment control |
| PostgreSQL | Persistent storage, relational integrity, evaluation history, approvals, and deployment governance |

The implementations are independent executable programs. The JavaScript, C++, Java, and SQL implementations provide complementary engineering models rather than requiring the Python program to execute them.

## 1. Classical Machine Learning Engineering

Classical machine learning uses statistical algorithms to learn patterns from structured observations.

The project uses supervised binary classification.

Each customer observation contains input features and a target:

- `0`: Customer does not churn.
- `1`: Customer churns.

The model estimates:

\[
P(Y=1\mid X)
\]

Where:

- \(Y\) represents the churn outcome.
- \(X\) represents the customer features.

The prediction is a probability. A classification threshold converts that probability into a binary decision.

For example:

\[
\hat{Y} =
\begin{cases}
1 & P(Y=1\mid X)\geq t\\
0 & P(Y=1\mid X)<t
\end{cases}
\]

Here, \(t\) represents the decision threshold.

The default threshold is 0.50, although the Python implementation selects a threshold using validation data.

### Engineering versus model development

A machine learning project contains more than an algorithm.

| Activity | Engineering responsibility |
|---|---|
| Data ingestion | Obtain data from an approved source |
| Data validation | Verify schema, completeness, and permitted values |
| Feature engineering | Prepare useful predictive variables |
| Preprocessing | Handle missing values and categorical variables |
| Training | Fit candidate statistical models |
| Validation | Compare models and select parameters |
| Testing | Estimate performance on unseen observations |
| Packaging | Store the trained model and metadata |
| Deployment | Make an approved model available for inference |
| Monitoring | Detect operational degradation and data changes |

The distinction matters because a statistically strong model can still fail in production due to incorrect input data, incompatible preprocessing, unavailable artifacts, excessive latency, or changing customer behavior.

## 2. End-to-End Pipeline Architecture

The main workflow is:

Data Source
    |
    v
Data Ingestion
    |
    v
Dataset Validation
    |
    v
Train / Validation / Test Separation
    |
    v
Preprocessing
    |
    v
Cross-Validation
    |
    v
Algorithm and Parameter Selection
    |
    v
Threshold Selection
    |
    v
Final Model Training
    |
    v
Independent Test Evaluation
    |
    v
Quality Assessment
    |
    v
Model Registration
    |
    v
Controlled Deployment
    |
    v
Prediction and Monitoring

The stages have separate responsibilities.

A failure in dataset validation should prevent training. A failure in model evaluation should prevent approval. A failed deployment should not silently replace an existing production model.

## 3. Dataset Management and Validation

### Data ingestion

The Python implementation accepts CSV input and can generate a reproducible demonstration dataset.

The dataset includes:

| Feature | Data type | Meaning |
|---|---|---|
| customer_id | Integer | Unique customer identifier |
| age | Integer | Customer age |
| tenure_months | Integer | Duration of service |
| monthly_charges | Numeric | Monthly service charges |
| support_calls | Integer | Number of support interactions |
| contract | Categorical | Contract duration |
| payment_method | Categorical | Payment category |
| internet_service | Categorical | Internet service type |
| churn | Binary | Prediction target |

The identifier is retained for traceability but excluded from predictive features.

This prevents an arbitrary identifier from being interpreted as a meaningful customer characteristic.

### Dataset validation

Validation checks include:

- Minimum observation count.
- Required identifier and target columns.
- Duplicate identifiers.
- Missing target values.
- Binary target values.
- Both target classes being present.
- Non-finite numerical values.
- High-cardinality categorical features.
- Missing prediction features.
- Unexpected prediction features.

The JavaScript, C++, Java, and SQL implementations provide corresponding contract and integrity checks.

### Dataset fingerprinting

The Python implementation calculates a SHA-256 fingerprint of the input dataset.

A fingerprint supports identification of the dataset used in a particular training run.

It is useful for reproducibility and auditing, although it does not establish the authenticity or authorization of a dataset.

### Dataset versioning

A production dataset should have an immutable version.

A version identifies a particular collection of observations and its associated schema.

The PostgreSQL implementation represents this through:

- `dataset`
- `dataset_version`
- `feature_definition`

The dataset version stores its fingerprint, row count, column count, target column, validation report, and class distribution.

## 4. Data Preprocessing

Preprocessing converts raw input into a representation accepted by the learning algorithm.

The Python implementation uses a `ColumnTransformer` containing separate numerical and categorical pipelines.

### Numerical preprocessing

Numerical variables use median imputation followed by standard scaling.

Median imputation:

\[
x_{\text{missing}} =
\operatorname{median}(X)
\]

Standardization:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) is the original value.
- \(\mu\) is the training mean.
- \(\sigma\) is the training standard deviation.

Median imputation is less sensitive to extreme observations than mean imputation.

Standardization is particularly useful for algorithms such as logistic regression.

### Categorical preprocessing

Categorical variables use:

1. Most-frequent-value imputation.
2. One-hot encoding.

One-hot encoding represents a category through binary indicator variables.

For example:

| Contract | monthly | annual | two_year |
|---|---:|---:|---:|
| monthly | 1 | 0 | 0 |
| annual | 0 | 1 | 0 |
| two_year | 0 | 0 | 1 |

The encoder uses `handle_unknown="ignore"` so previously unseen categories do not immediately break prediction.

### Leakage prevention

Preprocessing must be fitted using training observations rather than the complete dataset.

The Python implementation places preprocessing inside the scikit-learn pipeline.

Consequently, each cross-validation training fold learns its own preprocessing parameters.

This avoids using information from validation folds during fitting.

## 5. Dataset Partitioning and Validation

The Python implementation uses three partitions.

| Partition | Purpose |
|---|---|
| Training partition | Model fitting and cross-validation |
| Validation partition | Classification threshold selection |
| Test partition | Independent performance evaluation |

The test partition is separated before model selection.

Stratified splitting preserves the approximate distribution of the target classes across partitions.

The configured random seed supports reproducible splitting.

### Cross-validation

Stratified K-fold cross-validation divides the fitting data into K folds.

Each iteration uses:

- K-1 folds for training.
- One fold for validation.

The process repeats until every fold has served as the validation fold.

The Python implementation uses three folds and ROC-AUC as its model-selection metric.

Cross-validation provides a more stable model comparison than relying on one validation split.

### Hyperparameter selection

The implementation evaluates three classical algorithms:

| Algorithm | Main characteristics |
|---|---|
| Logistic Regression | Linear decision boundary and interpretable coefficients |
| Random Forest | Ensemble of decision trees |
| Histogram Gradient Boosting | Sequential boosting with histogram-based feature processing |

The parameter search evaluates different regularization strengths, tree depths, estimator counts, and learning rates.

The best candidate is selected according to mean cross-validation ROC-AUC.

The selected pipeline is subsequently refitted on the complete development partition.

The independent test partition remains unused during selection.

## 6. Classification Evaluation

A classification model should not be evaluated using accuracy alone.

The project calculates several complementary measures.

### Confusion matrix

| | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | True Positive | False Negative |
| Actual negative | False Positive | True Negative |

### Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

Accuracy measures the proportion of correct predictions.

It can be misleading when one target class is much more common than the other.

### Precision

\[
Precision=\frac{TP}{TP+FP}
\]

Precision measures the proportion of predicted positive cases that are actually positive.

### Recall

\[
Recall=\frac{TP}{TP+FN}
\]

Recall measures the proportion of actual positive cases identified by the model.

### F1-score

\[
F1=2\frac{Precision\times Recall}{Precision+Recall}
\]

F1 balances precision and recall through their harmonic mean.

### ROC-AUC

ROC-AUC measures the model's ability to rank positive observations above negative observations across possible classification thresholds.

AUC is independent of the particular classification threshold.

### Average precision

Average precision summarizes the precision-recall relationship and is useful when the positive class is relatively uncommon.

The Python implementation reports these metrics on held-out test observations.

The C++ and Java programs implement classification metrics and ROC-AUC calculations directly.

## 7. Classification Threshold Selection

A probability estimate is not itself a final classification.

The threshold determines how probability estimates become operational decisions.

A lower threshold generally increases the number of positive predictions. This can increase recall while also increasing false positives.

A higher threshold generally produces fewer positive predictions and may increase precision while reducing recall.

The Python implementation tests thresholds between 0.10 and 0.90 and selects the threshold that maximizes validation F1.

The test partition is not used to select the threshold.

This separation prevents test-set information from influencing the classification decision rule.

The selected threshold is stored in model metadata and reused during inference.

A production application may require a threshold based on financial cost, intervention capacity, or an explicitly defined business objective rather than F1 alone.

## 8. Python Implementation: Statistical Development and Inference

File: `ml_pipeline.py`

The Python program provides the complete statistical modeling workflow.

### Main components

| Component | Responsibility |
|---|---|
| `Config` | Centralized pipeline settings |
| `generate_dataset()` | Reproducible demonstration data |
| `load_data()` | CSV ingestion |
| `validate_dataset()` | Dataset contract enforcement |
| `dataset_fingerprint()` | Dataset identification |
| `build_preprocessor()` | Numerical and categorical transformations |
| `build_search()` | Algorithm and parameter selection |
| `select_threshold()` | Threshold optimization |
| `evaluate_model()` | Classification evaluation |
| `save_artifacts()` | Model and metadata persistence |
| `load_artifacts()` | Artifact retrieval |
| `train()` | End-to-end training |
| `predict()` | Batch inference |
| `monitor()` | Input schema monitoring |

### Model persistence

The model is stored as a Joblib artifact.

The accompanying JSON metadata records:

- Model name and version.
- Creation timestamp.
- Target variable.
- Feature names.
- Numerical features.
- Categorical features.
- Dataset fingerprint.
- Training and test row counts.
- Selected hyperparameters.
- Cross-validation score.
- Classification threshold.
- Test metrics.
- scikit-learn version.

The model and metadata are written through temporary files before replacement.

Joblib uses Python object serialization and must only load trusted artifacts. Production systems should use authenticated artifact storage, access controls, integrity verification, and signed model manifests.

### Execution

Training with demonstration data:

`python ml_pipeline.py train`

Training from a CSV:

`python ml_pipeline.py train --data customers.csv`

Batch prediction:

`python ml_pipeline.py predict --data new_customers.csv`

Schema monitoring:

`python ml_pipeline.py monitor --data customers.csv`

Basic assertions:

`python ml_pipeline.py test`

The prediction CSV includes customer identifiers, estimated churn probabilities, predicted labels, model versions, and prediction timestamps.

### Limitations

The demonstration dataset is synthetic. Its performance does not establish real-world predictive effectiveness.

The monitoring implementation checks input schema compatibility. It does not calculate comprehensive statistical drift or monitor actual prediction outcomes.

The artifact replacement operations are individually atomic but do not constitute a distributed transaction covering both files.

## 9. JavaScript Implementation: Event-Driven Orchestration

File: `ml_orchestrator.js`

The JavaScript implementation focuses on coordinating the pipeline rather than reproducing the Python statistical training implementation.

It demonstrates an orchestration layer that can communicate with an independent training service.

### Event-driven architecture

`PipelineRun` extends Node.js `EventEmitter`.

A pipeline state transition emits a `stateChanged` event.

The event records:

- Run identifier.
- Previous state.
- New state.
- Timestamp.
- Additional transition details.

This supports centralized audit logging and operational event handling.

### State management

The pipeline uses explicit states:

| State | Meaning |
|---|---|
| CREATED | Run initialized |
| VALIDATING | Dataset contract being checked |
| VALIDATED | Dataset accepted |
| TRAINING | Training operation executing |
| EVALUATING | Model metrics being assessed |
| APPROVED | Release conditions satisfied |
| DEPLOYING | Deployment in progress |
| DEPLOYED | Production promotion completed |
| REJECTED | Model failed quality requirements |
| FAILED | Execution encountered an error |

Only permitted state transitions are accepted.

This prevents invalid workflow sequences, such as deploying a model that has not passed evaluation.

### Dataset contracts

`DatasetContract` checks required fields, target values, row count, and duplicate identifiers.

It also generates a SHA-256 fingerprint.

The contract separates input validation from model execution.

### Asynchronous operations

The orchestrator uses Promises and asynchronous functions.

The retry mechanism implements exponential backoff:

\[
Delay_n=100\times 2^n
\]

The timeout wrapper limits the duration of individual asynchronous operations.

Retries are appropriate for transient failures, but not every failed operation should be repeated.

Production integrations must distinguish transient errors from permanent validation and authorization failures.

### Release policy

`ReleasePolicy` evaluates:

- ROC-AUC.
- Recall.
- Prediction latency.
- Feature drift.

The model is eligible only when every required condition passes.

### Model registry

`ModelRegistry` maintains registered model versions using a `Map`.

It supports:

- Version registration.
- Duplicate-version rejection.
- Production model promotion.
- Production version retrieval.
- Promotion event emission.

### Drift monitoring

The JavaScript implementation calculates Population Stability Index (PSI).

PSI compares the distribution of a numerical feature between a reference population and a current population.

\[
PSI=\sum_{i=1}^{N}
(A_i-E_i)\ln\left(\frac{A_i}{E_i}\right)
\]

Where:

- \(A_i\) is the observed proportion in bin \(i\).
- \(E_i\) is the expected proportion in bin \(i\).
- \(N\) is the number of bins.

PSI is a distribution-monitoring measure, not proof of model degradation.

The demonstration uses reference-derived bin boundaries.

### Important integration boundary

The trainer and evaluator in this implementation are adapters.

They demonstrate the orchestration contract and use deterministic sample metrics.

They do not execute a statistical classifier.

A production implementation would connect these adapters to an authenticated training and evaluation service.

The demonstration also uses in-memory execution state. Durable job queues, persistent idempotency records, deployment rollback, and distributed transaction handling are required for reliable multi-instance operation.

## 10. C++17 Implementation: Model Governance Engine

File: `ml_governance.cpp`

The C++ implementation emphasizes efficient evaluation and explicit release decisions.

It models the governance boundary between statistical evaluation and production deployment.

### Domain structures

| Structure | Purpose |
|---|---|
| `Observation` | Actual labels, features, and probabilities |
| `ModelMetrics` | Evaluation and operational metrics |
| `QualityCheck` | Individual release condition |
| `ReleaseDecision` | Overall eligibility result |
| `Model` | Registered model version and state |

### Dataset validation

`DatasetValidator` checks:

- Minimum observation count.
- Unique customer identifiers.
- Finite numerical values.
- Non-negative feature values.
- Binary target labels.
- Probability bounds.
- Presence of both classes.

An invalid dataset raises an exception before evaluation.

### Direct metric computation

`ModelEvaluator` calculates the confusion matrix, accuracy, precision, recall, F1, and ROC-AUC.

ROC-AUC uses sorted prediction scores and average ranks for tied observations.

Sorting requires:

\[
O(n\log n)
\]

The subsequent rank calculation requires:

\[
O(n)
\]

The total computational complexity is:

\[
O(n\log n)
\]

### Operational evaluation

`OperationalEvaluator` compares recall across two demonstration groups.

The difference between group recall values is used as a simple fairness indicator.

This is not a comprehensive fairness assessment.

A production fairness evaluation should use justified group definitions, adequate sample sizes, confidence intervals, relevant protected-group analysis, and domain-specific fairness criteria.

### Release policy

The release policy evaluates five independent conditions:

| Condition | Purpose |
|---|---|
| ROC-AUC | Ranking quality |
| Recall | Positive-class detection |
| Feature drift | Input distribution stability |
| Latency | Operational performance |
| Fairness gap | Difference in group recall |

A failed condition prevents approval.

### Model registry

The registry stores model versions in an ordered map.

It supports:

- Model registration.
- State transitions.
- Production promotion.
- Previous-version retirement.
- Rollback.

The state machine prevents invalid deployment transitions.

### Limitations

The program evaluates supplied prediction probabilities. It does not train a classifier.

Its demonstration fairness groups use identifier ranges to keep the program self-contained. They must not be interpreted as meaningful demographic groups.

The demonstration rollback method promotes an existing version but does not retain a complete production deployment history. A real registry should store immutable deployment records and support atomic rollback.

## 11. Java 17 Implementation: Enterprise Workflow

File: `MLEnterprisePipeline.java`

The Java implementation models an enterprise ML governance service with explicit domain types and controlled service interactions.

### Domain modeling

Java records represent immutable data structures:

- `Dataset`
- `Observation`
- `ModelMetrics`
- `QualityResult`
- `Review`
- `ModelVersion`
- `AuditEvent`

Records reduce accidental modification of important workflow data.

Collections and immutable copies help protect domain boundaries.

### Dataset validation service

`DatasetValidator` verifies the dataset before training.

The implementation checks duplicate identifiers and the presence of both target classes.

Validation failures use a custom `ContractException`.

### Evaluation service

`EvaluationService` calculates classification metrics and ROC-AUC.

The evaluator handles tied probability scores using average ranks.

### Approval policy

Approval is a governance decision separate from model quality.

`ApprovalPolicy` defines:

- Required approval count.
- Eligible reviewers.
- Excluded authors.
- Review decisions.
- Reviewer eligibility.

The policy uses the latest review decision for each reviewer when calculating approval counts.

A reviewer who previously approved but subsequently requests changes is not counted as an approving reviewer.

The implementation rejects authors reviewing their own model.

### Release policy

`ReleasePolicy` evaluates technical and operational requirements.

The conditions include:

- Minimum ROC-AUC.
- Minimum recall.
- Maximum drift.
- Maximum latency.
- Maximum fairness gap.

Approval does not override a failed technical quality gate.

### Run state machine

`RunStateMachine` enforces allowed transitions.

This separates the current workflow state from the data used to evaluate the model.

An invalid transition raises `GovernanceException`.

### Model registry

`ModelRegistry` manages registered model versions and production promotion.

Before deployment, it checks:

1. Model registration.
2. Reviewer eligibility.
3. Approval requirements.
4. Model quality requirements.

Rollback selects an existing registered version.

The demonstration registry is in memory and does not replace a persistent artifact registry or an infrastructure deployment controller.

### Audit service

`AuditService` captures:

- Actor.
- Action.
- Details.
- Timestamp.
- Event identifier.

Audit events provide a record of workflow activity.

For production, audit storage should be persistent, access-controlled, tamper-resistant, and protected against unauthorized deletion.

## 12. PostgreSQL Implementation: Persistent ML Operations

File: `ml_pipeline.sql`

The SQL implementation provides a relational foundation for storing machine learning project information and enforcing important integrity conditions.

It uses the `ml_platform` schema.

### Relational architecture

Project
    |
    +-- Project Members
    |
    +-- Datasets
    |      |
    |      +-- Dataset Versions
    |      |
    |      +-- Feature Definitions
    |
    +-- Training Runs
    |      |
    |      +-- Model Artifacts
    |             |
    |             +-- Evaluations
    |             |
    |             +-- Reviews
    |             |
    |             +-- Quality Gates
    |             |
    |             +-- Deployments
    |                    |
    |                    +-- Predictions
    |                    |
    |                    +-- Drift Records
    |
    +-- Approval Policies
    |
    +-- Audit Events

### Main tables

| Table | Responsibility |
|---|---|
| `ml_project` | Project ownership |
| `project_member` | Project roles |
| `dataset` | Dataset registration |
| `dataset_version` | Immutable dataset versions |
| `feature_definition` | Feature contracts |
| `training_run` | Training execution records |
| `model_artifact` | Model registry |
| `model_evaluation` | Classification performance |
| `approval_policy` | Deployment requirements |
| `model_review` | Independent review decisions |
| `quality_gate` | Technical release checks |
| `model_deployment` | Deployment history |
| `prediction_log` | Prediction-level operational records |
| `drift_monitoring` | Feature distribution monitoring |
| `audit_event` | Governance event history |

### Database integrity

The schema uses:

- Primary keys.
- Foreign keys.
- Unique constraints.
- Check constraints.
- Enumerated types.
- JSONB fields.
- Partial unique indexes.
- Operational indexes.

For example, dataset versions require a non-negative row count and a positive column count.

Prediction probabilities must remain within the interval from zero to one.

A unique request identifier prevents duplicate prediction request records.

The partial unique index on active deployments permits only one active deployment per environment.

### Database triggers

The audit trigger records changes to model artifacts and deployments.

The reviewer validation trigger checks:

- Reviewer existence.
- Reviewer activity.
- Project membership.
- Eligible reviewer role.

The trigger prevents invalid review records from being inserted through ordinary application operations.

### Eligibility determination

`check_model_eligibility()` combines approval and technical quality requirements.

It determines whether:

- The required number of eligible reviewers approved the model.
- ROC-AUC meets the configured minimum.
- Recall meets the configured minimum.
- Drift remains within the permitted threshold.
- Latency remains within the permitted threshold.
- Fairness gap remains within the permitted threshold.

It returns an eligibility decision and explanatory failure reasons.

The function demonstrates policy evaluation inside the database.

The current demonstration uses the most recent evaluation record and does not implement independent confidence-interval calculations.

### Transactional deployment

`promote_model()` performs controlled promotion.

The procedure:

1. Acquires a transaction-level advisory lock.
2. Retrieves and locks the candidate model.
3. Validates its state.
4. Verifies policy ownership.
5. Evaluates deployment eligibility.
6. Retires the existing active deployment.
7. Inserts the new deployment.
8. Updates model status.
9. Records an audit event.

The transaction ensures that a failed promotion does not leave the database with a partially updated deployment state.

The procedure serializes promotions within an environment using an advisory lock.

### Reporting

The script includes:

- Dataset retrieval.
- Training history.
- Model performance.
- Review aggregation.
- Quality gate reporting.
- Eligibility determination.
- Model ranking using a window function.
- Training success aggregation.
- Deployment reporting.
- Prediction latency reporting.
- Drift alert reporting.
- Audit history.

The reporting views separate frequently used reporting logic from application queries.

### SQL limitations

The database stores artifact locations and checksums. It does not store the actual model binary.

The database cannot establish that an external artifact is safe merely because its checksum is recorded.

Artifact verification must occur in the application or artifact-management service.

The sample deployment endpoint uses an illustrative internal address.

Production systems should use restricted database roles, row-level security where appropriate, encrypted connections, controlled procedure execution, and separate application and administrative privileges.

## 13. Model Approval and Production Release

Model quality and release authorization are separate concerns.

A model can achieve strong predictive performance but still fail production governance.

Examples include:

- Missing required independent reviews.
- Excessive feature drift.
- High inference latency.
- Incomplete feature documentation.
- Invalid artifact checksums.
- Incompatible preprocessing schemas.
- Unauthorized deployment requests.

The release process therefore uses independent gates.

| Gate | Evaluates | Responsible component |
|---|---|---|
| Data gate | Dataset integrity and schema | Dataset validator |
| Training gate | Successful model fitting | Training service |
| Evaluation gate | Predictive performance | Evaluation service |
| Operational gate | Latency and drift | Monitoring service |
| Fairness gate | Group performance differences | Evaluation service |
| Approval gate | Independent review decisions | Governance service |
| Deployment gate | Final release authorization | Registry and deployment service |

An approval represents a decision by an authorized reviewer.

It does not automatically establish technical quality or deployment eligibility.

Similarly, a passing automated quality gate does not replace independent review.

## 14. Monitoring and Model Degradation

A production model should be evaluated after deployment.

Two important monitoring categories are data drift and model performance degradation.

### Data drift

Data drift occurs when the distribution of incoming features changes relative to the reference population.

Examples include:

- Increasing monthly charges.
- Changing customer contract distribution.
- More missing feature values.
- Changes in customer tenure.
- Changes in payment methods.

PSI provides one numerical measure of distribution change.

### Concept drift

Concept drift occurs when the relationship between input features and the target changes.

For example, the same customer characteristics may become less predictive of churn after a pricing-policy change.

Feature drift does not necessarily imply concept drift.

A stable feature distribution also does not guarantee stable model performance.

### Delayed outcome monitoring

Actual churn labels may become available only after a period of time.

The `prediction_log` table includes an optional actual label.

This supports retrospective evaluation when ground-truth outcomes become available.

Monitoring should distinguish:

- Prediction volume.
- Positive prediction rate.
- Feature drift.
- Prediction latency.
- Missing input values.
- Actual outcome availability.
- Precision and recall over time.

### Model rollback

A rollback returns production traffic to a previously approved model version.

The registry examples demonstrate version selection and promotion.

A production rollback must also restore the corresponding preprocessing configuration, feature schema, inference dependencies, and compatible serving configuration.

## 15. Security and Reliability Considerations

A production ML pipeline processes potentially sensitive information and controls a business-critical application.

Important controls include:

| Area | Required control |
|---|---|
| Data access | Least-privilege permissions |
| Dataset storage | Encryption and retention controls |
| Model artifacts | Trusted storage and checksum verification |
| Model deployment | Authorized release operations |
| API access | Authentication and authorization |
| Input validation | Schema and range enforcement |
| Logging | Avoid unnecessary sensitive information |
| Audit | Restricted and tamper-resistant event history |
| Dependencies | Version pinning and vulnerability management |
| Reproducibility | Dataset, code, parameter, and environment tracking |
| Monitoring | Drift, latency, errors, and delayed outcomes |
| Recovery | Previous approved model and deployment configuration |

The examples demonstrate selected controls, not a complete certified security architecture.

## 16. Performance and Scalability

Different components have different computational requirements.

| Operation | Typical complexity |
|---|---|
| Dataset validation | \(O(n \times d)\) |
| Numerical imputation | \(O(n)\) per feature |
| One-hot encoding | Depends on observations and category count |
| Logistic regression | Depends on observations, features, and optimization iterations |
| Random forest | Depends on tree count, depth, and observations |
| Histogram gradient boosting | Depends on observations, features, bins, and iterations |
| Sorting prediction scores | \(O(n\log n)\) |
| Registry lookup using a hash map | Average \(O(1)\) |
| Ordered registry lookup | \(O(\log m)\) |
| Relational lookup | Depends on indexes, selectivity, and query plan |

Where:

- \(n\) is the number of observations.
- \(d\) is the number of features.
- \(m\) is the number of registered model versions.

Scalability measures include:

- Batch prediction.
- Efficient numerical transformations.
- Sparse categorical representations.
- Indexed database retrieval.
- Asynchronous orchestration.
- Controlled retries.
- Persistent execution records.
- Separation of training and inference services.

Large datasets may require distributed storage, incremental processing, partitioned tables, and workload-specific model-serving infrastructure.

## 17. Execution and File Organization

The six deliverables can be stored as:

| File | Purpose |
|---|---|
| `ml_pipeline.py` | Statistical ML pipeline |
| `ml_orchestrator.js` | Event-driven orchestration |
| `ml_governance.cpp` | Model release engine |
| `MLEnterprisePipeline.java` | Enterprise governance workflow |
| `ml_pipeline.sql` | Relational ML platform |
| `README.md` | Project documentation |

### Python dependencies

Install the required packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`

The Python implementation uses `sparse_output=True`, which requires a compatible scikit-learn version.

### JavaScript

Requires Node.js 18 or later.

The implementation uses built-in Node.js modules and does not require npm packages.

### C++17

Compile with a C++17-compatible compiler.

The implementation uses the standard library and requires no external dependencies.

### Java 17

Compile and run using the Java 17 compiler and runtime.

The implementation uses standard Java APIs and requires no third-party dependencies.

### PostgreSQL

The SQL implementation uses PostgreSQL-specific capabilities including:

- JSONB.
- PL/pgSQL.
- Advisory transaction locks.
- Partial indexes.
- Identity columns.
- UUID values.
- Window functions.

The script uses `gen_random_uuid()` for run identifiers and prediction request identifiers.

## 18. Technical Learning Outcomes

The project demonstrates the relationship between statistical learning, software engineering, data management, and production governance.

| Area | Implemented capability |
|---|---|
| Supervised learning | Binary classification |
| Data engineering | CSV ingestion and dataset contracts |
| Feature engineering | Numerical and categorical preprocessing |
| Model development | Logistic regression, random forest, and gradient boosting |
| Model selection | Stratified cross-validation |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, and average precision |
| Decision engineering | Validation-based threshold selection |
| Reproducibility | Dataset fingerprints and model metadata |
| Persistence | Model artifacts and relational records |
| Orchestration | Asynchronous execution and state transitions |
| Quality assurance | Automated release policies |
| Governance | Independent approvals and audit records |
| Deployment | Controlled model promotion |
| Monitoring | Feature drift and inference latency |
| Reliability | Error handling, retries, and transactional operations |

The central engineering distinction is that training a model, evaluating a model, authorizing a model, and deploying a model are separate operations.

A production-style classical ML pipeline coordinates all four while preserving the evidence and state required to reproduce, inspect, and govern each model release.
