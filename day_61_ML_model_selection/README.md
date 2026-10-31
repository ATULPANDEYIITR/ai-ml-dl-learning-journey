# ML Model Selection: Cross-Validation, K-Fold, Stratified CV, and Repeated CV

## Scope

This project models practical model-selection workflows for binary classification. It focuses on how validation strategy affects the reliability of model comparison rather than treating cross-validation as a single interchangeable technique.

The five implementations use the same conceptual pipeline:

~~~text
Raw labelled dataset
        |
        v
Training / holdout separation
        |
        +------------------------------+
        |                              |
        v                              |
Cross-validation                      |
        |                              |
        +--> K-fold                    |
        +--> Stratified K-fold         |
        +--> Repeated Stratified CV   |
        |                              |
        v                              |
Compare candidate models               |
        |                              |
        v                              |
Select model using training data      |
        |                              |
        v                              |
Fit selected model on full training   |
        |                              |
        v                              |
Untouched holdout evaluation <--------+
~~~

The central distinction is:

| Mechanism | Primary purpose |
|---|---|
| K-fold cross-validation | Estimate generalization performance across multiple training and validation partitions |
| Stratified K-fold | Preserve class proportions across validation folds |
| Repeated CV | Measure how sensitive the estimate is to different fold assignments |
| Model selection | Choose among candidate algorithms or hyperparameters using validation results |
| Holdout evaluation | Produce a final estimate after model selection is complete |

The holdout dataset is kept separate from cross-validation so that it does not influence the selection decision.

## Core Concepts

### Cross-validation

Cross-validation repeatedly divides the available training data into training and validation subsets.

For `k` folds, each observation serves as validation data once and training data approximately `k - 1` times.

~~~text
Fold 1: VALIDATION | TRAIN | TRAIN | TRAIN | TRAIN
Fold 2: TRAIN     | VALIDATION | TRAIN | TRAIN | TRAIN
Fold 3: TRAIN     | TRAIN | VALIDATION | TRAIN | TRAIN
Fold 4: TRAIN     | TRAIN | TRAIN | VALIDATION | TRAIN
Fold 5: TRAIN     | TRAIN | TRAIN | TRAIN | VALIDATION
~~~

The resulting scores are aggregated to estimate expected model performance.

Cross-validation is especially useful when a separate validation dataset would leave too little data for training.

### K-fold cross-validation

Ordinary K-fold CV randomly distributes observations among folds.

Its basic workflow is:

1. Shuffle or partition the training observations.
2. Create `k` folds.
3. Train on `k - 1` folds.
4. Validate on the remaining fold.
5. Repeat until every fold has been used as validation.
6. Aggregate the scores.

The implementations use five folds because this provides a practical balance between computational cost and the number of independent validation partitions.

### Stratified K-fold cross-validation

Stratification is important for classification when class frequencies are unequal.

Instead of distributing all observations without considering labels, the implementation distributes observations separately by class.

For binary classification:

~~~text
Class 0 observations ---> distributed across all folds
Class 1 observations ---> distributed across all folds

                    Fold 1  Fold 2  Fold 3  Fold 4  Fold 5
Class 0               0       0       0       0       0
Class 1               0       0       0       0       0
~~~

The exact counts differ when the class total is not evenly divisible by the number of folds.

The important property is that a validation fold should contain the relevant classes when the dataset contains both classes.

This reduces a common failure mode in ordinary K-fold CV: a validation fold can accidentally contain too few positive examples or no positive examples at all.

### Repeated cross-validation

A single K-fold partition produces one estimate based on one particular assignment of observations to folds.

Repeated CV rebuilds the folds with different random seeds.

The project uses three repetitions:

~~~text
Repetition 1 -> 5-fold stratified CV -> mean F1
Repetition 2 -> 5-fold stratified CV -> mean F1
Repetition 3 -> 5-fold stratified CV -> mean F1
~~~

The mean describes the central estimate across repetitions. The standard deviation provides information about how sensitive the result is to the particular fold assignment.

Repeated CV does not automatically remove bias. Its main value here is stability assessment.

## Model Selection Workflow

The implementations use two candidate classifiers:

| Candidate | Main mechanism | Main characteristic |
|---|---|---|
| Logistic Regression | Linear decision boundary with L2 regularization | Compact parametric model |
| Decision Tree | Recursive feature-based partitions using Gini impurity | Non-linear rule-based model |

The models are intentionally different so that validation can demonstrate a genuine selection problem.

The workflow is:

~~~text
Complete dataset
      |
      v
80% training / 20% holdout
      |
      +--> Cross-validation on training only
      |
      +--> Compare Logistic Regression
      |
      +--> Compare Decision Tree
      |
      v
Select highest mean F1
      |
      v
Repeated Stratified CV
      |
      v
Fit selected model on complete training data
      |
      v
Evaluate once on untouched holdout
~~~

The holdout data does not participate in model selection.

## Evaluation Metrics

### Accuracy

Accuracy is the fraction of correctly classified observations.

`accuracy = correct predictions / total predictions`

It is useful when class distributions are reasonably balanced and both error types have similar consequences.

It can be misleading for strongly imbalanced classification.

### Precision

Precision measures the proportion of predicted positive observations that are actually positive.

`precision = true positives / (true positives + false positives)`

High precision means positive predictions are rarely false alarms.

### Recall

Recall measures the proportion of actual positive observations that the classifier identifies.

`recall = true positives / (true positives + false negatives)`

High recall means fewer positive cases are missed.

### F1 score

F1 combines precision and recall through their harmonic mean.

`F1 = 2 * precision * recall / (precision + recall)`

F1 is the primary selection metric in the implementations because it balances false-positive and false-negative behaviour.

### Log loss

Log loss evaluates the quality of predicted probabilities rather than only the final class.

A confidently wrong probability receives a large penalty.

This makes log loss useful when probability quality matters, even when classification accuracy is similar.

## Python Implementation

The Python program is a complete standard-library implementation.

### Dataset representation

The `Dataset` dataclass stores:

- Feature rows
- Binary labels

Validation checks detect:

- Empty datasets
- Mismatched feature and label counts
- Inconsistent feature widths
- Empty feature vectors
- Invalid binary labels

The program first creates a deterministic synthetic classification dataset and then separates a holdout set.

### Logistic regression

The logistic model implements batch gradient descent directly.

Its parameters include:

- Learning rate
- Number of epochs
- L2 regularization strength

The implementation clips the sigmoid input to prevent numerical overflow.

L2 regularization is applied to model weights during gradient calculation. The bias is not regularized.

### Decision tree

The Python tree implementation searches candidate feature thresholds and calculates Gini impurity.

A split is accepted only when it provides positive impurity reduction.

The tree stops growing when:

- Maximum depth is reached
- The node has too few observations
- All observations belong to one class
- No useful split exists

### Cross-validation

The `kfold_indices` function supports both ordinary and stratified partitioning.

For stratified validation, indices are grouped by class before being distributed across folds.

The script verifies that each validation fold contains both classes.

### Selection rule

Models are ranked by:

1. Mean F1, descending
2. F1 standard deviation, ascending

This means a higher central validation score is preferred, while lower variability acts as the tie-breaker.

## JavaScript Implementation

The JavaScript implementation uses Node.js standard-library functionality and emphasizes asynchronous evaluation and event-driven reporting.

### Event-driven architecture

`ModelSelectionPipeline` extends `EventEmitter`.

It emits:

- `phase`
- `model-evaluated`
- `complete`

This separates model-evaluation work from reporting.

The evaluation loop yields through `setImmediate`, allowing the process to remain cooperative when additional asynchronous operations are introduced.

### Data validation

The `Dataset` class validates:

- Feature and label array sizes
- Feature dimensions
- Numeric finiteness
- Binary labels

The validation is performed at construction time, preventing invalid dataset objects from entering the pipeline.

### Stratified fold generation

The JavaScript implementation groups observation indices by their class labels before distributing them among folds.

This preserves class representation as closely as possible across the validation partitions.

### Model evaluation

Each candidate classifier is evaluated using the same fold-generation strategy and metric calculations.

The pipeline records fold-level results and aggregates them to support model comparison.

### Repeated validation

Repeated stratified validation uses different deterministic seeds to rebuild the fold assignments.

The resulting mean and standard deviation provide a view of validation stability.

## C++17 Implementation

The C++17 implementation represents the workflow using explicit structures and standard-library containers.

### Data representation

The implementation uses structures to represent:

- Dataset
- Fold
- Metrics
- Cross-validation results
- Model evaluation results

The dataset validation checks feature dimensions, label counts, and binary class values.

### Model implementation

The classifiers use a shared interface so that the evaluation process can operate on different model types.

The logistic regression implementation uses gradient-based parameter updates.

The decision tree implementation evaluates candidate thresholds using Gini impurity.

### Cross-validation

Fold generation supports ordinary and stratified partitioning.

The implementation uses deterministic random-number generation to make experiments reproducible.

### Model selection

Candidate models are evaluated using mean F1.

The selection process prefers a higher mean F1 and uses lower standard deviation as a tie-breaker.

The selected classifier is retrained using the complete training partition before final holdout evaluation.

## Java 17 Implementation

The Java implementation models model selection as an enterprise-style domain service.

### Domain types

The implementation defines explicit types for:

- `ModelType`
- `ReviewState`
- `Dataset`
- `Fold`
- `Metrics`
- `CVResult`
- `ModelDecision`

Java records provide immutable representations for validation results and domain state.

### Classifier abstraction

The `Classifier` interface defines:

- `fit`
- `predict`
- `predictProbability`

This makes the evaluation service independent of a specific model implementation.

Both classifiers satisfy the same contract while maintaining different internal algorithms.

### Explicit model decision

The `ModelDecision` record captures:

- Selected model
- Cross-validation F1
- Repeated-CV F1
- Selection state

The explicit `ReviewState.SELECTED` value represents the outcome of the model-selection workflow rather than leaving the decision implicit in local variables.

### Enterprise-oriented selection

The model registry is represented by an `EnumMap`.

Model creation is centralized in `createModel`, which prevents model construction logic from being scattered through the evaluation workflow.

The selection comparator ranks:

1. Higher mean F1
2. Lower F1 standard deviation

This gives the evaluation service a deterministic decision policy.

## PostgreSQL Implementation

The PostgreSQL script represents model selection as relational data rather than executable model-training code.

### Relational model

The main relationships are:

~~~text
datasets
    |
    +--> cv_experiments
             |
             +--> cv_folds
             |
             +--> model_evaluations --> models
             |
             +--> model_selection --> models
             |
             +--> holdout_evaluations --> models
~~~

This separates the dataset, validation experiment, individual fold, model, selection result, and final holdout evaluation.

### Dataset table

`datasets` records:

- Dataset identity
- Feature count
- Sample count
- Positive-class count
- Derived negative-class count

The negative count is generated from the sample and positive counts so that the database maintains the relationship automatically.

### Model table

`models` stores:

- Model name
- Model family
- Hyperparameters
- Active state

Hyperparameters are represented as `JSONB`, which permits model-specific configuration while retaining relational identity and constraints.

### Cross-validation experiments

`cv_experiments` records:

- Dataset
- Validation strategy
- Number of folds
- Number of repetitions
- Random seed
- Workflow state

The experiment state distinguishes planned, running, completed, and failed execution states.

### Fold records

`cv_folds` stores the composition statistics of every validation fold.

The stored values include:

- Training sample count
- Validation sample count
- Training positive count
- Validation positive count

This makes class-balance validation auditable.

### Evaluation records

`model_evaluations` stores metrics at fold level.

This allows the database to calculate:

- Mean accuracy
- Mean precision
- Mean recall
- Mean F1
- F1 standard deviation
- Mean log loss

The `v_model_cv_summary` view performs this aggregation.

### Selection enforcement

The `model_selection` table records the selected candidate.

A trigger ensures that only one model can be marked selected for an experiment.

The selection ordering uses:

1. Highest mean F1
2. Lowest F1 standard deviation
3. Lowest mean log loss

This mirrors the selection policy implemented by the source programs.

### Stratification enforcement

The database trigger rejects a validation fold that contains only one class when the dataset contains both positive and negative observations.

This moves an important validation rule into the persistence layer.

### Transactional state

The final experiment state transition is performed transactionally.

The experiment cannot be considered complete unless a selected model exists.

This separates:

- Evaluation data
- Model-selection decision
- Workflow completion state

## Relationship Between the Validation Methods

These methods should not be treated as interchangeable.

| Method | What changes | Primary reason |
|---|---|---|
| K-fold | Random partitioning into `k` folds | Reuse limited training data for validation |
| Stratified K-fold | Fold construction respects class labels | Preserve classification balance |
| Repeated Stratified CV | Stratified folds are rebuilt repeatedly | Estimate sensitivity to partitioning |

A model can perform well under one partition and less well under another. Repeated validation makes that instability visible.

## Data Leakage Controls

The implementations maintain a strict separation between model selection and final evaluation.

The holdout set is created before cross-validation but is not used during CV.

The correct sequence is:

~~~text
Training data
    |
    +--> Fold training
    +--> Fold validation
    |
    +--> Model comparison
    |
    +--> Hyperparameter/model selection
    |
    v
Selected model
    |
    +--> Retrain on complete training data
    |
    v
Untouched holdout
~~~

Using the holdout repeatedly during model selection would cause information leakage and produce an optimistically biased estimate.

## Edge Cases

Important failure conditions represented by the implementations include:

- Empty datasets
- Mismatched feature and label counts
- Inconsistent feature dimensions
- Invalid binary labels
- Invalid fold counts
- Empty validation folds
- Missing classes in a stratified fold
- Unfitted models
- Invalid model hyperparameters
- Numerical overflow in logistic regression
- Zero denominators in precision or recall
- Extreme probabilities in log loss
- Trees with no useful split
- Too-small training nodes
- Invalid database states

Metrics explicitly guard against zero denominators instead of allowing undefined arithmetic to propagate.

## Performance Considerations

K-fold CV approximately multiplies model-training cost by the number of folds.

Repeated CV multiplies this cost again by the number of repetitions.

For example, five-fold CV repeated three times requires approximately fifteen model-training passes per candidate configuration.

Decision-tree threshold search can become expensive because many candidate feature thresholds are examined.

The implementations favor clarity and deterministic behaviour over optimized numerical computation.

For large datasets, practical optimizations would include:

- Vectorized numerical operations
- Efficient matrix storage
- Pre-sorted feature values
- Parallel fold evaluation
- Cached feature transformations
- Early stopping where appropriate
- Distributed validation for very large experiments

The PostgreSQL schema is optimized for reporting and auditability rather than for performing the numerical model training itself.

## Security and Reproducibility

Reproducibility is controlled through explicit random seeds.

Each implementation uses deterministic seed values for:

- Dataset generation
- Training/holdout partitioning
- Fold generation
- Repeated CV partitions

This allows a validation experiment to be recreated under the same implementation and parameters.

The PostgreSQL design stores the random seed with the experiment so that the validation configuration is auditable.

The database also uses constraints and triggers to prevent invalid workflow states from being silently persisted.

## Limitations

The classifiers are deliberately compact implementations rather than production numerical libraries.

The decision-tree probability output is a simple class-based estimate rather than calibrated probability estimation.

The synthetic dataset is useful for demonstrating validation mechanics but does not represent the complexity of a real operational dataset.

Cross-validation does not protect against leakage caused by incorrect feature engineering. Any transformation learned from data must be fitted inside the appropriate training partition.

For grouped observations, ordinary K-fold may be inappropriate because related records can appear in both training and validation sets. Group-aware validation is required when observations are not independent.

For time-dependent data, random K-fold can also violate temporal ordering. Time-series validation should preserve chronology.

## Practical Model-Selection Rules

A sound classification workflow should distinguish the following questions:

| Question | Appropriate mechanism |
|---|---|
| How well does a candidate model generalize? | Cross-validation |
| Are class proportions preserved? | Stratified CV |
| Is the validation estimate sensitive to fold assignment? | Repeated CV |
| Which candidate should be selected? | Validation-based model comparison |
| How does the selected model perform on unseen data? | Final holdout evaluation |

The key boundary is between selection and final evaluation. Cross-validation supports decisions about models and hyperparameters. The untouched holdout provides an independent final check after those decisions have already been made.
