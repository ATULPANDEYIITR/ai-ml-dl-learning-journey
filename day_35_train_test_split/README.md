# Train-Test Split: Training, Validation, Testing, Leakage, and Reproducibility

## Scope

A train-test split is an experimental boundary used to estimate how well a machine-learning system will generalize beyond the observations used to develop it. A practical experiment usually distinguishes three roles:

- **Training data** is used to learn model parameters and parameters of fitted preprocessing operations.
- **Validation data** is used to compare model configurations, feature choices, thresholds, hyperparameters, or other development decisions.
- **Test data** is held back until those decisions are complete and is used to estimate final generalization.

These partitions are related, but they are not interchangeable. The central objective is not merely to divide rows into three files. It is to preserve the information boundary that makes evaluation meaningful.

The six artifacts in this repository model this boundary from different technical perspectives. Python provides a complete executable experiment pipeline, JavaScript models an event-driven experiment, C++ implements a repository-change risk case study, Java represents the workflow through enterprise domain types and state transitions, and PostgreSQL represents the experiment boundary as relational data and database constraints.

---

## Core Experimental Boundary

Consider a dataset in which each observation describes a software repository change:

- lines changed
- files changed
- review comments
- previous failures
- developer experience
- an outcome label

The training partition can be used to estimate a classifier and preprocessing parameters. The validation partition can then answer questions such as whether one configuration is preferable to another. The test partition should not answer those development questions because repeatedly inspecting test performance effectively turns the test set into another validation set.

A useful conceptual flow is:

`raw observations → partition → training fit → validation decisions → final test evaluation`

The critical information direction is one-way. Information from validation or test observations must not influence parameters that are supposed to represent training.

---

## Random Splitting

A random split assumes that observations are sufficiently independent and that their ordering or entity identity does not create a dependency that crosses the partition boundary.

The Python implementation uses a seeded shuffle before constructing the partitions. The JavaScript implementation implements its own deterministic pseudo-random generator rather than relying on an uncontrolled source of randomness. The C++ implementation uses `std::mt19937` with an explicit seed.

Random splitting is useful when the dataset represents independently sampled observations from a reasonably stable population. It becomes dangerous when nearby observations are correlated, when the same entity appears repeatedly, or when the future distribution differs from the historical distribution.

A random split therefore answers a different question from a chronological split.

---

## Stratification

A stratified split attempts to preserve the distribution of the target label across partitions.

For binary classification, suppose the source data contains 80% class zero and 20% class one. A stratified partition attempts to retain approximately that relationship in training, validation, and testing.

This is particularly important when the positive class is uncommon. A purely random partition can produce a validation or test set with an unusually small number of positive examples, increasing metric variance.

Stratification does not solve every splitting problem. It does not prevent the same customer, patient, device, developer, household, or other repeated entity from appearing in multiple partitions. It also does not prevent temporal leakage.

The Python and JavaScript implementations explicitly distinguish stratification from group and chronological splitting.

---

## Training Data

Training data is the only partition allowed to determine fitted model parameters during the normal development workflow.

This rule applies to preprocessing as well as the final predictive model.

The Python `StandardScaler` calculates means and standard deviations from training features. Its `transform()` operation then applies those fixed parameters to validation and test observations.

The distinction is important:

`fit(training) → transform(training, validation, test)`

is valid, while:

`fit(all_data) → transform(training, validation, test)`

can leak information from held-out observations.

The second pattern may appear harmless because the target labels are not directly used. It is still a form of information leakage because the distribution of held-out observations affects the learned transformation.

---

## Validation Data

Validation data exists to support development decisions without exposing the final test estimate to those decisions.

Typical decisions include:

- selecting among model configurations
- choosing regularization strength
- selecting feature transformations
- selecting a classification threshold
- comparing preprocessing strategies
- comparing model families
- determining whether a particular configuration is suitable for further evaluation

The validation set can therefore influence the chosen system indirectly. This is why repeatedly optimizing against validation performance can eventually overfit the validation set.

When many configurations are compared, the validation score becomes part of the selection process. A final test set provides an independent checkpoint after those choices have been completed.

The Python `train_with_validation()` function deliberately does not accept a test partition. That design reduces the chance that test observations become part of model-selection logic.

---

## Test Data

The test partition represents the final held-out evaluation boundary.

A test result should be interpreted as evidence about generalization under the assumptions represented by the test distribution and split strategy.

A test set should not be used to decide which hyperparameter is best and then reported as if it were untouched. If the test set is repeatedly inspected during development, its role has changed.

The implementations therefore treat test evaluation as a final operation:

`trained model + frozen preprocessing + untouched test observations → final metrics`

The test set is not used to fit the scaler in any implementation.

---

## Data Leakage

Data leakage occurs when information unavailable to the model at the intended prediction time, or information that should remain outside the training boundary, influences model development or evaluation.

Leakage can occur in several technically different ways.

### Preprocessing leakage

Suppose feature standardization calculates its mean using training, validation, and test observations. The model has not directly seen test labels, but its input representation has been influenced by the test distribution.

The Python script intentionally demonstrates this by comparing a training-only scaler with a scaler fitted on all partitions.

The SQL model reinforces the rule through `preprocessing_fit.fitted_from`, which is constrained to `train`.

### Target leakage

A feature becomes problematic when it contains information that would only become available after the target event.

For example, a feature called `final_resolution_time` may be highly predictive of whether an incident will eventually breach a service-level target. If resolution time is only known after the incident has completed, it cannot legitimately be used for a prediction made at incident creation time.

Target leakage is therefore about information availability, not merely statistical correlation.

### Temporal leakage

A model intended to predict future events must not use future observations during training.

For example, if observations from December are used to train a model that is evaluated on November, the experiment can accidentally provide information from the future.

The C++ and Java implementations include chronological policies. The SQL script uses window functions to expose the temporal boundaries that a time-aware experiment would establish.

### Duplicate leakage

If the same underlying observation appears in both training and test, the test score can become artificially high.

Duplicates may be exact duplicates, repeated records, or near-duplicates created by preprocessing pipelines.

The Python implementation includes duplicate-signature detection to demonstrate why partitioning should occur after appropriate dataset identity and deduplication decisions.

### Entity leakage

Repeated entities create another boundary problem. A dataset may contain many rows for the same developer, customer, device, patient, machine, or account.

If the same entity occurs in training and test, the model may exploit entity-specific characteristics rather than learning behavior that generalizes to unseen entities.

The Python, C++, Java, and SQL implementations distinguish ordinary splitting from group-aware splitting.

---

## Group-Aware Splitting

A group-aware split keeps all observations associated with an entity in the same partition.

In the case-study data, `developer_id` or its equivalent is the group boundary.

If developer A has twenty observations, all twenty should be assigned to one partition when the experiment is intended to measure performance on unseen developers.

The objective changes from:

`Can the model generalize to new observations from known developers?`

to:

`Can the model generalize to developers not represented in training?`

That distinction is an experimental design decision, not merely an implementation detail.

---

## Chronological Splitting

Chronological splitting is appropriate when deployment involves predicting future observations from historical observations.

A typical arrangement is:

`January–June → training`

`July–August → validation`

`September–October → test`

The exact dates depend on the application.

The key property is that the training period precedes the validation period, which precedes the test period.

This arrangement is often more realistic for operational forecasting, financial data, telemetry, demand prediction, fraud detection, incident prediction, and other domains with time-dependent behavior.

A chronological split can also reveal distribution drift that a random split hides.

---

## Reproducibility

A reproducible experiment should be capable of reconstructing the same result when the relevant inputs and configuration are unchanged.

A random seed is one component of reproducibility, not the complete solution.

The experiment may also depend on:

- the exact dataset version
- row ordering
- partitioning algorithm
- preprocessing implementation
- model implementation
- feature definitions
- software runtime
- dependency versions
- numerical behavior
- configuration values

The JavaScript and C++ implementations explicitly seed their partitioning processes. The Python implementation uses `random.Random(seed)` rather than changing Python's global random state.

The SQL script stores the experiment seed and uses a deterministic hash-based assignment to make its example partition reproducible.

---

## Python Implementation

The Python program is a complete experiment pipeline rather than a collection of isolated examples.

`Record` represents an observation with numerical features, a target, an entity identifier, and a timestamp.

`random_split()` demonstrates an ordinary deterministic random partition.

`stratified_split()` preserves target-class proportions approximately.

`chronological_split()` sorts observations by timestamp before partitioning.

`group_split()` assigns complete developer groups to partitions.

`StandardScaler` demonstrates the fit/transform boundary. Its parameters are learned through `fit()` and reused by `transform()`.

`NearestCentroidClassifier` provides a small executable model so that the partition boundary can be observed through actual validation and test metrics rather than only through comments.

`train_with_validation()` intentionally accepts only training and validation observations. `evaluate_final()` is the separate test operation.

The script also compares partition distributions, detects duplicate signatures, demonstrates preprocessing leakage, and verifies repeatability with fixed seeds.

---

## JavaScript Implementation

The JavaScript implementation emphasizes event-driven experiment behavior.

`SeededRandom` provides deterministic shuffling without requiring an npm dependency.

`stratifiedSplit()` groups observations by target before partitioning them.

`chronologicalSplit()` establishes a temporal ordering.

`StandardScaler` separates `fit()` from `transform()`, making the preprocessing boundary explicit.

`ExperimentRunner` models an experiment as a sequence of events:

`split-created → validation-complete → test-complete`

This makes the lifecycle visible and demonstrates a useful JavaScript-specific pattern for systems that may eventually connect model evaluation to asynchronous jobs, dashboards, or event streams.

The test partition is deliberately transformed only after validation evaluation. The runner therefore keeps final evaluation conceptually separate from model-selection evidence.

---

## C++ Case Study

The C++ program models a repository-change risk engine.

Each `ChangeRecord` contains change size, file count, review comments, previous failures, developer experience, an outcome label, a developer identifier, and a day.

`DatasetSplitter` exposes three different experimental policies:

- random partitioning
- chronological partitioning
- group-aware partitioning

This is important because the correct split depends on the prediction problem.

`StandardScaler` implements training-derived normalization parameters. Its internal state is fitted once and then used for transformation.

`NearestCentroidModel` gives the case study a complete predictive mechanism. Each class is represented by a centroid, and predictions use squared Euclidean distance.

The evaluation layer calculates accuracy, precision, recall, and F1.

The case study also checks group overlap and demonstrates the difference between training-only and all-data preprocessing statistics.

---

## Java Implementation

The Java implementation uses explicit domain abstractions to model experiment state.

`DatasetPartition` identifies the role of an observation.

`ModelStage` prevents invalid lifecycle transitions such as evaluating the final test set before validation.

`SplitPolicy` defines a contract for partitioning behavior. `StratifiedRandomPolicy`, `ChronologicalPolicy`, and `GroupPolicy` implement different experimental assumptions.

`StandardScaler` again enforces the distinction between fitting preprocessing parameters and applying them to held-out observations.

`CentroidModel` owns model state and refuses prediction before training.

`ExperimentService` coordinates the experiment while keeping validation and final test evaluation as distinct operations.

This structure is useful in enterprise software because experimental policies become explicit domain components rather than being hidden inside a large conditional function.

---

## SQL Data Model

The PostgreSQL implementation treats the experiment as relational state.

The `repository` and `branch` tables provide the source context.

`observation` stores the machine-learning records and enforces non-negative feature values and a binary target through database constraints.

`experiment` records the experimental identity and random seed.

`experiment_partition` maps observations to train, validation, or test. Its composite primary key prevents the same observation from being assigned repeatedly within the same experiment.

`preprocessing_fit` records fitted preprocessing statistics. Its `CHECK` constraint requires the fitting partition to be `train`.

`model_evaluation` stores validation and test metrics separately.

Indexes support common access patterns such as time-based observation retrieval, entity lookup, target inspection, and partition filtering.

The SQL script also contains queries for target distribution, entity overlap, chronological boundaries, preprocessing provenance, and transactional rollback.

---

## Leakage and Split Selection

Leakage prevention starts before the model is trained.

A sound workflow asks:

- What information is available at prediction time?
- Are observations independent?
- Are there repeated entities?
- Does time determine what information was available?
- Can preprocessing estimate parameters from held-out rows?
- Can duplicate observations cross partition boundaries?
- Is the target itself encoded directly or indirectly in a feature?
- Is the test set being inspected repeatedly during development?

The answers determine whether random, stratified, chronological, group-aware, or another specialized split is appropriate.

A technically sophisticated experiment can still be invalid if the partition strategy does not match the data-generating process.

---

## Validation Versus Test Decisions

The most important distinction is the decision boundary.

| Activity | Training | Validation | Test |
|---|---:|---:|---:|
| Fit model parameters | Yes | No | No |
| Fit feature scaler | Yes | No | No |
| Compare model configurations | No | Yes | No |
| Tune a threshold | No | Yes | No |
| Select final configuration | Indirectly | Yes | No |
| Final generalization estimate | No | No | Yes |

The test partition should remain outside decisions that determine the final configuration.

If ten configurations are tested against the same test set and the best test result is selected, the test set has effectively become a development set. A new independent evaluation set would then be needed for an unbiased final estimate.

---

## Common Failure Modes

### Fitting preprocessing before splitting

Calculating scaling, imputation, feature-selection statistics, or other learned transformations on the complete dataset allows held-out information to influence training.

The safer pattern is to split first, fit preprocessing on training data, and apply the fitted transformation to validation and test data.

### Randomly splitting temporal observations

Randomly mixing historical and future observations can produce a model that has effectively learned from the future.

A chronological boundary is usually more appropriate when the deployment task is inherently temporal.

### Ignoring repeated entities

Randomly distributing records from the same entity can produce an evaluation that measures memorization of entity characteristics rather than generalization to unseen entities.

Group-aware splitting addresses that specific experimental question.

### Reusing the test set

Using test performance to decide which feature engineering strategy or hyperparameter is best compromises the independence of the final estimate.

Validation exists precisely so that development decisions do not require access to the test result.

### Treating a seed as complete reproducibility

A fixed random seed cannot compensate for changed source data, changed preprocessing, changed code, changed dependency versions, or changed partition algorithms.

Reproducibility requires preserving the experiment configuration and relevant inputs as well as the random seed.

### Confusing class balance with validity

A perfectly balanced train and test set can still contain severe leakage. Conversely, a naturally imbalanced dataset can be valid if its imbalance accurately represents the intended deployment population.

Stratification is a distribution-control mechanism, not a universal leakage-prevention mechanism.

---

## Performance Considerations

Splitting itself is usually inexpensive compared with model training, but large datasets change implementation choices.

Random shuffling may require memory proportional to the number of observations. Group-aware splitting requires maintaining group membership. Chronological splitting can often be implemented efficiently when the source data is already sorted by time.

Database-backed pipelines can use indexed timestamps, entity identifiers, and experiment partitions to avoid repeatedly scanning the entire dataset.

Preprocessing should also avoid accidental repeated fitting. A scaler fitted once on training data should be reused for validation and test transformations rather than recalculated independently.

For cross-validation, computational cost increases because several training and validation fits are required. This makes leakage-safe pipeline design even more important.

---

## Security and Governance Considerations

Evaluation data can contain sensitive or commercially important information. Access to the test partition should be controlled when its purpose depends on keeping it unseen during development.

Experiment metadata should identify the dataset version, split policy, random seed, feature definitions, preprocessing configuration, and model configuration.

Auditability is especially important when a metric influences a business decision. A reported score without its partitioning assumptions can be misleading.

A reproducible experiment should make it possible to answer not only "What was the score?" but also "Which observations were used, under which partitioning policy, with which preprocessing parameters, and under which configuration?"

---

## Practical Interpretation

A train-test split is an experimental design mechanism rather than a formatting operation.

The correct question is not simply:

`How should the dataset be divided?`

It is:

`What information would genuinely be available when this model is deployed, and what generalization claim should the evaluation support?`

Random splitting is appropriate for some independently sampled datasets. Stratification helps preserve target proportions. Group-aware splitting protects entity boundaries. Chronological splitting protects temporal boundaries. Training-only fitting protects preprocessing boundaries. Validation supports development decisions. Testing provides the final held-out estimate.

These mechanisms work together, but each protects a different assumption. Treating them as interchangeable is one of the easiest ways to produce an evaluation that looks precise while measuring the wrong thing.
