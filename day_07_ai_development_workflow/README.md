# AI Development Workflow

## Problem Definition, Data Collection, Preprocessing, Training, Evaluation, Deployment, and Monitoring

This study material demonstrates a complete artificial intelligence and machine-learning development workflow through a self-contained binary classification system. The implementation uses synthetic customer churn data and builds logistic regression from first principles using only Python's standard library.

The workflow is organized around the major stages of a production-oriented machine-learning lifecycle:

1. Problem definition
2. Data collection
3. Data understanding
4. Data validation and cleaning
5. Train, validation, and test splitting
6. Preprocessing
7. Feature scaling
8. Baseline development
9. Model training
10. Validation and evaluation
11. Threshold selection
12. Error analysis
13. Final testing
14. Model serialization
15. Inference validation
16. Batch deployment concepts
17. Monitoring and drift detection
18. Testing

The central example predicts whether a customer is likely to churn.

## 1. Problem Definition

Machine-learning development should begin by defining the problem precisely. Selecting a model before understanding the business and decision context can result in a technically functional system that does not solve a useful problem.

The script represents the problem using a `ProblemDefinition` data class.

The example defines:

- Business problem: identify customers at elevated risk of churn.
- Prediction task: binary classification.
- Unit of prediction: one customer.
- Target variable: `churned`.
- Positive class: customer churn.
- Primary metric: recall, while monitoring precision.
- Constraints: prediction-time data availability, reproducibility, and the costs of false positives and false negatives.

### Business Problem Versus Prediction Problem

A business problem describes an organizational objective. A prediction problem describes the measurable computational task used to support that objective.

For example:

- Business objective: reduce customer churn.
- Prediction task: estimate the probability that a customer will churn.

The distinction matters because a high-quality prediction is useful only when it can support a meaningful decision.

### Unit of Prediction

The unit of prediction defines what one observation represents.

Examples include:

- one customer
- one transaction
- one image
- one medical examination
- one loan application
- one network event

The customer churn example uses one customer as the unit of prediction.

### Target Variable

The target is the value that supervised learning attempts to predict.

In the script:

- `churned = 1` means the customer churned.
- `churned = 0` means the customer remained.

The quality of target labels is critical. Incorrect, delayed, inconsistent, or biased labels can limit model performance regardless of algorithm complexity.

### Prediction-Time Availability

A feature should normally contain information available when the prediction is made.

Using information that becomes available after the outcome occurs is known as target leakage or data leakage.

For example, a field recording the reason a customer cancelled an account cannot legitimately be used to predict whether the customer will later cancel. The model would learn from information that is unavailable at deployment time.

## 2. Data Collection

Data collection obtains observations needed to solve the defined problem.

Real machine-learning systems may collect data from:

- relational databases
- transactional systems
- APIs
- sensors
- application logs
- data warehouses
- documents
- images
- audio
- manually labeled datasets
- streaming systems

The script generates synthetic customer records to remain self-contained.

Each record contains:

| Feature | Meaning |
|---|---|
| `age` | Customer age |
| `monthly_spend` | Monthly spending |
| `support_tickets` | Number of support interactions |
| `months_as_customer` | Customer tenure |
| `uses_premium_plan` | Premium-plan indicator |
| `churned` | Binary target |

The synthetic target is generated from an underlying probabilistic relationship involving the features. Noise is intentionally added because real-world outcomes are rarely perfectly determined by observed features.

### Data Quality Risks

Data collection may introduce:

- missing values
- duplicates
- invalid measurements
- inconsistent schemas
- incorrect labels
- biased sampling
- stale observations
- unauthorized data collection
- privacy risks
- data leakage

The script deliberately introduces missing values, a duplicate observation, an invalid age, and a missing target to demonstrate preprocessing and validation.

## 3. Data Understanding

Data understanding is the process of examining the collected dataset before training.

The script calculates:

- record counts
- missing-value counts
- positive and negative class counts
- positive-class rate
- minimum feature values
- maximum feature values
- mean feature values

These checks help identify obvious problems before they become model failures.

### Class Distribution

The distribution of the target classes is important.

Suppose a dataset contains:

- 95% non-churned customers
- 5% churned customers

A model that predicts "non-churn" for every customer would achieve 95% accuracy while identifying no customers who actually churn.

This demonstrates why accuracy alone can be misleading.

### Distribution Inspection

Feature ranges should be checked against domain expectations.

Examples:

- age should not normally be negative.
- customer tenure should not be negative.
- support ticket counts should not be negative.
- a premium indicator should have valid category values.

Domain validation prevents some classes of data errors from silently entering the model.

## 4. Data Validation and Cleaning

The script validates every record through `validate_customer_record`.

Validation checks include:

- positive customer identifier
- plausible age range
- non-negative spending
- non-negative support-ticket count
- non-negative customer tenure
- binary premium-plan indicator
- binary target when a target is present

Invalid records are separated from valid records.

### Duplicate Handling

The script creates a duplicate signature using the observation values while excluding `customer_id`.

This illustrates an important principle: identifiers and observations are different concepts.

Two records can have different identifiers while containing identical observations.

In real systems, duplicate detection can be significantly more complex because duplicates may differ slightly due to timestamps, spelling, formatting, or measurement precision.

### Missing Target Values

Supervised learning requires known targets during training.

Records without labels may still be useful for:

- inference
- unsupervised learning
- semi-supervised learning
- data analysis

They cannot directly contribute to ordinary supervised loss calculation because the correct answer is unknown.

The script removes records with missing targets before supervised training.

## 5. Train, Validation, and Test Sets

The cleaned labeled dataset is divided into three subsets.

### Training Set

The training set is used to learn model parameters.

For logistic regression, these parameters are:

- feature weights
- bias or intercept

### Validation Set

The validation set is used for development decisions such as:

- hyperparameter selection
- threshold selection
- model comparison
- architecture selection

The validation set should not directly update model parameters.

### Test Set

The test set provides the final evaluation after major model decisions have been completed.

Repeatedly changing the model after observing test results gradually causes the test set to become part of the development process. This weakens its value as an independent estimate of generalization.

### Random Splitting

The script uses a shuffled random split.

Random splitting is appropriate for many independent and identically distributed datasets.

It is not automatically appropriate for time-dependent problems.

Examples where chronological splitting is often preferable include:

- stock prediction
- demand forecasting
- fraud detection with evolving behavior
- sensor streams
- customer behavior over time

Randomly placing future observations into the training set can create temporal leakage.

## 6. Missing-Value Imputation

The script calculates feature medians from the training data.

Missing values are then replaced with those medians.

### Why Median Imputation?

The median is less sensitive to extreme values than the mean.

For example, consider:

- 10
- 12
- 13
- 15
- 1000

The mean is strongly affected by the extreme value. The median is more robust.

Median imputation is not always optimal. Other approaches include:

- mean imputation
- mode imputation
- constant-value imputation
- predictive imputation
- model-based imputation
- dedicated missing-value indicators

The appropriate method depends on the feature and the meaning of missingness.

### Missingness Can Carry Information

A missing value is not always random.

For example, customers who fail to provide optional information may behave differently from customers who provide it.

A production model may therefore use both:

- an imputed numerical value
- a binary feature indicating whether the original value was missing

The example focuses on median imputation to keep the implementation compact and transparent.

## 7. Why Preprocessing Must Be Fit on Training Data

The script fits imputation medians and scaling parameters using only training records.

This prevents information leakage.

A common mistake is:

1. calculate statistics using the complete dataset
2. split the dataset afterward

This allows validation and test distributions to influence the training representation.

Even when target labels are not used directly, information about held-out distributions can still leak into the training process.

The correct pattern is:

1. split the data
2. fit preprocessing on training data
3. transform training data
4. use the same fitted preprocessing parameters for validation and test data

## 8. Feature Scaling

The script implements standardization using the formula:

    z = (x - mean) / standard deviation

After scaling, numerical features are centered relative to the training distribution.

### Why Scaling Matters

Features may have very different numerical ranges.

For example:

- age may range from 18 to 80.
- monthly spending may range from 10 to 500.
- premium-plan status may be 0 or 1.

Gradient-based optimization can behave poorly when feature scales differ substantially.

Scaling can improve:

- numerical stability
- convergence behavior
- comparability of optimization updates

### Constant Features

A constant feature has zero standard deviation.

The script replaces a zero standard deviation with `1.0` to avoid division by zero.

A constant feature normally provides little or no predictive variation because every observation has the same value.

## 9. Logistic Regression

The model implemented in the script is binary logistic regression.

Despite its name, logistic regression is commonly used for binary classification.

The model calculates a linear score:

    z = w1*x1 + w2*x2 + ... + wn*xn + b

where:

- `wi` is a learned feature weight.
- `xi` is a feature value.
- `b` is the bias or intercept.

The score is converted into a probability using the sigmoid function:

    sigmoid(z) = 1 / (1 + exp(-z))

The output lies between 0 and 1.

### Interpretation of the Output

A model output such as:

    0.82

can be interpreted as a predicted probability or score associated with the positive class, subject to the assumptions and calibration quality of the model.

A probability estimate and a classification decision are different things.

The probability can be converted into a binary decision by applying a threshold.

## 10. Numerical Stability of the Sigmoid Function

The direct sigmoid formula can overflow for extreme values.

The script implements a numerically stable version.

For non-negative inputs, it uses:

    1 / (1 + exp(-z))

For negative inputs, it uses an equivalent form based on:

    exp(z) / (1 + exp(z))

This reduces the risk of overflow.

Numerical stability is important because mathematically equivalent formulas can behave differently in finite-precision computer arithmetic.

## 11. Binary Cross-Entropy Loss

The script trains logistic regression using binary cross-entropy.

For one observation:

    loss = -[y*log(p) + (1-y)*log(1-p)]

where:

- `y` is the actual binary label.
- `p` is the predicted probability.

The loss strongly penalizes confident incorrect predictions.

### Why Probabilities Are Clamped

The logarithm of zero is undefined.

The script clamps probabilities to a small interval away from exactly zero and one before calculating logarithms.

This is another numerical stability technique.

## 12. Gradient Descent

The model is trained with batch gradient descent.

The workflow is:

1. initialize weights and bias
2. calculate predictions
3. calculate loss
4. calculate gradients
5. update parameters
6. repeat for multiple epochs

The update rule is:

    parameter = parameter - learning_rate * gradient

### Learning Rate

The learning rate controls the size of optimization updates.

If the learning rate is too small:

- training can be slow.

If it is too large:

- optimization may oscillate.
- the loss may diverge.
- training may become unstable.

### Epochs

An epoch is one complete optimization pass through the training data.

Too few epochs may result in undertraining.

Too many epochs can increase computational cost and may contribute to overfitting in some learning settings.

The script stores training loss history for inspection.

## 13. Baseline Models

The script evaluates a majority-class baseline.

A baseline provides a minimum reference for evaluating whether a more complex model is useful.

For binary classification, a majority-class baseline predicts whichever class appears most frequently in the training data.

A model should generally be compared against relevant alternatives rather than evaluated in isolation.

Possible baselines include:

- majority class
- random predictor
- simple heuristic
- previous production system
- linear model
- rule-based system

The correct baseline depends on the application.

## 14. Classification Metrics

The script calculates:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC

### Confusion Matrix

The confusion matrix contains four outcomes.

| Actual | Predicted | Category |
|---|---|---|
| Positive | Positive | True Positive |
| Negative | Negative | True Negative |
| Negative | Positive | False Positive |
| Positive | Negative | False Negative |

### Accuracy

Accuracy is:

    correct predictions / total predictions

It is useful when classes and error costs are reasonably balanced.

It can be misleading for highly imbalanced datasets.

### Precision

Precision is:

    true positives / predicted positives

Precision answers:

> When the model predicts the positive class, how often is it correct?

### Recall

Recall is:

    true positives / actual positives

Recall answers:

> Of all actual positive cases, how many did the model identify?

### F1 Score

F1 is the harmonic mean of precision and recall.

It is:

    2 * precision * recall / (precision + recall)

The harmonic mean penalizes situations where one metric is high and the other is very low.

### Zero-Denominator Cases

Metrics can encounter zero denominators.

For example, if a model predicts no positive cases, precision has no predicted positives in its denominator.

The script uses a safe division function and returns zero for such cases.

Production metric systems should document their handling of undefined cases because different conventions can produce different interpretations.

## 15. ROC-AUC

ROC-AUC measures ranking quality across classification thresholds.

A useful interpretation is the probability that a randomly selected positive observation receives a higher score than a randomly selected negative observation.

The script implements an approximate rank-based calculation.

ROC-AUC is threshold-independent, but it has limitations.

A high ROC-AUC does not guarantee:

- good probability calibration
- good performance at a specific operational threshold
- good precision in highly imbalanced applications
- low business cost

Metric selection should reflect the actual decision problem.

## 16. Classification Threshold Selection

The default classification threshold is often 0.5:

    probability >= 0.5

This is a convention, not a universal optimum.

The script evaluates thresholds between 0.01 and 0.99 and selects the threshold with the highest validation F1 score.

### Lower Thresholds

A lower threshold generally produces more positive predictions.

Possible effects include:

- increased recall
- decreased precision
- more false positives

### Higher Thresholds

A higher threshold generally produces fewer positive predictions.

Possible effects include:

- increased precision
- decreased recall
- more false negatives

The appropriate threshold depends on business costs.

For customer churn:

- a false positive may trigger an unnecessary retention offer.
- a false negative may fail to identify a customer who leaves.

A production threshold should therefore reflect the cost and value associated with different outcomes.

## 17. Validation Evaluation

The validation set is used to evaluate the trained model and select operational decisions.

The script calculates:

- validation accuracy
- validation precision
- validation recall
- validation F1
- validation ROC-AUC

It then selects a classification threshold.

The validation set should not be treated as another training set. Repeated experimentation can eventually overfit development decisions to validation data.

Cross-validation and carefully managed experiment tracking are often used when data is limited or model selection is extensive.

## 18. Final Test Evaluation

The test set is evaluated after the model and threshold are selected.

The script reports:

- confusion matrix
- accuracy
- precision
- recall
- F1
- ROC-AUC

The test set provides an estimate of performance on previously unseen data, assuming the test distribution is representative of future deployment data.

A strong test result does not guarantee production success because real-world data may change.

## 19. Error Analysis

Aggregate metrics do not explain why a model fails.

The script examines:

- false negatives
- false positives

For each example, it prints selected feature values and predicted probability.

### False Negatives

A false negative occurs when:

- the actual class is positive
- the model predicts negative

For churn, this means the customer churns but the model fails to identify the risk.

### False Positives

A false positive occurs when:

- the actual class is negative
- the model predicts positive

For churn, this may cause unnecessary intervention.

Error analysis can reveal:

- missing features
- weak labels
- unusual input ranges
- systematic subgroup failures
- insufficient training data
- incorrect preprocessing
- possible leakage

Model development should use error analysis to understand failures rather than relying only on aggregate scores.

## 20. Training-Serving Consistency

A major production risk is training-serving skew.

This occurs when training and deployment process data differently.

Examples include:

- different missing-value rules
- different feature order
- different category encoding
- different scaling statistics
- different units
- outdated deployment code

The script addresses this by storing preprocessing state inside `ChurnPreprocessor`.

The same fitted preprocessing parameters are used during training evaluation and inference.

## 21. Model Artifacts

A trained model is saved as a JSON artifact.

The artifact contains:

- model version
- feature names
- learned weights
- bias
- missing-value medians
- feature means
- feature standard deviations
- classification threshold
- creation timestamp

### Why Feature Order Matters

A trained model expects features in the same order used during training.

Suppose the model was trained with:

    [age, monthly_spend, support_tickets]

If deployment accidentally sends:

    [support_tickets, age, monthly_spend]

the model can produce incorrect predictions without necessarily raising an error.

The artifact explicitly stores feature names to reduce this risk.

## 22. Inference Input Validation

The deployment interface validates incoming records before prediction.

The inference function checks:

- required fields
- numeric conversion
- feature ranges
- binary categorical values

This prevents malformed input from silently reaching the model.

Examples of invalid input include:

- negative age
- missing fields
- non-numeric values
- invalid binary indicators

### Reliability and Security

Input validation is important for both reliability and security.

Production systems should also consider:

- authentication
- authorization
- rate limiting
- request-size limits
- logging controls
- dependency security
- secret management
- encrypted communication
- data privacy
- denial-of-service protection

A model-serving system should not assume that external input is trustworthy.

## 23. Deployment Patterns

The script demonstrates a reusable inference function and batch inference.

Common deployment patterns include the following.

### Batch Inference

Predictions are generated periodically for a large collection of observations.

Examples:

- nightly churn scoring
- daily fraud analysis
- weekly demand prediction

Advantages include:

- efficient processing of large datasets
- simpler operational architecture

Limitations include:

- predictions may become stale between batches

### Online Inference

Predictions are generated in response to individual requests.

Examples:

- recommendation systems
- fraud decisions
- interactive personalization

Advantages include:

- immediate predictions

Limitations include:

- latency requirements
- availability requirements
- higher operational complexity

### Streaming Inference

Predictions are generated continuously from incoming events.

Examples include:

- sensor monitoring
- transaction streams
- network monitoring

### Embedded Inference

The model runs directly within an application or device.

Examples include:

- mobile applications
- edge devices
- embedded systems

## 24. Batch Inference Error Handling

The script demonstrates batch processing with per-record error handling.

One invalid record does not necessarily need to terminate an entire batch.

The workflow returns either:

- a prediction result
- an error message associated with the input

This is useful for operational pipelines because malformed records can be logged and investigated while valid records continue through the system.

Error-handling policies should depend on the application.

For high-risk systems, an invalid input may need to stop processing and trigger a controlled failure.

## 25. Prediction Export

The script exports batch results to CSV.

CSV is useful for simple tabular interchange but has limitations:

- limited type information
- weak schema enforcement
- ambiguous missing values
- quoting and delimiter issues
- no native nested structure

JSON is more suitable for nested data structures.

Databases and columnar storage formats are often preferable for large-scale production pipelines.

## 26. Monitoring

Model deployment does not end the machine-learning workflow.

A deployed model can degrade even when the original test results were strong.

The script demonstrates a simple feature-mean comparison between training data and simulated current data.

A difference between feature means may indicate distribution change.

### Data Drift

Data drift occurs when input distributions change.

Examples:

- customer spending patterns change
- user demographics shift
- sensor behavior changes

### Concept Drift

Concept drift occurs when the relationship between features and outcomes changes.

For example, support tickets may once have been strongly associated with churn, but a change in customer service processes could alter that relationship.

Concept drift is more difficult to detect because input distributions may remain stable while predictive relationships change.

### Prediction Drift

Prediction drift occurs when the distribution of model outputs changes.

A sudden increase in predicted positive cases may indicate:

- real behavior changes
- upstream data changes
- schema failures
- model degradation

### Performance Monitoring

True performance can only be calculated after ground-truth labels become available.

For some systems, labels arrive immediately.

For others, labels may take weeks or months.

Monitoring should distinguish between:

- immediately available operational signals
- delayed quality signals

## 27. Testing Machine-Learning Systems

The script includes executable assertions.

The tests verify:

- sigmoid behavior
- loss behavior
- classification metrics
- invalid-record rejection

Machine-learning testing extends beyond checking model accuracy.

Useful test categories include:

### Unit Tests

Verify individual components.

Examples:

- preprocessing functions
- metric calculations
- validation rules

### Integration Tests

Verify that multiple components work together.

Examples:

- preprocessing followed by inference
- artifact loading followed by prediction

### Data Tests

Verify expected data properties.

Examples:

- required columns exist
- numerical ranges are plausible
- class labels are valid

### Regression Tests

Detect unintended changes in behavior after code changes.

### Operational Tests

Verify:

- inference latency
- API behavior
- failure handling
- artifact compatibility

## 28. Common Machine-Learning Workflow Mistakes

### Starting With the Algorithm

Selecting a sophisticated algorithm before defining the problem can produce a technically impressive but operationally irrelevant system.

The problem definition should determine the task, data, metric, and decision structure.

### Using Leaked Features

Features containing future information can produce unrealistically strong evaluation results.

Such models often fail after deployment.

### Fitting Preprocessing on All Data

Calculating means, medians, or encodings using validation and test data leaks held-out information.

Preprocessing parameters should be fitted on training data.

### Ignoring Baselines

A model should be compared with simpler alternatives.

Without a baseline, it may be unclear whether complexity provides meaningful improvement.

### Using Accuracy Alone

Accuracy can hide serious failures in imbalanced classification.

Precision, recall, F1, ROC-AUC, and domain-specific cost metrics may provide more useful perspectives.

### Reusing the Test Set for Model Selection

Repeated test-driven development can overfit decisions to the test set.

The validation set should be used for development choices.

### Ignoring Data Drift

A model can become outdated after deployment even when the code and model parameters remain unchanged.

### Training and Serving With Different Transformations

Different preprocessing implementations can silently invalidate deployment predictions.

Shared preprocessing artifacts reduce this risk.

### Treating Invalid Input as Normal Data

Production systems should validate input before inference.

## 29. Performance Considerations

The educational implementation uses Python loops and standard-library data structures.

This approach is transparent but not optimized for large datasets.

For larger workloads, common performance considerations include:

- vectorized numerical operations
- efficient matrix libraries
- batch processing
- parallel processing
- hardware acceleration
- efficient serialization
- caching
- asynchronous request handling

Performance optimization should follow measurement.

Premature optimization can make systems harder to understand without addressing the actual bottleneck.

## 30. Scalability Considerations

Scaling a machine-learning workflow involves more than selecting a faster model.

Potential bottlenecks include:

- data ingestion
- storage
- preprocessing
- training time
- memory consumption
- inference latency
- network communication
- model loading
- monitoring infrastructure

A model can be computationally efficient while the surrounding data pipeline is the true bottleneck.

## 31. Reproducibility

The synthetic data generator uses explicit random seeds.

Reproducibility is important for:

- debugging
- model comparison
- experiment verification
- incident investigation

Production reproducibility may require recording:

- dataset versions
- feature definitions
- preprocessing parameters
- random seeds
- model configuration
- training code version
- dependency versions
- evaluation results

A model artifact should be traceable to the conditions under which it was trained.

## 32. Model Versioning

The saved artifact includes a model version.

Versioning supports:

- rollback
- comparison
- controlled deployment
- incident investigation
- compatibility management

A production versioning strategy may separately track:

- model version
- dataset version
- feature schema version
- preprocessing version
- application version

These components can change independently.

## 33. Limitations of the Example

The script is intentionally educational and therefore uses simplified implementations.

Limitations include:

- synthetic data rather than real observations
- simple random splitting
- no hyperparameter search
- no regularization
- no cross-validation
- no probability calibration analysis
- simple drift comparison
- no distributed training
- no database integration
- no HTTP service
- no authentication or authorization layer
- no automated retraining pipeline

These limitations do not invalidate the workflow concepts. They distinguish the educational implementation from a complete enterprise machine-learning platform.

## 34. End-to-End Workflow Structure

The complete workflow demonstrated by the script is:

    Problem definition
        ↓
    Data collection
        ↓
    Data understanding
        ↓
    Validation and cleaning
        ↓
    Train / validation / test split
        ↓
    Fit preprocessing on training data
        ↓
    Transform datasets
        ↓
    Establish baseline
        ↓
    Train model
        ↓
    Evaluate validation performance
        ↓
    Select operational threshold
        ↓
    Evaluate final test performance
        ↓
    Perform error analysis
        ↓
    Save model and preprocessing artifact
        ↓
    Load artifact for inference
        ↓
    Validate production input
        ↓
    Run batch or online inference
        ↓
    Monitor data and prediction behavior
        ↓
    Investigate drift and performance degradation
        ↓
    Retrain or revise when justified by evidence

This lifecycle is iterative rather than strictly linear. Evaluation may reveal data problems. Deployment monitoring may reveal drift. Error analysis may reveal missing features. Business requirements may change the target or success metric.

A robust AI development workflow therefore treats models as one component within a broader system involving data, software, evaluation, operations, governance, security, and continuous monitoring.
