# XGBoost: Gradient Boosting Mechanics, Regularization, and Hyperparameters

## Scope

This project studies the mechanics behind XGBoost-style gradient-boosted decision trees through five executable implementations.

The central learning objective is to understand how a boosted tree model converts an existing prediction error into a new tree, how first- and second-order derivatives determine leaf values and split quality, and how regularization and hyperparameters control model complexity.

The four major mechanisms are:

| Mechanism | Role |
|---|---|
| Gradient boosting | Adds trees sequentially to correct residual model error |
| Second-order optimization | Uses gradients and Hessians to construct better leaf weights and split decisions |
| Regularization | Penalizes unnecessarily complex or extreme tree solutions |
| Hyperparameters | Control learning rate, tree depth, sampling, split thresholds, and model capacity |

The implementations use a binary classification objective because logistic loss makes the gradient and Hessian calculations explicit.

## Core XGBoost Objective

For a training record with prediction margin `f(x)`, the logistic probability is:

`p = 1 / (1 + exp(-f(x)))`

For binary logistic loss, the gradient for observation `i` is:

`g_i = p_i - y_i`

The Hessian is:

`h_i = p_i(1 - p_i)`

XGBoost uses a second-order Taylor approximation of the objective for each boosting iteration. For a leaf containing observations `I`, define:

`G = sum(g_i)`

`H = sum(h_i)`

With L2 regularization `lambda`, the optimal leaf weight is approximately:

`w* = -G / (H + lambda)`

L1 regularization modifies the gradient through soft thresholding. If `alpha` is the L1 coefficient, a non-zero effective gradient is reduced toward zero before calculating the leaf weight.

The resulting optimization is not simply residual fitting. Each tree is selected using the current model's derivatives.

## Gradient Boosting Workflow

The implementations follow this conceptual sequence:

`Initial margin`
`      ↓`
`Calculate probabilities`
`      ↓`
`Calculate gradients and Hessians`
`      ↓`
`Search candidate tree splits`
`      ↓`
`Evaluate regularized split gain`
`      ↓`
`Construct tree`
`      ↓`
`Multiply tree contribution by learning rate`
`      ↓`
`Update prediction margins`
`      ↓`
`Repeat for subsequent boosting rounds`

The model is additive. A new tree does not replace the previous trees.

`F_t(x) = F_(t-1)(x) + learning_rate × f_t(x)`

A smaller learning rate usually requires more boosting rounds to reach comparable training capacity, while a larger learning rate can reach a strong fit with fewer trees but may generalize less reliably.

## Split Gain

For a candidate split, XGBoost evaluates the quality of the parent and child nodes using accumulated gradients and Hessians.

A simplified regularized gain expression is:

`Gain = 1/2 [ G_L²/(H_L + lambda) + G_R²/(H_R + lambda) - G²/(H + lambda) ] - gamma`

L1 regularization modifies each gradient before this calculation.

The four quantities have distinct meanings:

| Quantity | Meaning |
|---|---|
| `G` | Aggregate first-order error signal |
| `H` | Aggregate curvature information |
| `lambda` | L2 penalty on leaf weights |
| `gamma` | Minimum improvement required for a split |

A candidate split that does not produce sufficient improvement is rejected.

## Regularization

Regularization is part of the optimization itself rather than a separate post-processing operation.

### L2 regularization

`reg_lambda` penalizes large leaf weights.

Increasing it generally makes the model more conservative because the denominator in the leaf-weight calculation increases.

The Python, JavaScript, Java, C++, and SQL implementations expose this mechanism explicitly.

### L1 regularization

`reg_alpha` performs gradient soft thresholding.

Small gradient sums can become zero after the L1 penalty. This can produce zero-valued leaves and reduce unnecessary model complexity.

### Split penalty

`gamma` requires a candidate split to provide sufficient objective improvement.

A larger `gamma` therefore makes tree growth more conservative.

### Minimum child weight

`min_child_weight` prevents a child from being created when its accumulated Hessian is too small.

This is especially useful when a proposed split isolates observations with insufficient optimization information.

## Hyperparameters

The implementations distinguish parameters that affect tree structure from parameters that affect the optimization process.

| Parameter | Effect |
|---|---|
| `learning_rate` | Scales every newly added tree |
| `n_estimators` | Number of boosting rounds |
| `max_depth` | Maximum structural depth of each tree |
| `min_child_weight` | Minimum Hessian required for a child |
| `reg_lambda` | L2 regularization |
| `reg_alpha` | L1 regularization |
| `gamma` | Minimum split gain |
| `subsample` | Fraction of observations sampled per boosting round |
| `colsample_bytree` | Fraction of features considered by a tree |

These parameters interact.

For example, increasing `max_depth` raises potential model capacity. Increasing `reg_lambda`, `reg_alpha`, `gamma`, or `min_child_weight` can counteract that capacity. Reducing `learning_rate` while increasing `n_estimators` changes the optimization trajectory without simply producing the same model.

## Python Implementation

The Python implementation provides the most complete educational training path.

It defines:

- A `Row` data structure for feature vectors and labels.
- A recursive tree representation.
- Second-order gradient and Hessian calculations.
- L1-regularized leaf weights.
- Regularized split-gain evaluation.
- Maximum-depth control.
- Minimum-child-Hessian validation.
- Row subsampling through `subsample`.
- Feature subsampling through `colsample_bytree`.
- Probability and class prediction.
- Log-loss evaluation.
- Model configuration persistence as JSON.

The tree builder explicitly examines candidate thresholds. This makes the split mathematics visible.

It is intentionally not a replacement for the optimized XGBoost library. Real XGBoost uses highly optimized tree construction, efficient memory layouts, parallel execution, specialized histogram algorithms, sparse-data handling, and additional production capabilities.

## JavaScript Implementation

The JavaScript implementation approaches the same learning problem through an event-driven model.

`ModelEventBus` emits prediction events after the model has been trained. This separates model calculation from consumers that may display, log, validate, or react to predictions.

The implementation includes:

- Input validation.
- Tree objects with recursive inference.
- Logistic probabilities.
- Gradient and Hessian calculation.
- L1 and L2 regularization.
- Gain-based split selection.
- Sampling configuration.
- Prediction events.
- Model metadata serialization.

The JavaScript version is deliberately structured as a Node.js executable rather than a browser-specific application.

## C++ Implementation

The C++ program treats the model as a numerical governance-risk engine.

The domain record represents measurable characteristics of a repository change:

- Changed lines.
- Test failure rate.
- Historical risk.
- Review coverage.
- Binary target.

The tree engine uses explicit vectors and recursive tree construction. It computes regularized leaf weights and split gains directly.

The implementation comments identify an important engineering distinction: exhaustive threshold evaluation is useful for understanding the mathematics but is not the preferred strategy for large production datasets.

Production-scale gradient boosting requires attention to:

- Memory consumption.
- Cache locality.
- Parallel split evaluation.
- Histogram construction.
- Sparse feature representation.
- Numerical stability.
- Dataset size.
- Tree depth.
- Number of boosting rounds.

The C++ program therefore emphasizes the algorithmic perspective rather than presenting a generic machine-learning wrapper.

## Java Implementation

The Java implementation models the boosting system using enterprise-oriented domain types.

`PullRequest` is an immutable record with constructor validation.

`HyperParameters` is another immutable record that rejects invalid model configurations before training.

The `Objective` interface separates the optimization objective from the tree engine. `LogisticObjective` supplies gradients and Hessians without coupling those calculations to tree construction.

The model contains:

- Explicit hyperparameter validation.
- Immutable domain records.
- An objective interface.
- Recursive tree nodes.
- L1 regularization.
- L2 regularization.
- Split-gain evaluation.
- Minimum-child-weight enforcement.
- Probability classification.
- Explicit risk classes.

This separation makes it possible to replace the objective while retaining the general tree-building architecture.

## PostgreSQL Implementation

The PostgreSQL script represents the model as relational metadata and numerical training components.

The schema separates:

`model_run`
→ model configuration and hyperparameters

`training_record`
→ observations and target values

`boosting_tree`
→ boosting iterations

`tree_node`
→ split and leaf structure

`prediction`
→ generated model outputs

This structure makes the relationship between configuration, boosting iterations, tree structure, and predictions queryable.

The database enforces important invariants with:

- Primary keys.
- Foreign keys.
- Unique constraints.
- Check constraints.
- Indexes.
- Immutable numerical functions.
- Views.
- Transactional execution.

The SQL functions `xgboost_leaf_weight` and `xgboost_split_gain` encode the central second-order formulas directly in PostgreSQL.

The SQL representation is a relational model of the algorithm rather than a replacement for an optimized XGBoost training engine. PostgreSQL is used to demonstrate how model configuration, training observations, tree metadata, derivative calculations, and prediction reporting can be represented and validated at the database layer.

## Model Capacity and Overfitting

Tree depth and estimator count directly affect capacity.

A shallow tree has limited interaction depth. A deeper tree can model more complex feature interactions but can also memorize training-specific patterns.

Increasing `n_estimators` adds more corrective stages.

Increasing `max_depth` allows each stage to express more complex rules.

Increasing `learning_rate` makes each tree contribution larger.

These parameters should therefore be considered jointly rather than independently.

A common conceptual trade-off is:

`Lower learning rate + more trees`
versus
`Higher learning rate + fewer trees`

The first approach can provide finer-grained optimization but increases computation.

## Subsampling

`subsample < 1` means each boosting iteration can train on a subset of observations.

`colsample_bytree < 1` means each tree can use a subset of available features.

These mechanisms reduce correlation between individual trees and can improve generalization in suitable datasets.

They also introduce stochasticity, so reproducibility requires controlled random-state handling in implementations that use random sampling.

## Edge Cases

The implementations explicitly guard against several invalid conditions:

- Empty training datasets.
- Missing feature dimensions.
- Inconsistent feature widths.
- Invalid binary labels.
- Invalid learning rates.
- Invalid tree depths.
- Invalid estimator counts.
- Negative regularization parameters.
- Invalid sampling fractions.
- Invalid classification thresholds.
- Child nodes with insufficient Hessian mass.
- Splits with empty child partitions.
- Models used before training.
- Non-finite or malformed numerical input where validation is applicable.

A production implementation would also need explicit policies for missing values, categorical features, extreme numeric magnitudes, class imbalance, distributed training, and model serialization compatibility.

## Numerical Considerations

The logistic function is implemented using a sign-aware form.

For positive margins, it evaluates:

`1 / (1 + exp(-x))`

For negative margins, it uses an equivalent expression that avoids unnecessary overflow in `exp(-x)`.

Hessians are bounded below by a small positive value in the implementations to prevent unstable division when the theoretical curvature becomes extremely small.

These details matter because boosted models repeatedly transform and accumulate prediction margins.

## Performance Considerations

The educational implementations deliberately favor transparency.

Candidate split searches can be computationally expensive because many feature-threshold combinations are evaluated.

For a dataset with `n` observations, `d` features, and `T` trees, exhaustive tree construction can become expensive as tree depth increases.

Production XGBoost reduces this cost through optimized data structures and histogram-based algorithms.

The practical performance trade-off is therefore not simply about programming language speed. Algorithmic tree construction, memory layout, parallelism, feature sparsity, dataset size, and hyperparameter choices dominate large-scale performance.

## Security and Operational Considerations

A model-training pipeline should treat training data and model artifacts as controlled assets.

Important controls include:

- Validate numeric ranges before training.
- Reject malformed records.
- Avoid untrusted dynamic code execution.
- Restrict filesystem permissions for serialized models.
- Record model hyperparameters and training versions.
- Preserve reproducible training configuration.
- Monitor prediction distributions after deployment.
- Detect feature drift.
- Detect changes in label distribution.
- Keep training and evaluation datasets separated.
- Avoid evaluating performance only on training data.

A high training accuracy does not establish that a model generalizes.

## Relationship Between the Five Implementations

The implementations intentionally use different perspectives.

| File | Primary perspective |
|---|---|
| Python | Complete educational training simulation |
| JavaScript | Event-driven model and prediction processing |
| C++ | Numerical tree and split-gain engine |
| Java | Enterprise domain and objective architecture |
| PostgreSQL | Relational model, constraints, formulas, and reporting |

They share the same mathematical foundation but do not simply translate the same program into five languages.

The common structure is:

`Objective`
→ `Gradient/Hessian`
→ `Candidate split`
→ `Regularized gain`
→ `Leaf weight`
→ `Tree`
→ `Learning-rate-scaled update`
→ `Next boosting round`

That shared structure is the core connection between gradient boosting mechanics, regularization, and hyperparameter behavior.
