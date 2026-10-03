# Encoding Categorical Data

Categorical data contains values that represent groups, classes, states, or labels rather than inherently continuous numerical quantities. Machine-learning algorithms frequently require numerical feature representations, so categorical values must be transformed before they can be consumed by many models.

This project examines four important encoding strategies:

- **Label encoding** maps categories to integer identifiers.
- **One-hot encoding** creates an independent binary feature for each category.
- **Ordinal encoding** maps categories according to a meaningful domain-defined order.
- **Target encoding** replaces a category with a statistic calculated from the training target.

The four techniques are related because they all transform categorical values into numerical representations, but they make very different assumptions about what those categories mean.

The implementations use a customer-churn scenario. Region and acquisition channel are treated as nominal categories, subscription plan is treated as an ordered category, and customer segment is represented using smoothed target encoding.

## Categorical Data and Its Semantics

The correct encoding depends on the semantics of the feature.

A feature such as `region` may contain `North`, `South`, `East`, and `West`. These values identify groups, but there is no meaningful mathematical statement that `West > East` or that the distance between `North` and `South` is equivalent to the distance between two other regions.

A feature such as `plan_level` can be different. If the business definition is `Basic < Standard < Premium`, the order itself contains information. An ordinal representation can preserve that relationship.

A feature such as `customer_segment` may have predictive relationship with a target such as churn. Target encoding can represent that relationship using statistics learned from training outcomes, but this introduces a critical leakage boundary because the target is being used to construct a feature.

The encoding decision therefore begins with feature meaning rather than with the numerical algorithm that happens to be convenient.

## Label Encoding

Label encoding assigns an integer identifier to each distinct category.

For example, a fitted mapping might contain:

| Category | Encoded value |
|---|---:|
| East | 0 |
| North | 1 |
| South | 2 |
| West | 3 |

The important property is that the integers are identifiers. For nominal data, the numbers do not establish a legitimate ranking.

This distinction matters because some algorithms can interpret numerical input as ordered or continuous. If a nominal region is converted to `0`, `1`, `2`, and `3` and passed directly into such an algorithm, the model can treat those numbers as if their differences had mathematical meaning.

The Python implementation uses `LabelEncoder` with an optional unknown-category value of `-1`. The JavaScript implementation stores the mapping in a `Map`, while the C++ case study uses `std::map`.

Label encoding is particularly useful when an integer representation is required for a categorical identifier or class label. It is not automatically the correct representation for every categorical feature.

### Unknown categories

An encoder fitted on:

`North`, `South`, `East`, `West`

cannot assume that a future value such as `Central` was known during training.

The implementations demonstrate explicit unknown-category policies rather than silently inventing a new training mapping at inference time.

Possible policies include:

- raising an error when an unexpected category is encountered;
- assigning a dedicated unknown identifier;
- representing an unknown one-hot category as an all-zero vector;
- using the global target statistic for an unknown target-encoded category.

The correct choice depends on the downstream model and production data contract.

## One-Hot Encoding

One-hot encoding represents each nominal category with its own binary feature.

For a region feature:

| Region | region_East | region_North | region_South | region_West |
|---|---:|---:|---:|---:|
| East | 1 | 0 | 0 | 0 |
| North | 0 | 1 | 0 | 0 |
| South | 0 | 0 | 1 | 0 |
| West | 0 | 0 | 0 | 1 |

The representation does not impose a numerical ordering between categories. This makes one-hot encoding a natural choice for nominal variables.

The Python and JavaScript implementations explicitly support unknown categories. With the `ignore` behavior, an unseen category produces zeros across the learned one-hot columns rather than changing the feature schema.

The C++ implementation represents the learned categories in a sorted vector and creates the binary vector using category lookup.

### Dropping a reference category

If a linear model includes an intercept and every category of a feature is represented by a one-hot column, the columns are linearly dependent because exactly one category is active for each valid observation.

For example:

`region_East + region_North + region_South + region_West = 1`

when every observation belongs to exactly one region.

The Python and C++ implementations therefore support the concept of dropping a reference category. With three remaining columns, the omitted category becomes the implicit reference group.

Dropping a category is primarily a modeling-design issue. It does not mean that the omitted category is less important. It changes the mathematical parameterization of the feature.

### Cardinality

One-hot encoding increases feature dimensionality as the number of categories increases.

A feature with four regions produces four binary columns under full one-hot encoding. A feature with hundreds of thousands of distinct identifiers can produce an impractically wide representation.

This makes cardinality an important consideration when choosing an encoding strategy.

## Ordinal Encoding

Ordinal encoding assigns numerical values according to an explicit semantic order.

The project defines:

`Basic < Standard < Premium`

and therefore maps the values to:

| Plan | Ordinal value |
|---|---:|
| Basic | 0 |
| Standard | 1 |
| Premium | 2 |

Unlike label encoding, this ordering is intentional.

The encoder does not infer the order from alphabetic sorting or from category frequency. The order is supplied by the domain definition.

This is important because an arbitrary category ordering would turn ordinal encoding into a misleading numerical representation.

For example, alphabetically sorting:

`Bronze`, `Gold`, `Silver`

would produce `Bronze < Gold < Silver`, which is not the intended economic ordering. A valid ordinal encoder must receive the actual domain order.

Unknown ordinal values are handled explicitly. The implementations can return `-1` or raise an error depending on the configured policy.

## Target Encoding

Target encoding is a supervised categorical representation. Instead of creating one column per category, the encoder replaces a category with a statistic calculated from the target.

For binary churn prediction, a simple category statistic is its observed churn rate.

Suppose:

| Segment | Observed churn |
|---|---|
| Student | 1, 0, 1, 1 |
| Professional | 0, 0, 0 |
| Enterprise | 0, 0, 0 |

The raw target means are approximately:

- Student: `0.75`
- Professional: `0.00`
- Enterprise: `0.00`

A target-encoded feature therefore contains a numerical estimate related to the target rather than an independent category indicator.

This can be much more compact than one-hot encoding for high-cardinality categorical features.

The major difference from one-hot encoding is that target encoding uses outcome information. That makes it powerful but also makes leakage prevention essential.

## Smoothing Target Encoding

Small category groups can produce unstable target statistics.

A category appearing once with target value `1` has a raw target mean of `1.0`, but one observation provides very little evidence.

The implementations use the smoothed estimate:

`encoded = (count × category_mean + smoothing × global_mean) / (count + smoothing)`

Here:

- `count` is the number of observations for the category;
- `category_mean` is its observed target mean;
- `global_mean` is the target mean across the training dataset;
- `smoothing` controls how strongly sparse categories are pulled toward the global mean.

When a category has many observations, its own target mean dominates.

When a category has very few observations, the global mean has greater influence.

With smoothing equal to zero, the formula reduces to the raw category mean.

Increasing smoothing does not eliminate category-specific information. It reduces the influence of unreliable estimates created by small sample sizes.

## Target-Encoding Leakage

Target encoding creates one of the most important leakage risks in categorical preprocessing.

Suppose the validation set contains customer segments and churn outcomes. If the target encoder is fitted using both training and validation targets, the validation outcomes influence the feature values used to evaluate the model.

That violates the separation between training information and evaluation information.

The safe workflow is:

`training categories + training targets -> fit encoder`

followed by:

`validation categories -> transform using fitted encoder`

The validation targets are not supplied to the encoder.

The Python program contains an explicit comparison between safe and leaky target encoding. The JavaScript implementation performs the same conceptual comparison. The C++ feature engine maintains a training-fitted `TargetEncoder` and applies it to validation customers without fitting again.

An unknown validation category receives the training global mean rather than a statistic calculated from its validation outcomes.

## Cross-Fitted Target Encoding

There is a second leakage problem that can occur entirely inside the training data.

If an encoder calculates a category statistic using every training row and then gives each of those same rows that statistic as a feature, the row's own target can contribute to its encoded value.

This can make the encoded training feature too closely related to the target.

Cross-fitting addresses this by partitioning training data into folds.

For each fold:

- the encoder is fitted using the other folds;
- the held-out fold is transformed using that encoder;
- the resulting values become out-of-fold encodings for those rows.

Every training row therefore receives an encoding produced without using its own target directly.

The Python and JavaScript implementations demonstrate this pattern using explicit validation folds.

For production machine-learning pipelines, cross-fitting must be coordinated carefully with the model's cross-validation strategy so that preprocessing remains inside the appropriate training boundaries.

## The Four Representations Compared

| Property | Label | One-hot | Ordinal | Target |
|---|---|---|---|---|
| Uses target during fitting | No | No | No | Yes |
| Preserves nominal independence | No, numerically | Yes | Not intended for nominal data | Produces a target statistic |
| Represents explicit order | Not necessarily | No | Yes | No |
| Feature width | One column | Number of categories | One column | Usually one column |
| Handles high cardinality efficiently | Compact | Potentially expensive | Compact | Compact |
| Leakage risk | Low | Low | Low | High if fitted incorrectly |
| Requires domain order | No | No | Yes | No |
| Useful for target-derived signal | No | No | No | Yes |

The table describes typical behavior rather than an absolute rule for every machine-learning algorithm.

## Python Implementation

The Python program implements four independent encoder classes.

`LabelEncoder` builds deterministic category-to-integer mappings and supports unknown values.

`OneHotEncoder` creates binary vectors and supports an optional reference category through `drop_first`.

`OrdinalEncoder` requires an explicit category order. It does not infer ordinal semantics from category names.

`TargetEncoder` stores category counts and raw target means, applies smoothing, and falls back to the training global mean for unseen categories.

The script also contains a small customer-churn dataset with:

- `region` as a nominal feature;
- `plan_level` as an ordinal feature;
- `acquisition_channel` as a nominal feature;
- `customer_segment` as a target-encoded feature;
- `churn` as the binary target.

The end-to-end feature builder intentionally combines different encodings according to the semantics of each feature rather than applying one encoding strategy to every column.

The Python implementation also demonstrates missing values, unknown values, mismatched feature/target lengths, target validation, category cardinality, smoothing, leakage, cross-fitting, and round-trip label encoding.

## JavaScript Implementation

The JavaScript implementation models the same encoding concepts from an application-oriented perspective rather than simply translating the Python classes.

`Map` is used for encoder state, making category mappings explicit and suitable for JavaScript application workflows.

The `EncodingWorkflow` class adds event-driven behavior around fitting and transformation. It emits `fit:start`, `fit:complete`, `transform:complete`, and `error` events.

This is useful when preprocessing is part of a larger JavaScript application or service in which encoding stages need to be observable without coupling encoder logic to a particular interface.

The workflow also uses asynchronous execution through `async` and `await`. The encoding operations themselves are synchronous, but the workflow structure reflects the fact that production JavaScript applications frequently combine preprocessing with asynchronous data retrieval or model-serving operations.

The JavaScript implementation includes unknown-category behavior, validation failures, smoothed target encoding, cross-fitted encoding, and a direct leakage demonstration.

For client-side applications, target-derived statistics deserve additional security consideration. Sensitive target information should not be exposed to an untrusted browser merely because the browser can technically perform the transformation.

## C++ Case Study

The C++ program models a customer-risk feature engine.

A `Customer` contains:

- `region`;
- `plan`;
- `acquisitionChannel`;
- `segment`;
- `churn`.

The `GovernanceFeatureEngine` owns four feature encoders:

- `OneHotEncoder` for region;
- `OrdinalEncoder` for subscription plan;
- `OneHotEncoder` for acquisition channel;
- `TargetEncoder` for customer segment.

A separate `LabelEncoder` demonstrates compact categorical identifiers.

The architecture separates the fitting phase from the transformation phase. Training customers are used to establish category mappings and target statistics. Validation customers are subsequently transformed using that already-fitted state.

The C++ target encoder uses the same smoothing relationship as the Python and JavaScript versions. It records the category count and raw target mean, then combines those statistics with the global training target mean.

The feature engine also demonstrates a realistic production condition: validation data contains `Central` as a previously unseen region, `Social` as a previously unseen acquisition channel, and `New Segment` as an unseen customer segment.

The nominal encoders handle these categories without modifying the learned training schema. The target encoder uses the training global mean for an unseen segment.

## Unknown Categories and Schema Stability

A production encoder should not refit itself simply because an unseen value arrives.

Refitting at inference time can change:

- feature ordering;
- numerical mappings;
- one-hot vector width;
- target statistics;
- downstream model expectations.

For example, a model trained with:

`region__East`, `region__North`, `region__South`, `region__West`

expects that exact feature schema.

Adding a new region and regenerating the one-hot mapping dynamically could shift columns or create a dimensionality mismatch.

The safer architecture is to persist the fitted preprocessing state and use a defined unknown-category policy during inference.

## Missing Values

Missing categorical values are a distinct data condition and should not automatically be treated as a legitimate business category.

The implementations normalize missing or empty values to an explicit internal token where appropriate.

This allows the encoder to distinguish:

- a genuinely observed category;
- an absent value;
- an unseen value at inference time.

Whether missingness should receive its own one-hot category, be imputed before encoding, or be handled through a separate missingness feature depends on the data-generating process and model requirements.

## Choosing an Encoding

A nominal feature such as geographic region usually needs a representation that does not invent an ordering. One-hot encoding provides that separation directly.

An ordered feature such as subscription tier can use ordinal encoding when the order is explicitly defined and meaningful.

Label encoding can be appropriate when the integer is an identifier rather than a claim that the categories have numerical magnitude.

Target encoding is useful when category-specific target information is valuable and the category cardinality makes one-hot expansion expensive. Its supervised nature requires strict separation between fitting data and evaluation data.

The decision should therefore be based on:

- semantic meaning;
- cardinality;
- model family;
- expected unseen categories;
- leakage risk;
- feature dimensionality;
- production schema requirements.

## Performance Considerations

Label and ordinal encoding generally produce a single value per categorical feature, so their memory footprint is compact.

One-hot encoding creates a number of columns proportional to the learned category count. Sparse representations are often important when the number of categories is large and each observation activates only a small number of columns.

Target encoding also produces a compact representation, but fitting requires grouping observations by category and calculating target statistics.

The practical cost of target encoding is therefore not just computational. The main engineering cost is maintaining correct statistical boundaries.

The implementations use maps and dictionaries for category lookup. In larger systems, these structures should be selected according to the language runtime, expected cardinality, serialization requirements, and inference latency constraints.

## Common Failure Modes

### Treating label encoding as a universal categorical solution

Assigning `0`, `1`, and `2` to nominal categories can introduce artificial order into algorithms that interpret the numbers numerically.

The issue is not the integer representation itself. The issue is the meaning assigned to those integers by the downstream model.

### Inferring ordinal order from spelling

Alphabetical order is not semantic order.

If the domain says:

`Basic < Standard < Premium`

that order should be supplied explicitly.

### Fitting target encoding on evaluation data

Using evaluation targets to construct evaluation features contaminates the evaluation process.

The encoder must be fitted using training outcomes only.

### Ignoring rare categories in target encoding

A category with one observation can produce an extreme raw mean. Smoothing reduces this instability by incorporating the global target rate.

### Rebuilding category mappings during inference

A model and its preprocessing state form a coupled artifact. Changing the category mapping after training can make previously learned model coefficients refer to the wrong columns or values.

### Assuming unknown categories cannot happen

Production data evolves. New regions, product tiers, channels, merchants, customers, and other categorical values can appear after model training.

Unknown-category behavior should therefore be an explicit design decision.

## Security and Data Governance

Categorical values can originate from external users, files, databases, APIs, or event streams. They should be treated as untrusted data.

The encoder implementations do not dynamically execute category values, generate source code from them, or use them as shell commands.

Target encoding requires additional governance because it stores statistics derived from outcomes. Depending on the application, category-level target statistics may reveal information about the target population.

Training artifacts containing target encodings should therefore be handled with the same access controls and data-governance requirements applicable to the underlying training data.

In browser applications, sensitive target-derived statistics should not automatically be transferred to clients simply to perform preprocessing locally.

## Production Workflow

A robust categorical preprocessing workflow separates fitting from inference.

The training stage establishes the encoder state:

`raw training categories -> fitted encoder -> serialized preprocessing state`

The model-training stage then consumes the encoded training representation.

The inference stage reuses the same encoder:

`new categorical value -> fitted encoder -> model-compatible numerical representation`

The encoder state should be versioned with the model because a model trained against one feature schema may not be compatible with a later independently generated encoding schema.

For target encoding, the training workflow must also specify how cross-fitting, validation folds, smoothing, and unseen categories are handled.

The resulting preprocessing contract is part of the deployed model rather than an incidental transformation performed before prediction.

## Practical Interpretation

The four encoding techniques answer different questions.

**Label encoding** asks: “What compact integer identifier represents this category?”

**One-hot encoding** asks: “Which independent nominal category is active?”

**Ordinal encoding** asks: “Where does this category sit in a known domain-defined order?”

**Target encoding** asks: “What target-derived statistical signal has been learned for this category from the permitted training data?”

Those questions should not be treated as interchangeable. The correct encoding is determined by the semantics of the categorical variable, the model's assumptions, the category cardinality, and the statistical boundary between training and evaluation data.
