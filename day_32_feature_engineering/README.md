# Feature Engineering

Feature engineering converts raw observations into structured variables that expose useful patterns to a statistical or machine-learning system. The central challenge is not producing more columns. It is creating features whose meaning is valid, whose computation matches the prediction moment, and whose behavior remains reliable when new data arrives.

This repository focuses on four related forms of feature engineering:

- **Feature creation** derives new variables from existing observations.
- **Feature transformation** changes the representation of a variable so that its distribution or encoding is more suitable for downstream modeling.
- **Feature aggregation** summarizes multiple observations into a single entity-level or time-window-level signal.
- **Domain features** encode relationships that have a meaningful interpretation in a particular business or technical domain.

The implementations use customer and transaction data because this setting naturally demonstrates static attributes, temporal behavior, categorical variables, numerical transformations, aggregation windows, ratios, risk signals, and leakage prevention.

---

## Technical Scope

The examples deliberately distinguish feature-engineering mechanisms rather than treating every generated column as the same type of feature.

A raw field such as `annual_income` describes an observation directly. A transformation such as `log_income` changes its numerical representation. An aggregation such as `spend_30d` summarizes many transactions for one customer. A domain feature such as `monthly_spend_to_credit_limit` expresses a relationship that becomes meaningful because the data represents financial behavior.

The same distinction is important when designing a production feature pipeline. A feature catalog should identify not only the feature name but also its source fields, transformation logic, temporal assumptions, and domain meaning.

---

## Feature Creation

Feature creation combines existing information into a new representation.

The Python implementation creates fields such as:

- `age_band` converts a continuous age value into semantically meaningful intervals.
- `tenure_days` represents the elapsed time between signup and the prediction date.
- `is_business_hour` and `is_night` derive behavioral information from transaction timestamps.
- `high_value_purchase_flag` converts a transaction-size condition into a binary behavioral indicator.
- `active_customer_flag` represents recent transaction activity as an explicit feature.

Creation is useful when the raw field does not directly expose the relationship needed by the model. For example, a raw timestamp contains the hour, day, and month, but a prediction system may benefit from explicitly representing whether an event occurred during normal business hours.

A created feature should have a clear semantic definition. A feature such as `is_night` is meaningful because the rule is explicit: timestamps before 06:00 or at or after 22:00 are classified as nighttime in the example.

Feature creation should not silently encode the future. If a feature is calculated for a prediction made on April 15, information that became available on April 16 cannot be part of that feature.

---

## Feature Transformation

Transformation changes the representation of an existing feature without necessarily changing the underlying observation.

### Log transformation

Financial amounts and income frequently have right-skewed distributions. A small number of observations can be much larger than the majority. The implementations use `log1p(x)` rather than `log(x)` so that zero remains valid.

The Python script applies this to annual income:

`log_income = log(1 + annual_income)`

The C++ case study implements the same mathematical mechanism through `std::log1p`.

The transformation does not claim that logarithmic scaling is always appropriate. Its usefulness depends on the distribution, model family, interpretation requirements, and downstream preprocessing strategy.

### Standardization

The Python implementation contains a small `StandardScaler` that calculates a mean and standard deviation from training observations and then produces standardized values:

`z = (x - mean) / standard_deviation`

The important engineering rule is that the statistics are learned from the training data. Test observations are transformed using those already-fitted statistics.

Calculating the mean and standard deviation from the complete dataset before splitting training and test observations allows information from the test population to influence the transformation. That is a form of data leakage even though the target variable itself may never have been used.

### Winsorization

The Python implementation also contains a reusable winsorization function. It clips values to supplied lower and upper bounds.

This can reduce the influence of extreme observations, but the bounds themselves can become a source of leakage if they are estimated from data that includes the test period. In production, clipping thresholds should have an explicit fitting policy.

### Categorical encoding

The Python and JavaScript implementations represent categorical values through one-hot encoding.

For a `region` field containing `North`, `South`, `East`, and `West`, the representation can contain columns such as:

`region__North`

`region__South`

`region__East`

`region__West`

An important edge case is an unseen category. The implementations explicitly create an `unknown` representation rather than failing unexpectedly when inference encounters a category that was absent from the fitting vocabulary.

---

## Feature Aggregation

Aggregation changes the unit of analysis.

A transaction is an event-level observation. A customer-level prediction may require information about many transactions. Aggregation converts those events into customer-level features.

The example calculates a trailing-window transaction set and derives:

- `transactions_30d`
- `spend_30d`
- `average_transaction_30d`
- `max_transaction_30d`
- `risky_transaction_count_30d`
- `risky_spend_30d`
- `unique_categories_30d`
- `category_spend_entropy_30d`
- `card_usage_ratio`
- `largest_purchase_share`

These features answer different behavioral questions.

`spend_30d` measures volume.

`average_transaction_30d` measures typical transaction size.

`max_transaction_30d` identifies the largest recent purchase.

`unique_categories_30d` measures category breadth.

`card_usage_ratio` measures payment-method behavior.

`largest_purchase_share` identifies whether recent spending is concentrated in one unusually large purchase.

The aggregation window is part of the feature definition. `spend_30d` and `spend_90d` are not interchangeable because they represent different temporal horizons.

---

## Temporal Aggregation and Leakage Prevention

Time-aware feature engineering requires a strict definition of the prediction timestamp.

Suppose a model produces a prediction at:

`2025-04-15 23:59`

A transaction occurring on:

`2025-04-16 10:00`

cannot be used in the feature vector for the April 15 prediction.

The Python implementation enforces this with a transaction timestamp condition. The JavaScript implementation uses `isWithinWindow`, while the C++ engine filters events before updating aggregation state.

The relevant relationship is:

`window_start <= transaction_time <= prediction_time`

The feature window also excludes transactions that are too old.

This distinction is critical because a feature can look mathematically correct while being operationally invalid. A future transaction included in `spend_30d` creates look-ahead leakage and can produce unrealistically strong model performance during offline evaluation.

The same principle applies to derived values. If a future transaction is first aggregated and then used to calculate a ratio, every downstream feature inherits the leakage.

---

## Domain Features

Domain features encode relationships whose meaning depends on the problem domain.

The financial customer example creates:

`monthly_spend_to_credit_limit`

This relates recent spending to credit capacity:

`spend_30d / credit_limit`

It also creates:

`annualized_spend_to_income`

which scales the recent spending observation against annual income:

`(spend_30d × 12) / annual_income`

Another domain feature is:

`merchant_risk_intensity`

which represents the fraction of recent spending associated with transactions whose merchant-risk value is at or above the example threshold.

These features are not simply renamed arithmetic. Their usefulness comes from the interpretation of the underlying financial process.

A domain feature should be reviewed with subject-matter experts because its formula can encode assumptions about behavior. A ratio can also become unstable when its denominator is zero or extremely small. The implementations therefore use safe-ratio behavior.

---

## Feature Lineage

Feature lineage describes where a generated feature came from and how it was calculated.

The Python script represents lineage through `FeatureSpec`, while the C++ program uses `FeatureDefinition`.

Examples include:

| Feature | Category | Source fields | Temporal |
|---|---|---|---|
| `log_income` | Transformation | `annual_income` | No |
| `spend_30d` | Aggregation | amount, customer, timestamp | Yes |
| `average_transaction_30d` | Aggregation | amount, customer, timestamp | Yes |
| `monthly_spend_to_credit_limit` | Domain | spend, credit limit | Yes |
| `merchant_risk_intensity` | Domain | risky spend, total spend | Yes |

Lineage is valuable for debugging, model governance, reproducibility, and production maintenance.

If a model begins producing unexpected predictions, knowing that `merchant_risk_intensity` depends on `risky_spend_30d` and `spend_30d` provides a concrete path for investigating the issue.

---

## Python Implementation

The Python implementation provides the most complete end-to-end feature-engineering pipeline.

It starts with typed customer and transaction records and then separates the major stages:

`create_customer_base_features` handles customer-level creation and transformation.

`aggregate_transactions` performs temporal customer-level aggregation.

`build_domain_features` creates financial relationships from the base and aggregated features.

`build_prediction_snapshot` combines these stages while enforcing the prediction-time boundary.

The `StandardScaler` demonstrates train-only fitting. The `OneHotEncoder` demonstrates deterministic categorical vocabulary creation and explicit unknown-category handling.

The script also validates generated feature rows and exports:

- a CSV feature matrix
- a JSON feature matrix
- a JSON feature catalog

The test suite checks temporal leakage protection, empty activity, pre-fit transformation failure, unknown categorical values, invalid windows, and domain-feature creation.

The script uses only the Python standard library and is executable without third-party packages.

---

## JavaScript Implementation

The JavaScript implementation takes a different architectural approach.

Instead of translating the Python classes directly, it models feature computation as an event-driven process.

`TransactionStream` provides an asynchronous publication mechanism. `CustomerAggregator` maintains incremental customer-level state using JavaScript `Map` objects.

This structure is relevant to online or streaming feature systems because new transactions can update aggregate state without rebuilding the entire transaction history after every event.

The JavaScript implementation also uses:

- `async` and `await` for asynchronous event processing
- immutable configuration through `Object.freeze`
- `Map` for keyed aggregation state
- `Set` for categorical vocabularies and listeners
- explicit validation for numeric and timestamp values
- unknown-category handling for one-hot encoding
- deterministic JSON serialization

The streaming implementation and batch snapshot are tested against one another so that both representations produce the same recent-spending result.

This is an important production consideration: online and offline feature calculations should agree on their definitions. If a training pipeline computes a feature differently from the real-time system, the model can experience training-serving skew.

---

## C++ Case Study

The C++ program models a typed feature-engineering engine for a financial analytics platform.

The central class is `FeatureEngine`.

It owns:

- validated customer records
- aggregation state for each customer
- transaction IDs already processed
- a prediction timestamp
- a feature-window duration

The transaction ingestion path performs validation before updating feature state.

Unknown customers are rejected because silently creating a new customer from an untrusted transaction could corrupt entity-level aggregation.

Duplicate transaction IDs are rejected because counting the same event twice would inflate behavioral features.

Future transactions are ignored because they were unavailable at prediction time.

Transactions outside the configured window are also ignored.

The resulting `FeatureRow` contains static attributes, transformed values, aggregate features, and financial domain features.

The C++ design separates aggregation state from final feature rows. This makes it possible to maintain efficient mutable state while exposing a stable prediction snapshot.

---

## C++ Data Structures and Complexity

The C++ engine uses `std::unordered_map` for customer lookup and aggregation state.

Customer lookup is expected O(1), and each accepted transaction updates a bounded number of hash-map entries. For `T` transactions in the prediction window, the primary aggregation stage is expected O(T).

The category and payment-method maps also provide expected constant-time updates.

The final feature matrix is sorted by customer ID for deterministic output. If there are `C` customers, this sorting step costs O(C log C).

Memory usage is approximately O(C + K), where `C` represents customer state and `K` represents the distinct customer/category/payment combinations stored in the aggregation maps.

The Python and JavaScript versions use comparable dictionary or `Map` structures for aggregation.

---

## Validation and Failure Handling

Feature engineering is a data-processing boundary, so invalid input must be treated as an engineering failure rather than silently converted into a plausible value.

The implementations validate:

- customer identifiers
- positive customer age
- positive income
- positive credit limits
- transaction identifiers
- transaction amounts
- merchant-risk ranges
- timestamp validity
- known customer references
- duplicate transaction identifiers
- positive aggregation-window sizes
- missing numeric values
- NaN-like invalid numeric states
- transformation-before-fitting errors

Zero denominators are handled explicitly in ratio features.

For example, a customer with no recent transactions receives zero for `average_transaction_30d` rather than causing a division-by-zero failure.

This behavior is a design choice and should be documented for a production model because zero can mean "no activity" rather than "true numerical ratio equals zero."

---

## Train, Validation, and Test Boundaries

Feature engineering must respect dataset boundaries.

Transformations that learn parameters include:

- means
- standard deviations
- medians
- clipping thresholds
- category vocabularies
- target-derived statistics

Those parameters should be fitted using the appropriate training partition.

The Python `StandardScaler` demonstrates this explicitly. Its `fit` method learns statistics, while `transform` applies them.

A common mistake is to transform the complete dataset first and split it afterward. Even when the transformation does not use the target, statistics from validation or test observations can influence the representation used for training.

Temporal datasets require an additional boundary: the feature calculation must only use information available at the prediction timestamp.

---

## Aggregation Grain

The grain of a feature identifies the entity represented by one row.

The transaction table has transaction grain:

`one row = one transaction`

The feature matrix has customer grain:

`one row = one customer at a prediction time`

A feature pipeline must preserve this distinction.

Joining transaction-level data directly to customer-level labels can accidentally duplicate customer rows. Aggregating transactions before constructing the customer feature matrix prevents this particular duplication problem.

A production system should document the grain of every feature and the keys required to compute it.

---

## Feature Interactions

Feature interactions represent relationships between variables.

The financial example contains interactions such as:

`spend_30d / credit_limit`

and:

`spend_30d × 12 / annual_income`

The first relates behavioral volume to available credit capacity. The second relates a recent spending rate to income.

These relationships can sometimes be useful even when the raw variables already exist separately.

Interaction design should be controlled rather than generating every possible mathematical combination. Exhaustive interaction generation can create very large feature spaces, increase computation, complicate interpretation, and amplify noise.

The strongest interactions usually have a defensible relationship to the domain or a clear modeling purpose.

---

## Categorical Feature Considerations

Categorical variables require a representation compatible with the downstream model.

The examples use one-hot encoding because it provides a transparent representation for low-cardinality fields such as region and acquisition channel.

An important inference-time case is a category that did not appear during training.

The implementations represent that case with an explicit unknown indicator. This avoids silently assigning the observation to an unrelated known category.

High-cardinality categorical fields require different considerations. Customer IDs, merchant IDs, product IDs, and similar identifiers can create extremely wide matrices or encourage memorization. They should not automatically be one-hot encoded simply because they are categorical.

---

## Missing Values

Missingness can itself contain information.

For numerical features, the Python example calculates training-derived medians and uses them for missing-value replacement.

The imputation statistic must be fitted on the correct training population.

Missing categorical values can be mapped to an explicit category such as `unknown`, provided that this meaning is appropriate for the feature.

A production pipeline should distinguish:

- genuinely missing observations
- structurally absent observations
- zero-valued observations
- invalid observations
- values outside the domain

These states do not necessarily have the same meaning.

---

## Temporal Features

Transaction timestamps provide several possible feature dimensions:

- hour
- day of week
- month
- weekend status
- business-hour status
- nighttime status

The JavaScript implementation exposes these through `createTemporalFeatures`.

The C++ case study focuses more heavily on temporal filtering because the main system requirement is a prediction-time feature window.

Calendar decomposition can be useful, but it should be driven by the behavior being modeled. Creating every possible timestamp component without a meaningful relationship to the problem can increase dimensionality without adding useful information.

---

## Common Feature-Engineering Failure Modes

### Target leakage

A feature directly or indirectly contains information about the future target.

The most important defense in the examples is strict prediction-time filtering.

### Temporal leakage

A feature uses observations that occurred after the prediction timestamp.

A future transaction must not contribute to a historical spending feature.

### Transformation leakage

A scaler, imputer, clipping threshold, or vocabulary is fitted using validation or test observations.

The Python scaler demonstrates the correct separation between fitting and transformation.

### Double counting

The same event enters an aggregation more than once.

The C++ engine rejects duplicate transaction IDs to make this failure explicit.

### Wrong entity grain

A transaction-level table is joined into a customer-level dataset without aggregation, causing customers to appear multiple times.

### Unstable ratios

A denominator can be zero or extremely small.

The examples use safe ratio handling, but production systems may need domain-specific thresholds and additional validation.

### Training-serving skew

Offline feature generation and production feature generation use different definitions.

The JavaScript implementation's batch and streaming paths illustrate why shared semantics matter.

### Overengineering

Large numbers of weak or redundant features can increase storage, computation, monitoring requirements, and model complexity.

Feature quantity should not be treated as a quality metric.

---

## Debugging Feature Pipelines

A feature value should be traceable back to its source observations.

For an unexpected `spend_30d` value, debugging should inspect:

- the customer identifier
- prediction timestamp
- aggregation window
- included transaction IDs
- transaction timestamps
- transaction amounts
- duplicate detection
- filtering boundaries

For a domain ratio, the numerator and denominator should be inspectable independently.

For transformed features, the fitted parameters should be persisted and versioned.

The feature catalog in the examples provides a compact semantic representation of these relationships.

---

## Production Considerations

A production feature-engineering system needs stronger controls than a self-contained educational program.

Feature definitions should be versioned so that a model trained with one definition can be reproduced later.

Temporal features should have an explicit timezone policy. The JavaScript example uses UTC timestamp operations to avoid silently mixing local and UTC interpretation.

Feature computation should be deterministic when the same source data and prediction timestamp are supplied.

Pipelines should monitor feature distributions because a feature can remain syntactically valid while its statistical behavior changes significantly.

Operational monitoring should distinguish data-quality failures from legitimate distribution changes.

Feature storage should preserve the feature definition, source version, computation timestamp, and model compatibility requirements where reproducibility matters.

Sensitive financial information should also be protected through appropriate access controls, logging policies, retention policies, and encryption requirements.

---

## Practical Relationship Between the Four Feature Types

The four categories form a pipeline rather than four isolated techniques.

A raw customer record may contain annual income.

A transformation can produce `log_income`.

A transaction history can be aggregated into `spend_30d`.

The domain relationship between `spend_30d` and `credit_limit` can produce `monthly_spend_to_credit_limit`.

The resulting feature chain can therefore be represented as:

`raw observation → transformed representation → temporal aggregation → domain relationship`

Each stage has different validation and leakage risks.

Transformation requires attention to learned parameters.

Aggregation requires attention to entity grain and time windows.

Domain features require attention to business meaning and denominator behavior.

Feature creation requires attention to whether the new variable actually represents information available at prediction time.

---

## Testing Strategy

The implementations include tests for the failure modes that are particularly important for feature engineering.

The Python tests verify that future transactions do not alter historical features, that empty activity is safe, that transformations require fitting, that unseen categories are handled, and that invalid windows fail.

The JavaScript tests compare batch and streaming feature results and explicitly verify future-event exclusion.

The C++ tests verify future-event exclusion, duplicate-event rejection, unknown-customer rejection, and safe behavior for customers without recent transactions.

These tests are more useful than checking only whether a function returns a value because feature-engineering failures often produce plausible-looking numbers.

---

## Limitations of the Demonstrations

The datasets are intentionally small and deterministic.

The examples do not implement a complete machine-learning training algorithm, distributed feature store, database-backed feature registry, production scheduler, or real-time event broker.

The financial domain features are illustrative rather than financial advice or a validated credit-risk methodology.

The aggregation thresholds and feature windows are example engineering choices. A real system would establish them through domain requirements, data analysis, validation, and model-governance processes.

The C++ time representation is intentionally simplified to keep the case study focused on feature computation. Production systems should use an explicit date-time representation with documented timezone and calendar semantics.

---

## Implementation Files

The Python program provides the broadest standalone feature-engineering pipeline, including transformation fitting, categorical encoding, aggregation, domain features, validation, export, and tests.

The JavaScript program emphasizes asynchronous event-driven processing and incremental aggregation, illustrating how feature computation can be structured differently in a runtime commonly used for services and event-driven applications.

The C++ program emphasizes typed domain modeling, deterministic aggregation, validation boundaries, feature lineage, and computational characteristics in a systems-oriented implementation.

Together, the implementations demonstrate that feature engineering is not merely the creation of additional columns. It is a controlled transformation of raw information into features with explicit semantics, temporal validity, data lineage, validation behavior, and operational assumptions.
