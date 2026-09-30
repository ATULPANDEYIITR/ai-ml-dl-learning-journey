# Data Cleaning: Missing Values, Duplicates, Outliers, and Inconsistent Values

## Introduction

Data cleaning converts raw observations into a representation that satisfies defined quality rules without destroying information that may still be analytically useful.

This project focuses on four different forms of data-quality problems:

| Data-quality problem | Core question | Typical treatment |
| --- | --- | --- |
| Missing values | Is a required observation absent? | Impute, retain as missing, or reject according to meaning |
| Duplicates | Does the dataset contain the same business event more than once? | Detect using an appropriate identity and retain the correct record |
| Outliers | Is an observation statistically unusual? | Flag, investigate, transform, cap, or remove only with justification |
| Inconsistent values | Do equivalent observations use different representations? | Normalize them to controlled canonical values |

These categories are related but should not be treated as interchangeable. A blank amount is a completeness problem. Two identical transactions are a duplication problem. A very large transaction is an outlier candidate. Values such as `IND`, `India`, and `IN` represent an inconsistency when the system requires one canonical country representation.

The implementations in this project use a transaction dataset to demonstrate how these problems can be detected, transformed, validated, and audited.

---

## Data Quality Model

A useful cleaning workflow separates observation, transformation, and validation.

The raw transaction enters the pipeline in an uncontrolled representation:

`raw transaction -> profiling -> normalization -> duplicate detection -> missing-value handling -> outlier analysis -> validation -> cleaned dataset`

The ordering is deliberate.

Inconsistent representations should generally be normalized before duplicate comparison. For example, `IND`, `India`, and `IN` should not be treated as different countries when the domain definition says they all mean `India`.

Numeric formatting should also be normalized before statistical analysis. A value such as `13,250` is a presentation representation of the numeric value `13250`, not a different transaction amount.

Outlier detection should happen after valid numeric values have been established. Final validation should occur after transformations so the output can be checked against explicit data-quality constraints.

---

## Missing Values

A missing value means that an observation expected by the dataset is absent or unusable.

The important distinction is between a missing value and a valid zero. A transaction amount of `0` is a numeric observation. An empty amount does not provide an amount. Treating the two as equivalent can introduce serious analytical errors.

Missingness can arise from several causes:

- A source system did not collect the value.
- A user skipped an optional field.
- A transformation removed the value.
- A source integration failed.
- A field was not applicable to a particular record.
- A value exists in another system but was not joined correctly.

The correct treatment depends on the meaning of the field.

### Numeric imputation

The Python, JavaScript, and C++ implementations use median imputation for a missing transaction amount.

The median is useful for transaction data because monetary distributions are frequently skewed. A single very large transaction can substantially change the arithmetic mean, while the median is more resistant to extreme observations.

For ordered values:

`x(1) <= x(2) <= ... <= x(n)`

the median is the central observation for an odd number of values. For an even number of observations, it is the mean of the two central values.

The implementations calculate the replacement from valid amounts rather than from missing values.

### Why dates are treated differently

The programs do not fabricate missing transaction dates.

A transaction date can affect monthly revenue, daily activity, financial reporting, retention calculations, and time-series analysis. Unless there is a reliable source from which the date can be reconstructed, replacing it with an arbitrary date would create false information.

The programs therefore retain a missing date and record the decision in the audit trail.

### Missing-value risks

Imputation can introduce artificial similarity. If a large percentage of customers receive exactly the same replacement amount, downstream models may interpret the repeated value as genuine behavior.

A production pipeline should therefore monitor:

- Missing rate by field
- Missing rate by source system
- Missing rate over time
- Percentage of records imputed
- Distribution before and after imputation
- Whether missingness itself has predictive or operational meaning

---

## Duplicate Records

A duplicate is not necessarily a byte-for-byte identical row.

Two records may represent the same business event even when formatting differs. Conversely, two records with similar customer information may be legitimate separate transactions.

Duplicate detection therefore requires a business identity.

The implementations use a combination of transaction identifier, normalized email, and transaction date to create a comparison key.

This is different from simply comparing the entire serialized row.

A robust production system should distinguish:

- Exact duplicates
- Business duplicates
- Near duplicates
- Legitimate repeated transactions

For example, two purchases by the same customer on the same day are not automatically duplicates. If the transactions have different transaction identifiers and supporting evidence, they may represent two genuine purchases.

### Retention policy

The examples retain the first occurrence of a duplicate business identity.

Real systems may require a more sophisticated rule, such as retaining:

- The record with the latest source timestamp
- The record from the most authoritative source
- The record with the greatest completeness
- The record that passed upstream validation
- A merged record containing complementary fields

Deduplication without a retention policy can accidentally remove the most useful representation.

---

## Outliers

An outlier is an observation that is unusually distant from the central distribution.

An outlier is not automatically an error.

The transaction amount `1500000` may be suspicious when most transactions are around a few thousand or tens of thousands, but it could still be a legitimate high-value purchase.

This distinction is important:

`unusual != incorrect`

The implementations therefore flag statistical outliers rather than automatically deleting them.

### IQR method

The examples use the interquartile range.

The first quartile is represented as `Q1`, and the third quartile as `Q3`.

The interquartile range is:

`IQR = Q3 - Q1`

A conventional detection rule is:

`lower bound = Q1 - 1.5 × IQR`

`upper bound = Q3 + 1.5 × IQR`

An observation below the lower bound or above the upper bound is flagged as an outlier candidate.

The multiplier can be changed according to the analytical purpose. A smaller multiplier identifies more observations as unusual, while a larger multiplier is less sensitive.

### Why outliers are not automatically deleted

Removing an outlier changes the dataset's distribution.

For financial, operational, scientific, or security data, an extreme observation can contain the most important information. It might represent:

- A genuine high-value customer
- A bulk purchase
- A fraud event
- A sensor malfunction
- A data-entry error
- A system integration failure

The correct response depends on the domain and available evidence.

A production workflow can therefore separate:

`detection -> investigation -> classification -> treatment`

rather than treating detection as proof of invalidity.

---

## Inconsistent Values

Inconsistent values represent the same conceptual value using different representations.

The dataset deliberately includes examples such as:

- `IND`, `IN`, `India`, and `INDIA`
- `UP`, `U.P.`, and `uttar pradesh`
- `CA` and `California`
- `Electronics`, `electronics`, and differently formatted versions
- `Grocery` and `Groceries`
- `13,250` and `13250`
- Multiple date formats

The correct treatment is domain-aware canonicalization.

### Canonical values

A canonical value is a representation selected as the standard storage form.

For country values, the examples map recognized aliases to values such as `India` and `United States`.

For categories, the examples establish:

`electronics -> Electronics`

`home and kitchen -> Home & Kitchen`

`groceries -> Grocery`

This is safer than arbitrary string manipulation because the mapping expresses business meaning.

### Why unrestricted fuzzy matching is risky

Fuzzy matching can be useful for discovering possible similarities, but it should not automatically convert every similar string into the same value.

For example, two customer names that differ by one character might represent:

- A typographical error
- Two different people
- A transliteration difference
- A genuine naming variation

Canonicalization should therefore use controlled domain rules when the set of valid values is known.

---

## Python Implementation

The Python implementation is a complete standard-library data-cleaning pipeline.

The `CleaningIssue` structure represents one transformation or data-quality finding. The audit model records the original value, cleaned value, action, and reason.

The `profile_dataset` function measures missingness and uniqueness before transformations. This establishes a baseline rather than relying only on the final cleaned dataset.

The missing-value implementation distinguishes between numeric and temporal fields. Missing amounts use median imputation, while missing dates remain explicitly missing.

The duplicate implementation builds a business-oriented identity key and removes subsequent records with the same identity.

The inconsistency implementation uses explicit country, state, and category mappings. Email values are normalized to lowercase, numeric amounts are converted into numeric values, and supported date formats are converted to ISO `YYYY-MM-DD`.

The outlier implementation uses the IQR method and records outliers as audit events with a `flag_only` action.

The final validator checks:

- Unique transaction identifiers
- Valid email structure
- Finite numeric amounts
- Non-negative amounts
- Supported canonical countries
- Supported canonical categories
- Valid ISO dates where dates are present

The script writes two files:

`data_cleaning_output/cleaned_transactions.csv`

`data_cleaning_output/cleaning_audit.csv`

The audit file is important because a production cleaning process should be traceable. A downstream user should be able to determine why a value changed rather than seeing only the final result.

---

## JavaScript Implementation

The JavaScript implementation takes a different architectural approach.

Instead of reproducing the Python pipeline as a collection of procedural functions, it models cleaning as an event-driven engine using Node.js's `EventEmitter`.

The `CleaningEngine` owns:

- Raw records
- Audit events
- Cleaning metrics
- Transformation stages
- Validation behavior

The engine emits events such as:

`pipelineStart`

`stageComplete`

`issue`

`validation`

`pipelineComplete`

This design is useful for applications where cleaning operations need to be observed by logging, monitoring, a user interface, or another processing component.

For example, every detected issue emits an audit event containing the record identifier, field, issue category, original value, cleaned value, action, and reason.

The JavaScript implementation also demonstrates a distinction between normalization and workflow orchestration. The normalization functions transform individual values, while the engine coordinates the complete data-quality process.

The cleaned records are written to CSV and the detailed audit events are written to JSON.

This creates a useful separation between:

`cleaned operational data`

and

`explainable transformation history`

---

## C++ Case Study

The C++ program models a transaction data-governance engine.

The `Transaction` structure represents the domain object. The use of `std::optional` is significant because it distinguishes an absent amount or date from an ordinary string or numeric value.

The `DataQualityEngine` owns the complete processing lifecycle.

Its architecture separates responsibilities into methods for:

- Canonicalizing inconsistent values
- Removing duplicate business identities
- Handling missing values
- Detecting statistical outliers
- Validating the resulting dataset

The program uses `std::map` for explicit canonical mappings, `std::unordered_set` for efficient duplicate-key membership checks, `std::vector` for records and numeric analysis, and `std::optional` for nullable fields.

The business identity is converted into a deterministic key before duplicate detection.

The outlier analysis calculates quartiles and the IQR boundary. It records an outlier as an audit event rather than destroying the observation.

The final validation stage ensures that the cleaning process did not leave unsupported categories, invalid countries, duplicate transaction identifiers, invalid amounts, or non-finite numeric values.

The audit trail is exported to `data_quality_audit.csv`.

This case study illustrates how data cleaning can be designed as a governed processing component rather than as a sequence of ad hoc transformations.

---

## Relationship Between the Four Problems

The four categories often interact, but their decisions should remain distinct.

A useful relationship is:

`inconsistent representation -> canonical representation`

`canonical representation -> reliable duplicate comparison`

`missing value -> explicit missing-value policy`

`normalized numeric value -> statistical analysis`

`statistical anomaly -> investigation or controlled treatment`

Consider a value such as `13,250`.

Before numeric normalization, a statistical routine cannot safely treat the string as a number. After normalization, it becomes `13250`.

Now consider two records containing `IND` and `India`. If the values are not canonicalized first, a grouping operation could incorrectly report two countries.

The distinction matters because cleaning order can affect the quality of subsequent operations.

---

## Cleaning Audit Trail

A cleaning process should not only produce a clean dataset. It should explain what happened.

The audit records in this project capture:

- Record identity
- Field affected
- Issue type
- Original value
- New value or retained value
- Action taken
- Reason for the action

This supports debugging and data lineage.

For example, an audit event can establish that a missing amount was replaced with a median rather than silently changing the number.

A production audit system can extend this model with:

- Pipeline version
- Source system
- Source file
- Ingestion timestamp
- Operator or service identity
- Cleaning rule identifier
- Rule version
- Validation result

These additions are particularly important when cleaned data supports financial, regulatory, scientific, or operational decisions.

---

## Validation Versus Cleaning

Cleaning and validation have different responsibilities.

Cleaning changes or classifies data according to defined rules.

Validation checks whether the resulting data satisfies a contract.

For example, converting `IND` to `India` is cleaning.

Checking that the resulting country is one of the permitted canonical countries is validation.

This separation prevents a transformation function from silently accepting invalid output.

A useful production architecture is:

`source -> schema validation -> normalization -> cleaning rules -> business validation -> quality report -> storage`

The exact order can vary when a transformation requires information that initial validation would otherwise reject, but the distinction between transformation and verification should remain explicit.

---

## Edge Cases

### All values are missing

Median imputation cannot work when no valid numeric observations exist. The implementations therefore raise an error rather than inventing a replacement.

A production pipeline could instead use a trusted reference value, a segment-level statistic, or a manually defined policy.

### Too few observations for IQR analysis

Quartile-based outlier detection is unreliable on extremely small samples. The examples therefore avoid IQR classification when fewer than four valid observations are available.

A production system should select statistical techniques according to sample size and distribution.

### Zero IQR

If `Q1 == Q3`, then the IQR is zero. Dividing the distribution using ordinary IQR boundaries provides no useful separation.

This case requires a separate policy rather than forcing a statistical classification.

### Invalid date formats

The implementations support explicitly recognized formats and do not silently reinterpret arbitrary strings.

This is important because date ambiguity can change the meaning of an observation. A string such as `03/04/2026` can be interpreted differently under day-first and month-first conventions.

### Negative transaction amounts

Negative amounts may represent refunds or reversals in some systems and invalid data in others.

The sample validation treats negative amounts as invalid because the modeled transaction domain represents positive purchase amounts. A real financial schema would need an explicit transaction-type model before applying such a restriction.

---

## Common Data-Cleaning Failure Modes

### Replacing every missing value with zero

This changes absence into a real numerical observation. It can reduce measured averages and distort distributions.

### Deleting every outlier

An outlier is a statistical classification, not proof of an invalid record. Automatic deletion can remove genuine business events.

### Deduplicating only by entire-row equality

A duplicate may contain different formatting or timestamps while representing the same business event.

### Lowercasing everything without domain rules

Lowercasing can standardize presentation but cannot resolve semantic aliases such as `UP` and `Uttar Pradesh`.

### Changing values without recording the original

Without an audit trail, data owners cannot determine why a value changed or reproduce a transformation decision.

### Applying a cleaning rule without validation

A transformation can itself introduce errors. Post-cleaning validation is therefore necessary.

### Using fuzzy matching as an automatic truth mechanism

Similarity does not establish identity. Candidate matches should be reviewed or governed by domain-specific rules.

---

## Performance Considerations

Let `n` represent the number of records and `k` the number of fields.

Field profiling is approximately `O(nk)` because each record-field combination may be inspected.

Hash-based duplicate detection is approximately `O(n)` average-case when business keys are stored in a hash set.

Sorting numeric values for median and quartile calculations requires `O(n log n)` time.

Memory consumption is primarily driven by the records, audit entries, normalized copies, and intermediate numeric arrays.

For large datasets, a production implementation may need:

- Streaming CSV processing
- Chunk-based transformations
- Database-side deduplication
- Distributed processing
- Incremental statistics
- External audit storage
- Partition-aware outlier detection

The appropriate architecture depends on data volume and latency requirements.

---

## Production Considerations

A production data-cleaning system should make its rules explicit and versioned.

A transformation such as `USA -> United States` should come from a controlled domain definition rather than an undocumented assumption in application code.

Cleaning policies should also distinguish between:

- Automatically correctable problems
- Automatically detectable but review-required problems
- Unrecoverable records

Outliers generally belong to the second category because their statistical unusualness does not establish whether they are erroneous.

Data quality should also be measured over time. A sudden increase in missing email values or an unusual number of duplicates may indicate an upstream system failure rather than a normal change in data.

The cleaning pipeline should therefore produce both the cleaned dataset and quality metrics.

---

## Practical Interpretation of the Example

The example transaction data intentionally combines all four problem classes.

A blank amount is handled as missing data and receives a median-based replacement.

The repeated `TX3009` or equivalent transaction is identified as a duplicate business record and removed.

The very large transaction amount is statistically unusual and therefore flagged as an outlier candidate without automatic deletion.

Country, state, category, email, numeric formatting, and date representations are normalized into canonical forms.

These operations produce a dataset that is more consistent and suitable for downstream analysis while preserving traceability for decisions that changed or classified the source observations.

---

## Implementation Characteristics

| Capability | Python | JavaScript | C++ |
| --- | --- | --- | --- |
| Missing-value handling | Median imputation and retained missing dates | Median imputation and event-driven audit | `std::optional` with median imputation |
| Duplicate detection | Business-key set | Event-driven business-key detection | `std::unordered_set` |
| Outlier detection | IQR calculation | IQR calculation | IQR calculation |
| Inconsistency handling | Domain mappings and format normalization | Canonical mappings and event emission | Explicit `std::map` mappings |
| Validation | Dedicated validation function | Engine validation stage | Engine validation method |
| Audit representation | Dataclass records and CSV | Objects and JSON | Structured C++ records and CSV |
| Persistence | Clean CSV plus audit CSV | Clean CSV plus audit JSON | Audit CSV |
| Architectural emphasis | Standard-library data pipeline | Event-driven processing | Strongly typed governance engine |

The implementations intentionally use different structures so that the data-cleaning concepts can be understood independently of one programming style.

---

## Running the Programs

The Python implementation requires Python 3.10 or newer because it uses modern type annotations.

Run it with:

`python data_cleaning.py`

The JavaScript implementation is designed for Node.js and uses only built-in modules.

Run it with:

`node data_cleaning.js`

The C++ implementation requires C++17 or newer.

A typical compilation command is:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic data_cleaning.cpp -o data_cleaning`

Run the resulting program with:

`./data_cleaning`

On Windows with a typical MinGW installation, the executable can be run as:

`data_cleaning.exe`

The programs create their audit outputs in the current working directory or in their respective output directories.

---

## Technical Principle

The central principle demonstrated by these implementations is that data cleaning is not simply deleting bad rows.

A reliable cleaning process distinguishes absence, duplication, statistical unusualness, and representational inconsistency. Each problem requires a different detection rule and a different treatment policy.

The strongest workflow is therefore not:

`find bad data -> delete bad data`

but:

`profile -> classify -> apply justified rule -> validate -> audit`

This approach preserves useful observations, makes transformations explainable, and creates a clear boundary between automated cleaning and decisions that require domain knowledge.
