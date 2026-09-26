# Information Theory: Entropy, Cross-Entropy, KL Divergence, and Mutual Information

## 1. Topic Introduction

Information theory is the mathematical study of information, uncertainty, communication, compression, and statistical dependence.

The central quantities covered in this project are:

- **Self-information**, which measures the information associated with one event.
- **Shannon entropy**, which measures the expected uncertainty of a probability distribution.
- **Cross-entropy**, which measures the expected coding or prediction cost when one distribution is used to represent another.
- **Kullback-Leibler divergence**, which measures the directional discrepancy between two probability distributions.
- **Mutual information**, which measures how much knowing one random variable tells us about another.
- **Conditional entropy**, which measures remaining uncertainty after another variable is known.
- **Information gain**, which measures the reduction in uncertainty produced by a partition or feature.
- **Numerically stable cross-entropy from logits**, which is important in machine-learning implementations.

The three implementations deliberately emphasize different aspects of the subject:

- Python provides a broad mathematical study environment with reusable functions, demonstrations, validation, empirical estimation, and advanced information-theoretic calculations.
- JavaScript demonstrates the same mathematical ideas in an application-oriented language and uses JavaScript data structures such as `Map`, classes, and arrays.
- C++ develops an industry-style analytics case study with validation, modular functions, classes, error handling, numerical stability, and executable correctness checks.

---

## 2. Probability as the Foundation

Information theory operates on probability distributions.

For a discrete random variable `X`, a probability distribution assigns a probability `P(X=x)` to every possible outcome.

A valid discrete distribution satisfies:

- Every probability is non-negative.
- The probabilities sum to 1.

For example:

`P(A) = 0.5`

`P(B) = 0.3`

`P(C) = 0.2`

The total is:

`0.5 + 0.3 + 0.2 = 1`

The implementations validate this condition before performing calculations.

This validation is important because information measures have mathematical meaning only when their inputs represent valid probability distributions.

---

## 3. Self-Information

The information associated with an event is defined as:

`I(x) = -log_b P(x)`

When the logarithm has base 2, the unit is **bits**.

When the logarithm has base `e`, the unit is **nats**.

A certain event has probability 1:

`I(x) = -log2(1) = 0`

Therefore, a certain event provides no surprise.

A rare event has a small probability and consequently a large information value.

For example:

`P(x) = 0.5`

gives:

`I(x) = 1 bit`

while:

`P(x) = 0.01`

gives approximately:

`6.64 bits`

The logarithm has an important structural property:

`log(ab) = log(a) + log(b)`

This allows information from independent events to be additive.

The Python implementation demonstrates this concept with the `self_information()` function. The JavaScript implementation provides the same operation through its own numerical helper.

---

## 4. Shannon Entropy

Shannon entropy is the expected self-information of a random variable.

For a discrete variable:

`H(X) = -Σ P(x) log2 P(x)`

Entropy measures uncertainty in a distribution.

A fair binary variable has:

`P(0) = 0.5`

`P(1) = 0.5`

and therefore:

`H(X) = 1 bit`

A deterministic variable has one outcome with probability 1 and therefore:

`H(X) = 0`

A six-sided fair die has:

`H(X) = log2(6)`

which is approximately `2.585 bits`.

### Interpretation

High entropy means probability mass is spread across outcomes.

Low entropy means probability mass is concentrated.

Entropy does not mean that every individual event contains exactly `H(X)` bits. It is an **expected** quantity.

---

## 5. Maximum Entropy

For a discrete variable with `n` possible outcomes, entropy satisfies:

`H(X) <= log2(n)`

Equality occurs for the uniform distribution.

For example, among all binary distributions, the maximum entropy is:

`log2(2) = 1 bit`

Among six-outcome distributions, the maximum is:

`log2(6)`

This principle is useful when interpreting how much uncertainty is theoretically possible in a system.

---

## 6. Handling Zero Probabilities

A subtle mathematical issue occurs when:

`P(x) = 0`

because `log2(0)` is undefined.

The entropy expression handles this using the limiting convention:

`0 log 0 = 0`

Therefore, zero-probability outcomes contribute nothing to entropy.

The Python, JavaScript, and C++ implementations explicitly skip zero-probability terms.

This is important because real datasets frequently contain categories that are theoretically possible but absent from a particular sample.

---

## 7. Cross-Entropy

Cross-entropy compares a reference distribution `P` with a model distribution `Q`.

It is defined as:

`H(P,Q) = -Σ P(x) log Q(x)`

The reference distribution describes what actually occurs or what is treated as the target.

The second distribution represents the model's beliefs or predictions.

Cross-entropy therefore measures the expected cost of using `Q` when the data are generated according to `P`.

A model that assigns high probability to events that actually occur generally has lower cross-entropy.

A model that assigns very low probability to events that actually occur has high cross-entropy.

---

## 8. Cross-Entropy and Classification

Cross-entropy is one of the standard loss functions for probabilistic classification.

Suppose a three-class classifier predicts:

`[0.80, 0.15, 0.05]`

and the true class is class 0.

The loss for that observation is:

`-log(0.80)`

If the model instead predicts:

`[0.01, 0.09, 0.90]`

for the same true class, the loss becomes much larger because the model assigned only `0.01` probability to the correct class.

This explains an important distinction between classification accuracy and cross-entropy.

Accuracy asks whether the highest-probability class was correct.

Cross-entropy also evaluates the quality of the probabilities.

A model can therefore have the same accuracy as another model while having substantially different cross-entropy.

---

## 9. Binary Cross-Entropy

For a binary target `y` and predicted probability `p`, binary cross-entropy is:

`-[y log(p) + (1-y) log(1-p)]`

For `y = 1`, the expression becomes:

`-log(p)`

For `y = 0`, it becomes:

`-log(1-p)`

This creates a strong penalty for confidently incorrect predictions.

The Python and JavaScript implementations explicitly clip probabilities near zero and one before taking logarithms. This avoids numerical problems involving `log(0)`.

---

## 10. Kullback-Leibler Divergence

KL divergence is defined as:

`D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))`

It measures the information discrepancy between `P` and `Q`.

Important properties include:

`D_KL(P || Q) >= 0`

and:

`D_KL(P || Q) = 0`

when the distributions agree on their relevant support.

KL divergence is **directional**:

`D_KL(P || Q)` is generally not equal to:

`D_KL(Q || P)`

This is one of its most important distinctions from ordinary distance measures.

KL divergence is therefore not a metric.

A metric normally satisfies symmetry:

`d(P,Q) = d(Q,P)`

KL divergence does not.

---

## 11. Cross-Entropy and KL Divergence

The most important relationship is:

`H(P,Q) = H(P) + D_KL(P || Q)`

This identity explains why cross-entropy cannot be smaller than the entropy of the true distribution:

`H(P,Q) >= H(P)`

because KL divergence is non-negative.

The Python, JavaScript, and C++ implementations explicitly calculate all three quantities and verify this identity numerically.

---

## 12. Support Mismatch

A critical edge case occurs when:

`P(x) > 0`

but:

`Q(x) = 0`

Then:

`D_KL(P || Q) = infinity`

and:

`H(P,Q) = infinity`

The interpretation is straightforward: the model says an event is impossible even though the reference distribution says it can occur.

This situation can occur in poorly smoothed probabilistic models.

It is particularly important when probability estimates come from finite data because an unseen category does not necessarily mean that the category is impossible.

---

## 13. Joint Distributions

Mutual information requires a joint distribution.

For two random variables `X` and `Y`, the joint probability is:

`P(X=x,Y=y)`

A joint distribution describes combinations of outcomes rather than individual variables independently.

From a joint distribution, we can obtain marginal distributions.

The marginal distribution of `X` is:

`P(X=x) = Σ_y P(X=x,Y=y)`

The marginal distribution of `Y` is:

`P(Y=y) = Σ_x P(X=x,Y=y)`

The implementations contain explicit functions for calculating both marginals.

---

## 14. Conditional Probability

Conditional probability is:

`P(Y=y | X=x) = P(X=x,Y=y) / P(X=x)`

provided:

`P(X=x) > 0`

Conditional probability describes the distribution of one variable after observing another.

The Python, JavaScript, and C++ implementations construct conditional distributions directly from joint distributions.

---

## 15. Conditional Entropy

Conditional entropy measures remaining uncertainty.

It is defined as:

`H(Y|X) = Σ_x P(x) H(Y|X=x)`

An important property is:

`H(Y|X) <= H(Y)`

Knowing `X` cannot increase the average uncertainty about `Y` in the standard discrete setting.

If `Y` is completely determined by `X`, then:

`H(Y|X) = 0`

If `X` and `Y` are independent, then:

`H(Y|X) = H(Y)`

---

## 16. Mutual Information

Mutual information measures dependence between random variables.

The definition is:

`I(X;Y) = Σ_x Σ_y P(x,y) log2(P(x,y)/(P(x)P(y)))`

An equivalent expression is:

`I(X;Y) = H(X) - H(X|Y)`

Another is:

`I(X;Y) = H(Y) - H(Y|X)`

Another important identity is:

`I(X;Y) = H(X) + H(Y) - H(X,Y)`

Mutual information is symmetric:

`I(X;Y) = I(Y;X)`

This differs from KL divergence.

Mutual information is always non-negative:

`I(X;Y) >= 0`

---

## 17. Mutual Information and Independence

Two variables are independent when:

`P(X,Y) = P(X)P(Y)`

When this condition holds:

`I(X;Y) = 0`

Therefore, mutual information can be interpreted as a measure of statistical dependence.

If two variables are perfectly related, mutual information can be large.

For a perfectly correlated binary variable with equal probabilities:

`I(X;Y) = 1 bit`

The JavaScript and C++ case studies explicitly construct both independent and perfectly correlated examples.

---

## 18. Mutual Information Versus Correlation

Mutual information and correlation should not be treated as interchangeable.

Correlation measures a particular form of statistical relationship, usually emphasizing linear association.

Mutual information is based on the full probability relationship and can detect nonlinear dependence.

Mutual information is therefore useful when the relationship between variables is not adequately described by a linear correlation coefficient.

The exact numerical value of mutual information depends on how the probability distributions are modeled or estimated.

---

## 19. Information Gain

Information gain measures how much uncertainty is reduced by a partition.

For a parent dataset divided into child groups:

`IG = H(parent) - Σ weight(child) H(child)`

This quantity is important in decision-tree learning.

A useful split tends to produce child groups that are more homogeneous.

If the child groups are pure, their entropy is low and information gain is high.

The Python and C++ implementations calculate information gain from categorical observations.

---

## 20. Feature Selection Using Mutual Information

Mutual information can also be used to estimate the relationship between a feature and a target.

For observations:

`X = [x1, x2, ..., xn]`

and:

`Y = [y1, y2, ..., yn]`

the empirical joint distribution can be estimated from observed pairs.

The resulting mutual information can provide a measure of dependence between a feature and a target.

This approach is common in feature-selection workflows, particularly for categorical or discretized variables.

Important limitation: empirical mutual information depends on the quality and quantity of available data.

Small datasets can produce unstable estimates.

High-cardinality variables can also create misleading estimates without appropriate statistical treatment.

---

## 21. Normalized Mutual Information

Raw mutual information depends on the entropy scale of the variables.

One possible normalization is:

`NMI = I(X;Y) / sqrt(H(X)H(Y))`

This is not the only normalization used in practice.

Different applications may use different denominators and therefore produce different normalized values.

The Python and JavaScript implementations demonstrate one common normalization for comparison purposes.

---

## 22. Coding Interpretation of Entropy

Entropy has a fundamental connection with lossless data compression.

For an idealized source with distribution `P`, entropy represents the theoretical average information rate:

`H(P) bits per symbol`

Practical prefix codes have average code lengths related to entropy.

The Python implementation constructs a distribution and a corresponding set of code lengths, then calculates expected code length.

Entropy should be understood as a fundamental statistical limit rather than a promise that every practical compression algorithm will exactly achieve it.

Finite blocks, code constraints, model mismatch, and implementation overhead affect actual compression performance.

---

## 23. Units: Bits and Nats

The logarithm base determines the unit.

Base 2:

`log2`

produces **bits**.

Base `e`:

`ln`

produces **nats**.

For the same probability distribution, the numerical values differ because the unit differs.

The conversion is:

`1 nat = log2(e) bits`

and:

`1 bit = ln(2) nats`

The mathematical meaning is unchanged by the choice of logarithm base.

---

## 24. Numerical Stability

Directly computing softmax can create numerical overflow.

Softmax is:

`softmax(z_i) = exp(z_i) / Σ exp(z_j)`

If a logit is extremely large, `exp(z_i)` can exceed the numerical range of the floating-point type.

A stable implementation subtracts the maximum logit:

`softmax(z_i) = exp(z_i - max(z)) / Σ exp(z_j - max(z))`

Subtracting the same constant from every logit does not change the final probabilities.

The Python, JavaScript, and C++ implementations demonstrate stable softmax.

---

## 25. Log-Sum-Exp

For logits, cross-entropy can be computed without explicitly constructing probabilities.

For true class `k`:

`loss = log(Σ exp(z_i)) - z_k`

A numerically stable version computes:

`log(Σ exp(z_i)) = m + log(Σ exp(z_i-m))`

where:

`m = max(z_i)`

This is called the **log-sum-exp** technique.

It is an important implementation detail in machine-learning systems because it reduces overflow and underflow problems.

---

## 26. Python Implementation

The Python script is structured as a complete information-theory study program.

### Core mathematical functions

Important functions include:

- `self_information()`
- `entropy()`
- `cross_entropy()`
- `kl_divergence()`
- `marginal_x()`
- `marginal_y()`
- `conditional_distribution_y_given_x()`
- `conditional_entropy_y_given_x()`
- `mutual_information()`
- `joint_entropy()`

These functions separate mathematical operations from demonstration code.

### Data analysis

The script also contains:

- Empirical distribution estimation.
- Information gain.
- Mutual information from samples.
- Normalized mutual information.
- Sampling experiments.
- Expected code-length calculations.

### Numerical methods

The script includes:

- Stable softmax.
- Stable cross-entropy from logits.
- Probability clipping where necessary.
- Explicit handling of infinite divergence.

### Object-oriented design

`DistributionComparison` encapsulates a reference distribution and candidate distribution.

Its properties expose:

- Reference entropy.
- Cross-entropy.
- KL divergence.

This demonstrates how mathematical operations can be organized into reusable application-level abstractions.

---

## 27. JavaScript Implementation

The JavaScript implementation focuses on practical execution and application-level data structures.

### Maps as probability distributions

JavaScript `Map` objects are used to represent discrete distributions.

For example, a conceptual distribution is represented as a mapping between an outcome and its probability.

This is preferable to relying on object property names when arbitrary keys or explicit map semantics are useful.

### Joint distributions

Joint observations are represented using structured records containing:

- `x`
- `y`
- `probability`

This makes the relationship between the mathematical representation and the application data structure explicit.

### Classification analyzer

The `ClassificationAnalyzer` class calculates:

- Categorical cross-entropy.
- Classification accuracy.

This illustrates an important distinction: accuracy examines predicted class labels, whereas cross-entropy examines predicted probabilities.

### JavaScript-specific numerical considerations

JavaScript uses IEEE 754 double-precision floating-point numbers for ordinary numeric calculations.

Consequently:

- Probability totals can have tiny rounding errors.
- Numerical comparisons should generally use tolerances.
- Logarithms of zero must be handled explicitly.
- Stable softmax is necessary for very large logits.

---

## 28. C++ Industry-Style Case Study

The C++ program models a probabilistic analytics service.

The scenario combines several information-theoretic operations in one application.

### Problem being modeled

The system receives:

1. A categorical traffic distribution.
2. A model forecast.
3. Joint weather and traffic observations.
4. Fraud classification labels.
5. Candidate feature partitions.
6. Probabilistic classifier outputs.

The system evaluates these data using information-theoretic measures.

### Major components

The program contains:

- Distribution validation.
- Joint-distribution validation.
- Entropy calculation.
- Cross-entropy calculation.
- KL divergence.
- Marginalization.
- Conditional distributions.
- Conditional entropy.
- Mutual information.
- Empirical distributions.
- Information gain.
- Classification analysis.
- Stable softmax.
- Stable cross-entropy from logits.
- Error handling.
- Correctness assertions.

---

## 29. C++ Data Structures

The C++ case study uses:

`std::map<std::string, double>`

for ordinary discrete probability distributions.

A joint distribution is represented by:

`std::map<std::pair<std::string, std::string>, double>`

This directly models the mathematical pair `(X,Y)`.

The `ClassificationRecord` structure groups an actual class with a probability vector.

This is a natural representation for multiclass prediction records.

---

## 30. C++ Classification Analyzer

`ClassificationAnalyzer` encapsulates a collection of classification records.

It provides:

- `crossEntropyLoss()`
- `accuracy()`

The class validates each probability vector before performing calculations.

The implementation also verifies that the true class index is valid.

This prevents malformed data from silently producing meaningless results.

---

## 31. C++ Error Handling

The C++ case study uses standard exceptions such as:

`std::invalid_argument`

for invalid probability distributions and invalid inputs.

The `main()` function catches:

`std::exception`

and reports the error.

This structure is appropriate for a small standalone analytical application because invalid input is handled explicitly instead of causing undefined behavior.

---

## 32. Edge Cases

Important edge cases demonstrated across the implementations include:

### Empty distributions

An empty probability distribution cannot define entropy.

### Negative probabilities

Negative probabilities are rejected.

### Probabilities that do not sum to one

They are rejected instead of being silently normalized.

### Zero probability in entropy

A zero-probability term contributes zero to entropy.

### Zero predicted probability

If the reference distribution assigns positive probability to an event but the model assigns zero probability, cross-entropy and KL divergence become infinite.

### Deterministic variables

A deterministic distribution has zero entropy.

### Independent variables

Independent variables have zero mutual information.

### Floating-point error

Probability totals can be slightly different from one due to floating-point arithmetic, so the implementations use tolerances.

### Large logits

Direct exponentiation can overflow, so stable softmax and log-sum-exp are used.

---

## 33. Important Distinctions

### Entropy versus cross-entropy

Entropy describes the uncertainty of one distribution.

Cross-entropy describes the expected cost of using one distribution to represent another.

### Cross-entropy versus KL divergence

They satisfy:

`H(P,Q) = H(P) + D_KL(P||Q)`

Cross-entropy includes the intrinsic uncertainty of `P`.

KL divergence measures the additional discrepancy caused by using `Q`.

### KL divergence versus mutual information

KL divergence compares distributions and is directional.

Mutual information measures dependence between random variables and is symmetric.

### Mutual information versus conditional entropy

Mutual information measures uncertainty reduction:

`I(X;Y) = H(Y) - H(Y|X)`

Conditional entropy measures what uncertainty remains after information about another variable is available.

### Accuracy versus cross-entropy

Accuracy uses the final predicted class.

Cross-entropy evaluates the predicted probability assigned to the true class.

---

## 34. Common Mistakes

### Treating KL divergence as symmetric

Incorrect assumption:

`D_KL(P||Q) = D_KL(Q||P)`

In general, this is false.

### Calling KL divergence a distance

KL divergence is not a metric because it is directional and does not satisfy the standard metric requirements.

### Ignoring zero probabilities

Expressions containing `log(0)` require careful handling.

### Computing softmax naively

Exponentiating very large logits can overflow.

### Assuming zero observations mean impossible events

A category absent from a finite sample may still have non-zero probability in the underlying population.

### Confusing entropy with information of an individual event

Entropy is an expectation over possible events.

### Ignoring estimation error

Empirical entropy and mutual information are estimates derived from finite observations, not necessarily exact population quantities.

### Assuming high mutual information proves causation

Mutual information measures statistical dependence.

It does not establish that one variable causes another.

---

## 35. Limitations

Information-theoretic measures depend on the quality of the probability model.

For empirical data, the probability distribution must be estimated.

This introduces statistical uncertainty.

Mutual information can be especially sensitive to:

- Sample size.
- Number of categories.
- Rare categories.
- Discretization choices.
- Estimation methodology.

KL divergence can become infinite when the second distribution assigns zero probability to events supported by the first distribution.

Cross-entropy can also become infinite in the same situation.

Numerical implementations therefore require careful handling of floating-point arithmetic and probability boundaries.

---

## 36. Performance Considerations

For a discrete distribution containing `n` outcomes:

- Entropy is generally `O(n)`.
- Cross-entropy is generally `O(n)`.
- KL divergence is generally `O(n)` when distribution lookup is efficient.
- Marginalization over `n` joint entries is `O(n)`.
- Mutual information is generally `O(n)` for an explicitly represented joint distribution.
- Empirical distribution construction is approximately `O(n)` with efficient map or hash-map operations.

The actual runtime depends on the underlying data structure.

The C++ implementation uses ordered `std::map`, which provides logarithmic lookup and insertion.

A production implementation processing very large categorical datasets could use hash-based structures such as `std::unordered_map` when ordering is not required.

Memory consumption is proportional to the number of explicitly represented outcomes or joint combinations.

---

## 37. Security and Production Considerations

Information-theoretic calculations do not inherently provide security.

When used in production systems:

- Validate externally supplied probabilities.
- Reject NaN and infinite values.
- Prevent malformed inputs from reaching numerical operations.
- Apply sensible bounds to category counts and dataset sizes.
- Consider denial-of-service risks from extremely large inputs.
- Protect sensitive data used to estimate distributions.
- Avoid logging confidential observations unnecessarily.
- Treat probability outputs as potentially sensitive when they describe users, transactions, or other protected information.

For security analytics, mutual information and entropy can help characterize data, but they should not be treated as standalone security decisions.

---

## 38. Practical Applications

### Machine learning

Cross-entropy is widely used to train probabilistic classifiers.

KL divergence is used in distribution matching, regularization, probabilistic modeling, and variational methods.

Mutual information is used for feature selection and dependence analysis.

### Data compression

Entropy provides a theoretical basis for lossless compression.

### Natural language processing

Token distributions, cross-entropy, and perplexity are important statistical measures for language modeling.

### Cybersecurity

Entropy can characterize randomness or concentration in data.

Mutual information can help investigate relationships between security-related variables.

KL divergence can compare observed and reference distributions.

### Decision trees

Entropy and information gain are used to evaluate candidate splits.

### Communication systems

Information theory provides mathematical foundations for source coding and channel coding.

### Anomaly detection

Distributional differences can be quantified using divergence measures, although the choice of measure depends on the application.

---

## 39. Cross-Entropy and Perplexity

For language modeling, cross-entropy is often expressed in bits or nats.

If cross-entropy is measured in nats:

`Perplexity = exp(cross-entropy)`

If cross-entropy is measured in bits:

`Perplexity = 2^(cross-entropy)`

Perplexity can be interpreted as an effective number of equally likely alternatives under certain modeling assumptions.

It is not simply another name for entropy and should be interpreted according to the underlying evaluation setup.

---

## 40. Conditional Mutual Information

A more advanced quantity is conditional mutual information:

`I(X;Y|Z)`

It measures the dependence between `X` and `Y` after conditioning on `Z`.

One definition is:

`I(X;Y|Z) = Σ P(x,y,z) log(P(x,y|z)/(P(x|z)P(y|z)))`

It is useful when a third variable affects the relationship being analyzed.

The Python implementation includes a complete calculation for a discrete three-variable distribution.

---

## 41. Mathematical Relationship Map

The major quantities are connected by the following relationships.

Self-information:

`I(x) = -log P(x)`

Entropy:

`H(X) = E[I(X)]`

Conditional entropy:

`H(Y|X) = H(X,Y) - H(X)`

Mutual information:

`I(X;Y) = H(Y) - H(Y|X)`

and:

`I(X;Y) = H(X) - H(X|Y)`

KL divergence:

`D_KL(P||Q) = H(P,Q) - H(P)`

Cross-entropy:

`H(P,Q) = H(P) + D_KL(P||Q)`

These relationships form the conceptual core of the project.

---

## 42. Implementation Comparison

| Concept | Python | JavaScript | C++ |
|---|---|---|---|
| Probability validation | Yes | Yes | Yes |
| Self-information | Yes | Yes | Yes |
| Entropy | Yes | Yes | Yes |
| Cross-entropy | Yes | Yes | Yes |
| KL divergence | Yes | Yes | Yes |
| Joint distributions | Yes | Yes | Yes |
| Conditional entropy | Yes | Yes | Yes |
| Mutual information | Yes | Yes | Yes |
| Information gain | Yes | Yes | Yes |
| Empirical distributions | Yes | Yes | Yes |
| Stable softmax | Yes | Yes | Yes |
| Stable logits loss | Yes | Yes | Yes |
| Classification analysis | Yes | Yes | Yes |
| Conditional mutual information | Yes | No | No |
| Sampling demonstration | Yes | No | No |
| Object-oriented analysis | Yes | Yes | Yes |
| Explicit exception handling | Yes | Yes | Yes |
| Executable correctness checks | Yes | Yes | Yes |

The implementations are intentionally not identical.

Python emphasizes mathematical exploration and a broad study program.

JavaScript emphasizes application-level data structures and reusable classification analysis.

C++ emphasizes an integrated technical system with explicit types, validation, exceptions, architecture, and performance-conscious implementation.

---

## 43. Best Practices

1. Validate every probability distribution.
2. Allow small floating-point tolerance when checking probability totals.
3. Handle zero probabilities explicitly.
4. Avoid naive softmax implementations for large logits.
5. Use log-sum-exp for stable log-domain calculations.
6. Distinguish entropy from cross-entropy.
7. Remember that KL divergence is directional.
8. Do not interpret mutual information as proof of causality.
9. Consider estimation bias and sample size when working with empirical distributions.
10. Keep units explicit, especially when switching between bits and nats.
11. Test mathematical identities with known distributions.
12. Separate mathematical functions from application logic.
13. Validate class indices in classification systems.
14. Avoid unnecessary probability normalization that can hide invalid upstream data.
15. Treat infinite divergence as meaningful information about support mismatch rather than merely as a numerical failure.

---

## 44. Core Equations Reference

### Self-information

`I(x) = -log_b P(x)`

### Shannon entropy

`H(X) = -Σ P(x) log_b P(x)`

### Cross-entropy

`H(P,Q) = -Σ P(x) log_b Q(x)`

### KL divergence

`D_KL(P||Q) = Σ P(x) log_b(P(x)/Q(x))`

### Conditional entropy

`H(Y|X) = Σ P(x)H(Y|X=x)`

### Mutual information

`I(X;Y) = Σ P(x,y)log(P(x,y)/(P(x)P(y)))`

### Information gain

`IG = H(parent) - weighted_child_entropy`

### Cross-entropy identity

`H(P,Q) = H(P) + D_KL(P||Q)`

### Mutual-information identity

`I(X;Y) = H(Y) - H(Y|X)`

### Conditional mutual information

`I(X;Y|Z) = Σ P(x,y,z)log(P(x,y|z)/(P(x|z)P(y|z)))`

---

## 45. Executable Study Structure

The Python program runs a sequence of demonstrations covering:

1. Self-information.
2. Entropy.
3. Cross-entropy.
4. KL divergence.
5. The cross-entropy/KL identity.
6. Mutual information.
7. Entropy identities and inequalities.
8. Numerical stability.
9. Multiclass classification.
10. Binary cross-entropy.
11. Empirical entropy.
12. Information gain.
13. Feature relevance.
14. Coding interpretation.
15. Edge cases.
16. Sampling.
17. Normalized mutual information.
18. Cross-entropy and perplexity.
19. Concept comparison.
20. Conditional mutual information.
21. Class-based distribution comparison.

The JavaScript program provides a parallel executable study using JavaScript-native structures and a reusable classification analyzer.

The C++ program integrates these concepts into a single analytics scenario involving traffic modeling, model evaluation, dependence analysis, fraud feature selection, classification, numerical stability, and validation.

The three programs include executable checks so that fundamental mathematical identities are not merely stated but tested programmatically.
