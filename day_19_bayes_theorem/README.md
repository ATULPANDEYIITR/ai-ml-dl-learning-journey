# Bayes' theorem: conditional probability, likelihood, posterior, and priors

## Topic introduction

Bayes' theorem is a mathematical rule for updating the probability of a hypothesis after new evidence is observed. It connects four central ideas:

- **Prior probability**: the probability assigned to a hypothesis before considering new evidence.
- **Likelihood**: the probability of observing the evidence assuming the hypothesis is true.
- **Evidence**: the overall probability of observing the evidence across the relevant hypotheses.
- **Posterior probability**: the updated probability of the hypothesis after the evidence is incorporated.

The fundamental relationship is:

`P(H | E) = P(E | H) P(H) / P(E)`

where `H` represents a hypothesis and `E` represents evidence.

Bayesian reasoning is not limited to one application area. It appears in diagnostic testing, spam filtering, fraud detection, search, machine learning, reliability engineering, scientific inference, forecasting, robotics, finance, cybersecurity, and many other systems where uncertainty must be represented explicitly.

---

## Fundamental probability concepts

### Probability

Probability is a numerical representation of uncertainty. For an event `A`:

`0 <= P(A) <= 1`

A probability of `0` represents an impossible event under the model, while `1` represents an event considered certain under the model.

For empirical data, probability can be estimated as:

`P(A) = number of favorable observations / total observations`

The Python implementation demonstrates this idea through validation and probability calculations rather than treating probability as an abstract mathematical expression only.

### Complement

The complement of event `A` is written as `not A`.

`P(not A) = 1 - P(A)`

For example, if the probability of fraud is `0.02`, the probability of a legitimate transaction under the same binary model is:

`1 - 0.02 = 0.98`

The complement operation is fundamental to binary Bayesian calculations.

### Intersection

The intersection of events `A` and `B` means that both occur.

It is commonly written as:

`P(A and B)`

The multiplication rule gives:

`P(A and B) = P(A) P(B | A)`

This rule is used repeatedly in the implementations.

---

## Conditional probability

Conditional probability answers a question under a condition.

`P(A | B)`

is read as:

"the probability of A given B."

The definition is:

`P(A | B) = P(A and B) / P(B)`

provided `P(B) > 0`.

The order matters. In general:

`P(A | B) != P(B | A)`

This distinction is one of the most important concepts behind Bayes' theorem.

For example:

- `P(Disease | Positive Test)` asks how likely disease is after a positive test.
- `P(Positive Test | Disease)` asks how likely a positive test is when disease is already known.

These are different conditional probabilities.

The Python and JavaScript implementations explicitly demonstrate this distinction.

---

## The multiplication rule

Starting with:

`P(A | B) = P(A and B) / P(B)`

we can rearrange it to obtain:

`P(A and B) = P(A | B) P(B)`

The equivalent ordering:

`P(A and B) = P(B | A) P(A)`

is also valid.

This relationship provides the mathematical foundation for deriving Bayes' theorem.

---

## Bayes' theorem

Bayes' theorem can be written as:

`P(H | E) = P(E | H) P(H) / P(E)`

The four components have distinct meanings.

### Prior

`P(H)`

The prior is the probability assigned to the hypothesis before considering the current evidence.

A prior can come from:

- historical frequency
- population prevalence
- previous observations
- domain knowledge
- an earlier posterior
- a formal probabilistic model

The prior is not the same thing as the likelihood.

### Likelihood

`P(E | H)`

The likelihood measures how compatible the observed evidence is with the hypothesis.

It asks:

"If the hypothesis were true, how probable would this evidence be?"

A likelihood is conditional on the hypothesis.

### Evidence

`P(E)`

The denominator represents the total probability of observing the evidence.

For a binary hypothesis:

`P(E) = P(E | H)P(H) + P(E | not H)P(not H)`

For several mutually exclusive hypotheses:

`P(E) = sum P(E | H_i)P(H_i)`

This is the law of total probability.

### Posterior

`P(H | E)`

The posterior is the updated probability of the hypothesis after incorporating the evidence.

It is the quantity normally sought when applying Bayes' theorem.

---

## Why the denominator matters

Suppose:

- prevalence is `1%`
- sensitivity is `99%`
- false-positive rate is `5%`

It is incorrect to conclude that a positive result means there is a `99%` probability of disease.

The sensitivity represents:

`P(Positive | Disease)`

The question of interest is:

`P(Disease | Positive)`

The denominator includes both true positives and false positives:

`P(Positive) = P(Positive | Disease)P(Disease) + P(Positive | Healthy)P(Healthy)`

The Python frequency-table example makes this effect concrete by constructing a population of `100,000` people and separating:

- true positives
- false positives
- false negatives
- true negatives

This is an important practical interpretation of the base-rate effect.

---

## Prior probability and base rates

The prior can substantially influence the posterior.

Suppose a condition is rare. Even an accurate test can generate many false positives if the healthy population is much larger than the affected population.

This does not mean that the test is necessarily poor. It means that sensitivity, specificity, prevalence, and the direction of the conditional probability answer different questions.

A Bayesian calculation combines them rather than treating one test characteristic as the complete answer.

---

## Sensitivity and specificity

For diagnostic testing:

### Sensitivity

`Sensitivity = P(Positive | Disease)`

It measures the probability that the test is positive when the condition is present.

### Specificity

`Specificity = P(Negative | No Disease)`

It measures the probability that the test is negative when the condition is absent.

### False-positive rate

`False Positive Rate = 1 - Specificity`

Therefore:

`P(Positive | No Disease) = 1 - Specificity`

### False-negative rate

`False Negative Rate = 1 - Sensitivity`

Therefore:

`P(Negative | Disease) = 1 - Sensitivity`

These quantities are not interchangeable with posterior probabilities.

---

## Positive predictive value

The positive predictive value asks:

`P(Disease | Positive)`

For prevalence `p`, sensitivity `s`, and false-positive rate `f`:

`PPV = sp / [sp + f(1-p)]`

The Python `DiagnosticTest` class implements this relationship.

The result depends on prevalence. Therefore a predictive value measured in one population may not transfer directly to another population with a different prevalence.

---

## Negative predictive value

The negative predictive value asks:

`P(No Disease | Negative)`

It can be written as:

`NPV = Specificity(1-p) / [Specificity(1-p) + (1-Sensitivity)p]`

The Python implementation exposes both positive and negative predictive values through the `DiagnosticTest` class.

---

## Multiple hypotheses

Bayesian inference does not require exactly two hypotheses.

Suppose there are hypotheses:

- `H1`
- `H2`
- `H3`

with priors:

`P(H1), P(H2), P(H3)`

and evidence likelihoods:

`P(E|H1), P(E|H2), P(E|H3)`

Then:

`P(Hi|E) = P(E|Hi)P(Hi) / sum[P(E|Hj)P(Hj)]`

The denominator normalizes the unnormalized posterior values so that:

`sum P(Hi|E) = 1`

The Python, JavaScript, and C++ implementations all contain examples of multiple competing hypotheses.

---

## Unnormalized posterior

Bayesian calculations are often easier to understand in proportional form:

`P(H | E) proportional to P(E | H)P(H)`

The quantity:

`P(E | H)P(H)`

is sometimes called an unnormalized posterior.

For multiple hypotheses, calculate this quantity for each hypothesis and normalize:

`Posterior_i = Unnormalized_i / Sum(Unnormalized_j)`

This form is especially useful in implementations because the same normalization denominator is shared by all hypotheses.

---

## Odds form of Bayes' theorem

Bayes' theorem can also be expressed using odds.

Probability-to-odds conversion:

`Odds(H) = P(H) / [1-P(H)]`

The likelihood ratio is:

`LR = P(E | H) / P(E | not H)`

The Bayesian odds equation is:

`Posterior Odds = Prior Odds × Likelihood Ratio`

This representation is useful when several independent pieces of evidence are incorporated sequentially.

The implementations demonstrate that a likelihood ratio greater than `1` increases the odds of the hypothesis, while a likelihood ratio below `1` decreases them.

---

## Likelihood ratios

A likelihood ratio compares how strongly evidence supports one state relative to another.

For binary hypotheses:

`LR = P(E|H) / P(E|not H)`

Interpretation:

- `LR > 1`: evidence is more compatible with `H`.
- `LR = 1`: evidence provides no likelihood-ratio distinction.
- `LR < 1`: evidence is more compatible with `not H`.

The likelihood ratio does not itself equal the posterior probability.

The posterior also depends on the prior odds.

---

## Sequential Bayesian updating

Bayesian inference can be performed repeatedly.

Suppose an initial prior is:

`P(H)`

After evidence `E1`:

`P(H|E1)`

The resulting posterior can become the prior for a subsequent observation `E2`:

`P(H|E1,E2)`

Under an appropriate conditional-independence assumption, evidence can be incorporated one observation at a time.

The Python and JavaScript implementations demonstrate a sequence containing positive and negative observations.

This pattern is useful for:

- sensor systems
- monitoring
- fraud detection
- diagnostics
- reliability analysis
- online classification
- repeated measurements

A critical modeling issue is dependence. If two observations contain essentially the same information, treating them as independent can cause excessive confidence.

---

## Bayesian updating and time

Bayesian inference naturally supports changing beliefs as information arrives.

A common structure is:

`Prior -> Evidence -> Posterior -> New Prior -> New Evidence -> New Posterior`

The mathematics does not require the observations to arrive simultaneously.

The implementation should nevertheless specify the assumptions under which sequential multiplication or updating is valid.

---

## Naive Bayes

Naive Bayes is a family of classifiers based on Bayes' theorem combined with a simplifying conditional-independence assumption.

For features `x1, x2, ..., xn` and class `C`:

`P(C | x1,...,xn) proportional to P(C)P(x1,...,xn | C)`

The naive assumption approximates:

`P(x1,...,xn | C) = P(x1|C)P(x2|C)...P(xn|C)`

Therefore:

`P(C | x1,...,xn) proportional to P(C) product P(xi|C)`

The Python and JavaScript implementations include a small text classifier that uses this approach.

---

## Laplace smoothing

A problem occurs when a word has never appeared in a particular class.

Without smoothing:

`P(word|class) = 0`

A single zero factor can make an entire product equal to zero.

Laplace smoothing addresses this by adding one to counts:

`P(word|class) = (count(word,class)+1) / (total words in class + vocabulary size)`

The Python and JavaScript Naive Bayes implementations use this method.

Smoothing is a modeling choice. It changes probability estimates and should not be treated as a universal solution for every statistical problem.

---

## Log probabilities

Bayesian models can involve products of many small probabilities:

`p1 × p2 × p3 × ... × pn`

With enough factors, ordinary floating-point arithmetic can underflow toward zero.

Logarithms transform multiplication into addition:

`log(p1p2...pn) = log(p1) + log(p2) + ... + log(pn)`

The Python and C++ implementations demonstrate log-space reasoning, and the JavaScript Naive Bayes classifier also uses logarithmic scores.

The ordering of numerical operations matters when probabilities are extremely small.

---

## Log-sum-exp

Bayesian normalization sometimes requires:

`log(exp(a1) + exp(a2) + ... + exp(an))`

Direct exponentiation can overflow or underflow.

A stable transformation uses the maximum value `m`:

`log(sum exp(ai)) = m + log(sum exp(ai-m))`

where:

`m = max(ai)`

The Python and C++ implementations contain explicit stable log-sum-exp functions.

This technique is common in numerical statistical computing.

---

## Continuous observations

For a continuous variable, the likelihood is often represented by a probability density rather than a probability assigned to one exact real-valued point.

The Python and JavaScript implementations use a normal density:

`f(x) = 1 / (sigma sqrt(2pi)) × exp[-(x-mu)^2/(2sigma^2)]`

For competing continuous hypotheses:

`Posterior_i proportional to Prior_i × Density_i(x)`

A density can be greater than `1` when the unit interval is sufficiently narrow or the distribution is concentrated. This is not a contradiction because a density is not itself a probability of an exact continuous point.

The posterior still results from normalization across the competing hypotheses.

---

## Bayesian decision theory

Probability estimation and decision-making are related but distinct.

A posterior might state:

`P(Fraud | Evidence) = 0.30`

That probability alone does not determine whether a transaction should be approved, reviewed, or declined.

Decision theory introduces a loss or cost function.

For an action `a`:

`Expected Loss(a) = sum P(state | evidence) × Loss(a,state)`

The action with the smallest expected loss can then be selected under the specified loss model.

The Python, JavaScript, and C++ implementations demonstrate this distinction.

Changing the loss matrix can change the selected action even when the posterior probability remains unchanged.

---

## Python implementation

The Python script is structured as a complete study implementation.

### Basic functions

`probability`, `complement`, `conditional_probability`, and `intersection_from_conditional` establish the basic probability operations.

Validation is performed before calculations so that invalid probabilities and impossible joint-probability relationships are detected explicitly.

### Bayes functions

`bayes_theorem` presents the direct formula.

`bayes_binary` handles the common two-state case and calculates the evidence denominator automatically.

This makes the implementation useful for diagnostic testing, fraud detection, classification, and other binary inference problems.

### Multiple hypotheses

`BayesianTerms` stores:

- hypothesis name
- prior
- likelihood
- unnormalized posterior
- posterior

`posterior_distribution` calculates normalized posterior probabilities for multiple competing hypotheses.

### Diagnostic testing

`DiagnosticTest` models:

- sensitivity
- specificity
- false-positive rate
- false-negative rate
- positive predictive value
- negative predictive value

This demonstrates how Bayes' theorem connects test characteristics with prevalence.

### Naive Bayes

`NaiveBayesTextClassifier` provides a complete small text classifier.

It implements:

- tokenization
- class counts
- word counts
- vocabulary construction
- prior probabilities
- Laplace smoothing
- log probabilities
- classification

### Sequential updating

`sequential_update` shows how a posterior can be updated as additional observations arrive.

The example deliberately contains both positive and negative evidence.

### Continuous inference

`normal_pdf` and `gaussian_hypothesis_posterior` demonstrate Bayesian reasoning with continuous observations.

The implementation correctly treats the normal density as a likelihood rather than incorrectly assuming that a density value must itself be a probability.

### Decision theory

`choose_minimum_expected_loss` separates the posterior from the operational action.

This is important in real systems because different errors can have different consequences.

### Simulation

`monte_carlo_bayes_check` provides an empirical validation of an analytical posterior.

The simulation demonstrates that repeated sampling can approximate the Bayesian result while also showing why analytical calculations are preferable when an exact formula is available.

### Evaluation

`brier_score` demonstrates a metric for evaluating probabilistic predictions rather than merely evaluating hard classifications.

---

## JavaScript implementation

The JavaScript implementation emphasizes application-oriented programming patterns.

### Core probability functions

`validateProbability`, `conditionalProbability`, `intersectionProbability`, and `bayesBinary` establish the basic mechanics.

JavaScript's `Number` type is used for floating-point calculations.

### Objects and classes

The multiple-hypothesis representation uses ordinary JavaScript objects.

`NaiveBayesClassifier` demonstrates how Bayesian state can be encapsulated in a class.

The implementation uses `Map` for class and word-count structures and `Set` for the vocabulary.

### Higher-level application behavior

The JavaScript implementation includes asynchronous evidence collection through Promises and `async`/`await`.

In a production application, asynchronous evidence could originate from:

- a database
- an HTTP API
- a sensor service
- a browser event
- an external scoring service

The demonstration uses deterministic local evidence rather than depending on an external service.

### Continuous likelihoods

`normalPdf` and `continuousPosterior` demonstrate Bayesian inference using probability densities.

### Decision theory

`minimumExpectedLoss` shows how a JavaScript application can turn posterior probabilities into expected-loss calculations.

The action itself depends on the supplied cost model rather than on the probability alone.

---

## C++ case study

The C++ implementation models an industry-style fraud detection workflow.

### Problem being modeled

A financial institution receives transactions containing:

- transaction identifier
- transaction type
- amount
- international status
- unusual location status
- unusual device status

The system estimates:

`P(Fraud | Evidence)`

and then evaluates candidate actions using expected loss.

### Domain structures

`Transaction` represents an individual transaction.

`EvidenceProfile` stores the Bayesian parameters.

`ScoredTransaction` combines transaction data, posterior probability, and the resulting decision.

`TransactionType` provides a strongly typed representation of transaction categories.

### Bayesian model

`FraudBayesianModel` encapsulates the fraud inference process.

Each evidence feature is converted into a conditional likelihood.

For example:

`P(Unusual Location | Fraud)`

is compared with:

`P(Unusual Location | Legitimate)`

The posterior is updated sequentially.

### Conditional independence assumption

The case study treats the evidence features as conditionally independent given the fraud state.

This is similar in spirit to the assumption used by Naive Bayes.

The assumption simplifies computation but may be inaccurate when features are correlated.

For example, an unusual device and unusual location might frequently occur together for the same underlying reason. Counting both as fully independent pieces of evidence could exaggerate their combined effect.

A production model should validate such assumptions against real data.

### Decision model

The C++ program uses a loss matrix containing:

- approval loss under fraud
- approval loss under legitimacy
- review loss under fraud
- review loss under legitimacy
- decline loss under fraud
- decline loss under legitimacy

Expected loss is calculated for each action.

This demonstrates the distinction between inference and decision-making.

### Data structures

The implementation uses standard C++ structures:

- `struct`
- `class`
- `std::vector`
- `std::map`
- `std::string`
- `std::pair`
- `std::accumulate`
- `std::max_element`

No third-party dependency is required.

### Error handling

The program uses exceptions such as `std::invalid_argument` to handle invalid probabilities, mismatched arrays, zero evidence, invalid odds, and other model failures.

The `main` function catches exceptions and returns a nonzero status for fatal failures.

---

## Important distinctions

### Prior vs likelihood

A prior describes belief in the hypothesis before current evidence.

A likelihood describes the probability of the observed evidence under the hypothesis.

They have different roles:

`Prior = P(H)`

`Likelihood = P(E|H)`

### Likelihood vs posterior

The likelihood is:

`P(E|H)`

The posterior is:

`P(H|E)`

The two expressions reverse the conditioning direction.

### Evidence vs likelihood

The evidence is the probability of the observed evidence under all relevant hypotheses:

`P(E) = sum P(E|Hi)P(Hi)`

It is the normalization factor in Bayes' theorem.

### Probability vs odds

Probability lies between `0` and `1`.

Odds are:

`p/(1-p)`

They represent the ratio of probability to complementary probability.

### Sensitivity vs positive predictive value

Sensitivity:

`P(Positive|Disease)`

Positive predictive value:

`P(Disease|Positive)`

These are not equivalent.

### Bayesian inference vs decision theory

Bayesian inference estimates uncertainty.

Decision theory combines uncertainty with consequences.

A posterior probability is not automatically an action recommendation.

---

## Edge cases

### Zero conditioning probability

`P(A|B)` is undefined when:

`P(B) = 0`

The implementations explicitly reject this case.

### Zero evidence probability

Bayes' theorem requires a nonzero evidence denominator.

If:

`P(E) = 0`

the posterior is undefined under the supplied model.

### Probability zero

A prior of zero means that the hypothesis receives zero probability under the model. Multiplication by a zero prior produces a zero posterior unless the model is changed.

### Probability one

A prior of one represents certainty under the model. Its complement is zero.

### Likelihood equal to zero

If:

`P(E|H) = 0`

the evidence is considered impossible under that hypothesis.

### Likelihood ratio equal to one

When:

`P(E|H) = P(E|not H)`

the likelihood ratio is `1`.

Such evidence does not change the prior odds.

### Continuous density

A probability density can exceed `1`. This does not mean that a probability exceeds `1`. A density and a probability are different mathematical quantities.

---

## Common mistakes

### Mistaking `P(E|H)` for `P(H|E)`

This is the classic inverse-probability error.

The test being highly likely under disease does not directly imply that disease is highly likely after a positive test.

### Ignoring the base rate

A likelihood or test accuracy cannot be interpreted independently of the prior population probability.

### Ignoring false positives

When the alternative population is large, even a relatively small false-positive rate can produce many false positives.

### Using probabilities without validation

Production code should validate probability inputs and model assumptions.

### Multiplying many small probabilities directly

Long probability products can underflow.

Log probabilities are generally safer for large models.

### Treating correlated evidence as independent

Sequential multiplication can become misleading if observations are strongly dependent.

### Treating a posterior as an action

A posterior describes uncertainty. An action requires a decision rule, cost function, threshold, or other operational criterion.

### Using an unsuitable prior

A prior can encode strong assumptions. Sensitivity analysis should be considered when prior selection materially affects the result.

### Confusing density and probability

Continuous distributions use densities. Integrating a density over an interval gives probability.

---

## Limitations

Bayesian calculations are only as sound as the probability model behind them.

Important limitations include:

- incorrect priors
- incorrect likelihood estimates
- insufficient data
- distribution shift
- dependent evidence
- measurement error
- model misspecification
- numerical underflow
- poorly calibrated probabilities
- inappropriate decision costs
- changing populations
- hidden variables
- selection bias

Bayes' theorem itself is mathematically straightforward. The difficult part in practical systems is often specifying and estimating the probability model correctly.

---

## Performance considerations

For a binary Bayesian update involving `k` independent evidence features, the basic sequential implementation is approximately:

`O(k)`

For `n` competing hypotheses and `k` features, a straightforward implementation is approximately:

`O(nk)`

Memory requirements depend on how likelihood parameters and feature statistics are stored.

Naive Bayes text classification can become expensive when:

- the vocabulary is large
- the number of classes is large
- documents are long
- training data is large

Sparse data structures such as maps can reduce unnecessary storage for absent features.

For numerical stability, logarithmic calculations can prevent underflow and make large-scale models more reliable.

---

## Security considerations

Bayesian inference can be used in security systems, but probabilistic scoring does not eliminate security risks.

Relevant concerns include:

- manipulated input evidence
- adversarial observations
- poisoned training data
- incorrect priors
- stale likelihood estimates
- model extraction
- privacy leakage
- excessive trust in automated scores
- threshold manipulation
- correlated attack signals

A fraud or security model should therefore include input validation, monitoring, access controls, auditability, model versioning, and mechanisms for detecting changes in data distributions.

The C++ case study demonstrates validation and explicit decision costs, but a production fraud system would require substantially more controls.

---

## Implementation considerations

A practical Bayesian system should make its assumptions explicit.

Important configuration items include:

- definition of the hypothesis
- definition of each evidence variable
- source of priors
- source of likelihood estimates
- independence assumptions
- missing-data behavior
- numerical precision
- normalization method
- update frequency
- decision costs
- monitoring metrics
- calibration procedure
- model version

The probability calculation should be separated from the operational action whenever possible. This makes the system easier to test and audit.

---

## Probabilistic calibration

A model can rank cases correctly while still producing poorly calibrated probabilities.

For example, among cases assigned a probability near `0.70`, a well-calibrated model should produce the event approximately `70%` of the time over a sufficiently large representative population.

The Python and JavaScript examples include the Brier score:

`Brier Score = mean((p-y)^2)`

where:

- `p` is the predicted probability
- `y` is the observed binary outcome

Lower Brier scores indicate better probabilistic prediction under this metric.

Calibration should be evaluated on data that represents the intended deployment population.

---

## Real-world applications

### Medical diagnostics

A prior prevalence can be combined with test sensitivity and specificity to estimate the probability of a condition after a test result.

### Spam filtering

Email features can be used to estimate:

`P(Spam | Words, Metadata)`

Naive Bayes is historically important in this area.

### Fraud detection

Transaction attributes can update the probability of fraudulent behavior.

The C++ case study demonstrates this use case.

### Cybersecurity

Evidence such as login anomalies, device changes, network behavior, and location changes can contribute to probabilistic risk models.

### Reliability engineering

Observed failures and sensor measurements can update beliefs about component failure states.

### Scientific inference

Experimental evidence can update probabilities assigned to competing hypotheses.

### Robotics

Sensor observations can update beliefs about position, objects, or environmental states.

### Finance

Bayesian models can be used for uncertainty estimation, regime inference, credit risk, and other probabilistic tasks.

The quality of these applications depends on the quality of their probability models and data.

---

## Implementation comparison

| Aspect | Python | JavaScript | C++ |
|---|---|---|---|
| Basic Bayes calculations | Direct and compact | Direct and application-oriented | Strongly typed implementation |
| Multiple hypotheses | Dataclasses and lists | Objects and arrays | Structs and vectors |
| Naive Bayes | Complete text classifier | Complete text classifier | Focuses on fraud case study |
| Numerical stability | Log-space functions | Logarithmic classifier scores | Explicit log-sum-exp |
| Asynchronous behavior | Not central | Promises and `async`/`await` | Not central |
| Domain modeling | Python classes | JavaScript classes | Classes, enums, structs |
| Error handling | Exceptions | Exceptions and rejected Promises | Standard C++ exceptions |
| Decision theory | Expected-loss model | Expected-loss model | Integrated into fraud case study |
| Continuous likelihoods | Normal density | Normal density | Log-space focus |
| Primary educational role | Mathematical and statistical exploration | Application and language behavior | Systems-oriented case study |

---

## Formula reference

### Conditional probability

`P(A|B) = P(A and B) / P(B)`

### Multiplication rule

`P(A and B) = P(A)P(B|A)`

### Law of total probability

`P(E) = sum P(E|Hi)P(Hi)`

### Bayes' theorem

`P(H|E) = P(E|H)P(H)/P(E)`

### Binary evidence denominator

`P(E) = P(E|H)P(H) + P(E|not H)P(not H)`

### Odds

`Odds(H) = P(H)/(1-P(H))`

### Likelihood ratio

`LR = P(E|H)/P(E|not H)`

### Bayesian odds

`Posterior Odds = Prior Odds × Likelihood Ratio`

### Posterior proportionality

`P(H|E) proportional to P(E|H)P(H)`

### Brier score

`Brier Score = mean((p-y)^2)`

---

## File structure

The four deliverables form a coherent study set:

- The Python script provides the broadest mathematical and statistical treatment.
- The JavaScript file demonstrates Bayesian calculations in an application-oriented runtime with classes, collections, asynchronous execution, and classification.
- The C++ program develops a complete fraud-detection case study with domain structures, Bayesian scoring, decision theory, validation, numerical stability, and complexity analysis.
- This README explains the concepts represented by the implementations and connects the individual examples to practical Bayesian reasoning.
