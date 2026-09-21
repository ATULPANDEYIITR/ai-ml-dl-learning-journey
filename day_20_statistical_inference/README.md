# Statistical inference

Statistical inference is the process of using information from a sample to draw conclusions about a larger population. The central problem is that the complete population is usually unknown or too expensive to observe. Instead, researchers collect a sample, calculate statistics from that sample, and quantify the uncertainty involved in extending those results to the population.

This study focuses on four closely connected areas:

- sampling
- confidence intervals
- hypothesis testing
- p-values

The implementations also demonstrate the Central Limit Theorem, Student's t distribution, one-sample and two-sample tests, proportion inference, effect sizes, bootstrap methods, permutation tests, statistical power, multiple-testing correction, validation, simulation, and practical quality-control analysis.

## Statistical inference versus descriptive statistics

Descriptive statistics describe observed data.

Examples include:

- mean
- median
- variance
- standard deviation
- minimum
- maximum
- proportions
- correlations

Inferential statistics go beyond the observed sample. They use probability models to make statements about an underlying population or data-generating process.

For example, suppose a factory measures the processing time of 25 production batches. The average of those 25 observations is a descriptive statistic. If the factory uses those observations to estimate the average processing time for all comparable batches, the analysis becomes inferential.

The distinction is important because inferential conclusions contain uncertainty.

## Population, sample, parameter, and statistic

A population is the complete collection of units relevant to a research question.

A sample is the subset actually observed.

A parameter is a numerical property of a population.

Common parameters include:

- population mean, written as `μ`
- population proportion, written as `p`
- population variance, written as `σ²`
- population standard deviation, written as `σ`

A statistic is calculated from a sample.

Common statistics include:

- sample mean, written as `x̄`
- sample proportion, written as `p̂`
- sample variance, written as `s²`
- sample standard deviation, written as `s`

The parameter is generally unknown. The statistic is observable.

The fundamental inferential problem is therefore:

`sample data → statistic → probability model → inference about population parameter`

## Sampling

Sampling determines which observations enter an analysis.

A statistical calculation can be mathematically correct and still produce misleading conclusions if the sample is systematically unrepresentative.

### Simple random sampling

In simple random sampling, each population member has an equal selection probability under the sampling design.

The Python implementation uses `random.sample()` to demonstrate sampling without replacement.

The C++ implementation creates a vector representing a production population and uses `std::shuffle()` to obtain a random sample.

### Systematic sampling

Systematic sampling selects observations according to a regular interval after an appropriate starting point.

For example, a study might select every 20th record from an ordered population.

Systematic sampling can be efficient, but periodic structure in the population can create problems if the sampling interval interacts with an underlying pattern.

### Stratified sampling

Stratified sampling divides a population into meaningful subgroups called strata and samples within each stratum.

For example, an organization might divide customers into geographic regions and sample each region separately.

Stratification can improve representation of important subgroups and can increase precision when strata are internally homogeneous.

### Cluster sampling

Cluster sampling selects naturally occurring groups rather than independently selecting individuals throughout the population.

Examples include:

- selecting schools and surveying students within selected schools
- selecting hospitals and studying patients within selected hospitals
- selecting manufacturing sites and studying production batches within selected sites

Cluster sampling can reduce collection cost, but observations within the same cluster may be correlated. This reduces the amount of independent information provided by a given number of observations.

## Sampling bias

Sampling bias occurs when the sampling mechanism systematically makes some population members more or less likely to enter the sample in a way related to the target quantity.

Examples include:

- voluntary-response bias
- undercoverage
- nonresponse bias
- convenience sampling
- survivorship bias

Increasing the sample size does not automatically eliminate systematic sampling bias.

A million biased observations can still provide a biased estimate.

This is one of the most important distinctions in statistical inference.

## Sampling variability

Even when sampling is properly designed, different random samples produce different statistics.

Suppose the true population mean is `μ`. One sample might produce:

`x̄ = 98.7`

Another might produce:

`x̄ = 101.2`

Another might produce:

`x̄ = 99.8`

This natural variation is sampling variability.

Statistical inference quantifies that uncertainty rather than pretending that a sample statistic is exactly equal to the population parameter.

## Standard deviation and standard error

Standard deviation describes variability among individual observations.

For a sample:

`s = sqrt[Σ(xᵢ - x̄)² / (n - 1)]`

Standard error describes variability of an estimator across repeated samples.

For a sample mean:

`SE(x̄) = s / sqrt(n)`

The distinction is essential.

If individual observations are highly variable, the standard deviation can be large. Increasing the sample size reduces the standard error of the mean even though it does not necessarily reduce the standard deviation of individual observations.

The Python and JavaScript implementations explicitly calculate both quantities.

## Why n - 1 appears in sample variance

When estimating population variance from a sample, the conventional unbiased estimator uses:

`s² = Σ(xᵢ - x̄)² / (n - 1)`

The subtraction of one reflects the loss of one degree of freedom caused by estimating the population mean using the same sample.

The sample deviations satisfy:

`Σ(xᵢ - x̄) = 0`

Therefore, once `n - 1` deviations are known, the final deviation is determined.

## The Central Limit Theorem

The Central Limit Theorem is one of the foundations of classical statistical inference.

Under appropriate conditions, the distribution of sample means becomes approximately normal as sample size increases, even when the underlying population is not normally distributed.

If the population has mean `μ` and finite standard deviation `σ`, then the sampling distribution of the sample mean is approximately:

`x̄ ≈ Normal(μ, σ / sqrt(n))`

The Python implementation constructs a strongly skewed population and repeatedly samples from it. It then examines the resulting distribution of sample means.

The JavaScript implementation performs the same type of simulation using a reproducible custom random-number generator.

The C++ implementation uses standard-library random facilities in its power and sampling simulations.

The Central Limit Theorem does not state that every dataset becomes normal merely because its sample size is large. It concerns the distribution of an estimator under repeated sampling and has assumptions and regularity conditions.

## Confidence intervals

A confidence interval provides an interval estimate produced by a procedure designed to have a specified long-run coverage probability.

A general form is:

`estimate ± margin of error`

The margin of error usually has the structure:

`critical value × standard error`

For a normal-based mean interval with known population standard deviation:

`x̄ ± z* σ / sqrt(n)`

When the population standard deviation is estimated from the sample, the Student's t distribution is commonly used:

`x̄ ± t* s / sqrt(n)`

The Python implementation provides both z-based and t-based mean intervals.

The JavaScript implementation provides known-standard-deviation and Student's t intervals.

The C++ case study uses a Student's t interval for the manufacturing process.

## Interpreting a 95% confidence interval

A common but incorrect interpretation is:

"There is a 95% probability that the fixed population mean is inside this particular interval."

Under the frequentist framework used here, the parameter is treated as fixed while the interval-producing procedure is random.

The appropriate interpretation is based on repeated sampling:

If the same sampling and interval-construction procedure were repeated many times under the same conditions, approximately 95% of the resulting intervals would contain the true parameter.

The confidence level describes the procedure, not a probability distribution assigned to the already-fixed parameter.

## Confidence level and interval width

Higher confidence generally requires a wider interval.

For a fixed sample and model:

- 90% confidence produces a narrower interval
- 95% confidence produces a wider interval
- 99% confidence produces an even wider interval

Higher confidence requires a larger critical value.

Larger samples generally reduce standard error and therefore produce narrower intervals, assuming other factors remain fixed.

## Margin of error

For a known population standard deviation:

`E = z* σ / sqrt(n)`

This equation shows several important relationships.

Increasing `n` reduces the margin of error.

Increasing `σ` increases the margin of error.

Increasing the confidence level increases the critical value and therefore increases the margin of error.

The relationship with sample size is proportional to `1 / sqrt(n)`, so reducing the margin of error substantially can require a much larger sample.

## Confidence interval for a proportion

For a binary outcome, the population parameter is a proportion `p`.

The observed proportion is:

`p̂ = x / n`

A simple Wald interval is:

`p̂ ± z* sqrt[p̂(1-p̂)/n]`

The Python and JavaScript implementations include this formula.

The Wald interval is useful for understanding the underlying mathematics but can perform poorly with small samples or proportions close to zero or one. In production analysis, alternative interval procedures such as Wilson or exact methods may be preferable depending on the application.

## Hypothesis testing

Hypothesis testing evaluates evidence concerning a specified null model.

A typical test has:

- a null hypothesis `H0`
- an alternative hypothesis `HA`
- a test statistic
- a reference distribution
- a significance level `α`
- a p-value
- a decision rule

A simple mean test might use:

`H0: μ = μ0`

against:

`HA: μ ≠ μ0`

or a one-sided alternative:

`HA: μ > μ0`

or:

`HA: μ < μ0`

The null hypothesis is not necessarily a claim that nothing is happening in an absolute philosophical sense. It is a specific statistical model or parameter restriction used to construct the reference distribution.

## Two-sided and one-sided hypotheses

A two-sided alternative asks whether the parameter differs in either direction.

Example:

`H0: μ = 50`

`HA: μ ≠ 50`

A one-sided alternative specifies a direction.

Example:

`H0: μ = 50`

`HA: μ < 50`

The alternative must be selected based on the scientific or operational question, not chosen after observing the data simply to obtain a smaller p-value.

## Test statistic

A test statistic measures how far the observed result is from what the null hypothesis predicts, relative to an appropriate measure of sampling variability.

For a one-sample t test:

`t = (x̄ - μ0) / (s / sqrt(n))`

A large absolute value of `t` indicates that the sample mean is far from the null value relative to its estimated standard error.

## Student's t distribution

When population standard deviation is unknown and estimated from the sample, the test statistic follows a Student's t distribution under the classical normal model.

The t distribution has heavier tails than the standard normal distribution.

Its shape depends on degrees of freedom.

As degrees of freedom increase, the t distribution approaches the standard normal distribution.

For a one-sample t test:

`df = n - 1`

The Python, JavaScript, and C++ implementations explicitly implement numerical t-distribution calculations rather than depending on external statistical packages.

## One-sample t test

The one-sample t test evaluates whether a population mean differs from a specified reference value.

The Python implementation uses:

`one_sample_t_test()`

The JavaScript implementation uses:

`oneSampleTTest()`

The C++ implementation uses:

`HypothesisTests::oneSampleT()`

All three implementations calculate the sample mean, standard error, t statistic, reference distribution, p-value, and decision.

## Independent two-sample tests

Suppose two independent groups are being compared.

The null hypothesis might be:

`H0: μ₁ = μ₂`

The observed difference is:

`x̄₁ - x̄₂`

An independent two-sample t procedure evaluates whether the observed difference is unusually large relative to expected sampling variation under the null model.

## Pooled versus Welch t tests

A pooled two-sample t test assumes equal population variances.

Its pooled variance is:

`sp² = [(n₁ - 1)s₁² + (n₂ - 1)s₂²] / (n₁ + n₂ - 2)`

Welch's t test does not require equal population variances.

Welch's standard error is based on:

`sqrt(s₁²/n₁ + s₂²/n₂)`

and the degrees of freedom are approximated using the Welch-Satterthwaite equation.

The Python and JavaScript implementations use Welch's test for the practical two-group comparison.

The C++ case study also uses Welch's test because unequal variability is often plausible in operational data.

## Paired observations

Paired data occur when observations naturally correspond.

Examples include:

- before and after measurements on the same individual
- two measurements from the same machine
- matched subjects
- repeated measurements on the same unit

The paired t test transforms each pair into a difference:

`dᵢ = afterᵢ - beforeᵢ`

It then performs a one-sample t test on the differences:

`H0: μd = 0`

The Python implementation includes `paired_t_test()`.

Pairing can remove variability associated with differences between matched units and can increase statistical efficiency when the pairing is appropriate.

## P-values

A p-value is calculated under the null model.

It measures the probability, under that null model, of obtaining a test statistic at least as extreme as the observed statistic in the direction defined by the alternative hypothesis.

For a two-sided test, the calculation considers extreme results in both directions.

A p-value is not:

- the probability that the null hypothesis is true
- the probability that the alternative hypothesis is true
- the probability that the result occurred by chance
- the size of the effect
- the probability that a replication will succeed

These distinctions are essential.

## Significance level

The significance level `α` is a pre-specified threshold used to control a Type I error rate under the assumptions of the testing procedure.

A common value is:

`α = 0.05`

The conventional decision rule is:

`p < α → reject H0`

`p >= α → do not reject H0`

The second result should not be described as "accepting H0" without additional justification. It means that the specified test did not provide sufficient evidence to reject the null under the selected criterion.

## Statistical significance versus practical significance

A very large sample can make a very small effect statistically significant.

For example, a difference of a fraction of a unit might produce a very small p-value if millions of observations are available.

Conversely, a practically important difference may fail to achieve statistical significance when the sample is small or highly variable.

Therefore, statistical inference should consider:

- effect size
- confidence intervals
- practical consequences
- domain-specific thresholds
- uncertainty
- study design

The Python and C++ implementations calculate Cohen's d to illustrate this distinction.

## Cohen's d

Cohen's d standardizes a difference between two means.

For independent groups using a pooled standard deviation:

`d = (x̄₁ - x̄₂) / sp`

Unlike a p-value, Cohen's d is intended to describe the magnitude of a difference on a standardized scale.

Its practical interpretation depends on the field and measurement context.

A universal numerical threshold should not be treated as a substitute for domain knowledge.

## Type I error

A Type I error occurs when a true null hypothesis is rejected.

In simplified terminology:

`Type I error = false positive`

If a procedure has a nominal significance level of 0.05 and its assumptions are satisfied, the long-run Type I error rate is controlled around 5% for the specified testing scenario.

The Python implementation simulates repeated samples under the null hypothesis and estimates the observed rejection rate.

## Type II error

A Type II error occurs when a false null hypothesis is not rejected.

In simplified terminology:

`Type II error = false negative`

The probability of a Type II error is commonly represented by `β`.

Statistical power is:

`Power = 1 - β`

## Statistical power

Power is the probability that a test rejects the null hypothesis under a specified alternative condition.

Power generally increases when:

- the true effect becomes larger
- the sample size increases
- measurement noise decreases
- the significance level increases

Power generally decreases when:

- the true effect becomes smaller
- the sample size decreases
- measurement variability increases
- the significance threshold becomes more stringent

The Python, JavaScript, and C++ implementations simulate power.

Power is always relative to a specified alternative. A statement such as "the test has 80% power" is incomplete unless the effect size, variability, sample size, significance level, and testing procedure are understood.

## Sample size for a mean

When population standard deviation is known or reasonably estimated, an approximate sample-size calculation for a desired margin of error is:

`n = (z*σ/E)²`

where:

- `n` is the required sample size
- `z*` is the critical value
- `σ` is the population standard deviation
- `E` is the desired margin of error

The Python implementation rounds the result upward because a sample size cannot be fractional.

## Sample size for a proportion

A common approximation is:

`n = z² p(1-p) / E²`

When no prior estimate of `p` is available, `p = 0.5` is often used because it maximizes:

`p(1-p)`

This gives a conservative sample-size requirement under the corresponding assumptions.

## Chi-square test of independence

Categorical variables can be analyzed using contingency tables.

For observed count `O` and expected count `E`, the Pearson statistic is:

`χ² = Σ (O-E)²/E`

For a two-dimensional table, the expected count for cell `(i,j)` is:

`Eᵢⱼ = row total × column total / grand total`

The Python implementation calculates the chi-square statistic and demonstrates a two-by-two p-value calculation.

Chi-square methods have assumptions concerning expected frequencies and independence. For very small expected counts, alternative procedures may be more appropriate.

## Correlation

Pearson correlation measures linear association between two quantitative variables.

Its value lies between:

`-1 ≤ r ≤ 1`

Values near `1` indicate strong positive linear association.

Values near `-1` indicate strong negative linear association.

Values near `0` indicate weak linear association, although a value near zero does not establish that no relationship exists.

Correlation does not establish causation.

Confounding, selection effects, reverse causation, nonlinear relationships, and measurement problems can all affect interpretation.

The Python implementation calculates Pearson correlation from first principles.

## Bootstrap inference

The bootstrap uses the observed sample as an empirical approximation to the population.

A bootstrap replicate is generated by sampling with replacement from the observed sample.

The process is repeated many times.

For a mean:

`observed sample → resample with replacement → calculate mean → repeat`

The resulting empirical distribution approximates the sampling distribution of the statistic.

A percentile confidence interval can then be obtained from the empirical bootstrap distribution.

Bootstrap methods are flexible but are not universally valid. Poor samples, dependence, boundary problems, very small samples, and inappropriate resampling schemes can produce misleading results.

## Permutation tests

Permutation tests use rearrangements of observed data under a null hypothesis concerning exchangeability.

For a two-group difference:

1. calculate the observed difference
2. pool the observations
3. randomly reassign observations to groups
4. calculate the difference again
5. repeat many times
6. count results at least as extreme as the observed difference

The resulting proportion estimates a randomization-based p-value.

The Python, JavaScript, and C++ implementations demonstrate this approach.

Permutation testing is especially useful when an analytical reference distribution is inconvenient, provided the exchangeability assumptions of the permutation scheme are appropriate.

## Multiple comparisons

Testing many hypotheses increases the opportunity for false positives.

If many independent tests are performed at `α = 0.05`, the probability of obtaining at least one false positive can become much larger than 5%.

The Bonferroni correction controls the familywise error rate by replacing the threshold with:

`α / m`

where `m` is the number of tests.

Equivalently, each p-value can be adjusted approximately as:

`adjusted p = min(mp, 1)`

The Python and JavaScript implementations demonstrate Bonferroni adjustment.

Other multiple-testing procedures exist and may provide greater power under specific objectives.

## Bootstrap versus permutation testing

Bootstrap and permutation methods are related computational techniques but answer different questions.

Bootstrap:

- samples with replacement
- approximates a sampling distribution
- is often used for confidence intervals
- can estimate uncertainty for statistics without simple analytical formulas

Permutation testing:

- rearranges labels or observations
- represents a null model based on exchangeability
- is often used for hypothesis testing
- generates a null distribution of a test statistic

They should not be treated as interchangeable procedures.

## Python implementation

The Python script is the broadest educational implementation.

It includes:

- population and sample demonstrations
- simple random sampling
- systematic sampling
- stratified sampling
- cluster sampling
- mean
- variance
- standard deviation
- standard error
- normal density
- normal CDF
- normal quantile
- Student's t density
- Student's t CDF
- Student's t quantile
- z confidence intervals
- t confidence intervals
- proportion confidence intervals
- one-sample z tests
- one-sample t tests
- pooled two-sample t tests
- Welch two-sample t tests
- paired t tests
- one-proportion tests
- chi-square calculations
- Pearson correlation
- Cohen's d
- bootstrap confidence intervals
- permutation testing
- Type I error simulation
- statistical power simulation
- Bonferroni correction
- sample-size calculations
- edge-case handling
- practical manufacturing analysis

Python is particularly convenient for statistical education because lists, functions, classes, random simulation, and numerical operations can be expressed concisely.

The implementation intentionally avoids external packages so that the underlying calculations remain visible.

## JavaScript implementation

The JavaScript implementation complements the Python version by emphasizing executable application-oriented numerical code.

It demonstrates:

- arrays
- classes
- reusable functions
- input validation
- deterministic random-number generation
- Fisher-Yates shuffling
- Box-Muller normal simulation
- normal distribution calculations
- Student's t calculations
- confidence intervals
- hypothesis tests
- Welch's test
- Cohen's d
- proportion inference
- bootstrap inference
- permutation testing
- power simulation
- multiple-testing correction
- a production-style case study

JavaScript is especially relevant when statistical inference is integrated into browser applications, dashboards, client-side analytics, interactive data tools, or Node.js services.

The implementation does not require npm packages.

## C++ case study

The C++ implementation models a manufacturing quality-control system.

Each production batch is represented by:

`QualityRecord`

The record contains:

- batch ID
- processing time
- inspection result

The `QualityControlSystem` class stores records and calculates operational metrics.

The system validates:

- batch identifiers
- positive processing times
- finite numerical values
- empty datasets

The statistical engine is separated into components.

`Statistics` provides:

- mean
- sample variance
- sample standard deviation
- standard error
- normal CDF
- normal quantile
- gamma function
- t density
- t CDF
- t quantile

`ConfidenceIntervals` provides analytical confidence intervals.

`HypothesisTests` provides one-sample and Welch tests.

The program also includes bootstrap inference, permutation testing, random sampling, effect size, and power simulation.

This separation demonstrates modular design: the operational data model is not tightly coupled to the numerical procedures.

## Manufacturing problem being solved

The simulated manufacturing process historically has a mean processing time of 50 minutes.

A process improvement is introduced.

A sample of processing times is observed.

The operational question is:

Has the average processing time decreased?

The statistical hypotheses are:

`H0: μ = 50`

`HA: μ < 50`

The one-sided alternative is appropriate for the specific operational question because the direction of interest is a decrease.

The program calculates:

- sample mean
- sample standard deviation
- standard error
- 95% confidence interval
- one-sample t statistic
- p-value
- decision relative to `α = 0.05`

## C++ design decisions

The C++ case study uses classes where they provide a meaningful separation of responsibilities.

The statistical calculations are static methods because they do not require persistent object state.

The quality-control system stores persistent records because it represents an actual analytical data source.

`std::vector` is used for observations because it provides contiguous storage and efficient iteration.

`std::mt19937` is used for reproducible pseudo-random simulations.

`std::shuffle` supports random permutation and sampling.

The implementation uses exceptions for invalid input and catches failures at the program boundary.

This structure provides a basic separation between:

- data
- statistical algorithms
- confidence intervals
- hypothesis testing
- reporting
- application logic

## Computational complexity

For `n` observations:

Mean calculation is:

`O(n)`

Variance calculation is:

`O(n)`

Standard deviation is:

`O(n)`

A single-pass calculation can reduce memory overhead compared with approaches that construct multiple intermediate arrays.

Sorting bootstrap statistics is:

`O(B log B)`

where `B` is the number of bootstrap repetitions.

Generating bootstrap samples requires approximately:

`O(Bn)`

where `n` is the original sample size.

Permutation testing is also approximately:

`O(Bn)` for `B` permutations.

Monte Carlo power simulation is approximately:

`O(Rn)`

where `R` is the number of simulation repetitions.

The numerical integration used to approximate the t distribution adds computational cost. Production statistical systems generally use specialized numerical implementations rather than repeatedly integrating a density from scratch.

## Numerical precision

Floating-point arithmetic is approximate.

Operations involving:

- very large numbers
- very small numbers
- subtraction of nearly equal values
- repeated numerical integration
- tail probabilities

can introduce rounding error.

The educational implementations use `double` in C++ and JavaScript's standard floating-point number type.

Python's floating-point arithmetic also uses binary floating-point for ordinary numerical values.

Production statistical software should use validated numerical libraries when high numerical accuracy is important, particularly for extreme tail probabilities and specialized distributions.

## Why the implementations calculate distributions directly

The educational scripts implement normal and Student's t calculations instead of hiding them behind external statistical packages.

This makes the mechanics visible:

- density evaluation
- cumulative probability
- inverse quantiles
- test statistics
- p-value construction

The approach is appropriate for learning and demonstrations.

It is not automatically the best implementation strategy for production statistical software.

Validated numerical libraries generally provide broader distribution coverage, stronger numerical testing, better tail handling, and more specialized algorithms.

## Common mistakes

### Treating a sample as the population

A sample statistic is not automatically the exact population parameter.

### Ignoring sampling design

Statistical formulas cannot automatically correct a fundamentally biased sample.

### Confusing standard deviation with standard error

Standard deviation describes individual observations.

Standard error describes uncertainty in an estimator.

### Treating a p-value as a probability that the null is true

A p-value is calculated conditional on the null model.

### Treating p < 0.05 as proof of an important effect

Statistical significance and practical significance are different concepts.

### Treating p >= 0.05 as proof of no effect

A non-significant result can arise from low power, high noise, small sample size, or a genuinely small effect.

### Changing the hypothesis after seeing the data

Choosing one-sided versus two-sided testing after observing the direction of the result can invalidate the intended Type I error interpretation.

### Ignoring multiple testing

Testing many hypotheses without adjustment can substantially increase false-positive risk.

### Assuming correlation means causation

Association alone does not establish a causal relationship.

### Ignoring dependence

Repeated measurements, clustered observations, family members, time-series observations, and matched observations may violate independence assumptions.

## Edge cases

The implementations explicitly handle several edge cases.

Empty samples are rejected.

Variance requires at least two observations.

A constant sample can have zero standard deviation.

When the standard error is zero, ordinary t-statistic calculations require special handling because division by zero occurs.

Proportion calculations validate:

`0 <= successes <= trials`

and require a positive number of trials.

Confidence levels must lie between zero and one.

Sample sizes must be positive.

Numerical routines validate probabilities before calculating quantiles.

These checks prevent invalid input from silently producing meaningless results.

## Assumptions

Statistical inference depends on assumptions.

Important assumptions include:

- appropriate sampling
- independence
- correct specification of the statistical model
- suitable distributional assumptions
- correct measurement
- adequate sample size
- appropriate variance assumptions when required

The exact assumptions depend on the method.

For example, Welch's t test does not require equal population variances, while the pooled t test does.

A paired t test requires the differences to satisfy the relevant assumptions rather than treating all before and after measurements as independent.

## Random sampling versus random assignment

These concepts should not be confused.

Random sampling concerns how observations are selected from a population.

It primarily supports statistical generalization from the sample to the population.

Random assignment concerns how experimental units are allocated to treatment conditions.

It primarily supports causal inference by helping balance confounding factors across groups.

A study can have random assignment without random sampling, or random sampling without random assignment.

The two concepts address different inferential goals.

## Statistical inference and causality

A statistical test can identify evidence of an association or difference without establishing a causal mechanism.

Causal conclusions require an appropriate research design and assumptions.

Randomized experiments can provide strong causal identification when properly designed and conducted.

Observational analyses may require additional methods and assumptions concerning confounding, selection, measurement, and treatment assignment.

## Security and production considerations

Statistical analysis can affect operational decisions, financial decisions, quality-control decisions, medical research, and public policy.

Production systems should therefore consider:

- input validation
- reproducible random seeds when simulations must be audited
- controlled data access
- protection of sensitive records
- versioned analysis code
- reproducible datasets
- logging of analysis parameters
- explicit handling of missing values
- numerical testing
- unit tests
- validation against trusted statistical software
- documented assumptions
- review of model changes
- protection against accidental data leakage

Statistical correctness is only one part of a production analytical system.

## Interpretation of the manufacturing case study

The manufacturing example illustrates the complete inferential workflow.

The observed sample is first described.

The sample mean estimates the population mean.

The standard error quantifies expected sampling variability.

The confidence interval describes uncertainty around the estimated mean using the selected interval procedure.

The hypothesis test evaluates the evidence against a specified historical benchmark.

The p-value describes how extreme the observed test statistic would be under the null model.

Cohen's d provides an effect-size perspective.

Bootstrap inference provides a resampling-based uncertainty estimate.

Permutation testing provides a simulation-based null distribution.

Power simulation demonstrates how test sensitivity changes under a specified alternative.

The different procedures answer related but distinct questions and should not be collapsed into a single number.

## Practical interpretation framework

A sound inferential analysis should consider the complete set of evidence:

1. How was the sample obtained?
2. What population does the sample represent?
3. What parameter is being estimated?
4. What is the point estimate?
5. What is the uncertainty around the estimate?
6. What null hypothesis is being tested?
7. Was the alternative hypothesis specified appropriately?
8. What is the p-value?
9. What is the effect size?
10. Is the effect practically meaningful?
11. What assumptions support the procedure?
12. Were multiple hypotheses tested?
13. Could dependence or selection bias affect the result?
14. Does the research design support the intended conclusion?

These questions help distinguish a mathematically calculated result from a statistically justified inference.

## Python, JavaScript, and C++ comparison

| Aspect | Python | JavaScript | C++ |
|---|---|---|---|
| Educational statistical calculations | Strong | Strong | Strong |
| Concise numerical code | High | High | Moderate |
| Browser integration | Limited without additional tooling | Strong | Limited |
| Systems-level control | Moderate | Limited | Strong |
| Numerical simulation | Strong | Strong | Strong |
| Memory control | Moderate | Managed | Strong |
| Production analytical services | Strong | Strong | Strong |
| Visualization ecosystem | Extensive with external packages | Strong in web environments | More specialized |
| Low-level performance control | Moderate | Limited | Strong |

The statistical concepts do not fundamentally change between languages.

The implementation style does.

Python emphasizes readability and rapid analytical development.

JavaScript emphasizes application integration and interactive environments.

C++ emphasizes explicit data structures, compiled execution, memory control, and systems-oriented design.

## Important distinctions

### Parameter versus statistic

A parameter describes the population.

A statistic describes the sample.

### Standard deviation versus standard error

Standard deviation describes observation-level variability.

Standard error describes estimator-level sampling variability.

### Confidence interval versus prediction interval

A confidence interval estimates a population parameter.

A prediction interval concerns a future observation and generally contains additional individual-level variability.

### Statistical significance versus practical significance

Statistical significance concerns compatibility with a null model.

Practical significance concerns the size and consequences of an effect.

### Association versus causation

Association describes statistical dependence.

Causation requires a stronger design or set of assumptions.

### Sampling versus assignment

Sampling concerns selection from a population.

Assignment concerns allocation to experimental conditions.

### Bootstrap versus permutation

Bootstrap resampling approximates sampling variability.

Permutation testing commonly constructs a null distribution through rearrangement.

## Limitations of the implementations

These programs are designed for learning and transparent demonstration.

They do not attempt to replace mature statistical libraries.

The t-distribution calculations use numerical integration and binary search, which are computationally expensive compared with specialized algorithms.

The proportion example uses the simple Wald interval, which has known limitations.

The chi-square example focuses on a two-by-two table.

The bootstrap uses the percentile method rather than more advanced bootstrap intervals.

The simulations use pseudo-random numbers and therefore approximate theoretical quantities.

The programs do not automatically diagnose all assumptions.

Real datasets may require handling of:

- missing observations
- censored observations
- repeated measures
- hierarchical structure
- time dependence
- nonresponse
- survey weights
- measurement error
- complex sampling designs
- multiple imputation
- nonparametric procedures
- regression models
- generalized linear models

Those issues require methods appropriate to the particular research design and data-generating process.

## Best practices

Use a sampling design that matches the target population.

Define the research question before inspecting the outcome.

Specify hypotheses and analysis rules before testing whenever possible.

Use the correct standard error.

Choose a test whose assumptions match the study design.

Prefer Welch's test when equal variances cannot reasonably be assumed for independent groups.

Use paired methods for genuinely paired observations.

Report effect sizes and confidence intervals alongside p-values.

Consider practical importance rather than relying only on a significance threshold.

Account for multiple comparisons when many hypotheses are tested.

Use simulation or bootstrap methods when they are appropriate to the research design.

Validate numerical implementations against trusted references before deploying them in high-stakes applications.

Document random seeds and simulation parameters when reproducibility matters.

## Real-world applications

Statistical inference is used across many domains.

### Manufacturing

- process improvement
- defect-rate estimation
- quality control
- reliability analysis
- production-time comparison

### Finance

- return estimation
- risk analysis
- event studies
- model validation
- portfolio comparisons

### Healthcare research

- treatment comparisons
- clinical trial analysis
- risk estimation
- diagnostic evaluation
- population studies

### Product analytics

- A/B testing
- conversion-rate analysis
- customer behavior
- retention analysis
- experiment evaluation

### Public policy

- survey estimation
- population characteristics
- program evaluation
- socioeconomic studies

### Engineering

- reliability testing
- process monitoring
- sensor validation
- performance comparisons

## Reproducibility

Simulation-based inference depends on random sampling.

The implementations use fixed seeds for educational reproducibility.

For example, the Python code uses:

`random.Random(42)`

The JavaScript implementation defines a seeded pseudo-random generator.

The C++ implementation uses:

`std::mt19937 generator(42);`

A fixed seed makes a simulation reproducible when the same algorithm, implementation, and parameters are used.

Reproducibility is valuable for:

- debugging
- testing
- educational demonstrations
- auditing
- research workflows

A seed does not make the underlying statistical model valid. It only makes the computational random sequence repeatable.

## Final implementation map

The Python implementation provides the broadest collection of statistical demonstrations.

The JavaScript implementation emphasizes reusable application-oriented numerical functions, deterministic simulation, and executable client/server-compatible logic.

The C++ implementation combines statistical methods with an industry-style quality-control architecture.

Together, the three implementations demonstrate that statistical inference is not simply the calculation of a p-value. It is a complete process involving study design, sampling, measurement, estimation, uncertainty, hypothesis formulation, probability models, computation, interpretation, and responsible reporting.
