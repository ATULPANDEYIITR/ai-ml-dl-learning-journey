# Probability distributions: Bernoulli, Binomial, Poisson, Gaussian, Uniform, Exponential

## Topic overview

Probability distributions provide mathematical models for random variables. A distribution describes which values a random variable can take and how probability is allocated among those values.

This study covers six important distributions:

| Distribution | Type | Main parameter(s) | Typical interpretation |
|---|---|---|---|
| Bernoulli | Discrete | `p` | One binary trial |
| Binomial | Discrete | `n`, `p` | Number of successes in fixed binary trials |
| Poisson | Discrete | `λ` | Number of events in an interval |
| Gaussian / Normal | Continuous | `μ`, `σ` | Symmetric continuous variation |
| Uniform | Continuous | `a`, `b` | Equal density across a bounded interval |
| Exponential | Continuous | `λ` | Waiting time between events |

The Python implementation develops the mathematical concepts and implements each distribution explicitly. The JavaScript implementation focuses on executable application-level demonstrations, deterministic simulation, numerical methods, and JavaScript-specific implementation patterns. The C++ implementation develops an industry-style service-monitoring case study using all six distributions in a coherent system model.

## Random variables

A random variable maps outcomes of a random experiment to numerical values.

A discrete random variable has a finite or countably infinite set of possible values.

Examples include:

- Number of failed requests
- Number of calls received
- Number of defective products
- Number of successes in ten trials

A continuous random variable can take values throughout an interval.

Examples include:

- Response latency
- Waiting time
- Temperature
- Measurement error

The distinction matters because discrete and continuous variables use probability differently.

For a discrete random variable, the probability of an individual value can be positive:

`P(X = k)`

For an ordinary continuous random variable:

`P(X = x) = 0`

Probability for a continuous variable is obtained over intervals, using the area under the probability density function.

## Probability mass function

A probability mass function, or PMF, applies to a discrete random variable.

It assigns a probability to each possible value:

`P(X = x)`

A valid PMF must satisfy:

`P(X = x) >= 0`

and

`sum P(X = x) = 1`

The Python and C++ implementations explicitly calculate PMFs for the discrete distributions.

## Probability density function

A probability density function, or PDF, applies to a continuous distribution.

A PDF value is not itself the probability of an exact point.

For an interval:

`P(a <= X <= b) = integral from a to b of f(x) dx`

A density may be greater than 1 when the interval is sufficiently narrow. That does not violate probability rules because probability is based on area, not simply the height of the density.

The Uniform, Exponential, and Gaussian implementations demonstrate this distinction.

## Cumulative distribution function

The cumulative distribution function is:

`F(x) = P(X <= x)`

For a discrete variable, the CDF is the sum of PMF values up to `x`.

For a continuous variable, it is the integral of the PDF from negative infinity to `x`.

The CDF is useful for threshold analysis:

`P(X <= x) = F(x)`

and:

`P(X > x) = 1 - F(x)`

The implementations use CDF calculations for capacity thresholds, response-time limits, failure counts, and waiting-time analysis.

## Expected value

The expected value is the long-run probability-weighted average of a random variable.

For a discrete variable:

`E[X] = sum x P(X=x)`

The expected value does not have to be an outcome that can actually occur.

For example, a Binomial random variable can have an expected value of `3.7`, even though an individual observation must be an integer.

## Variance and standard deviation

Variance measures the average squared distance from the mean:

`Var(X) = E[(X - E[X])²]`

Standard deviation is:

`SD(X) = sqrt(Var(X))`

Variance is measured in squared units. Standard deviation is measured in the original units.

For example, if response latency is measured in milliseconds, standard deviation is measured in milliseconds, while variance is measured in squared milliseconds.

---

# Bernoulli distribution

## Definition

The Bernoulli distribution models one experiment with exactly two possible outcomes.

Let:

`X ~ Bernoulli(p)`

where `p` is the probability of success.

The conventional encoding is:

- `X = 1` for success
- `X = 0` for failure

The PMF is:

`P(X=x) = p^x (1-p)^(1-x)`

for `x` equal to `0` or `1`.

## Mean

The Bernoulli mean is:

`E[X] = p`

## Variance

The variance is:

`Var(X) = p(1-p)`

The variance is largest when `p = 0.5` and becomes zero when `p = 0` or `p = 1`.

## Applications

Bernoulli models are appropriate for binary outcomes such as:

- Successful or failed transaction
- Defective or non-defective item
- Clicked or not clicked
- Passed or failed
- Available or unavailable
- Event occurred or did not occur

The important point is that Bernoulli describes one trial. A collection of Bernoulli trials leads naturally to the Binomial distribution when the required assumptions hold.

## Python implementation

The Python `BernoulliDistribution` class validates `p`, calculates the PMF and CDF, exposes the theoretical mean and variance, and provides a sampling method.

The implementation also handles the edge cases `p=0` and `p=1`.

## JavaScript implementation

The JavaScript `BernoulliDistribution` class provides equivalent probability calculations and accepts a seeded random generator for reproducible simulation.

## C++ implementation

The C++ `BernoulliDistribution` class uses the standard library's `std::bernoulli_distribution` for sampling while keeping the PMF, mean, and variance calculations explicit.

---

# Binomial distribution

## Definition

The Binomial distribution counts the number of successes in a fixed number of Bernoulli trials.

Let:

`X ~ Binomial(n,p)`

where:

- `n` is the number of trials
- `p` is the probability of success in each trial

The PMF is:

`P(X=k) = C(n,k) p^k (1-p)^(n-k)`

for:

`k = 0, 1, ..., n`

## Conditions for a Binomial model

A standard Binomial model requires:

- A fixed number of trials
- Two outcomes per trial
- The same success probability for each trial
- Independence between trials

If these assumptions are substantially violated, a Binomial model may not be appropriate.

## Mean

`E[X] = np`

## Variance

`Var(X) = np(1-p)`

## Example

Suppose each request has a failure probability of `0.02` and exactly `500` independent requests are examined.

Then:

`X ~ Binomial(500, 0.02)`

The expected number of failures is:

`500 × 0.02 = 10`

This does not mean exactly ten failures will occur. It is the population mean of the model.

## Numerical implementation

Directly computing:

`n! / (k!(n-k)!)`

can become numerically problematic for large values of `n`.

The Python, JavaScript, and C++ implementations therefore use logarithmic calculations for probability evaluation where appropriate.

Using:

`log(n!) = lgamma(n+1)`

avoids constructing enormous intermediate factorial values.

## Python implementation

The Python implementation includes:

- PMF
- CDF
- Probability of at least a specified number of successes
- Mean
- Variance
- Sampling
- Boundary handling
- Logarithmic probability calculation

## JavaScript implementation

The JavaScript implementation uses `BigInt` for exact integer combinations in its general combination utility, while probability calculations use logarithmic floating-point expressions.

This illustrates an important distinction between exact integer arithmetic and numerical probability calculations.

## C++ implementation

The C++ case study uses `std::binomial_distribution` for simulation while maintaining an explicit PMF implementation based on logarithmic gamma functions.

---

# Poisson distribution

## Definition

The Poisson distribution models event counts over a specified interval when an appropriate Poisson-process assumption is reasonable.

Let:

`X ~ Poisson(λ)`

The PMF is:

`P(X=k) = exp(-λ) λ^k / k!`

for:

`k = 0, 1, 2, ...`

## Mean and variance

The Poisson distribution has a distinctive relationship:

`E[X] = λ`

and:

`Var(X) = λ`

Consequently:

`SD(X) = sqrt(λ)`

The equality of mean and variance is a property of the standard Poisson model.

## Examples

Possible applications include:

- Requests arriving at a service
- Calls received by a call center
- Defects observed in a fixed production length
- Events recorded during a fixed time interval
- Network packets arriving during a measurement window

The event rate must be defined together with its unit.

For example:

`λ = 120 requests per minute`

is different from:

`λ = 120 requests per hour`

Unit consistency is essential.

## Poisson process

For a constant-rate Poisson process:

`N(t) ~ Poisson(λt)`

where `N(t)` is the number of events during an interval of length `t`.

If the rate is `120 requests/minute`, the expected number of requests during ten minutes is:

`120 × 10 = 1200`

The C++ case study uses this relationship for service-capacity analysis.

## Important limitations

A simple Poisson process assumes conditions such as:

- A meaningful event rate
- Appropriate independence of events
- No systematic rate changes within the modeled interval
- No unexplained clustering or inhibition that materially violates the model

Real systems can have:

- Traffic seasonality
- Bursty demand
- Time-varying rates
- Correlated events
- Multiple user populations
- Scheduled traffic
- Self-exciting events

Such processes may require more advanced models.

---

# Exponential distribution

## Definition

The Exponential distribution models waiting time between events in a constant-rate Poisson process.

Let:

`T ~ Exponential(λ)`

where `λ` is a rate.

The PDF is:

`f(t) = λ exp(-λt)`

for `t >= 0`.

The CDF is:

`F(t) = 1 - exp(-λt)`

The survival function is:

`P(T>t) = exp(-λt)`

## Mean

`E[T] = 1/λ`

## Variance

`Var(T) = 1/λ²`

## Poisson and Exponential relationship

The relationship can be stated simply:

- Poisson answers how many events occur.
- Exponential answers how long until the next event.

For a Poisson process with rate `λ`:

`N(t) ~ Poisson(λt)`

and:

`T ~ Exponential(λ)`

This distinction is demonstrated explicitly in all three implementations.

## Memoryless property

The Exponential distribution has the memoryless property:

`P(T>s+t | T>s) = P(T>t)`

The amount of time already spent waiting does not change the distribution of the additional waiting time.

This property is mathematically important but should not be assumed for arbitrary real-world waiting times.

## Numerical detail

The implementation uses `expm1` when calculating:

`1 - exp(-λx)`

because direct subtraction can lose numerical precision for very small values of `λx`.

---

# Uniform distribution

## Definition

A continuous Uniform distribution gives equal density to every point inside a finite interval.

Let:

`X ~ Uniform(a,b)`

with:

`a < b`

The PDF is:

`f(x) = 1/(b-a)`

for `a <= x <= b`.

## CDF

The CDF is:

`F(x) = 0` for `x < a`

`F(x) = (x-a)/(b-a)` for `a <= x < b`

`F(x) = 1` for `x >= b`

## Mean

`E[X] = (a+b)/2`

## Variance

`Var(X) = (b-a)²/12`

## Applications

Uniform models can be useful when equal density over a bounded interval is substantively justified.

They are also widely used as a mathematical building block for simulation.

A common modeling error is to assume that every uncertain quantity with known minimum and maximum should be Uniform. The bounds alone do not establish equal density.

---

# Gaussian / Normal distribution

## Definition

The Gaussian distribution, commonly called the Normal distribution, is a continuous symmetric distribution characterized by its mean and standard deviation.

Let:

`X ~ Normal(μ, σ²)`

where:

- `μ` is the mean
- `σ` is the standard deviation
- `σ > 0`

The PDF is:

`f(x) = [1/(σ sqrt(2π))] exp(-(x-μ)²/(2σ²))`

## Mean

`E[X] = μ`

## Variance

`Var(X) = σ²`

## Standardization

A value can be converted to a standard score:

`Z = (X-μ)/σ`

A z-score indicates how many standard deviations a value lies above or below the mean.

For example, if:

`μ = 100`

and:

`σ = 15`

then:

`x = 130`

has:

`z = (130-100)/15 = 2`

## Standard normal distribution

The standard Normal distribution has:

`μ = 0`

and:

`σ = 1`

Its CDF is frequently used for probability calculations and statistical inference.

## Empirical rule

For a Normal distribution, approximately:

- 68% lies within ±1 standard deviation
- 95% lies within ±2 standard deviations
- 99.7% lies within ±3 standard deviations

These percentages apply approximately to a Normal distribution. They should not be treated as universal properties of arbitrary datasets.

## Central Limit Theorem

The Central Limit Theorem explains why Normal approximations appear frequently in statistics.

Under appropriate conditions, properly normalized sums or sample means of independent observations tend toward a Normal distribution as sample size increases.

The original population does not need to be Normal.

The Python and JavaScript implementations demonstrate this by repeatedly calculating means from Uniform random observations.

## Limitations

A Normal distribution may be inappropriate when data are strongly:

- Skewed
- Heavy-tailed
- Bounded
- Discrete
- Multimodal
- Zero-inflated

For example, response latency is often non-negative and can have a long right tail. A Normal model can assign positive probability to negative latency, so its appropriateness must be checked against actual data.

---

# Sampling and simulation

Analytical probability calculations use mathematical formulas. Simulation generates random observations from a specified distribution.

Simulation is useful for:

- Monte Carlo estimation
- Capacity analysis
- Sensitivity analysis
- Testing probabilistic algorithms
- Model validation
- Understanding sampling variability

A simulation does not prove that the selected distribution is correct. It only shows what follows from the chosen model.

## Reproducibility

The Python implementation uses `random.Random(seed)` for deterministic simulations.

The JavaScript implementation includes a small seeded pseudo-random generator because the standard `Math.random()` API does not provide seed control.

The C++ implementation initializes `std::mt19937_64` with a fixed seed.

A fixed seed makes simulation results reproducible, which is useful for:

- Debugging
- Unit tests
- Controlled experiments
- Comparing algorithm implementations

A simulation seed is not a security mechanism.

---

# Law of Large Numbers

The Law of Large Numbers states, under appropriate conditions, that sample averages tend toward their expected values as the sample size increases.

For a Bernoulli variable with `p=0.3`, a sufficiently large sample's observed success rate tends to approach `0.3`.

This does not mean that every small sample will be close to `0.3`.

Random variation remains present at finite sample sizes.

The Python implementation explicitly compares observed success rates at several sample sizes.

---

# Central Limit Theorem demonstration

The Python and JavaScript programs generate Uniform observations and repeatedly calculate sample means.

An individual Uniform observation is not Gaussian.

The distribution of sufficiently large sample means becomes approximately Normal under suitable conditions.

For `Uniform(0,1)`:

`Var(X) = 1/12`

For a sample of size `n`, the approximate standard deviation of the sample mean is:

`1 / sqrt(12n)`

The implementation compares this theoretical value with the simulated standard deviation of the sample means.

---

# Distribution relationships

The six distributions are related but serve different modeling purposes.

## Bernoulli to Binomial

A Binomial random variable can be represented as a sum of independent Bernoulli variables:

`X = X₁ + X₂ + ... + Xₙ`

where each:

`Xᵢ ~ Bernoulli(p)`

Under the standard assumptions:

`X ~ Binomial(n,p)`

## Binomial to Poisson

When:

- `n` is large
- `p` is small
- `λ = np`

the Binomial distribution can often be approximated by:

`Poisson(λ)`

The Python, JavaScript, and C++ programs compare Binomial and Poisson probabilities for:

`n = 1000`

`p = 0.004`

which gives:

`λ = 4`

This is an approximation, not an identity.

## Poisson to Exponential

For a Poisson process:

`N(t) ~ Poisson(λt)`

while the waiting time between events is:

`T ~ Exponential(λ)`

Counts and waiting times therefore describe two complementary views of the same idealized process.

## Normal approximations

Under appropriate conditions, the Binomial distribution can also be approximated by a Normal distribution when its parameters produce sufficiently large expected counts of both successes and failures.

The approximation should be assessed rather than applied mechanically.

---

# Practical service-monitoring case study

The C++ program models a web service.

The scenario contains several distinct random variables.

## Request outcome

Each request is either successful or failed.

A Bernoulli model is used:

`X ~ Bernoulli(0.02)`

The model therefore assumes a 2% failure probability.

## Failures in a fixed batch

For exactly 500 independent requests:

`X ~ Binomial(500,0.02)`

The expected number of failures is:

`10`

The program calculates:

- Probability of exactly 10 failures
- Probability of no more than 5 failures
- A simulated batch result

## Request arrivals

The service receives an average of 120 requests per minute.

The count model is:

`N(1 minute) ~ Poisson(120)`

For one hour:

`N(60 minutes) ~ Poisson(7200)`

The program uses this model to calculate event-count probabilities and demonstrate capacity planning.

## Interarrival times

The corresponding Exponential model is:

`T ~ Exponential(120)`

where the unit is requests per minute.

The expected interarrival time is:

`1/120 minute`

or approximately:

`0.5 seconds`

This is a direct illustration of the Poisson-process relationship.

## Response latency

The case study models response latency as:

`Normal(200,40²)`

This means:

- Mean latency = 200 ms
- Standard deviation = 40 ms
- Variance = 1600 ms²

The program calculates the probability that response latency is at most 250 ms.

This is an assumption for the case study. Actual production latency should be analyzed before selecting a Normal model.

## Bounded configuration factor

The C++ program also uses a Uniform distribution over `[0,1]` to demonstrate bounded uncertainty.

This represents a controlled modeling assumption rather than a general rule about configuration uncertainty.

---

# Python implementation

The Python file is designed as a mathematical study and reference implementation.

## Main components

### `BernoulliDistribution`

Demonstrates:

- Probability validation
- PMF
- CDF
- Mean
- Variance
- Standard deviation
- Sampling

### `BinomialDistribution`

Demonstrates:

- Fixed trial count
- Success probability
- Logarithmic PMF calculation
- CDF
- Tail probability
- Mean
- Variance
- Simulation

### `PoissonDistribution`

Demonstrates:

- Event-count PMF
- CDF
- Mean
- Variance
- Sampling

The implementation uses an exact Knuth-style algorithm for moderate rates and a Normal approximation for larger rates. The latter is explicitly an approximation rather than an exact Poisson sampler.

### `ExponentialDistribution`

Demonstrates:

- PDF
- CDF
- Survival function
- Quantiles
- Mean
- Variance
- Sampling
- Memorylessness

### `UniformDistribution`

Demonstrates:

- PDF
- CDF
- Mean
- Variance
- Sampling
- Boundary behavior

### `GaussianDistribution`

Demonstrates:

- PDF
- CDF
- z-scores
- Mean
- Variance
- Normal sampling
- Approximate inverse CDF

The inverse Normal CDF uses a rational approximation so that the example remains self-contained without requiring SciPy.

## Python-specific strengths

Python makes the mathematical implementation compact and readable.

The standard library provides useful numerical functionality such as:

- `math.lgamma`
- `math.erf`
- `math.expm1`
- `math.log1p`
- `statistics`
- `random`

The program also uses dataclasses to make distribution objects concise and explicit.

---

# JavaScript implementation

The JavaScript file complements the Python implementation through application-oriented numerical code.

## Deterministic random generator

JavaScript's standard `Math.random()` does not expose a standard seed mechanism.

The implementation therefore includes `SeededRandom`.

This is useful for reproducible demonstrations.

It is not a cryptographically secure random-number generator.

## JavaScript numerical representation

Ordinary JavaScript numbers use IEEE-754 double-precision floating-point representation.

This means very large integers cannot always be represented exactly.

The JavaScript implementation uses `BigInt` for exact integer factorial and combination demonstrations.

Probability calculations continue to use floating-point values because probability values are naturally represented numerically and because logarithmic formulas avoid unnecessarily large intermediate integer values.

## Gaussian CDF

Standard JavaScript does not provide `Math.erf()`.

The implementation therefore includes an error-function approximation for the Gaussian CDF.

This demonstrates an important application-level issue: mathematical functionality available in one language's standard library may need an explicit implementation or external numerical library in another environment.

## Monte Carlo estimation

The JavaScript file estimates probabilities by repeated simulation.

For example, the Uniform model is used to estimate:

`P(X <= 15)`

for:

`X ~ Uniform(10,20)`

The analytical answer is:

`0.5`

The simulation should approach this value as the number of trials increases.

---

# C++ implementation

The C++ program is structured as a technical case study rather than as a collection of isolated formulas.

## Main architecture

The program contains separate classes for:

- `BernoulliDistribution`
- `BinomialDistribution`
- `PoissonDistribution`
- `ExponentialDistribution`
- `UniformDistribution`
- `GaussianDistribution`

It also contains reusable utilities for:

- Input validation
- Sample statistics
- Histogram generation
- Monte Carlo estimation
- Numerical probability calculations

## C++ random distributions

The standard library provides:

- `std::bernoulli_distribution`
- `std::binomial_distribution`
- `std::poisson_distribution`
- `std::exponential_distribution`
- `std::uniform_real_distribution`
- `std::normal_distribution`

The program uses these standard facilities for simulation while implementing probability functions separately for educational transparency.

## Engine selection

The program uses:

`std::mt19937_64`

with a fixed seed.

This provides a reproducible pseudo-random sequence for the case study.

The generator is appropriate for statistical simulation but should not be treated as a cryptographic random-number source.

## Statistics

The program calculates:

- Count
- Mean
- Population variance
- Population standard deviation
- Minimum
- Maximum

It then compares simulated statistics against theoretical distribution parameters.

---

# Important distinction: probability model versus random generator

A probability distribution is a mathematical model.

A random-number generator is an algorithm that produces a sequence of values intended to behave according to some random process.

These are different concepts.

For example:

`Normal(200,40²)`

defines the target statistical model.

`std::normal_distribution<double>`

provides a mechanism for generating samples from that model using a C++ random engine.

Similarly, a Python random generator or JavaScript seeded generator is a computational mechanism, not the probability distribution itself.

---

# Edge cases

## Bernoulli

When:

`p = 0`

the outcome is always failure.

When:

`p = 1`

the outcome is always success.

## Binomial

When:

`p = 0`

the only possible result is:

`X = 0`

When:

`p = 1`

the only possible result is:

`X = n`

Values outside:

`0 <= k <= n`

have probability zero.

## Poisson

When:

`λ = 0`

the only possible count is zero.

Negative event counts are impossible.

## Exponential

For:

`x < 0`

the CDF is zero because a waiting time cannot be negative.

The rate must be strictly positive.

## Uniform

The upper bound must exceed the lower bound.

Values below the lower bound have CDF zero.

Values at or above the upper boundary have CDF one.

## Gaussian

The standard deviation must be strictly positive.

Unlike the other continuous models in this study, the mathematical Gaussian distribution has support over the entire real line, including negative values.

That property can make it inappropriate for naturally non-negative quantities unless the approximation is justified.

---

# Common mistakes

## Treating a PDF value as a probability

For a continuous distribution, `f(x)` is a density.

It is not:

`P(X=x)`

The probability comes from integration over an interval.

## Confusing Poisson and Exponential

Poisson models event counts.

Exponential models waiting times.

They are related through a Poisson process but are not interchangeable.

## Assuming every binary dataset is Binomial

Binary observations alone do not guarantee a Binomial model.

The standard model also requires a fixed number of trials, consistent probability, and independence.

## Assuming a Poisson rate is always constant

Real event processes can have changing rates.

A single Poisson parameter may fail when traffic changes significantly across time.

## Assuming a large dataset is automatically Gaussian

A large sample size does not make the raw observations Gaussian.

The Central Limit Theorem concerns suitable sums or sample means, not automatic Normality of the original variable.

## Ignoring units

A rate such as:

`120 requests/minute`

must be converted correctly when calculating a quantity over seconds, hours, or days.

## Using direct factorial calculations for large values

Factorials grow extremely rapidly.

For probability calculations, logarithmic gamma functions or stable recurrence formulas are often preferable.

## Ignoring tail behavior

A model can fit the center of a dataset reasonably well while producing poor tail estimates.

This matters for:

- Capacity planning
- Reliability
- Anomaly detection
- Safety limits
- Service-level objectives
- Rare-event analysis

---

# Limitations of the models

These six distributions are fundamental but do not cover every real-world probability problem.

## Bernoulli limitations

A standard Bernoulli model represents only one binary trial.

It does not directly represent:

- Multiple outcome categories
- Dependence between trials
- Changing probabilities
- Continuous measurements

## Binomial limitations

The standard Binomial model assumes:

- Fixed `n`
- Constant `p`
- Independent trials
- Two outcomes

When sampling without replacement from a finite population, a Hypergeometric model may be more appropriate.

When probabilities vary between trials, a Poisson-binomial model can be appropriate.

## Poisson limitations

The basic Poisson model assumes a particular event-count structure.

Overdispersion or underdispersion relative to Poisson expectations can indicate that another model may be more appropriate.

A rate that changes with time may require a non-homogeneous Poisson process or another model.

## Exponential limitations

The Exponential distribution has constant hazard.

Real lifetimes and waiting times may have increasing or decreasing hazard.

Alternative distributions such as Weibull or Gamma can model richer behavior.

## Uniform limitations

Uniformity requires equal density across the specified interval.

Knowing only minimum and maximum values is not enough to justify a Uniform model.

## Gaussian limitations

The Gaussian distribution is:

- Symmetric
- Continuous
- Unbounded
- Unimodal

It may therefore be unsuitable for strongly skewed, bounded, heavy-tailed, multimodal, or discrete variables.

---

# Performance considerations

## Factorials

Factorials grow extremely quickly.

For example:

`100!`

is far too large for ordinary floating-point representation.

Logarithmic calculations such as:

`lgamma(n+1)`

avoid creating the enormous factorial explicitly.

## Binomial probabilities

The expression:

`C(n,k) p^k (1-p)^(n-k)`

can create numerical difficulties for large `n`.

Logarithmic evaluation is generally safer.

## Poisson probabilities

The expression:

`exp(-λ) λ^k / k!`

can also encounter overflow or underflow.

The implementations use logarithmic formulas where appropriate.

## Normal tail probabilities

For large positive values of `x`, a CDF may be extremely close to one.

Calculating:

`1 - F(x)`

can lose precision.

Production numerical libraries often provide dedicated survival functions or log-survival functions.

## Simulation cost

A simulation involving millions or billions of observations can be computationally expensive.

For high-volume numerical work, vectorized numerical systems, specialized algorithms, parallel processing, or optimized numerical libraries may be preferable.

The Python program intentionally favors clarity over maximum throughput.

The C++ program demonstrates a more performance-oriented compiled implementation.

---

# Security considerations

Statistical simulation and cryptographic randomness have different requirements.

A seeded pseudo-random generator is useful for:

- Reproducible tests
- Simulation
- Benchmarking
- Statistical experiments

It should not be used for:

- Authentication tokens
- Password reset codes
- Session secrets
- Encryption keys
- Security-sensitive identifiers

The Python implementation explicitly demonstrates the distinction between `random` for simulation and `secrets` for security-oriented random values.

The JavaScript implementation's seeded generator is also strictly a simulation mechanism.

The C++ program uses a deterministic `std::mt19937_64` engine for reproducible statistical simulation.

---

# Implementation considerations

A production probability service should distinguish between:

1. Model definition
2. Parameter estimation
3. Probability evaluation
4. Sampling
5. Validation
6. Monitoring
7. Numerical stability
8. Operational decision-making

The mathematical distribution should not be selected solely because it is easy to calculate.

Model assumptions should be compared against observed data.

Useful validation activities include:

- Comparing empirical and theoretical distributions
- Checking sample moments
- Inspecting quantiles
- Examining tail behavior
- Testing parameter stability
- Evaluating residuals
- Testing performance under changing conditions
- Checking whether observations are independent
- Examining whether event rates vary over time

---

# Model selection considerations

A practical decision process can begin with the variable type.

For one binary observation:

`Bernoulli`

For the number of successes in a fixed number of comparable binary trials:

`Binomial`

For event counts in an interval:

`Poisson`

For interarrival times under a constant-rate Poisson process:

`Exponential`

For bounded continuous uncertainty with equal density:

`Uniform`

For approximately symmetric continuous measurements:

`Gaussian`

This is a starting point, not a substitute for empirical validation.

---

# Comparison of implementation languages

## Python

Python is particularly suitable for mathematical teaching and rapid experimentation because its syntax is concise and its standard library contains useful numerical functions.

The Python implementation emphasizes:

- Mathematical definitions
- Explicit formulas
- Numerical stability
- Simulation
- Statistical summaries
- Self-checking assertions

## JavaScript

JavaScript is useful when probability models are part of web applications, interactive dashboards, client-side simulation, or application logic.

The JavaScript implementation emphasizes:

- Classes
- Runtime validation
- Deterministic simulation
- Floating-point behavior
- JavaScript-specific numerical limitations
- Monte Carlo estimation
- Application-level modeling

## C++

C++ is useful when probability calculations form part of high-performance systems, simulations, services, scientific applications, or infrastructure software.

The C++ implementation emphasizes:

- Strongly structured classes
- Standard-library distributions
- Explicit numerical calculations
- Efficient memory handling
- Reproducible simulation
- Statistical analysis
- Error handling
- A coherent service-monitoring scenario

---

# Real-world applications

These distributions appear in many technical domains.

## Web and distributed systems

- Poisson request counts
- Exponential interarrival times
- Bernoulli request failures
- Binomial batch failures
- Gaussian approximations for selected measurements

## Manufacturing

- Bernoulli defect outcomes
- Binomial defect counts
- Poisson defect counts over production length
- Continuous measurement models

## Reliability engineering

- Event counts
- Failure probabilities
- Waiting times
- Lifetime models

The Exponential model is particularly important when constant hazard is a reasonable assumption, while other lifetime distributions may be necessary for changing hazard.

## Telecommunications

- Packet arrivals
- Connection failures
- Interarrival times
- Queueing models

## Finance

Normal models are frequently used as mathematical approximations in some financial models, but financial returns can exhibit characteristics that depart from simple Gaussian assumptions, especially in their tails.

## Quality control

- Defect probability
- Defects per unit
- Batch failure rates
- Measurement variation

## Experimentation

Bernoulli and Binomial models are foundational for:

- Conversion experiments
- Click-through events
- Treatment response
- Success rates

---

# Practical interpretation of the C++ case study

The C++ program does not treat all observations as the same type of random variable.

Instead, it separates the service into different stochastic components:

`Request outcome -> Bernoulli`

`Failures in fixed batch -> Binomial`

`Requests per interval -> Poisson`

`Time between requests -> Exponential`

`Response latency -> Gaussian approximation`

`Bounded configuration factor -> Uniform`

This separation is important because a single operational system can contain many different random variables, each requiring its own probability model.

---

# Model assumptions versus observed reality

The equations in this study are exact mathematical properties of the specified distributions.

Whether a distribution accurately represents a real system is a separate question.

For example, the following statement is mathematically correct:

`If X ~ Poisson(120), then E[X] = 120.`

It does not establish that a particular production service actually follows a Poisson distribution with rate 120.

Likewise:

`If X ~ Normal(200,40²), then P(X<=250)` can be calculated exactly from the model.

That does not establish that real response latency follows that Normal model.

Model validity must therefore be established separately from mathematical calculation.

---

# Reproducibility

The examples use deterministic seeds for simulation.

This makes it possible to:

- Reproduce examples
- Compare implementations
- Debug algorithms
- Write deterministic tests
- Investigate numerical differences

Reproducibility does not imply that the simulation is statistically perfect.

Different random-number generators, algorithms, seeds, and floating-point implementations can produce different sample sequences while still representing the same intended distribution.

---

# Error handling

The implementations reject invalid parameters rather than silently producing meaningless results.

Examples include:

- Probability below zero
- Probability above one
- Negative Binomial trial count
- Negative Poisson rate
- Zero Exponential rate
- Invalid Uniform bounds
- Non-positive Gaussian standard deviation
- Empty samples
- Invalid simulation trial counts

Explicit validation is especially important in production systems because invalid statistical parameters can otherwise propagate incorrect calculations into downstream decisions.

---

# Numerical precision

Probability calculations are affected by finite computer precision.

A mathematically valid expression may still be numerically unstable if implemented directly.

Examples include:

`1 - exp(-x)`

for very small `x`.

The implementations use:

`expm1`

where appropriate.

Another example is:

`log(n!)`

which is implemented using:

`lgamma(n+1)`

rather than constructing `n!`.

These techniques are not merely performance optimizations. They can materially improve numerical correctness.

---

# Key formulas

## Bernoulli

`P(X=x) = p^x(1-p)^(1-x)`

`E[X] = p`

`Var(X) = p(1-p)`

## Binomial

`P(X=k) = C(n,k)p^k(1-p)^(n-k)`

`E[X] = np`

`Var(X) = np(1-p)`

## Poisson

`P(X=k) = exp(-λ)λ^k/k!`

`E[X] = λ`

`Var(X) = λ`

## Exponential

`f(x) = λ exp(-λx)`

for `x >= 0`.

`E[X] = 1/λ`

`Var(X) = 1/λ²`

`P(X>x) = exp(-λx)`

## Uniform

`f(x) = 1/(b-a)`

for `a <= x <= b`.

`E[X] = (a+b)/2`

`Var(X) = (b-a)²/12`

## Gaussian

`f(x) = [1/(σsqrt(2π))] exp(-(x-μ)²/(2σ²))`

`E[X] = μ`

`Var(X) = σ²`

`Z = (X-μ)/σ`

---

# Core distinctions

| Question | Distribution |
|---|---|
| Is this one binary trial? | Bernoulli |
| How many successes occur in `n` binary trials? | Binomial |
| How many events occur in a fixed interval? | Poisson |
| How long until the next event in a constant-rate Poisson process? | Exponential |
| Is a continuous variable equally dense throughout a bounded interval? | Uniform |
| Is a continuous variable reasonably symmetric and bell-shaped? | Gaussian |

The central modeling principle is that the probability distribution should correspond to the structure of the random process being modeled. Mathematical convenience alone is not sufficient evidence for model selection.
