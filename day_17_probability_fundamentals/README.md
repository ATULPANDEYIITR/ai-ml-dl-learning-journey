# Probability fundamentals

## Topic scope

Probability provides a formal mathematical framework for reasoning about uncertain outcomes. The central ideas in this implementation are probability spaces, events, conditional probability, and independence.

The three implementations approach the subject from different technical perspectives:

- The Python implementation develops a reusable finite probability-space abstraction and uses it to demonstrate the mathematical rules progressively.
- The JavaScript implementation presents the same core ideas with JavaScript collections, classes, functions, deterministic simulation, validation, and application-oriented examples.
- The C++ implementation develops a more structured reliability case study in which probability is used to model a production-style service with redundant application servers and a required database.

The examples use finite discrete probability models because they make the underlying mathematical structure explicit while still allowing practical computation.

## Probability spaces

A probability space is commonly represented as:

`(Ω, F, P)`

where:

- `Ω` is the sample space.
- `F` is the collection of events being considered.
- `P` is the probability measure.

For a finite experiment, the sample space can be represented directly as a collection of elementary outcomes.

For a fair six-sided die:

`Ω = {1, 2, 3, 4, 5, 6}`

Each elementary outcome has probability:

`P({i}) = 1/6`

for `i` from 1 through 6.

An event is a subset of the sample space. For example, the event of rolling an even number is:

`A = {2, 4, 6}`

Its probability is:

`P(A) = 3/6 = 1/2`

The Python `FiniteProbabilitySpace`, JavaScript `FiniteProbabilitySpace`, and C++ `FiniteProbabilitySpace` classes all represent this structure explicitly.

## Outcomes, sample spaces, and events

An elementary outcome is one possible result of an experiment.

Examples include:

- Heads from a coin flip
- A particular number from a die
- A particular pair of values from two dice
- A particular state of a system
- A particular combination of binary variables

The sample space contains all elementary outcomes.

An event is a set of one or more outcomes. Important special events include:

### Empty event

The empty event contains no outcomes.

`P(∅) = 0`

### Certain event

The entire sample space is an event.

`P(Ω) = 1`

### Complement

The complement of `A`, written `Aᶜ`, contains every outcome in `Ω` that is not in `A`.

The complement rule is:

`P(Aᶜ) = 1 - P(A)`

The implementations explicitly calculate complements rather than treating the rule as merely a formula.

## Probability axioms

The standard probability model follows three fundamental axioms.

### Non-negativity

For every event `A`:

`P(A) >= 0`

### Normalization

The entire sample space has probability one:

`P(Ω) = 1`

### Countable additivity

For mutually exclusive events, probabilities add.

For two disjoint events:

`P(A ∪ B) = P(A) + P(B)`

The finite implementations use these principles when calculating probabilities from elementary outcomes.

These axioms produce many familiar probability rules, including the complement rule and inclusion-exclusion.

## Event algebra

Events behave like sets.

For events `A` and `B`:

### Intersection

`A ∩ B`

means both events occur.

### Union

`A ∪ B`

means at least one of the events occurs.

### Complement

`Aᶜ`

means `A` does not occur.

### Mutually exclusive events

If:

`A ∩ B = ∅`

then `A` and `B` cannot occur simultaneously.

Mutual exclusivity should not be confused with independence.

## Inclusion-exclusion

For two events:

`P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`

The intersection is subtracted because it is counted once in `P(A)` and again in `P(B)`.

The Python, JavaScript, and C++ implementations explicitly verify this relationship.

For disjoint events, the intersection probability is zero, so the formula reduces to:

`P(A ∪ B) = P(A) + P(B)`

## Counting and equally likely outcomes

When all elementary outcomes are equally likely:

`P(A) = |A| / |Ω|`

This allows combinatorial counting to be converted directly into probability.

Two dice have:

`6 × 6 = 36`

ordered outcomes.

The Python, JavaScript, and C++ examples use finite enumeration and counting to demonstrate this idea.

For a fair coin flipped five times, the number of ways to obtain exactly three heads is:

`C(5, 3) = 10`

There are:

`2⁵ = 32`

possible sequences.

Therefore:

`P(exactly 3 heads) = C(5,3) / 2⁵ = 10/32 = 0.3125`

The JavaScript implementation includes a complete combinations function, while the Python implementation uses the standard-library `math.comb`.

## Conditional probability

Conditional probability measures the probability of one event when another event is known to have occurred.

It is defined as:

`P(A | B) = P(A ∩ B) / P(B)`

provided:

`P(B) > 0`

The event `B` changes the effective set of possibilities. Outcomes outside `B` are no longer considered possible under the condition.

For a die:

- `A = {2,4,6}`
- `B = {4,5,6}`

Then:

`A ∩ B = {4,6}`

and:

`P(A | B) = P({4,6}) / P({4,5,6})`

which gives:

`(2/6) / (3/6) = 2/3`

### Zero-probability condition

`P(A | B)` is not defined by the elementary ratio formula when:

`P(B) = 0`

The implementations deliberately reject this situation rather than returning an arbitrary numerical value.

## Multiplication rule

The conditional-probability definition can be rearranged to give:

`P(A ∩ B) = P(A | B)P(B)`

An equivalent expression is:

`P(A ∩ B) = P(B | A)P(A)`

This rule is useful when direct intersection probabilities are difficult to determine but conditional probabilities are known.

The implementations test this identity numerically.

## Independence

Two events `A` and `B` are independent when knowledge that one occurred does not change the probability of the other.

The principal definition is:

`P(A ∩ B) = P(A)P(B)`

When `P(B) > 0`, this is equivalent to:

`P(A | B) = P(A)`

Independence is a mathematical relationship between events. It should not be inferred merely because events appear unrelated in everyday language.

The probability-space implementations calculate both sides of the independence equation.

## Mutual exclusivity versus independence

These concepts are fundamentally different.

Mutually exclusive events satisfy:

`P(A ∩ B) = 0`

Independent events satisfy:

`P(A ∩ B) = P(A)P(B)`

If two mutually exclusive events both have positive probability, then:

`P(A)P(B) > 0`

while:

`P(A ∩ B) = 0`

Therefore they cannot be independent.

A common mistake is to interpret "cannot happen together" as meaning "one does not affect the other." Those statements describe different mathematical properties.

## Conditional probability and dependence

Conditional probability is especially important when trials are dependent.

Consider drawing cards from a standard 52-card deck without replacement.

Initially:

`P(first heart) = 13/52`

If the first card is known to be a heart, then 12 hearts remain among 51 cards:

`P(second heart | first heart) = 12/51`

The composition of the deck has changed, so the second event is dependent on the first.

The probability of two consecutive hearts is therefore:

`(13/52)(12/51)`

rather than:

`(13/52)²`

The latter expression would require an independence assumption that is not appropriate for sampling without replacement.

## Bayes' theorem

Bayes' theorem reverses the direction of conditional probability:

`P(A | B) = P(B | A)P(A) / P(B)`

This is important because `P(B | A)` and `P(A | B)` are generally not equal.

A diagnostic example illustrates the difference.

Suppose:

- `P(D) = 0.01`
- `P(+ | D) = 0.95`
- `P(+ | not D) = 0.10`

The overall probability of a positive result is obtained through the law of total probability:

`P(+) = P(+ | D)P(D) + P(+ | not D)P(not D)`

Then:

`P(D | +) = P(+ | D)P(D) / P(+)`

The posterior probability depends on both test characteristics and the prevalence of the condition.

This is the mathematical basis of many diagnostic, classification, filtering, and inference systems.

## Law of total probability

Suppose events `B₁, B₂, ..., Bₙ` form a partition of the sample space.

Then:

`P(A) = Σ P(A | Bᵢ)P(Bᵢ)`

The factory example uses three production sources:

- Factory A supplies 50% of production and has a 1% defect rate.
- Factory B supplies 30% and has a 3% defect rate.
- Factory C supplies 20% and has a 5% defect rate.

The overall defect probability is:

`P(D) = P(D|A)P(A) + P(D|B)P(B) + P(D|C)P(C)`

This structure appears frequently when a population is divided into known subpopulations.

## Pairwise independence and mutual independence

Independence becomes more subtle when more than two events are involved.

Events can be pairwise independent without being mutually independent.

The implementations construct three events from two fair binary variables:

- `A`: first bit equals 1
- `B`: second bit equals 1
- `C`: XOR of the two bits equals 1

Every pair is independent.

Yet all three events are not mutually independent because the triple intersection has probability zero while:

`P(A)P(B)P(C) = 1/8`

Thus checking every pair separately does not always establish mutual independence.

For a collection of events to be mutually independent, the required factorization must hold for every relevant finite subset.

## Conditional independence

Conditional independence is different from ordinary independence.

Events `A` and `B` can be conditionally independent given `C` when:

`P(A ∩ B | C) = P(A | C)P(B | C)`

This concept is important in probabilistic graphical models and diagnostic systems.

For example, two test results may be associated in the overall population because both are related to an underlying disease state. Once the disease state is fixed, it may be reasonable to model the test outcomes as conditionally independent.

Conditional independence is a modeling assumption and should be supported by domain knowledge or empirical evidence.

## Random variables

A random variable assigns a numerical value to outcomes.

For a six-sided die, define:

`X(outcome) = outcome`

Then `X` can take the values:

`1, 2, 3, 4, 5, 6`

with probability `1/6` each.

A random variable is therefore a function from outcomes to numerical values, not simply an outcome itself.

The Python, JavaScript, and C++ implementations calculate expectations and variance from discrete distributions.

## Expected value

For a discrete random variable:

`E[X] = Σ x P(X=x)`

For a fair six-sided die:

`E[X] = (1+2+3+4+5+6)/6 = 3.5`

The expected value does not necessarily have to be an outcome that can actually occur. A six-sided die cannot display 3.5, but 3.5 is its long-run average value under repeated independent trials.

## Indicator random variables

For an event `A`, define an indicator:

`I_A = 1` if `A` occurs

and:

`I_A = 0` otherwise.

A fundamental identity is:

`E[I_A] = P(A)`

The implementations explicitly calculate an indicator expectation and compare it with the event probability.

Indicator variables are useful for counting occurrences. If:

`I₁, I₂, ..., Iₙ`

indicate whether individual events occur, then:

`I₁ + I₂ + ... + Iₙ`

counts how many occur.

## Variance

Variance measures the spread of a random variable around its expected value.

It is defined as:

`Var(X) = E[(X - E[X])²]`

An equivalent expression is:

`Var(X) = E[X²] - (E[X])²`

Standard deviation is:

`σ = sqrt(Var(X))`

The C++ and Python implementations calculate variance directly from the definition.

## Covariance and correlation

For random variables `X` and `Y`:

`Cov(X,Y) = E[(X-E[X])(Y-E[Y])]`

An equivalent form is:

`Cov(X,Y) = E[XY] - E[X]E[Y]`

Correlation standardizes covariance:

`Corr(X,Y) = Cov(X,Y) / (σₓσᵧ)`

Independence implies zero covariance when the relevant expectations exist.

The reverse implication generally does not hold. Zero covariance does not, by itself, prove independence.

## Probability bounds

For any two events:

`max(0, P(A)+P(B)-1) <= P(A ∩ B) <= min(P(A),P(B))`

These are the Fréchet bounds.

The lower bound prevents an intersection from having a negative probability.

The upper bound follows because:

`A ∩ B`

cannot contain more probability mass than either `A` or `B`.

The union bound is:

`P(A ∪ B) <= P(A) + P(B)`

It remains valid even when the events are not independent.

## At-least-one calculations

For independent events with success probability `p` repeated `n` times:

`P(at least one success) = 1 - (1-p)ⁿ`

The complement is often easier to calculate than explicitly adding the probabilities for one success, two successes, and so on.

For example, with `p = 0.1` and `n = 10`:

`P(at least one) = 1 - 0.9¹⁰`

This complement technique is widely used in reliability and risk calculations.

## Monte Carlo simulation

A Monte Carlo simulation estimates probabilities by repeatedly sampling random outcomes.

For a fair coin:

`P(heads) = 0.5`

A simulation repeatedly generates outcomes and calculates:

`estimated probability = number of successes / number of trials`

The estimate generally approaches the true probability as the number of independent trials increases.

A key approximation for ordinary Monte Carlo estimation is:

`error scale ≈ 1/sqrt(n)`

Consequently, increasing the sample size by a factor of 100 generally reduces the characteristic Monte Carlo error by approximately a factor of 10.

Simulation is useful when analytical calculation is difficult, but it introduces sampling error and therefore should not be confused with an exact probability calculation.

## Python implementation

The Python implementation provides a reusable `FiniteProbabilitySpace` class.

Its principal operations include:

- `probability(event)`
- `complement(event)`
- `conditional_probability(event, condition)`
- `independent(event_a, event_b)`

The class accepts non-uniform elementary probabilities, so it can represent both fair and biased finite experiments.

The implementation validates:

- non-empty probability spaces
- non-negative probabilities
- finite numerical values
- probabilities summing to one
- events containing only valid outcomes
- non-zero conditioning probabilities

The script also demonstrates:

- event algebra
- counting
- Bayes' theorem
- total probability
- independence
- pairwise versus mutual independence
- random variables
- expectation
- variance
- covariance
- Monte Carlo estimation
- sampling without replacement
