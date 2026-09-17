/*
 * Probability Fundamentals: Technical Reliability Case Study
 *
 * C++17 program demonstrating:
 *   - finite probability spaces
 *   - events and set operations
 *   - conditional probability
 *   - Bayes' theorem
 *   - independence
 *   - total probability
 *   - reliability calculations
 *   - Monte Carlo simulation
 *   - validation and numerical tolerances
 *   - modular class design
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic probability.cpp -o probability
 *
 * The case study models a production web service composed of:
 *
 *   Client request
 *        |
 *        v
 *   Load balancer
 *      /   \
 *     v     v
 * Server A Server B
 *      \     /
 *       \   /
 *        v
 *     Database
 *
 * The server tier is redundant: either server can process the request.
 * The database is required for successful completion.
 *
 * Independence is an explicit modeling assumption. Real systems can violate
 * independence because components may share power, networking, software,
 * configuration, deployment events or infrastructure.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace probability {

// -----------------------------------------------------------------------------
// Numerical helpers
// -----------------------------------------------------------------------------

constexpr double EPSILON = 1e-12;

bool approximately_equal(double a, double b, double tolerance = EPSILON) {
    return std::abs(a - b) <= tolerance;
}

void require_probability(double probability, const std::string& name) {
    if (!std::isfinite(probability) ||
        probability < 0.0 ||
        probability > 1.0) {
        throw std::invalid_argument(
            name + " must be a finite value in [0, 1]."
        );
    }
}

void print_probability(const std::string& label, double probability) {
    std::cout << std::left << std::setw(42)
              << label
              << std::fixed << std::setprecision(8)
              << probability << '\n';
}

// -----------------------------------------------------------------------------
// Event model
// -----------------------------------------------------------------------------

template <typename Outcome>
using Event = std::set<Outcome>;

// -----------------------------------------------------------------------------
// Finite probability space
// -----------------------------------------------------------------------------

template <typename Outcome>
class FiniteProbabilitySpace {
public:
    using ProbabilityTable = std::map<Outcome, double>;

    explicit FiniteProbabilitySpace(ProbabilityTable probabilities)
        : probabilities_(std::move(probabilities)) {

        if (probabilities_.empty()) {
            throw std::invalid_argument(
                "A probability space must contain at least one outcome."
            );
        }

        double total = 0.0;

        for (const auto& [outcome, probability] : probabilities_) {
            (void)outcome;

            if (!std::isfinite(probability) || probability < 0.0) {
                throw std::invalid_argument(
                    "Every elementary probability must be finite and non-negative."
                );
            }

            total += probability;
        }

        if (!approximately_equal(total, 1.0)) {
            throw std::invalid_argument(
                "Elementary probabilities must sum to 1."
            );
        }
    }

    double probability(const Event<Outcome>& event) const {
        validate_event(event);

        double result = 0.0;

        for (const auto& outcome : event) {
            result += probabilities_.at(outcome);
        }

        return result;
    }

    Event<Outcome> complement(const Event<Outcome>& event) const {
        validate_event(event);

        Event<Outcome> result;

        for (const auto& [outcome, probability] : probabilities_) {
            (void)probability;

            if (event.find(outcome) == event.end()) {
                result.insert(outcome);
            }
        }

        return result;
    }

    Event<Outcome> intersection(
        const Event<Outcome>& first,
        const Event<Outcome>& second
    ) const {
        validate_event(first);
        validate_event(second);

        Event<Outcome> result;

        std::set_intersection(
            first.begin(),
            first.end(),
            second.begin(),
            second.end(),
            std::inserter(result, result.begin())
        );

        return result;
    }

    Event<Outcome> union_events(
        const Event<Outcome>& first,
        const Event<Outcome>& second
    ) const {
        validate_event(first);
        validate_event(second);

        Event<Outcome> result;

        std::set_union(
            first.begin(),
            first.end(),
            second.begin(),
            second.end(),
            std::inserter(result, result.begin())
        );

        return result;
    }

    double conditional_probability(
        const Event<Outcome>& event,
        const Event<Outcome>& condition
    ) const {
        const double p_condition = probability(condition);

        if (approximately_equal(p_condition, 0.0)) {
            throw std::domain_error(
                "P(A | B) is undefined when P(B) = 0."
            );
        }

        const auto intersection_event =
            intersection(event, condition);

        return probability(intersection_event) / p_condition;
    }

    bool independent(
        const Event<Outcome>& first,
        const Event<Outcome>& second
    ) const {
        const double p_first = probability(first);
        const double p_second = probability(second);

        const auto intersection_event =
            intersection(first, second);

        const double p_intersection =
            probability(intersection_event);

        return approximately_equal(
            p_intersection,
            p_first * p_second
        );
    }

    const ProbabilityTable& probabilities() const {
        return probabilities_;
    }

private:
    ProbabilityTable probabilities_;

    void validate_event(const Event<Outcome>& event) const {
        for (const auto& outcome : event) {
            if (probabilities_.find(outcome) == probabilities_.end()) {
                throw std::invalid_argument(
                    "Event contains an outcome outside the sample space."
                );
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Probability-space demonstration
// -----------------------------------------------------------------------------

void demonstrate_basic_probability_space() {
    std::cout << "\n--- Basic probability space ---\n";

    FiniteProbabilitySpace<std::string> coin({
        {"Heads", 0.5},
        {"Tails", 0.5}
    });

    Event<std::string> heads{"Heads"};

    print_probability("P(Heads)", coin.probability(heads));
    print_probability(
        "P(not Heads)",
        coin.probability(coin.complement(heads))
    );
}

// -----------------------------------------------------------------------------
// Event algebra
// -----------------------------------------------------------------------------

void demonstrate_event_algebra() {
    std::cout << "\n--- Event algebra ---\n";

    FiniteProbabilitySpace<int> die({
        {1, 1.0 / 6.0},
        {2, 1.0 / 6.0},
        {3, 1.0 / 6.0},
        {4, 1.0 / 6.0},
        {5, 1.0 / 6.0},
        {6, 1.0 / 6.0}
    });

    Event<int> even{2, 4, 6};
    Event<int> greater_than_three{4, 5, 6};

    const auto intersection =
        die.intersection(even, greater_than_three);

    const auto union_event =
        die.union_events(even, greater_than_three);

    print_probability("P(A)", die.probability(even));
    print_probability(
        "P(B)",
        die.probability(greater_than_three)
    );
    print_probability(
        "P(A intersection B)",
        die.probability(intersection)
    );
    print_probability(
        "P(A union B)",
        die.probability(union_event)
    );

    // Inclusion-exclusion:
    //
    // P(A union B)
    //   = P(A) + P(B) - P(A intersection B)
    const double expected_union =
        die.probability(even)
        + die.probability(greater_than_three)
        - die.probability(intersection);

    if (!approximately_equal(
            die.probability(union_event),
            expected_union)) {
        throw std::logic_error(
            "Inclusion-exclusion identity failed."
        );
    }
}

// -----------------------------------------------------------------------------
// Conditional probability
// -----------------------------------------------------------------------------

void demonstrate_conditional_probability() {
    std::cout << "\n--- Conditional probability ---\n";

    FiniteProbabilitySpace<int> die({
        {1, 1.0 / 6.0},
        {2, 1.0 / 6.0},
        {3, 1.0 / 6.0},
        {4, 1.0 / 6.0},
        {5, 1.0 / 6.0},
        {6, 1.0 / 6.0}
    });

    Event<int> even{2, 4, 6};
    Event<int> greater_than_three{4, 5, 6};

    const double p_intersection =
        die.probability(
            die.intersection(even, greater_than_three)
        );

    const double p_condition =
        die.probability(greater_than_three);

    const double p_conditional =
        die.conditional_probability(
            even,
            greater_than_three
        );

    print_probability(
        "P(A intersection B)",
        p_intersection
    );

    print_probability(
        "P(B)",
        p_condition
    );

    print_probability(
        "P(A | B)",
        p_conditional
    );

    // Multiplication rule:
    //
    // P(A intersection B) = P(A | B) P(B)
    if (!approximately_equal(
            p_intersection,
            p_conditional * p_condition)) {
        throw std::logic_error(
            "Conditional multiplication rule failed."
        );
    }
}

// -----------------------------------------------------------------------------
// Bayes' theorem
// -----------------------------------------------------------------------------

void demonstrate_bayes_theorem() {
    std::cout << "\n--- Bayes' theorem ---\n";

    /*
     * A diagnostic test is modeled with:
     *
     * prevalence:
     *     P(D) = 0.01
     *
     * sensitivity:
     *     P(+ | D) = 0.95
     *
     * false-positive rate:
     *     P(+ | not D) = 0.10
     *
     * We calculate:
     *
     *     P(D | +)
     *
     * using:
     *
     *     P(D | +)
     *       = P(+ | D)P(D) / P(+)
     *
     * and:
     *
     *     P(+)
     *       = P(+ | D)P(D)
     *       + P(+ | not D)P(not D)
     */

    const double prevalence = 0.01;
    const double sensitivity = 0.95;
    const double false_positive_rate = 0.10;

    require_probability(prevalence, "prevalence");
    require_probability(sensitivity, "sensitivity");
    require_probability(
        false_positive_rate,
        "false_positive_rate"
    );

    const double p_positive =
        sensitivity * prevalence
        + false_positive_rate * (1.0 - prevalence);

    const double posterior =
        sensitivity * prevalence / p_positive;

    print_probability("P(D)", prevalence);
    print_probability("P(+ | D)", sensitivity);
    print_probability("P(+ | not D)", false_positive_rate);
    print_probability("P(+)", p_positive);
    print_probability("P(D | +)", posterior);
}

// -----------------------------------------------------------------------------
// Law of total probability
// -----------------------------------------------------------------------------

void demonstrate_total_probability() {
    std::cout << "\n--- Law of total probability ---\n";

    struct Factory {
        double selection_probability;
        double defect_probability;
    };

    const std::map<std::string, Factory> factories{
        {"Factory A", {0.50, 0.01}},
        {"Factory B", {0.30, 0.03}},
        {"Factory C", {0.20, 0.05}}
    };

    double total_defect_probability = 0.0;

    for (const auto& [name, factory] : factories) {
        (void)name;

        total_defect_probability +=
            factory.selection_probability
            * factory.defect_probability;
    }

    print_probability(
        "P(defect)",
        total_defect_probability
    );
}

// -----------------------------------------------------------------------------
// Independence
// -----------------------------------------------------------------------------

void demonstrate_independence() {
    std::cout << "\n--- Independence ---\n";

    FiniteProbabilitySpace<int> die({
        {1, 1.0 / 6.0},
        {2, 1.0 / 6.0},
        {3, 1.0 / 6.0},
        {4, 1.0 / 6.0},
        {5, 1.0 / 6.0},
        {6, 1.0 / 6.0}
    });

    Event<int> even{2, 4, 6};
    Event<int> greater_than_three{4, 5, 6};

    const double p_a = die.probability(even);
    const double p_b = die.probability(greater_than_three);
    const double p_ab =
        die.probability(
            die.intersection(even, greater_than_three)
        );

    print_probability("P(A)", p_a);
    print_probability("P(B)", p_b);
    print_probability("P(A intersection B)", p_ab);
    print_probability("P(A)P(B)", p_a * p_b);

    std::cout
        << "Independent: "
        << std::boolalpha
        << die.independent(even, greater_than_three)
        << '\n';

    /*
     * Mutual exclusion is different from independence.
     *
     * If A and B cannot occur together:
     *
     *     P(A intersection B) = 0.
     *
     * For independence we would need:
     *
     *     P(A)P(B) = 0.
     *
     * Thus two mutually exclusive events with positive probabilities
     * cannot be independent.
     */
}

// -----------------------------------------------------------------------------
// Discrete random variables
// -----------------------------------------------------------------------------

template <typename Outcome, typename Function>
double expected_value(
    const FiniteProbabilitySpace<Outcome>& space,
    Function value_function
) {
    double result = 0.0;

    for (const auto& [outcome, probability] : space.probabilities()) {
        result += probability * value_function(outcome);
    }

    return result;
}

template <typename Outcome, typename Function>
double variance(
    const FiniteProbabilitySpace<Outcome>& space,
    Function value_function
) {
    const double mean =
        expected_value(space, value_function);

    double result = 0.0;

    for (const auto& [outcome, probability] :
         space.probabilities()) {

        const double value = value_function(outcome);
        result += probability * (value - mean) * (value - mean);
    }

    return result;
}

void demonstrate_random_variables() {
    std::cout << "\n--- Random variables ---\n";

    FiniteProbabilitySpace<int> die({
        {1, 1.0 / 6.0},
        {2, 1.0 / 6.0},
        {3, 1.0 / 6.0},
        {4, 1.0 / 6.0},
        {5, 1.0 / 6.0},
        {6, 1.0 / 6.0}
    });

    const double mean =
        expected_value(
            die,
            [](int outcome) {
                return static_cast<double>(outcome);
            }
        );

    const double variance_value =
        variance(
            die,
            [](int outcome) {
                return static_cast<double>(outcome);
            }
        );

    print_probability("E[X]", mean);
    print_probability("Var(X)", variance_value);
    print_probability("Std(X)", std::sqrt(variance_value));

    Event<int> at_least_four{4, 5, 6};

    /*
     * Indicator random variable:
     *
     *     I_A = 1 when A occurs
     *     I_A = 0 otherwise
     *
     * Its expectation is:
     *
     *     E[I_A] = P(A)
     */
    const double indicator_expectation =
        expected_value(
            die,
            [&at_least_four](int outcome) {
                return at_least_four.count(outcome)
                    ? 1.0
                    : 0.0;
            }
        );

    print_probability(
        "E[I_A]",
        indicator_expectation
    );

    print_probability(
        "P(X >= 4)",
        die.probability(at_least_four)
    );
}

// -----------------------------------------------------------------------------
// Reliability components
// -----------------------------------------------------------------------------

class Component {
public:
    Component(std::string name, double reliability)
        : name_(std::move(name)),
          reliability_(reliability) {

        require_probability(
            reliability_,
            "component reliability"
        );
    }

    const std::string& name() const {
        return name_;
    }

    double reliability() const {
        return reliability_;
    }

    double failure_probability() const {
        return 1.0 - reliability_;
    }

private:
    std::string name_;
    double reliability_;
};

// -----------------------------------------------------------------------------
// Series and parallel reliability
// -----------------------------------------------------------------------------

double series_reliability(
    const std::vector<Component>& components
) {
    /*
     * Series system:
     *
     * Every component must succeed.
     *
     * Assuming independent failures:
     *
     *     R_series = product(R_i)
     *
     * Empty series systems are mathematically assigned reliability 1,
     * because the empty conjunction is true.
     */
    double result = 1.0;

    for (const auto& component : components) {
        result *= component.reliability();
    }

    return result;
}

double parallel_reliability(
    const std::vector<Component>& components
) {
    /*
     * Parallel system:
     *
     * At least one component must succeed.
     *
     * It is easier to calculate the complement:
     *
     *     R_parallel
     *       = 1 - P(all components fail)
     *
     * Under independence:
     *
     *     P(all fail) = product(1 - R_i)
     */
    double all_fail = 1.0;

    for (const auto& component : components) {
        all_fail *= component.failure_probability();
    }

    return 1.0 - all_fail;
}

// -----------------------------------------------------------------------------
// Production-style service model
// -----------------------------------------------------------------------------

class ServiceReliabilityModel {
public:
    ServiceReliabilityModel(
        std::vector<Component> application_servers,
        Component database
    )
        : application_servers_(std::move(application_servers)),
          database_(std::move(database)) {

        if (application_servers_.empty()) {
            throw std::invalid_argument(
                "At least one application server is required."
            );
        }
    }

    double application_tier_reliability() const {
        return parallel_reliability(application_servers_);
    }

    double total_reliability() const {
        /*
         * The request succeeds when:
         *
         *     application tier succeeds
         *     AND
         *     database succeeds
         *
         * If these subsystems are independent:
         *
         *     R_total = R_application * R_database
         */
        return application_tier_reliability()
            * database_.reliability();
    }

    double failure_probability() const {
        return 1.0 - total_reliability();
    }

    void print_report() const {
        std::cout << "\n--- Production service reliability ---\n";

        for (const auto& server : application_servers_) {
            print_probability(
                "Reliability: " + server.name(),
                server.reliability()
            );
        }

        print_probability(
            "Reliability: " + database_.name(),
            database_.reliability()
        );

        print_probability(
            "Application tier reliability",
            application_tier_reliability()
        );

        print_probability(
            "Total service reliability",
            total_reliability()
        );

        print_probability(
            "Total service failure probability",
            failure_probability()
        );
    }

private:
    std::vector<Component> application_servers_;
    Component database_;
};

// -----------------------------------------------------------------------------
// Monte Carlo verification of the reliability model
// -----------------------------------------------------------------------------

struct SimulationResult {
    std::size_t trials;
    std::size_t successful_requests;

    double estimated_reliability() const {
        return static_cast<double>(successful_requests)
            / static_cast<double>(trials);
    }
};

SimulationResult simulate_service(
    const ServiceReliabilityModel& model,
    std::size_t trials,
    std::uint32_t seed
) {
    if (trials == 0) {
        throw std::invalid_argument(
            "Simulation requires at least one trial."
        );
    }

    /*
     * The model does not expose internal random behavior, so the simulation
     * below recreates the same architectural probabilities:
     *
     * - each application server independently succeeds;
     * - the database independently succeeds;
     * - a request succeeds when at least one server and the database succeed.
     *
     * For a production simulator, component configuration would normally be
     * exposed through a structured model rather than duplicated here.
     *
     * This case study intentionally keeps the architecture explicit.
     */

    // The application tier in this model consists of two servers.
    // The function is therefore validated against the corresponding
    // production configuration below.
    if (model.total_reliability() < 0.0 ||
        model.total_reliability() > 1.0) {
        throw std::logic_error(
            "Model produced an invalid reliability."
        );
    }

    /*
     * To make the simulation generic without exposing implementation details,
     * simulate the equivalent Bernoulli event using the exact system
     * reliability. This verifies the aggregate probability numerically.
     *
     * It does not simulate individual component failures. A component-level
     * simulator would be needed when failure causes, repair times or
     * correlated failures are part of the study.
     */
    std::mt19937 generator(seed);
    std::bernoulli_distribution request_succeeds(
        model.total_reliability()
    );

    std::size_t successes = 0;

    for (std::size_t trial = 0; trial < trials; ++trial) {
        if (request_succeeds(generator)) {
            ++successes;
        }
    }

    return {trials, successes};
}

// -----------------------------------------------------------------------------
// Conditional reliability analysis
// -----------------------------------------------------------------------------

double posterior_failure_probability(
    double prior_failure_probability,
    double alarm_probability_given_failure,
    double alarm_probability_given_success
) {
    /*
     * Bayes' theorem can also be used for reliability monitoring.
     *
     * Let F = component failure.
     * Let A = monitoring alarm.
     *
     * We calculate:
     *
     *     P(F | A)
     *
     * = P(A | F)P(F)
     *   ----------------------------
     *   P(A | F)P(F) + P(A | not F)P(not F)
     */
    require_probability(
        prior_failure_probability,
        "prior_failure_probability"
    );

    require_probability(
        alarm_probability_given_failure,
        "alarm_probability_given_failure"
    );

    require_probability(
        alarm_probability_given_success,
        "alarm_probability_given_success"
    );

    const double p_alarm =
        alarm_probability_given_failure
            * prior_failure_probability
        + alarm_probability_given_success
            * (1.0 - prior_failure_probability);

    if (approximately_equal(p_alarm, 0.0)) {
        throw std::domain_error(
            "Cannot condition on an event with zero probability."
        );
    }

    return
        alarm_probability_given_failure
        * prior_failure_probability
        / p_alarm;
}

// -----------------------------------------------------------------------------
// Edge-case demonstration
// -----------------------------------------------------------------------------

void demonstrate_edge_cases() {
    std::cout << "\n--- Edge cases ---\n";

    // Empty event has probability zero.
    FiniteProbabilitySpace<int> die({
        {1, 1.0 / 6.0},
        {2, 1.0 / 6.0},
        {3, 1.0 / 6.0},
        {4, 1.0 / 6.0},
        {5, 1.0 / 6.0},
        {6, 1.0 / 6.0}
    });

    const Event<int> empty_event;

    print_probability(
        "P(empty event)",
        die.probability(empty_event)
    );

    // Conditioning on an impossible event is undefined.
    FiniteProbabilitySpace<std::string> deterministic({
        {"Success", 1.0},
        {"Failure", 0.0}
    });

    try {
        deterministic.conditional_probability(
            Event<std::string>{"Success"},
            Event<std::string>{"Failure"}
        );
    } catch (const std::domain_error& error) {
        std::cout
            << "Correctly rejected zero-probability condition: "
            << error.what()
            << '\n';
    }

    // Events must be subsets of the sample space.
    try {
        die.probability(Event<int>{99});
    } catch (const std::invalid_argument& error) {
        std::cout
            << "Correctly rejected invalid event: "
            << error.what()
            << '\n';
    }
}

// -----------------------------------------------------------------------------
// Numerical considerations
// -----------------------------------------------------------------------------

void demonstrate_numerical_considerations() {
    std::cout << "\n--- Numerical considerations ---\n";

    const std::vector<double> probabilities{
        0.1, 0.2, 0.3, 0.4
    };

    double total = 0.0;

    for (double probability : probabilities) {
        total += probability;
    }

    std::cout
        << std::setprecision(17)
        << "Floating-point total: "
        << total
        << '\n';

    std::cout
        << "Approximately one: "
        << std::boolalpha
        << approximately_equal(total, 1.0)
        << '\n';

    /*
     * Exact equality is usually inappropriate when probabilities are
     * generated by floating-point arithmetic.
     *
     * A tolerance should be chosen according to the numerical scale and
     * accumulated error of the calculation.
     */
}

// -----------------------------------------------------------------------------
// Validation tests
// -----------------------------------------------------------------------------

void run_tests() {
    std::cout << "\n--- Self-tests ---\n";

    FiniteProbabilitySpace<int> die({
        {1, 1.0 / 6.0},
        {2, 1.0 / 6.0},
        {3, 1.0 / 6.0},
        {4, 1.0 / 6.0},
        {5, 1.0 / 6.0},
        {6, 1.0 / 6.0}
    });

    Event<int> all{1, 2, 3, 4, 5, 6};
    Event<int> empty;

    if (!approximately_equal(die.probability(all), 1.0)) {
        throw std::logic_error("P(Omega) != 1.");
    }

    if (!approximately_equal(die.probability(empty), 0.0)) {
        throw std::logic_error("P(empty) != 0.");
    }

    Event<int> even{2, 4, 6};
    Event<int> greater_than_three{4, 5, 6};

    const auto intersection =
        die.intersection(even, greater_than_three);

    const auto union_event =
        die.union_events(even, greater_than_three);

    const double inclusion_exclusion =
        die.probability(even)
        + die.probability(greater_than_three)
        - die.probability(intersection);

    if (!approximately_equal(
            die.probability(union_event),
            inclusion_exclusion)) {
        throw std::logic_error(
            "Inclusion-exclusion test failed."
        );
    }

    const double conditional =
        die.conditional_probability(
            even,
            greater_than_three
        );

    if (!approximately_equal(
            die.probability(intersection),
            conditional
                * die.probability(greater_than_three))) {
        throw std::logic_error(
            "Conditional probability test failed."
        );
    }

    if (die.independent(even, greater_than_three)) {
        throw std::logic_error(
            "Expected selected events to be dependent."
        );
    }

    std::cout << "All tests passed.\n";
}

// -----------------------------------------------------------------------------
// Main technical case study
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout
            << "============================================================\n"
            << "Probability Fundamentals: C++ Case Study\n"
            << "============================================================\n";

        demonstrate_basic_probability_space();
        demonstrate_event_algebra();
        demonstrate_conditional_probability();
        demonstrate_bayes_theorem();
        demonstrate_total_probability();
        demonstrate_independence();
        demonstrate_random_variables();

        /*
         * Production architecture:
         *
         * Two application servers are redundant. The database is a required
         * dependency. The calculations assume independent component states.
         */
        const std::vector<Component> servers{
            Component("Application Server A", 0.98),
            Component("Application Server B", 0.97)
        };

        Component database(
            "Primary Database",
            0.995
        );

        ServiceReliabilityModel service(
            servers,
            database
        );

        service.print_report();

        /*
         * Monte Carlo verification.
         *
         * The analytical value is exact under the model. Simulation should
         * approach that value as the number of trials increases.
         */
        const std::size_t trials = 1'000'000;

        const SimulationResult simulation =
            simulate_service(
                service,
                trials,
                20260917
            );

        std::cout << "\n--- Monte Carlo verification ---\n";

        print_probability(
            "Analytical reliability",
            service.total_reliability()
        );

        print_probability(
            "Simulated reliability",
            simulation.estimated_reliability()
        );

        print_probability(
            "Absolute simulation error",
            std::abs(
                simulation.estimated_reliability()
                - service.total_reliability()
            )
        );

        /*
         * Reliability monitoring example.
         *
         * A component has a 2% prior failure probability.
         * An alarm is raised:
         *   90% of the time when the component has failed.
         *   5% of the time when it has not failed.
         *
         * Bayes' theorem estimates the probability of actual failure
         * after observing an alarm.
         */
        const double posterior_failure =
            posterior_failure_probability(
                0.02,
                0.90,
                0.05
            );

        std::cout << "\n--- Reliability monitoring ---\n";

        print_probability(
            "P(failure | alarm)",
            posterior_failure
        );

        demonstrate_edge_cases();
        demonstrate_numerical_considerations();
        run_tests();

        std::cout
            << "\n============================================================\n"
            << "Case study completed successfully.\n"
            << "============================================================\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << '\n';

        return 1;
    }
}
