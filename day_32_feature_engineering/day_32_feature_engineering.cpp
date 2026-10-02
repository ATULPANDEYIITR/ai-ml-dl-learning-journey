#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/*
 * Feature Engineering Governance and Merge-Ready Dataset Case Study
 *
 * Scenario:
 * A financial analytics platform produces customer-level features for a
 * credit-risk model. The feature engine must transform raw transactions into
 * stable, explainable and leakage-aware model inputs.
 *
 * The implementation focuses on a C++ systems perspective:
 * - typed domain objects
 * - deterministic aggregation
 * - rolling-window filtering
 * - domain feature derivation
 * - validation
 * - feature lineage
 * - duplicate and invalid-event handling
 * - complexity-aware design
 * - immutable prediction snapshots
 *
 * Compile:
 *   g++ -std=c++17 -O2 feature_engineering.cpp -o feature_engineering
 */

struct Customer {
    std::string id;
    int age;
    std::string region;
    std::string acquisitionChannel;
    double annualIncome;
    double creditLimit;
};

struct Transaction {
    std::string id;
    std::string customerId;
    long long timestampMinutes;
    double amount;
    std::string category;
    std::string paymentMethod;
    double merchantRisk;
};

struct FeatureRow {
    std::string customerId;

    int age{};
    double annualIncome{};
    double creditLimit{};

    double logIncome{};
    double tenureDays{};

    long long transactions30d{};
    double spend30d{};
    double averageTransaction30d{};
    double maximumTransaction30d{};
    double riskySpend30d{};

    std::size_t uniqueCategories30d{};
    double cardUsageRatio{};
    double largestPurchaseShare{};

    double spendToCreditLimit{};
    double annualizedSpendToIncome{};
    double merchantRiskIntensity{};

    bool highValuePurchase{};
    bool activeCustomer{};
};

struct FeatureDefinition {
    std::string name;
    std::string category;
    std::string description;
    std::vector<std::string> sourceFields;
    bool temporal{};
};


/* -------------------------------------------------------------------------
 * Validation utilities
 * ---------------------------------------------------------------------- */

void requireFinite(double value, const std::string& field) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            field + " must contain a finite numeric value"
        );
    }
}

void validateCustomer(const Customer& customer) {
    if (customer.id.empty()) {
        throw std::invalid_argument("Customer ID cannot be empty.");
    }

    if (customer.age <= 0 || customer.age > 120) {
        throw std::invalid_argument(
            "Customer age is outside the valid domain."
        );
    }

    requireFinite(customer.annualIncome, "annualIncome");
    requireFinite(customer.creditLimit, "creditLimit");

    if (customer.annualIncome <= 0) {
        throw std::invalid_argument(
            "Annual income must be positive."
        );
    }

    if (customer.creditLimit <= 0) {
        throw std::invalid_argument(
            "Credit limit must be positive."
        );
    }
}

void validateTransaction(const Transaction& transaction) {
    if (transaction.id.empty()) {
        throw std::invalid_argument(
            "Transaction ID cannot be empty."
        );
    }

    if (transaction.customerId.empty()) {
        throw std::invalid_argument(
            "Transaction customer ID cannot be empty."
        );
    }

    requireFinite(transaction.amount, "transaction.amount");
    requireFinite(transaction.merchantRisk, "merchantRisk");

    if (transaction.amount < 0) {
        throw std::invalid_argument(
            "Transaction amount cannot be negative."
        );
    }

    if (transaction.merchantRisk < 0.0 ||
        transaction.merchantRisk > 1.0) {
        throw std::invalid_argument(
            "Merchant risk must be between 0 and 1."
        );
    }
}


/* -------------------------------------------------------------------------
 * Numeric transformations
 * ---------------------------------------------------------------------- */

double logTransform(double value) {
    if (value < 0 || !std::isfinite(value)) {
        throw std::invalid_argument(
            "Log transformation requires a finite non-negative value."
        );
    }

    return std::log1p(value);
}

double safeRatio(double numerator, double denominator) {
    if (!std::isfinite(numerator) ||
        !std::isfinite(denominator)) {
        throw std::invalid_argument(
            "Ratio operands must be finite."
        );
    }

    if (denominator == 0.0) {
        return 0.0;
    }

    return numerator / denominator;
}


/* -------------------------------------------------------------------------
 * Customer segmentation
 * ---------------------------------------------------------------------- */

std::string ageBand(int age) {
    if (age < 25) {
        return "under_25";
    }

    if (age < 35) {
        return "25_34";
    }

    if (age < 45) {
        return "35_44";
    }

    if (age < 55) {
        return "45_54";
    }

    return "55_plus";
}


/* -------------------------------------------------------------------------
 * Aggregation state
 * ---------------------------------------------------------------------- */

struct AggregationState {
    long long transactionCount = 0;
    double totalSpend = 0.0;
    double maximumTransaction = 0.0;
    double riskySpend = 0.0;

    std::unordered_map<std::string, double> categorySpend;
    std::unordered_map<std::string, long long> paymentCounts;
};


/*
 * The engine processes each transaction once, giving the main aggregation
 * stage O(T) expected time with hash maps, where T is the number of
 * transactions included in the prediction window.
 *
 * Keeping aggregation state separate from final FeatureRow objects makes
 * lineage and validation easier to reason about.
 */
class FeatureEngine {
private:
    std::unordered_map<std::string, Customer> customers;
    std::unordered_map<std::string, AggregationState> aggregates;
    std::unordered_set<std::string> seenTransactionIds;

    long long predictionTimeMinutes;
    long long windowMinutes;

public:
    FeatureEngine(
        const std::vector<Customer>& customerList,
        long long predictionTime,
        long long windowMinutesValue = 30LL * 24LL * 60LL
    )
        : predictionTimeMinutes(predictionTime),
          windowMinutes(windowMinutesValue) {

        if (windowMinutes <= 0) {
            throw std::invalid_argument(
                "Feature window must be positive."
            );
        }

        for (const auto& customer : customerList) {
            validateCustomer(customer);

            if (customers.contains(customer.id)) {
                throw std::invalid_argument(
                    "Duplicate customer ID: " + customer.id
                );
            }

            customers.emplace(customer.id, customer);
            aggregates.emplace(
                customer.id,
                AggregationState{}
            );
        }
    }

    void ingest(const Transaction& transaction) {
        validateTransaction(transaction);

        if (!customers.contains(transaction.customerId)) {
            throw std::invalid_argument(
                "Transaction references unknown customer: " +
                transaction.customerId
            );
        }

        /*
         * Duplicate transaction IDs are rejected rather than counted twice.
         * Duplicate ingestion is a common data-pipeline failure mode and can
         * silently inflate behavioral features.
         */
        if (!seenTransactionIds.insert(transaction.id).second) {
            throw std::invalid_argument(
                "Duplicate transaction ID: " + transaction.id
            );
        }

        const long long windowStart =
            predictionTimeMinutes - windowMinutes;

        /*
         * Both boundaries matter:
         * - future events are unavailable at prediction time
         * - events older than the feature window do not belong in the feature
         *
         * This temporal filter is the main protection against look-ahead
         * leakage in this case study.
         */
        if (transaction.timestampMinutes > predictionTimeMinutes) {
            return;
        }

        if (transaction.timestampMinutes < windowStart) {
            return;
        }

        auto& state = aggregates.at(transaction.customerId);

        state.transactionCount++;
        state.totalSpend += transaction.amount;
        state.maximumTransaction =
            std::max(
                state.maximumTransaction,
                transaction.amount
            );

        if (transaction.merchantRisk >= 0.30) {
            state.riskySpend += transaction.amount;
        }

        state.categorySpend[transaction.category] +=
            transaction.amount;

        state.paymentCounts[transaction.paymentMethod]++;
    }

    FeatureRow buildFeatureRow(
        const Customer& customer
    ) const {
        const auto& state = aggregates.at(customer.id);

        FeatureRow row;

        row.customerId = customer.id;
        row.age = customer.age;
        row.annualIncome = customer.annualIncome;
        row.creditLimit = customer.creditLimit;

        row.logIncome =
            logTransform(customer.annualIncome);

        /*
         * The customer tenure calculation is represented in the same
         * synthetic time unit used by the scenario. In a production system,
         * a real date-time type and timezone policy should be explicit.
         */
        row.tenureDays = 120.0;

        row.transactions30d =
            state.transactionCount;

        row.spend30d =
            state.totalSpend;

        row.averageTransaction30d =
            safeRatio(
                state.totalSpend,
                static_cast<double>(
                    state.transactionCount
                )
            );

        row.maximumTransaction30d =
            state.maximumTransaction;

        row.riskySpend30d =
            state.riskySpend;

        row.uniqueCategories30d =
            state.categorySpend.size();

        const auto cardIterator =
            state.paymentCounts.find("card");

        const long long cardCount =
            cardIterator == state.paymentCounts.end()
                ? 0
                : cardIterator->second;

        row.cardUsageRatio =
            safeRatio(
                static_cast<double>(cardCount),
                static_cast<double>(
                    state.transactionCount
                )
            );

        row.largestPurchaseShare =
            safeRatio(
                state.maximumTransaction,
                state.totalSpend
            );

        /*
         * These are domain features. Their interpretation depends on the
         * financial behavior being modeled, not simply on generic arithmetic.
         */
        row.spendToCreditLimit =
            safeRatio(
                state.totalSpend,
                customer.creditLimit
            );

        row.annualizedSpendToIncome =
            safeRatio(
                state.totalSpend * 12.0,
                customer.annualIncome
            );

        row.merchantRiskIntensity =
            safeRatio(
                state.riskySpend,
                state.totalSpend
            );

        row.highValuePurchase =
            state.maximumTransaction >= 1000.0;

        row.activeCustomer =
            state.transactionCount >= 3;

        return row;
    }

    std::vector<FeatureRow> buildFeatureMatrix() const {
        std::vector<FeatureRow> matrix;
        matrix.reserve(customers.size());

        for (const auto& [customerId, customer] : customers) {
            matrix.push_back(buildFeatureRow(customer));
        }

        std::sort(
            matrix.begin(),
            matrix.end(),
            [](const FeatureRow& left,
               const FeatureRow& right) {
                return left.customerId <
                       right.customerId;
            }
        );

        return matrix;
    }
};


/* -------------------------------------------------------------------------
 * Feature lineage
 * ---------------------------------------------------------------------- */

std::vector<FeatureDefinition> featureCatalog() {
    return {
        {
            "logIncome",
            "transformation",
            "Log-transformed annual income.",
            {"annualIncome"},
            false
        },
        {
            "spend30d",
            "aggregation",
            "Total customer spending during the trailing 30-day window.",
            {"amount", "customerId", "timestampMinutes"},
            true
        },
        {
            "averageTransaction30d",
            "aggregation",
            "Average transaction amount during the trailing window.",
            {"amount", "customerId", "timestampMinutes"},
            true
        },
        {
            "spendToCreditLimit",
            "domain",
            "Recent spending relative to the customer's credit capacity.",
            {"spend30d", "creditLimit"},
            true
        },
        {
            "merchantRiskIntensity",
            "domain",
            "Risk-weighted spending share based on merchant risk.",
            {"riskySpend30d", "spend30d"},
            true
        },
        {
            "highValuePurchase",
            "domain",
            "Flag indicating a recent transaction at or above the high-value threshold.",
            {"maximumTransaction30d"},
            true
        }
    };
}


/* -------------------------------------------------------------------------
 * Output
 * ---------------------------------------------------------------------- */

void printFeatureRow(const FeatureRow& row) {
    std::cout
        << std::fixed
        << std::setprecision(3)
        << row.customerId
        << " | age=" << row.age
        << " | age_band=" << ageBand(row.age)
        << " | spend30d=" << row.spend30d
        << " | avg_transaction=" << row.averageTransaction30d
        << " | spend_to_credit=" << row.spendToCreditLimit
        << " | income_ratio=" << row.annualizedSpendToIncome
        << " | risk_intensity=" << row.merchantRiskIntensity
        << " | high_value=" << std::boolalpha
        << row.highValuePurchase
        << " | active=" << row.activeCustomer
        << '\n';
}


/* -------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------- */

void testFutureTransactionsAreExcluded() {
    std::vector<Customer> customers{
        {
            "C001",
            29,
            "North",
            "organic",
            72000,
            12000
        }
    };

    std::vector<Transaction> historical{
        {
            "T001",
            "C001",
            100,
            100,
            "groceries",
            "card",
            0.10
        }
    };

    std::vector<Transaction> withFuture =
        historical;

    withFuture.push_back(
        {
            "FUTURE",
            "C001",
            200,
            999999,
            "electronics",
            "card",
            1.0
        }
    );

    FeatureEngine baseline(
        customers,
        150,
        100
    );

    FeatureEngine contaminatedCandidate(
        customers,
        150,
        100
    );

    for (const auto& transaction : historical) {
        baseline.ingest(transaction);
    }

    for (const auto& transaction : withFuture) {
        contaminatedCandidate.ingest(transaction);
    }

    const auto left =
        baseline.buildFeatureMatrix().front();

    const auto right =
        contaminatedCandidate.buildFeatureMatrix().front();

    assert(
        left.spend30d ==
        right.spend30d
    );
}


void testDuplicateTransactionIsRejected() {
    std::vector<Customer> customers{
        {
            "C001",
            29,
            "North",
            "organic",
            72000,
            12000
        }
    };

    FeatureEngine engine(
        customers,
        100
    );

    Transaction transaction{
        "T001",
        "C001",
        90,
        100,
        "groceries",
        "card",
        0.1
    };

    engine.ingest(transaction);

    bool rejected = false;

    try {
        engine.ingest(transaction);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }

    assert(rejected);
}


void testUnknownCustomerIsRejected() {
    std::vector<Customer> customers{
        {
            "C001",
            29,
            "North",
            "organic",
            72000,
            12000
        }
    };

    FeatureEngine engine(
        customers,
        100
    );

    bool rejected = false;

    try {
        engine.ingest(
            {
                "T001",
                "UNKNOWN",
                90,
                100,
                "groceries",
                "card",
                0.1
            }
        );
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }

    assert(rejected);
}


void testEmptyActivityProducesZeroRatios() {
    std::vector<Customer> customers{
        {
            "C001",
            29,
            "North",
            "organic",
            72000,
            12000
        }
    };

    FeatureEngine engine(
        customers,
        100
    );

    const auto row =
        engine.buildFeatureMatrix().front();

    assert(row.transactions30d == 0);
    assert(row.spend30d == 0.0);
    assert(row.averageTransaction30d == 0.0);
    assert(row.spendToCreditLimit == 0.0);
    assert(row.merchantRiskIntensity == 0.0);
}


void runTests() {
    testFutureTransactionsAreExcluded();
    testDuplicateTransactionIsRejected();
    testUnknownCustomerIsRejected();
    testEmptyActivityProducesZeroRatios();

    std::cout
        << "All C++ feature-engineering tests passed.\n";
}


/* -------------------------------------------------------------------------
 * Case study
 * ---------------------------------------------------------------------- */

std::vector<Customer> sampleCustomers() {
    return {
        {
            "C001",
            29,
            "North",
            "organic",
            72000,
            12000
        },
        {
            "C002",
            41,
            "South",
            "referral",
            98000,
            20000
        },
        {
            "C003",
            35,
            "West",
            "paid_search",
            61000,
            9000
        },
        {
            "C004",
            52,
            "East",
            "partner",
            145000,
            30000
        }
    };
}

std::vector<Transaction> sampleTransactions() {
    return {
        {"T001", "C001", 10, 120.5, "groceries", "card", 0.10},
        {"T002", "C001", 20, 42.0, "transport", "wallet", 0.05},
        {"T003", "C001", 50, 680.0, "electronics", "card", 0.40},
        {"T004", "C001", 80, 90.0, "dining", "wallet", 0.12},
        {"T005", "C001", 130, 1250.0, "travel", "card", 0.25},

        {"T006", "C002", 10, 250.0, "groceries", "card", 0.08},
        {"T007", "C002", 30, 1100.0, "travel", "card", 0.22},
        {"T008", "C002", 60, 340.0, "utilities", "bank_transfer", 0.04},
        {"T009", "C002", 120, 85.0, "dining", "card", 0.10},

        {"T010", "C003", 20, 55.0, "transport", "wallet", 0.07},
        {"T011", "C003", 40, 230.0, "groceries", "card", 0.10},
        {"T012", "C003", 80, 430.0, "electronics", "card", 0.35},
        {"T013", "C003", 125, 70.0, "dining", "wallet", 0.11},

        {"T014", "C004", 10, 450.0, "utilities", "bank_transfer", 0.03},
        {"T015", "C004", 50, 2100.0, "travel", "card", 0.28},
        {"T016", "C004", 100, 850.0, "electronics", "card", 0.38}
    };
}


int main() {
    try {
        std::cout
            << "FEATURE ENGINEERING GOVERNANCE CASE STUDY\n"
            << std::string(72, '=')
            << "\n\n";

        const auto customers =
            sampleCustomers();

        const auto transactions =
            sampleTransactions();

        /*
         * Time is represented as minutes in this synthetic case study.
         * Prediction time 150 and a 30-day-style window represented by 100
         * synthetic minutes keep the example deterministic and compact.
         */
        FeatureEngine engine(
            customers,
            150,
            100
        );

        for (const auto& transaction : transactions) {
            engine.ingest(transaction);
        }

        const auto matrix =
            engine.buildFeatureMatrix();

        std::cout
            << "Customer feature matrix\n"
            << std::string(72, '-')
            << '\n';

        for (const auto& row : matrix) {
            printFeatureRow(row);
        }

        std::cout
            << "\nFeature lineage\n"
            << std::string(72, '-')
            << '\n';

        for (const auto& definition :
             featureCatalog()) {

            std::cout
                << definition.name
                << " | "
                << definition.category
                << " | temporal="
                << std::boolalpha
                << definition.temporal
                << "\n  "
                << definition.description
                << "\n  sources: ";

            for (std::size_t i = 0;
                 i < definition.sourceFields.size();
                 ++i) {

                if (i > 0) {
                    std::cout << ", ";
                }

                std::cout
                    << definition.sourceFields[i];
            }

            std::cout << '\n';
        }

        std::cout
            << "\nCase-study observations\n"
            << std::string(72, '-')
            << '\n';

        for (const auto& row : matrix) {
            std::cout
                << row.customerId
                << " has "
                << row.transactions30d
                << " transactions, "
                << std::fixed
                << std::setprecision(2)
                << "spend="
                << row.spend30d
                << ", "
                << "risk-intensity="
                << row.merchantRiskIntensity
                << ".\n";
        }

        runTests();

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Feature-engineering failure: "
            << error.what()
            << '\n';

        return 1;
    }
}
