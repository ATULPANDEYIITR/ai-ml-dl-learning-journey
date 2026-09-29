/*
 * DATA UNDERSTANDING CASE STUDY
 *
 * Topic:
 *   Features, labels, observations, variables, and target variables
 *
 * Scenario:
 *   A financial-services company wants to understand customer records and
 *   build a purchase-propensity data pipeline.
 *
 * The program demonstrates:
 *   - observations
 *   - variables
 *   - features
 *   - target variables
 *   - identifiers
 *   - numerical and categorical data
 *   - validation
 *   - missing values
 *   - duplicates
 *   - feature engineering
 *   - train/test splitting
 *   - class distributions
 *   - data leakage detection
 *   - temporal considerations
 *   - data contracts
 *   - complexity and design considerations
 *
 * Compile:
 *   g++ -std=c++17 -O2 data_understanding.cpp -o data_understanding
 *
 * Run:
 *   ./data_understanding
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


// -----------------------------------------------------------------------------
// 1. DOMAIN MODEL
// -----------------------------------------------------------------------------

struct Customer {
    std::string customerId;
    int age;
    double annualIncome;
    std::string city;
    std::string membership;
    int visitsPerMonth;
    int purchased;
};


/*
 * An Observation represents one customer in this case study.
 *
 * The structure deliberately distinguishes:
 *
 *   customerId      -> identifier
 *   age             -> numerical feature
 *   annualIncome    -> numerical feature
 *   city            -> categorical feature
 *   membership      -> categorical feature
 *   visitsPerMonth  -> numerical feature
 *   purchased       -> target
 *
 * This distinction is semantic. C++ types alone do not determine whether
 * a variable is a feature or a target.
 */


// -----------------------------------------------------------------------------
// 2. DATASET CONTAINER
// -----------------------------------------------------------------------------

class CustomerDataset {
private:
    std::vector<Customer> rows;

public:
    explicit CustomerDataset(std::vector<Customer> customers)
        : rows(std::move(customers)) {}

    const std::vector<Customer>& observations() const {
        return rows;
    }

    std::size_t size() const {
        return rows.size();
    }

    bool empty() const {
        return rows.empty();
    }
};


// -----------------------------------------------------------------------------
// 3. DATA DICTIONARY
// -----------------------------------------------------------------------------

enum class VariableRole {
    Identifier,
    Feature,
    Target
};

enum class VariableType {
    Numerical,
    NominalCategorical,
    OrdinalCategorical,
    Temporal,
    Text
};

struct VariableDefinition {
    std::string name;
    VariableRole role;
    VariableType type;
    bool nullable;
    bool availableAtPrediction;
    std::string description;
};


std::string roleToString(VariableRole role) {
    switch (role) {
        case VariableRole::Identifier:
            return "identifier";
        case VariableRole::Feature:
            return "feature";
        case VariableRole::Target:
            return "target";
    }

    return "unknown";
}


std::string typeToString(VariableType type) {
    switch (type) {
        case VariableType::Numerical:
            return "numerical";
        case VariableType::NominalCategorical:
            return "nominal categorical";
        case VariableType::OrdinalCategorical:
            return "ordinal categorical";
        case VariableType::Temporal:
            return "temporal";
        case VariableType::Text:
            return "text";
    }

    return "unknown";
}


// -----------------------------------------------------------------------------
// 4. PRINT DATA DICTIONARY
// -----------------------------------------------------------------------------

void printDataDictionary(
    const std::vector<VariableDefinition>& definitions
) {
    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "DATA DICTIONARY\n";
    std::cout << std::string(80, '=') << "\n";

    for (const auto& definition : definitions) {
        std::cout
            << definition.name
            << " | role=" << roleToString(definition.role)
            << " | type=" << typeToString(definition.type)
            << " | nullable=" << std::boolalpha
            << definition.nullable
            << " | prediction-time="
            << definition.availableAtPrediction
            << "\n  "
            << definition.description
            << "\n";
    }
}


// -----------------------------------------------------------------------------
// 5. DATA VALIDATION
// -----------------------------------------------------------------------------

struct ValidationError {
    std::string field;
    std::string message;
};


std::vector<ValidationError> validateCustomer(
    const Customer& customer
) {
    std::vector<ValidationError> errors;

    if (customer.customerId.empty()) {
        errors.push_back({
            "customerId",
            "identifier cannot be empty"
        });
    }

    if (customer.age < 0 || customer.age > 120) {
        errors.push_back({
            "age",
            "age must be between 0 and 120"
        });
    }

    if (!std::isfinite(customer.annualIncome) ||
        customer.annualIncome < 0.0) {
        errors.push_back({
            "annualIncome",
            "income must be finite and non-negative"
        });
    }

    const std::set<std::string> validMemberships{
        "Basic",
        "Premium"
    };

    if (!validMemberships.contains(customer.membership)) {
        errors.push_back({
            "membership",
            "membership must be Basic or Premium"
        });
    }

    if (customer.visitsPerMonth < 0) {
        errors.push_back({
            "visitsPerMonth",
            "visits cannot be negative"
        });
    }

    if (customer.purchased != 0 &&
        customer.purchased != 1) {
        errors.push_back({
            "purchased",
            "binary target must be 0 or 1"
        });
    }

    return errors;
}


void printValidationErrors(
    const std::vector<ValidationError>& errors
) {
    if (errors.empty()) {
        std::cout << "Validation: PASS\n";
        return;
    }

    std::cout << "Validation: FAIL\n";

    for (const auto& error : errors) {
        std::cout
            << "  "
            << error.field
            << ": "
            << error.message
            << "\n";
    }
}


// -----------------------------------------------------------------------------
// 6. DUPLICATE DETECTION
// -----------------------------------------------------------------------------

std::vector<std::size_t> findDuplicateIds(
    const CustomerDataset& dataset
) {
    std::set<std::string> seen;
    std::vector<std::size_t> duplicates;

    for (std::size_t index = 0;
         index < dataset.observations().size();
         ++index) {

        const auto& customer =
            dataset.observations()[index];

        if (!seen.insert(customer.customerId).second) {
            duplicates.push_back(index);
        }
    }

    return duplicates;
}


// -----------------------------------------------------------------------------
// 7. CATEGORICAL VALUE COUNTS
// -----------------------------------------------------------------------------

template <typename Extractor>
std::map<std::string, std::size_t> countCategories(
    const CustomerDataset& dataset,
    Extractor extractor
) {
    std::map<std::string, std::size_t> counts;

    for (const auto& customer : dataset.observations()) {
        ++counts[extractor(customer)];
    }

    return counts;
}


void printCategoryCounts(
    const std::map<std::string, std::size_t>& counts
) {
    for (const auto& [category, count] : counts) {
        std::cout
            << "  "
            << category
            << ": "
            << count
            << "\n";
    }
}


// -----------------------------------------------------------------------------
// 8. NUMERICAL SUMMARY
// -----------------------------------------------------------------------------

struct NumericalSummary {
    double minimum;
    double maximum;
    double mean;
};


NumericalSummary summarize(
    const std::vector<double>& values
) {
    if (values.empty()) {
        throw std::invalid_argument(
            "Cannot summarize an empty vector."
        );
    }

    const auto [minimumIterator, maximumIterator] =
        std::minmax_element(
            values.begin(),
            values.end()
        );

    const double total =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        );

    return {
        *minimumIterator,
        *maximumIterator,
        total / static_cast<double>(values.size())
    };
}


// -----------------------------------------------------------------------------
// 9. FEATURE MATRIX
// -----------------------------------------------------------------------------

using FeatureVector = std::vector<double>;
using FeatureMatrix = std::vector<FeatureVector>;


FeatureMatrix createNumericFeatureMatrix(
    const CustomerDataset& dataset
) {
    FeatureMatrix matrix;

    for (const auto& customer : dataset.observations()) {
        /*
         * This is a deliberately numeric subset.
         *
         * Categorical variables require an encoding strategy before they can
         * be represented in a purely numerical matrix.
         */
        matrix.push_back({
            static_cast<double>(customer.age),
            customer.annualIncome,
            static_cast<double>(customer.visitsPerMonth)
        });
    }

    return matrix;
}


void printFeatureMatrix(
    const FeatureMatrix& matrix
) {
    std::cout
        << "Feature columns: "
        << "[age, annualIncome, visitsPerMonth]\n";

    for (const auto& row : matrix) {
        std::cout << "  ";

        for (double value : row) {
            std::cout
                << std::fixed
                << std::setprecision(2)
                << value
                << " ";
        }

        std::cout << "\n";
    }
}


// -----------------------------------------------------------------------------
// 10. TARGET VECTOR
// -----------------------------------------------------------------------------

std::vector<int> createTargetVector(
    const CustomerDataset& dataset
) {
    std::vector<int> targets;

    for (const auto& customer : dataset.observations()) {
        targets.push_back(customer.purchased);
    }

    return targets;
}


// -----------------------------------------------------------------------------
// 11. CLASS DISTRIBUTION
// -----------------------------------------------------------------------------

std::map<int, double> classDistribution(
    const std::vector<int>& targets
) {
    if (targets.empty()) {
        throw std::invalid_argument(
            "Target vector cannot be empty."
        );
    }

    std::map<int, std::size_t> counts;

    for (int target : targets) {
        ++counts[target];
    }

    std::map<int, double> proportions;

    for (const auto& [label, count] : counts) {
        proportions[label] =
            static_cast<double>(count) /
            static_cast<double>(targets.size());
    }

    return proportions;
}


// -----------------------------------------------------------------------------
// 12. FEATURE ENGINEERING
// -----------------------------------------------------------------------------

struct EnrichedCustomer {
    Customer customer;
    double incomePerVisit;
    bool highIncome;
    bool frequentVisitor;
};


EnrichedCustomer engineerFeatures(
    const Customer& customer
) {
    if (customer.visitsPerMonth <= 0) {
        throw std::domain_error(
            "Cannot calculate income-per-visit when "
            "visitsPerMonth is zero."
        );
    }

    return {
        customer,
        customer.annualIncome /
            static_cast<double>(customer.visitsPerMonth),
        customer.annualIncome >= 75000.0,
        customer.visitsPerMonth >= 6
    };
}


// -----------------------------------------------------------------------------
// 13. ONE-HOT ENCODING
// -----------------------------------------------------------------------------

struct EncodedMembership {
    double basic;
    double premium;
};


EncodedMembership encodeMembership(
    const Customer& customer
) {
    if (customer.membership != "Basic" &&
        customer.membership != "Premium") {
        throw std::invalid_argument(
            "Unknown membership category."
        );
    }

    return {
        customer.membership == "Basic" ? 1.0 : 0.0,
        customer.membership == "Premium" ? 1.0 : 0.0
    };
}


// -----------------------------------------------------------------------------
// 14. TRAIN/TEST SPLIT
// -----------------------------------------------------------------------------

struct DatasetSplit {
    std::vector<std::size_t> train;
    std::vector<std::size_t> test;
};


DatasetSplit randomSplit(
    std::size_t rowCount,
    double trainFraction,
    unsigned seed
) {
    if (rowCount < 2) {
        throw std::invalid_argument(
            "At least two observations are required."
        );
    }

    if (trainFraction <= 0.0 ||
        trainFraction >= 1.0) {
        throw std::invalid_argument(
            "Train fraction must be between 0 and 1."
        );
    }

    std::vector<std::size_t> indices(rowCount);

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    std::mt19937 generator(seed);

    std::shuffle(
        indices.begin(),
        indices.end(),
        generator
    );

    const auto trainSize =
        static_cast<std::size_t>(
            std::floor(
                rowCount * trainFraction
            )
        );

    return {
        std::vector<std::size_t>(
            indices.begin(),
            indices.begin() + trainSize
        ),
        std::vector<std::size_t>(
            indices.begin() + trainSize,
            indices.end()
        )
    };
}


// -----------------------------------------------------------------------------
// 15. STRATIFIED SPLIT
// -----------------------------------------------------------------------------

DatasetSplit stratifiedSplit(
    const std::vector<int>& labels,
    double testFraction,
    unsigned seed
) {
    if (labels.empty()) {
        throw std::invalid_argument(
            "Labels cannot be empty."
        );
    }

    if (testFraction <= 0.0 ||
        testFraction >= 1.0) {
        throw std::invalid_argument(
            "Test fraction must be between 0 and 1."
        );
    }

    std::map<int, std::vector<std::size_t>> groups;

    for (std::size_t index = 0;
         index < labels.size();
         ++index) {
        groups[labels[index]].push_back(index);
    }

    std::mt19937 generator(seed);

    DatasetSplit result;

    for (auto& [label, indices] : groups) {
        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
        );

        const auto testSize =
            std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::round(
                        indices.size() *
                        testFraction
                    )
                )
            );

        result.test.insert(
            result.test.end(),
            indices.begin(),
            indices.begin() + testSize
        );

        result.train.insert(
            result.train.end(),
            indices.begin() + testSize,
            indices.end()
        );
    }

    std::shuffle(
        result.train.begin(),
        result.train.end(),
        generator
    );

    std::shuffle(
        result.test.begin(),
        result.test.end(),
        generator
    );

    return result;
}


// -----------------------------------------------------------------------------
// 16. SIMPLE SCORING SYSTEM
// -----------------------------------------------------------------------------

double purchaseScore(const Customer& customer) {
    /*
     * This is an educational deterministic scoring rule rather than a trained
     * machine-learning model.
     *
     * Importantly, the function does not use customer.purchased.
     *
     * That reflects the conceptual direction:
     *
     *     features -> prediction
     *
     * while the observed target is used separately for evaluation.
     */

    double score = 0.0;

    if (customer.age >= 30) {
        score += 1.0;
    }

    if (customer.annualIncome >= 70000.0) {
        score += 1.0;
    }

    if (customer.membership == "Premium") {
        score += 1.5;
    }

    if (customer.visitsPerMonth >= 6) {
        score += 1.5;
    }

    return score;
}


// -----------------------------------------------------------------------------
// 17. CONFUSION MATRIX
// -----------------------------------------------------------------------------

struct ConfusionMatrix {
    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;
};


ConfusionMatrix evaluatePredictions(
    const CustomerDataset& dataset
) {
    ConfusionMatrix matrix;

    for (const auto& customer : dataset.observations()) {
        const int prediction =
            purchaseScore(customer) >= 2.5
                ? 1
                : 0;

        if (prediction == 1 &&
            customer.purchased == 1) {
            ++matrix.truePositive;
        } else if (prediction == 0 &&
                   customer.purchased == 0) {
            ++matrix.trueNegative;
        } else if (prediction == 1 &&
                   customer.purchased == 0) {
            ++matrix.falsePositive;
        } else if (prediction == 0 &&
                   customer.purchased == 1) {
            ++matrix.falseNegative;
        }
    }

    return matrix;
}


double accuracy(
    const ConfusionMatrix& matrix
) {
    const int total =
        matrix.truePositive +
        matrix.trueNegative +
        matrix.falsePositive +
        matrix.falseNegative;

    if (total == 0) {
        return 0.0;
    }

    return static_cast<double>(
        matrix.truePositive +
        matrix.trueNegative
    ) / static_cast<double>(total);
}


double precision(
    const ConfusionMatrix& matrix
) {
    const int denominator =
        matrix.truePositive +
        matrix.falsePositive;

    if (denominator == 0) {
        return 0.0;
    }

    return static_cast<double>(
        matrix.truePositive
    ) / static_cast<double>(denominator);
}


double recall(
    const ConfusionMatrix& matrix
) {
    const int denominator =
        matrix.truePositive +
        matrix.falseNegative;

    if (denominator == 0) {
        return 0.0;
    }

    return static_cast<double>(
        matrix.truePositive
    ) / static_cast<double>(denominator);
}


// -----------------------------------------------------------------------------
// 18. LEAKAGE DETECTION
// -----------------------------------------------------------------------------

struct FeatureAvailability {
    std::string featureName;
    bool availableAtPrediction;
    std::string reason;
};


std::vector<std::string> findPotentialLeakage(
    const std::vector<FeatureAvailability>& features
) {
    std::vector<std::string> leakage;

    for (const auto& feature : features) {
        if (!feature.availableAtPrediction) {
            leakage.push_back(
                feature.featureName +
                ": " +
                feature.reason
            );
        }
    }

    return leakage;
}


// -----------------------------------------------------------------------------
// 19. TEMPORAL DATA MODEL
// -----------------------------------------------------------------------------

struct SalesObservation {
    std::string timestamp;
    double sales;
};


bool isChronologicallyOrdered(
    const std::vector<SalesObservation>& observations
) {
    for (std::size_t i = 1;
         i < observations.size();
         ++i) {
        if (observations[i - 1].timestamp >
            observations[i].timestamp) {
            return false;
        }
    }

    return true;
}


// -----------------------------------------------------------------------------
// 20. COMPLETE CASE STUDY
// -----------------------------------------------------------------------------

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout
        << "DATA UNDERSTANDING CASE STUDY\n";
    std::cout << std::string(80, '=') << "\n";

    /*
     * Business problem:
     *
     * The organization has customer records and wants to understand which
     * variables can be used to predict whether a customer purchases a
     * product.
     *
     * Unit of analysis:
     *   one customer
     *
     * Target:
     *   purchased
     *
     * Features:
     *   age
     *   annualIncome
     *   city
     *   membership
     *   visitsPerMonth
     *
     * Identifier:
     *   customerId
     */

    CustomerDataset dataset({
        {
            "C001",
            28,
            52000.0,
            "Lucknow",
            "Basic",
            3,
            0
        },
        {
            "C002",
            35,
            76000.0,
            "Delhi",
            "Premium",
            8,
            1
        },
        {
            "C003",
            42,
            91000.0,
            "Mumbai",
            "Premium",
            10,
            1
        },
        {
            "C004",
            23,
            41000.0,
            "Lucknow",
            "Basic",
            2,
            0
        },
        {
            "C005",
            51,
            125000.0,
            "Delhi",
            "Premium",
            12,
            1
        },
        {
            "C006",
            31,
            68000.0,
            "Mumbai",
            "Basic",
            5,
            0
        }
    });

    std::cout
        << "\nObservations: "
        << dataset.size()
        << "\n";


    // -------------------------------------------------------------------------
    // 21. PRINT OBSERVATIONS
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "OBSERVATIONS\n"
        << std::string(80, '-')
        << "\n";

    for (const auto& customer : dataset.observations()) {
        std::cout
            << customer.customerId
            << " | age=" << customer.age
            << " | income=" << customer.annualIncome
            << " | city=" << customer.city
            << " | membership=" << customer.membership
            << " | visits=" << customer.visitsPerMonth
            << " | purchased=" << customer.purchased
            << "\n";
    }


    // -------------------------------------------------------------------------
    // 22. DATA DICTIONARY
    // -------------------------------------------------------------------------

    std::vector<VariableDefinition> dictionary{
        {
            "customerId",
            VariableRole::Identifier,
            VariableType::Text,
            false,
            true,
            "Unique customer identifier"
        },
        {
            "age",
            VariableRole::Feature,
            VariableType::Numerical,
            false,
            true,
            "Customer age in years"
        },
        {
            "annualIncome",
            VariableRole::Feature,
            VariableType::Numerical,
            false,
            true,
            "Annual income in currency units"
        },
        {
            "city",
            VariableRole::Feature,
            VariableType::NominalCategorical,
            false,
            true,
            "Customer city"
        },
        {
            "membership",
            VariableRole::Feature,
            VariableType::NominalCategorical,
            false,
            true,
            "Membership tier"
        },
        {
            "visitsPerMonth",
            VariableRole::Feature,
            VariableType::Numerical,
            false,
            true,
            "Average monthly visits"
        },
        {
            "purchased",
            VariableRole::Target,
            VariableType::Numerical,
            false,
            false,
            "Observed purchase outcome"
        }
    };

    printDataDictionary(dictionary);


    // -------------------------------------------------------------------------
    // 23. VALIDATE ALL OBSERVATIONS
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "VALIDATION\n"
        << std::string(80, '-')
        << "\n";

    for (const auto& customer : dataset.observations()) {
        std::cout
            << customer.customerId
            << ": ";

        printValidationErrors(
            validateCustomer(customer)
        );
    }


    // -------------------------------------------------------------------------
    // 24. IDENTIFIER ANALYSIS
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "IDENTIFIER ANALYSIS\n"
        << std::string(80, '-')
        << "\n";

    const auto duplicateIds =
        findDuplicateIds(dataset);

    if (duplicateIds.empty()) {
        std::cout
            << "All customer identifiers are unique.\n";
    } else {
        std::cout
            << "Duplicate identifier observations:\n";

        for (std::size_t index : duplicateIds) {
            std::cout
                << "  row "
                << index
                << "\n";
        }
    }


    // -------------------------------------------------------------------------
    // 25. CATEGORY ANALYSIS
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "CITY DISTRIBUTION\n"
        << std::string(80, '-')
        << "\n";

    printCategoryCounts(
        countCategories(
            dataset,
            [](const Customer& customer) {
                return customer.city;
            }
        )
    );


    std::cout
        << "\n"
        << "MEMBERSHIP DISTRIBUTION\n";

    printCategoryCounts(
        countCategories(
            dataset,
            [](const Customer& customer) {
                return customer.membership;
            }
        )
    );


    // -------------------------------------------------------------------------
    // 26. NUMERICAL PROFILING
    // -------------------------------------------------------------------------

    std::vector<double> incomes;

    for (const auto& customer : dataset.observations()) {
        incomes.push_back(customer.annualIncome);
    }

    const auto incomeSummary =
        summarize(incomes);

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "INCOME SUMMARY\n"
        << std::string(80, '-')
        << "\n"
        << "Minimum: "
        << incomeSummary.minimum
        << "\n"
        << "Maximum: "
        << incomeSummary.maximum
        << "\n"
        << "Mean: "
        << incomeSummary.mean
        << "\n";


    // -------------------------------------------------------------------------
    // 27. FEATURE MATRIX AND TARGET VECTOR
    // -------------------------------------------------------------------------

    const FeatureMatrix featureMatrix =
        createNumericFeatureMatrix(dataset);

    const auto targetVector =
        createTargetVector(dataset);

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "FEATURE MATRIX X\n"
        << std::string(80, '-')
        << "\n";

    printFeatureMatrix(featureMatrix);

    std::cout
        << "\nTarget vector y: ";

    for (int target : targetVector) {
        std::cout
            << target
            << " ";
    }

    std::cout << "\n";


    // -------------------------------------------------------------------------
    // 28. CLASS DISTRIBUTION
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << "TARGET DISTRIBUTION\n";

    for (const auto& [label, proportion] :
         classDistribution(targetVector)) {
        std::cout
            << "  class "
            << label
            << ": "
            << std::fixed
            << std::setprecision(2)
            << proportion * 100.0
            << "%\n";
    }


    // -------------------------------------------------------------------------
    // 29. FEATURE ENGINEERING
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "FEATURE ENGINEERING\n"
        << std::string(80, '-')
        << "\n";

    for (const auto& customer : dataset.observations()) {
        try {
            const auto enriched =
                engineerFeatures(customer);

            std::cout
                << enriched.customer.customerId
                << " | incomePerVisit="
                << enriched.incomePerVisit
                << " | highIncome="
                << std::boolalpha
                << enriched.highIncome
                << " | frequentVisitor="
                << enriched.frequentVisitor
                << "\n";
        } catch (const std::exception& error) {
            std::cout
                << "Feature engineering failed for "
                << customer.customerId
                << ": "
                << error.what()
                << "\n";
        }
    }


    // -------------------------------------------------------------------------
    // 30. CATEGORICAL ENCODING
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "ONE-HOT MEMBERSHIP ENCODING\n"
        << std::string(80, '-')
        << "\n";

    for (const auto& customer : dataset.observations()) {
        const auto encoded =
            encodeMembership(customer);

        std::cout
            << customer.customerId
            << " | Basic="
            << encoded.basic
            << " | Premium="
            << encoded.premium
            << "\n";
    }


    // -------------------------------------------------------------------------
    // 31. TRAIN/TEST SPLIT
    // -------------------------------------------------------------------------

    const auto randomSplitResult =
        randomSplit(
            dataset.size(),
            0.67,
            42
        );

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "RANDOM TRAIN/TEST SPLIT\n"
        << std::string(80, '-')
        << "\n";

    std::cout << "Train indices: ";

    for (auto index : randomSplitResult.train) {
        std::cout << index << " ";
    }

    std::cout << "\nTest indices: ";

    for (auto index : randomSplitResult.test) {
        std::cout << index << " ";
    }

    std::cout << "\n";


    // -------------------------------------------------------------------------
    // 32. STRATIFIED SPLIT
    // -------------------------------------------------------------------------

    const auto stratified =
        stratifiedSplit(
            targetVector,
            0.33,
            42
        );

    std::cout
        << "\n"
        << "STRATIFIED SPLIT\n";

    std::cout << "Train indices: ";

    for (auto index : stratified.train) {
        std::cout << index << " ";
    }

    std::cout << "\nTest indices: ";

    for (auto index : stratified.test) {
        std::cout << index << " ";
    }

    std::cout << "\n";


    // -------------------------------------------------------------------------
    // 33. PREDICTIONS
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "PREDICTIVE SCORING\n"
        << std::string(80, '-')
        << "\n";

    for (const auto& customer : dataset.observations()) {
        const double score =
            purchaseScore(customer);

        const int prediction =
            score >= 2.5 ? 1 : 0;

        std::cout
            << customer.customerId
            << " | score="
            << score
            << " | prediction="
            << prediction
            << " | actual="
            << customer.purchased
            << "\n";
    }


    // -------------------------------------------------------------------------
    // 34. EVALUATION
    // -------------------------------------------------------------------------

    const auto confusion =
        evaluatePredictions(dataset);

    std::cout
        << "\n"
        << "CONFUSION MATRIX\n"
        << "True positives: "
        << confusion.truePositive
        << "\n"
        << "True negatives: "
        << confusion.trueNegative
        << "\n"
        << "False positives: "
        << confusion.falsePositive
        << "\n"
        << "False negatives: "
        << confusion.falseNegative
        << "\n"
        << "Accuracy: "
        << accuracy(confusion)
        << "\n"
        << "Precision: "
        << precision(confusion)
        << "\n"
        << "Recall: "
        << recall(confusion)
        << "\n";


    // -------------------------------------------------------------------------
    // 35. LEAKAGE DETECTION
    // -------------------------------------------------------------------------

    std::vector<FeatureAvailability> featureAvailability{
        {
            "income",
            true,
            "available during application"
        },
        {
            "creditScore",
            true,
            "available during application"
        },
        {
            "collectionActionTaken",
            false,
            "created after a default may occur"
        }
    };

    const auto leakage =
        findPotentialLeakage(
            featureAvailability
        );

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "LEAKAGE REVIEW\n"
        << std::string(80, '-')
        << "\n";

    if (leakage.empty()) {
        std::cout
            << "No unavailable prediction-time features found.\n";
    } else {
        for (const auto& issue : leakage) {
            std::cout
                << "Potential leakage: "
                << issue
                << "\n";
        }
    }


    // -------------------------------------------------------------------------
    // 36. TEMPORAL CASE
    // -------------------------------------------------------------------------

    std::vector<SalesObservation> salesHistory{
        {"2026-01-01T09:00:00", 10.0},
        {"2026-01-02T09:00:00", 13.0},
        {"2026-01-03T09:00:00", 15.0}
    };

    std::cout
        << "\n"
        << "Temporal observations are ordered: "
        << std::boolalpha
        << isChronologicallyOrdered(salesHistory)
        << "\n";


    // -------------------------------------------------------------------------
    // 37. EDGE CASE: INVALID CUSTOMER
    // -------------------------------------------------------------------------

    Customer invalidCustomer{
        "C999",
        -10,
        -5000.0,
        "Lucknow",
        "Unknown",
        -3,
        4
    };

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "INVALID INPUT CASE\n"
        << std::string(80, '-')
        << "\n";

    printValidationErrors(
        validateCustomer(invalidCustomer)
    );


    // -------------------------------------------------------------------------
    // 38. PERFORMANCE CONSIDERATIONS
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "PERFORMANCE CONSIDERATIONS\n"
        << std::string(80, '-')
        << "\n";

    std::cout
        << "Validation is O(n) for n observations.\n"
        << "Identifier duplicate detection is approximately O(n log n) "
        << "using std::set.\n"
        << "Category counting is O(n log k) for k ordered categories.\n"
        << "Feature-matrix construction is O(n * p) for n observations "
        << "and p selected features.\n"
        << "Random shuffling is O(n).\n";


    /*
     * Architectural trade-offs:
     *
     * std::set:
     *   predictable ordered behavior and O(log n) lookup/insertion.
     *
     * std::unordered_set:
     *   expected O(1) lookup/insertion but no ordering guarantee.
     *
     * vector:
     *   excellent cache locality and compact storage for tabular records.
     *
     * optional:
     *   useful for nullable values because absence is represented explicitly
     *   rather than through arbitrary sentinel values.
     */


    // -------------------------------------------------------------------------
    // 39. DESIGN RULES
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '-')
        << "\n"
        << "DESIGN RULES\n"
        << std::string(80, '-')
        << "\n";

    const std::vector<std::string> rules{
        "Define the unit of analysis before interpreting rows.",
        "Document every variable.",
        "Separate identifiers from predictive features.",
        "Identify the target explicitly.",
        "Validate ranges and categories.",
        "Investigate missingness rather than treating it automatically.",
        "Check duplicates.",
        "Respect prediction-time information availability.",
        "Fit preprocessing using training data only.",
        "Use time-aware splitting when temporal order matters.",
        "Use class-aware evaluation for imbalanced targets."
    };

    for (const auto& rule : rules) {
        std::cout
            << "- "
            << rule
            << "\n";
    }


    // -------------------------------------------------------------------------
    // 40. FINAL CASE STUDY REPORT
    // -------------------------------------------------------------------------

    std::cout
        << "\n"
        << std::string(80, '=')
        << "\n"
        << "CASE STUDY REPORT\n"
        << std::string(80, '=')
        << "\n";

    std::cout
        << "Unit of analysis: one customer\n"
        << "Feature variables: age, annualIncome, city, membership, "
           "visitsPerMonth\n"
        << "Identifier: customerId\n"
        << "Target: purchased\n"
        << "Task type: binary classification\n"
        << "Primary risks: leakage, invalid values, missing data, "
           "class imbalance, incorrect feature semantics\n"
        << "Feature engineering demonstrated: income per visit, "
           "high-income flag, frequent-visitor flag\n"
        << "Categorical encoding demonstrated: one-hot membership encoding\n"
        << "Evaluation demonstrated: confusion matrix, accuracy, "
           "precision, recall\n";

    /*
     * The most important engineering principle demonstrated by this case
     * study is that data understanding happens before model selection.
     *
     * A model cannot repair a fundamentally incorrect definition of:
     *
     *   - what one row represents
     *   - what each variable means
     *   - what information is available at prediction time
     *   - which variable is the target
     *
     * Correct semantic interpretation is therefore part of the technical
     * implementation, not merely documentation.
     */

    std::cout
        << "\n"
        << std::string(80, '=')
        << "\n"
        << "END OF CASE STUDY\n"
        << std::string(80, '=')
        << "\n";

    return 0;
}
