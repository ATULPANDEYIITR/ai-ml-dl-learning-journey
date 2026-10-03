#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

/*
 * Repository-scale style case study:
 *
 * A customer-risk feature service receives categorical customer attributes
 * and converts them into numerical representations before a downstream
 * scoring engine consumes them.
 *
 * The system deliberately separates:
 *   - nominal categorical features -> one-hot encoding
 *   - ordered categorical features -> ordinal encoding
 *   - target-derived high-cardinality features -> smoothed target encoding
 *   - external identifiers -> label encoding when an identifier is required
 *
 * The case study also evaluates a validation dataset without using its
 * outcomes to fit target encoding, preventing target leakage.
 *
 * Compile:
 *   g++ -std=c++17 -O2 categorical_encoding.cpp -o categorical_encoding
 */

struct Customer {
    string region;
    string plan;
    string acquisitionChannel;
    string segment;
    int churn;
};

struct EncodedCustomer {
    vector<double> values;
    int target;
};

static void printHeader(const string& title) {
    cout << "\n" << string(78, '=') << "\n";
    cout << title << "\n";
    cout << string(78, '=') << "\n";
}

static double mean(const vector<double>& values) {
    if (values.empty()) {
        throw invalid_argument("Mean requires at least one observation.");
    }

    return accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

static void validateBinaryTarget(int value) {
    if (value != 0 && value != 1) {
        throw invalid_argument("Churn target must be binary: 0 or 1.");
    }
}

class LabelEncoder {
private:
    map<string, int> mapping;
    bool useUnknown;

public:
    explicit LabelEncoder(bool useUnknownCategory = true)
        : useUnknown(useUnknownCategory) {}

    void fit(const vector<string>& values) {
        set<string> categories(values.begin(), values.end());

        mapping.clear();

        int nextIndex = 0;
        for (const auto& category : categories) {
            mapping[category] = nextIndex++;
        }

        if (useUnknown) {
            mapping["__UNKNOWN__"] = -1;
        }
    }

    int transform(const string& value) const {
        auto it = mapping.find(value);

        if (it != mapping.end()) {
            return it->second;
        }

        if (useUnknown) {
            return -1;
        }

        throw invalid_argument("Unknown label category: " + value);
    }

    const map<string, int>& getMapping() const {
        return mapping;
    }
};


class OneHotEncoder {
private:
    vector<string> categories;
    bool ignoreUnknown;
    bool dropFirst;

public:
    OneHotEncoder(
        bool ignoreUnknownCategory = true,
        bool removeFirstCategory = false
    )
        : ignoreUnknown(ignoreUnknownCategory),
          dropFirst(removeFirstCategory) {}

    void fit(const vector<string>& values) {
        set<string> unique(values.begin(), values.end());
        categories.assign(unique.begin(), unique.end());

        if (categories.empty()) {
            throw invalid_argument("One-hot encoder needs categories.");
        }
    }

    vector<int> transform(const string& value) const {
        if (categories.empty()) {
            throw logic_error("OneHotEncoder must be fitted first.");
        }

        size_t start = dropFirst ? 1 : 0;
        vector<int> result(categories.size() - start, 0);

        auto it = find(categories.begin(), categories.end(), value);

        if (it == categories.end()) {
            if (ignoreUnknown) {
                return result;
            }

            throw invalid_argument("Unknown one-hot category: " + value);
        }

        size_t index = static_cast<size_t>(
            distance(categories.begin(), it)
        );

        if (index >= start) {
            result[index - start] = 1;
        }

        return result;
    }

    vector<string> featureNames(const string& prefix) const {
        size_t start = dropFirst ? 1 : 0;
        vector<string> names;

        for (size_t index = start; index < categories.size(); ++index) {
            names.push_back(prefix + "__" + categories[index]);
        }

        return names;
    }
};


class OrdinalEncoder {
private:
    unordered_map<string, int> mapping;
    bool useUnknown;

public:
    OrdinalEncoder(
        const vector<string>& orderedCategories,
        bool useUnknownCategory = true
    )
        : useUnknown(useUnknownCategory) {

        if (orderedCategories.empty()) {
            throw invalid_argument("Ordinal order cannot be empty.");
        }

        set<string> uniquenessCheck;

        for (size_t index = 0; index < orderedCategories.size(); ++index) {
            if (!uniquenessCheck.insert(orderedCategories[index]).second) {
                throw invalid_argument(
                    "Ordinal categories must be unique."
                );
            }

            mapping[orderedCategories[index]] =
                static_cast<int>(index);
        }
    }

    int transform(const string& value) const {
        auto it = mapping.find(value);

        if (it != mapping.end()) {
            return it->second;
        }

        if (useUnknown) {
            return -1;
        }

        throw invalid_argument(
            "Unknown ordinal category: " + value
        );
    }
};


class TargetEncoder {
private:
    struct Statistics {
        int count;
        double rawMean;
    };

    unordered_map<string, Statistics> statistics;
    double globalMean = 0.0;
    double smoothing;

public:
    explicit TargetEncoder(double smoothingStrength = 10.0)
        : smoothing(smoothingStrength) {

        if (!isfinite(smoothing) || smoothing < 0.0) {
            throw invalid_argument(
                "Target-encoding smoothing must be non-negative."
            );
        }
    }

    void fit(
        const vector<string>& categories,
        const vector<int>& target
    ) {
        if (categories.size() != target.size()) {
            throw invalid_argument(
                "Target encoding feature and target lengths differ."
            );
        }

        if (categories.empty()) {
            throw invalid_argument(
                "Target encoding needs training observations."
            );
        }

        vector<double> numericTarget;

        for (int value : target) {
            validateBinaryTarget(value);
            numericTarget.push_back(static_cast<double>(value));
        }

        globalMean = mean(numericTarget);
        statistics.clear();

        unordered_map<string, vector<double>> grouped;

        for (size_t index = 0; index < categories.size(); ++index) {
            grouped[categories[index]].push_back(numericTarget[index]);
        }

        for (const auto& [category, values] : grouped) {
            statistics[category] = {
                static_cast<int>(values.size()),
                mean(values)
            };
        }
    }

    double transform(const string& category) const {
        auto it = statistics.find(category);

        if (it == statistics.end()) {
            return globalMean;
        }

        const Statistics& stats = it->second;

        /*
         * Bayesian-style smoothing:
         *
         * The category estimate receives weight proportional to its sample
         * count, while the global mean acts as a prior for sparse categories.
         */
        return (
            stats.count * stats.rawMean +
            smoothing * globalMean
        ) / (stats.count + smoothing);
    }

    double getGlobalMean() const {
        return globalMean;
    }

    void printStatistics() const {
        vector<string> categories;

        for (const auto& [category, _] : statistics) {
            categories.push_back(category);
        }

        sort(categories.begin(), categories.end());

        cout << left
             << setw(18) << "Segment"
             << setw(10) << "Count"
             << setw(14) << "Raw mean"
             << setw(14) << "Smoothed"
             << "\n";

        cout << string(56, '-') << "\n";

        for (const auto& category : categories) {
            const auto& stats = statistics.at(category);

            cout << left
                 << setw(18) << category
                 << setw(10) << stats.count
                 << setw(14) << fixed << setprecision(4)
                 << stats.rawMean
                 << setw(14) << transform(category)
                 << "\n";
        }
    }
};


class GovernanceFeatureEngine {
private:
    OneHotEncoder regionEncoder;
    OrdinalEncoder planEncoder;
    OneHotEncoder channelEncoder;
    TargetEncoder segmentEncoder;
    LabelEncoder identifierEncoder;

public:
    GovernanceFeatureEngine()
        : regionEncoder(true, false),
          planEncoder(
              {"Basic", "Standard", "Premium"},
              true
          ),
          channelEncoder(true, false),
          segmentEncoder(5.0),
          identifierEncoder(true) {}

    void fit(
        const vector<Customer>& trainingCustomers
    ) {
        if (trainingCustomers.empty()) {
            throw invalid_argument(
                "Feature engine cannot be fitted without customers."
            );
        }

        vector<string> regions;
        vector<string> plans;
        vector<string> channels;
        vector<string> segments;
        vector<string> identifiers;
        vector<int> targets;

        for (size_t index = 0; index < trainingCustomers.size(); ++index) {
            const auto& customer = trainingCustomers[index];

            validateBinaryTarget(customer.churn);

            regions.push_back(customer.region);
            plans.push_back(customer.plan);
            channels.push_back(customer.acquisitionChannel);
            segments.push_back(customer.segment);
            identifiers.push_back(
                "customer_" + to_string(index + 1)
            );
            targets.push_back(customer.churn);
        }

        regionEncoder.fit(regions);
        channelEncoder.fit(channels);
        identifierEncoder.fit(identifiers);

        /*
         * Plan encoding is not fitted from observed frequency. Its semantic
         * order was explicitly defined in the encoder constructor.
         */
        segmentEncoder.fit(segments, targets);
    }

    EncodedCustomer transform(
        const Customer& customer,
        int target
    ) const {
        validateBinaryTarget(target);

        EncodedCustomer result;
        result.target = target;

        for (int value : regionEncoder.transform(customer.region)) {
            result.values.push_back(static_cast<double>(value));
        }

        /*
         * Ordinal encoding contributes one numerical feature because the
         * subscription tiers have meaningful ordered semantics.
         */
        result.values.push_back(
            static_cast<double>(planEncoder.transform(customer.plan))
        );

        for (int value : channelEncoder.transform(
                 customer.acquisitionChannel)) {
            result.values.push_back(static_cast<double>(value));
        }

        /*
         * Segment is encoded as a supervised statistic. It is crucial that
         * segmentEncoder was fitted only on training outcomes.
         */
        result.values.push_back(
            segmentEncoder.transform(customer.segment)
        );

        return result;
    }

    void printTargetEncodingStatistics() const {
        cout << "Global training churn rate: "
             << fixed << setprecision(4)
             << segmentEncoder.getGlobalMean()
             << "\n\n";

        segmentEncoder.printStatistics();
    }

    vector<string> featureNames() const {
        vector<string> names = regionEncoder.featureNames("region");

        names.push_back("plan__ordinal");

        vector<string> channels =
            channelEncoder.featureNames("channel");

        names.insert(
            names.end(),
            channels.begin(),
            channels.end()
        );

        names.push_back("segment__target_mean");

        return names;
    }
};


static vector<Customer> trainingCustomers() {
    return {
        {"North", "Basic", "Organic", "Student", 1},
        {"North", "Standard", "Referral", "Professional", 0},
        {"South", "Premium", "Partner", "Enterprise", 0},
        {"West", "Basic", "Paid Search", "Student", 1},
        {"East", "Standard", "Organic", "Professional", 0},
        {"South", "Basic", "Paid Search", "Student", 1},
        {"West", "Premium", "Referral", "Enterprise", 0},
        {"North", "Basic", "Partner", "Student", 1},
        {"East", "Premium", "Organic", "Enterprise", 0},
        {"South", "Standard", "Referral", "Professional", 0}
    };
}


static vector<Customer> validationCustomers() {
    return {
        {"West", "Standard", "Organic", "Student", 1},
        {"Central", "Premium", "Partner", "Enterprise", 0},
        {"East", "Basic", "Social", "New Segment", 1}
    };
}


static void printEncodedCustomer(
    const vector<string>& names,
    const EncodedCustomer& customer
) {
    cout << left
         << setw(28) << "Feature"
         << "Value\n";

    cout << string(42, '-') << "\n";

    for (size_t index = 0; index < names.size(); ++index) {
        cout << left
             << setw(28) << names[index]
             << fixed << setprecision(4)
             << customer.values[index]
             << "\n";
    }

    cout << "Target: " << customer.target << "\n";
}


static void demonstrateLabelEncoding() {
    printHeader("Identifier Encoding with LabelEncoder");

    LabelEncoder encoder(true);

    vector<string> customerIds = {
        "customer_001",
        "customer_002",
        "customer_003"
    };

    encoder.fit(customerIds);

    for (const auto& id : customerIds) {
        cout << id << " -> "
             << encoder.transform(id)
             << "\n";
    }

    cout << "customer_999 -> "
         << encoder.transform("customer_999")
         << " (unknown identifier)\n";

    cout << "\nThe identifier is encoded as a compact integer for internal "
            "categorical representation. It must not be interpreted as "
            "a meaningful numerical ranking of customers.\n";
}


static void demonstrateEncodingSemantics() {
    printHeader("Encoding Semantics");

    cout << "Region: one-hot because North/South/East/West are nominal.\n";
    cout << "Plan: ordinal because Basic < Standard < Premium is meaningful.\n";
    cout << "Segment: target encoding because segment-specific churn statistics "
            "can represent supervised signal without creating one column per "
            "segment.\n";
    cout << "Customer ID: label encoding only when the identifier is treated "
            "as a categorical key rather than a numeric quantity.\n";
}


static void demonstrateTargetLeakage() {
    printHeader("Target Leakage Boundary");

    vector<Customer> train = trainingCustomers();
    vector<Customer> validation = validationCustomers();

    vector<string> trainingSegments;
    vector<int> trainingTargets;

    for (const auto& customer : train) {
        trainingSegments.push_back(customer.segment);
        trainingTargets.push_back(customer.churn);
    }

    TargetEncoder safeEncoder(5.0);
    safeEncoder.fit(trainingSegments, trainingTargets);

    cout << left
         << setw(18) << "Validation segment"
         << setw(18) << "Safe encoding"
         << "\n";

    cout << string(36, '-') << "\n";

    for (const auto& customer : validation) {
        cout << left
             << setw(18) << customer.segment
             << setw(18) << fixed << setprecision(4)
             << safeEncoder.transform(customer.segment)
             << "\n";
    }

    /*
     * The validation targets are intentionally never passed to fit().
     * Including them would allow validation outcomes to influence feature
     * construction and would make the measured validation performance
     * optimistic.
     */
}


static void demonstrateEndToEndEngine() {
    printHeader("Customer Churn Feature Engine");

    const vector<Customer> train = trainingCustomers();
    const vector<Customer> validation = validationCustomers();

    GovernanceFeatureEngine engine;
    engine.fit(train);

    cout << "Feature schema:\n";
    for (const auto& name : engine.featureNames()) {
        cout << "  " << name << "\n";
    }

    cout << "\nTraining target statistics:\n";
    engine.printTargetEncodingStatistics();

    cout << "\nValidation feature vector:\n";

    for (const auto& customer : validation) {
        cout << "\nRegion=" << customer.region
             << ", Plan=" << customer.plan
             << ", Channel=" << customer.acquisitionChannel
             << ", Segment=" << customer.segment
             << "\n";

        EncodedCustomer encoded =
            engine.transform(customer, customer.churn);

        printEncodedCustomer(
            engine.featureNames(),
            encoded
        );
    }
}


static void demonstrateConstraints() {
    printHeader("Encoding Constraints and Trade-offs");

    cout << "One-hot encoding keeps nominal categories semantically separate, "
            "but feature width grows with cardinality.\n";

    cout << "Ordinal encoding is compact, but it should only be used when "
            "the numerical order reflects domain meaning.\n";

    cout << "Label encoding is compact and useful for category identifiers, "
            "but raw integer values can create false ordering when consumed "
            "by algorithms that treat them as continuous numbers.\n";

    cout << "Target encoding compresses category-specific target information "
            "into numerical statistics, but it is supervised and therefore "
            "must be controlled carefully to avoid target leakage.\n";

    cout << "Smoothing reduces variance for rare categories by moving their "
            "estimates toward the global target mean.\n";
}


int main() {
    try {
        demonstrateEncodingSemantics();
        demonstrateLabelEncoding();
        demonstrateEndToEndEngine();
        demonstrateTargetLeakage();
        demonstrateConstraints();

        printHeader("Case Study Complete");

        cout << "The feature engine separates category semantics from "
                "representation choice and keeps training-only target "
                "statistics isolated from validation data.\n";

        return 0;
    }
    catch (const exception& error) {
        cerr << "Fatal error: " << error.what() << "\n";
        return 1;
    }
}
