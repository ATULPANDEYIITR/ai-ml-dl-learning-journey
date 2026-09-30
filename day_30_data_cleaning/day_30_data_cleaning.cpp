#include <algorithm>
#include <cmath>
#include <fstream>
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
#include <unordered_set>
#include <vector>

/*
 * Repository-scale style data-quality case study:
 *
 * A transaction ingestion service receives records from multiple business
 * systems. Before records enter an analytical store, a governance engine
 * evaluates four distinct data-quality conditions:
 *
 *   missing values     -> imputation or explicit retention
 *   duplicates         -> business-identity deduplication
 *   outliers           -> statistical flagging
 *   inconsistent data  -> canonicalization
 *
 * The program is intentionally designed as a policy engine rather than a
 * collection of isolated syntax examples.
 */

struct Transaction {
    std::string id;
    std::string customer;
    std::string email;
    std::string country;
    std::string state;
    std::string category;
    std::optional<double> amount;
    std::optional<std::string> date;
};

struct AuditEntry {
    std::string id;
    std::string field;
    std::string issue;
    std::string action;
    std::string reason;
};

class DataQualityEngine {
private:
    std::vector<Transaction> records;
    std::vector<AuditEntry> audit;

    static std::string trim(const std::string& input) {
        const auto first = input.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }

        const auto last = input.find_last_not_of(" \t\r\n");
        return input.substr(first, last - first + 1);
    }

    static std::string lower(std::string value) {
        std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            }
        );

        return value;
    }

    static std::string normalizeText(const std::string& value) {
        return trim(value);
    }

    static std::string normalizeEmail(const std::string& value) {
        return lower(trim(value));
    }

    static std::string canonicalCountry(const std::string& value) {
        const std::string key = lower(trim(value));

        static const std::map<std::string, std::string> mapping{
            {"india", "India"},
            {"ind", "India"},
            {"in", "India"},
            {"usa", "United States"},
            {"us", "United States"},
            {"united states", "United States"}
        };

        const auto iterator = mapping.find(key);

        return iterator == mapping.end()
            ? trim(value)
            : iterator->second;
    }

    static std::string canonicalState(const std::string& value) {
        const std::string key = lower(trim(value));

        static const std::map<std::string, std::string> mapping{
            {"up", "Uttar Pradesh"},
            {"u.p.", "Uttar Pradesh"},
            {"uttar pradesh", "Uttar Pradesh"},
            {"delhi", "Delhi"},
            {"ca", "California"},
            {"california", "California"},
            {"gujarat", "Gujarat"},
            {"mh", "Maharashtra"},
            {"maharashtra", "Maharashtra"}
        };

        const auto iterator = mapping.find(key);

        return iterator == mapping.end()
            ? trim(value)
            : iterator->second;
    }

    static std::string canonicalCategory(const std::string& value) {
        const std::string key = lower(trim(value));

        static const std::map<std::string, std::string> mapping{
            {"electronics", "Electronics"},
            {"home and kitchen", "Home & Kitchen"},
            {"home & kitchen", "Home & Kitchen"},
            {"grocery", "Grocery"},
            {"groceries", "Grocery"}
        };

        const auto iterator = mapping.find(key);

        return iterator == mapping.end()
            ? trim(value)
            : iterator->second;
    }

    static std::string normalizeDate(const std::string& value) {
        const std::string text = trim(value);

        if (text.empty()) {
            return "";
        }

        // ISO dates already satisfy the storage contract.
        if (text.size() == 10 &&
            text[4] == '-' &&
            text[7] == '-') {
            return text;
        }

        // Convert DD/MM/YYYY and DD-MM-YYYY to YYYY-MM-DD.
        if (text.size() == 10 &&
            (text[2] == '/' || text[2] == '-') &&
            text[5] == text[2]) {
            const std::string day = text.substr(0, 2);
            const std::string month = text.substr(3, 2);
            const std::string year = text.substr(6, 4);

            return year + "-" + month + "-" + day;
        }

        throw std::runtime_error(
            "Unsupported date format: " + value
        );
    }

    static std::string businessKey(const Transaction& transaction) {
        std::ostringstream key;

        key << lower(trim(transaction.id))
            << '|'
            << normalizeEmail(transaction.email)
            << '|';

        if (transaction.date.has_value()) {
            key << transaction.date.value();
        }

        return key.str();
    }

    void log(
        const std::string& id,
        const std::string& field,
        const std::string& issue,
        const std::string& action,
        const std::string& reason
    ) {
        audit.push_back({
            id,
            field,
            issue,
            action,
            reason
        });
    }

    static double median(std::vector<double> values) {
        if (values.empty()) {
            throw std::runtime_error(
                "Median cannot be calculated from an empty dataset."
            );
        }

        std::sort(values.begin(), values.end());

        const std::size_t middle = values.size() / 2;

        if (values.size() % 2 == 0) {
            return (
                values[middle - 1] +
                values[middle]
            ) / 2.0;
        }

        return values[middle];
    }

    static std::pair<double, double> quartiles(
        const std::vector<double>& values
    ) {
        if (values.size() < 4) {
            throw std::runtime_error(
                "At least four observations are required for IQR analysis."
            );
        }

        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());

        const std::size_t middle = sorted.size() / 2;

        std::vector<double> lower(
            sorted.begin(),
            sorted.begin() + middle
        );

        std::vector<double> upper(
            sorted.begin() +
                (sorted.size() % 2 == 0 ? middle : middle + 1),
            sorted.end()
        );

        return {
            median(lower),
            median(upper)
        };
    }

    void normalizeInconsistentValues() {
        for (auto& record : records) {
            const std::string originalCustomer = record.customer;
            record.customer = normalizeText(record.customer);

            if (record.customer != originalCustomer) {
                log(
                    record.id,
                    "customer",
                    "inconsistent_value",
                    "canonicalize",
                    "Removed presentation-level whitespace."
                );
            }

            const std::string originalEmail = record.email;
            record.email = normalizeEmail(record.email);

            if (record.email != originalEmail) {
                log(
                    record.id,
                    "email",
                    "inconsistent_value",
                    "canonicalize",
                    "Normalized email casing and surrounding whitespace."
                );
            }

            const std::string originalCountry = record.country;
            record.country = canonicalCountry(record.country);

            if (record.country != originalCountry) {
                log(
                    record.id,
                    "country",
                    "inconsistent_value",
                    "canonicalize",
                    "Mapped an accepted country alias to a canonical value."
                );
            }

            const std::string originalState = record.state;
            record.state = canonicalState(record.state);

            if (record.state != originalState) {
                log(
                    record.id,
                    "state",
                    "inconsistent_value",
                    "canonicalize",
                    "Mapped an accepted state abbreviation or alias."
                );
            }

            const std::string originalCategory = record.category;
            record.category = canonicalCategory(record.category);

            if (record.category != originalCategory) {
                log(
                    record.id,
                    "category",
                    "inconsistent_value",
                    "canonicalize",
                    "Mapped category spelling and case variants."
                );
            }

            if (record.date.has_value()) {
                const std::string originalDate = record.date.value();

                try {
                    record.date = normalizeDate(originalDate);

                    if (record.date.value() != originalDate) {
                        log(
                            record.id,
                            "date",
                            "inconsistent_value",
                            "canonicalize",
                            "Converted supported date formats to ISO 8601."
                        );
                    }
                } catch (const std::exception& error) {
                    record.date.reset();

                    log(
                        record.id,
                        "date",
                        "inconsistent_value",
                        "invalidate",
                        error.what()
                    );
                }
            }
        }
    }

    void removeDuplicates() {
        std::unordered_set<std::string> seen;
        std::vector<Transaction> retained;

        for (const auto& record : records) {
            const std::string key = businessKey(record);

            if (seen.find(key) != seen.end()) {
                log(
                    record.id,
                    "__record__",
                    "duplicate",
                    "remove_duplicate",
                    "Business identity matches a previously retained record."
                );

                continue;
            }

            seen.insert(key);
            retained.push_back(record);
        }

        records = std::move(retained);
    }

    void handleMissingValues() {
        std::vector<double> amounts;

        for (const auto& record : records) {
            if (record.amount.has_value() &&
                std::isfinite(record.amount.value())) {
                amounts.push_back(record.amount.value());
            }
        }

        if (amounts.empty()) {
            throw std::runtime_error(
                "No valid amount exists for numeric imputation."
            );
        }

        const double replacement = median(amounts);

        for (auto& record : records) {
            if (!record.amount.has_value()) {
                record.amount = replacement;

                log(
                    record.id,
                    "amount",
                    "missing_value",
                    "median_imputation",
                    "Missing amount replaced with the dataset median."
                );
            }

            // A missing transaction date remains missing. Guessing a date
            // would create fabricated temporal information.
            if (!record.date.has_value()) {
                log(
                    record.id,
                    "date",
                    "missing_value",
                    "retain_missing",
                    "Date cannot be reliably reconstructed."
                );
            }
        }
    }

    void detectOutliers() {
        std::vector<double> amounts;

        for (const auto& record : records) {
            if (record.amount.has_value()) {
                amounts.push_back(record.amount.value());
            }
        }

        if (amounts.size() < 4) {
            return;
        }

        const auto [q1, q3] = quartiles(amounts);
        const double iqr = q3 - q1;

        if (iqr == 0.0) {
            return;
        }

        const double lowerBound = q1 - 1.5 * iqr;
        const double upperBound = q3 + 1.5 * iqr;

        for (const auto& record : records) {
            const double amount = record.amount.value();

            if (amount < lowerBound || amount > upperBound) {
                std::ostringstream reason;

                reason << std::fixed << std::setprecision(2)
                       << "IQR bounds: ["
                       << lowerBound
                       << ", "
                       << upperBound
                       << "].";

                log(
                    record.id,
                    "amount",
                    "outlier",
                    "flag_only",
                    reason.str()
                );
            }
        }
    }

    std::vector<std::string> validate() const {
        std::vector<std::string> errors;
        std::set<std::string> ids;

        for (const auto& record : records) {
            if (record.id.empty()) {
                errors.push_back("A record has no transaction ID.");
            }

            if (!ids.insert(record.id).second) {
                errors.push_back(
                    "Duplicate transaction ID remains: " + record.id
                );
            }

            if (!record.amount.has_value() ||
                !std::isfinite(record.amount.value()) ||
                record.amount.value() < 0.0) {
                errors.push_back(
                    record.id + ": amount is invalid."
                );
            }

            if (
                record.country != "India" &&
                record.country != "United States"
            ) {
                errors.push_back(
                    record.id +
                    ": country is not canonical."
                );
            }

            if (
                record.category != "Electronics" &&
                record.category != "Home & Kitchen" &&
                record.category != "Grocery"
            ) {
                errors.push_back(
                    record.id +
                    ": category is not canonical."
                );
            }
        }

        return errors;
    }

public:
    explicit DataQualityEngine(std::vector<Transaction> input)
        : records(std::move(input)) {}

    void run() {
        normalizeInconsistentValues();
        removeDuplicates();
        handleMissingValues();
        detectOutliers();

        const auto errors = validate();

        if (!errors.empty()) {
            std::ostringstream message;

            message << "Post-cleaning validation failed:";

            for (const auto& error : errors) {
                message << '\n' << error;
            }

            throw std::runtime_error(message.str());
        }
    }

    const std::vector<Transaction>& getRecords() const {
        return records;
    }

    const std::vector<AuditEntry>& getAudit() const {
        return audit;
    }
};

std::vector<Transaction> createDataset() {
    return {
        {
            "TX3001",
            "  Priya Sharma ",
            "PRIYA.SHARMA@EXAMPLE.COM",
            "IND",
            "UP",
            "electronics",
            12500.0,
            "2026-09-01"
        },
        {
            "TX3002",
            "Rahul Verma",
            "rahul.verma@example.com",
            "India",
            "uttar pradesh",
            "Electronics",
            13250.0,
            "01/09/2026"
        },
        {
            "TX3003",
            "Neha Singh",
            "neha.singh@example.com",
            "IN",
            "U.P.",
            "Home and Kitchen",
            std::nullopt,
            "2026-09-03"
        },
        {
            "TX3004",
            "Amit Kumar",
            "amit.kumar@example.com",
            "India",
            "Delhi",
            "home & kitchen",
            8900.0,
            std::nullopt
        },
        {
            "TX3005",
            "Sara Khan",
            "sara.khan@example.com",
            "India",
            "delhi",
            "HOME & KITCHEN",
            9100.0,
            "2026-09-05"
        },
        {
            "TX3006",
            "Vikram Rao",
            "vikram.rao@example.com",
            "United States",
            "California",
            "Electronics",
            1500000.0,
            "2026-09-06"
        },
        {
            "TX3007",
            "Meera Joshi",
            "MEERA.JOSHI@example.com",
            "USA",
            "CA",
            "electronics",
            11750.0,
            "2026-09-07"
        },
        {
            "TX3008",
            "Arjun Patel",
            "arjun.patel@example.com",
            "INDIA",
            "Gujarat",
            "Electronics",
            12100.0,
            "07-09-2026"
        },
        {
            "TX3009",
            "Kabir Mehta",
            "kabir.mehta@example.com",
            "IN",
            "MH",
            "Grocery",
            2150.0,
            "2026-09-08"
        },
        {
            "TX3009",
            "Kabir Mehta",
            "kabir.mehta@example.com",
            "IN",
            "MH",
            "Grocery",
            2150.0,
            "2026-09-08"
        }
    };
}

void printRecords(
    const std::vector<Transaction>& records
) {
    std::cout << "\nCLEANED TRANSACTION DATA\n";
    std::cout << std::string(80, '=') << '\n';

    std::cout
        << std::left
        << std::setw(10) << "ID"
        << std::setw(20) << "Customer"
        << std::setw(18) << "Country"
        << std::setw(18) << "Category"
        << std::setw(14) << "Amount"
        << "Date\n";

    for (const auto& record : records) {
        std::cout
            << std::left
            << std::setw(10) << record.id
            << std::setw(20) << record.customer
            << std::setw(18) << record.country
            << std::setw(18) << record.category
            << std::setw(14)
            << std::fixed
            << std::setprecision(2)
            << record.amount.value()
            << (record.date.has_value()
                ? record.date.value()
                : "MISSING")
            << '\n';
    }
}

void printAudit(
    const std::vector<AuditEntry>& audit
) {
    std::cout << "\nDATA-QUALITY AUDIT\n";
    std::cout << std::string(80, '=') << '\n';

    std::map<std::string, int> counts;

    for (const auto& entry : audit) {
        counts[entry.issue]++;
    }

    for (const auto& [issue, count] : counts) {
        std::cout
            << std::left
            << std::setw(22)
            << issue
            << count
            << '\n';
    }

    std::cout << "\nAudit events:\n";

    for (const auto& entry : audit) {
        std::cout
            << entry.id
            << '.'
            << entry.field
            << " | "
            << entry.issue
            << " | "
            << entry.action
            << " | "
            << entry.reason
            << '\n';
    }
}

void writeAuditCsv(
    const std::vector<AuditEntry>& audit,
    const std::string& filename
) {
    std::ofstream output(filename);

    if (!output) {
        throw std::runtime_error(
            "Unable to create audit file: " + filename
        );
    }

    output
        << "id,field,issue,action,reason\n";

    for (const auto& entry : audit) {
        output
            << '"'
            << entry.id
            << "\",\""
            << entry.field
            << "\",\""
            << entry.issue
            << "\",\""
            << entry.action
            << "\",\""
            << entry.reason
            << "\"\n";
    }
}

int main() {
    try {
        std::cout
            << "DATA QUALITY GOVERNANCE ENGINE\n"
            << std::string(80, '=') << '\n';

        auto dataset = createDataset();

        std::cout
            << "Input records: "
            << dataset.size()
            << '\n';

        DataQualityEngine engine(std::move(dataset));

        /*
         * The order matters:
         *
         * Canonicalization establishes comparable representations.
         * Deduplication can then compare normalized business identities.
         * Missing-value policies operate on normalized numeric data.
         * Outlier detection is performed after numeric normalization.
         * Final validation verifies that the output meets the data contract.
         */
        engine.run();

        printRecords(engine.getRecords());
        printAudit(engine.getAudit());

        writeAuditCsv(
            engine.getAudit(),
            "data_quality_audit.csv"
        );

        std::cout
            << "\nAudit file written to data_quality_audit.csv\n";

        std::cout
            << "\nCASE-STUDY DESIGN NOTES\n"
            << std::string(80, '=') << '\n'
            << "Missing amounts use median imputation because the median is "
               "less sensitive to extreme transaction values.\n"
            << "Missing dates remain explicit because temporal fabrication "
               "could corrupt period-based reporting.\n"
            << "Duplicates are identified through a business identity key, "
               "not arbitrary memory addresses or row positions.\n"
            << "Outliers are flagged instead of deleted because statistical "
               "extremeness does not prove that an observation is incorrect.\n"
            << "Inconsistent categorical representations are resolved through "
               "explicit domain mappings rather than unrestricted fuzzy matching.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "DATA QUALITY PIPELINE ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}
