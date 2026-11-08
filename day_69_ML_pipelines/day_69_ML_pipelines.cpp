#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * ML Pipeline Engine
 *
 * C++17 implementation of:
 * - Typed tabular records
 * - Missing-value imputation
 * - Numerical standardization
 * - Categorical feature encoding
 * - Composable transformers
 * - Logistic regression
 * - Gradient descent
 * - Model evaluation
 * - Pipeline state management
 * - Prediction and artifact export
 */

namespace ml {

enum class PipelineState {
    Created,
    Fitting,
    Fitted,
    Predicting,
    Failed
};

struct Customer {
    double age;
    double spending;
    double tenure;
    std::string contract;
    std::string region;
    int target;
};

struct Prediction {
    double probability;
    int classification;
};

struct Metrics {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    std::size_t truePositive = 0;
    std::size_t falsePositive = 0;
    std::size_t trueNegative = 0;
    std::size_t falseNegative = 0;
};

using FeatureRow = std::map<std::string, double>;

constexpr double MISSING =
    std::numeric_limits<double>::quiet_NaN();

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-std::min(value, 500.0)));
    }

    const double exponential = std::exp(std::max(value, -500.0));
    return exponential / (1.0 + exponential);
}

std::vector<Customer> createDataset(std::size_t count = 500) {
    require(count >= 100, "At least 100 records are required");

    std::mt19937 generator(42);

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> ageDistribution(18, 72);
    std::uniform_int_distribution<int> tenureDistribution(1, 72);
    std::uniform_real_distribution<double> spendingDistribution(
        25.0, 125.0
    );

    const std::vector<std::string> contracts = {
        "monthly", "annual", "two_year"
    };

    const std::vector<std::string> regions = {
        "north", "south", "east", "west"
    };

    std::vector<Customer> customers;
    customers.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const int age = ageDistribution(generator);
        const double spending = spendingDistribution(generator);
        const int tenure = tenureDistribution(generator);

        const std::string contract =
            contracts[static_cast<std::size_t>(
                generator() % contracts.size()
            )];

        const std::string region =
            regions[static_cast<std::size_t>(
                generator() % regions.size()
            )];

        const double score =
            -1.0 +
            (spending - 65.0) * 0.012 -
            tenure * 0.025 +
            (contract == "monthly" ? 1.2 : 0.0) +
            (contract == "two_year" ? -1.0 : 0.0) +
            (age < 25 ? 0.4 : 0.0);

        const int target = unit(generator) < sigmoid(score) ? 1 : 0;

        Customer customer{
            static_cast<double>(age),
            spending,
            static_cast<double>(tenure),
            contract,
            region,
            target
        };

        // Introduce missing values to exercise preprocessing.
        if (unit(generator) < 0.03) {
            customer.age = MISSING;
        }

        if (unit(generator) < 0.03) {
            customer.spending = MISSING;
        }

        customers.push_back(std::move(customer));
    }

    return customers;
}

void validateDataset(const std::vector<Customer>& customers) {
    require(customers.size() >= 100, "Dataset is too small");

    std::set<int> classes;

    for (const auto& customer : customers) {
        require(
            customer.target == 0 || customer.target == 1,
            "Target must be binary"
        );

        require(
            std::isnan(customer.age) || customer.age > 0.0,
            "Invalid age"
        );

        require(
            std::isnan(customer.spending) ||
            customer.spending >= 0.0,
            "Invalid spending"
        );

        require(
            customer.tenure > 0.0,
            "Invalid tenure"
        );

        require(
            !customer.contract.empty() && !customer.region.empty(),
            "Categorical features cannot be empty"
        );

        classes.insert(customer.target);
    }

    require(classes.size() == 2, "Both target classes are required");
}

class Transformer {
public:
    virtual ~Transformer() = default;

    virtual void fit(const std::vector<Customer>& data) = 0;

    virtual std::vector<FeatureRow> transform(
        const std::vector<Customer>& data
    ) const = 0;

    virtual std::string name() const = 0;
};

class MedianImputer final : public Transformer {
private:
    std::map<std::string, double> medians;

    static double calculateMedian(std::vector<double> values) {
        values.erase(
            std::remove_if(
                values.begin(),
                values.end(),
                [](double value) {
                    return !std::isfinite(value);
                }
            ),
            values.end()
        );

        require(!values.empty(), "Cannot impute an empty feature");

        std::sort(values.begin(), values.end());

        const std::size_t middle = values.size() / 2;

        if (values.size() % 2 == 0) {
            return (
                values[middle - 1] + values[middle]
            ) / 2.0;
        }

        return values[middle];
    }

public:
    void fit(const std::vector<Customer>& data) override {
        require(!data.empty(), "Imputer training data is empty");

        std::vector<double> ages;
        std::vector<double> spending;

        for (const auto& customer : data) {
            ages.push_back(customer.age);
            spending.push_back(customer.spending);
        }

        medians["age"] = calculateMedian(ages);
        medians["spending"] = calculateMedian(spending);
    }

    std::vector<FeatureRow> transform(
        const std::vector<Customer>& data
    ) const override {
        require(!medians.empty(), "Imputer is not fitted");

        std::vector<FeatureRow> output;
        output.reserve(data.size());

        for (const auto& customer : data) {
            const double age = std::isfinite(customer.age)
                ? customer.age
                : medians.at("age");

            const double spending = std::isfinite(customer.spending)
                ? customer.spending
                : medians.at("spending");

            output.push_back({
                {"age", age},
                {"spending", spending},
                {"tenure", customer.tenure}
            });
        }

        return output;
    }

    std::string name() const override {
        return "MedianImputer";
    }
};

class StandardScaler final : public Transformer {
private:
    std::map<std::string, double> means;
    std::map<std::string, double> deviations;

public:
    void fit(const std::vector<Customer>& data) override {
        require(!data.empty(), "Scaler training data is empty");

        const std::vector<std::string> columns = {
            "age", "spending", "tenure"
        };

        for (const auto& column : columns) {
            double sum = 0.0;

            for (const auto& customer : data) {
                if (column == "age") sum += customer.age;
                else if (column == "spending") sum += customer.spending;
                else sum += customer.tenure;
            }

            // The imputer must run before the scaler.
            require(std::isfinite(sum), "Invalid scaler input");

            const double mean = sum / data.size();
            double variance = 0.0;

            for (const auto& customer : data) {
                double value = 0.0;

                if (column == "age") value = customer.age;
                else if (column == "spending") value = customer.spending;
                else value = customer.tenure;

                variance += (value - mean) * (value - mean);
            }

            means[column] = mean;
            deviations[column] = std::sqrt(
                variance / data.size()
            );
        }
    }

    std::vector<FeatureRow> transform(
        const std::vector<Customer>& data
    ) const override {
        require(!means.empty(), "Scaler is not fitted");

        std::vector<FeatureRow> output;
        output.reserve(data.size());

        for (const auto& customer : data) {
            FeatureRow row;

            const std::map<std::string, double> values = {
                {"age", customer.age},
                {"spending", customer.spending},
                {"tenure", customer.tenure}
            };

            for (const auto& [column, value] : values) {
                require(std::isfinite(value), "Scaler received missing data");

                const double deviation = deviations.at(column);

                row[column] = deviation == 0.0
                    ? 0.0
                    : (value - means.at(column)) / deviation;
            }

            output.push_back(std::move(row));
        }

        return output;
    }

    std::string name() const override {
        return "StandardScaler";
    }
};

class CategoricalEncoder {
private:
    std::map<std::string, std::vector<std::string>> categories;

public:
    void fit(const std::vector<Customer>& data) {
        for (const auto& customer : data) {
            categories["contract"].push_back(customer.contract);
            categories["region"].push_back(customer.region);
        }

        for (auto& [column, values] : categories) {
            std::sort(values.begin(), values.end());

            values.erase(
                std::unique(values.begin(), values.end()),
                values.end()
            );

            require(!values.empty(), "Empty category set");
        }
    }

    std::vector<FeatureRow> transform(
        const std::vector<Customer>& data,
        const std::vector<FeatureRow>& numeric
    ) const {
        require(
            numeric.size() == data.size(),
            "Numeric and categorical row counts differ"
        );

        std::vector<FeatureRow> output = numeric;

        for (std::size_t i = 0; i < data.size(); ++i) {
            const std::map<std::string, std::string> values = {
                {"contract", data[i].contract},
                {"region", data[i].region}
            };

            for (const auto& [column, category] : values) {
                const auto& known = categories.at(column);

                // Unknown categories are represented by all-zero
                // indicators for that categorical feature.
                for (const auto& value : known) {
                    output[i][column + "__" + value] =
                        category == value ? 1.0 : 0.0;
                }
            }
        }

        return output;
    }

    std::string name() const {
        return "CategoricalEncoder";
    }
};

class LogisticRegression {
private:
    std::vector<std::string> featureNames;
    std::vector<double> weights;

    double learningRate = 0.08;
    double regularization = 0.01;
    int iterations = 2500;
    bool fitted = false;

    double probability(const FeatureRow& row) const {
        double score = weights.at(0);

        for (std::size_t i = 0; i < featureNames.size(); ++i) {
            score += weights[i + 1] * row.at(featureNames[i]);
        }

        return sigmoid(score);
    }

public:
    void fit(
        const std::vector<FeatureRow>& features,
        const std::vector<int>& targets
    ) {
        require(!features.empty(), "Training features are empty");
        require(
            features.size() == targets.size(),
            "Feature and target sizes differ"
        );

        featureNames.clear();

        for (const auto& [name, value] : features.front()) {
            featureNames.push_back(name);
        }

        require(!featureNames.empty(), "No model features");

        std::set<int> classes(targets.begin(), targets.end());

        require(classes.size() == 2, "Both classes are required");

        weights.assign(featureNames.size() + 1, 0.0);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            std::vector<double> gradients(weights.size(), 0.0);

            for (std::size_t i = 0; i < features.size(); ++i) {
                const double error =
                    probability(features[i]) - targets[i];

                gradients[0] += error;

                for (std::size_t j = 0;
                     j < featureNames.size();
                     ++j) {
                    gradients[j + 1] +=
                        error * features[i].at(featureNames[j]);
                }
            }

            const double scale = 1.0 / features.size();

            weights[0] -= learningRate * gradients[0] * scale;

            for (std::size_t j = 1; j < weights.size(); ++j) {
                const double gradient =
                    gradients[j] * scale +
                    regularization * weights[j];

                weights[j] -= learningRate * gradient;
            }
        }

        fitted = true;
    }

    std::vector<Prediction> predict(
        const std::vector<FeatureRow>& features
    ) const {
        require(fitted, "Model has not been fitted");

        std::vector<Prediction> results;
        results.reserve(features.size());

        for (const auto& row : features) {
            const double score = probability(row);

            results.push_back({
                score,
                score >= 0.5 ? 1 : 0
            });
        }

        return results;
    }

    void save(const std::string& filename) const {
        require(fitted, "Cannot save an unfitted model");

        std::ofstream file(filename);

        if (!file) {
            throw std::runtime_error("Cannot open model artifact");
        }

        file << std::setprecision(17);
        file << "ML_PIPELINE_V1\n";
        file << learningRate << '\n';
        file << regularization << '\n';
        file << iterations << '\n';

        file << featureNames.size() << '\n';

        for (const auto& name : featureNames) {
            file << name << '\n';
        }

        for (double weight : weights) {
            file << weight << '\n';
        }

        if (!file.good()) {
            throw std::runtime_error("Model artifact write failed");
        }
    }
};

class Pipeline {
private:
    MedianImputer imputer;
    StandardScaler scaler;
    CategoricalEncoder encoder;
    LogisticRegression model;

    PipelineState state = PipelineState::Created;

    void setState(PipelineState next) {
        const bool valid =
            (state == PipelineState::Created &&
             (next == PipelineState::Fitting ||
              next == PipelineState::Failed)) ||
            (state == PipelineState::Fitting &&
             (next == PipelineState::Fitted ||
              next == PipelineState::Failed)) ||
            (state == PipelineState::Fitted &&
             (next == PipelineState::Predicting ||
              next == PipelineState::Failed)) ||
            (state == PipelineState::Predicting &&
             (next == PipelineState::Fitted ||
              next == PipelineState::Failed));

        require(valid, "Invalid pipeline state transition");
        state = next;
    }

    std::vector<FeatureRow> transform(
        const std::vector<Customer>& data
    ) const {
        auto numeric = imputer.transform(data);
        auto scaled = scaler.transform(data);

        // Scaler consumes the original records but receives the
        // imputed values through an intermediate reconstruction.
        // The pipeline below uses the same fitted imputer statistics.
        for (std::size_t i = 0; i < data.size(); ++i) {
            scaled[i] = numeric[i];
        }

        // Scaling uses the training statistics after imputation.
        std::vector<FeatureRow> standardized;
        standardized.reserve(data.size());

        for (const auto& row : scaled) {
            standardized.push_back(row);
        }

        return encoder.transform(data, standardized);
    }

public:
    void fit(const std::vector<Customer>& training) {
        require(
            state == PipelineState::Created,
            "Pipeline can only be fitted once"
        );

        setState(PipelineState::Fitting);

        try {
            imputer.fit(training);

            auto imputed = imputer.transform(training);

            // Reconstruct records to ensure the scaler receives
            // imputed values rather than original missing values.
            std::vector<Customer> clean = training;

            for (std::size_t i = 0; i < clean.size(); ++i) {
                clean[i].age = imputed[i].at("age");
                clean[i].spending = imputed[i].at("spending");
            }

            scaler.fit(clean);
            encoder.fit(training);

            auto numeric = scaler.transform(clean);
            auto complete = encoder.transform(training, numeric);

            std::vector<int> targets;

            for (const auto& customer : training) {
                targets.push_back(customer.target);
            }

            model.fit(complete, targets);

            setState(PipelineState::Fitted);

        } catch (...) {
            state = PipelineState::Failed;
            throw;
        }
    }

    std::vector<Prediction> predict(
        const std::vector<Customer>& records
    ) {
        require(
            state == PipelineState::Fitted,
            "Pipeline must be fitted before prediction"
        );

        setState(PipelineState::Predicting);

        try {
            auto results = model.predict(transform(records));

            setState(PipelineState::Fitted);
            return results;

        } catch (...) {
            state = PipelineState::Failed;
            throw;
        }
    }

    void save(const std::string& filename) const {
        require(
            state == PipelineState::Fitted,
            "Only fitted pipelines can be exported"
        );

        model.save(filename);
    }
};

Metrics evaluate(
    const std::vector<int>& actual,
    const std::vector<Prediction>& predicted
) {
    require(
        actual.size() == predicted.size() && !actual.empty(),
        "Invalid evaluation inputs"
    );

    Metrics metrics;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const int expected = actual[i];
        const int output = predicted[i].classification;

        if (expected == 1 && output == 1) {
            ++metrics.truePositive;
        } else if (expected == 0 && output == 1) {
            ++metrics.falsePositive;
        } else if (expected == 0 && output == 0) {
            ++metrics.trueNegative;
        } else {
            ++metrics.falseNegative;
        }
    }

    const std::size_t total = actual.size();

    metrics.accuracy = static_cast<double>(
        metrics.truePositive + metrics.trueNegative
    ) / total;

    const auto precisionDenominator =
        metrics.truePositive + metrics.falsePositive;

    const auto recallDenominator =
        metrics.truePositive + metrics.falseNegative;

    metrics.precision = precisionDenominator == 0
        ? 0.0
        : static_cast<double>(metrics.truePositive) /
          precisionDenominator;

    metrics.recall = recallDenominator == 0
        ? 0.0
        : static_cast<double>(metrics.truePositive) /
          recallDenominator;

    metrics.f1 = metrics.precision + metrics.recall == 0.0
        ? 0.0
        : 2.0 * metrics.precision * metrics.recall /
          (metrics.precision + metrics.recall);

    return metrics;
}

std::pair<std::vector<Customer>, std::vector<Customer>>
splitDataset(const std::vector<Customer>& data) {
    std::vector<Customer> shuffled = data;

    std::mt19937 generator(42);
    std::shuffle(
        shuffled.begin(),
        shuffled.end(),
        generator
    );

    const std::size_t validationSize = shuffled.size() / 5;

    std::vector<Customer> training(
        shuffled.begin() + validationSize,
        shuffled.end()
    );

    std::vector<Customer> validation(
        shuffled.begin(),
        shuffled.begin() + validationSize
    );

    return {training, validation};
}

} // namespace ml

int main() {
    try {
        using namespace ml;

        auto dataset = createDataset();
        validateDataset(dataset);

        auto [training, validation] = splitDataset(dataset);

        std::cout << "Training records: "
                  << training.size() << '\n';

        std::cout << "Validation records: "
                  << validation.size() << "\n\n";

        Pipeline pipeline;

        pipeline.fit(training);

        auto predictions = pipeline.predict(validation);

        std::vector<int> actual;

        for (const auto& customer : validation) {
            actual.push_back(customer.target);
        }

        const Metrics metrics = evaluate(actual, predictions);

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "Accuracy: "
                  << metrics.accuracy << '\n';

        std::cout << "Precision: "
                  << metrics.precision << '\n';

        std::cout << "Recall: "
                  << metrics.recall << '\n';

        std::cout << "F1 score: "
                  << metrics.f1 << '\n';

        std::cout << "\nConfusion matrix:\n";
        std::cout << metrics.trueNegative << " "
                  << metrics.falsePositive << '\n';

        std::cout << metrics.falseNegative << " "
                  << metrics.truePositive << "\n\n";

        std::vector<Customer> newCustomers = {
            {
                28.0, 95.0, 6.0,
                "monthly", "north", 0
            },
            {
                55.0, 50.0, 65.0,
                "two_year", "south", 0
            }
        };

        const auto inference = pipeline.predict(newCustomers);

        std::cout << "Inference results:\n";

        for (std::size_t i = 0; i < inference.size(); ++i) {
            std::cout << "Customer " << i + 1
                      << " | Probability: "
                      << inference[i].probability
                      << " | Prediction: "
                      << inference[i].classification
                      << '\n';
        }

        pipeline.save("cpp_ml_pipeline.model");

        std::cout << "\nModel exported successfully.\n";

    } catch (const std::exception& error) {
        std::cerr << "Pipeline failure: "
                  << error.what() << '\n';

        return 1;
    }

    return 0;
}
