#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace classical_ml {

constexpr double TEST_FRACTION = 0.20;
constexpr double LEARNING_RATE = 0.05;
constexpr double L2_PENALTY = 0.01;
constexpr int MAX_EPOCHS = 2000;

struct Sample {
    std::vector<double> features;
    int target = 0;
};

struct Dataset {
    std::vector<std::string> feature_names;
    std::vector<Sample> rows;
};

struct Split {
    std::vector<Sample> train;
    std::vector<Sample> test;
};

struct ConfusionMatrix {
    std::size_t true_negative = 0;
    std::size_t false_positive = 0;
    std::size_t false_negative = 0;
    std::size_t true_positive = 0;
};

struct Evaluation {
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double roc_auc = 0.0;
    ConfusionMatrix confusion;
};

double sigmoid(double value) {
    // Clamping prevents overflow in exp() for extreme model scores.
    value = std::clamp(value, -35.0, 35.0);
    return 1.0 / (1.0 + std::exp(-value));
}

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inside_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];

        if (character == '"') {
            if (
                inside_quotes &&
                i + 1 < line.size() &&
                line[i + 1] == '"'
            ) {
                field.push_back('"');
                ++i;
            } else {
                inside_quotes = !inside_quotes;
            }
        } else if (character == ',' && !inside_quotes) {
            fields.push_back(trim(field));
            field.clear();
        } else {
            field.push_back(character);
        }
    }

    if (inside_quotes) {
        throw std::runtime_error("Malformed CSV: unclosed quoted field.");
    }

    fields.push_back(trim(field));
    return fields;
}

double parse_feature(const std::string& raw_value) {
    const std::string value = trim(raw_value);

    if (value.empty() || value == "NA" || value == "NaN") {
        return std::numeric_limits<double>::quiet_NaN();
    }

    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);

    if (
        errno == ERANGE ||
        end == value.c_str() ||
        *end != '\0' ||
        std::isinf(parsed)
    ) {
        throw std::runtime_error(
            "Invalid numeric feature value: " + value
        );
    }

    return parsed;
}

int parse_target(const std::string& raw_value) {
    const double value = parse_feature(raw_value);

    if (
        !std::isfinite(value) ||
        (value != 0.0 && value != 1.0)
    ) {
        throw std::runtime_error("Target must be either zero or one.");
    }

    return static_cast<int>(value);
}

Dataset load_csv(const std::string& file_path) {
    std::ifstream input(file_path);

    if (!input) {
        throw std::runtime_error("Cannot open dataset: " + file_path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("The CSV file is empty.");
    }

    const std::vector<std::string> header = parse_csv_line(line);
    const auto target_iterator = std::find(
        header.begin(),
        header.end(),
        "target"
    );

    if (target_iterator == header.end()) {
        throw std::runtime_error(
            "The CSV must contain a binary 'target' column."
        );
    }

    const std::size_t target_index =
        static_cast<std::size_t>(target_iterator - header.begin());

    Dataset dataset;

    for (std::size_t i = 0; i < header.size(); ++i) {
        if (i != target_index) {
            dataset.feature_names.push_back(header[i]);
        }
    }

    if (dataset.feature_names.empty()) {
        throw std::runtime_error("No feature columns were found.");
    }

    while (std::getline(input, line)) {
        if (trim(line).empty()) {
            continue;
        }

        const auto fields = parse_csv_line(line);

        if (fields.size() != header.size()) {
            throw std::runtime_error(
                "CSV row has a different number of columns than the header."
            );
        }

        Sample sample;
        sample.target = parse_target(fields[target_index]);

        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i != target_index) {
                sample.features.push_back(parse_feature(fields[i]));
            }
        }

        dataset.rows.push_back(std::move(sample));
    }

    if (dataset.rows.size() < 20) {
        throw std::runtime_error(
            "At least twenty valid dataset rows are required."
        );
    }

    return dataset;
}

Dataset make_demo_dataset() {
    // A deterministic synthetic binary dataset keeps the program executable
    // without requiring the Python-generated CSV.
    Dataset dataset;
    dataset.feature_names = {
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_compactness",
        "mean_concavity",
        "worst_radius"
    };

    std::mt19937 generator(42);
    std::bernoulli_distribution malignant_distribution(0.38);
    std::bernoulli_distribution missing_distribution(0.01);

    const std::vector<double> benign_means = {
        12.0, 18.0, 78.0, 450.0, 0.09, 0.04, 13.2
    };
    const std::vector<double> malignant_means = {
        17.5, 22.0, 115.0, 950.0, 0.16, 0.14, 20.0
    };
    const std::vector<double> deviations = {
        2.2, 3.8, 14.0, 230.0, 0.04, 0.06, 2.5
    };

    for (int row = 0; row < 700; ++row) {
        const int target = malignant_distribution(generator) ? 1 : 0;
        const auto& means =
            target == 1 ? malignant_means : benign_means;

        Sample sample;
        sample.target = target;

        for (std::size_t column = 0; column < means.size(); ++column) {
            std::normal_distribution<double> distribution(
                means[column],
                deviations[column]
            );

            double value = distribution(generator);

            if (missing_distribution(generator)) {
                value = std::numeric_limits<double>::quiet_NaN();
            }

            sample.features.push_back(value);
        }

        dataset.rows.push_back(std::move(sample));
    }

    return dataset;
}

void validate_dataset(const Dataset& dataset) {
    if (dataset.rows.empty() || dataset.feature_names.empty()) {
        throw std::runtime_error("Dataset is empty.");
    }

    bool has_benign = false;
    bool has_malignant = false;

    for (const auto& sample : dataset.rows) {
        if (sample.features.size() != dataset.feature_names.size()) {
            throw std::runtime_error("Inconsistent feature dimensions.");
        }

        if (sample.target == 0) {
            has_benign = true;
        } else if (sample.target == 1) {
            has_malignant = true;
        } else {
            throw std::runtime_error("Unexpected target label.");
        }
    }

    if (!has_benign || !has_malignant) {
        throw std::runtime_error(
            "Both target classes are required for binary evaluation."
        );
    }
}

Split stratified_split(
    const Dataset& dataset,
    unsigned int seed
) {
    std::map<int, std::vector<Sample>> by_class;

    for (const auto& sample : dataset.rows) {
        by_class[sample.target].push_back(sample);
    }

    std::mt19937 generator(seed);
    Split split;

    for (auto& entry : by_class) {
        auto& class_rows = entry.second;
        std::shuffle(
            class_rows.begin(),
            class_rows.end(),
            generator
        );

        if (class_rows.size() < 2) {
            throw std::runtime_error(
                "Each class needs at least two observations."
            );
        }

        std::size_t test_count = static_cast<std::size_t>(
            std::round(class_rows.size() * TEST_FRACTION)
        );
        test_count = std::clamp<std::size_t>(
            test_count,
            1,
            class_rows.size() - 1
        );

        split.test.insert(
            split.test.end(),
            class_rows.begin(),
            class_rows.begin() + static_cast<std::ptrdiff_t>(test_count)
        );

        split.train.insert(
            split.train.end(),
            class_rows.begin() + static_cast<std::ptrdiff_t>(test_count),
            class_rows.end()
        );
    }

    std::shuffle(split.train.begin(), split.train.end(), generator);
    std::shuffle(split.test.begin(), split.test.end(), generator);

    return split;
}

class MedianImputer {
public:
    void fit(const std::vector<Sample>& training_rows) {
        if (training_rows.empty()) {
            throw std::runtime_error(
                "Cannot fit an imputer on an empty training set."
            );
        }

        const std::size_t dimensions =
            training_rows.front().features.size();

        medians_.assign(dimensions, 0.0);

        for (std::size_t column = 0; column < dimensions; ++column) {
            std::vector<double> observed;

            for (const auto& row : training_rows) {
                const double value = row.features[column];

                if (std::isfinite(value)) {
                    observed.push_back(value);
                }
            }

            if (observed.empty()) {
                throw std::runtime_error(
                    "A feature is entirely missing in the training set."
                );
            }

            std::sort(observed.begin(), observed.end());
            const std::size_t middle = observed.size() / 2;

            if (observed.size() % 2 == 0) {
                medians_[column] = (
                    observed[middle - 1] + observed[middle]
                ) / 2.0;
            } else {
                medians_[column] = observed[middle];
            }
        }
    }

    std::vector<std::vector<double>> transform(
        const std::vector<Sample>& rows
    ) const {
        std::vector<std::vector<double>> result;

        for (const auto& row : rows) {
            if (row.features.size() != medians_.size()) {
                throw std::runtime_error(
                    "Feature count changed during imputation."
                );
            }

            std::vector<double> transformed = row.features;

            for (std::size_t column = 0; column < transformed.size(); ++column) {
                if (!std::isfinite(transformed[column])) {
                    transformed[column] = medians_[column];
                }
            }

            result.push_back(std::move(transformed));
        }

        return result;
    }

private:
    std::vector<double> medians_;
};

class Standardizer {
public:
    void fit(const std::vector<std::vector<double>>& training_data) {
        if (training_data.empty()) {
            throw std::runtime_error(
                "Cannot fit a standardizer on an empty training set."
            );
        }

        const std::size_t dimensions = training_data.front().size();
        means_.assign(dimensions, 0.0);
        scales_.assign(dimensions, 1.0);

        for (const auto& row : training_data) {
            if (row.size() != dimensions) {
                throw std::runtime_error(
                    "Inconsistent feature dimensions during scaling."
                );
            }

            for (std::size_t j = 0; j < dimensions; ++j) {
                means_[j] += row[j];
            }
        }

        for (double& mean : means_) {
            mean /= static_cast<double>(training_data.size());
        }

        for (const auto& row : training_data) {
            for (std::size_t j = 0; j < dimensions; ++j) {
                const double difference = row[j] - means_[j];
                scales_[j] += difference * difference;
            }
        }

        for (double& scale : scales_) {
            scale = std::sqrt(
                std::max(
                    0.0,
                    (scale - 1.0) /
                    static_cast<double>(training_data.size())
                )
            );

            if (scale < 1e-12) {
                scale = 1.0;
            }
        }
    }

    std::vector<std::vector<double>> transform(
        const std::vector<std::vector<double>>& data
    ) const {
        std::vector<std::vector<double>> result = data;

        for (auto& row : result) {
            if (row.size() != means_.size()) {
                throw std::runtime_error(
                    "Feature count changed during standardization."
                );
            }

            for (std::size_t j = 0; j < row.size(); ++j) {
                row[j] = (row[j] - means_[j]) / scales_[j];
            }
        }

        return result;
    }

private:
    std::vector<double> means_;
    std::vector<double> scales_;
};

class LogisticRegression {
public:
    void fit(
        const std::vector<std::vector<double>>& features,
        const std::vector<int>& targets
    ) {
        if (features.empty() || features.size() != targets.size()) {
            throw std::runtime_error("Invalid training data dimensions.");
        }

        const std::size_t dimensions = features.front().size();
        weights_.assign(dimensions, 0.0);
        bias_ = 0.0;

        double previous_loss = std::numeric_limits<double>::infinity();

        for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
            std::vector<double> gradient(dimensions, 0.0);
            double bias_gradient = 0.0;
            double loss = 0.0;

            for (std::size_t i = 0; i < features.size(); ++i) {
                const double probability = predict_probability(features[i]);
                const double error = probability - targets[i];

                for (std::size_t j = 0; j < dimensions; ++j) {
                    gradient[j] += error * features[i][j];
                }

                bias_gradient += error;

                const double bounded = std::clamp(
                    probability,
                    1e-12,
                    1.0 - 1e-12
                );

                loss -= (
                    targets[i] * std::log(bounded) +
                    (1 - targets[i]) * std::log(1.0 - bounded)
                );
            }

            const double count = static_cast<double>(features.size());
            loss /= count;

            for (std::size_t j = 0; j < dimensions; ++j) {
                gradient[j] = gradient[j] / count +
                    L2_PENALTY * weights_[j];
                loss += (
                    L2_PENALTY * weights_[j] * weights_[j]
                ) / 2.0;
            }

            bias_gradient /= count;

            for (std::size_t j = 0; j < dimensions; ++j) {
                weights_[j] -= LEARNING_RATE * gradient[j];
            }

            bias_ -= LEARNING_RATE * bias_gradient;

            if (
                epoch > 50 &&
                std::abs(previous_loss - loss) < 1e-10
            ) {
                break;
            }

            previous_loss = loss;
        }
    }

    double predict_probability(const std::vector<double>& features) const {
        if (features.size() != weights_.size()) {
            throw std::runtime_error(
                "Prediction feature count does not match the model."
            );
        }

        double score = bias_;

        for (std::size_t i = 0; i < features.size(); ++i) {
            score += weights_[i] * features[i];
        }

        return sigmoid(score);
    }

    const std::vector<double>& weights() const {
        return weights_;
    }

private:
    std::vector<double> weights_;
    double bias_ = 0.0;
};

std::vector<double> extract_probabilities(
    const LogisticRegression& model,
    const std::vector<std::vector<double>>& features
) {
    std::vector<double> probabilities;
    probabilities.reserve(features.size());

    for (const auto& row : features) {
        probabilities.push_back(model.predict_probability(row));
    }

    return probabilities;
}

double calculate_auc(
    const std::vector<int>& actual,
    const std::vector<double>& probabilities
) {
    if (actual.size() != probabilities.size() || actual.empty()) {
        throw std::runtime_error("Invalid ROC AUC input.");
    }

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(actual.size());

    int positives = 0;
    int negatives = 0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        ranked.emplace_back(probabilities[i], actual[i]);
        actual[i] == 1 ? ++positives : ++negatives;
    }

    if (positives == 0 || negatives == 0) {
        throw std::runtime_error(
            "ROC AUC requires both target classes."
        );
    }

    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        }
    );

    // Mann-Whitney rank formulation with average ranks for tied scores.
    double positive_rank_sum = 0.0;
    std::size_t position = 0;

    while (position < ranked.size()) {
        std::size_t end = position + 1;

        while (
            end < ranked.size() &&
            ranked[end].first == ranked[position].first
        ) {
            ++end;
        }

        const double average_rank =
            (static_cast<double>(position + 1) +
             static_cast<double>(end)) / 2.0;

        for (std::size_t i = position; i < end; ++i) {
            if (ranked[i].second == 1) {
                positive_rank_sum += average_rank;
            }
        }

        position = end;
    }

    const double numerator =
        positive_rank_sum -
        (static_cast<double>(positives) * (positives + 1)) / 2.0;

    return numerator /
        (static_cast<double>(positives) * negatives);
}

Evaluation evaluate(
    const std::vector<int>& actual,
    const std::vector<double>& probabilities
) {
    if (actual.size() != probabilities.size() || actual.empty()) {
        throw std::runtime_error("Invalid evaluation dimensions.");
    }

    Evaluation result;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const int predicted = probabilities[i] >= 0.5 ? 1 : 0;

        if (actual[i] == 1 && predicted == 1) {
            ++result.confusion.true_positive;
        } else if (actual[i] == 1 && predicted == 0) {
            ++result.confusion.false_negative;
        } else if (actual[i] == 0 && predicted == 1) {
            ++result.confusion.false_positive;
        } else {
            ++result.confusion.true_negative;
        }
    }

    const auto& matrix = result.confusion;
    const double count = static_cast<double>(actual.size());

    const double tp = static_cast<double>(matrix.true_positive);
    const double tn = static_cast<double>(matrix.true_negative);
    const double fp = static_cast<double>(matrix.false_positive);
    const double fn = static_cast<double>(matrix.false_negative);

    result.accuracy = (tp + tn) / count;
    result.precision = tp + fp == 0.0 ? 0.0 : tp / (tp + fp);
    result.recall = tp + fn == 0.0 ? 0.0 : tp / (tp + fn);

    result.f1 = (
        result.precision + result.recall == 0.0
    ) ? 0.0 : (
        2.0 * result.precision * result.recall /
        (result.precision + result.recall)
    );

    result.roc_auc = calculate_auc(actual, probabilities);

    return result;
}

std::vector<int> extract_targets(const std::vector<Sample>& rows) {
    std::vector<int> targets;
    targets.reserve(rows.size());

    for (const auto& row : rows) {
        targets.push_back(row.target);
    }

    return targets;
}

void print_evaluation(const Evaluation& result) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nHoldout evaluation\n";
    std::cout << "Accuracy:  " << result.accuracy << '\n';
    std::cout << "Precision: " << result.precision << '\n';
    std::cout << "Recall:    " << result.recall << '\n';
    std::cout << "F1 score:  " << result.f1 << '\n';
    std::cout << "ROC AUC:   " << result.roc_auc << '\n';

    std::cout << "\nConfusion matrix\n";
    std::cout << "                 Predicted 0   Predicted 1\n";
    std::cout << "Actual 0         "
              << result.confusion.true_negative << "             "
              << result.confusion.false_positive << '\n';
    std::cout << "Actual 1         "
              << result.confusion.false_negative << "             "
              << result.confusion.true_positive << '\n';
}

}  // namespace classical_ml

int main(int argc, char* argv[]) {
    using namespace classical_ml;

    try {
        if (argc > 2) {
            throw std::runtime_error(
                "Usage: ml_engine [optional_dataset.csv]"
            );
        }

        Dataset dataset;

        if (argc == 2) {
            dataset = load_csv(argv[1]);
            std::cout << "Dataset source: " << argv[1] << '\n';
        } else {
            dataset = make_demo_dataset();
            std::cout << "Dataset source: deterministic synthetic example\n";
        }

        validate_dataset(dataset);

        std::cout << "Rows: " << dataset.rows.size() << '\n';
        std::cout << "Features: " << dataset.feature_names.size() << '\n';
        std::cout << "Positive label: malignant (1)\n";

        const Split split = stratified_split(dataset, 42);

        MedianImputer imputer;
        imputer.fit(split.train);

        const auto imputed_train = imputer.transform(split.train);
        const auto imputed_test = imputer.transform(split.test);

        // Both preprocessing stages are fitted only on training rows.
        Standardizer standardizer;
        standardizer.fit(imputed_train);

        const auto training_features =
            standardizer.transform(imputed_train);
        const auto test_features =
            standardizer.transform(imputed_test);

        const auto training_targets = extract_targets(split.train);
        const auto test_targets = extract_targets(split.test);

        LogisticRegression model;
        model.fit(training_features, training_targets);

        const auto probabilities = extract_probabilities(
            model,
            test_features
        );

        const Evaluation result = evaluate(test_targets, probabilities);

        std::cout << "Training rows: " << split.train.size() << '\n';
        std::cout << "Test rows: " << split.test.size() << '\n';

        print_evaluation(result);

        std::vector<std::pair<double, std::string>> ranked_features;

        for (std::size_t i = 0; i < model.weights().size(); ++i) {
            ranked_features.emplace_back(
                std::abs(model.weights()[i]),
                dataset.feature_names[i]
            );
        }

        std::sort(
            ranked_features.begin(),
            ranked_features.end(),
            std::greater<>()
        );

        std::cout << "\nLargest absolute standardized coefficients\n";
        const std::size_t limit = std::min<std::size_t>(
            5,
            ranked_features.size()
        );

        for (std::size_t i = 0; i < limit; ++i) {
            std::cout << ranked_features[i].second
                      << ": " << ranked_features[i].first << '\n';
        }

        std::cout
            << "\nTraining complexity: O(epochs * rows * features)\n"
            << "AUC ranking complexity: O(test_rows * log(test_rows))\n"
            << "The 0.5 probability threshold is configurable in the source.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ML engine error: " << error.what() << '\n';
        return 1;
    }
}
