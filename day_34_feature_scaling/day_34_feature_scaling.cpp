/*
 * Feature Scaling: Standardization, Normalization, Robust Scaling,
 * and Transformations
 *
 * C++17 technical case study:
 *
 * A risk analytics service receives customer financial measurements from
 * different operational systems. The service must decide whether incoming
 * records are represented on a numerically suitable scale before a
 * distance-based anomaly detector and a simple scoring engine process them.
 *
 * The program models:
 * - Standardization
 * - Min-max normalization
 * - Robust scaling
 * - Log and signed-log transformations
 * - Training-only parameter fitting
 * - Data leakage prevention
 * - Outlier behavior
 * - Constant-feature handling
 * - Distance calculations
 * - Merge-like validation of preprocessing configurations
 * - A production-oriented preprocessing pipeline
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic feature_scaling.cpp -o feature_scaling
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

namespace math_utils {

void validate_matrix(const Matrix& matrix, const std::string& name) {
    if (matrix.empty()) {
        throw std::invalid_argument(name + " cannot be empty.");
    }

    if (matrix.front().empty()) {
        throw std::invalid_argument(name + " must contain features.");
    }

    const std::size_t width = matrix.front().size();

    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (matrix[row].size() != width) {
            throw std::invalid_argument(
                name + " is not rectangular at row " +
                std::to_string(row)
            );
        }

        for (std::size_t column = 0; column < width; ++column) {
            if (!std::isfinite(matrix[row][column])) {
                throw std::invalid_argument(
                    name + " contains a non-finite value."
                );
            }
        }
    }
}

Vector get_column(const Matrix& matrix, std::size_t index) {
    Vector result;
    result.reserve(matrix.size());

    for (const auto& row : matrix) {
        result.push_back(row.at(index));
    }

    return result;
}

double mean(const Vector& values) {
    if (values.empty()) {
        throw std::invalid_argument("Mean requires values.");
    }

    const double total =
        std::accumulate(values.begin(), values.end(), 0.0);

    return total / static_cast<double>(values.size());
}

double population_std(const Vector& values) {
    const double average = mean(values);

    double squared_sum = 0.0;

    for (double value : values) {
        const double difference = value - average;
        squared_sum += difference * difference;
    }

    return std::sqrt(
        squared_sum / static_cast<double>(values.size())
    );
}

double quantile(Vector values, double probability) {
    if (values.empty()) {
        throw std::invalid_argument("Quantile requires values.");
    }

    if (probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument(
            "Quantile probability must be in [0, 1]."
        );
    }

    std::sort(values.begin(), values.end());

    if (values.size() == 1) {
        return values.front();
    }

    const double position =
        (static_cast<double>(values.size()) - 1.0) * probability;

    const auto lower =
        static_cast<std::size_t>(std::floor(position));
    const auto upper =
        static_cast<std::size_t>(std::ceil(position));

    if (lower == upper) {
        return values[lower];
    }

    const double fraction =
        position - static_cast<double>(lower);

    return values[lower] +
           fraction * (values[upper] - values[lower]);
}

double median(Vector values) {
    return quantile(std::move(values), 0.5);
}

double euclidean_distance(
    const Vector& left,
    const Vector& right
) {
    if (left.size() != right.size()) {
        throw std::invalid_argument(
            "Distance vectors must have equal dimensions."
        );
    }

    double squared = 0.0;

    for (std::size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        squared += difference * difference;
    }

    return std::sqrt(squared);
}

void print_matrix(
    const std::string& title,
    const Matrix& matrix,
    int precision = 3
) {
    std::cout << "\n" << title << "\n";

    std::cout << std::fixed
              << std::setprecision(precision);

    for (const auto& row : matrix) {
        std::cout << "  ";

        for (double value : row) {
            std::cout << std::setw(12) << value << " ";
        }

        std::cout << "\n";
    }
}

}  // namespace math_utils

// -----------------------------------------------------------------------------
// Scaler interface
// -----------------------------------------------------------------------------

class FeatureScaler {
public:
    virtual ~FeatureScaler() = default;

    virtual void fit(const Matrix& training_data) = 0;

    virtual Matrix transform(const Matrix& data) const = 0;

    virtual Matrix inverse_transform(const Matrix& data) const = 0;

    virtual std::string name() const = 0;

protected:
    std::size_t feature_count_ = 0;
    bool fitted_ = false;

    void validate_input(const Matrix& data) const {
        math_utils::validate_matrix(data, "input");

        if (data.front().size() != feature_count_) {
            throw std::invalid_argument(
                "Input feature count does not match fitted scaler."
            );
        }
    }

    void require_fitted() const {
        if (!fitted_) {
            throw std::logic_error(
                "Scaler must be fitted before transformation."
            );
        }
    }
};

// -----------------------------------------------------------------------------
// Standardization
// -----------------------------------------------------------------------------

class StandardScaler final : public FeatureScaler {
private:
    Vector means_;
    Vector standard_deviations_;

public:
    void fit(const Matrix& training_data) override {
        math_utils::validate_matrix(
            training_data,
            "training_data"
        );

        feature_count_ = training_data.front().size();
        means_.clear();
        standard_deviations_.clear();

        for (std::size_t feature = 0;
             feature < feature_count_;
             ++feature) {

            const Vector values =
                math_utils::get_column(training_data, feature);

            const double average =
                math_utils::mean(values);

            const double deviation =
                math_utils::population_std(values);

            means_.push_back(average);

            // Constant features have zero variance. Mapping their scale
            // denominator to one prevents division by zero.
            standard_deviations_.push_back(
                deviation == 0.0 ? 1.0 : deviation
            );
        }

        fitted_ = true;
    }

    Matrix transform(const Matrix& data) const override {
        require_fitted();
        validate_input(data);

        Matrix result = data;

        for (auto& row : result) {
            for (std::size_t feature = 0;
                 feature < feature_count_;
                 ++feature) {

                row[feature] =
                    (row[feature] - means_[feature]) /
                    standard_deviations_[feature];
            }
        }

        return result;
    }

    Matrix inverse_transform(const Matrix& data) const override {
        require_fitted();
        validate_input(data);

        Matrix result = data;

        for (auto& row : result) {
            for (std::size_t feature = 0;
                 feature < feature_count_;
                 ++feature) {

                row[feature] =
                    row[feature] * standard_deviations_[feature] +
                    means_[feature];
            }
        }

        return result;
    }

    std::string name() const override {
        return "Standardization";
    }
};

// -----------------------------------------------------------------------------
// Min-max normalization
// -----------------------------------------------------------------------------

class MinMaxScaler final : public FeatureScaler {
private:
    Vector minimums_;
    Vector maximums_;
    double lower_;
    double upper_;

public:
    explicit MinMaxScaler(
        double lower = 0.0,
        double upper = 1.0
    )
        : lower_(lower), upper_(upper) {

        if (lower_ >= upper_) {
            throw std::invalid_argument(
                "Min-max lower bound must be smaller than upper bound."
            );
        }
    }

    void fit(const Matrix& training_data) override {
        math_utils::validate_matrix(
            training_data,
            "training_data"
        );

        feature_count_ = training_data.front().size();
        minimums_.clear();
        maximums_.clear();

        for (std::size_t feature = 0;
             feature < feature_count_;
             ++feature) {

            const Vector values =
                math_utils::get_column(training_data, feature);

            const auto limits =
                std::minmax_element(values.begin(), values.end());

            minimums_.push_back(*limits.first);
            maximums_.push_back(*limits.second);
        }

        fitted_ = true;
    }

    Matrix transform(const Matrix& data) const override {
        require_fitted();
        validate_input(data);

        Matrix result = data;

        for (auto& row : result) {
            for (std::size_t feature = 0;
                 feature < feature_count_;
                 ++feature) {

                const double minimum = minimums_[feature];
                const double maximum = maximums_[feature];

                if (minimum == maximum) {
                    row[feature] = lower_;
                    continue;
                }

                const double ratio =
                    (row[feature] - minimum) /
                    (maximum - minimum);

                row[feature] =
                    lower_ + ratio * (upper_ - lower_);
            }
        }

        return result;
    }

    Matrix inverse_transform(const Matrix& data) const override {
        require_fitted();
        validate_input(data);

        Matrix result = data;

        for (auto& row : result) {
            for (std::size_t feature = 0;
                 feature < feature_count_;
                 ++feature) {

                const double minimum = minimums_[feature];
                const double maximum = maximums_[feature];

                if (minimum == maximum) {
                    row[feature] = minimum;
                    continue;
                }

                const double ratio =
                    (row[feature] - lower_) /
                    (upper_ - lower_);

                row[feature] =
                    minimum + ratio * (maximum - minimum);
            }
        }

        return result;
    }

    std::string name() const override {
        return "Min-max normalization";
    }
};

// -----------------------------------------------------------------------------
// Robust scaling
// -----------------------------------------------------------------------------

class RobustScaler final : public FeatureScaler {
private:
    Vector medians_;
    Vector iqrs_;
    double lower_quantile_;
    double upper_quantile_;

public:
    explicit RobustScaler(
        double lower_quantile = 0.25,
        double upper_quantile = 0.75
    )
        : lower_quantile_(lower_quantile),
          upper_quantile_(upper_quantile) {

        if (
            lower_quantile_ < 0.0 ||
            upper_quantile_ > 1.0 ||
            lower_quantile_ >= upper_quantile_
        ) {
            throw std::invalid_argument(
                "Invalid robust-scaling quantiles."
            );
        }
    }

    void fit(const Matrix& training_data) override {
        math_utils::validate_matrix(
            training_data,
            "training_data"
        );

        feature_count_ = training_data.front().size();
        medians_.clear();
        iqrs_.clear();

        for (std::size_t feature = 0;
             feature < feature_count_;
             ++feature) {

            const Vector values =
                math_utils::get_column(training_data, feature);

            const double q1 =
                math_utils::quantile(
                    values,
                    lower_quantile_
                );

            const double q3 =
                math_utils::quantile(
                    values,
                    upper_quantile_
                );

            const double iqr = q3 - q1;

            medians_.push_back(
                math_utils::median(values)
            );

            iqrs_.push_back(
                iqr == 0.0 ? 1.0 : iqr
            );
        }

        fitted_ = true;
    }

    Matrix transform(const Matrix& data) const override {
        require_fitted();
        validate_input(data);

        Matrix result = data;

        for (auto& row : result) {
            for (std::size_t feature = 0;
                 feature < feature_count_;
                 ++feature) {

                row[feature] =
                    (row[feature] - medians_[feature]) /
                    iqrs_[feature];
            }
        }

        return result;
    }

    Matrix inverse_transform(const Matrix& data) const override {
        require_fitted();
        validate_input(data);

        Matrix result = data;

        for (auto& row : result) {
            for (std::size_t feature = 0;
                 feature < feature_count_;
                 ++feature) {

                row[feature] =
                    row[feature] * iqrs_[feature] +
                    medians_[feature];
            }
        }

        return result;
    }

    std::string name() const override {
        return "Robust scaling";
    }
};

// -----------------------------------------------------------------------------
// Transformation functions
// -----------------------------------------------------------------------------

Vector log_transform(const Vector& values) {
    Vector result;
    result.reserve(values.size());

    for (double value : values) {
        if (value <= 0.0) {
            throw std::domain_error(
                "Natural logarithm requires x > 0."
            );
        }

        result.push_back(std::log(value));
    }

    return result;
}

Vector signed_log_transform(const Vector& values) {
    Vector result;
    result.reserve(values.size());

    for (double value : values) {
        if (value == 0.0) {
            result.push_back(0.0);
        } else {
            result.push_back(
                std::copysign(
                    std::log1p(std::abs(value)),
                    value
                )
            );
        }
    }

    return result;
}

Vector sqrt_transform(const Vector& values) {
    Vector result;
    result.reserve(values.size());

    for (double value : values) {
        if (value < 0.0) {
            throw std::domain_error(
                "Square-root transformation requires x >= 0."
            );
        }

        result.push_back(std::sqrt(value));
    }

    return result;
}

// -----------------------------------------------------------------------------
// Feature preprocessing pipeline
// -----------------------------------------------------------------------------

class FeaturePipeline {
private:
    std::map<std::size_t, Vector(*)(const Vector&)> transformations_;
    std::unique_ptr<FeatureScaler> scaler_;

    Vector transform_column(
        const Vector& values,
        Vector(*function)(const Vector&)
    ) const {
        return function(values);
    }

    Matrix apply_transformations(const Matrix& data) const {
        math_utils::validate_matrix(data, "pipeline input");

        Matrix result = data;

        for (const auto& [feature, function] : transformations_) {
            if (feature >= data.front().size()) {
                throw std::out_of_range(
                    "Pipeline references a nonexistent feature."
                );
            }

            const Vector original =
                math_utils::get_column(data, feature);

            const Vector transformed =
                transform_column(original, function);

            for (std::size_t row = 0;
                 row < result.size();
                 ++row) {

                result[row][feature] = transformed[row];
            }
        }

        return result;
    }

public:
    FeaturePipeline(
        std::map<std::size_t, Vector(*)(const Vector&)> transformations,
        std::unique_ptr<FeatureScaler> scaler
    )
        : transformations_(std::move(transformations)),
          scaler_(std::move(scaler)) {

        if (!scaler_) {
            throw std::invalid_argument(
                "Pipeline requires a scaler."
            );
        }
    }

    void fit(const Matrix& training_data) {
        const Matrix transformed =
            apply_transformations(training_data);

        scaler_->fit(transformed);
    }

    Matrix transform(const Matrix& data) const {
        const Matrix transformed =
            apply_transformations(data);

        return scaler_->transform(transformed);
    }
};

// -----------------------------------------------------------------------------
// Governance-style preprocessing contract
// -----------------------------------------------------------------------------

struct PreprocessingPolicy {
    std::string expected_scaler;
    bool fit_only_on_training_data = true;
    bool allow_out_of_training_range = true;
    bool transform_skewed_feature_before_scaling = false;
};

class PreprocessingContract {
private:
    PreprocessingPolicy policy_;

public:
    explicit PreprocessingContract(
        PreprocessingPolicy policy
    )
        : policy_(std::move(policy)) {}

    void validate(
        const std::string& actual_scaler,
        bool fitted_with_test_data,
        bool transformation_order_is_valid
    ) const {
        if (actual_scaler != policy_.expected_scaler) {
            throw std::runtime_error(
                "Preprocessing policy violation: scaler type differs."
            );
        }

        if (
            policy_.fit_only_on_training_data &&
            fitted_with_test_data
        ) {
            throw std::runtime_error(
                "Preprocessing policy violation: test data influenced fit."
            );
        }

        if (
            policy_.transform_skewed_feature_before_scaling &&
            !transformation_order_is_valid
        ) {
            throw std::runtime_error(
                "Preprocessing policy violation: transformation order differs."
            );
        }
    }
};

// -----------------------------------------------------------------------------
// Case study data
// -----------------------------------------------------------------------------

Matrix build_training_data() {
    /*
     * Columns:
     *   income       - currency units
     *   transactions - monthly transaction count
     *   spend        - average transaction value
     *   account_age  - months
     *
     * The units differ by several orders of magnitude, while income contains
     * a deliberate high-value observation that exposes outlier sensitivity.
     */
    return {
        {28000.0, 8.0, 1200.0, 14.0},
        {35000.0, 12.0, 1450.0, 20.0},
        {42000.0, 15.0, 1700.0, 28.0},
        {48000.0, 20.0, 2100.0, 35.0},
        {55000.0, 24.0, 2300.0, 42.0},
        {63000.0, 31.0, 2600.0, 51.0},
        {75000.0, 38.0, 3000.0, 66.0},
        {95000.0, 47.0, 3500.0, 82.0},
        {140000.0, 75.0, 5100.0, 120.0},
        {850000.0, 110.0, 8900.0, 144.0}
    };
}

Matrix build_inference_data() {
    return {
        {55000.0, 18.0, 1900.0, 38.0},
        {100000.0, 42.0, 4200.0, 90.0},
        {200000.0, 85.0, 7000.0, 132.0}
    };
}

// -----------------------------------------------------------------------------
// Case study analysis
// -----------------------------------------------------------------------------

void compare_scalers(const Matrix& training) {
    std::vector<std::unique_ptr<FeatureScaler>> scalers;

    scalers.push_back(
        std::make_unique<StandardScaler>()
    );

    scalers.push_back(
        std::make_unique<MinMaxScaler>()
    );

    scalers.push_back(
        std::make_unique<RobustScaler>()
    );

    for (auto& scaler : scalers) {
        scaler->fit(training);

        const Matrix transformed =
            scaler->transform(training);

        math_utils::print_matrix(
            scaler->name(),
            transformed
        );
    }
}

void analyze_distance_effect(const Matrix& training) {
    const Vector first = training.at(0);
    const Vector second = training.at(1);

    std::cout << "\nDISTANCE EFFECT\n";

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout
        << "Raw Euclidean distance: "
        << math_utils::euclidean_distance(first, second)
        << "\n";

    std::vector<std::unique_ptr<FeatureScaler>> scalers;

    scalers.push_back(
        std::make_unique<StandardScaler>()
    );

    scalers.push_back(
        std::make_unique<MinMaxScaler>()
    );

    scalers.push_back(
        std::make_unique<RobustScaler>()
    );

    for (auto& scaler : scalers) {
        scaler->fit(training);

        const Matrix transformed =
            scaler->transform(training);

        std::cout
            << scaler->name()
            << " distance: "
            << math_utils::euclidean_distance(
                transformed.at(0),
                transformed.at(1)
            )
            << "\n";
    }
}

void analyze_outlier_sensitivity() {
    const Matrix ordinary = {
        {30000.0},
        {32000.0},
        {35000.0},
        {37000.0},
        {40000.0},
        {42000.0}
    };

    const Matrix with_outlier = {
        {30000.0},
        {32000.0},
        {35000.0},
        {37000.0},
        {40000.0},
        {42000.0},
        {2000000.0}
    };

    StandardScaler standard_without;
    StandardScaler standard_with;

    RobustScaler robust_without;
    RobustScaler robust_with;

    standard_without.fit(ordinary);
    standard_with.fit(with_outlier);

    robust_without.fit(ordinary);
    robust_with.fit(with_outlier);

    const Matrix standard_a =
        standard_without.transform(ordinary);

    const Matrix standard_b =
        standard_with.transform(ordinary);

    const Matrix robust_a =
        robust_without.transform(ordinary);

    const Matrix robust_b =
        robust_with.transform(ordinary);

    std::cout << "\nOUTLIER SENSITIVITY\n";

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout
        << "Standard first value without outlier: "
        << standard_a.front().front()
        << "\n";

    std::cout
        << "Standard first value with outlier:    "
        << standard_b.front().front()
        << "\n";

    std::cout
        << "Robust first value without outlier:   "
        << robust_a.front().front()
        << "\n";

    std::cout
        << "Robust first value with outlier:      "
        << robust_b.front().front()
        << "\n";
}

void demonstrate_transformations() {
    const Vector revenue = {
        100.0,
        250.0,
        500.0,
        1000.0,
        5000.0,
        25000.0,
        250000.0
    };

    const Vector signed_change = {
        -500.0,
        -100.0,
        -10.0,
        0.0,
        15.0,
        200.0,
        2000.0
    };

    const Vector counts = {
        0.0,
        1.0,
        4.0,
        9.0,
        25.0,
        100.0,
        400.0
    };

    const Vector logged =
        log_transform(revenue);

    const Vector signed_logged =
        signed_log_transform(signed_change);

    const Vector square_root =
        sqrt_transform(counts);

    std::cout << "\nTRANSFORMATIONS\n";

    std::cout << "Log revenue:\n  ";

    for (double value : logged) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << value << " ";
    }

    std::cout << "\nSigned-log changes:\n  ";

    for (double value : signed_logged) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << value << " ";
    }

    std::cout << "\nSquare-root counts:\n  ";

    for (double value : square_root) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << value << " ";
    }

    std::cout << "\n";
}

void demonstrate_train_test_discipline() {
    const Matrix training = {
        {20000.0, 10.0},
        {30000.0, 12.0},
        {40000.0, 14.0},
        {50000.0, 15.0}
    };

    const Matrix test = {
        {55000.0, 16.0},
        {200000.0, 17.0}
    };

    StandardScaler correct_scaler;
    StandardScaler leaked_scaler;

    correct_scaler.fit(training);

    Matrix combined = training;
    combined.insert(
        combined.end(),
        test.begin(),
        test.end()
    );

    leaked_scaler.fit(combined);

    const Matrix correct =
        correct_scaler.transform(test);

    const Matrix leaked =
        leaked_scaler.transform(test);

    std::cout << "\nTRAINING-ONLY FITTING\n";

    math_utils::print_matrix(
        "Correct test transformation",
        correct,
        4
    );

    math_utils::print_matrix(
        "Transformation after test-set leakage",
        leaked,
        4
    );
}

void demonstrate_inverse_transformation() {
    const Matrix data = {
        {10.0, 100.0},
        {20.0, 200.0},
        {30.0, 300.0}
    };

    StandardScaler scaler;
    scaler.fit(data);

    const Matrix scaled =
        scaler.transform(data);

    const Matrix restored =
        scaler.inverse_transform(scaled);

    math_utils::print_matrix(
        "Original data",
        data
    );

    math_utils::print_matrix(
        "Standardized data",
        scaled
    );

    math_utils::print_matrix(
        "Inverse-transformed data",
        restored
    );

    double maximum_error = 0.0;

    for (std::size_t row = 0; row < data.size(); ++row) {
        for (std::size_t feature = 0;
             feature < data.front().size();
             ++feature) {

            maximum_error =
                std::max(
                    maximum_error,
                    std::abs(
                        data[row][feature] -
                        restored[row][feature]
                    )
                );
        }
    }

    std::cout
        << "\nMaximum reconstruction error: "
        << std::setprecision(12)
        << maximum_error
        << "\n";
}

void demonstrate_pipeline(const Matrix& training) {
    const Matrix inference = build_inference_data();

    /*
     * The average-spend feature is strongly right-skewed. The pipeline applies
     * log transformation before standardization, rather than standardizing
     * the raw skewed feature first.
     */
    FeaturePipeline pipeline(
        {
            {2, log_transform}
        },
        std::make_unique<StandardScaler>()
    );

    pipeline.fit(training);

    const Matrix transformed =
        pipeline.transform(inference);

    math_utils::print_matrix(
        "Pipeline: log(spend) followed by standardization",
        transformed,
        4
    );
}

void demonstrate_policy_validation() {
    PreprocessingPolicy policy{
        "Standardization",
        true,
        true,
        true
    };

    PreprocessingContract contract(policy);

    contract.validate(
        "Standardization",
        false,
        true
    );

    std::cout
        << "\nPreprocessing contract: valid configuration accepted.\n";

    try {
        contract.validate(
            "Robust scaling",
            false,
            true
        );
    } catch (const std::exception& error) {
        std::cout
            << "Rejected wrong scaler: "
            << error.what()
            << "\n";
    }

    try {
        contract.validate(
            "Standardization",
            true,
            true
        );
    } catch (const std::exception& error) {
        std::cout
            << "Rejected test-data leakage: "
            << error.what()
            << "\n";
    }

    try {
        contract.validate(
            "Standardization",
            false,
            false
        );
    } catch (const std::exception& error) {
        std::cout
            << "Rejected incorrect transformation order: "
            << error.what()
            << "\n";
    }
}

void demonstrate_failure_conditions() {
    std::cout << "\nFAILURE CONDITIONS\n";

    try {
        StandardScaler scaler;
        scaler.transform({{1.0, 2.0}});
    } catch (const std::exception& error) {
        std::cout
            << "Unfitted scaler rejected: "
            << error.what()
            << "\n";
    }

    try {
        log_transform({10.0, 0.0, 20.0});
    } catch (const std::exception& error) {
        std::cout
            << "Invalid logarithm rejected: "
            << error.what()
            << "\n";
    }

    try {
        sqrt_transform({4.0, -1.0});
    } catch (const std::exception& error) {
        std::cout
            << "Invalid square root rejected: "
            << error.what()
            << "\n";
    }

    try {
        StandardScaler scaler;

        scaler.fit({
            {1.0, 2.0},
            {3.0}
        });
    } catch (const std::exception& error) {
        std::cout
            << "Non-rectangular input rejected: "
            << error.what()
            << "\n";
    }

    try {
        MinMaxScaler scaler(1.0, 1.0);
    } catch (const std::exception& error) {
        std::cout
            << "Invalid min-max range rejected: "
            << error.what()
            << "\n";
    }
}

void demonstrate_constant_feature() {
    const Matrix data = {
        {10.0, 100.0},
        {20.0, 100.0},
        {30.0, 100.0}
    };

    StandardScaler scaler;
    scaler.fit(data);

    const Matrix transformed =
        scaler.transform(data);

    math_utils::print_matrix(
        "Constant feature under standardization",
        transformed,
        4
    );
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    try {
        std::cout
            << "=====================================================================\n"
            << "FEATURE SCALING CASE STUDY\n"
            << "Standardization | Normalization | Robust Scaling | Transformations\n"
            << "=====================================================================\n";

        const Matrix training =
            build_training_data();

        math_utils::print_matrix(
            "Raw customer financial measurements",
            training,
            2
        );

        compare_scalers(training);
        analyze_distance_effect(training);
        analyze_outlier_sensitivity();
        demonstrate_transformations();
        demonstrate_train_test_discipline();
        demonstrate_inverse_transformation();
        demonstrate_pipeline(training);
        demonstrate_policy_validation();
        demonstrate_failure_conditions();
        demonstrate_constant_feature();

        std::cout
            << "\nCASE STUDY DESIGN NOTES\n"
            << "Standardization centers features using training means and "
               "scales them by training standard deviations.\n"
            << "Min-max normalization maps training extrema into a chosen "
               "interval, but future observations may fall outside it.\n"
            << "Robust scaling uses median and IQR so isolated extreme "
               "observations have less influence on learned parameters.\n"
            << "Transformations alter distribution shape and are therefore "
               "conceptually different from simple rescaling.\n"
            << "The preprocessing contract demonstrates that fitted "
               "parameters belong to the training pipeline and must not be "
               "recalculated from test or production observations.\n";

        std::cout
            << "\nExecution completed successfully.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << "\n";

        return 1;
    }
}
