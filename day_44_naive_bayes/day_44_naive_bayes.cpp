#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * Repository-independent technical case study:
 * classify equipment telemetry using Gaussian Naive Bayes.
 *
 * Each observation contains vibration RMS and operating temperature.
 * The model estimates class-conditional Gaussian densities and combines
 * their log likelihoods with class priors.
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic naive_bayes.cpp -o naive_bayes
 */

using Sample = std::vector<double>;
using Label = std::string;

struct Observation {
    Sample features;
    Label condition;
};

class GaussianNaiveBayes {
private:
    std::vector<Label> classes_;
    std::map<Label, double> log_priors_;
    std::map<Label, Sample> means_;
    std::map<Label, Sample> variances_;
    std::size_t feature_count_ = 0;
    double smoothing_ = 1e-9;
    bool fitted_ = false;

    static void validateSample(const Sample& sample) {
        if (sample.empty()) {
            throw std::invalid_argument("An observation must contain features.");
        }

        for (double value : sample) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Features must be finite.");
            }
        }
    }

public:
    explicit GaussianNaiveBayes(double smoothing = 1e-9)
        : smoothing_(smoothing) {
        if (!std::isfinite(smoothing_) || smoothing_ <= 0.0) {
            throw std::invalid_argument("Smoothing must be finite and positive.");
        }
    }

    void fit(const std::vector<Observation>& observations) {
        if (observations.empty()) {
            throw std::invalid_argument("Training data cannot be empty.");
        }

        feature_count_ = observations.front().features.size();
        if (feature_count_ == 0) {
            throw std::invalid_argument("At least one feature is required.");
        }

        std::map<Label, std::vector<Sample>> grouped;
        std::vector<Sample> all_samples;

        for (const auto& observation : observations) {
            validateSample(observation.features);

            if (observation.features.size() != feature_count_) {
                throw std::invalid_argument("Feature dimensions are inconsistent.");
            }
            if (observation.condition.empty()) {
                throw std::invalid_argument("Condition labels cannot be empty.");
            }

            grouped[observation.condition].push_back(observation.features);
            all_samples.push_back(observation.features);
        }

        Sample global_variances(feature_count_, 0.0);
        for (std::size_t column = 0; column < feature_count_; ++column) {
            double mean = 0.0;

            for (const auto& sample : all_samples) {
                mean += sample[column];
            }
            mean /= static_cast<double>(all_samples.size());

            for (const auto& sample : all_samples) {
                const double difference = sample[column] - mean;
                global_variances[column] += difference * difference;
            }
            global_variances[column] /= static_cast<double>(all_samples.size());
        }

        std::vector<Label> new_classes;
        std::map<Label, double> new_priors;
        std::map<Label, Sample> new_means;
        std::map<Label, Sample> new_variances;

        for (const auto& entry : grouped) {
            const Label& label = entry.first;
            const auto& rows = entry.second;
            const double class_size = static_cast<double>(rows.size());

            new_classes.push_back(label);
            new_priors[label] = std::log(class_size / all_samples.size());

            Sample means(feature_count_, 0.0);
            Sample variances(feature_count_, 0.0);

            for (std::size_t column = 0; column < feature_count_; ++column) {
                for (const auto& row : rows) {
                    means[column] += row[column];
                }
                means[column] /= class_size;

                for (const auto& row : rows) {
                    const double difference = row[column] - means[column];
                    variances[column] += difference * difference;
                }

                variances[column] /= class_size;
                const double scale = std::max(global_variances[column], 1.0);
                variances[column] = std::max(
                    variances[column],
                    smoothing_ * scale
                );
            }

            new_means[label] = std::move(means);
            new_variances[label] = std::move(variances);
        }

        // Publish model state only after every class has been computed.
        classes_ = std::move(new_classes);
        log_priors_ = std::move(new_priors);
        means_ = std::move(new_means);
        variances_ = std::move(new_variances);
        fitted_ = true;
    }

    std::map<Label, double> predictLogScores(const Sample& sample) const {
        if (!fitted_) {
            throw std::logic_error("Fit the model before prediction.");
        }

        validateSample(sample);
        if (sample.size() != feature_count_) {
            throw std::invalid_argument("Prediction feature count is incorrect.");
        }

        std::map<Label, double> scores;

        for (const auto& label : classes_) {
            double score = log_priors_.at(label);

            for (std::size_t column = 0; column < feature_count_; ++column) {
                const double mean = means_.at(label)[column];
                const double variance = variances_.at(label)[column];
                const double difference = sample[column] - mean;

                // The Gaussian density is evaluated in log space to avoid
                // underflow when several small densities are multiplied.
                score -= 0.5 * (
                    std::log(2.0 * std::acos(-1.0) * variance)
                    + difference * difference / variance
                );
            }

            scores[label] = score;
        }

        return scores;
    }

    std::map<Label, double> predictProbabilities(const Sample& sample) const {
        const auto scores = predictLogScores(sample);
        const double maximum = std::max_element(
            scores.begin(),
            scores.end(),
            [](const auto& left, const auto& right) {
                return left.second < right.second;
            }
        )->second;

        double denominator = 0.0;
        for (const auto& entry : scores) {
            denominator += std::exp(entry.second - maximum);
        }

        std::map<Label, double> probabilities;
        for (const auto& entry : scores) {
            probabilities[entry.first] =
                std::exp(entry.second - maximum) / denominator;
        }

        return probabilities;
    }

    Label predict(const Sample& sample) const {
        const auto scores = predictLogScores(sample);
        return std::max_element(
            scores.begin(),
            scores.end(),
            [](const auto& left, const auto& right) {
                return left.second < right.second;
            }
        )->first;
    }

    const std::vector<Label>& classes() const {
        if (!fitted_) {
            throw std::logic_error("The model has not been fitted.");
        }
        return classes_;
    }
};

struct MaintenanceDecision {
    std::string asset_id;
    Label predicted_condition;
    double fault_probability;
    std::string action;
};

class MaintenanceEngine {
private:
    const GaussianNaiveBayes& model_;
    double intervention_threshold_;

public:
    MaintenanceEngine(
        const GaussianNaiveBayes& model,
        double intervention_threshold
    )
        : model_(model), intervention_threshold_(intervention_threshold) {
        if (!std::isfinite(intervention_threshold_) ||
            intervention_threshold_ < 0.0 ||
            intervention_threshold_ > 1.0) {
            throw std::invalid_argument("Threshold must lie in [0, 1].");
        }
    }

    MaintenanceDecision assess(
        const std::string& asset_id,
        const Sample& telemetry
    ) const {
        if (asset_id.empty()) {
            throw std::invalid_argument("Asset ID cannot be empty.");
        }

        const auto probabilities = model_.predictProbabilities(telemetry);
        const auto fault = probabilities.find("fault");

        if (fault == probabilities.end()) {
            throw std::logic_error(
                "The model must be trained with a 'fault' class."
            );
        }

        const double fault_probability = fault->second;
        std::string action = "Continue routine monitoring";

        if (fault_probability >= intervention_threshold_) {
            action = "Raise maintenance alert and inspect asset";
        } else if (fault_probability >= intervention_threshold_ * 0.6) {
            action = "Increase telemetry sampling frequency";
        }

        return {
            asset_id,
            model_.predict(telemetry),
            fault_probability,
            action
        };
    }
};

static std::vector<Observation> trainingData() {
    return {
        {{1.1, 43.0}, "normal"},
        {{1.3, 45.0}, "normal"},
        {{1.5, 44.0}, "normal"},
        {{1.2, 46.0}, "normal"},
        {{1.4, 42.0}, "normal"},
        {{1.6, 47.0}, "normal"},
        {{1.0, 44.0}, "normal"},
        {{1.3, 43.0}, "normal"},
        {{1.7, 48.0}, "normal"},
        {{1.2, 41.0}, "normal"},
        {{4.2, 83.0}, "fault"},
        {{4.5, 87.0}, "fault"},
        {{4.7, 90.0}, "fault"},
        {{4.3, 85.0}, "fault"},
        {{4.8, 91.0}, "fault"},
        {{4.1, 82.0}, "fault"},
        {{4.6, 88.0}, "fault"},
        {{4.4, 86.0}, "fault"},
        {{4.0, 80.0}, "fault"},
        {{4.9, 92.0}, "fault"}
    };
}

int main() {
    try {
        const auto training = trainingData();
        GaussianNaiveBayes model;
        model.fit(training);

        MaintenanceEngine maintenance(model, 0.80);

        const std::vector<std::pair<std::string, Sample>> readings = {
            {"PUMP-104", {1.35, 44.0}},
            {"MOTOR-208", {3.20, 69.0}},
            {"COMPRESSOR-031", {4.35, 86.0}}
        };

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Predictive maintenance assessment\n";
        std::cout << "----------------------------------\n";

        for (const auto& reading : readings) {
            const auto decision = maintenance.assess(
                reading.first,
                reading.second
            );

            std::cout << "\nAsset: " << decision.asset_id
                      << "\nPredicted condition: "
                      << decision.predicted_condition
                      << "\nFault probability: "
                      << decision.fault_probability
                      << "\nAction: " << decision.action << '\n';

            const auto probabilities =
                model.predictProbabilities(reading.second);

            double total_probability = 0.0;
            for (const auto& entry : probabilities) {
                total_probability += entry.second;
            }

            if (std::abs(total_probability - 1.0) > 1e-10) {
                throw std::runtime_error("Probability normalization failed.");
            }
        }

        // A constant feature is handled by the variance floor.
        GaussianNaiveBayes constant_feature_model;
        constant_feature_model.fit({
            {{1.0, 10.0}, "normal"},
            {{1.1, 10.0}, "normal"},
            {{5.0, 10.0}, "fault"},
            {{5.1, 10.0}, "fault"}
        });

        if (constant_feature_model.predict({5.05, 10.0}) != "fault") {
            throw std::runtime_error("Constant-feature test failed.");
        }

        bool rejected_invalid_input = false;
        try {
            model.predict({std::numeric_limits<double>::quiet_NaN(), 40.0});
        } catch (const std::invalid_argument&) {
            rejected_invalid_input = true;
        }

        if (!rejected_invalid_input) {
            throw std::runtime_error("Non-finite telemetry was not rejected.");
        }

        std::cout << "\nNumerical stability and input validation tests passed.\n";
        std::cout
            << "Operational limitation: a probability is a model estimate, "
            << "not a calibrated guarantee of failure.\n";
    } catch (const std::exception& error) {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
