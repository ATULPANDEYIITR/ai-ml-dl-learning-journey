#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Repository-independent technical case study:
 * A municipal water utility predicts maintenance urgency from sensor
 * measurements and historical maintenance outcomes.
 *
 * k-NN is used to classify equipment risk and estimate a continuous
 * maintenance cost. The implementation includes feature standardization,
 * multiple distance metrics, deterministic neighborhood selection,
 * weighted voting, validation, and evaluation.
 *
 * Compile: g++ -std=c++17 -O2 knn_utility.cpp -o knn_utility
 */

using Vector = std::vector<double>;

enum class Metric {
    Euclidean,
    Manhattan,
    Chebyshev
};

enum class WeightMode {
    Uniform,
    InverseDistance
};

struct EquipmentRecord {
    std::string assetId;
    Vector measurements;
    std::string riskCategory;
    double maintenanceCost;
};

struct Neighbor {
    std::size_t index;
    double distance;
};

void validateVector(const Vector& values, const std::string& name) {
    if (values.empty()) {
        throw std::invalid_argument(name + " cannot be empty");
    }

    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(name + " contains a nonfinite value");
        }
    }
}

std::size_t validateDataset(const std::vector<Vector>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }

    const std::size_t dimensions = samples.front().size();
    if (dimensions == 0) {
        throw std::invalid_argument("Samples need at least one feature");
    }

    for (const auto& sample : samples) {
        validateVector(sample, "sample");
        if (sample.size() != dimensions) {
            throw std::invalid_argument("Inconsistent feature dimensions");
        }
    }

    return dimensions;
}

double distance(const Vector& a, const Vector& b, Metric metric) {
    validateVector(a, "left operand");
    validateVector(b, "right operand");

    if (a.size() != b.size()) {
        throw std::invalid_argument("Distance operands have different dimensions");
    }

    double result = 0.0;

    switch (metric) {
        case Metric::Euclidean:
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double difference = a[i] - b[i];
                result += difference * difference;
            }
            result = std::sqrt(result);
            break;

        case Metric::Manhattan:
            for (std::size_t i = 0; i < a.size(); ++i) {
                result += std::abs(a[i] - b[i]);
            }
            break;

        case Metric::Chebyshev:
            for (std::size_t i = 0; i < a.size(); ++i) {
                result = std::max(result, std::abs(a[i] - b[i]));
            }
            break;
    }

    if (!std::isfinite(result)) {
        throw std::overflow_error("Distance calculation overflowed");
    }
    return result;
}

class StandardScaler {
private:
    Vector means_;
    Vector scales_;
    bool fitted_ = false;

public:
    void fit(const std::vector<Vector>& samples) {
        const std::size_t dimensions = validateDataset(samples);
        means_.assign(dimensions, 0.0);
        scales_.assign(dimensions, 1.0);

        for (const auto& sample : samples) {
            for (std::size_t column = 0; column < dimensions; ++column) {
                means_[column] += sample[column];
            }
        }

        for (double& mean : means_) {
            mean /= static_cast<double>(samples.size());
        }

        for (const auto& sample : samples) {
            for (std::size_t column = 0; column < dimensions; ++column) {
                const double difference = sample[column] - means_[column];
                scales_[column] += difference * difference;
            }
        }

        for (std::size_t column = 0; column < dimensions; ++column) {
            // Population standard deviation; a constant feature receives scale 1.
            const double variance =
                (scales_[column] - 1.0) / static_cast<double>(samples.size());
            scales_[column] = variance > 0.0 ? std::sqrt(variance) : 1.0;
        }

        fitted_ = true;
    }

    Vector transform(const Vector& sample) const {
        if (!fitted_) {
            throw std::logic_error("Scaler has not been fitted");
        }

        validateVector(sample, "sample");
        if (sample.size() != means_.size()) {
            throw std::invalid_argument("Feature count differs from scaler");
        }

        Vector result(sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i) {
            result[i] = (sample[i] - means_[i]) / scales_[i];
        }
        return result;
    }

    std::vector<Vector> transform(const std::vector<Vector>& samples) const {
        validateDataset(samples);
        std::vector<Vector> transformed;
        transformed.reserve(samples.size());

        for (const auto& sample : samples) {
            transformed.push_back(transform(sample));
        }
        return transformed;
    }
};

class KNNClassifier {
private:
    std::vector<Vector> samples_;
    std::vector<std::string> labels_;
    std::vector<std::string> classes_;
    std::size_t dimensions_ = 0;
    std::size_t k_;
    Metric metric_;
    WeightMode weighting_;

    double neighborWeight(double value) const {
        if (weighting_ == WeightMode::Uniform || value == 0.0) {
            return 1.0;
        }
        return 1.0 / value;
    }

public:
    KNNClassifier(std::size_t k, Metric metric, WeightMode weighting)
        : k_(k), metric_(metric), weighting_(weighting) {
        if (k_ == 0) {
            throw std::invalid_argument("k must be positive");
        }
    }

    void fit(
        const std::vector<Vector>& samples,
        const std::vector<std::string>& labels
    ) {
        dimensions_ = validateDataset(samples);

        if (samples.size() != labels.size()) {
            throw std::invalid_argument("Samples and labels must align");
        }
        if (k_ > samples.size()) {
            throw std::invalid_argument("k exceeds training sample count");
        }

        for (const auto& label : labels) {
            if (label.empty()) {
                throw std::invalid_argument("Class labels cannot be empty");
            }
        }

        samples_ = samples;
        labels_ = labels;
        classes_ = labels_;
        std::sort(classes_.begin(), classes_.end());
        classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
    }

    std::vector<Neighbor> kneighbors(const Vector& query) const {
        if (samples_.empty()) {
            throw std::logic_error("Classifier has not been trained");
        }

        validateVector(query, "query");
        if (query.size() != dimensions_) {
            throw std::invalid_argument("Query has the wrong feature count");
        }

        std::vector<Neighbor> neighbors;
        neighbors.reserve(samples_.size());

        for (std::size_t i = 0; i < samples_.size(); ++i) {
            neighbors.push_back({i, distance(query, samples_[i], metric_)});
        }

        // Partial sorting selects only the closest k records.
        // The tie-break on index makes duplicate-distance results deterministic.
        std::partial_sort(
            neighbors.begin(),
            neighbors.begin() + static_cast<std::ptrdiff_t>(k_),
            neighbors.end(),
            [](const Neighbor& a, const Neighbor& b) {
                if (a.distance != b.distance) {
                    return a.distance < b.distance;
                }
                return a.index < b.index;
            }
        );

        neighbors.resize(k_);
        return neighbors;
    }

    std::map<std::string, double> predictProbabilities(const Vector& query) const {
        auto neighbors = kneighbors(query);

        if (weighting_ == WeightMode::InverseDistance) {
            std::vector<Neighbor> exact;
            std::copy_if(
                neighbors.begin(), neighbors.end(), std::back_inserter(exact),
                [](const Neighbor& neighbor) { return neighbor.distance == 0.0; }
            );
            if (!exact.empty()) {
                neighbors = exact;
            }
        }

        std::map<std::string, double> totals;
        std::map<std::string, double> distanceTotals;

        for (const auto& label : classes_) {
            totals[label] = 0.0;
            distanceTotals[label] = 0.0;
        }

        for (const auto& neighbor : neighbors) {
            const std::string& label = labels_[neighbor.index];
            totals[label] += neighborWeight(neighbor.distance);
            distanceTotals[label] += neighbor.distance;
        }

        double denominator = 0.0;
        for (const auto& entry : totals) {
            denominator += entry.second;
        }

        for (auto& entry : totals) {
            entry.second /= denominator;
        }
        return totals;
    }

    std::string predict(const Vector& query) const {
        const auto probabilities = predictProbabilities(query);
        const auto neighbors = kneighbors(query);
        std::map<std::string, double> distanceTotals;

        for (const auto& label : classes_) {
            distanceTotals[label] = 0.0;
        }
        for (const auto& neighbor : neighbors) {
            distanceTotals[labels_[neighbor.index]] += neighbor.distance;
        }

        std::string winner = classes_.front();
        for (const auto& candidate : classes_) {
            const double candidateProbability = probabilities.at(candidate);
            const double winnerProbability = probabilities.at(winner);

            if (candidateProbability > winnerProbability ||
                (candidateProbability == winnerProbability &&
                 distanceTotals[candidate] < distanceTotals[winner]) ||
                (candidateProbability == winnerProbability &&
                 distanceTotals[candidate] == distanceTotals[winner] &&
                 candidate < winner)) {
                winner = candidate;
            }
        }
        return winner;
    }
};

class KNNRegressor {
private:
    std::vector<Vector> samples_;
    std::vector<double> targets_;
    std::size_t dimensions_ = 0;
    std::size_t k_;
    Metric metric_;
    WeightMode weighting_;

public:
    KNNRegressor(std::size_t k, Metric metric, WeightMode weighting)
        : k_(k), metric_(metric), weighting_(weighting) {
        if (k_ == 0) {
            throw std::invalid_argument("k must be positive");
        }
    }

    void fit(
        const std::vector<Vector>& samples,
        const std::vector<double>& targets
    ) {
        dimensions_ = validateDataset(samples);

        if (samples.size() != targets.size() || k_ > samples.size()) {
            throw std::invalid_argument("Invalid training sample or k count");
        }

        for (double target : targets) {
            if (!std::isfinite(target)) {
                throw std::invalid_argument("Regression targets must be finite");
            }
        }

        samples_ = samples;
        targets_ = targets;
    }

    double predict(const Vector& query) const {
        if (samples_.empty()) {
            throw std::logic_error("Regressor has not been trained");
        }

        validateVector(query, "query");
        if (query.size() != dimensions_) {
            throw std::invalid_argument("Query has the wrong feature count");
        }

        std::vector<Neighbor> neighbors;
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            neighbors.push_back({i, distance(query, samples_[i], metric_)});
        }

        std::sort(
            neighbors.begin(), neighbors.end(),
            [](const Neighbor& a, const Neighbor& b) {
                if (a.distance != b.distance) return a.distance < b.distance;
                return a.index < b.index;
            }
        );
        neighbors.resize(k_);

        const bool hasExactMatch =
            weighting_ == WeightMode::InverseDistance &&
            std::any_of(neighbors.begin(), neighbors.end(),
                        [](const Neighbor& n) { return n.distance == 0.0; });

        double numerator = 0.0;
        double denominator = 0.0;

        for (const auto& neighbor : neighbors) {
            if (hasExactMatch && neighbor.distance != 0.0) {
                continue;
            }

            const double weight =
                weighting_ == WeightMode::Uniform || neighbor.distance == 0.0
                ? 1.0
                : 1.0 / neighbor.distance;

            numerator += weight * targets_[neighbor.index];
            denominator += weight;
        }

        return numerator / denominator;
    }
};

double classificationAccuracy(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& predicted
) {
    if (actual.empty() || actual.size() != predicted.size()) {
        throw std::invalid_argument("Invalid evaluation vectors");
    }

    std::size_t correct = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        correct += actual[i] == predicted[i];
    }
    return static_cast<double>(correct) / actual.size();
}

int main() {
    try {
        std::cout << std::fixed << std::setprecision(3);

        std::vector<EquipmentRecord> records = {
            {"PUMP-101", {2.0, 0.8, 12.0}, "normal", 1200.0},
            {"PUMP-102", {2.4, 1.0, 15.0}, "normal", 1400.0},
            {"PUMP-103", {3.2, 1.8, 25.0}, "watch", 2800.0},
            {"PUMP-104", {3.6, 2.0, 29.0}, "watch", 3200.0},
            {"PUMP-105", {5.5, 4.5, 55.0}, "urgent", 9000.0},
            {"PUMP-106", {6.0, 5.1, 61.0}, "urgent", 11000.0},
            {"PUMP-107", {6.4, 5.8, 70.0}, "urgent", 12500.0},
            {"PUMP-108", {2.8, 1.3, 19.0}, "normal", 2100.0}
        };

        std::vector<Vector> X;
        std::vector<std::string> y;
        std::vector<double> costs;

        for (const auto& record : records) {
            X.push_back(record.measurements);
            y.push_back(record.riskCategory);
            costs.push_back(record.maintenanceCost);
        }

        StandardScaler scaler;
        scaler.fit(X);
        const auto normalized = scaler.transform(X);

        KNNClassifier classifier(3, Metric::Euclidean, WeightMode::InverseDistance);
        classifier.fit(normalized, y);

        const Vector incomingMeasurements = {4.8, 3.9, 49.0};
        const Vector normalizedQuery = scaler.transform(incomingMeasurements);

        std::cout << "Municipal pump condition assessment\n";
        std::cout << "Asset: PUMP-NEW-209\n";
        std::cout << "Predicted category: " << classifier.predict(normalizedQuery) << "\n";

        std::cout << "Estimated class probabilities:\n";
        for (const auto& entry : classifier.predictProbabilities(normalizedQuery)) {
            std::cout << "  " << entry.first << ": " << entry.second << "\n";
        }

        std::cout << "Nearest reference assets:\n";
        for (const auto& neighbor : classifier.kneighbors(normalizedQuery)) {
            std::cout << "  " << records[neighbor.index].assetId
                      << ", distance=" << neighbor.distance
                      << ", class=" << records[neighbor.index].riskCategory
                      << "\n";
        }

        KNNRegressor costModel(3, Metric::Euclidean, WeightMode::InverseDistance);
        costModel.fit(normalized, costs);
        std::cout << "Estimated maintenance cost: "
                  << costModel.predict(normalizedQuery) << "\n";

        std::vector<std::string> predictedTrainingLabels;
        for (const auto& sample : normalized) {
            predictedTrainingLabels.push_back(classifier.predict(sample));
        }
        std::cout << "Training diagnostic accuracy: "
                  << classificationAccuracy(y, predictedTrainingLabels) << "\n";

        // A production service must evaluate on held-out assets. Training accuracy
        // is diagnostic only and should never be presented as generalization accuracy.

        try {
            distance({1.0, 2.0}, {1.0}, Metric::Euclidean);
        } catch (const std::invalid_argument& error) {
            std::cout << "Rejected invalid query: " << error.what() << "\n";
        }

        try {
            KNNClassifier invalidModel(100, Metric::Euclidean, WeightMode::Uniform);
            invalidModel.fit(normalized, y);
        } catch (const std::invalid_argument& error) {
            std::cout << "Rejected invalid neighborhood size: " << error.what() << "\n";
        }

        std::cout << "\nOperational interpretation:\n";
        std::cout << "High predicted risk prioritizes inspection, not automatic shutdown.\n";
        std::cout << "A technician must verify the condition before costly intervention.\n";
    } catch (const std::exception& error) {
        std::cerr << "Assessment failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
