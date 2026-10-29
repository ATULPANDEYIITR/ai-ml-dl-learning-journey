#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/*
 * Repository Governance Anomaly Detection Engine
 *
 * The program models repository activity as feature vectors and evaluates
 * merge-related telemetry with three detectors:
 *
 *   Isolation Forest
 *   Local Outlier Factor
 *   One-Class SVM-style RBF boundary
 *
 * The C++ design emphasizes explicit domain objects, deterministic random
 * generation, STL algorithms, validation, and a governance decision layer.
 */

using Vector = std::vector<double>;

class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& message)
        : std::runtime_error(message) {}
};

enum class DetectorType {
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM
};

std::string detectorName(DetectorType type) {
    switch (type) {
        case DetectorType::IsolationForest:
            return "Isolation Forest";
        case DetectorType::LocalOutlierFactor:
            return "Local Outlier Factor";
        case DetectorType::OneClassSVM:
            return "One-Class SVM";
    }
    return "Unknown";
}

struct Telemetry {
    int id{};
    Vector features;
    bool knownAnomaly{};
    std::string description;
};

struct DetectionResult {
    DetectorType detector;
    std::vector<double> scores;
    std::vector<bool> predictions;
    double threshold{};
};

void validateVector(const Vector& values, std::size_t expected = 0) {
    if (values.empty()) {
        throw ValidationError("Feature vector cannot be empty.");
    }

    if (expected != 0 && values.size() != expected) {
        throw ValidationError("Feature dimension mismatch.");
    }

    for (double value : values) {
        if (!std::isfinite(value)) {
            throw ValidationError("Features must be finite.");
        }
    }
}

void validateMatrix(const std::vector<Vector>& matrix) {
    if (matrix.size() < 2) {
        throw ValidationError("At least two observations are required.");
    }

    const auto dimensions = matrix.front().size();
    for (const auto& row : matrix) {
        validateVector(row, dimensions);
    }
}

double euclidean(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw ValidationError("Distance vectors have different dimensions.");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

double quantile(std::vector<double> values, double q) {
    if (values.empty() || q < 0.0 || q > 1.0) {
        throw ValidationError("Invalid quantile.");
    }

    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));

    if (lower == upper) {
        return values[lower];
    }

    const double fraction = position - static_cast<double>(lower);
    return values[lower] +
           fraction * (values[upper] - values[lower]);
}

class StandardScaler {
private:
    Vector means_;
    Vector standardDeviations_;

public:
    void fit(const std::vector<Vector>& rows) {
        validateMatrix(rows);

        const std::size_t dimensions = rows.front().size();
        means_.assign(dimensions, 0.0);
        standardDeviations_.assign(dimensions, 1.0);

        for (const auto& row : rows) {
            for (std::size_t column = 0; column < dimensions; ++column) {
                means_[column] += row[column];
            }
        }

        for (double& mean : means_) {
            mean /= static_cast<double>(rows.size());
        }

        for (std::size_t column = 0; column < dimensions; ++column) {
            double variance = 0.0;
            for (const auto& row : rows) {
                const double difference = row[column] - means_[column];
                variance += difference * difference;
            }
            variance /= static_cast<double>(rows.size());
            const double deviation = std::sqrt(variance);
            standardDeviations_[column] =
                deviation > 1e-12 ? deviation : 1.0;
        }
    }

    std::vector<Vector> transform(const std::vector<Vector>& rows) const {
        if (means_.empty()) {
            throw ValidationError("Scaler has not been fitted.");
        }

        std::vector<Vector> result;
        result.reserve(rows.size());

        for (const auto& row : rows) {
            validateVector(row, means_.size());
            Vector transformed(row.size());

            for (std::size_t i = 0; i < row.size(); ++i) {
                transformed[i] =
                    (row[i] - means_[i]) / standardDeviations_[i];
            }

            result.push_back(std::move(transformed));
        }

        return result;
    }
};

double harmonicNumber(int n) {
    if (n <= 1) return 0.0;

    double result = 0.0;
    for (int i = 1; i <= n; ++i) {
        result += 1.0 / static_cast<double>(i);
    }
    return result;
}

double expectedPathLength(int n) {
    if (n <= 1) return 0.0;
    return 2.0 * harmonicNumber(n - 1) -
           2.0 * static_cast<double>(n - 1) / static_cast<double>(n);
}

struct IsolationNode {
    int feature{-1};
    double split{0.0};
    std::unique_ptr<IsolationNode> left;
    std::unique_ptr<IsolationNode> right;
    int size{0};
};

class IsolationTree {
private:
    std::unique_ptr<IsolationNode> root_;
    int maxDepth_;
    std::mt19937& generator_;

    std::unique_ptr<IsolationNode> build(
        const std::vector<Vector>& rows,
        int depth
    ) {
        auto node = std::make_unique<IsolationNode>();
        node->size = static_cast<int>(rows.size());

        if (rows.size() <= 1 || depth >= maxDepth_) {
            return node;
        }

        std::uniform_int_distribution<int> featureDistribution(
            0,
            static_cast<int>(rows.front().size()) - 1
        );

        const int feature = featureDistribution(generator_);

        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();

        for (const auto& row : rows) {
            minimum = std::min(minimum, row[feature]);
            maximum = std::max(maximum, row[feature]);
        }

        if (maximum - minimum <= 1e-12) {
            return node;
        }

        std::uniform_real_distribution<double> splitDistribution(
            minimum,
            maximum
        );

        const double split = splitDistribution(generator_);

        std::vector<Vector> leftRows;
        std::vector<Vector> rightRows;

        for (const auto& row : rows) {
            if (row[feature] < split) {
                leftRows.push_back(row);
            } else {
                rightRows.push_back(row);
            }
        }

        if (leftRows.empty() || rightRows.empty()) {
            return node;
        }

        node->feature = feature;
        node->split = split;
        node->left = build(leftRows, depth + 1);
        node->right = build(rightRows, depth + 1);

        return node;
    }

public:
    IsolationTree(int maxDepth, std::mt19937& generator)
        : maxDepth_(maxDepth), generator_(generator) {}

    void fit(const std::vector<Vector>& rows) {
        if (rows.empty()) {
            throw ValidationError("Cannot train an empty tree.");
        }
        root_ = build(rows, 0);
    }

    double pathLength(const Vector& row) const {
        if (!root_) {
            throw ValidationError("Tree has not been fitted.");
        }

        const IsolationNode* node = root_.get();
        int depth = 0;

        while (
            node->feature >= 0 &&
            node->left &&
            node->right
        ) {
            ++depth;
            node =
                row[node->feature] < node->split
                    ? node->left.get()
                    : node->right.get();
        }

        return static_cast<double>(depth) +
               expectedPathLength(node->size);
    }
};

class IsolationForest {
private:
    int treeCount_;
    int sampleSize_;
    double contamination_;
    std::mt19937 generator_;
    std::vector<IsolationTree> trees_;
    double normalizer_{1.0};

public:
    IsolationForest(
        int treeCount,
        int sampleSize,
        double contamination,
        unsigned seed
    )
        : treeCount_(treeCount),
          sampleSize_(sampleSize),
          contamination_(contamination),
          generator_(seed) {
        if (treeCount_ < 1 || sampleSize_ < 2) {
            throw ValidationError("Invalid forest configuration.");
        }

        if (contamination_ <= 0.0 || contamination_ >= 0.5) {
            throw ValidationError("Invalid contamination.");
        }
    }

    void fit(const std::vector<Vector>& rows) {
        validateMatrix(rows);

        const int actualSample =
            std::min(sampleSize_, static_cast<int>(rows.size()));

        normalizer_ = expectedPathLength(actualSample);

        if (normalizer_ <= 0.0) {
            throw ValidationError("Invalid forest normalizer.");
        }

        trees_.clear();
        trees_.reserve(treeCount_);

        for (int i = 0; i < treeCount_; ++i) {
            std::vector<Vector> sample = rows;

            std::shuffle(sample.begin(), sample.end(), generator_);
            sample.resize(actualSample);

            trees_.emplace_back(
                static_cast<int>(std::ceil(std::log2(actualSample))),
                generator_
            );

            trees_.back().fit(sample);
        }
    }

    std::vector<double> score(const std::vector<Vector>& rows) const {
        if (trees_.empty()) {
            throw ValidationError("Forest has not been fitted.");
        }

        std::vector<double> scores;
        scores.reserve(rows.size());

        for (const auto& row : rows) {
            double totalPath = 0.0;

            for (const auto& tree : trees_) {
                totalPath += tree.pathLength(row);
            }

            const double averagePath =
                totalPath / static_cast<double>(trees_.size());

            scores.push_back(
                std::pow(2.0, -averagePath / normalizer_)
            );
        }

        return scores;
    }

    DetectionResult predict(const std::vector<Vector>& rows) const {
        auto scores = score(rows);
        const double threshold =
            quantile(scores, 1.0 - contamination_);

        std::vector<bool> predictions;
        predictions.reserve(scores.size());

        for (double score : scores) {
            predictions.push_back(score >= threshold);
        }

        return {
            DetectorType::IsolationForest,
            std::move(scores),
            std::move(predictions),
            threshold
        };
    }
};

class LocalOutlierFactor {
private:
    int k_;
    std::vector<Vector> training_;
    std::vector<double> kDistance_;
    std::vector<double> localReachabilityDensity_;

    std::vector<std::pair<int, double>> neighbors(
        int index
    ) const {
        std::vector<std::pair<int, double>> result;

        for (int i = 0; i < static_cast<int>(training_.size()); ++i) {
            if (i != index) {
                result.emplace_back(
                    i,
                    euclidean(training_[index], training_[i])
                );
            }
        }

        std::sort(
            result.begin(),
            result.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            }
        );

        if (result.size() > static_cast<std::size_t>(k_)) {
            result.resize(k_);
        }

        return result;
    }

public:
    explicit LocalOutlierFactor(int k) : k_(k) {
        if (k_ < 2) {
            throw ValidationError("LOF requires k >= 2.");
        }
    }

    void fit(const std::vector<Vector>& rows) {
        validateMatrix(rows);

        if (rows.size() <= static_cast<std::size_t>(k_)) {
            throw ValidationError(
                "LOF requires more rows than k."
            );
        }

        training_ = rows;
        kDistance_.assign(rows.size(), 0.0);

        std::vector<std::vector<std::pair<int, double>>> allNeighbors;

        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            auto current = neighbors(i);
            kDistance_[i] =
                current.empty() ? 0.0 : current.back().second;
            allNeighbors.push_back(std::move(current));
        }

        localReachabilityDensity_.clear();

        for (const auto& list : allNeighbors) {
            double reachabilitySum = 0.0;

            for (const auto& [index, distanceValue] : list) {
                reachabilitySum +=
                    std::max(kDistance_[index], distanceValue);
            }

            localReachabilityDensity_.push_back(
                reachabilitySum > 1e-12
                    ? static_cast<double>(list.size()) /
                      reachabilitySum
                    : std::numeric_limits<double>::infinity()
            );
        }
    }

    DetectionResult predict(
        const std::vector<Vector>& rows,
        double threshold = 1.5
    ) const {
        if (training_.empty()) {
            throw ValidationError("LOF has not been fitted.");
        }

        std::vector<double> scores;

        for (const auto& row : rows) {
            std::vector<std::pair<int, double>> candidates;

            for (int i = 0; i < static_cast<int>(training_.size()); ++i) {
                candidates.emplace_back(
                    i,
                    euclidean(row, training_[i])
                );
            }

            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const auto& a, const auto& b) {
                    return a.second < b.second;
                }
            );

            if (candidates.size() > static_cast<std::size_t>(k_)) {
                candidates.resize(k_);
            }

            double reachability = 0.0;
            double neighborDensity = 0.0;

            for (const auto& [index, distanceValue] : candidates) {
                reachability +=
                    std::max(kDistance_[index], distanceValue);
                neighborDensity +=
                    localReachabilityDensity_[index];
            }

            if (reachability <= 1e-12) {
                scores.push_back(1.0);
                continue;
            }

            const double queryDensity =
                static_cast<double>(candidates.size()) /
                reachability;

            neighborDensity /=
                static_cast<double>(candidates.size());

            scores.push_back(
                neighborDensity /
                std::max(queryDensity, 1e-12)
            );
        }

        std::vector<bool> predictions;

        for (double score : scores) {
            predictions.push_back(score >= threshold);
        }

        return {
            DetectorType::LocalOutlierFactor,
            std::move(scores),
            std::move(predictions),
            threshold
        };
    }
};

class OneClassSVM {
private:
    double gamma_;
    double nu_;
    std::vector<Vector> training_;

    double kernel(const Vector& a, const Vector& b) const {
        double squaredDistance = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            const double difference = a[i] - b[i];
            squaredDistance += difference * difference;
        }

        return std::exp(-gamma_ * squaredDistance);
    }

public:
    OneClassSVM(double gamma, double nu)
        : gamma_(gamma), nu_(nu) {
        if (gamma_ <= 0.0 || nu_ <= 0.0 || nu_ > 0.5) {
            throw ValidationError("Invalid One-Class SVM configuration.");
        }
    }

    void fit(const std::vector<Vector>& rows) {
        validateMatrix(rows);
        training_ = rows;
    }

    DetectionResult predict(const std::vector<Vector>& rows) const {
        if (training_.empty()) {
            throw ValidationError("One-Class SVM has not been fitted.");
        }

        std::vector<double> scores;
        scores.reserve(rows.size());

        for (const auto& row : rows) {
            double similarity = 0.0;

            for (const auto& train : training_) {
                similarity += kernel(row, train);
            }

            similarity /= static_cast<double>(training_.size());
            scores.push_back(1.0 - similarity);
        }

        const double threshold =
            quantile(scores, 1.0 - nu_);

        std::vector<bool> predictions;

        for (double score : scores) {
            predictions.push_back(score >= threshold);
        }

        return {
            DetectorType::OneClassSVM,
            std::move(scores),
            std::move(predictions),
            threshold
        };
    }
};

struct Metrics {
    int truePositive{};
    int trueNegative{};
    int falsePositive{};
    int falseNegative{};
    double precision{};
    double recall{};
    double f1{};
};

Metrics evaluate(
    const std::vector<bool>& actual,
    const std::vector<bool>& predicted
) {
    if (actual.size() != predicted.size()) {
        throw ValidationError("Evaluation vector size mismatch.");
    }

    Metrics result;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] && predicted[i]) ++result.truePositive;
        else if (!actual[i] && !predicted[i]) ++result.trueNegative;
        else if (!actual[i] && predicted[i]) ++result.falsePositive;
        else ++result.falseNegative;
    }

    result.precision =
        result.truePositive + result.falsePositive == 0
            ? 0.0
            : static_cast<double>(result.truePositive) /
              (result.truePositive + result.falsePositive);

    result.recall =
        result.truePositive + result.falseNegative == 0
            ? 0.0
            : static_cast<double>(result.truePositive) /
              (result.truePositive + result.falseNegative);

    result.f1 =
        result.precision + result.recall == 0.0
            ? 0.0
            : 2.0 * result.precision * result.recall /
              (result.precision + result.recall);

    return result;
}

std::vector<Telemetry> createTelemetry() {
    std::mt19937 generator(123);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<Telemetry> telemetry;

    for (int id = 1; id <= 160; ++id) {
        telemetry.push_back({
            id,
            {
                std::max(1.0, 8.0 + 3.0 * normal(generator)),
                std::max(1.0, 120.0 + 45.0 * normal(generator)),
                std::max(0.0, 55.0 + 20.0 * normal(generator)),
                std::max(0.2, 18.0 + 7.0 * normal(generator)),
                std::max(0.0, 0.4 + 0.8 * normal(generator)),
                std::max(0.0, 1.0 + 0.7 * normal(generator))
            },
            false,
            "normal repository operation"
        });
    }

    const std::vector<std::tuple<Vector, std::string>> anomalies = {
        {{90, 2500, 1700, 1, 14, 12}, "large security-sensitive change"},
        {{2, 4, 1, 260, 0, 0}, "extreme review latency"},
        {{60, 1800, 90, 2, 17, 10}, "rapid high-risk modification"},
        {{3, 7, 2, 0.2, 18, 1}, "abnormal status-check failures"},
        {{75, 3200, 15, 3, 0, 13}, "large asymmetric change"}
    };

    int id = 161;
    for (const auto& [features, description] : anomalies) {
        telemetry.push_back({
            id++,
            features,
            true,
            description
        });
    }

    return telemetry;
}

void printResult(
    const DetectionResult& result,
    const std::vector<Telemetry>& telemetry
) {
    std::vector<bool> actual;

    for (const auto& item : telemetry) {
        actual.push_back(item.knownAnomaly);
    }

    const Metrics metrics =
        evaluate(actual, result.predictions);

    std::cout << "\n" << detectorName(result.detector) << "\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Threshold: " << result.threshold << "\n";
    std::cout << std::setprecision(3)
              << "Precision: " << metrics.precision
              << " Recall: " << metrics.recall
              << " F1: " << metrics.f1 << "\n";

    std::vector<std::pair<double, int>> ranking;

    for (std::size_t i = 0; i < result.scores.size(); ++i) {
        ranking.emplace_back(result.scores[i], static_cast<int>(i));
    }

    std::sort(
        ranking.rbegin(),
        ranking.rend()
    );

    std::cout << "\nHighest scoring events:\n";

    for (std::size_t i = 0; i < std::min<std::size_t>(7, ranking.size()); ++i) {
        const auto [score, index] = ranking[i];
        const auto& event = telemetry[index];

        std::cout
            << "  event=" << std::setw(3) << event.id
            << " score=" << std::setprecision(4) << score
            << " "
            << (event.knownAnomaly ? "KNOWN " : "NORMAL")
            << " "
            << event.description
            << "\n";
    }
}

int main() {
    try {
        const auto telemetry = createTelemetry();

        std::vector<Vector> raw;

        for (const auto& event : telemetry) {
            raw.push_back(event.features);
        }

        StandardScaler scaler;
        scaler.fit(raw);
        const auto features = scaler.transform(raw);

        IsolationForest isolationForest(
            100,
            64,
            0.06,
            42
        );
        isolationForest.fit(features);

        LocalOutlierFactor lof(10);
        lof.fit(features);

        OneClassSVM svm(0.18, 0.08);
        svm.fit(features);

        const auto forestResult =
            isolationForest.predict(features);

        const auto lofResult =
            lof.predict(features, 1.45);

        const auto svmResult =
            svm.predict(features);

        const std::vector<DetectionResult> results = {
            forestResult,
            lofResult,
            svmResult
        };

        std::cout << "REPOSITORY TELEMETRY ANOMALY ENGINE\n";
        std::cout << std::string(72, '=') << "\n";
        std::cout << "Observations: " << telemetry.size() << "\n";

        for (const auto& result : results) {
            printResult(result, telemetry);
        }

        std::vector<bool> consensus(telemetry.size(), false);

        for (std::size_t i = 0; i < telemetry.size(); ++i) {
            int votes = 0;

            for (const auto& result : results) {
                if (result.predictions[i]) {
                    ++votes;
                }
            }

            consensus[i] = votes >= 2;
        }

        std::vector<bool> actual;

        for (const auto& event : telemetry) {
            actual.push_back(event.knownAnomaly);
        }

        const Metrics consensusMetrics =
            evaluate(actual, consensus);

        std::cout << "\nTwo-of-three detector consensus\n";
        std::cout << std::string(72, '-') << "\n";
        std::cout << std::setprecision(3)
                  << "Precision: " << consensusMetrics.precision
                  << " Recall: " << consensusMetrics.recall
                  << " F1: " << consensusMetrics.f1
                  << "\n";

        /*
         * Security and production controls:
         * - avoid logging sensitive repository content as raw feature data
         * - authenticate telemetry producers
         * - validate feature schema before model execution
         * - version models and thresholds
         * - monitor distribution drift
         * - separate anomaly detection from authorization decisions
         */
        if (results.size() != 3) {
            throw ValidationError("Detector set is incomplete.");
        }

        for (const auto& result : results) {
            if (result.scores.size() != telemetry.size()) {
                throw ValidationError("Score cardinality mismatch.");
            }
        }

        std::cout << "\nValidation completed successfully.\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
