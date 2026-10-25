#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Point = std::vector<double>;

enum class Initialization {
    Random,
    KMeansPlusPlus
};

struct ClusterResult {
    std::vector<Point> centroids;
    std::vector<std::size_t> labels;
    double inertia = std::numeric_limits<double>::infinity();
    int iterations = 0;
    bool converged = false;
};

class KMeans {
public:
    KMeans(std::size_t k, Initialization initialization,
           int maxIterations = 300, double tolerance = 1e-6,
           std::uint32_t seed = 42, int restarts = 5)
        : k_(k), initialization_(initialization),
          maxIterations_(maxIterations), tolerance_(tolerance),
          seed_(seed), restarts_(restarts) {
        if (k_ == 0) throw std::invalid_argument("K must be positive.");
        if (maxIterations_ < 1 || restarts_ < 1)
            throw std::invalid_argument("Iteration and restart counts must be positive.");
        if (!std::isfinite(tolerance_) || tolerance_ < 0)
            throw std::invalid_argument("Tolerance must be finite and non-negative.");
    }

    ClusterResult fit(const std::vector<Point>& input) const {
        validate(input);

        if (k_ > input.size())
            throw std::invalid_argument("K cannot exceed the sample count.");

        std::mt19937 master(seed_);
        ClusterResult best;

        for (int attempt = 0; attempt < restarts_; ++attempt) {
            std::mt19937 engine(master());
            ClusterResult candidate = fitOnce(input, engine);

            if (candidate.inertia < best.inertia) {
                best = std::move(candidate);
            }
        }

        return best;
    }

    std::vector<std::size_t> predict(
        const std::vector<Point>& points,
        const std::vector<Point>& centroids) const {
        validate(points);

        if (centroids.size() != k_)
            throw std::invalid_argument("Unexpected centroid count.");

        const std::size_t dimension = centroids.front().size();
        for (const auto& point : points) {
            if (point.size() != dimension)
                throw std::invalid_argument("Prediction dimensions do not match.");
        }

        std::vector<std::size_t> labels;
        labels.reserve(points.size());

        for (const auto& point : points) {
            labels.push_back(nearest(point, centroids));
        }

        return labels;
    }

private:
    std::size_t k_;
    Initialization initialization_;
    int maxIterations_;
    double tolerance_;
    std::uint32_t seed_;
    int restarts_;

    static void validate(const std::vector<Point>& points) {
        if (points.empty()) throw std::invalid_argument("Dataset is empty.");
        if (points.front().empty()) throw std::invalid_argument("Points have no features.");

        const std::size_t dimension = points.front().size();

        for (const auto& point : points) {
            if (point.size() != dimension)
                throw std::invalid_argument("Inconsistent feature dimensions.");

            for (double value : point) {
                if (!std::isfinite(value))
                    throw std::invalid_argument("Features must be finite.");
            }
        }
    }

    static double squaredDistance(const Point& a, const Point& b) {
        double distance = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            const double difference = a[i] - b[i];
            distance += difference * difference;
        }

        return distance;
    }

    static std::size_t nearest(const Point& point,
                               const std::vector<Point>& centroids) {
        std::size_t bestIndex = 0;
        double bestDistance = squaredDistance(point, centroids[0]);

        for (std::size_t i = 1; i < centroids.size(); ++i) {
            const double distance = squaredDistance(point, centroids[i]);

            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    std::vector<Point> initialize(const std::vector<Point>& points,
                                  std::mt19937& engine) const {
        std::vector<Point> centers;
        centers.reserve(k_);

        if (initialization_ == Initialization::Random) {
            std::vector<std::size_t> indices(points.size());
            for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;

            std::shuffle(indices.begin(), indices.end(), engine);

            for (std::size_t i = 0; i < k_; ++i)
                centers.push_back(points[indices[i]]);

            return centers;
        }

        std::uniform_int_distribution<std::size_t> first(0, points.size() - 1);
        centers.push_back(points[first(engine)]);

        while (centers.size() < k_) {
            std::vector<double> weights;
            weights.reserve(points.size());
            double total = 0.0;

            for (const auto& point : points) {
                double nearestDistance = std::numeric_limits<double>::infinity();

                for (const auto& center : centers) {
                    nearestDistance = std::min(
                        nearestDistance, squaredDistance(point, center));
                }

                weights.push_back(nearestDistance);
                total += nearestDistance;
            }

            if (total == 0.0) {
                // Duplicate-valued samples cannot all become distinct centers.
                std::uniform_int_distribution<std::size_t> choose(0, points.size() - 1);
                centers.push_back(points[choose(engine)]);
                continue;
            }

            std::discrete_distribution<std::size_t> choose(
                weights.begin(), weights.end());
            centers.push_back(points[choose(engine)]);
        }

        return centers;
    }

    ClusterResult fitOnce(const std::vector<Point>& points,
                          std::mt19937& engine) const {
        ClusterResult result;
        result.centroids = initialize(points, engine);
        result.labels.assign(points.size(), k_); // Invalid initial label.

        const std::size_t dimension = points.front().size();

        for (int iteration = 1; iteration <= maxIterations_; ++iteration) {
            std::vector<std::size_t> labels(points.size());
            std::vector<Point> sums(k_, Point(dimension, 0.0));
            std::vector<std::size_t> counts(k_, 0);

            for (std::size_t i = 0; i < points.size(); ++i) {
                const std::size_t cluster = nearest(points[i], result.centroids);
                labels[i] = cluster;
                ++counts[cluster];

                for (std::size_t feature = 0; feature < dimension; ++feature)
                    sums[cluster][feature] += points[i][feature];
            }

            std::vector<Point> nextCenters(k_, Point(dimension, 0.0));

            for (std::size_t cluster = 0; cluster < k_; ++cluster) {
                if (counts[cluster] == 0) {
                    // Empty-cluster recovery uses the point farthest from its
                    // nearest current centroid.
                    auto farthest = std::max_element(
                        points.begin(), points.end(),
                        [&](const Point& a, const Point& b) {
                            return squaredDistance(a, result.centroids[
                                nearest(a, result.centroids)]) <
                                   squaredDistance(b, result.centroids[
                                nearest(b, result.centroids)]);
                        });

                    nextCenters[cluster] = *farthest;
                } else {
                    for (std::size_t feature = 0; feature < dimension; ++feature)
                        nextCenters[cluster][feature] =
                            sums[cluster][feature] / counts[cluster];
                }
            }

            double maxShift = 0.0;
            for (std::size_t cluster = 0; cluster < k_; ++cluster) {
                maxShift = std::max(
                    maxShift,
                    squaredDistance(result.centroids[cluster], nextCenters[cluster]));
            }

            const bool stable = labels == result.labels;
            result.labels = std::move(labels);
            result.centroids = std::move(nextCenters);
            result.iterations = iteration;

            if (stable || maxShift <= tolerance_ * tolerance_) {
                result.converged = true;
                break;
            }
        }

        // Recompute labels against the final centroids before measuring inertia.
        result.inertia = 0.0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            result.labels[i] = nearest(points[i], result.centroids);
            result.inertia += squaredDistance(
                points[i], result.centroids[result.labels[i]]);
        }

        return result;
    }
};

int main() {
    try {
        const std::vector<Point> customers = {
            {1.0, 1.2}, {1.3, 0.8}, {0.7, 1.1}, {1.2, 1.5},
            {5.0, 5.2}, {5.4, 4.8}, {4.7, 5.1}, {5.2, 5.5},
            {9.0, 1.0}, {9.4, 1.4}, {8.6, 0.7}, {9.2, 0.6}
        };

        KMeans model(3, Initialization::KMeansPlusPlus, 300, 1e-6, 42, 10);
        const ClusterResult result = model.fit(customers);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Inertia: " << result.inertia << '\n';
        std::cout << "Iterations: " << result.iterations << '\n';
        std::cout << "Converged: " << std::boolalpha << result.converged << '\n';

        for (std::size_t c = 0; c < result.centroids.size(); ++c) {
            std::cout << "Centroid " << c << ": ";
            for (double value : result.centroids[c]) std::cout << value << ' ';
            std::cout << '\n';
        }

        std::vector<std::size_t> sizes(3, 0);
        for (std::size_t label : result.labels) ++sizes[label];

        std::cout << "Cluster sizes: ";
        for (std::size_t size : sizes) std::cout << size << ' ';
        std::cout << '\n';

        // The elbow table exposes the trade-off between K and within-cluster error.
        std::cout << "\nElbow candidates:\n";
        for (std::size_t k = 1; k <= 6; ++k) {
            KMeans candidate(k, Initialization::KMeansPlusPlus, 300, 1e-6, 42, 5);
            const auto evaluated = candidate.fit(customers);
            std::cout << "K=" << k << ", inertia=" << evaluated.inertia << '\n';
        }

        if (result.labels.size() != customers.size() ||
            !std::isfinite(result.inertia)) {
            throw std::runtime_error("Clustering invariant failed.");
        }
    } catch (const std::exception& error) {
        std::cerr << "Clustering error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
