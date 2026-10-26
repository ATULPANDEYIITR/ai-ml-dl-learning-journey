#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Point {
    int id;
    double x;
    double y;
    std::string label;

    double distanceTo(const Point& other) const {
        return std::hypot(x - other.x, y - other.y);
    }
};

enum class Linkage {
    Single,
    Complete,
    Average
};

struct Cluster {
    std::vector<Point> points;
};

struct DBSCANResult {
    std::unordered_map<int, int> labels;
    std::set<int> corePoints;
    std::set<int> noisePoints;
};

class ClusteringEngine {
public:
    static void validate(const std::vector<Point>& points) {
        if (points.empty()) {
            throw std::invalid_argument("Dataset cannot be empty.");
        }

        std::set<int> ids;

        for (const auto& point : points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                throw std::invalid_argument("Point coordinates must be finite.");
            }

            if (!ids.insert(point.id).second) {
                throw std::invalid_argument("Point IDs must be unique.");
            }
        }
    }

    static double linkageDistance(
        const Cluster& a,
        const Cluster& b,
        Linkage linkage
    ) {
        double minDistance = std::numeric_limits<double>::infinity();
        double maxDistance = 0.0;
        double total = 0.0;
        std::size_t count = 0;

        for (const auto& first : a.points) {
            for (const auto& second : b.points) {
                const double distance = first.distanceTo(second);

                minDistance = std::min(minDistance, distance);
                maxDistance = std::max(maxDistance, distance);
                total += distance;
                ++count;
            }
        }

        if (count == 0) {
            throw std::logic_error("Cannot calculate distance between empty clusters.");
        }

        switch (linkage) {
            case Linkage::Single:
                return minDistance;
            case Linkage::Complete:
                return maxDistance;
            case Linkage::Average:
                return total / static_cast<double>(count);
        }

        throw std::logic_error("Unsupported linkage.");
    }

    static std::vector<Cluster> hierarchical(
        const std::vector<Point>& points,
        std::size_t targetClusters,
        Linkage linkage
    ) {
        validate(points);

        if (targetClusters == 0 || targetClusters > points.size()) {
            throw std::invalid_argument("Invalid target cluster count.");
        }

        std::vector<Cluster> clusters;
        clusters.reserve(points.size());

        for (const auto& point : points) {
            clusters.push_back(Cluster{{point}});
        }

        while (clusters.size() > targetClusters) {
            std::size_t bestI = 0;
            std::size_t bestJ = 1;
            double bestDistance = std::numeric_limits<double>::infinity();

            for (std::size_t i = 0; i < clusters.size(); ++i) {
                for (std::size_t j = i + 1; j < clusters.size(); ++j) {
                    const double distance =
                        linkageDistance(clusters[i], clusters[j], linkage);

                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }

            clusters[bestI].points.insert(
                clusters[bestI].points.end(),
                clusters[bestJ].points.begin(),
                clusters[bestJ].points.end()
            );

            clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(bestJ));
        }

        return clusters;
    }

    static DBSCANResult dbscan(
        const std::vector<Point>& points,
        double eps,
        std::size_t minSamples
    ) {
        validate(points);

        if (!(eps > 0.0) || !std::isfinite(eps)) {
            throw std::invalid_argument("eps must be positive and finite.");
        }

        if (minSamples == 0) {
            throw std::invalid_argument("minSamples must be positive.");
        }

        std::unordered_map<int, std::vector<int>> neighborhoods;

        for (const auto& point : points) {
            for (const auto& candidate : points) {
                if (point.distanceTo(candidate) <= eps) {
                    neighborhoods[point.id].push_back(candidate.id);
                }
            }
        }

        DBSCANResult result;
        std::set<int> visited;

        for (const auto& point : points) {
            if (neighborhoods[point.id].size() >= minSamples) {
                result.corePoints.insert(point.id);
            }
        }

        int clusterId = 0;

        for (const auto& point : points) {
            if (visited.contains(point.id)) {
                continue;
            }

            visited.insert(point.id);

            const auto& neighbors = neighborhoods[point.id];

            if (neighbors.size() < minSamples) {
                result.labels[point.id] = -1;
                result.noisePoints.insert(point.id);
                continue;
            }

            result.labels[point.id] = clusterId;

            std::vector<int> queue = neighbors;
            std::size_t cursor = 0;

            while (cursor < queue.size()) {
                const int candidateId = queue[cursor++];

                if (!visited.contains(candidateId)) {
                    visited.insert(candidateId);

                    if (neighborhoods[candidateId].size() >= minSamples) {
                        for (int neighborId : neighborhoods[candidateId]) {
                            if (std::find(
                                    queue.begin() + static_cast<std::ptrdiff_t>(cursor),
                                    queue.end(),
                                    neighborId
                                ) == queue.end()) {
                                queue.push_back(neighborId);
                            }
                        }
                    }
                }

                auto existing = result.labels.find(candidateId);

                if (existing == result.labels.end() || existing->second == -1) {
                    result.labels[candidateId] = clusterId;
                    result.noisePoints.erase(candidateId);
                }
            }

            ++clusterId;
        }

        return result;
    }

    static double silhouette(
        const std::vector<Point>& points,
        const std::unordered_map<int, int>& labels
    ) {
        validate(points);

        std::map<int, std::vector<const Point*>> groups;

        for (const auto& point : points) {
            auto it = labels.find(point.id);

            if (it != labels.end() && it->second != -1) {
                groups[it->second].push_back(&point);
            }
        }

        if (groups.size() < 2) {
            return 0.0;
        }

        double totalScore = 0.0;
        std::size_t scoreCount = 0;

        for (const auto& point : points) {
            auto labelIt = labels.find(point.id);

            if (labelIt == labels.end() || labelIt->second == -1) {
                continue;
            }

            const int ownLabel = labelIt->second;
            const auto& ownCluster = groups.at(ownLabel);

            double a = 0.0;

            if (ownCluster.size() > 1) {
                double sum = 0.0;

                for (const Point* other : ownCluster) {
                    if (other->id != point.id) {
                        sum += point.distanceTo(*other);
                    }
                }

                a = sum / static_cast<double>(ownCluster.size() - 1);
            }

            double b = std::numeric_limits<double>::infinity();

            for (const auto& [label, cluster] : groups) {
                if (label == ownLabel) {
                    continue;
                }

                double sum = 0.0;

                for (const Point* other : cluster) {
                    sum += point.distanceTo(*other);
                }

                b = std::min(
                    b,
                    sum / static_cast<double>(cluster.size())
                );
            }

            const double denominator = std::max(a, b);
            const double score =
                denominator == 0.0 ? 0.0 : (b - a) / denominator;

            totalScore += score;
            ++scoreCount;
        }

        return scoreCount == 0
            ? 0.0
            : totalScore / static_cast<double>(scoreCount);
    }
};

static void printClusters(
    const std::string& title,
    const std::vector<Cluster>& clusters
) {
    std::cout << "\n" << title << "\n";

    for (std::size_t i = 0; i < clusters.size(); ++i) {
        std::vector<int> ids;

        for (const auto& point : clusters[i].points) {
            ids.push_back(point.id);
        }

        std::sort(ids.begin(), ids.end());

        std::cout << "  Cluster " << i << ": ";

        for (int id : ids) {
            std::cout << id << " ";
        }

        std::cout << "\n";
    }
}

static std::vector<Point> makeDataset() {
    std::mt19937 generator(42);
    std::normal_distribution<double> noise(0.0, 0.35);

    std::vector<Point> points;
    int id = 1;

    for (int i = 0; i < 10; ++i) {
        points.push_back({
            id++,
            5.0 + noise(generator),
            5.0 + noise(generator),
            "cluster-A"
        });
    }

    for (int i = 0; i < 10; ++i) {
        points.push_back({
            id++,
            15.0 + noise(generator),
            15.0 + noise(generator),
            "cluster-B"
        });
    }

    points.push_back({id++, 30.0, 2.0, "outlier"});
    points.push_back({id++, 25.0, 25.0, "outlier"});

    return points;
}

static void runValidationTests() {
    const std::vector<Point> points = {
        {1, 0.0, 0.0, ""},
        {2, 0.2, 0.1, ""},
        {3, 10.0, 10.0, ""},
        {4, 10.1, 10.2, ""}
    };

    const auto hierarchical =
        ClusteringEngine::hierarchical(points, 2, Linkage::Average);

    if (hierarchical.size() != 2) {
        throw std::runtime_error("Hierarchical clustering test failed.");
    }

    const auto density =
        ClusteringEngine::dbscan(points, 0.5, 2);

    if (density.labels.at(1) != density.labels.at(2)) {
        throw std::runtime_error("DBSCAN first cluster test failed.");
    }

    std::vector<Point> invalid = {
        {1, 0.0, 0.0, ""},
        {1, 1.0, 1.0, ""}
    };

    try {
        ClusteringEngine::validate(invalid);
        throw std::runtime_error("Duplicate-ID validation failed.");
    } catch (const std::invalid_argument&) {
        // Expected failure.
    }
}

int main() {
    try {
        runValidationTests();

        const auto points = makeDataset();

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Advanced Clustering Governance Case Study\n";
        std::cout << "Observations: " << points.size() << "\n";

        const auto hierarchical =
            ClusteringEngine::hierarchical(
                points,
                3,
                Linkage::Average
            );

        printClusters(
            "Agglomerative hierarchical clustering",
            hierarchical
        );

        std::unordered_map<int, int> hierarchicalLabels;

        for (std::size_t clusterId = 0;
             clusterId < hierarchical.size();
             ++clusterId) {
            for (const auto& point : hierarchical[clusterId].points) {
                hierarchicalLabels[point.id] =
                    static_cast<int>(clusterId);
            }
        }

        std::cout
            << "Hierarchical silhouette: "
            << ClusteringEngine::silhouette(points, hierarchicalLabels)
            << "\n";

        const auto density =
            ClusteringEngine::dbscan(points, 1.0, 4);

        std::map<int, std::vector<int>> grouped;

        for (const auto& [pointId, clusterId] : density.labels) {
            grouped[clusterId].push_back(pointId);
        }

        std::cout << "\nDBSCAN\n";

        for (auto& [clusterId, ids] : grouped) {
            std::sort(ids.begin(), ids.end());

            std::cout
                << "  "
                << (clusterId == -1
                        ? "Noise"
                        : "Cluster " + std::to_string(clusterId))
                << ": ";

            for (int id : ids) {
                std::cout << id << " ";
            }

            std::cout << "\n";
        }

        std::cout << "Core points: ";

        for (int id : density.corePoints) {
            std::cout << id << " ";
        }

        std::cout << "\nNoise points: ";

        for (int id : density.noisePoints) {
            std::cout << id << " ";
        }

        std::cout << "\nDBSCAN silhouette: "
                  << ClusteringEngine::silhouette(
                         points,
                         density.labels
                     )
                  << "\n";

        std::cout
            << "\nPerformance note: the direct implementation uses "
               "pairwise distance calculations and is intended for "
               "moderate datasets. Spatial indexes or approximate "
               "neighbor search are preferable for large datasets.\n";

    } catch (const std::exception& error) {
        std::cerr
            << "Clustering failed: "
            << error.what()
            << "\n";
        return 1;
    }

    return 0;
}
