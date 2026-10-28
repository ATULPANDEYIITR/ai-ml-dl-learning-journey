#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

enum class ReductionMethod {
    PCA,
    TSNE,
    UMAP_STYLE
};

struct ReductionConfig {
    std::size_t dimensions = 2;
    std::size_t iterations = 300;
    std::size_t neighbors = 8;
    double perplexity = 8.0;
    double learningRate = 50.0;
    double minDist = 0.1;
    unsigned seed = 42;
};

struct Point2D {
    double x{};
    double y{};
};

class DatasetValidator {
public:
    static void validate(const Matrix& data) {
        if (data.empty()) {
            throw std::invalid_argument("Dataset is empty");
        }

        if (data.front().size() < 2) {
            throw std::invalid_argument(
                "At least two source dimensions are required"
            );
        }

        const auto width = data.front().size();

        for (const auto& row : data) {
            if (row.size() != width) {
                throw std::invalid_argument(
                    "Rows have inconsistent dimensions"
                );
            }

            for (double value : row) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "Dataset contains a non-finite value"
                    );
                }
            }
        }
    }
};

class Graph {
public:
    explicit Graph(std::size_t size)
        : adjacency_(size) {}

    void addEdge(std::size_t source, std::size_t target, double weight) {
        adjacency_[source].push_back({target, weight});
    }

    const auto& neighbors(std::size_t node) const {
        return adjacency_[node];
    }

    std::size_t size() const {
        return adjacency_.size();
    }

private:
    std::vector<std::vector<std::pair<std::size_t, double>>> adjacency_;
};

class ManifoldEmbeddingEngine {
public:
    explicit ManifoldEmbeddingEngine(ReductionConfig config)
        : config_(std::move(config)),
          rng_(config_.seed) {}

    std::vector<Point2D> run(
        const Matrix& input,
        ReductionMethod method
    ) {
        DatasetValidator::validate(input);

        if (config_.dimensions != 2) {
            throw std::invalid_argument(
                "This case study produces two-dimensional embeddings"
            );
        }

        switch (method) {
            case ReductionMethod::PCA:
                return pca(input);

            case ReductionMethod::TSNE:
                return tsne(input);

            case ReductionMethod::UMAP_STYLE:
                return umapStyle(input);
        }

        throw std::invalid_argument("Unknown reduction method");
    }

private:
    ReductionConfig config_;
    std::mt19937 rng_;

    static Matrix standardize(const Matrix& input) {
        const std::size_t n = input.size();
        const std::size_t d = input.front().size();

        Vector means(d, 0.0);
        Vector scales(d, 0.0);

        for (const auto& row : input) {
            for (std::size_t j = 0; j < d; ++j) {
                means[j] += row[j];
            }
        }

        for (double& mean : means) {
            mean /= static_cast<double>(n);
        }

        for (const auto& row : input) {
            for (std::size_t j = 0; j < d; ++j) {
                const double delta = row[j] - means[j];
                scales[j] += delta * delta;
            }
        }

        for (double& scale : scales) {
            scale = std::sqrt(
                scale / static_cast<double>(std::max<std::size_t>(1, n - 1))
            );

            if (scale < 1e-12) {
                scale = 1.0;
            }
        }

        Matrix result = input;

        for (auto& row : result) {
            for (std::size_t j = 0; j < d; ++j) {
                row[j] = (row[j] - means[j]) / scales[j];
            }
        }

        return result;
    }

    static double distanceSquared(
        const Vector& a,
        const Vector& b
    ) {
        double result = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            const double delta = a[i] - b[i];
            result += delta * delta;
        }

        return result;
    }

    static Matrix distances(const Matrix& input) {
        const std::size_t n = input.size();
        Matrix result(n, Vector(n, 0.0));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                result[i][j] = distanceSquared(input[i], input[j]);
                result[j][i] = result[i][j];
            }
        }

        return result;
    }

    static Vector matrixVector(
        const Matrix& matrix,
        const Vector& vector
    ) {
        Vector result(matrix.size(), 0.0);

        for (std::size_t i = 0; i < matrix.size(); ++i) {
            for (std::size_t j = 0; j < vector.size(); ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    static double norm(const Vector& vector) {
        return std::sqrt(
            std::inner_product(
                vector.begin(),
                vector.end(),
                vector.begin(),
                0.0
            )
        );
    }

    Vector normalized(Vector vector) const {
        const double length = norm(vector);

        if (length < 1e-12) {
            throw std::runtime_error(
                "Cannot normalize a zero vector"
            );
        }

        for (double& value : vector) {
            value /= length;
        }

        return vector;
    }

    std::pair<double, Vector> powerIteration(
        Matrix matrix
    ) const {
        std::normal_distribution<double> distribution(-1.0, 1.0);

        Vector vector(matrix.size());

        for (double& value : vector) {
            value = distribution(rng_);
        }

        vector = normalized(vector);

        for (std::size_t iteration = 0; iteration < 200; ++iteration) {
            Vector candidate =
                normalized(matrixVector(matrix, vector));

            double change = 0.0;

            for (std::size_t i = 0; i < vector.size(); ++i) {
                change = std::max(
                    change,
                    std::abs(candidate[i] - vector[i])
                );
            }

            vector = candidate;

            if (change < 1e-10) {
                break;
            }
        }

        const Vector product = matrixVector(matrix, vector);

        const double eigenvalue =
            std::inner_product(
                vector.begin(),
                vector.end(),
                product.begin(),
                0.0
            );

        return {eigenvalue, vector};
    }

    static void deflate(
        Matrix& matrix,
        double eigenvalue,
        const Vector& vector
    ) {
        for (std::size_t i = 0; i < matrix.size(); ++i) {
            for (std::size_t j = 0; j < matrix.size(); ++j) {
                matrix[i][j] -=
                    eigenvalue * vector[i] * vector[j];
            }
        }
    }

    std::vector<Point2D> pca(const Matrix& input) const {
        Matrix data = standardize(input);

        const std::size_t n = data.size();
        const std::size_t d = data.front().size();

        Matrix covariance(d, Vector(d, 0.0));

        for (const auto& row : data) {
            for (std::size_t i = 0; i < d; ++i) {
                for (std::size_t j = 0; j < d; ++j) {
                    covariance[i][j] += row[i] * row[j];
                }
            }
        }

        const double divisor =
            static_cast<double>(std::max<std::size_t>(1, n - 1));

        for (auto& row : covariance) {
            for (double& value : row) {
                value /= divisor;
            }
        }

        std::vector<Vector> components;

        for (int component = 0; component < 2; ++component) {
            auto [eigenvalue, vector] =
                powerIteration(covariance);

            components.push_back(vector);
            deflate(covariance, eigenvalue, vector);
        }

        std::vector<Point2D> result(n);

        for (std::size_t i = 0; i < n; ++i) {
            result[i].x =
                std::inner_product(
                    data[i].begin(),
                    data[i].end(),
                    components[0].begin(),
                    0.0
                );

            result[i].y =
                std::inner_product(
                    data[i].begin(),
                    data[i].end(),
                    components[1].begin(),
                    0.0
                );
        }

        return result;
    }

    Graph buildGraph(const Matrix& input) const {
        const std::size_t n = input.size();

        if (config_.neighbors < 1 ||
            config_.neighbors >= n) {
            throw std::invalid_argument(
                "Invalid neighbor count"
            );
        }

        const Matrix d = distances(input);
        Graph graph(n);

        for (std::size_t i = 0; i < n; ++i) {
            std::vector<std::pair<double, std::size_t>> ordered;

            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) {
                    ordered.push_back({d[i][j], j});
                }
            }

            std::partial_sort(
                ordered.begin(),
                ordered.begin() + config_.neighbors,
                ordered.end()
            );

            const std::size_t limit = config_.neighbors;

            double sigma = 0.0;

            for (std::size_t k = 0; k < limit; ++k) {
                sigma += std::sqrt(ordered[k].first);
            }

            sigma /= static_cast<double>(limit);
            sigma = std::max(sigma, 1e-9);

            for (std::size_t k = 0; k < limit; ++k) {
                const double distance =
                    std::sqrt(ordered[k].first);

                const double weight =
                    std::exp(-distance / sigma);

                graph.addEdge(
                    i,
                    ordered[k].second,
                    weight
                );
            }
        }

        return graph;
    }

    std::vector<Point2D> initializeEmbedding(
        std::size_t n
    ) const {
        std::uniform_real_distribution<double> distribution(
            -1.0,
            1.0
        );

        std::vector<Point2D> result(n);

        for (auto& point : result) {
            point.x = distribution(rng_);
            point.y = distribution(rng_);
        }

        return result;
    }

    std::vector<Point2D> umapStyle(
        const Matrix& input
    ) const {
        const Matrix standardized = standardize(input);
        const Graph graph = buildGraph(standardized);

        auto embedding =
            initializeEmbedding(standardized.size());

        for (std::size_t iteration = 0;
             iteration < config_.iterations;
             ++iteration) {

            std::vector<Point2D> gradient(
                embedding.size()
            );

            for (std::size_t i = 0;
                 i < graph.size();
                 ++i) {

                for (const auto& [j, weight] :
                     graph.neighbors(i)) {

                    const double dx =
                        embedding[i].x - embedding[j].x;
                    const double dy =
                        embedding[i].y - embedding[j].y;

                    const double distance =
                        std::sqrt(dx * dx + dy * dy) + 1e-9;

                    const double attraction =
                        weight *
                        (distance - config_.minDist);

                    gradient[i].x +=
                        attraction * dx / distance;

                    gradient[i].y +=
                        attraction * dy / distance;
                }
            }

            const double rate =
                config_.learningRate /
                (1.0 + static_cast<double>(iteration) / 100.0);

            for (std::size_t i = 0;
                 i < embedding.size();
                 ++i) {

                embedding[i].x -=
                    rate * 0.01 * gradient[i].x;

                embedding[i].y -=
                    rate * 0.01 * gradient[i].y;
            }
        }

        return embedding;
    }

    std::vector<Point2D> tsne(
        const Matrix& input
    ) const {
        const Matrix data = standardize(input);
        const std::size_t n = data.size();

        if (config_.perplexity <= 1.0 ||
            config_.perplexity >= static_cast<double>(n)) {
            throw std::invalid_argument(
                "Perplexity must be between 1 and sample count"
            );
        }

        const Matrix d = distances(data);

        Matrix probabilities(n, Vector(n, 0.0));

        for (std::size_t i = 0; i < n; ++i) {
            double beta = 1.0;

            for (std::size_t iteration = 0;
                 iteration < 50;
                 ++iteration) {

                double total = 0.0;

                for (std::size_t j = 0; j < n; ++j) {
                    if (i == j) {
                        probabilities[i][j] = 0.0;
                    } else {
                        probabilities[i][j] =
                            std::exp(-beta * d[i][j]);
                        total += probabilities[i][j];
                    }
                }

                if (total < 1e-300) {
                    beta *= 0.5;
                    continue;
                }

                double entropy = 0.0;

                for (std::size_t j = 0; j < n; ++j) {
                    probabilities[i][j] /= total;

                    if (probabilities[i][j] > 0.0) {
                        entropy -=
                            probabilities[i][j] *
                            std::log(probabilities[i][j]);
                    }
                }

                const double difference =
                    entropy -
                    std::log(config_.perplexity);

                if (std::abs(difference) < 1e-5) {
                    break;
                }

                beta *= difference > 0.0 ? 2.0 : 0.5;
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                probabilities[i][j] =
                    (probabilities[i][j] +
                     probabilities[j][i]) /
                    (2.0 * n);
            }
        }

        auto embedding = initializeEmbedding(n);
        std::vector<Point2D> velocity(n);

        for (std::size_t iteration = 0;
             iteration < config_.iterations;
             ++iteration) {

            Matrix q(n, Vector(n, 0.0));
            double totalQ = 0.0;

            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i + 1;
                     j < n;
                     ++j) {

                    const double dx =
                        embedding[i].x - embedding[j].x;
                    const double dy =
                        embedding[i].y - embedding[j].y;

                    const double value =
                        1.0 / (1.0 + dx * dx + dy * dy);

                    q[i][j] = value;
                    q[j][i] = value;
                    totalQ += 2.0 * value;
                }
            }

            std::vector<Point2D> gradient(n);

            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    if (i == j) {
                        continue;
                    }

                    const double normalizedQ =
                        q[i][j] / std::max(totalQ, 1e-12);

                    const double multiplier =
                        4.0 *
                        (probabilities[i][j] - normalizedQ) *
                        q[i][j];

                    gradient[i].x +=
                        multiplier *
                        (embedding[i].x - embedding[j].x);

                    gradient[i].y +=
                        multiplier *
                        (embedding[i].y - embedding[j].y);
                }
            }

            const double momentum =
                iteration < 100 ? 0.5 : 0.8;

            for (std::size_t i = 0; i < n; ++i) {
                velocity[i].x =
                    momentum * velocity[i].x -
                    config_.learningRate * gradient[i].x;

                velocity[i].y =
                    momentum * velocity[i].y -
                    config_.learningRate * gradient[i].y;

                embedding[i].x += velocity[i].x;
                embedding[i].y += velocity[i].y;
            }
        }

        return embedding;
    }
};

Matrix createSyntheticManifold(
    std::size_t samplesPerCurve,
    unsigned seed
) {
    if (samplesPerCurve < 5) {
        throw std::invalid_argument(
            "At least five samples per curve are required"
        );
    }

    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, 0.05);

    Matrix data;

    for (int curve = 0; curve < 2; ++curve) {
        for (std::size_t i = 0;
             i < samplesPerCurve;
             ++i) {

            const double t =
                4.0 * M_PI * i /
                static_cast<double>(samplesPerCurve - 1);

            const double shift =
                curve == 0 ? 0.0 : M_PI;

            const double n = noise(generator);

            data.push_back({
                t * std::cos(t + shift) + n,
                t * std::sin(t + shift) + n,
                t / (4.0 * M_PI) +
                    (curve == 0 ? 0.0 : 1.5) + n,
                std::sin(t / 2.0 + shift) + n
            });
        }
    }

    return data;
}

void printEmbedding(
    const std::vector<Point2D>& embedding,
    std::size_t limit = 10
) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "id,x,y\n";

    const std::size_t count =
        std::min(limit, embedding.size());

    for (std::size_t i = 0; i < count; ++i) {
        std::cout
            << i << ','
            << embedding[i].x << ','
            << embedding[i].y << '\n';
    }
}

int main() {
    try {
        ReductionConfig config;
        config.iterations = 250;
        config.neighbors = 8;
        config.perplexity = 8.0;
        config.learningRate = 50.0;
        config.seed = 42;

        const Matrix dataset =
            createSyntheticManifold(20, config.seed);

        ManifoldEmbeddingEngine engine(config);

        std::cout
            << "PCA embedding\n";
        printEmbedding(
            engine.run(dataset, ReductionMethod::PCA)
        );

        std::cout
            << "\nt-SNE embedding\n";
        printEmbedding(
            engine.run(dataset, ReductionMethod::TSNE)
        );

        std::cout
            << "\nUMAP-style graph embedding\n";
        printEmbedding(
            engine.run(dataset, ReductionMethod::UMAP_STYLE)
        );

        /*
         * Exact t-SNE is quadratic in sample count because every pair
         * contributes to the low-dimensional probability distribution.
         * The graph-based method keeps explicit local edges, making its
         * representation substantially more sparse.
         */
    } catch (const std::exception& error) {
        std::cerr
            << "Dimensionality-reduction failure: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}
