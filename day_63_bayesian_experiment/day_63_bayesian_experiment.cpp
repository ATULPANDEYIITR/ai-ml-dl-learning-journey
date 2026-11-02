#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

enum class Direction {
    Maximize,
    Minimize
};

struct Bounds {
    double low;
    double high;

    void validate() const {
        if (!std::isfinite(low) || !std::isfinite(high) || low >= high) {
            throw std::invalid_argument("Invalid optimization bounds.");
        }
    }
};

struct Observation {
    std::vector<double> x;
    double y;
};

struct Prediction {
    double mean;
    double standard_deviation;
};

class GaussianProcess {
public:
    GaussianProcess(
        double length_scale = 0.8,
        double signal_variance = 1.0,
        double noise_variance = 1e-5)
        : length_scale_(length_scale),
          signal_variance_(signal_variance),
          noise_variance_(noise_variance) {
        if (length_scale_ <= 0.0 ||
            signal_variance_ <= 0.0 ||
            noise_variance_ < 0.0) {
            throw std::invalid_argument("Invalid GP hyperparameters.");
        }
    }

    void fit(const std::vector<Observation>& observations) {
        if (observations.empty()) {
            throw std::invalid_argument("GP requires observations.");
        }

        x_train_.clear();
        for (const auto& observation : observations) {
            x_train_.push_back(observation.x);
        }

        const std::size_t n = observations.size();
        std::vector<std::vector<double>> matrix(
            n, std::vector<double>(n, 0.0));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                matrix[i][j] = kernel(x_train_[i], x_train_[j]);
                if (i == j) {
                    matrix[i][j] += noise_variance_ + 1e-9;
                }
            }
        }

        L_ = cholesky(matrix);

        std::vector<double> y;
        y.reserve(n);
        for (const auto& observation : observations) {
            y.push_back(observation.y);
        }

        const auto forward = forward_substitution(L_, y);
        alpha_ = backward_substitution(L_, forward);
    }

    Prediction predict(const std::vector<double>& point) const {
        if (x_train_.empty()) {
            throw std::logic_error("GP has not been fitted.");
        }

        std::vector<double> k_star;
        k_star.reserve(x_train_.size());

        for (const auto& train_point : x_train_) {
            k_star.push_back(kernel(point, train_point));
        }

        double mean = 0.0;
        for (std::size_t i = 0; i < k_star.size(); ++i) {
            mean += k_star[i] * alpha_[i];
        }

        const auto solved = forward_substitution(L_, k_star);
        double variance = kernel(point, point);

        for (double value : solved) {
            variance -= value * value;
        }

        return {mean, std::sqrt(std::max(0.0, variance))};
    }

private:
    double length_scale_;
    double signal_variance_;
    double noise_variance_;

    std::vector<std::vector<double>> x_train_;
    std::vector<std::vector<double>> L_;
    std::vector<double> alpha_;

    double kernel(
        const std::vector<double>& a,
        const std::vector<double>& b) const {

        double distance = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            const double delta = a[i] - b[i];
            distance += delta * delta;
        }

        return signal_variance_ *
               std::exp(-0.5 * distance / (length_scale_ * length_scale_));
    }

    static std::vector<std::vector<double>> cholesky(
        const std::vector<std::vector<double>>& matrix) {

        const std::size_t n = matrix.size();
        std::vector<std::vector<double>> L(
            n, std::vector<double>(n, 0.0));

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = matrix[i][j];

                for (std::size_t k = 0; k < j; ++k) {
                    value -= L[i][k] * L[j][k];
                }

                if (i == j) {
                    if (value <= 1e-12) {
                        throw std::runtime_error(
                            "Kernel matrix is not positive definite.");
                    }
                    L[i][j] = std::sqrt(value);
                } else {
                    L[i][j] = value / L[j][j];
                }
            }
        }

        return L;
    }

    static std::vector<double> forward_substitution(
        const std::vector<std::vector<double>>& L,
        const std::vector<double>& b) {

        std::vector<double> result(b.size(), 0.0);

        for (std::size_t i = 0; i < b.size(); ++i) {
            double value = b[i];

            for (std::size_t j = 0; j < i; ++j) {
                value -= L[i][j] * result[j];
            }

            result[i] = value / L[i][i];
        }

        return result;
    }

    static std::vector<double> backward_substitution(
        const std::vector<std::vector<double>>& L,
        const std::vector<double>& b) {

        std::vector<double> result(b.size(), 0.0);

        for (std::size_t i = b.size(); i-- > 0;) {
            double value = b[i];

            for (std::size_t j = i + 1; j < b.size(); ++j) {
                value -= L[j][i] * result[j];
            }

            result[i] = value / L[i][i];
        }

        return result;
    }
};

class Acquisition {
public:
    static double normalPdf(double z) {
        return std::exp(-0.5 * z * z) /
               std::sqrt(2.0 * M_PI);
    }

    static double normalCdf(double z) {
        return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    }

    static double expectedImprovement(
        const Prediction& prediction,
        double best,
        double exploration = 0.01) {

        if (prediction.standard_deviation <= 1e-12) {
            return 0.0;
        }

        const double improvement =
            prediction.mean - best - exploration;

        const double z =
            improvement / prediction.standard_deviation;

        return improvement * normalCdf(z) +
               prediction.standard_deviation * normalPdf(z);
    }
};

class GovernanceOptimizer {
public:
    GovernanceOptimizer(
        std::vector<Bounds> bounds,
        unsigned seed = 42)
        : bounds_(std::move(bounds)), generator_(seed) {

        if (bounds_.empty()) {
            throw std::invalid_argument(
                "At least one optimization dimension is required.");
        }

        for (const auto& bound : bounds_) {
            bound.validate();
        }
    }

    double objective(const std::vector<double>& x) const {
        if (x.size() != 2) {
            throw std::invalid_argument(
                "This technical case study requires two variables.");
        }

        const double x1 = x[0];
        const double x2 = x[1];

        return
            1.8 * std::exp(
                -((x1 - 1.5) * (x1 - 1.5) +
                  (x2 - 2.0) * (x2 - 2.0)) / 1.4)
            +
            1.25 * std::exp(
                -((x1 + 2.0) * (x1 + 2.0) +
                  (x2 + 1.0) * (x2 + 1.0)) / 0.8)
            -
            0.08 * (x1 * x1 + x2 * x2)
            +
            0.015 * std::sin(17.0 * x1 + 11.0 * x2);
    }

    std::vector<double> optimize(
        int initial_points,
        int iterations,
        int grid_points) {

        if (initial_points < 1 ||
            iterations < 1 ||
            grid_points < 2) {
            throw std::invalid_argument(
                "Optimization parameters must be positive.");
        }

        for (int i = 0; i < initial_points; ++i) {
            observations_.push_back(
                {random_point(), 0.0});
            observations_.back().y =
                objective(observations_.back().x);
        }

        GaussianProcess gp;

        for (int iteration = 0; iteration < iterations; ++iteration) {
            gp.fit(observations_);

            double best_value = -std::numeric_limits<double>::infinity();

            for (const auto& observation : observations_) {
                best_value = std::max(best_value, observation.y);
            }

            auto candidates = generate_grid(grid_points);

            double best_acquisition =
                -std::numeric_limits<double>::infinity();

            std::vector<double> selected;

            for (const auto& candidate : candidates) {
                if (already_observed(candidate)) {
                    continue;
                }

                const auto prediction = gp.predict(candidate);
                const double acquisition =
                    Acquisition::expectedImprovement(
                        prediction, best_value);

                if (acquisition > best_acquisition) {
                    best_acquisition = acquisition;
                    selected = candidate;
                }
            }

            if (selected.empty()) {
                throw std::runtime_error(
                    "No unexplored point remains.");
            }

            observations_.push_back(
                {selected, objective(selected)});
        }

        const auto best = std::max_element(
            observations_.begin(),
            observations_.end(),
            [](const Observation& a, const Observation& b) {
                return a.y < b.y;
            });

        return best->x;
    }

    const std::vector<Observation>& observations() const {
        return observations_;
    }

    void save_csv(const std::string& path) const {
        std::ofstream file(path);

        if (!file) {
            throw std::runtime_error(
                "Unable to open history file.");
        }

        file << "iteration,x1,x2,objective\n";
        file << std::setprecision(12);

        for (std::size_t i = 0; i < observations_.size(); ++i) {
            file << i + 1 << ','
                 << observations_[i].x[0] << ','
                 << observations_[i].x[1] << ','
                 << observations_[i].y << '\n';
        }
    }

private:
    std::vector<Bounds> bounds_;
    std::mt19937 generator_;
    std::vector<Observation> observations_;

    std::vector<double> random_point() {
        std::vector<double> point;

        for (const auto& bound : bounds_) {
            std::uniform_real_distribution<double> distribution(
                bound.low, bound.high);
            point.push_back(distribution(generator_));
        }

        return point;
    }

    std::vector<std::vector<double>> generate_grid(
        int points_per_dimension) const {

        std::vector<std::vector<double>> axes;

        for (const auto& bound : bounds_) {
            std::vector<double> axis;

            for (int i = 0; i < points_per_dimension; ++i) {
                const double ratio =
                    static_cast<double>(i) /
                    static_cast<double>(points_per_dimension - 1);

                axis.push_back(
                    bound.low +
                    ratio * (bound.high - bound.low));
            }

            axes.push_back(axis);
        }

        std::vector<std::vector<double>> result;

        for (double x : axes[0]) {
            for (double y : axes[1]) {
                result.push_back({x, y});
            }
        }

        return result;
    }

    bool already_observed(
        const std::vector<double>& candidate) const {

        constexpr double tolerance = 1e-9;

        for (const auto& observation : observations_) {
            bool same = true;

            for (std::size_t i = 0; i < candidate.size(); ++i) {
                if (std::abs(candidate[i] - observation.x[i]) >
                    tolerance) {
                    same = false;
                    break;
                }
            }

            if (same) {
                return true;
            }
        }

        return false;
    }
};

int main() {
    try {
        GovernanceOptimizer optimizer({
            {-4.0, 4.0},
            {-4.0, 4.0}
        });

        const auto best = optimizer.optimize(6, 18, 17);

        const auto& history = optimizer.observations();

        auto best_record = std::max_element(
            history.begin(),
            history.end(),
            [](const Observation& a, const Observation& b) {
                return a.y < b.y;
            });

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Bayesian Optimization\n";
        std::cout << "=====================\n";
        std::cout << "Observations: "
                  << history.size() << '\n';
        std::cout << "Best x: ("
                  << best[0] << ", "
                  << best[1] << ")\n";
        std::cout << "Best objective: "
                  << best_record->y << '\n';

        optimizer.save_csv(
            "bayesian_optimization_history.csv");

        std::cout << "History written to "
                  << "bayesian_optimization_history.csv\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Optimization failed: "
                  << error.what() << '\n';
        return 1;
    }
}
