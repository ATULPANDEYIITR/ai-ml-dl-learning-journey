#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

enum class SearchStrategy {
    Grid,
    Random
};

struct Hyperparameters {
    double learningRate;
    int maxDepth;
    int estimators;

    bool operator==(const Hyperparameters& other) const {
        return learningRate == other.learningRate &&
               maxDepth == other.maxDepth &&
               estimators == other.estimators;
    }
};

struct Trial {
    Hyperparameters parameters;
    double score;
    double duration;
};

struct ParameterHash {
    std::size_t operator()(const Hyperparameters& p) const noexcept {
        std::size_t h1 = std::hash<double>{}(p.learningRate);
        std::size_t h2 = std::hash<int>{}(p.maxDepth);
        std::size_t h3 = std::hash<int>{}(p.estimators);

        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class SearchSpace {
private:
    std::vector<double> learningRates_;
    std::vector<int> maxDepths_;
    std::vector<int> estimators_;

public:
    SearchSpace(
        std::vector<double> learningRates,
        std::vector<int> maxDepths,
        std::vector<int> estimators
    )
        : learningRates_(std::move(learningRates)),
          maxDepths_(std::move(maxDepths)),
          estimators_(std::move(estimators)) {
        validate();
    }

    void validate() const {
        if (learningRates_.empty() ||
            maxDepths_.empty() ||
            estimators_.empty()) {
            throw std::invalid_argument(
                "Every search-space dimension must contain values."
            );
        }

        for (double rate : learningRates_) {
            if (!std::isfinite(rate) || rate <= 0.0) {
                throw std::invalid_argument(
                    "Learning rates must be positive finite values."
                );
            }
        }

        for (int depth : maxDepths_) {
            if (depth < 1) {
                throw std::invalid_argument(
                    "Maximum depth must be positive."
                );
            }
        }

        for (int count : estimators_) {
            if (count < 1) {
                throw std::invalid_argument(
                    "Estimator count must be positive."
                );
            }
        }
    }

    const auto& learningRates() const { return learningRates_; }
    const auto& maxDepths() const { return maxDepths_; }
    const auto& estimators() const { return estimators_; }

    std::size_t size() const {
        return learningRates_.size() *
               maxDepths_.size() *
               estimators_.size();
    }
};

class GovernanceEngine {
private:
    const SearchSpace& space_;
    std::mt19937 generator_;

    double randomNoise() {
        std::uniform_real_distribution<double> distribution(-0.003, 0.003);
        return distribution(generator_);
    }

public:
    GovernanceEngine(const SearchSpace& space, unsigned seed = 42)
        : space_(space), generator_(seed) {}

    std::vector<Hyperparameters> gridConfigurations() const {
        std::vector<Hyperparameters> configurations;

        for (double rate : space_.learningRates()) {
            for (int depth : space_.maxDepths()) {
                for (int count : space_.estimators()) {
                    configurations.push_back({rate, depth, count});
                }
            }
        }

        return configurations;
    }

    std::vector<Hyperparameters> randomConfigurations(
        std::size_t requested
    ) {
        if (requested == 0) {
            throw std::invalid_argument(
                "Random trial count must be greater than zero."
            );
        }

        requested = std::min(requested, space_.size());

        std::unordered_set<Hyperparameters, ParameterHash> unique;
        std::uniform_int_distribution<std::size_t> rateIndex(
            0, space_.learningRates().size() - 1
        );
        std::uniform_int_distribution<std::size_t> depthIndex(
            0, space_.maxDepths().size() - 1
        );
        std::uniform_int_distribution<std::size_t> estimatorIndex(
            0, space_.estimators().size() - 1
        );

        while (unique.size() < requested) {
            unique.insert({
                space_.learningRates()[rateIndex(generator_)],
                space_.maxDepths()[depthIndex(generator_)],
                space_.estimators()[estimatorIndex(generator_)]
            });
        }

        return {
            unique.begin(),
            unique.end()
        };
    }

    Trial evaluate(const Hyperparameters& p) {
        if (p.learningRate <= 0.0 ||
            p.maxDepth < 1 ||
            p.estimators < 1) {
            throw std::invalid_argument(
                "Invalid hyperparameter configuration."
            );
        }

        // Synthetic validation surface centered around the strongest
        // configuration. This isolates search behavior from ML libraries.
        const double learningRatePenalty =
            std::abs(std::log10(p.learningRate) - std::log10(0.05));

        const double depthPenalty =
            std::abs(p.maxDepth - 6) * 0.018;

        const double estimatorPenalty =
            std::abs(p.estimators - 200) / 10000.0;

        double score =
            0.96 -
            learningRatePenalty * 0.09 -
            depthPenalty -
            estimatorPenalty +
            randomNoise();

        score = std::clamp(score, 0.0, 1.0);

        const double duration =
            0.5 +
            p.maxDepth * 0.08 +
            p.estimators * 0.004;

        return {p, score, duration};
    }

    std::vector<Trial> runGrid() {
        std::vector<Trial> trials;

        for (const auto& configuration : gridConfigurations()) {
            trials.push_back(evaluate(configuration));
        }

        return trials;
    }

    std::vector<Trial> runRandom(std::size_t trialCount) {
        std::vector<Trial> trials;

        for (const auto& configuration : randomConfigurations(trialCount)) {
            trials.push_back(evaluate(configuration));
        }

        return trials;
    }
};

const Trial& bestTrial(const std::vector<Trial>& trials) {
    if (trials.empty()) {
        throw std::runtime_error("No trials were evaluated.");
    }

    return *std::max_element(
        trials.begin(),
        trials.end(),
        [](const Trial& a, const Trial& b) {
            return a.score < b.score;
        }
    );
}

void printTrial(const Trial& trial) {
    std::cout << std::fixed << std::setprecision(4)
              << "learning_rate=" << trial.parameters.learningRate
              << ", max_depth=" << trial.parameters.maxDepth
              << ", estimators=" << trial.parameters.estimators
              << ", score=" << trial.score
              << ", duration=" << trial.duration
              << '\n';
}

void printResults(
    const std::string& name,
    const std::vector<Trial>& trials
) {
    const Trial& best = bestTrial(trials);

    double totalScore = 0.0;
    for (const auto& trial : trials) {
        totalScore += trial.score;
    }

    std::cout << "\n" << name << " SEARCH\n";
    std::cout << std::string(55, '-') << '\n';
    std::cout << "Trials: " << trials.size() << '\n';
    std::cout << std::fixed << std::setprecision(4)
              << "Average score: "
              << totalScore / static_cast<double>(trials.size())
              << '\n';
    std::cout << "Best trial: ";
    printTrial(best);
}

int main() {
    try {
        SearchSpace space(
            {0.005, 0.01, 0.02, 0.05, 0.1, 0.2},
            {2, 4, 6, 8, 10},
            {50, 100, 200, 400}
        );

        GovernanceEngine engine(space, 42);

        const auto gridTrials = engine.runGrid();
        const auto randomTrials = engine.runRandom(20);

        printResults("GRID", gridTrials);
        printResults("RANDOM", randomTrials);

        std::cout << "\nSEARCH-SPACE ANALYSIS\n";
        std::cout << std::string(55, '-') << '\n';
        std::cout << "Possible configurations: "
                  << space.size() << '\n';
        std::cout << "Grid evaluations: "
                  << gridTrials.size() << '\n';
        std::cout << "Random evaluations: "
                  << randomTrials.size() << '\n';

        std::vector<Trial> ranked = randomTrials;

        std::sort(
            ranked.begin(),
            ranked.end(),
            [](const Trial& a, const Trial& b) {
                return a.score > b.score;
            }
        );

        std::cout << "\nTop random configurations:\n";

        const std::size_t displayCount =
            std::min<std::size_t>(3, ranked.size());

        for (std::size_t i = 0; i < displayCount; ++i) {
            std::cout << i + 1 << ". ";
            printTrial(ranked[i]);
        }

        // Large Cartesian spaces make exhaustive grid search expensive.
        // Random search trades exhaustive coverage for a fixed trial budget.
        std::cout << "\nComplexity:\n";
        std::cout << "Grid search evaluations = product of dimension sizes.\n";
        std::cout << "Random search evaluations = configured trial budget.\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Optimization failed: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
