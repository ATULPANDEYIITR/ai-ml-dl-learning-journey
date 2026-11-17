#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * C++17 Optimizer and Repository-Scale Training Engine
 *
 * Implements:
 * 1. Momentum
 * 2. AdaGrad
 * 3. RMSProp
 * 4. Adam
 * 5. AdamW
 *
 * Uses a vector-based numerical objective, polymorphic optimizer
 * interface, gradient clipping, validation, and merge-free state updates.
 *
 * Compile:
 * g++ -std=c++17 -O2 -Wall -Wextra optimizer_lab.cpp -o optimizer_lab
 *
 * Run:
 * ./optimizer_lab
 */

using Vector = std::vector<double>;

enum class OptimizerType {
    Momentum,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW
};

struct OptimizerConfig {
    double learning_rate = 0.01;
    double epsilon = 1e-8;
    double weight_decay = 0.0;
    double gradient_clip = 0.0;
};

struct TrainingResult {
    std::string name;
    Vector parameters;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    std::size_t iterations = 0;
    std::vector<double> history;
};

class VectorOperations {
public:
    static void validate(const Vector& values, const std::string& name) {
        if (values.empty()) {
            throw std::invalid_argument(name + " cannot be empty.");
        }

        for (double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    name + " contains a non-finite value."
                );
            }
        }
    }

    static void validateDimensions(
        const Vector& a,
        const Vector& b
    ) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vector dimensions differ.");
        }
    }

    static double norm(const Vector& values) {
        double squared = 0.0;

        for (double value : values) {
            squared += value * value;
        }

        return std::sqrt(squared);
    }

    static Vector clip(const Vector& gradient, double threshold) {
        if (threshold <= 0.0) {
            return gradient;
        }

        const double gradient_norm = norm(gradient);

        if (gradient_norm <= threshold) {
            return gradient;
        }

        const double scale = threshold / gradient_norm;

        Vector result = gradient;

        for (double& value : result) {
            value *= scale;
        }

        return result;
    }
};

class Objective {
public:
    Objective(Vector target, Vector curvature)
        : target_(std::move(target)),
          curvature_(std::move(curvature)) {

        VectorOperations::validate(target_, "Target");
        VectorOperations::validate(curvature_, "Curvature");

        VectorOperations::validateDimensions(target_, curvature_);

        for (double value : curvature_) {
            if (value <= 0.0) {
                throw std::invalid_argument(
                    "Curvature must be positive."
                );
            }
        }
    }

    double loss(const Vector& parameters) const {
        VectorOperations::validate(parameters, "Parameters");
        VectorOperations::validateDimensions(parameters, target_);

        double result = 0.0;

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            const double difference = parameters[i] - target_[i];

            result += 0.5 * curvature_[i] *
                      difference * difference;
        }

        return result;
    }

    Vector gradient(const Vector& parameters) const {
        VectorOperations::validate(parameters, "Parameters");
        VectorOperations::validateDimensions(parameters, target_);

        Vector result(parameters.size());

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            result[i] = curvature_[i] *
                        (parameters[i] - target_[i]);
        }

        return result;
    }

private:
    Vector target_;
    Vector curvature_;
};

class Optimizer {
public:
    explicit Optimizer(OptimizerConfig config)
        : config_(config) {

        if (!std::isfinite(config_.learning_rate) ||
            config_.learning_rate <= 0.0) {
            throw std::invalid_argument("Invalid learning rate.");
        }

        if (!std::isfinite(config_.epsilon) ||
            config_.epsilon <= 0.0) {
            throw std::invalid_argument("Invalid epsilon.");
        }

        if (!std::isfinite(config_.weight_decay) ||
            config_.weight_decay < 0.0) {
            throw std::invalid_argument("Invalid weight decay.");
        }

        if (!std::isfinite(config_.gradient_clip) ||
            config_.gradient_clip < 0.0) {
            throw std::invalid_argument("Invalid gradient clipping.");
        }
    }

    virtual ~Optimizer() = default;

    virtual Vector update(
        const Vector& parameters,
        const Vector& gradients
    ) = 0;

    virtual std::string name() const = 0;

protected:
    OptimizerConfig config_;
    std::size_t step_ = 0;

    Vector prepare(
        const Vector& parameters,
        const Vector& gradients
    ) const {
        VectorOperations::validate(parameters, "Parameters");
        VectorOperations::validate(gradients, "Gradients");
        VectorOperations::validateDimensions(parameters, gradients);

        return VectorOperations::clip(
            gradients,
            config_.gradient_clip
        );
    }

    static Vector validateOutput(Vector values) {
        VectorOperations::validate(values, "Updated parameters");
        return values;
    }
};

class Momentum : public Optimizer {
public:
    explicit Momentum(
        OptimizerConfig config,
        double beta = 0.9
    ) : Optimizer(config), beta_(beta) {
        if (beta_ < 0.0 || beta_ >= 1.0) {
            throw std::invalid_argument("Invalid momentum beta.");
        }
    }

    Vector update(
        const Vector& parameters,
        const Vector& gradients
    ) override {
        Vector g = prepare(parameters, gradients);

        if (velocity_.empty()) {
            velocity_.assign(parameters.size(), 0.0);
        }

        ++step_;

        Vector result(parameters.size());

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            velocity_[i] =
                beta_ * velocity_[i] + g[i];

            result[i] =
                parameters[i] -
                config_.learning_rate * velocity_[i];
        }

        return validateOutput(std::move(result));
    }

    std::string name() const override {
        return "Momentum";
    }

private:
    double beta_;
    Vector velocity_;
};

class AdaGrad : public Optimizer {
public:
    using Optimizer::Optimizer;

    Vector update(
        const Vector& parameters,
        const Vector& gradients
    ) override {
        Vector g = prepare(parameters, gradients);

        if (accumulator_.empty()) {
            accumulator_.assign(parameters.size(), 0.0);
        }

        ++step_;

        Vector result(parameters.size());

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            accumulator_[i] += g[i] * g[i];

            result[i] = parameters[i] -
                config_.learning_rate * g[i] /
                (std::sqrt(accumulator_[i]) + config_.epsilon);
        }

        return validateOutput(std::move(result));
    }

    std::string name() const override {
        return "AdaGrad";
    }

private:
    Vector accumulator_;
};

class RMSProp : public Optimizer {
public:
    explicit RMSProp(
        OptimizerConfig config,
        double beta = 0.9
    ) : Optimizer(config), beta_(beta) {
        if (beta_ < 0.0 || beta_ >= 1.0) {
            throw std::invalid_argument("Invalid RMSProp beta.");
        }
    }

    Vector update(
        const Vector& parameters,
        const Vector& gradients
    ) override {
        Vector g = prepare(parameters, gradients);

        if (square_average_.empty()) {
            square_average_.assign(parameters.size(), 0.0);
        }

        ++step_;

        Vector result(parameters.size());

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            square_average_[i] =
                beta_ * square_average_[i] +
                (1.0 - beta_) * g[i] * g[i];

            result[i] = parameters[i] -
                config_.learning_rate * g[i] /
                (std::sqrt(square_average_[i]) + config_.epsilon);
        }

        return validateOutput(std::move(result));
    }

    std::string name() const override {
        return "RMSProp";
    }

private:
    double beta_;
    Vector square_average_;
};

class Adam : public Optimizer {
public:
    explicit Adam(
        OptimizerConfig config,
        double beta1 = 0.9,
        double beta2 = 0.999
    ) : Optimizer(config),
        beta1_(beta1),
        beta2_(beta2) {

        if (beta1_ < 0.0 || beta1_ >= 1.0 ||
            beta2_ < 0.0 || beta2_ >= 1.0) {
            throw std::invalid_argument("Invalid Adam coefficients.");
        }
    }

    Vector update(
        const Vector& parameters,
        const Vector& gradients
    ) override {
        Vector g = prepare(parameters, gradients);

        if (first_moment_.empty()) {
            first_moment_.assign(parameters.size(), 0.0);
            second_moment_.assign(parameters.size(), 0.0);
        }

        ++step_;

        const double correction1 =
            1.0 - std::pow(beta1_, static_cast<double>(step_));

        const double correction2 =
            1.0 - std::pow(beta2_, static_cast<double>(step_));

        Vector result(parameters.size());

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            first_moment_[i] =
                beta1_ * first_moment_[i] +
                (1.0 - beta1_) * g[i];

            second_moment_[i] =
                beta2_ * second_moment_[i] +
                (1.0 - beta2_) * g[i] * g[i];

            const double m_hat =
                first_moment_[i] / correction1;

            const double v_hat =
                second_moment_[i] / correction2;

            result[i] = parameters[i] -
                config_.learning_rate * m_hat /
                (std::sqrt(v_hat) + config_.epsilon);
        }

        return validateOutput(std::move(result));
    }

    std::string name() const override {
        return "Adam";
    }

protected:
    double beta1_;
    double beta2_;
    Vector first_moment_;
    Vector second_moment_;
};

class AdamW : public Adam {
public:
    using Adam::Adam;

    Vector update(
        const Vector& parameters,
        const Vector& gradients
    ) override {
        Vector result = Adam::update(parameters, gradients);

        // Decoupled decay is applied directly to the original parameters.
        const double decay =
            config_.learning_rate * config_.weight_decay;

        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] -= decay * parameters[i];
        }

        return validateOutput(std::move(result));
    }

    std::string name() const override {
        return "AdamW";
    }
};

class TrainingEngine {
public:
    explicit TrainingEngine(const Objective& objective)
        : objective_(objective) {}

    TrainingResult run(
        Optimizer& optimizer,
        Vector initial_parameters,
        std::size_t max_iterations,
        double tolerance
    ) const {
        if (max_iterations == 0) {
            throw std::invalid_argument(
                "Iteration count must be positive."
            );
        }

        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument("Invalid tolerance.");
        }

        TrainingResult result;

        result.name = optimizer.name();
        result.parameters = std::move(initial_parameters);

        result.initial_loss =
            objective_.loss(result.parameters);

        result.history.push_back(result.initial_loss);

        for (std::size_t i = 0; i < max_iterations; ++i) {
            Vector gradients =
                objective_.gradient(result.parameters);

            result.parameters =
                optimizer.update(result.parameters, gradients);

            const double current_loss =
                objective_.loss(result.parameters);

            if (!std::isfinite(current_loss)) {
                throw std::runtime_error("Non-finite loss detected.");
            }

            result.history.push_back(current_loss);
            ++result.iterations;

            if (current_loss <= tolerance) {
                break;
            }
        }

        result.final_loss = result.history.back();

        return result;
    }

private:
    const Objective& objective_;
};

std::unique_ptr<Optimizer> createOptimizer(
    OptimizerType type,
    const OptimizerConfig& config
) {
    switch (type) {
        case OptimizerType::Momentum:
            return std::make_unique<Momentum>(config);

        case OptimizerType::AdaGrad:
            return std::make_unique<AdaGrad>(config);

        case OptimizerType::RMSProp:
            return std::make_unique<RMSProp>(config);

        case OptimizerType::Adam:
            return std::make_unique<Adam>(config);

        case OptimizerType::AdamW:
            return std::make_unique<AdamW>(config);
    }

    throw std::invalid_argument("Unsupported optimizer.");
}

void runTests() {
    const Vector parameters = {2.0, -1.0};
    const Vector gradients = {0.5, -0.25};

    OptimizerConfig config;
    config.learning_rate = 0.01;

    const std::vector<OptimizerType> types = {
        OptimizerType::Momentum,
        OptimizerType::AdaGrad,
        OptimizerType::RMSProp,
        OptimizerType::Adam,
        OptimizerType::AdamW
    };

    for (OptimizerType type : types) {
        auto optimizer = createOptimizer(type, config);

        Vector result = optimizer->update(parameters, gradients);

        VectorOperations::validate(result, "Test output");

        VectorOperations::validateDimensions(parameters, result);
    }

    try {
        OptimizerConfig invalid;
        invalid.learning_rate = -1.0;

        Adam optimizer(invalid);

        throw std::runtime_error("Invalid configuration accepted.");
    } catch (const std::invalid_argument&) {
    }

    try {
        Adam optimizer(config);
        optimizer.update({1.0}, {1.0, 2.0});

        throw std::runtime_error("Dimension mismatch accepted.");
    } catch (const std::invalid_argument&) {
    }

    OptimizerConfig decay_config;
    decay_config.learning_rate = 0.1;
    decay_config.weight_decay = 0.1;

    AdamW adamw(decay_config);

    Vector decayed = adamw.update({2.0}, {0.0});

    if (!(decayed[0] < 2.0)) {
        throw std::runtime_error("AdamW decay test failed.");
    }

    std::cout << "All C++17 tests passed.\n";
}

void printResults(const std::vector<TrainingResult>& results) {
    std::cout << "\nOPTIMIZER COMPARISON\n";
    std::cout << std::string(83, '=') << '\n';

    std::cout
        << std::left << std::setw(15) << "Optimizer"
        << std::right << std::setw(16) << "Initial Loss"
        << std::setw(18) << "Final Loss"
        << std::setw(14) << "Iterations"
        << '\n';

    std::cout << std::string(83, '-') << '\n';

    for (const auto& result : results) {
        std::cout
            << std::left << std::setw(15) << result.name
            << std::right << std::setw(16)
            << std::fixed << std::setprecision(6)
            << result.initial_loss
            << std::setw(18) << std::setprecision(10)
            << result.final_loss
            << std::setw(14) << result.iterations
            << '\n';
    }
}

int main() {
    try {
        runTests();

        Objective objective(
            {2.0, -3.0, 1.0},
            {1.0, 10.0, 50.0}
        );

        TrainingEngine engine(objective);

        struct Experiment {
            OptimizerType type;
            OptimizerConfig config;
        };

        OptimizerConfig momentum_config;
        momentum_config.learning_rate = 0.005;

        OptimizerConfig adagrad_config;
        adagrad_config.learning_rate = 0.8;

        OptimizerConfig rmsprop_config;
        rmsprop_config.learning_rate = 0.02;

        OptimizerConfig adam_config;
        adam_config.learning_rate = 0.08;

        OptimizerConfig adamw_config;
        adamw_config.learning_rate = 0.08;
        adamw_config.weight_decay = 0.001;

        std::vector<Experiment> experiments = {
            {OptimizerType::Momentum, momentum_config},
            {OptimizerType::AdaGrad, adagrad_config},
            {OptimizerType::RMSProp, rmsprop_config},
            {OptimizerType::Adam, adam_config},
            {OptimizerType::AdamW, adamw_config}
        };

        std::vector<TrainingResult> results;

        for (const auto& experiment : experiments) {
            auto optimizer = createOptimizer(
                experiment.type,
                experiment.config
            );

            results.push_back(
                engine.run(
                    *optimizer,
                    {10.0, 8.0, -5.0},
                    3000,
                    1e-10
                )
            );
        }

        printResults(results);

        std::cout << "\nFINAL PARAMETER STATES\n";

        for (const auto& result : results) {
            std::cout << result.name << ": ";

            for (double value : result.parameters) {
                std::cout << std::fixed << std::setprecision(6)
                          << value << " ";
            }

            std::cout << '\n';
        }

    } catch (const std::exception& error) {
        std::cerr << "Execution error: "
                  << error.what() << '\n';

        return 1;
    }

    return 0;
}
