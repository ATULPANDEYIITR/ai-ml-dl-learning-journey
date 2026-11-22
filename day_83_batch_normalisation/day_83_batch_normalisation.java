#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

enum class Mode {
    Training,
    Inference
};

struct Statistics {
    Vector mean;
    Vector variance;
    std::size_t count;
};

struct ForwardCache {
    Matrix normalized;
    Vector inverseStd;
    std::size_t sampleCount;
    Mode mode;
};

struct Gradients {
    Matrix input;
    Vector gamma;
    Vector beta;
};

class BatchNormalization {
private:
    std::size_t features_;
    double epsilon_;
    double momentum_;

    Vector gamma_;
    Vector beta_;

    Vector runningMean_;
    Vector runningVariance_;

    std::size_t batchesSeen_ = 0;

    void validateBatch(const Matrix& batch) const {
        if (batch.empty()) {
            throw std::invalid_argument("Batch cannot be empty");
        }

        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].size() != features_) {
                throw std::invalid_argument(
                    "Feature dimension mismatch at row " +
                    std::to_string(i)
                );
            }

            for (double value : batch[i]) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "Input contains a non-finite value"
                    );
                }
            }
        }
    }

    Statistics calculateStatistics(const Matrix& batch) const {
        const std::size_t count = batch.size();

        Vector mean(features_, 0.0);
        Vector m2(features_, 0.0);

        // Welford's online algorithm provides stable variance estimates.
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < features_; ++j) {
                double delta = batch[i][j] - mean[j];

                mean[j] += delta / static_cast<double>(i + 1);

                double deltaAfter = batch[i][j] - mean[j];

                m2[j] += delta * deltaAfter;
            }
        }

        Vector variance(features_, 0.0);

        for (std::size_t j = 0; j < features_; ++j) {
            variance[j] = std::max(
                0.0,
                m2[j] / static_cast<double>(count)
            );
        }

        return {mean, variance, count};
    }

    void updateRunningStatistics(const Statistics& statistics) {
        Vector nextMean = runningMean_;
        Vector nextVariance = runningVariance_;

        for (std::size_t j = 0; j < features_; ++j) {
            nextMean[j] =
                momentum_ * runningMean_[j]
                + (1.0 - momentum_) * statistics.mean[j];

            nextVariance[j] =
                momentum_ * runningVariance_[j]
                + (1.0 - momentum_) * statistics.variance[j];
        }

        runningMean_.swap(nextMean);
        runningVariance_.swap(nextVariance);

        ++batchesSeen_;
    }

public:
    explicit BatchNormalization(
        std::size_t features,
        double epsilon = 1e-5,
        double momentum = 0.9
    )
        : features_(features),
          epsilon_(epsilon),
          momentum_(momentum),
          gamma_(features, 1.0),
          beta_(features, 0.0),
          runningMean_(features, 0.0),
          runningVariance_(features, 1.0) {

        if (features == 0) {
            throw std::invalid_argument(
                "Feature count must be positive"
            );
        }

        if (!std::isfinite(epsilon) || epsilon <= 0) {
            throw std::invalid_argument("Invalid epsilon");
        }

        if (
            !std::isfinite(momentum) ||
            momentum < 0 ||
            momentum >= 1
        ) {
            throw std::invalid_argument("Invalid momentum");
        }
    }

    ForwardCache forward(
        const Matrix& batch,
        Mode mode,
        Matrix& output
    ) {
        validateBatch(batch);

        const std::size_t count = batch.size();

        Vector mean;
        Vector variance;

        if (mode == Mode::Training) {
            Statistics statistics = calculateStatistics(batch);

            mean = statistics.mean;
            variance = statistics.variance;

            updateRunningStatistics(statistics);
        } else {
            if (batchesSeen_ == 0) {
                throw std::runtime_error(
                    "Inference statistics are not initialized"
                );
            }

            mean = runningMean_;
            variance = runningVariance_;
        }

        Vector inverseStd(features_);

        for (std::size_t j = 0; j < features_; ++j) {
            inverseStd[j] = 1.0 / std::sqrt(
                variance[j] + epsilon_
            );
        }

        Matrix normalized(
            count,
            Vector(features_, 0.0)
        );

        output.assign(
            count,
            Vector(features_, 0.0)
        );

        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < features_; ++j) {
                normalized[i][j] =
                    (batch[i][j] - mean[j]) * inverseStd[j];

                output[i][j] =
                    gamma_[j] * normalized[i][j] + beta_[j];

                if (!std::isfinite(output[i][j])) {
                    throw std::runtime_error(
                        "Non-finite normalization result"
                    );
                }
            }
        }

        return {
            normalized,
            inverseStd,
            count,
            mode
        };
    }

    Gradients backward(
        const Matrix& upstream,
        const ForwardCache& cache
    ) const {
        if (cache.mode != Mode::Training) {
            throw std::logic_error(
                "Backward propagation requires training mode"
            );
        }

        if (upstream.size() != cache.sampleCount) {
            throw std::invalid_argument(
                "Gradient batch dimension mismatch"
            );
        }

        validateBatch(upstream);

        const std::size_t count = cache.sampleCount;

        Vector dGamma(features_, 0.0);
        Vector dBeta(features_, 0.0);

        for (std::size_t j = 0; j < features_; ++j) {
            for (std::size_t i = 0; i < count; ++i) {
                dGamma[j] +=
                    upstream[i][j] * cache.normalized[i][j];

                dBeta[j] += upstream[i][j];
            }
        }

        Matrix dInput(
            count,
            Vector(features_, 0.0)
        );

        for (std::size_t j = 0; j < features_; ++j) {
            for (std::size_t i = 0; i < count; ++i) {
                dInput[i][j] =
                    gamma_[j] *
                    cache.inverseStd[j] /
                    static_cast<double>(count) *
                    (
                        static_cast<double>(count) * upstream[i][j]
                        - dBeta[j]
                        - cache.normalized[i][j] * dGamma[j]
                    );
            }
        }

        return {dInput, dGamma, dBeta};
    }

    void updateParameters(
        const Gradients& gradients,
        double learningRate
    ) {
        if (
            !std::isfinite(learningRate) ||
            learningRate <= 0
        ) {
            throw std::invalid_argument("Invalid learning rate");
        }

        if (
            gradients.gamma.size() != features_ ||
            gradients.beta.size() != features_
        ) {
            throw std::invalid_argument("Gradient dimensions mismatch");
        }

        Vector nextGamma = gamma_;
        Vector nextBeta = beta_;

        for (std::size_t j = 0; j < features_; ++j) {
            if (
                !std::isfinite(gradients.gamma[j]) ||
                !std::isfinite(gradients.beta[j])
            ) {
                throw std::invalid_argument("Non-finite gradient");
            }

            nextGamma[j] -= learningRate * gradients.gamma[j];
            nextBeta[j] -= learningRate * gradients.beta[j];

            if (
                !std::isfinite(nextGamma[j]) ||
                !std::isfinite(nextBeta[j])
            ) {
                throw std::overflow_error(
                    "Parameter update overflow"
                );
            }
        }

        gamma_.swap(nextGamma);
        beta_.swap(nextBeta);
    }

    const Vector& runningMean() const {
        return runningMean_;
    }

    const Vector& runningVariance() const {
        return runningVariance_;
    }

    const Vector& gamma() const {
        return gamma_;
    }

    const Vector& beta() const {
        return beta_;
    }

    std::size_t batchesSeen() const {
        return batchesSeen_;
    }

    void printState() const {
        auto printVector = [](const std::string& label,
                              const Vector& values) {
            std::cout << label << ": ";

            for (double value : values) {
                std::cout << std::fixed
                          << std::setprecision(5)
                          << value << " ";
            }

            std::cout << '\n';
        };

        printVector("Gamma", gamma_);
        printVector("Beta", beta_);
        printVector("Running mean", runningMean_);
        printVector("Running variance", runningVariance_);

        std::cout << "Batches observed: "
                  << batchesSeen_ << '\n';
    }
};

void printMatrix(const std::string& title, const Matrix& matrix) {
    std::cout << "\n" << title << '\n';

    for (const auto& row : matrix) {
        for (double value : row) {
            std::cout << std::fixed
                      << std::setprecision(5)
                      << std::setw(12)
                      << value;
        }

        std::cout << '\n';
    }
}

int main() {
    try {
        std::cout << "BATCH NORMALIZATION ENGINE\n";

        BatchNormalization layer(3, 1e-5, 0.9);

        Matrix trainingBatch = {
            {2.0, 4.0, 6.0},
            {4.0, 8.0, 10.0},
            {6.0, 12.0, 14.0},
            {8.0, 16.0, 18.0}
        };

        Matrix trainingOutput;

        ForwardCache cache = layer.forward(
            trainingBatch,
            Mode::Training,
            trainingOutput
        );

        printMatrix("Training output", trainingOutput);

        std::cout << "\nRunning statistics after training\n";
        layer.printState();

        Matrix upstream(
            trainingBatch.size(),
            Vector(3, 1.0)
        );

        Gradients gradients = layer.backward(
            upstream,
            cache
        );

        printMatrix("Input gradients", gradients.input);

        layer.updateParameters(gradients, 0.01);

        std::cout << "\nUpdated affine parameters\n";
        layer.printState();

        Matrix inferenceBatch = {
            {5.0, 10.0, 12.0},
            {7.0, 13.0, 15.0}
        };

        Matrix inferenceOutput;

        layer.forward(
            inferenceBatch,
            Mode::Inference,
            inferenceOutput
        );

        printMatrix("Inference output", inferenceOutput);

        // Inference must not modify running statistics.
        const Vector originalMean = layer.runningMean();
        const Vector originalVariance = layer.runningVariance();

        Matrix secondInference;

        layer.forward(
            {{10.0, 20.0, 30.0}},
            Mode::Inference,
            secondInference
        );

        if (
            originalMean != layer.runningMean() ||
            originalVariance != layer.runningVariance()
        ) {
            throw std::runtime_error(
                "Inference unexpectedly modified running statistics"
            );
        }

        std::cout << "\nInference state isolation verified.\n";

        // A single sample has zero population variance.
        BatchNormalization constantLayer(1);

        Matrix constantOutput;

        constantLayer.forward(
            {{5.0}, {5.0}, {5.0}},
            Mode::Training,
            constantOutput
        );

        printMatrix("Constant feature output", constantOutput);

        // Invalid dimensions must fail before numerical processing.
        try {
            Matrix invalid = {{1.0, 2.0}};

            Matrix ignored;

            layer.forward(
                invalid,
                Mode::Training,
                ignored
            );
        } catch (const std::invalid_argument& error) {
            std::cout << "\nInvalid batch rejected: "
                      << error.what() << '\n';
        }

        std::cout << "\nAll tests completed successfully.\n";

    } catch (const std::exception& error) {
        std::cerr << "Execution failure: "
                  << error.what() << '\n';

        return 1;
    }

    return 0;
}
