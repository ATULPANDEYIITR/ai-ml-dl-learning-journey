#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct Dataset {
    Matrix X;
    Vector y;
};

struct Metrics {
    double trainRmse{};
    double testRmse{};
    double trainR2{};
    double testR2{};
    std::size_t zeroCoefficients{};
};

double mean(const Vector& values) {
    if (values.empty()) {
        throw std::invalid_argument("mean requires non-empty data");
    }

    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vector dimensions do not match");
    }

    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double mse(const Vector& actual, const Vector& predicted) {
    if (actual.size() != predicted.size() || actual.empty()) {
        throw std::invalid_argument("invalid MSE inputs");
    }

    double sum = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = actual[i] - predicted[i];
        sum += error * error;
    }

    return sum / static_cast<double>(actual.size());
}

double rmse(const Vector& actual, const Vector& predicted) {
    return std::sqrt(mse(actual, predicted));
}

double r2(const Vector& actual, const Vector& predicted) {
    const double baseline = mean(actual);

    double total = 0.0;
    double residual = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        total += std::pow(actual[i] - baseline, 2);
        residual += std::pow(actual[i] - predicted[i], 2);
    }

    if (total == 0.0) {
        return residual == 0.0 ? 1.0 : 0.0;
    }

    return 1.0 - residual / total;
}

void validateDataset(const Dataset& data) {
    if (data.X.empty() || data.y.empty()) {
        throw std::invalid_argument("dataset cannot be empty");
    }

    if (data.X.size() != data.y.size()) {
        throw std::invalid_argument("X and y row counts differ");
    }

    const std::size_t width = data.X.front().size();

    if (width == 0) {
        throw std::invalid_argument("dataset needs at least one feature");
    }

    for (const auto& row : data.X) {
        if (row.size() != width) {
            throw std::invalid_argument("inconsistent feature width");
        }

        for (double value : row) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("features must be finite");
            }
        }
    }
}

class StandardScaler {
private:
    Vector means_;
    Vector scales_;

public:
    void fit(const Matrix& X) {
        if (X.empty()) {
            throw std::invalid_argument("cannot scale empty matrix");
        }

        const std::size_t features = X.front().size();

        means_.assign(features, 0.0);
        scales_.assign(features, 1.0);

        for (const auto& row : X) {
            for (std::size_t j = 0; j < features; ++j) {
                means_[j] += row[j];
            }
        }

        for (double& value : means_) {
            value /= static_cast<double>(X.size());
        }

        for (const auto& row : X) {
            for (std::size_t j = 0; j < features; ++j) {
                const double difference = row[j] - means_[j];
                scales_[j] += difference * difference;
            }
        }

        for (double& scale : scales_) {
            scale = std::sqrt(
                (scale - 1.0) / static_cast<double>(X.size())
            );

            // A constant feature cannot be standardized through division
            // by zero. Leaving its scale at one preserves a finite value.
            if (scale < 1e-12) {
                scale = 1.0;
            }
        }
    }

    Matrix transform(const Matrix& X) const {
        if (means_.empty()) {
            throw std::logic_error("scaler has not been fitted");
        }

        Matrix result = X;

        for (auto& row : result) {
            if (row.size() != means_.size()) {
                throw std::invalid_argument("feature count mismatch");
            }

            for (std::size_t j = 0; j < row.size(); ++j) {
                row[j] = (row[j] - means_[j]) / scales_[j];
            }
        }

        return result;
    }
};

double softThreshold(double value, double threshold) {
    if (value > threshold) {
        return value - threshold;
    }

    if (value < -threshold) {
        return value + threshold;
    }

    return 0.0;
}

class RegressionModel {
protected:
    double intercept_{0.0};
    Vector coefficients_;

public:
    virtual ~RegressionModel() = default;

    virtual void fit(const Matrix& X, const Vector& y) = 0;

    Vector predict(const Matrix& X) const {
        Vector predictions;
        predictions.reserve(X.size());

        for (const auto& row : X) {
            predictions.push_back(
                intercept_ + dot(row, coefficients_)
            );
        }

        return predictions;
    }

    const Vector& coefficients() const {
        return coefficients_;
    }

    double intercept() const {
        return intercept_;
    }
};

class RidgeRegression final : public RegressionModel {
private:
    double alpha_;
    double learningRate_;
    int maxIterations_;

public:
    RidgeRegression(
        double alpha,
        double learningRate = 0.003,
        int maxIterations = 12000
    )
        : alpha_(alpha),
          learningRate_(learningRate),
          maxIterations_(maxIterations) {
        if (alpha < 0.0) {
            throw std::invalid_argument("Ridge alpha cannot be negative");
        }
    }

    void fit(const Matrix& X, const Vector& y) override {
        Dataset data{X, y};
        validateDataset(data);

        const std::size_t n = X.size();
        const std::size_t p = X.front().size();

        coefficients_.assign(p, 0.0);
        intercept_ = mean(y);

        double previousLoss =
            std::numeric_limits<double>::infinity();

        for (int iteration = 0; iteration < maxIterations_; ++iteration) {
            Vector errors(n);

            for (std::size_t i = 0; i < n; ++i) {
                errors[i] =
                    intercept_ +
                    dot(X[i], coefficients_) -
                    y[i];
            }

            double interceptGradient = 2.0 * mean(errors);
            intercept_ -= learningRate_ * interceptGradient;

            for (std::size_t j = 0; j < p; ++j) {
                double gradient = 0.0;

                for (std::size_t i = 0; i < n; ++i) {
                    gradient += errors[i] * X[i][j];
                }

                gradient =
                    2.0 * gradient / static_cast<double>(n) +
                    2.0 * alpha_ * coefficients_[j];

                coefficients_[j] -=
                    learningRate_ * gradient;
            }

            double dataLoss = 0.0;

            for (double error : errors) {
                dataLoss += error * error;
            }

            dataLoss /= static_cast<double>(n);

            double penalty = 0.0;

            for (double coefficient : coefficients_) {
                penalty += coefficient * coefficient;
            }

            const double loss = dataLoss + alpha_ * penalty;

            if (std::abs(previousLoss - loss) < 1e-9) {
                break;
            }

            previousLoss = loss;
        }
    }
};

class LassoRegression final : public RegressionModel {
private:
    double alpha_;
    int maxIterations_;
    double tolerance_;

public:
    LassoRegression(
        double alpha,
        int maxIterations = 5000,
        double tolerance = 1e-7
    )
        : alpha_(alpha),
          maxIterations_(maxIterations),
          tolerance_(tolerance) {
        if (alpha < 0.0) {
            throw std::invalid_argument("Lasso alpha cannot be negative");
        }
    }

    void fit(const Matrix& X, const Vector& y) override {
        Dataset data{X, y};
        validateDataset(data);

        const std::size_t n = X.size();
        const std::size_t p = X.front().size();

        coefficients_.assign(p, 0.0);
        intercept_ = mean(y);

        Vector residuals(n);

        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] = y[i] - intercept_;
        }

        for (int iteration = 0; iteration < maxIterations_; ++iteration) {
            Vector oldCoefficients = coefficients_;

            const double residualMean = mean(residuals);
            intercept_ += residualMean;

            for (double& residual : residuals) {
                residual -= residualMean;
            }

            for (std::size_t j = 0; j < p; ++j) {
                double rho = 0.0;
                double denominator = 0.0;

                for (std::size_t i = 0; i < n; ++i) {
                    residuals[i] +=
                        X[i][j] * coefficients_[j];

                    rho += X[i][j] * residuals[i];
                    denominator += X[i][j] * X[i][j];
                }

                if (denominator == 0.0) {
                    coefficients_[j] = 0.0;
                } else {
                    coefficients_[j] =
                        softThreshold(
                            rho,
                            alpha_ * static_cast<double>(n) / 2.0
                        ) /
                        denominator;
                }

                for (std::size_t i = 0; i < n; ++i) {
                    residuals[i] -=
                        X[i][j] * coefficients_[j];
                }
            }

            double largestChange = 0.0;

            for (std::size_t j = 0; j < p; ++j) {
                largestChange = std::max(
                    largestChange,
                    std::abs(
                        coefficients_[j] -
                        oldCoefficients[j]
                    )
                );
            }

            if (largestChange < tolerance_) {
                break;
            }
        }
    }
};

class ElasticNetRegression final : public RegressionModel {
private:
    double alpha_;
    double l1Ratio_;
    int maxIterations_;
    double tolerance_;

public:
    ElasticNetRegression(
        double alpha,
        double l1Ratio,
        int maxIterations = 5000,
        double tolerance = 1e-7
    )
        : alpha_(alpha),
          l1Ratio_(l1Ratio),
          maxIterations_(maxIterations),
          tolerance_(tolerance) {
        if (alpha < 0.0) {
            throw std::invalid_argument(
                "Elastic Net alpha cannot be negative"
            );
        }

        if (l1Ratio < 0.0 || l1Ratio > 1.0) {
            throw std::invalid_argument(
                "Elastic Net l1Ratio must be between zero and one"
            );
        }
    }

    void fit(const Matrix& X, const Vector& y) override {
        Dataset data{X, y};
        validateDataset(data);

        const std::size_t n = X.size();
        const std::size_t p = X.front().size();

        coefficients_.assign(p, 0.0);
        intercept_ = mean(y);

        Vector residuals(n);

        for (std::size_t i = 0; i < n; ++i) {
            residuals[i] = y[i] - intercept_;
        }

        const double l1 = alpha_ * l1Ratio_;
        const double l2 = alpha_ * (1.0 - l1Ratio_);

        for (int iteration = 0; iteration < maxIterations_; ++iteration) {
            Vector oldCoefficients = coefficients_;

            const double residualMean = mean(residuals);
            intercept_ += residualMean;

            for (double& residual : residuals) {
                residual -= residualMean;
            }

            for (std::size_t j = 0; j < p; ++j) {
                double rho = 0.0;
                double denominator = 0.0;

                for (std::size_t i = 0; i < n; ++i) {
                    residuals[i] +=
                        X[i][j] * coefficients_[j];

                    rho += X[i][j] * residuals[i];
                    denominator += X[i][j] * X[i][j];
                }

                denominator +=
                    static_cast<double>(n) * l2;

                coefficients_[j] =
                    denominator == 0.0
                        ? 0.0
                        : softThreshold(
                              rho,
                              static_cast<double>(n) * l1 / 2.0
                          ) /
                          denominator;

                for (std::size_t i = 0; i < n; ++i) {
                    residuals[i] -=
                        X[i][j] * coefficients_[j];
                }
            }

            double largestChange = 0.0;

            for (std::size_t j = 0; j < p; ++j) {
                largestChange = std::max(
                    largestChange,
                    std::abs(
                        coefficients_[j] -
                        oldCoefficients[j]
                    )
                );
            }

            if (largestChange < tolerance_) {
                break;
            }
        }
    }
};

Dataset generateDemandData(
    std::size_t observations,
    unsigned int seed
) {
    std::mt19937 generator(seed);

    std::normal_distribution<double> temperatureNoise(0.0, 4.0);
    std::normal_distribution<double> humidityNoise(0.0, 10.0);
    std::normal_distribution<double> pressureNoise(0.0, 3.0);
    std::normal_distribution<double> priceNoise(0.0, 7.0);
    std::normal_distribution<double> trafficNoise(0.0, 100.0);
    std::normal_distribution<double> targetNoise(0.0, 8.0);
    std::normal_distribution<double> standardNoise(0.0, 1.0);
    std::uniform_real_distribution<double> advertising(0.0, 100.0);
    std::uniform_real_distribution<double> discount(0.0, 30.0);
    std::bernoulli_distribution holiday(0.18);

    Dataset data;

    data.X.reserve(observations);
    data.y.reserve(observations);

    for (std::size_t i = 0; i < observations; ++i) {
        const double temperature =
            22.0 + temperatureNoise(generator);

        const double humidity =
            60.0 + humidityNoise(generator);

        // Pressure intentionally depends on temperature, creating
        // multicollinearity that exposes differences between penalties.
        const double pressure =
            1000.0 +
            1.8 * temperature +
            pressureNoise(generator);

        const double advertisingSpend =
            advertising(generator);

        const double discountRate =
            discount(generator);

        const double holidayFlag =
            holiday(generator) ? 1.0 : 0.0;

        const double competitorPrice =
            50.0 + priceNoise(generator);

        const double storeTraffic =
            500.0 + trafficNoise(generator);

        const double noiseA = standardNoise(generator);
        const double noiseB = standardNoise(generator);
        const double noiseC = standardNoise(generator);
        const double noiseD = standardNoise(generator);

        data.X.push_back({
            temperature,
            humidity,
            pressure,
            advertisingSpend,
            discountRate,
            holidayFlag,
            competitorPrice,
            storeTraffic,
            noiseA,
            noiseB,
            noiseC,
            noiseD
        });

        const double target =
            80.0 +
            2.8 * temperature -
            0.9 * humidity +
            0.6 * advertisingSpend +
            1.2 * discountRate +
            18.0 * holidayFlag -
            0.5 * competitorPrice +
            0.04 * storeTraffic +
            targetNoise(generator);

        data.y.push_back(target);
    }

    return data;
}

std::pair<Dataset, Dataset> split(
    const Dataset& data,
    double testRatio,
    unsigned int seed
) {
    std::vector<std::size_t> indices(data.X.size());

    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);

    const std::size_t testSize =
        std::max<std::size_t>(
            1,
            static_cast<std::size_t>(
                std::round(data.X.size() * testRatio)
            )
        );

    Dataset train;
    Dataset test;

    for (std::size_t position = 0; position < indices.size(); ++position) {
        const std::size_t sourceIndex = indices[position];

        if (position < testSize) {
            test.X.push_back(data.X[sourceIndex]);
            test.y.push_back(data.y[sourceIndex]);
        } else {
            train.X.push_back(data.X[sourceIndex]);
            train.y.push_back(data.y[sourceIndex]);
        }
    }

    return {train, test};
}

Metrics evaluate(
    RegressionModel& model,
    const Dataset& train,
    const Dataset& test
) {
    model.fit(train.X, train.y);

    const Vector trainPredictions =
        model.predict(train.X);

    const Vector testPredictions =
        model.predict(test.X);

    Metrics metrics;

    metrics.trainRmse =
        rmse(train.y, trainPredictions);

    metrics.testRmse =
        rmse(test.y, testPredictions);

    metrics.trainR2 =
        r2(train.y, trainPredictions);

    metrics.testR2 =
        r2(test.y, testPredictions);

    metrics.zeroCoefficients =
        static_cast<std::size_t>(
            std::count_if(
                model.coefficients().begin(),
                model.coefficients().end(),
                [](double value) {
                    return std::abs(value) < 1e-8;
                }
            )
        );

    return metrics;
}

void printMetrics(
    const std::string& name,
    const Metrics& metrics
) {
    std::cout
        << std::left
        << std::setw(14)
        << name
        << "train RMSE=" << std::setw(9)
        << std::fixed << std::setprecision(4)
        << metrics.trainRmse
        << " test RMSE=" << std::setw(9)
        << metrics.testRmse
        << " test R2=" << std::setw(9)
        << metrics.testR2
        << " zeros="
        << metrics.zeroCoefficients
        << '\n';
}

double crossValidateRidge(
    const Matrix& X,
    const Vector& y,
    double alpha,
    int folds
) {
    double totalError = 0.0;

    const std::size_t foldSize =
        X.size() / static_cast<std::size_t>(folds);

    for (int fold = 0; fold < folds; ++fold) {
        const std::size_t begin =
            static_cast<std::size_t>(fold) * foldSize;

        const std::size_t end =
            fold == folds - 1
                ? X.size()
                : begin + foldSize;

        Matrix trainX;
        Vector trainY;
        Matrix validationX;
        Vector validationY;

        for (std::size_t i = 0; i < X.size(); ++i) {
            if (i >= begin && i < end) {
                validationX.push_back(X[i]);
                validationY.push_back(y[i]);
            } else {
                trainX.push_back(X[i]);
                trainY.push_back(y[i]);
            }
        }

        RidgeRegression model(alpha);
        model.fit(trainX, trainY);

        totalError +=
            mse(validationY, model.predict(validationX));
    }

    return totalError / static_cast<double>(folds);
}

int main() {
    try {
        std::cout << "REGULARIZATION CASE STUDY\n";
        std::cout << "=========================\n\n";

        const Dataset raw =
            generateDemandData(180, 42);

        auto [trainRaw, testRaw] =
            split(raw, 0.25, 19);

        StandardScaler scaler;
        scaler.fit(trainRaw.X);

        const Matrix trainX =
            scaler.transform(trainRaw.X);

        const Matrix testX =
            scaler.transform(testRaw.X);

        const Dataset train{trainX, trainRaw.y};
        const Dataset test{testX, testRaw.y};

        RidgeRegression ridge(0.8);
        LassoRegression lasso(0.08);
        ElasticNetRegression elasticNet(0.08, 0.5);

        const Metrics ridgeMetrics =
            evaluate(ridge, train, test);

        const Metrics lassoMetrics =
            evaluate(lasso, train, test);

        const Metrics elasticMetrics =
            evaluate(elasticNet, train, test);

        printMetrics("Ridge", ridgeMetrics);
        printMetrics("Lasso", lassoMetrics);
        printMetrics("Elastic Net", elasticMetrics);

        std::cout << "\nRidge alpha search\n";

        const Vector alphaGrid{
            0.001, 0.01, 0.03, 0.1,
            0.3, 1.0, 3.0, 10.0
        };

        double bestAlpha = alphaGrid.front();
        double bestError =
            std::numeric_limits<double>::infinity();

        for (double alpha : alphaGrid) {
            const double error =
                crossValidateRidge(
                    train.X,
                    train.y,
                    alpha,
                    5
                );

            std::cout
                << "alpha="
                << std::setw(7)
                << alpha
                << " validation MSE="
                << std::fixed
                << std::setprecision(5)
                << error
                << '\n';

            if (error < bestError) {
                bestError = error;
                bestAlpha = alpha;
            }
        }

        std::cout
            << "Selected Ridge alpha: "
            << bestAlpha
            << "\n";

        std::cout << "\nLasso regularization path\n";

        for (double alpha :
             Vector{0.005, 0.02, 0.05, 0.1, 0.2, 0.5}) {
            LassoRegression model(alpha);
            model.fit(train.X, train.y);

            const auto active =
                std::count_if(
                    model.coefficients().begin(),
                    model.coefficients().end(),
                    [](double value) {
                        return std::abs(value) > 1e-8;
                    }
                );

            std::cout
                << "alpha="
                << std::setw(6)
                << alpha
                << " active coefficients="
                << active
                << '\n';
        }

        std::cout << "\nCoefficient comparison\n";

        const std::vector<std::string> featureNames{
            "temperature",
            "humidity",
            "pressure",
            "advertising",
            "discount",
            "holiday",
            "competitor_price",
            "store_traffic",
            "noise_a",
            "noise_b",
            "noise_c",
            "noise_d"
        };

        LassoRegression finalLasso(0.08);
        finalLasso.fit(train.X, train.y);

        ElasticNetRegression finalElastic(0.08, 0.5);
        finalElastic.fit(train.X, train.y);

        RidgeRegression finalRidge(0.8);
        finalRidge.fit(train.X, train.y);

        std::cout
            << std::left
            << std::setw(22)
            << "Feature"
            << std::setw(15)
            << "Ridge"
            << std::setw(15)
            << "Lasso"
            << std::setw(15)
            << "ElasticNet"
            << '\n';

        for (std::size_t j = 0; j < featureNames.size(); ++j) {
            std::cout
                << std::setw(22)
                << featureNames[j]
                << std::setw(15)
                << finalRidge.coefficients()[j]
                << std::setw(15)
                << finalLasso.coefficients()[j]
                << std::setw(15)
                << finalElastic.coefficients()[j]
                << '\n';
        }

        std::cout << "\nEdge-case validation\n";

        try {
            RidgeRegression invalid(-1.0);
        } catch (const std::exception& error) {
            std::cout
                << "Negative Ridge alpha rejected: "
                << error.what()
                << '\n';
        }

        try {
            ElasticNetRegression invalid(0.1, 1.5);
        } catch (const std::exception& error) {
            std::cout
                << "Invalid Elastic Net ratio rejected: "
                << error.what()
                << '\n';
        }

        std::cout << "\nCase-study interpretation\n";
        std::cout
            << "The model predicts operational demand from correlated "
               "environmental, commercial, and traffic features.\n";
        std::cout
            << "Ridge reduces coefficient magnitude without intentionally "
               "creating a sparse feature set.\n";
        std::cout
            << "Lasso can discard weak predictors by driving coefficients "
               "exactly to zero.\n";
        std::cout
            << "Elastic Net combines sparsity with the stabilizing effect "
               "of an L2 penalty when predictors are correlated.\n";
        std::cout
            << "The regularization penalty introduces bias but can reduce "
               "estimation variance enough to improve test performance.\n";
        std::cout
            << "Feature scaling is essential because the penalties operate "
               "on coefficient magnitudes.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << '\n';

        return 1;
    }
}
