#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

/*
 * Exploratory Data Analysis Case Study
 *
 * Scenario:
 * A commerce organization is evaluating sales transactions before building
 * a revenue forecasting model. The EDA engine examines the same operational
 * data at three levels:
 *
 * Univariate:
 *   What does each individual variable look like?
 *
 * Bivariate:
 *   What relationship exists between two variables?
 *
 * Multivariate:
 *   What happens when several variables are considered jointly?
 *
 * The implementation intentionally models an analytical engine rather than
 * isolated C++ syntax. It includes:
 * - missing-value representation
 * - numerical summaries
 * - categorical frequency analysis
 * - IQR outlier detection
 * - covariance
 * - Pearson correlation
 * - simple linear regression
 * - correlation matrices
 * - cross-tabulation
 * - multivariate regression with Gaussian elimination
 * - business-rule validation
 * - merge-style analytical interpretation
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic eda_case_study.cpp -o eda
 */

struct SaleRecord {
    int transactionId;
    std::string region;
    std::string channel;
    std::string product;

    std::optional<double> marketingSpend;
    std::optional<double> discountPct;
    std::optional<double> unitsSold;
    std::optional<double> customerRating;
    std::optional<double> deliveryDays;
    std::optional<double> revenue;

    bool returned;
};

struct NumericSummary {
    std::size_t count = 0;
    std::size_t missing = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double standardDeviation = 0.0;
    double q1 = 0.0;
    double q3 = 0.0;
    double iqr = 0.0;
};

struct RegressionResult {
    double intercept = 0.0;
    double slope = 0.0;
    double correlation = 0.0;
    double rSquared = 0.0;
    double residualStandardError = 0.0;
};

struct MultipleRegressionResult {
    double intercept = 0.0;
    std::vector<double> coefficients;
    double rSquared = 0.0;
    std::size_t observations = 0;
};

class EDAEngine {
private:
    std::vector<SaleRecord> records;
    std::mt19937 generator;

    static double normalCDF(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    static double mean(const std::vector<double>& values) {
        if (values.empty()) {
            throw std::invalid_argument(
                "Mean requires at least one observation."
            );
        }

        return std::accumulate(
            values.begin(),
            values.end(),
            0.0
        ) / static_cast<double>(values.size());
    }

    static double median(std::vector<double> values) {
        if (values.empty()) {
            throw std::invalid_argument(
                "Median requires at least one observation."
            );
        }

        std::sort(values.begin(), values.end());

        const std::size_t middle = values.size() / 2;

        if (values.size() % 2 == 0) {
            return (
                values[middle - 1] +
                values[middle]
            ) / 2.0;
        }

        return values[middle];
    }

    static double quantile(
        std::vector<double> values,
        double probability
    ) {
        if (values.empty()) {
            throw std::invalid_argument(
                "Quantile requires observations."
            );
        }

        std::sort(values.begin(), values.end());

        const double position =
            (static_cast<double>(values.size()) - 1.0) *
            probability;

        const std::size_t lower =
            static_cast<std::size_t>(std::floor(position));

        const std::size_t upper =
            std::min(lower + 1, values.size() - 1);

        const double fraction =
            position - static_cast<double>(lower);

        return values[lower] +
               fraction * (values[upper] - values[lower]);
    }

    static double sampleVariance(
        const std::vector<double>& values
    ) {
        if (values.size() < 2) {
            throw std::invalid_argument(
                "Sample variance requires at least two values."
            );
        }

        const double average = mean(values);

        double sum = 0.0;

        for (double value : values) {
            sum += std::pow(value - average, 2.0);
        }

        return sum /
               static_cast<double>(values.size() - 1);
    }

    static double sampleStandardDeviation(
        const std::vector<double>& values
    ) {
        return std::sqrt(sampleVariance(values));
    }

    static double covariance(
        const std::vector<double>& x,
        const std::vector<double>& y
    ) {
        if (x.size() != y.size()) {
            throw std::invalid_argument(
                "Covariance requires equal-length paired samples."
            );
        }

        if (x.size() < 2) {
            throw std::invalid_argument(
                "Covariance requires at least two observations."
            );
        }

        const double xMean = mean(x);
        const double yMean = mean(y);

        double sum = 0.0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            sum +=
                (x[i] - xMean) *
                (y[i] - yMean);
        }

        return sum /
               static_cast<double>(x.size() - 1);
    }

    static double correlation(
        const std::vector<double>& x,
        const std::vector<double>& y
    ) {
        if (x.size() != y.size()) {
            throw std::invalid_argument(
                "Correlation requires equal-length paired samples."
            );
        }

        if (x.size() < 2) {
            throw std::invalid_argument(
                "Correlation requires at least two observations."
            );
        }

        const double xMean = mean(x);
        const double yMean = mean(y);

        double numerator = 0.0;
        double xSquares = 0.0;
        double ySquares = 0.0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            const double xDeviation = x[i] - xMean;
            const double yDeviation = y[i] - yMean;

            numerator +=
                xDeviation * yDeviation;

            xSquares +=
                xDeviation * xDeviation;

            ySquares +=
                yDeviation * yDeviation;
        }

        const double denominator =
            std::sqrt(xSquares * ySquares);

        if (denominator == 0.0) {
            throw std::invalid_argument(
                "Correlation is undefined for a constant variable."
            );
        }

        return numerator / denominator;
    }

    static std::vector<double> columnValues(
        const std::vector<SaleRecord>& data,
        const std::string& column
    ) {
        std::vector<double> values;

        for (const SaleRecord& record : data) {
            std::optional<double> value;

            if (column == "marketingSpend") {
                value = record.marketingSpend;
            } else if (column == "discountPct") {
                value = record.discountPct;
            } else if (column == "unitsSold") {
                value = record.unitsSold;
            } else if (column == "customerRating") {
                value = record.customerRating;
            } else if (column == "deliveryDays") {
                value = record.deliveryDays;
            } else if (column == "revenue") {
                value = record.revenue;
            } else {
                throw std::invalid_argument(
                    "Unknown numerical column: " + column
                );
            }

            if (value.has_value()) {
                values.push_back(*value);
            }
        }

        return values;
    }

    static std::pair<
        std::vector<double>,
        std::vector<double>
    > pairedColumns(
        const std::vector<SaleRecord>& data,
        const std::string& xColumn,
        const std::string& yColumn
    ) {
        std::vector<double> x;
        std::vector<double> y;

        for (const SaleRecord& record : data) {
            std::optional<double> xValue;
            std::optional<double> yValue;

            auto getValue = [](
                const SaleRecord& current,
                const std::string& column
            ) -> std::optional<double> {
                if (column == "marketingSpend") {
                    return current.marketingSpend;
                }
                if (column == "discountPct") {
                    return current.discountPct;
                }
                if (column == "unitsSold") {
                    return current.unitsSold;
                }
                if (column == "customerRating") {
                    return current.customerRating;
                }
                if (column == "deliveryDays") {
                    return current.deliveryDays;
                }
                if (column == "revenue") {
                    return current.revenue;
                }

                throw std::invalid_argument(
                    "Unknown numerical column: " + column
                );
            };

            xValue = getValue(record, xColumn);
            yValue = getValue(record, yColumn);

            if (xValue.has_value() && yValue.has_value()) {
                x.push_back(*xValue);
                y.push_back(*yValue);
            }
        }

        return {x, y};
    }

    static RegressionResult linearRegression(
        const std::vector<double>& x,
        const std::vector<double>& y
    ) {
        if (x.size() != y.size() || x.size() < 3) {
            throw std::invalid_argument(
                "Linear regression requires at least three paired observations."
            );
        }

        const double xMean = mean(x);
        const double yMean = mean(y);

        double numerator = 0.0;
        double denominator = 0.0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            numerator +=
                (x[i] - xMean) *
                (y[i] - yMean);

            denominator +=
                std::pow(x[i] - xMean, 2.0);
        }

        if (denominator == 0.0) {
            throw std::invalid_argument(
                "Regression predictor has zero variance."
            );
        }

        const double slope =
            numerator / denominator;

        const double intercept =
            yMean - slope * xMean;

        const double correlationValue =
            correlation(x, y);

        double residualSumSquares = 0.0;
        double totalSumSquares = 0.0;

        for (std::size_t i = 0; i < x.size(); ++i) {
            const double prediction =
                intercept + slope * x[i];

            residualSumSquares +=
                std::pow(y[i] - prediction, 2.0);

            totalSumSquares +=
                std::pow(y[i] - yMean, 2.0);
        }

        const double residualStandardError =
            std::sqrt(
                residualSumSquares /
                static_cast<double>(x.size() - 2)
            );

        return {
            intercept,
            slope,
            correlationValue,
            correlationValue * correlationValue,
            residualStandardError
        };
    }

    /*
     * Solve A*x=b using Gaussian elimination with partial pivoting.
     *
     * Partial pivoting is important because a naïve implementation that
     * always divides by the current diagonal element can become numerically
     * unstable when the matrix contains small pivot values.
     */
    static std::vector<double> solveLinearSystem(
        std::vector<std::vector<double>> matrix,
        std::vector<double> rightHandSide
    ) {
        const std::size_t n = rightHandSide.size();

        for (std::size_t pivot = 0; pivot < n; ++pivot) {
            std::size_t bestRow = pivot;

            for (std::size_t row = pivot + 1; row < n; ++row) {
                if (
                    std::abs(matrix[row][pivot]) >
                    std::abs(matrix[bestRow][pivot])
                ) {
                    bestRow = row;
                }
            }

            if (
                std::abs(matrix[bestRow][pivot]) <
                1e-12
            ) {
                throw std::runtime_error(
                    "Multivariate regression matrix is singular or ill-conditioned."
                );
            }

            std::swap(
                matrix[pivot],
                matrix[bestRow]
            );

            std::swap(
                rightHandSide[pivot],
                rightHandSide[bestRow]
            );

            for (std::size_t row = pivot + 1; row < n; ++row) {
                const double factor =
                    matrix[row][pivot] /
                    matrix[pivot][pivot];

                for (
                    std::size_t column = pivot;
                    column < n;
                    ++column
                ) {
                    matrix[row][column] -=
                        factor * matrix[pivot][column];
                }

                rightHandSide[row] -=
                    factor * rightHandSide[pivot];
            }
        }

        std::vector<double> solution(n, 0.0);

        for (std::size_t reverse = n; reverse-- > 0;) {
            double value = rightHandSide[reverse];

            for (
                std::size_t column = reverse + 1;
                column < n;
                ++column
            ) {
                value -=
                    matrix[reverse][column] *
                    solution[column];
            }

            solution[reverse] =
                value / matrix[reverse][reverse];
        }

        return solution;
    }

    MultipleRegressionResult multipleRegression(
        const std::vector<std::string>& featureColumns,
        const std::string& targetColumn
    ) const {
        struct Observation {
            std::vector<double> features;
            double target;
        };

        std::vector<Observation> observations;

        for (const SaleRecord& record : records) {
            std::vector<double> features;
            bool complete = true;

            auto getValue = [&record](
                const std::string& column
            ) -> std::optional<double> {
                if (column == "marketingSpend") {
                    return record.marketingSpend;
                }
                if (column == "discountPct") {
                    return record.discountPct;
                }
                if (column == "unitsSold") {
                    return record.unitsSold;
                }
                if (column == "customerRating") {
                    return record.customerRating;
                }
                if (column == "deliveryDays") {
                    return record.deliveryDays;
                }
                if (column == "revenue") {
                    return record.revenue;
                }

                throw std::invalid_argument(
                    "Unknown regression column: " + column
                );
            };

            for (const std::string& column : featureColumns) {
                const auto value = getValue(column);

                if (!value.has_value()) {
                    complete = false;
                    break;
                }

                features.push_back(*value);
            }

            const auto target =
                getValue(targetColumn);

            if (!target.has_value()) {
                complete = false;
            }

            if (complete) {
                observations.push_back({
                    features,
                    *target
                });
            }
        }

        if (
            observations.size() <
            featureColumns.size() + 3
        ) {
            throw std::runtime_error(
                "Too few complete observations for multivariate regression."
            );
        }

        /*
         * X contains an intercept column followed by predictors.
         * The normal equation is:
         *
         *     beta = (X'X)^-1 X'y
         *
         * Rather than explicitly computing an inverse, the implementation
         * solves (X'X) beta = X'y. This avoids an unnecessary matrix inverse.
         */
        const std::size_t parameterCount =
            featureColumns.size() + 1;

        std::vector<std::vector<double>> xtx(
            parameterCount,
            std::vector<double>(
                parameterCount,
                0.0
            )
        );

        std::vector<double> xty(
            parameterCount,
            0.0
        );

        for (const Observation& observation : observations) {
            std::vector<double> designRow;
            designRow.push_back(1.0);

            designRow.insert(
                designRow.end(),
                observation.features.begin(),
                observation.features.end()
            );

            for (std::size_t i = 0; i < parameterCount; ++i) {
                xty[i] +=
                    designRow[i] *
                    observation.target;

                for (
                    std::size_t j = 0;
                    j < parameterCount;
                    ++j
                ) {
                    xtx[i][j] +=
                        designRow[i] *
                        designRow[j];
                }
            }
        }

        const std::vector<double> coefficients =
            solveLinearSystem(
                xtx,
                xty
            );

        const std::vector<double> targets = [&]() {
            std::vector<double> values;

            for (const auto& observation : observations) {
                values.push_back(
                    observation.target
                );
            }

            return values;
        }();

        const double targetMean =
            mean(targets);

        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;

        for (const Observation& observation : observations) {
            double prediction =
                coefficients[0];

            for (
                std::size_t i = 0;
                i < observation.features.size();
                ++i
            ) {
                prediction +=
                    coefficients[i + 1] *
                    observation.features[i];
            }

            residualSumSquares +=
                std::pow(
                    observation.target - prediction,
                    2.0
                );

            totalSumSquares +=
                std::pow(
                    observation.target - targetMean,
                    2.0
                );
        }

        const double rSquared =
            totalSumSquares == 0.0
                ? 0.0
                : 1.0 -
                  residualSumSquares /
                  totalSumSquares;

        return {
            coefficients[0],
            std::vector<double>(
                coefficients.begin() + 1,
                coefficients.end()
            ),
            rSquared,
            observations.size()
        };
    }

    void printTitle(
        const std::string& title
    ) const {
        std::cout
            << "\n"
            << std::string(78, '=')
            << "\n"
            << title
            << "\n"
            << std::string(78, '=')
            << "\n";
    }

public:
    explicit EDAEngine(
        std::vector<SaleRecord> data,
        unsigned int seed = 42
    )
        : records(std::move(data)),
          generator(seed) {}

    void datasetProfile() const {
        printTitle("DATASET PROFILE");

        std::cout
            << "Observations: "
            << records.size()
            << "\n";

        const std::vector<std::string> numericalColumns = {
            "marketingSpend",
            "discountPct",
            "unitsSold",
            "customerRating",
            "deliveryDays",
            "revenue"
        };

        std::cout
            << std::left
            << std::setw(24)
            << "Variable"
            << std::right
            << std::setw(12)
            << "Observed"
            << std::setw(12)
            << "Missing"
            << "\n";

        for (const std::string& column : numericalColumns) {
            const auto values =
                columnValues(records, column);

            std::cout
                << std::left
                << std::setw(24)
                << column
                << std::right
                << std::setw(12)
                << values.size()
                << std::setw(12)
                << records.size() - values.size()
                << "\n";
        }
    }

    void univariateAnalysis() const {
        printTitle("UNIVARIATE ANALYSIS");

        const std::vector<std::string> columns = {
            "marketingSpend",
            "discountPct",
            "unitsSold",
            "customerRating",
            "deliveryDays",
            "revenue"
        };

        std::cout
            << std::left
            << std::setw(24)
            << "Variable"
            << std::right
            << std::setw(8)
            << "N"
            << std::setw(13)
            << "Mean"
            << std::setw(13)
            << "Median"
            << std::setw(13)
            << "StdDev"
            << std::setw(13)
            << "Q1"
            << std::setw(13)
            << "Q3"
            << "\n";

        for (const std::string& column : columns) {
            const auto values =
                columnValues(records, column);

            const double q1 =
                quantile(values, 0.25);

            const double q3 =
                quantile(values, 0.75);

            std::cout
                << std::left
                << std::setw(24)
                << column
                << std::right
                << std::setw(8)
                << values.size()
                << std::setw(13)
                << std::fixed
                << std::setprecision(2)
                << mean(values)
                << std::setw(13)
                << median(values)
                << std::setw(13)
                << sampleStandardDeviation(values)
                << std::setw(13)
                << q1
                << std::setw(13)
                << q3
                << "\n";
        }

        std::map<std::string, int> regionCounts;
        std::map<std::string, int> channelCounts;

        for (const SaleRecord& record : records) {
            ++regionCounts[record.region];
            ++channelCounts[record.channel];
        }

        std::cout
            << "\nRegion frequency distribution:\n";

        for (const auto& [region, count] : regionCounts) {
            std::cout
                << "  "
                << region
                << ": "
                << count
                << "\n";
        }

        std::cout
            << "\nChannel frequency distribution:\n";

        for (const auto& [channel, count] : channelCounts) {
            std::cout
                << "  "
                << channel
                << ": "
                << count
                << "\n";
        }
    }

    void outlierAnalysis() const {
        printTitle("OUTLIER ANALYSIS");

        const std::vector<std::string> columns = {
            "marketingSpend",
            "discountPct",
            "unitsSold",
            "customerRating",
            "deliveryDays",
            "revenue"
        };

        for (const std::string& column : columns) {
            const auto values =
                columnValues(records, column);

            const double q1 =
                quantile(values, 0.25);

            const double q3 =
                quantile(values, 0.75);

            const double iqr =
                q3 - q1;

            const double lowerFence =
                q1 - 1.5 * iqr;

            const double upperFence =
                q3 + 1.5 * iqr;

            std::size_t count = 0;

            for (double value : values) {
                if (
                    value < lowerFence ||
                    value > upperFence
                ) {
                    ++count;
                }
            }

            std::cout
                << std::left
                << std::setw(24)
                << column
                << "IQR outliers: "
                << count
                << " | lower="
                << std::fixed
                << std::setprecision(2)
                << lowerFence
                << " | upper="
                << upperFence
                << "\n";
        }

        std::cout
            << "\nAn IQR outlier is an analytical signal, not an automatic "
               "data-cleaning instruction. A valid large transaction can be "
               "commercially important and should not be deleted merely "
               "because it is statistically unusual.\n";
    }

    void bivariateAnalysis() const {
        printTitle("BIVARIATE ANALYSIS");

        const auto [marketing, revenue] =
            pairedColumns(
                records,
                "marketingSpend",
                "revenue"
            );

        const double covarianceValue =
            covariance(
                marketing,
                revenue
            );

        const RegressionResult regression =
            linearRegression(
                marketing,
                revenue
            );

        std::cout
            << "Variables: marketingSpend and revenue\n"
            << "Complete pairs: "
            << marketing.size()
            << "\n"
            << "Covariance: "
            << std::fixed
            << std::setprecision(4)
            << covarianceValue
            << "\n"
            << "Pearson correlation: "
            << regression.correlation
            << "\n"
            << "Regression slope: "
            << regression.slope
            << "\n"
            << "Regression intercept: "
            << regression.intercept
            << "\n"
            << "R-squared: "
            << regression.rSquared
            << "\n"
            << "Residual standard error: "
            << regression.residualStandardError
            << "\n";

        /*
         * Grouped analysis is bivariate because the grouping variable and
         * numerical response are considered together. It is distinct from
         * the correlation calculation because no linear assumption is made
         * about a categorical grouping variable.
         */
        std::map<std::string, std::vector<double>> revenueByChannel;

        for (const SaleRecord& record : records) {
            if (record.revenue.has_value()) {
                revenueByChannel[
                    record.channel
                ].push_back(
                    *record.revenue
                );
            }
        }

        std::cout
            << "\nRevenue by sales channel:\n";

        for (const auto& [channel, values] : revenueByChannel) {
            std::cout
                << "  "
                << std::left
                << std::setw(12)
                << channel
                << "n="
                << std::setw(5)
                << values.size()
                << "mean="
                << std::fixed
                << std::setprecision(2)
                << mean(values)
                << "\n";
        }
    }

    void multivariateAnalysis() const {
        printTitle("MULTIVARIATE ANALYSIS");

        const std::vector<std::string> columns = {
            "marketingSpend",
            "discountPct",
            "unitsSold",
            "customerRating",
            "deliveryDays",
            "revenue"
        };

        std::cout
            << std::left
            << std::setw(22)
            << "Variable";

        for (const std::string& column : columns) {
            std::cout
                << std::right
                << std::setw(12)
                << column.substr(0, 10);
        }

        std::cout << "\n";

        for (const std::string& xColumn : columns) {
            std::cout
                << std::left
                << std::setw(22)
                << xColumn;

            for (const std::string& yColumn : columns) {
                const auto [x, y] =
                    pairedColumns(
                        records,
                        xColumn,
                        yColumn
                    );

                double value =
                    std::numeric_limits<double>::quiet_NaN();

                if (x.size() >= 2) {
                    try {
                        value =
                            correlation(x, y);
                    } catch (const std::exception&) {
                        value =
                            std::numeric_limits<double>::quiet_NaN();
                    }
                }

                if (std::isnan(value)) {
                    std::cout
                        << std::right
                        << std::setw(12)
                        << "NA";
                } else {
                    std::cout
                        << std::right
                        << std::setw(12)
                        << std::fixed
                        << std::setprecision(2)
                        << value;
                }
            }

            std::cout << "\n";
        }

        const std::vector<std::string> features = {
            "marketingSpend",
            "discountPct",
            "deliveryDays"
        };

        const MultipleRegressionResult model =
            multipleRegression(
                features,
                "revenue"
            );

        std::cout
            << "\nMultiple regression for revenue\n"
            << "Complete observations: "
            << model.observations
            << "\n"
            << "Intercept: "
            << model.intercept
            << "\n";

        for (
            std::size_t i = 0;
            i < features.size();
            ++i
        ) {
            std::cout
                << "Coefficient ["
                << features[i]
                << "]: "
                << model.coefficients[i]
                << "\n";
        }

        std::cout
            << "R-squared: "
            << model.rSquared
            << "\n";

        /*
         * A cross-tabulation examines the joint distribution of categorical
         * variables. It answers a different question from correlation because
         * region and channel are nominal categories rather than continuous
         * numerical measurements.
         */
        std::map<
            std::string,
            std::map<std::string, int>
        > regionChannel;

        for (const SaleRecord& record : records) {
            ++regionChannel[
                record.region
            ][record.channel];
        }

        std::cout
            << "\nRegion/channel cross-tabulation:\n";

        std::cout
            << std::left
            << std::setw(15)
            << "Region"
            << std::right
            << std::setw(12)
            << "Online"
            << std::setw(12)
            << "Retail"
            << std::setw(12)
            << "Partner"
            << "\n";

        for (const auto& [region, channels] :
             regionChannel) {
            std::cout
                << std::left
                << std::setw(15)
                << region
                << std::right
                << std::setw(12)
                << channels.at("Online")
                << std::setw(12)
                << channels.at("Retail")
                << std::setw(12)
                << channels.at("Partner")
                << "\n";
        }
    }

    void validateBusinessRules() const {
        printTitle("BUSINESS-RULE VALIDATION");

        std::size_t errorCount = 0;

        for (const SaleRecord& record : records) {
            if (
                record.customerRating.has_value() &&
                (
                    *record.customerRating < 1.0 ||
                    *record.customerRating > 5.0
                )
            ) {
                ++errorCount;

                std::cout
                    << "Transaction "
                    << record.transactionId
                    << ": customer rating outside [1,5]\n";
            }

            if (
                record.discountPct.has_value() &&
                (
                    *record.discountPct < 0.0 ||
                    *record.discountPct > 100.0
                )
            ) {
                ++errorCount;

                std::cout
                    << "Transaction "
                    << record.transactionId
                    << ": discount outside [0,100]\n";
            }

            if (
                record.deliveryDays.has_value() &&
                *record.deliveryDays < 0.0
            ) {
                ++errorCount;

                std::cout
                    << "Transaction "
                    << record.transactionId
                    << ": negative delivery time\n";
            }

            if (
                record.revenue.has_value() &&
                *record.revenue < 0.0
            ) {
                ++errorCount;

                std::cout
                    << "Transaction "
                    << record.transactionId
                    << ": negative revenue\n";
            }
        }

        if (errorCount == 0) {
            std::cout
                << "No business-rule violations detected.\n";
        } else {
            std::cout
                << "Total violations: "
                << errorCount
                << "\n";
        }
    }

    void run() const {
        datasetProfile();
        univariateAnalysis();
        outlierAnalysis();
        bivariateAnalysis();
        multivariateAnalysis();
        validateBusinessRules();

        printTitle("ANALYTICAL INTERPRETATION");

        std::cout
            << "Univariate analysis establishes the distribution, scale, "
               "missingness, and unusual observations of individual fields.\n";

        std::cout
            << "Bivariate analysis examines pairwise association or grouped "
               "differences without treating every relationship as causal.\n";

        std::cout
            << "Multivariate analysis evaluates several variables jointly, "
               "which can reveal relationships that are obscured by pairwise "
               "inspection alone.\n";

        std::cout
            << "The regression coefficients describe the fitted model under "
               "its mathematical assumptions. They should not be interpreted "
               "as causal effects without an appropriate identification design.\n";

        std::cout
            << "EDA should precede modeling because invalid ranges, missing "
               "observations, extreme values, duplicated records, and "
               "measurement inconsistencies can materially change analytical "
               "results.\n";
    }
};

// -----------------------------------------------------------------------------
// Synthetic business dataset
// -----------------------------------------------------------------------------

std::vector<SaleRecord> generateDataset(
    std::size_t size,
    unsigned int seed
) {
    std::mt19937 generator(seed);

    std::vector<std::string> regions = {
        "North",
        "South",
        "East",
        "West"
    };

    std::vector<std::string> channels = {
        "Online",
        "Retail",
        "Partner"
    };

    std::vector<std::string> products = {
        "Alpha",
        "Beta",
        "Gamma",
        "Delta"
    };

    std::map<std::string, double> prices = {
        {"Alpha", 72.0},
        {"Beta", 105.0},
        {"Gamma", 145.0},
        {"Delta", 185.0}
    };

    std::uniform_int_distribution<int> regionDistribution(
        0,
        static_cast<int>(regions.size()) - 1
    );

    std::uniform_int_distribution<int> channelDistribution(
        0,
        static_cast<int>(channels.size()) - 1
    );

    std::uniform_int_distribution<int> productDistribution(
        0,
        static_cast<int>(products.size()) - 1
    );

    std::normal_distribution<double> marketingDistribution(
        950.0,
        300.0
    );

    std::normal_distribution<double> discountDistribution(
        10.0,
        5.0
    );

    std::normal_distribution<double> unitNoise(
        0.0,
        8.0
    );

    std::normal_distribution<double> deliveryNoise(
        0.0,
        1.0
    );

    std::normal_distribution<double> ratingNoise(
        0.0,
        0.45
    );

    std::normal_distribution<double> revenueNoise(
        0.0,
        2.0
    );

    std::uniform_real_distribution<double> probability(
        0.0,
        1.0
    );

    std::vector<SaleRecord> rows;
    rows.reserve(size);

    for (std::size_t index = 0; index < size; ++index) {
        const std::string region =
            regions[regionDistribution(generator)];

        const std::string channel =
            channels[channelDistribution(generator)];

        const std::string product =
            products[productDistribution(generator)];

        double marketingSpend =
            std::max(
                50.0,
                marketingDistribution(generator)
            );

        double discountPct =
            std::clamp(
                discountDistribution(generator),
                0.0,
                30.0
            );

        double channelEffect = 0.0;

        if (channel == "Online") {
            channelEffect = 30.0;
        } else if (channel == "Retail") {
            channelEffect = 10.0;
        } else {
            channelEffect = -5.0;
        }

        const double expectedUnits =
            20.0 +
            marketingSpend * 0.018 +
            discountPct * 1.4 +
            channelEffect;

        const double units =
            std::max(
                1.0,
                std::round(
                    expectedUnits +
                    unitNoise(generator)
                )
            );

        double deliveryMean = 3.5;

        if (channel == "Retail") {
            deliveryMean = 2.5;
        } else if (channel == "Partner") {
            deliveryMean = 4.5;
        }

        const double deliveryDays =
            std::max(
                1.0,
                deliveryMean +
                deliveryNoise(generator)
            );

        const double customerRating =
            std::clamp(
                4.9 -
                deliveryDays * 0.22 +
                ratingNoise(generator),
                1.0,
                5.0
            );

        const double effectivePrice =
            prices[product] *
            (1.0 - discountPct / 100.0);

        const double revenue =
            std::max(
                0.0,
                units * effectivePrice +
                revenueNoise(generator) * prices[product]
            );

        const double returnProbability =
            std::min(
                0.65,
                0.025 +
                std::max(
                    0.0,
                    3.4 - customerRating
                ) * 0.08 +
                std::max(
                    0.0,
                    deliveryDays - 4.0
                ) * 0.015
            );

        const bool returned =
            probability(generator) <
            returnProbability;

        rows.push_back({
            static_cast<int>(index + 1),
            region,
            channel,
            product,
            marketingSpend,
            discountPct,
            units,
            customerRating,
            deliveryDays,
            revenue,
            returned
        });
    }

    /*
     * Missing values are explicit optional values rather than magic numbers.
     * Using zero for a missing rating or spend would incorrectly change the
     * distribution and bias statistics.
     */
    if (rows.size() >= 20) {
        rows[17].customerRating.reset();
        rows[53].marketingSpend.reset();
        rows[111].deliveryDays.reset();
    }

    /*
     * A genuine extreme observation is included to show why outlier
     * detection must be separated from automatic deletion.
     */
    if (rows.size() >= 100) {
        rows[99].marketingSpend = 4800.0;
        rows[99].unitsSold = 170.0;
        rows[99].revenue = 22000.0;
    }

    return rows;
}

// -----------------------------------------------------------------------------
// Program entry point
// -----------------------------------------------------------------------------

int main() {
    try {
        const std::vector<SaleRecord> dataset =
            generateDataset(
                160,
                42
            );

        EDAEngine engine(
            dataset,
            42
        );

        engine.run();

        std::cout
            << "\n"
            << std::string(78, '=')
            << "\n"
            << "CASE STUDY COMPLETE"
            << "\n"
            << std::string(78, '=')
            << "\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "EDA execution failed: "
            << error.what()
            << "\n";

        return 1;
    }
}
