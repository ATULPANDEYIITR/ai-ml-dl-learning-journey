import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

public class RegularizationEnterpriseDemo {

    enum ModelType {
        RIDGE,
        LASSO,
        ELASTIC_NET
    }

    record Evaluation(
        double trainRmse,
        double testRmse,
        double trainR2,
        double testR2,
        long zeroCoefficients
    ) {}

    record PredictionRequest(
        String modelName,
        double[] features
    ) {}

    record PolicyDecision(
        ModelType model,
        double alpha,
        double l1Ratio,
        String reason
    ) {}

    static double mean(double[] values) {
        if (values.length == 0) {
            throw new IllegalArgumentException("Empty numeric array");
        }

        double sum = 0.0;

        for (double value : values) {
            sum += value;
        }

        return sum / values.length;
    }

    static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                "Feature dimensions do not match"
            );
        }

        double result = 0.0;

        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }

        return result;
    }

    static void validate(double[][] X, double[] y) {
        if (X.length == 0 || y.length == 0) {
            throw new IllegalArgumentException(
                "Training data cannot be empty"
            );
        }

        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "Feature and target row counts differ"
            );
        }

        int width = X[0].length;

        if (width == 0) {
            throw new IllegalArgumentException(
                "At least one feature is required"
            );
        }

        for (double[] row : X) {
            if (row.length != width) {
                throw new IllegalArgumentException(
                    "Feature rows have inconsistent widths"
                );
            }

            for (double value : row) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Features must be finite"
                    );
                }
            }
        }
    }

    static double mse(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException(
                "Prediction length mismatch"
            );
        }

        double sum = 0.0;

        for (int i = 0; i < actual.length; i++) {
            double error = actual[i] - predicted[i];
            sum += error * error;
        }

        return sum / actual.length;
    }

    static double rmse(double[] actual, double[] predicted) {
        return Math.sqrt(mse(actual, predicted));
    }

    static double r2(double[] actual, double[] predicted) {
        double baseline = mean(actual);
        double total = 0.0;
        double residual = 0.0;

        for (int i = 0; i < actual.length; i++) {
            total += Math.pow(actual[i] - baseline, 2);
            residual += Math.pow(actual[i] - predicted[i], 2);
        }

        return total == 0.0
            ? (residual == 0.0 ? 1.0 : 0.0)
            : 1.0 - residual / total;
    }

    static double softThreshold(
        double value,
        double threshold
    ) {
        if (value > threshold) {
            return value - threshold;
        }

        if (value < -threshold) {
            return value + threshold;
        }

        return 0.0;
    }

    static class StandardScaler {
        private double[] means;
        private double[] scales;

        StandardScaler fit(double[][] X) {
            int features = X[0].length;

            means = new double[features];
            scales = new double[features];

            for (double[] row : X) {
                for (int j = 0; j < features; j++) {
                    means[j] += row[j];
                }
            }

            for (int j = 0; j < features; j++) {
                means[j] /= X.length;
            }

            for (double[] row : X) {
                for (int j = 0; j < features; j++) {
                    double difference = row[j] - means[j];
                    scales[j] += difference * difference;
                }
            }

            for (int j = 0; j < features; j++) {
                scales[j] =
                    Math.sqrt(scales[j] / X.length);

                if (scales[j] < 1e-12) {
                    scales[j] = 1.0;
                }
            }

            return this;
        }

        double[][] transform(double[][] X) {
            if (means == null) {
                throw new IllegalStateException(
                    "Scaler has not been fitted"
                );
            }

            double[][] result = new double[X.length][];

            for (int i = 0; i < X.length; i++) {
                result[i] = new double[X[i].length];

                for (int j = 0; j < X[i].length; j++) {
                    result[i][j] =
                        (X[i][j] - means[j]) / scales[j];
                }
            }

            return result;
        }
    }

    interface RegressionModel {
        void fit(double[][] X, double[] y);

        double[] predict(double[][] X);

        double[] coefficients();

        double intercept();
    }

    static abstract class AbstractRegressionModel
        implements RegressionModel {

        protected double intercept;
        protected double[] coefficients;

        @Override
        public double[] predict(double[][] X) {
            double[] result = new double[X.length];

            for (int i = 0; i < X.length; i++) {
                result[i] =
                    intercept +
                    dot(X[i], coefficients);
            }

            return result;
        }

        @Override
        public double[] coefficients() {
            return coefficients.clone();
        }

        @Override
        public double intercept() {
            return intercept;
        }
    }

    static final class RidgeModel
        extends AbstractRegressionModel {

        private final double alpha;
        private final double learningRate;
        private final int maxIterations;

        RidgeModel(
            double alpha,
            double learningRate,
            int maxIterations
        ) {
            if (alpha < 0.0) {
                throw new IllegalArgumentException(
                    "Ridge alpha cannot be negative"
                );
            }

            this.alpha = alpha;
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
        }

        @Override
        public void fit(double[][] X, double[] y) {
            validate(X, y);

            coefficients =
                new double[X[0].length];

            intercept = mean(y);

            double previousLoss =
                Double.POSITIVE_INFINITY;

            for (int iteration = 0;
                 iteration < maxIterations;
                 iteration++) {

                double[] errors =
                    new double[X.length];

                for (int i = 0; i < X.length; i++) {
                    errors[i] =
                        intercept +
                        dot(X[i], coefficients) -
                        y[i];
                }

                intercept -=
                    learningRate *
                    2.0 *
                    mean(errors);

                for (int j = 0;
                     j < coefficients.length;
                     j++) {

                    double gradient = 0.0;

                    for (int i = 0; i < X.length; i++) {
                        gradient +=
                            errors[i] * X[i][j];
                    }

                    gradient =
                        2.0 * gradient / X.length +
                        2.0 * alpha * coefficients[j];

                    coefficients[j] -=
                        learningRate * gradient;
                }

                double loss = 0.0;

                for (double error : errors) {
                    loss += error * error;
                }

                loss /= X.length;

                for (double coefficient : coefficients) {
                    loss +=
                        alpha *
                        coefficient *
                        coefficient;
                }

                if (Math.abs(previousLoss - loss) < 1e-9) {
                    break;
                }

                previousLoss = loss;
            }
        }
    }

    static final class LassoModel
        extends AbstractRegressionModel {

        private final double alpha;
        private final int maxIterations;

        LassoModel(
            double alpha,
            int maxIterations
        ) {
            if (alpha < 0.0) {
                throw new IllegalArgumentException(
                    "Lasso alpha cannot be negative"
                );
            }

            this.alpha = alpha;
            this.maxIterations = maxIterations;
        }

        @Override
        public void fit(double[][] X, double[] y) {
            validate(X, y);

            int n = X.length;
            int p = X[0].length;

            coefficients = new double[p];
            intercept = mean(y);

            double[] residuals =
                new double[n];

            for (int i = 0; i < n; i++) {
                residuals[i] =
                    y[i] - intercept;
            }

            for (int iteration = 0;
                 iteration < maxIterations;
                 iteration++) {

                double[] old =
                    coefficients.clone();

                double residualMean =
                    mean(residuals);

                intercept += residualMean;

                for (int i = 0; i < n; i++) {
                    residuals[i] -= residualMean;
                }

                for (int j = 0; j < p; j++) {
                    double rho = 0.0;
                    double denominator = 0.0;

                    for (int i = 0; i < n; i++) {
                        residuals[i] +=
                            X[i][j] *
                            coefficients[j];

                        rho +=
                            X[i][j] *
                            residuals[i];

                        denominator +=
                            X[i][j] *
                            X[i][j];
                    }

                    coefficients[j] =
                        denominator == 0.0
                            ? 0.0
                            : softThreshold(
                                rho,
                                alpha * n / 2.0
                            ) /
                            denominator;

                    for (int i = 0; i < n; i++) {
                        residuals[i] -=
                            X[i][j] *
                            coefficients[j];
                    }
                }

                double largestChange = 0.0;

                for (int j = 0; j < p; j++) {
                    largestChange =
                        Math.max(
                            largestChange,
                            Math.abs(
                                coefficients[j] -
                                old[j]
                            )
                        );
                }

                if (largestChange < 1e-7) {
                    break;
                }
            }
        }
    }

    static final class ElasticNetModel
        extends AbstractRegressionModel {

        private final double alpha;
        private final double l1Ratio;
        private final int maxIterations;

        ElasticNetModel(
            double alpha,
            double l1Ratio,
            int maxIterations
        ) {
            if (alpha < 0.0) {
                throw new IllegalArgumentException(
                    "Elastic Net alpha cannot be negative"
                );
            }

            if (l1Ratio < 0.0 || l1Ratio > 1.0) {
                throw new IllegalArgumentException(
                    "l1Ratio must be between zero and one"
                );
            }

            this.alpha = alpha;
            this.l1Ratio = l1Ratio;
            this.maxIterations = maxIterations;
        }

        @Override
        public void fit(double[][] X, double[] y) {
            validate(X, y);

            int n = X.length;
            int p = X[0].length;

            coefficients = new double[p];
            intercept = mean(y);

            double[] residuals =
                new double[n];

            for (int i = 0; i < n; i++) {
                residuals[i] =
                    y[i] - intercept;
            }

            double l1 = alpha * l1Ratio;
            double l2 = alpha * (1.0 - l1Ratio);

            for (int iteration = 0;
                 iteration < maxIterations;
                 iteration++) {

                double[] old =
                    coefficients.clone();

                double residualMean =
                    mean(residuals);

                intercept += residualMean;

                for (int i = 0; i < n; i++) {
                    residuals[i] -= residualMean;
                }

                for (int j = 0; j < p; j++) {
                    double rho = 0.0;
                    double denominator = 0.0;

                    for (int i = 0; i < n; i++) {
                        residuals[i] +=
                            X[i][j] *
                            coefficients[j];

                        rho +=
                            X[i][j] *
                            residuals[i];

                        denominator +=
                            X[i][j] *
                            X[i][j];
                    }

                    denominator += n * l2;

                    coefficients[j] =
                        denominator == 0.0
                            ? 0.0
                            : softThreshold(
                                rho,
                                n * l1 / 2.0
                            ) /
                            denominator;

                    for (int i = 0; i < n; i++) {
                        residuals[i] -=
                            X[i][j] *
                            coefficients[j];
                    }
                }

                double largestChange = 0.0;

                for (int j = 0; j < p; j++) {
                    largestChange =
                        Math.max(
                            largestChange,
                            Math.abs(
                                coefficients[j] -
                                old[j]
                            )
                        );
                }

                if (largestChange < 1e-7) {
                    break;
                }
            }
        }
    }

    static double[][] generateFeatures(
        int observations,
        Random random
    ) {
        double[][] X =
            new double[observations][12];

        for (int i = 0; i < observations; i++) {
            double temperature =
                22 + random.nextGaussian() * 4;

            double humidity =
                60 + random.nextGaussian() * 10;

            double pressure =
                1000 +
                temperature * 1.8 +
                random.nextGaussian() * 3;

            X[i][0] = temperature;
            X[i][1] = humidity;
            X[i][2] = pressure;
            X[i][3] = random.nextDouble() * 100;
            X[i][4] = random.nextDouble() * 30;
            X[i][5] =
                random.nextDouble() < 0.18
                    ? 1.0
                    : 0.0;
            X[i][6] =
                50 + random.nextGaussian() * 7;
            X[i][7] =
                500 + random.nextGaussian() * 100;
            X[i][8] = random.nextGaussian();
            X[i][9] = random.nextGaussian();
            X[i][10] = random.nextGaussian();
            X[i][11] = random.nextGaussian();
        }

        return X;
    }

    static double[] generateTargets(
        double[][] X,
        Random random
    ) {
        double[] y =
            new double[X.length];

        for (int i = 0; i < X.length; i++) {
            y[i] =
                80 +
                2.8 * X[i][0] -
                0.9 * X[i][1] +
                0.6 * X[i][3] +
                1.2 * X[i][4] +
                18 * X[i][5] -
                0.5 * X[i][6] +
                0.04 * X[i][7] +
                random.nextGaussian() * 8;
        }

        return y;
    }

    static Evaluation evaluate(
        RegressionModel model,
        double[][] trainX,
        double[] trainY,
        double[][] testX,
        double[] testY
    ) {
        model.fit(trainX, trainY);

        double[] trainPrediction =
            model.predict(trainX);

        double[] testPrediction =
            model.predict(testX);

        long zeros =
            Arrays.stream(model.coefficients())
                .filter(value -> Math.abs(value) < 1e-8)
                .count();

        return new Evaluation(
            rmse(trainY, trainPrediction),
            rmse(testY, testPrediction),
            r2(trainY, trainPrediction),
            r2(testY, testPrediction),
            zeros
        );
    }

    static PolicyDecision choosePolicy(
        int observations,
        int features,
        boolean correlatedFeatures,
        boolean needSparsity
    ) {
        if (needSparsity && correlatedFeatures) {
            return new PolicyDecision(
                ModelType.ELASTIC_NET,
                0.08,
                0.5,
                "Sparsity is required while correlated predictors need L2 stabilization."
            );
        }

        if (needSparsity) {
            return new PolicyDecision(
                ModelType.LASSO,
                0.08,
                1.0,
                "Feature selection is a primary modeling requirement."
            );
        }

        if (features > observations || correlatedFeatures) {
            return new PolicyDecision(
                ModelType.RIDGE,
                0.8,
                0.0,
                "The model is vulnerable to coefficient instability."
            );
        }

        return new PolicyDecision(
            ModelType.RIDGE,
            0.1,
            0.0,
            "A modest L2 penalty provides a stable baseline."
        );
    }

    public static void main(String[] args) {
        System.out.println("REGULARIZATION ENTERPRISE MODEL");
        System.out.println("===============================");

        Random random =
            new Random(42);

        double[][] X =
            generateFeatures(180, random);

        double[] y =
            generateTargets(X, random);

        int trainSize =
            (int) Math.round(X.length * 0.75);

        double[][] trainRaw =
            Arrays.copyOfRange(X, 0, trainSize);

        double[] trainY =
            Arrays.copyOfRange(y, 0, trainSize);

        double[][] testRaw =
            Arrays.copyOfRange(X, trainSize, X.length);

        double[] testY =
            Arrays.copyOfRange(y, trainSize, y.length);

        StandardScaler scaler =
            new StandardScaler();

        double[][] trainX =
            scaler.fit(trainRaw)
                  .transform(trainRaw);

        double[][] testX =
            scaler.transform(testRaw);

        Map<ModelType, RegressionModel> models =
            new EnumMap<>(ModelType.class);

        models.put(
            ModelType.RIDGE,
            new RidgeModel(0.8, 0.003, 12000)
        );

        models.put(
            ModelType.LASSO,
            new LassoModel(0.08, 5000)
        );

        models.put(
            ModelType.ELASTIC_NET,
            new ElasticNetModel(
                0.08,
                0.5,
                5000
            )
        );

        for (Map.Entry<ModelType, RegressionModel> entry :
             models.entrySet()) {

            Evaluation evaluation =
                evaluate(
                    entry.getValue(),
                    trainX,
                    trainY,
                    testX,
                    testY
                );

            System.out.printf(
                "%-14s train RMSE=%8.4f test RMSE=%8.4f " +
                "test R2=%8.4f zero coefficients=%d%n",
                entry.getKey(),
                evaluation.trainRmse(),
                evaluation.testRmse(),
                evaluation.testR2(),
                evaluation.zeroCoefficients()
            );
        }

        System.out.println("\nRegularization policy");

        PolicyDecision policy =
            choosePolicy(
                trainX.length,
                trainX[0].length,
                true,
                true
            );

        System.out.println(
            "Selected model: " + policy.model()
        );

        System.out.println(
            "Alpha: " + policy.alpha()
        );

        System.out.println(
            "L1 ratio: " + policy.l1Ratio()
        );

        System.out.println(
            "Reason: " + policy.reason()
        );

        System.out.println("\nLasso sparsity path");

        for (double alpha :
             new double[] {
                 0.005, 0.02, 0.05,
                 0.1, 0.2, 0.5
             }) {

            LassoModel model =
                new LassoModel(alpha, 5000);

            model.fit(trainX, trainY);

            long active =
                Arrays.stream(model.coefficients())
                    .filter(value -> Math.abs(value) > 1e-8)
                    .count();

            System.out.printf(
                "alpha=%-7.3f active features=%d%n",
                alpha,
                active
            );
        }

        System.out.println("\nCoefficient comparison");

        String[] featureNames = {
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

        for (int j = 0; j < featureNames.length; j++) {
            System.out.printf(
                "%-20s Ridge=%9.4f Lasso=%9.4f ElasticNet=%9.4f%n",
                featureNames[j],
                models.get(ModelType.RIDGE)
                    .coefficients()[j],
                models.get(ModelType.LASSO)
                    .coefficients()[j],
                models.get(ModelType.ELASTIC_NET)
                    .coefficients()[j]
            );
        }

        System.out.println("\nValidation behavior");

        try {
            new RidgeModel(-1.0, 0.003, 1000);
        } catch (IllegalArgumentException error) {
            System.out.println(
                "Rejected negative Ridge alpha: " +
                error.getMessage()
            );
        }

        try {
            new ElasticNetModel(
                0.1,
                1.5,
                1000
            );
        } catch (IllegalArgumentException error) {
            System.out.println(
                "Rejected invalid Elastic Net ratio: " +
                error.getMessage()
            );
        }

        System.out.println("\nEnterprise modeling interpretation");
        System.out.println(
            "Ridge is appropriate when coefficient stability is more important " +
            "than automatic feature elimination."
        );
        System.out.println(
            "Lasso is appropriate when a compact predictor set has operational " +
            "or governance value."
        );
        System.out.println(
            "Elastic Net is useful when sparsity is desired but predictors " +
            "are correlated enough to make pure L1 selection unstable."
        );
        System.out.println(
            "The penalty is a policy parameter: its value should be selected " +
            "from validation performance rather than from training error alone."
        );
        System.out.println(
            "Standardization is part of the modeling pipeline because otherwise " +
            "coefficient penalties depend on measurement units."
        );
    }
}
