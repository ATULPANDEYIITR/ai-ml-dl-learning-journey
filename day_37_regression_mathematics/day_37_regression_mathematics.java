/*
 * Regression Governance Enterprise Model
 *
 * Java 17+
 *
 * The program models an enterprise analytics service that estimates
 * delivery-time regression coefficients, evaluates residual quality,
 * calculates MSE/R²/adjusted R², applies explicit governance policies,
 * and rejects invalid or numerically unsafe models.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class RegressionGovernance {

    enum ModelStatus {
        FITTED,
        ELIGIBLE,
        REJECTED,
        INVALID
    }

    enum RejectionReason {
        LOW_R_SQUARED,
        LOW_ADJUSTED_R_SQUARED,
        HIGH_RMSE,
        LARGE_RESIDUAL,
        INSUFFICIENT_DEGREES_OF_FREEDOM,
        SINGULAR_DESIGN_MATRIX
    }

    record Observation(
        double processingHours,
        double itemCount,
        double observedDeliveryHours
    ) {
        Observation {
            if (!Double.isFinite(processingHours)
                    || !Double.isFinite(itemCount)
                    || !Double.isFinite(observedDeliveryHours)) {
                throw new IllegalArgumentException(
                    "Observation values must be finite."
                );
            }

            if (processingHours < 0 || itemCount < 0) {
                throw new IllegalArgumentException(
                    "Operational quantities cannot be negative."
                );
            }
        }

        List<Double> predictors() {
            return List.of(
                processingHours,
                itemCount
            );
        }
    }

    record RegressionResult(
        List<Double> coefficients,
        List<Double> predictions,
        List<Double> residuals,
        double sse,
        double mse,
        double rmse,
        double rSquared,
        double adjustedRSquared
    ) {
        RegressionResult {
            coefficients = List.copyOf(coefficients);
            predictions = List.copyOf(predictions);
            residuals = List.copyOf(residuals);
        }

        double intercept() {
            return coefficients.get(0);
        }

        double coefficient(int predictorIndex) {
            return coefficients.get(predictorIndex + 1);
        }

        double largestAbsoluteResidual() {
            return residuals.stream()
                .mapToDouble(Math::abs)
                .max()
                .orElse(0.0);
        }
    }

    record GovernancePolicy(
        double minimumRSquared,
        double minimumAdjustedRSquared,
        double maximumRMSE,
        double maximumAbsoluteResidual
    ) {
        GovernancePolicy {
            if (minimumRSquared < 0 || minimumRSquared > 1) {
                throw new IllegalArgumentException(
                    "Minimum R² must be between zero and one."
                );
            }

            if (minimumAdjustedRSquared < 0
                    || minimumAdjustedRSquared > 1) {
                throw new IllegalArgumentException(
                    "Minimum adjusted R² must be between zero and one."
                );
            }

            if (maximumRMSE <= 0
                    || maximumAbsoluteResidual <= 0) {
                throw new IllegalArgumentException(
                    "Error thresholds must be positive."
                );
            }
        }
    }

    record GovernanceDecision(
        ModelStatus status,
        Set<RejectionReason> reasons
    ) {
        GovernanceDecision {
            reasons = Collections.unmodifiableSet(
                EnumSet.copyOf(
                    reasons.isEmpty()
                        ? EnumSet.noneOf(RejectionReason.class)
                        : reasons
                )
            );
        }
    }

    static final class Matrix {

        private Matrix() {
        }

        static double[][] transpose(double[][] matrix) {
            if (matrix.length == 0) {
                return new double[0][0];
            }

            double[][] result =
                new double[matrix[0].length][matrix.length];

            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    result[j][i] = matrix[i][j];
                }
            }

            return result;
        }

        static double[][] multiply(
            double[][] left,
            double[][] right
        ) {
            if (left.length == 0
                    || right.length == 0
                    || left[0].length != right.length) {
                throw new IllegalArgumentException(
                    "Invalid matrix dimensions."
                );
            }

            double[][] result =
                new double[left.length][right[0].length];

            for (int i = 0; i < left.length; i++) {
                for (int k = 0; k < right.length; k++) {
                    for (int j = 0; j < right[0].length; j++) {
                        result[i][j] +=
                            left[i][k] * right[k][j];
                    }
                }
            }

            return result;
        }

        static double[][] identity(int size) {
            double[][] result = new double[size][size];

            for (int i = 0; i < size; i++) {
                result[i][i] = 1.0;
            }

            return result;
        }

        static double[][] inverse(double[][] matrix) {
            int n = matrix.length;

            if (n == 0) {
                throw new IllegalArgumentException(
                    "Cannot invert an empty matrix."
                );
            }

            double[][] augmented =
                new double[n][2 * n];

            double[][] identity = identity(n);

            for (int i = 0; i < n; i++) {
                if (matrix[i].length != n) {
                    throw new IllegalArgumentException(
                        "Matrix must be square."
                    );
                }

                System.arraycopy(
                    matrix[i],
                    0,
                    augmented[i],
                    0,
                    n
                );

                System.arraycopy(
                    identity[i],
                    0,
                    augmented[i],
                    n,
                    n
                );
            }

            for (int column = 0; column < n; column++) {
                int pivotRow = column;

                for (int row = column + 1; row < n; row++) {
                    if (Math.abs(
                            augmented[row][column]
                        ) > Math.abs(
                            augmented[pivotRow][column]
                        )) {
                        pivotRow = row;
                    }
                }

                if (Math.abs(
                        augmented[pivotRow][column]
                    ) < 1e-12) {
                    throw new IllegalStateException(
                        "Singular design matrix."
                    );
                }

                double[] temporary =
                    augmented[column];

                augmented[column] =
                    augmented[pivotRow];

                augmented[pivotRow] =
                    temporary;

                double pivot =
                    augmented[column][column];

                for (int j = 0; j < 2 * n; j++) {
                    augmented[column][j] /= pivot;
                }

                for (int row = 0; row < n; row++) {
                    if (row == column) {
                        continue;
                    }

                    double factor =
                        augmented[row][column];

                    for (int j = 0; j < 2 * n; j++) {
                        augmented[row][j] -=
                            factor * augmented[column][j];
                    }
                }
            }

            double[][] result =
                new double[n][n];

            for (int i = 0; i < n; i++) {
                System.arraycopy(
                    augmented[i],
                    n,
                    result[i],
                    0,
                    n
                );
            }

            return result;
        }
    }

    static final class RegressionService {

        RegressionResult fit(
            List<Observation> observations
        ) {
            validateObservations(observations);

            double[][] design =
                new double[observations.size()][3];

            double[][] response =
                new double[observations.size()][1];

            for (int i = 0; i < observations.size(); i++) {
                Observation observation =
                    observations.get(i);

                design[i][0] = 1.0;
                design[i][1] =
                    observation.processingHours();
                design[i][2] =
                    observation.itemCount();

                response[i][0] =
                    observation.observedDeliveryHours();
            }

            double[][] xT =
                Matrix.transpose(design);

            double[][] xTX =
                Matrix.multiply(xT, design);

            double[][] inverse =
                Matrix.inverse(xTX);

            double[][] beta =
                Matrix.multiply(
                    Matrix.multiply(inverse, xT),
                    response
                );

            List<Double> coefficients =
                Arrays.stream(beta)
                    .map(row -> row[0])
                    .boxed()
                    .toList();

            List<Double> predictions =
                observations.stream()
                    .map(observation ->
                        coefficients.get(0)
                        + coefficients.get(1)
                            * observation.processingHours()
                        + coefficients.get(2)
                            * observation.itemCount()
                    )
                    .toList();

            List<Double> residuals =
                new ArrayList<>();

            for (int i = 0; i < observations.size(); i++) {
                residuals.add(
                    observations.get(i)
                        .observedDeliveryHours()
                    - predictions.get(i)
                );
            }

            double sse =
                residuals.stream()
                    .mapToDouble(value -> value * value)
                    .sum();

            int n = observations.size();
            int p = 2;

            double modelMSE =
                sse / (n - p - 1);

            double predictiveMSE =
                sse / n;

            double rmse =
                Math.sqrt(predictiveMSE);

            double responseMean =
                observations.stream()
                    .mapToDouble(
                        Observation::observedDeliveryHours
                    )
                    .average()
                    .orElseThrow();

            double sst =
                observations.stream()
                    .mapToDouble(observation -> {
                        double difference =
                            observation.observedDeliveryHours()
                            - responseMean;

                        return difference * difference;
                    })
                    .sum();

            if (Math.abs(sst) < 1e-12) {
                throw new IllegalStateException(
                    "R² is undefined for a constant response."
                );
            }

            double rSquared =
                1.0 - sse / sst;

            double adjustedRSquared =
                1.0
                - (1.0 - rSquared)
                * ((n - 1.0) / (n - p - 1.0));

            return new RegressionResult(
                coefficients,
                predictions,
                residuals,
                sse,
                modelMSE,
                rmse,
                rSquared,
                adjustedRSquared
            );
        }

        private void validateObservations(
            List<Observation> observations
        ) {
            Objects.requireNonNull(
                observations,
                "Observations cannot be null."
            );

            if (observations.size() <= 3) {
                throw new IllegalArgumentException(
                    "At least four observations are required."
                );
            }

            if (observations.stream().anyMatch(
                    Objects::isNull
                )) {
                throw new IllegalArgumentException(
                    "Observation collection contains null."
                );
            }
        }
    }

    static final class GovernanceService {

        GovernanceDecision evaluate(
            RegressionResult result,
            GovernancePolicy policy
        ) {
            EnumSet<RejectionReason> reasons =
                EnumSet.noneOf(RejectionReason.class);

            if (result.rSquared()
                    < policy.minimumRSquared()) {
                reasons.add(
                    RejectionReason.LOW_R_SQUARED
                );
            }

            if (result.adjustedRSquared()
                    < policy.minimumAdjustedRSquared()) {
                reasons.add(
                    RejectionReason.LOW_ADJUSTED_R_SQUARED
                );
            }

            if (result.rmse()
                    > policy.maximumRMSE()) {
                reasons.add(
                    RejectionReason.HIGH_RMSE
                );
            }

            if (result.largestAbsoluteResidual()
                    > policy.maximumAbsoluteResidual()) {
                reasons.add(
                    RejectionReason.LARGE_RESIDUAL
                );
            }

            ModelStatus status =
                reasons.isEmpty()
                    ? ModelStatus.ELIGIBLE
                    : ModelStatus.REJECTED;

            return new GovernanceDecision(
                status,
                reasons
            );
        }
    }

    static void printResult(
        RegressionResult result
    ) {
        System.out.println(
            "\n" + "=".repeat(72)
        );

        System.out.printf(
            "Intercept: %.6f%n",
            result.intercept()
        );

        System.out.printf(
            "Processing-hours coefficient: %.6f%n",
            result.coefficient(0)
        );

        System.out.printf(
            "Item-count coefficient: %.6f%n",
            result.coefficient(1)
        );

        System.out.printf(
            "SSE: %.6f%n",
            result.sse()
        );

        System.out.printf(
            "Model-error MSE: %.6f%n",
            result.mse()
        );

        System.out.printf(
            "RMSE: %.6f%n",
            result.rmse()
        );

        System.out.printf(
            "R²: %.6f%n",
            result.rSquared()
        );

        System.out.printf(
            "Adjusted R²: %.6f%n",
            result.adjustedRSquared()
        );

        System.out.println(
            "\nResidual observations:"
        );

        for (int i = 0; i < result.residuals().size(); i++) {
            System.out.printf(
                "row=%02d predicted=%9.3f residual=%9.3f%n",
                i,
                result.predictions().get(i),
                result.residuals().get(i)
            );
        }
    }

    public static void main(String[] args) {
        List<Observation> observations = List.of(
            new Observation(2, 10, 8.4),
            new Observation(3, 12, 10.2),
            new Observation(4, 14, 11.7),
            new Observation(5, 18, 14.1),
            new Observation(6, 17, 14.8),
            new Observation(7, 21, 17.0),
            new Observation(8, 23, 18.9),
            new Observation(9, 24, 19.7),
            new Observation(10, 28, 23.0),
            new Observation(11, 29, 24.2),
            new Observation(12, 32, 26.0),
            new Observation(13, 35, 29.1)
        );

        RegressionService regressionService =
            new RegressionService();

        RegressionResult result =
            regressionService.fit(observations);

        printResult(result);

        GovernancePolicy policy =
            new GovernancePolicy(
                0.90,
                0.88,
                2.00,
                3.00
            );

        GovernanceService governanceService =
            new GovernanceService();

        GovernanceDecision decision =
            governanceService.evaluate(
                result,
                policy
            );

        System.out.println(
            "\nEnterprise governance decision: "
            + decision.status()
        );

        if (!decision.reasons().isEmpty()) {
            System.out.println(
                "Rejection reasons: "
                + decision.reasons()
            );
        }

        /*
         * The model is now used for new operational observations. This is
         * intentionally separate from fitting because prediction should not
         * silently retrain the model.
         */
        List<Observation> futureCases = List.of(
            new Observation(6.5, 20, 0),
            new Observation(9.5, 25, 0),
            new Observation(14, 38, 0)
        );

        System.out.println(
            "\nPredictions for future operational cases:"
        );

        for (Observation future : futureCases) {
            double prediction =
                result.intercept()
                + result.coefficient(0)
                    * future.processingHours()
                + result.coefficient(1)
                    * future.itemCount();

            System.out.printf(
                "processingHours=%.1f itemCount=%.1f "
                + "predictedDeliveryHours=%.3f%n",
                future.processingHours(),
                future.itemCount(),
                prediction
            );
        }

        /*
         * Demonstrate a failure state caused by perfect predictor
         * collinearity. In a production implementation, a QR or SVD solver
         * is generally preferable to explicitly forming (XᵀX)^(-1), because
         * the normal equation can amplify numerical conditioning problems.
         */
        try {
            double[][] collinear = {
                {1, 2},
                {2, 4},
                {3, 6}
            };

            Matrix.inverse(
                Matrix.multiply(
                    Matrix.transpose(collinear),
                    collinear
                )
            );
        } catch (IllegalStateException error) {
            System.out.println(
                "\nNumerical validation: "
                + error.getMessage()
            );
        }

        /*
         * Streams are used here for an audit-style view of observations whose
         * absolute residual exceeds a selected operational tolerance.
         */
        double residualTolerance = 2.0;

        List<Integer> largeResidualRows =
            java.util.stream.IntStream
                .range(0, result.residuals().size())
                .filter(index ->
                    Math.abs(
                        result.residuals().get(index)
                    ) > residualTolerance
                )
                .boxed()
                .toList();

        System.out.println(
            "\nRows exceeding residual tolerance: "
            + largeResidualRows
        );
    }
}
