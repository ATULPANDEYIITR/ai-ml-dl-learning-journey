import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/*
 * Java 17 enterprise-oriented bias-variance evaluation workflow.
 *
 * Domain:
 * A model-governance service evaluates polynomial demand predictors
 * against a controlled nonlinear signal. Each candidate receives an
 * independent evaluation record and a policy-based lifecycle decision.
 *
 * Compile:
 * javac BiasVarianceEnterpriseDemo.java
 *
 * Run:
 * java BiasVarianceEnterpriseDemo
 */

public class BiasVarianceEnterpriseDemo {

    private static final double NOISE_STDDEV = 0.20;
    private static final double RIDGE_LAMBDA = 1e-6;

    enum RiskLevel {
        UNDERFITTING,
        OVERFITTING,
        GENERALIZES_WELL
    }

    enum LifecycleState {
        CREATED,
        EVALUATING,
        EVALUATED,
        APPROVED,
        REJECTED
    }

    record Point(double x, double y) {
        Point {
            if (!Double.isFinite(x) || !Double.isFinite(y)) {
                throw new IllegalArgumentException(
                    "Point coordinates must be finite."
                );
            }

            if (x < -1.0 || x > 1.0) {
                throw new IllegalArgumentException(
                    "The independent variable must be in [-1, 1]."
                );
            }
        }
    }

    record ExperimentConfig(
        int sampleCount,
        int repetitions,
        int gridSize,
        double noiseStdDev,
        long seed
    ) {
        ExperimentConfig {
            if (sampleCount < 10) {
                throw new IllegalArgumentException(
                    "At least ten observations are required."
                );
            }

            if (repetitions < 2) {
                throw new IllegalArgumentException(
                    "At least two repetitions are required."
                );
            }

            if (gridSize < 3) {
                throw new IllegalArgumentException(
                    "Evaluation grid must contain at least three points."
                );
            }

            if (!Double.isFinite(noiseStdDev) || noiseStdDev < 0.0) {
                throw new IllegalArgumentException(
                    "Noise standard deviation must be non-negative."
                );
            }
        }
    }

    record Candidate(
        String identifier,
        int polynomialDegree,
        double ridgeLambda
    ) {
        Candidate {
            Objects.requireNonNull(identifier, "Candidate identifier is required.");

            if (identifier.isBlank()) {
                throw new IllegalArgumentException(
                    "Candidate identifier cannot be blank."
                );
            }

            if (polynomialDegree < 0 || polynomialDegree > 12) {
                throw new IllegalArgumentException(
                    "Polynomial degree must be between 0 and 12."
                );
            }

            if (!Double.isFinite(ridgeLambda) || ridgeLambda < 0.0) {
                throw new IllegalArgumentException(
                    "Invalid ridge regularization value."
                );
            }
        }

        int parameterCount() {
            return polynomialDegree + 1;
        }
    }

    record Metrics(
        double averageTrainingMse,
        double biasSquared,
        double predictionVariance,
        double irreducibleNoise,
        double signalPredictionMse,
        double expectedTestMse,
        double generalizationGap
    ) {
        Metrics {
            double[] values = {
                averageTrainingMse,
                biasSquared,
                predictionVariance,
                irreducibleNoise,
                signalPredictionMse,
                expectedTestMse,
                generalizationGap
            };

            for (double value : values) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Evaluation metrics must be finite."
                    );
                }
            }

            if (
                averageTrainingMse < 0.0 ||
                biasSquared < 0.0 ||
                predictionVariance < 0.0 ||
                irreducibleNoise < 0.0 ||
                signalPredictionMse < 0.0 ||
                expectedTestMse < 0.0
            ) {
                throw new IllegalArgumentException(
                    "Error components cannot be negative."
                );
            }
        }
    }

    record PolicyDecision(
        RiskLevel risk,
        boolean approved,
        double samplesPerParameter,
        String rationale
    ) {
        PolicyDecision {
            Objects.requireNonNull(risk);
            Objects.requireNonNull(rationale);

            if (!Double.isFinite(samplesPerParameter) ||
                samplesPerParameter <= 0.0) {
                throw new IllegalArgumentException(
                    "Samples per parameter must be positive."
                );
            }
        }
    }

    record ModelEvaluation(
        Candidate candidate,
        Metrics metrics,
        PolicyDecision decision,
        LifecycleState lifecycleState
    ) {
        ModelEvaluation {
            Objects.requireNonNull(candidate);
            Objects.requireNonNull(metrics);
            Objects.requireNonNull(decision);
            Objects.requireNonNull(lifecycleState);
        }
    }

    record EvaluationBatch(
        String batchIdentifier,
        int datasetCount,
        int observationsPerDataset,
        List<ModelEvaluation> evaluations
    ) {
        EvaluationBatch {
            if (batchIdentifier == null || batchIdentifier.isBlank()) {
                throw new IllegalArgumentException(
                    "Batch identifier is required."
                );
            }

            if (datasetCount < 2 || observationsPerDataset < 10) {
                throw new IllegalArgumentException(
                    "Invalid evaluation batch dimensions."
                );
            }

            evaluations = List.copyOf(evaluations);
        }
    }

    interface RegressionEstimator {
        void fit(List<Point> trainingData);

        double predict(double x);
    }

    interface GeneralizationPolicy {
        PolicyDecision assess(
            Candidate candidate,
            Metrics metrics,
            double bestExpectedTestMse,
            double bestTrainingMse,
            int sampleCount
        );
    }

    static double trueSignal(double x) {
        if (!Double.isFinite(x) || x < -1.0 || x > 1.0) {
            throw new IllegalArgumentException("Signal input must be in [-1, 1].");
        }

        return Math.sin(2.0 * Math.PI * x) + 0.25 * x;
    }

    static double[] polynomialFeatures(double x, int degree) {
        double[] features = new double[degree + 1];
        features[0] = 1.0;

        for (int index = 1; index <= degree; index++) {
            features[index] = features[index - 1] * x;
        }

        return features;
    }

    static double[] solveLinearSystem(
        double[][] sourceMatrix,
        double[] sourceVector
    ) {
        int size = sourceVector.length;

        if (size == 0 || sourceMatrix.length != size) {
            throw new IllegalArgumentException(
                "Invalid linear system dimensions."
            );
        }

        double[][] matrix = new double[size][size + 1];

        for (int row = 0; row < size; row++) {
            if (sourceMatrix[row].length != size) {
                throw new IllegalArgumentException(
                    "Coefficient matrix must be square."
                );
            }

            System.arraycopy(
                sourceMatrix[row],
                0,
                matrix[row],
                0,
                size
            );

            matrix[row][size] = sourceVector[row];
        }

        for (int column = 0; column < size; column++) {
            int pivot = column;

            for (int row = column + 1; row < size; row++) {
                if (
                    Math.abs(matrix[row][column]) >
                    Math.abs(matrix[pivot][column])
                ) {
                    pivot = row;
                }
            }

            if (Math.abs(matrix[pivot][column]) < 1e-14) {
                throw new ArithmeticException(
                    "Singular regression system."
                );
            }

            double[] temporary = matrix[column];
            matrix[column] = matrix[pivot];
            matrix[pivot] = temporary;

            double divisor = matrix[column][column];

            for (int row = column + 1; row < size; row++) {
                double factor = matrix[row][column] / divisor;

                for (int index = column; index <= size; index++) {
                    matrix[row][index] -= factor * matrix[column][index];
                }
            }
        }

        double[] solution = new double[size];

        for (int row = size - 1; row >= 0; row--) {
            double value = matrix[row][size];

            for (int column = row + 1; column < size; column++) {
                value -= matrix[row][column] * solution[column];
            }

            solution[row] = value / matrix[row][row];

            if (!Double.isFinite(solution[row])) {
                throw new ArithmeticException(
                    "Non-finite regression coefficient."
                );
            }
        }

        return solution;
    }

    static final class PolynomialRidgeRegressor
        implements RegressionEstimator {

        private final Candidate candidate;
        private double[] coefficients;

        PolynomialRidgeRegressor(Candidate candidate) {
            this.candidate = Objects.requireNonNull(candidate);
        }

        @Override
        public void fit(List<Point> trainingData) {
            if (trainingData == null || trainingData.size() < 2) {
                throw new IllegalArgumentException(
                    "At least two training observations are required."
                );
            }

            int parameterCount = candidate.parameterCount();
            double[][] matrix = new double[parameterCount][parameterCount];
            double[] rhs = new double[parameterCount];

            for (Point point : trainingData) {
                double[] features = polynomialFeatures(
                    point.x(),
                    candidate.polynomialDegree()
                );

                for (int row = 0; row < parameterCount; row++) {
                    rhs[row] += features[row] * point.y();

                    for (int column = 0; column < parameterCount; column++) {
                        matrix[row][column] += features[row] * features[column];
                    }
                }
            }

            // The intercept is not subject to the L2 penalty.
            for (int index = 1; index < parameterCount; index++) {
                matrix[index][index] += candidate.ridgeLambda();
            }

            coefficients = solveLinearSystem(matrix, rhs);
        }

        @Override
        public double predict(double x) {
            if (coefficients == null) {
                throw new IllegalStateException(
                    "Model must be fitted before prediction."
                );
            }

            if (!Double.isFinite(x) || x < -1.0 || x > 1.0) {
                throw new IllegalArgumentException(
                    "Prediction input must be in [-1, 1]."
                );
            }

            double[] features = polynomialFeatures(
                x,
                candidate.polynomialDegree()
            );

            double prediction = 0.0;

            for (int index = 0; index < coefficients.length; index++) {
                prediction += coefficients[index] * features[index];
            }

            if (!Double.isFinite(prediction)) {
                throw new ArithmeticException("Prediction is not finite.");
            }

            return prediction;
        }
    }

    static final class DatasetGenerator {
        private final Random random;

        DatasetGenerator(long seed) {
            this.random = new Random(seed);
        }

        List<Point> generate(int sampleCount, double noiseStdDev) {
            if (sampleCount < 2 || noiseStdDev < 0.0) {
                throw new IllegalArgumentException(
                    "Invalid dataset generation parameters."
                );
            }

            List<Point> points = new ArrayList<>(sampleCount);

            for (int index = 0; index < sampleCount; index++) {
                double x = -1.0 + 2.0 * random.nextDouble();
                double y = trueSignal(x) + random.nextGaussian() * noiseStdDev;

                points.add(new Point(x, y));
            }

            return List.copyOf(points);
        }
    }

    static double meanSquaredError(
        List<Double> actual,
        List<Double> predicted
    ) {
        if (
            actual == null ||
            predicted == null ||
            actual.isEmpty() ||
            actual.size() != predicted.size()
        ) {
            throw new IllegalArgumentException(
                "MSE inputs must have equal non-zero lengths."
            );
        }

        double sum = 0.0;

        for (int index = 0; index < actual.size(); index++) {
            double difference = actual.get(index) - predicted.get(index);
            sum += difference * difference;
        }

        return sum / actual.size();
    }

    static double average(List<Double> values) {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException(
                "Cannot average an empty collection."
            );
        }

        return values.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElseThrow();
    }

    static final class EvaluationService {
        private final ExperimentConfig config;

        EvaluationService(ExperimentConfig config) {
            this.config = Objects.requireNonNull(config);
        }

        Metrics evaluate(Candidate candidate) {
            int repetitions = config.repetitions();
            int gridSize = config.gridSize();

            DatasetGenerator generator = new DatasetGenerator(
                config.seed()
            );

            double[][] predictions = new double[repetitions][gridSize];
            List<Double> trainingErrors = new ArrayList<>(repetitions);

            List<Double> grid = new ArrayList<>(gridSize);
            List<Double> signalValues = new ArrayList<>(gridSize);

            for (int index = 0; index < gridSize; index++) {
                double x = -1.0 + 2.0 * index / (gridSize - 1);
                grid.add(x);
                signalValues.add(trueSignal(x));
            }

            for (int repetition = 0; repetition < repetitions; repetition++) {
                List<Point> dataset = generator.generate(
                    config.sampleCount(),
                    config.noiseStdDev()
                );

                RegressionEstimator model = new PolynomialRidgeRegressor(
                    candidate
                );

                model.fit(dataset);

                List<Double> actualTraining = dataset.stream()
                    .map(Point::y)
                    .toList();

                List<Double> predictedTraining = dataset.stream()
                    .map(point -> model.predict(point.x()))
                    .toList();

                trainingErrors.add(
                    meanSquaredError(actualTraining, predictedTraining)
                );

                for (int position = 0; position < gridSize; position++) {
                    predictions[repetition][position] =
                        model.predict(grid.get(position));
                }
            }

            double biasSquared = 0.0;
            double predictionVariance = 0.0;

            for (int position = 0; position < gridSize; position++) {
                double meanPrediction = 0.0;

                for (int repetition = 0; repetition < repetitions; repetition++) {
                    meanPrediction += predictions[repetition][position];
                }

                meanPrediction /= repetitions;

                double bias = meanPrediction - signalValues.get(position);
                biasSquared += bias * bias;

                double varianceAtPosition = 0.0;

                for (int repetition = 0; repetition < repetitions; repetition++) {
                    double difference =
                        predictions[repetition][position] - meanPrediction;

                    varianceAtPosition += difference * difference;
                }

                predictionVariance += varianceAtPosition / repetitions;
            }

            biasSquared /= gridSize;
            predictionVariance /= gridSize;

            double trainingMse = average(trainingErrors);
            double irreducibleNoise =
                config.noiseStdDev() * config.noiseStdDev();
            double signalPredictionMse =
                biasSquared + predictionVariance;
            double expectedTestMse =
                signalPredictionMse + irreducibleNoise;

            return new Metrics(
                trainingMse,
                biasSquared,
                predictionVariance,
                irreducibleNoise,
                signalPredictionMse,
                expectedTestMse,
                expectedTestMse - trainingMse
            );
        }
    }

    static final class DefaultGeneralizationPolicy
        implements GeneralizationPolicy {

        private final double testErrorRatioLimit;
        private final double underfitTrainingRatio;
        private final double minimumSamplesPerParameter;
        private final double maximumPredictionVariance;

        DefaultGeneralizationPolicy() {
            this(1.10, 1.20, 2.5, 0.05);
        }

        DefaultGeneralizationPolicy(
            double testErrorRatioLimit,
            double underfitTrainingRatio,
            double minimumSamplesPerParameter,
            double maximumPredictionVariance
        ) {
            if (
                testErrorRatioLimit < 1.0 ||
                underfitTrainingRatio < 1.0 ||
                minimumSamplesPerParameter <= 0.0 ||
                maximumPredictionVariance < 0.0
            ) {
                throw new IllegalArgumentException(
                    "Invalid governance policy configuration."
                );
            }

            this.testErrorRatioLimit = testErrorRatioLimit;
            this.underfitTrainingRatio = underfitTrainingRatio;
            this.minimumSamplesPerParameter = minimumSamplesPerParameter;
            this.maximumPredictionVariance = maximumPredictionVariance;
        }

        @Override
        public PolicyDecision assess(
            Candidate candidate,
            Metrics metrics,
            double bestExpectedTestMse,
            double bestTrainingMse,
            int sampleCount
        ) {
            if (sampleCount <= 0) {
                throw new IllegalArgumentException(
                    "Sample count must be positive."
                );
            }

            double samplesPerParameter =
                (double) sampleCount / candidate.parameterCount();

            boolean testErrorExceedsLimit =
                metrics.expectedTestMse() >
                bestExpectedTestMse * testErrorRatioLimit;

            RiskLevel risk;

            if (
                testErrorExceedsLimit &&
                metrics.averageTrainingMse() >
                    bestTrainingMse * underfitTrainingRatio
            ) {
                risk = RiskLevel.UNDERFITTING;
            } else if (
                testErrorExceedsLimit ||
                metrics.predictionVariance() > maximumPredictionVariance
            ) {
                risk = RiskLevel.OVERFITTING;
            } else {
                risk = RiskLevel.GENERALIZES_WELL;
            }

            List<String> failures = new ArrayList<>();

            if (testErrorExceedsLimit) {
                failures.add("Expected test error exceeds the policy limit");
            }

            if (metrics.predictionVariance() > maximumPredictionVariance) {
                failures.add("Prediction variance exceeds the policy limit");
            }

            if (samplesPerParameter < minimumSamplesPerParameter) {
                failures.add("Training observations per parameter are insufficient");
            }

            boolean approved = failures.isEmpty();

            String rationale = approved
                ? "Expected test error, variance and sample adequacy satisfy policy"
                : String.join("; ", failures);

            return new PolicyDecision(
                risk,
                approved,
                samplesPerParameter,
                rationale
            );
        }
    }

    static final class ModelLifecycle {
        private LifecycleState state = LifecycleState.CREATED;

        LifecycleState state() {
            return state;
        }

        void transition(LifecycleState next) {
            Objects.requireNonNull(next);

            boolean allowed = switch (state) {
                case CREATED -> next == LifecycleState.EVALUATING;
                case EVALUATING -> next == LifecycleState.EVALUATED;
                case EVALUATED ->
                    next == LifecycleState.APPROVED ||
                    next == LifecycleState.REJECTED;
                case APPROVED, REJECTED -> false;
            };

            if (!allowed) {
                throw new IllegalStateException(
                    "Invalid model lifecycle transition: " +
                    state + " -> " + next
                );
            }

            state = next;
        }
    }

    static final class ModelGovernanceService {
        private final EvaluationService evaluator;
        private final GeneralizationPolicy policy;

        ModelGovernanceService(
            EvaluationService evaluator,
            GeneralizationPolicy policy
        ) {
            this.evaluator = Objects.requireNonNull(evaluator);
            this.policy = Objects.requireNonNull(policy);
        }

        EvaluationBatch evaluateBatch(
            String batchIdentifier,
            ExperimentConfig config,
            List<Candidate> candidates
        ) {
            if (candidates == null || candidates.isEmpty()) {
                throw new IllegalArgumentException(
                    "At least one candidate is required."
                );
            }

            Set<String> identifiers = new HashSet<>();
            Set<Integer> degrees = new HashSet<>();

            for (Candidate candidate : candidates) {
                if (!identifiers.add(candidate.identifier())) {
                    throw new IllegalArgumentException(
                        "Candidate identifiers must be unique."
                    );
                }

                if (!degrees.add(candidate.polynomialDegree())) {
                    throw new IllegalArgumentException(
                        "The batch cannot contain duplicate polynomial degrees."
                    );
                }
            }

            List<Metrics> metricResults = candidates.stream()
                .map(evaluator::evaluate)
                .toList();

            double bestExpectedTestMse = metricResults.stream()
                .mapToDouble(Metrics::expectedTestMse)
                .min()
                .orElseThrow();

            double bestTrainingMse = metricResults.stream()
                .mapToDouble(Metrics::averageTrainingMse)
                .min()
                .orElseThrow();

            List<ModelEvaluation> evaluations = new ArrayList<>();

            for (int index = 0; index < candidates.size(); index++) {
                Candidate candidate = candidates.get(index);
                Metrics metrics = metricResults.get(index);
                ModelLifecycle lifecycle = new ModelLifecycle();

                lifecycle.transition(LifecycleState.EVALUATING);
                lifecycle.transition(LifecycleState.EVALUATED);

                PolicyDecision decision = policy.assess(
                    candidate,
                    metrics,
                    bestExpectedTestMse,
                    bestTrainingMse,
                    config.sampleCount()
                );

                lifecycle.transition(
                    decision.approved()
                        ? LifecycleState.APPROVED
                        : LifecycleState.REJECTED
                );

                evaluations.add(
                    new ModelEvaluation(
                        candidate,
                        metrics,
                        decision,
                        lifecycle.state()
                    )
                );
            }

            return new EvaluationBatch(
                batchIdentifier,
                config.repetitions(),
                config.sampleCount(),
                evaluations
            );
        }
    }

    static void validateFailureConditions() {
        try {
            new Candidate("invalid", 13, RIDGE_LAMBDA);
            throw new IllegalStateException(
                "Excessive model complexity was accepted."
            );
        } catch (IllegalArgumentException expected) {
            // Invalid candidate rejected.
        }

        try {
            new Point(2.0, 1.0);
            throw new IllegalStateException(
                "Out-of-domain training input was accepted."
            );
        } catch (IllegalArgumentException expected) {
            // Invalid input rejected.
        }

        ModelLifecycle lifecycle = new ModelLifecycle();

        try {
            lifecycle.transition(LifecycleState.APPROVED);
            throw new IllegalStateException(
                "Invalid lifecycle transition was accepted."
            );
        } catch (IllegalStateException expected) {
            if (lifecycle.state() != LifecycleState.CREATED) {
                throw expected;
            }
        }

        Candidate constantCandidate = new Candidate(
            "constant-test",
            0,
            0.0
        );

        PolynomialRidgeRegressor constantModel =
            new PolynomialRidgeRegressor(constantCandidate);

        constantModel.fit(
            List.of(
                new Point(-1.0, 5.0),
                new Point(1.0, 5.0)
            )
        );

        if (Math.abs(constantModel.predict(0.0) - 5.0) > 1e-9) {
            throw new IllegalStateException("Constant-model test failed.");
        }
    }

    static void printReport(EvaluationBatch batch) {
        System.out.println("ENTERPRISE MODEL-GOVERNANCE REPORT");
        System.out.println("Batch: " + batch.batchIdentifier());
        System.out.println("Independent evaluation datasets: " + batch.datasetCount());
        System.out.println(
            "Observations per dataset: " + batch.observationsPerDataset()
        );
        System.out.println();

        System.out.printf(
            "%-12s %8s %10s %10s %10s %12s %12s %-17s %-10s%n",
            "Candidate",
            "Degree",
            "Train MSE",
            "Bias^2",
            "Variance",
            "Test MSE",
            "Gap",
            "Risk",
            "Decision"
        );

        for (ModelEvaluation evaluation : batch.evaluations()) {
            Metrics metrics = evaluation.metrics();
            PolicyDecision decision = evaluation.decision();

            System.out.printf(
                "%-12s %8d %10.5f %10.5f %10.5f %12.5f %12.5f %-17s %-10s%n",
                evaluation.candidate().identifier(),
                evaluation.candidate().polynomialDegree(),
                metrics.averageTrainingMse(),
                metrics.biasSquared(),
                metrics.predictionVariance(),
                metrics.expectedTestMse(),
                metrics.generalizationGap(),
                decision.risk(),
                evaluation.lifecycleState()
            );

            System.out.printf(
                "  Samples per parameter: %.2f | %s%n",
                decision.samplesPerParameter(),
                decision.rationale()
            );
        }

        long approvedCount = batch.evaluations().stream()
            .filter(item -> item.lifecycleState() == LifecycleState.APPROVED)
            .count();

        System.out.println();
        System.out.println("Approved candidates: " + approvedCount);
        System.out.println(
            "Rejected candidates: " +
            (batch.evaluations().size() - approvedCount)
        );

        batch.evaluations().stream()
            .filter(item -> item.lifecycleState() == LifecycleState.APPROVED)
            .min((left, right) -> Double.compare(
                left.metrics().expectedTestMse(),
                right.metrics().expectedTestMse()
            ))
            .ifPresent(item -> System.out.println(
                "Best approved candidate: " +
                item.candidate().identifier()
            ));
    }

    public static void main(String[] args) {
        try {
            validateFailureConditions();

            ExperimentConfig config = new ExperimentConfig(
                36,
                80,
                81,
                NOISE_STDDEV,
                20261103L
            );

            List<Candidate> candidates = List.of(
                new Candidate("POLY-DEG-1", 1, RIDGE_LAMBDA),
                new Candidate("POLY-DEG-2", 2, RIDGE_LAMBDA),
                new Candidate("POLY-DEG-5", 5, RIDGE_LAMBDA),
                new Candidate("POLY-DEG-9", 9, RIDGE_LAMBDA)
            );

            EvaluationService evaluator = new EvaluationService(config);
            GeneralizationPolicy policy =
                new DefaultGeneralizationPolicy();

            ModelGovernanceService governance =
                new ModelGovernanceService(evaluator, policy);

            EvaluationBatch batch = governance.evaluateBatch(
                "DEMAND-MODEL-VALIDATION-2026-01",
                config,
                candidates
            );

            printReport(batch);

            /*
             * Enterprise design considerations:
             *
             * 1. Candidate identifiers are unique within an evaluation batch.
             * 2. Candidate configuration and evaluation records are immutable.
             * 3. Lifecycle transitions are explicitly validated.
             * 4. Approval is separate from statistical evaluation.
             * 5. Policy thresholds are configurable and independently testable.
             * 6. A production workflow should persist model versions,
             *    dataset fingerprints, code versions, and policy versions.
             * 7. Evaluation data should be isolated from training data.
             * 8. Repeated evaluations must not reuse validation outcomes
             *    for model tuning without accounting for selection bias.
             *
             * The normal-equation solver is appropriate for this small,
             * controlled polynomial example. Production systems handling
             * larger or poorly conditioned feature matrices should use
             * QR or singular-value decomposition.
             */

        } catch (Exception error) {
            System.err.println(
                "Governance evaluation failed: " + error.getMessage()
            );
            System.exit(1);
        }
    }
}
