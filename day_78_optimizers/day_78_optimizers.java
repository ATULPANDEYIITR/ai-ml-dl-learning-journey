import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/*
 * Enterprise Optimization Framework
 *
 * Java 17 implementation of:
 * Momentum, AdaGrad, RMSProp, Adam, and AdamW.
 *
 * Domain elements:
 * - Immutable optimizer configuration.
 * - Explicit optimizer policies.
 * - Stateful optimizer implementations.
 * - Training lifecycle management.
 * - Numerical validation.
 * - Evaluation records.
 * - Exception-based failure handling.
 *
 * Compile:
 * javac OptimizerEnterprise.java
 *
 * Execute:
 * java OptimizerEnterprise
 */

public class OptimizerEnterprise {

    public enum OptimizerType {
        MOMENTUM,
        ADAGRAD,
        RMSPROP,
        ADAM,
        ADAMW
    }

    public enum TrainingStatus {
        INITIALIZED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    public static final class ValidationException
            extends RuntimeException {

        public ValidationException(String message) {
            super(message);
        }
    }

    public static final class TrainingException
            extends RuntimeException {

        public TrainingException(String message) {
            super(message);
        }
    }

    public record OptimizerConfig(
            double learningRate,
            double epsilon,
            double weightDecay,
            double gradientClip
    ) {
        public OptimizerConfig {
            if (!Double.isFinite(learningRate) || learningRate <= 0) {
                throw new ValidationException("Invalid learning rate.");
            }

            if (!Double.isFinite(epsilon) || epsilon <= 0) {
                throw new ValidationException("Invalid epsilon.");
            }

            if (!Double.isFinite(weightDecay) || weightDecay < 0) {
                throw new ValidationException("Invalid weight decay.");
            }

            if (!Double.isFinite(gradientClip) || gradientClip < 0) {
                throw new ValidationException("Invalid gradient clipping.");
            }
        }

        public static OptimizerConfig standard(double learningRate) {
            return new OptimizerConfig(
                    learningRate,
                    1e-8,
                    0.0,
                    0.0
            );
        }
    }

    public record ObjectiveDefinition(
            List<Double> target,
            List<Double> curvature
    ) {
        public ObjectiveDefinition {
            target = List.copyOf(target);
            curvature = List.copyOf(curvature);

            validateVector(target, "Target");
            validateVector(curvature, "Curvature");

            if (target.size() != curvature.size()) {
                throw new ValidationException(
                        "Objective dimensions differ."
                );
            }

            for (double value : curvature) {
                if (value <= 0) {
                    throw new ValidationException(
                            "Curvature must be positive."
                    );
                }
            }
        }

        public double loss(List<Double> parameters) {
            validateVector(parameters, "Parameters");
            requireSameSize(parameters, target);

            double result = 0.0;

            for (int i = 0; i < parameters.size(); i++) {
                double difference = parameters.get(i) - target.get(i);

                result += 0.5 * curvature.get(i) *
                        difference * difference;
            }

            return result;
        }

        public List<Double> gradient(List<Double> parameters) {
            validateVector(parameters, "Parameters");
            requireSameSize(parameters, target);

            List<Double> result = new ArrayList<>();

            for (int i = 0; i < parameters.size(); i++) {
                result.add(
                        curvature.get(i) *
                        (parameters.get(i) - target.get(i))
                );
            }

            return List.copyOf(result);
        }
    }

    public interface OptimizerPolicy {
        List<Double> update(
                List<Double> parameters,
                List<Double> gradients
        );

        OptimizerType type();

        long stepCount();
    }

    public abstract static class AbstractOptimizer
            implements OptimizerPolicy {

        protected final OptimizerConfig config;
        protected long steps;

        protected AbstractOptimizer(OptimizerConfig config) {
            this.config = Objects.requireNonNull(config);
        }

        protected List<Double> prepare(
                List<Double> parameters,
                List<Double> gradients
        ) {
            validateVector(parameters, "Parameters");
            validateVector(gradients, "Gradients");
            requireSameSize(parameters, gradients);

            double norm = Math.sqrt(
                    gradients.stream()
                            .mapToDouble(g -> g * g)
                            .sum()
            );

            if (config.gradientClip() <= 0 ||
                    norm <= config.gradientClip()) {
                return List.copyOf(gradients);
            }

            double scale = config.gradientClip() / norm;

            return gradients.stream()
                    .map(g -> g * scale)
                    .toList();
        }

        protected List<Double> validateOutput(List<Double> values) {
            validateVector(values, "Updated parameters");
            return List.copyOf(values);
        }

        @Override
        public long stepCount() {
            return steps;
        }
    }

    public static final class Momentum
            extends AbstractOptimizer {

        private final double beta;
        private List<Double> velocity;

        public Momentum(OptimizerConfig config, double beta) {
            super(config);

            if (beta < 0 || beta >= 1) {
                throw new ValidationException(
                        "Invalid momentum coefficient."
                );
            }

            this.beta = beta;
        }

        @Override
        public List<Double> update(
                List<Double> parameters,
                List<Double> gradients
        ) {
            List<Double> g = prepare(parameters, gradients);

            if (velocity == null) {
                velocity = zeros(parameters.size());
            }

            steps++;

            List<Double> result = new ArrayList<>();

            for (int i = 0; i < parameters.size(); i++) {
                double nextVelocity =
                        beta * velocity.get(i) + g.get(i);

                velocity.set(i, nextVelocity);

                result.add(
                        parameters.get(i) -
                        config.learningRate() * nextVelocity
                );
            }

            return validateOutput(result);
        }

        @Override
        public OptimizerType type() {
            return OptimizerType.MOMENTUM;
        }
    }

    public static final class AdaGrad
            extends AbstractOptimizer {

        private List<Double> accumulator;

        public AdaGrad(OptimizerConfig config) {
            super(config);
        }

        @Override
        public List<Double> update(
                List<Double> parameters,
                List<Double> gradients
        ) {
            List<Double> g = prepare(parameters, gradients);

            if (accumulator == null) {
                accumulator = zeros(parameters.size());
            }

            steps++;

            List<Double> result = new ArrayList<>();

            for (int i = 0; i < parameters.size(); i++) {
                double value = accumulator.get(i) +
                        g.get(i) * g.get(i);

                accumulator.set(i, value);

                result.add(
                        parameters.get(i) -
                        config.learningRate() * g.get(i) /
                        (Math.sqrt(value) + config.epsilon())
                );
            }

            return validateOutput(result);
        }

        @Override
        public OptimizerType type() {
            return OptimizerType.ADAGRAD;
        }
    }

    public static final class RMSProp
            extends AbstractOptimizer {

        private final double beta;
        private List<Double> squareAverage;

        public RMSProp(OptimizerConfig config, double beta) {
            super(config);

            if (beta < 0 || beta >= 1) {
                throw new ValidationException(
                        "Invalid RMSProp coefficient."
                );
            }

            this.beta = beta;
        }

        @Override
        public List<Double> update(
                List<Double> parameters,
                List<Double> gradients
        ) {
            List<Double> g = prepare(parameters, gradients);

            if (squareAverage == null) {
                squareAverage = zeros(parameters.size());
            }

            steps++;

            List<Double> result = new ArrayList<>();

            for (int i = 0; i < parameters.size(); i++) {
                double average =
                        beta * squareAverage.get(i) +
                        (1 - beta) * g.get(i) * g.get(i);

                squareAverage.set(i, average);

                result.add(
                        parameters.get(i) -
                        config.learningRate() * g.get(i) /
                        (Math.sqrt(average) + config.epsilon())
                );
            }

            return validateOutput(result);
        }

        @Override
        public OptimizerType type() {
            return OptimizerType.RMSPROP;
        }
    }

    public static class Adam
            extends AbstractOptimizer {

        protected final double beta1;
        protected final double beta2;

        protected List<Double> firstMoment;
        protected List<Double> secondMoment;

        public Adam(
                OptimizerConfig config,
                double beta1,
                double beta2
        ) {
            super(config);

            if (beta1 < 0 || beta1 >= 1 ||
                    beta2 < 0 || beta2 >= 1) {
                throw new ValidationException(
                        "Invalid Adam coefficients."
                );
            }

            this.beta1 = beta1;
            this.beta2 = beta2;
        }

        @Override
        public List<Double> update(
                List<Double> parameters,
                List<Double> gradients
        ) {
            List<Double> g = prepare(parameters, gradients);

            if (firstMoment == null) {
                firstMoment = zeros(parameters.size());
                secondMoment = zeros(parameters.size());
            }

            steps++;

            double correction1 =
                    1 - Math.pow(beta1, steps);

            double correction2 =
                    1 - Math.pow(beta2, steps);

            List<Double> result = new ArrayList<>();

            for (int i = 0; i < parameters.size(); i++) {
                double m =
                        beta1 * firstMoment.get(i) +
                        (1 - beta1) * g.get(i);

                double v =
                        beta2 * secondMoment.get(i) +
                        (1 - beta2) * g.get(i) * g.get(i);

                firstMoment.set(i, m);
                secondMoment.set(i, v);

                double correctedM = m / correction1;
                double correctedV = v / correction2;

                result.add(
                        parameters.get(i) -
                        config.learningRate() * correctedM /
                        (Math.sqrt(correctedV) + config.epsilon())
                );
            }

            return validateOutput(result);
        }

        @Override
        public OptimizerType type() {
            return OptimizerType.ADAM;
        }
    }

    public static final class AdamW extends Adam {

        public AdamW(
                OptimizerConfig config,
                double beta1,
                double beta2
        ) {
            super(config, beta1, beta2);
        }

        @Override
        public List<Double> update(
                List<Double> parameters,
                List<Double> gradients
        ) {
            List<Double> adamResult =
                    super.update(parameters, gradients);

            double decay =
                    config.learningRate() * config.weightDecay();

            List<Double> result = new ArrayList<>();

            for (int i = 0; i < parameters.size(); i++) {
                result.add(
                        adamResult.get(i) -
                        decay * parameters.get(i)
                );
            }

            return validateOutput(result);
        }

        @Override
        public OptimizerType type() {
            return OptimizerType.ADAMW;
        }
    }

    public static final class OptimizerFactory {

        private OptimizerFactory() {}

        public static OptimizerPolicy create(
                OptimizerType type,
                OptimizerConfig config
        ) {
            return switch (type) {
                case MOMENTUM -> new Momentum(config, 0.9);
                case ADAGRAD -> new AdaGrad(config);
                case RMSPROP -> new RMSProp(config, 0.9);
                case ADAM -> new Adam(config, 0.9, 0.999);
                case ADAMW -> new AdamW(config, 0.9, 0.999);
            };
        }
    }

    public record TrainingResult(
            OptimizerType optimizer,
            TrainingStatus status,
            List<Double> parameters,
            double initialLoss,
            double finalLoss,
            int iterations,
            List<Double> lossHistory
    ) {
        public TrainingResult {
            parameters = List.copyOf(parameters);
            lossHistory = List.copyOf(lossHistory);
        }
    }

    public static final class TrainingService {

        private final ObjectiveDefinition objective;
        private TrainingStatus status = TrainingStatus.INITIALIZED;

        public TrainingService(ObjectiveDefinition objective) {
            this.objective = Objects.requireNonNull(objective);
        }

        public synchronized TrainingResult train(
                OptimizerPolicy optimizer,
                List<Double> initialParameters,
                int maximumIterations,
                double tolerance
        ) {
            if (status == TrainingStatus.RUNNING) {
                throw new TrainingException(
                        "Training service is already running."
                );
            }

            if (maximumIterations <= 0 ||
                    !Double.isFinite(tolerance) ||
                    tolerance < 0) {
                throw new ValidationException(
                        "Invalid training configuration."
                );
            }

            List<Double> parameters =
                    new ArrayList<>(initialParameters);

            objective.loss(parameters);

            List<Double> history = new ArrayList<>();

            double initialLoss = objective.loss(parameters);

            history.add(initialLoss);

            status = TrainingStatus.RUNNING;

            try {
                for (int iteration = 0;
                     iteration < maximumIterations;
                     iteration++) {

                    List<Double> gradients =
                            objective.gradient(parameters);

                    parameters = new ArrayList<>(
                            optimizer.update(parameters, gradients)
                    );

                    double currentLoss =
                            objective.loss(parameters);

                    if (!Double.isFinite(currentLoss)) {
                        throw new TrainingException(
                                "Non-finite training loss."
                        );
                    }

                    history.add(currentLoss);

                    if (currentLoss <= tolerance) {
                        break;
                    }
                }

                status = TrainingStatus.COMPLETED;

                return new TrainingResult(
                        optimizer.type(),
                        status,
                        parameters,
                        initialLoss,
                        history.get(history.size() - 1),
                        history.size() - 1,
                        history
                );

            } catch (RuntimeException exception) {
                status = TrainingStatus.FAILED;
                throw exception;
            }
        }

        public TrainingStatus status() {
            return status;
        }
    }

    private static List<Double> zeros(int size) {
        return new ArrayList<>(
                Collections.nCopies(size, 0.0)
        );
    }

    private static void validateVector(
            List<Double> values,
            String name
    ) {
        if (values == null || values.isEmpty()) {
            throw new ValidationException(
                    name + " cannot be empty."
            );
        }

        for (Double value : values) {
            if (value == null || !Double.isFinite(value)) {
                throw new ValidationException(
                        name + " contains invalid numbers."
                );
            }
        }
    }

    private static void requireSameSize(
            List<Double> first,
            List<Double> second
    ) {
        if (first.size() != second.size()) {
            throw new ValidationException(
                    "Vector dimensions differ."
            );
        }
    }

    private static void runTests() {
        OptimizerConfig config =
                OptimizerConfig.standard(0.01);

        for (OptimizerType type : OptimizerType.values()) {
            OptimizerPolicy optimizer =
                    OptimizerFactory.create(type, config);

            List<Double> updated = optimizer.update(
                    List.of(2.0, -1.0),
                    List.of(0.5, -0.25)
            );

            validateVector(updated, "Test result");

            requireSameSize(
                    List.of(2.0, -1.0),
                    updated
            );
        }

        try {
            new OptimizerConfig(-1, 1e-8, 0, 0);
            throw new IllegalStateException(
                    "Invalid learning rate accepted."
            );
        } catch (ValidationException expected) {
            // Expected validation failure.
        }

        try {
            new Adam(config, 0.9, 1.0);
            throw new IllegalStateException(
                    "Invalid beta accepted."
            );
        } catch (ValidationException expected) {
            // Expected validation failure.
        }

        OptimizerConfig decayConfig =
                new OptimizerConfig(0.1, 1e-8, 0.1, 0);

        AdamW adamw = new AdamW(decayConfig, 0.9, 0.999);

        List<Double> decayed =
                adamw.update(List.of(2.0), List.of(0.0));

        if (!(decayed.get(0) < 2.0)) {
            throw new IllegalStateException(
                    "AdamW decay validation failed."
            );
        }

        System.out.println("All Java 17 tests passed.");
    }

    public static void main(String[] args) {
        try {
            runTests();

            ObjectiveDefinition objective =
                    new ObjectiveDefinition(
                            List.of(2.0, -3.0, 1.0),
                            List.of(1.0, 10.0, 50.0)
                    );

            Map<OptimizerType, OptimizerConfig> configurations =
                    new EnumMap<>(OptimizerType.class);

            configurations.put(
                    OptimizerType.MOMENTUM,
                    OptimizerConfig.standard(0.005)
            );

            configurations.put(
                    OptimizerType.ADAGRAD,
                    OptimizerConfig.standard(0.8)
            );

            configurations.put(
                    OptimizerType.RMSPROP,
                    OptimizerConfig.standard(0.02)
            );

            configurations.put(
                    OptimizerType.ADAM,
                    OptimizerConfig.standard(0.08)
            );

            configurations.put(
                    OptimizerType.ADAMW,
                    new OptimizerConfig(0.08, 1e-8, 0.001, 0)
            );

            TrainingService service =
                    new TrainingService(objective);

            List<TrainingResult> results = new ArrayList<>();

            for (OptimizerType type : OptimizerType.values()) {
                OptimizerPolicy optimizer =
                        OptimizerFactory.create(
                                type,
                                configurations.get(type)
                        );

                TrainingResult result = service.train(
                        optimizer,
                        List.of(10.0, 8.0, -5.0),
                        3000,
                        1e-10
                );

                results.add(result);
            }

            System.out.println("\nENTERPRISE OPTIMIZER REPORT");
            System.out.println(
                    "Optimizer      Initial Loss     Final Loss    Iterations"
            );

            for (TrainingResult result : results) {
                System.out.printf(
                        "%-15s %14.6f %15.10f %10d%n",
                        result.optimizer(),
                        result.initialLoss(),
                        result.finalLoss(),
                        result.iterations()
                );
            }

            System.out.println("\nFINAL PARAMETER STATES");

            for (TrainingResult result : results) {
                System.out.println(
                        result.optimizer() + ": " +
                        result.parameters()
                );
            }

            System.out.println(
                    "\nTraining lifecycle: " + service.status()
            );

        } catch (RuntimeException exception) {
            System.err.println(
                    "Enterprise training failure: " +
                    exception.getMessage()
            );

            System.exit(1);
        }
    }
}
