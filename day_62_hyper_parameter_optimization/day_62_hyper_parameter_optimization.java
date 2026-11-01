import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

public class HyperparameterOptimization {

    enum SearchStrategy {
        GRID,
        RANDOM
    }

    enum TrialState {
        CREATED,
        RUNNING,
        COMPLETED,
        FAILED
    }

    record Hyperparameters(
        double learningRate,
        int maxDepth,
        int estimators
    ) {
        Hyperparameters {
            if (!Double.isFinite(learningRate) || learningRate <= 0) {
                throw new IllegalArgumentException(
                    "Learning rate must be positive and finite."
                );
            }
            if (maxDepth < 1) {
                throw new IllegalArgumentException(
                    "Maximum depth must be positive."
                );
            }
            if (estimators < 1) {
                throw new IllegalArgumentException(
                    "Estimator count must be positive."
                );
            }
        }
    }

    record TrialResult(
        long trialId,
        Hyperparameters parameters,
        double score,
        double duration,
        TrialState state
    ) {}

    static final class SearchSpace {
        private final List<Double> learningRates;
        private final List<Integer> maxDepths;
        private final List<Integer> estimators;

        SearchSpace(
            List<Double> learningRates,
            List<Integer> maxDepths,
            List<Integer> estimators
        ) {
            this.learningRates = List.copyOf(learningRates);
            this.maxDepths = List.copyOf(maxDepths);
            this.estimators = List.copyOf(estimators);
            validate();
        }

        private void validate() {
            if (
                learningRates.isEmpty() ||
                maxDepths.isEmpty() ||
                estimators.isEmpty()
            ) {
                throw new IllegalArgumentException(
                    "Every search-space dimension requires values."
                );
            }

            if (new HashSet<>(learningRates).size() != learningRates.size()
                || new HashSet<>(maxDepths).size() != maxDepths.size()
                || new HashSet<>(estimators).size() != estimators.size()) {
                throw new IllegalArgumentException(
                    "Search-space dimensions cannot contain duplicates."
                );
            }
        }

        long size() {
            return (long) learningRates.size()
                * maxDepths.size()
                * estimators.size();
        }

        List<Double> learningRates() {
            return learningRates;
        }

        List<Integer> maxDepths() {
            return maxDepths;
        }

        List<Integer> estimators() {
            return estimators;
        }
    }

    interface TrialEvaluator {
        TrialResult evaluate(long trialId, Hyperparameters parameters);
    }

    static final class EvaluationException extends RuntimeException {
        EvaluationException(String message) {
            super(message);
        }
    }

    static final class ValidationEvaluator implements TrialEvaluator {
        private final Random random;

        ValidationEvaluator(long seed) {
            this.random = new Random(seed);
        }

        @Override
        public TrialResult evaluate(
            long trialId,
            Hyperparameters parameters
        ) {
            double lrPenalty = Math.abs(
                Math.log10(parameters.learningRate())
                    - Math.log10(0.05)
            );

            double depthPenalty =
                Math.abs(parameters.maxDepth() - 6) * 0.018;

            double estimatorPenalty =
                Math.abs(parameters.estimators() - 200) / 10000.0;

            double noise = -0.003 + random.nextDouble() * 0.006;

            double score = Math.max(
                0.0,
                Math.min(
                    1.0,
                    0.96
                        - lrPenalty * 0.09
                        - depthPenalty
                        - estimatorPenalty
                        + noise
                )
            );

            double duration =
                0.5
                    + parameters.maxDepth() * 0.08
                    + parameters.estimators() * 0.004;

            return new TrialResult(
                trialId,
                parameters,
                score,
                duration,
                TrialState.COMPLETED
            );
        }
    }

    static final class OptimizationService {
        private final SearchSpace searchSpace;
        private final TrialEvaluator evaluator;
        private final Random random;
        private long nextTrialId = 1;

        OptimizationService(
            SearchSpace searchSpace,
            TrialEvaluator evaluator,
            long seed
        ) {
            this.searchSpace = Objects.requireNonNull(searchSpace);
            this.evaluator = Objects.requireNonNull(evaluator);
            this.random = new Random(seed);
        }

        List<Hyperparameters> grid() {
            List<Hyperparameters> configurations = new ArrayList<>();

            for (double rate : searchSpace.learningRates()) {
                for (int depth : searchSpace.maxDepths()) {
                    for (int count : searchSpace.estimators()) {
                        configurations.add(
                            new Hyperparameters(rate, depth, count)
                        );
                    }
                }
            }

            return configurations;
        }

        List<Hyperparameters> randomSearchSpace(int requested) {
            if (requested < 1) {
                throw new IllegalArgumentException(
                    "Random trial count must be positive."
                );
            }

            int target = (int) Math.min(requested, searchSpace.size());
            Set<Hyperparameters> configurations = new HashSet<>();

            while (configurations.size() < target) {
                double rate = searchSpace.learningRates().get(
                    random.nextInt(searchSpace.learningRates().size())
                );

                int depth = searchSpace.maxDepths().get(
                    random.nextInt(searchSpace.maxDepths().size())
                );

                int count = searchSpace.estimators().get(
                    random.nextInt(searchSpace.estimators().size())
                );

                configurations.add(
                    new Hyperparameters(rate, depth, count)
                );
            }

            return new ArrayList<>(configurations);
        }

        List<TrialResult> execute(
            SearchStrategy strategy,
            int randomTrials
        ) {
            List<Hyperparameters> configurations =
                strategy == SearchStrategy.GRID
                    ? grid()
                    : randomSearchSpace(randomTrials);

            List<TrialResult> results = new ArrayList<>();

            for (Hyperparameters configuration : configurations) {
                long id = nextTrialId++;

                try {
                    results.add(evaluator.evaluate(id, configuration));
                } catch (RuntimeException exception) {
                    results.add(
                        new TrialResult(
                            id,
                            configuration,
                            Double.NaN,
                            0.0,
                            TrialState.FAILED
                        )
                    );
                }
            }

            return results;
        }
    }

    static TrialResult best(List<TrialResult> trials) {
        return trials.stream()
            .filter(trial -> trial.state() == TrialState.COMPLETED)
            .max(Comparator.comparingDouble(TrialResult::score))
            .orElseThrow(
                () -> new EvaluationException("No successful trials.")
            );
    }

    static double averageScore(List<TrialResult> trials) {
        return trials.stream()
            .filter(trial -> trial.state() == TrialState.COMPLETED)
            .mapToDouble(TrialResult::score)
            .average()
            .orElseThrow(
                () -> new EvaluationException("No completed trials.")
            );
    }

    static void printReport(
        SearchStrategy strategy,
        List<TrialResult> trials
    ) {
        TrialResult best = best(trials);

        System.out.println("\n" + strategy + " SEARCH");
        System.out.println("-".repeat(60));
        System.out.println("Trials: " + trials.size());
        System.out.printf(
            "Average score: %.4f%n",
            averageScore(trials)
        );
        System.out.printf(
            "Best score: %.4f%n",
            best.score()
        );
        System.out.println(
            "Learning rate: " + best.parameters().learningRate()
        );
        System.out.println(
            "Max depth: " + best.parameters().maxDepth()
        );
        System.out.println(
            "Estimators: " + best.parameters().estimators()
        );
        System.out.printf(
            "Training cost units: %.2f%n",
            best.duration()
        );
    }

    static void validateBehavior() {
        SearchSpace space = new SearchSpace(
            List.of(0.01, 0.05),
            List.of(3, 6),
            List.of(100, 200)
        );

        if (space.size() != 8) {
            throw new AssertionError("Unexpected search-space size.");
        }

        OptimizationService service =
            new OptimizationService(
                space,
                new ValidationEvaluator(11),
                11
            );

        List<TrialResult> results =
            service.execute(SearchStrategy.GRID, 0);

        if (results.size() != 8) {
            throw new AssertionError("Grid evaluation count is incorrect.");
        }

        try {
            new Hyperparameters(-0.1, 4, 100);
            throw new AssertionError("Invalid learning rate was accepted.");
        } catch (IllegalArgumentException expected) {
            // Expected validation failure.
        }
    }

    public static void main(String[] args) {
        validateBehavior();

        SearchSpace space = new SearchSpace(
            List.of(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
            List.of(2, 4, 6, 8, 10),
            List.of(50, 100, 200, 400)
        );

        OptimizationService service =
            new OptimizationService(
                space,
                new ValidationEvaluator(42),
                42
            );

        List<TrialResult> gridResults =
            service.execute(SearchStrategy.GRID, 0);

        List<TrialResult> randomResults =
            service.execute(SearchStrategy.RANDOM, 20);

        printReport(SearchStrategy.GRID, gridResults);
        printReport(SearchStrategy.RANDOM, randomResults);

        System.out.println("\nENTERPRISE OPTIMIZATION VIEW");
        System.out.println("-".repeat(60));
        System.out.println(
            "Possible configurations: " + space.size()
        );
        System.out.println(
            "Grid evaluations: " + gridResults.size()
        );
        System.out.println(
            "Random evaluations: " + randomResults.size()
        );

        System.out.println("\nTop three random trials:");

        randomResults.stream()
            .filter(t -> t.state() == TrialState.COMPLETED)
            .sorted(
                Comparator.comparingDouble(TrialResult::score)
                    .reversed()
            )
            .limit(3)
            .forEach(trial ->
                System.out.printf(
                    "Trial %d: lr=%.3f depth=%d estimators=%d score=%.4f%n",
                    trial.trialId(),
                    trial.parameters().learningRate(),
                    trial.parameters().maxDepth(),
                    trial.parameters().estimators(),
                    trial.score()
                )
            );
    }
}
