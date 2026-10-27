import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public class PCARepositoryGovernance {

    private static final double EPSILON = 1e-12;

    enum ReviewState {
        DRAFT,
        OPEN,
        CHANGES_REQUESTED,
        APPROVED,
        MERGED,
        CLOSED
    }

    enum CheckState {
        PASS,
        FAIL,
        PENDING
    }

    record StatusCheck(String name, CheckState state) {
        StatusCheck {
            Objects.requireNonNull(name, "Check name is required");
            Objects.requireNonNull(state, "Check state is required");

            if (name.isBlank()) {
                throw new IllegalArgumentException("Check name cannot be blank");
            }
        }
    }

    record ReviewPolicy(
        int requiredApprovals,
        boolean requirePassingChecks
    ) {
        ReviewPolicy {
            if (requiredApprovals < 0) {
                throw new IllegalArgumentException(
                    "Required approvals cannot be negative"
                );
            }
        }
    }

    record PullRequest(
        long id,
        String sourceBranch,
        String targetBranch,
        int changedFiles,
        double reviewLatencyHours,
        double checkLatencyMinutes,
        int reviewComments,
        ReviewState state,
        List<StatusCheck> checks
    ) {
        PullRequest {
            if (id <= 0) {
                throw new IllegalArgumentException("Invalid Pull Request ID");
            }

            if (sourceBranch == null || sourceBranch.isBlank()) {
                throw new IllegalArgumentException("Source branch is required");
            }

            if (targetBranch == null || targetBranch.isBlank()) {
                throw new IllegalArgumentException("Target branch is required");
            }

            if (changedFiles < 0 ||
                reviewLatencyHours < 0 ||
                checkLatencyMinutes < 0 ||
                reviewComments < 0) {
                throw new IllegalArgumentException(
                    "Repository metrics cannot be negative"
                );
            }

            checks = List.copyOf(checks);
        }
    }

    static final class PCAException extends RuntimeException {
        PCAException(String message) {
            super(message);
        }
    }

    record PCAResult(
        double[] means,
        double[][] covariance,
        double[] eigenvalues,
        double[][] eigenvectors,
        double[] explainedVariance,
        double[] cumulativeVariance,
        double[][] transformed
    ) {}

    static final class MatrixOperations {

        static void validate(double[][] data) {
            if (data == null || data.length < 2) {
                throw new PCAException(
                    "At least two observations are required"
                );
            }

            if (data[0] == null || data[0].length == 0) {
                throw new PCAException("At least one feature is required");
            }

            int width = data[0].length;

            for (double[] row : data) {
                if (row == null || row.length != width) {
                    throw new PCAException(
                        "All observations must have equal dimensions"
                    );
                }

                for (double value : row) {
                    if (!Double.isFinite(value)) {
                        throw new PCAException(
                            "All observations must be finite"
                        );
                    }
                }
            }
        }

        static double dot(double[] a, double[] b) {
            if (a.length != b.length) {
                throw new PCAException("Vector dimension mismatch");
            }

            double result = 0.0;

            for (int i = 0; i < a.length; i++) {
                result += a[i] * b[i];
            }

            return result;
        }

        static double norm(double[] vector) {
            return Math.sqrt(dot(vector, vector));
        }

        static double[] normalize(double[] vector) {
            double length = norm(vector);

            if (length < EPSILON) {
                throw new PCAException("Cannot normalize zero vector");
            }

            double[] result = vector.clone();

            for (int i = 0; i < result.length; i++) {
                result[i] /= length;
            }

            return result;
        }

        static double[] multiply(double[][] matrix, double[] vector) {
            double[] result = new double[matrix.length];

            for (int i = 0; i < matrix.length; i++) {
                result[i] = dot(matrix[i], vector);
            }

            return result;
        }

        static double[][] copy(double[][] matrix) {
            double[][] result = new double[matrix.length][];

            for (int i = 0; i < matrix.length; i++) {
                result[i] = matrix[i].clone();
            }

            return result;
        }
    }

    static final class PCAService {

        PCAResult fit(double[][] data, int components) {
            MatrixOperations.validate(data);

            int dimensions = data[0].length;

            if (components < 1 || components > dimensions) {
                throw new PCAException("Invalid component count");
            }

            double[] means = new double[dimensions];

            for (double[] row : data) {
                for (int j = 0; j < dimensions; j++) {
                    means[j] += row[j];
                }
            }

            for (int j = 0; j < dimensions; j++) {
                means[j] /= data.length;
            }

            double[][] centered = new double[data.length][dimensions];

            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < dimensions; j++) {
                    centered[i][j] = data[i][j] - means[j];
                }
            }

            double[][] covariance = new double[dimensions][dimensions];

            for (int i = 0; i < dimensions; i++) {
                for (int j = i; j < dimensions; j++) {
                    double sum = 0.0;

                    for (double[] row : centered) {
                        sum += row[i] * row[j];
                    }

                    double value = sum / (data.length - 1);
                    covariance[i][j] = value;
                    covariance[j][i] = value;
                }
            }

            EigenDecomposition decomposition =
                eigenDecompose(covariance);

            double totalVariance = 0.0;

            for (double value : decomposition.values) {
                totalVariance += Math.max(0.0, value);
            }

            if (totalVariance <= EPSILON) {
                throw new PCAException(
                    "The dataset has no measurable variance"
                );
            }

            double[] explained =
                new double[decomposition.values.length];

            double[] cumulative =
                new double[decomposition.values.length];

            double running = 0.0;

            for (int i = 0; i < decomposition.values.length; i++) {
                explained[i] =
                    Math.max(0.0, decomposition.values[i])
                    / totalVariance;

                running += explained[i];
                cumulative[i] = running;
            }

            double[][] transformed =
                new double[data.length][components];

            for (int i = 0; i < centered.length; i++) {
                for (int component = 0;
                     component < components;
                     component++) {

                    transformed[i][component] =
                        MatrixOperations.dot(
                            centered[i],
                            decomposition.vectors[component]
                        );
                }
            }

            return new PCAResult(
                means,
                covariance,
                decomposition.values,
                decomposition.vectors,
                explained,
                cumulative,
                transformed
            );
        }

        private record EigenDecomposition(
            double[] values,
            double[][] vectors
        ) {}

        private EigenDecomposition eigenDecompose(double[][] input) {
            double[][] working = MatrixOperations.copy(input);

            List<Double> values = new ArrayList<>();
            List<double[]> vectors = new ArrayList<>();

            for (int component = 0;
                 component < input.length;
                 component++) {

                double[] vector = new double[input.length];

                for (int i = 0; i < vector.length; i++) {
                    vector[i] = 1.0 + (double) i / vector.length;
                }

                vector = MatrixOperations.normalize(vector);

                double eigenvalue = 0.0;

                boolean converged = false;

                for (int iteration = 0;
                     iteration < 10_000;
                     iteration++) {

                    double[] candidate =
                        MatrixOperations.normalize(
                            MatrixOperations.multiply(
                                working,
                                vector
                            )
                        );

                    double newEigenvalue =
                        MatrixOperations.dot(
                            candidate,
                            MatrixOperations.multiply(
                                working,
                                candidate
                            )
                        );

                    double difference = 0.0;

                    for (int i = 0; i < vector.length; i++) {
                        double delta =
                            candidate[i] - vector[i];

                        difference += delta * delta;
                    }

                    difference = Math.sqrt(difference);
                    vector = candidate;

                    if (
                        Math.abs(newEigenvalue - eigenvalue)
                            < EPSILON &&
                        difference < EPSILON
                    ) {
                        eigenvalue = newEigenvalue;
                        converged = true;
                        break;
                    }

                    eigenvalue = newEigenvalue;
                }

                if (!converged) {
                    throw new PCAException(
                        "Eigenvalue iteration failed to converge"
                    );
                }

                int pivot = 0;

                for (int i = 1; i < vector.length; i++) {
                    if (Math.abs(vector[i]) > Math.abs(vector[pivot])) {
                        pivot = i;
                    }
                }

                if (vector[pivot] < 0) {
                    for (int i = 0; i < vector.length; i++) {
                        vector[i] = -vector[i];
                    }
                }

                if (Math.abs(eigenvalue) < EPSILON) {
                    eigenvalue = 0.0;
                }

                values.add(eigenvalue);
                vectors.add(vector);

                for (int i = 0; i < working.length; i++) {
                    for (int j = 0; j < working.length; j++) {
                        working[i][j] -=
                            eigenvalue * vector[i] * vector[j];
                    }
                }
            }

            Integer[] order =
                new Integer[values.size()];

            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }

            Arrays.sort(
                order,
                Comparator.comparingDouble(
                    (Integer i) -> values.get(i)
                ).reversed()
            );

            double[] sortedValues =
                new double[values.size()];

            double[][] sortedVectors =
                new double[vectors.size()][];

            for (int i = 0; i < order.length; i++) {
                sortedValues[i] = values.get(order[i]);
                sortedVectors[i] =
                    vectors.get(order[i]).clone();
            }

            return new EigenDecomposition(
                sortedValues,
                sortedVectors
            );
        }

        int componentsForVariance(
            PCAResult result,
            double threshold
        ) {
            if (threshold <= 0 || threshold > 1) {
                throw new PCAException(
                    "Variance threshold must be in (0, 1]"
                );
            }

            for (int i = 0;
                 i < result.cumulativeVariance().length;
                 i++) {

                if (result.cumulativeVariance()[i] >= threshold) {
                    return i + 1;
                }
            }

            return result.eigenvalues().length;
        }
    }

    static final class MergeEligibilityService {

        boolean eligible(
            PullRequest pullRequest,
            ReviewPolicy policy,
            int validApprovals
        ) {
            if (pullRequest.state() == ReviewState.CLOSED ||
                pullRequest.state() == ReviewState.MERGED) {
                return false;
            }

            if (validApprovals < policy.requiredApprovals()) {
                return false;
            }

            if (
                policy.requirePassingChecks() &&
                pullRequest.checks().stream()
                    .anyMatch(check -> check.state() != CheckState.PASS)
            ) {
                return false;
            }

            return pullRequest.state() == ReviewState.APPROVED;
        }
    }

    static void printMatrix(String title, double[][] matrix) {
        System.out.println("\n" + title);

        for (double[] row : matrix) {
            System.out.printf("  ");

            for (double value : row) {
                System.out.printf("%10.5f ", value);
            }

            System.out.println();
        }
    }

    public static void main(String[] args) {
        try {
            List<StatusCheck> checks = List.of(
                new StatusCheck("unit-tests", CheckState.PASS),
                new StatusCheck("security-scan", CheckState.PASS),
                new StatusCheck("integration-tests", CheckState.PASS)
            );

            PullRequest pullRequest = new PullRequest(
                1042,
                "feature/repository-policy",
                "main",
                14,
                6.5,
                8.0,
                7,
                ReviewState.APPROVED,
                checks
            );

            ReviewPolicy policy = new ReviewPolicy(
                2,
                true
            );

            MergeEligibilityService eligibilityService =
                new MergeEligibilityService();

            boolean eligible =
                eligibilityService.eligible(
                    pullRequest,
                    policy,
                    2
                );

            System.out.println(
                "Enterprise repository governance state"
            );
            System.out.println(
                "Pull Request #" + pullRequest.id()
            );
            System.out.println(
                "Merge eligible: " + eligible
            );

            /*
             * PCA input represents repository operations metrics.
             * The dimensions are deliberately correlated so that the
             * principal directions have an interpretable operational meaning.
             */
            double[][] telemetry = {
                {2.0,  8.0,  4.0, 3.0},
                {3.0,  9.0,  5.0, 3.5},
                {1.5,  6.0,  3.0, 2.0},
                {5.0, 15.0, 8.0, 7.0},
                {6.0, 18.0, 9.0, 8.0},
                {7.0, 21.0, 11.0, 9.0},
                {2.5,  7.0,  4.0, 2.5},
                {4.5, 13.0, 7.0, 6.0}
            };

            PCAService pcaService = new PCAService();

            PCAResult result =
                pcaService.fit(
                    telemetry,
                    telemetry[0].length
                );

            printMatrix(
                "Covariance matrix",
                result.covariance()
            );

            System.out.println("\nPrincipal components");

            for (int i = 0;
                 i < result.eigenvalues().length;
                 i++) {

                System.out.printf(
                    "  PC%d eigenvalue=%.6f explained=%.2f%% cumulative=%.2f%%%n",
                    i + 1,
                    result.eigenvalues()[i],
                    result.explainedVariance()[i] * 100,
                    result.cumulativeVariance()[i] * 100
                );
            }

            System.out.println("\nEigenvectors");

            for (int i = 0;
                 i < result.eigenvectors().length;
                 i++) {

                System.out.print("  PC" + (i + 1) + ": ");

                for (double value :
                    result.eigenvectors()[i]) {

                    System.out.printf("%.6f ", value);
                }

                System.out.println();
            }

            int selected =
                pcaService.componentsForVariance(
                    result,
                    0.95
                );

            System.out.println(
                "\nComponents for 95% explained variance: "
                    + selected
            );

            double explainedSum =
                Arrays.stream(
                    result.explainedVariance()
                ).sum();

            if (Math.abs(explainedSum - 1.0) > 1e-9) {
                throw new PCAException(
                    "Explained variance does not sum to one"
                );
            }

            try {
                pcaService.fit(
                    new double[][] {
                        {1.0, 1.0},
                        {1.0, 1.0}
                    },
                    2
                );
            } catch (PCAException expected) {
                System.out.println(
                    "\nExpected PCA failure: "
                        + expected.getMessage()
                );
            }

            /*
             * The governance service and PCA service have different
             * responsibilities. Governance determines whether a PR can
             * merge; PCA analyzes operational measurements. PCA does not
             * replace branch protection or review policy enforcement.
             */
            System.out.println(
                "\nDesign distinction: merge policy enforcement is deterministic; "
                    + "PCA is an exploratory statistical transformation."
            );

        } catch (RuntimeException error) {
            System.err.println(
                "Processing failed: " + error.getMessage()
            );
            System.exit(1);
        }
    }
}
