/*
 * OPTIMIZATION MATHEMATICS
 * ========================
 *
 * Industry-style C++17 case study:
 *
 * A constrained resource-allocation optimizer for a small production
 * planning system.
 *
 * The system chooses production quantities for three products while:
 *
 * 1. Minimizing a convex quadratic production-cost function.
 * 2. Respecting resource constraints.
 * 3. Respecting lower and upper production bounds.
 * 4. Demonstrating unconstrained gradient descent.
 * 5. Demonstrating projected gradient descent.
 * 6. Demonstrating Lagrange/KKT interpretation.
 * 7. Reporting objective values, feasibility, and convergence.
 *
 * Compile:
 *
 *     g++ -std=c++17 -O2 optimization_case_study.cpp -o optimization_case_study
 *
 * Run:
 *
 *     ./optimization_case_study
 *
 * Windows with MinGW:
 *
 *     g++ -std=c++17 -O2 optimization_case_study.cpp -o optimization_case_study.exe
 *     optimization_case_study.exe
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// ============================================================================
// 1. BASIC VECTOR OPERATIONS
// ============================================================================

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions do not match."
        );
    }

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double norm(const Vector& vector) {
    return std::sqrt(dot(vector, vector));
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions do not match."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

Vector subtract(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector dimensions do not match."
        );
    }

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

Vector multiply(double scalar, const Vector& vector) {
    Vector result(vector.size());

    for (std::size_t i = 0; i < vector.size(); ++i) {
        result[i] = scalar * vector[i];
    }

    return result;
}

void printVector(
    const std::string& label,
    const Vector& vector
) {
    std::cout << label << "[";

    for (std::size_t i = 0; i < vector.size(); ++i) {
        std::cout << std::fixed
                  << std::setprecision(6)
                  << vector[i];

        if (i + 1 < vector.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}

// ============================================================================
// 2. MATRIX-VECTOR MULTIPLICATION
// ============================================================================

Vector matrixVectorMultiply(
    const Matrix& matrix,
    const Vector& vector
) {
    if (matrix.empty()) {
        return {};
    }

    const std::size_t columns = matrix.front().size();

    if (columns != vector.size()) {
        throw std::invalid_argument(
            "Matrix/vector dimensions do not match."
        );
    }

    Vector result(matrix.size(), 0.0);

    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (matrix[row].size() != columns) {
            throw std::invalid_argument(
                "Matrix rows have inconsistent dimensions."
            );
        }

        result[row] = dot(matrix[row], vector);
    }

    return result;
}

// ============================================================================
// 3. RESOURCE-ALLOCATION MODEL
// ============================================================================

struct Product {
    std::string name;

    // Linear unit cost.
    double unitCost;

    // Resource consumption per unit of product.
    Vector resourceUsage;

    // Business constraints.
    double minimumProduction;
    double maximumProduction;
};

class ProductionModel {
private:
    std::vector<Product> products;

    // Available quantities of each constrained resource.
    Vector resourceCapacity;

    /*
     * Quadratic cost matrix Q.
     *
     * The objective is:
     *
     *     f(x) = 1/2 x^T Q x + c^T x
     *
     * Q is chosen positive definite, making the objective strictly convex.
     *
     * This gives a unique unconstrained minimizer and makes local minimum
     * analysis much simpler.
     */
    Matrix Q;

    Vector linearCosts;

public:
    ProductionModel(
        std::vector<Product> products,
        Vector resourceCapacity,
        Matrix Q,
        Vector linearCosts
    )
        : products(std::move(products)),
          resourceCapacity(std::move(resourceCapacity)),
          Q(std::move(Q)),
          linearCosts(std::move(linearCosts)) {

        const std::size_t n = this->products.size();

        if (n == 0) {
            throw std::invalid_argument(
                "At least one product is required."
            );
        }

        if (this->linearCosts.size() != n) {
            throw std::invalid_argument(
                "Linear cost dimension is incorrect."
            );
        }

        if (this->Q.size() != n) {
            throw std::invalid_argument(
                "Quadratic matrix dimension is incorrect."
            );
        }

        for (const auto& row : this->Q) {
            if (row.size() != n) {
                throw std::invalid_argument(
                    "Quadratic matrix must be square."
                );
            }
        }

        for (const auto& product : this->products) {
            if (
                product.resourceUsage.size()
                != this->resourceCapacity.size()
            ) {
                throw std::invalid_argument(
                    "Product resource dimensions are inconsistent."
                );
            }

            if (
                product.minimumProduction
                > product.maximumProduction
            ) {
                throw std::invalid_argument(
                    "Minimum production exceeds maximum production."
                );
            }
        }
    }

    std::size_t dimension() const {
        return products.size();
    }

    const std::vector<Product>& getProducts() const {
        return products;
    }

    /*
     * Objective:
     *
     *     f(x) = 1/2 x^T Q x + c^T x
     *
     * The quadratic component models increasing marginal cost, such as
     * overtime, machine congestion, or procurement pressure.
     */
    double objective(const Vector& x) const {
        if (x.size() != dimension()) {
            throw std::invalid_argument(
                "Decision vector has incorrect dimension."
            );
        }

        Vector Qx = matrixVectorMultiply(Q, x);

        return 0.5 * dot(x, Qx)
            + dot(linearCosts, x);
    }

    /*
     * Gradient:
     *
     *     grad f(x) = Qx + c
     *
     * This is the central quantity used by gradient-based optimization.
     */
    Vector gradient(const Vector& x) const {
        Vector Qx = matrixVectorMultiply(Q, x);

        return add(Qx, linearCosts);
    }

    /*
     * Resource consumption:
     *
     *     resource_j = sum_i A_ji x_i
     */
    Vector resourceUsage(const Vector& x) const {
        Vector usage(resourceCapacity.size(), 0.0);

        for (std::size_t product = 0;
             product < products.size();
             ++product) {

            for (
                std::size_t resource = 0;
                resource < resourceCapacity.size();
                ++resource
            ) {
                usage[resource] +=
                    products[product]
                        .resourceUsage[resource]
                    * x[product];
            }
        }

        return usage;
    }

    /*
     * A point is feasible if:
     *
     * 1. Every production quantity satisfies its bounds.
     * 2. No resource capacity is exceeded.
     */
    bool isFeasible(
        const Vector& x,
        double tolerance = 1e-8
    ) const {
        if (x.size() != dimension()) {
            return false;
        }

        for (std::size_t i = 0; i < products.size(); ++i) {
            if (
                x[i]
                < products[i].minimumProduction - tolerance
            ) {
                return false;
            }

            if (
                x[i]
                > products[i].maximumProduction + tolerance
            ) {
                return false;
            }
        }

        Vector usage = resourceUsage(x);

        for (std::size_t i = 0;
             i < resourceCapacity.size();
             ++i) {

            if (
                usage[i]
                > resourceCapacity[i] + tolerance
            ) {
                return false;
            }
        }

        return true;
    }

    /*
     * Projection onto simple production bounds.
     *
     * Resource constraints require a more sophisticated projection, so this
     * operation handles the box constraints first.
     */
    Vector projectBounds(const Vector& x) const {
        Vector projected = x;

        for (std::size_t i = 0; i < products.size(); ++i) {
            projected[i] = std::max(
                products[i].minimumProduction,
                std::min(
                    products[i].maximumProduction,
                    projected[i]
                )
            );
        }

        return projected;
    }

    /*
     * A simple feasibility repair method.
     *
     * If a resource is over capacity, production quantities are reduced
     * proportionally among products consuming that resource.
     *
     * This is intentionally a transparent educational method. General
     * polyhedral projection is a separate optimization problem itself.
     */
    Vector repairResourceConstraints(
        const Vector& input
    ) const {
        Vector x = projectBounds(input);

        for (std::size_t resource = 0;
             resource < resourceCapacity.size();
             ++resource) {

            double usage = resourceUsage(x)[resource];

            if (
                usage
                <= resourceCapacity[resource]
            ) {
                continue;
            }

            double excess =
                usage - resourceCapacity[resource];

            double reducible = 0.0;

            for (std::size_t product = 0;
                 product < products.size();
                 ++product) {

                if (
                    products[product]
                        .resourceUsage[resource]
                    > 0.0
                ) {
                    reducible +=
                        (
                            x[product]
                            - products[product]
                                .minimumProduction
                        )
                        * products[product]
                            .resourceUsage[resource];
                }
            }

            if (reducible <= 1e-12) {
                continue;
            }

            for (std::size_t product = 0;
                 product < products.size();
                 ++product) {

                double resourcePerUnit =
                    products[product]
                        .resourceUsage[resource];

                if (resourcePerUnit <= 0.0) {
                    continue;
                }

                double availableReduction =
                    x[product]
                    - products[product]
                        .minimumProduction;

                double contribution =
                    availableReduction
                    * resourcePerUnit;

                double reductionFraction =
                    contribution / reducible;

                double reductionResource =
                    std::min(
                        contribution,
                        excess
                        * reductionFraction
                    );

                double quantityReduction =
                    reductionResource
                    / resourcePerUnit;

                x[product] -= quantityReduction;
            }
        }

        return projectBounds(x);
    }

    void printModel(const Vector& x) const {
        std::cout
            << "\nProduction plan:\n";

        for (std::size_t i = 0;
             i < products.size();
             ++i) {

            std::cout
                << "  "
                << products[i].name
                << ": "
                << std::fixed
                << std::setprecision(4)
                << x[i]
                << "\n";
        }

        Vector usage = resourceUsage(x);

        std::cout << "\nResource usage:\n";

        for (std::size_t i = 0;
             i < usage.size();
             ++i) {

            std::cout
                << "  Resource "
                << i + 1
                << ": "
                << usage[i]
                << " / "
                << resourceCapacity[i]
                << "\n";
        }

        std::cout
            << "\nObjective value: "
            << objective(x)
            << "\n";

        std::cout
            << "Feasible: "
            << (isFeasible(x) ? "yes" : "no")
            << "\n";
    }
};

// ============================================================================
// 4. PROJECTED GRADIENT SOLVER
// ============================================================================

struct OptimizationResult {
    Vector solution;
    std::size_t iterations;
    double objectiveValue;
    bool feasible;
};

OptimizationResult projectedGradientSolve(
    const ProductionModel& model,
    Vector start,
    double learningRate,
    std::size_t maxIterations,
    double tolerance
) {
    if (learningRate <= 0.0) {
        throw std::invalid_argument(
            "Learning rate must be positive."
        );
    }

    Vector x =
        model.repairResourceConstraints(start);

    for (
        std::size_t iteration = 0;
        iteration < maxIterations;
        ++iteration
    ) {
        Vector gradient =
            model.gradient(x);

        Vector candidate =
            subtract(
                x,
                multiply(
                    learningRate,
                    gradient
                )
            );

        /*
         * First apply simple box projection, then repair resource
         * constraints. This is a practical heuristic for the particular
         * production model rather than a general-purpose convex projection
         * algorithm.
         */
        candidate =
            model.repairResourceConstraints(
                candidate
            );

        double movement =
            norm(
                subtract(
                    candidate,
                    x
                )
            );

        x = candidate;

        if (movement <= tolerance) {
            return {
                x,
                iteration + 1,
                model.objective(x),
                model.isFeasible(x)
            };
        }
    }

    return {
        x,
        maxIterations,
        model.objective(x),
        model.isFeasible(x)
    };
}

// ============================================================================
// 5. RESOURCE-CONSTRAINED GREEDY BASELINE
// ============================================================================

Vector greedyFeasibleBaseline(
    const ProductionModel& model,
    std::size_t iterations = 10000
) {
    /*
     * This is not presented as an optimal algorithm.
     *
     * It demonstrates a useful engineering practice: compare an optimizer
     * against a simple baseline. A baseline helps determine whether a
     * sophisticated method is producing meaningful improvements.
     */
    Vector x(model.dimension(), 0.0);

    const auto& products = model.getProducts();

    for (std::size_t i = 0; i < products.size(); ++i) {
        x[i] = products[i].minimumProduction;
    }

    double bestObjective = model.objective(x);

    for (std::size_t iteration = 0;
         iteration < iterations;
         ++iteration) {

        Vector bestCandidate = x;
        bool foundImprovement = false;

        for (std::size_t product = 0;
             product < products.size();
             ++product) {

            if (
                x[product]
                >= products[product]
                    .maximumProduction
                - 1e-12
            ) {
                continue;
            }

            Vector candidate = x;
            candidate[product] += 0.1;

            if (
                !model.isFeasible(candidate)
            ) {
                continue;
            }

            double value =
                model.objective(candidate);

            if (value < bestObjective) {
                bestObjective = value;
                bestCandidate = candidate;
                foundImprovement = true;
            }
        }

        if (!foundImprovement) {
            break;
        }

        x = bestCandidate;
    }

    return x;
}

// ============================================================================
// 6. KKT INTERPRETATION
// ============================================================================

void explainKKTConditions() {
    std::cout
        << "\nKKT conditions for a constrained minimization problem:\n"
        << "  1. Stationarity\n"
        << "  2. Primal feasibility\n"
        << "  3. Dual feasibility\n"
        << "  4. Complementary slackness\n\n";

    std::cout
        << "For an inequality written as g(x) <= 0, its multiplier lambda\n"
        << "must satisfy lambda >= 0 under the standard minimization convention.\n\n";

    std::cout
        << "Complementary slackness requires:\n"
        << "  lambda_i * g_i(x) = 0\n\n";

    std::cout
        << "This means either a constraint is active, or its multiplier is zero.\n";
}

// ============================================================================
// 7. CONVERGENCE AND CONDITIONING DEMONSTRATION
// ============================================================================

void demonstrateLearningRateSensitivity() {
    std::cout
        << "\nLearning-rate sensitivity demonstration:\n";

    const double target = 3.0;

    for (double learningRate : {
        0.05,
        0.2,
        0.8,
        1.1
    }) {
        double x = -10.0;

        for (std::size_t iteration = 0;
             iteration < 100;
             ++iteration) {

            // f(x) = (x-3)^2
            // f'(x) = 2(x-3)
            double gradient =
                2.0 * (x - target);

            x -= learningRate * gradient;

            if (!std::isfinite(x)) {
                break;
            }
        }

        std::cout
            << "  learning rate = "
            << learningRate
            << ", x = "
            << x
            << "\n";
    }

    std::cout
        << "\nFor a quadratic objective, an excessively large step can cause\n"
        << "oscillation or divergence. The acceptable range depends on curvature.\n";
}

// ============================================================================
// 8. VALIDATION
// ============================================================================

void validateModel(
    const ProductionModel& model,
    const OptimizationResult& result
) {
    if (!result.feasible) {
        throw std::runtime_error(
            "Optimizer returned an infeasible production plan."
        );
    }

    if (!std::isfinite(result.objectiveValue)) {
        throw std::runtime_error(
            "Optimizer returned a non-finite objective."
        );
    }

    const auto& products =
        model.getProducts();

    for (std::size_t i = 0;
         i < products.size();
         ++i) {

        if (
            result.solution[i]
            < products[i].minimumProduction - 1e-7
            ||
            result.solution[i]
            > products[i].maximumProduction + 1e-7
        ) {
            throw std::runtime_error(
                "Production bound violation detected."
            );
        }
    }
}

// ============================================================================
// 9. MAIN CASE STUDY
// ============================================================================

int main() {
    try {
        std::cout
            << "============================================================\n"
            << "OPTIMIZATION MATHEMATICS: PRODUCTION PLANNING CASE STUDY\n"
            << "============================================================\n";

        /*
         * Three products:
         *
         * Product A: high machine consumption
         * Product B: high labor consumption
         * Product C: balanced resource consumption
         *
         * The objective is intentionally a cost minimization problem.
         *
         * In real planning systems, business revenue could be incorporated
         * by minimizing:
         *
         *     cost - revenue
         *
         * or equivalently maximizing:
         *
         *     revenue - cost
         *
         * Subject to resource and policy constraints.
         */
        std::vector<Product> products = {
            {
                "Product A",
                10.0,
                {3.0, 1.0, 2.0},
                0.0,
                20.0
            },
            {
                "Product B",
                8.0,
                {1.0, 4.0, 1.0},
                0.0,
                20.0
            },
            {
                "Product C",
                7.0,
                {2.0, 2.0, 4.0},
                0.0,
                20.0
            }
        };

        Vector resourceCapacity = {
            35.0,
            40.0,
            35.0
        };

        /*
         * Positive-definite quadratic cost matrix.
         *
         * Diagonal terms represent increasing marginal costs.
         * Off-diagonal terms represent interactions between product volumes.
         */
        Matrix Q = {
            {2.0, 0.2, 0.1},
            {0.2, 2.5, 0.15},
            {0.1, 0.15, 1.8}
        };

        Vector linearCosts = {
            10.0,
            8.0,
            7.0
        };

        ProductionModel model(
            products,
            resourceCapacity,
            Q,
            linearCosts
        );

        std::cout
            << "\nCASE STUDY:\n"
            << "A production manager must choose quantities for three products.\n"
            << "The goal is to minimize a convex quadratic cost while obeying\n"
            << "resource capacities and production bounds.\n";

        // --------------------------------------------------------------------
        // STEP 1: STARTING POINT
        // --------------------------------------------------------------------

        Vector initial = {
            5.0,
            5.0,
            5.0
        };

        std::cout
            << "\nInitial state:\n";

        model.printModel(initial);

        // --------------------------------------------------------------------
        // STEP 2: GREEDY BASELINE
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "GREEDY BASELINE\n"
            << "============================================================\n";

        Vector baseline =
            greedyFeasibleBaseline(model);

        model.printModel(baseline);

        // --------------------------------------------------------------------
        // STEP 3: PROJECTED GRADIENT OPTIMIZATION
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "PROJECTED GRADIENT OPTIMIZATION\n"
            << "============================================================\n";

        OptimizationResult result =
            projectedGradientSolve(
                model,
                initial,
                0.02,
                5000,
                1e-8
            );

        model.printModel(result.solution);

        std::cout
            << "\nIterations: "
            << result.iterations
            << "\n";

        validateModel(
            model,
            result
        );

        // --------------------------------------------------------------------
        // STEP 4: BASELINE COMPARISON
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "BASELINE VERSUS OPTIMIZER\n"
            << "============================================================\n";

        double baselineObjective =
            model.objective(baseline);

        double optimizedObjective =
            model.objective(result.solution);

        std::cout
            << "Baseline objective:   "
            << baselineObjective
            << "\n";

        std::cout
            << "Optimized objective:  "
            << optimizedObjective
            << "\n";

        /*
         * Because this is a cost-minimization problem, lower objective values
         * correspond to lower modeled cost. The comparison is descriptive;
         * the optimizer's mathematical guarantees depend on the assumptions
         * of the objective, feasible set, and algorithm.
         */
        std::cout
            << "Objective difference:  "
            << baselineObjective
            - optimizedObjective
            << "\n";

        // --------------------------------------------------------------------
        // STEP 5: KKT THEORY
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "CONSTRAINED OPTIMIZATION THEORY\n"
            << "============================================================\n";

        explainKKTConditions();

        // --------------------------------------------------------------------
        // STEP 6: LEARNING-RATE BEHAVIOR
        // --------------------------------------------------------------------

        demonstrateLearningRateSensitivity();

        // --------------------------------------------------------------------
        // STEP 7: COMPLEXITY DISCUSSION
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "COMPLEXITY AND ENGINEERING CONSIDERATIONS\n"
            << "============================================================\n";

        std::cout
            << "Vector operations are O(n).\n"
            << "A dense matrix-vector multiplication is O(n^2).\n"
            << "Each projected-gradient iteration therefore costs approximately\n"
            << "O(n^2) for a dense quadratic objective, excluding constraint repair.\n\n";

        std::cout
            << "For large sparse systems, sparse matrix representations can reduce\n"
            << "memory use and arithmetic cost. Production solvers also use more\n"
            << "sophisticated projections, line searches, active-set methods,\n"
            << "interior-point methods, or specialized quadratic programming methods.\n";

        // --------------------------------------------------------------------
        // STEP 8: NUMERICAL AND SECURITY CONSIDERATIONS
        // --------------------------------------------------------------------

        std::cout
            << "\n============================================================\n"
            << "NUMERICAL AND OPERATIONAL CONSIDERATIONS\n"
            << "============================================================\n";

        std::cout
            << "1. Validate all numerical inputs.\n"
            << "2. Reject NaN and infinite values in production systems.\n"
            << "3. Use tolerances instead of exact floating-point equality.\n"
            << "4. Monitor constraint violations independently from objective value.\n"
            << "5. Record solver iterations and stopping reasons.\n"
            << "6. Bound resource and quantity inputs to prevent pathological values.\n"
            << "7. Protect optimization APIs against unbounded user-controlled workloads.\n"
            << "8. Treat optimization output as a decision-support result that must\n"
            << "   remain traceable to its inputs, constraints, and objective definition.\n";

        // --------------------------------------------------------------------
        // STEP 9: FINAL VALIDATION
        // --------------------------------------------------------------------

        if (!model.isFeasible(result.solution)) {
            throw std::runtime_error(
                "Final feasibility validation failed."
            );
        }

        if (
            !std::isfinite(
                model.objective(result.solution)
            )
        ) {
            throw std::runtime_error(
                "Final objective validation failed."
            );
        }

        std::cout
            << "\n============================================================\n"
            << "CASE STUDY VALIDATION PASSED\n"
            << "============================================================\n";

        std::cout
            << "The final production plan satisfies the implemented\n"
            << "production bounds and resource-capacity constraints.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "\nERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}
