/*
 * Bayes' Theorem C++ Case Study
 *
 * Scenario:
 * A financial institution receives transaction evidence and must estimate
 * whether a transaction is fraudulent. The program models:
 *
 *   - prior probability
 *   - conditional probability
 *   - likelihood
 *   - evidence
 *   - posterior probability
 *   - multiple hypotheses
 *   - sequential evidence
 *   - likelihood ratios
 *   - Bayesian decision theory
 *   - numerical stability through logarithms
 *   - validation and edge cases
 *
 * Compile:
 *   g++ -std=c++17 -O2 -Wall -Wextra -pedantic bayes_theorem.cpp -o bayes
 *
 * Run:
 *   ./bayes
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

void validateProbability(double probability, const std::string& name) {
    if (!std::isfinite(probability) ||
        probability < 0.0 ||
        probability > 1.0) {
        throw std::invalid_argument(
            name + " must be between 0 and 1."
        );
    }
}

double probabilityToOdds(double probability) {
    validateProbability(probability, "Probability");

    if (probability == 0.0) {
        return 0.0;
    }

    if (probability == 1.0) {
        return std::numeric_limits<double>::infinity();
    }

    return probability / (1.0 - probability);
}

double oddsToProbability(double odds) {
    if (odds < 0.0 || std::isnan(odds)) {
        throw std::invalid_argument(
            "Odds cannot be negative."
        );
    }

    if (std::isinf(odds)) {
        return 1.0;
    }

    return odds / (1.0 + odds);
}


// ---------------------------------------------------------------------------
// Basic Bayesian calculations
// ---------------------------------------------------------------------------

double conditionalProbability(
    double jointProbability,
    double conditionProbability
) {
    validateProbability(jointProbability, "P(A and B)");
    validateProbability(conditionProbability, "P(B)");

    if (conditionProbability == 0.0) {
        throw std::invalid_argument(
            "Conditional probability is undefined when P(B) is zero."
        );
    }

    if (jointProbability > conditionProbability) {
        throw std::invalid_argument(
            "P(A and B) cannot exceed P(B)."
        );
    }

    return jointProbability / conditionProbability;
}

struct BayesResult {
    double prior;
    double likelihood;
    double competingLikelihood;
    double evidence;
    double posterior;
};

BayesResult bayesBinary(
    double prior,
    double likelihood,
    double competingLikelihood
) {
    validateProbability(prior, "Prior");
    validateProbability(likelihood, "Likelihood");
    validateProbability(
        competingLikelihood,
        "Competing likelihood"
    );

    const double complementaryPrior = 1.0 - prior;

    // Law of total probability:
    //
    // P(E) =
    //     P(E|H)P(H) +
    //     P(E|not H)P(not H)
    //
    // This denominator is essential. Dividing only by P(E|H) would
    // incorrectly treat sensitivity as the posterior probability.
    const double evidence =
        likelihood * prior +
        competingLikelihood * complementaryPrior;

    if (evidence == 0.0) {
        throw std::invalid_argument(
            "Evidence has zero probability under the model."
        );
    }

    const double posterior =
        likelihood * prior / evidence;

    return {
        prior,
        likelihood,
        competingLikelihood,
        evidence,
        posterior
    };
}


// ---------------------------------------------------------------------------
// Bayesian hypothesis representation
// ---------------------------------------------------------------------------

struct Hypothesis {
    std::string name;
    double prior;
    double likelihood;
    double unnormalizedPosterior;
    double posterior;
};

std::vector<Hypothesis> posteriorDistribution(
    const std::vector<std::string>& names,
    const std::vector<double>& priors,
    const std::vector<double>& likelihoods
) {
    if (names.size() != priors.size() ||
        priors.size() != likelihoods.size()) {
        throw std::invalid_argument(
            "Hypotheses, priors, and likelihoods must have equal sizes."
        );
    }

    if (names.empty()) {
        throw std::invalid_argument(
            "At least one hypothesis is required."
        );
    }

    const double priorSum =
        std::accumulate(
            priors.begin(),
            priors.end(),
            0.0
        );

    if (std::abs(priorSum - 1.0) > 1e-9) {
        throw std::invalid_argument(
            "Priors must sum to 1."
        );
    }

    std::vector<Hypothesis> result;
    result.reserve(names.size());

    double evidence = 0.0;

    for (std::size_t i = 0; i < names.size(); ++i) {
        validateProbability(
            priors[i],
            "Prior"
        );

        validateProbability(
            likelihoods[i],
            "Likelihood"
        );

        const double unnormalized =
            priors[i] * likelihoods[i];

        result.push_back({
            names[i],
            priors[i],
            likelihoods[i],
            unnormalized,
            0.0
        });

        evidence += unnormalized;
    }

    if (evidence == 0.0) {
        throw std::invalid_argument(
            "All hypotheses assign zero probability to the evidence."
        );
    }

    for (auto& hypothesis : result) {
        hypothesis.posterior =
            hypothesis.unnormalizedPosterior /
            evidence;
    }

    return result;
}


// ---------------------------------------------------------------------------
// Transaction fraud case-study domain model
// ---------------------------------------------------------------------------

enum class TransactionType {
    Retail,
    Transfer,
    CashWithdrawal
};

std::string transactionTypeToString(TransactionType type) {
    switch (type) {
        case TransactionType::Retail:
            return "Retail";
        case TransactionType::Transfer:
            return "Transfer";
        case TransactionType::CashWithdrawal:
            return "Cash withdrawal";
    }

    return "Unknown";
}

struct Transaction {
    std::string id;
    TransactionType type;
    double amount;
    bool international;
    bool unusualLocation;
    bool unusualDevice;
};

struct EvidenceProfile {
    double fraudPrior;
    double amountEvidenceGivenFraud;
    double amountEvidenceGivenLegitimate;

    double locationEvidenceGivenFraud;
    double locationEvidenceGivenLegitimate;

    double deviceEvidenceGivenFraud;
    double deviceEvidenceGivenLegitimate;
};


// ---------------------------------------------------------------------------
// Evidence likelihood model
// ---------------------------------------------------------------------------

class FraudBayesianModel {
public:
    explicit FraudBayesianModel(
        const EvidenceProfile& profile
    )
        : profile_(profile) {}

    double posteriorForEvidence(
        double prior,
        bool amountEvidence,
        bool locationEvidence,
        bool deviceEvidence
    ) const {
        /*
         * Sequential Bayes update.
         *
         * This implementation treats evidence features as conditionally
         * independent given the fraud state. That assumption is useful for
         * a simple model, but it is not automatically true in real data.
         */

        double posterior = prior;

        posterior = updateOneFeature(
            posterior,
            amountEvidence,
            profile_.amountEvidenceGivenFraud,
            profile_.amountEvidenceGivenLegitimate
        );

        posterior = updateOneFeature(
            posterior,
            locationEvidence,
            profile_.locationEvidenceGivenFraud,
            profile_.locationEvidenceGivenLegitimate
        );

        posterior = updateOneFeature(
            posterior,
            deviceEvidence,
            profile_.deviceEvidenceGivenFraud,
            profile_.deviceEvidenceGivenLegitimate
        );

        return posterior;
    }

private:
    EvidenceProfile profile_;

    static double updateOneFeature(
        double prior,
        bool observed,
        double probabilityGivenFraud,
        double probabilityGivenLegitimate
    ) {
        validateProbability(
            probabilityGivenFraud,
            "P(E|fraud)"
        );

        validateProbability(
            probabilityGivenLegitimate,
            "P(E|legitimate)"
        );

        const double pEvidenceGivenFraud =
            observed
                ? probabilityGivenFraud
                : 1.0 - probabilityGivenFraud;

        const double pEvidenceGivenLegitimate =
            observed
                ? probabilityGivenLegitimate
                : 1.0 - probabilityGivenLegitimate;

        return bayesBinary(
            prior,
            pEvidenceGivenFraud,
            pEvidenceGivenLegitimate
        ).posterior;
    }
};


// ---------------------------------------------------------------------------
// Decision theory
// ---------------------------------------------------------------------------

struct Decision {
    std::string action;
    double expectedLoss;
};

Decision minimumExpectedLoss(
    double probabilityFraud,
    const std::map<std::string, std::pair<double, double>>& lossMatrix
) {
    validateProbability(
        probabilityFraud,
        "Fraud posterior"
    );

    const double probabilityLegitimate =
        1.0 - probabilityFraud;

    Decision best{
        "undefined",
        std::numeric_limits<double>::infinity()
    };

    for (const auto& [action, losses] : lossMatrix) {
        const double lossIfFraud = losses.first;
        const double lossIfLegitimate = losses.second;

        const double expectedLoss =
            probabilityFraud * lossIfFraud +
            probabilityLegitimate * lossIfLegitimate;

        if (expectedLoss < best.expectedLoss) {
            best = {
                action,
                expectedLoss
            };
        }
    }

    return best;
}


// ---------------------------------------------------------------------------
// Log-space calculations
// ---------------------------------------------------------------------------

double logSumExp(
    const std::vector<double>& logValues
) {
    if (logValues.empty()) {
        throw std::invalid_argument(
            "logSumExp requires at least one value."
        );
    }

    const double maximum =
        *std::max_element(
            logValues.begin(),
            logValues.end()
        );

    if (maximum == -std::numeric_limits<double>::infinity()) {
        return maximum;
    }

    double sum = 0.0;

    for (double value : logValues) {
        sum += std::exp(value - maximum);
    }

    return maximum + std::log(sum);
}

double logBayesianEvidence(
    const std::vector<double>& logPriors,
    const std::vector<double>& logLikelihoods
) {
    if (logPriors.size() != logLikelihoods.size()) {
        throw std::invalid_argument(
            "Log-prior and log-likelihood arrays must match."
        );
    }

    std::vector<double> terms;
    terms.reserve(logPriors.size());

    for (std::size_t i = 0; i < logPriors.size(); ++i) {
        terms.push_back(
            logPriors[i] + logLikelihoods[i]
        );
    }

    return logSumExp(terms);
}


// ---------------------------------------------------------------------------
// Transaction scoring
// ---------------------------------------------------------------------------

struct ScoredTransaction {
    Transaction transaction;
    double posteriorFraud;
    Decision decision;
};

ScoredTransaction scoreTransaction(
    const Transaction& transaction,
    const FraudBayesianModel& model,
    double prior,
    const std::map<std::string, std::pair<double, double>>& lossMatrix
) {
    const bool amountEvidence =
        transaction.amount >= 5000.0;

    const double posterior =
        model.posteriorForEvidence(
            prior,
            amountEvidence,
            transaction.unusualLocation,
            transaction.unusualDevice
        );

    const Decision decision =
        minimumExpectedLoss(
            posterior,
            lossMatrix
        );

    return {
        transaction,
        posterior,
        decision
    };
}


// ---------------------------------------------------------------------------
// Main case study
// ---------------------------------------------------------------------------

int main() {
    try {
        std::cout << std::fixed
                  << std::setprecision(4);

        std::cout << "============================================================\n";
        std::cout << "BAYESIAN FRAUD-DETECTION CASE STUDY\n";
        std::cout << "============================================================\n\n";

        // ---------------------------------------------------------------
        // Stage 1: Direct Bayes calculation
        // ---------------------------------------------------------------

        std::cout << "STAGE 1: BASIC BAYES CALCULATION\n";

        const BayesResult basic =
            bayesBinary(
                0.02,   // Prior fraud probability
                0.90,   // P(alert | fraud)
                0.05    // P(alert | legitimate)
            );

        std::cout << "Prior P(Fraud): "
                  << basic.prior << "\n";

        std::cout << "Likelihood P(Alert | Fraud): "
                  << basic.likelihood << "\n";

        std::cout << "P(Alert | Legitimate): "
                  << basic.competingLikelihood << "\n";

        std::cout << "Evidence P(Alert): "
                  << basic.evidence << "\n";

        std::cout << "Posterior P(Fraud | Alert): "
                  << basic.posterior << "\n\n";


        // ---------------------------------------------------------------
        // Stage 2: Multiple competing hypotheses
        // ---------------------------------------------------------------

        std::cout << "STAGE 2: MULTIPLE HYPOTHESES\n";

        const std::vector<std::string> serverNames = {
            "Server A",
            "Server B",
            "Server C"
        };

        const std::vector<double> serverPriors = {
            0.50,
            0.30,
            0.20
        };

        const std::vector<double> serverLikelihoods = {
            0.20,
            0.70,
            0.40
        };

        const auto serverPosteriors =
            posteriorDistribution(
                serverNames,
                serverPriors,
                serverLikelihoods
            );

        for (const auto& hypothesis : serverPosteriors) {
            std::cout
                << std::left
                << std::setw(12)
                << hypothesis.name
                << " prior="
                << std::setw(8)
                << hypothesis.prior
                << " likelihood="
                << std::setw(8)
                << hypothesis.likelihood
                << " posterior="
                << hypothesis.posterior
                << "\n";
        }

        std::cout << "\n";


        // ---------------------------------------------------------------
        // Stage 3: Configure the fraud model
        // ---------------------------------------------------------------

        std::cout << "STAGE 3: FRAUD MODEL\n";

        EvidenceProfile profile{
            0.02,  // Fraud prior

            0.70,  // High amount given fraud
            0.10,  // High amount given legitimate

            0.80,  // Unusual location given fraud
            0.05,  // Unusual location given legitimate

            0.75,  // Unusual device given fraud
            0.08   // Unusual device given legitimate
        };

        FraudBayesianModel model(profile);

        std::cout
            << "Base fraud prior: "
            << profile.fraudPrior
            << "\n\n";


        // ---------------------------------------------------------------
        // Stage 4: Decision costs
        // ---------------------------------------------------------------

        /*
         * The posterior is a probability. It does not itself determine
         * the correct operational action.
         *
         * Decision theory introduces consequences:
         *
         *   approve:
         *       high loss if fraud occurs
         *
         *   review:
         *       moderate operational cost
         *
         *   decline:
         *       high cost when a legitimate transaction is rejected
         *
         * Different institutions can use different costs.
         */

        const std::map<
            std::string,
            std::pair<double, double>
        > lossMatrix = {
            // action -> {loss if fraud, loss if legitimate}
            {"approve", {100.0, 0.0}},
            {"review",  {10.0, 2.0}},
            {"decline", {0.0, 20.0}}
        };


        // ---------------------------------------------------------------
        // Stage 5: Process realistic transactions
        // ---------------------------------------------------------------

        std::cout << "STAGE 5: TRANSACTION SCORING\n";

        const std::vector<Transaction> transactions = {
            {
                "TX-1001",
                TransactionType::Retail,
                250.0,
                false,
                false,
                false
            },
            {
                "TX-1002",
                TransactionType::Transfer,
                8000.0,
                true,
                true,
                true
            },
            {
                "TX-1003",
                TransactionType::Retail,
                7500.0,
                false,
                true,
                false
            },
            {
                "TX-1004",
                TransactionType::CashWithdrawal,
                6000.0,
                true,
                false,
                true
            }
        };

        std::vector<ScoredTransaction> scored;

        for (const auto& transaction : transactions) {
            scored.push_back(
                scoreTransaction(
                    transaction,
                    model,
                    profile.fraudPrior,
                    lossMatrix
                )
            );
        }

        for (const auto& item : scored) {
            std::cout
                << "\nTransaction: "
                << item.transaction.id
                << "\n";

            std::cout
                << "Type: "
                << transactionTypeToString(
                    item.transaction.type
                )
                << "\n";

            std::cout
                << "Amount: "
                << item.transaction.amount
                << "\n";

            std::cout
                << "International: "
                << (item.transaction.international
                        ? "yes"
                        : "no")
                << "\n";

            std::cout
                << "Unusual location: "
                << (item.transaction.unusualLocation
                        ? "yes"
                        : "no")
                << "\n";

            std::cout
                << "Unusual device: "
                << (item.transaction.unusualDevice
                        ? "yes"
                        : "no")
                << "\n";

            std::cout
                << "Posterior fraud probability: "
                << item.posteriorFraud
                << "\n";

            std::cout
                << "Decision: "
                << item.decision.action
                << "\n";

            std::cout
                << "Expected loss: "
                << item.decision.expectedLoss
                << "\n";
        }


        // ---------------------------------------------------------------
        // Stage 6: Odds and likelihood ratios
        // ---------------------------------------------------------------

        std::cout << "\n\nSTAGE 6: ODDS FORM\n";

        const double prior = 0.02;
        const double pEvidenceGivenFraud = 0.90;
        const double pEvidenceGivenLegitimate = 0.05;

        const double priorOdds =
            probabilityToOdds(prior);

        const double likelihoodRatio =
            pEvidenceGivenFraud /
            pEvidenceGivenLegitimate;

        const double posteriorOdds =
            priorOdds * likelihoodRatio;

        const double posterior =
            oddsToProbability(posteriorOdds);

        std::cout
            << "Prior odds: "
            << priorOdds
            << "\n";

        std::cout
            << "Likelihood ratio: "
            << likelihoodRatio
            << "\n";

        std::cout
            << "Posterior odds: "
            << posteriorOdds
            << "\n";

        std::cout
            << "Posterior probability: "
            << posterior
            << "\n";


        // ---------------------------------------------------------------
        // Stage 7: Numerical stability
        // ---------------------------------------------------------------

        std::cout << "\nSTAGE 7: NUMERICAL STABILITY\n";

        const double tinyProbability = 1e-300;

        const double directProduct =
            tinyProbability * tinyProbability;

        const double logProduct =
            std::log(tinyProbability) +
            std::log(tinyProbability);

        std::cout
            << "Direct tiny product: "
            << directProduct
            << "\n";

        std::cout
            << "Log-space product representation: "
            << logProduct
            << "\n";

        /*
         * In larger Bayesian models, a product of many probabilities can
         * underflow to zero. Logarithms convert multiplication into
         * addition:
         *
         * log(a*b*c) = log(a) + log(b) + log(c)
         *
         * The log-sum-exp operation handles normalization safely.
         */

        const std::vector<double> logLikelihoods = {
            -1000.0,
            -1001.0,
            -1005.0
        };

        std::cout
            << "Stable log-sum-exp: "
            << logSumExp(logLikelihoods)
            << "\n";


        // ---------------------------------------------------------------
        // Stage 8: Validation and failure conditions
        // ---------------------------------------------------------------

        std::cout << "\nSTAGE 8: EDGE CASES\n";

        try {
            conditionalProbability(
                0.2,
                0.0
            );
        } catch (const std::exception& error) {
            std::cout
                << "Caught expected error: "
                << error.what()
                << "\n";
        }

        try {
            bayesBinary(
                0.5,
                0.0,
                0.0
            );
        } catch (const std::exception& error) {
            std::cout
                << "Caught expected zero-evidence error: "
                << error.what()
                << "\n";
        }


        // ---------------------------------------------------------------
        // Stage 9: Complexity discussion
        // ---------------------------------------------------------------

        std::cout << "\nSTAGE 9: COMPLEXITY\n";

        /*
         * For k independent evidence features:
         *
         *   Bayesian update: O(k)
         *
         * For n mutually exclusive hypotheses and k features:
         *
         *   naive posterior calculation: O(n*k)
         *
         * The memory requirement is generally O(n*k) when all likelihood
         * parameters are stored explicitly.
         *
         * Real production systems can introduce substantially larger costs
         * for feature extraction, model retrieval, distributed storage,
         * monitoring, and online scoring.
         */

        std::cout
            << "Single transaction with k features: O(k)\n";

        std::cout
            << "n hypotheses and k features: O(n*k)\n";


        // ---------------------------------------------------------------
        // Final formula reference
        // ---------------------------------------------------------------

        std::cout << "\n============================================================\n";
        std::cout << "FORMULA REFERENCE\n";
        std::cout << "============================================================\n";

        std::cout
            << "Conditional probability:\n"
            << "P(A | B) = P(A and B) / P(B)\n\n";

        std::cout
            << "Multiplication rule:\n"
            << "P(A and B) = P(A) P(B | A)\n\n";

        std::cout
            << "Bayes' theorem:\n"
            << "P(H | E) = P(E | H) P(H) / P(E)\n\n";

        std::cout
            << "Odds form:\n"
            << "Posterior odds = Prior odds * Likelihood ratio\n\n";

        std::cout
            << "The fraud case study demonstrates how probability estimation "
               "and operational decisions are related but distinct.\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Fatal error: "
            << error.what()
            << "\n";

        return 1;
    }
}
