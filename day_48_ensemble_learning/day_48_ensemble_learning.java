import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/*
 * Enterprise ensemble governance for payment-risk decisions.
 *
 * Compile and run:
 *   javac EnsembleLearning.java
 *   java EnsembleLearning
 *
 * The program separates base-model predictions, ensemble aggregation,
 * threshold-based decisions, and operational decision policy.
 */

public class EnsembleLearning {

    enum RiskDecision {
        APPROVE,
        REVIEW,
        BLOCK
    }

    record Payment(
            String transactionId,
            double amount,
            int accountAgeDays,
            int recentTransactionCount,
            boolean fraud
    ) {
        Payment {
            if (transactionId == null || transactionId.isBlank()) {
                throw new IllegalArgumentException("Transaction ID is required.");
            }
            if (!Double.isFinite(amount) || amount < 0 ||
                    accountAgeDays < 0 || recentTransactionCount < 0) {
                throw new IllegalArgumentException("Invalid payment features.");
            }
        }

        double[] features() {
            return new double[] {
                Math.log1p(amount) / 10.0,
                Math.log1p(accountAgeDays) / 10.0,
                Math.log1p(recentTransactionCount) / 3.0
            };
        }
    }

    record Prediction(double probability, String modelName) {
        Prediction {
            if (!Double.isFinite(probability) ||
                    probability < 0 || probability > 1) {
                throw new IllegalArgumentException("Invalid probability.");
            }
            if (modelName == null || modelName.isBlank()) {
                throw new IllegalArgumentException("Model name is required.");
            }
        }
    }

    interface ProbabilisticModel {
        String name();

        void train(List<Payment> trainingData);

        Prediction predict(Payment payment);
    }

    static final class LogisticModel implements ProbabilisticModel {
        private final double learningRate;
        private final int epochs;
        private final double regularization;
        private double[] weights = new double[3];
        private double bias;

        LogisticModel(double learningRate, int epochs, double regularization) {
            if (learningRate <= 0 || epochs < 1 || regularization < 0) {
                throw new IllegalArgumentException("Invalid model configuration.");
            }

            this.learningRate = learningRate;
            this.epochs = epochs;
            this.regularization = regularization;
        }

        @Override
        public String name() {
            return "LogisticRegression";
        }

        private static double sigmoid(double value) {
            if (value >= 0) {
                return 1.0 / (1.0 + Math.exp(-Math.min(value, 700)));
            }

            double exponential = Math.exp(Math.max(value, -700));
            return exponential / (1.0 + exponential);
        }

        @Override
        public void train(List<Payment> rows) {
            if (rows == null || rows.isEmpty()) {
                throw new IllegalArgumentException("Training data is required.");
            }

            weights = new double[3];
            bias = 0.0;

            for (int epoch = 0; epoch < epochs; epoch++) {
                double[] gradients = new double[3];
                double biasGradient = 0.0;

                for (Payment payment : rows) {
                    double[] features = payment.features();
                    double score = bias;

                    for (int j = 0; j < weights.length; j++) {
                        score += weights[j] * features[j];
                    }

                    double error = sigmoid(score) - (payment.fraud() ? 1 : 0);

                    for (int j = 0; j < weights.length; j++) {
                        gradients[j] += error * features[j];
                    }

                    biasGradient += error;
                }

                for (int j = 0; j < weights.length; j++) {
                    double gradient = gradients[j] / rows.size()
                            + regularization * weights[j];
                    weights[j] -= learningRate * gradient;
                }

                bias -= learningRate * biasGradient / rows.size();
            }
        }

        @Override
        public Prediction predict(Payment payment) {
            Objects.requireNonNull(payment, "Payment cannot be null.");

            double[] features = payment.features();
            double score = bias;

            for (int j = 0; j < weights.length; j++) {
                score += weights[j] * features[j];
            }

            return new Prediction(sigmoid(score), name());
        }
    }

    static final class ThresholdModel implements ProbabilisticModel {
        private double amountThreshold;
        private double ageThreshold;
        private double velocityThreshold;
        private boolean trained;

        @Override
        public String name() {
            return "OperationalRiskRules";
        }

        @Override
        public void train(List<Payment> trainingData) {
            if (trainingData == null || trainingData.isEmpty()) {
                throw new IllegalArgumentException("Training data is required.");
            }

            List<Double> amounts = new ArrayList<>();
            List<Double> ages = new ArrayList<>();
            List<Double> velocities = new ArrayList<>();

            for (Payment payment : trainingData) {
                amounts.add(payment.amount());
                ages.add((double) payment.accountAgeDays());
                velocities.add((double) payment.recentTransactionCount());
            }

            Collections.sort(amounts);
            Collections.sort(ages);
            Collections.sort(velocities);

            // Median-based rules provide a simple heterogeneous base learner.
            amountThreshold = amounts.get(amounts.size() / 2);
            ageThreshold = ages.get(ages.size() / 2);
            velocityThreshold = velocities.get(velocities.size() / 2);
            trained = true;
        }

        @Override
        public Prediction predict(Payment payment) {
            if (!trained) {
                throw new IllegalStateException("Train the rules before prediction.");
            }

            double riskScore = 0.0;
            if (payment.amount() > amountThreshold) riskScore += 0.4;
            if (payment.accountAgeDays() < ageThreshold) riskScore += 0.25;
            if (payment.recentTransactionCount() > velocityThreshold) {
                riskScore += 0.45;
            }

            return new Prediction(Math.min(1.0, riskScore), name());
        }
    }

    static final class VotingPolicy {
        private final double[] weights;

        VotingPolicy(double... weights) {
            if (weights == null || weights.length == 0) {
                throw new IllegalArgumentException("Voting weights are required.");
            }

            double sum = 0.0;
            for (double weight : weights) {
                if (!Double.isFinite(weight) || weight < 0) {
                    throw new IllegalArgumentException("Invalid voting weight.");
                }
                sum += weight;
            }

            if (sum <= 0) {
                throw new IllegalArgumentException("Weights cannot all be zero.");
            }

            this.weights = weights.clone();
        }

        double averageProbability(List<Prediction> predictions) {
            if (predictions == null || predictions.size() != weights.length) {
                throw new IllegalArgumentException(
                        "Prediction count must match policy weights.");
            }

            double weightedSum = 0.0;
            double weightSum = 0.0;

            for (int i = 0; i < weights.length; i++) {
                weightedSum += weights[i] * predictions.get(i).probability();
                weightSum += weights[i];
            }

            return weightedSum / weightSum;
        }
    }

    static final class DecisionService {
        private final VotingPolicy votingPolicy;
        private final double blockThreshold;
        private final double reviewThreshold;

        DecisionService(
                VotingPolicy votingPolicy,
                double reviewThreshold,
                double blockThreshold
        ) {
            if (votingPolicy == null ||
                    !Double.isFinite(reviewThreshold) ||
                    !Double.isFinite(blockThreshold) ||
                    reviewThreshold < 0 || blockThreshold > 1 ||
                    reviewThreshold >= blockThreshold) {
                throw new IllegalArgumentException("Invalid decision policy.");
            }

            this.votingPolicy = votingPolicy;
            this.reviewThreshold = reviewThreshold;
            this.blockThreshold = blockThreshold;
        }

        RiskDecision decide(List<Prediction> predictions) {
            double risk = votingPolicy.averageProbability(predictions);

            if (risk >= blockThreshold) return RiskDecision.BLOCK;
            if (risk >= reviewThreshold) return RiskDecision.REVIEW;
            return RiskDecision.APPROVE;
        }
    }

    static List<Payment> generatePayments(int count, long seed) {
        if (count < 2) throw new IllegalArgumentException("Dataset is too small.");

        Random random = new Random(seed);
        List<Payment> payments = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            double amount = 1 + random.nextDouble() * 3000;
            int age = random.nextInt(2000);
            int velocity = random.nextInt(16);

            double score = 0.0012 * amount - 0.001 * age
                    + 0.32 * velocity - 1.2 + random.nextGaussian() * 0.9;

            payments.add(new Payment(
                    "PAY-" + (100000 + i),
                    amount,
                    age,
                    velocity,
                    score > 1.0
            ));
        }

        Collections.shuffle(payments, new Random(seed + 1));
        return payments;
    }

    static double accuracy(List<Payment> rows, List<Integer> predictions) {
        if (rows.isEmpty() || rows.size() != predictions.size()) {
            throw new IllegalArgumentException("Invalid evaluation data.");
        }

        int correct = 0;

        for (int i = 0; i < rows.size(); i++) {
            int actual = rows.get(i).fraud() ? 1 : 0;
            if (actual == predictions.get(i)) correct++;
        }

        return (double) correct / rows.size();
    }

    public static void main(String[] args) {
        List<Payment> payments = generatePayments(1000, 91);

        int trainingSize = (int) (payments.size() * 0.70);
        int validationSize = (int) (payments.size() * 0.15);

        List<Payment> training = new ArrayList<>(payments.subList(0, trainingSize));
        List<Payment> validation = new ArrayList<>(
                payments.subList(trainingSize, trainingSize + validationSize));
        List<Payment> testing = new ArrayList<>(
                payments.subList(trainingSize + validationSize, payments.size()));

        ProbabilisticModel logistic = new LogisticModel(0.12, 500, 0.01);
        ProbabilisticModel rules = new ThresholdModel();

        logistic.train(training);
        rules.train(training);

        // Validation data is reserved for threshold selection, not base-model fitting.
        List<Double> validationScores = new ArrayList<>();

        for (Payment payment : validation) {
            validationScores.add(
                    0.65 * logistic.predict(payment).probability()
                            + 0.35 * rules.predict(payment).probability()
            );
        }

        double selectedReviewThreshold = 0.35;
        double selectedBlockThreshold = 0.70;

        DecisionService service = new DecisionService(
                new VotingPolicy(0.65, 0.35),
                selectedReviewThreshold,
                selectedBlockThreshold
        );

        List<Integer> logisticLabels = new ArrayList<>();
        List<Integer> ensembleLabels = new ArrayList<>();

        int reviewed = 0;
        int blocked = 0;

        for (Payment payment : testing) {
            Prediction logisticPrediction = logistic.predict(payment);
            Prediction rulePrediction = rules.predict(payment);

            logisticLabels.add(
                    logisticPrediction.probability() >= 0.5 ? 1 : 0
            );

            RiskDecision decision = service.decide(
                    List.of(logisticPrediction, rulePrediction)
            );

            ensembleLabels.add(decision == RiskDecision.BLOCK ? 1 : 0);

            if (decision == RiskDecision.REVIEW) reviewed++;
            if (decision == RiskDecision.BLOCK) blocked++;
        }

        System.out.println("Enterprise payment-risk ensemble");
        System.out.printf("Training: %d, validation: %d, test: %d%n",
                training.size(), validation.size(), testing.size());

        System.out.printf("Logistic classification accuracy: %.3f%n",
                accuracy(testing, logisticLabels));

        System.out.printf("Conservative block accuracy: %.3f%n",
                accuracy(testing, ensembleLabels));

        System.out.println("Transactions routed to review: " + reviewed);
        System.out.println("Transactions blocked: " + blocked);
        System.out.println("Validation scores evaluated: " + validationScores.size());
        System.out.println("Review and block thresholds: "
                + selectedReviewThreshold + " / " + selectedBlockThreshold);
        System.out.println(
                "Operational use requires calibration, fairness testing, "
                        + "monitoring, and cost-sensitive threshold validation."
        );
    }
}
