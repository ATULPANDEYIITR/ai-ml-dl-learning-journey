import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.ToDoubleFunction;

/*
 * Enterprise-oriented Naive Bayes model registry and decision service.
 *
 * Scenario:
 * A support organization routes incoming messages to technical, billing,
 * or account-management queues. A review policy can redirect low-confidence
 * predictions to manual triage.
 *
 * Compile and run:
 *   javac NaiveBayesEnterprise.java
 *   java NaiveBayesEnterprise
 *
 * Requires Java 17 or later. No external dependencies.
 */
public class NaiveBayesEnterprise {

    enum FeatureModel {
        MULTINOMIAL,
        BERNOULLI
    }

    record TrainingDocument(String text, String category) {
        TrainingDocument {
            if (text == null || category == null || category.isBlank()) {
                throw new IllegalArgumentException(
                    "Text and category must be provided."
                );
            }
        }
    }

    record Classification(
        String category,
        double confidence,
        Map<String, Double> probabilities
    ) {
        Classification {
            Objects.requireNonNull(category);
            if (!Double.isFinite(confidence) ||
                confidence < 0.0 ||
                confidence > 1.0) {
                throw new IllegalArgumentException("Invalid confidence.");
            }
            probabilities = Collections.unmodifiableMap(
                new LinkedHashMap<>(probabilities)
            );
        }
    }

    record RoutingDecision(
        String queue,
        boolean requiresManualReview,
        Classification classification,
        String reason
    ) {}

    static List<String> tokenize(String text) {
        String normalized = text.toLowerCase(java.util.Locale.ROOT)
            .replaceAll("[^\\p{L}\\p{N}]+", " ")
            .trim();

        if (normalized.isEmpty()) {
            return List.of();
        }

        return List.of(normalized.split("\\s+"));
    }

    static Map<String, Double> normalizeLogScores(
        Map<String, Double> scores
    ) {
        if (scores.isEmpty()) {
            throw new IllegalArgumentException("No classes are available.");
        }

        double maximum = scores.values().stream()
            .mapToDouble(Double::doubleValue)
            .max()
            .orElseThrow();

        if (!Double.isFinite(maximum)) {
            throw new IllegalStateException("Invalid log scores.");
        }

        Map<String, Double> exponentials = new LinkedHashMap<>();
        double denominator = 0.0;

        for (var entry : scores.entrySet()) {
            double value = Math.exp(entry.getValue() - maximum);
            exponentials.put(entry.getKey(), value);
            denominator += value;
        }

        Map<String, Double> probabilities = new LinkedHashMap<>();
        for (var entry : exponentials.entrySet()) {
            probabilities.put(entry.getKey(), entry.getValue() / denominator);
        }

        return probabilities;
    }

    static final class TextNaiveBayes {
        private final FeatureModel featureModel;
        private final double alpha;

        private List<String> classes = List.of();
        private Set<String> vocabulary = Set.of();
        private Map<String, Double> logPriors = Map.of();
        private Map<String, Map<String, Double>> logPresent = Map.of();
        private Map<String, Map<String, Double>> logAbsent = Map.of();
        private boolean fitted;

        TextNaiveBayes(FeatureModel featureModel, double alpha) {
            this.featureModel = Objects.requireNonNull(featureModel);

            if (!Double.isFinite(alpha) || alpha <= 0.0) {
                throw new IllegalArgumentException(
                    "Smoothing alpha must be finite and positive."
                );
            }
            this.alpha = alpha;
        }

        void fit(List<TrainingDocument> trainingData) {
            if (trainingData == null || trainingData.isEmpty()) {
                throw new IllegalArgumentException(
                    "Training data cannot be empty."
                );
            }

            Map<String, List<List<String>>> grouped = new TreeMap<>();
            Set<String> newVocabulary = new TreeSet<>();

            for (TrainingDocument document : trainingData) {
                List<String> tokens = tokenize(document.text());
                grouped.computeIfAbsent(
                    document.category(),
                    ignored -> new ArrayList<>()
                ).add(tokens);
                newVocabulary.addAll(tokens);
            }

            if (newVocabulary.isEmpty()) {
                throw new IllegalArgumentException(
                    "Training data contains no tokens."
                );
            }

            Map<String, Double> newPriors = new TreeMap<>();
            Map<String, Map<String, Double>> newPresent = new TreeMap<>();
            Map<String, Map<String, Double>> newAbsent = new TreeMap<>();

            for (var entry : grouped.entrySet()) {
                String category = entry.getKey();
                List<List<String>> documents = entry.getValue();

                newPriors.put(
                    category,
                    Math.log((double) documents.size() / trainingData.size())
                );

                Map<String, Double> presentLogs = new TreeMap<>();
                Map<String, Double> absentLogs = new TreeMap<>();

                if (featureModel == FeatureModel.MULTINOMIAL) {
                    Map<String, Integer> counts = new HashMap<>();
                    int totalTokens = 0;

                    for (List<String> tokens : documents) {
                        for (String token : tokens) {
                            counts.merge(token, 1, Integer::sum);
                            totalTokens++;
                        }
                    }

                    double denominator =
                        totalTokens + alpha * newVocabulary.size();

                    for (String token : newVocabulary) {
                        presentLogs.put(
                            token,
                            Math.log(
                                (counts.getOrDefault(token, 0) + alpha)
                                / denominator
                            )
                        );
                    }
                } else {
                    for (String token : newVocabulary) {
                        int presentCount = 0;

                        for (List<String> tokens : documents) {
                            // Set membership converts repeated occurrences
                            // into one binary feature per document.
                            if (new TreeSet<>(tokens).contains(token)) {
                                presentCount++;
                            }
                        }

                        double probability =
                            (presentCount + alpha)
                            / (documents.size() + 2.0 * alpha);

                        presentLogs.put(token, Math.log(probability));
                        absentLogs.put(token, Math.log1p(-probability));
                    }
                }

                newPresent.put(category, Collections.unmodifiableMap(presentLogs));
                newAbsent.put(category, Collections.unmodifiableMap(absentLogs));
            }

            // Construct new state locally; publish immutable snapshots only
            // after the complete training process succeeds.
            classes = List.copyOf(grouped.keySet());
            vocabulary = Collections.unmodifiableSet(new TreeSet<>(newVocabulary));
            logPriors = Collections.unmodifiableMap(newPriors);
            logPresent = Collections.unmodifiableMap(newPresent);
            logAbsent = Collections.unmodifiableMap(newAbsent);
            fitted = true;
        }

        Map<String, Double> logScores(String text) {
            if (!fitted) {
                throw new IllegalStateException("Model has not been fitted.");
            }
            if (text == null) {
                throw new IllegalArgumentException("Text cannot be null.");
            }

            List<String> tokens = tokenize(text);
            Map<String, Integer> counts = new HashMap<>();
            Set<String> present = new TreeSet<>(tokens);
            for (String token : tokens) {
                counts.merge(token, 1, Integer::sum);
            }

            Map<String, Double> scores = new LinkedHashMap<>();

            for (String category : classes) {
                double score = logPriors.get(category);

                for (String token : vocabulary) {
                    if (featureModel == FeatureModel.MULTINOMIAL) {
                        score += counts.getOrDefault(token, 0)
                            * logPresent.get(category).get(token);
                    } else if (present.contains(token)) {
                        score += logPresent.get(category).get(token);
                    } else {
                        score += logAbsent.get(category).get(token);
                    }
                }

                scores.put(category, score);
            }

            return scores;
        }

        Classification classify(String text) {
            Map<String, Double> probabilities =
                normalizeLogScores(logScores(text));

            String bestCategory = probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow()
                .getKey();

            return new Classification(
                bestCategory,
                probabilities.get(bestCategory),
                probabilities
            );
        }

        FeatureModel featureModel() {
            return featureModel;
        }
    }

    static final class RoutingPolicy {
        private final double minimumConfidence;
        private final Map<String, String> categoryQueues;

        RoutingPolicy(
            double minimumConfidence,
            Map<String, String> categoryQueues
        ) {
            if (!Double.isFinite(minimumConfidence) ||
                minimumConfidence < 0.0 ||
                minimumConfidence > 1.0) {
                throw new IllegalArgumentException(
                    "Minimum confidence must lie in [0, 1]."
                );
            }

            if (categoryQueues == null || categoryQueues.isEmpty()) {
                throw new IllegalArgumentException(
                    "At least one category-to-queue mapping is required."
                );
            }

            this.minimumConfidence = minimumConfidence;
            this.categoryQueues = Map.copyOf(categoryQueues);
        }

        RoutingDecision apply(Classification classification) {
            if (classification.confidence() < minimumConfidence) {
                return new RoutingDecision(
                    "manual-triage",
                    true,
                    classification,
                    "Prediction confidence is below the routing threshold."
                );
            }

            String queue = categoryQueues.get(classification.category());
            if (queue == null) {
                return new RoutingDecision(
                    "manual-triage",
                    true,
                    classification,
                    "No queue is configured for the predicted category."
                );
            }

            return new RoutingDecision(
                queue,
                false,
                classification,
                "Predicted category meets the routing policy."
            );
        }
    }

    static final class ClassificationService {
        private final TextNaiveBayes model;
        private final RoutingPolicy policy;

        ClassificationService(
            TextNaiveBayes model,
            RoutingPolicy policy
        ) {
            this.model = Objects.requireNonNull(model);
            this.policy = Objects.requireNonNull(policy);
        }

        RoutingDecision route(String message) {
            Classification classification = model.classify(message);
            return policy.apply(classification);
        }
    }

    private static List<TrainingDocument> trainingData() {
        return List.of(
            new TrainingDocument(
                "server latency timeout connection failure", "technical"
            ),
            new TrainingDocument(
                "database server connection error", "technical"
            ),
            new TrainingDocument(
                "network latency server", "technical"
            ),
            new TrainingDocument(
                "deployment failed timeout server", "technical"
            ),
            new TrainingDocument(
                "invoice payment receipt billing", "billing"
            ),
            new TrainingDocument(
                "overdue invoice account payment", "billing"
            ),
            new TrainingDocument(
                "billing receipt transaction", "billing"
            ),
            new TrainingDocument(
                "payment invoice account overdue", "billing"
            ),
            new TrainingDocument(
                "password reset login account", "account"
            ),
            new TrainingDocument(
                "authentication password security", "account"
            ),
            new TrainingDocument(
                "login access password", "account"
            ),
            new TrainingDocument(
                "account login authentication access", "account"
            )
        );
    }

    public static void main(String[] args) {
        TextNaiveBayes multinomial = new TextNaiveBayes(
            FeatureModel.MULTINOMIAL, 1.0
        );
        multinomial.fit(trainingData());

        TextNaiveBayes bernoulli = new TextNaiveBayes(
            FeatureModel.BERNOULLI, 1.0
        );
        bernoulli.fit(trainingData());

        Map<String, String> queues = Map.of(
            "technical", "engineering-support",
            "billing", "accounts-receivable",
            "account", "identity-support"
        );

        ClassificationService service = new ClassificationService(
            multinomial,
            new RoutingPolicy(0.55, queues)
        );

        List<String> incomingMessages = List.of(
            "server latency timeout after deployment",
            "invoice payment overdue",
            "unfamiliar phrase with unknown terminology"
        );

        for (String message : incomingMessages) {
            RoutingDecision decision = service.route(message);

            System.out.println("\nMessage: " + message);
            System.out.println("Queue: " + decision.queue());
            System.out.printf(
                "Confidence: %.4f%n",
                decision.classification().confidence()
            );
            System.out.println(
                "Manual review required: " + decision.requiresManualReview()
            );
            System.out.println("Reason: " + decision.reason());
            System.out.println(
                "Probabilities: " + decision.classification().probabilities()
            );
        }

        Classification repeatedMultinomial =
            multinomial.classify("payment payment invoice");
        Classification repeatedBernoulli =
            bernoulli.classify("payment payment invoice");

        System.out.println("\nRepeated-token comparison");
        System.out.println(
            "Multinomial: " + repeatedMultinomial.probabilities()
        );
        System.out.println(
            "Bernoulli: " + repeatedBernoulli.probabilities()
        );

        if (Math.abs(
            multinomial.classify("server timeout").probabilities()
                .values().stream().mapToDouble(Double::doubleValue).sum()
            - 1.0
        ) > 1e-10) {
            throw new IllegalStateException("Probability normalization failed.");
        }

        try {
            new TextNaiveBayes(FeatureModel.MULTINOMIAL, 0.0);
            throw new IllegalStateException("Invalid smoothing was accepted.");
        } catch (IllegalArgumentException expected) {
            System.out.println("\nInvalid smoothing correctly rejected.");
        }

        System.out.println(
            "Model: " + multinomial.featureModel()
            + ". Routing decisions remain subject to operational review."
        );
    }
}
