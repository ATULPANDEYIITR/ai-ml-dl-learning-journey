# Naive Bayes: Probabilistic Classification in Five Implementations

## Technical scope

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem. It estimates the probability of a class given observed features, using a simplifying assumption that the features are conditionally independent once the class is known.

The assumption makes parameter estimation efficient and prediction computationally inexpensive. It does not require features to be independent across the entire dataset. It assumes independence within each class when constructing the class-conditional likelihood.

This repository examines three related but distinct models:

- **Gaussian Naive Bayes** represents continuous features using class-specific normal distributions.
- **Multinomial Naive Bayes** represents discrete counts, especially word frequencies in documents.
- **Bernoulli Naive Bayes** represents binary feature presence or absence.

The implementations use equipment telemetry and support-message routing to demonstrate how the choice of probability model affects classification.

## Bayes' theorem and the classification rule

For a class \(C\) and observed feature vector \(X\), Bayes' theorem states

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

The terms have distinct meanings:

- \(P(C)\) is the prior probability of the class, estimated from the class distribution in the training data.
- \(P(X\mid C)\) is the likelihood of observing the features under that class.
- \(P(X)\) is the evidence, or marginal probability of the observed features.
- \(P(C\mid X)\) is the posterior probability of the class after observing the features.

For classification, the evidence is identical across candidate classes. Therefore, the predicted class can be selected by maximizing the unnormalized posterior:

\[
\hat C=\arg\max_C P(C)P(X\mid C)
\]

Under the Naive Bayes conditional-independence assumption,

\[
P(X\mid C)=\prod_{j=1}^{d}P(X_j\mid C)
\]

The resulting classification rule becomes

\[
\hat C=\arg\max_C P(C)\prod_{j=1}^{d}P(X_j\mid C)
\]

The product can become extremely small when many features contribute to the likelihood. Implementations therefore use logarithms:

\[
\hat C=\arg\max_C\left[\log P(C)+
\sum_{j=1}^{d}\log P(X_j\mid C)\right]
\]

Logarithms preserve the ordering of positive probabilities while turning multiplication into addition. They improve numerical stability but do not correct an inaccurate probability model.

## Conditional independence

Consider a support message containing the words `server`, `timeout`, and `latency`. A Naive Bayes classifier estimates how frequently these words occur under technical, billing, and account-related messages. It combines their class-conditional contributions as though the words were independent given the message category.

In real language, the words `server` and `timeout` are related. A Gaussian model can also encounter correlated measurements such as temperature and vibration. Conditional independence is therefore an approximation rather than a universal property of the data.

The approximation has an important computational advantage. A general joint distribution over many features requires substantially more data and parameters than a collection of per-feature conditional distributions. Naive Bayes can estimate its parameters from relatively small training datasets, provided that each class has enough representative examples.

A classifier can predict the correct class even when its independence assumption is imperfect. Probability estimates can nevertheless become poorly calibrated, particularly when multiple correlated features contribute similar evidence.

## Gaussian Naive Bayes

### Probability model

Gaussian Naive Bayes is appropriate when features are continuous measurements. For each class \(C\) and feature \(X_j\), it estimates a mean \(\mu_{C,j}\) and variance \(\sigma^2_{C,j}\).

The class-conditional density is

\[
P(X_j=x\mid C)=
\frac{1}{\sqrt{2\pi\sigma^2_{C,j}}}
\exp\left(
-\frac{(x-\mu_{C,j})^2}{2\sigma^2_{C,j}}
\right)
\]

The density is evaluated at the observed measurement. Unlike a discrete probability, a probability density value can exceed one, because probability is obtained by integrating the density over an interval.

### Python implementation

The Python program implements `GaussianNaiveBayes` with explicit training, prediction, posterior normalization, validation, and model persistence.

During training, the classifier groups observations by label and calculates the mean and population variance for each feature within each class. The prior is based on the proportion of training observations belonging to that class.

A variance floor prevents division by zero when a feature is constant within a class. The floor is scaled by the global feature variance, with a minimum scale of one. This is a numerical safeguard, not a substitute for investigating constant or nearly constant features.

The example uses vibration RMS and operating temperature to distinguish normal equipment operation from a fault condition. The model can be applied to a new telemetry reading without requiring an external machine-learning package.

The JSON persistence methods save the learned parameters and restore them for inference. The loader checks the basic model structure and verifies that variances are positive. In production, a serialized model should also have a version identifier, feature definitions, training-data provenance, and integrity verification.

### C++ case study

The C++ program models a predictive-maintenance service. `GaussianNaiveBayes` estimates class-specific Gaussian parameters, returns log scores, and converts them into normalized probabilities. `MaintenanceEngine` uses the resulting fault probability to select an operational action.

The engine separates statistical classification from maintenance policy. A model predicts a fault probability, while the service decides whether to continue routine monitoring, increase telemetry sampling, or raise an inspection alert.

This separation allows the intervention threshold to change without retraining the classifier. The implementation validates telemetry values, rejects non-finite measurements, checks feature dimensions, and tests a constant feature.

The variance floor prevents numerical failure, but the Gaussian assumption may be unsuitable for multimodal or strongly skewed measurements. Feature scaling is not mathematically mandatory for Gaussian Naive Bayes when every feature has its own fitted distribution. Sensor units and measurement quality still affect numerical conditioning and operational interpretation.

## Multinomial Naive Bayes

### Probability model

Multinomial Naive Bayes is designed for non-negative feature counts. Text classification commonly represents a document using the number of occurrences of each vocabulary term.

Let \(N_{C,t}\) be the number of occurrences of token \(t\) in training documents belonging to class \(C\). Let \(N_C\) be the total number of token occurrences in that class and let \(V\) be the training vocabulary.

With additive smoothing parameter \(\alpha>0\),

\[
P(t\mid C)=\frac{N_{C,t}+\alpha}{N_C+\alpha|V|}
\]

For a document represented by token counts \(n_t\), the log score is

\[
\log P(C)+\sum_{t\in V}n_t\log P(t\mid C)
\]

Repeated tokens contribute repeatedly to the score. A message containing `payment payment invoice` can therefore produce a different prediction from `payment invoice`.

### Python implementation

`MultinomialNaiveBayes` uses a `Counter` to accumulate token frequencies within each class. Its vocabulary is built exclusively from the training documents.

Additive smoothing assigns nonzero likelihoods to vocabulary terms that never appeared in a particular class. Without smoothing, a single zero-frequency token would give the entire class likelihood a value of zero.

The tokenizer lowercases text and separates words using a standard-library character transformation. It is deliberately simple: it does not perform stemming, lemmatization, language detection, or domain-specific tokenization.

At inference time, tokens outside the training vocabulary are ignored. This policy is valid but loses information about unfamiliar terminology. An explicit unknown-token feature can be useful when its construction is consistent across training and inference.

### JavaScript implementation

The JavaScript program focuses on asynchronous classification and event-driven routing. `MultinomialTextClassifier` stores a model snapshot and computes scores from token counts.

The `WorkflowEventBus` allows classification results to be consumed by other components without embedding queue-management behavior inside the classifier. `SupportRoutingService` publishes a `message:classified` event containing the predicted category, confidence, and manual-review decision.

The asynchronous service method illustrates integration with message queues or HTTP handlers. The underlying model calculation is synchronous; the asynchronous interface makes it possible to coordinate with asynchronous application workflows without claiming that the statistical computation itself is parallel.

The implementation validates inputs, rejects empty training vocabularies, uses log scores, and verifies that posterior probabilities sum to approximately one. Event listeners receive the completed result rather than partially computed model state.

## Bernoulli Naive Bayes

### Probability model

Bernoulli Naive Bayes treats each vocabulary term as a binary variable. A term is either present in a document or absent from it, regardless of how many times it occurs.

For a class \(C\), let \(N_C\) be the number of training documents in that class and \(D_{C,t}\) the number containing token \(t\). With additive smoothing,

\[
P(X_t=1\mid C)=\frac{D_{C,t}+\alpha}{N_C+2\alpha}
\]

The denominator contains \(2\alpha\) because the feature has two outcomes: present and absent.

For a binary document vector \(x\), the log likelihood is

\[
\log P(X\mid C)=
\sum_{t\in V}\left[
x_t\log p_{C,t}+
(1-x_t)\log(1-p_{C,t})
\right]
\]

Both feature presence and feature absence affect the score. A word that is missing from a document is not automatically ignored.

### Python and Java implementations

`BernoulliNaiveBayes` converts each document to a set of tokens during training. The set ensures that repeated occurrences count as one presence observation. The model stores both the logarithm of the probability of presence and the logarithm of the probability of absence.

The Java program implements both Multinomial and Bernoulli feature models through the `FeatureModel` enum. This keeps the choice of statistical model explicit and prevents the distinction from being hidden in arbitrary conditional logic at the application boundary.

The Bernoulli training procedure calculates document-presence counts instead of total token counts. Its scoring procedure iterates across the complete training vocabulary, adding either the presence or absence contribution.

This model can be useful when whether a term appears matters more than how frequently it occurs. It can behave differently from Multinomial Naive Bayes for long documents, repeated terms, and sparse binary feature representations.

## Additive smoothing and unseen evidence

Smoothing prevents a feature that was not observed in one class from making that class's likelihood exactly zero.

For Multinomial Naive Bayes, the probability denominator depends on the total number of tokens in the class and the vocabulary size. For Bernoulli Naive Bayes, each feature has two possible outcomes, so its denominator depends on the number of class documents and twice the smoothing parameter.

The smoothing parameter controls how strongly estimates are pulled away from zero and one. A very small value can leave estimates unstable when training data is sparse. A very large value can weaken useful differences between classes.

Smoothing cannot compensate for missing classes, severe label errors, an unrepresentative vocabulary, or an inappropriate likelihood distribution. It also does not guarantee calibrated posterior probabilities.

## Enterprise routing and model policy

The Java implementation separates the statistical model, classification result, routing policy, and application service.

`TrainingDocument` represents labeled training data. `Classification` carries the predicted category, confidence, and normalized class probabilities. `RoutingPolicy` maps categories to operational queues and checks a minimum confidence threshold. `ClassificationService` combines these components.

When the prediction confidence is below the configured threshold, the service routes the message to manual triage. It also sends messages to manual triage when no queue is configured for a predicted category.

A confidence threshold is a policy choice, not an intrinsic property of Naive Bayes. A production system should select the threshold based on the costs of false routing, missed incidents, delayed handling, and unnecessary manual review.

## PostgreSQL relational model

The SQL implementation stores the training corpus and derived statistics in normalized tables.

| Table | Purpose |
|---|---|
| `nb_classes` | Defines the permitted classification categories. |
| `nb_documents` | Stores document text, its optional label, and its training, test, or inference split. |
| `nb_vocabulary` | Stores normalized tokens and document-frequency counts. |
| `nb_document_terms` | Records term frequencies for each document and token. |
| `nb_class_statistics` | Stores the training document count and total token count for each class. |
| `nb_class_term_statistics` | Stores per-class token counts and document-presence counts. |
| `nb_model_configuration` | Defines the smoothing parameter, feature model, confidence threshold, and active state. |
| `nb_predictions` | Records predicted categories, confidence, and manual-review status. |

Primary keys and unique constraints prevent duplicate class names, vocabulary tokens, external document keys, and predictions for the same document-model pair. Foreign keys ensure that terms, labels, and predictions refer to existing records. Check constraints restrict feature counts and confidence values to valid ranges.

The schema stores both total token counts and document-presence counts because the two text models require different sufficient statistics. Multinomial probabilities use token frequencies; Bernoulli probabilities use the number of documents containing each token.

### Training leakage prevention

The training-statistics queries explicitly restrict labeled training calculations to documents whose `split_name` is `train`. Test and inference records must not influence the learned priors or likelihoods.

The SQL tokenization stage processes the entire demonstration corpus to create a shared vocabulary, while model parameters are derived from training documents only. A strict production evaluation pipeline should construct the vocabulary from training data alone and transform test and inference documents using that fixed vocabulary. The demonstration vocabulary procedure is therefore suitable for explaining the relational scoring mechanics but should be tightened before using the script as a leakage-free benchmark.

### SQL scoring

The Multinomial prediction view adds each observed term's frequency multiplied by its smoothed log likelihood to the class log prior.

The Bernoulli prediction view constructs a presence-or-absence row for each inference document and vocabulary term. It adds the log probability of presence or absence for every vocabulary feature.

Both views normalize log scores by subtracting the maximum score before exponentiation. This is the log-sum-exp technique: it reduces underflow while producing probabilities that sum to one, subject to ordinary floating-point precision.

The prediction table records the model configuration used and derives its manual-review flag from the configured confidence threshold. The table constraints enforce valid confidence values, but they do not guarantee that stored predictions remain current after a model is retrained. A production system should version model artifacts and predictions, then update them atomically.

## Evaluation and numerical reliability

The Python script includes a stratified holdout split, accuracy, and a confusion matrix. Stratification preserves representation of each class in both partitions, subject to the requirement that each class has enough examples.

Accuracy is the proportion of predictions that match the true labels. It can be misleading when class frequencies are highly unequal. A classifier that predicts only the majority class may achieve high accuracy while failing to identify a rare but important condition.

A confusion matrix reveals which categories are confused with each other. For support routing, sending a technical incident to billing may have a different operational cost from sending a billing request to account support.

A more complete evaluation should include class-level precision, recall, F1 score, and probability calibration. Training, model selection, threshold selection, and final evaluation should use appropriately separated data.

## Edge cases and limitations

**Constant numerical features:** Gaussian Naive Bayes requires positive variance to evaluate its density. The Python and C++ implementations introduce a small variance floor to avoid division by zero. The choice of floor should be examined against measurement precision and feature scale.

**Unseen text vocabulary:** Multinomial and Bernoulli models cannot learn class-specific statistics for words absent from the training vocabulary. The implementations ignore such words at prediction time. A carefully designed unknown-token policy can improve robustness.

**Empty documents:** An empty document supplies no text evidence. The text models effectively fall back to their class priors, provided the model was trained successfully. Empty training corpora and empty training vocabularies are rejected.

**Correlated features:** Multiple strongly related features may cause the classifier to accumulate redundant evidence. This can preserve useful rankings while producing overconfident probabilities.

**Imbalanced classes:** Priors reflect the class proportions in the training dataset. If the deployment distribution differs, the resulting predictions can be systematically biased toward the training distribution.

**Distribution shift:** Equipment sensors may drift, support terminology may change, and the frequency of operational incidents may vary. Model accuracy can decline even when the software remains unchanged.

**Threshold selection:** The manual-review threshold should reflect the relative cost of false positives and false negatives. A default threshold is not a universal operational standard.

## Performance characteristics

Let \(N\) be the number of training documents, \(T\) the total number of token occurrences, \(V\) the vocabulary size, and \(K\) the number of classes.

For Multinomial Naive Bayes, token counting is approximately linear in the total number of training tokens. Scoring a document is proportional to its in-vocabulary token occurrences when the implementation processes only observed terms.

For Bernoulli Naive Bayes, training and scoring often require iteration over the vocabulary for each class. The dense form of the algorithm is approximately proportional to \(K|V|\) per prediction, even when the input document contains few words.

Gaussian Naive Bayes requires storage proportional to the number of classes times the number of numerical features. Training is approximately linear in the number of observations times the number of features, and prediction has the same dependence on the feature count and number of classes.

The PostgreSQL Bernoulli view expands inference documents against the vocabulary. This is a direct relational representation of the binary feature model, but its intermediate result can become large. For large corpora, precomputed sparse statistics, batch scoring, or an application-side inference service may be more efficient.

## Security and operational considerations

Classification output should be treated as a decision aid rather than an unconditional authority. Low-confidence cases, unusual inputs, and high-impact decisions should have an appropriate human-review path.

Training data can contain personal information, confidential incident descriptions, or maliciously crafted content. Data collection and retention should follow applicable access controls and privacy policies. Logs should avoid unnecessarily recording raw sensitive messages.

Model files and database prediction records need integrity and access controls. The Python JSON loader checks the shape of model parameters but does not authenticate the file. A production deployment should verify artifact provenance and compatibility with the feature-extraction version.

Naive Bayes is attractive because its learned parameters are compact, its training process is transparent, and its prediction mechanism is straightforward to inspect. Its reliability still depends on representative data, appropriate feature distributions, sound evaluation, and a clear separation between statistical predictions and operational policy.
