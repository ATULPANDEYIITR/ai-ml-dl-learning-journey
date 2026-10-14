/*
 * PostgreSQL Naive Bayes classification laboratory.
 *
 * This schema stores a text-classification training corpus, vocabulary,
 * class priors, and class-conditional token statistics. It demonstrates
 * both Multinomial and Bernoulli Naive Bayes scoring in SQL.
 *
 * Execute in PostgreSQL 14 or later using psql:
 *     psql -v ON_ERROR_STOP=1 -f naive_bayes.sql
 *
 * The scoring queries use natural logarithms. PostgreSQL's LN() function
 * is used for log probabilities, and EXP() is used for normalization.
 */

BEGIN;

DROP VIEW IF EXISTS nb_model_diagnostics CASCADE;
DROP VIEW IF EXISTS nb_multinomial_predictions CASCADE;
DROP VIEW IF EXISTS nb_bernoulli_predictions CASCADE;
DROP TABLE IF EXISTS nb_predictions CASCADE;
DROP TABLE IF EXISTS nb_document_terms CASCADE;
DROP TABLE IF EXISTS nb_class_term_statistics CASCADE;
DROP TABLE IF EXISTS nb_vocabulary CASCADE;
DROP TABLE IF EXISTS nb_documents CASCADE;
DROP TABLE IF EXISTS nb_model_configuration CASCADE;
DROP TABLE IF EXISTS nb_class_statistics CASCADE;
DROP TABLE IF EXISTS nb_classes CASCADE;

CREATE TABLE nb_classes (
    class_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    class_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    CHECK (length(trim(class_name)) > 0)
);

CREATE TABLE nb_documents (
    document_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    external_key TEXT NOT NULL UNIQUE,
    document_text TEXT NOT NULL,
    class_id BIGINT REFERENCES nb_classes(class_id),
    split_name TEXT NOT NULL CHECK (split_name IN ('train', 'test', 'inference')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (length(trim(external_key)) > 0)
);

CREATE TABLE nb_vocabulary (
    token_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    document_frequency INTEGER NOT NULL DEFAULT 0 CHECK (document_frequency >= 0),
    CHECK (token = lower(token)),
    CHECK (length(token) > 0)
);

CREATE TABLE nb_document_terms (
    document_id BIGINT NOT NULL REFERENCES nb_documents(document_id)
        ON DELETE CASCADE,
    token_id BIGINT NOT NULL REFERENCES nb_vocabulary(token_id),
    term_frequency INTEGER NOT NULL CHECK (term_frequency > 0),
    PRIMARY KEY (document_id, token_id)
);

CREATE TABLE nb_class_statistics (
    class_id BIGINT PRIMARY KEY REFERENCES nb_classes(class_id)
        ON DELETE CASCADE,
    training_document_count INTEGER NOT NULL CHECK (training_document_count > 0),
    training_token_count BIGINT NOT NULL CHECK (training_token_count >= 0)
);

CREATE TABLE nb_class_term_statistics (
    class_id BIGINT NOT NULL REFERENCES nb_classes(class_id)
        ON DELETE CASCADE,
    token_id BIGINT NOT NULL REFERENCES nb_vocabulary(token_id)
        ON DELETE CASCADE,
    term_count BIGINT NOT NULL CHECK (term_count >= 0),
    document_presence_count INTEGER NOT NULL CHECK (document_presence_count >= 0),
    PRIMARY KEY (class_id, token_id)
);

CREATE TABLE nb_model_configuration (
    model_name TEXT PRIMARY KEY,
    smoothing_alpha DOUBLE PRECISION NOT NULL CHECK (smoothing_alpha > 0),
    feature_mode TEXT NOT NULL CHECK (feature_mode IN ('multinomial', 'bernoulli')),
    minimum_confidence DOUBLE PRECISION NOT NULL
        CHECK (minimum_confidence BETWEEN 0 AND 1),
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE nb_predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES nb_documents(document_id)
        ON DELETE CASCADE,
    model_name TEXT NOT NULL REFERENCES nb_model_configuration(model_name),
    predicted_class_id BIGINT NOT NULL REFERENCES nb_classes(class_id),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    requires_manual_review BOOLEAN NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, model_name)
);

CREATE INDEX idx_nb_documents_split_class
    ON nb_documents (split_name, class_id);

CREATE INDEX idx_nb_document_terms_token
    ON nb_document_terms (token_id, document_id);

CREATE INDEX idx_nb_class_term_token
    ON nb_class_term_statistics (token_id, class_id);

CREATE INDEX idx_nb_predictions_review_queue
    ON nb_predictions (requires_manual_review, predicted_at DESC);

INSERT INTO nb_classes (class_name, description) VALUES
    ('technical', 'Infrastructure, application, latency, and connection incidents'),
    ('billing', 'Invoices, payments, receipts, and overdue balances'),
    ('account', 'Passwords, authentication, login, and access');

INSERT INTO nb_documents (
    external_key, document_text, class_id, split_name
)
SELECT
    sample.external_key,
    sample.document_text,
    class_record.class_id,
    sample.split_name
FROM (
    VALUES
        ('T-001', 'server latency timeout connection failure', 'technical', 'train'),
        ('T-002', 'database server connection error', 'technical', 'train'),
        ('T-003', 'network latency server', 'technical', 'train'),
        ('T-004', 'deployment failed timeout server', 'technical', 'train'),
        ('T-005', 'server connection timeout', 'technical', 'test'),
        ('B-001', 'invoice payment receipt billing', 'billing', 'train'),
        ('B-002', 'overdue invoice account payment', 'billing', 'train'),
        ('B-003', 'billing receipt transaction', 'billing', 'train'),
        ('B-004', 'payment invoice account overdue', 'billing', 'train'),
        ('B-005', 'invoice payment overdue', 'billing', 'test'),
        ('A-001', 'password reset login account', 'account', 'train'),
        ('A-002', 'authentication password security', 'account', 'train'),
        ('A-003', 'login access password', 'account', 'train'),
        ('A-004', 'account login authentication access', 'account', 'train'),
        ('A-005', 'password login access', 'account', 'test'),
        ('I-001', 'server latency timeout after deployment', NULL, 'inference'),
        ('I-002', 'invoice payment overdue', NULL, 'inference'),
        ('I-003', 'unfamiliar phrase with unknown terminology', NULL, 'inference')
) AS sample(external_key, document_text, class_name, split_name)
LEFT JOIN nb_classes AS class_record
    ON class_record.class_name = sample.class_name;

-- Tokenization is deliberately performed in PostgreSQL so the training
-- statistics and inference statistics share one normalization rule.
-- The regular expression retains letters and digits and lowercases tokens.
WITH token_occurrences AS (
    SELECT
        document.document_id,
        lower(token_match.token) AS token,
        count(*)::INTEGER AS term_frequency
    FROM nb_documents AS document
    CROSS JOIN LATERAL regexp_matches(
        lower(document.document_text),
        '[[:alnum:]]+',
        'g'
    ) AS token_match(token)
    WHERE document.split_name IN ('train', 'test', 'inference')
    GROUP BY document.document_id, lower(token_match.token)
),
inserted_tokens AS (
    INSERT INTO nb_vocabulary (token, document_frequency)
    SELECT token, 0
    FROM token_occurrences
    GROUP BY token
    ON CONFLICT (token) DO NOTHING
    RETURNING token_id
)
INSERT INTO nb_document_terms (document_id, token_id, term_frequency)
SELECT occurrence.document_id, vocabulary.token_id, occurrence.term_frequency
FROM token_occurrences AS occurrence
JOIN nb_vocabulary AS vocabulary USING (token)
ON CONFLICT (document_id, token_id)
DO UPDATE SET term_frequency = EXCLUDED.term_frequency;

UPDATE nb_vocabulary AS vocabulary
SET document_frequency = (
    SELECT count(*)::INTEGER
    FROM nb_document_terms AS terms
    JOIN nb_documents AS document USING (document_id)
    WHERE terms.token_id = vocabulary.token_id
      AND document.split_name IN ('train', 'test', 'inference')
);

-- Training counts exclude test and inference documents to prevent leakage.
INSERT INTO nb_class_statistics (
    class_id, training_document_count, training_token_count
)
SELECT
    class_record.class_id,
    count(DISTINCT document.document_id)::INTEGER,
    COALESCE(sum(terms.term_frequency), 0)::BIGINT
FROM nb_classes AS class_record
JOIN nb_documents AS document
    ON document.class_id = class_record.class_id
   AND document.split_name = 'train'
LEFT JOIN nb_document_terms AS terms
    ON terms.document_id = document.document_id
GROUP BY class_record.class_id;

-- Every class receives a row for every vocabulary token. Missing counts are
-- represented by zero, allowing smoothing to be applied consistently.
INSERT INTO nb_class_term_statistics (
    class_id, token_id, term_count, document_presence_count
)
SELECT
    class_record.class_id,
    vocabulary.token_id,
    COALESCE(sum(terms.term_frequency), 0)::BIGINT,
    count(DISTINCT terms.document_id)::INTEGER
FROM nb_classes AS class_record
CROSS JOIN nb_vocabulary AS vocabulary
LEFT JOIN nb_documents AS document
    ON document.class_id = class_record.class_id
   AND document.split_name = 'train'
LEFT JOIN nb_document_terms AS terms
    ON terms.document_id = document.document_id
   AND terms.token_id = vocabulary.token_id
GROUP BY class_record.class_id, vocabulary.token_id;

INSERT INTO nb_model_configuration (
    model_name, smoothing_alpha, feature_mode, minimum_confidence
) VALUES
    ('multinomial_alpha1', 1.0, 'multinomial', 0.55),
    ('bernoulli_alpha1', 1.0, 'bernoulli', 0.55);

-- Multinomial Naive Bayes:
-- log P(class) + sum(term frequency * log P(token | class)).
-- Repeated words contribute repeatedly. Alpha smooths token counts.
CREATE VIEW nb_multinomial_predictions AS
WITH vocabulary_size AS (
    SELECT count(*)::DOUBLE PRECISION AS size
    FROM nb_vocabulary
),
class_priors AS (
    SELECT
        statistics.class_id,
        class_record.class_name,
        ln(
            statistics.training_document_count::DOUBLE PRECISION
            / sum(statistics.training_document_count) OVER ()
        ) AS log_prior,
        statistics.training_token_count
    FROM nb_class_statistics AS statistics
    JOIN nb_classes AS class_record USING (class_id)
),
inference_documents AS (
    SELECT document_id
    FROM nb_documents
    WHERE split_name = 'inference'
),
candidate_scores AS (
    SELECT
        document.document_id,
        document.external_key,
        document.document_text,
        prior.class_id,
        prior.class_name,
        prior.log_prior
        + COALESCE(sum(
            inference_terms.term_frequency::DOUBLE PRECISION
            * ln(
                (
                    class_terms.term_count::DOUBLE PRECISION
                    + configuration.smoothing_alpha
                )
                / (
                    prior.training_token_count
                    + configuration.smoothing_alpha * vocabulary_size.size
                )
            )
        ), 0.0) AS log_score
    FROM inference_documents AS inference
    JOIN nb_documents AS document USING (document_id)
    CROSS JOIN class_priors AS prior
    CROSS JOIN vocabulary_size
    JOIN nb_model_configuration AS configuration
        ON configuration.model_name = 'multinomial_alpha1'
       AND configuration.active
    LEFT JOIN nb_document_terms AS inference_terms
        ON inference_terms.document_id = inference.document_id
    LEFT JOIN nb_class_term_statistics AS class_terms
        ON class_terms.class_id = prior.class_id
       AND class_terms.token_id = inference_terms.token_id
    GROUP BY
        document.document_id,
        document.external_key,
        document.document_text,
        prior.class_id,
        prior.class_name,
        prior.log_prior,
        prior.training_token_count,
        configuration.smoothing_alpha,
        vocabulary_size.size
),
stable_scores AS (
    SELECT
        candidate_scores.*,
        max(log_score) OVER (PARTITION BY document_id) AS maximum_score
    FROM candidate_scores
),
normalized AS (
    SELECT
        stable_scores.*,
        exp(log_score - maximum_score)
        / sum(exp(log_score - maximum_score))
            OVER (PARTITION BY document_id) AS probability
    FROM stable_scores
)
SELECT
    document_id,
    external_key,
    document_text,
    class_id AS predicted_class_id,
    class_name AS predicted_class,
    probability AS confidence,
    log_score
FROM (
    SELECT
        normalized.*,
        row_number() OVER (
            PARTITION BY document_id
            ORDER BY probability DESC, class_name
        ) AS rank_in_document
    FROM normalized
) AS ranked
WHERE rank_in_document = 1;

-- Bernoulli Naive Bayes:
-- Each token is present or absent in a document.
-- Repeated occurrences have no additional effect.
-- Both presence and absence contribute to the class score.
CREATE VIEW nb_bernoulli_predictions AS
WITH vocabulary_size AS (
    SELECT count(*)::DOUBLE PRECISION AS size
    FROM nb_vocabulary
),
class_priors AS (
    SELECT
        statistics.class_id,
        class_record.class_name,
        ln(
            statistics.training_document_count::DOUBLE PRECISION
            / sum(statistics.training_document_count) OVER ()
        ) AS log_prior,
        statistics.training_document_count
    FROM nb_class_statistics AS statistics
    JOIN nb_classes AS class_record USING (class_id)
),
inference_documents AS (
    SELECT document_id
    FROM nb_documents
    WHERE split_name = 'inference'
),
presence_features AS (
    SELECT
        inference.document_id,
        vocabulary.token_id,
        (inference_terms.token_id IS NOT NULL) AS is_present
    FROM inference_documents AS inference
    CROSS JOIN nb_vocabulary AS vocabulary
    LEFT JOIN nb_document_terms AS inference_terms
        ON inference_terms.document_id = inference.document_id
       AND inference_terms.token_id = vocabulary.token_id
),
candidate_scores AS (
    SELECT
        document.document_id,
        document.external_key,
        document.document_text,
        prior.class_id,
        prior.class_name,
        prior.log_prior
        + sum(
            CASE
                WHEN features.is_present THEN
                    ln(
                        (
                            class_terms.document_presence_count
                            + configuration.smoothing_alpha
                        )
                        / (
                            prior.training_document_count
                            + 2.0 * configuration.smoothing_alpha
                        )
                    )
                ELSE
                    ln(
                        1.0
                        - (
                            (
                                class_terms.document_presence_count
                                + configuration.smoothing_alpha
                            )
                            / (
                                prior.training_document_count
                                + 2.0 * configuration.smoothing_alpha
                            )
                        )
                    )
            END
        ) AS log_score
    FROM inference_documents AS inference
    JOIN nb_documents AS document USING (document_id)
    CROSS JOIN class_priors AS prior
    CROSS JOIN presence_features AS features
    JOIN nb_model_configuration AS configuration
        ON configuration.model_name = 'bernoulli_alpha1'
       AND configuration.active
    JOIN nb_class_term_statistics AS class_terms
        ON class_terms.class_id = prior.class_id
       AND class_terms.token_id = features.token_id
    WHERE features.document_id = inference.document_id
    GROUP BY
        document.document_id,
        document.external_key,
        document.document_text,
        prior.class_id,
        prior.class_name,
        prior.log_prior,
        prior.training_document_count,
        configuration.smoothing_alpha
),
stable_scores AS (
    SELECT
        candidate_scores.*,
        max(log_score) OVER (PARTITION BY document_id) AS maximum_score
    FROM candidate_scores
),
normalized AS (
    SELECT
        stable_scores.*,
        exp(log_score - maximum_score)
        / sum(exp(log_score - maximum_score))
            OVER (PARTITION BY document_id) AS probability
    FROM stable_scores
)
SELECT
    document_id,
    external_key,
    document_text,
    class_id AS predicted_class_id,
    class_name AS predicted_class,
    probability AS confidence,
    log_score
FROM (
    SELECT
        normalized.*,
        row_number() OVER (
            PARTITION BY document_id
            ORDER BY probability DESC, class_name
        ) AS rank_in_document
    FROM normalized
) AS ranked
WHERE rank_in_document = 1;

-- Compare the two feature assumptions on the same inference documents.
SELECT
    multinomial.external_key,
    multinomial.document_text,
    multinomial.predicted_class AS multinomial_prediction,
    round(multinomial.confidence::NUMERIC, 4) AS multinomial_confidence,
    bernoulli.predicted_class AS bernoulli_prediction,
    round(bernoulli.confidence::NUMERIC, 4) AS bernoulli_confidence
FROM nb_multinomial_predictions AS multinomial
JOIN nb_bernoulli_predictions AS bernoulli USING (document_id)
ORDER BY multinomial.external_key;

-- Persist one model's predictions for operational review.
INSERT INTO nb_predictions (
    document_id,
    model_name,
    predicted_class_id,
    confidence,
    requires_manual_review
)
SELECT
    prediction.document_id,
    'multinomial_alpha1',
    prediction.predicted_class_id,
    prediction.confidence,
    prediction.confidence < configuration.minimum_confidence
FROM nb_multinomial_predictions AS prediction
JOIN nb_model_configuration AS configuration
    ON configuration.model_name = 'multinomial_alpha1'
ON CONFLICT (document_id, model_name)
DO UPDATE SET
    predicted_class_id = EXCLUDED.predicted_class_id,
    confidence = EXCLUDED.confidence,
    requires_manual_review = EXCLUDED.requires_manual_review,
    predicted_at = CURRENT_TIMESTAMP;

-- Diagnostics expose vocabulary size, class balance, and training token volume.
CREATE VIEW nb_model_diagnostics AS
SELECT
    class_record.class_name,
    statistics.training_document_count,
    statistics.training_token_count,
    (SELECT count(*) FROM nb_vocabulary) AS vocabulary_size,
    round(
        statistics.training_document_count::NUMERIC
        / sum(statistics.training_document_count) OVER (),
        4
    ) AS empirical_class_prior
FROM nb_class_statistics AS statistics
JOIN nb_classes AS class_record USING (class_id);

SELECT *
FROM nb_model_diagnostics
ORDER BY class_name;

SELECT
    document.external_key,
    class_record.class_name AS predicted_class,
    round(prediction.confidence::NUMERIC, 4) AS confidence,
    prediction.requires_manual_review,
    prediction.predicted_at
FROM nb_predictions AS prediction
JOIN nb_documents AS document USING (document_id)
JOIN nb_classes AS class_record
    ON class_record.class_id = prediction.predicted_class_id
ORDER BY document.external_key;

-- The following query returns class-conditional probabilities for inspection.
-- The multinomial denominator is total class tokens plus alpha times |V|.
SELECT
    class_record.class_name,
    vocabulary.token,
    statistics.term_count,
    round(
        (
            (
                statistics.term_count::DOUBLE PRECISION
                + configuration.smoothing_alpha
            )
            / (
                class_statistics.training_token_count
                + configuration.smoothing_alpha
                * (SELECT count(*) FROM nb_vocabulary)
            )
        )::NUMERIC,
        6
    ) AS smoothed_multinomial_probability
FROM nb_class_term_statistics AS statistics
JOIN nb_classes AS class_record USING (class_id)
JOIN nb_vocabulary AS vocabulary USING (token_id)
JOIN nb_class_statistics AS class_statistics USING (class_id)
JOIN nb_model_configuration AS configuration
    ON configuration.model_name = 'multinomial_alpha1'
WHERE vocabulary.token IN ('server', 'invoice', 'password')
ORDER BY vocabulary.token, class_record.class_name;

-- Constraints prevent invalid smoothing values, invalid confidence scores,
-- unknown class references, and duplicate prediction rows.
-- The database does not guarantee that a client has recomputed predictions
-- after retraining. Model versioning and atomic deployment must handle that.
COMMIT;
