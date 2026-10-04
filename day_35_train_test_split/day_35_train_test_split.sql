DROP SCHEMA IF EXISTS ml_split_demo CASCADE;
CREATE SCHEMA ml_split_demo;
SET search_path TO ml_split_demo;

-- PostgreSQL-compatible relational model for an experiment in which a model
-- predicts whether a repository change requires additional intervention.
CREATE TABLE repository (
    repository_id BIGSERIAL PRIMARY KEY,
    repository_name TEXT NOT NULL UNIQUE
);

CREATE TABLE branch (
    branch_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repository(repository_id),
    branch_name TEXT NOT NULL,
    is_protected BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (repository_id, branch_name)
);

CREATE TABLE observation (
    observation_id BIGSERIAL PRIMARY KEY,
    repository_id BIGINT NOT NULL REFERENCES repository(repository_id),
    branch_id BIGINT NOT NULL REFERENCES branch(branch_id),
    observed_at TIMESTAMPTZ NOT NULL,
    entity_key TEXT NOT NULL,
    lines_changed INTEGER NOT NULL CHECK (lines_changed >= 0),
    files_changed INTEGER NOT NULL CHECK (files_changed >= 0),
    review_comments INTEGER NOT NULL CHECK (review_comments >= 0),
    previous_failures INTEGER NOT NULL CHECK (previous_failures >= 0),
    experience_score NUMERIC(8, 3) NOT NULL CHECK (experience_score >= 0),
    target_label SMALLINT NOT NULL CHECK (target_label IN (0, 1))
);

CREATE TABLE experiment (
    experiment_id BIGSERIAL PRIMARY KEY,
    experiment_name TEXT NOT NULL UNIQUE,
    random_seed INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TYPE partition_type AS ENUM ('train', 'validation', 'test');

CREATE TABLE experiment_partition (
    experiment_id BIGINT NOT NULL REFERENCES experiment(experiment_id),
    observation_id BIGINT NOT NULL REFERENCES observation(observation_id),
    partition partition_type NOT NULL,
    PRIMARY KEY (experiment_id, observation_id)
);

CREATE TABLE preprocessing_fit (
    experiment_id BIGINT NOT NULL REFERENCES experiment(experiment_id),
    feature_name TEXT NOT NULL,
    mean_value NUMERIC(18, 8) NOT NULL,
    standard_deviation NUMERIC(18, 8) NOT NULL CHECK (standard_deviation > 0),
    fitted_from partition_type NOT NULL,
    CHECK (fitted_from = 'train'),
    PRIMARY KEY (experiment_id, feature_name)
);

CREATE TABLE model_evaluation (
    evaluation_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES experiment(experiment_id),
    partition partition_type NOT NULL,
    accuracy NUMERIC(10, 6) CHECK (accuracy BETWEEN 0 AND 1),
    precision_score NUMERIC(10, 6) CHECK (precision_score BETWEEN 0 AND 1),
    recall_score NUMERIC(10, 6) CHECK (recall_score BETWEEN 0 AND 1),
    f1_score NUMERIC(10, 6) CHECK (f1_score BETWEEN 0 AND 1),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (experiment_id, partition)
);

CREATE INDEX idx_observation_time
    ON observation (repository_id, observed_at);

CREATE INDEX idx_observation_entity
    ON observation (repository_id, entity_key);

CREATE INDEX idx_partition_lookup
    ON experiment_partition (experiment_id, partition);

CREATE INDEX idx_target_label
    ON observation (target_label);

INSERT INTO repository (repository_name)
VALUES ('risk-engine');

INSERT INTO branch (repository_id, branch_name, is_protected)
VALUES
    (1, 'main', TRUE),
    (1, 'develop', FALSE);

INSERT INTO observation (
    repository_id,
    branch_id,
    observed_at,
    entity_key,
    lines_changed,
    files_changed,
    review_comments,
    previous_failures,
    experience_score,
    target_label
)
SELECT
    1,
    CASE WHEN id % 2 = 0 THEN 1 ELSE 2 END,
    TIMESTAMPTZ '2026-01-01 09:00:00+00'
        + (id || ' days')::interval,
    'developer-' || (id % 8),
    20 + (id * 17) % 280,
    1 + id % 9,
    id % 7,
    id % 5,
    2 + (id % 9),
    CASE
        WHEN
            0.025 * (20 + (id * 17) % 280)
            + 0.55 * (1 + id % 9)
            + 0.75 * (id % 7)
            + 0.9 * (id % 5)
            - 0.65 * (2 + id % 9)
            > 3
        THEN 1
        ELSE 0
    END
FROM generate_series(0, 119) AS generated(id);

INSERT INTO experiment (experiment_name, random_seed)
VALUES ('stratified-risk-baseline', 2026);

-- A deterministic hash-based assignment makes the example reproducible.
-- In a real experiment, the partitioning algorithm should be versioned with
-- the experiment metadata so that the same dataset can be reconstructed.
INSERT INTO experiment_partition (experiment_id, observation_id, partition)
SELECT
    1,
    observation_id,
    CASE
        WHEN mod(abs(hashtextextended(observation_id::text, 2026)), 100) < 70
            THEN 'train'::partition_type
        WHEN mod(abs(hashtextextended(observation_id::text, 2026)), 100) < 85
            THEN 'validation'::partition_type
        ELSE 'test'::partition_type
    END
FROM observation;

-- The database can verify that every observation belongs to one and only one
-- partition for an experiment because the composite primary key prevents
-- duplicate experiment/observation assignments.
SELECT
    partition,
    COUNT(*) AS observations,
    ROUND(AVG(target_label::numeric), 4) AS positive_rate
FROM experiment_partition ep
JOIN observation o
    ON o.observation_id = ep.observation_id
WHERE ep.experiment_id = 1
GROUP BY partition
ORDER BY partition;

-- Check whether a random split accidentally separates the same entity across
-- train and test. Such overlap can make an evaluation optimistic when entity
-- identity carries predictive information.
SELECT DISTINCT
    train.entity_key
FROM experiment_partition train_partition
JOIN observation train
    ON train.observation_id = train_partition.observation_id
JOIN experiment_partition test_partition
    ON test_partition.experiment_id = train_partition.experiment_id
JOIN observation test
    ON test.observation_id = test_partition.observation_id
WHERE train_partition.experiment_id = 1
  AND train_partition.partition = 'train'
  AND test_partition.partition = 'test'
  AND train.entity_key = test.entity_key
ORDER BY train.entity_key;

-- Fit preprocessing statistics using training observations only.
INSERT INTO preprocessing_fit (
    experiment_id,
    feature_name,
    mean_value,
    standard_deviation,
    fitted_from
)
WITH train_rows AS (
    SELECT o.*
    FROM observation o
    JOIN experiment_partition ep
      ON ep.observation_id = o.observation_id
    WHERE ep.experiment_id = 1
      AND ep.partition = 'train'
),
feature_values AS (
    SELECT 'lines_changed' AS feature_name,
           lines_changed::numeric AS value
    FROM train_rows
    UNION ALL
    SELECT 'files_changed', files_changed::numeric
    FROM train_rows
    UNION ALL
    SELECT 'review_comments', review_comments::numeric
    FROM train_rows
    UNION ALL
    SELECT 'previous_failures', previous_failures::numeric
    FROM train_rows
    UNION ALL
    SELECT 'experience_score', experience_score
    FROM train_rows
)
SELECT
    1,
    feature_name,
    AVG(value),
    GREATEST(STDDEV_POP(value), 0.000001),
    'train'::partition_type
FROM feature_values
GROUP BY feature_name;

-- The following query intentionally detects a prohibited situation:
-- preprocessing parameters should never be recorded as fitted from validation
-- or test data. The CHECK constraint prevents such a row from being inserted.
SELECT *
FROM preprocessing_fit
WHERE fitted_from <> 'train';

-- Training and test label rates can be compared to detect severe distribution
-- differences. A difference is a diagnostic signal, not automatic proof that
-- the split is invalid.
WITH rates AS (
    SELECT
        ep.partition,
        AVG(o.target_label::numeric) AS positive_rate
    FROM experiment_partition ep
    JOIN observation o
      ON o.observation_id = ep.observation_id
    WHERE ep.experiment_id = 1
    GROUP BY ep.partition
)
SELECT
    partition,
    ROUND(positive_rate, 4) AS positive_rate
FROM rates
ORDER BY partition;

-- A chronological split avoids using future observations to predict earlier
-- observations. This query identifies the time boundaries that such a split
-- would establish.
WITH ordered AS (
    SELECT
        o.*,
        ROW_NUMBER() OVER (ORDER BY observed_at, observation_id) AS row_number,
        COUNT(*) OVER () AS total_rows
    FROM observation o
    WHERE repository_id = 1
)
SELECT
    MIN(observed_at) FILTER (WHERE row_number <= total_rows * 0.70) AS train_start,
    MAX(observed_at) FILTER (WHERE row_number <= total_rows * 0.70) AS train_end,
    MIN(observed_at) FILTER (
        WHERE row_number > total_rows * 0.70
          AND row_number <= total_rows * 0.85
    ) AS validation_start,
    MAX(observed_at) FILTER (
        WHERE row_number > total_rows * 0.70
          AND row_number <= total_rows * 0.85
    ) AS validation_end,
    MIN(observed_at) FILTER (WHERE row_number > total_rows * 0.85) AS test_start
FROM ordered;

-- Group-aware splitting requires the entity boundary to be treated as a
-- first-class constraint. This report exposes entities crossing train/test.
SELECT
    o.entity_key,
    COUNT(*) FILTER (WHERE ep.partition = 'train') AS train_rows,
    COUNT(*) FILTER (WHERE ep.partition = 'test') AS test_rows
FROM observation o
JOIN experiment_partition ep
  ON ep.observation_id = o.observation_id
WHERE ep.experiment_id = 1
GROUP BY o.entity_key
HAVING COUNT(*) FILTER (WHERE ep.partition = 'train') > 0
   AND COUNT(*) FILTER (WHERE ep.partition = 'test') > 0
ORDER BY o.entity_key;

-- Validation metrics are model-selection evidence. Test metrics are final
-- evidence and should be recorded separately.
INSERT INTO model_evaluation (
    experiment_id,
    partition,
    accuracy,
    precision_score,
    recall_score,
    f1_score
)
VALUES
    (1, 'validation', 0.833333, 0.812500, 0.866667, 0.838710),
    (1, 'test',       0.816667, 0.800000, 0.842105, 0.820513);

SELECT
    experiment_id,
    partition,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    evaluated_at
FROM model_evaluation
WHERE experiment_id = 1
ORDER BY
    CASE partition
        WHEN 'validation' THEN 1
        WHEN 'test' THEN 2
        ELSE 3
    END;

-- Transactional integrity demonstration. The attempted assignment below is
-- rolled back, leaving the experiment unchanged.
BEGIN;

INSERT INTO experiment_partition (
    experiment_id,
    observation_id,
    partition
)
VALUES (1, 1, 'test');

ROLLBACK;

SELECT
    experiment_id,
    observation_id,
    partition
FROM experiment_partition
WHERE experiment_id = 1
  AND observation_id = 1;

-- Leakage principle:
-- Any transformation that estimates parameters from the complete dataset,
-- including scaling, imputation, feature selection, target encoding, or
-- dimensionality reduction, can leak held-out distribution information.
-- The preprocessing_fit table intentionally encodes the permitted boundary:
-- fitted_from must be TRAIN.
