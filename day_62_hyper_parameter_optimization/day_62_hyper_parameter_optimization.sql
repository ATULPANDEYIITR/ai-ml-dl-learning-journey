DROP SCHEMA IF EXISTS hpo CASCADE;
CREATE SCHEMA hpo;

CREATE TYPE hpo.search_strategy AS ENUM ('grid', 'random');
CREATE TYPE hpo.trial_status AS ENUM ('queued', 'running', 'completed', 'failed');

CREATE TABLE hpo.experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    strategy hpo.search_strategy NOT NULL,
    budget INTEGER NOT NULL CHECK (budget > 0),
    metric_name TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('maximize', 'minimize')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE hpo.hyperparameters (
    hyperparameter_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    data_type TEXT NOT NULL CHECK (
        data_type IN ('float', 'integer', 'categorical')
    )
);

CREATE TABLE hpo.search_space_values (
    value_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hyperparameter_id BIGINT NOT NULL
        REFERENCES hpo.hyperparameters(hyperparameter_id)
        ON DELETE CASCADE,
    numeric_value DOUBLE PRECISION,
    integer_value INTEGER,
    categorical_value TEXT,
    CHECK (
        (numeric_value IS NOT NULL)::INTEGER +
        (integer_value IS NOT NULL)::INTEGER +
        (categorical_value IS NOT NULL)::INTEGER = 1
    )
);

CREATE TABLE hpo.trials (
    trial_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES hpo.experiments(experiment_id)
        ON DELETE CASCADE,
    trial_number INTEGER NOT NULL CHECK (trial_number > 0),
    status hpo.trial_status NOT NULL DEFAULT 'queued',
    learning_rate DOUBLE PRECISION NOT NULL CHECK (learning_rate > 0),
    max_depth INTEGER NOT NULL CHECK (max_depth > 0),
    estimators INTEGER NOT NULL CHECK (estimators > 0),
    validation_score DOUBLE PRECISION,
    duration_seconds NUMERIC(12, 3),
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE (experiment_id, trial_number)
);

CREATE TABLE hpo_trial_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trial_id BIGINT NOT NULL REFERENCES hpo.trials(trial_id)
        ON DELETE CASCADE,
    old_status hpo.trial_status,
    new_status hpo.trial_status NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message TEXT
);

CREATE OR REPLACE FUNCTION hpo.validate_trial_status()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'completed' AND NEW.status <> 'completed' THEN
        RAISE EXCEPTION 'A completed trial cannot be moved to another state';
    END IF;

    IF NEW.status = 'completed' THEN
        IF NEW.validation_score IS NULL THEN
            RAISE EXCEPTION
                'Completed trial % requires validation_score',
                NEW.trial_id;
        END IF;

        IF NEW.completed_at IS NULL THEN
            NEW.completed_at := CURRENT_TIMESTAMP;
        END IF;
    END IF;

    IF NEW.status = 'running' AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_trial_status
BEFORE UPDATE ON hpo.trials
FOR EACH ROW
EXECUTE FUNCTION hpo.validate_trial_status();

CREATE OR REPLACE FUNCTION hpo.log_trial_status_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO hpo_trial_events (
            trial_id,
            old_status,
            new_status,
            message
        )
        VALUES (
            NEW.trial_id,
            OLD.status,
            NEW.status,
            'Trial state transition'
        );
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_log_trial_status
AFTER UPDATE ON hpo.trials
FOR EACH ROW
EXECUTE FUNCTION hpo.log_trial_status_change();

CREATE INDEX idx_trials_experiment_score
ON hpo.trials (experiment_id, validation_score DESC)
WHERE status = 'completed';

CREATE INDEX idx_trials_status
ON hpo.trials (status);

CREATE INDEX idx_search_space_parameter
ON hpo.search_space_values (hyperparameter_id);

INSERT INTO hpo.experiments (
    name,
    strategy,
    budget,
    metric_name,
    direction
)
VALUES
    ('gradient_boosting_grid', 'grid', 60, 'validation_accuracy', 'maximize'),
    ('gradient_boosting_random', 'random', 20, 'validation_accuracy', 'maximize');

INSERT INTO hpo.hyperparameters (name, data_type)
VALUES
    ('learning_rate', 'float'),
    ('max_depth', 'integer'),
    ('estimators', 'integer');

INSERT INTO hpo.search_space_values (
    hyperparameter_id,
    numeric_value
)
SELECT
    h.hyperparameter_id,
    value
FROM hpo.hyperparameters h
CROSS JOIN LATERAL (
    VALUES
        (0.005::DOUBLE PRECISION),
        (0.01::DOUBLE PRECISION),
        (0.02::DOUBLE PRECISION),
        (0.05::DOUBLE PRECISION),
        (0.1::DOUBLE PRECISION),
        (0.2::DOUBLE PRECISION)
) AS values(value)
WHERE h.name = 'learning_rate';

INSERT INTO hpo.search_space_values (
    hyperparameter_id,
    integer_value
)
SELECT
    h.hyperparameter_id,
    value
FROM hpo.hyperparameters h
CROSS JOIN LATERAL (
    VALUES (2), (4), (6), (8), (10)
) AS values(value)
WHERE h.name = 'max_depth';

INSERT INTO hpo.search_space_values (
    hyperparameter_id,
    integer_value
)
SELECT
    h.hyperparameter_id,
    value
FROM hpo.hyperparameters h
CROSS JOIN LATERAL (
    VALUES (50), (100), (200), (400)
) AS values(value)
WHERE h.name = 'estimators';

INSERT INTO hpo.trials (
    experiment_id,
    trial_number,
    status,
    learning_rate,
    max_depth,
    estimators,
    validation_score,
    duration_seconds,
    started_at,
    completed_at
)
SELECT
    e.experiment_id,
    trial_number,
    'completed',
    learning_rate,
    max_depth,
    estimators,
    validation_score,
    duration_seconds,
    CURRENT_TIMESTAMP - INTERVAL '5 minutes',
    CURRENT_TIMESTAMP
FROM hpo.experiments e
CROSS JOIN (
    VALUES
        (1, 0.01::DOUBLE PRECISION, 4, 100, 0.9120, 5.10),
        (2, 0.02::DOUBLE PRECISION, 6, 100, 0.9340, 6.80),
        (3, 0.05::DOUBLE PRECISION, 6, 200, 0.9580, 9.20),
        (4, 0.10::DOUBLE PRECISION, 8, 400, 0.9210, 17.40),
        (5, 0.005::DOUBLE PRECISION, 10, 50, 0.8840, 5.00),
        (6, 0.20::DOUBLE PRECISION, 2, 100, 0.8710, 4.00)
) AS trials(
    trial_number,
    learning_rate,
    max_depth,
    estimators,
    validation_score,
    duration_seconds
)
WHERE e.name = 'gradient_boosting_grid';

INSERT INTO hpo.trials (
    experiment_id,
    trial_number,
    status,
    learning_rate,
    max_depth,
    estimators,
    validation_score,
    duration_seconds,
    started_at,
    completed_at
)
SELECT
    e.experiment_id,
    trial_number,
    'completed',
    learning_rate,
    max_depth,
    estimators,
    validation_score,
    duration_seconds,
    CURRENT_TIMESTAMP - INTERVAL '3 minutes',
    CURRENT_TIMESTAMP
FROM hpo.experiments e
CROSS JOIN (
    VALUES
        (1, 0.01::DOUBLE PRECISION, 8, 200, 0.9300, 10.10),
        (2, 0.05::DOUBLE PRECISION, 6, 200, 0.9610, 9.30),
        (3, 0.20::DOUBLE PRECISION, 4, 50, 0.8500, 3.40),
        (4, 0.02::DOUBLE PRECISION, 6, 400, 0.9440, 12.90),
        (5, 0.10::DOUBLE PRECISION, 10, 100, 0.9020, 10.40)
) AS trials(
    trial_number,
    learning_rate,
    max_depth,
    estimators,
    validation_score,
    duration_seconds
)
WHERE e.name = 'gradient_boosting_random';

CREATE OR REPLACE VIEW hpo.completed_trial_ranking AS
SELECT
    e.name AS experiment,
    e.strategy,
    t.trial_number,
    t.learning_rate,
    t.max_depth,
    t.estimators,
    t.validation_score,
    t.duration_seconds,
    RANK() OVER (
        PARTITION BY t.experiment_id
        ORDER BY t.validation_score DESC
    ) AS score_rank
FROM hpo.trials t
JOIN hpo.experiments e
    ON e.experiment_id = t.experiment_id
WHERE t.status = 'completed';

CREATE OR REPLACE VIEW hpo.experiment_summary AS
SELECT
    e.experiment_id,
    e.name,
    e.strategy,
    e.budget,
    COUNT(t.trial_id) AS completed_trials,
    MAX(t.validation_score) AS best_score,
    AVG(t.validation_score) AS average_score,
    SUM(t.duration_seconds) AS total_duration
FROM hpo.experiments e
LEFT JOIN hpo.trials t
    ON t.experiment_id = e.experiment_id
   AND t.status = 'completed'
GROUP BY
    e.experiment_id,
    e.name,
    e.strategy,
    e.budget;

-- Count the theoretical Cartesian-product size of a search space.
WITH dimension_counts AS (
    SELECT
        h.name,
        COUNT(*) AS value_count
    FROM hpo.hyperparameters h
    JOIN hpo.search_space_values v
      ON v.hyperparameter_id = h.hyperparameter_id
    GROUP BY h.name
)
SELECT
    EXP(SUM(LN(value_count)))::BIGINT AS total_grid_configurations
FROM dimension_counts;

-- Best configuration for each experiment.
SELECT DISTINCT ON (experiment_id)
    experiment_id,
    trial_number,
    learning_rate,
    max_depth,
    estimators,
    validation_score
FROM hpo.trials
WHERE status = 'completed'
ORDER BY
    experiment_id,
    validation_score DESC,
    trial_number;

-- Compare search strategies by quality and evaluation cost.
SELECT
    e.strategy,
    COUNT(t.trial_id) AS trials,
    ROUND(MAX(t.validation_score)::NUMERIC, 4) AS best_score,
    ROUND(AVG(t.validation_score)::NUMERIC, 4) AS average_score,
    ROUND(SUM(t.duration_seconds), 2) AS total_duration
FROM hpo.experiments e
JOIN hpo.trials t
  ON t.experiment_id = e.experiment_id
WHERE t.status = 'completed'
GROUP BY e.strategy
ORDER BY best_score DESC;

-- Find trials that are substantially below the experiment's best score.
WITH experiment_scores AS (
    SELECT
        experiment_id,
        MAX(validation_score) AS best_score
    FROM hpo.trials
    WHERE status = 'completed'
    GROUP BY experiment_id
)
SELECT
    t.trial_id,
    t.experiment_id,
    t.trial_number,
    t.validation_score,
    s.best_score,
    s.best_score - t.validation_score AS score_gap
FROM hpo.trials t
JOIN experiment_scores s
  ON s.experiment_id = t.experiment_id
WHERE t.status = 'completed'
  AND s.best_score - t.validation_score >= 0.05
ORDER BY score_gap DESC;

-- Demonstrate a transactional state transition.
BEGIN;

UPDATE hpo.trials
SET status = 'running'
WHERE trial_id = (
    SELECT MIN(trial_id)
    FROM hpo.trials
    WHERE status = 'completed'
);

-- The transaction is intentionally rolled back so the sample dataset remains
-- unchanged after execution.
ROLLBACK;

-- Validation query for invalid parameter conditions.
SELECT
    trial_id,
    CASE
        WHEN learning_rate <= 0 THEN 'invalid learning rate'
        WHEN max_depth <= 0 THEN 'invalid depth'
        WHEN estimators <= 0 THEN 'invalid estimator count'
        ELSE 'valid'
    END AS parameter_validation
FROM hpo.trials
ORDER BY trial_id;
