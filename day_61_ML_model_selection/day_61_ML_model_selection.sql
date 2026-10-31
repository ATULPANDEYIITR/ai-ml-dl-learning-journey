DROP SCHEMA IF EXISTS ml_model_selection CASCADE;
CREATE SCHEMA ml_model_selection;

SET search_path = ml_model_selection, public;

CREATE TYPE model_family AS ENUM (
    'logistic_regression',
    'decision_tree'
);

CREATE TYPE cv_strategy AS ENUM (
    'k_fold',
    'stratified_k_fold',
    'repeated_stratified_k_fold'
);

CREATE TYPE evaluation_state AS ENUM (
    'planned',
    'running',
    'completed',
    'failed'
);

CREATE TABLE datasets (
    dataset_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    positive_count INTEGER NOT NULL CHECK (
        positive_count >= 0
        AND positive_count <= sample_count
    ),
    negative_count INTEGER GENERATED ALWAYS AS
        (sample_count - positive_count) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE models (
    model_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    family model_family NOT NULL,
    hyperparameters JSONB NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    CHECK (jsonb_typeof(hyperparameters) = 'object')
);

CREATE TABLE cv_experiments (
    experiment_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    experiment_name TEXT NOT NULL,
    strategy cv_strategy NOT NULL,
    folds INTEGER NOT NULL CHECK (folds BETWEEN 2 AND 20),
    repetitions INTEGER NOT NULL DEFAULT 1 CHECK (repetitions BETWEEN 1 AND 50),
    random_seed INTEGER NOT NULL,
    state evaluation_state NOT NULL DEFAULT 'planned',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, experiment_name)
);

CREATE TABLE cv_folds (
    fold_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES cv_experiments(experiment_id)
        ON DELETE CASCADE,
    repetition INTEGER NOT NULL CHECK (repetition >= 1),
    fold_number INTEGER NOT NULL CHECK (fold_number >= 1),
    training_samples INTEGER NOT NULL CHECK (training_samples > 0),
    validation_samples INTEGER NOT NULL CHECK (validation_samples > 0),
    training_positive INTEGER NOT NULL CHECK (training_positive >= 0),
    validation_positive INTEGER NOT NULL CHECK (validation_positive >= 0),
    UNIQUE(experiment_id, repetition, fold_number)
);

CREATE TABLE model_evaluations (
    evaluation_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES cv_experiments(experiment_id)
        ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES models(model_id),
    fold_id BIGINT REFERENCES cv_folds(fold_id)
        ON DELETE CASCADE,
    accuracy NUMERIC(8, 6) CHECK (accuracy BETWEEN 0 AND 1),
    precision_score NUMERIC(8, 6) CHECK (precision_score BETWEEN 0 AND 1),
    recall_score NUMERIC(8, 6) CHECK (recall_score BETWEEN 0 AND 1),
    f1_score NUMERIC(8, 6) CHECK (f1_score BETWEEN 0 AND 1),
    log_loss NUMERIC(12, 8) CHECK (log_loss >= 0),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, model_id, fold_id)
);

CREATE TABLE model_selection (
    selection_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES cv_experiments(experiment_id)
        ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES models(model_id),
    mean_f1 NUMERIC(8, 6) NOT NULL CHECK (mean_f1 BETWEEN 0 AND 1),
    std_f1 NUMERIC(8, 6) NOT NULL CHECK (std_f1 >= 0),
    mean_log_loss NUMERIC(12, 8) NOT NULL CHECK (mean_log_loss >= 0),
    selected BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, model_id)
);

CREATE TABLE holdout_evaluations (
    holdout_evaluation_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES cv_experiments(experiment_id)
        ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES models(model_id),
    accuracy NUMERIC(8, 6) NOT NULL CHECK (accuracy BETWEEN 0 AND 1),
    precision_score NUMERIC(8, 6) NOT NULL CHECK (precision_score BETWEEN 0 AND 1),
    recall_score NUMERIC(8, 6) NOT NULL CHECK (recall_score BETWEEN 0 AND 1),
    f1_score NUMERIC(8, 6) NOT NULL CHECK (f1_score BETWEEN 0 AND 1),
    log_loss NUMERIC(12, 8) NOT NULL CHECK (log_loss >= 0),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cv_folds_experiment
    ON cv_folds(experiment_id, repetition, fold_number);

CREATE INDEX idx_model_evaluations_experiment_model
    ON model_evaluations(experiment_id, model_id);

CREATE INDEX idx_model_selection_selected
    ON model_selection(experiment_id, selected)
    WHERE selected = TRUE;

CREATE OR REPLACE FUNCTION enforce_single_selected_model()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.selected THEN
        UPDATE model_selection
        SET selected = FALSE
        WHERE experiment_id = NEW.experiment_id
          AND selection_id <> COALESCE(NEW.selection_id, -1);
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_single_selected_model
BEFORE INSERT OR UPDATE OF selected
ON model_selection
FOR EACH ROW
EXECUTE FUNCTION enforce_single_selected_model();

CREATE OR REPLACE FUNCTION validate_experiment_state()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.state = 'completed' THEN
        IF NOT EXISTS (
            SELECT 1
            FROM model_selection
            WHERE experiment_id = NEW.experiment_id
              AND selected = TRUE
        ) THEN
            RAISE EXCEPTION
                'Experiment % cannot be completed without a selected model',
                NEW.experiment_id;
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE CONSTRAINT TRIGGER trg_completed_experiment_validation
AFTER UPDATE OF state
ON cv_experiments
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_state();

CREATE OR REPLACE FUNCTION validate_stratification()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    dataset_positive INTEGER;
    dataset_negative INTEGER;
BEGIN
    SELECT positive_count, negative_count
    INTO dataset_positive, dataset_negative
    FROM datasets d
    JOIN cv_experiments e
      ON e.dataset_id = d.dataset_id
    WHERE e.experiment_id = NEW.experiment_id;

    IF NEW.validation_samples <= 0 THEN
        RAISE EXCEPTION 'Validation fold cannot be empty';
    END IF;

    IF NEW.validation_positive < 0
       OR NEW.validation_positive > NEW.validation_samples THEN
        RAISE EXCEPTION 'Invalid validation class counts';
    END IF;

    IF NEW.training_positive < 0
       OR NEW.training_positive > NEW.training_samples THEN
        RAISE EXCEPTION 'Invalid training class counts';
    END IF;

    IF dataset_positive > 0
       AND dataset_negative > 0
       AND NEW.validation_positive IN (0, NEW.validation_samples) THEN
        RAISE EXCEPTION
            'Stratified validation fold must contain both classes';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_stratification
BEFORE INSERT OR UPDATE
ON cv_folds
FOR EACH ROW
EXECUTE FUNCTION validate_stratification();

INSERT INTO datasets (
    dataset_name,
    feature_count,
    sample_count,
    positive_count
)
VALUES (
    'customer_churn_binary_v1',
    12,
    180,
    82
);

INSERT INTO models (
    model_name,
    family,
    hyperparameters
)
VALUES
(
    'logistic_regression_l2',
    'logistic_regression',
    '{"learning_rate": 0.08, "epochs": 600, "l2": 0.02}'
),
(
    'decision_tree_depth4',
    'decision_tree',
    '{"max_depth": 4, "min_samples_split": 6}'
);

INSERT INTO cv_experiments (
    dataset_id,
    experiment_name,
    strategy,
    folds,
    repetitions,
    random_seed,
    state
)
SELECT
    dataset_id,
    'baseline_5_fold_stratified',
    'stratified_k_fold',
    5,
    1,
    42,
    'running'
FROM datasets
WHERE dataset_name = 'customer_churn_binary_v1';

INSERT INTO cv_folds (
    experiment_id,
    repetition,
    fold_number,
    training_samples,
    validation_samples,
    training_positive,
    validation_positive
)
SELECT
    experiment_id,
    1,
    fold_number,
    144,
    36,
    training_positive,
    validation_positive
FROM (
    SELECT
        experiment_id,
        generate_series(1, 5) AS fold_number,
        66 AS training_positive,
        16 AS validation_positive
    FROM cv_experiments
    WHERE experiment_name = 'baseline_5_fold_stratified'
) AS fold_source;

INSERT INTO model_evaluations (
    experiment_id,
    model_id,
    fold_id,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
SELECT
    e.experiment_id,
    m.model_id,
    f.fold_id,
    v.accuracy,
    v.precision_score,
    v.recall_score,
    v.f1_score,
    v.log_loss
FROM cv_experiments e
JOIN cv_folds f
  ON f.experiment_id = e.experiment_id
JOIN models m
  ON m.model_name = 'logistic_regression_l2'
CROSS JOIN LATERAL (
    SELECT
        0.8610::NUMERIC AS accuracy,
        0.8420::NUMERIC AS precision_score,
        0.8290::NUMERIC AS recall_score,
        0.8350::NUMERIC AS f1_score,
        0.3420::NUMERIC AS log_loss
) v
WHERE e.experiment_name = 'baseline_5_fold_stratified';

INSERT INTO model_evaluations (
    experiment_id,
    model_id,
    fold_id,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
SELECT
    e.experiment_id,
    m.model_id,
    f.fold_id,
    v.accuracy,
    v.precision_score,
    v.recall_score,
    v.f1_score,
    v.log_loss
FROM cv_experiments e
JOIN cv_folds f
  ON f.experiment_id = e.experiment_id
JOIN models m
  ON m.model_name = 'decision_tree_depth4'
CROSS JOIN LATERAL (
    SELECT
        0.8440::NUMERIC AS accuracy,
        0.8160::NUMERIC AS precision_score,
        0.8110::NUMERIC AS recall_score,
        0.8130::NUMERIC AS f1_score,
        0.3880::NUMERIC AS log_loss
) v
WHERE e.experiment_name = 'baseline_5_fold_stratified';

CREATE VIEW v_model_cv_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    m.model_id,
    m.model_name,
    m.family,
    COUNT(me.evaluation_id) AS fold_count,
    ROUND(AVG(me.accuracy), 6) AS mean_accuracy,
    ROUND(AVG(me.precision_score), 6) AS mean_precision,
    ROUND(AVG(me.recall_score), 6) AS mean_recall,
    ROUND(AVG(me.f1_score), 6) AS mean_f1,
    ROUND(STDDEV_SAMP(me.f1_score), 6) AS std_f1,
    ROUND(AVG(me.log_loss), 6) AS mean_log_loss
FROM cv_experiments e
JOIN model_evaluations me
  ON me.experiment_id = e.experiment_id
JOIN models m
  ON m.model_id = me.model_id
GROUP BY
    e.experiment_id,
    e.experiment_name,
    m.model_id,
    m.model_name,
    m.family;

CREATE VIEW v_stratification_report AS
SELECT
    e.experiment_name,
    f.repetition,
    f.fold_number,
    f.training_samples,
    f.validation_samples,
    f.training_positive,
    f.validation_positive,
    ROUND(
        f.training_positive::NUMERIC
        / NULLIF(f.training_samples, 0),
        4
    ) AS training_positive_rate,
    ROUND(
        f.validation_positive::NUMERIC
        / NULLIF(f.validation_samples, 0),
        4
    ) AS validation_positive_rate
FROM cv_experiments e
JOIN cv_folds f
  ON f.experiment_id = e.experiment_id
WHERE e.strategy = 'stratified_k_fold';

CREATE OR REPLACE FUNCTION select_best_model(
    p_experiment_id BIGINT
)
RETURNS TABLE (
    model_id BIGINT,
    model_name TEXT,
    mean_f1 NUMERIC,
    std_f1 NUMERIC,
    mean_log_loss NUMERIC
)
LANGUAGE SQL
AS $$
    SELECT
        s.model_id,
        m.model_name,
        s.mean_f1,
        s.std_f1,
        s.mean_log_loss
    FROM model_selection s
    JOIN models m
      ON m.model_id = s.model_id
    WHERE s.experiment_id = p_experiment_id
    ORDER BY
        s.mean_f1 DESC,
        s.std_f1 ASC,
        s.mean_log_loss ASC
    LIMIT 1;
$$;

INSERT INTO model_selection (
    experiment_id,
    model_id,
    mean_f1,
    std_f1,
    mean_log_loss,
    selected
)
SELECT
    summary.experiment_id,
    summary.model_id,
    summary.mean_f1,
    COALESCE(summary.std_f1, 0),
    summary.mean_log_loss,
    summary.model_name = 'logistic_regression_l2'
FROM v_model_cv_summary summary
WHERE summary.experiment_name = 'baseline_5_fold_stratified';

INSERT INTO holdout_evaluations (
    experiment_id,
    model_id,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
SELECT
    e.experiment_id,
    m.model_id,
    0.856000,
    0.838000,
    0.821000,
    0.829000,
    0.351000
FROM cv_experiments e
JOIN models m
  ON m.model_name = 'logistic_regression_l2'
WHERE e.experiment_name = 'baseline_5_fold_stratified';

UPDATE cv_experiments
SET state = 'completed'
WHERE experiment_name = 'baseline_5_fold_stratified';

SELECT
    experiment_name,
    model_name,
    family,
    fold_count,
    mean_accuracy,
    mean_precision,
    mean_recall,
    mean_f1,
    std_f1,
    mean_log_loss
FROM v_model_cv_summary
ORDER BY mean_f1 DESC, std_f1 ASC;

SELECT *
FROM v_stratification_report
ORDER BY experiment_name, repetition, fold_number;

SELECT *
FROM select_best_model(
    (
        SELECT experiment_id
        FROM cv_experiments
        WHERE experiment_name = 'baseline_5_fold_stratified'
    )
);

WITH ranked_models AS (
    SELECT
        s.experiment_id,
        s.model_id,
        m.model_name,
        s.mean_f1,
        s.std_f1,
        s.mean_log_loss,
        RANK() OVER (
            PARTITION BY s.experiment_id
            ORDER BY
                s.mean_f1 DESC,
                s.std_f1 ASC,
                s.mean_log_loss ASC
        ) AS selection_rank
    FROM model_selection s
    JOIN models m
      ON m.model_id = s.model_id
)
SELECT *
FROM ranked_models
ORDER BY experiment_id, selection_rank;

SELECT
    e.experiment_name,
    m.model_name,
    h.accuracy,
    h.precision_score,
    h.recall_score,
    h.f1_score,
    h.log_loss
FROM holdout_evaluations h
JOIN cv_experiments e
  ON e.experiment_id = h.experiment_id
JOIN models m
  ON m.model_id = h.model_id
WHERE e.state = 'completed';

BEGIN;

UPDATE cv_experiments
SET state = 'running'
WHERE experiment_name = 'baseline_5_fold_stratified'
  AND state = 'completed';

UPDATE cv_experiments
SET state = 'completed'
WHERE experiment_name = 'baseline_5_fold_stratified'
  AND EXISTS (
      SELECT 1
      FROM model_selection
      WHERE experiment_id = cv_experiments.experiment_id
        AND selected = TRUE
  );

COMMIT;
