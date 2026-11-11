/*
 * PERCEPTRON LEARNING RULE AND LINEAR SEPARABILITY
 *
 * PostgreSQL 15+
 *
 * Relational model for:
 * - Binary classification datasets.
 * - Feature vectors and labeled observations.
 * - Perceptron training runs.
 * - Epoch-by-epoch weight updates.
 * - Model parameters and bias.
 * - Predictions and evaluation metrics.
 * - Convergence and maximum-epoch termination.
 * - Linear separability experiments.
 *
 * Training uses the online perceptron learning rule:
 *
 *   error = target - prediction
 *
 *   weight_new = weight_old + learning_rate * error * feature
 *
 *   bias_new = bias_old + learning_rate * error
 *
 * All training operations are performed inside transactions.
 */

BEGIN;

CREATE SCHEMA IF NOT EXISTS perceptron_lab;

SET search_path TO perceptron_lab, public;

DROP VIEW IF EXISTS training_summary CASCADE;
DROP VIEW IF EXISTS prediction_report CASCADE;

DROP PROCEDURE IF EXISTS train_perceptron(BIGINT, INTEGER);
DROP FUNCTION IF EXISTS predict_sample(BIGINT, BIGINT);
DROP FUNCTION IF EXISTS evaluate_model(BIGINT);
DROP FUNCTION IF EXISTS activation_function(DOUBLE PRECISION);
DROP FUNCTION IF EXISTS validate_training_run();
DROP FUNCTION IF EXISTS record_epoch_update();
DROP FUNCTION IF EXISTS validate_sample_dimension();

DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS epoch_updates CASCADE;
DROP TABLE IF EXISTS model_weights CASCADE;
DROP TABLE IF EXISTS training_runs CASCADE;
DROP TABLE IF EXISTS samples CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(120) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE samples (
    sample_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    feature_values DOUBLE PRECISION[] NOT NULL,
    target_label SMALLINT NOT NULL CHECK (target_label IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_feature_vector CHECK (
        cardinality(feature_values) > 0
        AND array_ndims(feature_values) = 1
    )
);

CREATE INDEX idx_samples_dataset
    ON samples(dataset_id, sample_id);

CREATE TABLE training_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id) ON DELETE RESTRICT,

    algorithm_name VARCHAR(50) NOT NULL DEFAULT 'Perceptron',
    learning_rate DOUBLE PRECISION NOT NULL
        CHECK (learning_rate > 0 AND learning_rate < 'Infinity'::DOUBLE PRECISION),

    maximum_epochs INTEGER NOT NULL CHECK (maximum_epochs > 0),

    epochs_completed INTEGER NOT NULL DEFAULT 0
        CHECK (epochs_completed >= 0),

    model_bias DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    training_status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        CHECK (
            training_status IN (
                'PENDING',
                'RUNNING',
                'CONVERGED',
                'MAX_EPOCHS',
                'FAILED'
            )
        ),

    final_mistakes INTEGER CHECK (final_mistakes >= 0),

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_completion_time CHECK (
        completed_at IS NULL OR started_at IS NOT NULL
    )
);

CREATE INDEX idx_training_runs_dataset_status
    ON training_runs(dataset_id, training_status);

CREATE TABLE model_weights (
    run_id BIGINT NOT NULL
        REFERENCES training_runs(run_id) ON DELETE CASCADE,
    feature_index INTEGER NOT NULL CHECK (feature_index >= 1),
    weight_value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, feature_index)
);

CREATE TABLE epoch_updates (
    update_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES training_runs(run_id) ON DELETE CASCADE,

    epoch_number INTEGER NOT NULL CHECK (epoch_number > 0),
    sample_id BIGINT NOT NULL
        REFERENCES samples(sample_id) ON DELETE RESTRICT,

    predicted_label SMALLINT NOT NULL
        CHECK (predicted_label IN (0, 1)),

    target_label SMALLINT NOT NULL
        CHECK (target_label IN (0, 1)),

    prediction_error SMALLINT NOT NULL
        CHECK (prediction_error IN (-1, 0, 1)),

    updated_bias DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_sample_epoch
        UNIQUE (run_id, epoch_number, sample_id),

    CONSTRAINT consistent_error CHECK (
        prediction_error = target_label - predicted_label
    )
);

CREATE INDEX idx_epoch_updates_run_epoch
    ON epoch_updates(run_id, epoch_number);

CREATE TABLE predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES training_runs(run_id) ON DELETE CASCADE,

    sample_id BIGINT NOT NULL
        REFERENCES samples(sample_id) ON DELETE CASCADE,

    predicted_label SMALLINT NOT NULL
        CHECK (predicted_label IN (0, 1)),

    confidence_score DOUBLE PRECISION,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, sample_id),

    CHECK (
        confidence_score IS NULL OR
        confidence_score BETWEEN 0 AND 1
    )
);

CREATE OR REPLACE FUNCTION activation_function(
    input_value DOUBLE PRECISION
)
RETURNS SMALLINT
LANGUAGE SQL
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
    SELECT CASE WHEN input_value >= 0 THEN 1 ELSE 0 END::SMALLINT;
$$;

-- Ensure every observation matches its dataset's feature dimension.
CREATE OR REPLACE FUNCTION validate_sample_dimension()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_dimensions INTEGER;
BEGIN
    SELECT feature_count
    INTO expected_dimensions
    FROM datasets
    WHERE dataset_id = NEW.dataset_id;

    IF expected_dimensions IS NULL THEN
        RAISE EXCEPTION 'Dataset % does not exist', NEW.dataset_id;
    END IF;

    IF cardinality(NEW.feature_values) <> expected_dimensions THEN
        RAISE EXCEPTION
            'Expected % features, received %',
            expected_dimensions,
            cardinality(NEW.feature_values);
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(NEW.feature_values) AS feature(value)
        WHERE NOT (
            feature.value > '-Infinity'::DOUBLE PRECISION
            AND feature.value < 'Infinity'::DOUBLE PRECISION
        )
    ) THEN
        RAISE EXCEPTION 'Feature values must be finite.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_sample_dimension
BEFORE INSERT OR UPDATE ON samples
FOR EACH ROW
EXECUTE FUNCTION validate_sample_dimension();

-- Every epoch update must describe the actual classification error.
CREATE OR REPLACE FUNCTION record_epoch_update()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    actual_label SMALLINT;
BEGIN
    SELECT target_label
    INTO actual_label
    FROM samples
    WHERE sample_id = NEW.sample_id;

    IF actual_label IS NULL OR actual_label <> NEW.target_label THEN
        RAISE EXCEPTION 'Epoch target does not match sample label.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_record_epoch_update
BEFORE INSERT ON epoch_updates
FOR EACH ROW
EXECUTE FUNCTION record_epoch_update();

CREATE OR REPLACE FUNCTION validate_training_run()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.training_status = 'RUNNING'
       AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.training_status IN (
        'CONVERGED',
        'MAX_EPOCHS',
        'FAILED'
    ) THEN
        IF NEW.started_at IS NULL THEN
            RAISE EXCEPTION 'Completed runs require a start time.';
        END IF;

        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_training_run
BEFORE INSERT OR UPDATE ON training_runs
FOR EACH ROW
EXECUTE FUNCTION validate_training_run();

-- AND and XOR represent different geometric classification problems.
INSERT INTO datasets (
    dataset_name,
    description,
    feature_count
)
VALUES
(
    'AND_GATE',
    'Linearly separable binary logical AND dataset.',
    2
),
(
    'XOR_GATE',
    'Non-linearly separable binary exclusive OR dataset.',
    2
);

INSERT INTO samples (dataset_id, feature_values, target_label)
SELECT d.dataset_id, input.features, input.label
FROM datasets d
CROSS JOIN (
    VALUES
        ('AND_GATE', ARRAY[0.0, 0.0]::DOUBLE PRECISION[], 0::SMALLINT),
        ('AND_GATE', ARRAY[0.0, 1.0]::DOUBLE PRECISION[], 0::SMALLINT),
        ('AND_GATE', ARRAY[1.0, 0.0]::DOUBLE PRECISION[], 0::SMALLINT),
        ('AND_GATE', ARRAY[1.0, 1.0]::DOUBLE PRECISION[], 1::SMALLINT),

        ('XOR_GATE', ARRAY[0.0, 0.0]::DOUBLE PRECISION[], 0::SMALLINT),
        ('XOR_GATE', ARRAY[0.0, 1.0]::DOUBLE PRECISION[], 1::SMALLINT),
        ('XOR_GATE', ARRAY[1.0, 0.0]::DOUBLE PRECISION[], 1::SMALLINT),
        ('XOR_GATE', ARRAY[1.0, 1.0]::DOUBLE PRECISION[], 0::SMALLINT)
) AS input(dataset_name, features, label)
WHERE d.dataset_name = input.dataset_name;

INSERT INTO training_runs (
    dataset_id,
    learning_rate,
    maximum_epochs
)
SELECT dataset_id, 0.1, 100
FROM datasets
ORDER BY dataset_id;

-- Initialize one zero-valued weight per feature.
INSERT INTO model_weights (
    run_id,
    feature_index,
    weight_value
)
SELECT
    tr.run_id,
    dimension.index,
    0.0
FROM training_runs tr
JOIN datasets d USING (dataset_id)
CROSS JOIN LATERAL
    generate_series(1, d.feature_count) AS dimension(index);

-- The training procedure locks its run row and executes all
-- epoch updates within the calling transaction.
CREATE OR REPLACE PROCEDURE train_perceptron(
    p_run_id BIGINT,
    p_maximum_epochs INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_dataset_id BIGINT;
    v_learning_rate DOUBLE PRECISION;
    v_epoch INTEGER;
    v_mistakes INTEGER;
    v_prediction SMALLINT;
    v_error INTEGER;
    v_bias DOUBLE PRECISION;
    v_features DOUBLE PRECISION[];
    v_sample RECORD;
    v_weight DOUBLE PRECISION;
    v_feature_index INTEGER;
BEGIN
    IF p_maximum_epochs < 1 THEN
        RAISE EXCEPTION 'Maximum epochs must be positive.';
    END IF;

    SELECT dataset_id, learning_rate, model_bias
    INTO v_dataset_id, v_learning_rate, v_bias
    FROM training_runs
    WHERE run_id = p_run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run % does not exist.', p_run_id;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM model_weights WHERE run_id = p_run_id
    ) THEN
        RAISE EXCEPTION 'Model weights have not been initialized.';
    END IF;

    IF (
        SELECT COUNT(DISTINCT target_label)
        FROM samples
        WHERE dataset_id = v_dataset_id
    ) <> 2 THEN
        RAISE EXCEPTION 'Training requires both binary classes.';
    END IF;

    UPDATE training_runs
    SET training_status = 'RUNNING',
        epochs_completed = 0,
        final_mistakes = NULL,
        model_bias = 0.0,
        completed_at = NULL
    WHERE run_id = p_run_id;

    UPDATE model_weights
    SET weight_value = 0.0
    WHERE run_id = p_run_id;

    v_bias := 0.0;

    FOR v_epoch IN 1..p_maximum_epochs LOOP
        v_mistakes := 0;

        FOR v_sample IN
            SELECT sample_id, feature_values, target_label
            FROM samples
            WHERE dataset_id = v_dataset_id
            ORDER BY sample_id
        LOOP
            v_features := v_sample.feature_values;

            SELECT
                activation_function(
                    COALESCE(SUM(
                        mw.weight_value * feature.value
                    ), 0.0) + v_bias
                )
            INTO v_prediction
            FROM model_weights mw
            CROSS JOIN LATERAL
                unnest(v_features) WITH ORDINALITY
                AS feature(value, position)
            WHERE mw.run_id = p_run_id
              AND mw.feature_index = feature.position;

            v_error := v_sample.target_label - v_prediction;

            IF v_error <> 0 THEN
                v_mistakes := v_mistakes + 1;

                UPDATE model_weights mw
                SET weight_value =
                    mw.weight_value +
                    v_learning_rate *
                    v_error *
                    feature.value
                FROM unnest(v_features) WITH ORDINALITY
                    AS feature(value, position)
                WHERE mw.run_id = p_run_id
                  AND mw.feature_index = feature.position;

                v_bias := v_bias + v_learning_rate * v_error;
            END IF;

            INSERT INTO epoch_updates (
                run_id,
                epoch_number,
                sample_id,
                predicted_label,
                target_label,
                prediction_error,
                updated_bias
            )
            VALUES (
                p_run_id,
                v_epoch,
                v_sample.sample_id,
                v_prediction,
                v_sample.target_label,
                v_error,
                v_bias
            );
        END LOOP;

        UPDATE training_runs
        SET epochs_completed = v_epoch,
            model_bias = v_bias,
            final_mistakes = v_mistakes
        WHERE run_id = p_run_id;

        IF v_mistakes = 0 THEN
            UPDATE training_runs
            SET training_status = 'CONVERGED'
            WHERE run_id = p_run_id;

            RETURN;
        END IF;
    END LOOP;

    UPDATE training_runs
    SET training_status = 'MAX_EPOCHS',
        model_bias = v_bias
    WHERE run_id = p_run_id;
END;
$$;

CREATE OR REPLACE FUNCTION predict_sample(
    p_run_id BIGINT,
    p_sample_id BIGINT
)
RETURNS SMALLINT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_result DOUBLE PRECISION;
    v_features DOUBLE PRECISION[];
    v_bias DOUBLE PRECISION;
    v_dataset_id BIGINT;
BEGIN
    SELECT dataset_id, model_bias
    INTO v_dataset_id, v_bias
    FROM training_runs
    WHERE run_id = p_run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run does not exist.';
    END IF;

    SELECT feature_values
    INTO v_features
    FROM samples
    WHERE sample_id = p_sample_id
      AND dataset_id = v_dataset_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Sample does not belong to training dataset.';
    END IF;

    SELECT COALESCE(SUM(
        mw.weight_value * feature.value
    ), 0.0) + v_bias
    INTO v_result
    FROM model_weights mw
    CROSS JOIN LATERAL
        unnest(v_features) WITH ORDINALITY
        AS feature(value, position)
    WHERE mw.run_id = p_run_id
      AND mw.feature_index = feature.position;

    RETURN activation_function(v_result);
END;
$$;

CREATE OR REPLACE FUNCTION evaluate_model(
    p_run_id BIGINT
)
RETURNS TABLE (
    true_positive BIGINT,
    true_negative BIGINT,
    false_positive BIGINT,
    false_negative BIGINT,
    accuracy NUMERIC
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM training_runs
        WHERE run_id = p_run_id
    ) THEN
        RAISE EXCEPTION 'Training run does not exist.';
    END IF;

    RETURN QUERY
    WITH classification AS (
        SELECT
            s.target_label,
            predict_sample(p_run_id, s.sample_id) AS prediction
        FROM samples s
        JOIN training_runs tr USING (dataset_id)
        WHERE tr.run_id = p_run_id
    ),
    counts AS (
        SELECT
            COUNT(*) FILTER (
                WHERE target_label = 1 AND prediction = 1
            ) AS tp,
            COUNT(*) FILTER (
                WHERE target_label = 0 AND prediction = 0
            ) AS tn,
            COUNT(*) FILTER (
                WHERE target_label = 0 AND prediction = 1
            ) AS fp,
            COUNT(*) FILTER (
                WHERE target_label = 1 AND prediction = 0
            ) AS fn
        FROM classification
    )
    SELECT
        tp,
        tn,
        fp,
        fn,
        ROUND(
            (tp + tn)::NUMERIC /
            NULLIF(tp + tn + fp + fn, 0),
            4
        )
    FROM counts;
END;
$$;

CREATE VIEW prediction_report AS
SELECT
    tr.run_id,
    d.dataset_name,
    s.sample_id,
    s.feature_values,
    s.target_label,
    predict_sample(tr.run_id, s.sample_id) AS predicted_label,
    (
        s.target_label =
        predict_sample(tr.run_id, s.sample_id)
    ) AS correct
FROM training_runs tr
JOIN datasets d USING (dataset_id)
JOIN samples s USING (dataset_id);

CREATE VIEW training_summary AS
SELECT
    tr.run_id,
    d.dataset_name,
    tr.algorithm_name,
    tr.learning_rate,
    tr.epochs_completed,
    tr.maximum_epochs,
    tr.final_mistakes,
    tr.training_status,
    tr.model_bias,
    tr.started_at,
    tr.completed_at
FROM training_runs tr
JOIN datasets d USING (dataset_id);

-- Execute each training run independently.
CALL train_perceptron(
    (SELECT run_id
     FROM training_runs tr
     JOIN datasets d USING (dataset_id)
     WHERE d.dataset_name = 'AND_GATE'),
    100
);

CALL train_perceptron(
    (SELECT run_id
     FROM training_runs tr
     JOIN datasets d USING (dataset_id)
     WHERE d.dataset_name = 'XOR_GATE'),
    100
);

-- Populate persisted predictions for reporting.
INSERT INTO predictions (
    run_id,
    sample_id,
    predicted_label
)
SELECT
    run_id,
    sample_id,
    predict_sample(run_id, sample_id)
FROM prediction_report
ON CONFLICT (run_id, sample_id)
DO UPDATE SET
    predicted_label = EXCLUDED.predicted_label,
    evaluated_at = CURRENT_TIMESTAMP;

-- Training-state retrieval.
SELECT * FROM training_summary ORDER BY run_id;

-- Model weights.
SELECT
    run_id,
    feature_index,
    weight_value
FROM model_weights
ORDER BY run_id, feature_index;

-- Confusion matrix and accuracy.
SELECT
    tr.run_id,
    d.dataset_name,
    metrics.*
FROM training_runs tr
JOIN datasets d USING (dataset_id)
CROSS JOIN LATERAL evaluate_model(tr.run_id) AS metrics
ORDER BY tr.run_id;

-- Sample-level prediction analysis.
SELECT *
FROM prediction_report
ORDER BY run_id, sample_id;

-- Aggregate training mistakes by epoch.
SELECT
    run_id,
    epoch_number,
    COUNT(*) FILTER (
        WHERE prediction_error <> 0
    ) AS mistakes,
    COUNT(*) AS observations
FROM epoch_updates
GROUP BY run_id, epoch_number
ORDER BY run_id, epoch_number;

-- Eligibility is based on actual training convergence.
SELECT
    tr.run_id,
    d.dataset_name,
    tr.training_status,
    CASE
        WHEN tr.training_status = 'CONVERGED'
             AND tr.final_mistakes = 0
        THEN TRUE
        ELSE FALSE
    END AS converged_within_budget
FROM training_runs tr
JOIN datasets d USING (dataset_id)
ORDER BY tr.run_id;

-- Detect samples whose feature dimensions do not match the dataset.
SELECT
    s.sample_id,
    d.dataset_name,
    cardinality(s.feature_values) AS actual_dimensions,
    d.feature_count AS expected_dimensions
FROM samples s
JOIN datasets d USING (dataset_id)
WHERE cardinality(s.feature_values) <> d.feature_count;

-- Transactional integrity demonstration.
-- A failed insert is isolated by a savepoint.
SAVEPOINT invalid_sample_test;

DO $$
BEGIN
    BEGIN
        INSERT INTO samples (
            dataset_id,
            feature_values,
            target_label
        )
        SELECT dataset_id, ARRAY[1.0], 1
        FROM datasets
        WHERE dataset_name = 'AND_GATE';

        RAISE EXCEPTION 'Invalid sample unexpectedly accepted.';
    EXCEPTION
        WHEN raise_exception THEN
            RAISE NOTICE 'Invalid feature dimension rejected.';
    END;
END;
$$;

ROLLBACK TO SAVEPOINT invalid_sample_test;
RELEASE SAVEPOINT invalid_sample_test;

COMMIT;
