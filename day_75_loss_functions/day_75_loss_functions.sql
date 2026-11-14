/*
===============================================================================
LOSS FUNCTION EVALUATION AND MODEL GOVERNANCE DATABASE
PostgreSQL 14+

Implements:
- Regression loss evaluation: MSE and MAE
- Classification loss evaluation: BCE and CCE
- Model and experiment management
- Training runs and predictions
- Numerical input validation
- Loss calculation functions
- Evaluation policies
- Threshold-based acceptance
- Evaluation audit history
- Triggers and reporting views
===============================================================================
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS ml_governance;

SET search_path TO ml_governance, public;

-- ============================================================================
-- DOMAIN TYPES
-- ============================================================================

CREATE TYPE loss_type AS ENUM (
    'MSE',
    'MAE',
    'BINARY_CROSS_ENTROPY',
    'CATEGORICAL_CROSS_ENTROPY'
);

CREATE TYPE task_type AS ENUM (
    'REGRESSION',
    'BINARY_CLASSIFICATION',
    'MULTICLASS_CLASSIFICATION'
);

CREATE TYPE run_status AS ENUM (
    'CREATED',
    'TRAINING',
    'COMPLETED',
    'FAILED'
);

CREATE TYPE evaluation_status AS ENUM (
    'PENDING',
    'ACCEPTED',
    'REJECTED'
);

CREATE TYPE reduction_type AS ENUM (
    'MEAN',
    'SUM'
);

-- ============================================================================
-- REPOSITORY OF MODELS AND EXPERIMENTS
-- ============================================================================

CREATE TABLE models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name VARCHAR(150) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    task task_type NOT NULL,
    owner_name VARCHAR(120) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN NOT NULL DEFAULT TRUE,

    UNIQUE (model_name, model_version)
);

CREATE TABLE loss_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name VARCHAR(150) NOT NULL UNIQUE,

    maximum_mse DOUBLE PRECISION,
    maximum_mae DOUBLE PRECISION,
    maximum_bce DOUBLE PRECISION,
    maximum_cce DOUBLE PRECISION,

    minimum_samples INTEGER NOT NULL DEFAULT 1,

    require_complete_predictions BOOLEAN NOT NULL DEFAULT TRUE,
    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (
        maximum_mse IS NULL OR
        (maximum_mse >= 0 AND maximum_mse < 'Infinity'::float8)
    ),

    CHECK (
        maximum_mae IS NULL OR
        (maximum_mae >= 0 AND maximum_mae < 'Infinity'::float8)
    ),

    CHECK (
        maximum_bce IS NULL OR
        (maximum_bce >= 0 AND maximum_bce < 'Infinity'::float8)
    ),

    CHECK (
        maximum_cce IS NULL OR
        (maximum_cce >= 0 AND maximum_cce < 'Infinity'::float8)
    ),

    CHECK (minimum_samples > 0)
);

CREATE TABLE training_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL REFERENCES models(model_id),

    run_name VARCHAR(150) NOT NULL,
    status run_status NOT NULL DEFAULT 'CREATED',

    dataset_name VARCHAR(200) NOT NULL,
    random_seed INTEGER NOT NULL,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (model_id, run_name),

    CHECK (
        completed_at IS NULL OR
        started_at IS NULL OR
        completed_at >= started_at
    )
);

-- ============================================================================
-- PREDICTION DATA
-- ============================================================================
-- Each row represents one sample in one training run.
-- Regression uses actual_value and predicted_value.
-- Binary classification uses binary_label and predicted_probability.
-- Multiclass classification uses target_class and probability_array.

CREATE TABLE predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL REFERENCES training_runs(run_id),

    sample_reference VARCHAR(150) NOT NULL,

    actual_value DOUBLE PRECISION,
    predicted_value DOUBLE PRECISION,

    binary_label SMALLINT,
    predicted_probability DOUBLE PRECISION,

    target_class INTEGER,
    probability_array DOUBLE PRECISION[],

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, sample_reference),

    CHECK (
        actual_value IS NULL OR
        abs(actual_value) < 'Infinity'::float8
    ),

    CHECK (
        predicted_value IS NULL OR
        abs(predicted_value) < 'Infinity'::float8
    ),

    CHECK (
        binary_label IS NULL OR
        binary_label IN (0, 1)
    ),

    CHECK (
        predicted_probability IS NULL OR
        (
            predicted_probability >= 0 AND
            predicted_probability <= 1
        )
    ),

    CHECK (target_class IS NULL OR target_class >= 0),

    CHECK (
        probability_array IS NULL OR
        cardinality(probability_array) >= 2
    )
);

-- ============================================================================
-- EVALUATION RECORDS
-- ============================================================================

CREATE TABLE loss_evaluations (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL REFERENCES training_runs(run_id),

    policy_id BIGINT NOT NULL REFERENCES loss_policies(policy_id),

    metric loss_type NOT NULL,

    reduction reduction_type NOT NULL DEFAULT 'MEAN',

    sample_count INTEGER NOT NULL,

    loss_value DOUBLE PRECISION,

    threshold_value DOUBLE PRECISION,

    status evaluation_status NOT NULL DEFAULT 'PENDING',

    evaluated_by VARCHAR(120) NOT NULL,

    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, policy_id, metric),

    CHECK (sample_count > 0),

    CHECK (
        loss_value IS NULL OR
        (
            loss_value >= 0 AND
            loss_value < 'Infinity'::float8
        )
    ),

    CHECK (
        threshold_value IS NULL OR
        (
            threshold_value >= 0 AND
            threshold_value < 'Infinity'::float8
        )
    )
);

CREATE TABLE evaluation_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    evaluation_id BIGINT NOT NULL
        REFERENCES loss_evaluations(evaluation_id),

    old_status evaluation_status,
    new_status evaluation_status NOT NULL,

    event_description TEXT NOT NULL,

    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX idx_runs_model_status
ON training_runs(model_id, status);

CREATE INDEX idx_predictions_run
ON predictions(run_id);

CREATE INDEX idx_evaluations_run_status
ON loss_evaluations(run_id, status);

CREATE INDEX idx_evaluations_metric
ON loss_evaluations(metric, evaluated_at DESC);

CREATE INDEX idx_events_evaluation
ON evaluation_events(evaluation_id, occurred_at DESC);

-- ============================================================================
-- VALIDATION FUNCTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION validate_finite_array(
    values DOUBLE PRECISION[]
)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT COALESCE(
        bool_and(
            value IS NOT NULL
            AND value >= '-Infinity'::float8
            AND value <= 'Infinity'::float8
            AND value <> 'NaN'::float8
        ),
        FALSE
    )
    FROM unnest(values) AS value;
$$;

CREATE OR REPLACE FUNCTION validate_probability_array(
    probabilities DOUBLE PRECISION[]
)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT
        cardinality(probabilities) >= 2
        AND validate_finite_array(probabilities)
        AND COALESCE(
            bool_and(value BETWEEN 0 AND 1),
            FALSE
        )
        AND abs(sum(value) - 1.0) <= 0.00000001
    FROM unnest(probabilities) AS value;
$$;

-- ============================================================================
-- TASK-SPECIFIC PREDICTION VALIDATION
-- ============================================================================

CREATE OR REPLACE FUNCTION enforce_prediction_task()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    selected_task task_type;
BEGIN
    SELECT m.task
    INTO selected_task
    FROM training_runs r
    JOIN models m ON m.model_id = r.model_id
    WHERE r.run_id = NEW.run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run does not exist.';
    END IF;

    IF selected_task = 'REGRESSION' THEN

        IF NEW.actual_value IS NULL
           OR NEW.predicted_value IS NULL THEN
            RAISE EXCEPTION
                'Regression predictions require actual and predicted values.';
        END IF;

        IF NEW.binary_label IS NOT NULL
           OR NEW.target_class IS NOT NULL
           OR NEW.probability_array IS NOT NULL
           OR NEW.predicted_probability IS NOT NULL THEN
            RAISE EXCEPTION
                'Classification fields are not allowed for regression.';
        END IF;

    ELSIF selected_task = 'BINARY_CLASSIFICATION' THEN

        IF NEW.binary_label IS NULL
           OR NEW.predicted_probability IS NULL THEN
            RAISE EXCEPTION
                'Binary classification requires label and probability.';
        END IF;

        IF NEW.actual_value IS NOT NULL
           OR NEW.predicted_value IS NOT NULL
           OR NEW.target_class IS NOT NULL
           OR NEW.probability_array IS NOT NULL THEN
            RAISE EXCEPTION
                'Unexpected fields for binary classification.';
        END IF;

    ELSIF selected_task = 'MULTICLASS_CLASSIFICATION' THEN

        IF NEW.target_class IS NULL
           OR NEW.probability_array IS NULL THEN
            RAISE EXCEPTION
                'Multiclass prediction requires class and probabilities.';
        END IF;

        IF NEW.target_class >= cardinality(NEW.probability_array) THEN
            RAISE EXCEPTION 'Target class is outside probability array.';
        END IF;

        IF NOT validate_probability_array(NEW.probability_array) THEN
            RAISE EXCEPTION
                'Multiclass probabilities must be finite, valid and sum to one.';
        END IF;

        IF NEW.actual_value IS NOT NULL
           OR NEW.predicted_value IS NOT NULL
           OR NEW.binary_label IS NOT NULL
           OR NEW.predicted_probability IS NOT NULL THEN
            RAISE EXCEPTION
                'Unexpected fields for multiclass classification.';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_prediction
BEFORE INSERT OR UPDATE ON predictions
FOR EACH ROW
EXECUTE FUNCTION enforce_prediction_task();

-- ============================================================================
-- RUN STATE TRANSITION VALIDATION
-- ============================================================================

CREATE OR REPLACE FUNCTION enforce_run_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'COMPLETED' AND NEW.status <> 'COMPLETED' THEN
        RAISE EXCEPTION 'Completed runs cannot change status.';
    END IF;

    IF OLD.status = 'FAILED' AND NEW.status <> 'FAILED' THEN
        RAISE EXCEPTION 'Failed runs are terminal.';
    END IF;

    IF OLD.status = 'CREATED'
       AND NEW.status NOT IN ('CREATED', 'TRAINING', 'FAILED') THEN
        RAISE EXCEPTION 'Invalid transition from CREATED.';
    END IF;

    IF OLD.status = 'TRAINING'
       AND NEW.status NOT IN ('TRAINING', 'COMPLETED', 'FAILED') THEN
        RAISE EXCEPTION 'Invalid transition from TRAINING.';
    END IF;

    IF NEW.status = 'TRAINING' AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN ('COMPLETED', 'FAILED')
       AND NEW.completed_at IS NULL THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_run_transition
BEFORE UPDATE ON training_runs
FOR EACH ROW
EXECUTE FUNCTION enforce_run_transition();

-- ============================================================================
-- LOSS CALCULATION FUNCTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION calculate_mse(
    actual_values DOUBLE PRECISION[],
    predicted_values DOUBLE PRECISION[],
    reduction reduction_type DEFAULT 'MEAN'
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result DOUBLE PRECISION;
BEGIN
    IF actual_values IS NULL
       OR predicted_values IS NULL
       OR cardinality(actual_values) = 0
       OR cardinality(actual_values) <> cardinality(predicted_values) THEN
        RAISE EXCEPTION 'Invalid MSE dimensions.';
    END IF;

    IF NOT validate_finite_array(actual_values)
       OR NOT validate_finite_array(predicted_values) THEN
        RAISE EXCEPTION 'MSE inputs must be finite.';
    END IF;

    SELECT sum(power(a - p, 2))
    INTO result
    FROM unnest(actual_values, predicted_values) AS t(a, p);

    IF reduction = 'MEAN' THEN
        result := result / cardinality(actual_values);
    END IF;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION calculate_mae(
    actual_values DOUBLE PRECISION[],
    predicted_values DOUBLE PRECISION[],
    reduction reduction_type DEFAULT 'MEAN'
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result DOUBLE PRECISION;
BEGIN
    IF actual_values IS NULL
       OR predicted_values IS NULL
       OR cardinality(actual_values) = 0
       OR cardinality(actual_values) <> cardinality(predicted_values) THEN
        RAISE EXCEPTION 'Invalid MAE dimensions.';
    END IF;

    IF NOT validate_finite_array(actual_values)
       OR NOT validate_finite_array(predicted_values) THEN
        RAISE EXCEPTION 'MAE inputs must be finite.';
    END IF;

    SELECT sum(abs(a - p))
    INTO result
    FROM unnest(actual_values, predicted_values) AS t(a, p);

    IF reduction = 'MEAN' THEN
        result := result / cardinality(actual_values);
    END IF;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION calculate_binary_cross_entropy(
    actual_labels SMALLINT[],
    predicted_probabilities DOUBLE PRECISION[],
    reduction reduction_type DEFAULT 'MEAN'
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result DOUBLE PRECISION;
BEGIN
    IF actual_labels IS NULL
       OR predicted_probabilities IS NULL
       OR cardinality(actual_labels) = 0
       OR cardinality(actual_labels) <> cardinality(predicted_probabilities) THEN
        RAISE EXCEPTION 'Invalid BCE dimensions.';
    END IF;

    IF NOT validate_finite_array(predicted_probabilities) THEN
        RAISE EXCEPTION 'BCE probabilities must be finite.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(actual_labels) AS label
        WHERE label NOT IN (0, 1)
    ) THEN
        RAISE EXCEPTION 'Binary labels must be 0 or 1.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(predicted_probabilities) AS p
        WHERE p < 0 OR p > 1
    ) THEN
        RAISE EXCEPTION 'Invalid binary probability.';
    END IF;

    SELECT sum(
        -(
            label * ln(greatest(p, 0.000000000001))
            +
            (1 - label) * ln(greatest(1 - p, 0.000000000001))
        )
    )
    INTO result
    FROM unnest(
        actual_labels,
        predicted_probabilities
    ) AS t(label, p);

    IF reduction = 'MEAN' THEN
        result := result / cardinality(actual_labels);
    END IF;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION calculate_categorical_cross_entropy(
    actual_classes INTEGER[],
    probability_rows DOUBLE PRECISION[][],
    reduction reduction_type DEFAULT 'MEAN'
)
RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    sample_index INTEGER;
    class_index INTEGER;
    probabilities DOUBLE PRECISION[];
    probability_sum DOUBLE PRECISION;
    total_loss DOUBLE PRECISION := 0;
    sample_count INTEGER;
BEGIN
    sample_count := cardinality(actual_classes);

    IF actual_classes IS NULL
       OR probability_rows IS NULL
       OR sample_count = 0
       OR cardinality(probability_rows) <> sample_count THEN
        RAISE EXCEPTION 'Invalid CCE dimensions.';
    END IF;

    FOR sample_index IN 1..sample_count LOOP

        probabilities := probability_rows[sample_index];

        IF probabilities IS NULL
           OR cardinality(probabilities) < 2 THEN
            RAISE EXCEPTION 'At least two class probabilities required.';
        END IF;

        class_index := actual_classes[sample_index] + 1;

        IF class_index < 1
           OR class_index > cardinality(probabilities) THEN
            RAISE EXCEPTION 'Invalid target class.';
        END IF;

        IF NOT validate_finite_array(probabilities) THEN
            RAISE EXCEPTION 'Non-finite class probability.';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM unnest(probabilities) AS p
            WHERE p < 0 OR p > 1
        ) THEN
            RAISE EXCEPTION 'Invalid class probability.';
        END IF;

        SELECT sum(p)
        INTO probability_sum
        FROM unnest(probabilities) AS p;

        IF abs(probability_sum - 1.0) > 0.00000001 THEN
            RAISE EXCEPTION 'Class probabilities must sum to one.';
        END IF;

        total_loss := total_loss -
            ln(greatest(probabilities[class_index], 0.000000000001));
    END LOOP;

    IF reduction = 'MEAN' THEN
        total_loss := total_loss / sample_count;
    END IF;

    RETURN total_loss;
END;
$$;

-- ============================================================================
-- LOSS EVALUATION PROCEDURE
-- ============================================================================

CREATE OR REPLACE PROCEDURE evaluate_training_run(
    p_run_id BIGINT,
    p_policy_id BIGINT,
    p_evaluator VARCHAR(120)
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_task task_type;
    v_status run_status;
    v_minimum_samples INTEGER;
    v_require_complete BOOLEAN;

    v_metric loss_type;
    v_threshold DOUBLE PRECISION;
    v_loss DOUBLE PRECISION;
    v_sample_count INTEGER;

    v_actual DOUBLE PRECISION[];
    v_predicted DOUBLE PRECISION[];

    v_labels SMALLINT[];
    v_probabilities DOUBLE PRECISION[];

    v_classes INTEGER[];
    v_probability_rows DOUBLE PRECISION[][];

    v_policy RECORD;
    v_existing_id BIGINT;
BEGIN
    IF p_evaluator IS NULL OR length(trim(p_evaluator)) = 0 THEN
        RAISE EXCEPTION 'Evaluator identity is required.';
    END IF;

    SELECT m.task, r.status
    INTO v_task, v_status
    FROM training_runs r
    JOIN models m ON m.model_id = r.model_id
    WHERE r.run_id = p_run_id
    FOR UPDATE OF r;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run not found.';
    END IF;

    IF v_status <> 'COMPLETED' THEN
        RAISE EXCEPTION 'Only completed runs can be evaluated.';
    END IF;

    SELECT *
    INTO v_policy
    FROM loss_policies
    WHERE policy_id = p_policy_id
      AND active = TRUE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Active policy not found.';
    END IF;

    v_minimum_samples := v_policy.minimum_samples;
    v_require_complete := v_policy.require_complete_predictions;

    SELECT count(*)
    INTO v_sample_count
    FROM predictions
    WHERE run_id = p_run_id;

    IF v_sample_count < v_minimum_samples THEN
        RAISE EXCEPTION
            'Insufficient samples: found %, required %.',
            v_sample_count,
            v_minimum_samples;
    END IF;

    -- Regression evaluation.
    IF v_task = 'REGRESSION' THEN

        v_metric := 'MSE';
        v_threshold := v_policy.maximum_mse;

        IF v_threshold IS NOT NULL THEN

            IF v_require_complete AND EXISTS (
                SELECT 1
                FROM predictions
                WHERE run_id = p_run_id
                  AND (
                      actual_value IS NULL OR
                      predicted_value IS NULL
                  )
            ) THEN
                RAISE EXCEPTION 'Incomplete regression predictions.';
            END IF;

            SELECT
                array_agg(actual_value ORDER BY prediction_id),
                array_agg(predicted_value ORDER BY prediction_id)
            INTO v_actual, v_predicted
            FROM predictions
            WHERE run_id = p_run_id
              AND actual_value IS NOT NULL
              AND predicted_value IS NOT NULL;

            v_loss := calculate_mse(v_actual, v_predicted);

            INSERT INTO loss_evaluations (
                run_id,
                policy_id,
                metric,
                sample_count,
                loss_value,
                threshold_value,
                status,
                evaluated_by
            )
            VALUES (
                p_run_id,
                p_policy_id,
                'MSE',
                cardinality(v_actual),
                v_loss,
                v_threshold,
                CASE
                    WHEN v_loss <= v_threshold THEN 'ACCEPTED'
                    ELSE 'REJECTED'
                END,
                p_evaluator
            )
            ON CONFLICT (run_id, policy_id, metric)
            DO UPDATE SET
                sample_count = EXCLUDED.sample_count,
                loss_value = EXCLUDED.loss_value,
                threshold_value = EXCLUDED.threshold_value,
                status = EXCLUDED.status,
                evaluated_by = EXCLUDED.evaluated_by,
                evaluated_at = CURRENT_TIMESTAMP;

        END IF;

        v_metric := 'MAE';
        v_threshold := v_policy.maximum_mae;

        IF v_threshold IS NOT NULL THEN

            v_loss := calculate_mae(v_actual, v_predicted);

            INSERT INTO loss_evaluations (
                run_id,
                policy_id,
                metric,
                sample_count,
                loss_value,
                threshold_value,
                status,
                evaluated_by
            )
            VALUES (
                p_run_id,
                p_policy_id,
                'MAE',
                cardinality(v_actual),
                v_loss,
                v_threshold,
                CASE
                    WHEN v_loss <= v_threshold THEN 'ACCEPTED'
                    ELSE 'REJECTED'
                END,
                p_evaluator
            )
            ON CONFLICT (run_id, policy_id, metric)
            DO UPDATE SET
                sample_count = EXCLUDED.sample_count,
                loss_value = EXCLUDED.loss_value,
                threshold_value = EXCLUDED.threshold_value,
                status = EXCLUDED.status,
                evaluated_by = EXCLUDED.evaluated_by,
                evaluated_at = CURRENT_TIMESTAMP;
        END IF;

    -- Binary classification evaluation.
    ELSIF v_task = 'BINARY_CLASSIFICATION' THEN

        v_metric := 'BINARY_CROSS_ENTROPY';
        v_threshold := v_policy.maximum_bce;

        IF v_threshold IS NOT NULL THEN

            IF v_require_complete AND EXISTS (
                SELECT 1
                FROM predictions
                WHERE run_id = p_run_id
                  AND (
                      binary_label IS NULL OR
                      predicted_probability IS NULL
                  )
            ) THEN
                RAISE EXCEPTION 'Incomplete binary predictions.';
            END IF;

            SELECT
                array_agg(binary_label ORDER BY prediction_id),
                array_agg(
                    predicted_probability ORDER BY prediction_id
                )
            INTO v_labels, v_probabilities
            FROM predictions
            WHERE run_id = p_run_id
              AND binary_label IS NOT NULL
              AND predicted_probability IS NOT NULL;

            v_loss := calculate_binary_cross_entropy(
                v_labels,
                v_probabilities
            );

            INSERT INTO loss_evaluations (
                run_id,
                policy_id,
                metric,
                sample_count,
                loss_value,
                threshold_value,
                status,
                evaluated_by
            )
            VALUES (
                p_run_id,
                p_policy_id,
                'BINARY_CROSS_ENTROPY',
                cardinality(v_labels),
                v_loss,
                v_threshold,
                CASE
                    WHEN v_loss <= v_threshold THEN 'ACCEPTED'
                    ELSE 'REJECTED'
                END,
                p_evaluator
            )
            ON CONFLICT (run_id, policy_id, metric)
            DO UPDATE SET
                sample_count = EXCLUDED.sample_count,
                loss_value = EXCLUDED.loss_value,
                threshold_value = EXCLUDED.threshold_value,
                status = EXCLUDED.status,
                evaluated_by = EXCLUDED.evaluated_by,
                evaluated_at = CURRENT_TIMESTAMP;
        END IF;

    -- Multiclass classification evaluation.
    ELSIF v_task = 'MULTICLASS_CLASSIFICATION' THEN

        v_metric := 'CATEGORICAL_CROSS_ENTROPY';
        v_threshold := v_policy.maximum_cce;

        IF v_threshold IS NOT NULL THEN

            IF v_require_complete AND EXISTS (
                SELECT 1
                FROM predictions
                WHERE run_id = p_run_id
                  AND (
                      target_class IS NULL OR
                      probability_array IS NULL
                  )
            ) THEN
                RAISE EXCEPTION 'Incomplete multiclass predictions.';
            END IF;

            SELECT
                array_agg(target_class ORDER BY prediction_id),
                array_agg(
                    probability_array ORDER BY prediction_id
                )
            INTO v_classes, v_probability_rows
            FROM predictions
            WHERE run_id = p_run_id
              AND target_class IS NOT NULL
              AND probability_array IS NOT NULL;

            v_loss := calculate_categorical_cross_entropy(
                v_classes,
                v_probability_rows
            );

            INSERT INTO loss_evaluations (
                run_id,
                policy_id,
                metric,
                sample_count,
                loss_value,
                threshold_value,
                status,
                evaluated_by
            )
            VALUES (
                p_run_id,
                p_policy_id,
                'CATEGORICAL_CROSS_ENTROPY',
                cardinality(v_classes),
                v_loss,
                v_threshold,
                CASE
                    WHEN v_loss <= v_threshold THEN 'ACCEPTED'
                    ELSE 'REJECTED'
                END,
                p_evaluator
            )
            ON CONFLICT (run_id, policy_id, metric)
            DO UPDATE SET
                sample_count = EXCLUDED.sample_count,
                loss_value = EXCLUDED.loss_value,
                threshold_value = EXCLUDED.threshold_value,
                status = EXCLUDED.status,
                evaluated_by = EXCLUDED.evaluated_by,
                evaluated_at = CURRENT_TIMESTAMP;
        END IF;
    END IF;

    -- Prevent a policy from silently evaluating no metrics.
    IF NOT EXISTS (
        SELECT 1
        FROM loss_evaluations
        WHERE run_id = p_run_id
          AND policy_id = p_policy_id
    ) THEN
        RAISE EXCEPTION 'Policy contains no applicable loss threshold.';
    END IF;
END;
$$;

-- ============================================================================
-- AUDIT TRIGGER
-- ============================================================================

CREATE OR REPLACE FUNCTION audit_evaluation_status()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN

        INSERT INTO evaluation_events (
            evaluation_id,
            old_status,
            new_status,
            event_description
        )
        VALUES (
            NEW.evaluation_id,
            NULL,
            NEW.status,
            'Evaluation created.'
        );

    ELSIF OLD.status IS DISTINCT FROM NEW.status THEN

        INSERT INTO evaluation_events (
            evaluation_id,
            old_status,
            new_status,
            event_description
        )
        VALUES (
            NEW.evaluation_id,
            OLD.status,
            NEW.status,
            'Evaluation status changed.'
        );

    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_evaluation_audit
AFTER INSERT OR UPDATE ON loss_evaluations
FOR EACH ROW
EXECUTE FUNCTION audit_evaluation_status();

-- ============================================================================
-- REPORTING VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_model_loss_report AS
SELECT
    m.model_name,
    m.model_version,
    m.task,
    r.run_name,
    r.status AS run_status,
    e.metric,
    e.sample_count,
    e.loss_value,
    e.threshold_value,
    e.status AS evaluation_status,
    e.evaluated_at
FROM models m
JOIN training_runs r ON r.model_id = m.model_id
LEFT JOIN loss_evaluations e ON e.run_id = r.run_id;

CREATE OR REPLACE VIEW v_policy_performance AS
SELECT
    p.policy_name,
    e.metric,
    count(*) AS evaluation_count,
    count(*) FILTER (
        WHERE e.status = 'ACCEPTED'
    ) AS accepted_count,
    count(*) FILTER (
        WHERE e.status = 'REJECTED'
    ) AS rejected_count,
    round(avg(e.loss_value)::numeric, 6) AS average_loss,
    round(max(e.loss_value)::numeric, 6) AS maximum_loss
FROM loss_policies p
JOIN loss_evaluations e ON e.policy_id = p.policy_id
GROUP BY p.policy_name, e.metric;

CREATE OR REPLACE VIEW v_run_eligibility AS
SELECT
    r.run_id,
    r.run_name,
    m.model_name,
    p.policy_name,
    count(e.evaluation_id) AS evaluated_metrics,
    count(e.evaluation_id) FILTER (
        WHERE e.status = 'ACCEPTED'
    ) AS accepted_metrics,
    count(e.evaluation_id) FILTER (
        WHERE e.status = 'REJECTED'
    ) AS rejected_metrics,
    CASE
        WHEN count(e.evaluation_id) = 0 THEN 'NOT_EVALUATED'
        WHEN bool_and(e.status = 'ACCEPTED') THEN 'ELIGIBLE'
        ELSE 'NOT_ELIGIBLE'
    END AS eligibility
FROM training_runs r
JOIN models m ON m.model_id = r.model_id
JOIN loss_evaluations e ON e.run_id = r.run_id
JOIN loss_policies p ON p.policy_id = e.policy_id
GROUP BY
    r.run_id,
    r.run_name,
    m.model_name,
    p.policy_name;

-- ============================================================================
-- SAMPLE MODELS
-- ============================================================================

INSERT INTO models (
    model_name,
    model_version,
    task,
    owner_name
)
VALUES
    ('DemandForecast', '2.1.0', 'REGRESSION', 'ForecastingTeam'),
    ('FraudDetection', '3.2.0', 'BINARY_CLASSIFICATION', 'RiskAnalytics'),
    ('DocumentClassifier', '1.4.0', 'MULTICLASS_CLASSIFICATION', 'DocumentTeam');

INSERT INTO loss_policies (
    policy_name,
    maximum_mse,
    maximum_mae,
    maximum_bce,
    maximum_cce,
    minimum_samples
)
VALUES
    ('ProductionQualityPolicy', 1.0, 1.0, 0.5, 0.5, 2),
    ('StrictQualityPolicy', 0.1, 0.2, 0.2, 0.2, 3);

-- ============================================================================
-- SAMPLE TRAINING RUNS
-- ============================================================================

INSERT INTO training_runs (
    model_id,
    run_name,
    status,
    dataset_name,
    random_seed,
    started_at,
    completed_at
)
SELECT
    model_id,
    'validation-run-001',
    'COMPLETED',
    'enterprise_validation_dataset',
    42,
    CURRENT_TIMESTAMP - INTERVAL '2 hours',
    CURRENT_TIMESTAMP - INTERVAL '1 hour'
FROM models
WHERE model_name = 'DemandForecast';

INSERT INTO training_runs (
    model_id,
    run_name,
    status,
    dataset_name,
    random_seed,
    started_at,
    completed_at
)
SELECT
    model_id,
    'validation-run-002',
    'COMPLETED',
    'fraud_validation_dataset',
    43,
    CURRENT_TIMESTAMP - INTERVAL '2 hours',
    CURRENT_TIMESTAMP - INTERVAL '1 hour'
FROM models
WHERE model_name = 'FraudDetection';

INSERT INTO training_runs (
    model_id,
    run_name,
    status,
    dataset_name,
    random_seed,
    started_at,
    completed_at
)
SELECT
    model_id,
    'validation-run-003',
    'COMPLETED',
    'document_validation_dataset',
    44,
    CURRENT_TIMESTAMP - INTERVAL '2 hours',
    CURRENT_TIMESTAMP - INTERVAL '1 hour'
FROM models
WHERE model_name = 'DocumentClassifier';

-- ============================================================================
-- SAMPLE PREDICTIONS
-- ============================================================================

INSERT INTO predictions (
    run_id,
    sample_reference,
    actual_value,
    predicted_value
)
SELECT
    run_id,
    sample_reference,
    actual_value,
    predicted_value
FROM training_runs
CROSS JOIN (
    VALUES
        ('REG-001', 3.0, 2.5),
        ('REG-002', -0.5, 0.0),
        ('REG-003', 2.0, 2.0),
        ('REG-004', 7.0, 8.0)
) AS samples(sample_reference, actual_value, predicted_value)
WHERE run_name = 'validation-run-001';

INSERT INTO predictions (
    run_id,
    sample_reference,
    binary_label,
    predicted_probability
)
SELECT
    run_id,
    sample_reference,
    binary_label,
    predicted_probability
FROM training_runs
CROSS JOIN (
    VALUES
        ('BIN-001', 1, 0.95),
        ('BIN-002', 0, 0.10),
        ('BIN-003', 1, 0.80),
        ('BIN-004', 0, 0.20)
) AS samples(sample_reference, binary_label, predicted_probability)
WHERE run_name = 'validation-run-002';

INSERT INTO predictions (
    run_id,
    sample_reference,
    target_class,
    probability_array
)
SELECT
    run_id,
    sample_reference,
    target_class,
    probability_array
FROM training_runs
CROSS JOIN (
    VALUES
        ('CLS-001', 0, ARRAY[0.70, 0.20, 0.10]::float8[]),
        ('CLS-002', 2, ARRAY[0.10, 0.20, 0.70]::float8[]),
        ('CLS-003', 1, ARRAY[0.15, 0.75, 0.10]::float8[])
) AS samples(sample_reference, target_class, probability_array)
WHERE run_name = 'validation-run-003';

-- ============================================================================
-- EVALUATION WORKFLOW
-- ============================================================================

-- Each procedure call is atomic.
-- If validation fails, its changes are rolled back by the surrounding
-- transaction.

CALL evaluate_training_run(
    (
        SELECT run_id
        FROM training_runs
        WHERE run_name = 'validation-run-001'
    ),
    (
        SELECT policy_id
        FROM loss_policies
        WHERE policy_name = 'ProductionQualityPolicy'
    ),
    'QualityAssurance'
);

CALL evaluate_training_run(
    (
        SELECT run_id
        FROM training_runs
        WHERE run_name = 'validation-run-002'
    ),
    (
        SELECT policy_id
        FROM loss_policies
        WHERE policy_name = 'ProductionQualityPolicy'
    ),
    'RiskEvaluation'
);

CALL evaluate_training_run(
    (
        SELECT run_id
        FROM training_runs
        WHERE run_name = 'validation-run-003'
    ),
    (
        SELECT policy_id
        FROM loss_policies
        WHERE policy_name = 'ProductionQualityPolicy'
    ),
    'ClassificationEvaluation'
);

-- ============================================================================
-- RETRIEVAL QUERIES
-- ============================================================================

SELECT *
FROM v_model_loss_report
ORDER BY model_name, metric;

SELECT *
FROM v_policy_performance
ORDER BY policy_name, metric;

SELECT *
FROM v_run_eligibility
ORDER BY run_id;

-- ============================================================================
-- DIRECT LOSS CALCULATION QUERIES
-- ============================================================================

SELECT calculate_mse(
    ARRAY[3.0, -0.5, 2.0, 7.0],
    ARRAY[2.5, 0.0, 2.0, 8.0]
) AS mean_squared_error;

SELECT calculate_mae(
    ARRAY[3.0, -0.5, 2.0, 7.0],
    ARRAY[2.5, 0.0, 2.0, 8.0]
) AS mean_absolute_error;

SELECT calculate_binary_cross_entropy(
    ARRAY[1, 0, 1, 0]::smallint[],
    ARRAY[0.95, 0.10, 0.80, 0.20]::float8[]
) AS binary_cross_entropy;

SELECT calculate_categorical_cross_entropy(
    ARRAY[0, 2, 1],
    ARRAY[
        ARRAY[0.70, 0.20, 0.10],
        ARRAY[0.10, 0.20, 0.70],
        ARRAY[0.15, 0.75, 0.10]
    ]::float8[][]
) AS categorical_cross_entropy;

-- ============================================================================
-- AGGREGATION AND FAILURE ANALYSIS
-- ============================================================================

SELECT
    metric,
    count(*) AS total_evaluations,
    round(avg(loss_value)::numeric, 6) AS mean_loss,
    round(stddev_pop(loss_value)::numeric, 6) AS loss_stddev,
    min(loss_value) AS minimum_loss,
    max(loss_value) AS maximum_loss
FROM loss_evaluations
GROUP BY metric
ORDER BY metric;

SELECT
    model_name,
    metric,
    loss_value,
    threshold_value,
    status
FROM v_model_loss_report
WHERE evaluation_status = 'REJECTED';

-- ============================================================================
-- AUDIT HISTORY
-- ============================================================================

SELECT
    e.evaluation_id,
    e.metric,
    ev.old_status,
    ev.new_status,
    ev.event_description,
    ev.occurred_at
FROM evaluation_events ev
JOIN loss_evaluations e
    ON e.evaluation_id = ev.evaluation_id
ORDER BY ev.occurred_at;

-- ============================================================================
-- VALIDATION FAILURE EXAMPLES
-- ============================================================================
-- Run individually inside a savepoint if testing exceptions manually.
--
-- Invalid binary label:
-- SELECT calculate_binary_cross_entropy(
--     ARRAY[2]::smallint[],
--     ARRAY[0.5]::float8[]
-- );
--
-- Invalid probability:
-- SELECT calculate_binary_cross_entropy(
--     ARRAY[1]::smallint[],
--     ARRAY[1.5]::float8[]
-- );
--
-- Invalid probability distribution:
-- SELECT calculate_categorical_cross_entropy(
--     ARRAY[0],
--     ARRAY[ARRAY[0.8, 0.8]]::float8[][]
-- );
--
-- Incomplete regression prediction:
-- INSERT INTO predictions (
--     run_id,
--     sample_reference,
--     actual_value
-- )
-- VALUES (1, 'INVALID-001', 10.0);

COMMIT;

-- ============================================================================
-- END OF LOSS GOVERNANCE DATABASE
-- ============================================================================
