-- ================================================================
-- REGULARIZATION IN DEEP LEARNING
-- PostgreSQL 15+
--
-- Relational experiment-management system covering:
-- 1. Training datasets and model architectures
-- 2. Regularization configurations
-- 3. Training and validation metrics
-- 4. Dropout configuration
-- 5. L2 and decoupled weight decay
-- 6. Data augmentation
-- 7. Early stopping and checkpoint selection
-- 8. Experiment eligibility and reporting
--
-- The database stores experiment evidence and evaluates policies.
-- Numerical neural-network optimization is executed by application code.
-- ================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS dl_regularization;

SET search_path TO dl_regularization, public;

-- ================================================================
-- 1. ENUM TYPES
-- ================================================================

CREATE TYPE regularization_method AS ENUM (
    'DROPOUT',
    'L2',
    'WEIGHT_DECAY',
    'DATA_AUGMENTATION',
    'EARLY_STOPPING'
);

CREATE TYPE optimizer_type AS ENUM (
    'SGD',
    'ADAM',
    'ADAMW'
);

CREATE TYPE experiment_status AS ENUM (
    'CREATED',
    'VALIDATING',
    'RUNNING',
    'EARLY_STOPPED',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE dataset_role AS ENUM (
    'TRAINING',
    'VALIDATION',
    'TEST'
);

CREATE TYPE checkpoint_status AS ENUM (
    'CANDIDATE',
    'SELECTED',
    'REJECTED'
);

-- ================================================================
-- 2. DATASETS
-- ================================================================

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_name VARCHAR(150) NOT NULL UNIQUE,

    feature_count INTEGER NOT NULL
        CHECK (feature_count > 0),

    sample_count BIGINT NOT NULL
        CHECK (sample_count > 0),

    class_count INTEGER NOT NULL
        CHECK (class_count >= 2),

    content_hash VARCHAR(128) NOT NULL,

    augmentation_compatible BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (length(content_hash) >= 16)
);

CREATE TABLE dataset_assignments (
    assignment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),

    role dataset_role NOT NULL,

    partition_version INTEGER NOT NULL DEFAULT 1
        CHECK (partition_version > 0),

    sample_count BIGINT NOT NULL
        CHECK (sample_count > 0),

    UNIQUE (dataset_id, role, partition_version)
);

-- ================================================================
-- 3. MODEL ARCHITECTURES
-- ================================================================

CREATE TABLE model_architectures (
    architecture_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    architecture_name VARCHAR(150) NOT NULL UNIQUE,

    input_dimension INTEGER NOT NULL
        CHECK (input_dimension > 0),

    parameter_count BIGINT NOT NULL
        CHECK (parameter_count > 0),

    hidden_layer_count INTEGER NOT NULL
        CHECK (hidden_layer_count >= 0),

    output_dimension INTEGER NOT NULL
        CHECK (output_dimension > 0),

    description TEXT
);

-- ================================================================
-- 4. REGULARIZATION POLICIES
-- ================================================================

CREATE TABLE regularization_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    policy_name VARCHAR(150) NOT NULL UNIQUE,

    dropout_rate NUMERIC(6,5) NOT NULL DEFAULT 0
        CHECK (dropout_rate >= 0 AND dropout_rate < 1),

    weight_decay NUMERIC(12,8) NOT NULL DEFAULT 0
        CHECK (weight_decay >= 0),

    augmentation_noise NUMERIC(12,8) NOT NULL DEFAULT 0
        CHECK (augmentation_noise >= 0),

    augmentation_enabled BOOLEAN NOT NULL DEFAULT FALSE,

    patience INTEGER NOT NULL DEFAULT 10
        CHECK (patience > 0),

    min_delta NUMERIC(12,8) NOT NULL DEFAULT 0.00001
        CHECK (min_delta >= 0),

    optimizer optimizer_type NOT NULL DEFAULT 'ADAM',

    maximum_epochs INTEGER NOT NULL DEFAULT 100
        CHECK (maximum_epochs > 0),

    learning_rate NUMERIC(12,8) NOT NULL DEFAULT 0.001
        CHECK (learning_rate > 0),

    description TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (
        augmentation_enabled = TRUE
        OR augmentation_noise = 0
    )
);

-- ================================================================
-- 5. EXPERIMENTS
-- ================================================================

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_name VARCHAR(180) NOT NULL,

    architecture_id BIGINT NOT NULL
        REFERENCES model_architectures(architecture_id),

    policy_id BIGINT NOT NULL
        REFERENCES regularization_policies(policy_id),

    training_dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),

    validation_dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),

    test_dataset_id BIGINT
        REFERENCES datasets(dataset_id),

    status experiment_status NOT NULL DEFAULT 'CREATED',

    random_seed INTEGER NOT NULL DEFAULT 42,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    failure_reason TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_name),

    CHECK (training_dataset_id <> validation_dataset_id),

    CHECK (
        test_dataset_id IS NULL
        OR (
            test_dataset_id <> training_dataset_id
            AND test_dataset_id <> validation_dataset_id
        )
    ),

    CHECK (
        completed_at IS NULL
        OR started_at IS NOT NULL
    )
);

CREATE INDEX idx_experiments_status
ON experiments(status, created_at DESC);

CREATE INDEX idx_experiments_policy
ON experiments(policy_id);

-- ================================================================
-- 6. REGULARIZATION METHOD REGISTRY
-- ================================================================

CREATE TABLE experiment_methods (
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    method regularization_method NOT NULL,

    enabled BOOLEAN NOT NULL DEFAULT TRUE,

    method_configuration JSONB NOT NULL DEFAULT '{}'::jsonb,

    PRIMARY KEY (experiment_id, method),

    CHECK (jsonb_typeof(method_configuration) = 'object')
);

-- ================================================================
-- 7. EPOCH METRICS
-- ================================================================

CREATE TABLE epoch_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    epoch_number INTEGER NOT NULL
        CHECK (epoch_number > 0),

    training_loss NUMERIC(16,10) NOT NULL
        CHECK (training_loss >= 0),

    validation_loss NUMERIC(16,10) NOT NULL
        CHECK (validation_loss >= 0),

    training_accuracy NUMERIC(8,6)
        CHECK (training_accuracy BETWEEN 0 AND 1),

    validation_accuracy NUMERIC(8,6)
        CHECK (validation_accuracy BETWEEN 0 AND 1),

    learning_rate NUMERIC(12,8) NOT NULL
        CHECK (learning_rate > 0),

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, epoch_number)
);

CREATE INDEX idx_epoch_metrics_experiment_epoch
ON epoch_metrics(experiment_id, epoch_number);

CREATE INDEX idx_epoch_metrics_validation_loss
ON epoch_metrics(experiment_id, validation_loss);

-- ================================================================
-- 8. CHECKPOINTS
-- ================================================================

CREATE TABLE model_checkpoints (
    checkpoint_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    epoch_number INTEGER NOT NULL
        CHECK (epoch_number > 0),

    validation_loss NUMERIC(16,10) NOT NULL
        CHECK (validation_loss >= 0),

    checkpoint_uri TEXT NOT NULL,

    content_hash VARCHAR(128) NOT NULL,

    status checkpoint_status NOT NULL DEFAULT 'CANDIDATE',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, epoch_number),

    CHECK (length(content_hash) >= 16)
);

CREATE UNIQUE INDEX idx_one_selected_checkpoint
ON model_checkpoints(experiment_id)
WHERE status = 'SELECTED';

-- ================================================================
-- 9. EARLY STOPPING AUDIT
-- ================================================================

CREATE TABLE early_stopping_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    observed_epoch INTEGER NOT NULL
        CHECK (observed_epoch > 0),

    best_epoch INTEGER NOT NULL
        CHECK (best_epoch > 0),

    best_validation_loss NUMERIC(16,10) NOT NULL
        CHECK (best_validation_loss >= 0),

    stale_epochs INTEGER NOT NULL
        CHECK (stale_epochs >= 0),

    patience INTEGER NOT NULL
        CHECK (patience > 0),

    stopping_triggered BOOLEAN NOT NULL,

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (best_epoch <= observed_epoch),

    CHECK (
        stopping_triggered = FALSE
        OR stale_epochs >= patience
    )
);

-- ================================================================
-- 10. EXPERIMENT STATUS VALIDATION
-- ================================================================

CREATE OR REPLACE FUNCTION validate_experiment_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status = OLD.status THEN
        RETURN NEW;
    END IF;

    IF NOT (
        (OLD.status = 'CREATED'
            AND NEW.status IN ('VALIDATING', 'CANCELLED', 'FAILED'))
        OR
        (OLD.status = 'VALIDATING'
            AND NEW.status IN ('RUNNING', 'FAILED', 'CANCELLED'))
        OR
        (OLD.status = 'RUNNING'
            AND NEW.status IN (
                'EARLY_STOPPED',
                'COMPLETED',
                'FAILED',
                'CANCELLED'
            ))
    ) THEN
        RAISE EXCEPTION
            'Invalid experiment transition: % -> %',
            OLD.status,
            NEW.status;
    END IF;

    IF NEW.status IN ('RUNNING', 'COMPLETED', 'EARLY_STOPPED')
       AND NEW.started_at IS NULL THEN
        RAISE EXCEPTION
            'A running or completed experiment requires started_at';
    END IF;

    IF NEW.status IN ('COMPLETED', 'EARLY_STOPPED', 'FAILED', 'CANCELLED')
       AND NEW.completed_at IS NULL THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_experiment_transition
BEFORE UPDATE OF status ON experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_transition();

-- ================================================================
-- 11. DATASET COMPATIBILITY VALIDATION
-- ================================================================

CREATE OR REPLACE FUNCTION validate_experiment_datasets()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    training_dimension INTEGER;
    validation_dimension INTEGER;
    model_dimension INTEGER;
    training_classes INTEGER;
    validation_classes INTEGER;
BEGIN
    SELECT feature_count, class_count
    INTO training_dimension, training_classes
    FROM datasets
    WHERE dataset_id = NEW.training_dataset_id;

    SELECT feature_count, class_count
    INTO validation_dimension, validation_classes
    FROM datasets
    WHERE dataset_id = NEW.validation_dataset_id;

    SELECT input_dimension
    INTO model_dimension
    FROM model_architectures
    WHERE architecture_id = NEW.architecture_id;

    IF training_dimension <> validation_dimension
       OR training_dimension <> model_dimension THEN
        RAISE EXCEPTION
            'Architecture and dataset feature dimensions differ';
    END IF;

    IF training_classes <> validation_classes THEN
        RAISE EXCEPTION
            'Training and validation class counts differ';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_experiment_datasets
BEFORE INSERT OR UPDATE OF
    architecture_id,
    training_dataset_id,
    validation_dataset_id
ON experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_datasets();

-- ================================================================
-- 12. EXPERIMENT ELIGIBILITY
-- ================================================================

CREATE OR REPLACE FUNCTION evaluate_experiment_eligibility(
    requested_experiment BIGINT
)
RETURNS TABLE (
    experiment_id BIGINT,
    eligible BOOLEAN,
    reason TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    exp RECORD;
    policy RECORD;
    training_count BIGINT;
    validation_count BIGINT;
    architecture_dimension INTEGER;
BEGIN
    SELECT *
    INTO exp
    FROM experiments
    WHERE experiments.experiment_id = requested_experiment;

    IF NOT FOUND THEN
        RETURN QUERY SELECT
            requested_experiment,
            FALSE,
            'Experiment does not exist'::TEXT;
        RETURN;
    END IF;

    SELECT *
    INTO policy
    FROM regularization_policies
    WHERE policy_id = exp.policy_id;

    SELECT sample_count, feature_count
    INTO training_count, architecture_dimension
    FROM datasets
    WHERE dataset_id = exp.training_dataset_id;

    SELECT sample_count
    INTO validation_count
    FROM datasets
    WHERE dataset_id = exp.validation_dataset_id;

    IF exp.status NOT IN ('CREATED', 'VALIDATING') THEN
        RETURN QUERY SELECT
            requested_experiment,
            FALSE,
            'Experiment is not in an eligible initial state'::TEXT;
        RETURN;
    END IF;

    IF training_count < 2 OR validation_count < 2 THEN
        RETURN QUERY SELECT
            requested_experiment,
            FALSE,
            'Insufficient training or validation samples'::TEXT;
        RETURN;
    END IF;

    IF policy.augmentation_enabled = FALSE
       AND policy.augmentation_noise <> 0 THEN
        RETURN QUERY SELECT
            requested_experiment,
            FALSE,
            'Augmentation configuration is inconsistent'::TEXT;
        RETURN;
    END IF;

    RETURN QUERY SELECT
        requested_experiment,
        TRUE,
        'Configuration and datasets passed validation'::TEXT;
END;
$$;

-- ================================================================
-- 13. EARLY STOPPING EVALUATION
-- ================================================================

CREATE OR REPLACE FUNCTION evaluate_early_stopping(
    requested_experiment BIGINT
)
RETURNS TABLE (
    best_epoch INTEGER,
    best_validation_loss NUMERIC,
    current_epoch INTEGER,
    stale_epochs INTEGER,
    should_stop BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    policy RECORD;
    best RECORD;
    current RECORD;
BEGIN
    SELECT p.patience, p.min_delta
    INTO policy
    FROM experiments e
    JOIN regularization_policies p
        ON p.policy_id = e.policy_id
    WHERE e.experiment_id = requested_experiment;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Experiment not found';
    END IF;

    SELECT
        epoch_number,
        validation_loss
    INTO best
    FROM epoch_metrics
    WHERE experiment_id = requested_experiment
    ORDER BY validation_loss ASC, epoch_number ASC
    LIMIT 1;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No epoch metrics available';
    END IF;

    SELECT epoch_number
    INTO current
    FROM epoch_metrics
    WHERE experiment_id = requested_experiment
    ORDER BY epoch_number DESC
    LIMIT 1;

    RETURN QUERY
    SELECT
        best.epoch_number::INTEGER,
        best.validation_loss::NUMERIC,
        current.epoch_number::INTEGER,
        (current.epoch_number - best.epoch_number)::INTEGER,
        (
            current.epoch_number - best.epoch_number
            >= policy.patience
        )::BOOLEAN;
END;
$$;

-- ================================================================
-- 14. BEST CHECKPOINT SELECTION
-- ================================================================

CREATE OR REPLACE FUNCTION select_best_checkpoint(
    requested_experiment BIGINT
)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    selected_checkpoint BIGINT;
BEGIN
    SELECT checkpoint_id
    INTO selected_checkpoint
    FROM model_checkpoints
    WHERE experiment_id = requested_experiment
    ORDER BY validation_loss ASC, epoch_number ASC
    LIMIT 1;

    IF selected_checkpoint IS NULL THEN
        RAISE EXCEPTION 'No checkpoint available';
    END IF;

    -- Ensure only the best checkpoint is selected.
    UPDATE model_checkpoints
    SET status = CASE
        WHEN checkpoint_id = selected_checkpoint THEN 'SELECTED'
        ELSE 'REJECTED'
    END
    WHERE experiment_id = requested_experiment;

    RETURN selected_checkpoint;
END;
$$;

-- ================================================================
-- 15. EXPERIMENT REPORTING VIEW
-- ================================================================

CREATE OR REPLACE VIEW experiment_performance AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.status,
    ma.architecture_name,
    p.policy_name,
    p.dropout_rate,
    p.weight_decay,
    p.augmentation_noise,
    p.optimizer,
    p.patience,

    COUNT(em.metric_id) AS epochs_recorded,

    MIN(em.training_loss) AS minimum_training_loss,

    MIN(em.validation_loss) AS minimum_validation_loss,

    (
        SELECT epoch_number
        FROM epoch_metrics best
        WHERE best.experiment_id = e.experiment_id
        ORDER BY validation_loss ASC, epoch_number ASC
        LIMIT 1
    ) AS best_epoch,

    MAX(em.validation_accuracy) AS maximum_validation_accuracy,

    MAX(em.recorded_at) AS last_metric_recorded

FROM experiments e

JOIN model_architectures ma
    ON ma.architecture_id = e.architecture_id

JOIN regularization_policies p
    ON p.policy_id = e.policy_id

LEFT JOIN epoch_metrics em
    ON em.experiment_id = e.experiment_id

GROUP BY
    e.experiment_id,
    e.experiment_name,
    e.status,
    ma.architecture_name,
    p.policy_name,
    p.dropout_rate,
    p.weight_decay,
    p.augmentation_noise,
    p.optimizer,
    p.patience;

-- ================================================================
-- 16. SAMPLE DATASETS
-- ================================================================

INSERT INTO datasets (
    dataset_name,
    feature_count,
    sample_count,
    class_count,
    content_hash
)
VALUES
(
    'Synthetic Training Dataset',
    2,
    180,
    2,
    'aabbccddeeff00112233445566778899'
),
(
    'Synthetic Validation Dataset',
    2,
    60,
    2,
    '11223344556677889900aabbccddeeff'
),
(
    'Synthetic Test Dataset',
    2,
    60,
    2,
    'ffeeddccbbaa00998877665544332211'
);

INSERT INTO dataset_assignments (
    dataset_id,
    role,
    sample_count
)
SELECT dataset_id, 'TRAINING', 180
FROM datasets
WHERE dataset_name = 'Synthetic Training Dataset';

INSERT INTO dataset_assignments (
    dataset_id,
    role,
    sample_count
)
SELECT dataset_id, 'VALIDATION', 60
FROM datasets
WHERE dataset_name = 'Synthetic Validation Dataset';

INSERT INTO dataset_assignments (
    dataset_id,
    role,
    sample_count
)
SELECT dataset_id, 'TEST', 60
FROM datasets
WHERE dataset_name = 'Synthetic Test Dataset';

-- ================================================================
-- 17. SAMPLE ARCHITECTURE
-- ================================================================

INSERT INTO model_architectures (
    architecture_name,
    input_dimension,
    parameter_count,
    hidden_layer_count,
    output_dimension,
    description
)
VALUES (
    'Two-Layer Binary Classifier',
    2,
    33,
    1,
    1,
    'Two-dimensional binary classification network'
);

-- ================================================================
-- 18. REGULARIZATION POLICIES
-- ================================================================

INSERT INTO regularization_policies (
    policy_name,
    dropout_rate,
    weight_decay,
    augmentation_noise,
    augmentation_enabled,
    patience,
    min_delta,
    optimizer,
    maximum_epochs,
    learning_rate,
    description
)
VALUES
(
    'Baseline',
    0,
    0,
    0,
    FALSE,
    10,
    0.00001,
    'SGD',
    100,
    0.035,
    'No explicit regularization'
),
(
    'Dropout Policy',
    0.25,
    0,
    0,
    FALSE,
    12,
    0.00001,
    'SGD',
    100,
    0.035,
    'Hidden-layer inverted dropout'
),
(
    'L2 Policy',
    0,
    0.01,
    0,
    FALSE,
    12,
    0.00001,
    'ADAM',
    100,
    0.035,
    'Gradient-based L2 regularization'
),
(
    'AdamW Combined Policy',
    0.20,
    0.005,
    0.04,
    TRUE,
    12,
    0.00001,
    'ADAMW',
    100,
    0.035,
    'Dropout, decoupled weight decay and augmentation'
);

-- ================================================================
-- 19. SAMPLE EXPERIMENTS
-- ================================================================

INSERT INTO experiments (
    experiment_name,
    architecture_id,
    policy_id,
    training_dataset_id,
    validation_dataset_id,
    test_dataset_id
)
SELECT
    'Baseline Experiment',
    ma.architecture_id,
    p.policy_id,
    train.dataset_id,
    validation.dataset_id,
    test.dataset_id
FROM model_architectures ma
CROSS JOIN regularization_policies p
CROSS JOIN datasets train
CROSS JOIN datasets validation
CROSS JOIN datasets test
WHERE ma.architecture_name = 'Two-Layer Binary Classifier'
AND p.policy_name = 'Baseline'
AND train.dataset_name = 'Synthetic Training Dataset'
AND validation.dataset_name = 'Synthetic Validation Dataset'
AND test.dataset_name = 'Synthetic Test Dataset';

INSERT INTO experiments (
    experiment_name,
    architecture_id,
    policy_id,
    training_dataset_id,
    validation_dataset_id,
    test_dataset_id
)
SELECT
    'Combined Regularization Experiment',
    ma.architecture_id,
    p.policy_id,
    train.dataset_id,
    validation.dataset_id,
    test.dataset_id
FROM model_architectures ma
CROSS JOIN regularization_policies p
CROSS JOIN datasets train
CROSS JOIN datasets validation
CROSS JOIN datasets test
WHERE ma.architecture_name = 'Two-Layer Binary Classifier'
AND p.policy_name = 'AdamW Combined Policy'
AND train.dataset_name = 'Synthetic Training Dataset'
AND validation.dataset_name = 'Synthetic Validation Dataset'
AND test.dataset_name = 'Synthetic Test Dataset';

-- ================================================================
-- 20. REGISTER EXPERIMENT METHODS
-- ================================================================

INSERT INTO experiment_methods (
    experiment_id,
    method,
    enabled,
    method_configuration
)
SELECT
    e.experiment_id,
    m.method,
    TRUE,
    CASE m.method
        WHEN 'DROPOUT' THEN
            jsonb_build_object('rate', p.dropout_rate)
        WHEN 'WEIGHT_DECAY' THEN
            jsonb_build_object(
                'coefficient', p.weight_decay,
                'optimizer', p.optimizer
            )
        WHEN 'DATA_AUGMENTATION' THEN
            jsonb_build_object(
                'noise_std', p.augmentation_noise
            )
        WHEN 'EARLY_STOPPING' THEN
            jsonb_build_object(
                'patience', p.patience,
                'min_delta', p.min_delta
            )
        WHEN 'L2' THEN
            jsonb_build_object(
                'lambda', p.weight_decay
            )
    END
FROM experiments e
JOIN regularization_policies p
    ON p.policy_id = e.policy_id
CROSS JOIN (
    VALUES
        ('DROPOUT'::regularization_method),
        ('L2'::regularization_method),
        ('WEIGHT_DECAY'::regularization_method),
        ('DATA_AUGMENTATION'::regularization_method),
        ('EARLY_STOPPING'::regularization_method)
) AS m(method)
WHERE
    (m.method = 'DROPOUT' AND p.dropout_rate > 0)
    OR
    (m.method IN ('L2', 'WEIGHT_DECAY') AND p.weight_decay > 0)
    OR
    (m.method = 'DATA_AUGMENTATION' AND p.augmentation_enabled)
    OR
    m.method = 'EARLY_STOPPING';

-- ================================================================
-- 21. VALIDATE EXPERIMENTS AND START EXECUTION
-- ================================================================

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT experiment_id
        FROM experiments
        ORDER BY experiment_id
    LOOP
        IF NOT (
            SELECT eligible
            FROM evaluate_experiment_eligibility(item.experiment_id)
        ) THEN
            RAISE EXCEPTION
                'Experiment % failed eligibility',
                item.experiment_id;
        END IF;

        UPDATE experiments
        SET status = 'VALIDATING'
        WHERE experiment_id = item.experiment_id;

        UPDATE experiments
        SET status = 'RUNNING',
            started_at = CURRENT_TIMESTAMP
        WHERE experiment_id = item.experiment_id;
    END LOOP;
END;
$$;

-- ================================================================
-- 22. SAMPLE EPOCH METRICS
-- ================================================================

INSERT INTO epoch_metrics (
    experiment_id,
    epoch_number,
    training_loss,
    validation_loss,
    training_accuracy,
    validation_accuracy,
    learning_rate
)
SELECT
    e.experiment_id,
    epoch,
    0.85 / (epoch + 2) + 0.08,
    0.75 / (epoch + 2) + 0.12
        + GREATEST(0, epoch - 12) * 0.003,
    LEAST(0.98, 0.65 + epoch * 0.002),
    LEAST(0.95, 0.60 + epoch * 0.012),
    0.035
FROM experiments e
CROSS JOIN generate_series(1, 20) AS epoch;

-- ================================================================
-- 23. SAMPLE CHECKPOINTS
-- ================================================================

INSERT INTO model_checkpoints (
    experiment_id,
    epoch_number,
    validation_loss,
    checkpoint_uri,
    content_hash,
    status
)
SELECT
    metric.experiment_id,
    metric.epoch_number,
    metric.validation_loss,
    'models/experiment-' ||
        metric.experiment_id ||
        '/checkpoint-' ||
        metric.epoch_number,
    md5(
        metric.experiment_id::TEXT ||
        '-' ||
        metric.epoch_number::TEXT
    ),
    'CANDIDATE'
FROM epoch_metrics metric
WHERE metric.epoch_number IN (8, 12, 16, 20);

-- ================================================================
-- 24. RECORD EARLY STOPPING DECISIONS
-- ================================================================

INSERT INTO early_stopping_events (
    experiment_id,
    observed_epoch,
    best_epoch,
    best_validation_loss,
    stale_epochs,
    patience,
    stopping_triggered
)
SELECT
    e.experiment_id,
    latest.epoch_number,
    best.epoch_number,
    best.validation_loss,
    latest.epoch_number - best.epoch_number,
    p.patience,
    latest.epoch_number - best.epoch_number >= p.patience
FROM experiments e
JOIN regularization_policies p
    ON p.policy_id = e.policy_id
JOIN LATERAL (
    SELECT epoch_number
    FROM epoch_metrics
    WHERE experiment_id = e.experiment_id
    ORDER BY epoch_number DESC
    LIMIT 1
) latest ON TRUE
JOIN LATERAL (
    SELECT epoch_number, validation_loss
    FROM epoch_metrics
    WHERE experiment_id = e.experiment_id
    ORDER BY validation_loss ASC, epoch_number ASC
    LIMIT 1
) best ON TRUE;

-- ================================================================
-- 25. SELECT BEST CHECKPOINTS AND COMPLETE EXPERIMENTS
-- ================================================================

DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT experiment_id
        FROM experiments
        WHERE status = 'RUNNING'
    LOOP
        PERFORM select_best_checkpoint(item.experiment_id);

        UPDATE experiments
        SET status = 'COMPLETED',
            completed_at = CURRENT_TIMESTAMP
        WHERE experiment_id = item.experiment_id;
    END LOOP;
END;
$$;

COMMIT;

-- ================================================================
-- 26. RETRIEVAL QUERIES
-- ================================================================

-- Retrieve all registered policies.
SELECT
    policy_name,
    dropout_rate,
    weight_decay,
    augmentation_noise,
    optimizer,
    patience
FROM regularization_policies
ORDER BY policy_id;

-- Retrieve experiment details.
SELECT *
FROM experiment_performance
ORDER BY minimum_validation_loss ASC NULLS LAST;

-- Retrieve epoch history.
SELECT
    e.experiment_name,
    m.epoch_number,
    m.training_loss,
    m.validation_loss,
    m.training_accuracy,
    m.validation_accuracy
FROM epoch_metrics m
JOIN experiments e
    ON e.experiment_id = m.experiment_id
ORDER BY e.experiment_name, m.epoch_number;

-- ================================================================
-- 27. VALIDATION QUERIES
-- ================================================================

-- Detect invalid or inconsistent augmentation configurations.
SELECT
    policy_id,
    policy_name,
    'Invalid augmentation configuration' AS validation_error
FROM regularization_policies
WHERE
    augmentation_enabled = FALSE
    AND augmentation_noise <> 0;

-- Detect architecture and dataset incompatibility.
SELECT
    e.experiment_name,
    d.feature_count,
    a.input_dimension
FROM experiments e
JOIN datasets d
    ON d.dataset_id = e.training_dataset_id
JOIN model_architectures a
    ON a.architecture_id = e.architecture_id
WHERE d.feature_count <> a.input_dimension;

-- Detect missing checkpoint selections.
SELECT
    e.experiment_name
FROM experiments e
LEFT JOIN model_checkpoints c
    ON c.experiment_id = e.experiment_id
    AND c.status = 'SELECTED'
WHERE e.status IN ('COMPLETED', 'EARLY_STOPPED')
GROUP BY e.experiment_id, e.experiment_name
HAVING COUNT(c.checkpoint_id) <> 1;

-- ================================================================
-- 28. AGGREGATION AND POLICY EVALUATION
-- ================================================================

-- Compare regularization strategies.
SELECT
    p.policy_name,
    p.dropout_rate,
    p.weight_decay,
    p.augmentation_noise,
    COUNT(DISTINCT e.experiment_id) AS experiment_count,
    ROUND(AVG(best.validation_loss), 6) AS average_best_validation_loss,
    ROUND(AVG(best.validation_accuracy), 6) AS average_best_accuracy
FROM experiments e
JOIN regularization_policies p
    ON p.policy_id = e.policy_id
JOIN LATERAL (
    SELECT
        validation_loss,
        validation_accuracy
    FROM epoch_metrics
    WHERE experiment_id = e.experiment_id
    ORDER BY validation_loss ASC, epoch_number ASC
    LIMIT 1
) best ON TRUE
GROUP BY
    p.policy_id,
    p.policy_name,
    p.dropout_rate,
    p.weight_decay,
    p.augmentation_noise
ORDER BY average_best_validation_loss ASC;

-- Compare training and validation loss gaps.
SELECT
    e.experiment_name,
    MIN(m.training_loss) AS best_training_loss,
    MIN(m.validation_loss) AS best_validation_loss,
    MIN(m.validation_loss) - MIN(m.training_loss)
        AS approximate_generalization_gap
FROM experiments e
JOIN epoch_metrics m
    ON m.experiment_id = e.experiment_id
GROUP BY e.experiment_id, e.experiment_name;

-- ================================================================
-- 29. EARLY STOPPING REPORT
-- ================================================================

SELECT *
FROM evaluate_early_stopping(1);

SELECT
    e.experiment_name,
    s.best_epoch,
    s.observed_epoch,
    s.stale_epochs,
    s.patience,
    s.stopping_triggered
FROM early_stopping_events s
JOIN experiments e
    ON e.experiment_id = s.experiment_id;

-- ================================================================
-- 30. CHECKPOINT REPORT
-- ================================================================

SELECT
    e.experiment_name,
    c.epoch_number,
    c.validation_loss,
    c.status,
    c.checkpoint_uri
FROM model_checkpoints c
JOIN experiments e
    ON e.experiment_id = c.experiment_id
ORDER BY e.experiment_name, c.validation_loss;

-- ================================================================
-- 31. FAILURE-CASE DEMONSTRATION
-- ================================================================

-- The following query identifies invalid configurations.
-- The CHECK constraint prevents insertion of the invalid row.

SELECT
    policy_name,
    dropout_rate,
    weight_decay,
    augmentation_noise,
    patience
FROM regularization_policies
WHERE
    dropout_rate >= 1
    OR weight_decay < 0
    OR patience <= 0;

-- Eligibility evaluation for an existing experiment.
SELECT *
FROM evaluate_experiment_eligibility(1);

-- ================================================================
-- 32. OPERATIONAL NOTES
-- ================================================================

-- 1. Epoch metrics are append-oriented training evidence.
-- 2. A checkpoint URI does not guarantee that the external object exists.
--    Application code must verify the artifact and its cryptographic hash.
-- 3. Database constraints validate metadata, not neural-network mathematics.
-- 4. Test data must not participate in hyperparameter selection.
-- 5. Dataset immutability and partition isolation should be enforced
--    by application permissions and data-governance controls.
-- 6. For large experiments, partition epoch_metrics by experiment
--    or time after evaluating the operational query workload.
-- 7. Keep credentials outside SQL source files.
-- 8. Use role-based access control for experiment modification.
