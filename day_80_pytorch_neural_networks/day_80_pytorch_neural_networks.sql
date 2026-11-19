/*
===============================================================================
PYTORCH NEURAL NETWORK EXPERIMENT TRACKING
===============================================================================

PostgreSQL relational model for:
- Neural network architectures
- Modules and layers
- Layer configuration
- Training datasets
- Training runs
- Epoch-level metrics
- Optimizers and loss functions
- Validation results
- Model checkpoints
- Training state transitions
- Experiment comparison

Requires PostgreSQL 14+.
===============================================================================
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS neural_lab;

SET search_path TO neural_lab, public;

-- ============================================================================
-- 1. ENUMERATIONS
-- ============================================================================

CREATE TYPE model_status AS ENUM (
    'DRAFT',
    'CONFIGURED',
    'READY',
    'RETIRED'
);

CREATE TYPE training_status AS ENUM (
    'CREATED',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE layer_type AS ENUM (
    'LINEAR',
    'CONVOLUTION',
    'EMBEDDING',
    'BATCH_NORMALIZATION',
    'LAYER_NORMALIZATION',
    'RELU',
    'GELU',
    'SIGMOID',
    'TANH',
    'DROPOUT',
    'MAX_POOL',
    'FLATTEN',
    'SOFTMAX'
);

CREATE TYPE optimizer_type AS ENUM (
    'SGD',
    'ADAM',
    'ADAMW',
    'RMSPROP'
);

CREATE TYPE loss_type AS ENUM (
    'CROSS_ENTROPY',
    'BINARY_CROSS_ENTROPY',
    'MSE',
    'MAE',
    'HUBER'
);

CREATE TYPE checkpoint_status AS ENUM (
    'SAVED',
    'VERIFIED',
    'ARCHIVED',
    'CORRUPTED'
);

-- ============================================================================
-- 2. USERS AND PROJECTS
-- ============================================================================

CREATE TABLE researchers (
    researcher_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    username VARCHAR(80) NOT NULL UNIQUE,

    email VARCHAR(254) NOT NULL UNIQUE,

    role VARCHAR(30) NOT NULL CHECK (
        role IN ('ADMIN', 'RESEARCHER', 'REVIEWER', 'VIEWER')
    ),

    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE projects (
    project_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    project_name VARCHAR(150) NOT NULL UNIQUE,

    description TEXT,

    owner_id BIGINT NOT NULL REFERENCES researchers(researcher_id),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 3. DATASETS
-- ============================================================================

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    project_id BIGINT NOT NULL REFERENCES projects(project_id),

    dataset_name VARCHAR(150) NOT NULL,

    version VARCHAR(50) NOT NULL,

    sample_count BIGINT NOT NULL CHECK (sample_count > 0),

    feature_count INTEGER NOT NULL CHECK (feature_count > 0),

    class_count INTEGER CHECK (class_count > 0),

    data_format VARCHAR(30) NOT NULL CHECK (
        data_format IN ('CSV', 'PARQUET', 'IMAGE', 'TENSOR', 'OTHER')
    ),

    content_sha256 CHAR(64) NOT NULL CHECK (
        content_sha256 ~ '^[a-fA-F0-9]{64}$'
    ),

    train_fraction NUMERIC(5,4) NOT NULL DEFAULT 0.8 CHECK (
        train_fraction > 0 AND train_fraction < 1
    ),

    validation_fraction NUMERIC(5,4) NOT NULL DEFAULT 0.1 CHECK (
        validation_fraction >= 0 AND validation_fraction < 1
    ),

    test_fraction NUMERIC(5,4) NOT NULL DEFAULT 0.1 CHECK (
        test_fraction >= 0 AND test_fraction < 1
    ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_id, dataset_name, version),

    CHECK (
        train_fraction + validation_fraction + test_fraction = 1.0
    )
);

-- ============================================================================
-- 4. MODEL ARCHITECTURES
-- ============================================================================

CREATE TABLE models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    project_id BIGINT NOT NULL REFERENCES projects(project_id),

    model_name VARCHAR(150) NOT NULL,

    version VARCHAR(50) NOT NULL,

    description TEXT,

    input_features INTEGER NOT NULL CHECK (input_features > 0),

    output_classes INTEGER NOT NULL CHECK (output_classes > 0),

    status model_status NOT NULL DEFAULT 'DRAFT',

    created_by BIGINT NOT NULL REFERENCES researchers(researcher_id),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_id, model_name, version)
);

CREATE TABLE model_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL REFERENCES models(model_id)
        ON DELETE CASCADE,

    layer_name VARCHAR(100) NOT NULL,

    sequence_number INTEGER NOT NULL CHECK (sequence_number >= 0),

    type layer_type NOT NULL,

    input_dimensions JSONB NOT NULL,

    output_dimensions JSONB NOT NULL,

    configuration JSONB NOT NULL DEFAULT '{}'::jsonb,

    trainable BOOLEAN NOT NULL DEFAULT TRUE,

    UNIQUE(model_id, sequence_number),

    CHECK (jsonb_typeof(input_dimensions) = 'array'),

    CHECK (jsonb_typeof(output_dimensions) = 'array'),

    CHECK (jsonb_typeof(configuration) = 'object')
);

-- ============================================================================
-- 5. MODEL TRAINING CONFIGURATION
-- ============================================================================

CREATE TABLE training_configurations (
    configuration_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL REFERENCES models(model_id),

    optimizer optimizer_type NOT NULL,

    loss_function loss_type NOT NULL,

    learning_rate NUMERIC(14,10) NOT NULL CHECK (
        learning_rate > 0
    ),

    batch_size INTEGER NOT NULL CHECK (batch_size > 0),

    epochs INTEGER NOT NULL CHECK (epochs > 0),

    weight_decay NUMERIC(14,10) NOT NULL DEFAULT 0 CHECK (
        weight_decay >= 0
    ),

    dropout_rate NUMERIC(5,4) NOT NULL DEFAULT 0 CHECK (
        dropout_rate >= 0 AND dropout_rate < 1
    ),

    gradient_clip_norm NUMERIC(12,6) CHECK (
        gradient_clip_norm > 0
    ),

    random_seed BIGINT NOT NULL,

    configuration JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (jsonb_typeof(configuration) = 'object')
);

-- ============================================================================
-- 6. TRAINING RUNS
-- ============================================================================

CREATE TABLE training_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL REFERENCES models(model_id),

    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),

    configuration_id BIGINT NOT NULL
        REFERENCES training_configurations(configuration_id),

    initiated_by BIGINT NOT NULL REFERENCES researchers(researcher_id),

    run_name VARCHAR(150) NOT NULL,

    status training_status NOT NULL DEFAULT 'CREATED',

    device VARCHAR(30) NOT NULL CHECK (
        device IN ('CPU', 'CUDA', 'MPS', 'OTHER')
    ),

    current_epoch INTEGER NOT NULL DEFAULT 0 CHECK (
        current_epoch >= 0
    ),

    best_validation_loss NUMERIC(18,10),

    failure_reason TEXT,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_id, run_name),

    CHECK (
        completed_at IS NULL OR started_at IS NOT NULL
    ),

    CHECK (
        completed_at IS NULL OR completed_at >= started_at
    ),

    CHECK (
        best_validation_loss IS NULL OR best_validation_loss >= 0
    ),

    CHECK (
        (status = 'FAILED' AND failure_reason IS NOT NULL)
        OR
        (status <> 'FAILED')
    )
);

-- ============================================================================
-- 7. EPOCH METRICS
-- ============================================================================

CREATE TABLE epoch_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL REFERENCES training_runs(run_id)
        ON DELETE CASCADE,

    epoch_number INTEGER NOT NULL CHECK (epoch_number > 0),

    training_loss NUMERIC(18,10) NOT NULL CHECK (
        training_loss >= 0
    ),

    validation_loss NUMERIC(18,10) CHECK (
        validation_loss >= 0
    ),

    training_accuracy NUMERIC(8,7) CHECK (
        training_accuracy BETWEEN 0 AND 1
    ),

    validation_accuracy NUMERIC(8,7) CHECK (
        validation_accuracy BETWEEN 0 AND 1
    ),

    learning_rate NUMERIC(14,10) NOT NULL CHECK (
        learning_rate > 0
    ),

    gradient_norm NUMERIC(18,10) CHECK (
        gradient_norm >= 0
    ),

    execution_seconds NUMERIC(14,4) NOT NULL CHECK (
        execution_seconds >= 0
    ),

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(run_id, epoch_number)
);

-- ============================================================================
-- 8. CHECKPOINT MANAGEMENT
-- ============================================================================

CREATE TABLE model_checkpoints (
    checkpoint_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL REFERENCES training_runs(run_id),

    epoch_number INTEGER NOT NULL CHECK (epoch_number >= 0),

    storage_uri TEXT NOT NULL,

    content_sha256 CHAR(64) NOT NULL CHECK (
        content_sha256 ~ '^[a-fA-F0-9]{64}$'
    ),

    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),

    status checkpoint_status NOT NULL DEFAULT 'SAVED',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(run_id, epoch_number)
);

-- ============================================================================
-- 9. PERFORMANCE INDEXES
-- ============================================================================

CREATE INDEX idx_models_project
    ON models(project_id, status);

CREATE INDEX idx_layers_model_sequence
    ON model_layers(model_id, sequence_number);

CREATE INDEX idx_training_runs_status
    ON training_runs(status, created_at DESC);

CREATE INDEX idx_training_runs_model
    ON training_runs(model_id, created_at DESC);

CREATE INDEX idx_epoch_metrics_run_epoch
    ON epoch_metrics(run_id, epoch_number DESC);

CREATE INDEX idx_epoch_metrics_validation
    ON epoch_metrics(validation_loss)
    WHERE validation_loss IS NOT NULL;

CREATE INDEX idx_checkpoints_run
    ON model_checkpoints(run_id, epoch_number DESC);

CREATE INDEX idx_dataset_project
    ON datasets(project_id, dataset_name);

-- ============================================================================
-- 10. TRAINING STATE TRANSITION VALIDATION
-- ============================================================================

CREATE OR REPLACE FUNCTION validate_training_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN

        IF OLD.status IN ('COMPLETED', 'FAILED', 'CANCELLED')
           AND NEW.status <> OLD.status THEN

            RAISE EXCEPTION
                'Terminal training state cannot be changed from %',
                OLD.status;
        END IF;

        IF NEW.status = 'RUNNING'
           AND OLD.status NOT IN ('CREATED') THEN

            RAISE EXCEPTION
                'Only CREATED runs can enter RUNNING state';
        END IF;

        IF NEW.status = 'COMPLETED'
           AND OLD.status <> 'RUNNING' THEN

            RAISE EXCEPTION
                'Only RUNNING runs can complete';
        END IF;

        IF NEW.status = 'FAILED'
           AND OLD.status <> 'RUNNING' THEN

            RAISE EXCEPTION
                'Only RUNNING runs can fail';
        END IF;

        IF NEW.status = 'CANCELLED'
           AND OLD.status NOT IN ('CREATED', 'RUNNING') THEN

            RAISE EXCEPTION
                'Invalid cancellation transition';
        END IF;

    END IF;

    IF NEW.status = 'RUNNING' AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN ('COMPLETED', 'FAILED', 'CANCELLED')
       AND NEW.completed_at IS NULL THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status = 'FAILED'
       AND NULLIF(TRIM(NEW.failure_reason), '') IS NULL THEN
        RAISE EXCEPTION 'Failure reason is required';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_training_state
BEFORE UPDATE ON training_runs
FOR EACH ROW
EXECUTE FUNCTION validate_training_transition();

-- ============================================================================
-- 11. EPOCH RECORD VALIDATION
-- ============================================================================

CREATE OR REPLACE FUNCTION validate_epoch_metric()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    run_status training_status;
    configured_epochs INTEGER;
    previous_epoch INTEGER;
BEGIN
    SELECT
        tr.status,
        tc.epochs
    INTO
        run_status,
        configured_epochs
    FROM training_runs tr
    JOIN training_configurations tc
      ON tc.configuration_id = tr.configuration_id
    WHERE tr.run_id = NEW.run_id
    FOR UPDATE OF tr;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run does not exist';
    END IF;

    IF run_status <> 'RUNNING' THEN
        RAISE EXCEPTION
            'Epoch metrics can only be recorded for RUNNING runs';
    END IF;

    IF NEW.epoch_number > configured_epochs THEN
        RAISE EXCEPTION
            'Epoch exceeds configured training limit';
    END IF;

    SELECT MAX(epoch_number)
    INTO previous_epoch
    FROM epoch_metrics
    WHERE run_id = NEW.run_id;

    IF previous_epoch IS NOT NULL
       AND NEW.epoch_number <> previous_epoch + 1 THEN

        RAISE EXCEPTION
            'Epochs must be recorded sequentially';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_epoch_validation
BEFORE INSERT ON epoch_metrics
FOR EACH ROW
EXECUTE FUNCTION validate_epoch_metric();

-- ============================================================================
-- 12. UPDATE RUN PROGRESS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_run_progress()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE training_runs
    SET
        current_epoch = GREATEST(current_epoch, NEW.epoch_number),

        best_validation_loss = CASE
            WHEN NEW.validation_loss IS NULL THEN best_validation_loss
            WHEN best_validation_loss IS NULL THEN NEW.validation_loss
            ELSE LEAST(best_validation_loss, NEW.validation_loss)
        END

    WHERE run_id = NEW.run_id;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_update_run_progress
AFTER INSERT ON epoch_metrics
FOR EACH ROW
EXECUTE FUNCTION update_run_progress();

-- ============================================================================
-- 13. MODEL ARCHITECTURE VALIDATION
-- ============================================================================

CREATE OR REPLACE FUNCTION validate_model_architecture(
    p_model_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    layer_count INTEGER;
    first_sequence INTEGER;
    last_sequence INTEGER;
BEGIN
    SELECT
        COUNT(*),
        MIN(sequence_number),
        MAX(sequence_number)
    INTO
        layer_count,
        first_sequence,
        last_sequence
    FROM model_layers
    WHERE model_id = p_model_id;

    IF layer_count < 2 THEN
        RETURN FALSE;
    END IF;

    IF first_sequence <> 0 THEN
        RETURN FALSE;
    END IF;

    IF last_sequence <> layer_count - 1 THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
END;
$$;

-- ============================================================================
-- 14. ELIGIBILITY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW training_run_eligibility AS
SELECT
    tr.run_id,
    tr.run_name,
    tr.status,
    m.model_name,
    d.dataset_name,
    tc.optimizer,
    tc.loss_function,
    tc.learning_rate,
    tc.batch_size,
    tc.epochs,

    (
        m.status = 'READY'
        AND r.active
        AND validate_model_architecture(m.model_id)
        AND d.sample_count >= 2
        AND d.feature_count = m.input_features
    ) AS eligible_for_training

FROM training_runs tr

JOIN models m
    ON m.model_id = tr.model_id

JOIN datasets d
    ON d.dataset_id = tr.dataset_id

JOIN training_configurations tc
    ON tc.configuration_id = tr.configuration_id

JOIN researchers r
    ON r.researcher_id = tr.initiated_by;

-- ============================================================================
-- 15. EXPERIMENT REPORTING VIEW
-- ============================================================================

CREATE OR REPLACE VIEW experiment_performance AS
SELECT
    tr.run_id,
    tr.run_name,
    m.model_name,
    tr.status,
    tr.current_epoch,

    ROUND(AVG(em.training_loss), 6) AS average_training_loss,

    ROUND(AVG(em.validation_loss), 6)
        AS average_validation_loss,

    ROUND(MAX(em.validation_accuracy), 6)
        AS best_validation_accuracy,

    ROUND(MIN(em.validation_loss), 6)
        AS minimum_validation_loss,

    SUM(em.execution_seconds) AS total_execution_seconds,

    tr.best_validation_loss

FROM training_runs tr

JOIN models m
    ON m.model_id = tr.model_id

LEFT JOIN epoch_metrics em
    ON em.run_id = tr.run_id

GROUP BY
    tr.run_id,
    tr.run_name,
    m.model_name,
    tr.status,
    tr.current_epoch,
    tr.best_validation_loss;

-- ============================================================================
-- 16. STORED PROCEDURE: START TRAINING
-- ============================================================================

CREATE OR REPLACE PROCEDURE start_training(
    p_run_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    eligibility BOOLEAN;
BEGIN
    SELECT eligible_for_training
    INTO eligibility
    FROM training_run_eligibility
    WHERE run_id = p_run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run not found';
    END IF;

    IF NOT eligibility THEN
        RAISE EXCEPTION
            'Model or dataset is not eligible for training';
    END IF;

    UPDATE training_runs
    SET status = 'RUNNING'
    WHERE run_id = p_run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unable to start training';
    END IF;
END;
$$;

-- ============================================================================
-- 17. STORED PROCEDURE: COMPLETE TRAINING
-- ============================================================================

CREATE OR REPLACE PROCEDURE complete_training(
    p_run_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    current_status training_status;
    actual_epoch INTEGER;
    configured_epoch INTEGER;
BEGIN
    SELECT
        tr.status,
        tr.current_epoch,
        tc.epochs
    INTO
        current_status,
        actual_epoch,
        configured_epoch
    FROM training_runs tr
    JOIN training_configurations tc
      ON tc.configuration_id = tr.configuration_id
    WHERE tr.run_id = p_run_id
    FOR UPDATE OF tr;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run not found';
    END IF;

    IF current_status <> 'RUNNING' THEN
        RAISE EXCEPTION 'Run is not active';
    END IF;

    IF actual_epoch = 0 THEN
        RAISE EXCEPTION 'Cannot complete without training epochs';
    END IF;

    UPDATE training_runs
    SET status = 'COMPLETED'
    WHERE run_id = p_run_id;
END;
$$;

-- ============================================================================
-- 18. SAMPLE RECORDS
-- ============================================================================

INSERT INTO researchers (
    username,
    email,
    role
)
VALUES
    ('researcher_one', 'researcher@example.org', 'RESEARCHER'),
    ('reviewer_one', 'reviewer@example.org', 'REVIEWER'),
    ('administrator', 'admin@example.org', 'ADMIN');

INSERT INTO projects (
    project_name,
    description,
    owner_id
)
VALUES (
    'Neural Classification Laboratory',
    'Feedforward neural network classification experiments.',
    1
);

INSERT INTO datasets (
    project_id,
    dataset_name,
    version,
    sample_count,
    feature_count,
    class_count,
    data_format,
    content_sha256
)
VALUES
(
    1,
    'Synthetic Classification',
    '1.0',
    1000,
    2,
    2,
    'CSV',
    repeat('a', 64)
),
(
    1,
    'Extended Classification',
    '1.0',
    2000,
    2,
    2,
    'CSV',
    repeat('b', 64)
);

INSERT INTO models (
    project_id,
    model_name,
    version,
    description,
    input_features,
    output_classes,
    status,
    created_by
)
VALUES
(
    1,
    'Feedforward MLP',
    '1.0',
    'Three-layer classification network.',
    2,
    2,
    'READY',
    1
),
(
    1,
    'Experimental MLP',
    '1.0',
    'Experimental architecture awaiting configuration.',
    2,
    2,
    'DRAFT',
    1
);

INSERT INTO model_layers (
    model_id,
    layer_name,
    sequence_number,
    type,
    input_dimensions,
    output_dimensions,
    configuration
)
VALUES
(
    1,
    'input_dense',
    0,
    'LINEAR',
    '[2]'::jsonb,
    '[32]'::jsonb,
    '{"in_features":2,"out_features":32}'::jsonb
),
(
    1,
    'hidden_relu',
    1,
    'RELU',
    '[32]'::jsonb,
    '[32]'::jsonb,
    '{}'::jsonb
),
(
    1,
    'hidden_dense',
    2,
    'LINEAR',
    '[32]'::jsonb,
    '[32]'::jsonb,
    '{"in_features":32,"out_features":32}'::jsonb
),
(
    1,
    'output_dense',
    3,
    'LINEAR',
    '[32]'::jsonb,
    '[2]'::jsonb,
    '{"in_features":32,"out_features":2}'::jsonb
);

INSERT INTO training_configurations (
    model_id,
    optimizer,
    loss_function,
    learning_rate,
    batch_size,
    epochs,
    weight_decay,
    dropout_rate,
    gradient_clip_norm,
    random_seed
)
VALUES
(
    1,
    'ADAM',
    'CROSS_ENTROPY',
    0.0100000000,
    32,
    100,
    0.0001000000,
    0.1500,
    5.000000,
    42
),
(
    2,
    'SGD',
    'CROSS_ENTROPY',
    0.0100000000,
    16,
    50,
    0.0000000000,
    0.0000,
    5.000000,
    21
);

INSERT INTO training_runs (
    model_id,
    dataset_id,
    configuration_id,
    initiated_by,
    run_name,
    status,
    device
)
VALUES
(
    1,
    1,
    1,
    1,
    'mlp-adam-baseline',
    'CREATED',
    'CPU'
),
(
    2,
    2,
    2,
    1,
    'experimental-run',
    'CREATED',
    'CPU'
);

-- ============================================================================
-- 19. VALIDATION QUERIES
-- ============================================================================

-- Model architecture validation.
SELECT
    model_id,
    model_name,
    validate_model_architecture(model_id) AS architecture_valid
FROM models;

-- Training eligibility.
SELECT *
FROM training_run_eligibility
ORDER BY run_id;

-- Model layer inspection.
SELECT
    m.model_name,
    ml.sequence_number,
    ml.layer_name,
    ml.type,
    ml.input_dimensions,
    ml.output_dimensions
FROM model_layers ml
JOIN models m
    ON m.model_id = ml.model_id
ORDER BY m.model_id, ml.sequence_number;

-- ============================================================================
-- 20. TRANSACTIONAL TRAINING WORKFLOW
-- ============================================================================

BEGIN;

CALL start_training(1);

INSERT INTO epoch_metrics (
    run_id,
    epoch_number,
    training_loss,
    validation_loss,
    training_accuracy,
    validation_accuracy,
    learning_rate,
    gradient_norm,
    execution_seconds
)
VALUES
(
    1,
    1,
    0.6820000000,
    0.6710000000,
    0.6100000,
    0.6300000,
    0.0100000000,
    1.2500000000,
    0.4200
),
(
    1,
    2,
    0.5310000000,
    0.5180000000,
    0.7900000,
    0.8100000,
    0.0100000000,
    0.9500000000,
    0.3900
),
(
    1,
    3,
    0.3910000000,
    0.3760000000,
    0.8700000,
    0.8900000,
    0.0100000000,
    0.7200000000,
    0.3700
);

INSERT INTO model_checkpoints (
    run_id,
    epoch_number,
    storage_uri,
    content_sha256,
    file_size_bytes,
    status
)
VALUES (
    1,
    3,
    'file:///models/mlp/best-model.pt',
    repeat('c', 64),
    1048576,
    'VERIFIED'
);

-- The sample run has only three recorded epochs.
-- Mark it completed as a demonstration of a partial training run.
UPDATE training_runs
SET status = 'COMPLETED'
WHERE run_id = 1;

COMMIT;

-- ============================================================================
-- 21. FAILURE HANDLING EXAMPLE
-- ============================================================================

-- Demonstrates a failed training run with a mandatory failure reason.
-- This transaction is rolled back so the main sample data remains intact.

BEGIN;

UPDATE training_runs
SET status = 'RUNNING'
WHERE run_id = 2;

UPDATE training_runs
SET
    status = 'FAILED',
    failure_reason = 'Simulated numerical instability: non-finite loss.'
WHERE run_id = 2;

ROLLBACK;

-- ============================================================================
-- 22. RETRIEVAL AND REPORTING
-- ============================================================================

-- Training progress.
SELECT
    run_id,
    run_name,
    status,
    current_epoch,
    best_validation_loss
FROM training_runs
ORDER BY created_at DESC;

-- Epoch-level performance.
SELECT
    run_id,
    epoch_number,
    training_loss,
    validation_loss,
    training_accuracy,
    validation_accuracy
FROM epoch_metrics
ORDER BY run_id, epoch_number;

-- Aggregate experiment performance.
SELECT *
FROM experiment_performance
ORDER BY minimum_validation_loss NULLS LAST;

-- Best validation epoch for every run.
WITH ranked_epochs AS (
    SELECT
        run_id,
        epoch_number,
        validation_loss,
        validation_accuracy,

        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY validation_loss ASC NULLS LAST
        ) AS ranking

    FROM epoch_metrics
    WHERE validation_loss IS NOT NULL
)
SELECT
    run_id,
    epoch_number AS best_epoch,
    validation_loss,
    validation_accuracy
FROM ranked_epochs
WHERE ranking = 1;

-- Optimizer comparison.
SELECT
    tc.optimizer,
    COUNT(DISTINCT tr.run_id) AS run_count,
    ROUND(AVG(em.validation_loss), 6) AS average_validation_loss,
    ROUND(AVG(em.validation_accuracy), 6) AS average_validation_accuracy
FROM training_runs tr
JOIN training_configurations tc
    ON tc.configuration_id = tr.configuration_id
LEFT JOIN epoch_metrics em
    ON em.run_id = tr.run_id
GROUP BY tc.optimizer
ORDER BY average_validation_loss NULLS LAST;

-- Checkpoint verification.
SELECT
    mc.checkpoint_id,
    tr.run_name,
    mc.epoch_number,
    mc.status,
    mc.file_size_bytes
FROM model_checkpoints mc
JOIN training_runs tr
    ON tr.run_id = mc.run_id
ORDER BY mc.created_at DESC;

-- ============================================================================
-- 23. OPERATIONAL DATA QUALITY CHECKS
-- ============================================================================

-- Detect missing epoch records.
SELECT
    tr.run_id,
    tr.run_name,
    tr.current_epoch,
    COUNT(em.metric_id) AS recorded_epochs
FROM training_runs tr
LEFT JOIN epoch_metrics em
    ON em.run_id = tr.run_id
GROUP BY tr.run_id, tr.run_name, tr.current_epoch
HAVING COUNT(em.metric_id) <> tr.current_epoch;

-- Identify failed experiments.
SELECT
    run_id,
    run_name,
    failure_reason,
    completed_at
FROM training_runs
WHERE status = 'FAILED';

-- Identify models without valid architectures.
SELECT
    model_id,
    model_name
FROM models
WHERE NOT validate_model_architecture(model_id);

-- ============================================================================
-- 24. ROLE-BASED ACCESS CONTROL EXAMPLE
-- ============================================================================

-- Create a reporting role when permitted by the database administrator.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'neural_lab_reader'
    ) THEN
        CREATE ROLE neural_lab_reader NOLOGIN;
    END IF;
END;
$$;

GRANT USAGE ON SCHEMA neural_lab TO neural_lab_reader;

GRANT SELECT ON
    projects,
    datasets,
    models,
    model_layers,
    training_configurations,
    training_runs,
    epoch_metrics,
    model_checkpoints,
    training_run_eligibility,
    experiment_performance
TO neural_lab_reader;

COMMIT;
