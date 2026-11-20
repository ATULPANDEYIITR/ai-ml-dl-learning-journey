-- ============================================================
-- DEEP LEARNING TRAINING MANAGEMENT DATABASE
-- PostgreSQL 14+
--
-- Domain:
-- Experiments -> Datasets -> Training Runs -> Epochs
--             -> Batches -> Metrics -> Checkpoints
--
-- Features:
-- Training configuration validation
-- Epoch and batch tracking
-- Learning-rate scheduling
-- Validation metrics
-- Checkpoint lifecycle
-- Recovery metadata
-- Early stopping
-- Training reports
-- Transactional state transitions
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS deep_learning;

SET search_path TO deep_learning, public;

-- ------------------------------------------------------------
-- ENUMERATIONS
-- ------------------------------------------------------------

CREATE TYPE run_status AS ENUM (
    'created',
    'queued',
    'running',
    'completed',
    'early_stopped',
    'failed',
    'cancelled'
);

CREATE TYPE checkpoint_status AS ENUM (
    'available',
    'restored',
    'superseded',
    'corrupted'
);

CREATE TYPE metric_type AS ENUM (
    'training_loss',
    'validation_loss',
    'training_accuracy',
    'validation_accuracy',
    'gradient_norm'
);

CREATE TYPE batch_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

-- ------------------------------------------------------------
-- REPOSITORY-STYLE EXPERIMENT OWNERSHIP
-- ------------------------------------------------------------

CREATE TABLE experiment (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_name VARCHAR(150) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    owner_name VARCHAR(120) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT experiment_name_not_empty
        CHECK (length(trim(experiment_name)) > 0)
);

CREATE TABLE dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(150) NOT NULL,
    dataset_version VARCHAR(40) NOT NULL,
    sample_count BIGINT NOT NULL,
    feature_count INTEGER NOT NULL,
    training_samples BIGINT NOT NULL,
    validation_samples BIGINT NOT NULL,
    test_samples BIGINT NOT NULL DEFAULT 0,
    content_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT dataset_version_unique
        UNIQUE (dataset_name, dataset_version),

    CONSTRAINT valid_dataset_counts
        CHECK (
            sample_count > 0
            AND feature_count > 0
            AND training_samples > 0
            AND validation_samples > 0
            AND test_samples >= 0
            AND training_samples + validation_samples + test_samples
                = sample_count
        ),

    CONSTRAINT dataset_hash_not_empty
        CHECK (length(trim(content_hash)) > 0)
);

-- ------------------------------------------------------------
-- TRAINING RUNS
-- ------------------------------------------------------------

CREATE TABLE training_run (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiment(experiment_id),

    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id),

    run_name VARCHAR(150) NOT NULL UNIQUE,

    model_architecture VARCHAR(100) NOT NULL,

    status run_status NOT NULL DEFAULT 'created',

    total_epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,

    learning_rate NUMERIC(16, 10) NOT NULL,
    minimum_learning_rate NUMERIC(16, 10) NOT NULL DEFAULT 0.00001,
    learning_rate_decay NUMERIC(12, 10) NOT NULL DEFAULT 1.0,

    optimizer VARCHAR(40) NOT NULL DEFAULT 'SGD',
    loss_function VARCHAR(60) NOT NULL DEFAULT 'MSE',

    early_stopping_patience INTEGER NOT NULL DEFAULT 8,
    min_delta NUMERIC(16, 10) NOT NULL DEFAULT 0.00000001,

    random_seed BIGINT NOT NULL DEFAULT 42,

    completed_epochs INTEGER NOT NULL DEFAULT 0,

    best_epoch INTEGER,
    best_validation_loss NUMERIC(20, 12),

    failure_reason TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_training_configuration
        CHECK (
            total_epochs > 0
            AND batch_size > 0
            AND learning_rate > 0
            AND minimum_learning_rate > 0
            AND learning_rate_decay > 0
            AND learning_rate_decay <= 1
            AND early_stopping_patience > 0
            AND min_delta >= 0
        ),

    CONSTRAINT valid_training_progress
        CHECK (
            completed_epochs >= 0
            AND completed_epochs <= total_epochs
            AND (
                best_epoch IS NULL
                OR (
                    best_epoch > 0
                    AND best_epoch <= completed_epochs
                )
            )
        ),

    CONSTRAINT valid_best_loss
        CHECK (
            best_validation_loss IS NULL
            OR best_validation_loss >= 0
        )
);

CREATE INDEX idx_training_run_experiment
    ON training_run(experiment_id, created_at DESC);

CREATE INDEX idx_training_run_status
    ON training_run(status, created_at DESC);

CREATE INDEX idx_training_run_dataset
    ON training_run(dataset_id);

-- ------------------------------------------------------------
-- EPOCH EXECUTION
-- ------------------------------------------------------------

CREATE TABLE training_epoch (
    epoch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES training_run(run_id) ON DELETE CASCADE,

    epoch_number INTEGER NOT NULL,

    learning_rate NUMERIC(16, 10) NOT NULL,

    training_loss NUMERIC(20, 12),
    validation_loss NUMERIC(20, 12),

    training_accuracy NUMERIC(12, 10),
    validation_accuracy NUMERIC(12, 10),

    gradient_norm NUMERIC(20, 12),

    total_batches INTEGER NOT NULL,

    completed_batches INTEGER NOT NULL DEFAULT 0,

    epoch_started_at TIMESTAMPTZ,
    epoch_completed_at TIMESTAMPTZ,

    CONSTRAINT unique_run_epoch
        UNIQUE(run_id, epoch_number),

    CONSTRAINT valid_epoch_number
        CHECK (epoch_number > 0),

    CONSTRAINT valid_epoch_learning_rate
        CHECK (learning_rate > 0),

    CONSTRAINT valid_epoch_batches
        CHECK (
            total_batches > 0
            AND completed_batches >= 0
            AND completed_batches <= total_batches
        ),

    CONSTRAINT valid_epoch_losses
        CHECK (
            (training_loss IS NULL OR training_loss >= 0)
            AND (validation_loss IS NULL OR validation_loss >= 0)
            AND (gradient_norm IS NULL OR gradient_norm >= 0)
        ),

    CONSTRAINT valid_epoch_accuracy
        CHECK (
            (training_accuracy IS NULL OR
                training_accuracy BETWEEN 0 AND 1)
            AND
            (validation_accuracy IS NULL OR
                validation_accuracy BETWEEN 0 AND 1)
        )
);

CREATE INDEX idx_training_epoch_history
    ON training_epoch(run_id, epoch_number);

-- ------------------------------------------------------------
-- MINI-BATCH EXECUTION
-- ------------------------------------------------------------

CREATE TABLE training_batch (
    batch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    epoch_id BIGINT NOT NULL
        REFERENCES training_epoch(epoch_id) ON DELETE CASCADE,

    batch_number INTEGER NOT NULL,

    sample_count INTEGER NOT NULL,

    batch_loss NUMERIC(20, 12),

    status batch_status NOT NULL DEFAULT 'pending',

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    error_message TEXT,

    CONSTRAINT unique_epoch_batch
        UNIQUE(epoch_id, batch_number),

    CONSTRAINT valid_batch_number
        CHECK (batch_number > 0 AND sample_count > 0),

    CONSTRAINT valid_batch_loss
        CHECK (batch_loss IS NULL OR batch_loss >= 0)
);

CREATE INDEX idx_batch_epoch_status
    ON training_batch(epoch_id, status);

-- ------------------------------------------------------------
-- METRIC HISTORY
-- ------------------------------------------------------------

CREATE TABLE training_metric (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    epoch_id BIGINT NOT NULL
        REFERENCES training_epoch(epoch_id) ON DELETE CASCADE,

    metric metric_type NOT NULL,

    metric_value NUMERIC(24, 12) NOT NULL,

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT metric_value_finite
        CHECK (
            metric_value > '-Infinity'::numeric
            AND metric_value < 'Infinity'::numeric
            AND metric_value <> 'NaN'::numeric
        ),

    CONSTRAINT accuracy_range
        CHECK (
            metric NOT IN (
                'training_accuracy',
                'validation_accuracy'
            )
            OR metric_value BETWEEN 0 AND 1
        ),

    CONSTRAINT unique_epoch_metric
        UNIQUE(epoch_id, metric)
);

CREATE INDEX idx_metric_type_time
    ON training_metric(metric, recorded_at DESC);

-- ------------------------------------------------------------
-- CHECKPOINT MANAGEMENT
-- ------------------------------------------------------------

CREATE TABLE model_checkpoint (
    checkpoint_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES training_run(run_id) ON DELETE CASCADE,

    epoch_id BIGINT NOT NULL
        REFERENCES training_epoch(epoch_id),

    checkpoint_name VARCHAR(200) NOT NULL UNIQUE,

    storage_uri TEXT NOT NULL,

    checksum VARCHAR(128) NOT NULL,

    model_version VARCHAR(50) NOT NULL,

    checkpoint_status checkpoint_status NOT NULL DEFAULT 'available',

    file_size_bytes BIGINT NOT NULL,

    validation_loss NUMERIC(20, 12) NOT NULL,

    is_best BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_checkpoint_size
        CHECK (file_size_bytes > 0),

    CONSTRAINT valid_checkpoint_loss
        CHECK (validation_loss >= 0),

    CONSTRAINT checkpoint_checksum_not_empty
        CHECK (length(trim(checksum)) > 0),

    CONSTRAINT checkpoint_uri_not_empty
        CHECK (length(trim(storage_uri)) > 0)
);

CREATE UNIQUE INDEX idx_one_best_checkpoint_per_run
    ON model_checkpoint(run_id)
    WHERE is_best = TRUE;

CREATE INDEX idx_checkpoint_run_epoch
    ON model_checkpoint(run_id, epoch_id DESC);

-- ------------------------------------------------------------
-- AUDIT EVENTS
-- ------------------------------------------------------------

CREATE TABLE training_audit (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES training_run(run_id) ON DELETE CASCADE,

    event_type VARCHAR(80) NOT NULL,

    event_details JSONB NOT NULL DEFAULT '{}'::jsonb,

    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_audit_run
    ON training_audit(run_id, occurred_at DESC);

-- ------------------------------------------------------------
-- STATE TRANSITION VALIDATION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_training_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status IS DISTINCT FROM OLD.status THEN

        IF NOT (
            (OLD.status = 'created'
                AND NEW.status IN ('queued', 'running', 'cancelled', 'failed'))
            OR
            (OLD.status = 'queued'
                AND NEW.status IN ('running', 'cancelled', 'failed'))
            OR
            (OLD.status = 'running'
                AND NEW.status IN (
                    'completed',
                    'early_stopped',
                    'failed',
                    'cancelled'
                ))
            OR
            (OLD.status IN ('completed', 'early_stopped', 'failed', 'cancelled')
                AND NEW.status = 'queued')
        ) THEN
            RAISE EXCEPTION
                'Invalid training transition: % -> %',
                OLD.status,
                NEW.status;
        END IF;

    END IF;

    IF NEW.completed_epochs < OLD.completed_epochs THEN
        IF NEW.status <> 'queued' THEN
            RAISE EXCEPTION
                'Training progress cannot decrease outside a new execution.';
        END IF;
    END IF;

    IF NEW.status = 'running' AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN (
        'completed',
        'early_stopped',
        'failed',
        'cancelled'
    ) THEN
        NEW.completed_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status = 'failed'
        AND COALESCE(length(trim(NEW.failure_reason)), 0) = 0 THEN
        RAISE EXCEPTION 'Failure reason is required.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_training_state_transition
BEFORE UPDATE ON training_run
FOR EACH ROW
EXECUTE FUNCTION validate_training_transition();

-- ------------------------------------------------------------
-- EPOCH PROGRESS INTEGRITY
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_epoch_progress()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_batches INTEGER;
    expected_samples BIGINT;
    completed_samples BIGINT;
BEGIN
    SELECT
        CEIL(d.training_samples::NUMERIC / r.batch_size)::INTEGER,
        d.training_samples
    INTO expected_batches, expected_samples
    FROM training_run r
    JOIN dataset d ON d.dataset_id = r.dataset_id
    WHERE r.run_id = NEW.run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run or dataset does not exist.';
    END IF;

    IF NEW.total_batches <> expected_batches THEN
        RAISE EXCEPTION
            'Expected % batches, received %.',
            expected_batches,
            NEW.total_batches;
    END IF;

    IF NEW.completed_batches = NEW.total_batches THEN
        SELECT COALESCE(SUM(sample_count), 0)
        INTO completed_samples
        FROM training_batch
        WHERE epoch_id = NEW.epoch_id
          AND status = 'completed';

        IF NEW.epoch_id IS NOT NULL
           AND completed_samples > expected_samples THEN
            RAISE EXCEPTION 'Completed batch sample count exceeds dataset size.';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_epoch_progress
BEFORE INSERT OR UPDATE ON training_epoch
FOR EACH ROW
EXECUTE FUNCTION validate_epoch_progress();

-- ------------------------------------------------------------
-- TRAINING RUN LIFECYCLE PROCEDURE
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE transition_training_run(
    p_run_id BIGINT,
    p_new_status run_status,
    p_failure_reason TEXT DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE training_run
    SET
        status = p_new_status,
        failure_reason = CASE
            WHEN p_new_status = 'failed' THEN p_failure_reason
            ELSE failure_reason
        END
    WHERE run_id = p_run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run % does not exist.', p_run_id;
    END IF;

    INSERT INTO training_audit (
        run_id,
        event_type,
        event_details
    )
    VALUES (
        p_run_id,
        'STATUS_TRANSITION',
        jsonb_build_object(
            'new_status', p_new_status,
            'failure_reason', p_failure_reason
        )
    );
END;
$$;

-- ------------------------------------------------------------
-- EPOCH REGISTRATION PROCEDURE
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE register_training_epoch(
    p_run_id BIGINT,
    p_epoch_number INTEGER,
    p_learning_rate NUMERIC,
    p_training_loss NUMERIC,
    p_validation_loss NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_run training_run%ROWTYPE;
    v_expected_batches INTEGER;
    v_epoch_id BIGINT;
BEGIN
    SELECT *
    INTO v_run
    FROM training_run
    WHERE run_id = p_run_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Training run not found.';
    END IF;

    IF v_run.status <> 'running' THEN
        RAISE EXCEPTION 'Only running training runs accept epochs.';
    END IF;

    IF p_epoch_number <> v_run.completed_epochs + 1 THEN
        RAISE EXCEPTION 'Epochs must be registered sequentially.';
    END IF;

    IF p_epoch_number > v_run.total_epochs THEN
        RAISE EXCEPTION 'Configured epoch limit exceeded.';
    END IF;

    IF p_learning_rate <= 0 OR
       p_training_loss < 0 OR
       p_validation_loss < 0 THEN
        RAISE EXCEPTION 'Invalid epoch metrics.';
    END IF;

    v_expected_batches := (
        SELECT CEIL(d.training_samples::NUMERIC / v_run.batch_size)::INTEGER
        FROM dataset d
        WHERE d.dataset_id = v_run.dataset_id
    );

    INSERT INTO training_epoch (
        run_id,
        epoch_number,
        learning_rate,
        training_loss,
        validation_loss,
        total_batches,
        completed_batches,
        epoch_started_at,
        epoch_completed_at
    )
    VALUES (
        p_run_id,
        p_epoch_number,
        p_learning_rate,
        p_training_loss,
        p_validation_loss,
        v_expected_batches,
        v_expected_batches,
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    )
    RETURNING epoch_id INTO v_epoch_id;

    INSERT INTO training_metric(epoch_id, metric, metric_value)
    VALUES
        (v_epoch_id, 'training_loss', p_training_loss),
        (v_epoch_id, 'validation_loss', p_validation_loss);

    UPDATE training_run
    SET
        completed_epochs = p_epoch_number,
        best_epoch = CASE
            WHEN best_validation_loss IS NULL
                OR p_validation_loss < best_validation_loss
            THEN p_epoch_number
            ELSE best_epoch
        END,
        best_validation_loss = LEAST(
            COALESCE(best_validation_loss, p_validation_loss),
            p_validation_loss
        )
    WHERE run_id = p_run_id;

    INSERT INTO training_audit (
        run_id,
        event_type,
        event_details
    )
    VALUES (
        p_run_id,
        'EPOCH_COMPLETED',
        jsonb_build_object(
            'epoch', p_epoch_number,
            'training_loss', p_training_loss,
            'validation_loss', p_validation_loss,
            'learning_rate', p_learning_rate
        )
    );
END;
$$;

-- ------------------------------------------------------------
-- CHECKPOINT REGISTRATION PROCEDURE
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE register_checkpoint(
    p_run_id BIGINT,
    p_epoch_number INTEGER,
    p_checkpoint_name VARCHAR,
    p_storage_uri TEXT,
    p_checksum VARCHAR,
    p_model_version VARCHAR,
    p_file_size BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_epoch_id BIGINT;
    v_validation_loss NUMERIC;
BEGIN
    SELECT epoch_id, validation_loss
    INTO v_epoch_id, v_validation_loss
    FROM training_epoch
    WHERE run_id = p_run_id
      AND epoch_number = p_epoch_number;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Checkpoint epoch does not exist.';
    END IF;

    IF p_file_size <= 0 OR
       length(trim(p_checksum)) = 0 OR
       length(trim(p_storage_uri)) = 0 THEN
        RAISE EXCEPTION 'Invalid checkpoint metadata.';
    END IF;

    UPDATE model_checkpoint
    SET is_best = FALSE
    WHERE run_id = p_run_id
      AND is_best = TRUE;

    INSERT INTO model_checkpoint (
        run_id,
        epoch_id,
        checkpoint_name,
        storage_uri,
        checksum,
        model_version,
        file_size_bytes,
        validation_loss,
        is_best
    )
    VALUES (
        p_run_id,
        v_epoch_id,
        p_checkpoint_name,
        p_storage_uri,
        p_checksum,
        p_model_version,
        p_file_size,
        v_validation_loss,
        TRUE
    );

    INSERT INTO training_audit (
        run_id,
        event_type,
        event_details
    )
    VALUES (
        p_run_id,
        'CHECKPOINT_REGISTERED',
        jsonb_build_object(
            'epoch', p_epoch_number,
            'checkpoint', p_checkpoint_name
        )
    );
END;
$$;

-- ------------------------------------------------------------
-- MERGE-FREE TRAINING REPORTING VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW training_run_report AS
SELECT
    r.run_id,
    r.run_name,
    e.experiment_name,
    d.dataset_name,
    d.dataset_version,
    r.model_architecture,
    r.status,
    r.total_epochs,
    r.completed_epochs,
    r.batch_size,
    r.learning_rate,
    r.best_epoch,
    r.best_validation_loss,
    COUNT(DISTINCT te.epoch_id) AS recorded_epochs,
    COUNT(DISTINCT cp.checkpoint_id) AS checkpoint_count,
    MIN(te.training_loss) AS minimum_training_loss,
    MIN(te.validation_loss) AS minimum_validation_loss,
    r.started_at,
    r.completed_at
FROM training_run r
JOIN experiment e ON e.experiment_id = r.experiment_id
JOIN dataset d ON d.dataset_id = r.dataset_id
LEFT JOIN training_epoch te ON te.run_id = r.run_id
LEFT JOIN model_checkpoint cp ON cp.run_id = r.run_id
GROUP BY
    r.run_id,
    e.experiment_name,
    d.dataset_name,
    d.dataset_version;

-- ------------------------------------------------------------
-- ELIGIBILITY VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW training_completion_eligibility AS
SELECT
    r.run_id,
    r.run_name,
    r.status,
    r.total_epochs,
    r.completed_epochs,
    CASE
        WHEN r.status <> 'running'
            THEN FALSE
        WHEN r.completed_epochs >= r.total_epochs
            THEN TRUE
        ELSE FALSE
    END AS epoch_limit_reached,
    CASE
        WHEN r.best_validation_loss IS NOT NULL
            THEN TRUE
        ELSE FALSE
    END AS has_validation_result
FROM training_run r;

-- ------------------------------------------------------------
-- SAMPLE EXPERIMENTS AND DATASETS
-- ------------------------------------------------------------

INSERT INTO experiment (
    experiment_name,
    description,
    owner_name
)
VALUES
(
    'Image Classification Baseline',
    'Mini-batch optimization and checkpoint evaluation.',
    'Research Team'
),
(
    'Learning Rate Evaluation',
    'Comparison of learning-rate configurations.',
    'ML Engineering'
);

INSERT INTO dataset (
    dataset_name,
    dataset_version,
    sample_count,
    feature_count,
    training_samples,
    validation_samples,
    test_samples,
    content_hash
)
VALUES
(
    'Synthetic Classification Dataset',
    '1.0',
    1000,
    32,
    700,
    200,
    100,
    'sha256:synthetic-dataset-v1'
),
(
    'Synthetic Regression Dataset',
    '1.0',
    500,
    12,
    350,
    100,
    50,
    'sha256:synthetic-regression-v1'
);

-- ------------------------------------------------------------
-- SAMPLE TRAINING RUNS
-- ------------------------------------------------------------

INSERT INTO training_run (
    experiment_id,
    dataset_id,
    run_name,
    model_architecture,
    status,
    total_epochs,
    batch_size,
    learning_rate,
    minimum_learning_rate,
    learning_rate_decay,
    optimizer,
    loss_function,
    early_stopping_patience
)
VALUES
(
    1, 1,
    'cnn-baseline-run-001',
    'CNN',
    'created',
    20,
    32,
    0.0100000000,
    0.0000100000,
    0.9500000000,
    'SGD',
    'CrossEntropy',
    5
),
(
    1, 1,
    'cnn-baseline-run-002',
    'CNN',
    'created',
    30,
    64,
    0.0010000000,
    0.0000100000,
    0.9800000000,
    'Adam',
    'CrossEntropy',
    8
),
(
    2, 2,
    'regression-experiment-001',
    'LinearRegression',
    'created',
    15,
    16,
    0.0500000000,
    0.0000100000,
    0.9800000000,
    'SGD',
    'MSE',
    4
);

-- ------------------------------------------------------------
-- SAMPLE VALID TRAINING WORKFLOW
-- ------------------------------------------------------------

CALL transition_training_run(1, 'queued');

CALL transition_training_run(1, 'running');

CALL register_training_epoch(
    1,
    1,
    0.0100000000,
    1.240000000000,
    1.310000000000
);

CALL register_training_epoch(
    1,
    2,
    0.0095000000,
    0.940000000000,
    1.020000000000
);

CALL register_training_epoch(
    1,
    3,
    0.0090250000,
    0.730000000000,
    0.810000000000
);

CALL register_checkpoint(
    1,
    2,
    'cnn-baseline-epoch-002',
    's3://training-checkpoints/run-001/epoch-002',
    'sha256:checkpoint-002',
    'cnn-v1',
    1048576
);

CALL register_checkpoint(
    1,
    3,
    'cnn-baseline-epoch-003',
    's3://training-checkpoints/run-001/epoch-003',
    'sha256:checkpoint-003',
    'cnn-v1',
    2097152
);

-- ------------------------------------------------------------
-- BATCH EXECUTION RECORDS
-- ------------------------------------------------------------

INSERT INTO training_batch (
    epoch_id,
    batch_number,
    sample_count,
    batch_loss,
    status
)
SELECT
    te.epoch_id,
    batch_number,
    CASE
        WHEN batch_number = te.total_batches
        THEN (
            SELECT
                d.training_samples -
                (te.total_batches - 1) * r.batch_size
            FROM training_run r
            JOIN dataset d ON d.dataset_id = r.dataset_id
            WHERE r.run_id = te.run_id
        )
        ELSE (
            SELECT batch_size
            FROM training_run
            WHERE run_id = te.run_id
        )
    END,
    te.training_loss,
    'completed'
FROM training_epoch te
CROSS JOIN generate_series(1, 4) AS batch_number
WHERE te.run_id = 1
  AND te.epoch_number = 1
  AND batch_number <= te.total_batches;

-- ------------------------------------------------------------
-- TRANSACTIONAL STATUS COMPLETION
-- ------------------------------------------------------------

CALL transition_training_run(1, 'completed');

-- Failed run with an explicit failure reason.
CALL transition_training_run(2, 'queued');
CALL transition_training_run(2, 'running');
CALL transition_training_run(
    2,
    'failed',
    'Numerical instability caused non-finite gradient values.'
);

-- Early-stopped run.
CALL transition_training_run(3, 'queued');
CALL transition_training_run(3, 'running');

CALL register_training_epoch(
    3,
    1,
    0.0500000000,
    0.850000000000,
    0.900000000000
);

CALL transition_training_run(3, 'early_stopped');

COMMIT;

-- ============================================================
-- OPERATIONAL QUERIES
-- ============================================================

-- 1. Retrieve training runs and their states.

SELECT *
FROM training_run_report
ORDER BY created_at DESC;

-- 2. Inspect epoch history.

SELECT
    r.run_name,
    te.epoch_number,
    te.learning_rate,
    te.training_loss,
    te.validation_loss,
    te.total_batches,
    te.completed_batches
FROM training_epoch te
JOIN training_run r ON r.run_id = te.run_id
ORDER BY r.run_id, te.epoch_number;

-- 3. Compare training and validation loss.

SELECT
    run_id,
    epoch_number,
    training_loss,
    validation_loss,
    validation_loss - training_loss AS generalization_gap
FROM training_epoch
WHERE training_loss IS NOT NULL
  AND validation_loss IS NOT NULL
ORDER BY run_id, epoch_number;

-- 4. Aggregate epoch metrics.

SELECT
    run_id,
    COUNT(*) AS epochs_recorded,
    AVG(training_loss) AS average_training_loss,
    AVG(validation_loss) AS average_validation_loss,
    MIN(validation_loss) AS best_validation_loss
FROM training_epoch
GROUP BY run_id
ORDER BY run_id;

-- 5. Identify the best epoch using a window function.

WITH ranked_epochs AS (
    SELECT
        run_id,
        epoch_number,
        validation_loss,
        RANK() OVER (
            PARTITION BY run_id
            ORDER BY validation_loss ASC
        ) AS loss_rank
    FROM training_epoch
    WHERE validation_loss IS NOT NULL
)
SELECT *
FROM ranked_epochs
WHERE loss_rank = 1;

-- 6. Calculate the effective learning rate at a given epoch.

SELECT
    run_id,
    run_name,
    total_epochs,
    learning_rate,
    learning_rate_decay,
    ROUND(
        GREATEST(
            minimum_learning_rate,
            learning_rate * POWER(
                learning_rate_decay,
                9
            )
        ),
        10
    ) AS learning_rate_at_epoch_10
FROM training_run;

-- 7. Retrieve the latest available checkpoint.

SELECT DISTINCT ON (run_id)
    run_id,
    checkpoint_name,
    epoch_id,
    validation_loss,
    checksum,
    storage_uri,
    is_best
FROM model_checkpoint
WHERE checkpoint_status IN ('available', 'restored')
ORDER BY run_id, created_at DESC;

-- 8. Retrieve the best checkpoint for every run.

SELECT
    run_id,
    checkpoint_name,
    validation_loss,
    model_version
FROM model_checkpoint
WHERE is_best = TRUE;

-- 9. Identify training failures.

SELECT
    run_name,
    status,
    failure_reason,
    started_at,
    completed_at
FROM training_run
WHERE status IN ('failed', 'cancelled');

-- 10. Detect incomplete epoch execution.

SELECT
    r.run_name,
    te.epoch_number,
    te.total_batches,
    te.completed_batches
FROM training_epoch te
JOIN training_run r ON r.run_id = te.run_id
WHERE te.completed_batches < te.total_batches;

-- 11. Training eligibility report.

SELECT *
FROM training_completion_eligibility
ORDER BY run_id;

-- 12. Checkpoint recovery candidates.

SELECT
    r.run_name,
    cp.checkpoint_name,
    cp.epoch_id,
    cp.validation_loss,
    cp.checkpoint_status
FROM model_checkpoint cp
JOIN training_run r ON r.run_id = cp.run_id
WHERE cp.checkpoint_status = 'available'
ORDER BY r.run_name, cp.validation_loss ASC;

-- 13. Audit trail.

SELECT
    run_id,
    event_type,
    event_details,
    occurred_at
FROM training_audit
ORDER BY occurred_at DESC;

-- 14. Dataset utilization and effective batch counts.

SELECT
    r.run_name,
    d.sample_count,
    d.training_samples,
    r.batch_size,
    CEIL(
        d.training_samples::NUMERIC / r.batch_size
    ) AS batches_per_epoch
FROM training_run r
JOIN dataset d ON d.dataset_id = r.dataset_id;

-- 15. Training versus validation loss trend.

SELECT
    run_id,
    epoch_number,
    training_loss,
    validation_loss,
    CASE
        WHEN validation_loss > training_loss
            THEN 'VALIDATION_LOSS_HIGHER'
        WHEN validation_loss < training_loss
            THEN 'VALIDATION_LOSS_LOWER'
        ELSE 'EQUAL'
    END AS convergence_indicator
FROM training_epoch
ORDER BY run_id, epoch_number;
