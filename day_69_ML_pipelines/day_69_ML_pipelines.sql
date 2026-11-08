-- ============================================================
-- ML PIPELINE MANAGEMENT AND EXECUTION DATABASE
-- PostgreSQL 15+
--
-- Covers:
--   Pipeline definitions and versions
--   Transformer configurations
--   Preprocessing workflows
--   Training datasets
--   Train/validation/test partitions
--   Model configurations
--   Pipeline execution and stage tracking
--   Evaluation metrics
--   Model artifacts
--   Inference requests
--   Validation and eligibility rules
--   Transaction-safe workflow processing
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS ml_platform;

SET search_path TO ml_platform, public;

-- ------------------------------------------------------------
-- Enumerated workflow states
-- ------------------------------------------------------------

CREATE TYPE pipeline_status AS ENUM (
    'draft',
    'validated',
    'ready',
    'retired'
);

CREATE TYPE run_status AS ENUM (
    'queued',
    'running',
    'succeeded',
    'failed',
    'cancelled'
);

CREATE TYPE stage_status AS ENUM (
    'pending',
    'running',
    'succeeded',
    'failed',
    'skipped'
);

CREATE TYPE split_type AS ENUM (
    'train',
    'validation',
    'test'
);

CREATE TYPE artifact_type AS ENUM (
    'preprocessor',
    'model',
    'complete_pipeline',
    'metrics',
    'feature_schema'
);

-- ------------------------------------------------------------
-- Organizations and users
-- ------------------------------------------------------------

CREATE TABLE organizations (
    organization_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_name VARCHAR(150) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE platform_users (
    user_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id BIGINT NOT NULL
        REFERENCES organizations(organization_id),
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (organization_id, username),
    UNIQUE (organization_id, email)
);

-- ------------------------------------------------------------
-- Dataset registry
-- ------------------------------------------------------------

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id BIGINT NOT NULL
        REFERENCES organizations(organization_id),
    dataset_name VARCHAR(150) NOT NULL,
    version VARCHAR(40) NOT NULL,
    storage_uri TEXT NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    record_count BIGINT NOT NULL CHECK (record_count > 0),
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    target_column VARCHAR(100) NOT NULL,
    classification_type VARCHAR(30) NOT NULL
        CHECK (
            classification_type IN (
                'binary',
                'multiclass',
                'regression'
            )
        ),
    schema_definition JSONB NOT NULL,
    approved BOOLEAN NOT NULL DEFAULT FALSE,
    created_by BIGINT NOT NULL
        REFERENCES platform_users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (organization_id, dataset_name, version),
    UNIQUE (checksum_sha256),
    CHECK (checksum_sha256 ~ '^[a-f0-9]{64}$'),
    CHECK (jsonb_typeof(schema_definition) = 'object')
);

CREATE INDEX idx_datasets_organization
    ON datasets(organization_id, created_at DESC);

CREATE INDEX idx_datasets_approved
    ON datasets(organization_id, approved)
    WHERE approved = TRUE;

-- ------------------------------------------------------------
-- Pipeline definitions
-- ------------------------------------------------------------

CREATE TABLE pipelines (
    pipeline_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    organization_id BIGINT NOT NULL
        REFERENCES organizations(organization_id),
    pipeline_name VARCHAR(150) NOT NULL,
    description TEXT,
    status pipeline_status NOT NULL DEFAULT 'draft',
    owner_id BIGINT NOT NULL
        REFERENCES platform_users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (organization_id, pipeline_name)
);

CREATE TABLE pipeline_versions (
    pipeline_version_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_id BIGINT NOT NULL
        REFERENCES pipelines(pipeline_id),
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    random_seed INTEGER NOT NULL DEFAULT 42,
    train_ratio NUMERIC(5,4) NOT NULL DEFAULT 0.70,
    validation_ratio NUMERIC(5,4) NOT NULL DEFAULT 0.15,
    test_ratio NUMERIC(5,4) NOT NULL DEFAULT 0.15,
    configuration JSONB NOT NULL,
    created_by BIGINT NOT NULL REFERENCES platform_users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (pipeline_id, version_number),
    CHECK (train_ratio > 0 AND validation_ratio >= 0 AND test_ratio > 0),
    CHECK (
        train_ratio + validation_ratio + test_ratio = 1.0000
    ),
    CHECK (jsonb_typeof(configuration) = 'object')
);

-- ------------------------------------------------------------
-- Transformer registry
-- ------------------------------------------------------------

CREATE TABLE transformer_definitions (
    transformer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transformer_name VARCHAR(100) NOT NULL,
    transformer_category VARCHAR(40) NOT NULL
        CHECK (
            transformer_category IN (
                'imputation',
                'scaling',
                'encoding',
                'selection',
                'dimensionality_reduction',
                'custom'
            )
        ),
    description TEXT NOT NULL,
    deterministic BOOLEAN NOT NULL DEFAULT TRUE,
    version VARCHAR(40) NOT NULL,
    UNIQUE (transformer_name, version)
);

CREATE TABLE pipeline_stages (
    stage_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_version_id BIGINT NOT NULL
        REFERENCES pipeline_versions(pipeline_version_id)
        ON DELETE CASCADE,
    transformer_id BIGINT NOT NULL
        REFERENCES transformer_definitions(transformer_id),
    stage_name VARCHAR(100) NOT NULL,
    stage_order INTEGER NOT NULL CHECK (stage_order > 0),
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (pipeline_version_id, stage_order),
    UNIQUE (pipeline_version_id, stage_name),
    CHECK (jsonb_typeof(parameters) = 'object')
);

CREATE INDEX idx_pipeline_stages_execution
    ON pipeline_stages(
        pipeline_version_id,
        stage_order
    )
    WHERE enabled = TRUE;

-- ------------------------------------------------------------
-- Model configurations
-- ------------------------------------------------------------

CREATE TABLE model_definitions (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name VARCHAR(120) NOT NULL,
    model_family VARCHAR(60) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    supports_probability BOOLEAN NOT NULL DEFAULT FALSE,
    version VARCHAR(40) NOT NULL,
    UNIQUE (model_name, version)
);

CREATE TABLE model_configurations (
    model_configuration_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_version_id BIGINT NOT NULL UNIQUE
        REFERENCES pipeline_versions(pipeline_version_id)
        ON DELETE CASCADE,
    model_id BIGINT NOT NULL REFERENCES model_definitions(model_id),
    hyperparameters JSONB NOT NULL,
    objective_metric VARCHAR(50) NOT NULL,
    maximum_iterations INTEGER NOT NULL DEFAULT 1000
        CHECK (maximum_iterations > 0),
    UNIQUE (pipeline_version_id, model_id),
    CHECK (jsonb_typeof(hyperparameters) = 'object')
);

-- ------------------------------------------------------------
-- Dataset partition management
-- ------------------------------------------------------------

CREATE TABLE dataset_partitions (
    partition_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_version_id BIGINT NOT NULL
        REFERENCES pipeline_versions(pipeline_version_id)
        ON DELETE CASCADE,
    split split_type NOT NULL,
    record_count BIGINT NOT NULL CHECK (record_count > 0),
    partition_checksum VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (pipeline_version_id, split),
    UNIQUE (partition_checksum),
    CHECK (partition_checksum ~ '^[a-f0-9]{64}$')
);

-- ------------------------------------------------------------
-- Pipeline execution
-- ------------------------------------------------------------

CREATE TABLE pipeline_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline_version_id BIGINT NOT NULL
        REFERENCES pipeline_versions(pipeline_version_id),
    status run_status NOT NULL DEFAULT 'queued',
    triggered_by BIGINT NOT NULL REFERENCES platform_users(user_id),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    execution_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        completed_at IS NULL OR started_at IS NOT NULL
    ),
    CHECK (
        completed_at IS NULL OR completed_at >= started_at
    ),
    CHECK (
        status <> 'succeeded' OR completed_at IS NOT NULL
    ),
    CHECK (jsonb_typeof(execution_metadata) = 'object')
);

CREATE INDEX idx_pipeline_runs_status
    ON pipeline_runs(status, created_at DESC);

CREATE INDEX idx_pipeline_runs_version
    ON pipeline_runs(pipeline_version_id, created_at DESC);

CREATE TABLE run_stages (
    run_stage_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES pipeline_runs(run_id)
        ON DELETE CASCADE,
    stage_id BIGINT NOT NULL
        REFERENCES pipeline_stages(stage_id),
    status stage_status NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    input_records BIGINT,
    output_records BIGINT,
    execution_time_ms BIGINT,
    error_message TEXT,
    stage_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, stage_id),
    CHECK (input_records IS NULL OR input_records >= 0),
    CHECK (output_records IS NULL OR output_records >= 0),
    CHECK (
        execution_time_ms IS NULL OR execution_time_ms >= 0
    ),
    CHECK (
        completed_at IS NULL OR started_at IS NOT NULL
    ),
    CHECK (
        completed_at IS NULL OR completed_at >= started_at
    )
);

CREATE INDEX idx_run_stages_status
    ON run_stages(run_id, status);

-- ------------------------------------------------------------
-- Model evaluation
-- ------------------------------------------------------------

CREATE TABLE evaluation_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES pipeline_runs(run_id),
    split split_type NOT NULL,
    metric_name VARCHAR(80) NOT NULL,
    metric_value NUMERIC(16,8) NOT NULL,
    evaluation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, split, metric_name),
    CHECK (
        metric_name IN (
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'mae',
            'rmse',
            'log_loss'
        )
    ),
    CHECK (
        metric_value >= 0
    ),
    CHECK (
        metric_name NOT IN (
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc'
        )
        OR metric_value <= 1
    )
);

CREATE INDEX idx_metrics_run
    ON evaluation_metrics(run_id, split);

-- ------------------------------------------------------------
-- Model artifact registry
-- ------------------------------------------------------------

CREATE TABLE model_artifacts (
    artifact_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES pipeline_runs(run_id),
    artifact_type artifact_type NOT NULL,
    artifact_uri TEXT NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),
    registered_by BIGINT NOT NULL REFERENCES platform_users(user_id),
    registered_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, artifact_type),
    CHECK (checksum_sha256 ~ '^[a-f0-9]{64}$')
);

-- ------------------------------------------------------------
-- Inference requests
-- ------------------------------------------------------------

CREATE TABLE inference_requests (
    inference_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    artifact_id BIGINT NOT NULL
        REFERENCES model_artifacts(artifact_id),
    request_id UUID NOT NULL UNIQUE,
    input_payload JSONB NOT NULL,
    prediction JSONB,
    prediction_probability NUMERIC(10,8),
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (
            processing_status IN (
                'pending',
                'processing',
                'succeeded',
                'failed'
            )
        ),
    error_message TEXT,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    CHECK (jsonb_typeof(input_payload) = 'object'),
    CHECK (
        prediction_probability IS NULL OR
        prediction_probability BETWEEN 0 AND 1
    )
);

CREATE INDEX idx_inference_requests_status
    ON inference_requests(processing_status, requested_at DESC);

-- ------------------------------------------------------------
-- Updated timestamp trigger
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION update_modified_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_pipeline_updated
BEFORE UPDATE ON pipelines
FOR EACH ROW
EXECUTE FUNCTION update_modified_timestamp();

-- ------------------------------------------------------------
-- Pipeline configuration validation
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_pipeline_configuration()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    stage_count INTEGER;
    model_count INTEGER;
    dataset_approved BOOLEAN;
BEGIN
    SELECT COUNT(*)
    INTO stage_count
    FROM pipeline_stages
    WHERE pipeline_version_id = NEW.pipeline_version_id
      AND enabled = TRUE;

    SELECT COUNT(*)
    INTO model_count
    FROM model_configurations
    WHERE pipeline_version_id = NEW.pipeline_version_id;

    SELECT d.approved
    INTO dataset_approved
    FROM pipeline_versions pv
    JOIN datasets d ON d.dataset_id = pv.dataset_id
    WHERE pv.pipeline_version_id = NEW.pipeline_version_id;

    IF stage_count = 0 THEN
        RAISE EXCEPTION
            'Pipeline version % has no enabled stages',
            NEW.pipeline_version_id;
    END IF;

    IF model_count <> 1 THEN
        RAISE EXCEPTION
            'Pipeline version must have exactly one model configuration';
    END IF;

    IF dataset_approved IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION
            'Pipeline dataset must be approved';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_pipeline_configuration_validation
BEFORE UPDATE OF status ON pipelines
FOR EACH ROW
WHEN (NEW.status = 'ready')
EXECUTE FUNCTION validate_pipeline_configuration();

-- ------------------------------------------------------------
-- Run lifecycle enforcement
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION enforce_run_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IN ('succeeded', 'failed', 'cancelled')
       AND NEW.status <> OLD.status THEN
        RAISE EXCEPTION
            'Terminal run state % cannot be changed',
            OLD.status;
    END IF;

    IF OLD.status = 'queued' AND NEW.status NOT IN (
        'queued', 'running', 'cancelled'
    ) THEN
        RAISE EXCEPTION 'Invalid queued run transition';
    END IF;

    IF OLD.status = 'running' AND NEW.status NOT IN (
        'running', 'succeeded', 'failed', 'cancelled'
    ) THEN
        RAISE EXCEPTION 'Invalid running run transition';
    END IF;

    IF NEW.status = 'running' AND NEW.started_at IS NULL THEN
        NEW.started_at = CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN ('succeeded', 'failed', 'cancelled')
       AND NEW.completed_at IS NULL THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status = 'failed'
       AND COALESCE(NEW.error_message, '') = '' THEN
        RAISE EXCEPTION
            'Failed runs must contain an error message';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_run_transition
BEFORE UPDATE ON pipeline_runs
FOR EACH ROW
EXECUTE FUNCTION enforce_run_transition();

-- ------------------------------------------------------------
-- Eligibility function
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION is_pipeline_eligible(
    p_pipeline_version_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    dataset_approved BOOLEAN;
    enabled_stages INTEGER;
    configured_models INTEGER;
    partition_count INTEGER;
BEGIN
    SELECT d.approved
    INTO dataset_approved
    FROM pipeline_versions pv
    JOIN datasets d ON d.dataset_id = pv.dataset_id
    WHERE pv.pipeline_version_id = p_pipeline_version_id;

    IF NOT FOUND OR dataset_approved IS DISTINCT FROM TRUE THEN
        RETURN FALSE;
    END IF;

    SELECT COUNT(*)
    INTO enabled_stages
    FROM pipeline_stages
    WHERE pipeline_version_id = p_pipeline_version_id
      AND enabled = TRUE;

    SELECT COUNT(*)
    INTO configured_models
    FROM model_configurations
    WHERE pipeline_version_id = p_pipeline_version_id;

    SELECT COUNT(*)
    INTO partition_count
    FROM dataset_partitions
    WHERE pipeline_version_id = p_pipeline_version_id;

    RETURN enabled_stages > 0
       AND configured_models = 1
       AND partition_count = 3;
END;
$$;

-- ------------------------------------------------------------
-- Start a pipeline run
-- ------------------------------------------------------------

CREATE OR REPLACE PROCEDURE start_pipeline_run(
    p_pipeline_version_id BIGINT,
    p_user_id BIGINT,
    INOUT p_run_id BIGINT DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
DECLARE
    current_stage RECORD;
BEGIN
    IF NOT is_pipeline_eligible(p_pipeline_version_id) THEN
        RAISE EXCEPTION
            'Pipeline version % is not eligible',
            p_pipeline_version_id;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM platform_users
        WHERE user_id = p_user_id
          AND active = TRUE
    ) THEN
        RAISE EXCEPTION 'Inactive or unknown user';
    END IF;

    INSERT INTO pipeline_runs (
        pipeline_version_id,
        triggered_by,
        status
    )
    VALUES (
        p_pipeline_version_id,
        p_user_id,
        'queued'
    )
    RETURNING run_id INTO p_run_id;

    FOR current_stage IN
        SELECT stage_id
        FROM pipeline_stages
        WHERE pipeline_version_id = p_pipeline_version_id
          AND enabled = TRUE
        ORDER BY stage_order
    LOOP
        INSERT INTO run_stages (
            run_id,
            stage_id,
            status
        )
        VALUES (
            p_run_id,
            current_stage.stage_id,
            'pending'
        );
    END LOOP;
END;
$$;

-- ------------------------------------------------------------
-- Reporting view
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW pipeline_run_report AS
SELECT
    r.run_id,
    p.pipeline_name,
    pv.version_number,
    r.status AS run_status,
    u.username AS triggered_by,
    r.started_at,
    r.completed_at,
    COUNT(DISTINCT rs.run_stage_id) AS total_stages,
    COUNT(DISTINCT rs.run_stage_id)
        FILTER (WHERE rs.status = 'succeeded') AS successful_stages,
    COUNT(DISTINCT rs.run_stage_id)
        FILTER (WHERE rs.status = 'failed') AS failed_stages,
    MAX(em.metric_value)
        FILTER (
            WHERE em.metric_name = 'accuracy'
              AND em.split = 'test'
        ) AS test_accuracy,
    MAX(em.metric_value)
        FILTER (
            WHERE em.metric_name = 'f1'
              AND em.split = 'test'
        ) AS test_f1
FROM pipeline_runs r
JOIN pipeline_versions pv
    ON pv.pipeline_version_id = r.pipeline_version_id
JOIN pipelines p
    ON p.pipeline_id = pv.pipeline_id
JOIN platform_users u
    ON u.user_id = r.triggered_by
LEFT JOIN run_stages rs
    ON rs.run_id = r.run_id
LEFT JOIN evaluation_metrics em
    ON em.run_id = r.run_id
GROUP BY
    r.run_id,
    p.pipeline_name,
    pv.version_number,
    r.status,
    u.username,
    r.started_at,
    r.completed_at;

-- ------------------------------------------------------------
-- Sample organizational records
-- ------------------------------------------------------------

INSERT INTO organizations (organization_name)
VALUES ('Enterprise Analytics')
RETURNING organization_id;

INSERT INTO platform_users (
    organization_id,
    username,
    email
)
VALUES
(
    1,
    'pipeline_admin',
    'admin@example.com'
),
(
    1,
    'ml_engineer',
    'engineer@example.com'
);

INSERT INTO datasets (
    organization_id,
    dataset_name,
    version,
    storage_uri,
    checksum_sha256,
    record_count,
    feature_count,
    target_column,
    classification_type,
    schema_definition,
    approved,
    created_by
)
VALUES (
    1,
    'customer_churn',
    '1.0',
    's3://enterprise-data/customer-churn-v1',
    repeat('a', 64),
    10000,
    5,
    'churn',
    'binary',
    '{
        "age": "numeric",
        "spending": "numeric",
        "tenure": "numeric",
        "contract": "categorical",
        "region": "categorical",
        "churn": "binary"
    }'::jsonb,
    TRUE,
    1
);

INSERT INTO pipelines (
    organization_id,
    pipeline_name,
    description,
    owner_id
)
VALUES (
    1,
    'customer_churn_classifier',
    'Customer churn preprocessing and classification workflow',
    1
);

INSERT INTO pipeline_versions (
    pipeline_id,
    version_number,
    dataset_id,
    random_seed,
    train_ratio,
    validation_ratio,
    test_ratio,
    configuration,
    created_by
)
VALUES (
    1,
    1,
    1,
    42,
    0.7000,
    0.1500,
    0.1500,
    '{
        "missing_value_strategy": "median",
        "scaling": "standard",
        "categorical_encoding": "one_hot",
        "leakage_prevention": true
    }'::jsonb,
    1
);

INSERT INTO transformer_definitions (
    transformer_name,
    transformer_category,
    description,
    deterministic,
    version
)
VALUES
(
    'MedianImputer',
    'imputation',
    'Replaces missing numerical values with training median',
    TRUE,
    '1.0'
),
(
    'StandardScaler',
    'scaling',
    'Standardizes numerical features',
    TRUE,
    '1.0'
),
(
    'OneHotEncoder',
    'encoding',
    'Converts categorical features into binary indicators',
    TRUE,
    '1.0'
);

INSERT INTO pipeline_stages (
    pipeline_version_id,
    transformer_id,
    stage_name,
    stage_order,
    parameters
)
VALUES
(
    1,
    1,
    'numerical_imputation',
    1,
    '{"strategy": "median"}'::jsonb
),
(
    1,
    2,
    'numerical_scaling',
    2,
    '{"with_mean": true, "with_std": true}'::jsonb
),
(
    1,
    3,
    'categorical_encoding',
    3,
    '{"handle_unknown": "ignore"}'::jsonb
);

INSERT INTO model_definitions (
    model_name,
    model_family,
    algorithm,
    supports_probability,
    version
)
VALUES (
    'LogisticRegression',
    'linear',
    'logistic_regression',
    TRUE,
    '1.0'
);

INSERT INTO model_configurations (
    pipeline_version_id,
    model_id,
    hyperparameters,
    objective_metric,
    maximum_iterations
)
VALUES (
    1,
    1,
    '{
        "C": 1.0,
        "solver": "lbfgs",
        "class_weight": "balanced"
    }'::jsonb,
    'f1',
    2000
);

-- Dataset partitions must be registered before execution.

INSERT INTO dataset_partitions (
    pipeline_version_id,
    split,
    record_count,
    partition_checksum
)
VALUES
(1, 'train', 7000, repeat('b', 64)),
(1, 'validation', 1500, repeat('c', 64)),
(1, 'test', 1500, repeat('d', 64));

-- ------------------------------------------------------------
-- Approve pipeline configuration and prepare execution
-- ------------------------------------------------------------

UPDATE pipelines
SET status = 'ready'
WHERE pipeline_id = 1;

SELECT
    pipeline_id,
    pipeline_name,
    status
FROM pipelines;

SELECT
    is_pipeline_eligible(1) AS eligible;

-- ------------------------------------------------------------
-- Transaction-safe run creation
-- ------------------------------------------------------------

DO $$
DECLARE
    new_run_id BIGINT;
BEGIN
    CALL start_pipeline_run(
        1,
        2,
        new_run_id
    );

    RAISE NOTICE 'Created pipeline run: %', new_run_id;

    UPDATE pipeline_runs
    SET status = 'running'
    WHERE run_id = new_run_id;
END;
$$;

-- ------------------------------------------------------------
-- Execution stage processing
-- Demonstrates successful and failed stage states.
-- ------------------------------------------------------------

UPDATE run_stages
SET
    status = 'succeeded',
    started_at = CURRENT_TIMESTAMP,
    completed_at = CURRENT_TIMESTAMP,
    input_records = 7000,
    output_records = 7000,
    execution_time_ms = 125
WHERE run_id = 1
  AND stage_id = 1;

UPDATE run_stages
SET
    status = 'succeeded',
    started_at = CURRENT_TIMESTAMP,
    completed_at = CURRENT_TIMESTAMP,
    input_records = 7000,
    output_records = 7000,
    execution_time_ms = 210
WHERE run_id = 1
  AND stage_id = 2;

UPDATE run_stages
SET
    status = 'succeeded',
    started_at = CURRENT_TIMESTAMP,
    completed_at = CURRENT_TIMESTAMP,
    input_records = 7000,
    output_records = 7000,
    execution_time_ms = 95
WHERE run_id = 1
  AND stage_id = 3;

-- ------------------------------------------------------------
-- Evaluation results
-- ------------------------------------------------------------

INSERT INTO evaluation_metrics (
    run_id,
    split,
    metric_name,
    metric_value,
    evaluation_metadata
)
VALUES
(
    1,
    'validation',
    'accuracy',
    0.84200000,
    '{"threshold": 0.5}'::jsonb
),
(
    1,
    'validation',
    'f1',
    0.79100000,
    '{"threshold": 0.5}'::jsonb
),
(
    1,
    'test',
    'accuracy',
    0.83500000,
    '{"threshold": 0.5}'::jsonb
),
(
    1,
    'test',
    'precision',
    0.81000000,
    '{}'::jsonb
),
(
    1,
    'test',
    'recall',
    0.77500000,
    '{}'::jsonb
),
(
    1,
    'test',
    'f1',
    0.79200000,
    '{}'::jsonb
),
(
    1,
    'test',
    'roc_auc',
    0.88100000,
    '{}'::jsonb
);

-- ------------------------------------------------------------
-- Artifact registration
-- ------------------------------------------------------------

INSERT INTO model_artifacts (
    run_id,
    artifact_type,
    artifact_uri,
    checksum_sha256,
    file_size_bytes,
    registered_by
)
VALUES
(
    1,
    'complete_pipeline',
    's3://enterprise-models/churn/pipeline-v1',
    repeat('e', 64),
    1048576,
    2
),
(
    1,
    'metrics',
    's3://enterprise-models/churn/metrics-v1.json',
    repeat('f', 64),
    4096,
    2
);

-- ------------------------------------------------------------
-- Complete the run
-- ------------------------------------------------------------

UPDATE pipeline_runs
SET status = 'succeeded'
WHERE run_id = 1;

-- ------------------------------------------------------------
-- Retrieval: pipeline configuration and ordered transformers
-- ------------------------------------------------------------

SELECT
    p.pipeline_name,
    pv.version_number,
    ps.stage_order,
    ps.stage_name,
    td.transformer_name,
    td.transformer_category,
    ps.parameters
FROM pipelines p
JOIN pipeline_versions pv
    ON pv.pipeline_id = p.pipeline_id
JOIN pipeline_stages ps
    ON ps.pipeline_version_id = pv.pipeline_version_id
JOIN transformer_definitions td
    ON td.transformer_id = ps.transformer_id
WHERE p.pipeline_id = 1
ORDER BY ps.stage_order;

-- ------------------------------------------------------------
-- Validation: detect incomplete pipeline stages
-- ------------------------------------------------------------

SELECT
    r.run_id,
    rs.stage_id,
    ps.stage_name,
    rs.status
FROM pipeline_runs r
JOIN run_stages rs ON rs.run_id = r.run_id
JOIN pipeline_stages ps ON ps.stage_id = rs.stage_id
WHERE r.run_id = 1
  AND rs.status <> 'succeeded';

-- ------------------------------------------------------------
-- Aggregation: model performance
-- ------------------------------------------------------------

SELECT
    split,
    metric_name,
    ROUND(AVG(metric_value), 4) AS average_value,
    MIN(metric_value) AS minimum_value,
    MAX(metric_value) AS maximum_value
FROM evaluation_metrics
WHERE run_id = 1
GROUP BY split, metric_name
ORDER BY split, metric_name;

-- ------------------------------------------------------------
-- Evaluation: compare validation and test performance
-- ------------------------------------------------------------

SELECT
    metric_name,
    MAX(metric_value) FILTER (
        WHERE split = 'validation'
    ) AS validation_value,
    MAX(metric_value) FILTER (
        WHERE split = 'test'
    ) AS test_value,
    MAX(metric_value) FILTER (
        WHERE split = 'validation'
    ) -
    MAX(metric_value) FILTER (
        WHERE split = 'test'
    ) AS performance_difference
FROM evaluation_metrics
WHERE run_id = 1
GROUP BY metric_name
ORDER BY metric_name;

-- ------------------------------------------------------------
-- Reporting: complete pipeline execution history
-- ------------------------------------------------------------

SELECT *
FROM pipeline_run_report
ORDER BY run_id DESC;

-- ------------------------------------------------------------
-- Workflow processing: queue pending pipeline runs
-- ------------------------------------------------------------

SELECT
    run_id,
    pipeline_version_id,
    created_at
FROM pipeline_runs
WHERE status = 'queued'
ORDER BY created_at
LIMIT 20
FOR UPDATE SKIP LOCKED;

-- ------------------------------------------------------------
-- Demonstrate failure handling in a separate transaction.
-- This transaction is rolled back, preserving the successful run.
-- ------------------------------------------------------------

SAVEPOINT before_failure_demo;

INSERT INTO pipeline_runs (
    pipeline_version_id,
    triggered_by,
    status
)
VALUES (
    1,
    2,
    'queued'
);

-- The run is marked failed with a meaningful diagnostic.
UPDATE pipeline_runs
SET
    status = 'running'
WHERE run_id = (
    SELECT MAX(run_id) FROM pipeline_runs
);

UPDATE pipeline_runs
SET
    status = 'failed',
    error_message = 'Demonstration: transformer encountered invalid input'
WHERE run_id = (
    SELECT MAX(run_id) FROM pipeline_runs
);

ROLLBACK TO SAVEPOINT before_failure_demo;

-- ------------------------------------------------------------
-- Operational integrity checks
-- ------------------------------------------------------------

SELECT
    COUNT(*) AS total_runs,
    COUNT(*) FILTER (
        WHERE status = 'succeeded'
    ) AS successful_runs,
    COUNT(*) FILTER (
        WHERE status = 'failed'
    ) AS failed_runs,
    COUNT(*) FILTER (
        WHERE status = 'running'
    ) AS active_runs
FROM pipeline_runs;

SELECT
    artifact_type,
    artifact_uri,
    checksum_sha256,
    file_size_bytes
FROM model_artifacts
WHERE run_id = 1;

COMMIT;

-- ============================================================
-- SECURITY AND OPERATIONS NOTES
--
-- 1. Dataset and artifact URIs are references, not uploaded files.
-- 2. Verify checksums before loading external artifacts.
-- 3. Store credentials outside SQL configuration JSON.
-- 4. Restrict access using PostgreSQL roles and grants.
-- 5. Never deserialize untrusted executable model artifacts.
-- 6. Training and validation transformations must use training-
--    fitted statistics only.
-- 7. Preserve dataset versions and preprocessing configurations.
-- 8. Use EXPLAIN ANALYZE for production query optimization.
-- ============================================================
