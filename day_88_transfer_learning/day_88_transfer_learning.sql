-- ============================================================
-- TRANSFER LEARNING MANAGEMENT DATABASE
-- PostgreSQL 14+
--
-- Domain:
-- Pretrained models, feature extraction, fine-tuning,
-- datasets, model layers, experiments, evaluations,
-- approval decisions and deployment governance.
--
-- Execute:
-- psql -d your_database -f transfer_learning.sql
--
-- The script creates an isolated schema and sample workflow.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS transfer_learning;

SET search_path TO transfer_learning, public;

-- ============================================================
-- 1. ENUMERATIONS
-- ============================================================

CREATE TYPE model_status AS ENUM (
    'REGISTERED',
    'VERIFIED',
    'DEPRECATED',
    'REVOKED'
);

CREATE TYPE transfer_strategy AS ENUM (
    'FEATURE_EXTRACTION',
    'FINE_TUNING'
);

CREATE TYPE experiment_status AS ENUM (
    'CREATED',
    'VALIDATED',
    'TRAINING',
    'EVALUATED',
    'APPROVED',
    'REJECTED',
    'DEPLOYED',
    'FAILED'
);

CREATE TYPE layer_type AS ENUM (
    'CONVOLUTION',
    'DENSE',
    'NORMALIZATION',
    'ACTIVATION',
    'POOLING'
);

CREATE TYPE review_decision AS ENUM (
    'APPROVED',
    'REJECTED',
    'REQUIRES_CHANGES'
);

-- ============================================================
-- 2. MODEL REGISTRY
-- ============================================================

CREATE TABLE pretrained_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_name VARCHAR(150) NOT NULL,

    architecture VARCHAR(100) NOT NULL,

    source_domain VARCHAR(150) NOT NULL,

    framework VARCHAR(60) NOT NULL,

    model_version VARCHAR(40) NOT NULL,

    weight_checksum VARCHAR(128) NOT NULL,

    input_shape INTEGER[] NOT NULL,

    parameter_count BIGINT NOT NULL,

    license_name VARCHAR(100) NOT NULL,

    source_uri TEXT,

    status model_status NOT NULL DEFAULT 'REGISTERED',

    registered_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_model_version
        UNIQUE (model_name, model_version),

    CONSTRAINT valid_parameter_count
        CHECK (parameter_count > 0),

    CONSTRAINT valid_input_shape
        CHECK (
            cardinality(input_shape) > 0
            AND 0 < ALL(input_shape)
        ),

    CONSTRAINT valid_checksum
        CHECK (length(weight_checksum) >= 16)
);

CREATE INDEX idx_models_status
    ON pretrained_models(status);

CREATE INDEX idx_models_architecture
    ON pretrained_models(architecture);

CREATE INDEX idx_models_source_domain
    ON pretrained_models(source_domain);

-- ============================================================
-- 3. PRETRAINED MODEL LAYERS
-- ============================================================

CREATE TABLE model_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES pretrained_models(model_id)
        ON DELETE RESTRICT,

    layer_name VARCHAR(120) NOT NULL,

    layer_position INTEGER NOT NULL,

    layer_category layer_type NOT NULL,

    input_dimension INTEGER NOT NULL,

    output_dimension INTEGER NOT NULL,

    parameter_count BIGINT NOT NULL,

    is_pretrained BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT unique_layer_position
        UNIQUE (model_id, layer_position),

    CONSTRAINT unique_layer_name
        UNIQUE (model_id, layer_name),

    CONSTRAINT valid_layer_dimensions
        CHECK (
            input_dimension > 0
            AND output_dimension > 0
            AND parameter_count >= 0
            AND layer_position >= 0
        )
);

CREATE INDEX idx_layers_model_position
    ON model_layers(model_id, layer_position);

-- ============================================================
-- 4. DATASETS
-- ============================================================

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_name VARCHAR(150) NOT NULL,

    domain VARCHAR(150) NOT NULL,

    dataset_version VARCHAR(40) NOT NULL,

    input_shape INTEGER[] NOT NULL,

    sample_count INTEGER NOT NULL,

    training_samples INTEGER NOT NULL,

    validation_samples INTEGER NOT NULL,

    test_samples INTEGER NOT NULL,

    class_count INTEGER NOT NULL,

    data_checksum VARCHAR(128) NOT NULL,

    data_license VARCHAR(100) NOT NULL,

    contains_sensitive_data BOOLEAN NOT NULL DEFAULT FALSE,

    access_approved BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_dataset_version
        UNIQUE (dataset_name, dataset_version),

    CONSTRAINT valid_sample_count
        CHECK (
            sample_count > 0
            AND training_samples > 0
            AND validation_samples > 0
            AND test_samples > 0
        ),

    CONSTRAINT valid_split_count
        CHECK (
            training_samples +
            validation_samples +
            test_samples = sample_count
        ),

    CONSTRAINT valid_class_count
        CHECK (class_count >= 2),

    CONSTRAINT valid_dataset_shape
        CHECK (
            cardinality(input_shape) > 0
            AND 0 < ALL(input_shape)
        )
);

CREATE INDEX idx_datasets_domain
    ON datasets(domain);

CREATE INDEX idx_datasets_access
    ON datasets(access_approved);

-- ============================================================
-- 5. TRANSFER LEARNING EXPERIMENTS
-- ============================================================

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_name VARCHAR(180) NOT NULL,

    model_id BIGINT NOT NULL
        REFERENCES pretrained_models(model_id)
        ON DELETE RESTRICT,

    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id)
        ON DELETE RESTRICT,

    strategy transfer_strategy NOT NULL,

    status experiment_status NOT NULL DEFAULT 'CREATED',

    frozen_layers INTEGER NOT NULL,

    epochs INTEGER NOT NULL,

    backbone_learning_rate NUMERIC(12, 10) NOT NULL,

    head_learning_rate NUMERIC(12, 10) NOT NULL,

    batch_size INTEGER NOT NULL,

    random_seed INTEGER NOT NULL DEFAULT 42,

    created_by VARCHAR(100) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    failure_reason TEXT,

    CONSTRAINT valid_training_configuration
        CHECK (
            frozen_layers >= 0
            AND epochs > 0
            AND backbone_learning_rate >= 0
            AND head_learning_rate > 0
            AND batch_size > 0
        ),

    CONSTRAINT valid_feature_extraction_rate
        CHECK (
            strategy <> 'FEATURE_EXTRACTION'
            OR backbone_learning_rate = 0
        )
);

CREATE INDEX idx_experiments_model
    ON experiments(model_id);

CREATE INDEX idx_experiments_dataset
    ON experiments(dataset_id);

CREATE INDEX idx_experiments_status
    ON experiments(status);

CREATE INDEX idx_experiments_created
    ON experiments(created_at DESC);

-- ============================================================
-- 6. LAYER-LEVEL TRANSFER CONFIGURATION
-- ============================================================

CREATE TABLE experiment_layer_settings (
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    layer_id BIGINT NOT NULL
        REFERENCES model_layers(layer_id)
        ON DELETE RESTRICT,

    trainable BOOLEAN NOT NULL DEFAULT FALSE,

    learning_rate NUMERIC(12, 10) NOT NULL DEFAULT 0,

    PRIMARY KEY (experiment_id, layer_id),

    CONSTRAINT valid_layer_learning_rate
        CHECK (
            learning_rate >= 0
            AND (
                trainable = TRUE
                OR learning_rate = 0
            )
        )
);

-- ============================================================
-- 7. EVALUATION METRICS
-- ============================================================

CREATE TABLE experiment_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    evaluation_split VARCHAR(20) NOT NULL,

    accuracy NUMERIC(8, 6) NOT NULL,

    precision_score NUMERIC(8, 6) NOT NULL,

    recall_score NUMERIC(8, 6) NOT NULL,

    f1_score NUMERIC(8, 6) NOT NULL,

    loss NUMERIC(12, 8) NOT NULL,

    evaluated_samples INTEGER NOT NULL,

    training_duration_seconds NUMERIC(12, 3),

    inference_latency_ms NUMERIC(12, 4),

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_evaluation_split
        CHECK (
            evaluation_split IN (
                'TRAIN',
                'VALIDATION',
                'TEST',
                'EXTERNAL'
            )
        ),

    CONSTRAINT valid_metric_ranges
        CHECK (
            accuracy BETWEEN 0 AND 1
            AND precision_score BETWEEN 0 AND 1
            AND recall_score BETWEEN 0 AND 1
            AND f1_score BETWEEN 0 AND 1
            AND loss >= 0
            AND evaluated_samples > 0
        )
);

CREATE INDEX idx_metrics_experiment
    ON experiment_metrics(experiment_id, recorded_at DESC);

CREATE INDEX idx_metrics_accuracy
    ON experiment_metrics(accuracy DESC);

-- ============================================================
-- 8. APPROVAL AND GOVERNANCE
-- ============================================================

CREATE TABLE approval_requirements (
    requirement_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    requirement_name VARCHAR(120) NOT NULL UNIQUE,

    minimum_accuracy NUMERIC(8, 6) NOT NULL,

    maximum_loss NUMERIC(12, 8) NOT NULL,

    minimum_evaluation_coverage NUMERIC(5, 4) NOT NULL,

    requires_external_validation BOOLEAN NOT NULL DEFAULT FALSE,

    requires_data_access_approval BOOLEAN NOT NULL DEFAULT TRUE,

    active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT valid_approval_thresholds
        CHECK (
            minimum_accuracy BETWEEN 0 AND 1
            AND maximum_loss >= 0
            AND minimum_evaluation_coverage BETWEEN 0 AND 1
        )
);

CREATE TABLE experiment_reviews (
    review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    reviewer VARCHAR(100) NOT NULL,

    decision review_decision NOT NULL,

    comments TEXT NOT NULL,

    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, reviewer)
);

CREATE INDEX idx_reviews_experiment
    ON experiment_reviews(experiment_id);

-- ============================================================
-- 9. DEPLOYMENT REGISTRY
-- ============================================================

CREATE TABLE deployments (
    deployment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL UNIQUE
        REFERENCES experiments(experiment_id)
        ON DELETE RESTRICT,

    environment VARCHAR(40) NOT NULL,

    deployment_version VARCHAR(50) NOT NULL,

    deployed_by VARCHAR(100) NOT NULL,

    deployed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT valid_environment
        CHECK (
            environment IN (
                'STAGING',
                'PRODUCTION',
                'RESEARCH'
            )
        )
);

-- ============================================================
-- 10. DATA VALIDATION TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION validate_experiment_configuration()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_model pretrained_models%ROWTYPE;
    v_dataset datasets%ROWTYPE;
    v_layer_count INTEGER;
BEGIN
    SELECT *
    INTO v_model
    FROM pretrained_models
    WHERE model_id = NEW.model_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Pretrained model does not exist.';
    END IF;

    SELECT *
    INTO v_dataset
    FROM datasets
    WHERE dataset_id = NEW.dataset_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Target dataset does not exist.';
    END IF;

    IF v_model.status <> 'VERIFIED' THEN
        RAISE EXCEPTION
            'Only verified pretrained models can be transferred.';
    END IF;

    IF v_dataset.access_approved = FALSE THEN
        RAISE EXCEPTION
            'Target dataset access has not been approved.';
    END IF;

    IF v_model.input_shape <> v_dataset.input_shape THEN
        RAISE EXCEPTION
            'Source and target input shapes are incompatible.';
    END IF;

    SELECT COUNT(*)
    INTO v_layer_count
    FROM model_layers
    WHERE model_id = NEW.model_id;

    IF NEW.frozen_layers > v_layer_count THEN
        RAISE EXCEPTION
            'Frozen layer count exceeds model depth.';
    END IF;

    IF NEW.strategy = 'FEATURE_EXTRACTION'
       AND NEW.frozen_layers <> v_layer_count THEN

        RAISE EXCEPTION
            'Feature extraction requires all backbone layers frozen.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_experiment
BEFORE INSERT ON experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_configuration();

-- ============================================================
-- 11. EXPERIMENT STATE TRANSITION VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION validate_experiment_state_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status = OLD.status THEN
        RETURN NEW;
    END IF;

    IF NOT (
        (OLD.status = 'CREATED'
            AND NEW.status IN ('VALIDATED', 'FAILED'))
        OR
        (OLD.status = 'VALIDATED'
            AND NEW.status IN ('TRAINING', 'FAILED'))
        OR
        (OLD.status = 'TRAINING'
            AND NEW.status IN ('EVALUATED', 'FAILED'))
        OR
        (OLD.status = 'EVALUATED'
            AND NEW.status IN ('APPROVED', 'REJECTED', 'FAILED'))
        OR
        (OLD.status = 'APPROVED'
            AND NEW.status IN ('DEPLOYED', 'FAILED'))
    ) THEN
        RAISE EXCEPTION
            'Invalid experiment transition: % -> %',
            OLD.status,
            NEW.status;
    END IF;

    IF NEW.status = 'TRAINING' THEN
        NEW.started_at = CURRENT_TIMESTAMP;
    END IF;

    IF NEW.status IN (
        'EVALUATED',
        'REJECTED',
        'DEPLOYED',
        'FAILED'
    ) THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_experiment_state_transition
BEFORE UPDATE OF status ON experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_state_transition();

-- ============================================================
-- 12. APPROVAL ELIGIBILITY FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION check_experiment_eligibility(
    p_experiment_id BIGINT,
    p_requirement_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    reason TEXT,
    observed_accuracy NUMERIC,
    observed_loss NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_experiment experiments%ROWTYPE;
    v_requirement approval_requirements%ROWTYPE;
    v_dataset datasets%ROWTYPE;

    v_test experiment_metrics%ROWTYPE;

    v_approved_reviews INTEGER;
    v_rejected_reviews INTEGER;
    v_external_count INTEGER;

    v_eligible BOOLEAN := TRUE;
    v_reason TEXT := 'All approval conditions satisfied.';
BEGIN
    SELECT *
    INTO v_experiment
    FROM experiments
    WHERE experiment_id = p_experiment_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Experiment not found.', NULL::NUMERIC,
               NULL::NUMERIC;
        RETURN;
    END IF;

    SELECT *
    INTO v_requirement
    FROM approval_requirements
    WHERE requirement_id = p_requirement_id
      AND active = TRUE;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Active approval policy not found.',
               NULL::NUMERIC, NULL::NUMERIC;
        RETURN;
    END IF;

    SELECT *
    INTO v_dataset
    FROM datasets
    WHERE dataset_id = v_experiment.dataset_id;

    IF v_experiment.status <> 'EVALUATED' THEN
        v_eligible := FALSE;
        v_reason := 'Experiment has not completed evaluation.';

    ELSIF v_requirement.requires_data_access_approval
          AND NOT v_dataset.access_approved THEN

        v_eligible := FALSE;
        v_reason := 'Dataset access approval is missing.';
    END IF;

    SELECT *
    INTO v_test
    FROM experiment_metrics
    WHERE experiment_id = p_experiment_id
      AND evaluation_split = 'TEST'
    ORDER BY recorded_at DESC
    LIMIT 1;

    IF v_eligible AND NOT FOUND THEN
        v_eligible := FALSE;
        v_reason := 'Test evaluation metrics are missing.';
    END IF;

    IF v_eligible THEN
        IF v_test.accuracy < v_requirement.minimum_accuracy THEN
            v_eligible := FALSE;
            v_reason := 'Accuracy threshold not satisfied.';

        ELSIF v_test.loss > v_requirement.maximum_loss THEN
            v_eligible := FALSE;
            v_reason := 'Maximum loss threshold exceeded.';

        ELSIF v_test.evaluated_samples <
              CEIL(
                  v_dataset.sample_count *
                  v_requirement.minimum_evaluation_coverage
              ) THEN

            v_eligible := FALSE;
            v_reason := 'Evaluation coverage is insufficient.';
        END IF;
    END IF;

    SELECT COUNT(*) FILTER (
        WHERE decision = 'APPROVED'
    ),
    COUNT(*) FILTER (
        WHERE decision IN ('REJECTED', 'REQUIRES_CHANGES')
    )
    INTO v_approved_reviews, v_rejected_reviews
    FROM experiment_reviews
    WHERE experiment_id = p_experiment_id;

    IF v_eligible AND v_rejected_reviews > 0 THEN
        v_eligible := FALSE;
        v_reason := 'A reviewer rejected the experiment or requested changes.';

    ELSIF v_eligible AND v_approved_reviews = 0 THEN
        v_eligible := FALSE;
        v_reason := 'At least one reviewer approval is required.';
    END IF;

    IF v_eligible AND v_requirement.requires_external_validation THEN

        SELECT COUNT(*)
        INTO v_external_count
        FROM experiment_metrics
        WHERE experiment_id = p_experiment_id
          AND evaluation_split = 'EXTERNAL';

        IF v_external_count = 0 THEN
            v_eligible := FALSE;
            v_reason := 'External validation is required.';
        END IF;
    END IF;

    RETURN QUERY
    SELECT
        v_eligible,
        v_reason,
        v_test.accuracy,
        v_test.loss;
END;
$$;

-- ============================================================
-- 13. DEPLOYMENT ENFORCEMENT TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION enforce_deployment_approval()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_status experiment_status;
BEGIN
    SELECT status
    INTO v_status
    FROM experiments
    WHERE experiment_id = NEW.experiment_id
    FOR UPDATE;

    IF v_status <> 'APPROVED' THEN
        RAISE EXCEPTION
            'Only approved experiments can be deployed.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_enforce_deployment
BEFORE INSERT ON deployments
FOR EACH ROW
EXECUTE FUNCTION enforce_deployment_approval();

-- ============================================================
-- 14. REPORTING VIEWS
-- ============================================================

CREATE VIEW experiment_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    pm.model_name,
    pm.architecture,
    pm.source_domain,
    d.dataset_name,
    d.domain AS target_domain,
    e.strategy,
    e.status,
    e.frozen_layers,
    e.epochs,
    e.created_at,
    e.completed_at
FROM experiments e
JOIN pretrained_models pm
    ON pm.model_id = e.model_id
JOIN datasets d
    ON d.dataset_id = e.dataset_id;

CREATE VIEW experiment_performance AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.strategy,
    m.evaluation_split,
    m.accuracy,
    m.precision_score,
    m.recall_score,
    m.f1_score,
    m.loss,
    m.evaluated_samples,
    m.inference_latency_ms
FROM experiments e
JOIN experiment_metrics m
    ON m.experiment_id = e.experiment_id;

CREATE VIEW model_transfer_usage AS
SELECT
    pm.model_id,
    pm.model_name,
    pm.source_domain,
    COUNT(e.experiment_id) AS experiment_count,
    COUNT(*) FILTER (
        WHERE e.strategy = 'FEATURE_EXTRACTION'
    ) AS feature_extraction_count,
    COUNT(*) FILTER (
        WHERE e.strategy = 'FINE_TUNING'
    ) AS fine_tuning_count,
    COUNT(*) FILTER (
        WHERE e.status = 'DEPLOYED'
    ) AS deployed_count
FROM pretrained_models pm
LEFT JOIN experiments e
    ON e.model_id = pm.model_id
GROUP BY
    pm.model_id,
    pm.model_name,
    pm.source_domain;

-- ============================================================
-- 15. SAMPLE PRETRAINED MODELS
-- ============================================================

INSERT INTO pretrained_models (
    model_name,
    architecture,
    source_domain,
    framework,
    model_version,
    weight_checksum,
    input_shape,
    parameter_count,
    license_name,
    status
)
VALUES
(
    'VisionBackbone',
    'CNN',
    'General Image Recognition',
    'PyTorch',
    '1.0.0',
    'abcdef0123456789abcdef0123456789',
    ARRAY[224, 224, 3],
    15600000,
    'Research License',
    'VERIFIED'
),
(
    'SensorBackbone',
    'MLP',
    'Industrial Sensor Analytics',
    'TensorFlow',
    '2.0.0',
    '1234567890abcdef1234567890abcdef',
    ARRAY[12],
    850000,
    'Apache-2.0',
    'VERIFIED'
),
(
    'UnverifiedVisionModel',
    'CNN',
    'General Image Recognition',
    'PyTorch',
    '0.5.0',
    '9876543210abcdef9876543210abcdef',
    ARRAY[224, 224, 3],
    5000000,
    'Unknown',
    'REGISTERED'
);

-- ============================================================
-- 16. MODEL LAYER RECORDS
-- ============================================================

INSERT INTO model_layers (
    model_id,
    layer_name,
    layer_position,
    layer_category,
    input_dimension,
    output_dimension,
    parameter_count
)
SELECT
    model_id,
    layer_name,
    layer_position,
    layer_category,
    input_dimension,
    output_dimension,
    parameter_count
FROM (
    SELECT
        model_id,
        'conv_block_1'::VARCHAR AS layer_name,
        0 AS layer_position,
        'CONVOLUTION'::layer_type AS layer_category,
        3 AS input_dimension,
        64 AS output_dimension,
        12000::BIGINT AS parameter_count
    FROM pretrained_models
    WHERE model_name = 'VisionBackbone'

    UNION ALL

    SELECT
        model_id,
        'conv_block_2',
        1,
        'CONVOLUTION'::layer_type,
        64,
        128,
        48000::BIGINT
    FROM pretrained_models
    WHERE model_name = 'VisionBackbone'

    UNION ALL

    SELECT
        model_id,
        'feature_projection',
        2,
        'DENSE'::layer_type,
        128,
        512,
        66048::BIGINT
    FROM pretrained_models
    WHERE model_name = 'VisionBackbone'

    UNION ALL

    SELECT
        model_id,
        'sensor_encoder',
        0,
        'DENSE'::layer_type,
        12,
        64,
        832::BIGINT
    FROM pretrained_models
    WHERE model_name = 'SensorBackbone'

    UNION ALL

    SELECT
        model_id,
        'sensor_projection',
        1,
        'DENSE'::layer_type,
        64,
        32,
        2080::BIGINT
    FROM pretrained_models
    WHERE model_name = 'SensorBackbone'
) layer_data;

-- ============================================================
-- 17. SAMPLE TARGET DATASETS
-- ============================================================

INSERT INTO datasets (
    dataset_name,
    domain,
    dataset_version,
    input_shape,
    sample_count,
    training_samples,
    validation_samples,
    test_samples,
    class_count,
    data_checksum,
    data_license,
    access_approved
)
VALUES
(
    'IndustrialDefectImages',
    'Industrial Quality Control',
    '1.0',
    ARRAY[224, 224, 3],
    5000,
    3500,
    750,
    750,
    4,
    'datasetimagechecksum1234567890',
    'CC-BY-4.0',
    TRUE
),
(
    'IndustrialSensorReadings',
    'Industrial Sensor Analytics',
    '1.0',
    ARRAY[12],
    10000,
    7000,
    1500,
    1500,
    2,
    'datasetsensorchecksum1234567890',
    'Internal Research License',
    TRUE
),
(
    'RestrictedMedicalImages',
    'Medical Imaging',
    '1.0',
    ARRAY[224, 224, 3],
    2000,
    1400,
    300,
    300,
    3,
    'restrictedchecksum1234567890abcd',
    'Restricted',
    FALSE
);

-- ============================================================
-- 18. APPROVAL POLICIES
-- ============================================================

INSERT INTO approval_requirements (
    requirement_name,
    minimum_accuracy,
    maximum_loss,
    minimum_evaluation_coverage,
    requires_external_validation,
    requires_data_access_approval
)
VALUES
(
    'ResearchModelApproval',
    0.80,
    0.50,
    0.95,
    FALSE,
    TRUE
),
(
    'ProductionModelApproval',
    0.90,
    0.25,
    0.95,
    TRUE,
    TRUE
);

-- ============================================================
-- 19. CREATE EXPERIMENTS
-- ============================================================

-- Feature extraction experiment.
INSERT INTO experiments (
    experiment_name,
    model_id,
    dataset_id,
    strategy,
    frozen_layers,
    epochs,
    backbone_learning_rate,
    head_learning_rate,
    batch_size,
    created_by
)
SELECT
    'Vision Feature Extraction',
    pm.model_id,
    d.dataset_id,
    'FEATURE_EXTRACTION',
    3,
    20,
    0,
    0.001,
    32,
    'research_engineer'
FROM pretrained_models pm
JOIN datasets d
    ON d.dataset_name = 'IndustrialDefectImages'
WHERE pm.model_name = 'VisionBackbone';

-- Fine-tuning experiment.
INSERT INTO experiments (
    experiment_name,
    model_id,
    dataset_id,
    strategy,
    frozen_layers,
    epochs,
    backbone_learning_rate,
    head_learning_rate,
    batch_size,
    created_by
)
SELECT
    'Vision Partial Fine Tuning',
    pm.model_id,
    d.dataset_id,
    'FINE_TUNING',
    2,
    30,
    0.0001,
    0.001,
    32,
    'research_engineer'
FROM pretrained_models pm
JOIN datasets d
    ON d.dataset_name = 'IndustrialDefectImages'
WHERE pm.model_name = 'VisionBackbone';

-- This attempt is expected to fail because the dataset is not approved.
-- Execute separately to observe the integrity trigger:
--
-- INSERT INTO experiments (
--     experiment_name, model_id, dataset_id, strategy,
--     frozen_layers, epochs, backbone_learning_rate,
--     head_learning_rate, batch_size, created_by
-- )
-- SELECT
--     'Restricted Dataset Experiment',
--     pm.model_id, d.dataset_id, 'FINE_TUNING',
--     2, 10, 0.0001, 0.001, 16, 'research_engineer'
-- FROM pretrained_models pm
-- CROSS JOIN datasets d
-- WHERE pm.model_name = 'VisionBackbone'
-- AND d.dataset_name = 'RestrictedMedicalImages';

-- ============================================================
-- 20. CONFIGURE LAYER TRAINABILITY
-- ============================================================

INSERT INTO experiment_layer_settings (
    experiment_id,
    layer_id,
    trainable,
    learning_rate
)
SELECT
    e.experiment_id,
    l.layer_id,
    CASE
        WHEN e.strategy = 'FINE_TUNING'
             AND l.layer_position >= e.frozen_layers
        THEN TRUE
        ELSE FALSE
    END,
    CASE
        WHEN e.strategy = 'FINE_TUNING'
             AND l.layer_position >= e.frozen_layers
        THEN e.backbone_learning_rate
        ELSE 0
    END
FROM experiments e
JOIN model_layers l
    ON l.model_id = e.model_id;

-- ============================================================
-- 21. PROCESS EXPERIMENT STATES
-- ============================================================

UPDATE experiments
SET status = 'VALIDATED'
WHERE status = 'CREATED';

UPDATE experiments
SET status = 'TRAINING'
WHERE status = 'VALIDATED';

-- ============================================================
-- 22. RECORD EVALUATION METRICS
-- ============================================================

INSERT INTO experiment_metrics (
    experiment_id,
    evaluation_split,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    loss,
    evaluated_samples,
    training_duration_seconds,
    inference_latency_ms
)
SELECT
    experiment_id,
    'TEST',
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 0.86
        ELSE 0.93
    END,
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 0.84
        ELSE 0.92
    END,
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 0.85
        ELSE 0.91
    END,
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 0.845
        ELSE 0.915
    END,
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 0.32
        ELSE 0.18
    END,
    750,
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 120
        ELSE 280
    END,
    CASE
        WHEN strategy = 'FEATURE_EXTRACTION' THEN 4.5
        ELSE 5.2
    END
FROM experiments;

UPDATE experiments
SET status = 'EVALUATED'
WHERE status = 'TRAINING';

-- ============================================================
-- 23. RECORD REVIEW DECISIONS
-- ============================================================

INSERT INTO experiment_reviews (
    experiment_id,
    reviewer,
    decision,
    comments
)
SELECT
    experiment_id,
    'senior_researcher',
    'APPROVED',
    'Evaluation metrics and transfer configuration reviewed.'
FROM experiments;

-- ============================================================
-- 24. ELIGIBILITY EVALUATION
-- ============================================================

-- Determine eligibility under research approval policy.
SELECT
    e.experiment_id,
    e.experiment_name,
    result.eligible,
    result.reason,
    result.observed_accuracy,
    result.observed_loss
FROM experiments e
CROSS JOIN approval_requirements ar
CROSS JOIN LATERAL check_experiment_eligibility(
    e.experiment_id,
    ar.requirement_id
) AS result
WHERE ar.requirement_name = 'ResearchModelApproval'
ORDER BY e.experiment_id;

-- ============================================================
-- 25. APPROVE ELIGIBLE EXPERIMENTS
-- ============================================================

UPDATE experiments e
SET status = 'APPROVED'
FROM approval_requirements ar
CROSS JOIN LATERAL (
    SELECT 1
) AS policy
WHERE ar.requirement_name = 'ResearchModelApproval'
  AND e.status = 'EVALUATED'
  AND EXISTS (
      SELECT 1
      FROM check_experiment_eligibility(
          e.experiment_id,
          ar.requirement_id
      ) result
      WHERE result.eligible = TRUE
  );

-- ============================================================
-- 26. DEPLOY APPROVED EXPERIMENTS
-- ============================================================

-- The deployment trigger independently enforces approval.
INSERT INTO deployments (
    experiment_id,
    environment,
    deployment_version,
    deployed_by
)
SELECT
    experiment_id,
    'STAGING',
    '1.0.0',
    'deployment_service'
FROM experiments
WHERE status = 'APPROVED';

UPDATE experiments e
SET status = 'DEPLOYED'
WHERE status = 'APPROVED'
  AND EXISTS (
      SELECT 1
      FROM deployments d
      WHERE d.experiment_id = e.experiment_id
  );

-- ============================================================
-- 27. RETRIEVAL QUERIES
-- ============================================================

-- Model registry.
SELECT
    model_id,
    model_name,
    architecture,
    source_domain,
    model_version,
    parameter_count,
    status
FROM pretrained_models
ORDER BY model_name;

-- Model architecture and layer structure.
SELECT
    pm.model_name,
    ml.layer_position,
    ml.layer_name,
    ml.layer_category,
    ml.parameter_count
FROM pretrained_models pm
JOIN model_layers ml
    ON ml.model_id = pm.model_id
ORDER BY pm.model_id, ml.layer_position;

-- Experiment and dataset details.
SELECT *
FROM experiment_summary
ORDER BY experiment_id;

-- ============================================================
-- 28. VALIDATION QUERIES
-- ============================================================

-- Identify incompatible source and target shapes.
SELECT
    pm.model_name,
    d.dataset_name,
    pm.input_shape AS source_shape,
    d.input_shape AS target_shape
FROM pretrained_models pm
CROSS JOIN datasets d
WHERE pm.input_shape <> d.input_shape;

-- Identify datasets without access approval.
SELECT
    dataset_name,
    domain,
    contains_sensitive_data,
    access_approved
FROM datasets
WHERE access_approved = FALSE;

-- Identify experiments with invalid trainability settings.
SELECT
    e.experiment_id,
    e.experiment_name,
    e.strategy,
    els.layer_id,
    els.trainable,
    els.learning_rate
FROM experiments e
JOIN experiment_layer_settings els
    ON els.experiment_id = e.experiment_id
WHERE (
    e.strategy = 'FEATURE_EXTRACTION'
    AND els.trainable = TRUE
)
OR (
    els.trainable = FALSE
    AND els.learning_rate <> 0
);

-- ============================================================
-- 29. AGGREGATION AND PERFORMANCE REPORTING
-- ============================================================

-- Compare both transfer strategies.
SELECT
    e.strategy,
    COUNT(DISTINCT e.experiment_id) AS experiments,
    ROUND(AVG(m.accuracy), 4) AS average_accuracy,
    ROUND(AVG(m.loss), 4) AS average_loss,
    ROUND(AVG(m.training_duration_seconds), 2)
        AS average_training_seconds,
    ROUND(AVG(m.inference_latency_ms), 3)
        AS average_inference_latency_ms
FROM experiments e
JOIN experiment_metrics m
    ON m.experiment_id = e.experiment_id
WHERE m.evaluation_split = 'TEST'
GROUP BY e.strategy
ORDER BY e.strategy;

-- Layer-level trainable parameter distribution.
SELECT
    e.experiment_name,
    e.strategy,
    COUNT(*) FILTER (
        WHERE els.trainable = TRUE
    ) AS trainable_layers,
    COUNT(*) FILTER (
        WHERE els.trainable = FALSE
    ) AS frozen_layers,
    SUM(ml.parameter_count) FILTER (
        WHERE els.trainable = TRUE
    ) AS trainable_parameters,
    SUM(ml.parameter_count) FILTER (
        WHERE els.trainable = FALSE
    ) AS frozen_parameters
FROM experiments e
JOIN experiment_layer_settings els
    ON els.experiment_id = e.experiment_id
JOIN model_layers ml
    ON ml.layer_id = els.layer_id
GROUP BY
    e.experiment_id,
    e.experiment_name,
    e.strategy
ORDER BY e.experiment_id;

-- ============================================================
-- 30. CTE: EXPERIMENT RANKING
-- ============================================================

WITH ranked_experiments AS (
    SELECT
        e.experiment_id,
        e.experiment_name,
        e.strategy,
        m.accuracy,
        m.loss,
        RANK() OVER (
            PARTITION BY e.strategy
            ORDER BY m.accuracy DESC, m.loss ASC
        ) AS strategy_rank
    FROM experiments e
    JOIN experiment_metrics m
        ON m.experiment_id = e.experiment_id
    WHERE m.evaluation_split = 'TEST'
)
SELECT *
FROM ranked_experiments
ORDER BY strategy, strategy_rank;

-- ============================================================
-- 31. CTE: SOURCE-TO-TARGET DOMAIN TRANSFER REPORT
-- ============================================================

WITH transfer_pairs AS (
    SELECT
        pm.model_name,
        pm.source_domain,
        d.domain AS target_domain,
        e.strategy,
        e.experiment_id,
        e.status
    FROM experiments e
    JOIN pretrained_models pm
        ON pm.model_id = e.model_id
    JOIN datasets d
        ON d.dataset_id = e.dataset_id
)
SELECT
    source_domain,
    target_domain,
    strategy,
    COUNT(*) AS experiment_count,
    COUNT(*) FILTER (
        WHERE status = 'DEPLOYED'
    ) AS deployed_count
FROM transfer_pairs
GROUP BY
    source_domain,
    target_domain,
    strategy
ORDER BY source_domain, target_domain;

-- ============================================================
-- 32. STORED PROCEDURE: CONTROLLED EXPERIMENT FAILURE
-- ============================================================

CREATE OR REPLACE PROCEDURE fail_experiment(
    p_experiment_id BIGINT,
    p_reason TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_reason IS NULL OR length(trim(p_reason)) = 0 THEN
        RAISE EXCEPTION 'Failure reason is required.';
    END IF;

    UPDATE experiments
    SET
        status = 'FAILED',
        failure_reason = p_reason
    WHERE experiment_id = p_experiment_id
      AND status IN ('CREATED', 'VALIDATED', 'TRAINING', 'EVALUATED', 'APPROVED');

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Experiment not found or cannot enter FAILED state.';
    END IF;
END;
$$;

-- ============================================================
-- 33. TRANSACTIONAL EXAMPLE
-- ============================================================

-- Demonstrates atomic state updates.
-- The deployment operation and experiment state change
-- succeed together or roll back together.

-- Example:
--
-- BEGIN;
--
-- INSERT INTO deployments (
--     experiment_id,
--     environment,
--     deployment_version,
--     deployed_by
-- )
-- SELECT
--     experiment_id,
--     'PRODUCTION',
--     '1.0.1',
--     'production_service'
-- FROM experiments
-- WHERE experiment_id = 2
-- AND status = 'APPROVED';
--
-- UPDATE experiments
-- SET status = 'DEPLOYED'
-- WHERE experiment_id = 2
-- AND status = 'APPROVED';
--
-- COMMIT;

-- ============================================================
-- 34. FINAL DEPLOYMENT REPORT
-- ============================================================

SELECT
    e.experiment_id,
    e.experiment_name,
    pm.model_name,
    d.dataset_name,
    e.strategy,
    e.status,
    dep.environment,
    dep.deployment_version,
    dep.deployed_at
FROM experiments e
JOIN pretrained_models pm
    ON pm.model_id = e.model_id
JOIN datasets d
    ON d.dataset_id = e.dataset_id
LEFT JOIN deployments dep
    ON dep.experiment_id = e.experiment_id
ORDER BY e.experiment_id;

-- ============================================================
-- 35. PERFORMANCE AND OPERATIONAL NOTES
-- ============================================================

-- Important indexes:
-- experiments(status) supports lifecycle monitoring.
-- experiment_metrics(experiment_id, recorded_at DESC)
-- supports latest evaluation retrieval.
-- model_layers(model_id, layer_position)
-- supports ordered architecture reconstruction.
-- experiment_layer_settings(experiment_id, layer_id)
-- supports layer configuration retrieval.
--
-- Large deployments should consider:
-- 1. Partitioning experiment_metrics by recorded_at.
-- 2. Separating immutable model artifacts from metadata.
-- 3. Restricting model registry writes through database roles.
-- 4. Enforcing tenant-level access using row-level security.
-- 5. Recording artifact checksums and provenance.
-- 6. Using dedicated audit records for approval changes.
-- 7. Running external validation before production deployment.
--
-- The database records experiment metadata and measured results.
-- It does not execute neural-network training.

COMMIT;
