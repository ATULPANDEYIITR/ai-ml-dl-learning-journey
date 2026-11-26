/*
===============================================================================
COMPUTER VISION PIPELINE
PostgreSQL 15+

Scope:
1. Image ingestion and dataset management.
2. Preprocessing configuration and execution.
3. Data augmentation tracking.
4. Feature extraction.
5. Classification model lifecycle.
6. Predictions and evaluation metrics.
7. Dataset leakage prevention through split assignments.
8. Operational reporting and model governance.

All objects are created in the computer_vision schema.
===============================================================================
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS computer_vision;

SET search_path TO computer_vision, public;

-- ============================================================================
-- 1. DOMAIN TYPES
-- ============================================================================

CREATE TYPE dataset_split AS ENUM (
    'training',
    'validation',
    'testing'
);

CREATE TYPE image_status AS ENUM (
    'uploaded',
    'validated',
    'rejected',
    'processed'
);

CREATE TYPE preprocessing_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

CREATE TYPE augmentation_type AS ENUM (
    'horizontal_flip',
    'vertical_flip',
    'rotation',
    'brightness',
    'contrast',
    'crop',
    'noise'
);

CREATE TYPE model_status AS ENUM (
    'created',
    'training',
    'trained',
    'validated',
    'active',
    'retired',
    'failed'
);

CREATE TYPE prediction_status AS ENUM (
    'completed',
    'failed',
    'review_required'
);

-- ============================================================================
-- 2. REPOSITORY-STYLE DATASET AND IMAGE MANAGEMENT
-- ============================================================================

CREATE TABLE dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_name VARCHAR(150) NOT NULL UNIQUE,

    description TEXT,

    dataset_version VARCHAR(30) NOT NULL,

    owner_name VARCHAR(120) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT dataset_version_not_empty
        CHECK (LENGTH(TRIM(dataset_version)) > 0)
);

CREATE TABLE image_class (
    class_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    class_name VARCHAR(100) NOT NULL UNIQUE,

    description TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT class_name_not_empty
        CHECK (LENGTH(TRIM(class_name)) > 0)
);

CREATE TABLE image_asset (
    image_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id),

    class_id BIGINT
        REFERENCES image_class(class_id),

    file_name VARCHAR(255) NOT NULL,

    storage_uri TEXT NOT NULL,

    content_sha256 CHAR(64) NOT NULL,

    mime_type VARCHAR(100) NOT NULL,

    width INTEGER NOT NULL,

    height INTEGER NOT NULL,

    channels SMALLINT NOT NULL DEFAULT 3,

    bit_depth SMALLINT NOT NULL DEFAULT 8,

    file_size_bytes BIGINT NOT NULL,

    status image_status NOT NULL DEFAULT 'uploaded',

    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    validated_at TIMESTAMPTZ,

    rejection_reason TEXT,

    CONSTRAINT image_dimensions_valid
        CHECK (
            width BETWEEN 1 AND 100000
            AND height BETWEEN 1 AND 100000
        ),

    CONSTRAINT channel_count_valid
        CHECK (channels IN (1, 3, 4)),

    CONSTRAINT bit_depth_valid
        CHECK (bit_depth IN (8, 16, 32)),

    CONSTRAINT image_size_valid
        CHECK (file_size_bytes > 0),

    CONSTRAINT image_hash_valid
        CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),

    CONSTRAINT image_filename_not_empty
        CHECK (LENGTH(TRIM(file_name)) > 0),

    CONSTRAINT image_storage_uri_not_empty
        CHECK (LENGTH(TRIM(storage_uri)) > 0),

    CONSTRAINT image_validation_consistency
        CHECK (
            (status <> 'rejected' OR rejection_reason IS NOT NULL)
            AND
            (status <> 'validated' OR validated_at IS NOT NULL)
        ),

    UNIQUE(dataset_id, content_sha256)
);

CREATE TABLE dataset_assignment (
    assignment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    image_id BIGINT NOT NULL UNIQUE
        REFERENCES image_asset(image_id),

    split dataset_split NOT NULL,

    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    assignment_method VARCHAR(50) NOT NULL,

    CONSTRAINT assignment_method_not_empty
        CHECK (LENGTH(TRIM(assignment_method)) > 0)
);

-- ============================================================================
-- 3. PREPROCESSING CONFIGURATION
-- ============================================================================

CREATE TABLE preprocessing_profile (
    profile_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    profile_name VARCHAR(120) NOT NULL UNIQUE,

    target_width INTEGER NOT NULL,

    target_height INTEGER NOT NULL,

    resize_method VARCHAR(40) NOT NULL,

    color_space VARCHAR(30) NOT NULL DEFAULT 'RGB',

    normalization_method VARCHAR(40) NOT NULL,

    mean_red NUMERIC(12,8),

    mean_green NUMERIC(12,8),

    mean_blue NUMERIC(12,8),

    standard_deviation_red NUMERIC(12,8),

    standard_deviation_green NUMERIC(12,8),

    standard_deviation_blue NUMERIC(12,8),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT preprocessing_dimensions_valid
        CHECK (
            target_width BETWEEN 1 AND 4096
            AND target_height BETWEEN 1 AND 4096
        ),

    CONSTRAINT resize_method_valid
        CHECK (
            resize_method IN (
                'nearest',
                'bilinear',
                'bicubic',
                'lanczos'
            )
        ),

    CONSTRAINT normalization_valid
        CHECK (
            normalization_method IN (
                'zero_one',
                'minus_one_one',
                'standardization'
            )
        ),

    CONSTRAINT rgb_statistics_valid
        CHECK (
            (
                mean_red IS NULL
                AND mean_green IS NULL
                AND mean_blue IS NULL
            )
            OR
            (
                mean_red BETWEEN 0 AND 1
                AND mean_green BETWEEN 0 AND 1
                AND mean_blue BETWEEN 0 AND 1
            )
        )
);

CREATE TABLE preprocessing_run (
    preprocessing_run_id BIGINT
        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    image_id BIGINT NOT NULL
        REFERENCES image_asset(image_id),

    profile_id BIGINT NOT NULL
        REFERENCES preprocessing_profile(profile_id),

    status preprocessing_status NOT NULL DEFAULT 'pending',

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    output_uri TEXT,

    error_message TEXT,

    execution_duration_ms BIGINT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT preprocessing_timestamps_valid
        CHECK (
            completed_at IS NULL
            OR started_at IS NULL
            OR completed_at >= started_at
        ),

    CONSTRAINT preprocessing_completion_valid
        CHECK (
            status <> 'completed'
            OR (
                completed_at IS NOT NULL
                AND output_uri IS NOT NULL
                AND execution_duration_ms IS NOT NULL
            )
        ),

    CONSTRAINT preprocessing_failure_valid
        CHECK (
            status <> 'failed'
            OR error_message IS NOT NULL
        ),

    CONSTRAINT preprocessing_duration_valid
        CHECK (
            execution_duration_ms IS NULL
            OR execution_duration_ms >= 0
        )
);

-- ============================================================================
-- 4. DATA AUGMENTATION
-- ============================================================================

CREATE TABLE augmentation_policy (
    augmentation_policy_id BIGINT
        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    policy_name VARCHAR(120) NOT NULL UNIQUE,

    horizontal_flip BOOLEAN NOT NULL DEFAULT FALSE,

    vertical_flip BOOLEAN NOT NULL DEFAULT FALSE,

    rotation_enabled BOOLEAN NOT NULL DEFAULT FALSE,

    maximum_rotation_degrees NUMERIC(6,2) NOT NULL DEFAULT 0,

    brightness_enabled BOOLEAN NOT NULL DEFAULT FALSE,

    minimum_brightness NUMERIC(6,3) NOT NULL DEFAULT 1,

    maximum_brightness NUMERIC(6,3) NOT NULL DEFAULT 1,

    probability NUMERIC(5,4) NOT NULL DEFAULT 1,

    random_seed BIGINT NOT NULL DEFAULT 42,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT rotation_range_valid
        CHECK (
            maximum_rotation_degrees BETWEEN 0 AND 360
        ),

    CONSTRAINT brightness_range_valid
        CHECK (
            minimum_brightness >= 0
            AND maximum_brightness >= minimum_brightness
            AND maximum_brightness <= 3
        ),

    CONSTRAINT augmentation_probability_valid
        CHECK (probability BETWEEN 0 AND 1)
);

CREATE TABLE augmentation_run (
    augmentation_run_id BIGINT
        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    source_image_id BIGINT NOT NULL
        REFERENCES image_asset(image_id),

    generated_image_id BIGINT NOT NULL
        REFERENCES image_asset(image_id),

    augmentation_policy_id BIGINT NOT NULL
        REFERENCES augmentation_policy(augmentation_policy_id),

    operation augmentation_type NOT NULL,

    random_seed BIGINT NOT NULL,

    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT augmentation_source_target_different
        CHECK (source_image_id <> generated_image_id),

    CONSTRAINT augmentation_parameters_object
        CHECK (jsonb_typeof(parameters) = 'object')
);

-- ============================================================================
-- 5. FEATURE EXTRACTION
-- ============================================================================

CREATE TABLE feature_vector (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    image_id BIGINT NOT NULL
        REFERENCES image_asset(image_id),

    preprocessing_run_id BIGINT NOT NULL
        REFERENCES preprocessing_run(preprocessing_run_id),

    feature_version VARCHAR(40) NOT NULL,

    feature_values JSONB NOT NULL,

    feature_dimension INTEGER NOT NULL,

    extraction_duration_ms BIGINT NOT NULL,

    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT feature_values_object
        CHECK (jsonb_typeof(feature_values) = 'object'),

    CONSTRAINT feature_dimension_valid
        CHECK (feature_dimension > 0),

    CONSTRAINT feature_duration_valid
        CHECK (extraction_duration_ms >= 0),

    UNIQUE(image_id, preprocessing_run_id, feature_version)
);

-- ============================================================================
-- 6. MODEL GOVERNANCE
-- ============================================================================

CREATE TABLE classification_model (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_name VARCHAR(150) NOT NULL,

    model_version VARCHAR(40) NOT NULL,

    architecture VARCHAR(120) NOT NULL,

    framework VARCHAR(80),

    preprocessing_profile_id BIGINT NOT NULL
        REFERENCES preprocessing_profile(profile_id),

    augmentation_policy_id BIGINT
        REFERENCES augmentation_policy(augmentation_policy_id),

    status model_status NOT NULL DEFAULT 'created',

    model_artifact_uri TEXT,

    training_dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id),

    training_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    trained_at TIMESTAMPTZ,

    activated_at TIMESTAMPTZ,

    retired_at TIMESTAMPTZ,

    UNIQUE(model_name, model_version),

    CONSTRAINT model_parameters_object
        CHECK (jsonb_typeof(training_parameters) = 'object'),

    CONSTRAINT model_training_timestamp_valid
        CHECK (
            status NOT IN ('trained', 'validated', 'active', 'retired')
            OR (
                trained_at IS NOT NULL
                AND model_artifact_uri IS NOT NULL
            )
        ),

    CONSTRAINT model_activation_timestamp_valid
        CHECK (
            status NOT IN ('active', 'retired')
            OR activated_at IS NOT NULL
        )
);

CREATE UNIQUE INDEX one_active_model_per_architecture
ON classification_model(architecture)
WHERE status = 'active';

CREATE TABLE model_evaluation (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES classification_model(model_id),

    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id),

    evaluated_split dataset_split NOT NULL DEFAULT 'validation',

    accuracy NUMERIC(8,6),

    precision_score NUMERIC(8,6),

    recall_score NUMERIC(8,6),

    f1_score NUMERIC(8,6),

    macro_f1 NUMERIC(8,6),

    inference_latency_ms NUMERIC(12,4),

    total_samples BIGINT NOT NULL,

    correct_predictions BIGINT NOT NULL,

    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT metric_range_valid
        CHECK (
            (accuracy IS NULL OR accuracy BETWEEN 0 AND 1)
            AND (precision_score IS NULL OR precision_score BETWEEN 0 AND 1)
            AND (recall_score IS NULL OR recall_score BETWEEN 0 AND 1)
            AND (f1_score IS NULL OR f1_score BETWEEN 0 AND 1)
            AND (macro_f1 IS NULL OR macro_f1 BETWEEN 0 AND 1)
        ),

    CONSTRAINT evaluation_counts_valid
        CHECK (
            total_samples > 0
            AND correct_predictions BETWEEN 0 AND total_samples
        ),

    CONSTRAINT inference_latency_valid
        CHECK (
            inference_latency_ms IS NULL
            OR inference_latency_ms >= 0
        )
);

-- ============================================================================
-- 7. CLASSIFICATION PREDICTIONS
-- ============================================================================

CREATE TABLE prediction (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    image_id BIGINT NOT NULL
        REFERENCES image_asset(image_id),

    model_id BIGINT NOT NULL
        REFERENCES classification_model(model_id),

    predicted_class_id BIGINT
        REFERENCES image_class(class_id),

    actual_class_id BIGINT
        REFERENCES image_class(class_id),

    confidence NUMERIC(10,8),

    probabilities JSONB NOT NULL DEFAULT '{}'::jsonb,

    status prediction_status NOT NULL DEFAULT 'completed',

    inference_latency_ms NUMERIC(12,4),

    error_message TEXT,

    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT confidence_valid
        CHECK (
            confidence IS NULL
            OR confidence BETWEEN 0 AND 1
        ),

    CONSTRAINT prediction_probabilities_object
        CHECK (jsonb_typeof(probabilities) = 'object'),

    CONSTRAINT prediction_latency_valid
        CHECK (
            inference_latency_ms IS NULL
            OR inference_latency_ms >= 0
        ),

    CONSTRAINT prediction_state_valid
        CHECK (
            (
                status = 'failed'
                AND error_message IS NOT NULL
            )
            OR
            (
                status <> 'failed'
                AND predicted_class_id IS NOT NULL
                AND confidence IS NOT NULL
            )
        )
);

-- ============================================================================
-- 8. PERFORMANCE INDEXES
-- ============================================================================

CREATE INDEX idx_image_dataset_status
ON image_asset(dataset_id, status);

CREATE INDEX idx_image_class
ON image_asset(class_id);

CREATE INDEX idx_dataset_split
ON dataset_assignment(split, image_id);

CREATE INDEX idx_preprocessing_status
ON preprocessing_run(status, created_at);

CREATE INDEX idx_preprocessing_image
ON preprocessing_run(image_id, profile_id);

CREATE INDEX idx_augmentation_source
ON augmentation_run(source_image_id, created_at);

CREATE INDEX idx_feature_image
ON feature_vector(image_id);

CREATE INDEX idx_feature_json
ON feature_vector USING GIN(feature_values);

CREATE INDEX idx_model_status
ON classification_model(status, created_at);

CREATE INDEX idx_prediction_model_time
ON prediction(model_id, predicted_at DESC);

CREATE INDEX idx_prediction_class
ON prediction(predicted_class_id);

CREATE INDEX idx_evaluation_model
ON model_evaluation(model_id, evaluated_at DESC);

-- ============================================================================
-- 9. AUDIT LOG
-- ============================================================================

CREATE TABLE model_status_audit (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES classification_model(model_id),

    old_status model_status,

    new_status model_status NOT NULL,

    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    database_user TEXT NOT NULL DEFAULT CURRENT_USER
);

CREATE OR REPLACE FUNCTION audit_model_status()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO model_status_audit (
            model_id,
            old_status,
            new_status
        )
        VALUES (
            NEW.model_id,
            OLD.status,
            NEW.status
        );
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_model_status_audit
AFTER UPDATE OF status
ON classification_model
FOR EACH ROW
EXECUTE FUNCTION audit_model_status();

-- ============================================================================
-- 10. MODEL STATE TRANSITION ENFORCEMENT
-- ============================================================================

CREATE OR REPLACE FUNCTION enforce_model_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = NEW.status THEN
        RETURN NEW;
    END IF;

    IF NOT (
        (OLD.status = 'created' AND NEW.status IN ('training', 'failed'))
        OR
        (OLD.status = 'training' AND NEW.status IN ('trained', 'failed'))
        OR
        (OLD.status = 'trained' AND NEW.status IN ('validated', 'failed'))
        OR
        (OLD.status = 'validated' AND NEW.status IN ('active', 'failed'))
        OR
        (OLD.status = 'active' AND NEW.status = 'retired')
        OR
        (OLD.status = 'retired' AND NEW.status = 'active')
    ) THEN
        RAISE EXCEPTION
            'Invalid model transition: % -> %',
            OLD.status,
            NEW.status;
    END IF;

    IF NEW.status IN ('trained', 'validated', 'active', 'retired')
       AND NEW.trained_at IS NULL THEN
        RAISE EXCEPTION 'Trained timestamp is required.';
    END IF;

    IF NEW.status IN ('active', 'retired')
       AND NEW.activated_at IS NULL THEN
        RAISE EXCEPTION 'Activation timestamp is required.';
    END IF;

    IF NEW.status = 'active' THEN
        IF NEW.model_artifact_uri IS NULL THEN
            RAISE EXCEPTION 'Active model requires an artifact.';
        END IF;

        IF NOT EXISTS (
            SELECT 1
            FROM model_evaluation evaluation
            WHERE evaluation.model_id = NEW.model_id
              AND evaluation.evaluated_split = 'validation'
              AND evaluation.accuracy IS NOT NULL
              AND evaluation.accuracy >= 0.80
        ) THEN
            RAISE EXCEPTION
                'Model cannot activate without validation accuracy >= 0.80.';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_enforce_model_transition
BEFORE UPDATE OF status
ON classification_model
FOR EACH ROW
EXECUTE FUNCTION enforce_model_transition();

-- ============================================================================
-- 11. PREPROCESSING WORKFLOW
-- ============================================================================

CREATE OR REPLACE FUNCTION complete_preprocessing(
    p_run_id BIGINT,
    p_output_uri TEXT,
    p_duration_ms BIGINT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_output_uri IS NULL OR LENGTH(TRIM(p_output_uri)) = 0 THEN
        RAISE EXCEPTION 'Output URI is required.';
    END IF;

    IF p_duration_ms < 0 THEN
        RAISE EXCEPTION 'Duration cannot be negative.';
    END IF;

    UPDATE preprocessing_run
    SET
        status = 'completed',
        completed_at = NOW(),
        output_uri = p_output_uri,
        execution_duration_ms = p_duration_ms,
        error_message = NULL
    WHERE preprocessing_run_id = p_run_id
      AND status = 'running';

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Preprocessing run % is not running.',
            p_run_id;
    END IF;

    UPDATE image_asset
    SET status = 'processed'
    WHERE image_id = (
        SELECT image_id
        FROM preprocessing_run
        WHERE preprocessing_run_id = p_run_id
    );
END;
$$;

-- ============================================================================
-- 12. AUGMENTATION ELIGIBILITY
-- ============================================================================

CREATE OR REPLACE FUNCTION validate_augmentation_source()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    source_split dataset_split;
    generated_split dataset_split;
BEGIN
    SELECT assignment.split
    INTO source_split
    FROM dataset_assignment assignment
    WHERE assignment.image_id = NEW.source_image_id;

    SELECT assignment.split
    INTO generated_split
    FROM dataset_assignment assignment
    WHERE assignment.image_id = NEW.generated_image_id;

    IF source_split IS NULL THEN
        RAISE EXCEPTION 'Source image has no dataset assignment.';
    END IF;

    IF generated_split IS NULL THEN
        RAISE EXCEPTION 'Generated image has no dataset assignment.';
    END IF;

    IF source_split <> 'training' THEN
        RAISE EXCEPTION
            'Augmentation must not use validation or testing images.';
    END IF;

    IF generated_split <> 'training' THEN
        RAISE EXCEPTION
            'Augmented images must belong to training split.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_augmentation_source
BEFORE INSERT
ON augmentation_run
FOR EACH ROW
EXECUTE FUNCTION validate_augmentation_source();

-- ============================================================================
-- 13. MODEL ELIGIBILITY FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION model_activation_eligibility(
    p_model_id BIGINT
)
RETURNS TABLE (
    model_id BIGINT,
    eligible BOOLEAN,
    validation_accuracy NUMERIC,
    validation_f1 NUMERIC,
    reason TEXT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        model.model_id,

        (
            model.status = 'validated'
            AND evaluation.accuracy >= 0.80
            AND evaluation.f1_score >= 0.75
            AND evaluation.total_samples >= 10
        ) AS eligible,

        evaluation.accuracy,

        evaluation.f1_score,

        CASE
            WHEN model.status <> 'validated'
                THEN 'Model is not in validated state'
            WHEN evaluation.evaluation_id IS NULL
                THEN 'Validation evaluation is missing'
            WHEN evaluation.accuracy < 0.80
                THEN 'Validation accuracy below threshold'
            WHEN evaluation.f1_score < 0.75
                THEN 'Validation F1 below threshold'
            WHEN evaluation.total_samples < 10
                THEN 'Insufficient validation samples'
            ELSE 'Eligible for activation'
        END AS reason

    FROM classification_model model

    LEFT JOIN LATERAL (
        SELECT e.*
        FROM model_evaluation e
        WHERE e.model_id = model.model_id
          AND e.evaluated_split = 'validation'
        ORDER BY e.evaluated_at DESC
        LIMIT 1
    ) evaluation ON TRUE

    WHERE model.model_id = p_model_id;
$$;

-- ============================================================================
-- 14. OPERATIONAL REPORTING VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW dataset_distribution AS
SELECT
    dataset.dataset_id,
    dataset.dataset_name,
    assignment.split,
    COUNT(image.image_id) AS image_count,
    COUNT(DISTINCT image.class_id) AS distinct_classes

FROM dataset
LEFT JOIN image_asset image
    ON image.dataset_id = dataset.dataset_id

LEFT JOIN dataset_assignment assignment
    ON assignment.image_id = image.image_id

GROUP BY
    dataset.dataset_id,
    dataset.dataset_name,
    assignment.split;

CREATE OR REPLACE VIEW preprocessing_performance AS
SELECT
    profile.profile_name,

    COUNT(run.preprocessing_run_id) AS total_runs,

    COUNT(*) FILTER (
        WHERE run.status = 'completed'
    ) AS completed_runs,

    COUNT(*) FILTER (
        WHERE run.status = 'failed'
    ) AS failed_runs,

    ROUND(
        AVG(run.execution_duration_ms)
            FILTER (WHERE run.status = 'completed'),
        2
    ) AS average_duration_ms

FROM preprocessing_profile profile

LEFT JOIN preprocessing_run run
    ON run.profile_id = profile.profile_id

GROUP BY profile.profile_id, profile.profile_name;

CREATE OR REPLACE VIEW model_performance_report AS
SELECT
    model.model_id,
    model.model_name,
    model.model_version,
    model.architecture,
    model.status,
    evaluation.accuracy,
    evaluation.precision_score,
    evaluation.recall_score,
    evaluation.f1_score,
    evaluation.macro_f1,
    evaluation.inference_latency_ms,
    evaluation.total_samples,
    evaluation.evaluated_at

FROM classification_model model

LEFT JOIN LATERAL (
    SELECT e.*
    FROM model_evaluation e
    WHERE e.model_id = model.model_id
    ORDER BY e.evaluated_at DESC
    LIMIT 1
) evaluation ON TRUE;

CREATE OR REPLACE VIEW classification_error_report AS
SELECT
    actual.class_name AS actual_class,
    predicted.class_name AS predicted_class,
    COUNT(*) AS error_count,
    ROUND(AVG(prediction.confidence), 4) AS average_confidence

FROM prediction

JOIN image_class actual
    ON actual.class_id = prediction.actual_class_id

JOIN image_class predicted
    ON predicted.class_id = prediction.predicted_class_id

WHERE prediction.status = 'completed'
  AND prediction.actual_class_id <> prediction.predicted_class_id

GROUP BY actual.class_name, predicted.class_name;

-- ============================================================================
-- 15. REPORTING AND EVALUATION FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION classification_accuracy(
    p_model_id BIGINT
)
RETURNS TABLE (
    total_predictions BIGINT,
    correct_predictions BIGINT,
    incorrect_predictions BIGINT,
    accuracy NUMERIC
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        COUNT(*) AS total_predictions,

        COUNT(*) FILTER (
            WHERE actual_class_id = predicted_class_id
        ) AS correct_predictions,

        COUNT(*) FILTER (
            WHERE actual_class_id <> predicted_class_id
        ) AS incorrect_predictions,

        ROUND(
            COUNT(*) FILTER (
                WHERE actual_class_id = predicted_class_id
            )::NUMERIC
            /
            NULLIF(COUNT(*), 0),
            4
        ) AS accuracy

    FROM prediction

    WHERE model_id = p_model_id
      AND status = 'completed'
      AND actual_class_id IS NOT NULL;
$$;

-- ============================================================================
-- 16. SAMPLE DATA
-- ============================================================================

INSERT INTO dataset (
    dataset_name,
    description,
    dataset_version,
    owner_name
)
VALUES
(
    'RGB Object Classification',
    'Controlled RGB classification dataset.',
    '1.0',
    'Computer Vision Research Team'
),
(
    'Industrial Surface Inspection',
    'Dataset for visual inspection experiments.',
    '1.0',
    'Quality Engineering Team'
);

INSERT INTO image_class(class_name, description)
VALUES
('red', 'Predominantly red images'),
('green', 'Predominantly green images'),
('blue', 'Predominantly blue images'),
('defect', 'Images containing industrial surface defects');

INSERT INTO image_asset (
    dataset_id,
    class_id,
    file_name,
    storage_uri,
    content_sha256,
    mime_type,
    width,
    height,
    channels,
    bit_depth,
    file_size_bytes,
    status,
    validated_at
)
VALUES
(
    1, 1, 'red_001.ppm',
    'storage/datasets/red_001.ppm',
    repeat('a', 64), 'image/x-portable-pixmap',
    256, 256, 3, 8, 196608, 'validated', NOW()
),
(
    1, 2, 'green_001.ppm',
    'storage/datasets/green_001.ppm',
    repeat('b', 64), 'image/x-portable-pixmap',
    256, 256, 3, 8, 196608, 'validated', NOW()
),
(
    1, 3, 'blue_001.ppm',
    'storage/datasets/blue_001.ppm',
    repeat('c', 64), 'image/x-portable-pixmap',
    256, 256, 3, 8, 196608, 'validated', NOW()
),
(
    1, 1, 'red_002.ppm',
    'storage/datasets/red_002.ppm',
    repeat('d', 64), 'image/x-portable-pixmap',
    256, 256, 3, 8, 196608, 'validated', NOW()
),
(
    2, 4, 'surface_defect_001.ppm',
    'storage/inspection/defect_001.ppm',
    repeat('e', 64), 'image/x-portable-pixmap',
    512, 512, 3, 8, 786432, 'validated', NOW()
),
(
    2, NULL, 'rejected_image.ppm',
    'storage/inspection/rejected.ppm',
    repeat('f', 64), 'image/x-portable-pixmap',
    512, 512, 3, 8, 786432, 'rejected', NULL
);

INSERT INTO dataset_assignment (
    image_id,
    split,
    assignment_method
)
VALUES
(1, 'training', 'stratified'),
(2, 'training', 'stratified'),
(3, 'validation', 'stratified'),
(4, 'testing', 'stratified'),
(5, 'training', 'manual'),
(6, 'training', 'manual');

INSERT INTO preprocessing_profile (
    profile_name,
    target_width,
    target_height,
    resize_method,
    color_space,
    normalization_method,
    mean_red,
    mean_green,
    mean_blue
)
VALUES
(
    'RGB 224 Standard',
    224,
    224,
    'bilinear',
    'RGB',
    'zero_one',
    0.485,
    0.456,
    0.406
),
(
    'RGB 32 Baseline',
    32,
    32,
    'nearest',
    'RGB',
    'minus_one_one',
    NULL,
    NULL,
    NULL
);

INSERT INTO augmentation_policy (
    policy_name,
    horizontal_flip,
    vertical_flip,
    rotation_enabled,
    maximum_rotation_degrees,
    brightness_enabled,
    minimum_brightness,
    maximum_brightness,
    probability,
    random_seed
)
VALUES
(
    'Training Augmentation',
    TRUE,
    TRUE,
    TRUE,
    15,
    TRUE,
    0.8,
    1.2,
    0.75,
    42
),
(
    'No Augmentation',
    FALSE,
    FALSE,
    FALSE,
    0,
    FALSE,
    1,
    1,
    1,
    42
);

INSERT INTO preprocessing_run (
    image_id,
    profile_id,
    status,
    started_at
)
VALUES
(1, 1, 'running', NOW()),
(2, 1, 'running', NOW()),
(3, 1, 'pending', NULL);

-- ============================================================================
-- 17. SAMPLE MODEL AND EVALUATION
-- ============================================================================

INSERT INTO classification_model (
    model_name,
    model_version,
    architecture,
    framework,
    preprocessing_profile_id,
    augmentation_policy_id,
    status,
    model_artifact_uri,
    training_dataset_id
)
VALUES
(
    'RGB Centroid Classifier',
    '1.0.0',
    'NearestCentroid',
    'Custom',
    2,
    1,
    'created',
    'models/rgb-centroid-1.0.0.bin',
    1
);

-- Demonstrate valid model state transitions.

UPDATE classification_model
SET status = 'training'
WHERE model_id = 1;

UPDATE classification_model
SET
    status = 'trained',
    trained_at = NOW()
WHERE model_id = 1;

INSERT INTO model_evaluation (
    model_id,
    dataset_id,
    evaluated_split,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    macro_f1,
    inference_latency_ms,
    total_samples,
    correct_predictions
)
VALUES
(
    1, 1, 'validation',
    0.92, 0.91, 0.90, 0.905,
    0.905, 12.5, 100, 92
);

UPDATE classification_model
SET status = 'validated'
WHERE model_id = 1;

UPDATE classification_model
SET
    status = 'active',
    activated_at = NOW()
WHERE model_id = 1;

-- ============================================================================
-- 18. SAMPLE AUGMENTATION RECORD
-- ============================================================================

INSERT INTO augmentation_run (
    source_image_id,
    generated_image_id,
    augmentation_policy_id,
    operation,
    random_seed,
    parameters
)
VALUES
(
    1,
    2,
    1,
    'horizontal_flip',
    42,
    '{"axis": "horizontal", "probability": 0.75}'::jsonb
);

-- ============================================================================
-- 19. SAMPLE FEATURE RECORD
-- ============================================================================

INSERT INTO feature_vector (
    image_id,
    preprocessing_run_id,
    feature_version,
    feature_values,
    feature_dimension,
    extraction_duration_ms
)
VALUES
(
    1,
    1,
    'rgb-statistics-v1',
    '{
        "red_mean": 0.90,
        "green_mean": 0.08,
        "blue_mean": 0.06,
        "red_std": 0.10,
        "green_std": 0.03,
        "blue_std": 0.02
    }'::jsonb,
    6,
    5
);

-- ============================================================================
-- 20. SAMPLE PREDICTIONS
-- ============================================================================

INSERT INTO prediction (
    image_id,
    model_id,
    predicted_class_id,
    actual_class_id,
    confidence,
    probabilities,
    status,
    inference_latency_ms
)
VALUES
(
    1, 1, 1, 1, 0.96,
    '{"red": 0.96, "green": 0.03, "blue": 0.01}'::jsonb,
    'completed',
    10.5
),
(
    2, 1, 2, 2, 0.89,
    '{"red": 0.05, "green": 0.89, "blue": 0.06}'::jsonb,
    'completed',
    11.2
),
(
    3, 1, 1, 3, 0.73,
    '{"red": 0.73, "green": 0.12, "blue": 0.15}'::jsonb,
    'completed',
    14.8
),
(
    4, 1, NULL, NULL, NULL,
    '{}'::jsonb,
    'failed',
    0
);

-- ============================================================================
-- 21. TRANSACTIONAL PREPROCESSING COMPLETION
-- ============================================================================

SAVEPOINT preprocessing_completion;

SELECT complete_preprocessing(
    1,
    'processed/red_001_224.ppm',
    15
);

-- The savepoint permits rollback of this operation independently.
RELEASE SAVEPOINT preprocessing_completion;

COMMIT;

-- ============================================================================
-- 22. RETRIEVAL QUERIES
-- ============================================================================

-- A. Dataset inventory.

SELECT
    dataset_name,
    dataset_version,
    owner_name,
    is_active
FROM dataset
ORDER BY dataset_name;

-- B. Image inventory with labels and split assignments.

SELECT
    image.file_name,
    dataset.dataset_name,
    class.class_name,
    assignment.split,
    image.status
FROM image_asset image
JOIN dataset
    ON dataset.dataset_id = image.dataset_id
LEFT JOIN image_class class
    ON class.class_id = image.class_id
LEFT JOIN dataset_assignment assignment
    ON assignment.image_id = image.image_id
ORDER BY image.image_id;

-- C. Preprocessing queue.

SELECT
    run.preprocessing_run_id,
    image.file_name,
    profile.profile_name,
    run.status,
    run.created_at
FROM preprocessing_run run
JOIN image_asset image
    ON image.image_id = run.image_id
JOIN preprocessing_profile profile
    ON profile.profile_id = run.profile_id
WHERE run.status IN ('pending', 'running')
ORDER BY run.created_at;

-- ============================================================================
-- 23. VALIDATION QUERIES
-- ============================================================================

-- A. Images missing dataset split assignments.

SELECT image_id, file_name
FROM image_asset
WHERE status <> 'rejected'
  AND NOT EXISTS (
      SELECT 1
      FROM dataset_assignment assignment
      WHERE assignment.image_id = image_asset.image_id
  );

-- B. Dataset class distribution and imbalance.

WITH class_counts AS (
    SELECT
        dataset.dataset_name,
        class.class_name,
        COUNT(image.image_id) AS image_count

    FROM dataset
    JOIN image_asset image
        ON image.dataset_id = dataset.dataset_id

    JOIN image_class class
        ON class.class_id = image.class_id

    GROUP BY dataset.dataset_name, class.class_name
)
SELECT
    dataset_name,
    class_name,
    image_count,
    ROUND(
        image_count::NUMERIC
        /
        NULLIF(
            SUM(image_count) OVER (PARTITION BY dataset_name),
            0
        ),
        4
    ) AS class_proportion
FROM class_counts
ORDER BY dataset_name, class_name;

-- C. Check preprocessing failures.

SELECT *
FROM preprocessing_run
WHERE status = 'failed'
ORDER BY created_at DESC;

-- D. Check generated images with incorrect split assignments.

SELECT
    augmentation_run.augmentation_run_id,
    source_assignment.split AS source_split,
    generated_assignment.split AS generated_split
FROM augmentation_run
JOIN dataset_assignment source_assignment
    ON source_assignment.image_id =
       augmentation_run.source_image_id
JOIN dataset_assignment generated_assignment
    ON generated_assignment.image_id =
       augmentation_run.generated_image_id
WHERE source_assignment.split <> 'training'
   OR generated_assignment.split <> 'training';

-- ============================================================================
-- 24. ELIGIBILITY AND EVALUATION
-- ============================================================================

-- A. Model activation eligibility.

SELECT *
FROM model_activation_eligibility(1);

-- B. Model evaluation report.

SELECT *
FROM model_performance_report
ORDER BY model_id;

-- C. Prediction accuracy.

SELECT *
FROM classification_accuracy(1);

-- D. Misclassification analysis.

SELECT *
FROM classification_error_report
ORDER BY error_count DESC;

-- E. Average inference latency.

SELECT
    model_id,
    COUNT(*) AS predictions,
    ROUND(AVG(inference_latency_ms), 3) AS average_latency_ms,
    ROUND(
        PERCENTILE_CONT(0.95)
        WITHIN GROUP (ORDER BY inference_latency_ms)::NUMERIC,
        3
    ) AS p95_latency_ms
FROM prediction
WHERE status = 'completed'
GROUP BY model_id;

-- ============================================================================
-- 25. AUGMENTATION AND PREPROCESSING REPORTING
-- ============================================================================

SELECT *
FROM preprocessing_performance
ORDER BY profile_name;

SELECT
    policy.policy_name,
    augmentation.operation,
    COUNT(*) AS generated_variants
FROM augmentation_run augmentation
JOIN augmentation_policy policy
    ON policy.augmentation_policy_id =
       augmentation.augmentation_policy_id
GROUP BY policy.policy_name, augmentation.operation
ORDER BY policy.policy_name, augmentation.operation;

-- ============================================================================
-- 26. MODEL AUDIT HISTORY
-- ============================================================================

SELECT
    model.model_name,
    audit.old_status,
    audit.new_status,
    audit.changed_at,
    audit.database_user
FROM model_status_audit audit
JOIN classification_model model
    ON model.model_id = audit.model_id
ORDER BY audit.changed_at;

-- ============================================================================
-- 27. ADDITIONAL OPERATIONAL CTE
-- ============================================================================

WITH image_quality AS (
    SELECT
        dataset_id,
        COUNT(*) AS total_images,
        COUNT(*) FILTER (
            WHERE status = 'validated'
        ) AS validated_images,
        COUNT(*) FILTER (
            WHERE status = 'rejected'
        ) AS rejected_images,
        AVG(width * height) AS average_pixel_count
    FROM image_asset
    GROUP BY dataset_id
)
SELECT
    dataset.dataset_name,
    COALESCE(quality.total_images, 0) AS total_images,
    COALESCE(quality.validated_images, 0) AS validated_images,
    COALESCE(quality.rejected_images, 0) AS rejected_images,
    ROUND(quality.average_pixel_count, 2) AS average_pixel_count
FROM dataset
LEFT JOIN image_quality quality
    ON quality.dataset_id = dataset.dataset_id
ORDER BY dataset.dataset_name;

-- ============================================================================
-- 28. DATABASE SECURITY AND OPERATIONAL NOTES
-- ============================================================================

/*
Production deployment controls:

1. Store image binaries in controlled object storage rather than database
   JSON or unrestricted large binary columns.

2. Store only trusted storage identifiers in storage_uri.

3. Use dedicated database roles for ingestion, processing, inference,
   evaluation, and administration.

4. Grant EXECUTE on workflow functions selectively.

5. Use row-level security when dataset ownership requires tenant isolation.

6. Keep database credentials outside application source files.

7. Record immutable dataset versions and model artifacts.

8. Enforce image content verification in the ingestion service.

9. Never trust a file extension as proof of image format.

10. Validate augmentation parameters in the image-processing application.

11. Apply model promotion approvals and deployment controls separately
    from numerical evaluation.

12. The activation threshold in this example is illustrative and must be
    calibrated against operational requirements.

13. Database constraints protect relational integrity but cannot verify
    that an external image file contains the declared pixel data.

14. Split assignments should be immutable after training begins unless
    an audited dataset-versioning operation is performed.

15. For large workloads, partition prediction and audit tables by time,
    and use retention policies.
*/

SET search_path TO public;
