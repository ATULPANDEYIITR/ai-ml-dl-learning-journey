/*
 * Modern CNN Architecture Research and Experiment Database
 *
 * PostgreSQL 15+
 *
 * Covers:
 * - CNN architecture registry
 * - ResNet, DenseNet, EfficientNet families
 * - Architecture configurations and compound scaling
 * - Layer-level computational metadata
 * - Residual and dense connectivity
 * - MBConv and squeeze-and-excitation
 * - Dataset and experiment management
 * - Training runs and evaluation metrics
 * - Resource budgets and deployment policies
 * - Model comparison and experiment eligibility
 * - Transactional experiment registration
 *
 * Run:
 * psql -U postgres -d modern_cnn_lab -f modern_cnn.sql
 */

BEGIN;

CREATE SCHEMA IF NOT EXISTS cnn_lab;

SET search_path TO cnn_lab, public;

CREATE TYPE architecture_family AS ENUM (
    'RESNET',
    'DENSENET',
    'EFFICIENTNET'
);

CREATE TYPE layer_type AS ENUM (
    'CONVOLUTION',
    'DEPTHWISE',
    'POOLING',
    'NORMALIZATION',
    'ACTIVATION',
    'RESIDUAL_ADD',
    'CONCATENATION',
    'SQUEEZE_EXCITATION',
    'DENSE'
);

CREATE TYPE experiment_status AS ENUM (
    'CREATED',
    'VALIDATING',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE deployment_target AS ENUM (
    'EDGE',
    'MOBILE',
    'SERVER',
    'CLOUD'
);

CREATE TYPE run_status AS ENUM (
    'QUEUED',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TABLE researchers (
    researcher_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    department VARCHAR(150) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT researcher_email_check
        CHECK (POSITION('@' IN email) > 1)
);

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(150) NOT NULL UNIQUE,
    domain VARCHAR(100) NOT NULL,
    input_channels INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    image_width INTEGER NOT NULL,
    class_count INTEGER NOT NULL,
    training_samples BIGINT NOT NULL,
    validation_samples BIGINT NOT NULL,
    test_samples BIGINT NOT NULL,
    license_name VARCHAR(150) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT dataset_dimensions_check CHECK (
        input_channels > 0
        AND image_height > 0
        AND image_width > 0
        AND class_count > 1
    ),

    CONSTRAINT dataset_samples_check CHECK (
        training_samples > 0
        AND validation_samples >= 0
        AND test_samples >= 0
    )
);

CREATE TABLE architectures (
    architecture_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    architecture_name VARCHAR(120) NOT NULL UNIQUE,
    family architecture_family NOT NULL,
    description TEXT NOT NULL,
    input_channels INTEGER NOT NULL,
    input_height INTEGER NOT NULL,
    input_width INTEGER NOT NULL,
    output_classes INTEGER NOT NULL,
    total_parameters BIGINT NOT NULL DEFAULT 0,
    total_flops BIGINT NOT NULL DEFAULT 0,
    layer_count INTEGER NOT NULL DEFAULT 0,
    estimated_activation_bytes BIGINT NOT NULL DEFAULT 0,
    is_validated BOOLEAN NOT NULL DEFAULT FALSE,
    created_by BIGINT REFERENCES researchers(researcher_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT architecture_dimensions_check CHECK (
        input_channels > 0
        AND input_height > 0
        AND input_width > 0
        AND output_classes > 1
    ),

    CONSTRAINT architecture_metrics_check CHECK (
        total_parameters >= 0
        AND total_flops >= 0
        AND layer_count >= 0
        AND estimated_activation_bytes >= 0
    )
);

CREATE TABLE architecture_configurations (
    configuration_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    architecture_id BIGINT NOT NULL
        REFERENCES architectures(architecture_id) ON DELETE CASCADE,
    configuration_name VARCHAR(100) NOT NULL,
    version_number INTEGER NOT NULL DEFAULT 1,
    growth_rate INTEGER,
    compression_factor NUMERIC(5,4),
    width_multiplier NUMERIC(6,3),
    depth_multiplier NUMERIC(6,3),
    resolution_multiplier NUMERIC(6,3),
    expansion_factor INTEGER,
    se_reduction INTEGER,
    dropout_rate NUMERIC(5,4),
    configuration JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT configuration_version_check CHECK (
        version_number > 0
    ),

    CONSTRAINT configuration_growth_check CHECK (
        growth_rate IS NULL OR growth_rate > 0
    ),

    CONSTRAINT configuration_compression_check CHECK (
        compression_factor IS NULL
        OR compression_factor BETWEEN 0.01 AND 1.0
    ),

    CONSTRAINT configuration_scaling_check CHECK (
        (width_multiplier IS NULL OR width_multiplier > 0)
        AND (depth_multiplier IS NULL OR depth_multiplier > 0)
        AND (resolution_multiplier IS NULL OR resolution_multiplier > 0)
    ),

    CONSTRAINT configuration_dropout_check CHECK (
        dropout_rate IS NULL OR dropout_rate BETWEEN 0 AND 1
    ),

    UNIQUE (architecture_id, configuration_name, version_number)
);

CREATE TABLE architecture_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    architecture_id BIGINT NOT NULL
        REFERENCES architectures(architecture_id) ON DELETE CASCADE,
    layer_name VARCHAR(150) NOT NULL,
    layer_index INTEGER NOT NULL,
    type layer_type NOT NULL,
    input_channels INTEGER NOT NULL,
    input_height INTEGER NOT NULL,
    input_width INTEGER NOT NULL,
    output_channels INTEGER NOT NULL,
    output_height INTEGER NOT NULL,
    output_width INTEGER NOT NULL,
    kernel_size INTEGER,
    stride INTEGER,
    groups INTEGER NOT NULL DEFAULT 1,
    parameter_count BIGINT NOT NULL DEFAULT 0,
    flop_count BIGINT NOT NULL DEFAULT 0,
    configuration JSONB NOT NULL DEFAULT '{}'::JSONB,

    CONSTRAINT layer_index_check CHECK (layer_index >= 0),

    CONSTRAINT layer_dimensions_check CHECK (
        input_channels > 0
        AND input_height > 0
        AND input_width > 0
        AND output_channels > 0
        AND output_height > 0
        AND output_width > 0
    ),

    CONSTRAINT layer_cost_check CHECK (
        parameter_count >= 0 AND flop_count >= 0
    ),

    CONSTRAINT layer_group_check CHECK (groups > 0),

    UNIQUE (architecture_id, layer_index),
    UNIQUE (architecture_id, layer_name)
);

-- Connectivity edges preserve the actual computation graph.
CREATE TABLE layer_connections (
    connection_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    architecture_id BIGINT NOT NULL
        REFERENCES architectures(architecture_id) ON DELETE CASCADE,
    source_layer_id BIGINT NOT NULL
        REFERENCES architecture_layers(layer_id) ON DELETE CASCADE,
    target_layer_id BIGINT NOT NULL
        REFERENCES architecture_layers(layer_id) ON DELETE CASCADE,
    connection_type VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT connection_type_check CHECK (
        connection_type IN (
            'SEQUENTIAL',
            'IDENTITY_SHORTCUT',
            'PROJECTION_SHORTCUT',
            'DENSE_CONCATENATION'
        )
    ),

    CONSTRAINT no_self_connection CHECK (
        source_layer_id <> target_layer_id
    ),

    UNIQUE (
        architecture_id,
        source_layer_id,
        target_layer_id,
        connection_type
    )
);

CREATE TABLE deployment_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name VARCHAR(120) NOT NULL UNIQUE,
    target deployment_target NOT NULL,
    maximum_parameters BIGINT NOT NULL,
    maximum_flops BIGINT NOT NULL,
    maximum_activation_bytes BIGINT NOT NULL,
    maximum_input_resolution INTEGER NOT NULL,
    require_validated_architecture BOOLEAN NOT NULL DEFAULT TRUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT policy_limits_check CHECK (
        maximum_parameters > 0
        AND maximum_flops > 0
        AND maximum_activation_bytes > 0
        AND maximum_input_resolution > 0
    )
);

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_code VARCHAR(60) NOT NULL UNIQUE,
    architecture_id BIGINT NOT NULL
        REFERENCES architectures(architecture_id),
    configuration_id BIGINT
        REFERENCES architecture_configurations(configuration_id),
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    researcher_id BIGINT NOT NULL REFERENCES researchers(researcher_id),
    policy_id BIGINT NOT NULL REFERENCES deployment_policies(policy_id),
    status experiment_status NOT NULL DEFAULT 'CREATED',
    learning_rate NUMERIC(12,10) NOT NULL,
    batch_size INTEGER NOT NULL,
    epochs INTEGER NOT NULL,
    optimizer VARCHAR(60) NOT NULL,
    random_seed INTEGER NOT NULL,
    failure_reason TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT experiment_training_check CHECK (
        learning_rate > 0
        AND learning_rate <= 1
        AND batch_size > 0
        AND epochs > 0
    ),

    CONSTRAINT experiment_completion_check CHECK (
        completed_at IS NULL
        OR (started_at IS NOT NULL AND completed_at >= started_at)
    ),

    CONSTRAINT experiment_failure_check CHECK (
        status <> 'FAILED' OR failure_reason IS NOT NULL
    )
);

CREATE TABLE training_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    run_number INTEGER NOT NULL,
    status run_status NOT NULL DEFAULT 'QUEUED',
    hardware_name VARCHAR(120) NOT NULL,
    accelerator_type VARCHAR(60),
    accelerator_memory_gb NUMERIC(8,2),
    peak_memory_gb NUMERIC(8,2),
    duration_seconds NUMERIC(14,3),
    checkpoint_uri TEXT,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT run_number_check CHECK (run_number > 0),

    CONSTRAINT run_memory_check CHECK (
        accelerator_memory_gb IS NULL OR accelerator_memory_gb > 0
    ),

    CONSTRAINT run_duration_check CHECK (
        duration_seconds IS NULL OR duration_seconds >= 0
    ),

    CONSTRAINT run_error_check CHECK (
        status <> 'FAILED' OR error_message IS NOT NULL
    ),

    UNIQUE (experiment_id, run_number)
);

CREATE TABLE evaluation_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES training_runs(run_id) ON DELETE CASCADE,
    metric_name VARCHAR(80) NOT NULL,
    metric_value NUMERIC(18,8) NOT NULL,
    evaluation_split VARCHAR(20) NOT NULL,
    epoch_number INTEGER,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT metric_split_check CHECK (
        evaluation_split IN ('TRAIN', 'VALIDATION', 'TEST')
    ),

    CONSTRAINT metric_epoch_check CHECK (
        epoch_number IS NULL OR epoch_number >= 0
    ),

    UNIQUE (run_id, metric_name, evaluation_split, epoch_number)
);

CREATE TABLE experiment_eligibility (
    eligibility_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    policy_id BIGINT NOT NULL
        REFERENCES deployment_policies(policy_id),
    eligible BOOLEAN NOT NULL,
    rejection_reason TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT eligibility_reason_check CHECK (
        eligible = TRUE OR rejection_reason IS NOT NULL
    ),

    UNIQUE (experiment_id, policy_id)
);

-- Indexed access paths for experiment and architecture analysis.
CREATE INDEX idx_architecture_family
    ON architectures(family);

CREATE INDEX idx_architecture_parameters
    ON architectures(total_parameters);

CREATE INDEX idx_layer_architecture_order
    ON architecture_layers(architecture_id, layer_index);

CREATE INDEX idx_layer_connections_target
    ON layer_connections(target_layer_id);

CREATE INDEX idx_experiment_status_created
    ON experiments(status, created_at DESC);

CREATE INDEX idx_experiment_architecture
    ON experiments(architecture_id, dataset_id);

CREATE INDEX idx_runs_experiment_status
    ON training_runs(experiment_id, status);

CREATE INDEX idx_metrics_run_name
    ON evaluation_metrics(run_id, metric_name);

CREATE INDEX idx_configuration_json
    ON architecture_configurations USING GIN(configuration);

CREATE OR REPLACE FUNCTION validate_layer_connection()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    source_architecture BIGINT;
    target_architecture BIGINT;
    source_index INTEGER;
    target_index INTEGER;
BEGIN
    SELECT architecture_id, layer_index
    INTO source_architecture, source_index
    FROM architecture_layers
    WHERE layer_id = NEW.source_layer_id;

    SELECT architecture_id, layer_index
    INTO target_architecture, target_index
    FROM architecture_layers
    WHERE layer_id = NEW.target_layer_id;

    IF source_architecture IS NULL OR target_architecture IS NULL THEN
        RAISE EXCEPTION 'Connection references an unknown layer.';
    END IF;

    IF source_architecture <> NEW.architecture_id
       OR target_architecture <> NEW.architecture_id THEN
        RAISE EXCEPTION
            'Connection layers must belong to the specified architecture.';
    END IF;

    IF source_index >= target_index THEN
        RAISE EXCEPTION
            'Connections must follow topological layer order.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_layer_connection
BEFORE INSERT OR UPDATE ON layer_connections
FOR EACH ROW
EXECUTE FUNCTION validate_layer_connection();

CREATE OR REPLACE FUNCTION refresh_architecture_metrics()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    affected_architecture BIGINT;
BEGIN
    affected_architecture := COALESCE(
        NEW.architecture_id,
        OLD.architecture_id
    );

    UPDATE architectures
    SET
        total_parameters = (
            SELECT COALESCE(SUM(parameter_count), 0)
            FROM architecture_layers
            WHERE architecture_id = affected_architecture
        ),
        total_flops = (
            SELECT COALESCE(SUM(flop_count), 0)
            FROM architecture_layers
            WHERE architecture_id = affected_architecture
        ),
        layer_count = (
            SELECT COUNT(*)
            FROM architecture_layers
            WHERE architecture_id = affected_architecture
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE architecture_id = affected_architecture;

    RETURN COALESCE(NEW, OLD);
END;
$$;

CREATE TRIGGER trg_refresh_architecture_metrics
AFTER INSERT OR UPDATE OR DELETE ON architecture_layers
FOR EACH ROW
EXECUTE FUNCTION refresh_architecture_metrics();

CREATE OR REPLACE FUNCTION evaluate_architecture_eligibility(
    p_architecture_id BIGINT,
    p_policy_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    rejection_reason TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    a architectures%ROWTYPE;
    p deployment_policies%ROWTYPE;
    reason TEXT := NULL;
BEGIN
    SELECT *
    INTO a
    FROM architectures
    WHERE architecture_id = p_architecture_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Architecture % does not exist.',
            p_architecture_id;
    END IF;

    SELECT *
    INTO p
    FROM deployment_policies
    WHERE policy_id = p_policy_id
      AND is_active = TRUE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Active deployment policy not found.';
    END IF;

    IF a.total_parameters > p.maximum_parameters THEN
        reason := 'Parameter budget exceeded';
    ELSIF a.total_flops > p.maximum_flops THEN
        reason := 'FLOP budget exceeded';
    ELSIF a.estimated_activation_bytes >
          p.maximum_activation_bytes THEN
        reason := 'Activation memory budget exceeded';
    ELSIF GREATEST(a.input_height, a.input_width) >
          p.maximum_input_resolution THEN
        reason := 'Input resolution exceeded';
    ELSIF p.require_validated_architecture
          AND NOT a.is_validated THEN
        reason := 'Architecture validation required';
    END IF;

    RETURN QUERY
    SELECT reason IS NULL, reason;
END;
$$;

CREATE OR REPLACE PROCEDURE register_experiment(
    p_code VARCHAR,
    p_architecture_id BIGINT,
    p_dataset_id BIGINT,
    p_researcher_id BIGINT,
    p_policy_id BIGINT,
    p_learning_rate NUMERIC,
    p_batch_size INTEGER,
    p_epochs INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    decision RECORD;
BEGIN
    IF p_learning_rate <= 0 OR p_learning_rate > 1 THEN
        RAISE EXCEPTION 'Learning rate must be in (0, 1].';
    END IF;

    IF p_batch_size <= 0 OR p_epochs <= 0 THEN
        RAISE EXCEPTION 'Batch size and epochs must be positive.';
    END IF;

    SELECT *
    INTO decision
    FROM evaluate_architecture_eligibility(
        p_architecture_id,
        p_policy_id
    );

    IF NOT decision.eligible THEN
        RAISE EXCEPTION
            'Experiment rejected: %',
            decision.rejection_reason;
    END IF;

    INSERT INTO experiments (
        experiment_code,
        architecture_id,
        dataset_id,
        researcher_id,
        policy_id,
        learning_rate,
        batch_size,
        epochs,
        optimizer,
        random_seed
    )
    VALUES (
        p_code,
        p_architecture_id,
        p_dataset_id,
        p_researcher_id,
        p_policy_id,
        p_learning_rate,
        p_batch_size,
        p_epochs,
        'AdamW',
        42
    );

END;
$$;

-- Reports architectures with their estimated resource profiles.
CREATE OR REPLACE VIEW architecture_comparison AS
SELECT
    a.architecture_id,
    a.architecture_name,
    a.family,
    a.input_height,
    a.input_width,
    a.output_classes,
    a.total_parameters,
    ROUND(a.total_parameters / 1000000.0, 3)
        AS parameters_millions,
    a.total_flops,
    ROUND(a.total_flops / 1000000000.0, 3)
        AS gflops,
    a.layer_count,
    a.estimated_activation_bytes,
    a.is_validated
FROM architectures a;

-- Experiment performance report.
CREATE OR REPLACE VIEW experiment_performance AS
SELECT
    e.experiment_code,
    a.architecture_name,
    d.dataset_name,
    e.status AS experiment_status,
    COUNT(DISTINCT tr.run_id) AS run_count,
    AVG(tr.duration_seconds) AS average_duration_seconds,
    MAX(
        CASE
            WHEN em.metric_name = 'accuracy'
             AND em.evaluation_split = 'VALIDATION'
            THEN em.metric_value
        END
    ) AS validation_accuracy,
    MAX(
        CASE
            WHEN em.metric_name = 'accuracy'
             AND em.evaluation_split = 'TEST'
            THEN em.metric_value
        END
    ) AS test_accuracy
FROM experiments e
JOIN architectures a
    ON a.architecture_id = e.architecture_id
JOIN datasets d
    ON d.dataset_id = e.dataset_id
LEFT JOIN training_runs tr
    ON tr.experiment_id = e.experiment_id
LEFT JOIN evaluation_metrics em
    ON em.run_id = tr.run_id
GROUP BY
    e.experiment_code,
    a.architecture_name,
    d.dataset_name,
    e.status;

-- Seed researchers.
INSERT INTO researchers (
    full_name, email, department
)
VALUES
    ('Ananya Sharma', 'ananya@example.org', 'Computer Vision'),
    ('Rahul Verma', 'rahul@example.org', 'Deep Learning'),
    ('Meera Singh', 'meera@example.org', 'Model Optimization');

-- Seed datasets.
INSERT INTO datasets (
    dataset_name,
    domain,
    input_channels,
    image_height,
    image_width,
    class_count,
    training_samples,
    validation_samples,
    test_samples,
    license_name
)
VALUES
    (
        'ImageNet-1K',
        'Natural Images',
        3, 224, 224, 1000,
        1281167, 50000, 50000,
        'Research Dataset License'
    ),
    (
        'Industrial-Defects',
        'Industrial Inspection',
        3, 128, 128, 12,
        24000, 4000, 4000,
        'Internal Research License'
    );

-- Architecture records.
INSERT INTO architectures (
    architecture_name,
    family,
    description,
    input_channels,
    input_height,
    input_width,
    output_classes,
    created_by
)
SELECT
    'ResNet-18',
    'RESNET',
    'Basic residual blocks with identity and projection shortcuts.',
    3, 224, 224, 1000,
    researcher_id
FROM researchers
WHERE email = 'ananya@example.org';

INSERT INTO architectures (
    architecture_name,
    family,
    description,
    input_channels,
    input_height,
    input_width,
    output_classes,
    created_by
)
SELECT
    'DenseNet-121',
    'DENSENET',
    'Dense blocks with feature reuse and transition compression.',
    3, 224, 224, 1000,
    researcher_id
FROM researchers
WHERE email = 'rahul@example.org';

INSERT INTO architectures (
    architecture_name,
    family,
    description,
    input_channels,
    input_height,
    input_width,
    output_classes,
    created_by
)
SELECT
    'EfficientNet-B0',
    'EFFICIENTNET',
    'Compound scaling with MBConv and squeeze-excitation.',
    3, 224, 224, 1000,
    researcher_id
FROM researchers
WHERE email = 'meera@example.org';

-- Architecture configuration examples.
INSERT INTO architecture_configurations (
    architecture_id,
    configuration_name,
    version_number,
    configuration
)
SELECT
    architecture_id,
    'residual-basic',
    1,
    '{"block_type":"basic","stages":[2,2,2,2]}'::JSONB
FROM architectures
WHERE architecture_name = 'ResNet-18';

INSERT INTO architecture_configurations (
    architecture_id,
    configuration_name,
    version_number,
    growth_rate,
    compression_factor,
    configuration
)
SELECT
    architecture_id,
    'dense-standard',
    1,
    32,
    0.5,
    '{"blocks":[6,12,24,16]}'::JSONB
FROM architectures
WHERE architecture_name = 'DenseNet-121';

INSERT INTO architecture_configurations (
    architecture_id,
    configuration_name,
    version_number,
    width_multiplier,
    depth_multiplier,
    resolution_multiplier,
    expansion_factor,
    se_reduction,
    configuration
)
SELECT
    architecture_id,
    'efficientnet-b0',
    1,
    1.0,
    1.0,
    1.0,
    6,
    4,
    '{"variant":"B0","resolution":224,"activation":"SiLU"}'::JSONB
FROM architectures
WHERE architecture_name = 'EfficientNet-B0';

-- Seed deployment policies.
INSERT INTO deployment_policies (
    policy_name,
    target,
    maximum_parameters,
    maximum_flops,
    maximum_activation_bytes,
    maximum_input_resolution,
    require_validated_architecture
)
VALUES
    (
        'Mobile-Lite',
        'MOBILE',
        10000000,
        5000000000,
        128 * 1024 * 1024,
        300,
        TRUE
    ),
    (
        'Server-Standard',
        'SERVER',
        100000000,
        100000000000,
        1024 * 1024 * 1024,
        600,
        TRUE
    ),
    (
        'Edge-Restricted',
        'EDGE',
        5000000,
        2000000000,
        64 * 1024 * 1024,
        224,
        TRUE
    );

-- Representative layer metadata for ResNet-18.
INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    layer_index,
    type,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    kernel_size,
    stride,
    parameter_count,
    flop_count
)
SELECT
    architecture_id,
    'stem_conv',
    0,
    'CONVOLUTION',
    3, 224, 224,
    64, 112, 112,
    7, 2,
    9408,
    236027904
FROM architectures
WHERE architecture_name = 'ResNet-18';

INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    layer_index,
    type,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    kernel_size,
    stride,
    parameter_count,
    flop_count
)
SELECT
    architecture_id,
    'stage1_block1_conv1',
    1,
    'CONVOLUTION',
    64, 56, 56,
    64, 56, 56,
    3, 1,
    36864,
    231211008
FROM architectures
WHERE architecture_name = 'ResNet-18';

-- Additional layer examples distinguish the two connectivity mechanisms.
INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    layer_index,
    type,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    parameter_count,
    flop_count
)
SELECT
    architecture_id,
    'dense_feature_concat',
    0,
    'CONCATENATION',
    256, 28, 28,
    288, 28, 28,
    0,
    225792
FROM architectures
WHERE architecture_name = 'DenseNet-121';

INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    layer_index,
    type,
    input_channels,
    input_height,
    input_width,
    output_channels,
    output_height,
    output_width,
    kernel_size,
    stride,
    groups,
    parameter_count,
    flop_count
)
SELECT
    architecture_id,
    'mbconv_depthwise',
    0,
    'DEPTHWISE',
    96, 56, 56,
    96, 56, 56,
    3, 1, 96,
    864,
    5419008
FROM architectures
WHERE architecture_name = 'EfficientNet-B0';

-- Example connectivity: residual projection and identity paths.
INSERT INTO layer_connections (
    architecture_id,
    source_layer_id,
    target_layer_id,
    connection_type
)
SELECT
    source.architecture_id,
    source.layer_id,
    target.layer_id,
    'SEQUENTIAL'
FROM architecture_layers source
JOIN architecture_layers target
    ON target.architecture_id = source.architecture_id
WHERE source.layer_name = 'stem_conv'
  AND target.layer_name = 'stage1_block1_conv1'
  AND source.architecture_id = (
      SELECT architecture_id
      FROM architectures
      WHERE architecture_name = 'ResNet-18'
  );

-- Update validation state after architecture checks.
UPDATE architectures
SET
    is_validated = TRUE,
    estimated_activation_bytes = 64 * 56 * 56 * 4
WHERE architecture_name = 'ResNet-18';

UPDATE architectures
SET
    is_validated = TRUE,
    estimated_activation_bytes = 288 * 28 * 28 * 4
WHERE architecture_name = 'DenseNet-121';

UPDATE architectures
SET
    is_validated = TRUE,
    estimated_activation_bytes = 96 * 56 * 56 * 4
WHERE architecture_name = 'EfficientNet-B0';

-- Transactional registration of an experiment.
CALL register_experiment(
    'EXP-RESNET-001',
    (
        SELECT architecture_id
        FROM architectures
        WHERE architecture_name = 'ResNet-18'
    ),
    (
        SELECT dataset_id
        FROM datasets
        WHERE dataset_name = 'Industrial-Defects'
    ),
    (
        SELECT researcher_id
        FROM researchers
        WHERE email = 'ananya@example.org'
    ),
    (
        SELECT policy_id
        FROM deployment_policies
        WHERE policy_name = 'Server-Standard'
    ),
    0.001,
    32,
    50
);

-- Experiment with a resource-policy rejection is recorded separately.
INSERT INTO experiments (
    experiment_code,
    architecture_id,
    dataset_id,
    researcher_id,
    policy_id,
    status,
    learning_rate,
    batch_size,
    epochs,
    optimizer,
    random_seed
)
SELECT
    'EXP-DENSE-FAILED',
    a.architecture_id,
    d.dataset_id,
    r.researcher_id,
    p.policy_id,
    'FAILED',
    0.001,
    32,
    30,
    'AdamW',
    42
FROM architectures a
CROSS JOIN datasets d
CROSS JOIN researchers r
CROSS JOIN deployment_policies p
WHERE a.architecture_name = 'DenseNet-121'
  AND d.dataset_name = 'Industrial-Defects'
  AND r.email = 'rahul@example.org'
  AND p.policy_name = 'Server-Standard';

-- The failure reason is mandatory for failed experiments.
UPDATE experiments
SET failure_reason = 'Illustrative failure: training data validation failed.'
WHERE experiment_code = 'EXP-DENSE-FAILED';

-- Record a rejected mobile deployment eligibility decision.
INSERT INTO experiment_eligibility (
    experiment_id,
    policy_id,
    eligible,
    rejection_reason
)
SELECT
    e.experiment_id,
    p.policy_id,
    FALSE,
    'Architecture exceeds mobile resource budget.'
FROM experiments e
CROSS JOIN deployment_policies p
WHERE e.experiment_code = 'EXP-RESNET-001'
  AND p.policy_name = 'Mobile-Lite';

-- Successful experiment eligibility decision.
INSERT INTO experiment_eligibility (
    experiment_id,
    policy_id,
    eligible,
    rejection_reason
)
SELECT
    e.experiment_id,
    p.policy_id,
    TRUE,
    NULL
FROM experiments e
CROSS JOIN deployment_policies p
WHERE e.experiment_code = 'EXP-RESNET-001'
  AND p.policy_name = 'Server-Standard';

-- Create a training run.
INSERT INTO training_runs (
    experiment_id,
    run_number,
    status,
    hardware_name,
    accelerator_type,
    accelerator_memory_gb,
    peak_memory_gb,
    started_at
)
SELECT
    experiment_id,
    1,
    'RUNNING',
    'NVIDIA Training Server',
    'GPU',
    24,
    11.5,
    CURRENT_TIMESTAMP
FROM experiments
WHERE experiment_code = 'EXP-RESNET-001';

-- Complete the run.
UPDATE training_runs
SET
    status = 'COMPLETED',
    duration_seconds = 3600.5,
    completed_at = CURRENT_TIMESTAMP,
    checkpoint_uri = 'checkpoints/exp-resnet-001/model.pt'
WHERE run_id = (
    SELECT tr.run_id
    FROM training_runs tr
    JOIN experiments e
        ON e.experiment_id = tr.experiment_id
    WHERE e.experiment_code = 'EXP-RESNET-001'
);

UPDATE experiments
SET
    status = 'COMPLETED',
    started_at = CURRENT_TIMESTAMP - INTERVAL '1 hour',
    completed_at = CURRENT_TIMESTAMP
WHERE experiment_code = 'EXP-RESNET-001';

-- Evaluation records.
INSERT INTO evaluation_metrics (
    run_id,
    metric_name,
    metric_value,
    evaluation_split,
    epoch_number
)
SELECT
    run_id,
    'accuracy',
    0.9475,
    'VALIDATION',
    50
FROM training_runs
WHERE experiment_id = (
    SELECT experiment_id
    FROM experiments
    WHERE experiment_code = 'EXP-RESNET-001'
);

INSERT INTO evaluation_metrics (
    run_id,
    metric_name,
    metric_value,
    evaluation_split,
    epoch_number
)
SELECT
    run_id,
    'accuracy',
    0.9320,
    'TEST',
    50
FROM training_runs
WHERE experiment_id = (
    SELECT experiment_id
    FROM experiments
    WHERE experiment_code = 'EXP-RESNET-001'
);

-- Validate experiment registration and eligibility.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM experiments
        WHERE experiment_code = 'EXP-RESNET-001'
    ) THEN
        RAISE EXCEPTION 'Experiment registration failed.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM architectures
        WHERE total_parameters < 0 OR total_flops < 0
    ) THEN
        RAISE EXCEPTION 'Invalid architecture cost detected.';
    END IF;
END;
$$;

COMMIT;

-- ============================================================
-- RETRIEVAL QUERIES
-- ============================================================

-- Compare architecture families and computational requirements.
SELECT *
FROM architecture_comparison
ORDER BY total_parameters ASC;

-- Retrieve all layers of an architecture.
SELECT
    layer_index,
    layer_name,
    type,
    input_channels,
    output_channels,
    parameter_count,
    flop_count
FROM architecture_layers
WHERE architecture_id = (
    SELECT architecture_id
    FROM architectures
    WHERE architecture_name = 'ResNet-18'
)
ORDER BY layer_index;

-- Inspect dense and residual connectivity.
SELECT
    a.architecture_name,
    source.layer_name AS source_layer,
    target.layer_name AS target_layer,
    c.connection_type
FROM layer_connections c
JOIN architectures a
    ON a.architecture_id = c.architecture_id
JOIN architecture_layers source
    ON source.layer_id = c.source_layer_id
JOIN architecture_layers target
    ON target.layer_id = c.target_layer_id;

-- ============================================================
-- VALIDATION AND ELIGIBILITY
-- ============================================================

SELECT
    a.architecture_name,
    p.policy_name,
    decision.eligible,
    decision.rejection_reason
FROM architectures a
CROSS JOIN deployment_policies p
CROSS JOIN LATERAL evaluate_architecture_eligibility(
    a.architecture_id,
    p.policy_id
) decision
WHERE p.is_active
ORDER BY a.architecture_name, p.policy_name;

-- Detect inconsistent architecture metric totals.
SELECT
    a.architecture_name,
    a.total_parameters,
    COALESCE(SUM(l.parameter_count), 0) AS calculated_parameters,
    a.total_flops,
    COALESCE(SUM(l.flop_count), 0) AS calculated_flops
FROM architectures a
LEFT JOIN architecture_layers l
    ON l.architecture_id = a.architecture_id
GROUP BY
    a.architecture_id,
    a.architecture_name,
    a.total_parameters,
    a.total_flops
HAVING
    a.total_parameters <> COALESCE(SUM(l.parameter_count), 0)
    OR a.total_flops <> COALESCE(SUM(l.flop_count), 0);

-- ============================================================
-- AGGREGATION AND REPORTING
-- ============================================================

SELECT
    family,
    COUNT(*) AS architecture_count,
    ROUND(AVG(total_parameters) / 1000000.0, 3)
        AS average_parameters_millions,
    ROUND(AVG(total_flops) / 1000000000.0, 3)
        AS average_gflops
FROM architectures
GROUP BY family
ORDER BY family;

SELECT *
FROM experiment_performance
ORDER BY experiment_code;

-- Best validation accuracy per architecture.
SELECT
    a.architecture_name,
    MAX(em.metric_value) AS best_validation_accuracy
FROM architectures a
JOIN experiments e
    ON e.architecture_id = a.architecture_id
JOIN training_runs tr
    ON tr.experiment_id = e.experiment_id
JOIN evaluation_metrics em
    ON em.run_id = tr.run_id
WHERE em.metric_name = 'accuracy'
  AND em.evaluation_split = 'VALIDATION'
GROUP BY a.architecture_name
ORDER BY best_validation_accuracy DESC NULLS LAST;

-- ============================================================
-- CTE: RESOURCE-CONSTRAINED MODEL SELECTION
-- ============================================================

WITH eligible_models AS (
    SELECT
        a.architecture_id,
        a.architecture_name,
        a.family,
        a.total_parameters,
        a.total_flops
    FROM architectures a
    WHERE a.is_validated
      AND a.total_parameters <= 100000000
      AND a.total_flops <= 100000000000
),
ranked_models AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY family
            ORDER BY total_parameters, total_flops
        ) AS family_rank
    FROM eligible_models
)
SELECT *
FROM ranked_models
WHERE family_rank = 1
ORDER BY family;

-- JSON configuration inspection.
SELECT
    a.architecture_name,
    c.configuration_name,
    c.configuration ->> 'variant' AS variant,
    c.configuration -> 'blocks' AS dense_blocks,
    c.width_multiplier,
    c.depth_multiplier
FROM architecture_configurations c
JOIN architectures a
    ON a.architecture_id = c.architecture_id
ORDER BY a.architecture_name;

-- Failed and cancelled experiment monitoring.
SELECT
    experiment_code,
    status,
    failure_reason,
    created_at
FROM experiments
WHERE status IN ('FAILED', 'CANCELLED')
ORDER BY created_at DESC;

-- Rollback demonstration: a failed registration leaves no partial record.
BEGIN;

SAVEPOINT before_invalid_registration;

DO $$
BEGIN
    BEGIN
        INSERT INTO experiments (
            experiment_code,
            architecture_id,
            dataset_id,
            researcher_id,
            policy_id,
            learning_rate,
            batch_size,
            epochs,
            optimizer,
            random_seed
        )
        SELECT
            'EXP-INVALID',
            a.architecture_id,
            d.dataset_id,
            r.researcher_id,
            p.policy_id,
            -0.01,
            0,
            10,
            'AdamW',
            42
        FROM architectures a
        CROSS JOIN datasets d
        CROSS JOIN researchers r
        CROSS JOIN deployment_policies p
        WHERE a.architecture_name = 'ResNet-18'
          AND d.dataset_name = 'Industrial-Defects'
          AND r.email = 'ananya@example.org'
          AND p.policy_name = 'Server-Standard';

    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE
            'Invalid experiment rejected by database constraints.';
    END;
END;
$$;

ROLLBACK TO SAVEPOINT before_invalid_registration;
COMMIT;
