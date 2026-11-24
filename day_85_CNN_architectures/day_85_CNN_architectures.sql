-- ============================================================
-- CNN ARCHITECTURE RESEARCH AND EXPERIMENT MANAGEMENT
-- PostgreSQL 14+
--
-- Models:
-- LeNet-5
-- AlexNet
-- VGG16
-- Inception v1
--
-- Includes:
-- Architecture catalog
-- Layer definitions
-- Model configurations
-- Training experiments
-- Dataset metadata
-- Evaluation metrics
-- Quality reviews
-- Deployment governance
-- Approval policies
-- Audit events
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS cnn_lab;

SET search_path TO cnn_lab, public;

-- ============================================================
-- ENUMERATIONS
-- ============================================================

CREATE TYPE architecture_family AS ENUM (
    'LENET',
    'ALEXNET',
    'VGG16',
    'INCEPTION'
);

CREATE TYPE layer_type AS ENUM (
    'CONVOLUTION',
    'POOLING',
    'DENSE',
    'ACTIVATION',
    'DROPOUT',
    'NORMALIZATION',
    'BRANCH_CONCATENATION'
);

CREATE TYPE experiment_status AS ENUM (
    'CREATED',
    'VALIDATED',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE review_status AS ENUM (
    'PENDING',
    'PASSED',
    'FAILED'
);

CREATE TYPE deployment_status AS ENUM (
    'DRAFT',
    'VALIDATED',
    'APPROVED',
    'DEPLOYED',
    'REJECTED',
    'RETIRED'
);

CREATE TYPE review_category AS ENUM (
    'ACCURACY',
    'SECURITY',
    'REPRODUCIBILITY',
    'RESOURCE',
    'DATA_GOVERNANCE'
);

-- ============================================================
-- REPOSITORY AND ARCHITECTURE CATALOG
-- ============================================================

CREATE TABLE architectures (
    architecture_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    family architecture_family NOT NULL UNIQUE,

    architecture_name VARCHAR(100) NOT NULL,

    publication_year SMALLINT NOT NULL,

    minimum_input_size SMALLINT NOT NULL,

    expected_channels SMALLINT NOT NULL,

    description TEXT NOT NULL,

    primary_innovation TEXT NOT NULL,

    CONSTRAINT valid_publication_year
        CHECK (publication_year BETWEEN 1990 AND 2100),

    CONSTRAINT valid_input_size
        CHECK (minimum_input_size >= 28),

    CONSTRAINT valid_channels
        CHECK (expected_channels IN (1, 3))
);

INSERT INTO architectures (
    family,
    architecture_name,
    publication_year,
    minimum_input_size,
    expected_channels,
    description,
    primary_innovation
)
VALUES
(
    'LENET',
    'LeNet-5',
    1998,
    32,
    1,
    'Early convolutional architecture for handwritten digit recognition.',
    'Convolution, average pooling and compact fully connected layers.'
),
(
    'ALEXNET',
    'AlexNet',
    2012,
    224,
    3,
    'Deep image classification network trained using GPUs.',
    'ReLU, dropout, data augmentation and large-scale GPU training.'
),
(
    'VGG16',
    'VGG16',
    2014,
    224,
    3,
    'Deep convolutional network using repeated small filters.',
    'Thirteen 3x3 convolutions and five max-pooling stages.'
),
(
    'INCEPTION',
    'Inception v1',
    2014,
    224,
    3,
    'Multi-branch convolutional neural network.',
    'Parallel convolution branches and 1x1 bottleneck operations.'
);

-- ============================================================
-- ARCHITECTURE LAYERS
-- ============================================================

CREATE TABLE architecture_layers (
    layer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    architecture_id BIGINT NOT NULL
        REFERENCES architectures(architecture_id),

    layer_name VARCHAR(100) NOT NULL,

    sequence_number INTEGER NOT NULL,

    layer_kind layer_type NOT NULL,

    input_channels INTEGER,

    output_channels INTEGER,

    kernel_size SMALLINT,

    stride SMALLINT NOT NULL DEFAULT 1,

    padding SMALLINT NOT NULL DEFAULT 0,

    activation VARCHAR(30),

    branch_name VARCHAR(50),

    CONSTRAINT unique_architecture_layer
        UNIQUE (architecture_id, sequence_number),

    CONSTRAINT valid_sequence
        CHECK (sequence_number > 0),

    CONSTRAINT valid_channel_counts
        CHECK (
            (input_channels IS NULL OR input_channels > 0)
            AND
            (output_channels IS NULL OR output_channels > 0)
        ),

    CONSTRAINT valid_kernel
        CHECK (kernel_size IS NULL OR kernel_size > 0),

    CONSTRAINT valid_stride
        CHECK (stride > 0),

    CONSTRAINT valid_padding
        CHECK (padding >= 0)
);

CREATE INDEX idx_layers_architecture
ON architecture_layers(architecture_id, sequence_number);

-- LeNet convolutional layers.

INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    sequence_number,
    layer_kind,
    input_channels,
    output_channels,
    kernel_size,
    activation
)
SELECT
    a.architecture_id,
    v.layer_name,
    v.sequence_number,
    'CONVOLUTION',
    v.input_channels,
    v.output_channels,
    v.kernel_size,
    'tanh'
FROM architectures a
CROSS JOIN (
    VALUES
        ('conv1', 1, 1, 6, 5),
        ('conv2', 2, 6, 16, 5),
        ('conv3', 3, 16, 120, 5)
) AS v(layer_name, sequence_number, input_channels, output_channels, kernel_size)
WHERE a.family = 'LENET';

-- AlexNet convolutional layers.

INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    sequence_number,
    layer_kind,
    input_channels,
    output_channels,
    kernel_size,
    stride,
    padding,
    activation
)
SELECT
    a.architecture_id,
    v.layer_name,
    v.sequence_number,
    'CONVOLUTION',
    v.input_channels,
    v.output_channels,
    v.kernel_size,
    v.stride,
    v.padding,
    'relu'
FROM architectures a
CROSS JOIN (
    VALUES
        ('conv1', 1, 3, 64, 11, 4, 2),
        ('conv2', 2, 64, 192, 5, 1, 2),
        ('conv3', 3, 192, 384, 3, 1, 1),
        ('conv4', 4, 384, 256, 3, 1, 1),
        ('conv5', 5, 256, 256, 3, 1, 1)
) AS v(layer_name, sequence_number, input_channels, output_channels,
       kernel_size, stride, padding)
WHERE a.family = 'ALEXNET';

-- VGG16: thirteen convolutional layers.

INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    sequence_number,
    layer_kind,
    input_channels,
    output_channels,
    kernel_size,
    padding,
    activation
)
SELECT
    a.architecture_id,
    v.layer_name,
    v.sequence_number,
    'CONVOLUTION',
    v.input_channels,
    v.output_channels,
    3,
    1,
    'relu'
FROM architectures a
CROSS JOIN (
    VALUES
        ('conv1_1', 1, 3, 64),
        ('conv1_2', 2, 64, 64),
        ('conv2_1', 3, 64, 128),
        ('conv2_2', 4, 128, 128),
        ('conv3_1', 5, 128, 256),
        ('conv3_2', 6, 256, 256),
        ('conv3_3', 7, 256, 256),
        ('conv4_1', 8, 256, 512),
        ('conv4_2', 9, 512, 512),
        ('conv4_3', 10, 512, 512),
        ('conv5_1', 11, 512, 512),
        ('conv5_2', 12, 512, 512),
        ('conv5_3', 13, 512, 512)
) AS v(layer_name, sequence_number, input_channels, output_channels)
WHERE a.family = 'VGG16';

-- Inception v1: first representative multi-branch module.

INSERT INTO architecture_layers (
    architecture_id,
    layer_name,
    sequence_number,
    layer_kind,
    input_channels,
    output_channels,
    kernel_size,
    padding,
    activation,
    branch_name
)
SELECT
    a.architecture_id,
    v.layer_name,
    v.sequence_number,
    'CONVOLUTION',
    v.input_channels,
    v.output_channels,
    v.kernel_size,
    v.padding,
    'relu',
    v.branch_name
FROM architectures a
CROSS JOIN (
    VALUES
        ('branch1_1x1', 1, 192, 64, 1, 0, '1x1'),
        ('branch2_reduce', 2, 192, 96, 1, 0, '3x3'),
        ('branch2_3x3', 3, 96, 128, 3, 1, '3x3'),
        ('branch3_reduce', 4, 192, 16, 1, 0, '5x5'),
        ('branch3_5x5', 5, 16, 32, 5, 2, '5x5'),
        ('branch4_projection', 6, 192, 32, 1, 0, 'pool')
) AS v(layer_name, sequence_number, input_channels, output_channels,
       kernel_size, padding, branch_name)
WHERE a.family = 'INCEPTION';

-- ============================================================
-- DATASET CATALOG
-- ============================================================

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_name VARCHAR(150) NOT NULL UNIQUE,

    dataset_version VARCHAR(50) NOT NULL,

    image_count BIGINT NOT NULL,

    class_count INTEGER NOT NULL,

    image_height INTEGER NOT NULL,

    image_width INTEGER NOT NULL,

    channels SMALLINT NOT NULL,

    license_name VARCHAR(150) NOT NULL,

    checksum VARCHAR(128),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_dataset_size CHECK (image_count > 0),

    CONSTRAINT valid_dataset_classes CHECK (class_count >= 2),

    CONSTRAINT valid_dataset_dimensions CHECK (
        image_height > 0 AND image_width > 0
    ),

    CONSTRAINT valid_dataset_channels CHECK (channels IN (1, 3))
);

INSERT INTO datasets (
    dataset_name,
    dataset_version,
    image_count,
    class_count,
    image_height,
    image_width,
    channels,
    license_name
)
VALUES
(
    'MNIST',
    'Original',
    70000,
    10,
    28,
    28,
    1,
    'CC BY-SA 3.0'
),
(
    'CIFAR-10',
    '1.0',
    60000,
    10,
    32,
    32,
    3,
    'CIFAR-10 Dataset Terms'
),
(
    'ImageNet',
    'ILSVRC-2012',
    1281167,
    1000,
    224,
    224,
    3,
    'ImageNet access terms'
);

-- ============================================================
-- MODEL CONFIGURATIONS
-- ============================================================

CREATE TABLE model_configurations (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    architecture_id BIGINT NOT NULL
        REFERENCES architectures(architecture_id),

    model_name VARCHAR(150) NOT NULL,

    input_channels SMALLINT NOT NULL,

    input_height INTEGER NOT NULL,

    input_width INTEGER NOT NULL,

    class_count INTEGER NOT NULL,

    optimizer VARCHAR(50) NOT NULL,

    learning_rate NUMERIC(12, 8) NOT NULL,

    batch_size INTEGER NOT NULL,

    weight_decay NUMERIC(12, 8) NOT NULL DEFAULT 0,

    random_seed INTEGER NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_model_name UNIQUE (model_name),

    CONSTRAINT valid_model_dimensions CHECK (
        input_channels IN (1, 3)
        AND input_height > 0
        AND input_width > 0
    ),

    CONSTRAINT valid_model_classes CHECK (class_count >= 2),

    CONSTRAINT valid_learning_rate CHECK (
        learning_rate > 0 AND learning_rate <= 1
    ),

    CONSTRAINT valid_batch_size CHECK (
        batch_size BETWEEN 1 AND 4096
    ),

    CONSTRAINT valid_weight_decay CHECK (weight_decay >= 0)
);

INSERT INTO model_configurations (
    architecture_id,
    model_name,
    input_channels,
    input_height,
    input_width,
    class_count,
    optimizer,
    learning_rate,
    batch_size,
    weight_decay,
    random_seed
)
SELECT
    a.architecture_id,
    v.model_name,
    v.channels,
    v.image_size,
    v.image_size,
    v.classes,
    v.optimizer,
    v.learning_rate,
    v.batch_size,
    v.weight_decay,
    42
FROM (
    VALUES
        ('LENET', 'LeNet-MNIST', 1, 32, 10, 'SGD', 0.01, 64, 0.0001),
        ('ALEXNET', 'AlexNet-ImageNet', 3, 224, 1000, 'SGD', 0.001, 32, 0.0005),
        ('VGG16', 'VGG16-ImageNet', 3, 224, 1000, 'SGD', 0.0001, 16, 0.0005),
        ('INCEPTION', 'Inception-ImageNet', 3, 224, 1000, 'RMSPROP', 0.001, 32, 0.0001)
) AS v(family, model_name, channels, image_size, classes,
       optimizer, learning_rate, batch_size, weight_decay)
JOIN architectures a ON a.family::TEXT = v.family;

-- ============================================================
-- EXPERIMENT MANAGEMENT
-- ============================================================

CREATE TABLE experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_configurations(model_id),

    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id),

    experiment_name VARCHAR(150) NOT NULL UNIQUE,

    status experiment_status NOT NULL DEFAULT 'CREATED',

    epoch_limit INTEGER NOT NULL,

    completed_epochs INTEGER NOT NULL DEFAULT 0,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    failure_reason TEXT,

    CONSTRAINT valid_epoch_limit CHECK (epoch_limit > 0),

    CONSTRAINT valid_completed_epochs CHECK (
        completed_epochs >= 0
        AND completed_epochs <= epoch_limit
    ),

    CONSTRAINT valid_experiment_timestamps CHECK (
        completed_at IS NULL
        OR started_at IS NULL
        OR completed_at >= started_at
    )
);

CREATE INDEX idx_experiments_status
ON experiments(status, created_at);

CREATE INDEX idx_experiments_model
ON experiments(model_id, experiment_id);

-- ============================================================
-- EXPERIMENT METRICS
-- ============================================================

CREATE TABLE experiment_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,

    epoch_number INTEGER NOT NULL,

    training_loss NUMERIC(14, 8),

    validation_loss NUMERIC(14, 8),

    accuracy NUMERIC(8, 6),

    precision_score NUMERIC(8, 6),

    recall_score NUMERIC(8, 6),

    f1_score NUMERIC(8, 6),

    inference_latency_ms NUMERIC(12, 4),

    peak_memory_mb NUMERIC(14, 2),

    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_experiment_epoch
        UNIQUE (experiment_id, epoch_number),

    CONSTRAINT valid_metric_epoch CHECK (epoch_number > 0),

    CONSTRAINT valid_accuracy CHECK (
        accuracy IS NULL OR accuracy BETWEEN 0 AND 1
    ),

    CONSTRAINT valid_precision CHECK (
        precision_score IS NULL OR precision_score BETWEEN 0 AND 1
    ),

    CONSTRAINT valid_recall CHECK (
        recall_score IS NULL OR recall_score BETWEEN 0 AND 1
    ),

    CONSTRAINT valid_f1 CHECK (
        f1_score IS NULL OR f1_score BETWEEN 0 AND 1
    ),

    CONSTRAINT valid_losses CHECK (
        (training_loss IS NULL OR training_loss >= 0)
        AND
        (validation_loss IS NULL OR validation_loss >= 0)
    ),

    CONSTRAINT valid_latency CHECK (
        inference_latency_ms IS NULL OR inference_latency_ms >= 0
    )
);

CREATE INDEX idx_metrics_best_accuracy
ON experiment_metrics(experiment_id, accuracy DESC);

-- ============================================================
-- REVIEWERS AND QUALITY GOVERNANCE
-- ============================================================

CREATE TABLE reviewers (
    reviewer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    reviewer_name VARCHAR(120) NOT NULL,

    organization_email VARCHAR(254) NOT NULL UNIQUE,

    role_name VARCHAR(80) NOT NULL,

    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO reviewers (
    reviewer_name,
    organization_email,
    role_name
)
VALUES
(
    'Model Evaluation Reviewer',
    'evaluation@example.org',
    'MODEL_EVALUATOR'
),
(
    'Security Reviewer',
    'security@example.org',
    'SECURITY_ENGINEER'
),
(
    'Research Governance Reviewer',
    'governance@example.org',
    'RESEARCH_GOVERNANCE'
),
(
    'Infrastructure Reviewer',
    'infrastructure@example.org',
    'INFRASTRUCTURE_ENGINEER'
);

CREATE TABLE model_reviews (
    review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_configurations(model_id),

    reviewer_id BIGINT NOT NULL
        REFERENCES reviewers(reviewer_id),

    category review_category NOT NULL,

    status review_status NOT NULL DEFAULT 'PENDING',

    comments TEXT,

    reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_model_review
        UNIQUE (model_id, reviewer_id, category)
);

CREATE INDEX idx_reviews_model_status
ON model_reviews(model_id, status, category);

-- ============================================================
-- DEPLOYMENT GOVERNANCE
-- ============================================================

CREATE TABLE deployment_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    policy_name VARCHAR(120) NOT NULL UNIQUE,

    required_accuracy NUMERIC(8, 6) NOT NULL,

    maximum_latency_ms NUMERIC(12, 4) NOT NULL,

    maximum_memory_mb NUMERIC(14, 2) NOT NULL,

    require_security_review BOOLEAN NOT NULL DEFAULT TRUE,

    require_reproducibility BOOLEAN NOT NULL DEFAULT TRUE,

    minimum_completed_epochs INTEGER NOT NULL DEFAULT 1,

    active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT valid_policy_accuracy CHECK (
        required_accuracy BETWEEN 0 AND 1
    ),

    CONSTRAINT valid_policy_latency CHECK (maximum_latency_ms > 0),

    CONSTRAINT valid_policy_memory CHECK (maximum_memory_mb > 0),

    CONSTRAINT valid_policy_epochs CHECK (
        minimum_completed_epochs > 0
    )
);

INSERT INTO deployment_policies (
    policy_name,
    required_accuracy,
    maximum_latency_ms,
    maximum_memory_mb,
    minimum_completed_epochs
)
VALUES
(
    'STANDARD_CLASSIFIER',
    0.850000,
    100.0,
    8192.0,
    10
),
(
    'EDGE_CLASSIFIER',
    0.800000,
    20.0,
    1024.0,
    10
);

CREATE TABLE deployments (
    deployment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_configurations(model_id),

    experiment_id BIGINT NOT NULL
        REFERENCES experiments(experiment_id),

    policy_id BIGINT NOT NULL
        REFERENCES deployment_policies(policy_id),

    status deployment_status NOT NULL DEFAULT 'DRAFT',

    deployment_environment VARCHAR(80) NOT NULL,

    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    approved_at TIMESTAMPTZ,

    deployed_at TIMESTAMPTZ,

    CONSTRAINT unique_model_deployment
        UNIQUE (model_id, experiment_id, deployment_environment)
);

CREATE INDEX idx_deployments_status
ON deployments(status, requested_at);

-- ============================================================
-- AUDIT LOG
-- ============================================================

CREATE TABLE audit_events (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    entity_type VARCHAR(80) NOT NULL,

    entity_id BIGINT NOT NULL,

    action VARCHAR(80) NOT NULL,

    old_state TEXT,

    new_state TEXT,

    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    database_user TEXT NOT NULL DEFAULT CURRENT_USER
);

CREATE INDEX idx_audit_entity
ON audit_events(entity_type, entity_id, event_timestamp DESC);

-- ============================================================
-- AUDIT TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION record_experiment_state_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO audit_events (
            entity_type,
            entity_id,
            action,
            old_state,
            new_state
        )
        VALUES (
            'EXPERIMENT',
            NEW.experiment_id,
            'STATUS_CHANGE',
            OLD.status::TEXT,
            NEW.status::TEXT
        );
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_experiment_state_audit
AFTER UPDATE ON experiments
FOR EACH ROW
EXECUTE FUNCTION record_experiment_state_change();

-- ============================================================
-- VALIDATION FUNCTIONS
-- ============================================================

CREATE OR REPLACE FUNCTION validate_model_configuration(
    p_model_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_valid BOOLEAN;
BEGIN
    SELECT
        m.input_channels = a.expected_channels
        AND m.input_height >= a.minimum_input_size
        AND m.input_width >= a.minimum_input_size
        AND m.class_count >= 2
    INTO v_valid
    FROM model_configurations m
    JOIN architectures a
      ON a.architecture_id = m.architecture_id
    WHERE m.model_id = p_model_id;

    RETURN COALESCE(v_valid, FALSE);
END;
$$;

CREATE OR REPLACE FUNCTION calculate_layer_parameters(
    p_input_channels INTEGER,
    p_output_channels INTEGER,
    p_kernel INTEGER
)
RETURNS BIGINT
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    IF p_input_channels <= 0
       OR p_output_channels <= 0
       OR p_kernel <= 0 THEN
        RAISE EXCEPTION 'Invalid convolution dimensions';
    END IF;

    RETURN
        p_input_channels::BIGINT *
        p_output_channels::BIGINT *
        p_kernel::BIGINT *
        p_kernel::BIGINT +
        p_output_channels;
END;
$$;

-- ============================================================
-- ELIGIBILITY FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION check_deployment_eligibility(
    p_deployment_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    reason TEXT,
    measured_accuracy NUMERIC,
    measured_latency NUMERIC,
    measured_memory NUMERIC
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_deployment deployments%ROWTYPE;
    v_policy deployment_policies%ROWTYPE;

    v_accuracy NUMERIC;
    v_latency NUMERIC;
    v_memory NUMERIC;

    v_completed_epochs INTEGER;

    v_security BOOLEAN;
    v_reproducibility BOOLEAN;
    v_accuracy_review BOOLEAN;
BEGIN
    SELECT *
    INTO v_deployment
    FROM deployments
    WHERE deployment_id = p_deployment_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Deployment does not exist',
               NULL::NUMERIC, NULL::NUMERIC, NULL::NUMERIC;
        RETURN;
    END IF;

    SELECT *
    INTO v_policy
    FROM deployment_policies
    WHERE policy_id = v_deployment.policy_id
      AND active = TRUE;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Deployment policy is inactive',
               NULL::NUMERIC, NULL::NUMERIC, NULL::NUMERIC;
        RETURN;
    END IF;

    SELECT
        MAX(em.accuracy),
        MIN(em.inference_latency_ms),
        MIN(em.peak_memory_mb),
        e.completed_epochs
    INTO
        v_accuracy,
        v_latency,
        v_memory,
        v_completed_epochs
    FROM experiments e
    LEFT JOIN experiment_metrics em
      ON em.experiment_id = e.experiment_id
    WHERE e.experiment_id = v_deployment.experiment_id
      AND e.model_id = v_deployment.model_id
      AND e.status = 'COMPLETED'
    GROUP BY e.completed_epochs;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Experiment is not completed',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    SELECT
        COALESCE(BOOL_OR(
            category = 'SECURITY' AND status = 'PASSED'
        ), FALSE),
        COALESCE(BOOL_OR(
            category = 'REPRODUCIBILITY' AND status = 'PASSED'
        ), FALSE),
        COALESCE(BOOL_OR(
            category = 'ACCURACY' AND status = 'PASSED'
        ), FALSE)
    INTO
        v_security,
        v_reproducibility,
        v_accuracy_review
    FROM model_reviews
    WHERE model_id = v_deployment.model_id;

    IF v_completed_epochs < v_policy.minimum_completed_epochs THEN
        RETURN QUERY
        SELECT FALSE, 'Insufficient completed epochs',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    IF v_accuracy IS NULL OR v_accuracy < v_policy.required_accuracy THEN
        RETURN QUERY
        SELECT FALSE, 'Accuracy threshold not satisfied',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    IF v_latency IS NULL OR v_latency > v_policy.maximum_latency_ms THEN
        RETURN QUERY
        SELECT FALSE, 'Inference latency threshold not satisfied',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    IF v_memory IS NULL OR v_memory > v_policy.maximum_memory_mb THEN
        RETURN QUERY
        SELECT FALSE, 'Memory threshold not satisfied',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    IF v_policy.require_security_review AND NOT v_security THEN
        RETURN QUERY
        SELECT FALSE, 'Security review missing or failed',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    IF v_policy.require_reproducibility AND NOT v_reproducibility THEN
        RETURN QUERY
        SELECT FALSE, 'Reproducibility review missing or failed',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    IF NOT v_accuracy_review THEN
        RETURN QUERY
        SELECT FALSE, 'Accuracy review missing or failed',
               v_accuracy, v_latency, v_memory;
        RETURN;
    END IF;

    RETURN QUERY
    SELECT TRUE, 'All deployment requirements satisfied',
           v_accuracy, v_latency, v_memory;
END;
$$;

-- ============================================================
-- CONTROLLED DEPLOYMENT PROCEDURE
-- ============================================================

CREATE OR REPLACE PROCEDURE approve_deployment(
    p_deployment_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_eligible BOOLEAN;
    v_reason TEXT;
    v_status deployment_status;
BEGIN
    SELECT status
    INTO v_status
    FROM deployments
    WHERE deployment_id = p_deployment_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Deployment % not found', p_deployment_id;
    END IF;

    IF v_status NOT IN ('DRAFT', 'VALIDATED') THEN
        RAISE EXCEPTION 'Deployment cannot be approved from state %',
            v_status;
    END IF;

    SELECT e.eligible, e.reason
    INTO v_eligible, v_reason
    FROM check_deployment_eligibility(p_deployment_id) e;

    IF NOT COALESCE(v_eligible, FALSE) THEN
        RAISE EXCEPTION 'Deployment rejected: %', v_reason;
    END IF;

    UPDATE deployments
    SET status = 'APPROVED',
        approved_at = NOW()
    WHERE deployment_id = p_deployment_id;

    INSERT INTO audit_events (
        entity_type,
        entity_id,
        action,
        old_state,
        new_state
    )
    VALUES (
        'DEPLOYMENT',
        p_deployment_id,
        'APPROVAL',
        v_status::TEXT,
        'APPROVED'
    );
END;
$$;

-- ============================================================
-- REPORTING VIEWS
-- ============================================================

CREATE VIEW architecture_catalog_report AS
SELECT
    a.architecture_name,
    a.family,
    a.publication_year,
    a.expected_channels,
    a.minimum_input_size,
    COUNT(l.layer_id) FILTER (
        WHERE l.layer_kind = 'CONVOLUTION'
    ) AS convolution_layers,
    COUNT(DISTINCT l.branch_name) FILTER (
        WHERE l.branch_name IS NOT NULL
    ) AS parallel_branches
FROM architectures a
LEFT JOIN architecture_layers l
    ON l.architecture_id = a.architecture_id
GROUP BY
    a.architecture_id,
    a.architecture_name,
    a.family,
    a.publication_year,
    a.expected_channels,
    a.minimum_input_size;

CREATE VIEW experiment_performance_report AS
SELECT
    e.experiment_id,
    e.experiment_name,
    a.architecture_name,
    d.dataset_name,
    e.status,
    e.completed_epochs,
    MAX(m.accuracy) AS best_accuracy,
    MIN(m.validation_loss) AS best_validation_loss,
    MIN(m.inference_latency_ms) AS minimum_latency_ms,
    MIN(m.peak_memory_mb) AS minimum_memory_mb
FROM experiments e
JOIN model_configurations mc
    ON mc.model_id = e.model_id
JOIN architectures a
    ON a.architecture_id = mc.architecture_id
JOIN datasets d
    ON d.dataset_id = e.dataset_id
LEFT JOIN experiment_metrics m
    ON m.experiment_id = e.experiment_id
GROUP BY
    e.experiment_id,
    e.experiment_name,
    a.architecture_name,
    d.dataset_name,
    e.status,
    e.completed_epochs;

-- ============================================================
-- SAMPLE EXPERIMENTS
-- ============================================================

INSERT INTO experiments (
    model_id,
    dataset_id,
    experiment_name,
    status,
    epoch_limit,
    completed_epochs,
    started_at,
    completed_at
)
SELECT
    mc.model_id,
    d.dataset_id,
    v.experiment_name,
    v.status::experiment_status,
    v.epoch_limit,
    v.completed_epochs,
    NOW() - INTERVAL '2 days',
    NOW()
FROM (
    VALUES
        ('LeNet-MNIST', 'MNIST', 'LeNet baseline experiment',
         'COMPLETED', 20, 20),
        ('AlexNet-ImageNet', 'ImageNet', 'AlexNet baseline experiment',
         'COMPLETED', 30, 30),
        ('VGG16-ImageNet', 'ImageNet', 'VGG16 baseline experiment',
         'COMPLETED', 30, 30),
        ('Inception-ImageNet', 'ImageNet', 'Inception baseline experiment',
         'COMPLETED', 30, 30)
) AS v(model_name, dataset_name, experiment_name,
       status, epoch_limit, completed_epochs)
JOIN model_configurations mc
    ON mc.model_name = v.model_name
JOIN datasets d
    ON d.dataset_name = v.dataset_name;

-- ============================================================
-- SAMPLE METRICS
-- ============================================================

INSERT INTO experiment_metrics (
    experiment_id,
    epoch_number,
    training_loss,
    validation_loss,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    inference_latency_ms,
    peak_memory_mb
)
SELECT
    e.experiment_id,
    v.epoch_number,
    v.training_loss,
    v.validation_loss,
    v.accuracy,
    v.precision_score,
    v.recall_score,
    v.f1_score,
    v.latency,
    v.memory
FROM (
    VALUES
        ('LeNet baseline experiment', 20, 0.080, 0.095, 0.985, 0.984, 0.985, 0.984, 4.0, 256.0),
        ('AlexNet baseline experiment', 30, 0.320, 0.350, 0.870, 0.868, 0.870, 0.869, 45.0, 4096.0),
        ('VGG16 baseline experiment', 30, 0.270, 0.310, 0.910, 0.908, 0.910, 0.909, 90.0, 7168.0),
        ('Inception baseline experiment', 30, 0.240, 0.280, 0.920, 0.918, 0.920, 0.919, 35.0, 3072.0)
) AS v(experiment_name, epoch_number, training_loss, validation_loss,
       accuracy, precision_score, recall_score, f1_score, latency, memory)
JOIN experiments e
    ON e.experiment_name = v.experiment_name;

-- ============================================================
-- QUALITY REVIEW RECORDS
-- ============================================================

INSERT INTO model_reviews (
    model_id,
    reviewer_id,
    category,
    status,
    comments,
    reviewed_at
)
SELECT
    mc.model_id,
    r.reviewer_id,
    v.category::review_category,
    v.status::review_status,
    v.comments,
    NOW()
FROM (
    VALUES
        ('Inception-ImageNet', 'SECURITY_ENGINEER',
         'SECURITY', 'PASSED', 'Security controls verified.'),
        ('Inception-ImageNet', 'RESEARCH_GOVERNANCE',
         'REPRODUCIBILITY', 'PASSED', 'Training configuration recorded.'),
        ('Inception-ImageNet', 'MODEL_EVALUATOR',
         'ACCURACY', 'PASSED', 'Evaluation threshold satisfied.')
) AS v(model_name, role_name, category, status, comments)
JOIN model_configurations mc
    ON mc.model_name = v.model_name
JOIN reviewers r
    ON r.role_name = v.role_name;

-- ============================================================
-- SAMPLE DEPLOYMENT REQUESTS
-- ============================================================

INSERT INTO deployments (
    model_id,
    experiment_id,
    policy_id,
    status,
    deployment_environment
)
SELECT
    mc.model_id,
    e.experiment_id,
    p.policy_id,
    'DRAFT',
    'STAGING'
FROM model_configurations mc
JOIN experiments e
    ON e.model_id = mc.model_id
JOIN deployment_policies p
    ON p.policy_name = 'STANDARD_CLASSIFIER'
WHERE mc.model_name = 'Inception-ImageNet';

-- ============================================================
-- RETRIEVAL AND VALIDATION QUERIES
-- ============================================================

-- 1. Architecture catalog.

SELECT *
FROM architecture_catalog_report
ORDER BY publication_year;

-- 2. Ordered layer definitions.

SELECT
    a.architecture_name,
    l.sequence_number,
    l.layer_name,
    l.layer_kind,
    l.input_channels,
    l.output_channels,
    l.kernel_size,
    l.branch_name
FROM architecture_layers l
JOIN architectures a
    ON a.architecture_id = l.architecture_id
ORDER BY a.architecture_name, l.sequence_number;

-- 3. Validate model input contracts.

SELECT
    model_name,
    validate_model_configuration(model_id) AS configuration_valid
FROM model_configurations;

-- 4. Calculate convolution parameter counts.

SELECT
    a.architecture_name,
    SUM(
        calculate_layer_parameters(
            l.input_channels,
            l.output_channels,
            l.kernel_size
        )
    ) AS convolution_parameters
FROM architecture_layers l
JOIN architectures a
    ON a.architecture_id = l.architecture_id
WHERE l.layer_kind = 'CONVOLUTION'
GROUP BY a.architecture_name
ORDER BY convolution_parameters;

-- 5. Experiment performance report.

SELECT *
FROM experiment_performance_report
ORDER BY best_accuracy DESC NULLS LAST;

-- 6. Architecture-level performance aggregation.

SELECT
    architecture_name,
    COUNT(*) AS experiments,
    ROUND(AVG(best_accuracy), 4) AS average_accuracy,
    ROUND(AVG(minimum_latency_ms), 2) AS average_latency_ms
FROM experiment_performance_report
GROUP BY architecture_name
ORDER BY average_accuracy DESC;

-- 7. Review status.

SELECT
    mc.model_name,
    mr.category,
    mr.status,
    r.reviewer_name
FROM model_reviews mr
JOIN model_configurations mc
    ON mc.model_id = mr.model_id
JOIN reviewers r
    ON r.reviewer_id = mr.reviewer_id
ORDER BY mc.model_name, mr.category;

-- 8. Deployment eligibility.

SELECT
    d.deployment_id,
    d.deployment_environment,
    e.eligible,
    e.reason,
    e.measured_accuracy,
    e.measured_latency,
    e.measured_memory
FROM deployments d
CROSS JOIN LATERAL check_deployment_eligibility(
    d.deployment_id
) e;

-- 9. CTE: rank models using multiple evaluation metrics.

WITH model_scores AS (
    SELECT
        mc.model_name,
        a.architecture_name,
        MAX(em.accuracy) AS accuracy,
        MIN(em.inference_latency_ms) AS latency,
        MIN(em.peak_memory_mb) AS memory
    FROM model_configurations mc
    JOIN architectures a
        ON a.architecture_id = mc.architecture_id
    JOIN experiments ex
        ON ex.model_id = mc.model_id
    JOIN experiment_metrics em
        ON em.experiment_id = ex.experiment_id
    WHERE ex.status = 'COMPLETED'
    GROUP BY mc.model_name, a.architecture_name
),
ranked_models AS (
    SELECT
        *,
        RANK() OVER (
            ORDER BY accuracy DESC, latency ASC
        ) AS performance_rank
    FROM model_scores
)
SELECT *
FROM ranked_models
ORDER BY performance_rank;

-- 10. Incomplete review identification.

SELECT
    mc.model_name,
    required.category
FROM model_configurations mc
CROSS JOIN (
    VALUES
        ('ACCURACY'::review_category),
        ('SECURITY'::review_category),
        ('REPRODUCIBILITY'::review_category)
) AS required(category)
LEFT JOIN model_reviews mr
    ON mr.model_id = mc.model_id
   AND mr.category = required.category
   AND mr.status = 'PASSED'
WHERE mr.review_id IS NULL
ORDER BY mc.model_name, required.category;

-- ============================================================
-- TRANSACTIONAL DEPLOYMENT APPROVAL
-- ============================================================

-- The procedure locks the deployment row and verifies all
-- requirements before changing its state.
--
-- Uncomment to approve the sample eligible Inception deployment:
--
-- BEGIN;
--
-- SELECT deployment_id
-- FROM deployments
-- WHERE deployment_environment = 'STAGING';
--
-- CALL approve_deployment(1);
--
-- COMMIT;
--
-- A failed eligibility check raises an exception. The
-- transaction must then be rolled back by the caller.

-- ============================================================
-- DATA INTEGRITY EXAMPLES
-- ============================================================

-- These statements intentionally demonstrate invalid data.
-- They are kept commented so the complete script executes.

-- Invalid class count:
-- INSERT INTO model_configurations (...)
-- VALUES (..., 1, ...);

-- Invalid learning rate:
-- INSERT INTO model_configurations (...)
-- VALUES (..., -0.01, ...);

-- Duplicate epoch:
-- INSERT INTO experiment_metrics (
--     experiment_id, epoch_number, accuracy
-- ) VALUES (1, 20, 0.90);

-- Invalid accuracy:
-- INSERT INTO experiment_metrics (
--     experiment_id, epoch_number, accuracy
-- ) VALUES (1, 21, 1.50);

-- ============================================================
-- FINAL DATABASE STATE
-- ============================================================

COMMIT;

SELECT
    architecture_name,
    publication_year,
    convolution_layers,
    parallel_branches
FROM architecture_catalog_report
ORDER BY publication_year;

SELECT
    experiment_name,
    architecture_name,
    best_accuracy,
    minimum_latency_ms,
    status
FROM experiment_performance_report
ORDER BY best_accuracy DESC NULLS LAST;
