/*
Production-style classical machine learning pipeline database.

PostgreSQL 14+

Domain:
Customer churn classification and model lifecycle governance.

Capabilities:
- Dataset and dataset-version management
- Feature contracts
- Training runs
- Model registry
- Evaluation metrics
- Approval records
- Quality gates
- Deployment management
- Prediction logging
- Feature drift monitoring
- Audit history
- Transactional model promotion
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS ml_platform;

SET search_path TO ml_platform, public;

-- -----------------------------------------------------
-- Enumerated domain states
-- -----------------------------------------------------

CREATE TYPE dataset_status AS ENUM (
    'REGISTERED',
    'VALIDATED',
    'REJECTED',
    'ARCHIVED'
);

CREATE TYPE run_status AS ENUM (
    'CREATED',
    'VALIDATING',
    'TRAINING',
    'TRAINED',
    'EVALUATING',
    'APPROVED',
    'REJECTED',
    'FAILED'
);

CREATE TYPE model_status AS ENUM (
    'REGISTERED',
    'VALIDATING',
    'APPROVED',
    'REJECTED',
    'DEPLOYED',
    'RETIRED'
);

CREATE TYPE review_decision AS ENUM (
    'APPROVE',
    'REQUEST_CHANGES',
    'REJECT'
);

CREATE TYPE deployment_environment AS ENUM (
    'DEVELOPMENT',
    'STAGING',
    'PRODUCTION'
);

CREATE TYPE check_status AS ENUM (
    'PENDING',
    'PASSED',
    'FAILED',
    'WARNING'
);

-- -----------------------------------------------------
-- Repository-like project organization
-- -----------------------------------------------------

CREATE TABLE ml_project (
    project_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    project_name VARCHAR(150) NOT NULL UNIQUE,

    description TEXT,

    owner VARCHAR(100) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE project_member (
    member_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    project_id BIGINT NOT NULL
        REFERENCES ml_project(project_id),

    username VARCHAR(100) NOT NULL,

    role VARCHAR(30) NOT NULL CHECK (
        role IN (
            'DATA_ENGINEER',
            'ML_ENGINEER',
            'RESEARCHER',
            'REVIEWER',
            'ADMIN',
            'OPERATOR'
        )
    ),

    active BOOLEAN NOT NULL DEFAULT TRUE,

    UNIQUE(project_id, username)
);

-- -----------------------------------------------------
-- Dataset registry
-- -----------------------------------------------------

CREATE TABLE dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    project_id BIGINT NOT NULL
        REFERENCES ml_project(project_id),

    dataset_name VARCHAR(150) NOT NULL,

    description TEXT,

    source_type VARCHAR(40) NOT NULL CHECK (
        source_type IN (
            'CSV',
            'DATABASE',
            'OBJECT_STORAGE',
            'API'
        )
    ),

    owner VARCHAR(100) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(project_id, dataset_name)
);

CREATE TABLE dataset_version (
    dataset_version_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id),

    version VARCHAR(40) NOT NULL,

    fingerprint CHAR(64) NOT NULL,

    row_count BIGINT NOT NULL CHECK (row_count >= 0),

    column_count INTEGER NOT NULL CHECK (column_count > 0),

    target_column VARCHAR(100) NOT NULL,

    status dataset_status NOT NULL DEFAULT 'REGISTERED',

    class_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,

    validation_report JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(dataset_id, version),

    UNIQUE(dataset_id, fingerprint),

    CHECK (jsonb_typeof(class_distribution) = 'object'),

    CHECK (jsonb_typeof(validation_report) = 'object')
);

CREATE INDEX idx_dataset_version_status
    ON dataset_version(status, created_at DESC);

-- -----------------------------------------------------
-- Feature contracts
-- -----------------------------------------------------

CREATE TABLE feature_definition (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    dataset_id BIGINT NOT NULL
        REFERENCES dataset(dataset_id),

    feature_name VARCHAR(100) NOT NULL,

    data_type VARCHAR(30) NOT NULL CHECK (
        data_type IN (
            'INTEGER',
            'NUMERIC',
            'BOOLEAN',
            'TEXT',
            'CATEGORY',
            'TIMESTAMP'
        )
    ),

    is_required BOOLEAN NOT NULL DEFAULT TRUE,

    nullable BOOLEAN NOT NULL DEFAULT TRUE,

    sensitive BOOLEAN NOT NULL DEFAULT FALSE,

    allowed_values JSONB,

    UNIQUE(dataset_id, feature_name)
);

-- -----------------------------------------------------
-- Training runs
-- -----------------------------------------------------

CREATE TABLE training_run (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    project_id BIGINT NOT NULL
        REFERENCES ml_project(project_id),

    dataset_version_id BIGINT NOT NULL
        REFERENCES dataset_version(dataset_version_id),

    run_name VARCHAR(150) NOT NULL,

    algorithm VARCHAR(100) NOT NULL,

    target_column VARCHAR(100) NOT NULL,

    random_seed INTEGER NOT NULL,

    test_size NUMERIC(5,4) NOT NULL CHECK (
        test_size > 0 AND test_size < 1
    ),

    validation_size NUMERIC(5,4) NOT NULL CHECK (
        validation_size > 0 AND validation_size < 1
    ),

    status run_status NOT NULL DEFAULT 'CREATED',

    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,

    training_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    error_message TEXT,

    created_by VARCHAR(100) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        test_size + validation_size < 1
    ),

    CHECK (
        completed_at IS NULL OR
        started_at IS NULL OR
        completed_at >= started_at
    )
);

CREATE INDEX idx_training_run_project
    ON training_run(project_id, created_at DESC);

CREATE INDEX idx_training_run_status
    ON training_run(status, created_at DESC);

CREATE INDEX idx_training_run_dataset
    ON training_run(dataset_version_id);

-- -----------------------------------------------------
-- Model artifacts
-- -----------------------------------------------------

CREATE TABLE model_artifact (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    run_id UUID NOT NULL UNIQUE
        REFERENCES training_run(run_id),

    model_name VARCHAR(150) NOT NULL,

    version VARCHAR(40) NOT NULL,

    algorithm VARCHAR(100) NOT NULL,

    artifact_uri TEXT NOT NULL,

    artifact_checksum CHAR(64) NOT NULL,

    framework VARCHAR(60) NOT NULL,

    framework_version VARCHAR(40) NOT NULL,

    feature_schema JSONB NOT NULL,

    preprocessing_schema JSONB NOT NULL,

    model_status model_status NOT NULL DEFAULT 'REGISTERED',

    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(model_name, version),

    CHECK (jsonb_typeof(feature_schema) = 'object'),

    CHECK (jsonb_typeof(preprocessing_schema) = 'object')
);

CREATE INDEX idx_model_status
    ON model_artifact(model_status, registered_at DESC);

-- -----------------------------------------------------
-- Evaluation metrics
-- -----------------------------------------------------

CREATE TABLE model_evaluation (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_artifact(model_id),

    evaluation_dataset_version_id BIGINT NOT NULL
        REFERENCES dataset_version(dataset_version_id),

    accuracy NUMERIC(8,6) CHECK (
        accuracy BETWEEN 0 AND 1
    ),

    precision_score NUMERIC(8,6) CHECK (
        precision_score BETWEEN 0 AND 1
    ),

    recall_score NUMERIC(8,6) CHECK (
        recall_score BETWEEN 0 AND 1
    ),

    f1_score NUMERIC(8,6) CHECK (
        f1_score BETWEEN 0 AND 1
    ),

    roc_auc NUMERIC(8,6) CHECK (
        roc_auc BETWEEN 0 AND 1
    ),

    average_precision NUMERIC(8,6) CHECK (
        average_precision BETWEEN 0 AND 1
    ),

    classification_threshold NUMERIC(8,6) NOT NULL CHECK (
        classification_threshold BETWEEN 0 AND 1
    ),

    confusion_matrix JSONB NOT NULL,

    evaluation_details JSONB NOT NULL DEFAULT '{}'::jsonb,

    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (jsonb_typeof(confusion_matrix) = 'object')
);

CREATE INDEX idx_evaluation_model
    ON model_evaluation(model_id, evaluated_at DESC);

CREATE INDEX idx_evaluation_auc
    ON model_evaluation(roc_auc DESC);

-- -----------------------------------------------------
-- Approval requirements
-- -----------------------------------------------------

CREATE TABLE approval_policy (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    project_id BIGINT NOT NULL
        REFERENCES ml_project(project_id),

    policy_name VARCHAR(150) NOT NULL,

    required_approvals INTEGER NOT NULL CHECK (
        required_approvals >= 1
    ),

    minimum_roc_auc NUMERIC(8,6) NOT NULL CHECK (
        minimum_roc_auc BETWEEN 0 AND 1
    ),

    minimum_recall NUMERIC(8,6) NOT NULL CHECK (
        minimum_recall BETWEEN 0 AND 1
    ),

    maximum_drift NUMERIC(10,6) NOT NULL CHECK (
        maximum_drift >= 0
    ),

    maximum_latency_ms NUMERIC(12,3) NOT NULL CHECK (
        maximum_latency_ms > 0
    ),

    maximum_fairness_gap NUMERIC(8,6) NOT NULL CHECK (
        maximum_fairness_gap BETWEEN 0 AND 1
    ),

    require_independent_review BOOLEAN NOT NULL DEFAULT TRUE,

    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(project_id, policy_name)
);

CREATE TABLE model_review (
    review_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_artifact(model_id),

    reviewer_id BIGINT NOT NULL
        REFERENCES project_member(member_id),

    decision review_decision NOT NULL,

    comments TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(model_id, reviewer_id)
);

CREATE INDEX idx_model_review_decision
    ON model_review(model_id, decision);

-- -----------------------------------------------------
-- Quality gate results
-- -----------------------------------------------------

CREATE TABLE quality_gate (
    gate_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_artifact(model_id),

    policy_id BIGINT NOT NULL
        REFERENCES approval_policy(policy_id),

    check_name VARCHAR(100) NOT NULL,

    actual_value NUMERIC(20,8),

    expected_value NUMERIC(20,8),

    comparison VARCHAR(20) NOT NULL CHECK (
        comparison IN (
            'GREATER_EQUAL',
            'LESS_EQUAL',
            'EQUAL'
        )
    ),

    status check_status NOT NULL DEFAULT 'PENDING',

    details TEXT,

    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(model_id, policy_id, check_name)
);

CREATE INDEX idx_quality_gate_model
    ON quality_gate(model_id, status);

-- -----------------------------------------------------
-- Deployment management
-- -----------------------------------------------------

CREATE TABLE model_deployment (
    deployment_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_artifact(model_id),

    environment deployment_environment NOT NULL,

    deployment_version VARCHAR(40) NOT NULL,

    active BOOLEAN NOT NULL DEFAULT FALSE,

    deployed_by VARCHAR(100) NOT NULL,

    deployed_at TIMESTAMPTZ,

    retired_at TIMESTAMPTZ,

    endpoint TEXT,

    UNIQUE(environment, deployment_version),

    CHECK (
        retired_at IS NULL OR
        deployed_at IS NULL OR
        retired_at >= deployed_at
    )
);

CREATE UNIQUE INDEX idx_one_active_model_per_environment
    ON model_deployment(environment)
    WHERE active = TRUE;

CREATE INDEX idx_deployment_model
    ON model_deployment(model_id, deployed_at DESC);

-- -----------------------------------------------------
-- Prediction logs
-- -----------------------------------------------------

CREATE TABLE prediction_log (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    deployment_id BIGINT NOT NULL
        REFERENCES model_deployment(deployment_id),

    request_id UUID NOT NULL UNIQUE,

    prediction_probability NUMERIC(8,6) NOT NULL CHECK (
        prediction_probability BETWEEN 0 AND 1
    ),

    prediction_label SMALLINT NOT NULL CHECK (
        prediction_label IN (0, 1)
    ),

    inference_latency_ms NUMERIC(12,3) NOT NULL CHECK (
        inference_latency_ms >= 0
    ),

    actual_label SMALLINT CHECK (
        actual_label IN (0, 1)
    ),

    feature_reference JSONB NOT NULL DEFAULT '{}'::jsonb,

    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_prediction_time
    ON prediction_log(predicted_at DESC);

CREATE INDEX idx_prediction_deployment
    ON prediction_log(deployment_id, predicted_at DESC);

-- -----------------------------------------------------
-- Feature drift monitoring
-- -----------------------------------------------------

CREATE TABLE drift_monitoring (
    drift_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    deployment_id BIGINT NOT NULL
        REFERENCES model_deployment(deployment_id),

    feature_name VARCHAR(100) NOT NULL,

    metric_name VARCHAR(50) NOT NULL CHECK (
        metric_name IN (
            'PSI',
            'KS_STATISTIC',
            'MEAN_SHIFT',
            'MISSING_RATE'
        )
    ),

    metric_value NUMERIC(20,8) NOT NULL,

    warning_threshold NUMERIC(20,8) NOT NULL,

    critical_threshold NUMERIC(20,8) NOT NULL,

    window_start TIMESTAMPTZ NOT NULL,

    window_end TIMESTAMPTZ NOT NULL,

    status check_status NOT NULL,

    CHECK (window_end > window_start),

    CHECK (warning_threshold <= critical_threshold)
);

CREATE INDEX idx_drift_deployment_time
    ON drift_monitoring(
        deployment_id,
        window_end DESC
    );

-- -----------------------------------------------------
-- Audit trail
-- -----------------------------------------------------

CREATE TABLE audit_event (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    actor VARCHAR(100) NOT NULL,

    action VARCHAR(100) NOT NULL,

    entity_type VARCHAR(60) NOT NULL,

    entity_reference TEXT NOT NULL,

    details JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_entity
    ON audit_event(entity_type, entity_reference, created_at DESC);

-- -----------------------------------------------------
-- Generic audit trigger
-- -----------------------------------------------------

CREATE OR REPLACE FUNCTION record_model_audit()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    reference_value TEXT;
BEGIN
    IF TG_OP = 'DELETE' THEN
        reference_value := OLD.model_id::TEXT;
    ELSE
        reference_value := NEW.model_id::TEXT;
    END IF;

    INSERT INTO audit_event (
        actor,
        action,
        entity_type,
        entity_reference,
        details
    )
    VALUES (
        CURRENT_USER,
        TG_OP,
        TG_TABLE_NAME,
        reference_value,
        jsonb_build_object(
            'operation', TG_OP,
            'database_timestamp', NOW()
        )
    );

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_model_audit
AFTER INSERT OR UPDATE OR DELETE
ON model_artifact
FOR EACH ROW
EXECUTE FUNCTION record_model_audit();

CREATE TRIGGER trg_deployment_audit
AFTER INSERT OR UPDATE OR DELETE
ON model_deployment
FOR EACH ROW
EXECUTE FUNCTION record_model_audit();

-- -----------------------------------------------------
-- Review eligibility trigger
-- -----------------------------------------------------

CREATE OR REPLACE FUNCTION validate_model_reviewer()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    reviewer_active BOOLEAN;
    reviewer_role VARCHAR(30);
    model_project BIGINT;
    reviewer_project BIGINT;
BEGIN
    SELECT
        pm.active,
        pm.role,
        pm.project_id
    INTO
        reviewer_active,
        reviewer_role,
        reviewer_project
    FROM project_member pm
    WHERE pm.member_id = NEW.reviewer_id;

    SELECT p.project_id
    INTO model_project
    FROM model_artifact ma
    JOIN training_run tr
        ON tr.run_id = ma.run_id
    JOIN ml_project p
        ON p.project_id = tr.project_id
    WHERE ma.model_id = NEW.model_id;

    IF reviewer_active IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'Reviewer is inactive or missing.';
    END IF;

    IF reviewer_project IS DISTINCT FROM model_project THEN
        RAISE EXCEPTION 'Reviewer belongs to another project.';
    END IF;

    IF reviewer_role NOT IN ('REVIEWER', 'ADMIN') THEN
        RAISE EXCEPTION 'Reviewer role is not eligible.';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_model_reviewer
BEFORE INSERT OR UPDATE
ON model_review
FOR EACH ROW
EXECUTE FUNCTION validate_model_reviewer();

-- -----------------------------------------------------
-- Approval and release eligibility function
-- -----------------------------------------------------

CREATE OR REPLACE FUNCTION check_model_eligibility(
    p_model_id BIGINT,
    p_policy_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    approval_count BIGINT,
    quality_passed BOOLEAN,
    failure_reasons TEXT[]
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_required INTEGER;
    v_min_auc NUMERIC;
    v_min_recall NUMERIC;
    v_max_drift NUMERIC;
    v_max_latency NUMERIC;
    v_max_fairness NUMERIC;

    v_approvals BIGINT;
    v_auc NUMERIC;
    v_recall NUMERIC;
    v_drift NUMERIC;
    v_latency NUMERIC;
    v_fairness NUMERIC;

    v_reasons TEXT[] := ARRAY[]::TEXT[];
    v_quality BOOLEAN := TRUE;
BEGIN
    SELECT
        required_approvals,
        minimum_roc_auc,
        minimum_recall,
        maximum_drift,
        maximum_latency_ms,
        maximum_fairness_gap
    INTO
        v_required,
        v_min_auc,
        v_min_recall,
        v_max_drift,
        v_max_latency,
        v_max_fairness
    FROM approval_policy
    WHERE policy_id = p_policy_id
      AND active = TRUE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Active approval policy not found.';
    END IF;

    SELECT COUNT(DISTINCT mr.reviewer_id)
    INTO v_approvals
    FROM model_review mr
    JOIN project_member pm
        ON pm.member_id = mr.reviewer_id
    WHERE mr.model_id = p_model_id
      AND mr.decision = 'APPROVE'
      AND pm.active = TRUE
      AND pm.role IN ('REVIEWER', 'ADMIN');

    SELECT
        me.roc_auc,
        me.recall_score,
        COALESCE(MAX(dm.metric_value), 0),
        COALESCE(MAX(pl.inference_latency_ms), 0),
        0
    INTO
        v_auc,
        v_recall,
        v_drift,
        v_latency,
        v_fairness
    FROM model_evaluation me
    LEFT JOIN model_artifact ma
        ON ma.model_id = me.model_id
    LEFT JOIN training_run tr
        ON tr.run_id = ma.run_id
    LEFT JOIN model_deployment md
        ON md.model_id = ma.model_id
    LEFT JOIN drift_monitoring dm
        ON dm.deployment_id = md.deployment_id
    LEFT JOIN prediction_log pl
        ON pl.deployment_id = md.deployment_id
    WHERE me.model_id = p_model_id
    GROUP BY
        me.roc_auc,
        me.recall_score
    ORDER BY MAX(me.evaluated_at) DESC
    LIMIT 1;

    IF v_auc IS NULL THEN
        v_quality := FALSE;
        v_reasons := array_append(
            v_reasons,
            'Evaluation metrics missing'
        );
    ELSE
        IF v_auc < v_min_auc THEN
            v_quality := FALSE;
            v_reasons := array_append(
                v_reasons,
                'ROC-AUC requirement failed'
            );
        END IF;

        IF v_recall < v_min_recall THEN
            v_quality := FALSE;
            v_reasons := array_append(
                v_reasons,
                'Recall requirement failed'
            );
        END IF;

        IF v_drift > v_max_drift THEN
            v_quality := FALSE;
            v_reasons := array_append(
                v_reasons,
                'Feature drift requirement failed'
            );
        END IF;

        IF v_latency > v_max_latency THEN
            v_quality := FALSE;
            v_reasons := array_append(
                v_reasons,
                'Latency requirement failed'
            );
        END IF;

        IF v_fairness > v_max_fairness THEN
            v_quality := FALSE;
            v_reasons := array_append(
                v_reasons,
                'Fairness requirement failed'
            );
        END IF;
    END IF;

    IF v_approvals < v_required THEN
        v_reasons := array_append(
            v_reasons,
            'Insufficient independent approvals'
        );
    END IF;

    eligible :=
        v_quality AND
        v_approvals >= v_required;

    approval_count := v_approvals;
    quality_passed := v_quality;
    failure_reasons := v_reasons;

    RETURN NEXT;
END;
$$;

-- -----------------------------------------------------
-- Transactional deployment procedure
-- -----------------------------------------------------

CREATE OR REPLACE PROCEDURE promote_model(
    p_model_id BIGINT,
    p_policy_id BIGINT,
    p_environment deployment_environment,
    p_actor VARCHAR,
    p_endpoint TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_eligible BOOLEAN;
    v_current_model_status model_status;
    v_project_id BIGINT;
    v_version VARCHAR(40);
    v_deployment_id BIGINT;
BEGIN
    -- Serialize concurrent production promotions.
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            p_environment::TEXT,
            0
        )
    );

    SELECT ma.model_status, tr.project_id, ma.version
    INTO
        v_current_model_status,
        v_project_id,
        v_version
    FROM model_artifact ma
    JOIN training_run tr
        ON tr.run_id = ma.run_id
    WHERE ma.model_id = p_model_id
    FOR UPDATE OF ma;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Model does not exist.';
    END IF;

    IF v_current_model_status NOT IN (
        'REGISTERED',
        'APPROVED',
        'DEPLOYED'
    ) THEN
        RAISE EXCEPTION 'Model state prohibits promotion.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM approval_policy ap
        WHERE ap.policy_id = p_policy_id
          AND ap.project_id = v_project_id
          AND ap.active = TRUE
    ) THEN
        RAISE EXCEPTION 'Policy does not belong to model project.';
    END IF;

    SELECT ce.eligible
    INTO v_eligible
    FROM check_model_eligibility(
        p_model_id,
        p_policy_id
    ) ce;

    IF v_eligible IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'Model failed deployment eligibility.';
    END IF;

    -- Retire the previous active deployment.
    UPDATE model_deployment
    SET
        active = FALSE,
        retired_at = NOW()
    WHERE environment = p_environment
      AND active = TRUE;

    UPDATE model_artifact
    SET model_status = 'RETIRED'
    WHERE model_status = 'DEPLOYED'
      AND model_id <> p_model_id
      AND model_id IN (
          SELECT model_id
          FROM model_deployment
          WHERE environment = p_environment
      );

    INSERT INTO model_deployment (
        model_id,
        environment,
        deployment_version,
        active,
        deployed_by,
        deployed_at,
        endpoint
    )
    VALUES (
        p_model_id,
        p_environment,
        v_version,
        TRUE,
        p_actor,
        NOW(),
        p_endpoint
    )
    RETURNING deployment_id INTO v_deployment_id;

    UPDATE model_artifact
    SET model_status = 'DEPLOYED'
    WHERE model_id = p_model_id;

    INSERT INTO audit_event (
        actor,
        action,
        entity_type,
        entity_reference,
        details
    )
    VALUES (
        p_actor,
        'MODEL_PROMOTED',
        'MODEL',
        p_model_id::TEXT,
        jsonb_build_object(
            'environment', p_environment,
            'deployment_id', v_deployment_id,
            'version', v_version
        )
    );

    RAISE NOTICE
        'Deployment completed: deployment ID %',
        v_deployment_id;
END;
$$;

-- -----------------------------------------------------
-- Reporting views
-- -----------------------------------------------------

CREATE VIEW model_performance_report AS
SELECT
    ma.model_id,
    ma.model_name,
    ma.version,
    ma.algorithm,
    ma.model_status,
    me.accuracy,
    me.precision_score,
    me.recall_score,
    me.f1_score,
    me.roc_auc,
    me.average_precision,
    me.classification_threshold,
    me.evaluated_at
FROM model_artifact ma
LEFT JOIN LATERAL (
    SELECT *
    FROM model_evaluation e
    WHERE e.model_id = ma.model_id
    ORDER BY e.evaluated_at DESC
    LIMIT 1
) me ON TRUE;

CREATE VIEW active_deployment_report AS
SELECT
    md.deployment_id,
    md.environment,
    md.deployment_version,
    md.deployed_by,
    md.deployed_at,
    ma.model_name,
    ma.version AS model_version,
    ma.model_status
FROM model_deployment md
JOIN model_artifact ma
    ON ma.model_id = md.model_id
WHERE md.active = TRUE;

CREATE VIEW project_training_summary AS
SELECT
    mp.project_name,
    COUNT(tr.run_id) AS total_runs,
    COUNT(*) FILTER (
        WHERE tr.status = 'TRAINED'
    ) AS trained_runs,
    COUNT(*) FILTER (
        WHERE tr.status = 'FAILED'
    ) AS failed_runs,
    COUNT(*) FILTER (
        WHERE tr.status = 'REJECTED'
    ) AS rejected_runs,
    COUNT(ma.model_id) AS registered_models
FROM ml_project mp
LEFT JOIN training_run tr
    ON tr.project_id = mp.project_id
LEFT JOIN model_artifact ma
    ON ma.run_id = tr.run_id
GROUP BY mp.project_id, mp.project_name;

-- -----------------------------------------------------
-- Sample project and team
-- -----------------------------------------------------

INSERT INTO ml_project (
    project_name,
    description,
    owner
)
VALUES (
    'Customer Churn Prediction',
    'Classical supervised learning and controlled model deployment.',
    'ml-platform-admin'
);

INSERT INTO project_member (
    project_id,
    username,
    role
)
SELECT project_id, username, role
FROM ml_project
CROSS JOIN (
    VALUES
        ('engineer-01', 'ML_ENGINEER'),
        ('engineer-02', 'DATA_ENGINEER'),
        ('reviewer-01', 'REVIEWER'),
        ('reviewer-02', 'REVIEWER'),
        ('reviewer-03', 'REVIEWER'),
        ('operator-01', 'OPERATOR'),
        ('admin-01', 'ADMIN')
) AS members(username, role)
WHERE project_name = 'Customer Churn Prediction';

-- -----------------------------------------------------
-- Sample dataset registration
-- -----------------------------------------------------

INSERT INTO dataset (
    project_id,
    dataset_name,
    description,
    source_type,
    owner
)
SELECT
    project_id,
    'customer_churn',
    'Customer-level churn prediction dataset.',
    'CSV',
    'engineer-02'
FROM ml_project
WHERE project_name = 'Customer Churn Prediction';

INSERT INTO dataset_version (
    dataset_id,
    version,
    fingerprint,
    row_count,
    column_count,
    target_column,
    status,
    class_distribution,
    validation_report
)
SELECT
    dataset_id,
    '1.0.0',
    repeat('a', 64),
    10000,
    8,
    'churn',
    'VALIDATED',
    '{"0": 7500, "1": 2500}'::jsonb,
    '{"missing_values_checked": true, "duplicates_checked": true, "target_validated": true}'::jsonb
FROM dataset
WHERE dataset_name = 'customer_churn';

-- -----------------------------------------------------
-- Feature contract
-- -----------------------------------------------------

INSERT INTO feature_definition (
    dataset_id,
    feature_name,
    data_type,
    is_required,
    nullable,
    sensitive
)
SELECT
    d.dataset_id,
    f.feature_name,
    f.data_type,
    f.is_required,
    f.nullable,
    f.sensitive
FROM dataset d
CROSS JOIN (
    VALUES
        ('tenure_months', 'INTEGER', TRUE, FALSE, FALSE),
        ('monthly_charges', 'NUMERIC', TRUE, TRUE, FALSE),
        ('support_calls', 'INTEGER', TRUE, TRUE, FALSE),
        ('contract', 'CATEGORY', TRUE, FALSE, FALSE),
        ('payment_method', 'CATEGORY', TRUE, TRUE, FALSE),
        ('internet_service', 'CATEGORY', TRUE, TRUE, FALSE)
) AS f(
    feature_name,
    data_type,
    is_required,
    nullable,
    sensitive
)
WHERE d.dataset_name = 'customer_churn';

-- -----------------------------------------------------
-- Release policy
-- -----------------------------------------------------

INSERT INTO approval_policy (
    project_id,
    policy_name,
    required_approvals,
    minimum_roc_auc,
    minimum_recall,
    maximum_drift,
    maximum_latency_ms,
    maximum_fairness_gap,
    require_independent_review
)
SELECT
    project_id,
    'production-classifier-policy',
    2,
    0.75,
    0.60,
    0.20,
    250,
    0.15,
    TRUE
FROM ml_project
WHERE project_name = 'Customer Churn Prediction';

-- -----------------------------------------------------
-- Training run
-- -----------------------------------------------------

INSERT INTO training_run (
    run_id,
    project_id,
    dataset_version_id,
    run_name,
    algorithm,
    target_column,
    random_seed,
    test_size,
    validation_size,
    status,
    hyperparameters,
    training_parameters,
    started_at,
    completed_at,
    created_by
)
SELECT
    '11111111-1111-4111-8111-111111111111',
    mp.project_id,
    dv.dataset_version_id,
    'churn-baseline-run',
    'LogisticRegression',
    'churn',
    42,
    0.20,
    0.20,
    'TRAINED',
    '{"C": 1.0, "class_weight": "balanced"}'::jsonb,
    '{"cross_validation": 3, "scoring": "roc_auc"}'::jsonb,
    NOW() - INTERVAL '10 minutes',
    NOW(),
    'engineer-01'
FROM ml_project mp
JOIN dataset d
    ON d.project_id = mp.project_id
JOIN dataset_version dv
    ON dv.dataset_id = d.dataset_id
WHERE mp.project_name = 'Customer Churn Prediction';

-- -----------------------------------------------------
-- Model registration
-- -----------------------------------------------------

INSERT INTO model_artifact (
    run_id,
    model_name,
    version,
    algorithm,
    artifact_uri,
    artifact_checksum,
    framework,
    framework_version,
    feature_schema,
    preprocessing_schema,
    model_status
)
VALUES (
    '11111111-1111-4111-8111-111111111111',
    'customer-churn-classifier',
    '1.0.0',
    'LogisticRegression',
    's3://ml-artifacts/churn/1.0.0/model.joblib',
    repeat('b', 64),
    'scikit-learn',
    '1.x',
    '{"features": ["tenure_months", "monthly_charges", "support_calls", "contract", "payment_method", "internet_service"]}'::jsonb,
    '{"numeric": "median_imputation_and_standard_scaling", "categorical": "mode_imputation_and_one_hot_encoding"}'::jsonb,
    'REGISTERED'
);

-- -----------------------------------------------------
-- Evaluation
-- -----------------------------------------------------

INSERT INTO model_evaluation (
    model_id,
    evaluation_dataset_version_id,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    average_precision,
    classification_threshold,
    confusion_matrix,
    evaluation_details
)
SELECT
    ma.model_id,
    dv.dataset_version_id,
    0.84,
    0.79,
    0.76,
    0.775,
    0.86,
    0.72,
    0.50,
    '{"tn": 1400, "fp": 100, "fn": 120, "tp": 380}'::jsonb,
    '{"test_partition": "held_out", "leakage_check": "passed"}'::jsonb
FROM model_artifact ma
JOIN training_run tr
    ON tr.run_id = ma.run_id
JOIN dataset_version dv
    ON dv.dataset_version_id = tr.dataset_version_id
WHERE ma.version = '1.0.0';

-- -----------------------------------------------------
-- Independent model reviews
-- -----------------------------------------------------

INSERT INTO model_review (
    model_id,
    reviewer_id,
    decision,
    comments
)
SELECT
    ma.model_id,
    pm.member_id,
    'APPROVE',
    CASE pm.username
        WHEN 'reviewer-01'
        THEN 'Statistical evaluation reviewed.'
        ELSE 'Operational and validation evidence reviewed.'
    END
FROM model_artifact ma
CROSS JOIN project_member pm
WHERE ma.version = '1.0.0'
  AND pm.username IN ('reviewer-01', 'reviewer-02');

-- -----------------------------------------------------
-- Quality gate records
-- -----------------------------------------------------

INSERT INTO quality_gate (
    model_id,
    policy_id,
    check_name,
    actual_value,
    expected_value,
    comparison,
    status,
    details
)
SELECT
    ma.model_id,
    ap.policy_id,
    checks.check_name,
    checks.actual_value,
    checks.expected_value,
    checks.comparison,
    CASE
        WHEN checks.actual_value >= checks.expected_value
             AND checks.comparison = 'GREATER_EQUAL'
            THEN 'PASSED'::check_status

        WHEN checks.actual_value <= checks.expected_value
             AND checks.comparison = 'LESS_EQUAL'
            THEN 'PASSED'::check_status

        ELSE 'FAILED'::check_status
    END,
    checks.details
FROM model_artifact ma
JOIN training_run tr
    ON tr.run_id = ma.run_id
JOIN approval_policy ap
    ON ap.project_id = tr.project_id
CROSS JOIN (
    VALUES
        (
            'ROC-AUC',
            0.86::NUMERIC,
            0.75::NUMERIC,
            'GREATER_EQUAL',
            'ROC-AUC threshold'
        ),
        (
            'Recall',
            0.76::NUMERIC,
            0.60::NUMERIC,
            'GREATER_EQUAL',
            'Recall threshold'
        ),
        (
            'Feature drift',
            0.08::NUMERIC,
            0.20::NUMERIC,
            'LESS_EQUAL',
            'Drift threshold'
        ),
        (
            'Inference latency',
            45::NUMERIC,
            250::NUMERIC,
            'LESS_EQUAL',
            'Latency threshold'
        ),
        (
            'Fairness gap',
            0.05::NUMERIC,
            0.15::NUMERIC,
            'LESS_EQUAL',
            'Fairness threshold'
        )
) AS checks(
    check_name,
    actual_value,
    expected_value,
    comparison,
    details
)
WHERE ma.version = '1.0.0'
  AND ap.policy_name = 'production-classifier-policy';

-- -----------------------------------------------------
-- Workflow queries
-- -----------------------------------------------------

-- 1. Dataset retrieval and validation.
SELECT
    d.dataset_name,
    dv.version,
    dv.row_count,
    dv.status,
    dv.class_distribution,
    dv.validation_report
FROM dataset d
JOIN dataset_version dv
    ON dv.dataset_id = d.dataset_id;

-- 2. Training history.
SELECT
    run_name,
    algorithm,
    status,
    hyperparameters,
    created_at,
    completed_at
FROM training_run
ORDER BY created_at DESC;

-- 3. Latest model performance.
SELECT *
FROM model_performance_report
ORDER BY roc_auc DESC NULLS LAST;

-- 4. Model review aggregation.
SELECT
    ma.version,
    COUNT(*) FILTER (
        WHERE mr.decision = 'APPROVE'
    ) AS approvals,
    COUNT(*) FILTER (
        WHERE mr.decision = 'REQUEST_CHANGES'
    ) AS requested_changes,
    COUNT(*) FILTER (
        WHERE mr.decision = 'REJECT'
    ) AS rejections
FROM model_artifact ma
LEFT JOIN model_review mr
    ON mr.model_id = ma.model_id
GROUP BY ma.model_id, ma.version;

-- 5. Quality gate evaluation.
SELECT
    ma.version,
    qg.check_name,
    qg.actual_value,
    qg.expected_value,
    qg.status
FROM quality_gate qg
JOIN model_artifact ma
    ON ma.model_id = qg.model_id
ORDER BY ma.version, qg.check_name;

-- 6. Eligibility determination.
SELECT
    ma.version,
    eligibility.*
FROM model_artifact ma
CROSS JOIN approval_policy ap
CROSS JOIN LATERAL check_model_eligibility(
    ma.model_id,
    ap.policy_id
) eligibility
WHERE ma.version = '1.0.0'
  AND ap.policy_name = 'production-classifier-policy';

-- 7. CTE-based model ranking.
WITH ranked_models AS (
    SELECT
        ma.model_id,
        ma.version,
        me.roc_auc,
        me.f1_score,
        RANK() OVER (
            ORDER BY me.roc_auc DESC
        ) AS performance_rank
    FROM model_artifact ma
    JOIN model_evaluation me
        ON me.model_id = ma.model_id
)
SELECT *
FROM ranked_models
ORDER BY performance_rank;

-- 8. Training success aggregation.
SELECT
    project_name,
    total_runs,
    trained_runs,
    failed_runs,
    rejected_runs,
    registered_models
FROM project_training_summary;

-- -----------------------------------------------------
-- Controlled deployment
-- -----------------------------------------------------

DO $$
DECLARE
    v_model_id BIGINT;
    v_policy_id BIGINT;
BEGIN
    SELECT model_id
    INTO v_model_id
    FROM model_artifact
    WHERE version = '1.0.0';

    SELECT policy_id
    INTO v_policy_id
    FROM approval_policy
    WHERE policy_name = 'production-classifier-policy';

    CALL promote_model(
        v_model_id,
        v_policy_id,
        'PRODUCTION',
        'admin-01',
        'https://internal.example.com/churn/v1'
    );
END;
$$;

-- -----------------------------------------------------
-- Prediction logging
-- -----------------------------------------------------

INSERT INTO prediction_log (
    deployment_id,
    request_id,
    prediction_probability,
    prediction_label,
    inference_latency_ms,
    feature_reference
)
SELECT
    deployment_id,
    gen_random_uuid(),
    0.83,
    1,
    35.5,
    '{"tenure_months": 4, "monthly_charges": 110}'::jsonb
FROM model_deployment
WHERE environment = 'PRODUCTION'
  AND active = TRUE;

-- -----------------------------------------------------
-- Drift monitoring sample
-- -----------------------------------------------------

INSERT INTO drift_monitoring (
    deployment_id,
    feature_name,
    metric_name,
    metric_value,
    warning_threshold,
    critical_threshold,
    window_start,
    window_end,
    status
)
SELECT
    deployment_id,
    'monthly_charges',
    'PSI',
    0.08,
    0.10,
    0.20,
    NOW() - INTERVAL '24 hours',
    NOW(),
    'PASSED'
FROM model_deployment
WHERE environment = 'PRODUCTION'
  AND active = TRUE;

-- -----------------------------------------------------
-- Operational reporting
-- -----------------------------------------------------

-- Deployment state.
SELECT *
FROM active_deployment_report;

-- Prediction volume and latency.
SELECT
    md.environment,
    COUNT(pl.prediction_id) AS prediction_count,
    ROUND(
        AVG(pl.inference_latency_ms),
        3
    ) AS average_latency_ms,
    ROUND(
        MAX(pl.inference_latency_ms),
        3
    ) AS maximum_latency_ms,
    ROUND(
        AVG(pl.prediction_label::NUMERIC),
        4
    ) AS positive_prediction_rate
FROM model_deployment md
LEFT JOIN prediction_log pl
    ON pl.deployment_id = md.deployment_id
GROUP BY md.environment;

-- Drift alerts.
SELECT
    feature_name,
    metric_name,
    metric_value,
    warning_threshold,
    critical_threshold,
    status,
    window_end
FROM drift_monitoring
WHERE status IN ('FAILED', 'WARNING')
ORDER BY window_end DESC;

-- Audit history.
SELECT
    actor,
    action,
    entity_type,
    entity_reference,
    details,
    created_at
FROM audit_event
ORDER BY created_at DESC;

-- -----------------------------------------------------
-- Transaction failure demonstration
-- -----------------------------------------------------

DO $$
BEGIN
    BEGIN
        INSERT INTO dataset_version (
            dataset_id,
            version,
            fingerprint,
            row_count,
            column_count,
            target_column,
            status
        )
        SELECT
            dataset_id,
            'invalid-version',
            repeat('c', 64),
            -100,
            8,
            'churn',
            'REGISTERED'
        FROM dataset
        WHERE dataset_name = 'customer_churn';

        RAISE EXCEPTION
            'The invalid sample should have been rejected.';

    EXCEPTION
        WHEN check_violation THEN
            RAISE NOTICE
                'Invalid dataset version rejected by database constraint.';
    END;
END;
$$;

-- -----------------------------------------------------
-- Security and deployment considerations
-- -----------------------------------------------------

/*
Production deployment should additionally implement:

1. Dedicated database roles and least-privilege grants.
2. Row-level security for multi-project isolation.
3. External secret management.
4. Encrypted artifact storage.
5. Immutable artifact checksums.
6. Signed model manifests.
7. Restricted deployment procedure execution.
8. Retention and deletion policies.
9. Partitioning for high-volume prediction logs.
10. Controlled schema migrations.

The eligibility function and deployment procedure are
governance components. Database administrators should
restrict direct modifications to governance tables.
*/

COMMIT;
