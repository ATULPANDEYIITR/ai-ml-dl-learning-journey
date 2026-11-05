/*
    MODEL INTERPRETABILITY DATABASE
    PostgreSQL 14+

    Domain:
    Enterprise model governance and interpretability.

    Capabilities:
    - Model and feature registry
    - Model versions and evaluation datasets
    - Interpretation executions
    - Coefficient importance
    - Permutation importance
    - Partial dependence
    - Interpretation policies
    - Validation triggers
    - Audit events
    - Reporting and eligibility views

    Run this script in a PostgreSQL database.
*/

BEGIN;

CREATE SCHEMA IF NOT EXISTS interpretability;

SET search_path TO interpretability, public;

-- ============================================================
-- 1. DOMAIN TYPES
-- ============================================================

CREATE TYPE model_family AS ENUM (
    'LINEAR_REGRESSION',
    'LOGISTIC_REGRESSION',
    'DECISION_TREE',
    'RANDOM_FOREST',
    'GRADIENT_BOOSTING',
    'NEURAL_NETWORK',
    'OTHER'
);

CREATE TYPE analysis_method AS ENUM (
    'COEFFICIENT',
    'TREE_BASED',
    'PERMUTATION',
    'PARTIAL_DEPENDENCE'
);

CREATE TYPE execution_status AS ENUM (
    'CREATED',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'REJECTED'
);

CREATE TYPE approval_status AS ENUM (
    'PENDING',
    'APPROVED',
    'REJECTED'
);

-- ============================================================
-- 2. MODEL REGISTRY
-- ============================================================

CREATE TABLE model_registry (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_name VARCHAR(150) NOT NULL,

    model_family model_family NOT NULL,

    business_purpose TEXT NOT NULL,

    owner_name VARCHAR(150) NOT NULL,

    production_status BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (model_name, owner_name),

    CHECK (length(trim(model_name)) > 0),
    CHECK (length(trim(business_purpose)) > 0)
);

CREATE TABLE model_versions (
    version_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    model_id BIGINT NOT NULL
        REFERENCES model_registry(model_id),

    version_number VARCHAR(50) NOT NULL,

    artifact_hash VARCHAR(128) NOT NULL,

    training_dataset_reference VARCHAR(250) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (model_id, version_number),

    CHECK (length(trim(artifact_hash)) >= 16)
);

-- ============================================================
-- 3. FEATURE REGISTRY
-- ============================================================

CREATE TABLE features (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    feature_name VARCHAR(150) NOT NULL UNIQUE,

    description TEXT NOT NULL,

    data_type VARCHAR(40) NOT NULL,

    unit VARCHAR(50),

    sensitive BOOLEAN NOT NULL DEFAULT FALSE,

    business_owner VARCHAR(150),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (length(trim(feature_name)) > 0)
);

CREATE TABLE model_features (
    version_id BIGINT NOT NULL
        REFERENCES model_versions(version_id),

    feature_id BIGINT NOT NULL
        REFERENCES features(feature_id),

    feature_position INTEGER NOT NULL,

    included BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (version_id, feature_id),

    UNIQUE (version_id, feature_position),

    CHECK (feature_position >= 0)
);

-- ============================================================
-- 4. INTERPRETATION POLICIES
-- ============================================================

CREATE TABLE interpretation_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    policy_name VARCHAR(150) NOT NULL UNIQUE,

    minimum_samples INTEGER NOT NULL DEFAULT 20,

    minimum_permutation_repeats INTEGER NOT NULL DEFAULT 5,

    maximum_permutation_repeats INTEGER NOT NULL DEFAULT 100,

    minimum_allowed_importance DOUBLE PRECISION NOT NULL DEFAULT 0,

    allow_sensitive_features BOOLEAN NOT NULL DEFAULT FALSE,

    require_test_dataset BOOLEAN NOT NULL DEFAULT TRUE,

    require_approval BOOLEAN NOT NULL DEFAULT TRUE,

    active BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (minimum_samples >= 2),

    CHECK (minimum_permutation_repeats >= 1),

    CHECK (
        maximum_permutation_repeats >=
        minimum_permutation_repeats
    ),

    CHECK (
        minimum_allowed_importance >= 0
        AND minimum_allowed_importance <= 1
    )
);

-- ============================================================
-- 5. EVALUATION DATASETS
-- ============================================================

CREATE TABLE evaluation_datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    dataset_name VARCHAR(150) NOT NULL,

    dataset_reference VARCHAR(250) NOT NULL,

    dataset_hash VARCHAR(128) NOT NULL,

    dataset_role VARCHAR(20) NOT NULL,

    sample_count INTEGER NOT NULL,

    feature_count INTEGER NOT NULL,

    contains_sensitive_data BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (dataset_role IN ('TRAIN', 'VALIDATION', 'TEST')),

    CHECK (sample_count > 0),

    CHECK (feature_count > 0)
);

-- ============================================================
-- 6. INTERPRETATION EXECUTIONS
-- ============================================================

CREATE TABLE interpretation_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    version_id BIGINT NOT NULL
        REFERENCES model_versions(version_id),

    dataset_id BIGINT NOT NULL
        REFERENCES evaluation_datasets(dataset_id),

    policy_id BIGINT NOT NULL
        REFERENCES interpretation_policies(policy_id),

    requested_by VARCHAR(150) NOT NULL,

    execution_status execution_status NOT NULL DEFAULT 'CREATED',

    sample_count INTEGER NOT NULL,

    baseline_mse DOUBLE PRECISION,

    baseline_mae DOUBLE PRECISION,

    baseline_rmse DOUBLE PRECISION,

    permutation_repeats INTEGER NOT NULL DEFAULT 10,

    random_seed BIGINT NOT NULL DEFAULT 42,

    failure_reason TEXT,

    started_at TIMESTAMPTZ,

    completed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (sample_count > 0),

    CHECK (permutation_repeats > 0),

    CHECK (
        baseline_mse IS NULL OR baseline_mse >= 0
    ),

    CHECK (
        baseline_mae IS NULL OR baseline_mae >= 0
    ),

    CHECK (
        baseline_rmse IS NULL OR baseline_rmse >= 0
    ),

    CHECK (
        baseline_mse IS NULL OR
        baseline_mae IS NULL OR
        baseline_rmse IS NULL OR
        abs(baseline_rmse * baseline_rmse - baseline_mse) < 0.000001
    ),

    CHECK (
        execution_status <> 'COMPLETED'
        OR (
            baseline_mse IS NOT NULL
            AND baseline_mae IS NOT NULL
            AND baseline_rmse IS NOT NULL
            AND completed_at IS NOT NULL
        )
    )
);

-- ============================================================
-- 7. FEATURE IMPORTANCE RESULTS
-- ============================================================

CREATE TABLE feature_importance_results (
    importance_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES interpretation_runs(run_id)
        ON DELETE CASCADE,

    feature_id BIGINT NOT NULL
        REFERENCES features(feature_id),

    method analysis_method NOT NULL,

    importance_score DOUBLE PRECISION NOT NULL,

    normalized_score DOUBLE PRECISION NOT NULL DEFAULT 0,

    standard_deviation DOUBLE PRECISION,

    experiment_count INTEGER NOT NULL DEFAULT 1,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, feature_id, method),

    CHECK (importance_score >= '-Infinity'::DOUBLE PRECISION),

    CHECK (importance_score <= 'Infinity'::DOUBLE PRECISION),

    CHECK (
        normalized_score >= 0
        AND normalized_score <= 1
    ),

    CHECK (
        standard_deviation IS NULL OR standard_deviation >= 0
    ),

    CHECK (experiment_count >= 1)
);

-- ============================================================
-- 8. PARTIAL DEPENDENCE RESULTS
-- ============================================================

CREATE TABLE partial_dependence_results (
    dependence_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES interpretation_runs(run_id)
        ON DELETE CASCADE,

    feature_id BIGINT NOT NULL
        REFERENCES features(feature_id),

    grid_position INTEGER NOT NULL,

    feature_value DOUBLE PRECISION NOT NULL,

    average_prediction DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, feature_id, grid_position),

    CHECK (grid_position >= 0),

    CHECK (
        feature_value >= '-Infinity'::DOUBLE PRECISION
        AND feature_value <= 'Infinity'::DOUBLE PRECISION
    ),

    CHECK (
        average_prediction >= '-Infinity'::DOUBLE PRECISION
        AND average_prediction <= 'Infinity'::DOUBLE PRECISION
    )
);

-- ============================================================
-- 9. GOVERNANCE APPROVALS
-- ============================================================

CREATE TABLE interpretation_approvals (
    approval_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT NOT NULL
        REFERENCES interpretation_runs(run_id),

    reviewer VARCHAR(150) NOT NULL,

    decision approval_status NOT NULL DEFAULT 'PENDING',

    comments TEXT,

    decided_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, reviewer),

    CHECK (
        decision = 'PENDING'
        OR decided_at IS NOT NULL
    )
);

-- ============================================================
-- 10. AUDIT EVENTS
-- ============================================================

CREATE TABLE interpretation_audit (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    run_id BIGINT
        REFERENCES interpretation_runs(run_id)
        ON DELETE SET NULL,

    event_type VARCHAR(100) NOT NULL,

    actor VARCHAR(150) NOT NULL,

    event_details JSONB NOT NULL DEFAULT '{}'::JSONB,

    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 11. PERFORMANCE INDEXES
-- ============================================================

CREATE INDEX idx_model_versions_model
    ON model_versions(model_id, created_at DESC);

CREATE INDEX idx_model_features_feature
    ON model_features(feature_id, version_id);

CREATE INDEX idx_runs_status_created
    ON interpretation_runs(execution_status, created_at DESC);

CREATE INDEX idx_runs_version
    ON interpretation_runs(version_id, created_at DESC);

CREATE INDEX idx_importance_run_method
    ON feature_importance_results(run_id, method);

CREATE INDEX idx_importance_feature_score
    ON feature_importance_results(feature_id, importance_score DESC);

CREATE INDEX idx_dependence_run_feature
    ON partial_dependence_results(run_id, feature_id, grid_position);

CREATE INDEX idx_approvals_status
    ON interpretation_approvals(run_id, decision);

CREATE INDEX idx_audit_timestamp
    ON interpretation_audit(event_timestamp DESC);

CREATE INDEX idx_audit_details
    ON interpretation_audit USING GIN(event_details);

-- ============================================================
-- 12. DATA INTEGRITY FUNCTIONS
-- ============================================================

CREATE OR REPLACE FUNCTION validate_interpretation_run()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_minimum_samples INTEGER;
    v_minimum_repeats INTEGER;
    v_maximum_repeats INTEGER;
    v_allow_sensitive BOOLEAN;
    v_dataset_role VARCHAR(20);
    v_dataset_samples INTEGER;
    v_dataset_features INTEGER;
    v_sensitive_count INTEGER;
    v_expected_features INTEGER;
BEGIN
    SELECT
        p.minimum_samples,
        p.minimum_permutation_repeats,
        p.maximum_permutation_repeats,
        p.allow_sensitive_features
    INTO
        v_minimum_samples,
        v_minimum_repeats,
        v_maximum_repeats,
        v_allow_sensitive
    FROM interpretation_policies p
    WHERE p.policy_id = NEW.policy_id
      AND p.active = TRUE;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Interpretation policy is missing or inactive';
    END IF;

    SELECT
        dataset_role,
        sample_count,
        feature_count
    INTO
        v_dataset_role,
        v_dataset_samples,
        v_dataset_features
    FROM evaluation_datasets
    WHERE dataset_id = NEW.dataset_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Evaluation dataset does not exist';
    END IF;

    SELECT count(*)
    INTO v_expected_features
    FROM model_features
    WHERE version_id = NEW.version_id
      AND included = TRUE;

    IF v_expected_features = 0 THEN
        RAISE EXCEPTION 'Model version has no registered features';
    END IF;

    IF NEW.sample_count <> v_dataset_samples THEN
        RAISE EXCEPTION
            'Run sample count differs from dataset sample count';
    END IF;

    IF NEW.sample_count < v_minimum_samples THEN
        RAISE EXCEPTION 'Dataset is below minimum sample threshold';
    END IF;

    IF v_dataset_features <> v_expected_features THEN
        RAISE EXCEPTION
            'Dataset and model feature counts do not match';
    END IF;

    IF NEW.permutation_repeats NOT BETWEEN
        v_minimum_repeats AND v_maximum_repeats THEN
        RAISE EXCEPTION 'Permutation repeat count violates policy';
    END IF;

    IF v_dataset_role <> 'TEST' THEN
        RAISE EXCEPTION
            'Interpretability analysis requires a test dataset';
    END IF;

    IF NOT v_allow_sensitive THEN
        SELECT count(*)
        INTO v_sensitive_count
        FROM model_features mf
        JOIN features f ON f.feature_id = mf.feature_id
        WHERE mf.version_id = NEW.version_id
          AND mf.included = TRUE
          AND f.sensitive = TRUE;

        IF v_sensitive_count > 0 THEN
            RAISE EXCEPTION
                'Sensitive features are prohibited by policy';
        END IF;
    END IF;

    IF NEW.execution_status = 'RUNNING'
       AND NEW.started_at IS NULL THEN
        NEW.started_at := CURRENT_TIMESTAMP;
    END IF;

    IF NEW.execution_status = 'COMPLETED' THEN
        NEW.completed_at := COALESCE(
            NEW.completed_at,
            CURRENT_TIMESTAMP
        );
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_interpretation_run
BEFORE INSERT OR UPDATE ON interpretation_runs
FOR EACH ROW
EXECUTE FUNCTION validate_interpretation_run();

-- ============================================================
-- 13. IMPORTANCE VALIDATION TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION validate_importance_feature()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_version_id BIGINT;
    v_status execution_status;
    v_feature_exists BOOLEAN;
BEGIN
    SELECT version_id, execution_status
    INTO v_version_id, v_status
    FROM interpretation_runs
    WHERE run_id = NEW.run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Interpretation run does not exist';
    END IF;

    IF v_status <> 'RUNNING' THEN
        RAISE EXCEPTION
            'Importance results can only be inserted during execution';
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM model_features
        WHERE version_id = v_version_id
          AND feature_id = NEW.feature_id
          AND included = TRUE
    )
    INTO v_feature_exists;

    IF NOT v_feature_exists THEN
        RAISE EXCEPTION
            'Feature is not registered for this model version';
    END IF;

    IF NEW.importance_score = 'NaN'::DOUBLE PRECISION
       OR NEW.importance_score = 'Infinity'::DOUBLE PRECISION
       OR NEW.importance_score = '-Infinity'::DOUBLE PRECISION THEN
        RAISE EXCEPTION 'Importance score must be finite';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_importance_feature
BEFORE INSERT OR UPDATE ON feature_importance_results
FOR EACH ROW
EXECUTE FUNCTION validate_importance_feature();

-- ============================================================
-- 14. AUDIT TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION audit_run_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO interpretation_audit (
        run_id,
        event_type,
        actor,
        event_details
    )
    VALUES (
        NEW.run_id,
        'RUN_STATUS_CHANGED',
        NEW.requested_by,
        jsonb_build_object(
            'old_status',
            CASE
                WHEN TG_OP = 'INSERT' THEN NULL
                ELSE OLD.execution_status
            END,
            'new_status',
            NEW.execution_status,
            'policy_id',
            NEW.policy_id,
            'dataset_id',
            NEW.dataset_id
        )
    );

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_audit_run_changes
AFTER INSERT OR UPDATE OF execution_status
ON interpretation_runs
FOR EACH ROW
EXECUTE FUNCTION audit_run_changes();

-- ============================================================
-- 15. APPROVAL VALIDATION
-- ============================================================

CREATE OR REPLACE FUNCTION validate_interpretation_approval()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_status execution_status;
BEGIN
    SELECT execution_status
    INTO v_status
    FROM interpretation_runs
    WHERE run_id = NEW.run_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Interpretation run not found';
    END IF;

    IF v_status <> 'COMPLETED' THEN
        RAISE EXCEPTION
            'Only completed interpretation runs can be approved';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_interpretation_approval
BEFORE INSERT OR UPDATE ON interpretation_approvals
FOR EACH ROW
WHEN (NEW.decision <> 'PENDING')
EXECUTE FUNCTION validate_interpretation_approval();

-- ============================================================
-- 16. APPROVAL AND ELIGIBILITY VIEW
-- ============================================================

CREATE OR REPLACE VIEW interpretation_eligibility AS
SELECT
    r.run_id,
    r.version_id,
    r.execution_status,
    r.policy_id,
    p.require_approval,

    count(a.approval_id) FILTER (
        WHERE a.decision = 'APPROVED'
    ) AS approved_count,

    count(a.approval_id) FILTER (
        WHERE a.decision = 'REJECTED'
    ) AS rejected_count,

    CASE
        WHEN r.execution_status <> 'COMPLETED'
            THEN FALSE

        WHEN p.require_approval = FALSE
            THEN TRUE

        WHEN count(a.approval_id) FILTER (
            WHERE a.decision = 'REJECTED'
        ) > 0
            THEN FALSE

        WHEN count(a.approval_id) FILTER (
            WHERE a.decision = 'APPROVED'
        ) >= 1
            THEN TRUE

        ELSE FALSE
    END AS eligible_for_acceptance

FROM interpretation_runs r
JOIN interpretation_policies p
    ON p.policy_id = r.policy_id
LEFT JOIN interpretation_approvals a
    ON a.run_id = r.run_id

GROUP BY
    r.run_id,
    r.version_id,
    r.execution_status,
    r.policy_id,
    p.require_approval;

-- ============================================================
-- 17. FEATURE IMPORTANCE REPORTING VIEW
-- ============================================================

CREATE OR REPLACE VIEW feature_importance_report AS
SELECT
    r.run_id,
    mr.model_name,
    mv.version_number,
    f.feature_name,
    fir.method,
    fir.importance_score,
    fir.normalized_score,
    fir.standard_deviation,
    fir.experiment_count,
    fir.created_at

FROM feature_importance_results fir

JOIN interpretation_runs r
    ON r.run_id = fir.run_id

JOIN model_versions mv
    ON mv.version_id = r.version_id

JOIN model_registry mr
    ON mr.model_id = mv.model_id

JOIN features f
    ON f.feature_id = fir.feature_id;

-- ============================================================
-- 18. PARTIAL DEPENDENCE REPORTING VIEW
-- ============================================================

CREATE OR REPLACE VIEW partial_dependence_report AS
SELECT
    r.run_id,
    mr.model_name,
    mv.version_number,
    f.feature_name,
    p.grid_position,
    p.feature_value,
    p.average_prediction

FROM partial_dependence_results p

JOIN interpretation_runs r
    ON r.run_id = p.run_id

JOIN model_versions mv
    ON mv.version_id = r.version_id

JOIN model_registry mr
    ON mr.model_id = mv.model_id

JOIN features f
    ON f.feature_id = p.feature_id;

-- ============================================================
-- 19. WORKFLOW FUNCTIONS
-- ============================================================

CREATE OR REPLACE FUNCTION start_interpretation(
    p_version_id BIGINT,
    p_dataset_id BIGINT,
    p_policy_id BIGINT,
    p_requested_by VARCHAR,
    p_repeats INTEGER DEFAULT 10,
    p_seed BIGINT DEFAULT 42
)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    v_samples INTEGER;
    v_run_id BIGINT;
BEGIN
    SELECT sample_count
    INTO v_samples
    FROM evaluation_datasets
    WHERE dataset_id = p_dataset_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Dataset not found';
    END IF;

    INSERT INTO interpretation_runs (
        version_id,
        dataset_id,
        policy_id,
        requested_by,
        sample_count,
        permutation_repeats,
        random_seed,
        execution_status
    )
    VALUES (
        p_version_id,
        p_dataset_id,
        p_policy_id,
        p_requested_by,
        v_samples,
        p_repeats,
        p_seed,
        'RUNNING'
    )
    RETURNING run_id INTO v_run_id;

    RETURN v_run_id;
END;
$$;

CREATE OR REPLACE FUNCTION complete_interpretation(
    p_run_id BIGINT,
    p_mse DOUBLE PRECISION,
    p_mae DOUBLE PRECISION,
    p_rmse DOUBLE PRECISION
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_mse < 0 OR p_mae < 0 OR p_rmse < 0 THEN
        RAISE EXCEPTION 'Evaluation metrics cannot be negative';
    END IF;

    IF abs(p_rmse * p_rmse - p_mse) > 0.000001 THEN
        RAISE EXCEPTION 'RMSE must be the square root of MSE';
    END IF;

    UPDATE interpretation_runs
    SET
        baseline_mse = p_mse,
        baseline_mae = p_mae,
        baseline_rmse = p_rmse,
        execution_status = 'COMPLETED',
        completed_at = CURRENT_TIMESTAMP
    WHERE run_id = p_run_id
      AND execution_status = 'RUNNING';

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Run does not exist or is not currently running';
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION fail_interpretation(
    p_run_id BIGINT,
    p_reason TEXT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_reason IS NULL OR length(trim(p_reason)) = 0 THEN
        RAISE EXCEPTION 'Failure reason is required';
    END IF;

    UPDATE interpretation_runs
    SET
        execution_status = 'FAILED',
        failure_reason = p_reason,
        completed_at = CURRENT_TIMESTAMP
    WHERE run_id = p_run_id
      AND execution_status IN ('CREATED', 'RUNNING');

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Run cannot transition to FAILED';
    END IF;
END;
$$;

-- ============================================================
-- 20. SAMPLE DATA
-- ============================================================

INSERT INTO model_registry (
    model_name,
    model_family,
    business_purpose,
    owner_name,
    production_status
)
VALUES (
    'Business Credit Risk Model',
    'LINEAR_REGRESSION',
    'Estimate business credit-risk scores using financial indicators.',
    'Risk Analytics Team',
    TRUE
);

INSERT INTO model_versions (
    model_id,
    version_number,
    artifact_hash,
    training_dataset_reference
)
SELECT
    model_id,
    '2.1.0',
    'a7c92d4e8b10f623ab91c456',
    'credit_training_2026'
FROM model_registry
WHERE model_name = 'Business Credit Risk Model';

INSERT INTO features (
    feature_name,
    description,
    data_type,
    unit,
    sensitive,
    business_owner
)
VALUES
(
    'revenue',
    'Annual business revenue',
    'NUMERIC',
    'thousand USD',
    FALSE,
    'Finance Analytics'
),
(
    'debt_ratio',
    'Debt relative to assets',
    'NUMERIC',
    'ratio',
    FALSE,
    'Finance Analytics'
),
(
    'customer_retention',
    'Customer retention rate',
    'NUMERIC',
    'ratio',
    FALSE,
    'Customer Analytics'
),
(
    'protected_attribute',
    'Restricted demographic attribute',
    'NUMERIC',
    'category',
    TRUE,
    'Governance Office'
);

INSERT INTO model_features (
    version_id,
    feature_id,
    feature_position,
    included
)
SELECT
    mv.version_id,
    f.feature_id,
    CASE f.feature_name
        WHEN 'revenue' THEN 0
        WHEN 'debt_ratio' THEN 1
        WHEN 'customer_retention' THEN 2
    END,
    TRUE
FROM model_versions mv
CROSS JOIN features f
WHERE mv.version_number = '2.1.0'
  AND f.feature_name IN (
      'revenue',
      'debt_ratio',
      'customer_retention'
  );

INSERT INTO interpretation_policies (
    policy_name,
    minimum_samples,
    minimum_permutation_repeats,
    maximum_permutation_repeats,
    minimum_allowed_importance,
    allow_sensitive_features,
    require_test_dataset,
    require_approval
)
VALUES
(
    'Enterprise Standard',
    20,
    5,
    100,
    0.01,
    FALSE,
    TRUE,
    TRUE
),
(
    'Exploratory Analysis',
    10,
    3,
    50,
    0,
    FALSE,
    TRUE,
    FALSE
);

INSERT INTO evaluation_datasets (
    dataset_name,
    dataset_reference,
    dataset_hash,
    dataset_role,
    sample_count,
    feature_count,
    contains_sensitive_data
)
VALUES
(
    'Credit Risk Test Set',
    'credit_test_2026',
    'aabbccddeeff001122334455',
    'TEST',
    120,
    3,
    FALSE
),
(
    'Credit Risk Training Set',
    'credit_train_2026',
    '11223344556677889900aabb',
    'TRAIN',
    360,
    3,
    FALSE
);

-- ============================================================
-- 21. EXECUTION TRANSACTION
-- ============================================================

DO $$
DECLARE
    v_version BIGINT;
    v_dataset BIGINT;
    v_policy BIGINT;
    v_run BIGINT;
    v_revenue BIGINT;
    v_debt BIGINT;
    v_retention BIGINT;
BEGIN
    SELECT version_id
    INTO v_version
    FROM model_versions
    WHERE version_number = '2.1.0';

    SELECT dataset_id
    INTO v_dataset
    FROM evaluation_datasets
    WHERE dataset_name = 'Credit Risk Test Set';

    SELECT policy_id
    INTO v_policy
    FROM interpretation_policies
    WHERE policy_name = 'Enterprise Standard';

    SELECT feature_id INTO v_revenue
    FROM features WHERE feature_name = 'revenue';

    SELECT feature_id INTO v_debt
    FROM features WHERE feature_name = 'debt_ratio';

    SELECT feature_id INTO v_retention
    FROM features WHERE feature_name = 'customer_retention';

    v_run := start_interpretation(
        v_version,
        v_dataset,
        v_policy,
        'risk_analyst',
        15,
        42
    );

    -- Coefficient-based importance.
    INSERT INTO feature_importance_results (
        run_id,
        feature_id,
        method,
        importance_score,
        normalized_score,
        experiment_count
    )
    VALUES
    (
        v_run,
        v_revenue,
        'COEFFICIENT',
        28.50,
        0.45,
        1
    ),
    (
        v_run,
        v_debt,
        'COEFFICIENT',
        25.00,
        0.39,
        1
    ),
    (
        v_run,
        v_retention,
        'COEFFICIENT',
        10.10,
        0.16,
        1
    );

    -- Permutation importance represents the increase in MSE.
    INSERT INTO feature_importance_results (
        run_id,
        feature_id,
        method,
        importance_score,
        normalized_score,
        standard_deviation,
        experiment_count
    )
    VALUES
    (
        v_run,
        v_revenue,
        'PERMUTATION',
        142.40,
        0.48,
        9.50,
        15
    ),
    (
        v_run,
        v_debt,
        'PERMUTATION',
        105.20,
        0.35,
        7.20,
        15
    ),
    (
        v_run,
        v_retention,
        'PERMUTATION',
        50.50,
        0.17,
        5.10,
        15
    );

    -- Partial dependence observations.
    INSERT INTO partial_dependence_results (
        run_id,
        feature_id,
        grid_position,
        feature_value,
        average_prediction
    )
    VALUES
    (v_run, v_revenue, 0, 100, 10),
    (v_run, v_revenue, 1, 300, 17),
    (v_run, v_revenue, 2, 500, 24),
    (v_run, v_revenue, 3, 700, 31),
    (v_run, v_revenue, 4, 900, 38),

    (v_run, v_debt, 0, 0.10, 45),
    (v_run, v_debt, 1, 0.30, 21),
    (v_run, v_debt, 2, 0.50, -3),
    (v_run, v_debt, 3, 0.70, -27),
    (v_run, v_debt, 4, 0.90, -51),

    (v_run, v_retention, 0, 0.40, 10),
    (v_run, v_retention, 1, 0.55, 16),
    (v_run, v_retention, 2, 0.70, 23),
    (v_run, v_retention, 3, 0.85, 30),
    (v_run, v_retention, 4, 0.99, 36);

    complete_interpretation(
        v_run,
        125.0,
        8.2,
        sqrt(125.0)
    );

    INSERT INTO interpretation_approvals (
        run_id,
        reviewer,
        decision,
        comments,
        decided_at
    )
    VALUES (
        v_run,
        'governance_reviewer',
        'APPROVED',
        'Interpretation results reviewed for methodological consistency.',
        CURRENT_TIMESTAMP
    );

    RAISE NOTICE 'Interpretation run completed: %', v_run;
END;
$$;

COMMIT;

-- ============================================================
-- 22. RETRIEVAL QUERIES
-- ============================================================

-- Retrieve registered models and their versions.
SELECT
    mr.model_name,
    mr.model_family,
    mv.version_number,
    mv.artifact_hash
FROM model_registry mr
JOIN model_versions mv
    ON mv.model_id = mr.model_id
ORDER BY mr.model_name, mv.created_at DESC;

-- Retrieve features included in a particular model version.
SELECT
    mv.version_number,
    f.feature_name,
    f.description,
    f.unit,
    f.sensitive,
    mf.feature_position
FROM model_features mf
JOIN model_versions mv
    ON mv.version_id = mf.version_id
JOIN features f
    ON f.feature_id = mf.feature_id
WHERE mv.version_number = '2.1.0'
  AND mf.included = TRUE
ORDER BY mf.feature_position;

-- ============================================================
-- 23. VALIDATION QUERIES
-- ============================================================

-- Identify model versions with missing feature metadata.
SELECT
    mv.version_id,
    mv.version_number,
    count(mf.feature_id) AS registered_features
FROM model_versions mv
LEFT JOIN model_features mf
    ON mf.version_id = mv.version_id
   AND mf.included = TRUE
GROUP BY mv.version_id, mv.version_number
HAVING count(mf.feature_id) = 0;

-- Detect importance scores outside the expected reporting range.
SELECT *
FROM feature_importance_results
WHERE normalized_score < 0
   OR normalized_score > 1
   OR importance_score = 'NaN'::DOUBLE PRECISION;

-- ============================================================
-- 24. IMPORTANCE AGGREGATION
-- ============================================================

-- Compare coefficient and permutation results independently.
SELECT
    model_name,
    version_number,
    feature_name,
    method,
    importance_score,
    normalized_score,
    standard_deviation
FROM feature_importance_report
ORDER BY method, importance_score DESC;

-- Compare methods using conditional aggregation.
SELECT
    model_name,
    version_number,
    feature_name,

    max(normalized_score) FILTER (
        WHERE method = 'COEFFICIENT'
    ) AS coefficient_importance,

    max(normalized_score) FILTER (
        WHERE method = 'PERMUTATION'
    ) AS permutation_importance

FROM feature_importance_report

GROUP BY
    model_name,
    version_number,
    feature_name

ORDER BY permutation_importance DESC NULLS LAST;

-- ============================================================
-- 25. PARTIAL DEPENDENCE EVALUATION
-- ============================================================

SELECT
    model_name,
    feature_name,
    grid_position,
    feature_value,
    average_prediction,

    average_prediction - lag(average_prediction) OVER (
        PARTITION BY run_id, feature_name
        ORDER BY grid_position
    ) AS prediction_change

FROM partial_dependence_report
ORDER BY run_id, feature_name, grid_position;

-- ============================================================
-- 26. CTE-BASED GOVERNANCE REPORT
-- ============================================================

WITH importance_totals AS (
    SELECT
        run_id,
        method,
        sum(greatest(importance_score, 0)) AS total_positive_importance
    FROM feature_importance_results
    GROUP BY run_id, method
),

importance_checks AS (
    SELECT
        fir.run_id,
        fir.method,
        count(*) AS feature_count,
        sum(fir.normalized_score) AS normalized_total,
        it.total_positive_importance
    FROM feature_importance_results fir
    JOIN importance_totals it
        ON it.run_id = fir.run_id
       AND it.method = fir.method
    GROUP BY
        fir.run_id,
        fir.method,
        it.total_positive_importance
)

SELECT
    run_id,
    method,
    feature_count,
    normalized_total,
    total_positive_importance,
    abs(normalized_total - 1.0) < 0.0001 AS normalization_valid
FROM importance_checks
ORDER BY run_id, method;

-- ============================================================
-- 27. MERGE-LIKE INTERPRETATION ACCEPTANCE ELIGIBILITY
-- ============================================================

SELECT
    e.run_id,
    e.execution_status,
    e.require_approval,
    e.approved_count,
    e.rejected_count,
    e.eligible_for_acceptance
FROM interpretation_eligibility e
ORDER BY e.run_id;

-- ============================================================
-- 28. WORKFLOW AND FAILURE REPORTING
-- ============================================================

SELECT
    run_id,
    execution_status,
    requested_by,
    sample_count,
    started_at,
    completed_at,
    failure_reason
FROM interpretation_runs
ORDER BY created_at DESC;

-- Find failed runs.
SELECT
    run_id,
    requested_by,
    failure_reason,
    created_at
FROM interpretation_runs
WHERE execution_status = 'FAILED';

-- ============================================================
-- 29. AUDIT REPORTING
-- ============================================================

SELECT
    audit_id,
    run_id,
    event_type,
    actor,
    event_details,
    event_timestamp
FROM interpretation_audit
ORDER BY event_timestamp DESC;

-- ============================================================
-- 30. DEMONSTRATION OF TRANSACTION ROLLBACK
-- ============================================================

-- An invalid repeat count is rejected by the run-validation trigger.
-- The exception is contained in a savepoint, preserving the transaction.

BEGIN;

SAVEPOINT invalid_analysis_attempt;

DO $$
DECLARE
    v_version BIGINT;
    v_dataset BIGINT;
    v_policy BIGINT;
    v_samples INTEGER;
BEGIN
    SELECT version_id
    INTO v_version
    FROM model_versions
    WHERE version_number = '2.1.0';

    SELECT dataset_id, sample_count
    INTO v_dataset, v_samples
    FROM evaluation_datasets
    WHERE dataset_name = 'Credit Risk Test Set';

    SELECT policy_id
    INTO v_policy
    FROM interpretation_policies
    WHERE policy_name = 'Enterprise Standard';

    BEGIN
        INSERT INTO interpretation_runs (
            version_id,
            dataset_id,
            policy_id,
            requested_by,
            sample_count,
            permutation_repeats,
            execution_status
        )
        VALUES (
            v_version,
            v_dataset,
            v_policy,
            'invalid_test_operator',
            v_samples,
            1000,
            'RUNNING'
        );

    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Expected validation failure: %', SQLERRM;
    END;
END;
$$;

RELEASE SAVEPOINT invalid_analysis_attempt;

COMMIT;

-- ============================================================
-- 31. OPERATIONAL REPORT
-- ============================================================

SELECT
    mr.model_name,
    mv.version_number,
    count(DISTINCT r.run_id) AS total_runs,

    count(DISTINCT r.run_id) FILTER (
        WHERE r.execution_status = 'COMPLETED'
    ) AS completed_runs,

    count(DISTINCT r.run_id) FILTER (
        WHERE r.execution_status = 'FAILED'
    ) AS failed_runs,

    round(
        avg(r.baseline_rmse)::NUMERIC,
        4
    ) AS average_rmse

FROM model_registry mr

JOIN model_versions mv
    ON mv.model_id = mr.model_id

LEFT JOIN interpretation_runs r
    ON r.version_id = mv.version_id

GROUP BY
    mr.model_name,
    mv.version_number

ORDER BY mr.model_name, mv.version_number;
