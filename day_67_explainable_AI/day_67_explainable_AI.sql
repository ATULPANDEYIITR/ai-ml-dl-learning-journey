-- ============================================================
-- EXPLAINABLE AI: SHAP, LIME AND GLOBAL EXPLANATION DATABASE
-- PostgreSQL 15+
--
-- Models:
-- 1. Prediction records
-- 2. Model versions
-- 3. Feature definitions and observations
-- 4. SHAP local attributions
-- 5. LIME local surrogate explanations
-- 6. Global feature importance
-- 7. Explanation policies
-- 8. Explanation validation and audit
--
-- SHAP and LIME values are stored separately because their
-- semantics and interpretation are different.
-- ============================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS explainability;

SET search_path TO explainability, public;

-- ------------------------------------------------------------
-- ENUM TYPES
-- ------------------------------------------------------------

CREATE TYPE explanation_method AS ENUM (
    'SHAP',
    'LIME',
    'GLOBAL'
);

CREATE TYPE explanation_status AS ENUM (
    'PENDING',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'REJECTED'
);

CREATE TYPE explanation_scope AS ENUM (
    'LOCAL',
    'GLOBAL'
);

CREATE TYPE prediction_decision AS ENUM (
    'APPROVED',
    'REVIEW_REQUIRED',
    'REJECTED'
);

CREATE TYPE audit_event_type AS ENUM (
    'PREDICTION_CREATED',
    'EXPLANATION_STARTED',
    'EXPLANATION_COMPLETED',
    'EXPLANATION_FAILED',
    'VALIDATION_FAILED',
    'POLICY_CHANGED'
);

-- ------------------------------------------------------------
-- MODEL REGISTRY
-- ------------------------------------------------------------

CREATE TABLE model_versions (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name VARCHAR(150) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_description TEXT NOT NULL,
    model_type VARCHAR(80) NOT NULL,
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (model_name, version)
);

CREATE TABLE feature_definitions (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES model_versions(model_id),
    feature_name VARCHAR(100) NOT NULL,
    data_type VARCHAR(30) NOT NULL,
    minimum_value NUMERIC(20, 8) NOT NULL,
    maximum_value NUMERIC(20, 8) NOT NULL,
    baseline_value NUMERIC(20, 8) NOT NULL,
    description TEXT NOT NULL,

    CHECK (minimum_value < maximum_value),

    CHECK (
        baseline_value >= minimum_value
        AND baseline_value <= maximum_value
    ),

    UNIQUE (model_id, feature_name),
    UNIQUE (model_id, feature_id)
);

-- ------------------------------------------------------------
-- PREDICTION AND OBSERVATION RECORDS
-- ------------------------------------------------------------

CREATE TABLE predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES model_versions(model_id),
    subject_reference VARCHAR(150) NOT NULL,
    predicted_probability NUMERIC(12, 10) NOT NULL,
    threshold NUMERIC(12, 10) NOT NULL DEFAULT 0.5,
    decision prediction_decision NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (
        predicted_probability >= 0
        AND predicted_probability <= 1
    ),

    CHECK (threshold > 0 AND threshold < 1),

    UNIQUE (prediction_id, model_id)
);

CREATE INDEX idx_predictions_model_date
ON predictions(model_id, predicted_at DESC);

CREATE INDEX idx_predictions_decision
ON predictions(decision, predicted_at DESC);

CREATE TABLE prediction_features (
    prediction_id BIGINT NOT NULL,
    model_id BIGINT NOT NULL,
    feature_id BIGINT NOT NULL,
    feature_value NUMERIC(20, 8) NOT NULL,

    PRIMARY KEY (prediction_id, feature_id),

    FOREIGN KEY (prediction_id, model_id)
        REFERENCES predictions(prediction_id, model_id),

    FOREIGN KEY (model_id, feature_id)
        REFERENCES feature_definitions(model_id, feature_id)
);

CREATE INDEX idx_prediction_features_feature
ON prediction_features(feature_id, prediction_id);

-- ------------------------------------------------------------
-- BACKGROUND DATASET
-- ------------------------------------------------------------

CREATE TABLE background_datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES model_versions(model_id),
    dataset_name VARCHAR(150) NOT NULL,
    dataset_version VARCHAR(50) NOT NULL,
    observation_count INTEGER NOT NULL
        CHECK (observation_count > 0),
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (model_id, dataset_name, dataset_version),
    UNIQUE (dataset_id, model_id)
);

CREATE TABLE background_observations (
    background_observation_id BIGINT
        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL,
    model_id BIGINT NOT NULL,
    observation_reference VARCHAR(150) NOT NULL,

    FOREIGN KEY (dataset_id, model_id)
        REFERENCES background_datasets(dataset_id, model_id),

    UNIQUE (dataset_id, observation_reference),
    UNIQUE (background_observation_id, model_id)
);

CREATE TABLE background_features (
    background_observation_id BIGINT NOT NULL,
    model_id BIGINT NOT NULL,
    feature_id BIGINT NOT NULL,
    feature_value NUMERIC(20, 8) NOT NULL,

    PRIMARY KEY (background_observation_id, feature_id),

    FOREIGN KEY (background_observation_id, model_id)
        REFERENCES background_observations(
            background_observation_id,
            model_id
        ),

    FOREIGN KEY (model_id, feature_id)
        REFERENCES feature_definitions(model_id, feature_id)
);

-- ------------------------------------------------------------
-- EXPLANATION POLICIES
-- ------------------------------------------------------------

CREATE TABLE explanation_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES model_versions(model_id),
    policy_name VARCHAR(150) NOT NULL,
    minimum_background_samples INTEGER NOT NULL DEFAULT 1,
    maximum_exact_shap_features INTEGER NOT NULL DEFAULT 15,
    minimum_lime_perturbations INTEGER NOT NULL DEFAULT 100,
    maximum_lime_perturbations INTEGER NOT NULL DEFAULT 100000,
    additivity_tolerance NUMERIC(15, 12) NOT NULL DEFAULT 0.000001,
    require_validation BOOLEAN NOT NULL DEFAULT TRUE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (minimum_background_samples > 0),

    CHECK (
        maximum_exact_shap_features BETWEEN 1 AND 20
    ),

    CHECK (
        minimum_lime_perturbations >= 20
        AND maximum_lime_perturbations >= minimum_lime_perturbations
    ),

    CHECK (additivity_tolerance > 0),

    UNIQUE (model_id, policy_name),
    UNIQUE (policy_id, model_id)
);

CREATE UNIQUE INDEX idx_one_active_policy_per_model
ON explanation_policies(model_id)
WHERE active = TRUE;

-- ------------------------------------------------------------
-- EXPLANATION EXECUTIONS
-- ------------------------------------------------------------

CREATE TABLE explanations (
    explanation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id BIGINT NOT NULL,
    model_id BIGINT NOT NULL,
    policy_id BIGINT NOT NULL,
    method explanation_method NOT NULL,
    scope explanation_scope NOT NULL,
    status explanation_status NOT NULL DEFAULT 'PENDING',

    baseline_value NUMERIC(15, 12),
    reconstructed_value NUMERIC(15, 12),
    model_output NUMERIC(15, 12),

    perturbation_count INTEGER,
    kernel_width NUMERIC(12, 8),
    random_seed BIGINT,

    algorithm_version VARCHAR(50) NOT NULL,
    execution_duration_ms NUMERIC(15, 3),
    failure_reason TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    FOREIGN KEY (prediction_id, model_id)
        REFERENCES predictions(prediction_id, model_id),

    FOREIGN KEY (policy_id, model_id)
        REFERENCES explanation_policies(policy_id, model_id),

    CHECK (
        perturbation_count IS NULL
        OR perturbation_count >= 0
    ),

    CHECK (
        kernel_width IS NULL
        OR kernel_width > 0
    ),

    CHECK (
        execution_duration_ms IS NULL
        OR execution_duration_ms >= 0
    ),

    CHECK (
        (scope = 'LOCAL' AND method IN ('SHAP', 'LIME'))
        OR
        (scope = 'GLOBAL' AND method = 'GLOBAL')
    ),

    CHECK (
        status <> 'COMPLETED'
        OR (
            baseline_value IS NOT NULL
            AND model_output IS NOT NULL
            AND completed_at IS NOT NULL
        )
    ),

    CHECK (
        status <> 'FAILED'
        OR failure_reason IS NOT NULL
    ),

    UNIQUE (explanation_id, model_id)
);

CREATE INDEX idx_explanations_prediction
ON explanations(prediction_id, method, created_at DESC);

CREATE INDEX idx_explanations_status
ON explanations(status, created_at)
WHERE status IN ('PENDING', 'RUNNING');

-- ------------------------------------------------------------
-- LOCAL FEATURE ATTRIBUTIONS
-- ------------------------------------------------------------

CREATE TABLE feature_attributions (
    attribution_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    explanation_id BIGINT NOT NULL
        REFERENCES explanations(explanation_id),
    feature_id BIGINT NOT NULL
        REFERENCES feature_definitions(feature_id),
    attribution_value NUMERIC(20, 12) NOT NULL,
    feature_value NUMERIC(20, 8) NOT NULL,
    attribution_rank INTEGER NOT NULL CHECK (attribution_rank > 0),
    direction VARCHAR(20) NOT NULL,

    CHECK (direction IN ('POSITIVE', 'NEGATIVE', 'ZERO')),

    UNIQUE (explanation_id, feature_id),
    UNIQUE (explanation_id, attribution_rank)
);

CREATE INDEX idx_attributions_feature
ON feature_attributions(feature_id, attribution_value);

CREATE INDEX idx_attributions_explanation_rank
ON feature_attributions(explanation_id, attribution_rank);

-- ------------------------------------------------------------
-- GLOBAL FEATURE IMPORTANCE
-- ------------------------------------------------------------

CREATE TABLE global_importance (
    importance_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL
        REFERENCES model_versions(model_id),
    explanation_id BIGINT NOT NULL
        REFERENCES explanations(explanation_id),
    feature_id BIGINT NOT NULL
        REFERENCES feature_definitions(feature_id),
    dataset_id BIGINT NOT NULL
        REFERENCES background_datasets(dataset_id),
    mean_absolute_shap NUMERIC(20, 12) NOT NULL,
    importance_rank INTEGER NOT NULL CHECK (importance_rank > 0),
    evaluated_sample_count INTEGER NOT NULL
        CHECK (evaluated_sample_count > 0),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (mean_absolute_shap >= 0),

    UNIQUE (explanation_id, feature_id),
    UNIQUE (explanation_id, importance_rank)
);

CREATE INDEX idx_global_importance_ranking
ON global_importance(
    model_id,
    calculated_at DESC,
    importance_rank
);

-- ------------------------------------------------------------
-- EXPLANATION VALIDATION
-- ------------------------------------------------------------

CREATE TABLE explanation_validations (
    validation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    explanation_id BIGINT NOT NULL
        REFERENCES explanations(explanation_id),
    validation_name VARCHAR(120) NOT NULL,
    passed BOOLEAN NOT NULL,
    observed_value NUMERIC(20, 12),
    expected_value NUMERIC(20, 12),
    tolerance NUMERIC(20, 12),
    validation_message TEXT NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (explanation_id, validation_name)
);

CREATE INDEX idx_validation_failures
ON explanation_validations(validated_at DESC)
WHERE passed = FALSE;

-- ------------------------------------------------------------
-- AUDIT LOG
-- ------------------------------------------------------------

CREATE TABLE explanation_audit (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    explanation_id BIGINT
        REFERENCES explanations(explanation_id),
    prediction_id BIGINT
        REFERENCES predictions(prediction_id),
    event_type audit_event_type NOT NULL,
    actor VARCHAR(120) NOT NULL,
    event_details JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_explanation_date
ON explanation_audit(explanation_id, created_at DESC);

CREATE INDEX idx_audit_event_type
ON explanation_audit(event_type, created_at DESC);

-- ------------------------------------------------------------
-- FEATURE VALIDATION TRIGGER
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_prediction_feature()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    feature_record feature_definitions%ROWTYPE;
BEGIN
    SELECT *
    INTO feature_record
    FROM feature_definitions
    WHERE feature_id = NEW.feature_id
      AND model_id = NEW.model_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Feature does not belong to prediction model';
    END IF;

    IF NEW.feature_value <
        feature_record.minimum_value
       OR NEW.feature_value >
        feature_record.maximum_value THEN

        RAISE EXCEPTION
            'Feature % value % outside permitted range',
            feature_record.feature_name,
            NEW.feature_value;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_prediction_feature
BEFORE INSERT OR UPDATE
ON prediction_features
FOR EACH ROW
EXECUTE FUNCTION validate_prediction_feature();

-- ------------------------------------------------------------
-- BACKGROUND FEATURE VALIDATION TRIGGER
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_background_feature()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    feature_record feature_definitions%ROWTYPE;
BEGIN
    SELECT *
    INTO feature_record
    FROM feature_definitions
    WHERE feature_id = NEW.feature_id
      AND model_id = NEW.model_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Background feature does not belong to model';
    END IF;

    IF NEW.feature_value <
        feature_record.minimum_value
       OR NEW.feature_value >
        feature_record.maximum_value THEN

        RAISE EXCEPTION
            'Invalid background feature value for %',
            feature_record.feature_name;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_background_feature
BEFORE INSERT OR UPDATE
ON background_features
FOR EACH ROW
EXECUTE FUNCTION validate_background_feature();

-- ------------------------------------------------------------
-- ATTRIBUTION DIRECTION VALIDATION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_attribution_direction()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.attribution_value > 0
       AND NEW.direction <> 'POSITIVE' THEN

        RAISE EXCEPTION
            'Positive attribution must have POSITIVE direction';
    END IF;

    IF NEW.attribution_value < 0
       AND NEW.direction <> 'NEGATIVE' THEN

        RAISE EXCEPTION
            'Negative attribution must have NEGATIVE direction';
    END IF;

    IF NEW.attribution_value = 0
       AND NEW.direction <> 'ZERO' THEN

        RAISE EXCEPTION
            'Zero attribution must have ZERO direction';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_attribution_direction
BEFORE INSERT OR UPDATE
ON feature_attributions
FOR EACH ROW
EXECUTE FUNCTION validate_attribution_direction();

-- ------------------------------------------------------------
-- EXPLANATION WORKFLOW TRANSITIONS
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION enforce_explanation_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IN ('COMPLETED', 'FAILED', 'REJECTED')
       AND NEW.status <> OLD.status THEN

        RAISE EXCEPTION
            'Terminal explanation status cannot be changed';
    END IF;

    IF OLD.status = 'PENDING'
       AND NEW.status NOT IN (
           'PENDING', 'RUNNING', 'REJECTED', 'FAILED'
       ) THEN

        RAISE EXCEPTION 'Invalid PENDING transition';
    END IF;

    IF OLD.status = 'RUNNING'
       AND NEW.status NOT IN (
           'RUNNING', 'COMPLETED', 'FAILED'
       ) THEN

        RAISE EXCEPTION 'Invalid RUNNING transition';
    END IF;

    IF NEW.status = 'COMPLETED'
       AND NEW.completed_at IS NULL THEN

        RAISE EXCEPTION
            'Completed explanation requires completion time';
    END IF;

    IF NEW.status = 'FAILED'
       AND NEW.failure_reason IS NULL THEN

        RAISE EXCEPTION
            'Failed explanation requires failure reason';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_explanation_transition
BEFORE UPDATE ON explanations
FOR EACH ROW
EXECUTE FUNCTION enforce_explanation_transition();

-- ------------------------------------------------------------
-- SHAP ADDITIVITY VALIDATION FUNCTION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION validate_shap_additivity(
    p_explanation_id BIGINT
)
RETURNS TABLE (
    baseline NUMERIC,
    total_contribution NUMERIC,
    reconstructed NUMERIC,
    model_output NUMERIC,
    additivity_error NUMERIC,
    is_valid BOOLEAN
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        e.baseline_value,
        COALESCE(SUM(a.attribution_value), 0),
        e.baseline_value +
            COALESCE(SUM(a.attribution_value), 0),
        e.model_output,
        ABS(
            e.model_output -
            (
                e.baseline_value +
                COALESCE(SUM(a.attribution_value), 0)
            )
        ),
        ABS(
            e.model_output -
            (
                e.baseline_value +
                COALESCE(SUM(a.attribution_value), 0)
            )
        ) <= p.additivity_tolerance
    FROM explanations e
    JOIN explanation_policies p
      ON p.policy_id = e.policy_id
    LEFT JOIN feature_attributions a
      ON a.explanation_id = e.explanation_id
    WHERE e.explanation_id = p_explanation_id
      AND e.method = 'SHAP'
    GROUP BY
        e.baseline_value,
        e.model_output,
        p.additivity_tolerance;
$$;

-- ------------------------------------------------------------
-- EXPLANATION ELIGIBILITY FUNCTION
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION check_explanation_eligibility(
    p_prediction_id BIGINT,
    p_method explanation_method
)
RETURNS TABLE (
    eligible BOOLEAN,
    reason TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    prediction_record predictions%ROWTYPE;
    policy_record explanation_policies%ROWTYPE;
    feature_total INTEGER;
    background_total INTEGER;
BEGIN
    SELECT *
    INTO prediction_record
    FROM predictions
    WHERE prediction_id = p_prediction_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'Prediction does not exist';
        RETURN;
    END IF;

    SELECT *
    INTO policy_record
    FROM explanation_policies
    WHERE model_id = prediction_record.model_id
      AND active = TRUE;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT FALSE, 'No active explanation policy';
        RETURN;
    END IF;

    SELECT COUNT(*)
    INTO feature_total
    FROM feature_definitions
    WHERE model_id = prediction_record.model_id;

    IF p_method = 'SHAP'
       AND feature_total >
           policy_record.maximum_exact_shap_features THEN

        RETURN QUERY
        SELECT FALSE, 'Feature count exceeds exact SHAP policy';
        RETURN;
    END IF;

    SELECT COUNT(*)
    INTO background_total
    FROM background_datasets
    WHERE model_id = prediction_record.model_id
      AND observation_count >=
          policy_record.minimum_background_samples;

    IF background_total = 0 THEN
        RETURN QUERY
        SELECT FALSE, 'Insufficient background dataset';
        RETURN;
    END IF;

    RETURN QUERY
    SELECT TRUE, 'Explanation eligible';
END;
$$;

-- ------------------------------------------------------------
-- GLOBAL IMPORTANCE AGGREGATION VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW v_global_feature_importance AS
SELECT
    gi.model_id,
    mv.model_name,
    mv.version,
    fd.feature_name,
    AVG(gi.mean_absolute_shap) AS average_importance,
    MIN(gi.importance_rank) AS best_observed_rank,
    MAX(gi.importance_rank) AS worst_observed_rank,
    SUM(gi.evaluated_sample_count) AS total_evaluated_samples
FROM global_importance gi
JOIN model_versions mv
    ON mv.model_id = gi.model_id
JOIN feature_definitions fd
    ON fd.feature_id = gi.feature_id
GROUP BY
    gi.model_id,
    mv.model_name,
    mv.version,
    fd.feature_name;

-- ------------------------------------------------------------
-- LOCAL EXPLANATION REPORT VIEW
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW v_local_explanation_report AS
SELECT
    e.explanation_id,
    e.prediction_id,
    e.method,
    e.status,
    mv.model_name,
    mv.version,
    p.subject_reference,
    p.predicted_probability,
    fd.feature_name,
    a.feature_value,
    a.attribution_value,
    a.attribution_rank,
    a.direction,
    e.created_at
FROM explanations e
JOIN predictions p
    ON p.prediction_id = e.prediction_id
JOIN model_versions mv
    ON mv.model_id = e.model_id
LEFT JOIN feature_attributions a
    ON a.explanation_id = e.explanation_id
LEFT JOIN feature_definitions fd
    ON fd.feature_id = a.feature_id
WHERE e.scope = 'LOCAL';

-- ------------------------------------------------------------
-- REPORTING: LOCAL ATTRIBUTION RETRIEVAL
-- ------------------------------------------------------------

-- Example:
-- SELECT *
-- FROM v_local_explanation_report
-- WHERE prediction_id = 1
-- ORDER BY explanation_id, attribution_rank;

-- ------------------------------------------------------------
-- REPORTING: GLOBAL FEATURE RANKING
-- ------------------------------------------------------------

-- Example:
-- SELECT *
-- FROM v_global_feature_importance
-- WHERE model_id = 1
-- ORDER BY average_importance DESC;

-- ------------------------------------------------------------
-- REPORTING: FAILED VALIDATIONS
-- ------------------------------------------------------------

-- Example:
-- SELECT *
-- FROM explanation_validations
-- WHERE passed = FALSE
-- ORDER BY validated_at DESC;

-- ------------------------------------------------------------
-- REPORTING: EXPLANATION METHOD COMPARISON
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW v_explanation_method_metrics AS
SELECT
    model_id,
    method,
    COUNT(*) AS execution_count,
    COUNT(*) FILTER (
        WHERE status = 'COMPLETED'
    ) AS completed_count,
    COUNT(*) FILTER (
        WHERE status = 'FAILED'
    ) AS failed_count,
    AVG(execution_duration_ms) AS average_duration_ms
FROM explanations
GROUP BY model_id, method;

-- ------------------------------------------------------------
-- SAMPLE DATA
-- ------------------------------------------------------------

INSERT INTO model_versions (
    model_name,
    version,
    model_description,
    model_type,
    feature_count
)
VALUES (
    'Credit Risk Classifier',
    '1.0.0',
    'Binary logistic credit-risk demonstration model',
    'LOGISTIC_CLASSIFICATION',
    5
);

INSERT INTO feature_definitions (
    model_id,
    feature_name,
    data_type,
    minimum_value,
    maximum_value,
    baseline_value,
    description
)
SELECT
    m.model_id,
    f.feature_name,
    'NUMERIC',
    f.minimum_value,
    f.maximum_value,
    f.baseline_value,
    f.description
FROM model_versions m
CROSS JOIN (
    VALUES
    (
        'income',
        10000::NUMERIC,
        200000::NUMERIC,
        60000::NUMERIC,
        'Annual income'
    ),
    (
        'debt_ratio',
        0::NUMERIC,
        1::NUMERIC,
        0.35::NUMERIC,
        'Debt relative to income'
    ),
    (
        'credit_history',
        0::NUMERIC,
        30::NUMERIC,
        8::NUMERIC,
        'Credit history in years'
    ),
    (
        'missed_payments',
        0::NUMERIC,
        20::NUMERIC,
        1::NUMERIC,
        'Number of missed payments'
    ),
    (
        'employment_years',
        0::NUMERIC,
        40::NUMERIC,
        5::NUMERIC,
        'Employment duration'
    )
) AS f(
    feature_name,
    minimum_value,
    maximum_value,
    baseline_value,
    description
)
WHERE m.model_name = 'Credit Risk Classifier'
AND m.version = '1.0.0';

INSERT INTO explanation_policies (
    model_id,
    policy_name,
    minimum_background_samples,
    maximum_exact_shap_features,
    minimum_lime_perturbations,
    maximum_lime_perturbations,
    additivity_tolerance,
    require_validation
)
SELECT
    model_id,
    'STANDARD_EXPLANATION_POLICY',
    3,
    15,
    100,
    100000,
    0.000001,
    TRUE
FROM model_versions
WHERE model_name = 'Credit Risk Classifier';

INSERT INTO background_datasets (
    model_id,
    dataset_name,
    dataset_version,
    observation_count,
    description
)
SELECT
    model_id,
    'CREDIT_RISK_REFERENCE',
    '1.0',
    4,
    'Reference observations for interventional explanations'
FROM model_versions
WHERE model_name = 'Credit Risk Classifier';

INSERT INTO background_observations (
    dataset_id,
    model_id,
    observation_reference
)
SELECT
    d.dataset_id,
    d.model_id,
    'REFERENCE_' || series_number
FROM background_datasets d
CROSS JOIN generate_series(1, 4) AS series_number
WHERE d.dataset_name = 'CREDIT_RISK_REFERENCE';

INSERT INTO background_features (
    background_observation_id,
    model_id,
    feature_id,
    feature_value
)
SELECT
    bo.background_observation_id,
    bo.model_id,
    fd.feature_id,
    values_table.feature_value
FROM background_observations bo
JOIN feature_definitions fd
    ON fd.model_id = bo.model_id
JOIN (
    VALUES
    ('REFERENCE_1', 'income', 45000::NUMERIC),
    ('REFERENCE_1', 'debt_ratio', 0.55::NUMERIC),
    ('REFERENCE_1', 'credit_history', 3::NUMERIC),
    ('REFERENCE_1', 'missed_payments', 4::NUMERIC),
    ('REFERENCE_1', 'employment_years', 2::NUMERIC),

    ('REFERENCE_2', 'income', 70000::NUMERIC),
    ('REFERENCE_2', 'debt_ratio', 0.25::NUMERIC),
    ('REFERENCE_2', 'credit_history', 12::NUMERIC),
    ('REFERENCE_2', 'missed_payments', 0::NUMERIC),
    ('REFERENCE_2', 'employment_years', 10::NUMERIC),

    ('REFERENCE_3', 'income', 55000::NUMERIC),
    ('REFERENCE_3', 'debt_ratio', 0.40::NUMERIC),
    ('REFERENCE_3', 'credit_history', 7::NUMERIC),
    ('REFERENCE_3', 'missed_payments', 2::NUMERIC),
    ('REFERENCE_3', 'employment_years', 5::NUMERIC),

    ('REFERENCE_4', 'income', 95000::NUMERIC),
    ('REFERENCE_4', 'debt_ratio', 0.20::NUMERIC),
    ('REFERENCE_4', 'credit_history', 18::NUMERIC),
    ('REFERENCE_4', 'missed_payments', 0::NUMERIC),
    ('REFERENCE_4', 'employment_years', 15::NUMERIC)
) AS values_table(
    observation_reference,
    feature_name,
    feature_value
)
    ON values_table.observation_reference =
        bo.observation_reference
   AND values_table.feature_name = fd.feature_name;

INSERT INTO predictions (
    model_id,
    subject_reference,
    predicted_probability,
    threshold,
    decision
)
SELECT
    model_id,
    'APPLICANT_001',
    0.8500000000,
    0.5000000000,
    'APPROVED'
FROM model_versions
WHERE model_name = 'Credit Risk Classifier';

INSERT INTO prediction_features (
    prediction_id,
    model_id,
    feature_id,
    feature_value
)
SELECT
    p.prediction_id,
    p.model_id,
    fd.feature_id,
    values_table.feature_value
FROM predictions p
JOIN feature_definitions fd
    ON fd.model_id = p.model_id
JOIN (
    VALUES
    ('income', 85000::NUMERIC),
    ('debt_ratio', 0.30::NUMERIC),
    ('credit_history', 14::NUMERIC),
    ('missed_payments', 1::NUMERIC),
    ('employment_years', 9::NUMERIC)
) AS values_table(feature_name, feature_value)
    ON values_table.feature_name = fd.feature_name
WHERE p.subject_reference = 'APPLICANT_001';

-- A separate case illustrates a review-required decision.
INSERT INTO predictions (
    model_id,
    subject_reference,
    predicted_probability,
    threshold,
    decision
)
SELECT
    model_id,
    'APPLICANT_002',
    0.3500000000,
    0.5000000000,
    'REVIEW_REQUIRED'
FROM model_versions
WHERE model_name = 'Credit Risk Classifier';

-- ------------------------------------------------------------
-- EXPLANATION EXECUTION RECORDS
-- ------------------------------------------------------------

INSERT INTO explanations (
    prediction_id,
    model_id,
    policy_id,
    method,
    scope,
    status,
    baseline_value,
    reconstructed_value,
    model_output,
    algorithm_version,
    execution_duration_ms,
    completed_at
)
SELECT
    p.prediction_id,
    p.model_id,
    ep.policy_id,
    'SHAP',
    'LOCAL',
    'COMPLETED',
    0.650000000000,
    0.850000000000,
    0.850000000000,
    'EXACT_SHAP_1.0',
    12.500,
    NOW()
FROM predictions p
JOIN explanation_policies ep
    ON ep.model_id = p.model_id
WHERE p.subject_reference = 'APPLICANT_001';

INSERT INTO explanations (
    prediction_id,
    model_id,
    policy_id,
    method,
    scope,
    status,
    baseline_value,
    reconstructed_value,
    model_output,
    perturbation_count,
    kernel_width,
    random_seed,
    algorithm_version,
    execution_duration_ms,
    completed_at
)
SELECT
    p.prediction_id,
    p.model_id,
    ep.policy_id,
    'LIME',
    'LOCAL',
    'COMPLETED',
    0.720000000000,
    0.840000000000,
    0.850000000000,
    1500,
    0.75,
    42,
    'LIME_LOCAL_1.0',
    18.250,
    NOW()
FROM predictions p
JOIN explanation_policies ep
    ON ep.model_id = p.model_id
WHERE p.subject_reference = 'APPLICANT_001';

-- A failure case is retained for operational reporting.
INSERT INTO explanations (
    prediction_id,
    model_id,
    policy_id,
    method,
    scope,
    status,
    algorithm_version,
    failure_reason
)
SELECT
    p.prediction_id,
    p.model_id,
    ep.policy_id,
    'SHAP',
    'LOCAL',
    'FAILED',
    'EXACT_SHAP_1.0',
    'Reference dataset did not satisfy a separate execution requirement'
FROM predictions p
JOIN explanation_policies ep
    ON ep.model_id = p.model_id
WHERE p.subject_reference = 'APPLICANT_002';

-- ------------------------------------------------------------
-- SHAP ATTRIBUTIONS
-- ------------------------------------------------------------

INSERT INTO feature_attributions (
    explanation_id,
    feature_id,
    attribution_value,
    feature_value,
    attribution_rank,
    direction
)
SELECT
    e.explanation_id,
    fd.feature_id,
    values_table.attribution_value,
    values_table.feature_value,
    values_table.attribution_rank,
    values_table.direction
FROM explanations e
JOIN predictions p
    ON p.prediction_id = e.prediction_id
JOIN feature_definitions fd
    ON fd.model_id = e.model_id
JOIN (
    VALUES
    ('income', 0.120000000000::NUMERIC, 85000::NUMERIC, 1, 'POSITIVE'),
    ('debt_ratio', 0.080000000000::NUMERIC, 0.30::NUMERIC, 2, 'POSITIVE'),
    ('credit_history', 0.050000000000::NUMERIC, 14::NUMERIC, 3, 'POSITIVE'),
    ('missed_payments', -0.070000000000::NUMERIC, 1::NUMERIC, 4, 'NEGATIVE'),
    ('employment_years', 0.020000000000::NUMERIC, 9::NUMERIC, 5, 'POSITIVE')
) AS values_table(
    feature_name,
    attribution_value,
    feature_value,
    attribution_rank,
    direction
)
    ON values_table.feature_name = fd.feature_name
WHERE p.subject_reference = 'APPLICANT_001'
  AND e.method = 'SHAP'
  AND e.status = 'COMPLETED';

-- ------------------------------------------------------------
-- GLOBAL EXPLANATION
-- ------------------------------------------------------------

INSERT INTO explanations (
    prediction_id,
    model_id,
    policy_id,
    method,
    scope,
    status,
    baseline_value,
    reconstructed_value,
    model_output,
    algorithm_version,
    execution_duration_ms,
    completed_at
)
SELECT
    p.prediction_id,
    p.model_id,
    ep.policy_id,
    'GLOBAL',
    'GLOBAL',
    'COMPLETED',
    0,
    0,
    0,
    'MEAN_ABSOLUTE_SHAP_1.0',
    35.500,
    NOW()
FROM predictions p
JOIN explanation_policies ep
    ON ep.model_id = p.model_id
WHERE p.subject_reference = 'APPLICANT_001';

INSERT INTO global_importance (
    model_id,
    explanation_id,
    feature_id,
    dataset_id,
    mean_absolute_shap,
    importance_rank,
    evaluated_sample_count
)
SELECT
    e.model_id,
    e.explanation_id,
    fd.feature_id,
    d.dataset_id,
    values_table.importance,
    values_table.importance_rank,
    100
FROM explanations e
JOIN feature_definitions fd
    ON fd.model_id = e.model_id
JOIN background_datasets d
    ON d.model_id = e.model_id
JOIN (
    VALUES
    ('income', 0.135000000000::NUMERIC, 1),
    ('debt_ratio', 0.092000000000::NUMERIC, 2),
    ('credit_history', 0.071000000000::NUMERIC, 3),
    ('missed_payments', 0.052000000000::NUMERIC, 4),
    ('employment_years', 0.026000000000::NUMERIC, 5)
) AS values_table(feature_name, importance, importance_rank)
    ON values_table.feature_name = fd.feature_name
WHERE e.method = 'GLOBAL'
  AND d.dataset_name = 'CREDIT_RISK_REFERENCE';

-- ------------------------------------------------------------
-- VALIDATION RECORDS
-- ------------------------------------------------------------

INSERT INTO explanation_validations (
    explanation_id,
    validation_name,
    passed,
    observed_value,
    expected_value,
    tolerance,
    validation_message
)
SELECT
    e.explanation_id,
    'SHAP_ADDITIVITY',
    TRUE,
    0.850000000000,
    0.850000000000,
    0.000001,
    'Illustrative additivity validation record'
FROM explanations e
JOIN predictions p
    ON p.prediction_id = e.prediction_id
WHERE p.subject_reference = 'APPLICANT_001'
  AND e.method = 'SHAP';

INSERT INTO explanation_validations (
    explanation_id,
    validation_name,
    passed,
    observed_value,
    expected_value,
    tolerance,
    validation_message
)
SELECT
    e.explanation_id,
    'LIME_SURROGATE_FIDELITY',
    FALSE,
    0.840000000000,
    0.850000000000,
    0.005000,
    'Surrogate error exceeds configured tolerance'
FROM explanations e
JOIN predictions p
    ON p.prediction_id = e.prediction_id
WHERE p.subject_reference = 'APPLICANT_001'
  AND e.method = 'LIME';

-- ------------------------------------------------------------
-- AUDIT RECORDS
-- ------------------------------------------------------------

INSERT INTO explanation_audit (
    explanation_id,
    prediction_id,
    event_type,
    actor,
    event_details
)
SELECT
    e.explanation_id,
    e.prediction_id,
    'EXPLANATION_COMPLETED',
    'EXPLAINABILITY_SERVICE',
    jsonb_build_object(
        'method', e.method,
        'algorithm_version', e.algorithm_version
    )
FROM explanations e
WHERE e.status = 'COMPLETED';

INSERT INTO explanation_audit (
    explanation_id,
    prediction_id,
    event_type,
    actor,
    event_details
)
SELECT
    e.explanation_id,
    e.prediction_id,
    'EXPLANATION_FAILED',
    'EXPLAINABILITY_SERVICE',
    jsonb_build_object(
        'reason', e.failure_reason
    )
FROM explanations e
WHERE e.status = 'FAILED';

COMMIT;

-- ============================================================
-- OPERATIONAL QUERIES
-- ============================================================

-- 1. Retrieve local explanations for an applicant.

SELECT *
FROM v_local_explanation_report
WHERE subject_reference = 'APPLICANT_001'
ORDER BY explanation_id, attribution_rank;

-- 2. Retrieve global importance.

SELECT *
FROM v_global_feature_importance
ORDER BY model_id, average_importance DESC;

-- 3. Compare SHAP and LIME execution performance.

SELECT *
FROM v_explanation_method_metrics
ORDER BY model_id, method;

-- 4. Detect failed explanation validation.

SELECT
    e.explanation_id,
    e.method,
    ev.validation_name,
    ev.observed_value,
    ev.expected_value,
    ev.validation_message
FROM explanation_validations ev
JOIN explanations e
    ON e.explanation_id = ev.explanation_id
WHERE ev.passed = FALSE
ORDER BY ev.validated_at DESC;

-- 5. Evaluate SHAP additivity.

SELECT *
FROM validate_shap_additivity(1);

-- 6. Determine whether a prediction is eligible for SHAP.

SELECT *
FROM check_explanation_eligibility(1, 'SHAP');

-- 7. Aggregate positive and negative local contributions.

SELECT
    p.subject_reference,
    fd.feature_name,
    SUM(
        CASE
            WHEN a.attribution_value > 0
            THEN a.attribution_value
            ELSE 0
        END
    ) AS positive_contribution,
    SUM(
        CASE
            WHEN a.attribution_value < 0
            THEN a.attribution_value
            ELSE 0
        END
    ) AS negative_contribution
FROM feature_attributions a
JOIN explanations e
    ON e.explanation_id = a.explanation_id
JOIN predictions p
    ON p.prediction_id = e.prediction_id
JOIN feature_definitions fd
    ON fd.feature_id = a.feature_id
WHERE e.method = 'SHAP'
GROUP BY
    p.subject_reference,
    fd.feature_name
ORDER BY p.subject_reference, fd.feature_name;

-- 8. Identify models with frequent explanation failures.

SELECT
    mv.model_name,
    mv.version,
    e.method,
    COUNT(*) AS total_executions,
    COUNT(*) FILTER (
        WHERE e.status = 'FAILED'
    ) AS failures,
    ROUND(
        100.0 *
        COUNT(*) FILTER (WHERE e.status = 'FAILED') /
        NULLIF(COUNT(*), 0),
        2
    ) AS failure_percentage
FROM explanations e
JOIN model_versions mv
    ON mv.model_id = e.model_id
GROUP BY
    mv.model_name,
    mv.version,
    e.method
ORDER BY failure_percentage DESC;

-- 9. Review the explanation audit trail.

SELECT
    audit_id,
    explanation_id,
    prediction_id,
    event_type,
    actor,
    event_details,
    created_at
FROM explanation_audit
ORDER BY created_at DESC;

-- 10. Check feature coverage for a prediction.

SELECT
    p.prediction_id,
    p.subject_reference,
    COUNT(DISTINCT pf.feature_id) AS supplied_features,
    mv.feature_count AS expected_features,
    COUNT(DISTINCT pf.feature_id) = mv.feature_count AS complete
FROM predictions p
JOIN model_versions mv
    ON mv.model_id = p.model_id
LEFT JOIN prediction_features pf
    ON pf.prediction_id = p.prediction_id
GROUP BY
    p.prediction_id,
    p.subject_reference,
    mv.feature_count;

-- 11. Find explanations still waiting for execution.

SELECT
    explanation_id,
    prediction_id,
    method,
    created_at
FROM explanations
WHERE status IN ('PENDING', 'RUNNING')
ORDER BY created_at;

-- 12. Demonstrate transaction rollback for invalid data.

BEGIN;

-- This statement intentionally violates the feature range.
-- The surrounding transaction is rolled back below.

SAVEPOINT invalid_feature_test;

DO $$
BEGIN
    BEGIN
        INSERT INTO prediction_features (
            prediction_id,
            model_id,
            feature_id,
            feature_value
        )
        SELECT
            p.prediction_id,
            p.model_id,
            fd.feature_id,
            999999999
        FROM predictions p
        JOIN feature_definitions fd
            ON fd.model_id = p.model_id
        WHERE p.subject_reference = 'APPLICANT_001'
          AND fd.feature_name = 'income';

    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE
            'Expected validation failure: %',
            SQLERRM;
    END;
END;
$$;

ROLLBACK TO SAVEPOINT invalid_feature_test;
ROLLBACK;
