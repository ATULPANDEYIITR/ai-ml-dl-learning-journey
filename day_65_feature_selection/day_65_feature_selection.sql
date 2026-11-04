/*
 * FEATURE SELECTION MANAGEMENT DATABASE
 *
 * PostgreSQL 15+
 *
 * Models:
 * - Datasets and feature definitions
 * - Feature-selection experiments
 * - Filter, wrapper, embedded and RFE methods
 * - Feature metrics and rankings
 * - Selected and rejected features
 * - RFE elimination history
 * - Experiment evaluation
 * - Governance policies
 * - Audit history
 *
 * All objects are contained in the feature_selection schema.
 */

BEGIN;

CREATE SCHEMA IF NOT EXISTS feature_selection;

SET search_path TO feature_selection, public;

-- ============================================================
-- 1. ENUMERATIONS
-- ============================================================

CREATE TYPE selection_method AS ENUM (
    'FILTER',
    'WRAPPER',
    'EMBEDDED',
    'RFE'
);

CREATE TYPE experiment_status AS ENUM (
    'CREATED',
    'VALIDATING',
    'RUNNING',
    'COMPLETED',
    'FAILED',
    'CANCELLED'
);

CREATE TYPE feature_status AS ENUM (
    'CANDIDATE',
    'SELECTED',
    'REJECTED',
    'ELIMINATED'
);

CREATE TYPE metric_type AS ENUM (
    'VARIANCE',
    'CORRELATION',
    'ANOVA_F',
    'MUTUAL_INFORMATION',
    'COEFFICIENT',
    'VALIDATION_ACCURACY',
    'RECURSIVE_RANK'
);

-- ============================================================
-- 2. DATASET MANAGEMENT
-- ============================================================

CREATE TABLE datasets (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_name VARCHAR(150) NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    sample_count INTEGER NOT NULL CHECK (sample_count >= 3),
    target_column VARCHAR(100) NOT NULL,
    target_type VARCHAR(30) NOT NULL DEFAULT 'BINARY',
    source_reference TEXT,
    contains_sensitive_data BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dataset_features (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL
        REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    feature_name VARCHAR(150) NOT NULL,
    data_type VARCHAR(40) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_sensitive BOOLEAN NOT NULL DEFAULT FALSE,
    is_target BOOLEAN NOT NULL DEFAULT FALSE,
    missing_percentage NUMERIC(7,4) NOT NULL DEFAULT 0
        CHECK (missing_percentage BETWEEN 0 AND 100),
    variance NUMERIC(20,10)
        CHECK (variance IS NULL OR variance >= 0),
    feature_status feature_status NOT NULL DEFAULT 'CANDIDATE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (dataset_id, feature_name),
    UNIQUE (dataset_id, feature_id),

    CHECK (NOT (is_target AND is_sensitive))
);

CREATE INDEX idx_features_dataset_status
    ON dataset_features(dataset_id, feature_status);

CREATE INDEX idx_features_sensitive
    ON dataset_features(dataset_id, is_sensitive)
    WHERE is_sensitive = TRUE;

-- ============================================================
-- 3. GOVERNANCE POLICIES
-- ============================================================

CREATE TABLE selection_policies (
    policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name VARCHAR(150) NOT NULL UNIQUE,
    maximum_features INTEGER NOT NULL CHECK (maximum_features >= 1),
    minimum_variance NUMERIC(20,10) NOT NULL DEFAULT 0
        CHECK (minimum_variance >= 0),
    maximum_correlation NUMERIC(8,6) NOT NULL DEFAULT 0.95
        CHECK (maximum_correlation BETWEEN 0 AND 1),
    minimum_mutual_information NUMERIC(12,8) NOT NULL DEFAULT 0
        CHECK (minimum_mutual_information >= 0),
    coefficient_threshold NUMERIC(12,8) NOT NULL DEFAULT 0.01
        CHECK (coefficient_threshold >= 0),
    allow_sensitive_features BOOLEAN NOT NULL DEFAULT FALSE,
    require_validation BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 4. EXPERIMENTS
-- ============================================================

CREATE TABLE selection_experiments (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES datasets(dataset_id),
    policy_id BIGINT NOT NULL REFERENCES selection_policies(policy_id),
    experiment_name VARCHAR(180) NOT NULL,
    method selection_method NOT NULL,
    status experiment_status NOT NULL DEFAULT 'CREATED',
    requested_feature_count INTEGER NOT NULL
        CHECK (requested_feature_count >= 1),
    random_seed INTEGER NOT NULL DEFAULT 42,
    training_sample_count INTEGER,
    validation_sample_count INTEGER,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (
        completed_at IS NULL OR
        started_at IS NULL OR
        completed_at >= started_at
    ),

    CHECK (
        status <> 'COMPLETED' OR completed_at IS NOT NULL
    ),

    CHECK (
        status <> 'FAILED' OR failure_reason IS NOT NULL
    ),

    UNIQUE (experiment_id, dataset_id)
);

CREATE INDEX idx_experiments_dataset_method
    ON selection_experiments(dataset_id, method, status);

CREATE INDEX idx_experiments_created
    ON selection_experiments(created_at DESC);

-- ============================================================
-- 5. FEATURE METRICS
-- ============================================================

CREATE TABLE feature_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES selection_experiments(experiment_id) ON DELETE CASCADE,
    feature_id BIGINT NOT NULL
        REFERENCES dataset_features(feature_id),
    metric metric_type NOT NULL,
    metric_value NUMERIC(24,12) NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, feature_id, metric),

    CHECK (
        metric <> 'CORRELATION' OR
        metric_value BETWEEN -1 AND 1
    ),

    CHECK (
        metric <> 'VALIDATION_ACCURACY' OR
        metric_value BETWEEN 0 AND 1
    ),

    CHECK (
        metric <> 'VARIANCE' OR metric_value >= 0
    ),

    CHECK (
        metric <> 'MUTUAL_INFORMATION' OR metric_value >= 0
    )
);

CREATE INDEX idx_feature_metrics_experiment_metric
    ON feature_metrics(experiment_id, metric, metric_value DESC);

CREATE INDEX idx_feature_metrics_feature
    ON feature_metrics(feature_id);

-- ============================================================
-- 6. SELECTION DECISIONS
-- ============================================================

CREATE TABLE feature_selection_decisions (
    decision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES selection_experiments(experiment_id) ON DELETE CASCADE,
    feature_id BIGINT NOT NULL
        REFERENCES dataset_features(feature_id),
    decision feature_status NOT NULL,
    ranking_position INTEGER CHECK (ranking_position >= 1),
    selection_score NUMERIC(24,12),
    decision_reason TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, feature_id),

    CHECK (
        decision <> 'SELECTED' OR ranking_position IS NOT NULL
    )
);

CREATE INDEX idx_decisions_selected
    ON feature_selection_decisions(experiment_id, ranking_position)
    WHERE decision = 'SELECTED';

CREATE INDEX idx_decisions_status
    ON feature_selection_decisions(experiment_id, decision);

-- ============================================================
-- 7. RFE ELIMINATION HISTORY
-- ============================================================

CREATE TABLE rfe_elimination_history (
    elimination_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES selection_experiments(experiment_id) ON DELETE CASCADE,
    feature_id BIGINT NOT NULL
        REFERENCES dataset_features(feature_id),
    iteration_number INTEGER NOT NULL CHECK (iteration_number >= 1),
    importance_score NUMERIC(24,12) NOT NULL
        CHECK (importance_score >= 0),
    remaining_feature_count INTEGER NOT NULL
        CHECK (remaining_feature_count >= 1),
    elimination_reason TEXT NOT NULL,
    eliminated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, iteration_number),
    UNIQUE (experiment_id, feature_id)
);

CREATE INDEX idx_rfe_history_experiment
    ON rfe_elimination_history(experiment_id, iteration_number);

-- ============================================================
-- 8. EXPERIMENT EVALUATION
-- ============================================================

CREATE TABLE experiment_evaluations (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT NOT NULL
        REFERENCES selection_experiments(experiment_id) ON DELETE CASCADE,
    evaluation_dataset VARCHAR(150) NOT NULL,
    evaluation_accuracy NUMERIC(8,6)
        CHECK (evaluation_accuracy BETWEEN 0 AND 1),
    precision_score NUMERIC(8,6)
        CHECK (precision_score BETWEEN 0 AND 1),
    recall_score NUMERIC(8,6)
        CHECK (recall_score BETWEEN 0 AND 1),
    f1_score NUMERIC(8,6)
        CHECK (f1_score BETWEEN 0 AND 1),
    roc_auc NUMERIC(8,6)
        CHECK (roc_auc BETWEEN 0 AND 1),
    selected_feature_count INTEGER NOT NULL
        CHECK (selected_feature_count >= 1),
    evaluation_notes TEXT NOT NULL DEFAULT '',
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (experiment_id, evaluation_dataset)
);

CREATE INDEX idx_evaluations_accuracy
    ON experiment_evaluations(evaluation_accuracy DESC);

-- ============================================================
-- 9. AUDIT LOG
-- ============================================================

CREATE TABLE selection_audit_log (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id BIGINT
        REFERENCES selection_experiments(experiment_id)
        ON DELETE SET NULL,
    event_type VARCHAR(80) NOT NULL,
    event_details JSONB NOT NULL DEFAULT '{}'::JSONB,
    performed_by VARCHAR(120) NOT NULL DEFAULT CURRENT_USER,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_experiment_time
    ON selection_audit_log(experiment_id, occurred_at DESC);

-- ============================================================
-- 10. AUDIT TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION audit_selection_decision()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO selection_audit_log (
        experiment_id,
        event_type,
        event_details
    )
    VALUES (
        NEW.experiment_id,
        'FEATURE_DECISION',
        jsonb_build_object(
            'feature_id', NEW.feature_id,
            'decision', NEW.decision,
            'ranking_position', NEW.ranking_position,
            'score', NEW.selection_score,
            'reason', NEW.decision_reason
        )
    );

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_audit_selection_decision
AFTER INSERT ON feature_selection_decisions
FOR EACH ROW
EXECUTE FUNCTION audit_selection_decision();

-- ============================================================
-- 11. VALIDATION TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION validate_experiment_feature()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    experiment_dataset BIGINT;
    feature_dataset BIGINT;
    sensitive_allowed BOOLEAN;
BEGIN
    SELECT dataset_id
    INTO experiment_dataset
    FROM selection_experiments
    WHERE experiment_id = NEW.experiment_id;

    SELECT dataset_id
    INTO feature_dataset
    FROM dataset_features
    WHERE feature_id = NEW.feature_id;

    IF experiment_dataset IS DISTINCT FROM feature_dataset THEN
        RAISE EXCEPTION
            'Feature % does not belong to experiment dataset %',
            NEW.feature_id,
            experiment_dataset;
    END IF;

    SELECT p.allow_sensitive_features
    INTO sensitive_allowed
    FROM selection_experiments e
    JOIN selection_policies p USING (policy_id)
    WHERE e.experiment_id = NEW.experiment_id;

    IF EXISTS (
        SELECT 1
        FROM dataset_features
        WHERE feature_id = NEW.feature_id
          AND (is_target OR
               (is_sensitive AND NOT sensitive_allowed))
    ) THEN
        RAISE EXCEPTION
            'Target or policy-prohibited sensitive feature cannot be selected';
    END IF;

    IF NEW.decision = 'SELECTED'
       AND NEW.ranking_position IS NULL THEN
        RAISE EXCEPTION
            'Selected features require a ranking position';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_validate_feature_decision
BEFORE INSERT OR UPDATE ON feature_selection_decisions
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_feature();

-- ============================================================
-- 12. EXPERIMENT STATE TRANSITIONS
-- ============================================================

CREATE OR REPLACE FUNCTION validate_experiment_transition()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = NEW.status THEN
        RETURN NEW;
    END IF;

    IF OLD.status = 'CREATED'
       AND NEW.status IN ('VALIDATING', 'CANCELLED') THEN
        RETURN NEW;
    END IF;

    IF OLD.status = 'VALIDATING'
       AND NEW.status IN ('RUNNING', 'FAILED', 'CANCELLED') THEN
        RETURN NEW;
    END IF;

    IF OLD.status = 'RUNNING'
       AND NEW.status IN ('COMPLETED', 'FAILED', 'CANCELLED') THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION
        'Invalid experiment state transition: % -> %',
        OLD.status,
        NEW.status;
END;
$$;

CREATE TRIGGER trg_experiment_transition
BEFORE UPDATE OF status ON selection_experiments
FOR EACH ROW
EXECUTE FUNCTION validate_experiment_transition();

-- ============================================================
-- 13. EXPERIMENT ELIGIBILITY FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION check_experiment_eligibility(
    p_experiment_id BIGINT
)
RETURNS TABLE (
    eligible BOOLEAN,
    selected_count BIGINT,
    requested_count INTEGER,
    reason TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_status experiment_status;
    v_requested INTEGER;
    v_maximum INTEGER;
    v_selected BIGINT;
    v_dataset BIGINT;
BEGIN
    SELECT
        e.status,
        e.requested_feature_count,
        p.maximum_features,
        e.dataset_id
    INTO
        v_status,
        v_requested,
        v_maximum,
        v_dataset
    FROM selection_experiments e
    JOIN selection_policies p USING (policy_id)
    WHERE e.experiment_id = p_experiment_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT
            FALSE,
            0::BIGINT,
            0,
            'Experiment does not exist'::TEXT;
        RETURN;
    END IF;

    SELECT COUNT(*)
    INTO v_selected
    FROM feature_selection_decisions d
    JOIN dataset_features f USING (feature_id)
    WHERE d.experiment_id = p_experiment_id
      AND d.decision = 'SELECTED'
      AND f.dataset_id = v_dataset;

    IF v_status <> 'COMPLETED' THEN
        RETURN QUERY SELECT
            FALSE,
            v_selected,
            v_requested,
            'Experiment must be completed before acceptance'::TEXT;
        RETURN;
    END IF;

    IF v_requested > v_maximum THEN
        RETURN QUERY SELECT
            FALSE,
            v_selected,
            v_requested,
            'Requested feature count exceeds policy limit'::TEXT;
        RETURN;
    END IF;

    IF v_selected = 0 OR v_selected > v_maximum THEN
        RETURN QUERY SELECT
            FALSE,
            v_selected,
            v_requested,
            'Selected feature count violates policy'::TEXT;
        RETURN;
    END IF;

    RETURN QUERY SELECT
        TRUE,
        v_selected,
        v_requested,
        'Selection satisfies eligibility checks'::TEXT;
END;
$$;

-- ============================================================
-- 14. CONTROLLED EXPERIMENT COMPLETION
-- ============================================================

CREATE OR REPLACE PROCEDURE complete_experiment(
    p_experiment_id BIGINT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_eligible BOOLEAN;
    v_reason TEXT;
BEGIN
    SELECT eligible, reason
    INTO v_eligible, v_reason
    FROM check_experiment_eligibility(p_experiment_id);

    IF NOT v_eligible THEN
        RAISE EXCEPTION 'Experiment cannot be completed: %', v_reason;
    END IF;

    UPDATE selection_experiments
    SET status = 'COMPLETED',
        completed_at = CURRENT_TIMESTAMP
    WHERE experiment_id = p_experiment_id
      AND status = 'RUNNING';

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Experiment must be RUNNING before completion';
    END IF;
END;
$$;

-- ============================================================
-- 15. REPORTING VIEWS
-- ============================================================

CREATE OR REPLACE VIEW v_selected_features AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.method,
    d.dataset_name,
    f.feature_name,
    s.ranking_position,
    s.selection_score,
    s.decision_reason
FROM feature_selection_decisions s
JOIN selection_experiments e
    ON e.experiment_id = s.experiment_id
JOIN datasets d
    ON d.dataset_id = e.dataset_id
JOIN dataset_features f
    ON f.feature_id = s.feature_id
WHERE s.decision = 'SELECTED';

CREATE OR REPLACE VIEW v_experiment_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    d.dataset_name,
    e.method,
    e.status,
    COUNT(s.decision_id) FILTER (
        WHERE s.decision = 'SELECTED'
    ) AS selected_features,
    COUNT(s.decision_id) FILTER (
        WHERE s.decision IN ('REJECTED', 'ELIMINATED')
    ) AS rejected_features,
    MAX(ev.evaluation_accuracy) AS validation_accuracy,
    MAX(ev.f1_score) AS validation_f1
FROM selection_experiments e
JOIN datasets d
    ON d.dataset_id = e.dataset_id
LEFT JOIN feature_selection_decisions s
    ON s.experiment_id = e.experiment_id
LEFT JOIN experiment_evaluations ev
    ON ev.experiment_id = e.experiment_id
GROUP BY
    e.experiment_id,
    e.experiment_name,
    d.dataset_name,
    e.method,
    e.status;

CREATE OR REPLACE VIEW v_feature_method_comparison AS
SELECT
    d.dataset_name,
    e.method,
    f.feature_name,
    MAX(m.metric_value) FILTER (
        WHERE m.metric = 'MUTUAL_INFORMATION'
    ) AS mutual_information,
    MAX(m.metric_value) FILTER (
        WHERE m.metric = 'VARIANCE'
    ) AS variance,
    MAX(m.metric_value) FILTER (
        WHERE m.metric = 'COEFFICIENT'
    ) AS coefficient,
    MAX(m.metric_value) FILTER (
        WHERE m.metric = 'VALIDATION_ACCURACY'
    ) AS validation_accuracy,
    MAX(s.decision) AS decision
FROM selection_experiments e
JOIN datasets d USING (dataset_id)
JOIN dataset_features f USING (dataset_id)
LEFT JOIN feature_metrics m
    ON m.experiment_id = e.experiment_id
   AND m.feature_id = f.feature_id
LEFT JOIN feature_selection_decisions s
    ON s.experiment_id = e.experiment_id
   AND s.feature_id = f.feature_id
GROUP BY d.dataset_name, e.method, f.feature_name;

-- ============================================================
-- 16. SAMPLE DATASET
-- ============================================================

INSERT INTO datasets (
    dataset_name,
    description,
    sample_count,
    target_column,
    target_type,
    source_reference,
    contains_sensitive_data
)
VALUES (
    'Customer Risk Classification',
    'Synthetic customer dataset for binary risk classification.',
    200,
    'risk_label',
    'BINARY',
    'Synthetic reproducible dataset',
    TRUE
);

INSERT INTO dataset_features (
    dataset_id,
    feature_name,
    data_type,
    description,
    is_sensitive,
    missing_percentage,
    variance
)
SELECT
    d.dataset_id,
    x.feature_name,
    x.data_type,
    x.description,
    x.is_sensitive,
    x.missing_percentage,
    x.variance
FROM datasets d
CROSS JOIN (
    VALUES
    ('income', 'NUMERIC', 'Annual customer income', FALSE, 0.0, 1250000000.0),
    ('age', 'NUMERIC', 'Customer age', FALSE, 0.0, 210.0),
    ('account_balance', 'NUMERIC', 'Account balance', FALSE, 2.0, 250000000.0),
    ('transaction_count', 'NUMERIC', 'Monthly transactions', FALSE, 0.0, 850.0),
    ('income_duplicate', 'NUMERIC', 'Highly correlated income measurement', FALSE, 0.0, 1250000000.0),
    ('constant_feature', 'NUMERIC', 'Constant control field', FALSE, 0.0, 0.0),
    ('postal_code', 'NUMERIC', 'Sensitive geographic proxy', TRUE, 5.0, 50000.0)
) AS x(feature_name, data_type, description, is_sensitive, missing_percentage, variance)
WHERE d.dataset_name = 'Customer Risk Classification';

-- ============================================================
-- 17. SAMPLE POLICIES
-- ============================================================

INSERT INTO selection_policies (
    policy_name,
    maximum_features,
    minimum_variance,
    maximum_correlation,
    minimum_mutual_information,
    coefficient_threshold,
    allow_sensitive_features,
    require_validation
)
VALUES
(
    'Standard Research Policy',
    3,
    0.00000001,
    0.95,
    0,
    0.01,
    FALSE,
    TRUE
),
(
    'Exploratory Research Policy',
    5,
    0,
    0.99,
    0,
    0.001,
    FALSE,
    FALSE
);

-- ============================================================
-- 18. SAMPLE EXPERIMENTS
-- ============================================================

INSERT INTO selection_experiments (
    dataset_id,
    policy_id,
    experiment_name,
    method,
    status,
    requested_feature_count,
    training_sample_count,
    validation_sample_count,
    started_at
)
SELECT
    d.dataset_id,
    p.policy_id,
    x.experiment_name,
    x.method,
    'RUNNING',
    x.requested_count,
    150,
    50,
    CURRENT_TIMESTAMP
FROM datasets d
CROSS JOIN selection_policies p
CROSS JOIN (
    VALUES
    ('Filter Experiment', 'FILTER'::selection_method, 3),
    ('Wrapper Experiment', 'WRAPPER'::selection_method, 3),
    ('Embedded Experiment', 'EMBEDDED'::selection_method, 3),
    ('RFE Experiment', 'RFE'::selection_method, 3)
) AS x(experiment_name, method, requested_count)
WHERE d.dataset_name = 'Customer Risk Classification'
  AND p.policy_name = 'Standard Research Policy';

-- ============================================================
-- 19. METRIC SAMPLE RECORDS
-- ============================================================

INSERT INTO feature_metrics (
    experiment_id,
    feature_id,
    metric,
    metric_value
)
SELECT
    e.experiment_id,
    f.feature_id,
    x.metric,
    x.metric_value
FROM selection_experiments e
JOIN dataset_features f
    ON f.dataset_id = e.dataset_id
CROSS JOIN LATERAL (
    VALUES
    ('VARIANCE'::metric_type,
        CASE f.feature_name
            WHEN 'income' THEN 1250000000.0
            WHEN 'age' THEN 210.0
            WHEN 'account_balance' THEN 250000000.0
            WHEN 'transaction_count' THEN 850.0
            WHEN 'income_duplicate' THEN 1250000000.0
            WHEN 'constant_feature' THEN 0.0
            ELSE 50000.0
        END),
    ('MUTUAL_INFORMATION'::metric_type,
        CASE f.feature_name
            WHEN 'income' THEN 0.42
            WHEN 'age' THEN 0.18
            WHEN 'account_balance' THEN 0.35
            WHEN 'transaction_count' THEN 0.02
            WHEN 'income_duplicate' THEN 0.41
            WHEN 'constant_feature' THEN 0.0
            ELSE 0.03
        END)
) AS x(metric, metric_value);

-- ============================================================
-- 20. SAMPLE SELECTION DECISIONS
-- ============================================================

-- Filter selection: removes constant and redundant features.
INSERT INTO feature_selection_decisions (
    experiment_id,
    feature_id,
    decision,
    ranking_position,
    selection_score,
    decision_reason
)
SELECT
    e.experiment_id,
    f.feature_id,
    CASE
        WHEN f.feature_name IN (
            'income',
            'account_balance',
            'age'
        ) THEN 'SELECTED'::feature_status
        ELSE 'REJECTED'::feature_status
    END,
    CASE f.feature_name
        WHEN 'income' THEN 1
        WHEN 'account_balance' THEN 2
        WHEN 'age' THEN 3
        ELSE NULL
    END,
    CASE f.feature_name
        WHEN 'income' THEN 0.42
        WHEN 'account_balance' THEN 0.35
        WHEN 'age' THEN 0.18
        ELSE 0
    END,
    CASE
        WHEN f.feature_name = 'constant_feature'
            THEN 'Rejected due to zero variance'
        WHEN f.feature_name = 'income_duplicate'
            THEN 'Rejected due to high correlation'
        WHEN f.feature_name = 'postal_code'
            THEN 'Rejected by sensitive-feature policy'
        WHEN f.feature_name = 'transaction_count'
            THEN 'Low target association'
        ELSE 'Passed filter evaluation'
    END
FROM selection_experiments e
JOIN dataset_features f
    ON f.dataset_id = e.dataset_id
WHERE e.method = 'FILTER'
  AND f.feature_name <> 'postal_code';

-- Wrapper experiment.
INSERT INTO feature_selection_decisions (
    experiment_id,
    feature_id,
    decision,
    ranking_position,
    selection_score,
    decision_reason
)
SELECT
    e.experiment_id,
    f.feature_id,
    CASE
        WHEN f.feature_name IN (
            'income',
            'age',
            'account_balance'
        ) THEN 'SELECTED'::feature_status
        ELSE 'REJECTED'::feature_status
    END,
    CASE f.feature_name
        WHEN 'income' THEN 1
        WHEN 'age' THEN 2
        WHEN 'account_balance' THEN 3
        ELSE NULL
    END,
    CASE f.feature_name
        WHEN 'income' THEN 0.86
        WHEN 'age' THEN 0.84
        WHEN 'account_balance' THEN 0.85
        ELSE 0.40
    END,
    CASE
        WHEN f.feature_name IN (
            'income',
            'age',
            'account_balance'
        ) THEN 'Improved wrapper validation objective'
        ELSE 'Did not improve the selected subset objective'
    END
FROM selection_experiments e
JOIN dataset_features f
    ON f.dataset_id = e.dataset_id
WHERE e.method = 'WRAPPER'
  AND f.feature_name <> 'postal_code';

-- Embedded experiment.
INSERT INTO feature_selection_decisions (
    experiment_id,
    feature_id,
    decision,
    ranking_position,
    selection_score,
    decision_reason
)
SELECT
    e.experiment_id,
    f.feature_id,
    CASE
        WHEN f.feature_name IN (
            'income',
            'account_balance',
            'age'
        ) THEN 'SELECTED'::feature_status
        ELSE 'REJECTED'::feature_status
    END,
    CASE f.feature_name
        WHEN 'income' THEN 1
        WHEN 'account_balance' THEN 2
        WHEN 'age' THEN 3
        ELSE NULL
    END,
    CASE f.feature_name
        WHEN 'income' THEN 1.25
        WHEN 'account_balance' THEN 0.92
        WHEN 'age' THEN 0.38
        ELSE 0
    END,
    'Regularized model coefficient evaluation'
FROM selection_experiments e
JOIN dataset_features f
    ON f.dataset_id = e.dataset_id
WHERE e.method = 'EMBEDDED'
  AND f.feature_name <> 'postal_code';

-- RFE experiment.
INSERT INTO feature_selection_decisions (
    experiment_id,
    feature_id,
    decision,
    ranking_position,
    selection_score,
    decision_reason
)
SELECT
    e.experiment_id,
    f.feature_id,
    CASE
        WHEN f.feature_name IN (
            'income',
            'account_balance',
            'age'
        ) THEN 'SELECTED'::feature_status
        ELSE 'ELIMINATED'::feature_status
    END,
    CASE f.feature_name
        WHEN 'income' THEN 1
        WHEN 'account_balance' THEN 2
        WHEN 'age' THEN 3
        ELSE NULL
    END,
    CASE f.feature_name
        WHEN 'income' THEN 1.20
        WHEN 'account_balance' THEN 0.91
        WHEN 'age' THEN 0.36
        ELSE 0.01
    END,
    CASE
        WHEN f.feature_name IN (
            'income',
            'account_balance',
            'age'
        ) THEN 'Retained after iterative elimination'
        ELSE 'Removed due to low model importance'
    END
FROM selection_experiments e
JOIN dataset_features f
    ON f.dataset_id = e.dataset_id
WHERE e.method = 'RFE'
  AND f.feature_name <> 'postal_code';

-- ============================================================
-- 21. RFE ELIMINATION HISTORY
-- ============================================================

INSERT INTO rfe_elimination_history (
    experiment_id,
    feature_id,
    iteration_number,
    importance_score,
    remaining_feature_count,
    elimination_reason
)
SELECT
    e.experiment_id,
    f.feature_id,
    x.iteration_number,
    x.importance_score,
    x.remaining_count,
    x.reason
FROM selection_experiments e
JOIN (
    VALUES
    ('constant_feature', 1, 0.00, 6, 'Zero variance'),
    ('transaction_count', 2, 0.04, 5, 'Low model importance'),
    ('income_duplicate', 3, 0.12, 4, 'Redundant predictive information')
) AS x(feature_name, iteration_number, importance_score, remaining_count, reason)
    ON TRUE
JOIN dataset_features f
    ON f.dataset_id = e.dataset_id
   AND f.feature_name = x.feature_name
WHERE e.method = 'RFE';

-- ============================================================
-- 22. EVALUATION RECORDS
-- ============================================================

INSERT INTO experiment_evaluations (
    experiment_id,
    evaluation_dataset,
    evaluation_accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    selected_feature_count,
    evaluation_notes
)
SELECT
    experiment_id,
    'Held-out validation dataset',
    CASE method
        WHEN 'FILTER' THEN 0.86
        WHEN 'WRAPPER' THEN 0.88
        WHEN 'EMBEDDED' THEN 0.87
        WHEN 'RFE' THEN 0.87
    END,
    0.85,
    0.84,
    0.845,
    CASE method
        WHEN 'FILTER' THEN 0.89
        WHEN 'WRAPPER' THEN 0.91
        WHEN 'EMBEDDED' THEN 0.90
        WHEN 'RFE' THEN 0.90
    END,
    3,
    'Illustrative synthetic evaluation record; not a measured model result.'
FROM selection_experiments;

-- ============================================================
-- 23. CONTROLLED COMPLETION
-- ============================================================

-- Transition through VALIDATING before completing.
UPDATE selection_experiments
SET status = 'VALIDATING'
WHERE status = 'RUNNING';

-- The sample experiments have three selected features each.
UPDATE selection_experiments
SET status = 'RUNNING'
WHERE status = 'VALIDATING';

-- Eligibility checks and procedure-based completion.
DO $$
DECLARE
    experiment_record RECORD;
BEGIN
    FOR experiment_record IN
        SELECT experiment_id
        FROM selection_experiments
        ORDER BY experiment_id
    LOOP
        CALL complete_experiment(
            experiment_record.experiment_id
        );
    END LOOP;
END;
$$;

-- ============================================================
-- 24. RETRIEVAL QUERIES
-- ============================================================

-- Selected features per experiment.
SELECT *
FROM v_selected_features
ORDER BY experiment_id, ranking_position;

-- Full experiment report.
SELECT *
FROM v_experiment_summary
ORDER BY experiment_id;

-- ============================================================
-- 25. VALIDATION QUERIES
-- ============================================================

-- Detect constant or near-constant features.
SELECT
    dataset_id,
    feature_name,
    variance,
    CASE
        WHEN variance IS NULL THEN 'NOT_MEASURED'
        WHEN variance <= 0.00000001 THEN 'REJECT'
        ELSE 'ELIGIBLE'
    END AS variance_decision
FROM dataset_features
ORDER BY dataset_id, feature_name;

-- Detect sensitive features excluded by policy.
SELECT
    f.feature_name,
    p.policy_name,
    p.allow_sensitive_features,
    CASE
        WHEN f.is_sensitive AND NOT p.allow_sensitive_features
            THEN 'BLOCKED'
        ELSE 'ALLOWED'
    END AS eligibility
FROM dataset_features f
CROSS JOIN selection_policies p;

-- Detect experiments exceeding their policy feature limit.
SELECT
    e.experiment_id,
    e.experiment_name,
    e.requested_feature_count,
    p.maximum_features
FROM selection_experiments e
JOIN selection_policies p USING (policy_id)
WHERE e.requested_feature_count > p.maximum_features;

-- ============================================================
-- 26. AGGREGATION AND COMPARISON
-- ============================================================

SELECT
    method,
    COUNT(*) AS experiment_count,
    AVG(validation_accuracy) AS average_accuracy,
    AVG(validation_f1) AS average_f1,
    AVG(selected_features) AS average_selected_features
FROM v_experiment_summary
GROUP BY method
ORDER BY average_accuracy DESC NULLS LAST;

-- Compare feature selection frequency by method.
SELECT
    feature_name,
    COUNT(*) FILTER (
        WHERE decision = 'SELECTED'
    ) AS selected_count,
    COUNT(DISTINCT experiment_id) AS experiment_count
FROM v_selected_features
GROUP BY feature_name
ORDER BY selected_count DESC, feature_name;

-- ============================================================
-- 27. RFE REPORT
-- ============================================================

SELECT
    e.experiment_name,
    h.iteration_number,
    f.feature_name,
    h.importance_score,
    h.remaining_feature_count,
    h.elimination_reason
FROM rfe_elimination_history h
JOIN selection_experiments e
    ON e.experiment_id = h.experiment_id
JOIN dataset_features f
    ON f.feature_id = h.feature_id
ORDER BY e.experiment_id, h.iteration_number;

-- ============================================================
-- 28. FEATURE METHOD COMPARISON
-- ============================================================

SELECT *
FROM v_feature_method_comparison
ORDER BY dataset_name, feature_name, method;

-- ============================================================
-- 29. ELIGIBILITY EVALUATION
-- ============================================================

SELECT
    e.experiment_name,
    result.eligible,
    result.selected_count,
    result.requested_count,
    result.reason
FROM selection_experiments e
CROSS JOIN LATERAL
    check_experiment_eligibility(e.experiment_id) AS result
ORDER BY e.experiment_id;

-- ============================================================
-- 30. AUDIT REPORT
-- ============================================================

SELECT
    audit_id,
    experiment_id,
    event_type,
    event_details,
    performed_by,
    occurred_at
FROM selection_audit_log
ORDER BY occurred_at DESC;

-- ============================================================
-- 31. FAILURE CASE DEMONSTRATION
-- ============================================================

-- This query reports sensitive features blocked by policy.
-- It does not modify or delete records.
SELECT
    f.feature_name,
    f.is_sensitive,
    p.policy_name,
    'INELIGIBLE' AS status,
    'Sensitive features are prohibited by this policy'
        AS failure_reason
FROM dataset_features f
CROSS JOIN selection_policies p
WHERE f.is_sensitive
  AND NOT p.allow_sensitive_features;

-- ============================================================
-- 32. TRANSACTION COMPLETION
-- ============================================================

COMMIT;
